#include "shell/App.h"
#include <GL/glew.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <vector>
#include <glm/gtc/matrix_transform.hpp>
#include <imgui.h>

#include "remo/CudaCheck.h"
#include "remo/unsuck.hpp"
#include "io/LasReader.h"
#include "pipelines/CudalodPipeline.h"
#include "pipelines/FlatPipeline.h"
#include "pipelines/RemolodPipeline.h"
#include "pipelines/SimlodPipeline.h"

namespace fs = std::filesystem;

namespace remo {
namespace {

mat4 toDeviceMat(const glm::dmat4& m) {
	mat4 out;
	for (int r = 0; r < 4; ++r) {
		for (int c = 0; c < 4; ++c) {
			out.rows[r][c] = static_cast<float>(m[c][r]);
		}
	}
	return out;
}

}

App::App() = default;
App::~App() = default;

bool App::init(const AppOptions& options, std::string* err) {
	m_options = options;

	m_settings.showBoundingBox = options.showBoundingBox;
	m_settings.showPoints = !options.hidePoints;

	if (!m_renderer.init("RemoBench", options.width, options.height, err)) {
		return false;
	}

	m_cuda = std::make_unique<CudaContext>();
	printf("remobench: CUDA on %s, %d SMs, sm_%d%d, %.1f GB free of %.1f GB\n",
	       m_cuda->deviceName().c_str(), m_cuda->numSMs(), m_cuda->ccMajor(),
	       m_cuda->ccMinor(), m_cuda->freeMemory() / 1e9,
	       m_cuda->totalMemory() / 1e9);

	registerPipelines();

	m_renderer.onFileDrop([this](const std::vector<std::string>& files) {
		if (files.empty()) return;
		requestLoad(files.front(), fs::path(files.front()).filename().string());
	});

	std::string loadErr;
	bool loaded = false;
	if (!options.files.empty()) {
		loaded = loadCloud(options.files, &loadErr);
	} else if (options.syntheticPoints > 0) {
		loaded = loadSynthetic(options.syntheticPoints, &loadErr);
	} else {
		loaded = loadSynthetic(1'000'000, &loadErr);
		if (loaded) {
			m_status = "no input given; showing 1M synthetic points "
			           "(--open <file> or --synthetic N)";
		}
	}
	if (!loaded) {
		m_status = loadErr;
		m_statusIsError = true;
		m_startupFailed = true;
		m_startupError = loadErr;
		fprintf(stderr, "remobench: %s\n", loadErr.c_str());
	}

	return true;
}

void App::registerPipelines() {
	m_registry.add([this] { return std::make_unique<FlatPipeline>(*m_cuda); });
	m_registry.add([this] {
		auto p = std::make_unique<RemolodPipeline>(*m_cuda);
		p->setAccumEnabled(!m_options.remolodNoAccum);
		p->setPhaseTimings(m_options.remolodPhaseTimings);
		p->setFixedDepth(m_options.remolodFixedDepth);
		return p;
	});
	m_registry.add([this] { return std::make_unique<CudalodPipeline>(*m_cuda); });
	m_registry.add([this] { return std::make_unique<SimlodPipeline>(*m_cuda); });
}

DeviceBudget App::computeBudget() const {
	DeviceBudget budget;
	budget.vramTotal = m_cuda->totalMemory();
	budget.vramFreeAtStartup = m_cuda->freeMemory();

	if (m_options.deviceBudgetBytes != 0) {
		budget.bytes = m_options.deviceBudgetBytes;
		return budget;
	}

	constexpr double kUsableFraction = 0.85;
	const size_t reserve = 512ull << 20;
	const size_t free = budget.vramFreeAtStartup;
	budget.bytes = free > reserve
	                   ? static_cast<size_t>((free - reserve) * kUsableFraction)
	                   : 0;
	return budget;
}

bool App::loadSynthetic(uint64_t numPoints, std::string* err) {
	m_source = makeSyntheticSource(*m_cuda, numPoints);
	if (!m_source) {
		if (err) *err = "could not create the synthetic point source";
		return false;
	}
	m_meta = m_source->meta();
	return activateCloud(err);
}

bool App::loadCloud(const std::vector<std::string>& files, std::string* err) {
	std::vector<std::string> accepted;
	for (const std::string& f : files) {
		if (iEndsWith(f, ".simlod") || iEndsWith(f, ".las") || iEndsWith(f, ".laz")) {
			accepted.push_back(f);
		}
	}
	if (accepted.empty()) {
		if (err) *err = "no .las / .laz / .simlod among the given files";
		return false;
	}

	const double tStart = now();
	std::unique_ptr<PointSource> source = openPointSource(*m_cuda, accepted, err);
	if (!source) return false;

	m_source = std::move(source);
	m_meta = m_source->meta();

	if (!activateCloud(err)) return false;

	const double elapsed = std::max(1e-6, now() - tStart);
	const std::string points = formatNumber(static_cast<double>(m_meta.numPoints));
	const double mps = static_cast<double>(m_meta.numPoints) / 1e6 / elapsed;

	// A streaming source has read coordinates and nothing else at this point, so calling
	// that a load at N MP/s would be quoting a throughput for work that has not happened
	// -- the points arrive a ring slot at a time while the tree is being built.
	if (m_source->isFullyResident()) {
		m_status = std::format("loaded {} points in {:.2f}s ({:.0f} MP/s)", points,
		                       elapsed, mps);
		printf("remobench: loaded %s points from %s in %.2fs (%.0f MP/s)\n",
		       points.c_str(), accepted.front().c_str(), elapsed, mps);
	} else {
		const uint32_t slots = m_source->view().numSlots;
		m_status = std::format("opened {} points in {:.2f}s (box scan); streaming "
		                       "through a {}-slot ring",
		                       points, elapsed, slots);
		printf("remobench: opened %s points from %s in %.2fs (box scan only, "
		       "%.0f MP/s); streaming through a %u-slot ring\n",
		       points.c_str(), accepted.front().c_str(), elapsed, mps, slots);
	}
	m_statusIsError = false;
	fflush(stdout);
	return true;
}

bool App::activateCloud(std::string* err) {
	m_budget = computeBudget();

	m_profiler.clear();
	m_frameTimeStats.clear();
	m_frameHistory.clear();

	// Ingest is started inside switchTo(), from the incoming pipeline's PipelineInfo:
	// whether the cloud has to be resident, and how deep its ring is, are the
	// pipeline's requirements and only it knows them.
	const std::string wanted =
		m_registry.activeId().empty() ? m_options.pipeline : m_registry.activeId();
	if (!m_registry.switchTo(wanted, m_source.get(), m_meta, m_budget, err)) {
		return false;
	}

	// Clearing the gate is not the same as fitting. Say so up front rather than letting
	// a silently truncated tree be read as a complete one.
	if (const PipelineInfo* info = m_registry.find(wanted)) {
		const std::string warning =
			m_registry.completionWarning(*info, m_meta, m_budget);
		if (!warning.empty()) {
			printf("remobench: WARNING -- %s: %s\n", wanted.c_str(), warning.c_str());
			fflush(stdout);
		}
	}

	m_renderer.controls().frameBox(
		glm::dvec3(0.0, 0.0, 0.0),
		glm::dvec3(m_meta.boxSize[0], m_meta.boxSize[1], m_meta.boxSize[2]),
		m_renderer.camera().fovyRad());

	const double extent = std::max({static_cast<double>(m_meta.boxSize[0]),
	                                static_cast<double>(m_meta.boxSize[1]),
	                                static_cast<double>(m_meta.boxSize[2]), 1.0});
	m_renderer.camera().near = extent * 1e-4;
	m_renderer.camera().far = extent * 100.0;

	m_hasFrozen = false;
	return true;
}

SharedUniforms App::buildUniforms() const {
	const Camera& cam = m_renderer.camera();

	SharedUniforms u = {};
	u.width = static_cast<float>(m_renderer.framebuffer().width());
	u.height = static_cast<float>(m_renderer.framebuffer().height());
	u.fovyRad = static_cast<float>(cam.fovyRad());
	u.time = static_cast<float>(now());

	u.view = toDeviceMat(cam.view);
	u.proj = toDeviceMat(cam.proj);
	u.transform = toDeviceMat(cam.proj * cam.view);

	u.boxMin = {0.0f, 0.0f, 0.0f};
	u.boxMax = {m_meta.boxSize[0], m_meta.boxSize[1], m_meta.boxSize[2]};

	u.frameCounter = m_frameCounter;

	u.lodPixelBudget = m_settings.lodPixelBudget;
	u.minNodeSize = m_settings.minNodeSize;
	u.lodScale = m_settings.lodScale;

	u.pointSize = m_settings.pointSize;
	u.colorMode = m_settings.colorMode;
	u.edlStrength = m_settings.edlStrength;

	u.doUpdateVisibility = m_settings.doUpdateVisibility ? 1 : 0;
	u.useHighQualityShading = m_settings.useHighQualityShading ? 1 : 0;
	u.enableEDL = m_settings.enableEDL ? 1 : 0;
	u.showBoundingBox = m_settings.showBoundingBox ? 1 : 0;
	u.showPoints = m_settings.showPoints ? 1 : 0;

	return u;
}

std::vector<DatasetEntry> scanDatasetDir(const std::string& dir) {
	std::vector<DatasetEntry> out;

	std::error_code ec;
	const fs::path root(dir);
	if (!fs::exists(root, ec)) return out;

	for (fs::recursive_directory_iterator it(root, ec), end; it != end;
	     it.increment(ec)) {
		if (ec) break;
		if (!it->is_regular_file(ec)) continue;

		std::string ext = it->path().extension().string();
		std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
			return static_cast<char>(std::tolower(c));
		});
		if (ext != ".simlod" && ext != ".las" && ext != ".laz") continue;

		DatasetEntry entry;
		entry.path = fs::absolute(it->path(), ec).string();
		entry.label = fs::relative(it->path(), root, ec).string();
		if (entry.label.empty()) entry.label = it->path().filename().string();
		entry.bytes = it->file_size(ec);

		if (ext == ".simlod") {
			entry.supported = true;
			if (entry.bytes > 24) entry.numPoints = (entry.bytes - 24) / 16;
		} else {
			LasHeaderInfo info;
			std::string headerErr;
			if (readLasHeader(entry.path, info, &headerErr)) {
				entry.supported = true;
				entry.numPoints = info.numPoints;
			} else {
				entry.supported = false;
				entry.note = headerErr;
			}
		}

		out.push_back(std::move(entry));
	}

	std::sort(out.begin(), out.end(),
	          [](const DatasetEntry& a, const DatasetEntry& b) {
		          if (a.supported != b.supported) return a.supported;
		          return a.label < b.label;
	          });
	return out;
}

void App::scanDatasets() {
	m_datasetsScanned = true;

	if (const char* env = std::getenv("REMOBENCH_DATA_DIR")) {
		if (*env) m_datasetDir = env;
	}

	m_datasets = scanDatasetDir(m_datasetDir);

	std::error_code ec;
	m_selectedDataset = -1;
	if (!m_meta.files.empty()) {
		const std::string current = fs::absolute(m_meta.files.front(), ec).string();
		for (size_t i = 0; i < m_datasets.size(); ++i) {
			if (m_datasets[i].path == current) {
				m_selectedDataset = static_cast<int>(i);
				break;
			}
		}
	}
}

void App::requestLoad(const std::string& pathOrSynthetic, const std::string& label) {
	m_pendingLoadPath = pathOrSynthetic;
	m_pendingLoadDelayFrames = 1;
	m_status = "loading " + label + "...";
	m_statusIsError = false;
}

void App::applyPendingLoad() {
	if (m_pendingLoadPath.empty()) return;
	if (m_pendingLoadDelayFrames > 0) {
		--m_pendingLoadDelayFrames;
		return;
	}

	const std::string request = m_pendingLoadPath;
	m_pendingLoadPath.clear();

	std::string err;
	bool ok = false;
	if (request.rfind("synthetic:", 0) == 0) {
		const uint64_t n = std::strtoull(request.c_str() + 10, nullptr, 10);
		ok = loadSynthetic(n, &err);
		if (ok) {
			m_status = std::format("{} synthetic points",
			                       formatNumber(static_cast<double>(n)));
			m_statusIsError = false;
		}
	} else {
		ok = loadCloud({request}, &err);
	}

	if (!ok) {
		m_status = err;
		m_statusIsError = true;
		fprintf(stderr, "remobench: %s\n", err.c_str());
	} else {
		m_startupFailed = false;
		m_startupError.clear();
	}

	if (ok && m_registry.active()) {
		for (const PipelineInfo& info : m_registry.list()) {
			if (info.id != m_registry.activeId()) continue;
			if (!m_registry.unsupportedReason(info, m_meta, m_budget).empty()) {
				m_pendingPipeline = "flat";
				m_status += "  (fell back to flat: " +
				            m_registry.unsupportedReason(info, m_meta, m_budget) + ")";
			}
		}
	}

	m_datasetsScanned = false;
}

void App::applyPendingPipelineSwitch() {
	if (m_pendingPipeline.empty()) return;

	const std::string id = m_pendingPipeline;
	m_pendingPipeline.clear();
	if (id == m_registry.activeId()) return;

	std::string err;
	if (!m_registry.switchTo(id, m_source.get(), m_meta, m_budget, &err)) {
		m_status = "could not switch to " + id + ": " + err;
		m_statusIsError = true;
		fprintf(stderr, "remobench: %s\n", m_status.c_str());
		return;
	}
	m_status = "switched to " + id;
	m_statusIsError = false;
}

int App::run() {
	auto update = [this] {
		if (!m_options.switchToPipeline.empty() &&
		    m_frameCounter == static_cast<uint64_t>(m_options.switchAfterFrames)) {
			m_pendingPipeline = m_options.switchToPipeline;
		}
		applyPendingLoad();
		applyPendingPipelineSwitch();
	};

	auto render = [this] {
		const Regime regime =
			m_options.strictTiming ? Regime::Strict : Regime::Deferred;
		m_profiler.beginFrame(m_frameCounter, regime, nullptr);
		m_frameTimeStats.add(m_renderer.frameMs());
		m_frameHistory.add(static_cast<float>(m_renderer.frameMs()));

		SharedUniforms uniforms = buildUniforms();

		if (m_settings.doUpdateVisibility || !m_hasFrozen) {
			m_frozen = uniforms;
			m_hasFrozen = true;
		}
		uniforms.transformFrozen = m_frozen.transform;
		uniforms.transformFrozenInv = m_frozen.transform;

		ILodPipeline* pipeline = m_registry.active();

		if (pipeline && m_source) {
			Framebuffer& fb = m_renderer.framebuffer();
			std::string interopErr;
			if (m_interop.bind(fb.colorTexture(), fb.width(), fb.height(),
			                   &interopErr) &&
			    m_interop.map(nullptr, &interopErr)) {

				FrameContext frame;
				frame.uniforms = uniforms;
				frame.targets.surface = m_interop.surface();
				frame.targets.width = fb.width();
				frame.targets.height = fb.height();
				frame.stream = nullptr;
				frame.numSMs = m_cuda->numSMs();
				frame.strictTiming = m_options.strictTiming;
				frame.profiler = &m_profiler;

				pipeline->render(frame);
				// Refill the ring before the consumer runs, within whatever window
				// the last frame's consumption signal opened. A no-op for a
				// whole-resident pipeline.
				m_source->pump();
				pipeline->build(*m_source, frame);

				m_interop.unmap(nullptr);
			} else if (!interopErr.empty()) {
				m_status = interopErr;
				m_statusIsError = true;
			}
		}

		m_profiler.endFrame();

		drawGui();
		++m_frameCounter;
	};

	while (m_renderer.runFrame(update, render)) {
		if (!m_options.dumpFramePath.empty() &&
		    m_frameCounter >= static_cast<uint64_t>(m_options.dumpAfterFrames)) {
			const bool ok = dumpFrame(m_options.dumpFramePath);
			m_renderer.requestClose();
			if (!ok) return 1;
		}
	}
	// A cloud or pipeline asked for on the command line that never activated is a
	// failure, whatever the window did afterwards.
	return m_startupFailed ? 1 : 0;
}

bool App::dumpFrame(const std::string& path) {
	const Framebuffer& fb = m_renderer.framebuffer();
	const int w = fb.width();
	const int h = fb.height();
	if (w <= 0 || h <= 0 || fb.fbo() == 0) {
		fprintf(stderr, "remobench: nothing to dump (framebuffer not ready)\n");
		return false;
	}

	std::vector<unsigned char> rgba(static_cast<size_t>(w) * h * 4);
	if (m_options.dumpIncludeGui) {
		glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
		const GLboolean debugWasOn = glIsEnabled(GL_DEBUG_OUTPUT);
		if (debugWasOn) glDisable(GL_DEBUG_OUTPUT);

		glReadBuffer(GL_FRONT);
		glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
		glReadBuffer(GL_BACK);
		while (glGetError() != GL_NO_ERROR) {
		}

		if (debugWasOn) glEnable(GL_DEBUG_OUTPUT);
	} else {
		glBindFramebuffer(GL_READ_FRAMEBUFFER, fb.fbo());
		glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
	}
	glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);

	FILE* out = fopen(path.c_str(), "wb");
	if (!out) {
		fprintf(stderr, "remobench: cannot write %s\n", path.c_str());
		return false;
	}
	fprintf(out, "P6\n%d %d\n255\n", w, h);
	std::vector<unsigned char> row(static_cast<size_t>(w) * 3);
	for (int y = h - 1; y >= 0; --y) {
		const unsigned char* src = rgba.data() + static_cast<size_t>(y) * w * 4;
		for (int x = 0; x < w; ++x) {
			row[x * 3 + 0] = src[x * 4 + 0];
			row[x * 3 + 1] = src[x * 4 + 1];
			row[x * 3 + 2] = src[x * 4 + 2];
		}
		fwrite(row.data(), 1, row.size(), out);
	}
	fclose(out);
	printf("remobench: wrote %s (%dx%d)\n", path.c_str(), w, h);

	// The budget moves with whatever else was on the GPU, so a capture that does not
	// name it cannot be interpreted afterwards. Printed outside the pipeline block
	// because it is exactly what a refusal has to be read against.
	printf("  device budget       %.3f GB (%.3f GB free of %.3f GB at startup%s)\n",
	       static_cast<double>(m_budget.bytes) / 1e9,
	       static_cast<double>(m_budget.vramFreeAtStartup) / 1e9,
	       static_cast<double>(m_budget.vramTotal) / 1e9,
	       m_options.deviceBudgetBytes ? ", pinned by --device-budget" : "");

	if (!m_registry.active()) {
		printf("remobench: no active pipeline\n");
		if (!m_startupError.empty()) {
			printf("  refused             %s\n", m_startupError.c_str());
		}
		if (!m_status.empty() && m_status != m_startupError) {
			printf("  status              %s\n", m_status.c_str());
		}
		fflush(stdout);
		return true;
	}

	if (const ILodPipeline* pipeline = m_registry.active()) {
		const PipelineStats& s = pipeline->stats();
		printf("remobench: pipeline=%s\n", m_registry.activeId().c_str());
		printf("  points              %s\n",
		       formatNumber(static_cast<double>(s.numPoints)).c_str());
		printf("  voxels              %s\n",
		       formatNumber(static_cast<double>(s.numVoxels)).c_str());
		printf("  nodes               %s (inner %s, leaves %s)\n",
		       formatNumber(static_cast<double>(s.numNodes)).c_str(),
		       formatNumber(static_cast<double>(s.numInner)).c_str(),
		       formatNumber(static_cast<double>(s.numLeaves)).c_str());
		printf("  visible samples     %s\n",
		       formatNumber(static_cast<double>(s.numVisiblePoints +
		                                       s.numVisibleVoxels)).c_str());
		printf("  visible nodes       %s (drawn this frame)\n",
		       formatNumber(static_cast<double>(s.numVisibleNodes)).c_str());
		printf("  max points/node     %s\n",
		       formatNumber(static_cast<double>(s.maxPointsPerNode)).c_str());

		for (const std::string& line : pipeline->diagnostics()) {
			const size_t tab = line.find('\t');
			if (tab == std::string::npos) {
				printf("  %s\n", line.c_str());
			} else {
				printf("  %-19s %s\n", line.substr(0, tab).c_str(),
				       line.substr(tab + 1).c_str());
			}
		}

		const TimingScopes scopes = pipeline->timingScopes();
		const BuildTotals build = buildTotals(m_profiler, scopes);
		if (build.measured) {
			printf("  build device ms     %.2f (%llu launches)\n", build.ms,
			       static_cast<unsigned long long>(build.launches));
		} else {
			printf("  build device ms     not measured\n");
		}

		const ScopeStats* render =
			scopes.render.empty() ? nullptr : m_profiler.find(scopes.render);
		if (render) {
			printf("  render device ms    %.2f  (median %.2f, p95 %.2f, n=%llu, %s)\n",
			       render->last(), render->median(), render->percentile(0.95),
			       static_cast<unsigned long long>(render->count()),
			       regimeName(m_profiler.regime()));
		} else {
			printf("  render device ms    not measured (%s regime)\n",
			       regimeName(m_profiler.regime()));
		}

		printf("  frame wall ms       %.2f  (median %.2f, p95 %.2f, n=%llu)\n",
		       m_frameTimeStats.last(), m_frameTimeStats.median(),
		       m_frameTimeStats.percentile(0.95),
		       static_cast<unsigned long long>(m_frameTimeStats.count()));

		printf("  device high water   %.3f GB of %.3f GB\n",
		       static_cast<double>(s.bytesHighWater) / 1e9,
		       static_cast<double>(s.bytesAllocated) / 1e9);

		// Predicted against observed, so the 26 B/pt floor is a claim the dump checks
		// rather than a number quoted from a paper. A large disagreement means the
		// coefficient is wrong for this build, which is worth knowing early.
		if (const PipelineInfo* info = m_registry.find(m_registry.activeId())) {
			if (m_meta.numPoints > 0 && info->minBytesPerPointEstimate > 0.0) {
				const double predicted =
					m_registry.predictedCompletion(*info, m_meta, m_budget);
				const double observed = static_cast<double>(s.numPointsIngested) /
				                        static_cast<double>(m_meta.numPoints);
				printf("  ingested            %.1f%% of the cloud (predicted %.1f%% "
				       "at %.0f B/pt)\n",
				       observed * 100.0, predicted * 100.0,
				       info->minBytesPerPointEstimate);
			}
		}
		if (m_profiler.droppedScopes() > 0) {
			printf("  WARNING %llu profiler sample(s) dropped\n",
			       static_cast<unsigned long long>(m_profiler.droppedScopes()));
		}

		printf("  scopes (%s regime)\n", regimeName(m_profiler.regime()));
		for (const std::string& name : m_profiler.scopeNames()) {
			const ScopeStats* st = m_profiler.find(name);
			if (!st) continue;
			printf("    %-20s n=%-6llu last %7.3f  med %7.3f  p95 %7.3f  "
			       "min %7.3f  max %7.3f  total %9.2f\n",
			       name.c_str(), static_cast<unsigned long long>(st->count()),
			       st->last(), st->median(), st->percentile(0.95), st->min(), st->max(),
			       st->total());
		}
		if (s.allocOverflow) printf("  WARNING allocator overflow\n");
		if (s.nodeCapacityReached) printf("  WARNING node pool exhausted\n");
		if (s.memCapacityReached) printf("  WARNING memory budget reached\n");
		fflush(stdout);
	}
	return true;
}

}
