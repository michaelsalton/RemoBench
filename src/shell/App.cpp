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

// glm is column-major; SharedUniforms::mat4 is row-major (matching how the device
// code multiplies). Transposing here means the device side never has to think about
// it -- upstream does the same, via glm::transpose at getUniforms().
mat4 toDeviceMat(const glm::dmat4& m) {
	mat4 out;
	for (int r = 0; r < 4; ++r) {
		for (int c = 0; c < 4; ++c) {
			out.rows[r][c] = static_cast<float>(m[c][r]);
		}
	}
	return out;
}

}  // namespace

App::App() = default;
App::~App() = default;

bool App::init(const AppOptions& options, std::string* err) {
	m_options = options;

	// Seed the shared settings the panel then owns, so --show-bounds and --hide-points
	// give a scripted run the same view a click would.
	m_settings.showBoundingBox = options.showBoundingBox;
	m_settings.showPoints = !options.hidePoints;

	if (!m_renderer.init("RemoBench", options.width, options.height, err)) {
		return false;
	}

	// The CUDA context must be created AFTER the GL context, so CUDA-GL interop can
	// find it.
	m_cuda = std::make_unique<CudaContext>();
	printf("remobench: CUDA on %s, %d SMs, sm_%d%d, %.1f GB free of %.1f GB\n",
	       m_cuda->deviceName().c_str(), m_cuda->numSMs(), m_cuda->ccMajor(),
	       m_cuda->ccMinor(), m_cuda->freeMemory() / 1e9,
	       m_cuda->totalMemory() / 1e9);

	registerPipelines();

	// Drops go through the same deferred path as the dropdown, so there is exactly one
	// place where a cloud gets loaded and one place where the pipeline-destroying
	// switchTo() can happen.
	m_renderer.onFileDrop([this](const std::vector<std::string>& files) {
		if (files.empty()) return;
		requestLoad(files.front(), fs::path(files.front()).filename().string());
	});

	// Load whatever was asked for. A failure here is reported and survivable: the
	// window still comes up, which matters because that is where the error is shown.
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
		fprintf(stderr, "remobench: %s\n", loadErr.c_str());
	}

	return true;
}

void App::registerPipelines() {
	// One line per pipeline. Everything the shell needs to know about it -- id,
	// display name, whether it streams, how much device memory it wants per point --
	// comes from its own info(), so there is nothing to keep in sync here.
	//
	// Four, in the order they mean something:
	//   flat     the control condition and image-quality ground truth -- no LOD at all
	//   remolod  RemoBench's own pipeline, and what the research is about
	//   cudalod  external comparison, batch build         } vendored byte-identical,
	//   simlod   external comparison, progressive build   } see bench/check_vendored.sh
	//
	// RemoLOD may pull whatever it needs out of the two comparison pipelines, but never
	// changes them -- they are only worth having while they still reproduce their published
	// numbers against bench/reference/.
	m_registry.add([this] { return std::make_unique<FlatPipeline>(*m_cuda); });
	m_registry.add([this] {
		auto p = std::make_unique<RemolodPipeline>(*m_cuda);
		// Applied in the factory, not after construction, because a pipeline is rebuilt from
		// the factory on every switch -- setting it once at startup would be lost the first
		// time the user switched away and back.
		p->setAccumEnabled(!m_options.remolodNoAccum);
		return p;
	});
	m_registry.add([this] { return std::make_unique<CudalodPipeline>(*m_cuda); });
	m_registry.add([this] { return std::make_unique<SimlodPipeline>(*m_cuda); });
}

DeviceBudget App::computeBudget() const {
	DeviceBudget budget;
	budget.vramTotal = m_cuda->totalMemory();
	budget.vramFreeAtStartup = m_cuda->freeMemory();

	// Reserve headroom for GL framebuffers, the driver, and the desktop compositor,
	// then hand every pipeline the SAME number.
	//
	// This deliberately replaces two upstream land grabs: SimLOD takes 80% of
	// whatever happens to be free at startup, and CudaLOD takes a hardcoded slab.
	// Neither is a budget; the first also makes a run depend on what else was on the
	// GPU at the time, which is fatal for reproducible measurement.
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
	// Keep only the extensions we understand, so dropping a folder full of mixed
	// files does something sensible.
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
	m_status = std::format("loaded {} points in {:.2f}s ({:.0f} MP/s)",
	                       formatNumber(static_cast<double>(m_meta.numPoints)),
	                       elapsed,
	                       static_cast<double>(m_meta.numPoints) / 1e6 / elapsed);
	m_statusIsError = false;

	// Also to stdout, so a script can assert on the count without screen-scraping a
	// GUI. This is the sort of thing that makes the difference between a viewer and
	// something a benchmark can drive.
	printf("remobench: loaded %s points from %s in %.2fs (%.0f MP/s)\n",
	       formatNumber(static_cast<double>(m_meta.numPoints)).c_str(),
	       accepted.front().c_str(), elapsed,
	       static_cast<double>(m_meta.numPoints) / 1e6 / elapsed);
	fflush(stdout);
	return true;
}

bool App::activateCloud(std::string* err) {
	m_budget = computeBudget();

	// A new scene makes every retained sample non-comparable. Note this is NOT done on a
	// pipeline switch: there the cloud is unchanged, so keeping flat.render alongside
	// simlod.render is the whole point -- the control condition and the thing being
	// measured, on the same camera and the same pixel budget.
	m_profiler.clear();
	m_frameTimeStats.clear();

	if (!m_source->start(PointSource::Mode::Whole, m_budget.bytes, err)) {
		return false;
	}

	const std::string wanted =
		m_registry.activeId().empty() ? m_options.pipeline : m_registry.activeId();
	if (!m_registry.switchTo(wanted, m_source.get(), m_meta, m_budget, err)) {
		return false;
	}

	// Frame the new cloud.
	m_renderer.controls().frameBox(
		glm::dvec3(0.0, 0.0, 0.0),
		glm::dvec3(m_meta.boxSize[0], m_meta.boxSize[1], m_meta.boxSize[2]),
		m_renderer.camera().fovyRad());

	// Near/far scaled to the scene, or a large cloud is clipped away and a small one
	// z-fights.
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
			// The format infers its count from file size: 24-byte header, 16 bytes per
			// point. Showing it up front means the dropdown can be sanity-checked
			// against the reference without reading the file.
			if (entry.bytes > 24) entry.numPoints = (entry.bytes - 24) / 16;
		} else {
			// LAS/LAZ carry an exact count in the header, so read it -- 375 bytes per
			// file, and it makes the dropdown's counts authoritative rather than
			// inferred. A header that will not parse is the same thing as a file that
			// will not load, so report it here instead of at load time.
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

	// Supported first, then by name, so usable entries sit at the top of the dropdown.
	std::sort(out.begin(), out.end(),
	          [](const DatasetEntry& a, const DatasetEntry& b) {
		          if (a.supported != b.supported) return a.supported;
		          return a.label < b.label;
	          });
	return out;
}

void App::scanDatasets() {
	m_datasetsScanned = true;

	// REMOBENCH_DATA_DIR lets a scripted run or a different checkout point elsewhere.
	if (const char* env = std::getenv("REMOBENCH_DATA_DIR")) {
		if (*env) m_datasetDir = env;
	}

	m_datasets = scanDatasetDir(m_datasetDir);

	// Preselect whatever is currently loaded, so the combo shows the truth.
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
	// One presented frame of delay, so the status below is visible before ingest blocks.
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
	}

	// The active pipeline may have become unsupported for the new cloud (a big cloud can
	// price CudaLOD out of the budget). Fall back rather than leaving a pipeline selected
	// that cannot run.
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

	// Rescan so the combo's selection tracks what is actually loaded.
	m_datasetsScanned = false;
}

void App::applyPendingPipelineSwitch() {
	if (m_pendingPipeline.empty()) return;

	const std::string id = m_pendingPipeline;
	m_pendingPipeline.clear();
	if (id == m_registry.activeId()) return;

	// Exclusive residency: the outgoing pipeline frees its device memory before the
	// incoming one allocates, so both are offered the SAME budget and their memory
	// figures stay comparable.
	std::string err;
	if (!m_registry.switchTo(id, m_source.get(), m_meta, m_budget, &err)) {
		m_status = "could not switch to " + id + ": " + err;
		m_statusIsError = true;
		// Also to stderr: a refused switch is exactly the kind of thing a scripted run
		// needs to see, and the GUI status line is invisible to one.
		fprintf(stderr, "remobench: %s\n", m_status.c_str());
		return;
	}
	m_status = "switched to " + id;
	m_statusIsError = false;
}

int App::run() {
	// Applied here rather than inside drawGui: a switch destroys the active pipeline, and
	// drawGui holds a raw pointer to it across its whole body.
	auto update = [this] {
		// Scripted switch request, for --switch-to.
		if (!m_options.switchToPipeline.empty() &&
		    m_frameCounter == static_cast<uint64_t>(m_options.switchAfterFrames)) {
			m_pendingPipeline = m_options.switchToPipeline;
		}
		// Load before switching: a new cloud can make the active pipeline unsupported,
		// and applyPendingLoad queues the fallback.
		applyPendingLoad();
		applyPendingPipelineSwitch();
	};

	auto render = [this] {
		// One regime for the whole process, since --strict-timing is a startup option.
		// Kept per-frame anyway so the profiler files samples under the regime that
		// produced them rather than trusting the caller to be consistent.
		const Regime regime =
			m_options.strictTiming ? Regime::Strict : Regime::Deferred;
		m_profiler.beginFrame(m_frameCounter, regime, nullptr);
		m_frameTimeStats.add(m_renderer.frameMs());

		SharedUniforms uniforms = buildUniforms();

		// Freeze the LOD-selection transform when asked, so the cut can be locked
		// and inspected while flying the camera. Best debugging feature in either
		// upstream repo.
		if (m_settings.doUpdateVisibility || !m_hasFrozen) {
			m_frozen = uniforms;
			m_hasFrozen = true;
		}
		uniforms.transformFrozen = m_frozen.transform;
		uniforms.transformFrozenInv = m_frozen.transform;  // TODO: real inverse

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

				// Render BEFORE build, matching upstream's ordering: both are on the
				// null stream so they serialise anyway, and this way a frame shows
				// the structure as of the previous build step rather than stalling on
				// this one.
				pipeline->render(frame);
				pipeline->build(*m_source, frame);

				m_interop.unmap(nullptr);
			} else if (!interopErr.empty()) {
				m_status = interopErr;
				m_statusIsError = true;
			}
		}

		// Closed BEFORE the GUI, so that in the strict regime this frame's own samples
		// are the ones the panel shows rather than the previous frame's.
		m_profiler.endFrame();

		drawGui();
		++m_frameCounter;
	};

	while (m_renderer.runFrame(update, render)) {
		// Dump after the render callback has run for this frame, so what lands on
		// disk is what was drawn rather than a half-cleared target.
		if (!m_options.dumpFramePath.empty() &&
		    m_frameCounter >= static_cast<uint64_t>(m_options.dumpAfterFrames)) {
			const bool ok = dumpFrame(m_options.dumpFramePath);
			m_renderer.requestClose();
			if (!ok) return 1;
		}
	}
	return 0;
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
		// The presented frame, ImGui overlay included. ImGui draws to the default
		// framebuffer AFTER the point cloud is blitted there, so the FBO alone never
		// contains the UI -- which made the UI unverifiable without a human looking at
		// the window.
		glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
		// GL_FRONT because the dump runs after glfwSwapBuffers, so the frame just drawn
		// is the front buffer.
		//
		// This driver flags reading FRONT on a double-buffered context as
		// GL_INVALID_OPERATION while still returning the pixels correctly. Debug output
		// is muted across the read rather than draining glGetError() afterwards, because
		// the message comes from the debug CALLBACK, which fires when the error is
		// generated -- draining the queue later is too late to suppress it.
		const GLboolean debugWasOn = glIsEnabled(GL_DEBUG_OUTPUT);
		if (debugWasOn) glDisable(GL_DEBUG_OUTPUT);

		glReadBuffer(GL_FRONT);
		glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
		glReadBuffer(GL_BACK);
		while (glGetError() != GL_NO_ERROR) {
		}

		if (debugWasOn) glEnable(GL_DEBUG_OUTPUT);
	} else {
		// Just the rendered cloud: what a golden-image comparison should compare.
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
	// GL's origin is bottom-left; PPM's is top-left, so emit rows in reverse.
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

	// Dump the active pipeline's stats alongside the image, so a scripted run can assert
	// on structure and timing without screen-scraping the GUI. This is the readout the
	// benchmark harness will formalise.
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

		// Whatever this pipeline alone can report. RemoLOD's accumulator invariants come
		// through here, which is what makes them assertable from a script rather than only
		// visible in the panel.
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

		// Printed as "not measured" rather than 0.00 when no sample exists. The
		// distinction matters: in the deferred regime a render scope may legitimately
		// have nothing harvested yet, and the previous code's 0.00 was indistinguishable
		// from a kernel that genuinely cost nothing.
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
		if (m_profiler.droppedScopes() > 0) {
			printf("  WARNING %llu profiler sample(s) dropped\n",
			       static_cast<unsigned long long>(m_profiler.droppedScopes()));
		}

		// Every scope, not just the two aggregates. This is what a scripted run asserts
		// against -- bench/reference/README.md's per-strategy split and voxelize medians
		// are a table of exactly these numbers, so they double as an oracle for the
		// instrument itself.
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
		// Health flags invalidate a run: the structure was silently truncated.
		if (s.allocOverflow) printf("  WARNING allocator overflow\n");
		if (s.nodeCapacityReached) printf("  WARNING node pool exhausted\n");
		if (s.memCapacityReached) printf("  WARNING memory budget reached\n");
		fflush(stdout);
	}
	return true;
}

}  // namespace remo
