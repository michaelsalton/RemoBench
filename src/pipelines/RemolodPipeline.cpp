#include "pipelines/RemolodPipeline.h"

#include <imgui.h>
#include <vector_functions.h>
#include <vector_types.h>

#include <algorithm>
#include <cstring>

#include "remo/CudaCheck.h"
#include "remo/CudaContext.h"
#include "remo/GpuProfiler.h"
#include "remo/PointSource.h"
#include "remo/unsuck.hpp"
#include "shell/TimingUi.h"

#include "../../kernels/simlod/HostDeviceInterface.h"
#include "../../kernels/remolod/remolod_layout.h"

namespace remo {
namespace {

constexpr uint64_t kMomentaryBytes = 512ull << 20;

constexpr uint32_t kMaxNodes = remolod::kMaxNodes;

constexpr double kBytesPerPointPersistent = 48.0;
constexpr uint64_t kMinPersistentBytes = 512ull << 20;

constexpr uint64_t kBytesPerPixelScratch = 64;
constexpr uint64_t kMinScratchBytes = 64ull << 20;

}

RemolodPipeline::RemolodPipeline(CudaContext& cuda) : m_cuda(cuda) {}
RemolodPipeline::~RemolodPipeline() { release(); }

PipelineInfo RemolodPipeline::info() const {
	PipelineInfo info;
	info.id = "remolod";
	info.displayName = "RemoLOD (detail-aware)";
	info.progressive = true;
	info.needsWholeCloudResident = true;
	info.bytesPerPointEstimate = kBytesPerPointPersistent + 16.0;
	return info;
}

bool RemolodPipeline::initPrograms(std::string* err) {
	struct Spec {
		std::unique_ptr<CudaModularProgram>* target;
		const char* module;
		std::vector<std::string> kernels;
	};
	const Spec specs[] = {
		{&m_resetProgram, "remolod/remolod_reset.cu", {"kernel"}},
		{&m_constructProgram, "remolod/remolod_octree.cu", {"kernel_construct"}},
		{&m_accumProgram, "remolod/remolod_accum.cu",
		 {"kernel_accumulate", "kernel_accum_verify"}},
		{&m_renderProgram, "remolod/remolod_render.cu", {"kernel_render"}},
	};

	for (const Spec& spec : specs) {
		KernelProgramDesc desc;
		desc.modules = {spec.module};
		desc.kernels = spec.kernels;
		*spec.target = std::make_unique<CudaModularProgram>(std::move(desc));
		if (!(*spec.target)->ok()) {
			if (err) *err = (*spec.target)->lastError();
			return false;
		}
	}

	m_constructProgram->onCompile([this] { m_needsReset = true; });
	m_accumProgram->onCompile([this] { m_needsReset = true; });

	return true;
}

bool RemolodPipeline::allocate(const CloudMeta& meta, const DeviceBudget& budget,
                               std::string* err) {
	release();

	m_stats = PipelineStats{};
	m_stats.bytesCapacity = budget.bytes;
	m_numPoints = meta.numPoints;

	if (meta.numPoints == 0) {
		if (err) *err = "cloud is empty";
		return false;
	}

	const uint64_t inputBytes = meta.numPoints * 16ull;
	const uint64_t available =
		budget.bytes > inputBytes ? budget.bytes - inputBytes : 0;

	m_nodesBytes = static_cast<uint64_t>(kMaxNodes) * remolod::kNodeBytes;
	m_nodeAccumsBytes = static_cast<uint64_t>(kMaxNodes) * sizeof(NodeAccum);
	m_momentaryBytes = kMomentaryBytes;

	uint64_t wantPersistent = static_cast<uint64_t>(
		kBytesPerPointPersistent * static_cast<double>(meta.numPoints));
	wantPersistent = std::max(wantPersistent, kMinPersistentBytes);

	const uint64_t fixed = m_nodesBytes + m_momentaryBytes + m_nodeAccumsBytes;
	if (available < fixed + kMinPersistentBytes) {
		if (err) {
			*err = "not enough device memory: needs at least " +
			       std::to_string((fixed + kMinPersistentBytes) / (1024 * 1024)) +
			       " MB, " + std::to_string(available / (1024 * 1024)) + " MB available";
		}
		return false;
	}
	m_persistentBytes = std::min(wantPersistent, available - fixed);

	auto alloc = [&](CUdeviceptr* ptr, uint64_t bytes, const char* what) {
		if (REMO_CU(cuMemAlloc(ptr, bytes)) != CUDA_SUCCESS) {
			if (err) {
				*err = std::string("cuMemAlloc failed for ") + what + " (" +
				       std::to_string(bytes / (1024 * 1024)) + " MB)";
			}
			return false;
		}
		REMO_CU(cuMemsetD8(*ptr, 0, bytes));
		return true;
	};

	if (!alloc(&m_momentary, m_momentaryBytes, "the momentary buffer") ||
	    !alloc(&m_persistent, m_persistentBytes, "the persistent octree store") ||
	    !alloc(&m_nodes, m_nodesBytes, "the node pool") ||
	    !alloc(&m_nodeAccums, m_nodeAccumsBytes, "the accumulator side array") ||
	    !alloc(&m_accumGlobals, sizeof(AccumGlobals), "the accumulator globals") ||
	    !alloc(&m_statsBuffer, sizeof(Stats), "stats") ||
	    !alloc(&m_frameStart, 8, "the frame timestamp") ||
	    !alloc(&m_cudaPrint, 1024, "the CudaPrint buffer") ||
	    !alloc(&m_diagnostics, sizeof(DeviceDiagnostics), "diagnostics") ||
	    !alloc(&m_resetBatchSizes, uint64_t(remolod::kBatchStreamSize) * 4,
	           "reset's batchSizes scratch") ||
	    !alloc(&m_resetNumUploaded, 4, "reset's numBatchesUploaded scratch")) {
		release();
		return false;
	}

	m_stats.numPoints = 0;
	m_stats.bytesAllocated =
		m_persistentBytes + m_momentaryBytes + m_nodesBytes + m_nodeAccumsBytes;
	m_needsReset = true;
	m_complete = false;
	m_accum = AccumGlobals{};
	return true;
}

void RemolodPipeline::release() {
	for (CUdeviceptr* p : {&m_momentary, &m_persistent, &m_nodes, &m_nodeAccums,
	                       &m_accumGlobals, &m_statsBuffer, &m_frameStart, &m_cudaPrint,
	                       &m_scratch, &m_diagnostics, &m_resetBatchSizes,
	                       &m_resetNumUploaded}) {
		if (*p) {
			REMO_CU(cuMemFree(*p));
			*p = 0;
		}
	}
	m_momentaryBytes = m_persistentBytes = m_nodesBytes = m_scratchBytes = 0;
	m_nodeAccumsBytes = 0;

	m_complete = false;
	m_needsReset = true;
}

void RemolodPipeline::reset() {
	m_needsReset = true;
	m_complete = false;
	m_batchesConsumed = 0;
}

TimingScopes RemolodPipeline::timingScopes() const {
	TimingScopes scopes;
	scopes.build = {"remolod.reset", "remolod.construct", "remolod.accumulate"};
	scopes.render = "remolod.render";
	return scopes;
}

void RemolodPipeline::fillUniforms(const FrameContext& frame, void* out) const {
	Uniforms* u = static_cast<Uniforms*>(out);
	std::memset(u, 0, sizeof(Uniforms));

	u->boxMin = make_float3(frame.uniforms.boxMin.x, frame.uniforms.boxMin.y,
	                        frame.uniforms.boxMin.z);
	u->boxMax = make_float3(frame.uniforms.boxMax.x, frame.uniforms.boxMax.y,
	                        frame.uniforms.boxMax.z);
	u->frameCounter = frame.uniforms.frameCounter;
	u->persistentBufferCapacity = m_persistentBytes;
	u->momentaryBufferCapacity = m_momentaryBytes;
	u->width = frame.uniforms.width;
	u->height = frame.uniforms.height;
}

void RemolodPipeline::runAccumulator(const FrameContext& frame) {
	if (!m_accumEnabled) return;
	if (!m_accumProgram || !m_accumProgram->ok()) return;
	if (!m_nodeAccums || !m_accumGlobals) return;

	CUfunction fold = m_accumProgram->kernel("kernel_accumulate");
	CUfunction verify = m_accumProgram->kernel("kernel_accum_verify");
	if (!fold || !verify) return;

	AccumGlobals g = m_accum;
	g.sumLeafCounts = 0;
	g.innerWithSums = 0;
	g.pointsFolded = 0;
	g.numLeavesFolded = 0;
	REMO_CU(cuMemcpyHtoD(m_accumGlobals, &g, sizeof(g)));

	const SharedUniforms& u = frame.uniforms;
	const float octreeSize =
		std::max({u.boxMax.x - u.boxMin.x, u.boxMax.y - u.boxMin.y,
		          u.boxMax.z - u.boxMin.z});

	AccumArgs args = {};
	args.nodes = reinterpret_cast<void*>(m_nodes);
	args.accums = reinterpret_cast<void*>(m_nodeAccums);
	args.globals = reinterpret_cast<void*>(m_accumGlobals);
	args.numNodes = m_stats.numNodes;
	args.batchIndex = m_batchesConsumed;
	args.octreeMinX = u.boxMin.x;
	args.octreeMinY = u.boxMin.y;
	args.octreeMinZ = u.boxMin.z;
	args.octreeSize = octreeSize;

	void* kernelArgs[] = {&args};

	const int grid = m_cuda.gridForKernel(fold, kAccumBlockSize);
	{
		GpuScope scope(frame.profiler, "remolod.accumulate");
		REMO_CU(cuLaunchKernel(fold, static_cast<unsigned>(grid), 1, 1,
		                       static_cast<unsigned>(kAccumBlockSize), 1, 1, 0, 0,
		                       kernelArgs, nullptr));
		REMO_CU(cuLaunchKernel(verify, static_cast<unsigned>(grid), 1, 1,
		                       static_cast<unsigned>(kAccumBlockSize), 1, 1, 0, 0,
		                       kernelArgs, nullptr));
	}

	const CUresult sync = cuCtxSynchronize();
	if (isStickyError(sync)) {
		reportDeadContextAndExit(sync, "RemoLOD kernel_accumulate");
	}
	REMO_CU(sync);

	REMO_CU(cuMemcpyDtoH(&m_accum, m_accumGlobals, sizeof(m_accum)));
}

bool RemolodPipeline::build(PointSource& source, const FrameContext& frame) {
	if (!m_constructProgram || !m_constructProgram->ok()) return false;
	if (!m_resetProgram || !m_resetProgram->ok()) return false;
	if (!m_momentary || !m_persistent || !m_nodes) return false;

	const BatchView view = source.view();
	if (view.slots == 0 || view.batchSizes == 0 || view.numBatchesUploaded == 0) {
		return true;
	}
	m_batchesTotal = view.numBatchesTotal;

	Uniforms uniforms;
	fillUniforms(frame, &uniforms);

	CUdeviceptr persistent = m_persistent, nodes = m_nodes;
	CUdeviceptr statsPtr = m_statsBuffer;
	CUdeviceptr frameStart = m_frameStart, cudaPrint = m_cudaPrint;
	CUdeviceptr batchSizes = view.batchSizes;
	CUdeviceptr numUploaded = view.numBatchesUploaded;

	if (m_needsReset) {
		CUfunction resetKernel = m_resetProgram->kernel("kernel");
		if (!resetKernel) return false;

		CUdeviceptr resetSizes = m_resetBatchSizes;
		CUdeviceptr resetUploaded = m_resetNumUploaded;
		void* resetArgs[] = {&uniforms,  &persistent,    &nodes,     &statsPtr,
		                     &cudaPrint, &resetUploaded, &resetSizes};

		if (frame.profiler) frame.profiler->clearPrefix("remolod.");

		{
			GpuScope scope(frame.profiler, "remolod.reset");
			REMO_CU(cuLaunchCooperativeKernel(resetKernel, 1, 1, 1, 1, 1, 1, 0, 0,
			                                  resetArgs));
		}

		const CUresult sync = cuCtxSynchronize();
		if (isStickyError(sync)) {
			reportDeadContextAndExit(sync, "RemoLOD reset kernel");
		}
		REMO_CU(sync);

		if (m_nodeAccums) REMO_CU(cuMemsetD8(m_nodeAccums, 0, m_nodeAccumsBytes));
		if (m_accumGlobals) REMO_CU(cuMemsetD8(m_accumGlobals, 0, sizeof(AccumGlobals)));
		m_accum = AccumGlobals{};

		m_needsReset = false;
		m_complete = false;
		m_batchesConsumed = 0;
	}

	if (m_complete) return false;

	CUfunction construct = m_constructProgram->kernel("kernel_construct");
	if (!construct) return false;

	const uint64_t nowNs = static_cast<uint64_t>(now() * 1e9);
	REMO_CU(cuMemcpyHtoD(m_frameStart, &nowNs, sizeof(nowNs)));

	CUdeviceptr points = view.slots;
	CUdeviceptr momentary = m_momentary;

	void* args[] = {&uniforms, &points,     &momentary, &persistent,  &nodes,
	                &statsPtr, &frameStart, &cudaPrint, &numUploaded, &batchSizes};

	const int grid = m_cuda.gridForKernel(construct, m_blockSize, 1);

	{
		GpuScope scope(frame.profiler, "remolod.construct");
		REMO_CU(cuLaunchCooperativeKernel(construct, static_cast<unsigned>(grid), 1, 1,
		                                  static_cast<unsigned>(m_blockSize), 1, 1, 0, 0,
		                                  args));
	}

	const CUresult sync = cuCtxSynchronize();
	if (isStickyError(sync)) {
		reportDeadContextAndExit(sync, "RemoLOD kernel_construct");
	}
	REMO_CU(sync);

	readStats();

	runAccumulator(frame);

	if (m_batchesTotal > 0 && m_batchesConsumed >= m_batchesTotal) {
		m_complete = true;
		return false;
	}
	return true;
}

void RemolodPipeline::readStats() {
	if (!m_statsBuffer) return;

	Stats s = {};
	if (REMO_CU(cuMemcpyDtoH(&s, m_statsBuffer, sizeof(Stats))) != CUDA_SUCCESS) return;

	m_batchesConsumed = s.batchletIndex;

	m_stats.numPoints = s.numPoints;
	m_stats.numVoxels = s.numVoxels;
	m_stats.numPointsIngested = s.numPointsProcessed;
	m_stats.numNodes = s.numNodes;
	m_stats.numInner = s.numInner;
	m_stats.numLeaves = s.numLeaves;
	m_stats.numVisibleNodes = s.numVisibleNodes;
	m_stats.numVisiblePoints = s.numVisiblePoints;
	m_stats.numVisibleVoxels = s.numVisibleVoxels;

	m_stats.bytesHighWater = s.allocatedBytes_persistent;
	m_stats.bytesAllocated =
		m_persistentBytes + m_momentaryBytes + m_nodesBytes + m_nodeAccumsBytes;
	m_stats.memCapacityReached = s.memCapacityReached;

	if (m_stats.numNodes >= kMaxNodes) m_stats.nodeCapacityReached = true;
}

void RemolodPipeline::ensureScratch(int width, int height) {
	const uint64_t pixels =
		static_cast<uint64_t>(width) * static_cast<uint64_t>(height);
	uint64_t needed = std::max(pixels * kBytesPerPixelScratch, kMinScratchBytes);
	needed += 16ull << 20;
	if (needed <= m_scratchBytes) return;

	if (m_scratch) REMO_CU(cuMemFree(m_scratch));
	m_scratch = 0;
	if (REMO_CU(cuMemAlloc(&m_scratch, needed)) != CUDA_SUCCESS) {
		m_scratchBytes = 0;
		return;
	}
	m_scratchBytes = needed;
}

void RemolodPipeline::render(const FrameContext& frame) {
	if (!m_renderProgram || !m_renderProgram->ok()) return;
	if (frame.targets.surface == 0) return;
	if (!m_nodes || m_needsReset) return;

	ensureScratch(frame.targets.width, frame.targets.height);
	if (!m_scratch) return;

	CUfunction kernel = m_renderProgram->kernel("kernel_render");
	if (!kernel) return;

	RenderArgs args;
	args.uniforms = frame.uniforms;
	args.scratch = reinterpret_cast<uint32_t*>(m_scratch);
	args.scratchCapacity = m_scratchBytes;
	args.surface = frame.targets.surface;

	CUdeviceptr nodes = m_nodes, statsPtr = m_statsBuffer, diag = m_diagnostics;
	void* kernelArgs[] = {&args, &nodes, &statsPtr, &diag};

	const int grid = m_cuda.gridForKernel(kernel, m_blockSize);
	{
		GpuScope scope(frame.profiler, "remolod.render");
		REMO_CU(cuLaunchCooperativeKernel(kernel, static_cast<unsigned>(grid), 1, 1,
		                                  static_cast<unsigned>(m_blockSize), 1, 1, 0, 0,
		                                  kernelArgs));
	}

	if (frame.strictTiming) {
		const CUresult sync = cuCtxSynchronize();
		if (isStickyError(sync)) {
			reportDeadContextAndExit(sync, "RemoLOD kernel_render");
		}
		REMO_CU(sync);
	}

	DeviceDiagnostics d = {};
	if (cuMemcpyDtoH(&d, m_diagnostics, sizeof(DeviceDiagnostics)) == CUDA_SUCCESS) {
		if (d.allocOverflow) m_stats.allocOverflow = true;
		m_stats.numVisibleNodes = d.drawItems;
		m_stats.numVisiblePoints = d.drawSamples;
	}
}

std::vector<std::string> RemolodPipeline::diagnostics() const {
	std::vector<std::string> out;
	char buf[128];

	if (!m_accumEnabled) {
		out.push_back("accumulator\toff (--remolod-no-accum)");
		return out;
	}

	const bool countsAgree = m_accum.sumLeafCounts == m_stats.numPoints;
	snprintf(buf, sizeof(buf), "%llu / %llu  %s",
	         static_cast<unsigned long long>(m_accum.sumLeafCounts),
	         static_cast<unsigned long long>(m_stats.numPoints),
	         countsAgree ? "ok" : "MISMATCH");
	out.push_back(std::string("accum sum/points\t") + buf);

	snprintf(buf, sizeof(buf), "%u  %s", m_accum.innerWithSums,
	         m_accum.innerWithSums == 0 ? "ok" : "MISMATCH");
	out.push_back(std::string("accum inner w/ sums\t") + buf);

	snprintf(buf, sizeof(buf), "%llu",
	         static_cast<unsigned long long>(m_accum.numAccumulated));
	out.push_back(std::string("accum folded total\t") + buf);

	snprintf(buf, sizeof(buf), "0x%llx",
	         static_cast<unsigned long long>(m_accum.mortonWatermark));
	out.push_back(std::string("accum watermark\t") + buf);

	return out;
}

void RemolodPipeline::gui(const GpuProfiler& profiler) {
	ImGui::TextUnformatted(
		"RemoBench's own pipeline. Progressive construction forked from\n"
		"SimLOD, plus the per-node accumulator the detail-aware work needs.");
	ImGui::Separator();

	if (m_batchesTotal > 0) {
		const float progress =
			static_cast<float>(m_batchesConsumed) / static_cast<float>(m_batchesTotal);
		char overlay[64];
		snprintf(overlay, sizeof(overlay), "%u / %u batches", m_batchesConsumed,
		         m_batchesTotal);
		ImGui::ProgressBar(progress, ImVec2(-1.0f, 0.0f), overlay);
	}

	ImGui::Checkbox("accumulate (per-node sums)", &m_accumEnabled);
	ImGui::SameLine();
	ImGui::TextDisabled("(?)");
	if (ImGui::IsItemHovered()) {
		ImGui::SetTooltip(
			"Off leaves the tree identical -- the pass mutates nothing.\n"
			"That is the acceptance test: --dump-frame must be byte-identical\n"
			"with it on and off. Also reachable as --remolod-no-accum.");
	}

	if (ImGui::BeginTable("remolod_timing", 2, ImGuiTableFlags_SizingStretchProp)) {
		auto row = [](const char* label, const char* fmt, double v) {
			ImGui::TableNextRow();
			ImGui::TableNextColumn();
			ImGui::TextUnformatted(label);
			ImGui::TableNextColumn();
			ImGui::Text(fmt, v);
		};

		timingRow(profiler, "reset (ms)", "remolod.reset");
		timingRow(profiler, "construct (ms)", "remolod.construct");
		timingRow(profiler, "accumulate (ms)", "remolod.accumulate");
		timingRow(profiler, "render (ms)", "remolod.render");

		const BuildTotals totals = buildTotals(profiler, timingScopes());
		row("build total", "%.2f ms", totals.ms);
		row("launches", "%.0f", static_cast<double>(totals.launches));

		const double mps = totals.ms > 0.0
		                       ? double(m_stats.numPointsIngested) / 1e6 /
		                             (totals.ms / 1000.0)
		                       : 0.0;
		row("throughput (build total)", "%.0f MP/s", mps);
		row("persistent", "%.2f GB", double(m_persistentBytes) / 1e9);
		row("accumulators", "%.1f MB", double(m_nodeAccumsBytes) / 1e6);
		row("high water", "%.2f GB", double(m_stats.bytesHighWater) / 1e9);
		ImGui::EndTable();
	}

	ImGui::Separator();
	ImGui::TextUnformatted("accumulator invariants");

	if (ImGui::BeginTable("remolod_accum", 2, ImGuiTableFlags_SizingStretchProp)) {
		auto srow = [](const char* label, const char* value, bool bad) {
			ImGui::TableNextRow();
			ImGui::TableNextColumn();
			ImGui::TextUnformatted(label);
			ImGui::TableNextColumn();
			if (bad) {
				ImGui::TextColored(ImVec4(1, 0.4f, 0.2f, 1), "%s", value);
			} else {
				ImGui::TextUnformatted(value);
			}
		};
		char buf[96];

		const bool countsAgree = m_accum.sumLeafCounts == m_stats.numPoints;
		snprintf(buf, sizeof(buf), "%llu / %llu",
		         static_cast<unsigned long long>(m_accum.sumLeafCounts),
		         static_cast<unsigned long long>(m_stats.numPoints));
		srow("sum(leaf counts) / points", buf, !countsAgree);

		snprintf(buf, sizeof(buf), "%u", m_accum.innerWithSums);
		srow("inner nodes holding sums", buf, m_accum.innerWithSums != 0);

		snprintf(buf, sizeof(buf), "%u leaves, %u points",
		         m_accum.numLeavesFolded, m_accum.pointsFolded);
		srow("folded (last launch)", buf, false);

		snprintf(buf, sizeof(buf), "0x%llx",
		         static_cast<unsigned long long>(m_accum.mortonWatermark));
		srow("morton watermark", buf, false);
		ImGui::EndTable();
	}

	ImGui::TextDisabled(
		"the watermark is maintained but is NOT a closure oracle yet --\n"
		"no reader sorts points into Morton order");

	if (ImGui::Button("rebuild")) m_needsReset = true;

	ImGui::TextDisabled(
		"ingest is the resident path, not the paper's overlapped\n"
		"streaming loader -- construction is progressive, loading is not");

	if (m_renderProgram && m_renderProgram->isStale()) {
		ImGui::TextColored(ImVec4(1, 0.4f, 0.2f, 1),
		                   "render kernel failed to recompile;\n"
		                   "showing the previously loaded version");
	}
}

}
