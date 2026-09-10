#include "pipelines/CudalodPipeline.h"

#include <imgui.h>
#include <vector_functions.h>
#include <vector_types.h>

#include <algorithm>
#include <cstring>

#include "remo/CudaCheck.h"
#include "remo/CudaContext.h"
#include "remo/GpuProfiler.h"
#include "remo/PointSource.h"
#include "shell/TimingUi.h"

// The pipeline's own host/device contract, vendored unmodified. Deliberately NOT merged
// into remo/HostDeviceCommon.h: rewriting the struct the reference kernels read is how a
// port silently stops reproducing its published numbers.
#include "../../kernels/cudalod/common.h"

namespace remo {
namespace {

// Bytes of device slab per input point.
//
// Not a guess. Measured watermarks for 36,200,706 points (see bench/reference/README.md):
//   strategy 0 FIRST_COME             1.20 GB   ~33 B/pt
//   strategy 2 AVERAGE_SINGLECELL     3.82 GB  ~106 B/pt
//   strategy 3 WEIGHTED_NEIGHBORHOOD  3.82 GB  ~106 B/pt
//
// Sized for the WORST strategy, because switching strategy rebuilds and the device
// allocator has no bounds check -- so a slab that only fits strategy 0 corrupts memory
// the moment you press another button. Upstream's own rule of thumb (~28 B/pt) describes
// only the default strategy and is why its 4GB default appeared to work.
constexpr double kBytesPerPointSlab = 128.0;

// Phase 1 allocates count+node grids for every level 0..8 as scratch: ~200MB regardless
// of point count. A small cloud must still get that.
constexpr uint64_t kMinSlabBytes = 512ull << 20;

constexpr uint64_t kBytesPerPixelScratch = 64;
constexpr uint64_t kMinScratchBytes = 64ull << 20;

const char* strategyName(int s) {
	switch (s) {
		case FIRST_COME: return "0  FIRST_COME";
		case RANDOM: return "1  RANDOM";
		case AVERAGE_SINGLECELL: return "2  AVERAGE_SINGLECELL";
		case WEIGHTED_NEIGHBORHOOD: return "3  WEIGHTED_NEIGHBORHOOD";
		default: return "?";
	}
}

}  // namespace

CudalodPipeline::CudalodPipeline(CudaContext& cuda) : m_cuda(cuda) {}
CudalodPipeline::~CudalodPipeline() { release(); }

PipelineInfo CudalodPipeline::info() const {
	PipelineInfo info;
	info.id = "cudalod";
	info.displayName = "CudaLOD (batch)";
	info.progressive = false;
	info.needsWholeCloudResident = true;
	// Slab + the resident input points the source holds. On a 16GB card this is what
	// makes the 350M cloud correctly report as not fitting rather than crashing.
	info.bytesPerPointEstimate = kBytesPerPointSlab + 16.0;
	return info;
}

bool CudalodPipeline::initPrograms(std::string* err) {
	{
		KernelProgramDesc desc;
		desc.modules = {"cudalod/lib.cu", "cudalod/kernel.cu"};
		desc.kernels = {"kernel2", "kernel3"};
		m_buildProgram = std::make_unique<CudaModularProgram>(std::move(desc));
		if (!m_buildProgram->ok()) {
			if (err) *err = m_buildProgram->lastError();
			return false;
		}
	}
	{
		KernelProgramDesc desc;
		desc.modules = {"cudalod/lib.cu", "cudalod/cudalod_render.cu"};
		desc.kernels = {"kernel_render"};
		m_renderProgram = std::make_unique<CudaModularProgram>(std::move(desc));
		if (!m_renderProgram->ok()) {
			if (err) *err = m_renderProgram->lastError();
			return false;
		}
	}

	// Editing a construct kernel invalidates the tree it built, so rebuild on reload.
	// Upstream SimLOD does not do this and leaves a tree built by the previous version of
	// the code on screen, which is a confusing thing to debug.
	m_buildProgram->onCompile([this] { m_rebuildRequested = true; });

	return true;
}

bool CudalodPipeline::allocate(const CloudMeta& meta, const DeviceBudget& budget,
                               std::string* err) {
	release();

	m_stats = PipelineStats{};
	m_stats.bytesCapacity = budget.bytes;

	if (meta.numPoints == 0) {
		if (err) *err = "cloud is empty";
		return false;
	}
	// The device side is uint32_t/int throughout, so refuse rather than wrap.
	if (meta.numPoints > 0xFFFFFFFFull) {
		if (err) {
			*err = "CudaLOD's device code indexes points with 32-bit types; " +
			       std::to_string(meta.numPoints) + " points does not fit";
		}
		return false;
	}

	// Slab sized from the cloud, clamped to the budget minus what the resident input
	// points already cost (the source owns those, but they come out of the same VRAM).
	const uint64_t inputBytes = meta.numPoints * 16ull;
	uint64_t want = static_cast<uint64_t>(kBytesPerPointSlab *
	                                     static_cast<double>(meta.numPoints));
	want = std::max(want, kMinSlabBytes);

	const uint64_t available =
		budget.bytes > inputBytes ? budget.bytes - inputBytes : 0;
	if (available < kMinSlabBytes) {
		if (err) {
			*err = "not enough device memory for the LOD slab (" +
			       std::to_string(available / (1024 * 1024)) + " MB available)";
		}
		return false;
	}
	m_slabBytes = std::min(want, available);

	if (REMO_CU(cuMemAlloc(&m_slab, m_slabBytes)) != CUDA_SUCCESS) {
		if (err) {
			*err = "cuMemAlloc failed for a " +
			       std::to_string(m_slabBytes / (1024 * 1024)) + " MB LOD slab";
		}
		m_slabBytes = 0;
		return false;
	}

	auto allocCell = [&](CUdeviceptr* ptr, size_t bytes) {
		if (REMO_CU(cuMemAlloc(ptr, bytes)) != CUDA_SUCCESS) return false;
		REMO_CU(cuMemsetD8(*ptr, 0, bytes));
		return true;
	};

	if (!allocCell(&m_results, sizeof(Results)) ||
	    !allocCell(&m_numNodes, 4) || !allocCell(&m_nodes, 8) ||
	    !allocCell(&m_sorted, 8) || !allocCell(&m_allocOffset, 8) ||
	    !allocCell(&m_debugPoints, 8) || !allocCell(&m_debugLines, 8) ||
	    !allocCell(&m_diagnostics, sizeof(DeviceDiagnostics))) {
		if (err) *err = "cuMemAlloc failed for a CudaLOD indirection cell";
		release();
		return false;
	}

	m_stats.numPoints = meta.numPoints;
	m_stats.bytesAllocated = m_slabBytes;
	return true;
}

void CudalodPipeline::release() {
	for (CUdeviceptr* p : {&m_slab, &m_results, &m_scratch, &m_diagnostics,
	                       &m_numNodes, &m_nodes, &m_sorted, &m_allocOffset,
	                       &m_debugPoints, &m_debugLines}) {
		if (*p) {
			REMO_CU(cuMemFree(*p));
			*p = 0;
		}
	}
	m_slabBytes = 0;
	m_scratchBytes = 0;

	m_inputPoints = 0;
	m_numPoints = 0;
	m_built = false;
}

void CudalodPipeline::reset() {
	m_built = false;
	m_rebuildRequested = false;
	if (m_numNodes) REMO_CU(cuMemsetD8(m_numNodes, 0, 4));
	if (m_nodes) REMO_CU(cuMemsetD8(m_nodes, 0, 8));
	if (m_allocOffset) REMO_CU(cuMemsetD8(m_allocOffset, 0, 8));
	if (m_diagnostics) REMO_CU(cuMemsetD8(m_diagnostics, 0, sizeof(DeviceDiagnostics)));
	m_clearTimingRequested = true;
}

TimingScopes CudalodPipeline::timingScopes() const {
	TimingScopes scopes;
	// Kept as two scopes, not one total, because the paper reports them separately and
	// because they scale very differently across strategies: split is flat at ~5 ms
	// while voxelize spans 4 ms to 61 ms. A single "build" number would hide the only
	// thing the strategy choice actually changes.
	scopes.build = {"cudalod.split", "cudalod.voxelize"};
	scopes.render = "cudalod.render";
	return scopes;
}

bool CudalodPipeline::build(PointSource& source, const FrameContext& frame) {
	if (!m_buildProgram || !m_buildProgram->ok()) return false;
	if (!m_slab) return false;

	// Batch pipeline: nothing to do until the whole cloud is on the device.
	if (!source.isFullyResident()) return true;

	if (m_built && !m_rebuildRequested) return false;
	m_rebuildRequested = false;

	// Repeated rebuilds at a FIXED strategy accumulate into one distribution on purpose
	// -- press rebuild ten times and the median is worth more than any single run. Only
	// a change that makes the samples describe different work clears them.
	if (m_clearTimingRequested) {
		m_clearTimingRequested = false;
		if (frame.profiler) frame.profiler->clearPrefix("cudalod.");
	}

	m_inputPoints = source.residentPoints();
	m_numPoints = source.meta().numPoints;
	if (m_inputPoints == 0 || m_numPoints == 0) return false;

	CUfunction kernel2 = m_buildProgram->kernel("kernel2");
	CUfunction kernel3 = m_buildProgram->kernel("kernel3");
	if (!kernel2 || !kernel3) return false;

	const CloudMeta& meta = source.meta();

	State state = {};
	state.metadata.numPoints = static_cast<uint32_t>(m_numPoints);
	// Device-space bounds: the loader already translated so the minimum is the origin.
	state.metadata.min_x = 0.0f;
	state.metadata.min_y = 0.0f;
	state.metadata.min_z = 0.0f;
	state.metadata.max_x = meta.boxSize[0];
	state.metadata.max_y = meta.boxSize[1];
	state.metadata.max_z = meta.boxSize[2];
	state.imageSize = make_int2(frame.targets.width, frame.targets.height);
	state.strategy = static_cast<SamplingStrategy>(m_strategy);
	state.LOD = frame.uniforms.lodScale;
	std::memset(&state.transform, 0, sizeof(state.transform));

	// Reset the arena watermark before phase 1, or a rebuild appends to the previous
	// build's allocations and runs off the end of the slab.
	REMO_CU(cuMemsetD8(m_allocOffset, 0, 8));
	REMO_CU(cuMemsetD8(m_numNodes, 0, 4));

	CUdeviceptr slab = m_slab, results = m_results, input = m_inputPoints;
	CUdeviceptr nodes = m_nodes, numNodes = m_numNodes, sorted = m_sorted;
	CUdeviceptr allocOffset = m_allocOffset;
	CUdeviceptr dbgPoints = m_debugPoints, dbgLines = m_debugLines;

	void* args[] = {&state, &slab,        &results,     &input,     &nodes,
	                &numNodes, &sorted,   &allocOffset, &dbgPoints, &dbgLines};

	// Phase 1: split. Grid from occupancy.
	const int gridSplit = m_cuda.gridForKernel(kernel2, m_blockSize);
	{
		GpuScope scope(frame.profiler, "cudalod.split");
		REMO_CU(cuLaunchCooperativeKernel(kernel2, static_cast<unsigned>(gridSplit), 1, 1,
		                                  static_cast<unsigned>(m_blockSize), 1, 1, 0, 0,
		                                  args));
	}

	// Check for a dead context after EACH phase, not just at the end.
	//
	// Two reasons. It attributes the fault to a specific kernel, which a stack trace
	// cannot do for device code. And it closes a race: once the context dies, the CUDA
	// driver's teardown path corrupts the heap arenas that the file-watcher threads
	// allocate on, so the process can die of "heap corruption in an unrelated thread"
	// before a check placed only at the end is ever reached. That is precisely how this
	// fault first presented, and it sent the investigation in the wrong direction.
	{
		const CUresult sync = cuCtxSynchronize();
		if (isStickyError(sync)) {
			reportDeadContextAndExit(sync, "CudaLOD kernel2 (split / counting sort)");
		}
		REMO_CU(sync);
	}

	// Phase 2: voxelise. EXACTLY one block per SM -- the strategies allocate a
	// per-block sampling grid from the slab, so more blocks would overrun it. This is
	// upstream's constraint and it is load-bearing.
	const int gridVoxelize = m_cuda.gridForKernel(kernel3, m_blockSize, 1);
	{
		GpuScope scope(frame.profiler, "cudalod.voxelize");
		REMO_CU(cuLaunchCooperativeKernel(kernel3, static_cast<unsigned>(gridVoxelize), 1,
		                                  1, static_cast<unsigned>(m_blockSize), 1, 1, 0,
		                                  0, args));
	}

	// Synchronise and check for a dead context BEFORE touching any result. A device
	// fault here poisons everything downstream, so this is the only place it can be
	// reported usefully.
	{
		const CUresult sync = cuCtxSynchronize();
		if (isStickyError(sync)) {
			reportDeadContextAndExit(
				sync, "CudaLOD kernel3 (voxelize)");
		}
		REMO_CU(sync);
	}

	readResults();

	m_built = true;
	return false;  // construction is complete in one shot
}

void CudalodPipeline::readResults() {
	if (!m_results) return;

	Results r = {};
	if (REMO_CU(cuMemcpyDtoH(&r, m_results, sizeof(Results))) != CUDA_SUCCESS) return;

	m_stats.numPoints = r.points;
	m_stats.numVoxels = r.voxels;
	m_stats.numPointsIngested = m_numPoints;
	m_stats.numNodes = static_cast<uint32_t>(r.nodes);
	m_stats.numInner = static_cast<uint32_t>(r.innerNodes);
	m_stats.numLeaves = static_cast<uint32_t>(r.leafNodes);
	m_stats.maxPointsPerNode = static_cast<uint32_t>(r.maxPoints);

	m_allocatedSplitting = r.allocatedMemory_splitting;
	m_allocatedVoxelization = r.allocatedMemory_voxelization;
	m_stats.bytesHighWater =
		std::max(r.allocatedMemory_splitting, r.allocatedMemory_voxelization);
	m_stats.bytesAllocated = m_slabBytes;

	// The device allocator has no bounds check, so the watermark exceeding the slab is
	// the only signal that it wrote outside it. Treat that as invalidating the run.
	if (m_stats.bytesHighWater > m_slabBytes) m_stats.allocOverflow = true;

	// MAX_NODES is a hard cap in methods_common.h.cu, also unchecked on device.
	constexpr uint32_t kMaxNodes = 200'000;
	if (m_stats.numNodes >= kMaxNodes) m_stats.nodeCapacityReached = true;

	for (int i = 0; i < 24 && i < 50; ++i) {
		m_stats.samplesPerLevel[i] = r.pointsPerLevel[i] + r.voxelsPerLevel[i];
	}
}

void CudalodPipeline::ensureScratch(int width, int height) {
	const uint64_t pixels =
		static_cast<uint64_t>(width) * static_cast<uint64_t>(height);
	uint64_t needed = pixels * kBytesPerPixelScratch;
	needed = std::max(needed, kMinScratchBytes);
	// The render kernel also allocates a MAX_NODES visibility byte array.
	needed += 256ull * 1024ull;
	if (needed <= m_scratchBytes) return;

	if (m_scratch) REMO_CU(cuMemFree(m_scratch));
	m_scratch = 0;
	if (REMO_CU(cuMemAlloc(&m_scratch, needed)) != CUDA_SUCCESS) {
		m_scratchBytes = 0;
		return;
	}
	m_scratchBytes = needed;
}

void CudalodPipeline::render(const FrameContext& frame) {
	if (!m_renderProgram || !m_renderProgram->ok()) return;
	if (frame.targets.surface == 0) return;
	if (!m_built) return;  // nothing to draw yet; the shell clears the target

	ensureScratch(frame.targets.width, frame.targets.height);
	if (!m_scratch) return;

	CUfunction kernel = m_renderProgram->kernel("kernel_render");
	if (!kernel) return;

	RenderArgs args;
	args.uniforms = frame.uniforms;
	args.scratch = reinterpret_cast<uint32_t*>(m_scratch);
	args.scratchCapacity = m_scratchBytes;
	args.surface = frame.targets.surface;

	CUdeviceptr nodes = m_nodes, numNodes = m_numNodes, diag = m_diagnostics;
	void* kernelArgs[] = {&args, &nodes, &numNodes, &diag};

	const int grid = m_cuda.gridForKernel(kernel, m_blockSize);
	{
		// As with SimLOD, this launch previously had no events on either side, so
		// CudaLOD's render time was never measured at all.
		GpuScope scope(frame.profiler, "cudalod.render");
		REMO_CU(cuLaunchCooperativeKernel(kernel, static_cast<unsigned>(grid), 1, 1,
		                                  static_cast<unsigned>(m_blockSize), 1, 1, 0, 0,
		                                  kernelArgs));
	}

	if (frame.strictTiming) {
		const CUresult sync = cuCtxSynchronize();
		if (isStickyError(sync)) {
			reportDeadContextAndExit(sync, "CudaLOD render (kernel_render)");
		}
		REMO_CU(sync);
	}

	DeviceDiagnostics d = {};
	if (cuMemcpyDtoH(&d, m_diagnostics, sizeof(DeviceDiagnostics)) == CUDA_SUCCESS) {
		if (d.allocOverflow) m_stats.allocOverflow = true;
		// Our render kernel owns selection, so these come from it.
		m_stats.numVisibleNodes = d.drawItems;
		m_stats.numVisiblePoints = d.drawSamples;
	}
}

void CudalodPipeline::gui(const GpuProfiler& profiler) {
	ImGui::TextUnformatted(
		"Batch construction: the whole cloud is resident, then\n"
		"split (kernel2) and voxelise (kernel3) build the tree in one shot.");
	ImGui::Separator();

	// The sampling strategy is genuinely pipeline-owned, so it lives here rather than in
	// the shared panel. Upstream puts these four buttons in its RENDERER
	// (Renderer.cpp:546-609), which is what has to be undone to make anything swappable.
	ImGui::TextUnformatted("sampling strategy");
	const int previous = m_strategy;
	for (int s = 0; s <= 3; ++s) {
		if (ImGui::RadioButton(strategyName(s), m_strategy == s)) m_strategy = s;
	}
	if (m_strategy != previous) {
		// Changing strategy rebuilds the whole tree. Worth knowing that it is not free
		// and not equivalent: WEIGHTED_NEIGHBORHOOD voxelises ~13x slower than
		// FIRST_COME for bit-identical tree structure. That is also why the timing
		// samples are dropped -- a median pooled across two strategies describes neither.
		m_rebuildRequested = true;
		m_clearTimingRequested = true;
	}

	ImGui::Separator();
	if (ImGui::BeginTable("cudalod_timing", 2, ImGuiTableFlags_SizingStretchProp)) {
		auto row = [](const char* label, const char* fmt, double v) {
			ImGui::TableNextRow();
			ImGui::TableNextColumn();
			ImGui::TextUnformatted(label);
			ImGui::TableNextColumn();
			ImGui::Text(fmt, v);
		};
		timingRow(profiler, "split (ms)", "cudalod.split");
		timingRow(profiler, "voxelize (ms)", "cudalod.voxelize");
		timingRow(profiler, "render (ms)", "cudalod.render");

		// Throughput derives from the MEDIAN of one build, not the sum over rebuilds:
		// the cloud is voxelised once per build, so summing ten rebuilds would divide
		// 36M points by ten builds' worth of time. SimLOD's row is the opposite case and
		// correctly uses the total, since there the launches partition one ingest.
		const ScopeStats* split = profiler.find("cudalod.split");
		const ScopeStats* voxelize = profiler.find("cudalod.voxelize");
		const double buildMedian = (split ? split->median() : 0.0) +
		                           (voxelize ? voxelize->median() : 0.0);
		row("build (median split+voxelize)", "%.2f ms", buildMedian);
		const double mps =
			buildMedian > 0.0 ? double(m_numPoints) / 1e6 / (buildMedian / 1000.0) : 0.0;
		row("throughput (median build)", "%.0f MP/s", mps);
		row("slab", "%.2f GB", double(m_slabBytes) / 1e9);
		row("watermark split", "%.2f GB", double(m_allocatedSplitting) / 1e9);
		row("watermark voxelize", "%.2f GB", double(m_allocatedVoxelization) / 1e9);
		ImGui::EndTable();
	}

	if (ImGui::Button("rebuild")) m_rebuildRequested = true;

	if (m_renderProgram && m_renderProgram->isStale()) {
		ImGui::TextColored(ImVec4(1, 0.4f, 0.2f, 1),
		                   "render kernel failed to recompile;\n"
		                   "showing the previously loaded version");
	}
}

}  // namespace remo
