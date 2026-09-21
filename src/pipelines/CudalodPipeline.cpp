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

#include "../../kernels/cudalod/common.h"

namespace remo {
namespace {

constexpr double kBytesPerPointSlab = 128.0;

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

}

CudalodPipeline::CudalodPipeline(CudaContext& cuda) : m_cuda(cuda) {}
CudalodPipeline::~CudalodPipeline() { release(); }

PipelineInfo CudalodPipeline::info() const {
	PipelineInfo info;
	info.id = "cudalod";
	info.displayName = "CudaLOD (batch)";
	info.progressive = false;
	info.needsWholeCloudResident = true;
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
	if (meta.numPoints > 0xFFFFFFFFull) {
		if (err) {
			*err = "CudaLOD's device code indexes points with 32-bit types; " +
			       std::to_string(meta.numPoints) + " points does not fit";
		}
		return false;
	}

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
	scopes.build = {"cudalod.split", "cudalod.voxelize"};
	scopes.render = "cudalod.render";
	return scopes;
}

bool CudalodPipeline::build(PointSource& source, const FrameContext& frame) {
	if (!m_buildProgram || !m_buildProgram->ok()) return false;
	if (!m_slab) return false;

	if (!source.isFullyResident()) return true;

	if (m_built && !m_rebuildRequested) return false;
	m_rebuildRequested = false;

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

	REMO_CU(cuMemsetD8(m_allocOffset, 0, 8));
	REMO_CU(cuMemsetD8(m_numNodes, 0, 4));

	CUdeviceptr slab = m_slab, results = m_results, input = m_inputPoints;
	CUdeviceptr nodes = m_nodes, numNodes = m_numNodes, sorted = m_sorted;
	CUdeviceptr allocOffset = m_allocOffset;
	CUdeviceptr dbgPoints = m_debugPoints, dbgLines = m_debugLines;

	void* args[] = {&state, &slab,        &results,     &input,     &nodes,
	                &numNodes, &sorted,   &allocOffset, &dbgPoints, &dbgLines};

	const int gridSplit = m_cuda.gridForKernel(kernel2, m_blockSize);
	{
		GpuScope scope(frame.profiler, "cudalod.split");
		REMO_CU(cuLaunchCooperativeKernel(kernel2, static_cast<unsigned>(gridSplit), 1, 1,
		                                  static_cast<unsigned>(m_blockSize), 1, 1, 0, 0,
		                                  args));
	}

	{
		const CUresult sync = cuCtxSynchronize();
		if (isStickyError(sync)) {
			reportDeadContextAndExit(sync, "CudaLOD kernel2 (split / counting sort)");
		}
		REMO_CU(sync);
	}

	const int gridVoxelize = m_cuda.gridForKernel(kernel3, m_blockSize, 1);
	{
		GpuScope scope(frame.profiler, "cudalod.voxelize");
		REMO_CU(cuLaunchCooperativeKernel(kernel3, static_cast<unsigned>(gridVoxelize), 1,
		                                  1, static_cast<unsigned>(m_blockSize), 1, 1, 0,
		                                  0, args));
	}

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
	return false;
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

	if (m_stats.bytesHighWater > m_slabBytes) m_stats.allocOverflow = true;

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
	if (!m_built) return;

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
		m_stats.numVisibleNodes = d.drawItems;
		m_stats.numVisiblePoints = d.drawSamples;
	}
}

void CudalodPipeline::gui(const GpuProfiler& profiler) {
	ImGui::TextUnformatted(
		"Batch construction: the whole cloud is resident, then\n"
		"split (kernel2) and voxelise (kernel3) build the tree in one shot.");
	ImGui::Separator();

	ImGui::TextUnformatted("sampling strategy");
	const int previous = m_strategy;
	for (int s = 0; s <= 3; ++s) {
		if (ImGui::RadioButton(strategyName(s), m_strategy == s)) m_strategy = s;
	}
	if (m_strategy != previous) {
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

}
