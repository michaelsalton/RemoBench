#include "pipelines/FlatPipeline.h"

#include <imgui.h>

#include "remo/CudaCheck.h"
#include "remo/CudaContext.h"
#include "remo/GpuProfiler.h"
#include "remo/PointSource.h"
#include "shell/GuiWidgets.h"
#include "shell/TimingUi.h"

namespace remo {

namespace {
constexpr uint64_t kBytesPerPixel = 64;
constexpr uint64_t kMinScratchBytes = 64ull << 20;
}

FlatPipeline::FlatPipeline(CudaContext& cuda) : m_cuda(cuda) {}

FlatPipeline::~FlatPipeline() { release(); }

PipelineInfo FlatPipeline::info() const {
	PipelineInfo info;
	info.id = "flat";
	info.displayName = "Flat (no LOD)";
	info.progressive = false;
	info.needsWholeCloudResident = true;
	info.bytesPerPointEstimate = 16.0;
	return info;
}

bool FlatPipeline::initPrograms(std::string* err) {
	KernelProgramDesc desc;
	desc.modules = {"flat/flat_render.cu"};
	desc.kernels = {"kernel_render"};

	m_program = std::make_unique<CudaModularProgram>(std::move(desc));
	if (!m_program->ok()) {
		if (err) *err = m_program->lastError();
		return false;
	}
	return true;
}

bool FlatPipeline::allocate(const CloudMeta& meta, const DeviceBudget& budget,
                            std::string* err) {
	m_stats = PipelineStats{};
	m_stats.numPoints = meta.numPoints;

	if (!m_diagnostics) {
		if (REMO_CU(cuMemAlloc(&m_diagnostics, sizeof(DeviceDiagnostics))) !=
		    CUDA_SUCCESS) {
			if (err) *err = "cuMemAlloc failed for diagnostics";
			return false;
		}
	}
	REMO_CU(cuMemsetD8(m_diagnostics, 0, sizeof(DeviceDiagnostics)));

	m_stats.bytesCapacity = budget.bytes;
	return true;
}

void FlatPipeline::release() {
	if (m_scratch) {
		REMO_CU(cuMemFree(m_scratch));
		m_scratch = 0;
		m_scratchBytes = 0;
	}
	if (m_diagnostics) {
		REMO_CU(cuMemFree(m_diagnostics));
		m_diagnostics = 0;
	}
	m_points = 0;
	m_numPoints = 0;
}

void FlatPipeline::reset() {
	if (m_diagnostics) {
		REMO_CU(cuMemsetD8(m_diagnostics, 0, sizeof(DeviceDiagnostics)));
	}
	m_stats.allocOverflow = false;
}

TimingScopes FlatPipeline::timingScopes() const {
	TimingScopes scopes;
	scopes.render = "flat.render";
	return scopes;
}

bool FlatPipeline::build(PointSource& source, const FrameContext&) {
	if (!source.isFullyResident()) return true;
	if (m_points != 0) return false;

	m_points = source.residentPoints();
	m_numPoints = source.meta().numPoints;
	m_stats.numPoints = m_numPoints;
	m_stats.numPointsIngested = m_numPoints;
	m_stats.numVisiblePoints = m_numPoints;
	return false;
}

void FlatPipeline::render(const FrameContext& frame) {
	if (!m_program || !m_program->ok()) return;
	if (m_points == 0 || m_numPoints == 0) return;
	if (frame.targets.surface == 0) return;

	const uint64_t pixels = static_cast<uint64_t>(frame.targets.width) *
	                        static_cast<uint64_t>(frame.targets.height);
	const uint64_t needed =
		pixels * kBytesPerPixel < kMinScratchBytes ? kMinScratchBytes
		                                           : pixels * kBytesPerPixel;
	if (needed > m_scratchBytes) {
		if (m_scratch) REMO_CU(cuMemFree(m_scratch));
		m_scratch = 0;
		if (REMO_CU(cuMemAlloc(&m_scratch, needed)) != CUDA_SUCCESS) {
			m_scratchBytes = 0;
			return;
		}
		m_scratchBytes = needed;
	}
	m_stats.bytesAllocated = m_scratchBytes;

	CUfunction kernel = m_program->kernel("kernel_render");
	if (!kernel) return;

	RenderArgs args;
	args.uniforms = frame.uniforms;
	args.scratch = reinterpret_cast<uint32_t*>(m_scratch);
	args.scratchCapacity = m_scratchBytes;
	args.surface = frame.targets.surface;

	CUdeviceptr points = m_points;
	uint64_t numPoints = m_numPoints;
	CUdeviceptr diag = m_diagnostics;

	void* kernelArgs[] = {&args, &points, &numPoints, &diag};

	const int grid = m_cuda.gridForKernel(kernel, m_blockSize);

	{
		GpuScope scope(frame.profiler, "flat.render");
		REMO_CU(cuLaunchCooperativeKernel(kernel, static_cast<unsigned>(grid), 1, 1,
		                                  static_cast<unsigned>(m_blockSize), 1, 1, 0,
		                                  0, kernelArgs));
	}

	if (frame.strictTiming) {
		REMO_CU(cuCtxSynchronize());
	}

	DeviceDiagnostics diagnostics = {};
	if (cuMemcpyDtoH(&diagnostics, m_diagnostics, sizeof(DeviceDiagnostics)) ==
	    CUDA_SUCCESS) {
		m_stats.allocOverflow = diagnostics.allocOverflow != 0;
		m_stats.numVisibleNodes = diagnostics.drawItems;
		m_stats.numVisiblePoints = diagnostics.drawSamples;
		m_stats.bytesHighWater = diagnostics.allocHighWater;
	}
}

void FlatPipeline::guiControls() {
	paragraph(
		"No LOD: every point is rasterised every frame. This is the control "
		"condition and the image-quality ground truth that LOD pipelines are "
		"compared against.");

	if (m_program && m_program->isStale()) {
		ImGui::TextColored(ImVec4(1, 0.4f, 0.2f, 1),
		                   "kernel source changed but failed to compile;\n"
		                   "showing the previously loaded version");
	}
}

void FlatPipeline::guiStats(const GpuProfiler& profiler) {
	if (ImGui::BeginTable("flat_timing", 2, ImGuiTableFlags_SizingStretchProp)) {
		timingRow(profiler, "render (ms)", "flat.render");
		ImGui::EndTable();
	}

	ImGui::Text("render scratch: %.1f MB",
	            static_cast<double>(m_scratchBytes) / (1024.0 * 1024.0));
}

}
