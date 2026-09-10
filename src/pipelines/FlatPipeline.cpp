#include "pipelines/FlatPipeline.h"

#include <imgui.h>

#include "remo/CudaCheck.h"
#include "remo/CudaContext.h"
#include "remo/GpuProfiler.h"
#include "remo/PointSource.h"
#include "shell/TimingUi.h"

namespace remo {
	
namespace {
// Render scratch: 8 bytes per pixel for the packed framebuffer, plus slack for the
// accumulation targets a high-quality path will want. Derived from the viewport
// rather than being a magic constant -- CudaLOD's fixed 100MB render buffer is
// exactly this decision made badly: renderHQS wants 28 bytes/pixel, which fits at
// 1920x1080 and overruns at 3840x1600.
constexpr uint64_t kBytesPerPixel = 64;
constexpr uint64_t kMinScratchBytes = 64ull << 20;
}  // namespace

FlatPipeline::FlatPipeline(CudaContext& cuda) : m_cuda(cuda) {}

FlatPipeline::~FlatPipeline() { release(); }

PipelineInfo FlatPipeline::info() const {
	PipelineInfo info;
	info.id = "flat";
	info.displayName = "Flat (no LOD)";
	info.progressive = false;
	info.needsWholeCloudResident = true;
	// Only the input points; no LOD structure of its own.
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
	// No structure to clear. Diagnostics are cleared so a previously reported
	// overflow does not stick around and mislabel a good run.
	if (m_diagnostics) {
		REMO_CU(cuMemsetD8(m_diagnostics, 0, sizeof(DeviceDiagnostics)));
	}
	m_stats.allocOverflow = false;
}

TimingScopes FlatPipeline::timingScopes() const {
	TimingScopes scopes;
	// Nothing to build -- build() only latches a device pointer, which is host work.
	scopes.render = "flat.render";
	return scopes;
}

bool FlatPipeline::build(PointSource& source, const FrameContext&) {
	// There is nothing to build. All this does is latch the device pointer once the
	// source is fully resident, because render() is not handed the source.
	if (!source.isFullyResident()) return true;  // still uploading; try again
	if (m_points != 0) return false;             // done

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

	// Size scratch from the actual viewport, growing on resize.
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

	// Cooperative launch: the kernel grid.sync()s between clear, rasterise, EDL and
	// resolve, so the whole grid must be resident. gridForKernel resolves that from
	// occupancy rather than hardcoding it.
	const int grid = m_cuda.gridForKernel(kernel, m_blockSize);

	{
		GpuScope scope(frame.profiler, "flat.render");
		REMO_CU(cuLaunchCooperativeKernel(kernel, static_cast<unsigned>(grid), 1, 1,
		                                  static_cast<unsigned>(m_blockSize), 1, 1, 0,
		                                  0, kernelArgs));
	}

	// The profiler harvests without stalling, so there is nothing to read here: in the
	// deferred regime the sample lands on a later frame's pass, and in the strict regime
	// on this frame's endFrame(). This is what the old code MEANT to do -- it queried
	// the event pair it had just re-recorded, got CUDA_ERROR_NOT_READY every time, and
	// so never assigned a render time at all outside --strict-timing.
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

void FlatPipeline::gui(const GpuProfiler& profiler) {
	ImGui::TextUnformatted(
		"No LOD: every point is rasterised every frame.\n"
		"This is the control condition and the image-quality ground truth\n"
		"that LOD pipelines are compared against.");
	ImGui::Separator();

	if (ImGui::BeginTable("flat_timing", 2, ImGuiTableFlags_SizingStretchProp)) {
		// The baseline every LOD renderer's cost is read against: this is what drawing
		// the whole cloud with no selection at all costs on this scene.
		timingRow(profiler, "render (ms)", "flat.render");
		ImGui::EndTable();
	}

	ImGui::Text("render scratch: %.1f MB",
	            static_cast<double>(m_scratchBytes) / (1024.0 * 1024.0));
	if (m_program && m_program->isStale()) {
		ImGui::TextColored(ImVec4(1, 0.4f, 0.2f, 1),
		                   "kernel source changed but failed to compile;\n"
		                   "showing the previously loaded version");
	}
}

}  // namespace remo
