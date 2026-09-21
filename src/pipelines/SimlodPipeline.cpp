#include "pipelines/SimlodPipeline.h"

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
#include "shell/GuiWidgets.h"
#include "shell/TimingUi.h"

#include "../../kernels/simlod/HostDeviceInterface.h"
#include "../../kernels/simlod/simlod_layout.h"

namespace remo {
namespace {

constexpr uint64_t kMomentaryBytes = 512ull << 20;

constexpr uint32_t kMaxNodes = simlod::kMaxNodes;

constexpr double kBytesPerPointPersistent = 48.0;
constexpr uint64_t kMinPersistentBytes = 512ull << 20;

// The measured floor for the persistent store, against which kBytesPerPointPersistent is
// an appetite. SimLOD's Table 5 reports 9.1 GB for the 350M Morro Bay cloud -- 26 B/pt
// including unused chunk capacity -- and RemoBench's own 36M tree comes to 25.7 B/pt.
// Only this number refuses a cloud; see PipelineInfo.
constexpr double kMinBytesPerPointPersistent = 26.0;

constexpr uint64_t kBytesPerPixelScratch = 64;
constexpr uint64_t kMinScratchBytes = 64ull << 20;

}

SimlodPipeline::SimlodPipeline(CudaContext& cuda) : m_cuda(cuda) {}
SimlodPipeline::~SimlodPipeline() { release(); }

PipelineInfo SimlodPipeline::info() const {
	PipelineInfo info;
	info.id = "simlod";
	info.displayName = "SimLOD (progressive)";
	info.progressive = true;
	// kernel_construct consumes nothing but BatchView, so it never needed the cloud
	// resident -- upstream streams into it, and holding the whole cloud was RemoBench's
	// own choice. The ring depth is not ours to pick: kernel_construct computes
	// `batchIndex % BATCH_STREAM_SIZE` itself, so the device ring must have exactly
	// that many slots or the addressing reads outside it.
	info.needsWholeCloudResident = false;
	info.ringSlots = simlod::kBatchStreamSize;
	info.bytesPerPointEstimate = kBytesPerPointPersistent;
	info.minBytesPerPointEstimate = kMinBytesPerPointPersistent;
	info.minStoreBytes = kMinPersistentBytes;
	info.fixedBytesEstimate =
		kMomentaryBytes + static_cast<uint64_t>(kMaxNodes) * simlod::kNodeBytes;
	return info;
}

bool SimlodPipeline::initPrograms(std::string* err) {
	struct Spec {
		std::unique_ptr<CudaModularProgram>* target;
		const char* module;
		const char* kernel;
	};
	const Spec specs[] = {
		{&m_resetProgram, "simlod/reset.cu", "kernel"},
		{&m_constructProgram, "simlod/progressive_octree_voxels.cu", "kernel_construct"},
		{&m_renderProgram, "simlod/simlod_render.cu", "kernel_render"},
	};

	for (const Spec& spec : specs) {
		KernelProgramDesc desc;
		desc.modules = {spec.module};
		desc.kernels = {spec.kernel};
		*spec.target = std::make_unique<CudaModularProgram>(std::move(desc));
		if (!(*spec.target)->ok()) {
			if (err) *err = (*spec.target)->lastError();
			return false;
		}
	}

	m_constructProgram->onCompile([this] { m_needsReset = true; });

	return true;
}

bool SimlodPipeline::allocate(const CloudMeta& meta, const DeviceBudget& budget,
                              std::string* err) {
	release();

	m_stats = PipelineStats{};
	m_stats.bytesCapacity = budget.bytes;
	m_numPoints = meta.numPoints;

	if (meta.numPoints == 0) {
		if (err) *err = "cloud is empty";
		return false;
	}

	// The input term no longer scales with the cloud: what PointSource takes out of the
	// shared budget is the ring, and the ring is the same size for 36M points as for
	// 350M. That is the whole of what streaming buys.
	const uint64_t batches =
		(meta.numPoints + kSlotCapacity - 1) / kSlotCapacity;
	const uint64_t slots = std::min<uint64_t>(simlod::kBatchStreamSize, batches);
	const uint64_t inputBytes = slots * kSlotCapacity * sizeof(Point);
	const uint64_t available =
		budget.bytes > inputBytes ? budget.bytes - inputBytes : 0;

	m_nodesBytes = static_cast<uint64_t>(kMaxNodes) * simlod::kNodeBytes;
	m_momentaryBytes = kMomentaryBytes;

	uint64_t wantPersistent = static_cast<uint64_t>(
		kBytesPerPointPersistent * static_cast<double>(meta.numPoints));
	wantPersistent = std::max(wantPersistent, kMinPersistentBytes);

	const uint64_t fixed = m_nodesBytes + m_momentaryBytes;
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
	    !alloc(&m_statsBuffer, sizeof(Stats), "stats") ||
	    !alloc(&m_frameStart, 8, "the frame timestamp") ||
	    !alloc(&m_cudaPrint, 1024, "the CudaPrint buffer") ||
	    !alloc(&m_diagnostics, sizeof(DeviceDiagnostics), "diagnostics") ||
	    !alloc(&m_resetBatchSizes, uint64_t(simlod::kBatchStreamSize) * 4,
	           "reset's batchSizes scratch") ||
	    !alloc(&m_resetNumUploaded, 4, "reset's numBatchesUploaded scratch")) {
		release();
		return false;
	}

	m_stats.numPoints = 0;
	m_stats.bytesAllocated = m_persistentBytes + m_momentaryBytes + m_nodesBytes;
	m_needsReset = true;
	m_complete = false;
	return true;
}

void SimlodPipeline::release() {
	for (CUdeviceptr* p : {&m_momentary, &m_persistent, &m_nodes, &m_statsBuffer,
	                       &m_frameStart, &m_cudaPrint, &m_scratch, &m_diagnostics,
	                       &m_resetBatchSizes, &m_resetNumUploaded}) {
		if (*p) {
			REMO_CU(cuMemFree(*p));
			*p = 0;
		}
	}
	m_momentaryBytes = m_persistentBytes = m_nodesBytes = m_scratchBytes = 0;

	m_complete = false;
	m_needsReset = true;
}

void SimlodPipeline::reset() {
	m_needsReset = true;
	m_complete = false;
	m_batchesConsumed = 0;
}

TimingScopes SimlodPipeline::timingScopes() const {
	TimingScopes scopes;
	scopes.build = {"simlod.reset", "simlod.construct"};
	scopes.render = "simlod.render";
	return scopes;
}

void SimlodPipeline::fillUniforms(const FrameContext& frame, void* out) const {
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

bool SimlodPipeline::build(PointSource& source, const FrameContext& frame) {
	if (!m_constructProgram || !m_constructProgram->ok()) return false;
	if (!m_resetProgram || !m_resetProgram->ok()) return false;
	if (!m_momentary || !m_persistent || !m_nodes) return false;

	const BatchView view = source.view();
	if (view.slots == 0 || view.batchSizes == 0 || view.numBatchesUploaded == 0) {
		return true;
	}

	// The one thing a consumer can check about the ring it was handed.
	//
	// kernel_construct reads batch B from slot B % BATCH_STREAM_SIZE, always. The host
	// writes batch B to slot B % numSlots. Those agree in exactly two cases: the ring is
	// BATCH_STREAM_SIZE deep, or the whole cloud fits in it and neither side wraps at
	// all. Anything else reads a slot holding some other batch -- with no fault, no
	// allocation error, and a plausible node count out the far end. Note that a ring
	// DEEPER than BATCH_STREAM_SIZE is just as wrong as a shallower one, which is why
	// this tests the kernel's constant and not BatchView::wrapping.
	const bool ringMatchesKernel =
		view.numSlots == simlod::kBatchStreamSize ||
		(view.numBatchesTotal <= simlod::kBatchStreamSize &&
		 view.numSlots >= view.numBatchesTotal);
	if (!ringMatchesKernel) {
		if (!m_ringMismatchReported) {
			m_ringMismatchReported = true;
			fprintf(stderr,
			        "remobench: SimLOD was handed a %u-slot ring but kernel_construct "
			        "wraps at %u; refusing to build from it.\n",
			        view.numSlots, simlod::kBatchStreamSize);
		}
		return false;
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

		if (frame.profiler) frame.profiler->clearPrefix("simlod.");

		{
			GpuScope scope(frame.profiler, "simlod.reset");
			REMO_CU(cuLaunchCooperativeKernel(resetKernel, 1, 1, 1, 1, 1, 1, 0, 0,
			                                  resetArgs));
		}

		const CUresult sync = cuCtxSynchronize();
		if (isStickyError(sync)) {
			reportDeadContextAndExit(sync, "SimLOD reset kernel");
		}
		REMO_CU(sync);

		m_needsReset = false;
		m_complete = false;
		m_batchesConsumed = 0;

		// The reset kernel put the device's batch counter back to zero, so the
		// producer has to go back to batch 0 too. Without this a rebuild keeps filling
		// slots ahead of a consumer that has restarted, and the tree comes out of the
		// wrong points -- silently.
		source.rewind();
		return true;
	}

	if (m_complete) return false;

	CUfunction construct = m_constructProgram->kernel("kernel_construct");
	if (!construct) return false;

	const uint64_t nowNs = static_cast<uint64_t>(now() * 1e9);
	REMO_CU(cuMemcpyHtoD(m_frameStart, &nowNs, sizeof(nowNs)));

	CUdeviceptr points = view.slots;
	CUdeviceptr momentary = m_momentary;

	void* args[] = {&uniforms, &points,     &momentary,  &persistent, &nodes,
	                &statsPtr, &frameStart, &cudaPrint,  &numUploaded, &batchSizes};

	const int grid = m_cuda.gridForKernel(construct, m_blockSize, 1);

	{
		GpuScope scope(frame.profiler, "simlod.construct");
		REMO_CU(cuLaunchCooperativeKernel(construct, static_cast<unsigned>(grid), 1, 1,
		                                  static_cast<unsigned>(m_blockSize), 1, 1, 0, 0,
		                                  args));
	}

	const CUresult sync = cuCtxSynchronize();
	if (isStickyError(sync)) {
		reportDeadContextAndExit(sync, "SimLOD kernel_construct");
	}
	REMO_CU(sync);

	readStats();

	// The only consumption signal there is, handed straight back to the producer: it
	// opens the window for the next pump(). stats->batchletIndex is incremented once
	// per batch actually folded into the tree, so a launch cut short by the device time
	// budget narrows the window rather than widening it.
	source.setBatchesConsumed(m_batchesConsumed);

	if (m_batchesTotal > 0 && m_batchesConsumed >= m_batchesTotal) {
		m_complete = true;
		return false;
	}
	return true;
}

void SimlodPipeline::readStats() {
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
	// Deliberately NOT copying s.numVisible* here. No kernel writes those Stats fields
	// -- visibility is counted by the render kernel into DeviceDiagnostics -- so copying
	// them zeroes what render() just measured. It went unnoticed while every build
	// completed, because a completed build stops calling readStats and the last render's
	// numbers survived. A build that stops on memCapacityReached never completes, and
	// then the dump reports 0 visible nodes for a tree that is plainly on screen.

	m_stats.bytesHighWater = s.allocatedBytes_persistent;
	m_stats.bytesAllocated = m_persistentBytes + m_momentaryBytes + m_nodesBytes;
	m_stats.memCapacityReached = s.memCapacityReached;

	if (m_stats.numNodes >= kMaxNodes) m_stats.nodeCapacityReached = true;
}

void SimlodPipeline::ensureScratch(int width, int height) {
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

void SimlodPipeline::render(const FrameContext& frame) {
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
		GpuScope scope(frame.profiler, "simlod.render");
		REMO_CU(cuLaunchCooperativeKernel(kernel, static_cast<unsigned>(grid), 1, 1,
		                                  static_cast<unsigned>(m_blockSize), 1, 1, 0, 0,
		                                  kernelArgs));
	}

	if (frame.strictTiming) {
		const CUresult sync = cuCtxSynchronize();
		if (isStickyError(sync)) {
			reportDeadContextAndExit(sync, "SimLOD kernel_render");
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

void SimlodPipeline::guiControls() {
	paragraph(
		"Progressive construction: one bounded launch per frame inserts batches "
		"into the octree while it is being rendered.");

	if (ImGui::Button("rebuild")) m_needsReset = true;

	hint("ingest is the resident path, not the paper's overlapped streaming "
	     "loader -- construction is progressive, loading is not");

	if (m_renderProgram && m_renderProgram->isStale()) {
		ImGui::TextColored(ImVec4(1, 0.4f, 0.2f, 1),
		                   "render kernel failed to recompile;\n"
		                   "showing the previously loaded version");
	}
}

void SimlodPipeline::guiStats(const GpuProfiler& profiler) {
	if (m_batchesTotal > 0) {
		const float progress =
			static_cast<float>(m_batchesConsumed) / static_cast<float>(m_batchesTotal);
		char overlay[64];
		snprintf(overlay, sizeof(overlay), "%u / %u batches", m_batchesConsumed,
		         m_batchesTotal);
		ImGui::ProgressBar(progress, ImVec2(-1.0f, 0.0f), overlay);
	}

	if (ImGui::BeginTable("simlod_timing", 2, ImGuiTableFlags_SizingStretchProp)) {
		auto row = [](const char* label, const char* fmt, double v) {
			ImGui::TableNextRow();
			ImGui::TableNextColumn();
			ImGui::TextUnformatted(label);
			ImGui::TableNextColumn();
			ImGui::Text(fmt, v);
		};

		timingRow(profiler, "reset (ms)", "simlod.reset");
		timingRow(profiler, "construct (ms)", "simlod.construct");
		timingRow(profiler, "render (ms)", "simlod.render");

		const BuildTotals totals = buildTotals(profiler, timingScopes());
		row("build total", "%.2f ms", totals.ms);
		row("launches", "%.0f", static_cast<double>(totals.launches));

		const double mps = totals.ms > 0.0
		                       ? double(m_stats.numPointsIngested) / 1e6 /
		                             (totals.ms / 1000.0)
		                       : 0.0;
		row("throughput (build total)", "%.0f MP/s", mps);
		row("persistent", "%.2f GB", double(m_persistentBytes) / 1e9);
		row("high water", "%.2f GB", double(m_stats.bytesHighWater) / 1e9);
		ImGui::EndTable();
	}
}

}
