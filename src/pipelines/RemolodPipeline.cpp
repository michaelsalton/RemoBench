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

// SimLOD's host/device contract, pulled in unchanged. RemoLOD has not needed to fork Uniforms
// or Stats yet; when Analysis wants fields of its own, this is the next file to copy into
// kernels/remolod/.
#include "../../kernels/simlod/HostDeviceInterface.h"
// RemoLOD's OWN device-layout numbers -- not simlod_layout.h, because RemoLOD raises
// BATCH_STREAM_SIZE and defines a node-pool capacity that SimLOD does not have.
#include "../../kernels/remolod/remolod_layout.h"

namespace remo {
namespace {

// Momentary (per-launch) scratch for construction.
//
// Upstream uses 300MB, and its kernel allocates roughly 409MB from it -- 10M voxel-backlog
// entries at 16B, 10M targets at 8B, 10M spilled points at 16B, a 1M-entry chunk queue and
// assorted counters. It survives only because those arrays are never filled anywhere near
// capacity, with chunkQueue's base pointer landing entirely outside the allocation and its
// writes going into whatever cuMemAlloc returned next.
//
// Sized here from the actual allocation sum with headroom, so the buffer genuinely contains
// what the kernel hands out. Do not lower it to upstream's 300MB.
constexpr uint64_t kMomentaryBytes = 512ull << 20;

constexpr uint32_t kMaxNodes = remolod::kMaxNodes;

// Persistent octree store, per input point. Chunks are 16KB for 1000 points (~16 B/pt), and
// every inner node also carries a 256KB occupancy grid, which dominates for a deep tree.
constexpr double kBytesPerPointPersistent = 48.0;
constexpr uint64_t kMinPersistentBytes = 512ull << 20;

constexpr uint64_t kBytesPerPixelScratch = 64;
constexpr uint64_t kMinScratchBytes = 64ull << 20;

}  // namespace

RemolodPipeline::RemolodPipeline(CudaContext& cuda) : m_cuda(cuda) {}
RemolodPipeline::~RemolodPipeline() { release(); }

PipelineInfo RemolodPipeline::info() const {
	PipelineInfo info;
	info.id = "remolod";
	info.displayName = "RemoLOD (detail-aware)";
	info.progressive = true;
	// False in principle -- the whole point is that it streams. True for now, because ingest
	// is the resident path until the pinned-pool loader lands.
	info.needsWholeCloudResident = true;
	// Persistent store, the resident input points, and the accumulator side array. The last
	// is a fixed 21MB rather than per-point, so it rounds to nothing here.
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
		// One module, two entry points: the fold and the invariant check. They are separate
		// kernels because the check must observe every block's writes, and this is a plain
		// launch with no grid.sync() to order the two phases inside one kernel.
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

	// A tree built by the previous version of the construct kernel is not valid input to the
	// new one, so a hot reload must reset. Upstream does not do this and leaves the old tree
	// on screen, which is a confusing thing to debug.
	m_constructProgram->onCompile([this] { m_needsReset = true; });
	// The accumulator's sums are keyed to the tree AND to the arithmetic that produced them,
	// so editing the pass invalidates what is already accumulated. Resetting is the honest
	// response; without it a hot reload silently mixes two versions' sums in one array.
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
	    // CudaPrint is a no-op on both ends but is threaded through the octree kernel's
	    // signature, so it gets a small real allocation rather than a null pointer.
	    !alloc(&m_cudaPrint, 1024, "the CudaPrint buffer") ||
	    !alloc(&m_diagnostics, sizeof(DeviceDiagnostics), "diagnostics") ||
	    // Must hold BATCH_STREAM_SIZE entries: reset writes all of them. See the member
	    // comment in the header for why reset does not get the real buffer.
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
	// The timing samples are dropped by build(), where the profiler is in hand -- see the
	// reset branch there.
	m_needsReset = true;
	m_complete = false;
	m_batchesConsumed = 0;
}

TimingScopes RemolodPipeline::timingScopes() const {
	TimingScopes scopes;
	// Every construction launch, or the reported build total under-reports. The accumulator
	// is listed here because it IS construction cost -- leaving it out would make RemoLOD
	// look free next to SimLOD, which is the opposite of what the comparison is for.
	scopes.build = {"remolod.reset", "remolod.construct", "remolod.accumulate"};
	scopes.render = "remolod.render";
	return scopes;
}

void RemolodPipeline::fillUniforms(const FrameContext& frame, void* out) const {
	Uniforms* u = static_cast<Uniforms*>(out);
	std::memset(u, 0, sizeof(Uniforms));

	// Only four fields are read by kernel_construct: boxMin, boxMax, frameCounter and
	// persistentBufferCapacity. The rest exist for upstream's own renderer, which we do not
	// use -- shading is driven from SharedUniforms instead, so they stay zero rather than
	// becoming controls that appear to do something.
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

	// sumLeafCounts and innerWithSums are recomputed from scratch every launch, so they are
	// zeroed before the verify pass rather than carried. The cumulative fields
	// (mortonWatermark, numAccumulated) are NOT touched here -- they describe the whole build.
	// pointsFolded and numLeavesFolded describe only this launch, so they reset too.
	AccumGlobals g = m_accum;
	g.sumLeafCounts = 0;
	g.innerWithSums = 0;
	g.pointsFolded = 0;
	g.numLeavesFolded = 0;
	REMO_CU(cuMemcpyHtoD(m_accumGlobals, &g, sizeof(g)));

	// The octree root cube, derived EXACTLY as remolod_render.cu derives it from the same
	// uniforms: the cloud is pre-translated so its minimum is the origin, and the box is
	// cubed on its longest axis. Deriving it here from a second source -- CloudMeta, say --
	// would give the accumulator's node-local normalisation a different cube from the one the
	// octree's cell indexing uses, and every sum would be silently off by that ratio.
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

	// A PLAIN launch, not a cooperative one. Nothing here grid-syncs: blocks are independent
	// because each owns a whole node, which is the property that made the separate pass
	// cheaper than the in-kernel hook it replaces.
	const int grid = m_cuda.gridForKernel(fold, kAccumBlockSize);
	{
		GpuScope scope(frame.profiler, "remolod.accumulate");
		REMO_CU(cuLaunchKernel(fold, static_cast<unsigned>(grid), 1, 1,
		                       static_cast<unsigned>(kAccumBlockSize), 1, 1, 0, 0,
		                       kernelArgs, nullptr));
		// Same stream, so the verify pass is ordered after every fold block has retired.
		// It is inside the same scope because it is part of the accumulator's cost and
		// pretending otherwise would understate it.
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
		// The source has not published a batch view. Nothing to do; not an error, since a
		// streaming source may simply not have started yet.
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

	// --- reset, if needed -------------------------------------------------
	if (m_needsReset) {
		CUfunction resetKernel = m_resetProgram->kernel("kernel");
		if (!resetKernel) return false;

		// Scratch, not the source's real metadata -- see the header.
		CUdeviceptr resetSizes = m_resetBatchSizes;
		CUdeviceptr resetUploaded = m_resetNumUploaded;
		void* resetArgs[] = {&uniforms,  &persistent,    &nodes,     &statsPtr,
		                     &cudaPrint, &resetUploaded, &resetSizes};

		// The tree the previous samples describe no longer exists, and a distribution
		// spanning two different trees describes neither. Drop them here rather than in
		// reset(), which does not have the profiler.
		if (frame.profiler) frame.profiler->clearPrefix("remolod.");

		// One block, one thread -- but a COOPERATIVE launch, because remolod_reset.cu calls
		// grid.sync(). A plain cuLaunchKernel makes that undefined, and the failure mode is a
		// bare CUDA_ERROR_LAUNCH_FAILED ("unspecified launch failure") that says nothing about
		// the cause.
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

		// The reset kernel clears the octree but knows nothing about the accumulator -- that
		// is RemoLOD's own state, and keeping it out of the forked kernel keeps the fork's
		// diff against SimLOD to the two constants. Zeroed here instead.
		if (m_nodeAccums) REMO_CU(cuMemsetD8(m_nodeAccums, 0, m_nodeAccumsBytes));
		if (m_accumGlobals) REMO_CU(cuMemsetD8(m_accumGlobals, 0, sizeof(AccumGlobals)));
		m_accum = AccumGlobals{};

		m_needsReset = false;
		m_complete = false;
		m_batchesConsumed = 0;
	}

	if (m_complete) return false;

	// --- one bounded construction step ------------------------------------
	CUfunction construct = m_constructProgram->kernel("kernel_construct");
	if (!construct) return false;

	// The device-side 10ms budget is measured against this, so it must be refreshed every
	// launch or the kernel believes it is already out of time.
	const uint64_t nowNs = static_cast<uint64_t>(now() * 1e9);
	REMO_CU(cuMemcpyHtoD(m_frameStart, &nowNs, sizeof(nowNs)));

	CUdeviceptr points = view.slots;
	CUdeviceptr momentary = m_momentary;

	void* args[] = {&uniforms, &points,     &momentary, &persistent,  &nodes,
	                &statsPtr, &frameStart, &cudaPrint, &numUploaded, &batchSizes};

	// Upstream launches this at exactly 1 block per SM.
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

	// --- fold the new points into the per-node sums ------------------------
	//
	// After construct, not inside it: the pass reads the finished tree, and reading it after
	// the launch that wrote it is what removes the need to touch the octree kernel at all.
	runAccumulator(frame);

	// Progressive: done once every batch has been consumed.
	if (m_batchesTotal > 0 && m_batchesConsumed >= m_batchesTotal) {
		m_complete = true;
		return false;
	}
	return true;  // more to do next frame
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

	// numNodes is a bump index grown by `atomicAdd(&stats->numNodes, 8)` with no capacity
	// check on the device, so this is the only place the pool running out can be noticed.
	if (m_stats.numNodes >= kMaxNodes) m_stats.nodeCapacityReached = true;
}

void RemolodPipeline::ensureScratch(int width, int height) {
	const uint64_t pixels =
		static_cast<uint64_t>(width) * static_cast<uint64_t>(height);
	uint64_t needed = std::max(pixels * kBytesPerPixelScratch, kMinScratchBytes);
	// The render kernel also allocates the draw list plus two MAX_NODES_CAPACITY flag arrays.
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
		// Our render kernel owns selection, so these come from it rather than from SimLOD's
		// Stats (which our renderer never writes).
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

	// The invariant that matters most: sum of per-leaf counts against the point total the
	// octree itself reports. numPoints is the sum of node->numPoints over leaves, so this
	// checks the accumulation against the insertion it shadows, per point, over the whole
	// cloud. Printed as a verdict as well as a pair, so a script can grep for "MISMATCH"
	// rather than having to parse and compare two numbers.
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

		// A progressive build is hundreds of bounded launches, each governed by the
		// device-side MAX_PROCESSING_TIME budget. The interesting question is the SHAPE of
		// that distribution -- does the budget hold, and how does per-launch cost evolve as
		// the octree deepens -- which a running sum cannot answer.
		timingRow(profiler, "reset (ms)", "remolod.reset");
		timingRow(profiler, "construct (ms)", "remolod.construct");
		timingRow(profiler, "accumulate (ms)", "remolod.accumulate");
		timingRow(profiler, "render (ms)", "remolod.render");

		const BuildTotals totals = buildTotals(profiler, timingScopes());
		row("build total", "%.2f ms", totals.ms);
		row("launches", "%.0f", static_cast<double>(totals.launches));

		// Ingest throughput is a whole-build figure, so it divides by the SUM over every
		// launch -- unlike CudaLOD's, which is one build and uses the median.
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

	// These are the whole verification surface for the accumulator: tests/unit/ is empty, so
	// an invariant that is not displayed is an invariant nobody checks.
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

}  // namespace remo
