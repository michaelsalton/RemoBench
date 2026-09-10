// SimLOD: progressive LOD construction -- batches are inserted into a LIVE octree.
//
// From "SimLOD: Simultaneous LOD Generation and Loading" (Schuetz et al. 2024). Structurally
// the opposite of CudaLOD, which is what makes the pair worth comparing:
//
//   CudaLOD  the whole cloud must be resident, then two launches build the tree in one shot
//   SimLOD   one bounded launch per FRAME inserts up to 20 million-point batches into a tree
//            that is simultaneously being rendered, under a 10ms device-side time budget
//
// So SimLOD shows a usable image almost immediately and refines, while CudaLOD shows nothing
// until it is done. That is the comparison neither upstream repo can make, because they are
// separate binaries with different loaders and different rasterisers.
//
// THIS IS AN EXTERNAL COMPARISON BASELINE. Its device code is vendored BYTE-IDENTICAL to
// external/SimLOD -- bench/check_vendored.sh asserts it -- so the numbers stay verifiable
// against bench/reference/. Only the renderer is replaced, by a selection pass feeding the
// shared rasteriser.
//
// Do not edit kernels/simlod/ to make something else work. RemoLOD is the pipeline that gets
// to change things, and it forks into kernels/remolod/. The accumulator was once spliced into
// kernel_construct here; it added two parameters the host never passed, and this pipeline
// silently built no tree at all for a whole commit. See kernels/simlod/VENDORED.md.
//
// THE 50M POINT CEILING is a direct consequence of staying byte-identical. kernel_construct
// addresses batch N at (N % BATCH_STREAM_SIZE) with upstream's BATCH_STREAM_SIZE == 50, and
// RemoBench feeds it a resident cloud with no wrapping, so past 50 batches it re-reads slot 0
// and builds from the wrong points -- silently, since nothing faults.
// PipelineRegistry::unsupportedReason refuses such clouds. Lifting the limit means a genuinely
// wrapping ring in PointSource, NOT another edit to structures.cuh.
//
// HOW IT IS FED, and the current limitation:
//
// kernel_construct does not care that our points are already resident -- it walks batches,
// reading batchSizes[slot] and numBatchesUploaded. ResidentSource publishes exactly that for
// the whole cloud with every batch pre-declared as uploaded, which is the "whole cloud is a
// ring with non-wrapping slot addresses" idea from PointSource.h. Construction is therefore
// genuinely progressive across frames, but ingest is not overlapped with it.
//
// That is a legitimate measurement mode in its own right -- it isolates construction cost
// from streaming cost -- but it is NOT the paper's headline claim, which is that loading and
// LOD generation overlap. Measuring that needs the real pinned-pool streaming loader.
//
// Host-side changes from upstream, all about making measurement trustworthy:
//   - Buffer sizes come from DeviceBudget instead of an 80%-of-free-VRAM land grab, which
//     upstream does and which makes a run depend on what else was on the GPU at the time.
//   - The momentary buffer is sized from the kernel's ACTUAL allocation sum. Upstream's
//     kernel allocates ~409MB from a 300MB buffer; it only survives because the backlog
//     arrays are never filled, and chunkQueue's base pointer sits entirely outside the
//     allocation.
//   - numNodes is clamped and nodeCapacityReached is reported, rather than letting the
//     unchecked `atomicAdd(&stats->numNodes, 8)` walk off the end of the pool.

#pragma once

#include <memory>
#include <string>

#include "remo/CudaModularProgram.h"
#include "remo/ILodPipeline.h"

namespace remo {

class CudaContext;

class SimlodPipeline final : public ILodPipeline {
public:
	explicit SimlodPipeline(CudaContext& cuda);
	~SimlodPipeline() override;

	PipelineInfo info() const override;

	bool initPrograms(std::string* err) override;
	bool allocate(const CloudMeta& meta, const DeviceBudget& budget,
	              std::string* err) override;
	void release() override;
	void reset() override;

	bool build(PointSource& source, const FrameContext& frame) override;
	void render(const FrameContext& frame) override;

	const PipelineStats& stats() const override { return m_stats; }
	TimingScopes timingScopes() const override;
	void gui(const GpuProfiler& profiler) override;

private:
	void readStats();
	void ensureScratch(int width, int height);
	// Builds the Uniforms struct SimLOD's kernels expect. Separate from SharedUniforms
	// because it is upstream's struct, kept unmodified.
	void fillUniforms(const FrameContext& frame, void* outUniforms) const;

	CudaContext& m_cuda;

	// Three programs, matching upstream: reset, construct, render. Declared in
	// kernels/simlod/programs.txt.
	std::unique_ptr<CudaModularProgram> m_resetProgram;
	std::unique_ptr<CudaModularProgram> m_constructProgram;
	std::unique_ptr<CudaModularProgram> m_renderProgram;

	// --- device memory -----------------------------------------------------
	CUdeviceptr m_momentary = 0;    // per-launch scratch for construction
	uint64_t m_momentaryBytes = 0;
	CUdeviceptr m_persistent = 0;   // the octree itself: chunks and occupancy grids
	uint64_t m_persistentBytes = 0;
	CUdeviceptr m_nodes = 0;        // flat Node pool
	uint64_t m_nodesBytes = 0;
	CUdeviceptr m_statsBuffer = 0;   // device-side Stats, read back each launch
	CUdeviceptr m_frameStart = 0;   // nanotime, for the device-side time budget
	CUdeviceptr m_cudaPrint = 0;    // dummy; the facility is a no-op both ends

	// Scratch handed to the RESET kernel in place of the source's real batch metadata.
	//
	// reset.cu unconditionally zeroes batchSizes[0 .. BATCH_STREAM_SIZE-1] and
	// numBatchesUploaded. Upstream relies on that, because its uploader republishes both as
	// batches stream in. Our source publishes them once, up front, for an already-resident
	// cloud -- so letting reset touch them would wipe every batch size to zero and nothing
	// would ever be ingested. Reset gets these instead, and construct gets the real ones.
	CUdeviceptr m_resetBatchSizes = 0;
	CUdeviceptr m_resetNumUploaded = 0;
	CUdeviceptr m_scratch = 0;      // render scratch
	uint64_t m_scratchBytes = 0;
	CUdeviceptr m_diagnostics = 0;

	bool m_needsReset = true;
	bool m_complete = false;
	uint64_t m_numPoints = 0;

	PipelineStats m_stats;
	int m_blockSize = 256;

	uint32_t m_batchesConsumed = 0;
	uint32_t m_batchesTotal = 0;
};

}  // namespace remo
