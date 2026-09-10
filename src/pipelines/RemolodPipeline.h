// RemoLOD: RemoBench's own pipeline, and the one the research is about.
//
// It is a progressive octree builder forked from SimLOD, plus the passes SimLOD does not
// have. Today that is one pass -- the accumulator. Analysis and Refinement land next to it.
//
//   Rasterize   every frame, unconditionally      exists (remolod_render.cu)
//   Update      on batch completion               exists (remolod_octree.cu)
//   Accumulate  after each Update                 exists (remolod_accum.cu)
//   Analysis    every frame, per node             to build
//   Refinement  only if budget remains            to build
//
// WHY IT IS A FORK RATHER THAN A HOOK
//
// The comparison baselines -- kernels/simlod/ and kernels/cudalod/ -- are only worth having
// while they still reproduce their published numbers against bench/reference/. So RemoLOD
// takes copies of what it needs and changes those, and bench/check_vendored.sh asserts the
// originals never move. An earlier version of the accumulator was edited straight into
// SimLOD's kernel_construct; it added two parameters that the host launch never passed, and
// the SimLOD pipeline silently built nothing for a whole commit. That is the failure mode
// this separation exists to make impossible.
//
// RemoLOD still #includes SimLOD's vendored math, allocator and Uniforms/Stats headers.
// Reading a header is pulling, not changing; the fork covers the files RemoLOD needs to
// diverge in (structures, the octree kernel, reset, selection).
//
// WHAT IT ALREADY DIVERGES ON
//
//   BATCH_STREAM_SIZE  8192, against upstream's 50. The kernel addresses batch N at
//                      (N % BATCH_STREAM_SIZE), and RemoBench feeds it a resident cloud with
//                      no wrapping, so at 50 anything past 50M points re-read slot 0.
//                      SimLOD keeps upstream's value and SimlodPipeline refuses such clouds;
//                      RemoLOD raises it and takes them.
//   MAX_NODES_CAPACITY a constant upstream does not have at all. Refinement's job is to split
//                      more eagerly where geometry warrants, which walks toward the pool
//                      bound on purpose, so the bound has to be nameable.
//
// The octree kernel itself is otherwise identical to SimLOD's today, deliberately: while that
// holds, any difference between the two pipelines is attributable to the passes around it.
//
// HOW IT IS FED, and the current limitation:
//
// kernel_construct does not care that our points are already resident -- it walks batches,
// reading batchSizes[slot] and numBatchesUploaded. ResidentSource publishes exactly that for
// the whole cloud with every batch pre-declared as uploaded. Construction is therefore
// genuinely progressive across frames, but ingest is not overlapped with it. That isolates
// construction cost from streaming cost, which is a legitimate measurement mode but NOT the
// paper's overlapped-loading claim.

#pragma once

#include <memory>
#include <string>

#include "remo/CudaModularProgram.h"
#include "remo/ILodPipeline.h"
#include "remo/RemoAccum.h"

namespace remo {

class CudaContext;

class RemolodPipeline final : public ILodPipeline {
public:
	explicit RemolodPipeline(CudaContext& cuda);
	~RemolodPipeline() override;

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
	std::vector<std::string> diagnostics() const override;

	// The accumulator can be switched off, and must be reachable from the command line
	// (--remolod-no-accum) as well as the panel. It is the "with and without" half of the
	// acceptance test: the pass mutates no tree state, so the structural counts must be
	// unchanged with it off. A GUI-only toggle would make that test unscriptable.
	void setAccumEnabled(bool on) { m_accumEnabled = on; }
	bool accumEnabled() const { return m_accumEnabled; }

private:
	void readStats();
	void ensureScratch(int width, int height);
	// Builds the Uniforms struct the octree kernels expect. Separate from SharedUniforms
	// because it is SimLOD's struct, which RemoLOD has not needed to fork yet.
	void fillUniforms(const FrameContext& frame, void* outUniforms) const;

	// The accumulator launch, plus its verification launch. Separate from build() so the
	// ordering (construct -> accumulate -> verify, each fully ordered against the last) is
	// stated in one place.
	void runAccumulator(const FrameContext& frame);

	CudaContext& m_cuda;

	std::unique_ptr<CudaModularProgram> m_resetProgram;
	std::unique_ptr<CudaModularProgram> m_constructProgram;
	std::unique_ptr<CudaModularProgram> m_accumProgram;
	std::unique_ptr<CudaModularProgram> m_renderProgram;

	// --- device memory -----------------------------------------------------
	CUdeviceptr m_momentary = 0;    // per-launch scratch for construction
	uint64_t m_momentaryBytes = 0;
	CUdeviceptr m_persistent = 0;   // the octree itself: chunks and occupancy grids
	uint64_t m_persistentBytes = 0;
	CUdeviceptr m_nodes = 0;        // flat Node pool
	uint64_t m_nodesBytes = 0;
	CUdeviceptr m_statsBuffer = 0;  // device-side Stats, read back each launch
	CUdeviceptr m_frameStart = 0;   // nanotime, for the device-side time budget
	CUdeviceptr m_cudaPrint = 0;    // dummy; the facility is a no-op both ends

	// The accumulator's own state. A side array indexed by node index, NOT fields on Node --
	// see the banner in remo/RemoAccum.h.
	CUdeviceptr m_nodeAccums = 0;
	uint64_t m_nodeAccumsBytes = 0;
	CUdeviceptr m_accumGlobals = 0;

	// Scratch handed to the RESET kernel in place of the source's real batch metadata.
	//
	// remolod_reset.cu unconditionally zeroes batchSizes[0 .. BATCH_STREAM_SIZE-1] and
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
	// Must equal ACCUM_BLOCK_SIZE in remolod_accum.cu, which sizes its shared staging from it.
	static constexpr int kAccumBlockSize = 256;

	// Last read-back of the accumulator's invariants. Displayed in the panel and printed by
	// --dump-frame, because tests/unit/ is empty: if these are not observable from the host
	// they are not checked at all.
	AccumGlobals m_accum = {};
	bool m_accumEnabled = true;

	uint32_t m_batchesConsumed = 0;
	uint32_t m_batchesTotal = 0;
};

}  // namespace remo
