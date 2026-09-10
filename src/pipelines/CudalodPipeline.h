// CudaLOD: batch LOD construction -- hierarchical counting-sort split, then voxelisation.
//
// From "GPU-Accelerated LOD Generation for Point Clouds" (Schuetz et al. 2023). The whole
// cloud must be resident in device memory first, then two cooperative kernel launches
// build the entire octree in one shot:
//
//   kernel2  split      dense 256^3 counting grid + sparse 16^3 subgrids -> depth 12
//   kernel3  voxelise   inner nodes, by one of four sampling strategies
//
// Contrast SimLOD, which inserts batches into a live tree across frames. That difference
// is the most interesting comparison this project can make, and it is also why the two
// need different things from the loader -- see PointSource.
//
// The device code is vendored UNMODIFIED (kernels/cudalod/), so the numbers stay
// verifiable against bench/reference/. Only the renderer is replaced, by a ~170-line
// selection pass feeding the shared rasteriser.
//
// Host-side changes from upstream, all of which fix things that made measurement
// unreliable rather than changing the algorithm:
//
//   - The device slab comes from DeviceBudget, not a compile-time #define. That deletes
//     a whole bug class: upstream's MAX_BUFFER_SIZE was shadowed to 2.147GB by a dead
//     include, its documented -D override never reached the compiler, and sizing it for
//     the default strategy meant pressing a strategy button overran it (strategy 2 needs
//     3.82GB against strategy 0's 1.20GB, measured).
//   - numPoints is u64 on the host with a checked narrow. Upstream is uint32_t/int
//     throughout the device side, which is one bit from overflow at 4.29B points.
//   - The render grid is computed from occupancy instead of a hardcoded 80 blocks, which
//     on an 84-SM card is under-subscribed.
//
// kernel3's grid stays at exactly numSMs. That is load-bearing, not an oversight: it
// allocates a per-block sampling grid from the slab, so more blocks than SMs would
// overrun it.

#pragma once

#include <memory>
#include <string>

#include "remo/CudaModularProgram.h"
#include "remo/ILodPipeline.h"

namespace remo {

class CudaContext;

class CudalodPipeline final : public ILodPipeline {
public:
	explicit CudalodPipeline(CudaContext& cuda);
	~CudalodPipeline() override;

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
	void readResults();
	void ensureScratch(int width, int height);

	CudaContext& m_cuda;

	// Two programs, matching upstream: the builder and the renderer, each linked with
	// lib.cu. Declared in kernels/cudalod/programs.txt.
	std::unique_ptr<CudaModularProgram> m_buildProgram;
	std::unique_ptr<CudaModularProgram> m_renderProgram;

	// --- device memory -----------------------------------------------------
	CUdeviceptr m_slab = 0;          // the LOD arena; sub-allocated on device
	uint64_t m_slabBytes = 0;
	CUdeviceptr m_results = 0;       // Results readback
	CUdeviceptr m_scratch = 0;       // render scratch (framebuffer + draw list)
	uint64_t m_scratchBytes = 0;
	CUdeviceptr m_diagnostics = 0;

	// Indirection cells the device writes its own pointers back into.
	CUdeviceptr m_numNodes = 0;      // uint32_t
	CUdeviceptr m_nodes = 0;         // Node**
	CUdeviceptr m_sorted = 0;        // Point**
	CUdeviceptr m_allocOffset = 0;   // uint64_t, watermark handed kernel2 -> kernel3
	CUdeviceptr m_debugPoints = 0;   // upstream debug-draw cells, unused but in the ABI
	CUdeviceptr m_debugLines = 0;

	CUdeviceptr m_inputPoints = 0;   // borrowed from PointSource; not owned
	uint64_t m_numPoints = 0;

	bool m_built = false;
	bool m_rebuildRequested = false;

	// Set when the previous samples stop describing the same work -- changing the
	// sampling strategy, above all. WEIGHTED_NEIGHBORHOOD voxelises ~13x slower than
	// FIRST_COME for a bit-identical tree, so pooling the two into one distribution
	// produces a median that describes neither. Honoured in build(), which is where the
	// profiler is in hand.
	bool m_clearTimingRequested = false;

	// Pipeline-owned tunable. Exposed in gui(); changing it rebuilds.
	int m_strategy = 0;  // SamplingStrategy

	PipelineStats m_stats;
	int m_blockSize = 256;

	uint64_t m_allocatedSplitting = 0;
	uint64_t m_allocatedVoxelization = 0;
};

}  // namespace remo
