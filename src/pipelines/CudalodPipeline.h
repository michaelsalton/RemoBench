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

	std::unique_ptr<CudaModularProgram> m_buildProgram;
	std::unique_ptr<CudaModularProgram> m_renderProgram;

	CUdeviceptr m_slab = 0;
	uint64_t m_slabBytes = 0;
	CUdeviceptr m_results = 0;
	CUdeviceptr m_scratch = 0;
	uint64_t m_scratchBytes = 0;
	CUdeviceptr m_diagnostics = 0;

	CUdeviceptr m_numNodes = 0;
	CUdeviceptr m_nodes = 0;
	CUdeviceptr m_sorted = 0;
	CUdeviceptr m_allocOffset = 0;
	CUdeviceptr m_debugPoints = 0;
	CUdeviceptr m_debugLines = 0;

	CUdeviceptr m_inputPoints = 0;
	uint64_t m_numPoints = 0;

	bool m_built = false;
	bool m_rebuildRequested = false;

	bool m_clearTimingRequested = false;

	int m_strategy = 0;

	PipelineStats m_stats;
	int m_blockSize = 256;

	uint64_t m_allocatedSplitting = 0;
	uint64_t m_allocatedVoxelization = 0;
};

}
