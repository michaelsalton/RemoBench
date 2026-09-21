#pragma once

#include <memory>
#include <string>

#include "remo/CudaModularProgram.h"
#include "remo/ILodPipeline.h"

namespace remo {

class CudaContext;

class FlatPipeline final : public ILodPipeline {
public:
	explicit FlatPipeline(CudaContext& cuda);
	~FlatPipeline() override;

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
	CudaContext& m_cuda;
	std::unique_ptr<CudaModularProgram> m_program;

	CUdeviceptr m_scratch = 0;
	uint64_t m_scratchBytes = 0;

	CUdeviceptr m_diagnostics = 0;

	CUdeviceptr m_points = 0;
	uint64_t m_numPoints = 0;

	PipelineStats m_stats;
	int m_blockSize = 256;
};

}
