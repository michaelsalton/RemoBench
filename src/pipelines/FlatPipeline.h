// The `flat` pipeline: no LOD, draw every point.
//
// The control condition. See kernels/flat/flat_render.cu for why it earns its keep
// rather than being a placeholder.
//
// It is also the reference implementation of ILodPipeline, and deliberately the
// smallest possible one: build() does nothing, there is no data structure, and the
// only device memory it owns is render scratch. Anyone writing a custom pipeline
// should be able to read this in one sitting.

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

	// Set by build() once the source is resident, since render() is not handed the
	// source.
	CUdeviceptr m_points = 0;
	uint64_t m_numPoints = 0;

	PipelineStats m_stats;
	int m_blockSize = 256;
};

}  // namespace remo
