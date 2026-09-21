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
	void guiControls() override;
	void guiStats(const GpuProfiler& profiler) override;

private:
	void readStats();
	void ensureScratch(int width, int height);
	void fillUniforms(const FrameContext& frame, void* outUniforms) const;

	CudaContext& m_cuda;

	std::unique_ptr<CudaModularProgram> m_resetProgram;
	std::unique_ptr<CudaModularProgram> m_constructProgram;
	std::unique_ptr<CudaModularProgram> m_renderProgram;

	CUdeviceptr m_momentary = 0;
	uint64_t m_momentaryBytes = 0;
	CUdeviceptr m_persistent = 0;
	uint64_t m_persistentBytes = 0;
	CUdeviceptr m_nodes = 0;
	uint64_t m_nodesBytes = 0;
	CUdeviceptr m_statsBuffer = 0;
	CUdeviceptr m_frameStart = 0;
	CUdeviceptr m_cudaPrint = 0;

	CUdeviceptr m_resetBatchSizes = 0;
	CUdeviceptr m_resetNumUploaded = 0;
	CUdeviceptr m_scratch = 0;
	uint64_t m_scratchBytes = 0;
	CUdeviceptr m_diagnostics = 0;

	bool m_needsReset = true;
	bool m_complete = false;
	uint64_t m_numPoints = 0;

	PipelineStats m_stats;
	int m_blockSize = 256;

	uint32_t m_batchesConsumed = 0;
	uint32_t m_batchesTotal = 0;

	// Reported once, not once per frame: a ring the kernel's addressing does not match
	// stops the build for as long as it is wrong.
	bool m_ringMismatchReported = false;
};

}
