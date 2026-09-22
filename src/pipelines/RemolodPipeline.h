#pragma once

#include <memory>
#include <string>

#include "remo/CudaModularProgram.h"
#include "remo/HostDeviceCommon.h"
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
	void guiControls() override;
	void guiStats(const GpuProfiler& profiler) override;
	std::vector<std::string> diagnostics() const override;

	void setAccumEnabled(bool on) { m_accumEnabled = on; }
	bool accumEnabled() const { return m_accumEnabled; }

	// Must be set before initPrograms(): it selects the -DREMO_PROFILE variant
	// of the construct kernel, which is what emits the phase marks.
	void setPhaseTimings(bool on) { m_phaseTimings = on; }

	// Likewise before initPrograms(): the depth is a compile-time constant, so it
	// selects a compiled variant of the construct kernel and cannot be a GUI
	// control. 0 leaves the default, iterative expand() in place.
	// See plans/07_HardCodingExpandStage.md.
	void setFixedDepth(int depth) { m_fixedDepth = depth; }
	int fixedDepth() const { return m_fixedDepth; }

private:
	void readStats();
	void ensureScratch(int width, int height);
	void fillUniforms(const FrameContext& frame, void* outUniforms) const;

	void runAccumulator(const FrameContext& frame);
	void readTimeline(const FrameContext& frame);

	CudaContext& m_cuda;

	std::unique_ptr<CudaModularProgram> m_resetProgram;
	std::unique_ptr<CudaModularProgram> m_constructProgram;
	std::unique_ptr<CudaModularProgram> m_accumProgram;
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

	CUdeviceptr m_nodeAccums = 0;
	uint64_t m_nodeAccumsBytes = 0;
	CUdeviceptr m_accumGlobals = 0;

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
	static constexpr int kAccumBlockSize = 256;

	AccumGlobals m_accum = {};
	bool m_accumEnabled = true;

	// Intra-kernel phase timing. See plans/05_HardCodedTest.md.
	CUdeviceptr m_timeline = 0;
	bool m_phaseTimings = false;
	double m_phaseMs[kNumConstructPhases] = {};
	double m_expandIterMs[REMO_MAX_EXPAND_ITERS] = {};
	double m_constructMsSeen = 0.0;
	uint64_t m_phaseBatches = 0;
	uint64_t m_phaseExpandIters = 0;
	uint64_t m_phaseSpilledPoints = 0;
	uint64_t m_phaseNodesSplit = 0;
	uint32_t m_phaseOverflow = 0;

	// The fixed-depth arm. m_nodePoolOverflow and m_occupiedCells come off the
	// timeline's counter block, which is read in every variant.
	int m_fixedDepth = 0;
	double m_fixedBucketMs = 0.0;
	double m_fixedPyramidMs = 0.0;
	double m_fixedMaterialiseMs = 0.0;
	double m_fixedSeedMs = 0.0;
	uint64_t m_occupiedCells = 0;
	bool m_nodePoolOverflow = false;
	uint32_t m_counterUnderflows = 0;

	uint32_t m_batchesConsumed = 0;
	uint32_t m_batchesTotal = 0;

	// Reported once, not once per frame: a ring the kernel's addressing does not match
	// stops the build for as long as it is wrong.
	bool m_ringMismatchReported = false;
};

}
