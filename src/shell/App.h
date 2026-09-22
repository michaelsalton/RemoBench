#pragma once

#include <memory>
#include <string>
#include <vector>

#include "remo/CudaContext.h"
#include "remo/GLInterop.h"
#include "remo/GpuProfiler.h"
#include "remo/HostDeviceCommon.h"
#include "remo/ILodPipeline.h"
#include "remo/PipelineRegistry.h"
#include "remo/PointSource.h"
#include "shell/GLRenderer.h"

namespace remo {

struct AppOptions {
	std::vector<std::string> files;
	uint64_t syntheticPoints = 0;
	std::string pipeline = "flat";
	int width = 1600;
	int height = 900;
	bool strictTiming = false;

	// Overrides computeBudget() when non-zero, so a benchmark run stops depending on
	// what else happened to be on the GPU. The budget is the one number every memory
	// figure is relative to, so a capture that does not pin it is not comparable.
	size_t deviceBudgetBytes = 0;

	bool remolodNoAccum = false;
	bool remolodPhaseTimings = false;

	// 0 = off, the default iterative expand(). 1..8 selects the fixed-depth arm,
	// which is a compiled variant of the construct kernel and so cannot be a GUI
	// control. See plans/07_HardCodingExpandStage.md.
	int remolodFixedDepth = 0;

	std::string dumpFramePath;
	int dumpAfterFrames = 8;
	bool dumpIncludeGui = false;

	std::string switchToPipeline;
	int switchAfterFrames = 0;

	bool showBoundingBox = false;
	bool hidePoints = false;
};

struct SharedSettings {
	float lodPixelBudget = 128.0f;
	float minNodeSize = 64.0f;
	float lodScale = 0.5f;
	int pointSize = 1;
	int colorMode = COLOR_RGB;
	bool doUpdateVisibility = true;
	bool useHighQualityShading = false;
	bool enableEDL = true;
	float edlStrength = 0.4f;
	bool showBoundingBox = false;
	bool showPoints = true;
};

struct FrameHistory {
	static constexpr int kCapacity = 240;

	float ms[kCapacity] = {};
	int head = 0;
	int count = 0;

	void add(float value) {
		ms[head] = value;
		head = (head + 1) % kCapacity;
		if (count < kCapacity) ++count;
	}

	void clear() {
		head = 0;
		count = 0;
	}
};

struct DatasetEntry {
	std::string path;
	std::string label;
	uint64_t bytes = 0;
	uint64_t numPoints = 0;
	bool supported = false;
	std::string note;
};
std::vector<DatasetEntry> scanDatasetDir(const std::string& dir);

class App {
public:
	App();
	~App();

	bool init(const AppOptions& options, std::string* err);
	int run();

private:
	void registerPipelines();
	bool loadCloud(const std::vector<std::string>& files, std::string* err);
	bool loadSynthetic(uint64_t numPoints, std::string* err);
	bool activateCloud(std::string* err);

	SharedUniforms buildUniforms() const;
	DeviceBudget computeBudget() const;

	void drawGui();
	void drawControlPanel();
	void drawStatsPanel();

	bool dumpFrame(const std::string& path);

	AppOptions m_options;
	SharedSettings m_settings;

	std::unique_ptr<CudaContext> m_cuda;
	GLRenderer m_renderer;
	GLInterop m_interop;

	PipelineRegistry m_registry;
	std::unique_ptr<PointSource> m_source;
	CloudMeta m_meta;
	DeviceBudget m_budget;

	GpuProfiler m_profiler;

	ScopeStats m_frameTimeStats;
	FrameHistory m_frameHistory;

	std::string m_status;
	bool m_statusIsError = false;

	// A cloud or pipeline named on the command line that never activated. Kept so the
	// process can exit non-zero: a refusal used to leave a window with no pipeline and
	// still exit 0, which made every gate change unverifiable from a script.
	bool m_startupFailed = false;
	std::string m_startupError;

	std::vector<DatasetEntry> m_datasets;
	std::string m_datasetDir = "data";
	bool m_datasetsScanned = false;
	int m_selectedDataset = -1;

	std::string m_pendingLoadPath;
	int m_pendingLoadDelayFrames = 0;

	void scanDatasets();
	void applyPendingLoad();
	void requestLoad(const std::string& pathOrSynthetic, const std::string& label);

	std::string m_pendingPipeline;

	void applyPendingPipelineSwitch();

	uint64_t m_frameCounter = 0;
	SharedUniforms m_frozen = {};
	bool m_hasFrozen = false;
};

}
