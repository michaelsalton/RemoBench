#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <cuda.h>

#include "remo/GpuProfiler.h"
#include "remo/HostDeviceCommon.h"

namespace remo {

class PointSource;
struct CloudMeta;

struct PipelineInfo {
	std::string id;
	std::string displayName;

	bool progressive = false;
	bool needsWholeCloudResident = false;

	double bytesPerPointEstimate = 0.0;
};

struct DeviceBudget {
	size_t vramTotal = 0;
	size_t vramFreeAtStartup = 0;
	size_t bytes = 0;
};

struct RenderTargets {
	uint64_t surface = 0;
	int width = 0;
	int height = 0;
};

struct FrameContext {
	SharedUniforms uniforms;
	RenderTargets targets;
	CUstream stream = nullptr;
	int numSMs = 0;

	bool strictTiming = false;

	GpuProfiler* profiler = nullptr;
};

struct PipelineStats {
	uint64_t numPoints = 0;
	uint64_t numVoxels = 0;
	uint64_t numPointsIngested = 0;

	uint32_t numNodes = 0;
	uint32_t numInner = 0;
	uint32_t numLeaves = 0;
	uint32_t maxPointsPerNode = 0;

	uint32_t numVisibleNodes = 0;
	uint64_t numVisiblePoints = 0;
	uint64_t numVisibleVoxels = 0;

	uint64_t bytesAllocated = 0;
	uint64_t bytesHighWater = 0;
	uint64_t bytesCapacity = 0;

	uint64_t samplesPerLevel[24] = {};

	bool memCapacityReached = false;
	bool nodeCapacityReached = false;
	bool allocOverflow = false;
};

struct TimingScopes {
	std::vector<std::string> build;
	std::string render;
};

class ILodPipeline {
public:
	virtual ~ILodPipeline() = default;

	virtual PipelineInfo info() const = 0;

	virtual bool initPrograms(std::string* err) = 0;

	virtual bool allocate(const CloudMeta& meta, const DeviceBudget& budget,
	                      std::string* err) = 0;

	virtual void release() = 0;

	virtual void reset() = 0;

	virtual bool build(PointSource& source, const FrameContext& frame) = 0;

	virtual void render(const FrameContext& frame) = 0;

	virtual const PipelineStats& stats() const = 0;

	virtual TimingScopes timingScopes() const = 0;

	virtual void guiControls() {}

	virtual void guiStats(const GpuProfiler& profiler) { (void)profiler; }

	virtual std::vector<std::string> diagnostics() const { return {}; }
};

struct BuildTotals {
	double ms = 0.0;
	uint64_t launches = 0;
	bool measured = false;
};

inline BuildTotals buildTotals(const GpuProfiler& profiler,
                               const TimingScopes& scopes) {
	BuildTotals totals;
	for (const std::string& name : scopes.build) {
		const ScopeStats* s = profiler.find(name);
		if (!s) continue;
		totals.ms += s->total();
		totals.launches += s->count();
		totals.measured = true;
	}
	return totals;
}

using PipelineFactory = std::function<std::unique_ptr<ILodPipeline>()>;

}
