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

// What the shell needs to know about a pipeline without knowing which one it is.
//
// The two byte-per-point numbers are deliberately separate, because one constant was
// doing both jobs and the two pull in opposite directions: raising it to describe what
// allocate() wants also raised the bar a cloud has to clear to be accepted at all, and
// lowering it to accept a cloud silently shrank every persistent buffer.
//
//   bytesPerPointEstimate     appetite   -- the coefficient allocate() sizes with, and it
//                                          clamps to whatever the budget leaves.
//   minBytesPerPointEstimate  floor      -- what the structure costs per point actually
//                                          stored. fits() refuses on this and nothing else.
//
// The floor is measured rather than guessed: SimLOD's Table 5 reports 9.1 GB for the 350M
// Morro Bay cloud (26 B/pt), and RemoBench's own 36M tree independently comes to 25.7 B/pt.
struct PipelineInfo {
	std::string id;
	std::string displayName;

	bool progressive = false;
	bool needsWholeCloudResident = false;

	// Device ring depth, in slots of PointSource::kSlotCapacity points. Zero means the
	// pipeline takes the whole cloud resident and the input term scales with the cloud;
	// non-zero means the input term is a flat ringSlots * kSlotCapacity * sizeof(Point)
	// no matter how large the cloud is. That difference is the whole point of streaming.
	uint32_t ringSlots = 0;

	double bytesPerPointEstimate = 0.0;
	double minBytesPerPointEstimate = 0.0;

	// The smallest store the pipeline will run with at all. Non-zero only for a
	// pipeline whose ingest truncates cleanly -- the progressive ones stop on
	// memCapacityReached with a valid, smaller tree, so a budget below the full
	// requirement is a smaller result rather than a failure, and refusing it would
	// throw away the run this plan exists to make possible.
	//
	// Zero means the pipeline cannot truncate: a batch builder voxelizes the whole
	// cloud or overruns its slab, so its full per-point requirement IS its floor.
	uint64_t minStoreBytes = 0;

	// Overhead that does not scale with the cloud: the momentary buffer, the node pool,
	// per-node side arrays. Part of the floor, but not per point.
	uint64_t fixedBytesEstimate = 0;
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
