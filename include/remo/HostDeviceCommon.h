#pragma once

#ifdef __CUDACC_RTC__
#include <cuda/std/cstdint>
using cuda::std::int32_t;
using cuda::std::uint32_t;
using cuda::std::uint64_t;
using cuda::std::uint8_t;
#else
#include <cstdint>
#endif

namespace remo {

struct Point {
	float x, y, z;
	uint32_t color;
};

struct mat4 {
	float rows[4][4];
};

struct vec3f {
	float x, y, z;
};

struct SharedUniforms {
	float width;
	float height;
	float fovyRad;
	float time;

	mat4 transform;
	mat4 view;
	mat4 proj;

	mat4 transformFrozen;
	mat4 transformFrozenInv;

	vec3f boxMin;
	vec3f boxMax;

	uint64_t frameCounter;

	float lodPixelBudget;
	float minNodeSize;
	float lodScale;

	int32_t pointSize;
	int32_t colorMode;
	float edlStrength;

	int32_t doUpdateVisibility;
	int32_t useHighQualityShading;
	int32_t enableEDL;
	int32_t showBoundingBox;
	int32_t showPoints;
	int32_t pad0;
};

enum ColorMode : int32_t {
	COLOR_RGB = 0,
	COLOR_BY_NODE = 1,
	COLOR_BY_LOD = 2,
	COLOR_WHITE = 3,
};

struct RenderArgs {
	SharedUniforms uniforms;

	uint32_t* scratch;
	uint64_t scratchCapacity;

	uint64_t surface;
};

struct DeviceDiagnostics {
	uint64_t allocHighWater;
	uint64_t allocCapacity;
	uint32_t allocOverflow;
	uint32_t nodePoolOverflow;
	uint32_t drawListOverflow;

	uint32_t drawItems;
	uint64_t drawSamples;
};

// Intra-kernel phase timing. See plans/05_HardCodedTest.md.
//
// GpuScope cannot see inside a launch: cuEventRecord is stream-ordered and
// kernel_construct is one cooperative launch. These marks are the only way to
// attribute time to the phases within it. They are written through REMO_MARK
// (kernels/shared/remo_prelude.cuh), which compiles to nothing unless
// REMO_PROFILE is defined, so a default build does none of the timing work. The
// sink pointer is passed to the kernel either way, so the signature does not vary
// with the build -- see plans/05_HardCodedTest.md on why that matters.

// A mark is a grid-wide phase boundary ONLY when it sits immediately after a
// grid.sync(). A mark that cannot be placed after a barrier must not be placed.

constexpr uint32_t REMO_MAX_MARKS = 256;
constexpr uint32_t REMO_MAX_EXPAND_ITERS = 20;  // == the loop bound in expand()

// The phase that BEGINS at a mark. Differencing consecutive marks gives each
// phase's duration; kPhaseBatchEnd terminates a batch and has no duration.
enum ConstructPhase : uint32_t {
	kPhaseBatchBegin = 0,
	kPhaseExpand,
	kPhaseVoxelSampling,
	kPhaseAllocPointChunks,
	kPhaseAllocVoxelChunks,
	kPhaseInsertPoints,
	kPhaseInsertVoxels,
	kPhaseBatchEnd,
	kNumConstructPhases
};

struct TimelineMark {
	uint32_t phase;
	uint32_t pad;
	uint64_t ns;
};

struct DeviceTimeline {
	uint32_t numMarks;
	uint32_t overflow;  // more marks than REMO_MAX_MARKS
	TimelineMark marks[REMO_MAX_MARKS];

	// What milliseconds alone cannot answer: predicted depth would delete
	// expand iterations 2..N and the spill re-descent, not all of expand.
	//
	// Everything from here down is the COUNTER BLOCK. The mark array above is
	// written only under REMO_PROFILE; these are written, zeroed and read back in
	// every variant, because nodePoolOverflow is a safety report and not a
	// measurement. kernel_construct is not given a DeviceDiagnostics* to carry
	// them instead: an arity change to that kernel is the defect CLAUDE.md
	// records as having silently built no tree for a whole commit.
	uint32_t batches;      // batches folded into this launch
	uint32_t expandIters;  // summed over batches
	uint32_t nodesSplit;   // summed over batches
	uint32_t maxPointsPerNode;  // largest leaf, at the end of this launch
	uint64_t spilledPoints;                        // summed over batches
	uint64_t expandIterNs[REMO_MAX_EXPAND_ITERS];  // ns by iteration index

	// The fixed-depth arm (-DREMO_FIXED_DEPTH=N, plans/07_HardCodingExpandStage.md).
	// Named fields rather than reused expandIterNs[] slots: two meanings on one
	// field reads fine today and misleads a year from now.
	uint64_t bucketNs;       // summed over batches
	uint64_t pyramidNs;
	uint64_t materialiseNs;
	uint64_t seedNs;
	uint32_t occupiedCells;     // depth-D cells that took points, summed over batches
	uint32_t nodePoolOverflow;  // set in BOTH variants; see doSplitting

	// Leaves with counter < numPoints at the end of the launch. allocatePointChunks
	// sizes a leaf's chunk list from counter alone, so one of these is an
	// under-allocated leaf -- and the only complaint the kernel would otherwise
	// make goes through CudaPrint::print(), which returns on its first line.
	uint32_t counterUnderflows;
	uint32_t pad2;
};

static_assert(sizeof(TimelineMark) == 16, "TimelineMark layout changed");
static_assert(sizeof(DeviceTimeline) == 4336,
              "DeviceTimeline layout changed; host and device must agree");

}
