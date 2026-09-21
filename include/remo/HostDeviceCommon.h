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

}
