// The ONLY header shared between host C++ and NVRTC-compiled device code.
//
// Both research repos have an equivalent (SimLOD's HostDeviceInterface.h,
// CudaLOD's common.h) and both work well; the shape here follows them. Two rules
// keep it working:
//
//   1. Nothing in here may include a host-only header. NVRTC has no libstdc++.
//   2. Layout must be identical on both sides. No virtuals, no std:: types, and
//      explicit padding where alignment would otherwise differ.
//
// SharedUniforms is the important design decision. It holds only knobs that are
// IDENTICAL for every pipeline on every frame -- camera, viewport, LOD budget,
// shading toggles. That is what makes an A/B between two pipelines attributable to
// the LOD algorithm rather than to one of them being handed a different camera or
// a different point size. Pipeline-specific tunables live in the pipeline, not
// here.
//
// A note on what upstream got wrong here, because it matters for a comparison
// tool: SimLOD's Uniforms carries LOD, doProgressive, colorWhite, updateStats,
// enableEDL and edlStrength, and NO kernel reads any of them -- EDL hardcodes 0.4
// internally. Six GUI controls that silently do nothing. In a tool whose purpose
// is measurement, a knob that does nothing is not a cosmetic bug, it invalidates
// experiments. Every field below is read by device code, and the plan is to keep
// it that way by grepping the kernel sources for each field name.

#pragma once

#ifdef __CUDACC_RTC__
// NVRTC has no libstdc++. Pull the fixed-width types from CCCL, which is on the
// NVRTC include path (both $CUDA/include and $CUDA/include/cccl are passed, the
// latter because CUDA 13 relocated these headers).
//
// Do NOT be tempted to hand-write `typedef unsigned int uint32_t` here instead:
// cooperative_groups pulls in <cuda/std/cstdint> transitively, and a conflicting
// typedef is exactly the breakage that patches/cudalod-linux-port.patch has to fix
// in two of CudaLOD's kernel headers.
#include <cuda/std/cstdint>
using cuda::std::int32_t;
using cuda::std::uint32_t;
using cuda::std::uint64_t;
using cuda::std::uint8_t;
#else
#include <cstdint>
#endif

namespace remo {

// ---------------------------------------------------------------------------
// Point format
//
// 16 bytes: three float32 coordinates plus packed RGBA8. Identical in SimLOD and
// CudaLOD, and identical in the .simlod on-disk format, so it is the natural
// interchange type -- and keeping it means throughput numbers stay comparable to
// the published figures.
//
// Coordinates are PRE-TRANSLATED on the host so the global bounding-box minimum
// sits at the origin. Device code therefore never needs doubles. The f64
// translation that was applied lives in CloudMeta on the host side; it is never
// stored in float.
// ---------------------------------------------------------------------------
struct Point {
	float x, y, z;
	uint32_t color;  // RGBA8, alpha in the high byte
};

// Row-major, to match how both repos' device code multiplies. Host side transposes
// glm's column-major matrices on the way in -- see getUniforms().
struct mat4 {
	float rows[4][4];
};

struct vec3f {
	float x, y, z;
};

// ---------------------------------------------------------------------------
// Cross-pipeline uniforms
// ---------------------------------------------------------------------------
struct SharedUniforms {
	// Viewport / camera
	float width;
	float height;
	float fovyRad;
	float time;

	mat4 transform;      // proj * view, live camera
	mat4 view;
	mat4 proj;

	// LOD selection is evaluated against a FROZEN transform when
	// doUpdateVisibility is off, so you can lock the cut and fly the camera to
	// inspect where a pipeline chose its boundaries. This is the single best
	// debugging feature in either upstream repo and is worth keeping.
	mat4 transformFrozen;
	mat4 transformFrozenInv;

	// Post-translation bounds. boxMin is the origin by construction.
	vec3f boxMin;
	vec3f boxMax;

	uint64_t frameCounter;

	// --- LOD ---------------------------------------------------------------
	// The canonical currency is the projected screen-space extent of a node, in
	// PIXELS. This exists because the two upstream metrics are not comparable:
	// SimLOD tests `dx > 2 * minNodeSize` in WORLD units, so the same slider
	// position means a different cut on every dataset; CudaLOD tests
	// `cubeSize / distance < 1 - 0.97 * LOD`, which is angular but not calibrated
	// to the viewport. Without one shared rule, "both at LOD 0.5" is not the same
	// cut, and every render-side measurement is quietly incomparable.
	//
	// The two native metrics remain selectable per pipeline at NVRTC compile time
	// (REMO_LOD_SIMLOD_NATIVE / REMO_LOD_CUDALOD_NATIVE) so a port can be
	// validated against its published behaviour before being switched over.
	float lodPixelBudget;  // default 128.0
	float minNodeSize;     // SimLOD-native fallback, world units
	float lodScale;        // CudaLOD-native fallback, 0..1

	// --- Shading -----------------------------------------------------------
	int32_t pointSize;
	int32_t colorMode;  // see ColorMode
	float edlStrength;

	// Packed as int32 rather than bool: bool has no guaranteed size across the
	// host/NVRTC boundary, and a layout mismatch here is a silent disaster.
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

// ---------------------------------------------------------------------------
// Render arguments
//
// One struct instead of upstream's seven positional void* args. Adding a scratch
// buffer to an experiment then touches one header rather than every launch site,
// which removes the most tedious friction in SimLOD's codebase.
// ---------------------------------------------------------------------------
struct RenderArgs {
	SharedUniforms uniforms;

	// Per-launch scratch for the framebuffer, accumulation targets and draw list.
	// scratchCapacity is NOT decoration: the device-side bump allocator checks
	// against it. Upstream has no such check, and as a direct result SimLOD's
	// momentary allocator hands out ~409MB from a 300MB buffer -- chunkQueue's base
	// pointer lands entirely outside the allocation and its writes go into whatever
	// cuMemAlloc returned next. That only "works" because the preceding arrays are
	// never filled to capacity.
	uint32_t* scratch;
	uint64_t scratchCapacity;

	// GL colour attachment, registered once and reused (upstream re-registers the
	// image every single frame).
	uint64_t surface;  // cudaSurfaceObject_t, opaque here to avoid a CUDA include
};

// ---------------------------------------------------------------------------
// Device -> host diagnostics
//
// Read back every frame. These exist so that overrunning a buffer or exhausting a
// node pool reports itself instead of corrupting memory and being discovered as a
// visual artefact three days later.
// ---------------------------------------------------------------------------
struct DeviceDiagnostics {
	uint64_t allocHighWater;   // peak bump-allocator offset
	uint64_t allocCapacity;
	uint32_t allocOverflow;    // an allocation exceeded capacity
	uint32_t nodePoolOverflow; // node pool bump index hit its limit
	uint32_t drawListOverflow; // more visible nodes than the draw list can hold

	// How many DrawItems the selection pass emitted this frame, and how many samples they
	// reference. The first question to ask when a pipeline renders nothing is whether
	// selection produced anything at all, and without this it cannot be answered from the
	// host.
	uint32_t drawItems;
	uint64_t drawSamples;
};

}  // namespace remo
