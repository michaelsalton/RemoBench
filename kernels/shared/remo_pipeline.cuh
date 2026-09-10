// The one header a pipeline's device code needs to include.
//
//   #include "shared/remo_pipeline.cuh"
//   using namespace remo;
//
// It pulls in, in dependency order: the host/device contract and grid helpers, the
// bounds-checked bump allocators, device math and frustum culling, the shared
// software rasteriser, and the octree wireframe overlay.
//
// Sharing happens HERE, at NVRTC compile time, rather than by dispatching inside a
// kernel. That is not a stylistic preference -- see the note in
// include/remo/ILodPipeline.h: cooperative launches cannot be composed, and the
// non-atomic allocator forbids branch-dependent allocation. Textual inclusion gives
// every pipeline the same rasteriser with zero runtime dispatch and no
// uniform-control-flow hazard.

#pragma once

#include "shared/remo_prelude.cuh"
#include "shared/remo_alloc.cuh"
#include "shared/remo_math.cuh"
#include "shared/remo_framebuffer.cuh"
#include "shared/remo_draw.cuh"
#include "shared/remo_lines.cuh"
