// Common prelude for every RemoBench kernel module.
//
// Include this first from any .cu under kernels/. It pulls in the host/device
// contract, cooperative groups, and the grid-stride helper, and brings the remo
// namespace into scope so kernel code can write `Point` rather than `remo::Point`.
//
// Reminder about the compile environment, because it surprises everyone: these
// files are compiled by NVRTC with -default-device, so unannotated functions are
// implicitly __device__ and the sources carry no __device__ markers at all. They
// read like plain C++ and will NOT compile under nvcc as-is. That is inherited from
// upstream and kept, because it genuinely is nicer to author.

#pragma once

#include <cooperative_groups.h>

#include "remo/HostDeviceCommon.h"

namespace cg = cooperative_groups;

// Deliberately NO `using remo::Point;` and friends at global scope.
//
// A ported pipeline may bring its own type of the same name: CudaLOD's device headers
// define a global `Point` and `vec3` of their own, and hoisting ours alongside them is an
// ambiguity error at the first use. Since those types are layout-identical, the port
// works fine by qualifying -- but only if this header stays out of the global namespace.
//
// So: a kernel that owns its whole translation unit says `using namespace remo;` (see
// kernels/remolod/remolod_render.cu); one sharing scope with foreign headers qualifies or
// imports selectively (see kernels/cudalod/cudalod_render.cu).

namespace remo {

// ---------------------------------------------------------------------------
// Grid-stride iteration
//
// Note this is a block-contiguous partition, not a classic grid-stride loop: each
// thread handles a run of consecutive indices. That is coalescing-hostile for
// streaming reads but harmless for the atomic-heavy traversal workloads it was
// written for, and it is what upstream's numbers were measured with -- so it is
// kept as-is to avoid changing the thing being compared. Use processRangeStrided
// for genuinely bandwidth-bound passes.
// ---------------------------------------------------------------------------
template <typename Fn>
void processRange(uint64_t first, uint64_t last, Fn&& fn) {
	auto grid = cg::this_grid();
	const uint64_t count = last > first ? last - first : 0ull;
	const uint64_t threads = grid.num_threads();
	const uint64_t perThread = (count + threads - 1ull) / threads;
	const uint64_t start = first + grid.thread_rank() * perThread;
	const uint64_t end = start + perThread < last ? start + perThread : last;
	for (uint64_t i = start; i < end; ++i) fn(i);
}

template <typename Fn>
void processRange(uint64_t count, Fn&& fn) {
	processRange(0ull, count, fn);
}

// Coalesced variant: consecutive threads touch consecutive elements. Prefer this
// for anything that streams points.
template <typename Fn>
void processRangeStrided(uint64_t count, Fn&& fn) {
	auto grid = cg::this_grid();
	for (uint64_t i = grid.thread_rank(); i < count; i += grid.num_threads()) {
		fn(i);
	}
}

// Nanosecond clock, for the device-side time budgets that make progressive
// construction bound its own frame cost.
inline uint64_t remoNanotime() {
	uint64_t ns;
	asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ns));
	return ns;
}

// ---------------------------------------------------------------------------
// Colour helpers
// ---------------------------------------------------------------------------
inline uint32_t remoPackRGBA(uint32_t r, uint32_t g, uint32_t b, uint32_t a) {
	return r | (g << 8) | (b << 16) | (a << 24);
}

// Cheap integer hash, for colour-by-node / colour-by-LOD debug shading and the octree
// wireframe.
//
// Two things here are corrections, not decoration, and both are about a debug view being
// USELESS rather than merely ugly when it goes wrong:
//
//   1. The key is OFFSET before hashing. A multiply-xor hash maps 0 to 0, so key 0 came
//      out pure black -- and key 0 is not a rare case: `remolod` reports every slice as
//      level 0, so colour-by-LOD rendered the entire control condition invisible, and
//      CudaLOD holds the root unconditionally visible, so its outermost wireframe cube
//      was always the one you could not see.
//
//   2. Channels are lifted into [96,255]. An unconstrained hash produces near-black
//      colours for ~1 in 20 keys, which on the dark clear colour are indistinguishable
//      from nothing being drawn there. Some hue distinguishability is traded for the
//      guarantee that every node is actually visible; for a debug view that is the right
//      way round.
//
// Exact colours therefore differ from earlier builds. Nothing measured depends on them
// (bench/reference/ holds structural counts, not images), but a screenshot from before
// this change will not match one from after.
inline uint32_t remoHashColor(uint64_t key) {
	uint64_t h = (key + 0x9E3779B97F4A7C15ull) * 0x9E3779B97F4A7C15ull;
	h ^= h >> 29;
	h *= 0xBF58476D1CE4E5B9ull;
	h ^= h >> 32;

	// byte -> [96, 255]
	auto channel = [](uint64_t bits) {
		return 96u + ((static_cast<uint32_t>(bits) & 0xFFu) * 159u) / 255u;
	};
	return remoPackRGBA(channel(h), channel(h >> 8), channel(h >> 16), 255u);
}

}  // namespace remo
