// Adapted from SimLOD: modules/progressive_octree/utils.h.cu
// Upstream: https://github.com/m-schuetz/SimLOD @ fa7891613c138bd41775ca72a47cd89e32a5a647
// Copyright 2023 Markus Schuetz and Lukas Herzberger -- MIT (see THIRD_PARTY.md)
//
// Device-side bump allocators.
//
// ###########################################################################
// #                                                                         #
// #  READ THIS BEFORE USING RemoAllocator IN A NEW PIPELINE.                #
// #                                                                         #
// #  RemoAllocator IS NOT ATOMIC, AND THAT IS DELIBERATE.                   #
// #                                                                         #
// #  Every thread constructs it from the same base pointer and walks the     #
// #  IDENTICAL allocation sequence, so all threads independently arrive at   #
// #  identical pointers with zero atomics and zero broadcast. It is an       #
// #  elegant trick and it is load-bearing for performance.                   #
// #                                                                         #
// #  The price is a hard requirement: EVERY THREAD MUST EXECUTE EVERY        #
// #  alloc() CALL, IN THE SAME ORDER. An allocation inside a branch that     #
// #  only some threads take silently desynchronises the pointers, and the    #
// #  result is threads writing over each other's buffers.                    #
// #                                                                         #
// #  So: no alloc() inside `if(threadIdx.x == 0)`, no alloc() behind a       #
// #  data-dependent condition, no alloc() in a loop with a varying trip      #
// #  count. Hoist the allocation out and branch around the *use*.            #
// #                                                                         #
// #  Compile with -DREMO_ALLOC_DEBUG=1 to have this checked at runtime.      #
// #                                                                         #
// ###########################################################################
//
// Changes from upstream, both non-negotiable for a tool whose purpose is to let you
// change tunables and see what happens:
//
//   1. A capacity, and a bounds check. Upstream has neither, and the consequence is
//      live in the shipped code: SimLOD's momentary allocator hands out ~409MB from
//      a 300MB buffer. `chunkQueue`'s base pointer lands ENTIRELY OUTSIDE the
//      allocation, so its writes go into whatever cuMemAlloc returned next. It only
//      "works" because the preceding backlog arrays are never filled near capacity
//      -- i.e. it is one tunable change away from silent corruption. CudaLOD has the
//      same shape, and there a 4GB slab overruns as soon as you press a
//      sampling-strategy button that needs 3.8GB.
//
//   2. Overflow REPORTS itself, into DeviceDiagnostics, which the host reads back
//      every frame. An out-of-range allocation returns nullptr rather than a wild
//      pointer, so the failure mode is a visible diagnostic instead of memory
//      corruption discovered days later as a visual artefact.

#pragma once

#include "shared/remo_prelude.cuh"

namespace remo {

// 16-byte alignment, matching upstream, so float4/uint64 loads stay aligned.
inline uint64_t remoAlignUp16(uint64_t offset) {
	const uint64_t rem = offset % 16ull;
	return rem == 0ull ? offset : offset + (16ull - rem);
}

// Per-launch scratch allocator. See the banner above.
struct RemoAllocator {
	uint8_t* buffer;
	uint64_t offset;
	uint64_t capacity;
	DeviceDiagnostics* diag;  // may be nullptr

	RemoAllocator(void* base, uint64_t capacityBytes,
	              DeviceDiagnostics* diagnostics = nullptr)
		: buffer(reinterpret_cast<uint8_t*>(base)),
		  offset(0),
		  capacity(capacityBytes),
		  diag(diagnostics) {}

	template <typename T>
	T alloc(uint64_t size) {
		const uint64_t start = offset;
		const uint64_t end = remoAlignUp16(start + size);

		if (end > capacity) {
			// Report once per launch; thread 0 is enough and avoids an atomic storm
			// when every thread notices the same overflow simultaneously.
			if (diag != nullptr && threadIdx.x == 0 && blockIdx.x == 0) {
				diag->allocOverflow = 1u;
				diag->allocCapacity = capacity;
				diag->allocHighWater = end;
			}
			// Still advance, so that all threads stay in lockstep and subsequent
			// allocations agree with each other -- desynchronising here would turn
			// one reported bug into an unreportable one.
			offset = end;
			return nullptr;
		}

		offset = end;

		if (diag != nullptr && threadIdx.x == 0 && blockIdx.x == 0) {
			if (end > diag->allocHighWater) diag->allocHighWater = end;
			diag->allocCapacity = capacity;
		}

		return reinterpret_cast<T>(buffer + start);
	}

	// Reclaim scratch by rewinding, the way upstream's split phase does. Same
	// uniform-control-flow rule applies.
	void rewindTo(uint64_t savedOffset) { offset = savedOffset; }
};

// Atomic bump allocator for state that must persist ACROSS launches (the LOD
// structure itself). Lives inside the buffer it manages, so the host only has to
// hand over one pointer.
struct RemoAllocatorGlobal {
	uint64_t offset;
	uint64_t capacity;
	uint8_t* buffer;
	uint32_t overflow;
	uint32_t pad0;

	// Called once, single-threaded, from a reset kernel.
	void init(void* base, uint64_t capacityBytes) {
		buffer = reinterpret_cast<uint8_t*>(base);
		capacity = capacityBytes;
		// Skip past this header so it is never handed out.
		offset = remoAlignUp16(sizeof(RemoAllocatorGlobal));
		overflow = 0u;
	}

	template <typename T>
	T alloc(uint64_t size) {
		const uint64_t aligned = remoAlignUp16(size);
		const uint64_t start =
			atomicAdd(reinterpret_cast<unsigned long long*>(&offset),
			          static_cast<unsigned long long>(aligned));
		if (start + aligned > capacity) {
			overflow = 1u;
			return nullptr;  // callers MUST check; see remo_prelude.cuh
		}
		return reinterpret_cast<T>(buffer + start);
	}
};

}  // namespace remo
