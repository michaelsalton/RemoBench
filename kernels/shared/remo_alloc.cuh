// Adapted from SimLOD: modules/progressive_octree/utils.h.cu
// Upstream: https://github.com/m-schuetz/SimLOD @ fa7891613c138bd41775ca72a47cd89e32a5a647
// Copyright 2023 Markus Schuetz and Lukas Herzberger -- MIT (see THIRD_PARTY.md)

#pragma once

#include "shared/remo_prelude.cuh"

namespace remo {

inline uint64_t remoAlignUp16(uint64_t offset) {
	const uint64_t rem = offset % 16ull;
	return rem == 0ull ? offset : offset + (16ull - rem);
}

struct RemoAllocator {
	uint8_t* buffer;
	uint64_t offset;
	uint64_t capacity;
	DeviceDiagnostics* diag;

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
			if (diag != nullptr && threadIdx.x == 0 && blockIdx.x == 0) {
				diag->allocOverflow = 1u;
				diag->allocCapacity = capacity;
				diag->allocHighWater = end;
			}
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

	void rewindTo(uint64_t savedOffset) { offset = savedOffset; }
};

struct RemoAllocatorGlobal {
	uint64_t offset;
	uint64_t capacity;
	uint8_t* buffer;
	uint32_t overflow;
	uint32_t pad0;

	void init(void* base, uint64_t capacityBytes) {
		buffer = reinterpret_cast<uint8_t*>(base);
		capacity = capacityBytes;
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
			return nullptr;
		}
		return reinterpret_cast<T>(buffer + start);
	}
};

}
