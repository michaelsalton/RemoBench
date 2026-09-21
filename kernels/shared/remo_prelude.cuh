#pragma once

#include <cooperative_groups.h>

#include "remo/HostDeviceCommon.h"

namespace cg = cooperative_groups;

namespace remo {

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

template <typename Fn>
void processRangeStrided(uint64_t count, Fn&& fn) {
	auto grid = cg::this_grid();
	for (uint64_t i = grid.thread_rank(); i < count; i += grid.num_threads()) {
		fn(i);
	}
}

inline uint64_t remoNanotime() {
	uint64_t ns;
	asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ns));
	return ns;
}

inline uint32_t remoPackRGBA(uint32_t r, uint32_t g, uint32_t b, uint32_t a) {
	return r | (g << 8) | (b << 16) | (a << 24);
}

inline uint32_t remoHashColor(uint64_t key) {
	uint64_t h = (key + 0x9E3779B97F4A7C15ull) * 0x9E3779B97F4A7C15ull;
	h ^= h >> 29;
	h *= 0xBF58476D1CE4E5B9ull;
	h ^= h >> 32;

	auto channel = [](uint64_t bits) {
		return 96u + ((static_cast<uint32_t>(bits) & 0xFFu) * 159u) / 255u;
	};
	return remoPackRGBA(channel(h), channel(h >> 8), channel(h >> 16), 255u);
}

}
