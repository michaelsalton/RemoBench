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

// Intra-kernel phase timing. See plans/05_HardCodedTest.md.
//
// REMO_MARK expands to nothing unless REMO_PROFILE is defined, so a default build
// does none of the timing work -- not the stores and not the clock reads, which
// are asm volatile and would otherwise survive. The kernel signature is the one
// thing that does NOT vary: the sink pointer is passed either way, so there is a
// single host launch path. The profiling variant is requested
// through KernelProgramDesc::defines, which is part of the compile cache key,
// so both variants cache side by side and hot-reload independently.
//
// Place a mark ONLY immediately after a grid.sync(). One thread's clock read is
// a grid-wide phase boundary only when every other thread has reached the same
// point. REMO_PROFILE_ONLY carries the same requirement.
#ifdef REMO_PROFILE

#define REMO_MARK(tl, phaseId)                                              \
	do {                                                                    \
		if ((tl) != nullptr && cg::this_grid().thread_rank() == 0) {        \
			const uint32_t _i = (tl)->numMarks;                             \
			if (_i < remo::REMO_MAX_MARKS) {                                \
				(tl)->marks[_i].phase = (phaseId);                          \
				(tl)->marks[_i].pad = 0;                                    \
				(tl)->marks[_i].ns = remo::remoNanotime();                  \
				(tl)->numMarks = _i + 1;                                    \
			} else {                                                        \
				(tl)->overflow = 1;                                         \
			}                                                               \
		}                                                                   \
	} while (0)

#define REMO_PROFILE_ONLY(...)                                              \
	do {                                                                    \
		__VA_ARGS__                                                         \
	} while (0)

#else

#define REMO_MARK(tl, phaseId) ((void)0)
#define REMO_PROFILE_ONLY(...) ((void)0)

#endif

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
