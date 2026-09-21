// Adapted from SimLOD: modules/progressive_octree/render.cu
// Upstream: https://github.com/m-schuetz/SimLOD @ fa7891613c138bd41775ca72a47cd89e32a5a647
// Copyright 2023 Markus Schuetz and Lukas Herzberger -- MIT (see THIRD_PARTY.md)

#pragma once

#include "shared/remo_math.cuh"

namespace remo {

constexpr uint64_t REMO_FB_CLEAR = 0x7f800000'00332211ull;

inline uint32_t remoFbColor(uint64_t pixel) {
	return static_cast<uint32_t>(pixel & 0xFFFFFFFFull);
}

inline float remoFbDepth(uint64_t pixel) {
	const uint32_t bits = static_cast<uint32_t>(pixel >> 32);
	return __int_as_float(static_cast<int>(bits));
}

inline uint64_t remoFbPack(float depth, uint32_t color) {
	const uint64_t bits = static_cast<uint32_t>(__float_as_int(depth));
	return (bits << 32) | static_cast<uint64_t>(color);
}

inline void remoClearFramebuffer(uint64_t* fb, const SharedUniforms& u) {
	const uint64_t numPixels =
		static_cast<uint64_t>(u.width) * static_cast<uint64_t>(u.height);
	processRangeStrided(numPixels, [&](uint64_t i) { fb[i] = REMO_FB_CLEAR; });
}

inline void remoDrawPoint(uint64_t* fb, const SharedUniforms& u, float x, float y,
                          float z, uint32_t color) {
	float px, py, depth;
	if (!remoProject(u.transform, x, y, z, u.width, u.height, px, py, depth)) return;

	const int32_t half = u.pointSize / 2;
	const int32_t ix = static_cast<int32_t>(px);
	const int32_t iy = static_cast<int32_t>(py);
	const int32_t w = static_cast<int32_t>(u.width);
	const int32_t h = static_cast<int32_t>(u.height);

	if (ix + half < 0 || iy + half < 0 || ix - half >= w || iy - half >= h) return;

	const uint64_t packed = remoFbPack(depth, color);

	for (int32_t oy = -half; oy <= half; ++oy) {
		const int32_t sy = iy + oy;
		if (sy < 0 || sy >= h) continue;
		for (int32_t ox = -half; ox <= half; ++ox) {
			const int32_t sx = ix + ox;
			if (sx < 0 || sx >= w) continue;
			const uint64_t index =
				static_cast<uint64_t>(sy) * static_cast<uint64_t>(w) +
				static_cast<uint64_t>(sx);
			atomicMin(reinterpret_cast<unsigned long long*>(&fb[index]),
			          static_cast<unsigned long long>(packed));
		}
	}
}

inline void remoResolve(uint64_t* fb, const SharedUniforms& u,
                        cudaSurfaceObject_t surface) {
	const uint32_t width = static_cast<uint32_t>(u.width);
	const uint32_t height = static_cast<uint32_t>(u.height);
	const uint64_t numPixels =
		static_cast<uint64_t>(width) * static_cast<uint64_t>(height);

	processRangeStrided(numPixels, [&](uint64_t i) {
		const uint32_t x = static_cast<uint32_t>(i % width);
		const uint32_t y = static_cast<uint32_t>(i / width);
		uint32_t color = remoFbColor(fb[i]);
		surf2Dwrite(color, surface, static_cast<int>(x) * 4, static_cast<int>(y));
	});
}

inline void remoApplyEDL(uint64_t* fb, const SharedUniforms& u) {
	if (u.enableEDL == 0) return;

	const int32_t width = static_cast<int32_t>(u.width);
	const int32_t height = static_cast<int32_t>(u.height);
	const uint64_t numPixels =
		static_cast<uint64_t>(width) * static_cast<uint64_t>(height);

	processRangeStrided(numPixels, [&](uint64_t i) {
		const int32_t x = static_cast<int32_t>(i % static_cast<uint64_t>(width));
		const int32_t y = static_cast<int32_t>(i / static_cast<uint64_t>(width));

		const uint64_t pixel = fb[i];
		const float depth = remoFbDepth(pixel);
		if (!(depth < 3.0e38f)) return;

		const float logDepth = __log2f(depth);
		float response = 0.0f;
		int taps = 0;

		const int32_t offsets[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
		for (int t = 0; t < 4; ++t) {
			const int32_t nx = x + offsets[t][0];
			const int32_t ny = y + offsets[t][1];
			if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
			const float nd = remoFbDepth(
				fb[static_cast<uint64_t>(ny) * static_cast<uint64_t>(width) +
				   static_cast<uint64_t>(nx)]);
			if (!(nd < 3.0e38f)) continue;
			response += fmaxf(0.0f, logDepth - __log2f(nd));
			++taps;
		}
		if (taps == 0) return;
		response /= static_cast<float>(taps);

		constexpr float kEdlScale = 24.0f;
		const float shade = __expf(-response * kEdlScale * u.edlStrength);

		const uint32_t color = remoFbColor(pixel);
		const uint32_t r = static_cast<uint32_t>((color & 0xFFu) * shade);
		const uint32_t g = static_cast<uint32_t>(((color >> 8) & 0xFFu) * shade);
		const uint32_t b = static_cast<uint32_t>(((color >> 16) & 0xFFu) * shade);
		fb[i] = remoFbPack(depth, remoPackRGBA(r, g, b, 255u));
	});
}

}
