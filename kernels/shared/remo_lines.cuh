#pragma once

#include "shared/remo_draw.cuh"

namespace remo {

inline uint32_t remoLevelColor(uint32_t level) {
	return remoHashColor(static_cast<uint64_t>(level) * 2654435761ull);
}

inline void remoBoxEdge(vec3f boxMin, float size, uint32_t edge, vec3f& outA,
                        vec3f& outB) {
	const uint32_t axis = edge / 4u;
	const uint32_t rest = edge % 4u;

	const uint32_t other0 = (axis + 1u) % 3u;
	const uint32_t other1 = (axis + 2u) % 3u;

	float offset[3] = {0.0f, 0.0f, 0.0f};
	offset[other0] = (rest & 1u) ? size : 0.0f;
	offset[other1] = (rest & 2u) ? size : 0.0f;

	outA = {boxMin.x + offset[0], boxMin.y + offset[1], boxMin.z + offset[2]};

	offset[axis] = size;
	outB = {boxMin.x + offset[0], boxMin.y + offset[1], boxMin.z + offset[2]};
}

inline bool remoClipSegment2D(float x0, float y0, float x1, float y1, float xmax,
                              float ymax, float& t0, float& t1) {
	const float dx = x1 - x0;
	const float dy = y1 - y0;
	t0 = 0.0f;
	t1 = 1.0f;

	auto clip = [&](float p, float q) {
		if (p == 0.0f) return q >= 0.0f;
		const float r = q / p;
		if (p < 0.0f) {
			if (r > t1) return false;
			if (r > t0) t0 = r;
		} else {
			if (r < t0) return false;
			if (r < t1) t1 = r;
		}
		return true;
	};

	if (!clip(-dx, x0)) return false;
	if (!clip(dx, xmax - x0)) return false;
	if (!clip(-dy, y0)) return false;
	if (!clip(dy, ymax - y0)) return false;
	return true;
}

inline void remoDrawSegment(uint64_t* fb, const SharedUniforms& u, vec3f a, vec3f b,
                            uint32_t color) {
	float4v c0 = remoMatMul(u.transform, a.x, a.y, a.z, 1.0f);
	float4v c1 = remoMatMul(u.transform, b.x, b.y, b.z, 1.0f);

	constexpr float kMinW = 1.0e-6f;
	if (c0.w < kMinW && c1.w < kMinW) return;
	if (c0.w < kMinW) {
		const float t = (kMinW - c0.w) / (c1.w - c0.w);
		c0 = {c0.x + t * (c1.x - c0.x), c0.y + t * (c1.y - c0.y),
		      c0.z + t * (c1.z - c0.z), kMinW};
	} else if (c1.w < kMinW) {
		const float t = (kMinW - c1.w) / (c0.w - c1.w);
		c1 = {c1.x + t * (c0.x - c1.x), c1.y + t * (c0.y - c1.y),
		      c1.z + t * (c0.z - c1.z), kMinW};
	}

	const float invW0 = 1.0f / c0.w;
	const float invW1 = 1.0f / c1.w;

	const float sx0 = (c0.x * invW0 * 0.5f + 0.5f) * u.width;
	const float sy0 = (c0.y * invW0 * 0.5f + 0.5f) * u.height;
	const float sx1 = (c1.x * invW1 * 0.5f + 0.5f) * u.width;
	const float sy1 = (c1.y * invW1 * 0.5f + 0.5f) * u.height;

	const int32_t width = static_cast<int32_t>(u.width);
	const int32_t height = static_cast<int32_t>(u.height);
	if (width <= 0 || height <= 0) return;

	float t0, t1;
	if (!remoClipSegment2D(sx0, sy0, sx1, sy1, static_cast<float>(width - 1),
	                       static_cast<float>(height - 1), t0, t1)) {
		return;
	}

	const float ax = sx0 + t0 * (sx1 - sx0);
	const float ay = sy0 + t0 * (sy1 - sy0);
	const float bx = sx0 + t1 * (sx1 - sx0);
	const float by = sy0 + t1 * (sy1 - sy0);

	const int32_t steps =
		static_cast<int32_t>(fmaxf(fabsf(bx - ax), fabsf(by - ay))) + 1;
	const float invSteps = 1.0f / static_cast<float>(steps);

	for (int32_t s = 0; s <= steps; ++s) {
		const float f = static_cast<float>(s) * invSteps;

		const int32_t ix = static_cast<int32_t>(ax + f * (bx - ax));
		const int32_t iy = static_cast<int32_t>(ay + f * (by - ay));
		if (ix < 0 || iy < 0 || ix >= width || iy >= height) continue;

		const float sT = t0 + f * (t1 - t0);
		const float invW = invW0 + sT * (invW1 - invW0);
		if (!(invW > 0.0f)) continue;

		const uint64_t index =
			static_cast<uint64_t>(iy) * static_cast<uint64_t>(width) +
			static_cast<uint64_t>(ix);
		atomicMin(reinterpret_cast<unsigned long long*>(&fb[index]),
		          static_cast<unsigned long long>(
					  remoFbPack(1.0f / invW, color)));
	}
}

inline void remoDrawListWireframe(const DrawList& list, uint64_t* fb,
                                  const SharedUniforms& u) {
	if (u.showBoundingBox == 0) return;

	uint32_t numItems = *list.numItems;
	if (numItems > list.capacity) numItems = list.capacity;

	processRangeStrided(static_cast<uint64_t>(numItems) * 12ull, [&](uint64_t i) {
		const DrawItem& item = list.items[i / 12ull];

		if (!(item.nodeSize > 0.0f)) return;

		vec3f edgeA, edgeB;
		remoBoxEdge(item.nodeMin, item.nodeSize, static_cast<uint32_t>(i % 12ull),
		            edgeA, edgeB);
		remoDrawSegment(fb, u, edgeA, edgeB, remoLevelColor(item.level));
	});
}

}
