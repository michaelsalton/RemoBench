// Adapted from SimLOD: modules/progressive_octree/math.cuh
// Upstream: https://github.com/m-schuetz/SimLOD @ fa7891613c138bd41775ca72a47cd89e32a5a647
// Copyright 2023 Markus Schuetz and Lukas Herzberger -- MIT (see THIRD_PARTY.md)
// Frustum extraction adapted from three.js (MIT).
//
// Minimal device math: row-major mat4 * vec4, and frustum culling. Deliberately
// small -- glm does not go through NVRTC cleanly, and this is all the device side
// needs.

#pragma once

#include "shared/remo_prelude.cuh"

namespace remo {

struct float4v {
	float x, y, z, w;
};

// Row-major, matching SharedUniforms::mat4. The host transposes glm's
// column-major matrices on the way in.
inline float4v remoMatMul(const mat4& m, float x, float y, float z, float w) {
	float4v r;
	r.x = m.rows[0][0] * x + m.rows[0][1] * y + m.rows[0][2] * z + m.rows[0][3] * w;
	r.y = m.rows[1][0] * x + m.rows[1][1] * y + m.rows[1][2] * z + m.rows[1][3] * w;
	r.z = m.rows[2][0] * x + m.rows[2][1] * y + m.rows[2][2] * z + m.rows[2][3] * w;
	r.w = m.rows[3][0] * x + m.rows[3][1] * y + m.rows[3][2] * z + m.rows[3][3] * w;
	return r;
}

// Project a world position to pixel coordinates. Returns false if the point is
// behind or on the near plane, in which case pixel/depth are not written.
inline bool remoProject(const mat4& transform, float x, float y, float z,
                        float width, float height, float& outX, float& outY,
                        float& outDepth) {
	const float4v clip = remoMatMul(transform, x, y, z, 1.0f);
	if (clip.w <= 0.0f) return false;

	const float invW = 1.0f / clip.w;
	const float ndcX = clip.x * invW;
	const float ndcY = clip.y * invW;

	outX = (ndcX * 0.5f + 0.5f) * width;
	outY = (ndcY * 0.5f + 0.5f) * height;
	// Linear eye-space depth. Used directly as the atomicMin key, so it must be
	// monotonically increasing with distance and non-negative -- see
	// remo_framebuffer.cuh on why the float bit pattern compares correctly.
	outDepth = clip.w;
	return true;
}

// ---------------------------------------------------------------------------
// Frustum
// ---------------------------------------------------------------------------
struct Plane {
	float nx, ny, nz, d;

	float distance(float x, float y, float z) const {
		return nx * x + ny * y + nz * z + d;
	}
};

struct Frustum {
	Plane planes[6];

	// Extract world-space planes from a row-major view-projection matrix.
	static Frustum fromViewProj(const mat4& m) {
		Frustum f;
		auto set = [&](int i, float a, float b, float c, float d) {
			const float len = sqrtf(a * a + b * b + c * c);
			const float inv = len > 0.0f ? 1.0f / len : 0.0f;
			f.planes[i] = {a * inv, b * inv, c * inv, d * inv};
		};
		const float(*r)[4] = m.rows;
		set(0, r[3][0] - r[0][0], r[3][1] - r[0][1], r[3][2] - r[0][2], r[3][3] - r[0][3]);
		set(1, r[3][0] + r[0][0], r[3][1] + r[0][1], r[3][2] + r[0][2], r[3][3] + r[0][3]);
		set(2, r[3][0] + r[1][0], r[3][1] + r[1][1], r[3][2] + r[1][2], r[3][3] + r[1][3]);
		set(3, r[3][0] - r[1][0], r[3][1] - r[1][1], r[3][2] - r[1][2], r[3][3] - r[1][3]);
		set(4, r[3][0] - r[2][0], r[3][1] - r[2][1], r[3][2] - r[2][2], r[3][3] - r[2][3]);
		set(5, r[3][0] + r[2][0], r[3][1] + r[2][1], r[3][2] + r[2][2], r[3][3] + r[2][3]);
		return f;
	}

	// Conservative AABB test: rejects only boxes fully outside a plane.
	bool intersectsBox(vec3f boxMin, vec3f boxMax) const {
		for (int i = 0; i < 6; ++i) {
			const Plane& p = planes[i];
			// Farthest corner along the plane normal.
			const float px = p.nx > 0.0f ? boxMax.x : boxMin.x;
			const float py = p.ny > 0.0f ? boxMax.y : boxMin.y;
			const float pz = p.nz > 0.0f ? boxMax.z : boxMin.z;
			if (p.distance(px, py, pz) < 0.0f) return false;
		}
		return true;
	}
};

}  // namespace remo
