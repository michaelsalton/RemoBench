// Octree wireframe: the LOD structure itself, drawn as node cubes.
//
// Colour-by-node shows you which samples belong to which node, but it cannot show you
// the node's EXTENT, and extent is what a split criterion actually decides. This draws
// the cut instead of inferring it: one cube per emitted DrawItem, so what you see is
// exactly the set of nodes the pipeline's selection pass chose, at the depths it chose
// them. Combined with `update visibility` off it is a structural inspector -- freeze the
// frontier, fly inside it, and look at where the boundaries fell.
//
// ---------------------------------------------------------------------------
// WHY THERE IS NO Lines BUFFER
//
// Both upstream repos accumulate debug geometry into a `Lines` struct with an atomic
// vertex counter and rasterise it in a later pass (SimLOD's rasterization.cuh,
// CudaLOD's lib.cu). That indirection earns its keep there because line segments are
// emitted from scattered places across several kernels and have to be collected
// somewhere.
//
// Here there is exactly ONE source of lines -- the draw list -- and it is already
// materialised in scratch by the time this runs. Going straight from DrawItem to pixels
// costs no memory, has no capacity to overflow, and needs no overflow diagnostic. A
// 131k-item frontier would otherwise want 1.57M segments (~50MB of vertices) to describe
// cubes we can derive from 16 bytes of DrawItem each.
//
// So: one thread per cube EDGE (not per cube), striped over the grid. Node sample counts
// vary by orders of magnitude but every cube has exactly twelve edges of bounded screen
// length, so a flat stripe is already balanced and no work stealing is needed.
// ---------------------------------------------------------------------------

#pragma once

#include "shared/remo_draw.cuh"

namespace remo {

// Wireframe colour for a node at this level.
//
// Deliberately the SAME expression remoSampleColor uses for COLOR_BY_LOD, so that with
// `colour: by LOD` selected a node's cube and the samples inside it come out the same
// colour. That correspondence is the point -- it is what lets you see that a patch of
// colour really is one node's worth of samples rather than two nodes that happen to hash
// alike.
inline uint32_t remoLevelColor(uint32_t level) {
	return remoHashColor(static_cast<uint64_t>(level) * 2654435761ull);
}

// Endpoints of one of a cube's twelve edges.
//
// Corners are indexed x = bit 0, y = bit 1, z = bit 2 -- the shared convention from
// remoOctantOf, so edge and octant indices agree rather than being two orders to keep
// straight. Edges 0-3 run along x, 4-7 along y, 8-11 along z.
inline void remoBoxEdge(vec3f boxMin, float size, uint32_t edge, vec3f& outA,
                        vec3f& outB) {
	const uint32_t axis = edge / 4u;  // 0 = x, 1 = y, 2 = z
	const uint32_t rest = edge % 4u;  // which of the four edges parallel to it

	// The two axes the edge does NOT run along; their corner bits come from `rest`.
	const uint32_t other0 = (axis + 1u) % 3u;
	const uint32_t other1 = (axis + 2u) % 3u;

	float offset[3] = {0.0f, 0.0f, 0.0f};
	offset[other0] = (rest & 1u) ? size : 0.0f;
	offset[other1] = (rest & 2u) ? size : 0.0f;

	outA = {boxMin.x + offset[0], boxMin.y + offset[1], boxMin.z + offset[2]};

	offset[axis] = size;
	outB = {boxMin.x + offset[0], boxMin.y + offset[1], boxMin.z + offset[2]};
}

// Parametric clip of a 2D segment to [0,xmax] x [0,ymax] (Liang-Barsky).
//
// Returns false when the segment is entirely outside; otherwise narrows [t0,t1] to the
// visible span. Clipping FIRST, rather than stepping the whole segment and rejecting
// out-of-bounds pixels, is what bounds the cost: a cube edge that starts near the eye
// can be tens of thousands of pixels long, almost all of it off-screen. Upstream instead
// clamps the step count to 400 (rasterization.cuh), which bounds the cost by drawing a
// dotted line -- long edges visibly break up.
inline bool remoClipSegment2D(float x0, float y0, float x1, float y1, float xmax,
                              float ymax, float& t0, float& t1) {
	const float dx = x1 - x0;
	const float dy = y1 - y0;
	t0 = 0.0f;
	t1 = 1.0f;

	// Narrow [t0,t1] against one half-plane, expressed as p*t <= q.
	auto clip = [&](float p, float q) {
		if (p == 0.0f) return q >= 0.0f;  // parallel: in or out wholesale
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

	if (!clip(-dx, x0)) return false;         // x >= 0
	if (!clip(dx, xmax - x0)) return false;   // x <= xmax
	if (!clip(-dy, y0)) return false;         // y >= 0
	if (!clip(dy, ymax - y0)) return false;   // y <= ymax
	return true;
}

// Rasterise one world-space segment into the packed uint64 framebuffer.
//
// Depth-tests against the geometry through the same atomicMin as every point, so an edge
// behind a wall is correctly hidden by it. That is the reason for the care taken over
// depth below: a wireframe with wrong depths does not look like a wrong wireframe, it
// looks like the LOD structure is in the wrong place.
inline void remoDrawSegment(uint64_t* fb, const SharedUniforms& u, vec3f a, vec3f b,
                            uint32_t color) {
	float4v c0 = remoMatMul(u.transform, a.x, a.y, a.z, 1.0f);
	float4v c1 = remoMatMul(u.transform, b.x, b.y, b.z, 1.0f);

	// Near clip in CLIP space, before the perspective divide.
	//
	// Not optional. Dividing by a negative w mirrors the endpoint through the origin, so
	// an edge crossing behind the eye is drawn to the wrong side of the screen -- a long
	// bogus streak across the viewport, which reads as a rendering bug rather than as a
	// clipping one. remoProject sidesteps this by rejecting w <= 0 outright, which is
	// right for a point and wrong for a segment with one endpoint in front.
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

	// One step per pixel along the major axis: gap-free, and bounded by the screen
	// diagonal now that the segment is clipped to the viewport.
	const int32_t steps =
		static_cast<int32_t>(fmaxf(fabsf(bx - ax), fabsf(by - ay))) + 1;
	const float invSteps = 1.0f / static_cast<float>(steps);

	for (int32_t s = 0; s <= steps; ++s) {
		const float f = static_cast<float>(s) * invSteps;

		const int32_t ix = static_cast<int32_t>(ax + f * (bx - ax));
		const int32_t iy = static_cast<int32_t>(ay + f * (by - ay));
		if (ix < 0 || iy < 0 || ix >= width || iy >= height) continue;

		// Perspective-correct depth. 1/w is the quantity that interpolates LINEARLY in
		// screen space, so lerp that and invert; lerping w directly bows the edge in
		// depth and makes it punch through the surface it is supposed to bound near the
		// silhouette. `sT` is the parameter along the unclipped screen segment, which is
		// the parameter 1/w is linear in -- f alone is relative to the clipped span.
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

// Draw a cube per DrawItem, coloured by level. No-op unless u.showBoundingBox is set.
//
// CALL THIS AFTER remoApplyEDL, not before. EDL shades from depth discontinuities and a
// wireframe is nothing but depth discontinuities, so shading it rings every edge with a
// dark halo and dims the lines themselves. An overlay should not be lit.
//
// Grid-wide, so the caller must grid.sync() before remoResolve.
inline void remoDrawListWireframe(const DrawList& list, uint64_t* fb,
                                  const SharedUniforms& u) {
	if (u.showBoundingBox == 0) return;

	uint32_t numItems = *list.numItems;
	if (numItems > list.capacity) numItems = list.capacity;

	processRangeStrided(static_cast<uint64_t>(numItems) * 12ull, [&](uint64_t i) {
		const DrawItem& item = list.items[i / 12ull];

		// `remolod` has no tree: its items are fixed-size slices of the point array and
		// carry nodeSize 0. Draw nothing rather than a degenerate cube at the origin --
		// the toggle is correctly inert for the control condition.
		if (!(item.nodeSize > 0.0f)) return;

		vec3f edgeA, edgeB;
		remoBoxEdge(item.nodeMin, item.nodeSize, static_cast<uint32_t>(i % 12ull),
		            edgeA, edgeB);
		remoDrawSegment(fb, u, edgeA, edgeB, remoLevelColor(item.level));
	});
}

}  // namespace remo
