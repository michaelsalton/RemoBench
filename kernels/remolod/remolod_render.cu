// RemoLOD's LOD selection, RemoBench's shared rasteriser.
//
// FORKED from kernels/simlod/simlod_render.cu, which is itself RemoBench's port of SimLOD's
// selection pass. This copy is RemoLOD's to evolve -- once Analysis writes per-node scores,
// selection is where they get spent, and that is a change SimLOD's baseline must not see.
//
// IDENTICAL TO THE SIMLOD ONE TODAY apart from the includes and the scope name. While it
// stays that way, a difference between the two pipelines' images is attributable to
// construction rather than to selection.
//
// WHAT MAKES SIMLOD'S SELECTION DIFFERENT, and why it needs no octant mask:
//
// CudaLOD marks every node above a size threshold visible, so a parent and its children can
// both be drawn and the parent must mask away octants its children already cover. SimLOD
// instead emits a DISJOINT frontier directly (render.cu:904-933):
//
//   large inner node  -> emit each child that is visible and NOT large
//   large visible leaf-> emit itself
//
// "large" is per-node, so the set of emitted nodes is mutually exclusive by construction.
// voxelOctantMask therefore stays 0xFF here -- there is nothing to mask.
//
// Node bounds are not stored: SimLOD keeps integer grid coordinates (level, X, Y, Z) and
// recomputes world bounds on the fly, which is what nodeBounds() below does.


#include "../simlod/utils.h.cu"
#include "builtin_types.h"
#include "../simlod/helper_math.h"
#include "../simlod/HostDeviceInterface.h"
#include "../simlod/math.cuh"
#include "remolod_structures.cuh"
#include "../CudaPrint/CudaPrint.cuh"

#include "shared/remo_pipeline.cuh"

// SimLOD's device headers define their own Point, mat4 and math helpers at global scope, so
// remo:: names are imported selectively rather than with `using namespace remo` -- a bare
// using-directive makes Point and mat4 ambiguous. The two Point definitions are
// layout-identical, which is what makes the shared rasteriser usable on them.
using remo::RemoAllocator;
using remo::DrawItem;
using remo::DrawList;
using remo::RenderArgs;
using remo::SharedUniforms;

static_assert(sizeof(Point) == sizeof(remo::Point),
              "SimLOD's Point must match remo::Point for the shared rasteriser");

// Leaves and inner nodes both store samples as a linked list of POINTS_PER_CHUNK Chunks, so
// one walker covers both. This is the case the template Walker exists for -- CudaLOD's
// samples are contiguous slices instead.
using SimlodWalker = remo::RemoChunkedWalker<Chunk, POINTS_PER_CHUNK>;

constexpr uint32_t REMOLOD_DRAWLIST_CAPACITY = 131072;

struct NodeBounds {
	float3 min;
	float3 max;
	float size;
};

// SimLOD stores (level, X, Y, Z) rather than bounds; recompute them.
static NodeBounds nodeBounds(const Node* node, float3 cubeMin, float cubeSize) {
	NodeBounds b;
	b.size = cubeSize / powf(2.0f, float(node->level));
	b.min = {cubeMin.x + float(node->X) * b.size,
	         cubeMin.y + float(node->Y) * b.size,
	         cubeMin.z + float(node->Z) * b.size};
	b.max = {b.min.x + b.size, b.min.y + b.size, b.min.z + b.size};
	return b;
}

// Screen-space extent of the node's projected bounding box, in pixels.
//
// Upstream projects all eight corners and takes the AABB of the result, which is what this
// reproduces -- a centre-and-radius approximation disagrees near the frustum edges and would
// change the cut.
static void projectedExtent(const NodeBounds& b, const remo::mat4& transform, float width,
                            float height, float& outDx, float& outDy) {
	outDx = 0.0f;
	outDy = 0.0f;

	float minX = 1e30f, minY = 1e30f, maxX = -1e30f, maxY = -1e30f;
	bool any = false;

	for (int corner = 0; corner < 8; ++corner) {
		const float x = (corner & 1) ? b.max.x : b.min.x;
		const float y = (corner & 2) ? b.max.y : b.min.y;
		const float z = (corner & 4) ? b.max.z : b.min.z;

		const remo::float4v clip = remo::remoMatMul(transform, x, y, z, 1.0f);
		if (clip.w <= 0.0f) continue;  // behind the eye; ignore, as upstream does

		const float invW = 1.0f / clip.w;
		const float sx = (clip.x * invW * 0.5f + 0.5f) * width;
		const float sy = (clip.y * invW * 0.5f + 0.5f) * height;

		minX = fminf(minX, sx);
		maxX = fmaxf(maxX, sx);
		minY = fminf(minY, sy);
		maxY = fmaxf(maxY, sy);
		any = true;
	}

	if (!any) return;
	outDx = maxX - minX;
	outDy = maxY - minY;
}

extern "C" __global__ void kernel_render(RenderArgs args, Node* nodes, Stats* stats,
                                         remo::DeviceDiagnostics* diag) {
	auto grid = cg::this_grid();

	const SharedUniforms& u = args.uniforms;

	// Uniform control flow: every thread reaches every alloc(), in the same order. See the
	// banner in kernels/shared/remo_alloc.cuh.
	RemoAllocator alloc(args.scratch, args.scratchCapacity, diag);
	const uint64_t numPixels =
		static_cast<uint64_t>(u.width) * static_cast<uint64_t>(u.height);
	uint64_t* framebuffer = alloc.alloc<uint64_t*>(8ull * numPixels);
	DrawList drawList = remo::remoAllocDrawList(alloc, REMOLOD_DRAWLIST_CAPACITY);
	// Per-node flags, mirroring upstream's node->visible / node->isLarge. Kept in scratch
	// rather than written back into the Node, so selection never mutates the tree that the
	// construct kernel owns -- upstream writes into the live nodes, which makes rendering
	// and construction race on the same memory.
	uint8_t* visibleFlags = alloc.alloc<uint8_t*>(MAX_NODES_CAPACITY);
	uint8_t* largeFlags = alloc.alloc<uint8_t*>(MAX_NODES_CAPACITY);
	// Total samples the emitted items reference. Answers "did selection emit items that
	// actually point at data" separately from "can the walker read them".
	uint64_t* sampleCount = alloc.alloc<uint64_t*>(8);

	if (framebuffer == nullptr || drawList.items == nullptr ||
	    visibleFlags == nullptr || largeFlags == nullptr || sampleCount == nullptr) {
		return;  // diag->allocOverflow is set; the host reads it back and reports it
	}

	remo::remoClearFramebuffer(framebuffer, u);
	remo::remoResetDrawList(drawList);
	if (grid.thread_rank() == 0) *sampleCount = 0ull;
	grid.sync();

	uint32_t numNodes = stats->numNodes;
	if (numNodes > MAX_NODES_CAPACITY) numNodes = MAX_NODES_CAPACITY;
	if (numNodes == 0u) {
		grid.sync();
		remo::remoResolve(framebuffer, u,
		                  static_cast<cudaSurfaceObject_t>(args.surface));
		return;
	}

	// The octree root cube. SimLOD pre-translates so the cloud's minimum is the origin, and
	// cubes the box on its longest axis.
	const float3 cubeMin = {u.boxMin.x, u.boxMin.y, u.boxMin.z};
	const float cubeSize =
		fmaxf(fmaxf(u.boxMax.x - u.boxMin.x, u.boxMax.y - u.boxMin.y),
		      u.boxMax.z - u.boxMin.z);

	// Selection runs against the FROZEN transform, so unchecking "update visibility" locks
	// the cut and lets you fly the camera to inspect where it fell.
	const remo::Frustum frustum = remo::Frustum::fromViewProj(u.transformFrozen);

	// Pass 1: visible + large, for every node.
	remo::processRangeStrided(numNodes, [&](uint64_t i) {
		const Node* node = &nodes[i];
		const NodeBounds b = nodeBounds(node, cubeMin, cubeSize);

		const bool hasSamples = node->numPoints > 0 || node->numVoxels > 0;
		remo::vec3f bmin = {b.min.x, b.min.y, b.min.z};
		remo::vec3f bmax = {b.max.x, b.max.y, b.max.z};
		visibleFlags[i] =
			(hasSamples && frustum.intersectsBox(bmin, bmax)) ? 1u : 0u;

		float dx = 0.0f, dy = 0.0f;
		projectedExtent(b, u.transformFrozen, u.width, u.height, dx, dy);

#ifdef REMO_LOD_SIMLOD_NATIVE
		// Upstream's own test, for validating the port against bench/reference/.
		// minNodeSize is in WORLD units, so the same slider value means a different cut on
		// every dataset -- which is exactly why the shared metric below is the default.
		largeFlags[i] = (dx > 2.0f * u.minNodeSize || dy > 2.0f * u.minNodeSize) ? 1u : 0u;
#else
		// Shared metric: projected extent in pixels against one budget, so "both pipelines
		// at the same LOD" means the same cut.
		largeFlags[i] = (dx > u.lodPixelBudget || dy > u.lodPixelBudget) ? 1u : 0u;
#endif
	});
	grid.sync();

	// Pass 2: the disjoint frontier (render.cu:904-933).
	remo::processRangeStrided(numNodes, [&](uint64_t i) {
		Node* node = &nodes[i];
		if (largeFlags[i] == 0u) return;

		const bool isLeaf = node->isLeafFn();

		auto emit = [&](Node* target, uint64_t key) {
			const NodeBounds b = nodeBounds(target, cubeMin, cubeSize);

			DrawItem item = {};
			item.nodeMin = {b.min.x, b.min.y, b.min.z};
			item.nodeSize = b.size;
			item.level = target->level;
			item.nodeKey = static_cast<uint32_t>(key);
			// Disjoint by construction, so nothing to mask. See the header comment.
			item.voxelOctantMask = 0xFFu;

			item.points.head = target->points;
			item.points.count = target->numPoints;
			item.voxels.head = target->voxelChunks;
			// numVoxels, matching upstream's drawNode call (render.cu:204).
			// numVoxelsStored is the insertion counter, not the drawable count.
			item.voxels.count = target->numVoxels;

			if (item.points.count == 0u && item.voxels.count == 0u) return;
			if (remo::remoDrawListAppend(drawList, item)) {
				atomicAdd(reinterpret_cast<unsigned long long*>(sampleCount),
				          static_cast<unsigned long long>(item.points.count) +
				              static_cast<unsigned long long>(item.voxels.count));
			}
		};

		if (!isLeaf) {
			// Emit the children that are visible but NOT large: the frontier.
			for (int c = 0; c < 8; ++c) {
				Node* child = node->children[c];
				if (child == nullptr) continue;

				const uint64_t childIndex = static_cast<uint64_t>(child - nodes);
				// Children are raw pointers into the pool; guard the index, because the
				// tree is being built concurrently and may be mid-split.
				if (childIndex >= numNodes) continue;
				if (largeFlags[childIndex] != 0u) continue;
				if (visibleFlags[childIndex] == 0u) continue;

				emit(child, childIndex);
			}
		} else if (visibleFlags[i] != 0u) {
			emit(node, i);
		}
	});
	grid.sync();

	remo::remoRasterizeDrawList<SimlodWalker>(drawList, framebuffer, u);
	grid.sync();

	remo::remoApplyEDL(framebuffer, u);
	grid.sync();

	// After EDL, deliberately -- see the note on remoDrawListWireframe. Because
	// SimLOD's frontier is disjoint, every cube drawn here is a leaf of the cut, so
	// what you see is the partition itself with no parent boxes overlaying it.
	remo::remoDrawListWireframe(drawList, framebuffer, u);
	grid.sync();

	remo::remoResolve(framebuffer, u,
	                  static_cast<cudaSurfaceObject_t>(args.surface));

	if (grid.thread_rank() == 0 && diag != nullptr) {
		diag->drawListOverflow = *drawList.overflowed;
		diag->drawItems = *drawList.numItems;
		diag->drawSamples = *sampleCount;
	}
}
