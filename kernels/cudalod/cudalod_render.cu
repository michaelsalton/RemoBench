// CudaLOD's LOD selection, RemoBench's shared rasteriser.
//
// This file replaces upstream's render.cu (1932 lines). Only the SELECTION half is
// ported, because that is the part under study -- deciding which nodes to draw is a real
// algorithmic choice with a real quality/performance trade-off. Everything after it
// (projection, splatting, the packed uint64 atomicMin depth test, EDL, the surface
// resolve) comes from kernels/shared, identical to every other pipeline.
//
// That split is the whole reason this project exists. Upstream's two renderers differ in
// ways that have nothing to do with LOD: CudaLOD uses a uint32 depth buffer plus a
// 16 byte/pixel accumulator with a Gaussian 3x3 splat resolve, SimLOD uses a packed
// uint64 with divide-by-count. Feed those the same octree and the images differ anyway,
// so no comparison between them is attributable to the LOD algorithm.
//
// Kept faithfully from upstream:
//   - the LOD test itself, cubeSize / distance < 1 - 0.97 * LOD (render.cu:1473)
//   - drawing an inner node's voxels only for octants whose child is not itself
//     visible (upstream's childMask)
//
// Deliberately NOT kept: upstream copies each visible node's whole 120-byte Node into a
// scratch array. A DrawItem is 56 bytes and holds only what the rasteriser reads.

#include "lib.h.cu"
#include "helper_math.h"
#include "common.h"
#include "methods_common.h.cu"

#include "shared/remo_pipeline.cuh"

// CudaLOD's device headers define their own vec3/Box3 and pull in cooperative_groups
// themselves, so `remo::` names are qualified rather than dumped into scope -- a bare
// `using namespace remo` would make `Point` ambiguous between the two definitions.
// They are layout-identical (float x,y,z + uint32 colour), which is why the
// reinterpret_casts below are sound.
using remo::RemoAllocator;
using remo::RemoContiguousWalker;
using remo::DrawItem;
using remo::DrawList;
using remo::Frustum;
using remo::RenderArgs;
using remo::SharedUniforms;

static_assert(sizeof(Point) == sizeof(remo::Point),
              "CudaLOD's Point must match remo::Point for the shared rasteriser");

// How many nodes the draw list can hold. MAX_NODES is 200'000 upstream, but the visible
// set is a small fraction of that; overflow is reported rather than silently truncating.
constexpr uint32_t CUDALOD_DRAWLIST_CAPACITY = 65536;

// Is this node worth subdividing past?
//
// REMO_LOD_CUDALOD_NATIVE reproduces upstream exactly, for validating the port against
// bench/reference/. The default is the shared pixel budget, which is what makes "both
// pipelines at the same LOD" mean the same cut -- upstream's lodScale is angular but not
// calibrated to the viewport, and SimLOD's minNodeSize is in world units and therefore
// means something different on every dataset.
static bool cudalodNodeVisible(const Node* node, const SharedUniforms& u,
                               const Frustum& frustum) {
	// The root is unconditionally visible, matching upstream (render.cu:1484). Without
	// this, a camera inside the cloud can reject the root on the frustum test and the
	// whole tree disappears -- the root's own box is behind you even though its contents
	// are not.
	if (node->level == 0) return true;

	remo::vec3f boxMin = {node->min.x, node->min.y, node->min.z};
	remo::vec3f boxMax = {node->max.x, node->max.y, node->max.z};
	if (!frustum.intersectsBox(boxMin, boxMax)) return false;

	const float cx = (node->min.x + node->max.x) * 0.5f;
	const float cy = (node->min.y + node->max.y) * 0.5f;
	const float cz = (node->min.z + node->max.z) * 0.5f;

	const remo::float4v clip = remo::remoMatMul(u.transformFrozen, cx, cy, cz, 1.0f);
	float distance = clip.w;
	if (distance < 0.1f) distance = 0.1f;

#ifdef REMO_LOD_CUDALOD_NATIVE
	const float lodFactor = 1.0f - 0.97f * u.lodScale;
	return node->cubeSize / distance >= lodFactor;
#else
	// Shared metric: the node's projected extent in pixels. proj.rows[1][1] is
	// 1/tan(fovy/2), so cubeSize/distance * that * (height/2) is the on-screen size.
	const float pixels =
		(node->cubeSize / distance) * u.proj.rows[1][1] * (u.height * 0.5f);
	return pixels >= u.lodPixelBudget;
#endif
}

extern "C" __global__ void kernel_render(RenderArgs args, void** nnodes,
                                         uint32_t* num_nodes,
                                         remo::DeviceDiagnostics* diag) {
	auto grid = cg::this_grid();

	const SharedUniforms& u = args.uniforms;

	// Uniform control flow: every thread reaches every alloc(), in the same order.
	// See the banner in kernels/shared/remo_alloc.cuh.
	RemoAllocator alloc(args.scratch, args.scratchCapacity, diag);
	const uint64_t numPixels =
		static_cast<uint64_t>(u.width) * static_cast<uint64_t>(u.height);
	uint64_t* framebuffer = alloc.alloc<uint64_t*>(8ull * numPixels);
	DrawList drawList = remo::remoAllocDrawList(alloc, CUDALOD_DRAWLIST_CAPACITY);
	// Samples referenced by the emitted items, for the stats panel.
	uint64_t* sampleCount = alloc.alloc<uint64_t*>(8);
	uint8_t* visibleFlags = alloc.alloc<uint8_t*>(MAX_NODES);

	if (framebuffer == nullptr || drawList.items == nullptr ||
	    visibleFlags == nullptr) {
		return;  // diag->allocOverflow is set; the host reports it
	}

	remo::remoClearFramebuffer(framebuffer, u);
	remo::remoResetDrawList(drawList);
	if (grid.thread_rank() == 0) *sampleCount = 0ull;
	grid.sync();

	Node* nodes = reinterpret_cast<Node*>(*nnodes);
	const uint32_t numNodes = *num_nodes;
	if (nodes == nullptr || numNodes == 0u) {
		grid.sync();
		remo::remoResolve(framebuffer, u,
		                  static_cast<cudaSurfaceObject_t>(args.surface));
		return;
	}

	// Pass 1: visibility for every node. Done for all nodes before any emission,
	// because the octant mask below needs to know whether a node's CHILDREN are
	// visible, which is not known until the whole pass is complete.
	const Frustum frustum = Frustum::fromViewProj(u.transformFrozen);
	remo::processRangeStrided(numNodes, [&](uint64_t i) {
		visibleFlags[i] = cudalodNodeVisible(&nodes[i], u, frustum) ? 1u : 0u;
	});
	grid.sync();

	// Pass 2: emit the visible set.
	remo::processRangeStrided(numNodes, [&](uint64_t i) {
		if (visibleFlags[i] == 0u) return;
		Node* node = &nodes[i];

		DrawItem item = {};
		item.nodeMin = {node->min.x, node->min.y, node->min.z};
		item.nodeSize = node->cubeSize;
		item.level = static_cast<uint32_t>(node->level);
		item.nodeKey = static_cast<uint32_t>(i);
		item.voxelOctantMask = 0xFFu;

		if (node->isLeaf()) {
			// Leaves hold original points, as a contiguous slice of the globally
			// counting-sorted array.
			item.points.head = node->points;
			item.points.count = static_cast<uint32_t>(node->numPoints);
		} else {
			// Inner nodes hold voxels. Skip the octants whose child is being drawn at
			// higher detail -- upstream's childMask.
			uint32_t mask = 0u;
			for (uint32_t c = 0; c < 8u; ++c) {
				Node* child = node->children[c];

				// PERMUTE INTO THE SHARED OCTANT ORDER. CudaLOD indexes children as
				// (ox << 2) | (oy << 1) | oz (split_countsort_blockwise.h.cu:520),
				// x in the HIGH bit; remoOctantOf uses x in the LOW bit. Using c
				// directly as a mask bit masks the wrong octants, which renders as
				// large holes while the LOD structure is provably fine.
				const uint32_t ox = (c >> 2) & 1u;
				const uint32_t oy = (c >> 1) & 1u;
				const uint32_t oz = c & 1u;
				const uint32_t sharedBit = ox | (oy << 1) | (oz << 2);

				if (child == nullptr) {
					// No child here, so nothing else will cover this octant.
					mask |= (1u << sharedBit);
					continue;
				}
				const uint64_t childIndex =
					static_cast<uint64_t>(child - nodes);
				// Guard the index: children are raw pointers into the pool, and a
				// half-built or corrupt tree would otherwise read out of bounds.
				const bool childVisible = childIndex < numNodes &&
				                          visibleFlags[childIndex] != 0u;
				if (!childVisible) mask |= (1u << sharedBit);
			}
			item.voxels.head = node->voxels;
			item.voxels.count = static_cast<uint32_t>(node->numVoxels);
			item.voxelOctantMask = mask;
		}

		if (item.points.count == 0u && item.voxels.count == 0u) return;
		if (remo::remoDrawListAppend(drawList, item)) {
			atomicAdd(reinterpret_cast<unsigned long long*>(sampleCount),
			          static_cast<unsigned long long>(item.points.count) +
			              static_cast<unsigned long long>(item.voxels.count));
		}
	});
	grid.sync();

	// Points and voxels are both contiguous Point arrays here, so one walker covers both.
	remo::remoRasterizeDrawList<RemoContiguousWalker>(drawList, framebuffer, u);
	grid.sync();

	remo::remoApplyEDL(framebuffer, u);
	grid.sync();

	// After EDL, deliberately -- see the note on remoDrawListWireframe. Unlike SimLOD's
	// disjoint frontier, CudaLOD marks a parent AND its children visible, so expect
	// nested cubes here. That is the selection difference made visible, not a bug.
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
