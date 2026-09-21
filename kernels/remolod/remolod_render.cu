#include "../simlod/utils.h.cu"
#include "builtin_types.h"
#include "../simlod/helper_math.h"
#include "../simlod/HostDeviceInterface.h"
#include "../simlod/math.cuh"
#include "remolod_structures.cuh"
#include "../CudaPrint/CudaPrint.cuh"

#include "shared/remo_pipeline.cuh"

using remo::RemoAllocator;
using remo::DrawItem;
using remo::DrawList;
using remo::RenderArgs;
using remo::SharedUniforms;

static_assert(sizeof(Point) == sizeof(remo::Point),
              "SimLOD's Point must match remo::Point for the shared rasteriser");

using SimlodWalker = remo::RemoChunkedWalker<Chunk, POINTS_PER_CHUNK>;

constexpr uint32_t REMOLOD_DRAWLIST_CAPACITY = 131072;

struct NodeBounds {
	float3 min;
	float3 max;
	float size;
};

static NodeBounds nodeBounds(const Node* node, float3 cubeMin, float cubeSize) {
	NodeBounds b;
	b.size = cubeSize / powf(2.0f, float(node->level));
	b.min = {cubeMin.x + float(node->X) * b.size,
	         cubeMin.y + float(node->Y) * b.size,
	         cubeMin.z + float(node->Z) * b.size};
	b.max = {b.min.x + b.size, b.min.y + b.size, b.min.z + b.size};
	return b;
}

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
		if (clip.w <= 0.0f) continue;

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

	RemoAllocator alloc(args.scratch, args.scratchCapacity, diag);
	const uint64_t numPixels =
		static_cast<uint64_t>(u.width) * static_cast<uint64_t>(u.height);
	uint64_t* framebuffer = alloc.alloc<uint64_t*>(8ull * numPixels);
	DrawList drawList = remo::remoAllocDrawList(alloc, REMOLOD_DRAWLIST_CAPACITY);
	uint8_t* visibleFlags = alloc.alloc<uint8_t*>(MAX_NODES_CAPACITY);
	uint8_t* largeFlags = alloc.alloc<uint8_t*>(MAX_NODES_CAPACITY);
	uint64_t* sampleCount = alloc.alloc<uint64_t*>(8);

	if (framebuffer == nullptr || drawList.items == nullptr ||
	    visibleFlags == nullptr || largeFlags == nullptr || sampleCount == nullptr) {
		return;
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

	const float3 cubeMin = {u.boxMin.x, u.boxMin.y, u.boxMin.z};
	const float cubeSize =
		fmaxf(fmaxf(u.boxMax.x - u.boxMin.x, u.boxMax.y - u.boxMin.y),
		      u.boxMax.z - u.boxMin.z);

	const remo::Frustum frustum = remo::Frustum::fromViewProj(u.transformFrozen);

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
		largeFlags[i] = (dx > 2.0f * u.minNodeSize || dy > 2.0f * u.minNodeSize) ? 1u : 0u;
#else
		largeFlags[i] = (dx > u.lodPixelBudget || dy > u.lodPixelBudget) ? 1u : 0u;
#endif
	});
	grid.sync();

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
			item.voxelOctantMask = 0xFFu;

			item.points.head = target->points;
			item.points.count = target->numPoints;
			item.voxels.head = target->voxelChunks;
			item.voxels.count = target->numVoxels;

			if (item.points.count == 0u && item.voxels.count == 0u) return;
			if (remo::remoDrawListAppend(drawList, item)) {
				atomicAdd(reinterpret_cast<unsigned long long*>(sampleCount),
				          static_cast<unsigned long long>(item.points.count) +
				              static_cast<unsigned long long>(item.voxels.count));
			}
		};

		if (!isLeaf) {
			for (int c = 0; c < 8; ++c) {
				Node* child = node->children[c];
				if (child == nullptr) continue;

				const uint64_t childIndex = static_cast<uint64_t>(child - nodes);
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
