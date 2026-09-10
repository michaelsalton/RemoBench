// The seam between a pipeline's LOD structure and the shared rasteriser.
//
// A pipeline decides WHICH nodes to draw (its own LOD selection -- genuinely part of
// what is being compared, so it stays pipeline-owned) and publishes them as a
// DrawList. Everything downstream -- projection, splatting, the packed uint64
// atomicMin depth test, EDL, the surface resolve -- is shared, which is what makes an
// image difference attributable to the LOD algorithm.
//
// ---------------------------------------------------------------------------
// WHY SAMPLE STORAGE IS WALKED BY A TEMPLATE RATHER THAN A COMMON STRUCT
//
// The two reference pipelines store node samples incompatibly:
//
//   SimLOD   linked list of Chunks, 1000 Points each
//            (structures.cuh: struct Chunk{ Point points[1000]; int size; Chunk* next; })
//   CudaLOD  a contiguous slice of one globally counting-sorted Point array
//            (methods_common.h.cu: Node{ int pointOffset; int numPoints; Point* points; })
//
// Three ways to bridge that, and only one is good:
//
//   1. Flatten to an array of spans. Rejected: 100k nodes x up to 50 chunks x 32B is
//      ~160MB of descriptors rebuilt every frame, to describe data we already have.
//   2. A runtime `kind` switch inside the inner loop. Rejected: it puts a branch on
//      the hottest path in the renderer, and it would force every pipeline to adopt a
//      single chunk layout.
//   3. A Walker type supplied by the pipeline, resolved at NVRTC compile time.
//      Chosen: zero runtime cost, and each pipeline keeps its own structs UNCHANGED,
//      which matters because those structs are what is being validated against
//      published numbers.
//
// Option 3 works precisely because sharing here is textual (#include), so templates
// specialise per pipeline with no dispatch. See remo_pipeline.cuh.
// ---------------------------------------------------------------------------

#pragma once

#include "shared/remo_framebuffer.cuh"

namespace remo {

// Opaque handle to one node's samples. The pipeline's Walker knows how to read it.
struct SampleSource {
	void* head;       // Point* for contiguous storage, chunk head for a chunk list
	uint32_t count;   // total samples reachable from head
	uint32_t stride;  // spare: element stride or chunk capacity, Walker's choice
};

// One node to draw. 48 bytes.
//
// Note this carries node GEOMETRY, not just samples: colour-by-node and colour-by-LOD
// need an identity and a level, and a later high-quality path needs the node size to
// derive a splat radius. Upstream instead copies the whole 152-byte Node by value into
// a scratch array (render.cu makeVisible), which is 3x the traffic for data the
// rasteriser does not read.
struct DrawItem {
	SampleSource points;
	SampleSource voxels;
	vec3f nodeMin;
	float nodeSize;
	uint32_t level;
	uint32_t nodeKey;  // stable per-node id, for colour-by-node

	// Which of the node's eight octants should have their VOXELS drawn, one bit each
	// (bit i = octant with x = i&1, y = (i>>1)&1, z = (i>>2)&1). 0xFF draws all.
	//
	// This exists because an inner node's voxels summarise its whole subtree, so
	// wherever a child is ALSO being drawn at higher detail the parent's voxels in
	// that octant are redundant. CudaLOD does the same thing with a childMask
	// (render.cu), and SimLOD's disjoint-frontier selection avoids the situation
	// structurally.
	//
	// Leaving it at 0xFF is not a correctness problem -- the depth test resolves the
	// overlap -- but it wastes samples, and "samples drawn" is one of the numbers
	// being compared, so it should not be inflated by avoidable overdraw.
	uint32_t voxelOctantMask;
};

// Which octant of [nodeMin, nodeMin + nodeSize) a position falls in.
//
// THE BIT ORDER HERE IS THE SHARED CONVENTION: x is bit 0, y is bit 1, z is bit 2.
//
// A pipeline whose own child array uses a different order MUST permute when it builds
// voxelOctantMask -- do not assume they agree. CudaLOD, for instance, indexes children as
// (ox << 2) | (oy << 1) | oz, i.e. exactly reversed, and using its slot index directly as
// a mask bit masks the WRONG octants. The symptom is not a crash but large holes in the
// render, with the LOD structure itself provably correct.
inline uint32_t remoOctantOf(const DrawItem& item, float x, float y, float z) {
	const float half = item.nodeSize * 0.5f;
	const uint32_t ix = x >= item.nodeMin.x + half ? 1u : 0u;
	const uint32_t iy = y >= item.nodeMin.y + half ? 1u : 0u;
	const uint32_t iz = z >= item.nodeMin.z + half ? 1u : 0u;
	return ix | (iy << 1) | (iz << 2);
}

struct DrawList {
	DrawItem* items;
	uint32_t* numItems;   // written by the pipeline's selection pass
	uint32_t capacity;    // bounds-checked on append; see remoDrawListAppend
	uint32_t* overflowed; // set if selection produced more nodes than capacity
};

// Allocate a draw list from per-launch scratch.
//
// Uniform control flow: every thread must reach this, in this order. See the banner in
// remo_alloc.cuh.
inline DrawList remoAllocDrawList(RemoAllocator& alloc, uint32_t capacity) {
	DrawList list;
	list.items = alloc.alloc<DrawItem*>(sizeof(DrawItem) * uint64_t(capacity));
	list.numItems = alloc.alloc<uint32_t*>(4);
	list.overflowed = alloc.alloc<uint32_t*>(4);
	list.capacity = capacity;
	return list;
}

inline void remoResetDrawList(const DrawList& list) {
	auto grid = cg::this_grid();
	if (grid.thread_rank() == 0) {
		*list.numItems = 0u;
		*list.overflowed = 0u;
	}
}

// Append one node. Safe to call from any thread.
//
// Returns false when the list is full, and records that fact rather than dropping
// nodes silently -- a truncated draw list means fewer samples drawn, which would show
// up as a suspiciously fast frame and a subtly wrong image. Upstream's equivalent
// scratch array has a fixed 100'000-entry capacity and no check at all.
inline bool remoDrawListAppend(const DrawList& list, const DrawItem& item) {
	const uint32_t index = atomicAdd(list.numItems, 1u);
	if (index >= list.capacity) {
		*list.overflowed = 1u;
		return false;
	}
	list.items[index] = item;
	return true;
}

// ---------------------------------------------------------------------------
// Walkers
// ---------------------------------------------------------------------------

// Samples in one contiguous Point array. CudaLOD's leaves and voxel arrays.
struct RemoContiguousWalker {
	// fn(index, Point) for a strided subset of the samples, so a whole block can
	// cooperate on one node.
	template <typename Fn>
	static void forEachStrided(const SampleSource& source, uint32_t offset,
	                           uint32_t stride, Fn&& fn) {
		const Point* points = reinterpret_cast<const Point*>(source.head);
		if (points == nullptr) return;
		for (uint32_t i = offset; i < source.count; i += stride) {
			fn(i, points[i]);
		}
	}
};

// Samples in a singly-linked list of fixed-capacity chunks. SimLOD's leaves and voxel
// chunks.
//
// ChunkT must expose exactly two things:
//   Point points[Capacity];
//   ChunkT* next;
//
// DELIBERATELY NOT a `size` member, even though SimLOD's Chunk has one. That field is
// NEVER WRITTEN by its construct kernel -- it is dead in the struct, and upstream's own
// drawNode ignores it, deriving the per-chunk count from the total sample count and
// POINTS_PER_CHUNK instead (render.cu:106-140). Reading it yields 0, every chunk looks
// empty, and the pipeline renders a blank frame while its tree and its draw list are both
// provably correct -- which is a genuinely confusing thing to debug. So the count comes
// from source.count here, and `size` is never touched.
//
// Templated rather than hardcoded so each pipeline's Chunk stays byte-identical to
// upstream.
template <typename ChunkT, uint32_t Capacity>
struct RemoChunkedWalker {
	template <typename Fn>
	static void forEachStrided(const SampleSource& source, uint32_t offset,
	                           uint32_t stride, Fn&& fn) {
		const ChunkT* chunk = reinterpret_cast<const ChunkT*>(source.head);
		if (chunk == nullptr || stride == 0u) return;

		// Walk chunks; within each, stride over the samples. The pointer chase is
		// per-chunk rather than per-sample, so a 1000-point chunk costs one dereference.
		uint32_t base = 0;
		while (chunk != nullptr && base < source.count) {
			const uint32_t remaining = source.count - base;
			const uint32_t inChunk = remaining < Capacity ? remaining : Capacity;

			// This thread owns the global indices g with g % stride == offset (offset is
			// its lane, so offset < stride). Solve for the first slot within this chunk:
			//   base + slot == offset  (mod stride)
			const uint32_t delta = (offset + stride - (base % stride)) % stride;

			for (uint32_t slot = delta; slot < inChunk; slot += stride) {
				// The chunk's element type is the PIPELINE's Point, a distinct type from
				// remo::Point even though the two are layout-identical (asserted in each
				// pipeline's render kernel). Cast, exactly as the contiguous walker does
				// when it reinterprets source.head -- otherwise every pipeline would have
				// to template its callback on its own point type.
				fn(base + slot,
				   *reinterpret_cast<const Point*>(&chunk->points[slot]));
			}

			base += inChunk;
			chunk = chunk->next;
		}
	}
};

// ---------------------------------------------------------------------------
// Rasterise a draw list
// ---------------------------------------------------------------------------

inline uint32_t remoSampleColor(const SharedUniforms& u, const DrawItem& item,
                                uint32_t sampleColor) {
	switch (u.colorMode) {
		case COLOR_WHITE:
			return 0xFFFFFFFFu;
		case COLOR_BY_NODE:
			return remoHashColor(item.nodeKey);
		case COLOR_BY_LOD:
			return remoHashColor(item.level * 2654435761ull);
		default:
			return sampleColor;
	}
}

// One workgroup per DrawItem, work-stolen through a shared counter.
//
// This is upstream's pattern (render.cu drawNodes) and it is kept for a reason: node
// sample counts vary by orders of magnitude, so a static partition leaves most blocks
// idle waiting for the fat ones. Work stealing also makes the Walker's `kind`
// BLOCK-uniform, so there is no intra-warp divergence from mixed storage layouts.
//
// PointWalker and VoxelWalker are separate template parameters because a pipeline may
// store leaf points and inner-node voxels differently -- SimLOD in fact does, using
// separate chunk lists for each.
template <typename PointWalker, typename VoxelWalker>
void remoRasterizeDrawList(const DrawList& list, uint64_t* fb,
                           const SharedUniforms& u) {
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	// A VIEW toggle, not a selection change. Selection has already run and the draw
	// list is intact, so diag->drawItems / drawSamples still report what the pipeline
	// CHOSE -- turning points off does not make a frame look cheaper than it was.
	// Hiding them is how the octree wireframe becomes readable: four million samples
	// bury the cubes that bound them.
	if (u.showPoints == 0) return;

	// Read the count once per block rather than once per iteration.
	__shared__ uint32_t sh_numItems;

	if (block.thread_rank() == 0) {
		sh_numItems = *list.numItems;
		if (sh_numItems > list.capacity) sh_numItems = list.capacity;
	}
	block.sync();

	const uint32_t numItems = sh_numItems;
	if (numItems == 0u) return;

	// Blocks claim items by striding over their own block index; no atomics needed and
	// it is deterministic, which matters for reproducible golden images.
	for (uint32_t itemIndex = blockIdx.x; itemIndex < numItems;
	     itemIndex += gridDim.x) {

		const DrawItem item = list.items[itemIndex];

		const uint32_t lane = block.thread_rank();
		const uint32_t stride = block.num_threads();

		PointWalker::forEachStrided(
			item.points, lane, stride, [&](uint32_t, const Point& p) {
				remoDrawPoint(fb, u, p.x, p.y, p.z,
				              remoSampleColor(u, item, p.color));
			});

		// The mask test is DrawItem-uniform, hence block-uniform here, so this branch
		// costs nothing in divergence and keeps the octant arithmetic off the hot path
		// for the common all-octants case.
		if (item.voxelOctantMask == 0xFFu) {
			VoxelWalker::forEachStrided(
				item.voxels, lane, stride, [&](uint32_t, const Point& p) {
					remoDrawPoint(fb, u, p.x, p.y, p.z,
					              remoSampleColor(u, item, p.color));
				});
		} else if (item.voxelOctantMask != 0u) {
			VoxelWalker::forEachStrided(
				item.voxels, lane, stride, [&](uint32_t, const Point& p) {
					const uint32_t octant = remoOctantOf(item, p.x, p.y, p.z);
					if ((item.voxelOctantMask & (1u << octant)) == 0u) return;
					remoDrawPoint(fb, u, p.x, p.y, p.z,
					              remoSampleColor(u, item, p.color));
				});
		}
	}
}

// Convenience for pipelines whose points and voxels share one storage layout.
template <typename Walker>
void remoRasterizeDrawList(const DrawList& list, uint64_t* fb,
                           const SharedUniforms& u) {
	remoRasterizeDrawList<Walker, Walker>(list, fb, u);
}

// Total samples referenced by the list, for the stats panel's "visible samples".
inline void remoCountDrawList(const DrawList& list, uint64_t* outPoints,
                              uint64_t* outVoxels) {
	const uint32_t numItems = *list.numItems < list.capacity ? *list.numItems
	                                                         : list.capacity;
	processRangeStrided(numItems, [&](uint64_t i) {
		atomicAdd(reinterpret_cast<unsigned long long*>(outPoints),
		          static_cast<unsigned long long>(list.items[i].points.count));
		atomicAdd(reinterpret_cast<unsigned long long*>(outVoxels),
		          static_cast<unsigned long long>(list.items[i].voxels.count));
	});
}

}  // namespace remo
