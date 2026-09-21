#pragma once

#include "shared/remo_framebuffer.cuh"

namespace remo {

struct SampleSource {
	void* head;
	uint32_t count;
	uint32_t stride;
};

struct DrawItem {
	SampleSource points;
	SampleSource voxels;
	vec3f nodeMin;
	float nodeSize;
	uint32_t level;
	uint32_t nodeKey;

	uint32_t voxelOctantMask;
};

inline uint32_t remoOctantOf(const DrawItem& item, float x, float y, float z) {
	const float half = item.nodeSize * 0.5f;
	const uint32_t ix = x >= item.nodeMin.x + half ? 1u : 0u;
	const uint32_t iy = y >= item.nodeMin.y + half ? 1u : 0u;
	const uint32_t iz = z >= item.nodeMin.z + half ? 1u : 0u;
	return ix | (iy << 1) | (iz << 2);
}

struct DrawList {
	DrawItem* items;
	uint32_t* numItems;
	uint32_t capacity;
	uint32_t* overflowed;
};

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

inline bool remoDrawListAppend(const DrawList& list, const DrawItem& item) {
	const uint32_t index = atomicAdd(list.numItems, 1u);
	if (index >= list.capacity) {
		*list.overflowed = 1u;
		return false;
	}
	list.items[index] = item;
	return true;
}

struct RemoContiguousWalker {
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

template <typename ChunkT, uint32_t Capacity>
struct RemoChunkedWalker {
	template <typename Fn>
	static void forEachStrided(const SampleSource& source, uint32_t offset,
	                           uint32_t stride, Fn&& fn) {
		const ChunkT* chunk = reinterpret_cast<const ChunkT*>(source.head);
		if (chunk == nullptr || stride == 0u) return;

		uint32_t base = 0;
		while (chunk != nullptr && base < source.count) {
			const uint32_t remaining = source.count - base;
			const uint32_t inChunk = remaining < Capacity ? remaining : Capacity;

			const uint32_t delta = (offset + stride - (base % stride)) % stride;

			for (uint32_t slot = delta; slot < inChunk; slot += stride) {
				fn(base + slot,
				   *reinterpret_cast<const Point*>(&chunk->points[slot]));
			}

			base += inChunk;
			chunk = chunk->next;
		}
	}
};

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

template <typename PointWalker, typename VoxelWalker>
void remoRasterizeDrawList(const DrawList& list, uint64_t* fb,
                           const SharedUniforms& u) {
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	if (u.showPoints == 0) return;

	__shared__ uint32_t sh_numItems;

	if (block.thread_rank() == 0) {
		sh_numItems = *list.numItems;
		if (sh_numItems > list.capacity) sh_numItems = list.capacity;
	}
	block.sync();

	const uint32_t numItems = sh_numItems;
	if (numItems == 0u) return;

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

template <typename Walker>
void remoRasterizeDrawList(const DrawList& list, uint64_t* fb,
                           const SharedUniforms& u) {
	remoRasterizeDrawList<Walker, Walker>(list, fb, u);
}

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

}
