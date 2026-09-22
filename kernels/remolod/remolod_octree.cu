// Derived from SimLOD @ fa7891613c138bd41775ca72a47cd89e32a5a647, MIT,
// Copyright 2023 Markus Schuetz and Lukas Herzberger (see THIRD_PARTY.md).

// Some code in this file, particularly frustum, ray and intersection tests,
// is adapted from three.js. Three.js is licensed under the MIT license
// This file this follows the three.js licensing
// License: MIT https://github.com/mrdoob/three.js/blob/dev/LICENSE

#include <cooperative_groups.h>
#include <curand_kernel.h>

#include "../simlod/utils.h.cu"
#include "builtin_types.h"
#include "../simlod/helper_math.h"
#include "../simlod/HostDeviceInterface.h"

#include "../simlod/math.cuh"
#include "remolod_structures.cuh"

#include "../CudaPrint/CudaPrint.cuh"

#include "shared/remo_prelude.cuh"

namespace cg = cooperative_groups;

constexpr uint64_t VOXEL_BACKLOG_CAPACITY = 10'000'000;
constexpr float MAX_PROCESSING_TIME = 10.0f;

Allocator* allocator;
uint32_t* errorValue;
AllocatorGlobal* allocator_persistent;
Stats* stats = nullptr;

Point* backlog_voxels      = nullptr;
Node** backlog_targets     = nullptr;
uint32_t* numBacklogVoxels = nullptr;
Chunk** chunkQueue         = nullptr;
CudaPrint* cudaprint       = nullptr;

constexpr uint32_t SPILLING_NODES_CAPACITY = 100'000;

// The fixed-depth arm: -DREMO_FIXED_DEPTH=N, plans/07_HardCodingExpandStage.md.
// An oracle, not a method -- the depth is a compile-time constant with no
// computation behind it, so what it measures is the ceiling on what any depth
// predictor could save. The default build contains none of it.
#ifdef REMO_FIXED_DEPTH

constexpr int FIXED_DEPTH = REMO_FIXED_DEPTH;

static_assert(FIXED_DEPTH >= 1 && FIXED_DEPTH <= 8,
	"REMO_FIXED_DEPTH is 1..8. D=9 is 613 MB of cell counters against a 512 MB "
	"momentary buffer, and its grid demand exceeds the card.");

// Sigma 8^L for L < level: where level L's counters start in the flat block.
constexpr uint32_t fixedLevelOffset(int level){
	uint32_t offset = 0;
	for(int l = 0; l < level; l++) offset += (1u << (3 * l));
	return offset;
}

constexpr uint32_t FIXED_NUM_CELLS =
	fixedLevelOffset(FIXED_DEPTH) + (1u << (3 * FIXED_DEPTH));

// Dense per-level counters over the occupied part of the octree, carved from the
// momentary buffer. Nothing else in the kernel reads it.
uint32_t* cellCount = nullptr;

// Nothing ever spills at a fixed depth: points only land at depth D, so no node
// below D holds points to evacuate. The allocation shrinks to one point rather
// than disappearing, so every pointer the shared passes take stays valid while
// the 160 MB it used to cost pays for cellCount instead.
constexpr uint64_t SPILLED_POINTS_CAPACITY = 1;

#else

constexpr uint64_t SPILLED_POINTS_CAPACITY = 10'000'000;

#endif

// Phase-timing sink. Always passed by the host, in both build variants, so the
// kernel signature never depends on REMO_PROFILE. Only the marks compile out.
remo::DeviceTimeline* timeline = nullptr;

uint32_t SPECTRAL[8] = {
	0x4f3ed5,
	0x436df4,
	0x61aefd,
	0x8be0fe,
	0x98f5e6,
	0xa4ddab,
	0xa5c266,
	0xbd8832,
};

void sampleVoxel(Node* node,
	uint32_t pX_full, uint32_t pY_full, uint32_t pZ_full,
	Point point,
	float3 octreeMin, float3 octreeMax, float octreeSize
){

	if(node->grid == nullptr) return;

	uint32_t pX_leveled = pX_full / (1 << ((MAX_DEPTH + 1) - node->level));
	uint32_t pY_leveled = pY_full / (1 << ((MAX_DEPTH + 1) - node->level));
	uint32_t pZ_leveled = pZ_full / (1 << ((MAX_DEPTH + 1) - node->level));

	uint32_t pX = pX_leveled % GRID_SIZE;
	uint32_t pY = pY_leveled % GRID_SIZE;
	uint32_t pZ = pZ_leveled % GRID_SIZE;

	uint32_t voxelIndex = pX + pY * GRID_SIZE + pZ * GRID_SIZE * GRID_SIZE;
	uint32_t voxelGridElementIndex = voxelIndex / 32;
	uint32_t voxelGridElementBitIndex = voxelIndex % 32;

	uint32_t bitmask = 1 << voxelGridElementBitIndex;
	uint32_t old_nonatomic = node->grid->values[voxelGridElementIndex];
	if((old_nonatomic & bitmask) != 0) return;

	uint32_t old = atomicOr(&node->grid->values[voxelGridElementIndex], bitmask);

	if((old & bitmask) == 0){
		atomicAdd(&node->numVoxels, 1);

		float nodeSize = octreeSize / pow(2.0f, float(node->level));

		float nodeMin_x = (float(node->X) + 0.0f) * nodeSize + octreeMin.x;
		float nodeMin_y = (float(node->Y) + 0.0f) * nodeSize + octreeMin.y;
		float nodeMin_z = (float(node->Z) + 0.0f) * nodeSize + octreeMin.z;

		Point voxel;
		voxel.x = nodeMin_x + nodeSize * (float(pX) + 0.5f) / float(GRID_SIZE);
		voxel.y = nodeMin_y + nodeSize * (float(pY) + 0.5f) / float(GRID_SIZE);
		voxel.z = nodeMin_z + nodeSize * (float(pZ) + 0.5f) / float(GRID_SIZE);
		voxel.color = point.color;

		uint32_t backlogIndex = atomicAdd(numBacklogVoxels, 1);
		backlog_voxels[backlogIndex] = voxel;
		backlog_targets[backlogIndex] = node;
	}
}

bool doCounting(
	Node* root, Point* points, int numPoints,
	float3 octreeMin, float3 octreeMax, float octreeSize,
	Node* nodes,
	Node** spillingNodes, uint32_t* numSpillingNodes,
	Point* spilledPoints, uint32_t* numSpilledPoints,
	uint32_t countIteration
){
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	auto t_00 = nanotime();

	float fGridSize = pow(2.0f, float(MAX_DEPTH));

	*numSpillingNodes = 0;

	grid.sync();

	auto countPoint = [&](Point point){

		uint32_t X = fGridSize * (point.x - octreeMin.x) / octreeSize;
		uint32_t Y = fGridSize * (point.y - octreeMin.y) / octreeSize;
		uint32_t Z = fGridSize * (point.z - octreeMin.z) / octreeSize;

		uint32_t pX = MAX_DEPTH_GRIDSIZE * (point.x - octreeMin.x) / octreeSize;
		uint32_t pY = MAX_DEPTH_GRIDSIZE * (point.y - octreeMin.y) / octreeSize;
		uint32_t pZ = MAX_DEPTH_GRIDSIZE * (point.z - octreeMin.z) / octreeSize;

		Node* current = root;

		int level = 0;
		uint32_t level_X;
		uint32_t level_Y;
		uint32_t level_Z;
		uint32_t child_X;
		uint32_t child_Y;
		uint32_t child_Z;
		uint32_t childIndex;

		for(; level < MAX_DEPTH; level++){

			level_X = X >> (MAX_DEPTH - level - 1);
			level_Y = Y >> (MAX_DEPTH - level - 1);
			level_Z = Z >> (MAX_DEPTH - level - 1);

			child_X = level_X & 1;
			child_Y = level_Y & 1;
			child_Z = level_Z & 1;

			childIndex = (child_X << 2) | (child_Y << 1) | child_Z;

			if(current->children[childIndex] == nullptr){
				break;
			}else{
				current = current->children[childIndex];
			}
		}

		Node* leaf = current;

		if(leaf->countIteration < countIteration){

			uint64_t leafptr = uint64_t(leaf);
			auto warp = cg::coalesced_threads();
			auto group = cg::labeled_partition(warp, leafptr);

			uint32_t old = 0;
			if(group.thread_rank() == 0){
				old = atomicAdd(&leaf->counter, group.num_threads());

				if(old <= MAX_POINTS_PER_NODE)
				if(old + group.num_threads() > MAX_POINTS_PER_NODE)
				{
					uint32_t spillIndex = atomicAdd(numSpillingNodes, 1);
					spillingNodes[spillIndex] = leaf;
				}
			}
		}
	};

	auto t_10 = nanotime();

	processRange(numPoints, [&](int pointID){
		Point point = points[pointID];

		countPoint(point);
	});

	grid.sync();

	auto t_20 = nanotime();

	processRange(*numSpilledPoints, [&](int pointID){
		Point point = spilledPoints[pointID];

		countPoint(point);
	});

	grid.sync();
	auto t_30 = nanotime();

	uint32_t realNumSpillingNodes = *numSpillingNodes;

	grid.sync();

	__shared__ int sh_nodeIndex;

	while(true){
		block.sync();

		if(block.thread_rank() == 0){
			sh_nodeIndex = atomicAdd(numSpillingNodes, 1) - realNumSpillingNodes;
		}

		block.sync();

		if(sh_nodeIndex >= realNumSpillingNodes) break;

		Node* node = spillingNodes[sh_nodeIndex];

		int numChunks = (node->numPoints + POINTS_PER_CHUNK - 1) / POINTS_PER_CHUNK;
		Chunk* chunk = node->points;
		int chunkIndex = 0;

		for(
			int pointIndex = block.thread_rank();
			pointIndex < node->numPoints;
			pointIndex += block.num_threads()
		){

			int targetChunkIndex = pointIndex / POINTS_PER_CHUNK;

			if(chunkIndex < targetChunkIndex){
				chunk = chunk->next;
				chunkIndex++;
			}

			int pIndex = pointIndex % POINTS_PER_CHUNK;
			Point point = chunk->points[pIndex];
			uint32_t spillID = atomicAdd(numSpilledPoints, 1);
			spilledPoints[spillID] = point;
		}
	}

	grid.sync();
	*numSpillingNodes = realNumSpillingNodes;

	grid.sync();
	auto t_40 = nanotime();

	processRange(stats->numNodes, [&](int nodeIndex){
		nodes[nodeIndex].countIteration = countIteration;
	});

	grid.sync();
	auto t_50 = nanotime();

	return *numSpillingNodes == 0;
}

bool doSplitting(Node* nodes, Node** spillingNodes, uint32_t* numSpillingNodes){
	auto grid = cg::this_grid();

	processRange(*numSpillingNodes, [&](int spillNodeIndex){
		Node* spillingNode = spillingNodes[spillNodeIndex];

		uint32_t childOffset = atomicAdd(&stats->numNodes, 8);

		// The node pool's only device-side bound. numNodes is a bump index and the
		// host clamp in readStats was the only place exhaustion was ever noticed --
		// by which point nodes[childOffset + i] below has already written past the
		// pool. Unconditional, in both arms: below MAX_NODES_CAPACITY it cannot
		// change what is built, which is how it is verified (the default tree must
		// still be 4,137 / 12,742,751). Pre-splitting to a fixed depth is exactly
		// the "splits more eagerly" case CLAUDE.md warns about.
		if(childOffset + 8 > MAX_NODES_CAPACITY){
			if(timeline != nullptr) timeline->nodePoolOverflow = 1;
			return;   // this node stays a leaf
		}

		for(int i = 0; i < 8; i++){

			int cx = (i >> 2) & 1;
			int cy = (i >> 1) & 1;
			int cz = (i >> 0) & 1;

			Node child;
			child.counter = 0;
			child.numPoints = 0;
			child.points = nullptr;
			memset(&child.children, 0, 8 * sizeof(Node*));
			child.level = spillingNode->level + 1;
			child.X = 2 * spillingNode->X + cx;
			child.Y = 2 * spillingNode->Y + cy;
			child.Z = 2 * spillingNode->Z + cz;
			child.countIteration = 0;
			memcpy(&child.name[0], &spillingNode->name[0], 20);
			child.name[child.level] = i + '0';
			child.numVoxels = 0;
			child.numVoxelsStored = 0;

			nodes[childOffset + i] = child;

			spillingNode->children[i] = &nodes[childOffset + i];
		}

		Chunk* chunk = spillingNode->points;
		while(chunk != nullptr){

			Chunk* next = chunk->next;

			chunk->next = nullptr;
			int32_t oldIndex = atomicAdd(&stats->numAllocatedChunks, -1);
			int32_t newIndex = oldIndex - 1;
			chunkQueue[newIndex] = chunk;

			chunk = next;
		}

		spillingNode->numPoints = 0;
		spillingNode->points = nullptr;

		if(spillingNode->grid == nullptr){
			spillingNode->grid = (OccupancyGrid*)allocator_persistent->alloc(sizeof(OccupancyGrid));
		}
	});

	grid.sync();

	uint32_t numElements = *numSpillingNodes * (GRID_NUM_CELLS / 32u);
	grid.sync();
	processRange(numElements, [&](int cellIndex){
		int gridIndex = cellIndex / (GRID_NUM_CELLS / 32u);
		int localCellIndex = cellIndex % (GRID_NUM_CELLS / 32u);

		Node* spillingNode = spillingNodes[gridIndex];

		// A node whose split the pool check refused never reached the grid
		// allocation above and is still a leaf with a null grid. Without this the
		// safety check is itself the fault: D=8 on morro_bay exhausts the pool and
		// every thread in its zeroing range dereferences nullptr.
		if(spillingNode->grid == nullptr) return;

		spillingNode->grid->values[localCellIndex] = 0;
	});

	grid.sync();

	// A refused split above still bumped numNodes, and every later pass sweeps
	// [0, numNodes). Pin it back to the pool so those sweeps stay inside it; a
	// further split still fails the check, because the clamped value is itself
	// over capacity by the time 8 more are asked for.
	if(grid.thread_rank() == 0 && stats->numNodes > MAX_NODES_CAPACITY){
		stats->numNodes = MAX_NODES_CAPACITY;
	}

	grid.sync();
}

void expand(
	Node* root, Point* points, int numPoints,
	float3 octreeMin, float3 octreeMax, float octreeSize,
	Node* nodes,
	Node** spillingNodes, uint32_t* numSpillingNodes,
	Point* spilledPoints, uint32_t* numSpilledPoints,
	uint32_t batchIndex
){
	auto grid = cg::this_grid();
	for(int i = 0; i < 20; i++){

		grid.sync();

		// Iteration 0's counting pass is irreducible; 1..N are what a predicted
		// depth would delete. They are separated here, not in the mark stream,
		// because a mark pair per iteration would overflow REMO_MAX_MARKS.
		// nanotime() is asm volatile, so the read itself is guarded too — the
		// default build must not pay for a clock read it never stores.
#ifdef REMO_PROFILE
		const uint64_t t_iter = nanotime();
#endif

		bool isFinished = doCounting(root, points, numPoints,
			octreeMin, octreeMax, octreeSize,
			nodes,
			spillingNodes, numSpillingNodes,
			spilledPoints, numSpilledPoints,
			batchIndex + 1);

		grid.sync();

		// Counting only. Splitting is deliberately outside this window, so
		// expand_total - sum(expandIterNs) isolates doSplitting's cost.
#ifdef REMO_PROFILE
		if(timeline != nullptr && grid.thread_rank() == 0){
			timeline->expandIters += 1;
			timeline->nodesSplit += isFinished ? 0u : *numSpillingNodes;
			if(i < int(remo::REMO_MAX_EXPAND_ITERS)){
				timeline->expandIterNs[i] += nanotime() - t_iter;
			}
		}
#endif

		if(isFinished) break;

		doSplitting(nodes, spillingNodes, numSpillingNodes);

		grid.sync();
	}

	grid.sync();

	// Once per batch, not per iteration: numSpilledPoints is reset in addBatch
	// and only ever grows within a batch, so adding it each iteration would
	// count the same points again at every pass.
#ifdef REMO_PROFILE
	if(timeline != nullptr && grid.thread_rank() == 0){
		timeline->spilledPoints += *numSpilledPoints;
	}
#endif
}

#ifdef REMO_FIXED_DEPTH

// The same bit order as childIndex in doCounting -- (x << 2) | (y << 1) | z, most
// significant level first. That is what makes a cell's parent exactly cell >> 3,
// so the pyramid below is a shift rather than a remap.
uint32_t fixedMorton(uint32_t x, uint32_t y, uint32_t z, int level){
	uint32_t code = 0;

	for(int l = 0; l < level; l++){
		uint32_t bit = level - 1 - l;
		code = (code << 3)
			| (((x >> bit) & 1u) << 2)
			| (((y >> bit) & 1u) << 1)
			| ((z >> bit) & 1u);
	}

	return code;
}

// expand() with the depth known in advance.
//
// The descent disappears, not just the loop: at a fixed depth the destination
// cell is X >> (MAX_DEPTH - D), computable from the point's own coordinates with
// no pointer chasing, so one analytic bucketing pass replaces the 3.46 root-to-
// leaf counting descents per batch that plans/05_ExpandMeasure.md measured. It
// also yields Node::counter for free, which is allocatePointChunks' only sizing
// input.
//
// doSplitting is called unchanged: child allocation, chunk recycling and the
// occupancy grid allocate-and-zero keep their existing behaviour, and this pass
// only chooses WHICH nodes go into spillingNodes.
//
// It takes no root, which is the whole point: nothing here walks the tree from
// the top, so there is nothing to walk it from.
void expandFixed(
	Point* points, int numPoints,
	float3 octreeMin, float octreeSize,
	Node* nodes,
	Node** spillingNodes, uint32_t* numSpillingNodes
){
	auto grid = cg::this_grid();

	const float fGridSize      = pow(2.0f, float(MAX_DEPTH));
	constexpr uint32_t shift    = MAX_DEPTH - FIXED_DEPTH;
	constexpr uint32_t maxCoord = (1u << FIXED_DEPTH) - 1u;
	constexpr uint32_t depthOffset = fixedLevelOffset(FIXED_DEPTH);

	// Per batch, not per launch: counter accumulates across batches and is reset
	// only when a node splits, so pass 4 adds this batch's counts to it.
	processRange(FIXED_NUM_CELLS, [&](int cell){
		cellCount[cell] = 0;
	});

	grid.sync();

	// 1. bucket -- proportional to points, and the only pass that is. No tree
	//    access at all.
#ifdef REMO_PROFILE
	const uint64_t t_bucket = nanotime();
#endif

	processRange(numPoints, [&](int pointID){
		Point point = points[pointID];

		uint32_t X = fGridSize * (point.x - octreeMin.x) / octreeSize;
		uint32_t Y = fGridSize * (point.y - octreeMin.y) / octreeSize;
		uint32_t Z = fGridSize * (point.z - octreeMin.z) / octreeSize;

		// Masked, not clamped. A point exactly on boxMax quantises to 2^MAX_DEPTH
		// and indexes one cell past the level; the descent this replaces absorbs
		// that in its per-level & 1, which WRAPS it to cell 0 rather than pinning
		// it to the last cell. Clamping instead left 30 leaves on morro_bay with
		// counter < numPoints -- insertPoints descended where this pass had not
		// counted, and allocatePointChunks under-allocated them. The rule is that
		// this index must be bit-for-bit what the descent computes, whatever the
		// descent computes.
		uint32_t cx = (X >> shift) & maxCoord;
		uint32_t cy = (Y >> shift) & maxCoord;
		uint32_t cz = (Z >> shift) & maxCoord;

		atomicAdd(&cellCount[depthOffset + fixedMorton(cx, cy, cz, FIXED_DEPTH)], 1u);
	});

	grid.sync();

	// 2. pyramid -- proportional to cells, independent of the point count.
#ifdef REMO_PROFILE
	const uint64_t t_pyramid = nanotime();
#endif

	for(int level = FIXED_DEPTH - 1; level >= 0; level--){
		const uint32_t numCells = 1u << (3 * level);
		uint32_t* dst = cellCount + fixedLevelOffset(level);
		uint32_t* src = cellCount + fixedLevelOffset(level + 1);

		processRange(numCells, [&](int cell){
			uint32_t sum = 0;
			for(int i = 0; i < 8; i++) sum += src[8 * cell + i];
			dst[cell] = sum;
		});

		grid.sync();
	}

	// 3. materialise -- D grid-wide rounds, root downwards. Every childless node
	//    at level L whose cell holds points is split, once.
#ifdef REMO_PROFILE
	const uint64_t t_materialise = nanotime();
	uint32_t splitThisBatch = 0;
#endif

	for(int level = 0; level < FIXED_DEPTH; level++){

		if(grid.thread_rank() == 0) *numSpillingNodes = 0;

		grid.sync();

		const uint32_t numNodes = stats->numNodes;
		const uint32_t levelOffset = fixedLevelOffset(level);

		// Every node, not just the ones this batch created: a leaf left empty by
		// an earlier batch sits at a level above D and has to split the moment
		// points reach it.
		processRange(numNodes, [&](int nodeIndex){
			Node* node = &nodes[nodeIndex];

			if(node->level != uint32_t(level)) return;
			if(!node->isLeafFn()) return;

			uint32_t cell = fixedMorton(node->X, node->Y, node->Z, level);
			if(cellCount[levelOffset + cell] == 0) return;

			uint32_t spillIndex = atomicAdd(numSpillingNodes, 1);
			if(spillIndex < SPILLING_NODES_CAPACITY){
				spillingNodes[spillIndex] = node;
			}
		});

		grid.sync();

		// The list is the one thing between here and a write past its end. A node
		// that does not fit stays a leaf above D, and pass 4 still gives it the
		// whole subtree's count, so the tree is shallower but not wrong.
		if(grid.thread_rank() == 0 && *numSpillingNodes > SPILLING_NODES_CAPACITY){
			*numSpillingNodes = SPILLING_NODES_CAPACITY;
			if(timeline != nullptr) timeline->nodePoolOverflow = 1;
		}

		grid.sync();

		if(*numSpillingNodes == 0) continue;

#ifdef REMO_PROFILE
		splitThisBatch += *numSpillingNodes;
#endif

		doSplitting(nodes, spillingNodes, numSpillingNodes);

		grid.sync();
	}

	// 4. seed -- proportional to nodes. allocatePointChunks sizes every leaf's
	//    chunk list from node->counter and nothing else, and the invariant it
	//    needs is counter >= the number of points insertPoints will push. The
	//    bucket pass satisfies it exactly, with no counting descent.
#ifdef REMO_PROFILE
	const uint64_t t_seed = nanotime();
#endif

	const uint32_t numNodes = stats->numNodes;

	processRange(numNodes, [&](int nodeIndex){
		Node* node = &nodes[nodeIndex];

		if(!node->isLeafFn()) return;
		if(node->level > uint32_t(FIXED_DEPTH)) return;

		// Every leaf, at whatever level it ended up. Normally that is depth D; a
		// leaf above D exists only where the node pool or the spilling list ran
		// out, and the pyramid means its count is already the subtree's total.
		uint32_t cell = fixedMorton(node->X, node->Y, node->Z, node->level);
		uint32_t count = cellCount[fixedLevelOffset(node->level) + cell];
		if(count == 0) return;

		node->counter += count;

		if(node->level == uint32_t(FIXED_DEPTH) && timeline != nullptr){
			atomicAdd(&timeline->occupiedCells, 1u);
		}
	});

	grid.sync();

#ifdef REMO_PROFILE
	const uint64_t t_end = nanotime();

	if(timeline != nullptr && grid.thread_rank() == 0){
		timeline->bucketNs      += t_pyramid - t_bucket;
		timeline->pyramidNs     += t_materialise - t_pyramid;
		timeline->materialiseNs += t_seed - t_materialise;
		timeline->seedNs        += t_end - t_seed;

		// One point-proportional pass over the batch, against 3.46 in the
		// baseline. That single counter is the clearest statement of what this
		// arm does.
		timeline->expandIters += 1;
		timeline->nodesSplit  += splitThisBatch;
	}
#endif
}

#endif

void voxelSampling(
	Node* root, Point* points, int numPoints,
	float3 octreeMin, float3 octreeMax, float octreeSize,
	Point* spilledPoints, uint32_t* numSpilledPoints
){
	auto grid = cg::this_grid();

	float fGridSize = pow(2.0f, float(MAX_DEPTH));

	auto traverse = [&](Point point){

		uint32_t X = fGridSize * (point.x - octreeMin.x) / octreeSize;
		uint32_t Y = fGridSize * (point.y - octreeMin.y) / octreeSize;
		uint32_t Z = fGridSize * (point.z - octreeMin.z) / octreeSize;

		uint32_t pX = MAX_DEPTH_GRIDSIZE * (point.x - octreeMin.x) / octreeSize;
		uint32_t pY = MAX_DEPTH_GRIDSIZE * (point.y - octreeMin.y) / octreeSize;
		uint32_t pZ = MAX_DEPTH_GRIDSIZE * (point.z - octreeMin.z) / octreeSize;

		Node* current = root;

		int level = 0;
		uint32_t level_X;
		uint32_t level_Y;
		uint32_t level_Z;
		uint32_t child_X;
		uint32_t child_Y;
		uint32_t child_Z;
		uint32_t childIndex;

		for(; level < MAX_DEPTH; level++){

			level_X = X >> (MAX_DEPTH - level - 1);
			level_Y = Y >> (MAX_DEPTH - level - 1);
			level_Z = Z >> (MAX_DEPTH - level - 1);

			child_X = level_X & 1;
			child_Y = level_Y & 1;
			child_Z = level_Z & 1;

			childIndex = (child_X << 2) | (child_Y << 1) | child_Z;

			sampleVoxel(current, pX, pY, pZ, point, octreeMin, octreeMax, octreeSize);

			if(current->children[childIndex] == nullptr){
				break;
			}else{
				current = current->children[childIndex];
			}
		}
	};

	processRange(numPoints, [&](int index){
		Point point = points[index];

		traverse(point);
	});

	processRange(*numSpilledPoints, [&](int index){
		Point point = spilledPoints[index];

		traverse(point);
	});
}

void allocatePointChunks(Node* root, Point* points, int numPoints, Node* nodes){
	auto grid = cg::this_grid();

	processRange(stats->numNodes, [&](int nodeIndex){
		Node* node = &nodes[nodeIndex];

		if(node->isLeafFn())
		if(node->numPoints < node->counter){

			int numRequiredChunks = (node->counter + POINTS_PER_CHUNK - 1) / POINTS_PER_CHUNK;
			int numExistingChunks = (node->numPoints + POINTS_PER_CHUNK - 1) / POINTS_PER_CHUNK;
			int numAdditionallyRequiredChunks = numRequiredChunks - numExistingChunks;

			if(numAdditionallyRequiredChunks > 0){

				Chunk* prevChunk = node->points;
				for(int i = 1; i < numExistingChunks; i++){
					prevChunk = prevChunk->next;
				}

				for(int i_chunk = 0; i_chunk < numAdditionallyRequiredChunks; i_chunk++){
					uint32_t chunkIndex = atomicAdd(&stats->numAllocatedChunks, 1);
					Chunk* chunk = nullptr;

					if(chunkIndex >= stats->chunkPoolSize){
						chunk = (Chunk*)allocator_persistent->alloc(sizeof(Chunk));
					}else{
						chunk = chunkQueue[chunkIndex];
					}

					chunk->next = nullptr;

					if(prevChunk == nullptr){
						node->points = chunk;
						prevChunk = chunk;
					}else{
						prevChunk->next = chunk;
						prevChunk = chunk;
					}
				}

			}
		}
	});

	grid.sync();

	if(grid.thread_rank() == 0){
		stats->chunkPoolSize = max(stats->chunkPoolSize, stats->numAllocatedChunks);
	}
}

void insertPoints(
	Node* root, Point* points, uint32_t numPoints,
	float3 octreeMin, float3 octreeMax, float octreeSize,
	Point* spilledPoints, uint32_t* numSpilledPoints,
	int batchIndex
){
	auto grid = cg::this_grid();

	float fGridSize = pow(2.0f, float(MAX_DEPTH));

	auto insertPoint = [&](Point point){
		uint32_t X = fGridSize * (point.x - octreeMin.x) / octreeSize;
		uint32_t Y = fGridSize * (point.y - octreeMin.y) / octreeSize;
		uint32_t Z = fGridSize * (point.z - octreeMin.z) / octreeSize;

		Node* current = root;

		int level = 0;
		uint32_t level_X;
		uint32_t level_Y;
		uint32_t level_Z;
		uint32_t child_X;
		uint32_t child_Y;
		uint32_t child_Z;
		uint32_t childIndex;

		for(; level < MAX_DEPTH; level++){

			level_X = X >> (MAX_DEPTH - level - 1);
			level_Y = Y >> (MAX_DEPTH - level - 1);
			level_Z = Z >> (MAX_DEPTH - level - 1);

			child_X = level_X & 1;
			child_Y = level_Y & 1;
			child_Z = level_Z & 1;

			childIndex = (child_X << 2) | (child_Y << 1) | child_Z;

			if(current->children[childIndex] == nullptr){
				break;
			}else{
				current = current->children[childIndex];
			}
		}

		Node* leaf = current;

		uint32_t pointInNodeIndex = atomicAdd(&leaf->numPoints, 1);
		uint32_t chunkIndex = pointInNodeIndex / POINTS_PER_CHUNK;
		uint32_t pointInChunkIndex = pointInNodeIndex % POINTS_PER_CHUNK;

		Chunk* chunk = leaf->points;

		if(chunk == nullptr){
			cudaprint->print("chunk is NULL: {} \n", (const char*)leaf->name);

			return;
		}

		for(int i = 0; i < chunkIndex; i++){
			if(chunk == 0) break;

			chunk = chunk->next;
		}

		chunk->points[pointInChunkIndex] = point;
	};

	processRange(numPoints, [&](int pointID){
		Point point = points[pointID];

		insertPoint(point);
	});

	grid.sync();

	processRange(*numSpilledPoints, [&](int pointID){

		if(pointID > 3'000'000){
			printf("too many spilled points \n");
			return;
		}

		Point point = spilledPoints[pointID];

		insertPoint(point);
	});

	grid.sync();
}

void allocateVoxelChunks(Node* nodes){
	auto grid = cg::this_grid();

	processRange(stats->numNodes, [&](int nodeIndex){

		Node* node = &nodes[nodeIndex];

		int requiredChunks = (node->numVoxels + POINTS_PER_CHUNK - 1) / POINTS_PER_CHUNK;

		if(requiredChunks == 0) return;

		if(node->voxelChunks == nullptr){
			node->voxelChunks = (Chunk*)allocator_persistent->alloc(sizeof(Chunk));
			node->voxelChunks->next = nullptr;
		}

		Chunk* chunk = node->voxelChunks;

		for(int chunkIndex = 1; chunkIndex < requiredChunks; chunkIndex++){

			if(chunk->next == nullptr){
				Chunk* newChunk = (Chunk*)allocator_persistent->alloc(sizeof(Chunk));
				newChunk->next = nullptr;
				chunk->next = newChunk;
			}

			chunk = chunk->next;

		}

	});
}

void insertVoxels(int batchIndex){
	auto grid = cg::this_grid();

	processRange(*numBacklogVoxels, [&](int index){
		Point voxel = backlog_voxels[index];
		Node* target = backlog_targets[index];

		uint32_t voxelIndex = atomicAdd(&target->numVoxelsStored, 1);
		uint32_t chunkIndex = voxelIndex / POINTS_PER_CHUNK;

		Chunk* chunk = target->voxelChunks;

		for(int i = 0; i < chunkIndex; i++){
			chunk = chunk->next;
		}

		uint32_t chunkLocalVoxelIndex = voxelIndex % POINTS_PER_CHUNK;

		chunk->points[chunkLocalVoxelIndex] = voxel;
	});
}

void addBatch(
	Point* points, uint32_t batchSize,
	int batchIndex,
	float3 octreeMin, float3 octreeMax, float octreeSize,
	Node* nodes,
	Node** spillingNodes, uint32_t* numSpillingNodes,
	Point* spilledPoints, uint32_t* numSpilledPoints
){

	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	auto t_00 = nanotime();
	REMO_MARK(timeline, remo::kPhaseBatchBegin);

	if(grid.thread_rank() == 0){
		*numSpilledPoints = 0;
		*numBacklogVoxels = 0;
	}

	grid.sync();

	Node* root = &nodes[0];

	auto t_10 = nanotime();
	REMO_MARK(timeline, remo::kPhaseExpand);

	// The two arms sit inside the same mark pair, so they are directly comparable
	// at the phase level. expand() is left exactly as it is.
#ifdef REMO_FIXED_DEPTH
	expandFixed(points, batchSize,
		octreeMin, octreeSize,
		nodes,
		spillingNodes, numSpillingNodes
	);
#else
	expand(root, points, batchSize,
		octreeMin, octreeMax, octreeSize,
		nodes,
		spillingNodes, numSpillingNodes,
		spilledPoints, numSpilledPoints,
		batchIndex
	);
#endif

	grid.sync();

	auto t_20 = nanotime();
	REMO_MARK(timeline, remo::kPhaseVoxelSampling);

	voxelSampling(root, points, batchSize,
		octreeMin, octreeMax, octreeSize,
		spilledPoints, numSpilledPoints
	);

	grid.sync();

	auto t_30 = nanotime();
	REMO_MARK(timeline, remo::kPhaseAllocPointChunks);

	allocatePointChunks(root, points, batchSize, nodes);

	grid.sync();

	auto t_40 = nanotime();
	REMO_MARK(timeline, remo::kPhaseAllocVoxelChunks);

	allocateVoxelChunks(nodes);

	grid.sync();

	auto t_50 = nanotime();
	REMO_MARK(timeline, remo::kPhaseInsertPoints);

	insertPoints(root, points, batchSize,
		octreeMin, octreeMax, octreeSize,
		spilledPoints, numSpilledPoints, batchIndex
	);

	grid.sync();

	auto t_60 = nanotime();
	REMO_MARK(timeline, remo::kPhaseInsertVoxels);

	insertVoxels(batchIndex);

	grid.sync();

	auto t_70 = nanotime();
	REMO_MARK(timeline, remo::kPhaseBatchEnd);

	grid.sync();

	REMO_PROFILE_ONLY(
		if(timeline != nullptr && grid.thread_rank() == 0){
			timeline->batches += 1;
		}
	);

	if(grid.thread_rank() == 0){
		float t_00_70 = double(t_70 - t_00) / 1'000'000.0;
		float t_00_10 = double(t_10 - t_00) / 1'000'000.0;
		float t_10_20 = double(t_20 - t_10) / 1'000'000.0;
		float t_20_30 = double(t_30 - t_20) / 1'000'000.0;
		float t_30_40 = double(t_40 - t_30) / 1'000'000.0;
		float t_40_50 = double(t_50 - t_40) / 1'000'000.0;
		float t_50_60 = double(t_60 - t_50) / 1'000'000.0;
		float t_60_70 = double(t_70 - t_60) / 1'000'000.0;

		cudaprint->print("t_00_70: {:.3f}, t_00_10: {:.3f}, expand: {:.3f}, createVoxels: {:.3f}, t_30_40: {:.3f}, t_40_50: {:.3f}, insertPoints: {:.3f}, t_60_70: {:.3f} \n",
			t_00_70,
			t_00_10,
			t_10_20,
			t_20_30,
			t_30_40,
			t_40_50,
			t_50_60,
			t_60_70
		);
	}
}

extern "C" __global__
void kernel_construct(
	const Uniforms uniforms,
	Point* points,
	uint32_t* buffer,
	uint8_t* buffer_persistent,
	Node* nodes,
	Stats* _stats,
	uint64_t* frameStartTimestamp,
	CudaPrint* _cudaprint,
	uint32_t* _numBatchesUploaded_volatile,
	uint32_t* batchSizes,
	remo::DeviceTimeline* _timeline
){
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();
	auto tStart = nanotime();

	cudaprint = _cudaprint;
	timeline = _timeline;

	// Per launch, not per cloud: the host reads the timeline after every
	// construct, so each launch starts from zero. remolod_reset.cu runs only on
	// cloud reset and cannot do this.
	//
	// Unconditional, unlike the marks it precedes: the counter block carries
	// nodePoolOverflow and maxPointsPerNode, which are written in every variant.
	// It is a few dozen stores by one thread, not a clock read in a hot loop.
	if(timeline != nullptr && grid.thread_rank() == 0){
		timeline->numMarks         = 0;
		timeline->overflow         = 0;
		timeline->batches          = 0;
		timeline->expandIters      = 0;
		timeline->nodesSplit       = 0;
		timeline->maxPointsPerNode = 0;
		timeline->spilledPoints    = 0;
		for(uint32_t i = 0; i < remo::REMO_MAX_EXPAND_ITERS; i++){
			timeline->expandIterNs[i] = 0;
		}
		timeline->bucketNs         = 0;
		timeline->pyramidNs        = 0;
		timeline->materialiseNs    = 0;
		timeline->seedNs           = 0;
		timeline->occupiedCells     = 0;
		timeline->nodePoolOverflow  = 0;
		timeline->counterUnderflows = 0;
		timeline->pad2              = 0;
	}

	if(grid.thread_rank() == 0){
		*frameStartTimestamp = tStart;
	}
	grid.sync();

	stats = _stats;

	Allocator _allocator(buffer, 0);
	allocator = &_allocator;
	allocator_persistent = (AllocatorGlobal*)buffer_persistent;

	uint32_t& numBatchesUploaded_global = *allocator->alloc<uint32_t*>(4);

	grid.sync();

	uint64_t& ellapsed_nanos     = *allocator->alloc<uint64_t*>(8);

	backlog_voxels     = allocator->alloc<Point*>(VOXEL_BACKLOG_CAPACITY * sizeof(Point));
	backlog_targets    = allocator->alloc<Node**>(VOXEL_BACKLOG_CAPACITY * sizeof(Node*));
	numBacklogVoxels   = allocator->alloc<uint32_t*>(4);

	Node** spillingNodes         =  allocator->alloc<Node**>(SPILLING_NODES_CAPACITY * sizeof(Node*));
	uint32_t* numSpillingNodes   =  allocator->alloc<uint32_t*>(4);

	Point* spilledPoints = allocator->alloc<Point*>(SPILLED_POINTS_CAPACITY * sizeof(Point));
	uint32_t* numSpilledPoints = allocator->alloc<uint32_t*>(4);

	chunkQueue = allocator->alloc<Chunk**>(sizeof(Chunk*) * 1'000'000);

	// RemoAllocator's uniform-control-flow rule is satisfied because the variant
	// is chosen at compile time: each variant walks one fixed allocation sequence.
#ifdef REMO_FIXED_DEPTH
	cellCount = allocator->alloc<uint32_t*>(uint64_t(FIXED_NUM_CELLS) * sizeof(uint32_t));
#endif

	grid.sync();

	float3 boxSize      = uniforms.boxMax - uniforms.boxMin;
	float octreeSize    = max(max(boxSize.x, boxSize.y), boxSize.z);
	float3 octreeMin    = uniforms.boxMin;
	float3 octreeMax    = octreeMin + octreeSize;
	float3 cubePosition = uniforms.boxMin + octreeSize * 0.5f;

	auto tStartExpand = nanotime();

	grid.sync();

	if(grid.thread_rank() == 0){
		numBatchesUploaded_global = *_numBatchesUploaded_volatile;
	}
	grid.sync();
	uint32_t numBatchesUploaded = atomicAdd(&numBatchesUploaded_global, 0);

	grid.sync();

	constexpr int MAX_SUBPATCH_SIZE      = 1'000'000;
	constexpr uint64_t MAX_BATCH_SIZE    = 1'000'000;

	uint32_t numBatches = min(numBatchesUploaded - stats->batchletIndex, 20);
	uint32_t firstBatch = stats->batchletIndex;
	uint32_t lastBatch = firstBatch + numBatches;

	auto tStart_addBatches = nanotime();

	for(int batchIndex = firstBatch; batchIndex < lastBatch; batchIndex++){

		uint32_t ringSlotIndex = batchIndex % BATCH_STREAM_SIZE;
		uint32_t batchSize = batchSizes[ringSlotIndex];
		Point* sub_points = points + ringSlotIndex * MAX_BATCH_SIZE;

		uint64_t memCapacity = uniforms.persistentBufferCapacity;
		uint64_t memUsed = allocator_persistent->offset;

		// The margin has to cover ONE batch's allocations, because it is only
		// tested between batches and AllocatorGlobal has no bounds check to fall
		// back on -- it hands out pointers past the end and the write faults. A
		// fixed-depth batch splits hundreds of nodes at once (525/batch at D=8
		// against 14 in the baseline), each taking a 256 KB occupancy grid, plus a
		// 16 KB chunk for every new leaf that takes points. Upstream's 200 MB does
		// not cover that: D=8 on morro_bay faults in allocatePointChunks' chunk
		// write, ~200 MB past the end, without ever printing the warning.
#ifdef REMO_FIXED_DEPTH
		uint64_t safetyMargin = 1'000'000'000;
#else
		uint64_t safetyMargin = 200'000'000;
#endif
		bool memCapacityReached = memUsed + safetyMargin >= memCapacity;

		if(memCapacityReached && grid.thread_rank() == 0){
			printf("persistent memory capacity almost reached, ignoring further points. capacity: %i MB, used: %i MB \n",
				int32_t(memCapacity / (1024llu * 1024llu)),
				int32_t(memUsed / (1024llu * 1024llu))
			);

			stats->memCapacityReached = true;
		}else if(!memCapacityReached){
			stats->memCapacityReached = false;
		}

		if(memCapacityReached) break;

		addBatch(
			sub_points, batchSize,
			stats->batchletIndex,
			octreeMin, octreeMax, octreeSize,
			nodes,
			spillingNodes, numSpillingNodes,
			spilledPoints, numSpilledPoints
		);

		grid.sync();

		if(grid.thread_rank() == 0){
			atomicAdd(&stats->batchletIndex, 1);
			atomicAdd(&stats->numPointsProcessed, batchSize);
		}

		grid.sync();

		if(grid.thread_rank() == 0){
			ellapsed_nanos = nanotime() - (*frameStartTimestamp);
		}

		grid.sync();

		float ellapsed_ms = float(ellapsed_nanos) / 1'000'000.0f;
		if(ellapsed_ms > MAX_PROCESSING_TIME){

			break;
		}
	}

	auto tEndExpand = nanotime();
	float durationExpandMS = double(tEndExpand - tStartExpand) / 1'000'000.0;

	grid.sync();

	uint32_t* counter_inner           = allocator->alloc<uint32_t*>(4);
	uint32_t* counter_leaves          = allocator->alloc<uint32_t*>(4);
	uint32_t* counter_nonempty_leaves = allocator->alloc<uint32_t*>(4);
	uint32_t* counter_points          = allocator->alloc<uint32_t*>(4);
	uint32_t* counter_voxels          = allocator->alloc<uint32_t*>(4);
	uint32_t* counter_chunks_points   = allocator->alloc<uint32_t*>(4);
	uint32_t* counter_chunks_voxels   = allocator->alloc<uint32_t*>(4);
	uint32_t* counter_max_points      = allocator->alloc<uint32_t*>(4);
	uint32_t* counter_underflows      = allocator->alloc<uint32_t*>(4);

	if(grid.thread_rank() == 0){
		*counter_inner           = 0;
		*counter_leaves          = 0;
		*counter_nonempty_leaves = 0;
		*counter_points          = 0;
		*counter_voxels          = 0;
		*counter_chunks_points   = 0;
		*counter_chunks_voxels   = 0;
		*counter_max_points      = 0;
		*counter_underflows      = 0;
	}
	grid.sync();

	processRange(stats->numNodes, [&](int nodeIndex){
		Node* node = &nodes[nodeIndex];

		if(node->isLeafFn()){
			atomicAdd(counter_leaves, 1);
			atomicAdd(counter_points, node->numPoints);
			atomicAdd(counter_chunks_points, (node->numPoints + POINTS_PER_CHUNK - 1) / POINTS_PER_CHUNK);
			atomicMax(counter_max_points, node->numPoints);

			// The invariant allocatePointChunks depends on, checked where the tree
			// is already being swept. Both arms: the baseline must report 0 too.
			if(node->counter < node->numPoints){
				atomicAdd(counter_underflows, 1);
			}

			if(node->numPoints > 0){
				atomicAdd(counter_nonempty_leaves, 1);
			}
		}else{
			atomicAdd(counter_inner, 1);
			atomicAdd(counter_voxels, node->numVoxels);
			atomicAdd(counter_chunks_voxels, (node->numVoxels + POINTS_PER_CHUNK - 1) / POINTS_PER_CHUNK);
		}

	});

	grid.sync();

	if(grid.thread_rank() == 0){
		stats->numInner                  = *counter_inner;
		stats->numLeaves                 = *counter_leaves;
		stats->numNonemptyLeaves         = *counter_nonempty_leaves;
		stats->numPoints                 = *counter_points;
		stats->numVoxels                 = *counter_voxels;
		stats->numChunksPoints           = *counter_chunks_points;
		stats->numChunksVoxels           = *counter_chunks_voxels;
		stats->allocatedBytes_momentary  = allocator->offset;
		stats->allocatedBytes_persistent = allocator_persistent->offset;
		stats->frameID                   = uniforms.frameCounter;
	}

	// The direct measure of a too-shallow fixed depth: a depth-D cell holding
	// 200k points stays one leaf, and only this says so. Stats is vendored from
	// SimLOD and has nowhere to put it, so it rides the timeline's counter block
	// rather than growing kernel_construct a twelfth argument.
	if(grid.thread_rank() == 0 && timeline != nullptr){
		timeline->maxPointsPerNode  = *counter_max_points;
		timeline->counterUnderflows = *counter_underflows;
	}
}
