// RemoLOD's fork of SimLOD's structures.cuh.
//
// FORKED, NOT VENDORED. RemoLOD is RemoBench's own pipeline and this file is ours to change;
// bench/check_vendored.sh does not police it, and it is NOT expected to stay identical to
// external/SimLOD. The comparison baseline lives in kernels/simlod/ and stays byte-identical
// -- that is the whole point of the split. Never "fix" one by editing the other.
//
// Derived from SimLOD @ fa7891613c138bd41775ca72a47cd89e32a5a647, MIT,
// Copyright 2023 Markus Schuetz and Lukas Herzberger (see THIRD_PARTY.md).
//
// Node / Chunk / OccupancyGrid, and the tunables RemoLOD raises.

#pragma once

constexpr float PI = 3.1415;
// constexpr int MAX_POINTS_PER_NODE = 100;
// constexpr int MAX_POINTS_PER_NODE = 5'000;
// constexpr uint32_t POINTS_PER_CHUNK = 1000;
constexpr bool RIGHTSIDE_BOXES = false;
constexpr bool RIGHTSIDE_NODECOLORS = false;

constexpr bool ENABLE_TRACE = false;
// constexpr int MAX_DEPTH = 20;
// constexpr float MAX_DEPTH_GRIDSIZE = 268'435'456.0f;

// constexpr int MAX_POINTS_PER_NODE       = 5'000;
// constexpr uint32_t POINTS_PER_CHUNK     = 256;
// constexpr uint32_t GRID_SIZE            = 64;
// constexpr uint32_t GRID_NUM_CELLS       = GRID_SIZE * GRID_SIZE * GRID_SIZE;
// constexpr int MAX_DEPTH                 = 17;
// constexpr float MAX_DEPTH_GRIDSIZE      = 16'777'216.0f;

constexpr int MAX_POINTS_PER_NODE    = 50'000;

// ADDED (SimLOD has no such constant).
//
// Upstream grows its flat Node pool with `atomicAdd(&stats->numNodes, 8)` and NO capacity
// check anywhere, so an eagerly splitting tree walks off the end of the allocation. This is
// the size the host allocates the pool to, so every pass can clamp against it and the host
// can report nodeCapacityReached instead of corrupting memory.
//
// It matters more for RemoLOD than for SimLOD: Refinement's whole job is to split more
// eagerly where geometry warrants it, which walks toward this bound on purpose.
constexpr uint32_t MAX_NODES_CAPACITY = 200'000;
constexpr uint32_t POINTS_PER_CHUNK  = 1000;
constexpr uint32_t GRID_SIZE         = 128;
constexpr uint32_t GRID_NUM_CELLS    = GRID_SIZE * GRID_SIZE * GRID_SIZE;
constexpr int MAX_DEPTH              = 20;
constexpr float MAX_DEPTH_GRIDSIZE   = 268'435'456.0f;

// CHANGED FROM SimLOD (was 50).
//
// kernel_construct addresses batch N at points + (N % BATCH_STREAM_SIZE) * MAX_BATCH_SIZE --
// a ring of this many 1M-point slots. RemoBench feeds the whole cloud already resident in
// device memory, where batch N lives at points + N * MAX_BATCH_SIZE with NO wrapping. Those
// two agree only while N < BATCH_STREAM_SIZE, so at 50 anything past 50M points silently
// re-read slot 0 and built a tree from the wrong points.
//
// Raising it makes the non-wrapping addressing exact for any realistic cloud. It is used in
// exactly two places -- that modulo and a clearing loop in remolod_reset.cu -- so it changes
// no algorithm; it only costs a larger batchSizes array (4 bytes per slot).
//
// SimLOD itself keeps upstream's 50 and SimlodPipeline refuses a cloud it cannot address.
// Revert this to 50 once PointSource grows a genuinely wrapping ring.
constexpr uint64_t BATCH_STREAM_SIZE = 8192;

struct Point{
	float x;
	float y;
	float z;
	uint32_t color;
};

struct Voxel{
	uint8_t X;
	uint8_t Y;
	uint8_t Z;
	uint8_t filler;
	uint32_t color;
};

struct Lines{
	unsigned int count = 0;
	unsigned int padding0;
	unsigned int padding1;
	unsigned int padding2;
	Point* vertices;
};

float4 operator*(const mat4& a, const float4& b){
	return make_float4(
		dot(a.rows[0], b),
		dot(a.rows[1], b),
		dot(a.rows[2], b),
		dot(a.rows[3], b)
	);
}

struct Chunk{
	Point points[POINTS_PER_CHUNK];
	int size;
	int padding_0;
	Chunk* next;
};

struct OccupancyGrid{
	// gridsize^3 occupancy grid; 1 bit per voxel
	uint32_t values[GRID_NUM_CELLS / 32u];
};

struct Node{
	Node* children[8];
	uint32_t counter = 0;
	// uint32_t counters[8] = {0, 0, 0, 0, 0, 0, 0, 0};

	uint32_t numPoints = 0;
	uint32_t level = 0;
	uint32_t X = 0;
	uint32_t Y = 0;
	uint32_t Z = 0;
	uint32_t countIteration = 0;
	uint32_t countFlag = 0;
	uint8_t name[20] = {'r', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	bool visible = false;
	bool isFiltered = false;
	bool isLeaf = true;
	bool isLarge = false;

	OccupancyGrid* grid = nullptr;
	
	Chunk* points = nullptr;
	Chunk* voxelChunks = nullptr;

	uint32_t numVoxels = 0;
	uint32_t numVoxelsStored = 0;

	// bool spilled(){
	// 	return counter > MAX_POINTS_PER_NODE;
	// }

	bool isLeafFn(){

		if(children[0] != nullptr) return false;
		if(children[1] != nullptr) return false;
		if(children[2] != nullptr) return false;
		if(children[3] != nullptr) return false;
		if(children[4] != nullptr) return false;
		if(children[5] != nullptr) return false;
		if(children[6] != nullptr) return false;
		if(children[7] != nullptr) return false;

		return true;
	}

	uint64_t getID(){
		uint64_t id = 0;

		id = id | ((name[ 0] == 'r' ? 1 : 0));
		id = id | ((name[ 1] - '0') <<  3);
		id = id | ((name[ 2] - '0') <<  6);
		id = id | ((name[ 3] - '0') <<  9);
		id = id | ((name[ 4] - '0') << 12);
		id = id | ((name[ 5] - '0') << 15);
		id = id | ((name[ 6] - '0') << 18);
		id = id | ((name[ 7] - '0') << 21);
		id = id | ((name[ 8] - '0') << 24);
		id = id | ((name[ 9] - '0') << 27);
		id = id | (uint64_t((name[10] - '0')) << 30);
		id = id | (uint64_t((name[11] - '0')) << 33);
		id = id | (uint64_t((name[12] - '0')) << 36);
		id = id | (uint64_t((name[13] - '0')) << 39);
		id = id | (uint64_t((name[14] - '0')) << 42);
		id = id | (uint64_t((name[15] - '0')) << 45);
		id = id | (uint64_t((name[16] - '0')) << 48);
		id = id | (uint64_t((name[17] - '0')) << 51);
		id = id | (uint64_t((name[18] - '0')) << 53);

		return id;
	}

};

// Guards kernels/remolod/remolod_layout.h, which the HOST uses to size the node pool and the
// accumulator side array, because this header is not host-compilable (Node's methods call
// dot() from helper_math.h). Drift becomes a compile error in the kernel -- which
// --check-kernels catches without a GPU -- rather than a silently mis-sized allocation.
#ifdef __CUDACC_RTC__
#include "remolod_layout.h"
static_assert(sizeof(Node) == remo::remolod::kNodeBytes,
              "update kNodeBytes in kernels/remolod/remolod_layout.h");
static_assert(BATCH_STREAM_SIZE == remo::remolod::kBatchStreamSize,
              "update kBatchStreamSize in kernels/remolod/remolod_layout.h");
static_assert(MAX_NODES_CAPACITY == remo::remolod::kMaxNodes,
              "update kMaxNodes in kernels/remolod/remolod_layout.h");
#endif
