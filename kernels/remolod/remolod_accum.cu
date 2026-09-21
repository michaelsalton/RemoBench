#include <cooperative_groups.h>
#include <curand_kernel.h>

#include "../simlod/utils.h.cu"
#include "builtin_types.h"
#include "../simlod/helper_math.h"
#include "../simlod/HostDeviceInterface.h"

#include "../simlod/math.cuh"
#include "remolod_structures.cuh"

#include "../CudaPrint/CudaPrint.cuh"

#include "remo/RemoAccum.h"

namespace cg = cooperative_groups;

using remo::AccumArgs;
using remo::AccumGlobals;
using remo::NodeAccum;

constexpr int ACCUM_BLOCK_SIZE = 256;

uint64_t remoMortonSpread(uint32_t v) {
	uint64_t x = uint64_t(v) & 0xFFFFFull;
	x = (x | (x << 32)) & 0x001F00000000FFFFull;
	x = (x | (x << 16)) & 0x001F0000FF0000FFull;
	x = (x | (x <<  8)) & 0x100F00F00F00F00Full;
	x = (x | (x <<  4)) & 0x10C30C30C30C30C3ull;
	x = (x | (x <<  2)) & 0x1249249249249249ull;
	return x;
}

uint64_t remoMortonCode(uint32_t X, uint32_t Y, uint32_t Z) {
	return (remoMortonSpread(X) << 2) | (remoMortonSpread(Y) << 1) | remoMortonSpread(Z);
}

void remoAccumClear(NodeAccum* a) {
	for (int i = 0; i < remo::kNumGeomSums; i++) a->g[i] = 0.0;
	for (int i = 0; i < remo::kNumColorSums; i++) a->c[i] = 0.0f;
	a->count = 0;
	a->lastTouchedBatch = 0;
	a->state = remo::kAccumOpen;
	a->score = 0.0f;
}

extern "C" __global__
void kernel_accumulate(AccumArgs args) {
	auto block = cg::this_thread_block();

	Node* nodes = (Node*)args.nodes;
	NodeAccum* accums = (NodeAccum*)args.accums;
	AccumGlobals* globals = (AccumGlobals*)args.globals;

	if (nodes == nullptr || accums == nullptr || globals == nullptr) return;

	const uint32_t numNodes = min(args.numNodes, MAX_NODES_CAPACITY);

	__shared__ double sg[remo::kNumGeomSums][ACCUM_BLOCK_SIZE / 32];
	__shared__ float  sc[remo::kNumColorSums][ACCUM_BLOCK_SIZE / 32];
	__shared__ unsigned long long sMorton[ACCUM_BLOCK_SIZE / 32];

	const int lane = threadIdx.x & 31;
	const int warp = threadIdx.x >> 5;
	const int numWarps = ACCUM_BLOCK_SIZE / 32;

	const bool writesShared = (warp < numWarps);

	for (uint32_t nodeIndex = blockIdx.x; nodeIndex < numNodes; nodeIndex += gridDim.x) {
		Node* node = &nodes[nodeIndex];
		NodeAccum* a = &accums[nodeIndex];

		if (!node->isLeafFn()) {
			if (block.thread_rank() == 0 && a->count != 0) remoAccumClear(a);
			block.sync();
			continue;
		}

		const uint32_t from = a->count;
		const uint32_t to = node->numPoints;
		if (to <= from) continue;

		const uint32_t firstChunk = from / POINTS_PER_CHUNK;

		double lg[remo::kNumGeomSums];
		float  lc[remo::kNumColorSums];
		for (int i = 0; i < remo::kNumGeomSums; i++) lg[i] = 0.0;
		for (int i = 0; i < remo::kNumColorSums; i++) lc[i] = 0.0f;
		unsigned long long lMorton = 0ull;

		const double invOctreeSize = 1.0 / double(args.octreeSize);
		const double levelScale = double(1ull << node->level);
		const double fGridSize = double(1u << MAX_DEPTH);

		Chunk* startChunk = node->points;
		for (uint32_t i = 0; i < firstChunk && startChunk != nullptr; i++) {
			startChunk = startChunk->next;
		}

		uint32_t chunkIndex = firstChunk;
		Chunk* chunk = startChunk;

		for (uint32_t base = from; base < to; base += blockDim.x) {
			const uint32_t pointIndex = base + threadIdx.x;

			if (pointIndex < to && chunk != nullptr) {
				const uint32_t targetChunk = pointIndex / POINTS_PER_CHUNK;
				Chunk* c = chunk;
				for (uint32_t k = chunkIndex; k < targetChunk && c != nullptr; k++) {
					c = c->next;
				}
				if (c != nullptr) {
					const Point point = c->points[pointIndex % POINTS_PER_CHUNK];

					double lx = (double(point.x) - double(args.octreeMinX)) * invOctreeSize;
					double ly = (double(point.y) - double(args.octreeMinY)) * invOctreeSize;
					double lz = (double(point.z) - double(args.octreeMinZ)) * invOctreeSize;

					const uint32_t maxCoord = (1u << MAX_DEPTH) - 1u;
					const uint32_t X = min(uint32_t(fGridSize * lx), maxCoord);
					const uint32_t Y = min(uint32_t(fGridSize * ly), maxCoord);
					const uint32_t Z = min(uint32_t(fGridSize * lz), maxCoord);
					const unsigned long long m = remoMortonCode(X, Y, Z);
					if (m > lMorton) lMorton = m;

					lx = lx * levelScale - double(node->X);
					ly = ly * levelScale - double(node->Y);
					lz = lz * levelScale - double(node->Z);

					lx = fmin(fmax(lx, 0.0), 1.0);
					ly = fmin(fmax(ly, 0.0), 1.0);
					lz = fmin(fmax(lz, 0.0), 1.0);

					lg[remo::kSumX]  += lx;
					lg[remo::kSumY]  += ly;
					lg[remo::kSumZ]  += lz;
					lg[remo::kSumXX] += lx * lx;
					lg[remo::kSumXY] += lx * ly;
					lg[remo::kSumXZ] += lx * lz;
					lg[remo::kSumYY] += ly * ly;
					lg[remo::kSumYZ] += ly * lz;
					lg[remo::kSumZZ] += lz * lz;

					const float cr = float((point.color >>  0) & 0xFFu) * (1.0f / 255.0f);
					const float cgc = float((point.color >>  8) & 0xFFu) * (1.0f / 255.0f);
					const float cb = float((point.color >> 16) & 0xFFu) * (1.0f / 255.0f);

					lc[remo::kSumR]  += cr;
					lc[remo::kSumG]  += cgc;
					lc[remo::kSumB]  += cb;
					lc[remo::kSumRR] += cr * cr;
					lc[remo::kSumGG] += cgc * cgc;
					lc[remo::kSumBB] += cb * cb;
				}
			}

			const uint32_t nextBase = base + blockDim.x;
			const uint32_t nextChunk = nextBase / POINTS_PER_CHUNK;
			while (chunkIndex < nextChunk && chunk != nullptr) {
				chunk = chunk->next;
				chunkIndex++;
			}
		}

		for (int i = 0; i < remo::kNumGeomSums; i++) {
			double v = lg[i];
			for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xFFFFFFFFu, v, off);
			if (lane == 0 && writesShared) sg[i][warp] = v;
		}
		for (int i = 0; i < remo::kNumColorSums; i++) {
			float v = lc[i];
			for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xFFFFFFFFu, v, off);
			if (lane == 0 && writesShared) sc[i][warp] = v;
		}
		{
			unsigned long long v = lMorton;
			for (int off = 16; off > 0; off >>= 1) {
				const unsigned long long o = __shfl_down_sync(0xFFFFFFFFu, v, off);
				if (o > v) v = o;
			}
			if (lane == 0 && writesShared) sMorton[warp] = v;
		}

		block.sync();

		if (block.thread_rank() == 0) {
			for (int i = 0; i < remo::kNumGeomSums; i++) {
				double v = 0.0;
				for (int w = 0; w < numWarps; w++) v += sg[i][w];
				a->g[i] += v;
			}
			for (int i = 0; i < remo::kNumColorSums; i++) {
				float v = 0.0f;
				for (int w = 0; w < numWarps; w++) v += sc[i][w];
				a->c[i] += v;
			}
			unsigned long long m = 0ull;
			for (int w = 0; w < numWarps; w++) if (sMorton[w] > m) m = sMorton[w];

			const uint32_t folded = to - from;
			a->count = to;
			a->lastTouchedBatch = args.batchIndex;

			atomicMax(&globals->mortonWatermark, m);
			atomicAdd(&globals->numAccumulated, (unsigned long long)folded);
			atomicAdd(&globals->pointsFolded, folded);
			atomicAdd(&globals->numLeavesFolded, 1u);
		}

		block.sync();
	}
}

extern "C" __global__
void kernel_accum_verify(AccumArgs args) {
	Node* nodes = (Node*)args.nodes;
	NodeAccum* accums = (NodeAccum*)args.accums;
	AccumGlobals* globals = (AccumGlobals*)args.globals;

	if (nodes == nullptr || accums == nullptr || globals == nullptr) return;

	const uint32_t numNodes = min(args.numNodes, MAX_NODES_CAPACITY);

	const uint32_t stride = blockDim.x * gridDim.x;
	for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < numNodes; i += stride) {
		Node* node = &nodes[i];
		if (node->isLeafFn()) {
			atomicAdd(&globals->sumLeafCounts, (unsigned long long)accums[i].count);
		} else if (accums[i].count != 0) {
			atomicAdd(&globals->innerWithSums, 1u);
		}
	}
}
