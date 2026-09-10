// RemoLOD's accumulator pass: per-node running sums, so the Analysis kernel has something to
// roll up. RemoBench's own file -- nothing here is derived from SimLOD.
//
// It computes nothing, decides nothing, and MUTATES NO TREE STATE. It writes only the side
// array in remo/RemoAccum.h. That is what makes "--dump-frame must stay byte-identical with
// the pass on and off" the acceptance test.
//
// WHY THIS IS A SEPARATE KERNEL AND NOT A HOOK INSIDE THE OCTREE KERNEL
//
// The first version of this was a subpass spliced into SimLOD's addBatch(), which meant
// editing vendored device code. It added two kernel_construct parameters without the matching
// host arguments, and every launch then failed CUDA_ERROR_INVALID_VALUE -- the SimLOD pipeline
// silently built no tree at all for a whole commit, while make succeeded, --check-kernels
// passed and --dump-frame still exited 0. See bench/check_vendored.sh.
//
// Pulling it out to its own launch turned out to be better on the merits, not merely tidier:
//
//   - No octree descent. The hook re-derived each point's leaf by walking the tree from the
//     root, 20 levels deep, per point per batch -- purely to rediscover where insertPoints
//     had already put it. This pass iterates leaves and reads the points they already hold.
//   - One atomic set per leaf per launch instead of one per point. The hook needed warp
//     aggregation (labeled_partition over coalesced_threads) to make 36M atomic bursts
//     survivable, and the aggregation only paid off when neighbouring threads shared a leaf --
//     which needs Morton-ordered input that no reader here produces. A block owns a whole
//     leaf, so the reduction is a plain block reduction and the write is uncontended.
//   - It is separately measurable. The hook's cost was buried inside simlod.construct; this
//     has its own GpuScope, which the evaluation needs because refinement budget is meant to
//     be the single independent variable.
//   - It is idempotent and interruptible. See the watermark note in remo/RemoAccum.h.
//
// ONE BLOCK PER DIRTY LEAF, blocks claiming leaves by striding over blockIdx.x. Leaf point
// counts vary by orders of magnitude (1 .. MAX_POINTS_PER_NODE), so a static partition would
// leave most blocks idle. This is the same striding the shared rasteriser uses over DrawItems,
// and for the same reason.
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

// ---------------------------------------------------------------------------
// Morton
// ---------------------------------------------------------------------------

// Spread the low 20 bits of v out with two zero bits between each, so three of these can be
// interleaved into a 60-bit code.
uint64_t remoMortonSpread(uint32_t v) {
	uint64_t x = uint64_t(v) & 0xFFFFFull;
	x = (x | (x << 32)) & 0x001F00000000FFFFull;
	x = (x | (x << 16)) & 0x001F0000FF0000FFull;
	x = (x | (x <<  8)) & 0x100F00F00F00F00Full;
	x = (x | (x <<  4)) & 0x10C30C30C30C30C3ull;
	x = (x | (x <<  2)) & 0x1249249249249249ull;
	return x;
}

// X MOST SIGNIFICANT, deliberately: it has to match
//   childIndex = (child_X << 2) | (child_Y << 1) | child_Z
// in the octree's own traversals. Interleaved the other way round, Morton order is not the
// octree's child order and the Analysis kernel's closure test would be comparing incomparable
// things.
//
// The watermark this feeds is well-defined and cheap, but it is NOT YET A CLOSURE ORACLE:
// nothing in RemoBench sorts points into Morton order (every reader hands them over in file
// order, which for .las/.laz is acquisition order along flight lines). Building closure on it
// needs either an ordering stage in the loader or a different criterion -- an Analysis
// decision. This layer only guarantees the watermark is maintained and correctly oriented.
uint64_t remoMortonCode(uint32_t X, uint32_t Y, uint32_t Z) {
	return (remoMortonSpread(X) << 2) | (remoMortonSpread(Y) << 1) | remoMortonSpread(Z);
}

// ---------------------------------------------------------------------------
// Clear
// ---------------------------------------------------------------------------

void remoAccumClear(NodeAccum* a) {
	for (int i = 0; i < remo::kNumGeomSums; i++) a->g[i] = 0.0;
	for (int i = 0; i < remo::kNumColorSums; i++) a->c[i] = 0.0f;
	a->count = 0;
	a->lastTouchedBatch = 0;
	a->state = remo::kAccumOpen;
	a->score = 0.0f;
}

// ---------------------------------------------------------------------------
// The pass
// ---------------------------------------------------------------------------
//
// AccumArgs is one struct shared with the host (remo/RemoAccum.h) rather than a parameter
// list, because a drifting parameter list is exactly the failure this file exists to undo.
extern "C" __global__
void kernel_accumulate(AccumArgs args) {
	auto block = cg::this_thread_block();

	Node* nodes = (Node*)args.nodes;
	NodeAccum* accums = (NodeAccum*)args.accums;
	AccumGlobals* globals = (AccumGlobals*)args.globals;

	if (nodes == nullptr || accums == nullptr || globals == nullptr) return;

	// The node pool is a bump index with NO device-side capacity check -- the host clamp in
	// RemolodPipeline::readStats is the only place exhaustion is noticed -- so bound the walk
	// rather than reading past the side array.
	const uint32_t numNodes = min(args.numNodes, MAX_NODES_CAPACITY);

	// Per-block staging for the reduction. float/double sums are reduced through shared
	// memory rather than __shfl, because the block is 256 threads and the sums are arrays:
	// a shared-memory tree reduction is one loop over the array, where shuffles would be
	// fifteen separate butterflies.
	__shared__ double sg[remo::kNumGeomSums][ACCUM_BLOCK_SIZE / 32];
	__shared__ float  sc[remo::kNumColorSums][ACCUM_BLOCK_SIZE / 32];
	__shared__ unsigned long long sMorton[ACCUM_BLOCK_SIZE / 32];

	const int lane = threadIdx.x & 31;
	const int warp = threadIdx.x >> 5;
	const int numWarps = ACCUM_BLOCK_SIZE / 32;

	// The shared staging above is sized for ACCUM_BLOCK_SIZE, so a launch with a larger block
	// would write past it. The host launches exactly ACCUM_BLOCK_SIZE threads
	// (RemolodPipeline::kAccumBlockSize, kept equal to this constant), so `writesShared` is
	// always true in practice -- it is here so that if that ever stops being true the failure
	// is dropped work rather than corrupted shared memory.
	//
	// Deliberately NOT an early return: there are block.sync() calls below, and a block whose
	// threads have exited cannot reach them.
	const bool writesShared = (warp < numWarps);

	for (uint32_t nodeIndex = blockIdx.x; nodeIndex < numNodes; nodeIndex += gridDim.x) {
		Node* node = &nodes[nodeIndex];
		NodeAccum* a = &accums[nodeIndex];

		// --- inner nodes hold no sums -------------------------------------------
		//
		// A leaf that spilled is now an inner node and its sums are meaningless: inner-node
		// statistics are derived by roll-up later, not stored. Its points were moved to the
		// spill buffer and re-inserted into the new leaves, which walk from count == 0 below,
		// so there is no double-counting to reason about.
		//
		// Done here rather than at the split site for three reasons. It keeps the octree
		// kernel untouched. It is idempotent and self-healing, so a clear that is somehow
		// missed is corrected on the next launch instead of silently poisoning a roll-up. And
		// it makes the invariant observable: innerWithSums, counted below, must be zero.
		if (!node->isLeafFn()) {
			if (block.thread_rank() == 0 && a->count != 0) remoAccumClear(a);
			block.sync();
			continue;
		}

		const uint32_t from = a->count;
		const uint32_t to = node->numPoints;
		if (to <= from) continue;   // nothing new since the last launch

		// --- walk to the chunk holding `from` -----------------------------------
		//
		// Points live in a linked list of POINTS_PER_CHUNK-point chunks, densely at
		// [0, numPoints). Every thread walks the same list to the same starting chunk, so the
		// traversal is block-uniform and costs one pass over at most numPoints/1000 links.
		const uint32_t firstChunk = from / POINTS_PER_CHUNK;

		double lg[remo::kNumGeomSums];
		float  lc[remo::kNumColorSums];
		for (int i = 0; i < remo::kNumGeomSums; i++) lg[i] = 0.0;
		for (int i = 0; i < remo::kNumColorSums; i++) lc[i] = 0.0f;
		unsigned long long lMorton = 0ull;

		// Node-local normalisation. THE SINGLE MOST IMPORTANT DETAIL IN THE PASS: sum raw
		// world coordinates -- UTM eastings in the hundreds of thousands -- and Sxx loses
		// precisely the small differences the smallest eigenvalue is made of.
		//
		// In double because the left term is a cancellation at depth: at level 15 it is
		// ~32768 and the answer is in [0,1], which float32 resolves to about 2e-3 of a node.
		const double invOctreeSize = 1.0 / double(args.octreeSize);
		const double levelScale = double(1ull << node->level);
		const double fGridSize = double(1u << MAX_DEPTH);

		// Each thread strides over the new points. The chunk pointer is advanced per thread
		// from firstChunk, which is why the loop steps by blockDim.x rather than partitioning:
		// consecutive threads then read consecutive points within a chunk.
		Chunk* startChunk = node->points;
		for (uint32_t i = 0; i < firstChunk && startChunk != nullptr; i++) {
			startChunk = startChunk->next;
		}

		uint32_t chunkIndex = firstChunk;
		Chunk* chunk = startChunk;

		for (uint32_t base = from; base < to; base += blockDim.x) {
			const uint32_t pointIndex = base + threadIdx.x;

			// Advance the block-uniform chunk cursor to cover [base, base + blockDim.x).
			// blockDim.x (256) is smaller than POINTS_PER_CHUNK (1000), so this window spans
			// at most two chunks and the inner walk is at most one link per iteration.
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

					// Morton code at MAX_DEPTH, from the same normalised position. Computed
					// before the node-local shift, because the watermark is a whole-cloud
					// quantity.
					//
					// Clamped to 20 bits: a point exactly on the box maximum normalises to
					// 1.0 and scales to 2^20, which remoMortonSpread's 0xFFFFF mask would
					// wrap to 0 -- turning the largest possible code into the smallest.
					//
					// This deliberately does NOT have to match insertPoints' float
					// arithmetic. The hook it replaces did, because it re-descended the tree
					// and a boundary point had to land in the same cell; this pass reads the
					// leaf the point is already in, so X/Y/Z feed the watermark only.
					const uint32_t maxCoord = (1u << MAX_DEPTH) - 1u;
					const uint32_t X = min(uint32_t(fGridSize * lx), maxCoord);
					const uint32_t Y = min(uint32_t(fGridSize * ly), maxCoord);
					const uint32_t Z = min(uint32_t(fGridSize * lz), maxCoord);
					const unsigned long long m = remoMortonCode(X, Y, Z);
					if (m > lMorton) lMorton = m;

					lx = lx * levelScale - double(node->X);
					ly = ly * levelScale - double(node->Y);
					lz = lz * levelScale - double(node->Z);

					// The float->uint32 truncation that picked the cell during insertion and
					// this normalisation are not the same arithmetic, so a point exactly on a
					// cell boundary can land marginally outside. One instruction each, and it
					// keeps a garbage term out of a sum that is never recomputed.
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

					// RGBA8, alpha in the high byte. Channels in [0,1] so the colour sums are
					// scale-free in the same way the geometry ones are.
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

			// Advance the shared cursor for the next window. Uniform across the block.
			const uint32_t nextBase = base + blockDim.x;
			const uint32_t nextChunk = nextBase / POINTS_PER_CHUNK;
			while (chunkIndex < nextChunk && chunk != nullptr) {
				chunk = chunk->next;
				chunkIndex++;
			}
		}

		// --- block reduction ----------------------------------------------------
		//
		// Warp-level first (no shared traffic, no barrier), then one value per warp through
		// shared memory. The partial sums INSIDE a warp reduction can stay float for colour:
		// at most 32 terms, so it contributes ~1e-5 absolute against the 0.05 that matters.
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
			// Only this block touches this node, so these are plain read-modify-writes rather
			// than atomics. That is the whole payoff of one-block-per-leaf.
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

// The two invariants, recomputed from scratch every launch.
//
// They are a separate kernel from the pass above because they must run AFTER every block has
// finished folding, and this pass is a plain (non-cooperative) launch -- there is no
// grid.sync() available to separate the two phases inside one kernel. Two launches is the
// honest way to express that ordering, and the check is cheap: one pass over at most
// MAX_NODES_CAPACITY nodes.
//
//   sumLeafCounts  must equal Stats::numPoints exactly. numPoints is the sum of node->numPoints
//                  over leaves, so this checks the accumulation against the insertion it
//                  shadows, per point, over the whole cloud. Anything that misses a chunk,
//                  double-counts, or clears the wrong entry shows up here.
//   innerWithSums  must be zero. Checks the clear-on-split path above.
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
