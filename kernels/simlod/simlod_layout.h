// RemoBench's own file. NOT vendored -- it carries no upstream banner and bench/check_vendored.sh
// does not police it.
//
// The SimLOD device-layout numbers that RemoBench needs but upstream does not define, in a
// header BOTH the host and NVRTC can compile.
//
// Two reasons it exists rather than the values living in structures.cuh, where an earlier
// version of this code put them:
//
//   1. structures.cuh is vendored and must stay byte-identical to the submodule, so it is
//      not ours to add constants to. See bench/check_vendored.sh for what went wrong when
//      that rule was broken.
//   2. structures.cuh is not host-compilable anyway -- its Node methods call dot() from
//      helper_math.h, so it only builds inside a CUDA translation unit. The host cannot
//      read sizeof(Node) from the real struct and has to be told.
//
// simlod_bridge.cuh static_asserts every number here against the real vendored types, so
// drift is a compile error in the kernel rather than a silently mis-sized allocation.
//
// NO <cstdint>, AND NO IMPORTED TYPE NAMES. NVRTC has no libstdc++, and CCCL's
// cuda::std::uint32_t is a different type from the `typedef unsigned int uint32_t` that
// SimLOD's utils.h.cu puts in scope -- a conflicting typedef, which is the same trap
// patches/cudalod-linux-port.patch exists to fix. Spelling the builtin types out sidesteps
// the question on both sides.
#pragma once

namespace remo {
namespace simlod {

// Node pool capacity, and the bound every pass clamps against.
//
// Upstream grows its flat Node pool with `atomicAdd(&stats->numNodes, 8)` and NO capacity
// check anywhere, so an eagerly splitting tree walks off the end of the allocation. This is
// the size the host allocates the pool to; the host clamp in SimlodPipeline::readStats is
// still the only place exhaustion is actually reported.
constexpr unsigned int kMaxNodes = 200000u;

// sizeof(Node). Verified against the real struct by simlod_bridge.cuh.
constexpr unsigned int kNodeBytes = 152u;

// Must equal BATCH_STREAM_SIZE in structures.cuh (asserted in simlod_bridge.cuh).
//
// UPSTREAM'S VALUE, and the reason the SimLOD pipeline has a point-count ceiling.
// kernel_construct addresses batch N at points + (N % BATCH_STREAM_SIZE) * MAX_BATCH_SIZE --
// a ring of this many 1M-point slots. RemoBench currently feeds it the whole cloud already
// resident in device memory, where batch N lives at points + N * MAX_BATCH_SIZE with NO
// wrapping. Those two agree only while N < BATCH_STREAM_SIZE, so past 50 batches the kernel
// silently re-reads slot 0 and builds a tree from the wrong points.
//
// This used to be raised to 8192 by editing structures.cuh. That worked, but it made the
// comparison baseline stop being SimLOD. SimlodPipeline now refuses a cloud it cannot
// address instead, and RemoLOD -- which owns its fork and may change what it likes -- raises
// it in kernels/remolod/remolod_structures.cuh. The real fix for SimLOD is a wrapping
// streaming ring in PointSource, which is the loader work the README already lists.
//
// The host also needs it because reset.cu unconditionally zeroes batchSizes[0 ..
// BATCH_STREAM_SIZE-1], so any buffer handed to it must be at least that large. Getting that
// wrong is an out-of-bounds device write presenting as CUDA_ERROR_LAUNCH_FAILED with no
// indication of where.
constexpr unsigned int kBatchStreamSize = 50u;

// The largest cloud SimLOD's ring addressing can describe without wrapping onto itself.
//
// MAX_BATCH_SIZE is a function-local constexpr inside kernel_construct
// (progressive_octree_voxels.cu:886), so no other translation unit can assert against it --
// it is mirrored here and must be kept equal to PointSource's slotCapacity by hand. The two
// are checked against each other at runtime in SimlodPipeline::allocate.
constexpr unsigned long long kMaxBatchSize = 1000000ull;
constexpr unsigned long long kMaxAddressablePoints =
	kMaxBatchSize * static_cast<unsigned long long>(kBatchStreamSize);

}  // namespace simlod
}  // namespace remo
