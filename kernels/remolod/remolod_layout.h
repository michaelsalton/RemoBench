// RemoLOD's device-layout numbers, in a header BOTH the host and NVRTC can compile.
//
// The host cannot include remolod_structures.cuh -- Node's methods call dot() from
// helper_math.h, so it only builds inside a CUDA translation unit. These are the numbers the
// host needs to size allocations, mirrored here and static_asserted against the real types at
// the bottom of remolod_structures.cuh. Drift is a compile error in the kernel, which
// --check-kernels catches without a GPU, a display or a point cloud.
//
// NO <cstdint>, AND NO IMPORTED TYPE NAMES. NVRTC has no libstdc++, and CCCL's
// cuda::std::uint32_t is a different type from the `typedef unsigned int uint32_t` that
// SimLOD's utils.h.cu puts in scope -- a conflicting typedef, the same trap
// patches/cudalod-linux-port.patch exists to fix. Spelling the builtin types out sidesteps
// the question on both sides.
#pragma once

namespace remo {
namespace remolod {

// Node pool capacity, and the bound every pass clamps against. Matches MAX_NODES_CAPACITY.
constexpr unsigned int kMaxNodes = 200000u;

// sizeof(Node), for sizing the pool host-side.
constexpr unsigned int kNodeBytes = 152u;

// Matches BATCH_STREAM_SIZE. Raised from SimLOD's 50 -- see the comment at its definition in
// remolod_structures.cuh. The host needs it because remolod_reset.cu unconditionally zeroes
// batchSizes[0 .. BATCH_STREAM_SIZE-1], so any buffer handed to reset must be at least that
// large; getting it wrong is an out-of-bounds device write presenting as
// CUDA_ERROR_LAUNCH_FAILED with no indication of where.
constexpr unsigned int kBatchStreamSize = 8192u;

}  // namespace remolod
}  // namespace remo
