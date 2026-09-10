// RemoBench's own file. NOT vendored -- it carries no upstream banner and
// bench/check_vendored.sh does not police it.
//
// The seam between SimLOD's vendored device code and RemoBench's own kernels. It holds the
// checks that used to be static_asserts inside structures.cuh, which is not ours to edit any
// more.
//
// Include it AFTER structures.cuh, from any RemoBench kernel that works on a SimLOD tree.
// Including it costs nothing at runtime: it defines two aliases and a handful of compile-time
// assertions.
//
// WHY THE ASSERTIONS MATTER MORE THAN THEY LOOK
//
// The host cannot include structures.cuh (its Node methods call dot()), so it sizes the
// 200k-node pool from simlod_layout.h's mirrored kNodeBytes instead. Nothing else connects
// the two numbers. A submodule bump that widens Node -- or changes BATCH_STREAM_SIZE -- would
// otherwise produce a pool that is the wrong size, and the failure mode is an out-of-bounds
// device write, not a diagnostic. These asserts turn that into a compile error in the kernel,
// which --check-kernels catches without a GPU, a display or a point cloud.
#pragma once

#include "simlod_layout.h"

// structures.cuh must already be included; these names come from it.
static_assert(sizeof(Node) == remo::simlod::kNodeBytes,
              "sizeof(Node) changed upstream; update kNodeBytes in simlod_layout.h "
              "(it sizes the host's node pool)");
static_assert(BATCH_STREAM_SIZE == remo::simlod::kBatchStreamSize,
              "BATCH_STREAM_SIZE changed upstream; update kBatchStreamSize in "
              "simlod_layout.h (it sizes the reset kernel's batchSizes buffer and caps "
              "how many points SimlodPipeline will accept)");

// The node-pool bound, under the name RemoBench's kernels use. Upstream has no such
// constant -- it grows the pool with atomicAdd and never checks -- so every RemoBench pass
// over the pool clamps against this instead.
constexpr uint32_t REMO_SIMLOD_MAX_NODES = remo::simlod::kMaxNodes;
