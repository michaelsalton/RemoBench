#pragma once

namespace remo {
namespace remolod {

constexpr unsigned int kMaxNodes = 200000u;

constexpr unsigned int kNodeBytes = 152u;

// The device ring depth, in 1M-point slots. Must equal BATCH_STREAM_SIZE in
// remolod_structures.cuh, which static_asserts against this.
//
// This was 8192, and came back DOWN to upstream's 50 -- the opposite direction to the
// usual fork edit, and it REMOVES a divergence rather than adding one.
//
// 8192 existed only to make a non-wrapping resident feed addressable past 50M points:
// kernel_construct reads batch N from slot N % BATCH_STREAM_SIZE, and a flat resident
// cloud puts batch N at slot N, so the two agree only while N < BATCH_STREAM_SIZE.
// Raising the constant raised that ceiling. As a real ring it would be 8192 * 1M * 16 B
// = 131 GB, which is not a buffer anyone allocates; the value never meant a ring depth,
// only an addressing ceiling.
//
// PointSource now wraps, so the ceiling is gone and 50 is correct again. RemoLOD and
// SimLOD are therefore apart by one constant instead of two, and the tree equality
// between them (4,137 nodes / 12,742,751 voxels on morro_bay 36M) becomes a free check
// that the ring is filling the right slots -- a wrong ring shows up as a different tree
// rather than as nothing at all.
constexpr unsigned int kBatchStreamSize = 50u;

}
}
