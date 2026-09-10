// The `flat` pipeline: rasterise every point, no LOD.
//
// Not a placeholder. It is the CONTROL CONDITION for every later comparison:
//
//   - It is ground truth for image quality. A LOD pipeline's output is supposed to
//     approximate this; the difference against it is the only honest measure of
//     sampling quality, as opposed to comparing two approximations to each other and
//     picking the prettier one.
//   - It is the upper bound on samples drawn, so "pipeline A drew 4.1M samples for the
//     same visual result" has a denominator.
//   - It exercises the whole shared path -- draw list, projection, the packed uint64
//     atomicMin framebuffer, EDL, the surface resolve -- before any octree exists to
//     confuse a bug with.
//
// It deliberately goes through the SAME DrawList seam that SimLOD and CudaLOD will use,
// rather than a private fast path. If flat had its own shortcut, the seam would be
// untested until the first real port depended on it, and a bug there would look like a
// bug in the port.
//
// Since flat has no tree, it fabricates one draw item per fixed-size slice of the point
// array. That is not busywork either: a single item would leave one block drawing all
// 36M points, so slicing is what distributes the work -- and it gives colour-by-node
// something to show, which makes the slicing visible.

#include "shared/remo_pipeline.cuh"

using namespace remo;

// Samples per synthetic "node". 64k keeps the draw list small (a 350M cloud needs
// ~5300 items) while still giving every SM many items to chew on.
constexpr uint32_t FLAT_SLICE = 65536;

// Enough for ~2.1 billion points at the slice size above.
constexpr uint32_t FLAT_DRAWLIST_CAPACITY = 32768;

extern "C" __global__ void kernel_render(RenderArgs args, Point* points,
                                         uint64_t numPoints,
                                         DeviceDiagnostics* diag) {
	auto grid = cg::this_grid();

	const SharedUniforms& u = args.uniforms;

	// Every thread walks the identical allocation sequence -- see the banner in
	// remo_alloc.cuh. Do not move any alloc() behind a condition.
	RemoAllocator alloc(args.scratch, args.scratchCapacity, diag);
	const uint64_t numPixels =
		static_cast<uint64_t>(u.width) * static_cast<uint64_t>(u.height);
	uint64_t* framebuffer = alloc.alloc<uint64_t*>(8ull * numPixels);
	DrawList drawList = remoAllocDrawList(alloc, FLAT_DRAWLIST_CAPACITY);
	// Samples referenced by the emitted items, for the stats panel.
	uint64_t* sampleCount = alloc.alloc<uint64_t*>(8);

	if (framebuffer == nullptr || drawList.items == nullptr) {
		// Scratch was too small. diag->allocOverflow is already set and the host reads
		// it back, so this reports itself instead of corrupting memory.
		return;
	}

	remoClearFramebuffer(framebuffer, u);
	remoResetDrawList(drawList);
	if (grid.thread_rank() == 0) *sampleCount = 0ull;
	grid.sync();

	// "Selection": no LOD, so every slice is visible. This is where a real pipeline
	// would run its own frustum test and LOD cut.
	const uint32_t numSlices =
		static_cast<uint32_t>((numPoints + FLAT_SLICE - 1) / FLAT_SLICE);

	processRangeStrided(numSlices, [&](uint64_t sliceIndex) {
		const uint64_t first = sliceIndex * FLAT_SLICE;
		const uint64_t remaining = numPoints - first;
		const uint32_t count = remaining < FLAT_SLICE
		                           ? static_cast<uint32_t>(remaining)
		                           : FLAT_SLICE;

		DrawItem item = {};
		item.points.head = points + first;
		item.points.count = count;
		item.voxels.head = nullptr;
		item.voxels.count = 0u;
		item.nodeMin = u.boxMin;
		item.nodeSize = 0.0f;
		item.level = 0u;  // flat is level 0 by definition
		item.nodeKey = static_cast<uint32_t>(sliceIndex);

		if (remoDrawListAppend(drawList, item)) {
			atomicAdd(reinterpret_cast<unsigned long long*>(sampleCount),
			          static_cast<unsigned long long>(item.points.count) +
			              static_cast<unsigned long long>(item.voxels.count));
		}
	});
	grid.sync();

	// Points and voxels share one storage layout here, so one walker covers both.
	remoRasterizeDrawList<RemoContiguousWalker>(drawList, framebuffer, u);
	grid.sync();

	remoApplyEDL(framebuffer, u);
	grid.sync();

	// Inert here by design: flat's items are point-array slices, not nodes, so they
	// carry no box. It goes through the same seam anyway rather than being omitted --
	// the control condition exercising the shared path is the whole point of `flat`.
	remoDrawListWireframe(drawList, framebuffer, u);
	grid.sync();

	remoResolve(framebuffer, u, static_cast<cudaSurfaceObject_t>(args.surface));

	// Publish what was drawn, so the stats panel is not guessing.
	if (grid.thread_rank() == 0 && diag != nullptr) {
		diag->drawListOverflow = *drawList.overflowed;
		diag->drawItems = *drawList.numItems;
		diag->drawSamples = *sampleCount;
	}
}
