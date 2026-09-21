#include "shared/remo_pipeline.cuh"

using namespace remo;

constexpr uint32_t FLAT_SLICE = 65536;

constexpr uint32_t FLAT_DRAWLIST_CAPACITY = 32768;

extern "C" __global__ void kernel_render(RenderArgs args, Point* points,
                                         uint64_t numPoints,
                                         DeviceDiagnostics* diag) {
	auto grid = cg::this_grid();

	const SharedUniforms& u = args.uniforms;

	RemoAllocator alloc(args.scratch, args.scratchCapacity, diag);
	const uint64_t numPixels =
		static_cast<uint64_t>(u.width) * static_cast<uint64_t>(u.height);
	uint64_t* framebuffer = alloc.alloc<uint64_t*>(8ull * numPixels);
	DrawList drawList = remoAllocDrawList(alloc, FLAT_DRAWLIST_CAPACITY);
	uint64_t* sampleCount = alloc.alloc<uint64_t*>(8);

	if (framebuffer == nullptr || drawList.items == nullptr) {
		return;
	}

	remoClearFramebuffer(framebuffer, u);
	remoResetDrawList(drawList);
	if (grid.thread_rank() == 0) *sampleCount = 0ull;
	grid.sync();

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
		item.level = 0u;
		item.nodeKey = static_cast<uint32_t>(sliceIndex);

		if (remoDrawListAppend(drawList, item)) {
			atomicAdd(reinterpret_cast<unsigned long long*>(sampleCount),
			          static_cast<unsigned long long>(item.points.count) +
			              static_cast<unsigned long long>(item.voxels.count));
		}
	});
	grid.sync();

	remoRasterizeDrawList<RemoContiguousWalker>(drawList, framebuffer, u);
	grid.sync();

	remoApplyEDL(framebuffer, u);
	grid.sync();

	remoDrawListWireframe(drawList, framebuffer, u);
	grid.sync();

	remoResolve(framebuffer, u, static_cast<cudaSurfaceObject_t>(args.surface));

	if (grid.thread_rank() == 0 && diag != nullptr) {
		diag->drawListOverflow = *drawList.overflowed;
		diag->drawItems = *drawList.numItems;
		diag->drawSamples = *sampleCount;
	}
}
