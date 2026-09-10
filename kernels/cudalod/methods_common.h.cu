// Vendored from CudaLOD: modules/simlod/sampling_cuda_nonprogressive/methods_common.h.cu
// Upstream: https://github.com/m-schuetz/CudaLOD @ cfaa4dc90c088b6e8aa8723c22e91b64eacac347
// Copyright 2022 Markus Schuetz -- MIT (see THIRD_PARTY.md)
// Copied from the PATCHED submodule tree: patches/cudalod-linux-port.patch
// fixes `typedef char int8_t` -> `signed char` here, which otherwise collides
// with cuda/std/cstdint under NVRTC.
//
// The LOD Node, plus MAX_NODES / MAX_POINTS_PER_NODE / VOXEL_GRID_SIZE.
//
// Leaves hold a contiguous slice of one globally counting-sorted Point array
// (pointOffset + numPoints), and inner nodes hold voxels as a plain Point*. That is
// why the shared rasteriser walks samples through a template Walker -- SimLOD uses a
// linked list of chunks instead. See kernels/shared/remo_draw.cuh.

#include "lib.h.cu"

constexpr bool PRINT_STATS = false;

static constexpr int MAX_NODES           = 200'000;
static constexpr int MAX_POINTS_PER_NODE = 50'000;
static constexpr int VOXEL_GRID_SIZE     = 128;
static constexpr int MAX_DEPTH           = 20;

struct Node{
	int pointOffset;
	int numPoints;
	Point* points;
	
	uint32_t dbg = 0;
	int numAdded;
	int level;
	int voxelIndex;
	vec3 min;
	vec3 max;
	float cubeSize;
	Node* children[8] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};

	int numVoxels = 0;
	Point* voxels = nullptr;

	bool visible = true;

	bool isLeaf(){
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
};

typedef Node* NodePtr;
typedef Node** NodePtrGrid;