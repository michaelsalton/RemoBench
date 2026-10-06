#pragma once

#include <cuda.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "remo/CudaCheck.h"

namespace remo {

// The deepest level present in a SimLOD-shaped node pool, read from the host.
//
// No device struct carries depth, and SimLOD's Stats is vendored, so this reads Node::level
// straight out of the pool instead. It costs 4 bytes per node, and each node is read ONCE:
// the pool is a bump index (`atomicAdd(&stats->numNodes, 8)`), a node's level is written in
// the same launch that allocates it and never changes, and nothing is freed. So after each
// construct launch only [scanned, numNodes) is new. A strided cuMemcpy2D pulls just the
// level field rather than whole 152-byte nodes.
//
// reset() must accompany every tree reset -- a rebuilt tree reuses the pool from index 0.
class OctreeDepth {
public:
	void reset() {
		m_scanned = 0;
		m_maxLevel = 0;
	}

	// numNodes is the device count, already clamped to the pool by the caller. Call only
	// after the construct launch has completed.
	void update(CUdeviceptr nodes, uint32_t numNodes, uint32_t nodeBytes,
	            uint32_t levelOffset) {
		if (!nodes || numNodes <= m_scanned) return;
		const uint32_t count = numNodes - m_scanned;
		m_levels.resize(count);

		CUDA_MEMCPY2D copy = {};
		copy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
		copy.srcDevice = nodes + static_cast<CUdeviceptr>(m_scanned) * nodeBytes + levelOffset;
		copy.srcPitch = nodeBytes;
		copy.dstMemoryType = CU_MEMORYTYPE_HOST;
		copy.dstHost = m_levels.data();
		copy.dstPitch = sizeof(uint32_t);
		copy.WidthInBytes = sizeof(uint32_t);
		copy.Height = count;
		if (REMO_CU(cuMemcpy2D(&copy)) != CUDA_SUCCESS) return;

		m_maxLevel = std::max(m_maxLevel, *std::max_element(m_levels.begin(), m_levels.end()));
		m_scanned = numNodes;
	}

	// Root is level 0, so a lone root has depth 0.
	uint32_t maxLevel() const { return m_maxLevel; }

private:
	uint32_t m_scanned = 0;
	uint32_t m_maxLevel = 0;
	std::vector<uint32_t> m_levels;
};

}
