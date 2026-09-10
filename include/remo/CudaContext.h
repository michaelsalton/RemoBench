// CUDA driver-API context, device properties, and the streams everything shares.
//
// Replaces SimLOD's initCuda() (main_progressive_octree.cpp:267-281) plus the pile
// of file-scope globals around it. Two things here are not cosmetic:
//
//   numSMs / maxBlocksPerSM: every kernel in this project is a COOPERATIVE launch
//       using cg::this_grid().sync(), which means the whole grid must be resident
//       on the device simultaneously. Exceed that and cuLaunchCooperativeKernel
//       fails outright with a non-obvious error. Both research repos partly
//       hardcode this -- SimLOD uses `1 * numSMs` for its update kernel, CudaLOD
//       uses a literal 80 for render, which on an 84-SM card is silently
//       *under*-subscribed. gridForKernel() resolves it from occupancy instead.
//
//   CUDA 13 changed cuCtxCreate's signature (the _v4 variant takes an extra
//       CUctxCreateParams*). Guarded below so CUDA 12 still builds -- the same fix
//       both submodule patches need.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include <cuda.h>

namespace remo {

class CudaContext {
public:
	// Creates the primary context on device 0. Fatal on failure: without a CUDA
	// context there is nothing this program can do.
	CudaContext();
	~CudaContext();

	CudaContext(const CudaContext&) = delete;
	CudaContext& operator=(const CudaContext&) = delete;

	CUcontext handle() const { return m_context; }
	CUdevice device() const { return m_device; }

	const std::string& deviceName() const { return m_deviceName; }
	int numSMs() const { return m_numSMs; }
	int ccMajor() const { return m_ccMajor; }
	int ccMinor() const { return m_ccMinor; }

	// Non-blocking streams for overlapping host<->device transfer with compute.
	// Kernels launch on the null stream, matching upstream's ordering assumptions.
	CUstream uploadStream() const { return m_upload; }
	CUstream downloadStream() const { return m_download; }

	size_t freeMemory() const;
	size_t totalMemory() const;

	// Largest cooperative grid that will fit for this kernel, in blocks:
	//   min(maxActiveBlocksPerSM * numSMs, numSMs * smFactor) when smFactor > 0,
	//   else maxActiveBlocksPerSM * numSMs.
	// Pass smFactor to express "exactly N blocks per SM", which some kernels
	// genuinely require -- CudaLOD's kernel3 allocates a per-block sampling grid
	// from global memory, so running more blocks than SMs would overrun it. That is
	// load-bearing, not an oversight.
	int gridForKernel(CUfunction kernel, int blockSize, int smFactor = 0) const;

private:
	CUdevice m_device = 0;
	CUcontext m_context = nullptr;
	CUstream m_upload = nullptr;
	CUstream m_download = nullptr;

	std::string m_deviceName;
	int m_numSMs = 0;
	int m_ccMajor = 0;
	int m_ccMinor = 0;
};

}  // namespace remo
