#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include <cuda.h>

namespace remo {

class CudaContext {
public:
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

	CUstream uploadStream() const { return m_upload; }
	CUstream downloadStream() const { return m_download; }

	size_t freeMemory() const;
	size_t totalMemory() const;

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

}
