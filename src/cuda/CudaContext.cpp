#include "remo/CudaContext.h"

#include <algorithm>

#include "remo/CudaCheck.h"

namespace remo {

CudaContext::CudaContext() {
	REMO_CU_FATAL(cuInit(0));
	REMO_CU_FATAL(cuDeviceGet(&m_device, 0));

	char name[256] = {};
	if (cuDeviceGetName(name, sizeof(name), m_device) == CUDA_SUCCESS) {
		m_deviceName = name;
	}

	cuDeviceGetAttribute(&m_numSMs, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
	                     m_device);
	cuDeviceGetAttribute(&m_ccMajor,
	                     CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, m_device);
	cuDeviceGetAttribute(&m_ccMinor,
	                     CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, m_device);

#if defined(CUDA_VERSION) && CUDA_VERSION >= 13000
	// CUDA 13: cuCtxCreate is the _v4 variant, taking CUctxCreateParams*.
	REMO_CU_FATAL(cuCtxCreate(&m_context, nullptr, 0, m_device));
#else
	REMO_CU_FATAL(cuCtxCreate(&m_context, 0, m_device));
#endif

	REMO_CU_FATAL(cuStreamCreate(&m_upload, CU_STREAM_NON_BLOCKING));
	REMO_CU_FATAL(cuStreamCreate(&m_download, CU_STREAM_NON_BLOCKING));
}

CudaContext::~CudaContext() {
	if (m_upload) cuStreamDestroy(m_upload);
	if (m_download) cuStreamDestroy(m_download);
	if (m_context) cuCtxDestroy(m_context);
}

size_t CudaContext::freeMemory() const {
	size_t free = 0, total = 0;
	if (cuMemGetInfo(&free, &total) != CUDA_SUCCESS) return 0;
	return free;
}

size_t CudaContext::totalMemory() const {
	size_t free = 0, total = 0;
	if (cuMemGetInfo(&free, &total) != CUDA_SUCCESS) return 0;
	return total;
}

int CudaContext::gridForKernel(CUfunction kernel, int blockSize,
                               int smFactor) const {
	int maxBlocksPerSM = 0;
	const CUresult r = cuOccupancyMaxActiveBlocksPerMultiprocessor(
		&maxBlocksPerSM, kernel, blockSize, 0);
	if (r != CUDA_SUCCESS || maxBlocksPerSM < 1) {
		REMO_CU(r);
		maxBlocksPerSM = 1;
	}

	const int residentMax = maxBlocksPerSM * m_numSMs;
	if (smFactor <= 0) return residentMax;

	// Requested blocks-per-SM, clamped to what can actually be resident. Clamping
	// rather than trusting the request is deliberate: a cooperative launch that
	// asks for more than fits does not degrade, it fails.
	const int requested = smFactor * m_numSMs;
	return std::min(requested, residentMax);
}

}  // namespace remo
