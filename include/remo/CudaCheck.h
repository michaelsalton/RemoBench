#pragma once

#include <cstdio>
#include <cstdlib>

#include <cuda.h>

namespace remo {

inline const char* cuErrorName(CUresult result) {
	const char* name = nullptr;
	if (cuGetErrorName(result, &name) == CUDA_SUCCESS && name) return name;
	static thread_local char buf[32];
	snprintf(buf, sizeof(buf), "CUresult(%d)", static_cast<int>(result));
	return buf;
}

inline const char* cuErrorString(CUresult result) {
	const char* str = nullptr;
	if (cuGetErrorString(result, &str) == CUDA_SUCCESS && str) return str;
	return "(no description)";
}

inline CUresult cuCheckImpl(CUresult result, const char* expr, const char* file,
                            int line) {
	if (result != CUDA_SUCCESS) {
		fprintf(stderr, "remobench: CUDA error %s at %s:%d\n  %s\n  %s\n",
		        cuErrorName(result), file, line, expr, cuErrorString(result));
	}
	return result;
}

[[noreturn]] inline void cuFatal(CUresult result, const char* expr,
                                 const char* file, int line) {
	fprintf(stderr, "remobench: fatal CUDA error %s at %s:%d\n  %s\n  %s\n",
	        cuErrorName(result), file, line, expr, cuErrorString(result));
	std::abort();
}

inline bool isStickyError(CUresult result) {
	switch (result) {
		case CUDA_ERROR_ILLEGAL_ADDRESS:
		case CUDA_ERROR_HARDWARE_STACK_ERROR:
		case CUDA_ERROR_ILLEGAL_INSTRUCTION:
		case CUDA_ERROR_MISALIGNED_ADDRESS:
		case CUDA_ERROR_INVALID_ADDRESS_SPACE:
		case CUDA_ERROR_INVALID_PC:
		case CUDA_ERROR_LAUNCH_FAILED:
		case CUDA_ERROR_LAUNCH_TIMEOUT:
		case CUDA_ERROR_ASSERT:
		case CUDA_ERROR_ECC_UNCORRECTABLE:
			return true;
		default:
			return false;
	}
}

[[noreturn]] void reportDeadContextAndExit(CUresult result, const char* what);

}

#define REMO_CU(expr) ::remo::cuCheckImpl((expr), #expr, __FILE__, __LINE__)

#define REMO_CU_OK(expr) (REMO_CU(expr) == CUDA_SUCCESS)

#define REMO_CU_FATAL(expr)                                            \
	do {                                                               \
		CUresult _remo_r = (expr);                                     \
		if (_remo_r != CUDA_SUCCESS)                                   \
			::remo::cuFatal(_remo_r, #expr, __FILE__, __LINE__);       \
	} while (0)
