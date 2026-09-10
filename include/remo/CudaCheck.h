// CUDA driver-API error handling.
//
// Both research codebases have a `cu_checked` that cout's an error code and
// carries on, which means a failed allocation or a bad launch surfaces later as
// mysterious corruption instead of at the call site. Given that the GPU-side
// allocators here have no bounds checks either, silent failure is the last thing
// this project needs.
//
//   REMO_CU(expr)      -- log with file/line/name, return the CUresult
//   REMO_CU_OK(expr)   -- as above, evaluates to true on success
//   REMO_CU_FATAL(expr)-- log and abort; only for genuinely unrecoverable setup

#pragma once

#include <cstdio>
#include <cstdlib>

#include <cuda.h>

namespace remo {

// Driver error enum -> symbolic name. cuGetErrorName can itself fail (e.g. before
// cuInit), so fall back to the numeric code rather than returning nullptr.
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

// A "sticky" error means the CUDA context is dead: a kernel made an illegal access, hit
// an assert, or was aborted. Nothing after that point is recoverable without recreating
// the context, and every subsequent driver call fails.
//
// This must be treated as terminal rather than logged and ignored. Carrying on produced,
// in order: a screenful of identical errors from every later call, then host heap
// corruption, then a SIGSEGV in an unrelated thread -- a debugging trail that points
// nowhere near the actual fault. Failing at the first sticky error keeps the diagnosis
// where the cause is.
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

// Checks for a dead context and, if found, reports what was running and exits.
//
// `what` should name the pipeline and phase, because the fault is in device code and the
// stack trace will not tell you which kernel it was.
[[noreturn]] void reportDeadContextAndExit(CUresult result, const char* what);

}  // namespace remo

#define REMO_CU(expr) ::remo::cuCheckImpl((expr), #expr, __FILE__, __LINE__)

#define REMO_CU_OK(expr) (REMO_CU(expr) == CUDA_SUCCESS)

#define REMO_CU_FATAL(expr)                                            \
	do {                                                               \
		CUresult _remo_r = (expr);                                     \
		if (_remo_r != CUDA_SUCCESS)                                   \
			::remo::cuFatal(_remo_r, #expr, __FILE__, __LINE__);       \
	} while (0)
