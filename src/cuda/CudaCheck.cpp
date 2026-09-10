#include "remo/CudaCheck.h"

#include <cstdio>
#include <cstdlib>

namespace remo {

void reportDeadContextAndExit(CUresult result, const char* what) {
	fflush(stdout);
	fprintf(stderr,
	        "\n"
	        "remobench: FATAL -- the CUDA context is dead (%s)\n"
	        "  while running: %s\n"
	        "  detail: %s\n"
	        "\n"
	        "A kernel made an illegal memory access or otherwise faulted. The context\n"
	        "cannot be recovered, so continuing would only produce a cascade of identical\n"
	        "errors from every later driver call, and eventually host heap corruption and\n"
	        "a crash somewhere unrelated. Exiting here instead, so the report points at\n"
	        "the cause.\n"
	        "\n"
	        "Most likely causes, in order:\n"
	        "  - a device-side buffer overrun. The bump allocators have no bounds check\n"
	        "    on the reference kernels' side; RemoBench's own (kernels/shared/remo_alloc.cuh)\n"
	        "    reports overflow through DeviceDiagnostics instead.\n"
	        "  - an input distribution the pipeline cannot handle. CudaLOD's split runs\n"
	        "    once at a fixed depth and its capacities are unchecked, so a degenerate\n"
	        "    cloud (coplanar or duplicated points) can walk off the end.\n"
	        "  - too small a device budget for the chosen sampling strategy.\n"
	        "\n"
	        "To narrow it down:\n"
	        "  compute-sanitizer ./build/remobench <same arguments>\n",
	        cuErrorName(result), what, cuErrorString(result));
	fflush(stderr);
	// _Exit, not exit: with a dead context, running static destructors invites the CUDA
	// driver's own teardown to fail on top of the original error.
	std::_Exit(70);  // EX_SOFTWARE
}

}  // namespace remo
