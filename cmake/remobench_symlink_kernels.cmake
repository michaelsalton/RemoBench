# POST_BUILD helper for remobench_link_kernels(). Run with -P; see the banner in
# remobench_kernels.cmake for why this is a symlink and must never become a copy.
#
# Best-effort on purpose. Windows refuses symlink creation unless Developer Mode is
# on or the shell is elevated, and that must not fail the build: the binary finds its
# kernels through the compiled-in absolute REMOBENCH_KERNEL_DIR, not through this
# link (see CudaModularProgram.cpp). The link only makes ./remobench convenient from
# inside the build dir.
#
# Expects REMOBENCH_KERNELS_SRC and REMOBENCH_KERNELS_LINK on the command line.

file(REMOVE "${REMOBENCH_KERNELS_LINK}")

file(CREATE_LINK
	"${REMOBENCH_KERNELS_SRC}" "${REMOBENCH_KERNELS_LINK}"
	SYMBOLIC RESULT _remobench_link_result)

# Capturing RESULT is what keeps a failure non-fatal -- without it, file(CREATE_LINK)
# raises a hard error and takes the build down with it.
if (NOT _remobench_link_result EQUAL 0)
	message(STATUS
		"remobench: could not symlink kernels/ next to the binary "
		"(${_remobench_link_result}).\n"
		"   Harmless: kernels are loaded from the compiled-in absolute path.\n"
		"   On Windows, enable Developer Mode to get the convenience link.")
endif ()
