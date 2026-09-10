# Make kernels/ reachable from the binary's working directory.
#
# The .cu files under kernels/ are NOT build inputs -- they are compiled at
# runtime by NVRTC and watched for changes so that saving a kernel recompiles it
# without restarting. That has one consequence worth stating explicitly:
#
#   Never COPY kernels/ next to the binary. A POST_BUILD copy goes stale the
#   moment you edit a kernel without relinking, so you end up hot-reloading a
#   file the running program isn't reading. Symlink instead.
#
# Upstream SimLOD copies (CMakeLists.txt:112-142) and has a broken
# symlink-detection branch; the CudaLOD Linux port symlinks, which is correct.
#
# We also bake an absolute path into the binary (REMOBENCH_KERNEL_DIR) so a build
# run from anywhere still finds its kernels. The symlink is a convenience for
# running the binary directly from the build dir; the compiled-in path is the
# thing that actually makes `--open` from any cwd work.

function(remobench_link_kernels target)
	set(_link "$<TARGET_FILE_DIR:${target}>/kernels")

	add_custom_command(TARGET ${target} POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E rm -f "${_link}"
		COMMAND ${CMAKE_COMMAND} -E create_symlink
			"${CMAKE_SOURCE_DIR}/kernels" "${_link}"
		COMMENT "symlinking kernels/ next to ${target}"
		VERBATIM)

	# Passed to NVRTC at runtime so a kernel can #include "remo/HostDeviceCommon.h",
	# the one header shared across the host/device boundary. Baked in at configure
	# time rather than resolved relative to the working directory, so the binary works
	# from anywhere.
	target_compile_definitions(${target} PRIVATE
		REMOBENCH_KERNEL_DIR="${CMAKE_SOURCE_DIR}/kernels"
		REMOBENCH_INCLUDE_DIR="${CMAKE_SOURCE_DIR}/include")
endfunction()
