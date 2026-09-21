# Dependency wiring for remobench.
#
# THE RULE (see also the comment in CMakeLists.txt):
#
#   remobench may reference external/*/libs/** only. It must never #include a file
#   that patches/*.patch modifies.
#
# Verified against both patch files: they touch CMakeLists.txt, include/*.h,
# modules/**, src/** -- never libs/. That keeps `make` working after a bare
# `git submodule update --init`, regardless of whether the submodule patches have
# been applied, and independent of the `ignore = dirty` submodule config.
#
# Anything we need to *edit* is copied into this repo instead (see THIRD_PARTY.md).

set(SIMLOD_LIBS "${CMAKE_SOURCE_DIR}/external/SimLOD/libs")

if (NOT EXISTS "${SIMLOD_LIBS}/glew/glew.c")
	message(FATAL_ERROR
		"external/SimLOD is empty. Run: git submodule update --init --recursive")
endif ()

# ---------------------------------------------------------------------------
# CUDA. Driver API + NVRTC only -- no .cu is compiled at build time. Every
# kernel under kernels/ is compiled at *runtime* by NVRTC and linked with
# nvJitLink (see src/cuda/CudaModularProgram.cpp), which is what makes hot
# reload possible.
#
# nvJitLink needs an explicit link line; upstream SimLOD uses the header but
# forgets to link it, which is one of the things patches/simlod-linux-sm120.patch
# has to fix. Get it right from the start here.
# ---------------------------------------------------------------------------
find_package(CUDAToolkit 12.4 REQUIRED)

# ---------------------------------------------------------------------------
# OpenGL. Used only for the window, the ImGui overlay, and blitting the
# CUDA-rendered texture to the backbuffer.
#
# GLU: libs/glew's glew.h includes <GL/glu.h> unconditionally, so libglu1-mesa-dev
# is a hard requirement even though we never call a GLU function. It is a separate
# package from libgl1-mesa-dev and easy to miss, so fail loudly and early.
#
# Linux only: the header ships with the Windows SDK, so there is nothing to check
# for and nothing to install. The path is hardcoded rather than found, so guarding
# on the platform is the fix -- FindOpenGL's OPENGL_glu_LIBRARY says whether the
# *library* is linkable, which is not what glew.h needs.
# ---------------------------------------------------------------------------
find_package(OpenGL REQUIRED)

if (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT EXISTS "/usr/include/GL/glu.h")
	message(FATAL_ERROR
		"GL/glu.h not found, but libs/glew/glew.h includes it unconditionally.\n"
		"        Run: sudo apt install libglu1-mesa-dev")
endif ()

# ---------------------------------------------------------------------------
# GLFW. Prefer a system package if one ever appears; otherwise build the
# external/glfw submodule (pinned to 3.4).
#
# Neither research submodule can supply this: their libs/glfw holds headers plus
# a prebuilt msvc2017_x64 .lib. SimLOD FetchContent's 3.3.2 at configure time,
# which needs network access on every fresh configure AND the
# -DCMAKE_POLICY_VERSION_MINIMUM=3.5 workaround, because 3.3.2 declares a
# pre-3.5 minimum that CMake 4 rejects outright. A pinned submodule at 3.4 avoids
# both, and has a better Wayland/X11 story.
# ---------------------------------------------------------------------------
find_package(glfw3 3.3 QUIET)

if (glfw3_FOUND)
	message(STATUS "remobench: using system glfw3 ${glfw3_VERSION}")
else ()
	if (NOT EXISTS "${CMAKE_SOURCE_DIR}/external/glfw/CMakeLists.txt")
		message(FATAL_ERROR
			"external/glfw is empty and no system glfw3 was found.\n"
			"        Run: git submodule update --init --recursive")
	endif ()
	message(STATUS "remobench: building external/glfw (submodule)")
	set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
	set(GLFW_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
	set(GLFW_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
	set(GLFW_INSTALL        OFF CACHE BOOL "" FORCE)
	add_subdirectory("${CMAKE_SOURCE_DIR}/external/glfw" glfw EXCLUDE_FROM_ALL)
endif ()

# ---------------------------------------------------------------------------
# laszip -- only needed for .laz. Built SHARED, which also keeps its LGPL-2.1
# obligation simple (see THIRD_PARTY.md).
# ---------------------------------------------------------------------------
add_subdirectory("${SIMLOD_LIBS}/laszip" laszip EXCLUDE_FROM_ALL)

# ---------------------------------------------------------------------------
# Header-only / compiled-in third-party sources, referenced in place from
# external/SimLOD/libs.
#
# imgui and implot are PINNED here deliberately and must not be upgraded before
# the shell is rewritten: SimLOD's GLRenderer uses ImPlot::SetNextPlotLimitsX and
# a 3-arg BeginPlot, both removed in modern ImPlot. Referencing the vendored copy
# keeps that pin automatic rather than aspirational.
# ---------------------------------------------------------------------------
add_library(remobench_thirdparty STATIC
	"${SIMLOD_LIBS}/glew/glew.c"
	"${SIMLOD_LIBS}/imgui/imgui.cpp"
	"${SIMLOD_LIBS}/imgui/imgui_draw.cpp"
	"${SIMLOD_LIBS}/imgui/imgui_tables.cpp"
	"${SIMLOD_LIBS}/imgui/imgui_widgets.cpp"
	"${SIMLOD_LIBS}/imgui/imgui_demo.cpp"
	"${SIMLOD_LIBS}/imgui/backends/imgui_impl_glfw.cpp"
	"${SIMLOD_LIBS}/imgui/backends/imgui_impl_opengl3.cpp"
	"${SIMLOD_LIBS}/implot/implot.cpp"
	"${SIMLOD_LIBS}/implot/implot_items.cpp")

target_include_directories(remobench_thirdparty SYSTEM PUBLIC
	"${SIMLOD_LIBS}/glew/include"
	"${SIMLOD_LIBS}/glm"
	"${SIMLOD_LIBS}/imgui"
	"${SIMLOD_LIBS}/imgui/backends"
	"${SIMLOD_LIBS}/implot"
	"${SIMLOD_LIBS}/laszip")

# GLEW_STATIC: we compile glew.c into this target rather than linking a .so.
target_compile_definitions(remobench_thirdparty PUBLIC GLEW_STATIC)

# Third-party code is not ours to keep warning-clean.
if (MSVC)
	target_compile_options(remobench_thirdparty PRIVATE /w)
else ()
	target_compile_options(remobench_thirdparty PRIVATE -w)
endif ()

target_link_libraries(remobench_thirdparty PUBLIC glfw OpenGL::GL)

# ---------------------------------------------------------------------------
# The interface target remobench links against.
# ---------------------------------------------------------------------------
add_library(remobench_deps INTERFACE)
target_link_libraries(remobench_deps INTERFACE
	remobench_thirdparty
	laszip
	CUDA::cuda_driver
	CUDA::nvrtc
	CUDA::nvJitLink)

# Note CUDAToolkit_INCLUDE_DIRS is a LIST, and on CUDA 13 it already contains
# include/cccl -- where the CCCL / libcu++ headers (<cuda/std/*>, pulled in by
# cooperative_groups) were relocated to. Do not append "/cccl" by hand; on CUDA 13
# that yields "<a>;<b>/cccl" and, in a compile definition, an unterminated string.
target_include_directories(remobench_deps SYSTEM INTERFACE
	"${CUDAToolkit_INCLUDE_DIRS}")

# Bake the toolkit include root in at configure time instead of reading CUDA_PATH
# from the environment at runtime, which is what both research repos do and why
# they need an env var set just to compile a kernel.
#
# Only the first element: this becomes a quoted C string, so it must be a single
# path. CudaModularProgram derives the cccl subdirectory from it (a nonexistent -I
# is harmless to NVRTC, so this is safe on CUDA 12 as well).
list(GET CUDAToolkit_INCLUDE_DIRS 0 REMOBENCH_CUDA_INCLUDE_ROOT)
target_compile_definitions(remobench_deps INTERFACE
	REMOBENCH_CUDA_INCLUDE_DIR="${REMOBENCH_CUDA_INCLUDE_ROOT}")

# ---------------------------------------------------------------------------
# Windows: put NVRTC and nvJitLink next to the binary.
#
# There is no rpath on Windows, so these are resolved from the directory of the
# executable or from PATH. PATH commonly holds an OLDER toolkit's bin -- a machine
# with 11.6 and 12.4 installed side by side will have exactly one of them first --
# and the failure mode is a bare 0xC0000135 at process start with no message about
# which DLL or which version. Copying the ones we linked against removes both the
# ordering question and the need to prepend anything to PATH before running.
#
# This is a copy of a redistributable runtime, not of anything under kernels/; the
# banner in remobench_kernels.cmake is about kernel SOURCE, which must stay a
# symlink so hot reload reads the file you are editing.
#
# Deliberately not an install() rule: nothing here is installed, the build tree is
# where the binary is run from.
# ---------------------------------------------------------------------------
function(remobench_copy_cuda_runtime target)
	if (NOT WIN32)
		return ()
	endif ()

	# Globbed rather than named: the soname carries the toolkit version
	# (nvrtc64_120_0.dll, nvrtc-builtins64_124.dll), so hardcoding it would silently
	# copy nothing after a toolkit bump. nvrtc-builtins is not linked against, but
	# nvrtc loads it at runtime and fails the first compile without it.
	file(GLOB _cuda_runtime_dlls
		"${CUDAToolkit_BIN_DIR}/nvrtc64_*.dll"
		"${CUDAToolkit_BIN_DIR}/nvrtc-builtins64_*.dll"
		"${CUDAToolkit_BIN_DIR}/nvJitLink_*.dll")

	if (NOT _cuda_runtime_dlls)
		message(WARNING
			"remobench: no NVRTC/nvJitLink DLLs found in ${CUDAToolkit_BIN_DIR}.\n"
			"        The binary will only start if they are reachable via PATH.")
		return ()
	endif ()

	add_custom_command(TARGET ${target} POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${_cuda_runtime_dlls} "$<TARGET_FILE_DIR:${target}>"
		COMMENT "copying CUDA runtime DLLs next to ${target}"
		VERBATIM)
endfunction()
