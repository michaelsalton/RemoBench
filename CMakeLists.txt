cmake_minimum_required(VERSION 3.22)
project(remobench LANGUAGES C CXX)

# ---------------------------------------------------------------------------
# RemoBench -- a point cloud viewer with swappable LOD generation pipelines.
#
# THE PROVENANCE RULE, worth reading before adding an include directory:
#
#   remobench may reference external/*/libs/** only. It must never #include a file
#   that patches/*.patch modifies.
#
# The two research submodules (external/SimLOD, external/CudaLOD) are patched at
# build time by `make simlod` / `make cudalod` and are declared `ignore = dirty`,
# so their working trees are expected to be modified. Building against a file
# whose content depends on patch-application state is not reproducible. Their
# libs/ directories are never touched by either patch, so those are safe to
# reference in place; everything else we need is copied into this repo with
# attribution (see THIRD_PARTY.md).
#
# CUDA is deliberately NOT enabled as a project language. No .cu file is compiled
# at build time -- every kernel under kernels/ is compiled at runtime by NVRTC and
# linked with nvJitLink, which is what makes save-to-reload iteration possible.
# ---------------------------------------------------------------------------

# C++23: the vendored unsuck.hpp uses <print> / <format>.
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif ()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(remobench_deps)
include(remobench_kernels)

# ---------------------------------------------------------------------------
# remobench
# ---------------------------------------------------------------------------
add_executable(remobench
	src/main.cpp
	src/platform/unsuck_platform_specific.cpp
	src/cuda/CudaCheck.cpp
	src/cuda/CudaModularProgram.cpp
	src/cuda/CudaContext.cpp
	src/cuda/GLInterop.cpp
	src/shell/App.cpp
	src/shell/GLRenderer.cpp
	src/shell/GpuProfiler.cpp
	src/shell/PipelineRegistry.cpp
	src/shell/SettingsPanel.cpp
	src/shell/TimingUi.cpp
	src/io/PointSource.cpp
	src/io/RawReader.cpp
	src/io/LasReader.cpp
	src/io/LazReader.cpp
	src/pipelines/FlatPipeline.cpp
	src/pipelines/RemolodPipeline.cpp
	src/pipelines/CudalodPipeline.cpp
	src/pipelines/SimlodPipeline.cpp)

target_include_directories(remobench PRIVATE include src)

target_link_libraries(remobench PRIVATE remobench_deps)

# Our own code we do keep warning-clean. Third-party sources are isolated in the
# remobench_thirdparty target, which is built with -w.
#
# /W4 is MSVC's nearest equivalent to -Wall -Wextra; /Wall is not, it warns on every
# system header. Note cl.exe only warns (D9002) on an unknown -W flag rather than
# failing, so getting this wrong is silently a no-op.
if (MSVC)
	target_compile_options(remobench PRIVATE /W4)
else ()
	target_compile_options(remobench PRIVATE -Wall -Wextra)
endif ()

remobench_link_kernels(remobench)

# No-op off Windows; see the banner in remobench_deps.cmake.
remobench_copy_cuda_runtime(remobench)

# ---------------------------------------------------------------------------
# Tests (Catch2 v3, optional -- absent on a fresh machine)
# ---------------------------------------------------------------------------
option(REMOBENCH_BUILD_TESTS "Build the test suite" OFF)

if (REMOBENCH_BUILD_TESTS)
	find_package(Catch2 3 QUIET)
	if (Catch2_FOUND)
		enable_testing()
		add_subdirectory(tests)
	else ()
		message(STATUS
			"remobench: Catch2 v3 not found, skipping tests "
			"(sudo apt install catch2)")
	endif ()
endif ()
