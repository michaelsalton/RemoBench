# RemoBench Makefile
#
# remobench itself is built by CMake -- the dependency set (CUDA driver API, NVRTC,
# nvJitLink, glfw, glew, imgui, implot, laszip) is not something to hand-roll in
# Make. This half of the file is a thin facade so `make`, `make run`, `make debug`
# and `make test` keep working; the second half drives the two reference
# submodules and is deliberately unchanged.

BUILD_DIR := build
TARGET    := $(BUILD_DIR)/remobench

# Extra args for `make run`, e.g.
#   make run ARGS="--open data/morro_bay_35M/morro_bay_36M.simlod"
ARGS ?=

CMAKE_FLAGS ?=

.PHONY: all debug run test check check-vendored check-kernels clean cmake-configure

all: cmake-configure
	cmake --build $(BUILD_DIR) --parallel

# Separate build dir so a debug configure does not thrash the release cache.
debug:
	cmake -S . -B $(BUILD_DIR)-debug -DCMAKE_BUILD_TYPE=Debug $(CMAKE_FLAGS)
	cmake --build $(BUILD_DIR)-debug --parallel

cmake-configure: $(BUILD_DIR)/CMakeCache.txt

$(BUILD_DIR)/CMakeCache.txt:
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release $(CMAKE_FLAGS)

run: all
	./$(TARGET) $(ARGS)

test: all
	ctest --test-dir $(BUILD_DIR) --output-on-failure

# What passes for a test suite until tests/unit/ has something in it.
#
# check-vendored needs no GPU and no build; check-kernels needs a CUDA context but no
# display and no point cloud. Run `make check` after touching anything under kernels/.
check: check-vendored check-kernels

# Assert the comparison baselines are still byte-identical to their submodules. See the
# banner in the script for the failure this exists to prevent.
check-vendored:
	@./bench/check_vendored.sh

# A successful `make` does NOT mean the kernels compile -- every .cu under kernels/ is
# compiled at runtime by NVRTC, so kernel errors are invisible to the build.
check-kernels: all
	@./$(TARGET) --check-kernels

clean:
	rm -rf $(BUILD_DIR) $(BUILD_DIR)-debug

# ---------------------------------------------------------------------------
# Subrepos (external/, git submodules)
#
# These are independent upstream projects with their own build systems; they
# are not compiled into remobench. `make subrepos` lists what's wired up.
# ---------------------------------------------------------------------------

CUDA_PATH       ?= /usr/local/cuda
SIMLOD_GPU_ARCH ?= compute_120

SIMLOD_SRC   := external/SimLOD
SIMLOD_BUILD := $(SIMLOD_SRC)/build
SIMLOD_PATCH := patches/simlod-linux-sm120.patch
CUDALOD_SRC  := external/CudaLOD

.PHONY: subrepos simlod simlod-check simlod-patch simlod-build simlod-clean cudalod

subrepos:
	@echo "Subrepo targets:"
	@echo "  make simlod    build + run SimLOD   ($(SIMLOD_SRC))"
	@echo "  make cudalod   build + run CudaLOD  ($(CUDALOD_SRC))"
	@echo "                 requires CUDALOD_LAS=/path/to/cloud.las"
	@echo
	@echo "Overrides:"
	@echo "  CUDA_PATH=$(CUDA_PATH)"
	@echo "  SIMLOD_GPU_ARCH=$(SIMLOD_GPU_ARCH)  CUDALOD_GPU_ARCH=$(CUDALOD_GPU_ARCH)"

# --- SimLOD ---------------------------------------------------------------

simlod-check:
	@test -f $(SIMLOD_SRC)/CMakeLists.txt || { \
		echo "error: $(SIMLOD_SRC) is empty."; \
		echo "       run: git submodule update --init --recursive"; exit 1; }
	@command -v cmake >/dev/null || { \
		echo "error: cmake not found."; \
		echo "       run: sudo apt install cmake"; exit 1; }
	@test -d $(CUDA_PATH)/include || { \
		echo "error: no CUDA toolkit at $(CUDA_PATH)"; \
		echo "       run: sudo apt install cuda-toolkit-13-1"; \
		echo "       or:  make simlod CUDA_PATH=/path/to/cuda"; exit 1; }
	@test -e /usr/include/GL/glu.h || { \
		echo "error: GL/glu.h not found (glew.h includes it unconditionally)."; \
		echo "       run: sudo apt install libglu1-mesa-dev"; exit 1; }

# Local fixes upstream doesn't have: link nvJitLink, null-safe CUDA_PATH,
# and a non-hardcoded GPU arch (upstream pins compute_89 / Ada).
#
# --ignore-whitespace on both the check and the apply: upstream sources are
# CRLF, and cudalod-linux-port.patch is LF-only. Without it the reverse-check
# never detects an already-applied patch, so a second `make cudalod` falls
# through to a forward apply that also fails ("already exists in working
# directory") -- i.e. the target only ever worked once.
GIT_APPLY := git apply --ignore-whitespace --whitespace=nowarn

simlod-patch:
	@cd $(SIMLOD_SRC) && \
	if $(GIT_APPLY) --reverse --check ../../$(SIMLOD_PATCH) 2>/dev/null; then \
		echo "SimLOD: patch already applied"; \
	else \
		$(GIT_APPLY) ../../$(SIMLOD_PATCH) && echo "SimLOD: patch applied"; \
	fi

# CMAKE_POLICY_VERSION_MINIMUM: glfw 3.3.2 is fetched at configure time and
# declares a pre-3.5 minimum, which CMake 4 rejects outright.
$(SIMLOD_BUILD)/CMakeCache.txt: | simlod-check simlod-patch
	cmake -S $(SIMLOD_SRC) -B $(SIMLOD_BUILD) \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_POLICY_VERSION_MINIMUM=3.5

simlod-build: $(SIMLOD_BUILD)/CMakeCache.txt
	cmake --build $(SIMLOD_BUILD) --parallel

# Run from the build dir: SimLOD resolves ./modules relative to the binary,
# and CMake's post-build step copies modules/ there.
simlod: simlod-build
	cd $(SIMLOD_BUILD) && \
	CUDA_PATH=$(CUDA_PATH) SIMLOD_GPU_ARCH=$(SIMLOD_GPU_ARCH) ./SimLOD

simlod-clean:
	rm -rf $(SIMLOD_BUILD)

# --- CudaLOD --------------------------------------------------------------
#
# Upstream ships only Visual Studio project files, so CMakeLists.txt is ours
# (see patches/cudalod-linux-port.patch). Note the build dir is cmake-build,
# not build/ -- upstream tracks build/ for the .sln.

CUDALOD_BUILD    := $(CUDALOD_SRC)/cmake-build
CUDALOD_PATCH    := patches/cudalod-linux-port.patch
CUDALOD_GPU_ARCH ?= compute_120
# Bytes for the CudaLOD device slab. The patch defaults to 4GB, which suits a
# 16GB card: 36M points only watermark at ~1.4GB, while >= 10GB builds the LOD
# correctly but then floods "illegal memory access" once rendering starts.
# Upstream's own value was 15GB (assumes a 24GB+ card).
CUDALOD_MAX_BUFFER ?=

# Default input: the smallest .las under data/ (ls -S -r sorts ascending by
# size). The larger clouds need MAX_BUFFER_SIZE raised past what 16GB of VRAM
# allows. Override per-run with `make cudalod CUDALOD_LAS=...`.
CUDALOD_LAS ?= $(shell ls -S -r data/*.las data/*/*.las 2>/dev/null | head -1)

.PHONY: cudalod cudalod-check cudalod-patch cudalod-build cudalod-clean

cudalod-check:
	@test -d $(CUDALOD_SRC)/src || { \
		echo "error: $(CUDALOD_SRC) is empty."; \
		echo "       run: git submodule update --init --recursive"; exit 1; }
	@command -v cmake >/dev/null || { \
		echo "error: cmake not found."; \
		echo "       run: sudo apt install cmake"; exit 1; }
	@test -d $(CUDA_PATH)/include || { \
		echo "error: no CUDA toolkit at $(CUDA_PATH)"; \
		echo "       run: sudo apt install cuda-toolkit-13-1"; exit 1; }
	@test -e /usr/include/GL/glu.h || { \
		echo "error: GL/glu.h not found."; \
		echo "       run: sudo apt install libglu1-mesa-dev"; exit 1; }

cudalod-patch:
	@cd $(CUDALOD_SRC) && \
	if $(GIT_APPLY) --reverse --check ../../$(CUDALOD_PATCH) 2>/dev/null; then \
		echo "CudaLOD: patch already applied"; \
	else \
		$(GIT_APPLY) ../../$(CUDALOD_PATCH) && echo "CudaLOD: patch applied"; \
	fi

$(CUDALOD_BUILD)/CMakeCache.txt: | cudalod-check cudalod-patch
	cmake -S $(CUDALOD_SRC) -B $(CUDALOD_BUILD) \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
		$(if $(CUDALOD_MAX_BUFFER),-DCUDALOD_MAX_BUFFER_SIZE=$(CUDALOD_MAX_BUFFER),)

cudalod-build: $(CUDALOD_BUILD)/CMakeCache.txt
	cmake --build $(CUDALOD_BUILD) --parallel

# CudaLOD takes no arguments and its input path is hardcoded upstream; the
# patch adds a CUDALOD_LAS override.
cudalod: cudalod-build
	@test -n "$(CUDALOD_LAS)" || { \
		echo "error: no point cloud found - CudaLOD needs one."; \
		echo "       drop a .las under data/, or run:"; \
		echo "       make cudalod CUDALOD_LAS=/path/to/cloud.las"; exit 1; }
	@test -f "$(CUDALOD_LAS)" || { \
		echo "error: no such file: $(CUDALOD_LAS)"; exit 1; }
	cd $(CUDALOD_BUILD) && \
	CUDA_PATH=$(CUDA_PATH) CUDALOD_GPU_ARCH=$(CUDALOD_GPU_ARCH) \
	CUDALOD_LAS=$(abspath $(CUDALOD_LAS)) ./CudaLOD

cudalod-clean:
	rm -rf $(CUDALOD_BUILD)
