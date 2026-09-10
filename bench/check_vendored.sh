#!/usr/bin/env bash
# Assert that every vendored kernel is byte-identical to the submodule it came from.
#
# This exists because inspection did not catch the drift that motivated it: the accumulator
# hook was edited straight into SimLOD's progressive_octree_voxels.cu, which added two
# kernel_construct parameters without the matching host arguments. Every launch then failed
# with CUDA_ERROR_INVALID_VALUE and the SimLOD pipeline built no tree at all -- while `make`
# succeeded, `--check-kernels` passed, and `--dump-frame` still exited 0. Nothing in the
# toolchain noticed for a whole commit.
#
# The rule the check enforces: a vendored file is HEADER_LINES of RemoBench provenance
# banner followed by the upstream file, unchanged. Notes about a vendored file go in the
# sidecar (kernels/*/VENDORED.md), never in the file, so that "is this still upstream?" is a
# diff rather than a judgement call.
#
# RemoLOD is free to FORK any of this into kernels/remolod/ and change it there. What it may
# not do is edit the comparison baselines in place -- they are only worth having while they
# still reproduce their published numbers against bench/reference/.
set -u

cd "$(dirname "$0")/.." || exit 2

HEADER_LINES=5   # 4 comment lines + 1 blank; upstream content starts on line 6
fail=0
checked=0

# dest <TAB> upstream-relative-path <TAB> submodule
while IFS=$'\t' read -r dest rel sub; do
	[ -z "${dest:-}" ] && continue
	case "$dest" in \#*) continue ;; esac

	up="external/$sub/$rel"
	if [ ! -f "$dest" ]; then
		printf '  MISSING   %s\n' "$dest"; fail=1; continue
	fi
	if [ ! -f "$up" ]; then
		printf '  NO UPSTREAM %s (run: git submodule update --init)\n' "$up"; fail=1; continue
	fi

	checked=$((checked + 1))
	if diff -q <(tail -n +$((HEADER_LINES + 1)) "$dest") "$up" >/dev/null 2>&1; then
		printf '  ok        %s\n' "$dest"
	else
		printf '  DRIFTED   %s\n' "$dest"
		diff <(tail -n +$((HEADER_LINES + 1)) "$dest") "$up" | head -40 | sed 's/^/              /'
		fail=1
	fi
done <<'MANIFEST'
kernels/simlod/progressive_octree_voxels.cu	modules/progressive_octree/progressive_octree_voxels.cu	SimLOD
kernels/simlod/structures.cuh	modules/progressive_octree/structures.cuh	SimLOD
kernels/simlod/reset.cu	modules/progressive_octree/reset.cu	SimLOD
kernels/simlod/utils.h.cu	modules/progressive_octree/utils.h.cu	SimLOD
kernels/simlod/HostDeviceInterface.h	modules/progressive_octree/HostDeviceInterface.h	SimLOD
kernels/simlod/math.cuh	modules/progressive_octree/math.cuh	SimLOD
kernels/simlod/rasterization.cuh	modules/progressive_octree/rasterization.cuh	SimLOD
kernels/simlod/helper_math.h	modules/progressive_octree/helper_math.h	SimLOD
kernels/CudaPrint/CudaPrint.cuh	modules/CudaPrint/CudaPrint.cuh	SimLOD
MANIFEST

if [ "$fail" -ne 0 ]; then
	printf 'remobench: vendored kernels have DRIFTED from their submodule (%d checked)\n' "$checked"
	printf '           RemoLOD forks go in kernels/remolod/; baselines are not edited in place.\n'
	exit 1
fi

printf 'remobench: %d vendored kernel(s) byte-identical to their submodule\n' "$checked"
