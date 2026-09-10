# Notes on the vendored SimLOD kernels

Everything in this directory listed below is **byte-identical to `external/SimLOD`** apart
from a 5-line provenance banner at the top of each file. `bench/check_vendored.sh`
(`make check-vendored`) asserts it.

These notes live here rather than in the files because a comment inside a vendored file is a
diff against upstream, and then "is this still upstream?" stops being answerable with `diff`
and becomes a judgement call. That judgement call is what let the accumulator hook sit inside
`progressive_octree_voxels.cu` unnoticed.

## Why byte-identity matters

`simlod` and `cudalod` are **external comparison pipelines**. They are worth having only
while they still reproduce their published numbers against `bench/reference/`. An edit to
either — however well intentioned — destroys the baseline the research is measured against.

RemoLOD is the pipeline that gets to change things. When it needs behaviour these kernels do
not have, it **forks** the file into `kernels/remolod/`. See
`kernels/remolod/remolod_structures.cuh` for the two constants it has already diverged on.

## What broke when this rule was not in place

The per-node accumulator was originally spliced into `addBatch()` as a seventh phase, which
meant adding two parameters to `kernel_construct`. The host launch was never updated to pass
them. `cuLaunchCooperativeKernel` reads one entry per declared parameter, found ten where the
kernel wanted twelve, and failed **every** launch with `CUDA_ERROR_INVALID_VALUE`.

The SimLOD pipeline built no tree at all — 0 points, 1 node — for a whole commit, while:

- `make` succeeded (no `.cu` is a build input; NVRTC compiles at runtime),
- `--check-kernels` passed (it compiles and links; it never launches),
- `--dump-frame` exited 0 and wrote a perfectly valid image of an empty tree.

The accumulator now lives in `kernels/remolod/remolod_accum.cu` as a separate launch that
reads the finished tree. Nothing in this directory had to be touched, and the pass is cheaper
than the hook was.

## The files

| file | what it is |
| --- | --- |
| `progressive_octree_voxels.cu` | `kernel_construct`: inserts up to 20 uploaded 1M-point batches into the LIVE octree per launch, under a 10 ms device-side wall-clock budget (`%%globaltimer`), advancing the persistent cursor `stats->batchletIndex`. Nothing is rebuilt — the tree is rendered while it is still being inserted into. `addBatch()` is six phases separated by `grid.sync()`: expand (doCounting ↔ doSplitting until no node overflows) → voxelSampling → allocate point chunks → allocate voxel chunks → insert points → insert voxels → stats. |
| `structures.cuh` | `Node` / `Chunk` / `OccupancyGrid`, and the tunables. Leaves hold original points and inner nodes hold voxels, both as a **linked list** of 1000-point `Chunk`s — not a contiguous slice like CudaLOD. That is why the shared rasteriser walks samples through a template Walker; see `kernels/shared/remo_draw.cuh`. |
| `reset.cu` | Clears the octree and initialises the persistent allocator in place. Factored out of the construct kernel upstream, which is why `progressive_octree_mno.cu` still tries to reset inline via a `Uniforms` field that no longer exists. |
| `utils.h.cu` | `processRange`, `nanotime`, and the two bump allocators. `Allocator` is **non-atomic by design**: every thread walks the identical allocation sequence so all threads derive identical pointers with no atomics. It requires uniform control flow, and it has **no capacity and no bounds check** — which is why upstream's momentary allocator quietly hands out ~409 MB from a 300 MB buffer. Kept as-is so the port stays verifiable; the host bounds the damage by sizing the buffers from the actual allocation sum. Our own kernels use `kernels/shared/remo_alloc.cuh`, which does check. |
| `HostDeviceInterface.h` | SimLOD's own host/device contract: `Uniforms` and `Stats`. Deliberately **not** merged into `remo/HostDeviceCommon.h` — rewriting the struct the reference kernels read is how a port silently stops reproducing its published numbers. Several `Uniforms` fields are plumbed but read by no kernel (`LOD`, `doProgressive`, `colorWhite`, `updateStats`, `enableEDL`, `edlStrength`); RemoBench drives shading from `SharedUniforms` instead, so those stay unused rather than becoming placebo controls. |
| `math.cuh` | Matrix and vector helpers. |
| `rasterization.cuh` | Upstream's software rasteriser. **Unused** — RemoBench rasterises through `kernels/shared/`. Kept because `kernels/shared/remo_lines.cuh` cites its 400-step clamp. |
| `helper_math.h` | NVIDIA's CUDA-samples `float2`/`3`/`4` operator header. |
| `../CudaPrint/CudaPrint.cuh` | A device→host `printf` ring buffer that is a **no-op on both ends**: `print()` returns immediately, and the host half is entirely commented out upstream. Vendored anyway, because it is threaded through both kernel signatures and called from `progressive_octree_voxels.cu` — excising it would mean editing the kernels being validated, for no benefit. The host passes a small dummy allocation. Device `printf()` works and is what to reach for instead. |

`CudaPrint.cuh` sits at `kernels/CudaPrint/` rather than in this directory so that upstream's
`#include "../CudaPrint/CudaPrint.cuh"` resolves **unchanged**. Mirroring upstream's layout
was cheaper than carrying an edited include line and a whitelist entry to excuse it.

## RemoBench's own files in this directory

Not vendored, not checked by `check_vendored.sh`, and free to change:

| file | what it is |
| --- | --- |
| `simlod_render.cu` | SimLOD's LOD **selection** only, feeding the shared rasteriser. Replaces upstream's 1356-line `render.cu`. |
| `simlod_layout.h` | The device-layout numbers the host needs, in a header both sides can compile. Holds `kMaxNodes`, `kNodeBytes`, `kBatchStreamSize`, and the 50M point ceiling that follows from the last of those. |
| `simlod_bridge.cuh` | `static_assert`s `simlod_layout.h` against the real vendored types. These checks used to live inside `structures.cuh`; they are here now because that file is not ours to edit. |
| `programs.txt` | The link groups `--check-kernels` reads. |
| `VENDORED.md` | this file |
