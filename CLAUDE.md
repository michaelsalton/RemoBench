# RemoBench

A real-time LOD generation and rendering program for lidar point clouds. It extends
SimLOD and borrows from CudaLOD. **The project's goal is to make SimLOD's octree
construction detail-aware** — geometry analysis driving per-node sample budgets and
depth, computed online while the cloud is still streaming in.

See [README.md](README.md) for the idea, the kernel architecture, build details and
current status.

## The four pipelines, and which ones you may touch

| id | role | may be changed |
| --- | --- | --- |
| `remolod` | **RemoLOD — this project's own pipeline.** The detail-aware work lives here. | yes, freely |
| `simlod` | external comparison baseline, progressive build | **no** |
| `cudalod` | external comparison baseline, batch build | **no** |
| `flat` | the control condition: no LOD, every point drawn | rarely |

**RemoLOD may pull anything it needs out of the comparison pipelines, but never changes
them.** They are only worth having while they still reproduce their published numbers
against `bench/reference/`; an edit to either destroys the baseline the research is
measured against. So RemoLOD *forks* what it needs into `kernels/remolod/` and changes the
fork. Reading a vendored header (`#include "../simlod/math.cuh"`) is pulling, not changing,
and is fine.

`./bench/check_vendored.sh` (also `make check-vendored`) asserts this mechanically: every
vendored kernel must be a 5-line provenance banner followed by the submodule's file, byte
for byte. Notes about a vendored file go in `kernels/simlod/VENDORED.md`, never in the file,
so "is this still upstream?" is a diff rather than a judgement call.

This rule exists because it was broken. The accumulator was originally edited straight into
SimLOD's `progressive_octree_voxels.cu`, adding two `kernel_construct` parameters that the
host launch never passed. Every launch then failed `CUDA_ERROR_INVALID_VALUE` and the SimLOD
pipeline silently built no tree at all for a whole commit — while `make` succeeded,
`--check-kernels` passed, and `--dump-frame` still exited 0.

`flat` is the control condition and the image-quality ground truth. It is not dead code and
is not a candidate for optimisation-by-deletion.

## What the work actually is

Five kernels. The first three exist; the last two are the project.

| kernel | when | state |
| --- | --- | --- |
| Rasterize | every frame | exists — `kernels/remolod/remolod_render.cu` |
| Update (octree construction) | on batch completion | exists — `kernels/remolod/remolod_octree.cu` |
| Accumulate | after each Update | exists — `kernels/remolod/remolod_accum.cu` |
| Analysis | every frame, per node | **to build** |
| Refinement | if budget remains | **to build** |

Analysis is cheap, unconditional and does not mutate the tree. Refinement is the only
thing that mutates it and the only thing that consumes budget — which is what makes
"how much refinement budget the system got" the single independent variable of the
evaluation. Keep that separation.

**The accumulator is a separate launch, not a hook.** It reads the finished tree after
`kernel_construct` returns, carrying a per-node watermark (`NodeAccum::count`) and folding
`[count, node->numPoints)` from each leaf's own chunk list. That is what lets
`kernels/simlod/` stay byte-identical, and it turned out cheaper than the hook it replaced:
no octree descent per point, one uncontended write per leaf instead of atomics per point,
and its own `GpuScope`. Prefer a separate pass whenever a hook is tempting.

Read `wiki` and `plans` for the current shape of the design before touching LOD selection
metrics, node budgets or split criteria.

## The shared path

**Everything outside LOD generation stays shared.** One loader, one camera, one
rasteriser (`kernels/shared/`), one pixel budget and one device-memory budget handed
unchanged to every pipeline. A change that gives one pipeline its own render path, its
own budget interpretation or its own loader destroys the only baseline the research has.

If a pipeline genuinely needs different behaviour, that is a finding to surface, not a
special case to add. Say so rather than diverging the shared path.

## Working here

```sh
make                       # release -> build/remobench
make debug                 # -O0 -g -> build-debug/
make check                 # check-vendored + check-kernels; run this after any kernels/ change
make run ARGS="--open data/morro_bay_35M/morro_bay_36M.simlod"
./build/remobench --check-kernels    # compile every kernel headlessly; exits non-zero on failure
./build/remobench --dump-frame ...   # headless frame capture for verification
```

**A successful `make` does not mean the kernels compile.** Every `.cu` under `kernels/`
is compiled at runtime by NVRTC — that is what makes hot reload work, and it means
kernel errors are invisible to the build. After touching anything under `kernels/`, run
`make check`. It needs no display, and `check-vendored` needs no GPU at all.

**And compiling is not launching.** `--check-kernels` compiles and links; it does not launch
anything, so it cannot see a host/device signature mismatch — which is exactly what took the
SimLOD pipeline down. A change to a kernel signature is only verified by actually running the
pipeline and looking at the structural counts.

There is no test suite yet: `tests/unit/` is empty and `REMOBENCH_BUILD_TESTS` is OFF, so
`make test` currently runs ctest against nothing. Verification today is `make check`,
`--dump-frame` and the structural counts in `bench/reference/`.

`.simlod`, `.las` and `.laz` all load, whole-cloud, and produce **bit-identical trees** from
the same cloud — the strongest verification the readers have, and worth re-running after
touching any of them.

**Compare structural counts, not frames, for anything with an octree.** Only `flat` is
run-to-run deterministic. `simlod`, `remolod` and `cudalod` all sample a voxel's colour from
the first point to reach its cell, so which *thread* gets there first decides the colour, and
two runs of the same binary on the same file produce different images. That is first-come
sampling doing exactly what the README criticises it for, and it is one of the things colour
filtering is meant to fix. Use `--dump-frame` plus `cmp` for `flat`; use the printed
point/voxel/node counts everywhere else.
Load rates on morro_bay: `.simlod` ~100 MP/s, `.las` ~110–140 MP/s (eight loader threads),
`.laz` ~5 MP/s (laszip, single-threaded by necessity — see the banner in
`src/io/LazReader.cpp`). Data lives under `data/` and is gitignored; do not assume a cloud
is present.

## Layout

| Path | Contents |
| --- | --- |
| `include/remo/` | Public headers — the pipeline SDK (`ILodPipeline.h` is the contract) |
| `src/shell/` | Window, orbit camera, ImGui panel, pipeline registry |
| `src/cuda/` | NVRTC wrapper, CUDA context, GL interop |
| `src/io/` | Point cloud readers |
| `src/pipelines/` | Host side of each pipeline (flat, remolod, cudalod, simlod) |
| `kernels/shared/` | Rasteriser + allocators every pipeline includes — **shared, keep it that way** |
| `kernels/remolod/` | **RemoLOD's device code.** Yours to change. Forked octree + the accumulator |
| `kernels/{simlod,cudalod}/` | Comparison baselines. Vendored byte-identical — **do not edit** |
| `kernels/CudaPrint/` | Vendored from SimLOD, at this path so its `../CudaPrint/` include resolves unchanged |
| `kernels/flat/` | The no-LOD control |
| `bench/check_vendored.sh` | Asserts the baselines still match their submodules |
| `bench/reference/` | Upstream baseline numbers the ports are validated against |
| `plans/` | Research direction and staged implementation plans |
| `references/` | Source papers |
| `external/`, `patches/` | Submodules and the Linux/CUDA-13 fixes applied to them |

## Rules for vendored code

- Vendored device code is **byte-identical** to the submodule, and `make check-vendored`
  enforces it. A vendored file is a **5-line provenance banner** (upstream path, commit,
  copyright, a pointer to the sidecar, blank) followed by the upstream file, unchanged.
- **Explanatory notes go in `kernels/simlod/VENDORED.md`, not in the file.** That is what
  keeps "is this still upstream?" a diff rather than a judgement call.
- **Never edit a baseline to make RemoLOD work.** Fork the file into `kernels/remolod/` and
  change the fork. RemoLOD already diverges on two constants this way — see
  `kernels/remolod/remolod_structures.cuh`.
- Prefer a **separate pass** over a hook, always. It is what makes the fork's diff reviewable,
  and in the accumulator's case it was also faster.
- **Licence trap:** one upstream file is CC BY-NC-SA and is deliberately not used,
  because it would infect this project's licence. Check
  [THIRD_PARTY.md](THIRD_PARTY.md) before copying anything new out of `external/`.
- Kernels must be taken from the **patched** submodule tree — the Linux port fixes a
  `typedef char int8_t` that otherwise collides with `cuda/std/cstdint`.

## Constraints the detail-aware work runs into

These are load-bearing and were established by reading the code; check them before
proposing a design that assumes otherwise.

- **Per-node accumulators do not belong on `Node`.** `Node` is 152 bytes and its size is
  mirrored host-side as `kNodeBytes` to size the 200k-node pool. Widening it costs the render
  kernel's hot traversal its 8-nodes-per-cache-line layout, for data that traversal never
  reads. Use a **side array indexed by node index** — `NodeAccum`, see `remo/RemoAccum.h`.
- **Anything reading the tree can be a separate launch.** The accumulator was originally a
  hook inside `kernel_construct`; it is now `kernel_accumulate`, which runs after construct
  and reads the finished tree. Both `Node::numPoints` and the chunk list are stable at that
  point, and a per-node watermark makes the pass incremental, idempotent and interruptible.
  Reach for this shape before reaching for a hook.
- **Colour averaging has a memory wall.** The occupancy grid is 1 bit per cell (256 KB
  per inner node at 128³). An RGBA-sum grid at the same resolution would be ~16 MB
  *per node*. Filtering therefore needs a sparse accumulator keyed off the voxel
  backlog, not a dense per-cell one — or it pays CudaLOD's cost (3.2× device memory,
  ~13× voxelisation time).
- **Collapsing needs a grid pool first.** Point/voxel chunks are already recycled
  through `chunkQueue` when a leaf splits, but the 256 KB `OccupancyGrid` allocated on
  split is never freed. There is nothing to return it to.
- **The node pool has no device-side capacity check.** `numNodes` is a bump index grown
  by `atomicAdd(&stats->numNodes, 8)`; the host clamping in `readStats` is the only place
  exhaustion is noticed. Anything that splits more eagerly must keep that reporting intact.
- **SimLOD caps at 50M points, on purpose.** `kernel_construct` addresses batch N at
  `(N % BATCH_STREAM_SIZE)` with `BATCH_STREAM_SIZE == 50`, and RemoBench feeds it a resident
  cloud with no wrapping, so past 50 batches it re-reads slot 0 and builds from the wrong
  points — silently, since nothing faults. `PipelineRegistry::unsupportedReason` refuses such
  clouds for `simlod`. RemoLOD raises the constant in its own fork and takes them. Lifting it
  for `simlod` means a genuinely wrapping ring in `PointSource`, which is the listed loader
  work — **not** another edit to `structures.cuh`, which is how it was "fixed" before.
- **Uniform control flow around `RemoAllocator`.** It is deliberately non-atomic: every
  thread walks the identical allocation sequence. No `alloc()` behind a branch, in a
  data-dependent condition, or in a loop with a varying trip count. See the banner in
  `kernels/shared/remo_alloc.cuh`.
- **The root cube assumes translated `boxMin` is exactly the origin**, and nothing on the
  device checks it. `CloudMeta::translation` is therefore exactly `-boxMinOrig`; the
  coarser rule it used to be (snap down to a power of two) left a 1.3 km UTM cloud sitting
  169 km off origin, which sized the root cube at 170424 units and collapsed 36M points
  into 29 nodes with 7.04M in one leaf. Any new reader must land every point in
  `[0, boxSize]`; nothing on the device will complain if it doesn't, because CudaLOD
  clamps the cell index and SimLOD's float→uint32 conversion saturates, so out-of-box
  points pile into cell 0 instead of faulting. `loadLasCloud` therefore checks the
  observed bounds itself and re-reads against a corrected box.

## Reporting numbers

- **Always name CudaLOD's sampling strategy when quoting a throughput figure.** The tree
  is bit-identical across all four, but `WEIGHTED_NEIGHBORHOOD` voxelizes ~13× slower
  than `FIRST_COME` and needs 3.2× the device memory. An unqualified "CudaLOD does X
  MP/s" is meaningless. The same will be true of any SimLOD number without the batch
  size and the device-side time budget.
- **The current pipeline comparison is not a clean quality A/B.** Both selection passes
  take the shared pixel budget, but SimLOD's projects all eight corners and takes the
  screen AABB while CudaLOD's estimates from the node centre, so they do not interpret
  it identically. Do not present those numbers as a quality result.
- The native metrics (`REMO_LOD_SIMLOD_NATIVE`, `REMO_LOD_CUDALOD_NATIVE`) exist in the
  kernels for validating a port against its published behaviour, but no pipeline
  populates `KernelProgramDesc::defines`, so that path is currently unreachable from the
  host.
- **Name whether the accumulator was on** for any RemoLOD construct-time figure. It adds an
  unconditional launch — ~2.5 ms per construct launch against ~10 ms for construct itself on
  morro_bay 36M — and `--remolod-no-accum` is how you difference the two.
- **RemoLOD and SimLOD currently build identical trees** (4,137 nodes / 12,742,751 voxels on
  morro_bay 36M), because RemoLOD's forked octree kernel is still line-for-line SimLOD's apart
  from two constants. That is the intended starting point: while it holds, any difference
  between the two pipelines is attributable to the passes around construction. When Refinement
  starts changing the tree, this stops being true and the fork's diff is what explains why.
- Recapture `bench/reference/` after a submodule bump or driver change.

## Known traps

- **CudaLOD faults on `--synthetic`** with `CUDA_ERROR_ILLEGAL_ADDRESS` in the split
  kernel, at every point count tried. Not root-caused; coplanarity and bbox-boundary
  causes ruled out. The pipeline is refused for that fixture on purpose. Use a real cloud.
- A device fault is unrecoverable, so RemoBench exits immediately naming the kernel rather
  than continuing. Continuing previously produced a cascade of errors, then host heap
  corruption, then a SIGSEGV in a file-watcher thread — a trail pointing nowhere near the
  cause. Keep that fail-fast behaviour.
- A GUI-only code path is an untested code path. `--switch-to` / `--switch-after`,
  `--dump-frame` and `--remolod-no-accum` exist so those paths are scriptable; keep new
  controls reachable from the command line, and give a pipeline-specific readout a
  `diagnostics()` line so `--dump-frame` prints it. **CudaLOD's sampling strategy is still
  GUI-only**, which is why only strategy 0 of the `bench/reference/` oracle can be checked
  from a script; `--bench` (Stage 2 of `plans/02_ProfilingTools.md`) has to reach it.
- **A kernel signature is not covered by any check.** `--check-kernels` compiles, it does not
  launch, and the driver only rejects an argument-count mismatch at `cuLaunchKernel`. Passing
  kernel arguments as one shared struct (see `remo::AccumArgs`) makes the whole class of
  mistake unrepresentable; prefer that to a long parameter list for anything new.

## Timing

Stage 1 of `plans/02_ProfilingTools.md` has landed, so timing is now shell-owned:

- **Every kernel launch goes inside a `GpuScope`** taken from `FrameContext::profiler`.
  A launch outside one does not appear in any total. `ILodPipeline::timingScopes()`
  declares the names, and a pipeline that adds a phase without adding it there silently
  stops being counted — that is the one way back into the defect this layer removed.
- **A scope with no samples reports as absent, not `0.00`.** `GpuProfiler::find()`
  returns null, `timingRow()` prints "not measured", and `BuildTotals::measured` says so.
  Do not reintroduce a code path that formats an unmeasured scope as a number.
- **Strict and deferred samples are never pooled.** `--strict-timing` synchronises and
  attributes a sample to the frame that produced it; the default reads whenever
  `cuEventQuery` says ready. Both regimes now produce render times for all three
  pipelines. Name the regime whenever quoting a number.
- Scope names are the data format (`simlod.construct`, `cudalod.voxelize`, …). Renaming
  one breaks comparison against runs already captured.

## Deeper context — read when relevant

- `plans/01_NoveltyAssessment.md` — the research thesis: content-adaptive per-node point
  budgets computed online during streaming LOD construction, the prior art it must be
  differentiated from (Sequential Point Trees, VoxelMap, Pauly et al.), and the
  evaluation plan. **Read before any work on LOD selection metrics, node budgets, or
  split criteria.**
- `plans/02_ProfilingTools.md` — staged plan for the measurement layer, with §7
  recording what Stage 1 actually measured. **Read before touching timing, benchmarking
  or `GpuProfiler`.** Stage 2 (`--bench`, NDJSON) and Stage 3 (`DeviceTimeline` under
  `-DREMO_PROFILE`) are still open.
- `bench/reference/README.md` — capture methodology and the machine baselines were taken on.
- `references/` — source papers (SimLOD, CudaLOD).
