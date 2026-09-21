# RemoBench

A real-time LOD generation and rendering program for point clouds, aimed at lidar
data. It is a research project: it extends [SimLOD](https://github.com/m-schuetz/SimLOD)
and draws on [CudaLOD](https://github.com/m-schuetz/CudaLOD).

**The goal is to make SimLOD's octree construction detail-aware** — to spend samples
and depth where the geometry warrants it, and to stay coarse where it does not, using
geometry analysis computed *online, while the cloud is still streaming in*.

## The idea

SimLOD builds an octree incrementally on the GPU and renders intermediate results in
the same frame, at up to hundreds of millions of points per second. It gets there by
making the coarse levels as cheap as possible: inner-node samples come from a uniform
128³ occupancy grid, one bit per cell, and a voxel simply takes the colour of **the
first point that lands in its cell**. That is not Poisson-disk, not averaging, not
error-driven — it is uniform bucketing with an arbitrary representative. The authors
list colour filtering and out-of-core processing as future work, and never claim
content-awareness at all.

So there is room above it. The interesting constraint is *causality*: a point is
inserted, and its voxel created, long before that cell's local neighbourhood has
arrived. Any curvature or planarity measure needs a neighbourhood. Computing one
offline is decades old (Sequential Point Trees, 2003; Pauly et al., 2002); computing
one **incrementally, under a streaming constraint, on the GPU, to drive a rendering
LOD budget** is the part that appears unoccupied. See
[plans/01_NoveltyAssessment.md](plans/01_NoveltyAssessment.md) for the prior-art
survey and the evaluation plan it implies.

The way out is that the useful statistics are *additive*. A running per-node
covariance (Σxxᵀ, Σx, count) needs no neighbour search and no second pass — only
atomics into per-node accumulators — and eigenvalues of that give surface variation
and planarity directly. This is the mechanism LiDAR-SLAM voxel mapping already proves
works under streaming ingestion; it has just never been pointed at rendering LOD.

## Geometry metrics under consideration

| metric | basis | streaming-friendly |
| --- | --- | --- |
| Surface variation λ₀/(λ₀+λ₁+λ₂) | covariance eigenvalues | yes — additive accumulators |
| Plane residual | covariance / fitted plane | yes |
| Dimensionality features (linearity / planarity / scattering) | covariance eigenvalues | yes |
| Normal | smallest eigenvector | yes |
| Points per voxel | occupancy counting | yes |
| Colour variance | additive sums | yes |

All six are per-node reductions over points that have already been inserted, which is
what keeps them compatible with a progressive builder.

## The four pipelines

RemoBench is the harness; the LOD algorithm is a swappable plugin. Four pipelines, all
switchable at runtime:

| id | role |
| --- | --- |
| `remolod` | **RemoLOD — this project's own pipeline.** The detail-aware work lives here |
| `simlod` | external comparison baseline, progressive build |
| `cudalod` | external comparison baseline, batch build |
| `flat` | the control condition: no LOD, every point drawn — and the image-quality ground truth |

**RemoLOD may pull anything it needs out of the comparison pipelines, but never changes
them.** They are only worth having while they still reproduce their published numbers against
[bench/reference/](bench/reference/README.md), so RemoLOD *forks* what it needs into
`kernels/remolod/` and changes the fork. `make check-vendored` asserts that every vendored
kernel is still byte-identical to its submodule.

RemoLOD's octree is currently line-for-line SimLOD's apart from two constants, which is the
intended starting point: while that holds, any difference between the two pipelines is
attributable to the passes around construction rather than to a divergence inside it.

## Kernel architecture

Five kernels. Three exist; the last two are the project.

| kernel | when | what it does |
| --- | --- | --- |
| **Rasterize** | every frame, unconditionally | frustum test; descend while a node projects larger than the pixel budget; one block per visible node, `atomicMin` splatting |
| **Update** | only when a batch completes | expand the octree (count → split → make room), voxel-sample the new points, acquire chunks, insert points and voxels |
| **Accumulate** | after each Update | fold each leaf's new points into per-node running sums — one block per dirty leaf, carrying a watermark so it only ever touches what changed |
| **Analysis** (new) | every frame, unconditionally, per node | advance the watermark; test closure; roll leaf accumulators up the tree and finalise; score = geometric score × screen coverage, and enqueue — rebuilt each frame |
| **Refinement** (new) | only if budget remains; may produce no changes | colour filtering; deepening (local split + redistribute); collapsing; compression, lagging closure by a widen margin |

**Accumulate is a separate launch, not a hook inside Update.** It reads the finished tree, so
it needs no access to the spill buffer, no batch index and no octree descent — the point is
already at its leaf. That is what lets the SimLOD baseline stay byte-identical, and it is
also simply cheaper: one uncontended write per leaf rather than atomics per point.

The split between Analysis and Refinement is deliberate. Analysis is cheap, unconditional
and side-effect-free on the tree; Refinement is the only thing that mutates it, and it is
the only thing that consumes budget. That makes "how much refinement did the system get"
a single knob — which is exactly the independent variable the evaluation needs.

## What this is meant to buy

- **Offline-quality LOD without giving up instant display.** SimLOD's selling point is
  that you see the cloud immediately. Any scheme that pays for quality with a
  pre-pass forfeits that, so refinement has to be incremental and interruptible.
- **Colour filtering.** Named as future work in the SimLOD paper. First-come sampling
  biases every voxel towards whichever scan arrived first.
- **Adaptive depth.** Detail where geometry warrants it, coarse where it does not.
- **A path to out-of-core rendering**, since closure and compression give nodes a
  well-defined "finished" state.

## Evaluation plan

- **Independent variable: how much refinement budget the system got.** Everything else
  — loader, camera, pixel budget, rasteriser — is held fixed by construction (see
  *The shared path* below), so a difference in the output is attributable to refinement.
- **Offline reference.** Build the tree offline with full knowledge of the dataset,
  applying the same criterion with no closure gate and no budget. That is the ceiling
  the real-time version is trying to approach, and the gap to it is the result.
- **Image quality.** Ground truth is a full-resolution render with no LOD at all — the
  `flat` pipeline. Comparing two approximations to each other and picking the prettier
  one is not a measurement.
- **Colour filtering.** The target is a scene with no visible scan seams. The test:
  process the same file twice with the point order reversed. First-come sampling gives
  two visibly different colourings; a filtered build should give the same one both times.

## The shared path

Everything outside LOD generation is shared and must stay shared: one loader, one
orbit camera, one software rasteriser (`kernels/shared/`), one pixel budget handed
unchanged to every pipeline, one device-memory budget computed once by the shell.
`flat` (no LOD) is the control condition and the image-quality ground truth.

This matters because the two reference implementations cannot be compared as they
ship. They are separate binaries with different loaders, different LOD metrics
(`dx > 2 * minNodeSize` in world units versus `cubeSize / distance < 1 - 0.97 * LOD`,
angular and not viewport-calibrated) and different rasterisers (a packed `uint64`
`atomicMin` framebuffer versus a `uint32` depth buffer plus a 16 byte/pixel
accumulator with a Gaussian 3×3 resolve). Feed those two the same octree and the
images still differ, for reasons that have nothing to do with LOD quality.

## Status

Working:

- Four pipelines, switchable at runtime: `flat` (no LOD, the control), `remolod` (this
  project's own), `cudalod` (batch: counting-sort split + voxelisation, four sampling
  strategies), `simlod` (progressive octree). All four go through the same `DrawList` seam
  and the same rasteriser.
- **The per-node accumulator.** `remolod_accum.cu` maintains fifteen running sums plus a
  count per node — the covariance and colour statistics the Analysis kernel will roll up.
  A separate launch over the finished tree, one block per dirty leaf, carrying a per-node
  watermark so it only ever folds points it has not seen. On `morro_bay_36M` it costs
  ~2.5 ms per construct launch against ~10 ms for construction itself, and its two
  invariants are printed by `--dump-frame`: `Σ leaf counts == numPoints` exactly
  (36,200,706), and no inner node holds sums.
- **Loading, all three formats.** `.simlod`, `.las` (eight loader threads, ~110–140 MP/s)
  and `.laz` (laszip, ~5 MP/s and sequential — see `src/io/LazReader.cpp`). The same cloud
  read through all three produces bit-identical trees (2,252 nodes, 12,742,500 voxels on
  `morro_bay_36M`), which is what says the parsers agree rather than merely all running.
  Compare structural counts rather than frames: only `flat` is run-to-run deterministic,
  because every octree pipeline colours a voxel from the first point to reach its cell and
  which *thread* wins is a scheduling accident. LAS/LAZ apply the
  translation in f64 inside the parse, so a UTM cloud keeps sub-millimetre resolution
  instead of the ~4 cm an f32 coordinate has at that magnitude.
- CUDA software rasteriser: packed `uint64` `atomicMin` framebuffer, EDL,
  `surf2Dwrite` into a GL texture registered once per resize.
- Kernel hot reload: edit a `.cu` or `.cuh`, save, and the next frame runs it. A broken
  kernel is non-fatal and leaves the previous version live.
- `--check-kernels` (headless compile+link of every declared program) and
  `--dump-frame` (headless frame capture) for verification without a human at a window.
- **Octree wireframe.** `--show-bounds` (panel: *node boxes*) draws a cube per node the
  selection pass emitted, coloured by level from the same hash colour-by-LOD gives the
  samples; `--hide-points` leaves the structure on its own. It renders the cut rather
  than inferring it from sample colours, and it makes the selection difference visible
  directly: SimLOD's disjoint frontier tiles, CudaLOD's parent-and-child visibility
  nests. Depth-tested through the shared framebuffer, so an edge behind a surface is
  hidden by it, and drawn after EDL so the overlay is not lit by its own edges.
- CudaLOD's port reproduces the upstream reference exactly on point, node and
  watermark counts — see [bench/reference/README.md](bench/reference/README.md).
- **GPU timing**, shell-owned. Every kernel launch is bracketed by a `GpuScope`, and the
  profiler keeps the distribution — median, p95, min/max, n — rather than a running sum.
  Strict and deferred samples are filed separately and never pooled, and a scope nobody
  measured reports as *not measured* instead of `0.00 ms`. `--dump-frame` prints the
  per-scope table. This reproduces the CudaLOD reference numbers for strategy 0 within
  ~2% (split 4.99 ms against 5.1, voxelize 4.08 against 4.1), which is the oracle that
  says the instrument is right.

Not there yet:

- **The Analysis and Refinement kernels.** This is the actual project.
- **The +388-voxel gap against the CudaLOD reference is still unexplained.** The `.las`
  reader was supposed to settle it, and it did settle what it is *not*: reading the very
  file the reference read gives 12,742,500 voxels, exactly what the `.simlod` gives, so
  the f32-vs-f64 bounding-box story recorded in
  [bench/reference/README.md](bench/reference/README.md) cannot be the cause —
  `Metadata::max_x` is a `float` in the vendored struct, which narrows both headers to the
  same value before any kernel sees them.
- **Parallel `.laz` decode.** laszip decodes sequentially here — ~70 s for the 350M cloud
  against 3.2 s for the same points as `.las`. Fanning out needs the chunk table read up
  front so a single-chunk file is not decoded N times over; laszip does not expose it and
  lazperf does, which is the concrete reason to make the swap THIRD_PARTY.md already
  contemplates.
- **No quality metric harness.** `--dump-frame` writes a PPM; there is no Chamfer,
  Hausdorff or PCQM comparison against `flat`.
- **Ingest is synchronous.** `PointSource` wraps now — `simlod` and `remolod` stream
  through a 50-slot ring and neither has a point ceiling any more — but the refill is a
  blocking `cuMemcpyHtoD` on the render thread, inside the serialisation every pipeline
  already does after construct. That is enough to make a cloud larger than the card
  correct; it is not the paper's loading/generation overlap, which needs a real copy
  stream (there is still no second `CUstream` in the repo). See the streaming-loader item
  below and `plans/06_ComparisonFixes.md` Stage 5.
- **No benchmark harness.** Stage 1 of
  [plans/02_ProfilingTools.md](plans/02_ProfilingTools.md) landed the instrument; there
  is still no `--bench` writing an NDJSON time series over a deterministic camera orbit
  (Stage 2), and no intra-kernel phase attribution — SimLOD's `expand` / `createVoxels`
  / `insertPoints` marks are still computed on-device every launch and thrown away by a
  no-op `CudaPrint` (Stage 3). CudaLOD's sampling strategy is also still GUI-only, so
  only one row of the reference oracle can be checked from a script.
- **The two LOD metrics are not yet interchangeable.** Both selection passes accept the
  shared pixel budget, but SimLOD's projects all eight corners and takes the screen
  AABB while CudaLOD's estimates from the node centre, so they do not interpret the
  budget identically. The native metrics are still selectable in the kernels
  (`REMO_LOD_SIMLOD_NATIVE`, `REMO_LOD_CUDALOD_NATIVE`) but nothing on the host passes
  them yet.
- The overlapped streaming loader. The ring exists and wraps, so construction is
  genuinely progressive over a cloud that never has to fit on the device — but the refill
  is synchronous, so loading and generation still do not overlap and the paper's
  throughput claim is not reproduced. `.laz` is not on this path at all: laszip decodes
  strictly sequentially with no seek, so a compressed cloud stays whole-cloud resident.

## Requirements

- A C++23 compiler (`g++`; developed against 15.2)
- CMake ≥ 3.22, GNU Make
- A CUDA toolkit providing the driver API, NVRTC and nvJitLink (developed against
  13.1). **No `.cu` is compiled at build time** — every kernel is compiled at runtime
  by NVRTC, which is what makes hot reload possible.
- GL development headers **including GLU**:
  ```sh
  sudo apt install cmake cuda-toolkit-13-1 libglu1-mesa-dev
  ```
  `libglu1-mesa-dev` is separate from `libgl1-mesa-dev` and easy to miss — the
  vendored `glew.h` includes `GL/glu.h` unconditionally.

## Building and running

```sh
make                              # release -> build/remobench
make debug                        # -O0 -g  -> build-debug/
make check                        # check-vendored + check-kernels
make run ARGS="--open data/morro_bay_35M/morro_bay_36M.simlod"
make clean
```

`make` is a thin façade over CMake. `remobench --help` lists the options; point clouds
can also be dropped on the window, but `--open` exists because a benchmark runner
cannot drag a file.

**A successful `make` does not mean the kernels compile.** They are NVRTC-compiled at
runtime, so kernel errors are invisible to the build. After touching anything under
`kernels/`:

```sh
make check                          # both checks below
./build/remobench --check-kernels   # no display, GPU context or point cloud needed
./bench/check_vendored.sh           # no GPU and no build needed either
```

And note that **compiling is not launching**: `--check-kernels` links every program but
launches none, so it cannot see a host/device signature mismatch. A kernel whose parameter
list changed is only verified by running the pipeline and reading the structural counts.

The reference implementations can be built and run for comparison:

```sh
make simlod                                     # patches, builds and runs upstream SimLOD
make cudalod CUDALOD_LAS=/path/to/cloud.las     # same for CudaLOD
```

## Layout

```
.
├── include/remo/    Public headers = the pipeline SDK (ILodPipeline.h is the contract)
├── src/
│   ├── shell/       Window, orbit camera, GUI, pipeline registry
│   ├── cuda/        NVRTC wrapper, CUDA context, GL interop
│   ├── io/          Point cloud readers
│   └── pipelines/   Host side of each pipeline
├── kernels/         Device code, NVRTC-compiled at runtime and hot-reloaded
│   ├── shared/      Rasteriser + allocators every pipeline #includes — keep it shared
│   ├── flat/        The no-LOD control pipeline
│   ├── remolod/     RemoLOD — ours to change: forked octree, accumulator, selection
│   ├── cudalod/     Comparison baseline, batch (vendored + our selection pass)
│   ├── simlod/      Comparison baseline, progressive (vendored + our selection pass)
│   └── CudaPrint/   Vendored from SimLOD; at this path so its include resolves unchanged
├── bench/reference/ Upstream baselines the ports are validated against
├── plans/           Research direction and staged implementation plans
├── references/      Source papers
├── external/        SimLOD, CudaLOD, glfw (git submodules)
└── patches/         Linux / CUDA-13 fixes applied to the submodules at build time
```

`data/` holds point clouds and is gitignored; do not assume a cloud is present.

## Provenance

RemoBench vendors code from both reference implementations, which are MIT. Every copied
file carries a three-line header naming its upstream path, commit and copyright. See
[THIRD_PARTY.md](THIRD_PARTY.md) for the full record — including one file that is
deliberately **not** used, because it is CC BY-NC-SA and would infect this project's
licence.

Clone with submodules:

```sh
git clone --recurse-submodules <url>
# or, in an existing checkout:
git submodule update --init --recursive
```

| path | upstream | pin |
| --- | --- | --- |
| `external/SimLOD` | https://github.com/m-schuetz/SimLOD | branch `ubuntu` |
| `external/CudaLOD` | https://github.com/m-schuetz/CudaLOD | branch `main` |
| `external/glfw` | https://github.com/glfw/glfw | tag `3.4` |

`remobench` references `external/*/libs/**` only, and never a file that `patches/*.patch`
modifies — so it builds after a bare `git submodule update --init`, whether or not the
submodule patches have been applied.