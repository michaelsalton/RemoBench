# Reference baselines

Upstream SimLOD / CudaLOD numbers captured on this machine, via `make simlod` and
`make cudalod`. These are the oracle RemoBench's ported pipelines are validated
against — not legacy cruft. Recapture after any submodule bump or driver change.

## Machine

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5080, 16303 MiB, compute capability 12.0 (`sm_120`), 84 SMs |
| Display | 3840x1600 (matters — see the render-buffer note below) |
| CUDA | 13.1 |
| Compiler | g++ 15.2.0, CMake 4.2.3 |
| Dataset | `data/morro_bay_35M/morro_bay_36M.las`, 36,200,706 points |

## CudaLOD — `cudalod_35M.txt`

Structural output, identical across every run and every sampling strategy:

| | |
|---|---|
| `#points` | 36,200,706 (exactly matches the LAS header) |
| `#voxels` | 12,742,112 |
| `#nodes` | 2,252 |
| `#allocated (splitting)` | 929,611,360 |
| `#allocated (voxelization)` | 1,402,630,112 |
| points/node min-avg-max | 1 - 20,864 - 49,739 |
| voxels/node min-avg-max | 1,046 - 24,646 - 56,067 |

Timings per sampling strategy. The app builds once in the ctor with strategy 0,
then rebuilds each time a strategy button is pressed:

| strategy | split | voxelize | total | MP/s | slab watermark |
|---|---|---|---|---|---|
| 0 `FIRST_COME` (ctor, cold) | 6.6 ms | 4.7 ms | 11.3 ms | 3,210 | 1.20 GB |
| 0 `FIRST_COME` (warm) | 5.1 ms | 4.1 ms | 9.3 ms | 3,912 | 1.20 GB |
| 2 `AVERAGE_SINGLECELL` | 4.9 ms | 20.1 ms | 25.0 ms | 1,448 | 3.82 GB |
| 3 `WEIGHTED_NEIGHBORHOOD` | 4.9 ms | 60.9 ms | 65.8 ms | 550 | 3.82 GB |

Strategy 1 (`RANDOM`) was not exercised in this capture.

Two things worth carrying into RemoBench's own benchmarking:

- **The strategies are not close.** `WEIGHTED_NEIGHBORHOOD` — the paper's quality
  contribution — voxelizes 13x slower than `FIRST_COME` (60.9 ms vs 4.7 ms) for
  bit-identical tree structure. Any "CudaLOD does N MP/s" claim is meaningless
  without naming the strategy. The README's previously recorded 3,332 MP/s was
  strategy 0.
- **Split cost is strategy-independent** (~5 ms) and the *cold* ctor build is
  ~25% slower than a warm rebuild. Measure warm, or measure both deliberately.

### Three upstream defects fixed to get this capture

The previously recorded baseline was produced with a silently misconfigured
build, and the render phase crashed. All three fixes are in
`patches/cudalod-linux-port.patch`:

1. **`MAX_BUFFER_SIZE` was shadowed.** `sampling_cuda/sampling_cuda.h`
   unconditionally `#define`d it to 2,147,483,647 and `SimLOD.h` included that
   *before* the `#ifndef`-guarded default, so the live pipeline ran with a
   2.147 GB slab regardless of intent. That dead include is now removed
   (`simlod_gentree_cuda::VoxelTreeGen` is only referenced from commented-out
   code).
2. **The override never worked.** `cmake -DMAX_BUFFER_SIZE=...` only sets a cache
   variable, so the documented knob was a no-op. It is now forwarded via
   `target_compile_definitions`, and renamed to `CUDALOD_MAX_BUFFER_SIZE` —
   `MAX_BUFFER_SIZE` is *also* a member of `ProgressiveFileBuffer`, so defining
   that name on the compiler command line breaks the build outright.
3. **Two unbounded-allocator overruns**, both invisible because the bump
   allocators in `lib.h.cu` have no capacity check — the symptom is a flood of
   `illegal memory access` from the render kernel, never an allocation failure:
   - The slab default is now **8 GB**, sized from the worst strategy's measured
     3.82 GB watermark rather than the default strategy's 1.20 GB. At 4 GB,
     pressing the strategy-2 button overran it.
   - `ptr_render_buffer` was a fixed 100 MB, but `renderHQS` allocates
     **28 bytes/pixel**. That is 58 MB at the author's 1920x1080 and fits; at
     3840x1600 it is 172 MB and does not. It is now sized from the actual
     framebuffer at 64 bytes/pixel.

With those, the run is clean: **0** illegal-access errors across all four builds.

## RemoBench's port vs the reference

`remobench --pipeline cudalod --open data/morro_bay_35M/morro_bay_36M.simlod --dump-frame x.ppm`

| metric | reference | RemoBench port | |
| ------ | --------- | ------------ | --- |
| points | 36,200,706 | 36,200,706 | exact |
| nodes | 2,252 | 2,252 (517 inner, 1,735 leaves) | exact |
| max points/node | 49,739 | 49,739 | exact |
| watermark, split | 929,611,360 | 929,611,360 | exact |
| watermark, voxelize | 1,402,630,112 | 1,402,6xx,xxx | exact to 3 s.f. |
| build, strategy 0 | 9.3 – 11.3 ms | 9.66 ms | within range |
| voxels | 12,742,112 | 12,742,500 | **+388 (+0.003%)** |

The port is deterministic (identical across runs), and the device code is vendored
unmodified, so the tree really is the same tree.

### The voxel delta is NOT bounding-box provenance — that was tested and refuted

This entry used to explain the +388 as an f32-vs-f64 bounding-box difference: the
reference reads the `.las`, whose header carries an f64 extent of `1399.9900000002235`,
while RemoBench read the `.simlod`, whose header is f32 (`1399.989990234375`); CudaLOD
derives `cubeSize` from the longest axis, so a last-bit difference would shift the 128³
grid's cell boundaries. It was recorded as needing the `.las` reader to confirm.

The reader landed. It does not confirm it:

| input | voxels | nodes | max points/node |
| ----- | ------ | ----- | --------------- |
| `morro_bay_36M.simlod` | 12,742,500 | 2,252 | 49,739 |
| `morro_bay_36M.las` | 12,742,500 | 2,252 | 49,739 |
| `morro_bay_36M.laz` | 12,742,500 | 2,252 | 49,739 |

All three are bit-identical, and the `--dump-frame` PPMs are byte-identical, at 36M and at
350M points. Reading the very file the reference read changes nothing, so the box cannot be
what differs.

The mechanism is now clear, and it is upstream's own: `Metadata::max_x` is a **`float`** in
`kernels/cudalod/common.h` (vendored unmodified, so the reference has the same field).
`1399.9900000002235` and `1399.989990234375` narrow to the *same* f32 before any kernel
sees either one. The f64 header extent never reaches the grid.

So the +388 (0.003%) is still open, and the remaining candidates are all downstream of the
box: `FIRST_COME` sampling resolving ties differently under a different thread schedule,
and the same unchecked capacities in `split_countsort_blockwise` that
`makeSyntheticSource` documents. Whatever it is, it is not the input.

The general lesson survives, just not the specific claim: bounding-box provenance *is* part
of the input, which is why `CloudMeta` keeps the box and the translation in f64. The
correction is that keeping f64 on the host buys nothing once the value crosses into an f32
device struct — check the width of the field the number actually lands in before attributing
a difference to precision.

## SimLOD

Not yet captured — SimLOD accepts no command-line arguments and loads only via
drag-and-drop onto its window, so it cannot be driven from a script. Run
`make simlod`, drop `data/morro_bay_35M/morro_bay_36M.simlod` on the window, and
save the stats panel output to `simlod_35M.txt`.

## RemoBench's own streaming captures — 350M

Not an upstream oracle: these are RemoBench against itself, and they exist because
`data/morro_bay_350M/` could not be run at all until `PointSource` grew a wrapping ring
(`plans/06_ComparisonFixes.md`). They are the check that the ring fills the right slots —
`simlod` and `remolod` must agree, and so must every reader.

**The budget is part of the capture, not context.** A progressive pipeline stops on
`memCapacityReached` at whatever fraction of the cloud the budget holds, so an unpinned
run truncates at a different point count every time, purely from free-VRAM drift. Two
unpinned 350M runs an hour apart here differed by a whole batch. Always `--device-budget`.

```sh
./build/remobench --open data/morro_bay_350M/morro_bay_350M.simlod \
    --pipeline remolod --device-budget 6G --dump-frame /tmp/x.ppm --dump-after 700
```

At `--device-budget 6G`, 350,360,028 points:

| pipeline | input | points ingested | voxels | nodes (inner / leaves) |
|---|---|---|---|---|
| `remolod` | `.simlod` | 169,000,000 | 59,548,545 | 20,633 (2,579 / 18,054) |
| `remolod` | `.las` | 169,000,000 | 59,548,545 | 20,633 (2,579 / 18,054) |

The cloud is truncated on purpose: 10.5 GB of budget would hold all of it, against 9.4 GB
of VRAM free on this machine with a desktop up. `memCapacityReached` is set and the tree is
valid — smaller, not wrong.

`.laz` is absent because it stays whole-cloud resident (5.6 GB of input on top of the
tree), so it does not fit a 6 GB budget at this size. The three-reader invariant is
therefore checked at 36M, where all three give 4,137 nodes / 12,742,751 voxels.

### The completion prediction is itself a check

The dump prints observed ingest beside the prediction from the 26 B/pt floor
(SimLOD Table 5: 9.1 GB / 350.36M; RemoBench's own 36M tree: 25.7 B/pt). Agreement so far:

| budget | predicted | observed |
|---|---|---|
| 6.00 GB (pinned) | 48.4% | 48.2% |
| 7.05 GB (free VRAM) | 60.2% | 59.7% |
| 7.07 GB (free VRAM) | 60.3% | 59.7% |

Within a point throughout, which is what makes 26 B/pt safe to quote. A large disagreement
would mean the coefficient is wrong for this build and should be re-derived before it is
used anywhere else.

### Wrapping, verified directly

51 batches into a 50-slot ring wraps exactly once — batch 50 lands in slot 0 on top of
batch 0. Built from the first 51M points of the 350M cloud:

| ring depth | wraps | points | voxels | nodes |
|---|---|---|---|---|
| 50 slots (`BATCH_STREAM_SIZE`) | yes, once | 51,000,000 | 16,938,284 | 5,665 |
| 51 slots (temporary `BATCH_STREAM_SIZE = 64`) | no | 51,000,000 | 16,938,284 | 5,665 |

Identical, so the slot mapping is right. This is worth redoing after any change to
`CloudSource` or to `BATCH_STREAM_SIZE`, because the failure mode is silent: a mis-mapped
ring builds a tree from the wrong points without faulting, without an allocation error, and
with plausible node counts.
