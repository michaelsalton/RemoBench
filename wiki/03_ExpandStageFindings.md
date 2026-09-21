# Where the update kernel's time goes

What `kernel_construct` actually spends its time on, measured rather than assumed.
[02_KernelPipeline.md](02_KernelPipeline.md) describes what the Update kernel *does*;
this page says what it *costs*, and is the evidence behind the gate in
[plans/05_HardCodedTest.md](../plans/05_HardCodedTest.md).

The question it was built to answer: **if the octree depth were known in advance, so
the iterative count-then-split loop could be skipped, how much would that save?** The
answer is 21%, not the 46% that a first look at `expand` suggests, and the difference
between those two numbers is the point of this page.

---

## 1. The headline

Measured on morro_bay 36M, RTX 5080, strict timing regime, accumulator off, five runs.
Percentages are of the sum of all phases.

| phase | mean ms | mean % | range |
|---|---:|---:|---|
| **expand** | 28.37 | **46.3%** | 41.8–51.0% |
| insertPoints | 12.47 | 20.6% | 19.4–23.6% |
| voxelSampling | 11.80 | 19.4% | 17.3–22.2% |
| insertVoxels | 5.52 | 9.1% | 6.4–12.2% |
| allocPointChunks | 1.36 | 2.3% | 1.9–3.2% |
| allocVoxelChunks | 1.29 | 2.2% | 1.2–6.2% |
| batchBegin | 0.04 | 0.1% | 0.1–0.1% |

Phase sum: 60.85 ms mean, over 37 batches.

## 2. `expand` is 46%, but only 21% of it is reachable

This is the distinction that matters, and the one that is easy to get wrong.
`expand` has three parts, and skipping the iterative deepening only removes one:

| part of `expand` | mean ms | mean % of construct | range | removed by knowing the depth? |
|---|---:|---:|---|---|
| counting, iteration 0 | 12.36 | 20.1% | 15.3–26.6% | **no** — you still must count once to place points |
| **counting, iterations 1..N** | **13.04** | **21.5%** | **16.4–24.4%** | **yes** — this is the re-counting |
| splitting | 2.97 | 4.8% | 3.0–7.2% | no — still happens, once instead of iteratively |
| = all of `expand` | 28.37 | 46.3% | 41.8–51.0% | |

The three parts account for 100.0% of `expand`; nothing is unattributed.

So the two numbers answer two different questions:

- *How much of the update kernel is the expand stage?* → **46%**
- *How much could a predicted depth actually save?* → **21%**

Quoting 46% as the saving would overstate it by more than double.

## 3. The stable number: 3.46 iterations per batch

The millisecond splits move between runs (§6). The structural counters do not — they
came out **identical in all five runs**:

| | |
|---|---|
| batches | 37 |
| **expand iterations / batch** | **3.46** |
| spilled points / batch | 233,070 |
| nodes split / batch | 14.0 |

3.46 iterations per batch means **71% of all counting passes are re-counts**. Each one
re-descends every point in the batch from the root, plus the spill — and the spill
averages 233k points, 23% of a 1M-point batch. That ratio is a property of the tree and
the 50k-point split threshold, not of the clock, which is why it reproduces exactly.

This is the number to reason from. The timing percentages are its consequence.

## 4. The part predicted depth does not touch

**`voxelSampling` + `insertPoints` + `insertVoxels` are 49.1% of construct** — more than
`expand`. All three are per-point root-to-leaf descents, and none of them goes away when
the depth is known in advance:

- `voxelSampling` (`remolod_octree.cu:393`) descends every point through every ancestor,
  writing into each inner node's 128³ occupancy grid. Those voxels *are* the LOD.
- `insertPoints` / `insertVoxels` descend again to place points and drain the voxel
  backlog.

If the goal is construct throughput rather than a density-only baseline arm, the descent
itself is the larger target. Skipping the expand loop addresses the smaller half.

## 5. How it is measured

`GpuScope` cannot see inside `kernel_construct`. `cuEventRecord` is stream-ordered and
construct is a **single cooperative launch** holding up to 20 batches of six phases each;
`ncu` cannot break it down either, because kernel replay does not support grid-wide sync.
Device-side timing written into a readback buffer is the only option. This is Layer 3 of
[plans/02_ProfilingTools.md](../plans/02_ProfilingTools.md), built for RemoLOD.

**The marks already existed and were being discarded.** `addBatch` had always computed
eight `nanotime()` deltas and passed them to `cudaprint->print(...)`. `CudaPrint::print()`
is `return;` on its first line (`kernels/CudaPrint/CudaPrint.cuh:107`) and the host
allocates a 1024-byte dummy, so every one of those numbers was computed and thrown away
on every launch since the port. The work was giving existing marks a sink.

| piece | where |
|---|---|
| `REMO_MARK`, compiled out unless `REMO_PROFILE` | `kernels/shared/remo_prelude.cuh` |
| `DeviceTimeline`, `ConstructPhase` | `include/remo/HostDeviceCommon.h` |
| marks + per-iteration counters | `kernels/remolod/remolod_octree.cu` |
| `GpuProfiler::addSample()` | `src/shell/GpuProfiler.cpp` |
| readback, GUI table, diagnostics rows | `src/pipelines/RemolodPipeline.cpp` |

Three details that are load-bearing:

- **A mark is only a grid-wide phase boundary when it sits immediately after a
  `grid.sync()`.** All eight do. A mark that cannot be placed after a barrier must not be
  placed at all.
- **The clock read is guarded, not just the store.** `nanotime()` is `asm volatile`, so a
  read left outside `#ifdef REMO_PROFILE` survives optimisation and the default build
  would pay for a timestamp it never keeps.
- **Per-iteration costs use counters, not marks.** A mark pair per expand iteration would
  overflow `REMO_MAX_MARKS` at 20 batches × 20 iterations, so `expandIterNs[]` accumulates
  by iteration index instead. It covers counting only, which is what lets
  `expand − Σ expandIterNs` isolate splitting.

The phase scopes are deliberately **not** in `TimingScopes::build`: `buildTotals()` sums
every name there, so registering sub-phases would double-count them against
`remolod.construct` and corrupt the throughput readout.

### Reproducing

```sh
./build/remobench --pipeline remolod --strict-timing --remolod-no-accum \
  --remolod-phase-timings --open data/morro_bay_35M/morro_bay_36M.simlod \
  --dump-frame /tmp/x.ppm
```

Without `--remolod-phase-timings` the kernel compiles without the marks and the rows
report as absent, not as `0.00`. The flag selects the `-DREMO_PROFILE` variant through
`KernelProgramDesc::defines` — the **first host-side user of that field**, which CLAUDE.md
had listed as unreachable. Both variants cache side by side, since `defines` is part of
the compile cache key.

## 6. Why the timings have a range and the counts do not

`MAX_PROCESSING_TIME = 10.0f` caps each construct launch, so how many batches land inside
one launch shifts between runs, and GPU clocks drift. That moves the millisecond split
without changing the work done. Hence: report means over repeats, quote the range, and
prefer the structural counters (§3) when a single number is needed.

The same cap has a consequence for any future optimisation: **removing work from construct
does not shorten the launch, it ingests more batches per launch.** Quote results as MP/s,
never as construct ms.

## 7. The cross-check that validates all of it

**The phase sum is 97.7% of the `remolod.construct` CUevent total** across five runs
(94.8–99.3%).

These are two independent clocks — device `%globaltimer` marks and stream-ordered CUDA
events — and their agreement to within a few percent is
[plans/02](../plans/02_ProfilingTools.md)'s own acceptance test for this layer. The
missing ~2.3% is the kernel prologue and the closing stats reduction, which sit outside
`addBatch` and carry no marks.

A large gap here would mean marks misplaced relative to their barriers, not a slow phase.

## 8. What was verified alongside

Adding the timeline pointer changed `kernel_construct`'s signature — the exact move
CLAUDE.md records as having silently built no tree for a whole commit, while `make`
succeeded, `--check-kernels` passed and `--dump-frame` exited 0. `--check-kernels`
compiles and links; only `cuLaunchKernel` rejects an argument-count mismatch.

| check | result |
|---|---|
| structural counts, 36M, with and without the flag | **4,137 nodes / 12,742,751 voxels** — unchanged |
| `simlod` | 4,137 / 12,742,751 — still identical to RemoLOD |
| `cudalod` | 2,252 / 12,742,500 — matches `bench/reference/` |
| `make check-vendored` | 9 kernels byte-identical |
| `DeviceTimeline::overflow` | 0 in every run (160 marks used of 256) |

The structural counts are the only evidence the launch happened at all.

## 9. Caveats

- **36M only.** The 350M cloud did not run: `remolod` still declares
  `needsWholeCloudResident` and needs 15.3 GB (26.00 B/pt structure + 5.6 GB resident
  cloud + 0.59 GB fixed) against 16.585 GB of card, most of the free part of which was in
  use. That is a memory ceiling, not a speed one. `expand`'s share should *grow* with
  depth as the counting descent lengthens, so 46% is a floor for larger clouds rather
  than a ceiling — untested.
- **Measured while the `PointSource` wrapping-ring work was in flight**, with
  `kBatchStreamSize` back down to 50 and `SimlodPipeline` already off resident mode. The
  tree was still 4,137 / 12,742,751 throughout, which is that work's own correctness
  check.
- **RemoLOD's breakdown is SimLOD's breakdown**, for now: the forked octree kernel is
  line-for-line SimLOD's apart from constants. When Refinement starts changing the tree
  this stops being true, and these numbers stop describing `simlod`.
- Strict regime, accumulator off, throughout. Both must be named when quoting any of
  these figures.
