# Measuring `expand`'s Share of the Update Kernel

## TL;DR

- The original sketch proposed hard-coding octree depth from a point-density table,
  skipping the expand stage, and verifying against SimLOD's published throughput. It had
  no Amdahl bound, and its oracle was wrong on three counts. §1 records why.
- Stage 0, per the added note "before we do any of this we need to measure the expand
  stage's share of the Update kernel", now delivers **the bound**.
- **Measured: `expand` is 46% of construct. The deletable part — counting iterations
  2..N, which is what knowing the depth in advance removes — is 21.5%.** Five runs,
  morro_bay 36M. §8.
- **That clears the 20% bar, but only just** (range 16.4–24.4%). The work is worth doing
  as the density-only baseline arm `plans/01_NoveltyAssessment.md` requires the adaptive
  method to beat — not as a contribution in itself. §9.
- The mechanism was not new: `plans/02_ProfilingTools.md` §2 Layer 3 already specified
  `DeviceTimeline` + `REMO_MARK` + `REMO_PROFILE`. This is that stage, scoped to
  RemoLOD's construct kernel — the one file plan 02 forgot to name, and the only octree
  kernel we are allowed to edit.

## 1. Why the original sketch was replaced

The five bullets were: hard-code octree values; skip the expand stage when depth is
known in advance; drive depth from a density table (`< 1000/chunk` → 1 level, else 3);
expect it to be faster; verify against the SimLOD paper.

**Density-driven depth is what an octree already is.** `MAX_POINTS_PER_NODE = 50'000`
(`kernels/remolod/remolod_structures.cuh:12`) is a density rule: `expand` counts points
per leaf and splits whatever exceeds it. A density→depth table does not make the tree
detail-aware; it reaches approximately the same tree by a cruder route. The detail-aware
thesis is depth driven by *geometric complexity*, which is decoupled from density — a
dense flat wall should get *less* depth, not more.
`plans/01_NoveltyAssessment.md:60` names "a simple density heuristic captures most of
the benefit" as the reviewer objection that must be killed, and its Recommendation 2
asks for precisely this as the **baseline arm**. That is what this line of work is.

**No ceiling.** "This should make the Octree generation faster theoretically" had no
bound attached. If `expand` had been 12% of construct, the whole idea would be capped at
12%.

**The saving is not the whole of `expand`.** `expand`
(`kernels/remolod/remolod_octree.cu:331`) loops up to 20 times. Each iteration runs
`doCounting` → `countPoint` (`:121`), a full root-to-leaf descent of every point in the
batch plus every spilled point; then `doSplitting` (`:263`) zeroes a 256 KB occupancy
grid per split node. Knowing the depth in advance removes iterations 1..N; iteration 0's
counting is irreducible unless the depth predictor replaces it outright, and splitting
still has to happen, just once. So the measurement had to yield iteration counts, not
only milliseconds.

**The oracle was wrong.** SimLOD's 580 MP/s is an RTX 4090 + PCIe 5 SSD *end-to-end*
figure. This machine is a 5080; RemoBench loads morro_bay at ~86–140 MP/s, so an
end-to-end comparison measures `src/io/`; and construct is time-budgeted at
`MAX_PROCESSING_TIME = 10.0f` (`:27`), so its wall time is near-constant by design. The
correct oracle is same-machine and in-harness: RemoBench's own `simlod` pipeline, which
shares the loader, camera and pixel budget, and currently builds an identical tree.

## 2. What this plan does not do

- No hard-coded depth table, no density bands, no change to split criteria. **Nothing
  here changes the tree**, and the structural counts prove it (§10).
- No `--bench` / NDJSON. That is plan 02 Stage 2 and stays there.
- No instrumentation of `simlod`, `cudalod` or `flat`. RemoLOD's octree kernel is
  line-for-line SimLOD's apart from two constants, so **RemoLOD's breakdown is SimLOD's
  breakdown**. Zero vendored churn.

## 3. Mechanism

Event-based `GpuScope` cannot break this launch down: `cuEventRecord` is stream-ordered
and `kernel_construct` is a single cooperative launch. `ncu` cannot either — kernel
replay does not support grid-wide sync. Device-side timing into a readback buffer is the
only option, as plan 02 §2 Layer 3 concluded.

**The marks already existed and were being thrown away.** `addBatch` computed eight
`nanotime()` deltas and handed them to `cudaprint->print(...)`. `CudaPrint::print()` is
`return;` on its first line (`kernels/CudaPrint/CudaPrint.cuh:107`), the host allocates a
1024-byte dummy, and nothing reads it back. The work was to give existing marks a sink,
not to find phase boundaries. The `cudaprint->print(...)` call is left as it is.

**Compile-time guard.** `REMO_MARK` in `kernels/shared/remo_prelude.cuh` expands to
`((void)0)` unless `REMO_PROFILE` is defined. The clock reads are guarded too —
`nanotime()` is `asm volatile` and would not be optimised away, so an unguarded read
would make the default build pay for a timestamp it never stores. The profiling variant
is requested through `KernelProgramDesc::defines`, which is part of the compile cache
key, so both variants cache side by side and hot-reload independently. **This is the
first host-side user of `defines`**, which CLAUDE.md notes was unreachable.

**The pointer is passed in both variants.** Only the marks compile out, never the
parameter — one host launch path, one kernel signature.

## 4. Data layout

`include/remo/HostDeviceCommon.h`, next to `DeviceDiagnostics`, following the
`RemoAccum.h` precedent (POD, host+device shared, explicit size assert):

```cpp
constexpr uint32_t REMO_MAX_MARKS = 256;
constexpr uint32_t REMO_MAX_EXPAND_ITERS = 20;

enum ConstructPhase : uint32_t {
    kPhaseBatchBegin = 0, kPhaseExpand, kPhaseVoxelSampling,
    kPhaseAllocPointChunks, kPhaseAllocVoxelChunks,
    kPhaseInsertPoints, kPhaseInsertVoxels, kPhaseBatchEnd,
    kNumConstructPhases
};

struct DeviceTimeline {
    uint32_t numMarks;
    uint32_t overflow;
    TimelineMark marks[REMO_MAX_MARKS];   // {phase, pad, ns}

    uint32_t batches, expandIters, nodesSplit, pad1;
    uint64_t spilledPoints;
    uint64_t expandIterNs[REMO_MAX_EXPAND_ITERS];
};
static_assert(sizeof(DeviceTimeline) == 4288, ...);
```

A mark records the phase that **begins** at it, so consecutive marks bound one phase.
`expandIterNs` is the load-bearing field: it separates iteration 0 from 1..N without a
mark pair per iteration, which would overflow 256 marks at 20 batches × 20 iterations.
Measured usage is 8 marks × 20 batches = 160, and `overflow` stayed 0 in every run.

`expandIterNs` covers **counting only** — `doSplitting` is deliberately outside the
window, so `expand − Σ expandIterNs` isolates splitting.

Zeroed per launch by thread 0 in the `kernel_construct` prologue: the host reads the
timeline after every construct, and `remolod_reset.cu` runs only on cloud reset.

## 5. Reporting

Read back in `RemolodPipeline::readTimeline()`, called next to `readStats()` after the
existing `cuCtxSynchronize()`, so no new sync point. Phase durations are summed over the
batches in a launch and handed to `GpuProfiler::addSample()` — a new entry point,
because `cuEventRecord` cannot bracket an intra-kernel phase — as one sample per phase
per launch, the same granularity as the `remolod.construct` CUevent they are
cross-checked against. `ScopeStats`' median/p95/n machinery and the "not measured"
semantics then apply with no special cases.

Scope names, following plan 02's nested convention. **These are the data format;
renaming one breaks comparison against runs already captured:**

```
remolod.construct.expand          remolod.construct.allocPointChunks
remolod.construct.voxelSampling   remolod.construct.allocVoxelChunks
remolod.construct.insertPoints    remolod.construct.insertVoxels
```

**They are deliberately NOT in `TimingScopes::build`.** `buildTotals()` sums every name
there; adding sub-phases would double-count them against `remolod.construct` and corrupt
the headline throughput figure.

Surfaced in `guiStats()` (a `remolod_phases` table) and in `diagnostics()`, which
`--dump-frame` prints. The phase lines are emitted *before* `diagnostics()`' early return
for a disabled accumulator, because the protocol runs with `--remolod-no-accum`.

## 6. Deviation from the approved plan

The approved plan said "no new CLI flag; `-DREMO_PROFILE` rides
`KernelProgramDesc::defines`". That was underspecified: `defines` is the mechanism, but
something has to set it. **`--remolod-phase-timings` was added.** The alternative —
always compiling with `REMO_PROFILE` — would have abandoned the unchanged-default-build
property the guard exists to protect, and CLAUDE.md requires new controls to be reachable
from the command line.

## 7. Protocol

```sh
./build/remobench --pipeline remolod --strict-timing --remolod-no-accum \
  --remolod-phase-timings --open data/morro_bay_35M/morro_bay_36M.simlod \
  --dump-frame /tmp/x.ppm
```

Strict regime throughout; accumulator off, so its ~2.5 ms is not folded into construct.
Five repeats, because the per-launch time split is noisy even though the structural
counts are not (§8).

## 8. Results — morro_bay 36M, RTX 5080, strict regime, accumulator off

Five runs. Percentages are of the phase sum.

| phase | mean | min | max |
| --- | ---: | ---: | ---: |
| **expand** | **46.3%** | 41.8% | 51.0% |
| insertPoints | 20.6% | 19.4% | 23.6% |
| voxelSampling | 19.4% | 17.3% | 22.2% |
| insertVoxels | 9.1% | 6.4% | 12.2% |
| allocPointChunks | 2.3% | 1.9% | 3.2% |
| allocVoxelChunks | 2.2% | 1.2% | 6.2% |
| batchBegin | 0.1% | — | — |

`expand` decomposed, in ms:

| component | mean | min | max | removed by knowing the depth? |
| --- | ---: | ---: | ---: | --- |
| counting, iteration 0 | 12.36 | 8.17 | 17.79 | no — some counting is unavoidable |
| **counting, iterations 1..N** | **13.04** | 10.98 | 16.21 | **yes** |
| splitting | 2.97 | 1.86 | 4.80 | partly — still happens, just once |

**Iterations 1..N are 21.5% of construct** (range 16.4–24.4%).

Structural counters, **identical in every run** — properties of the tree, not of the
clock:

| | |
| --- | --- |
| batches | 37 |
| expand iterations / batch | **3.46** |
| spilled points / batch | 233,070 |
| nodes split / batch | 14.0 |

3.46 iterations per batch means **71% of all counting passes are re-counts**. Each one
re-descends the whole batch plus the spill, and the spill is 23% of a batch.

**Cross-check (plan 02 Stage 3's own acceptance test): the phase sum is 97.7% of the
`remolod.construct` CUevent total** (range 94.8–99.3%). The missing ~2.3% is the kernel
prologue and the closing stats reduction, which sit outside `addBatch` and carry no
marks. Two independent clocks agreeing to that margin is what validates the numbers.

### The 350M arm could not be run

`remolod` still declares `needsWholeCloudResident`, so it needs 15.3 GB for 350,360,028
points (26.00 B/pt structure + 5.6 GB resident cloud + 0.59 GB fixed). The card has
16.585 GB total, and `PipelineRegistry` refuses the cloud at every budget that fits.
**This is a memory ceiling, not a speed one — a faster `expand` does not reduce the
26 B/pt structure.**

This is a moving target, not a settled limit. The wrapping-ring work in `PointSource`
was in flight in the working tree when this was measured: `SimlodPipeline` had already
been moved off resident mode (`needsWholeCloudResident = false`) and
`kBatchStreamSize` had come back down from 8192 to upstream's 50. At that moment
neither pipeline took 350M — `simlod` refused on the 50-slot ring and `remolod` on
memory. **Re-run this arm once `remolod` is on the ring**; the numbers here are 36M
only.

The measured share is therefore from 36M only. `expand`'s share should *grow* with depth,
since the counting descent lengthens, so 46% is a floor for larger clouds rather than a
ceiling. That expectation is untested.

## 9. The gate

**D = 21.5%** — counting iterations 1..N as a fraction of construct — against the 20%
bar. It clears, but the run-to-run range is 16.4–24.4%, so it straddles the bar rather
than clearing it comfortably.

Two things qualify what clearing it buys:

1. **Construct is time-budgeted** (`MAX_PROCESSING_TIME = 10.0f`). Removing a fifth of
   its work does not shorten the launch; it ingests more batches per launch. Quote the
   result as MP/s, never as construct ms.
2. **`voxelSampling` + `insertPoints` + `insertVoxels` are 49% of construct**, and all
   three are per-point root-to-leaf descents that knowing the depth does *not* remove. If
   the goal is construct throughput rather than a baseline arm, the descent itself is the
   larger target.

So: build predicted depth, as the density-only **baseline arm** that `plans/01`
Recommendation 2 requires adaptive to beat, with the thresholds delivered through
`defines` rather than recompiled by hand. Do not present it as a contribution — §1's
first point stands whatever the number turned out to be.

## 10. Verification performed

| check | result |
| --- | --- |
| `make check-vendored` | 9 vendored kernels byte-identical |
| `make check-kernels` | 10 programs, 0 failed |
| `-DREMO_PROFILE` variant compiles **and launches** | yes — the measurement runs are the proof |
| **structural counts after the signature change** | **4,137 nodes / 12,742,751 voxels — unchanged**, with and without the flag |
| phase sum vs `remolod.construct` CUevent | 97.7% (94.8–99.3%) |
| `DeviceTimeline::overflow` | 0 in every run |
| default build prints no phase rows | confirmed — gated on the flag, not formatted as `0.00` |

The structural-count check is the load-bearing one. **Adding a parameter to
`kernel_construct` is precisely the move CLAUDE.md records as having silently built no
tree for a whole commit** — `make` succeeded, `--check-kernels` passed and `--dump-frame`
exited 0 while every launch failed `CUDA_ERROR_INVALID_VALUE`. `--check-kernels` compiles
and links; only `cuLaunchKernel` rejects an argument-count mismatch. The counts are the
only evidence the launch happened.

## 11. Files

| File | Change |
| --- | --- |
| `include/remo/HostDeviceCommon.h` | `ConstructPhase`, `TimelineMark`, `DeviceTimeline`, size asserts |
| `kernels/shared/remo_prelude.cuh` | `REMO_MARK`, `REMO_PROFILE_ONLY` |
| `kernels/remolod/remolod_octree.cu` | marks at the eight existing `nanotime()` sites; iteration/spill/split counters; timeline parameter; prologue zeroing |
| `include/remo/GpuProfiler.h`, `src/shell/GpuProfiler.cpp` | `addSample()` — externally-measured samples |
| `src/pipelines/RemolodPipeline.{h,cpp}` | buffer, `defines`, launch arg, `readTimeline()`, GUI table, diagnostics lines |
| `src/main.cpp`, `src/shell/App.{h,cpp}` | `--remolod-phase-timings` |

Untouched: every vendored kernel, `kernels/{simlod,cudalod,flat}/`, the other three
pipelines, the shared rasteriser, the loader, and `TimingScopes::build`.

## 12. Open — the next plan

The trailing note "after we confirm that we need to …" is unfinished. What Stage 1 has
to settle before any kernel work:

- **Where does the predicted depth come from?** A density estimate is itself a pass over
  the batch; if it costs as much as the counting it replaces, D goes to zero.
- **What happens to the tree?** If predicted depth changes node/voxel counts, "faster" is
  uninterpretable — you can always go faster by building less tree. Either tune to
  reproduce 4,137 / 12,742,751 exactly, or report counts and time as a pair and carry a
  Chamfer/PCQM number for the quality cost.
- **Node pool.** Pre-splitting is more eager than iterative deepening; `numNodes` is a
  bump index with no device-side capacity check, and the host clamp in `readStats` is the
  only place exhaustion is noticed.
