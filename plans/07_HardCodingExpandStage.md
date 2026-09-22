# Hard-Coding the Expand Stage

## TL;DR

- **`plans/05_ExpandMeasure.md` bounded the prize at 21.5% of construct** — counting
  iterations 1..N, the re-counts that a known depth deletes. This plan builds the thing
  that collects it, as an A/B arm against the existing kernel.
- **The arm is an oracle, not a method.** Depth is a compile-time constant with zero
  computation behind it, so what it measures is the **ceiling** on what any depth
  predictor could save. If the ceiling is not ~21%, no predictor is worth building. §1.
- **Dense pre-split to a fixed depth is not implementable.** Uniform depth 6 is 299,593
  nodes against `MAX_NODES_CAPACITY = 200'000`, and 9.6 GB of occupancy grids against
  the reference tree's 132 MB. The pass must be **sparse** — only cells that contain
  points. §2.
- **The descent disappears, not just the loop.** At a fixed depth the target cell is
  `X >> (MAX_DEPTH - D)`, computable from the point's own coordinates with no pointer
  chasing. One analytic bucketing pass replaces 3.46 root-to-leaf counting descents, and
  it yields `Node::counter` for free — which is `allocatePointChunks`' only sizing
  input. §3.
- **Two effects will contaminate the headline number, and both are reported, not
  hidden.** Spilling vanishes entirely (233,070 points/batch, re-descended today by
  `voxelSampling` *and* `insertPoints`, both outside `expand`), which flatters the arm.
  And three O(numNodes)-per-batch passes get more expensive as the tree grows, which
  penalises it. §4, §5.
- **Expect a U-curve over D, not a monotone win.** D too shallow gives leaves far over
  the 50k threshold; D too deep multiplies the per-batch node sweeps. The experiment's
  real output is the curve, not a single number. §5, §8.

---

## 1. What is under test, and what it is not

`plans/05_ExpandMeasure.md` §8 measured, on morro_bay 36M, RTX 5080, strict regime,
accumulator off, five runs:

| component of `expand` | mean ms | % of construct | removed by knowing the depth? |
| --- | ---: | ---: | --- |
| counting, iteration 0 | 12.36 | 20.1% | no |
| **counting, iterations 1..N** | **13.04** | **21.5%** | **yes** |
| splitting | 2.97 | 4.8% | partly — once instead of iteratively |

The stable number behind it is **3.46 expand iterations per batch**, identical in all
five runs: 71% of all counting passes are re-counts. This plan's arm reduces that to
**one** point-proportional pass.

**This arm is deliberately an oracle.** Depth comes from a compile-time constant. There
is no density estimate, no table lookup, no analysis pass — nothing that costs
milliseconds. That is the point: it measures the **upper bound** on what any depth
predictor could ever recover, before anyone spends time building a predictor. A real
density-only baseline (the one `plans/01_NoveltyAssessment.md` Recommendation 2 requires
adaptive to beat) pays for its own estimate and will land below this ceiling.

> **A correction to how this was framed.** The original note asked for "hard coded
> depths … to see if having hard coded depths will maintain that 20%". Read literally
> that is a dense pre-split, which §2 shows cannot be built. The question the arm
> actually answers is narrower and more useful: *given a perfect, free depth oracle, how
> much of construct goes away?*

`plans/05` §1 already recorded why a density→depth table is not a contribution:
`MAX_POINTS_PER_NODE = 50'000` (`kernels/remolod/remolod_structures.cuh:12`) is itself a
density rule, so a density table reaches approximately the same tree by a cruder route.
That still stands. This is a baseline arm and a measurement, not a result.

---

## 2. Why the obvious shape does not work

A fixed depth D applied densely materialises every cell whether or not it holds a point.

| D | nodes | inner nodes | occupancy grids @ 256 KB |
| ---: | ---: | ---: | ---: |
| reference tree | 4,137 | 517 | 132 MB |
| 5 | 37,449 | 4,681 | 1.14 GB |
| 6 | **299,593** | 37,449 | **9.58 GB** |

`MAX_NODES_CAPACITY = 200'000` (`remolod_structures.cuh:14`), so **D = 6 exceeds the
node pool outright**, and its grid demand exceeds the whole card. D = 5 fits but is 9.1×
the reference node count for a tree that, at 36M / 32,768 leaves ≈ 1,099 points per
leaf, is far finer than the 50k threshold ever asked for.

`doSplitting` allocates a 256 KB `OccupancyGrid` per node that becomes inner
(`kernels/remolod/remolod_octree.cu:312-314`) out of the persistent bump allocator, and
**never frees it** — `CLAUDE.md` records that collapsing would need a grid pool that
does not exist. So grid memory is a one-way ratchet in D.

**Therefore: sparse.** Only paths to cells that actually contain points are materialised.
morro_bay is a scanned surface, so occupied cells grow roughly as 4^D rather than 8^D,
and the tree stays within budget for useful D.

---

## 3. Design — the sparse fixed-depth pass

A new `expandFixed()` in `kernels/remolod/remolod_octree.cu`, selected at compile time by
`REMO_FIXED_DEPTH`. `expand()` is left exactly as it is; `addBatch` calls one or the
other under `#ifdef`, inside the same `kPhaseExpand` mark pair so the two arms are
directly comparable at the phase level.

### 3.1 The analytic cell index is what removes the descent

`countPoint` already quantises the point to a 2^20 integer grid before it descends
(`remolod_octree.cu:123-125`):

```cpp
uint32_t X = fGridSize * (point.x - octreeMin.x) / octreeSize;   // fGridSize = 2^MAX_DEPTH
```

At a fixed depth D the destination cell is then just `X >> (MAX_DEPTH - D)` — the
root-to-leaf pointer chase at `:142-159` computes nothing the shift does not already
give. That is the mechanism: **the fixed arm never descends the tree during expand at
all.**

Cells are indexed in Morton order using the same bit order as `childIndex` (`:152`,
`(x << 2) | (y << 1) | z`), so a cell's parent is exactly `cell >> 3`. That is what makes
the pyramid in §3.2 a shift rather than a remap.

### 3.2 Four sub-passes, no loop

| # | pass | proportional to | what it does |
| --- | --- | --- | --- |
| 1 | **bucket** | points | `atomicAdd(&cellCount[D][morton(X>>s, Y>>s, Z>>s)], 1)` for every point in the batch. No tree access. |
| 2 | **pyramid** | cells | reduce level D up to level 0: `cellCount[L][c] = Σ cellCount[L+1][8c..8c+7]`. Independent of point count. |
| 3 | **materialise** | nodes | D grid-wide rounds, L = 0..D-1: every childless node at level L whose `cellCount[L][cell] > 0` is appended to `spillingNodes` and split by the **existing `doSplitting`**. |
| 4 | **seed** | nodes | every leaf at level D: `counter += cellCount[D][cell]`. |

Pass 4 is what keeps the rest of the kernel working. `allocatePointChunks` (`:458-508`)
sizes every leaf's chunk list from `node->counter` and nothing else, and the invariant it
needs is `counter >= the number of points insertPoints will push`. The bucket pass
satisfies it exactly, with no counting descent.

`counter` accumulates across batches and is only reset when a node splits (`:277`), so
pass 4 uses `+=`. `cellCount` is zeroed per batch (~8 MB at D=7, a few µs).

**Reuse rather than rebuild:** `doSplitting` (`:263-329`) is called unchanged, so child
allocation, `level`/`X`/`Y`/`Z`/`name` initialisation, chunk recycling and the 256 KB
grid allocate-and-zero all keep their existing, tested behaviour. Pass 3 only chooses
*which* nodes go into `spillingNodes`.

### 3.3 Scratch layout

`cellCount` is one flat block of Σ(8^L) for L = 0..D, carved from the momentary buffer
next to the existing allocations at `:824-834`. The `RemoAllocator` uniform-control-flow
rule (`kernels/shared/remo_alloc.cuh`) is satisfied because the variant is chosen at
compile time — each variant walks one fixed allocation sequence.

| D | `cellCount` bytes | fits in the 512 MB momentary buffer? |
| ---: | ---: | --- |
| 5 | 0.15 MB | yes |
| 6 | 1.2 MB | yes |
| 7 | 9.6 MB | yes |
| 8 | 76.7 MB | yes — the fixed arm also skips the 160 MB `spilledPoints` allocation |
| 9 | 613 MB | **no** — hard ceiling |

Today's carve is ≈409 MB of 512 MB (`backlog_voxels` 160 MB, `backlog_targets` 80 MB,
`spilledPoints` 160 MB, `chunkQueue` 8 MB, `spillingNodes` 0.8 MB). The fixed arm frees
`spilledPoints` outright (§4), so D ≤ 8 is comfortable and **D = 9 is refused**.

### 3.4 The node pool gets its missing check

`doSplitting:269` is `uint32_t childOffset = atomicAdd(&stats->numNodes, 8);` with **no
bound against `MAX_NODES_CAPACITY`**, and `nodes[childOffset + i] = child` at `:291` then
writes past the pool. `CLAUDE.md` names this and warns that "anything that splits more
eagerly must keep that reporting intact". Pre-splitting is precisely more eager.

Add the check **unconditionally, in both variants**:

```cpp
uint32_t childOffset = atomicAdd(&stats->numNodes, 8);
if(childOffset + 8 > MAX_NODES_CAPACITY){
    timeline->nodePoolOverflow = 1;   // always-written; see §6
    return;                           // this node stays a leaf
}
```

It cannot change behaviour below 200,000 nodes, so the default arm's tree must still be
4,137 / 12,742,751 — which is how the change is verified. `DeviceDiagnostics` already
declares an unused `nodePoolOverflow` (`include/remo/HostDeviceCommon.h:78-87`), but
`kernel_construct` is not passed a `DeviceDiagnostics*` and **will not be given one** —
see §6 on why no parameter is added.

---

## 4. What disappears, and why that flatters the arm

**Spilling and evacuation vanish completely.** In the baseline, a node that exceeds
50,000 points splits, and the points it already holds must be pulled back out into
`spilledPoints` (`doCounting:204-248`) so `insertPoints` can place them in the new
children. In the fixed arm, **points are only ever stored at depth D**, so a node below D
never holds points, so nothing is ever evacuated. `spilledPoints` is never written, the
evacuation loop is dead, and the `processRange(*numSpilledPoints, …)` passes in
`doCounting:195`, `voxelSampling` and `insertPoints:586-596` all see zero.

That is worth **233,070 points per batch — 23% of a 1M batch** (`plans/05` §8), and two
of the three places it is paid are **outside `expand`**. So the arm will beat 21.5%
end-to-end, for a reason that is not "skipping re-counts".

**This is reported as two numbers, never one:**

| number | what it is | compares against |
| --- | --- | --- |
| `remolod.construct.expand` delta | the 21.5% claim under test | `plans/05` §8's decomposition |
| construct MP/s delta | the arm's real end-to-end effect | the baseline arm in the same session |

The gap between them is the spill windfall, and it is stated explicitly. Quoting only the
second would make the 21.5% prediction uncheckable against anything.

---

## 5. What gets worse — the risk that decides the experiment

Three passes iterate **all `stats->numNodes` nodes, once per batch**, independent of how
many points arrive:

| pass | location | % of construct today |
| --- | --- | ---: |
| `allocatePointChunks` | `remolod_octree.cu:458-508` | 2.3% |
| `allocateVoxelChunks` | `:601-632` | 2.2% |
| closing stats reduction | `:925-976` | part of the unmarked ~2.3% |

At the reference 4,137 nodes they are cheap. A sparse fixed-depth tree is larger — and
these costs scale linearly with it. If D = 8 produces ~18× the nodes, `allocPointChunks`
alone goes from 2.3% to ~40% and **swallows the entire 21.5% saving**.

`voxelSampling` grows too, for a different reason: it writes into every ancestor's
occupancy grid on the way down (`:435`), so a deeper tree means more grids touched per
point and more voxels created. That shows up in the voxel count, which is why the
(nodes, voxels) pair travels with every timing figure.

**So the deliverable is a curve over D, not a number.** The plausible shape:

- **D too shallow** (5) — leaves far over the 50k threshold, poor LOD granularity,
  `maxPointsPerNode` blows up, few nodes so the sweeps stay cheap.
- **D about right** (6–7) — the saving is visible, node count within a small multiple of
  the reference.
- **D too deep** (8) — node sweeps dominate, grid memory ratchets, the arm is *slower*
  than the baseline.

If no D shows a net win, that is a real result and it retires the idea: the 21.5% is
reachable only by something that does not enlarge the tree.

---

## 6. Data and host plumbing

### Timeline fields

Extend `DeviceTimeline` in `include/remo/HostDeviceCommon.h:125-138`:

```cpp
uint64_t bucketNs, pyramidNs, materialiseNs, seedNs;  // fixed-depth sub-passes
uint32_t occupiedCells;        // at depth D, this launch
uint32_t nodePoolOverflow;     // always written, both variants
```

and update `static_assert(sizeof(DeviceTimeline) == 4288, …)` at `:141` to **4328**. That
assert is the only thing keeping host and device agreed, so it moves deliberately.

Sub-pass timings go in named fields rather than reusing `expandIterNs[]` slots. Two
meanings on one field is exactly the kind of thing that reads fine today and misleads a
year from now.

**`expandIters` keeps its meaning — point-proportional passes over the batch.** It is
3.46 per batch in the baseline and **1** in the fixed arm. That single counter is the
clearest statement of what the arm does.

### No new kernel parameter

`kernel_construct` already takes 11 positional arguments (`remolod_octree.cu:768-781`),
and `CLAUDE.md` records an arity change as the defect that silently built no tree for a
whole commit — `make` succeeded, `--check-kernels` passed, `--dump-frame` exited 0, and
every launch failed `CUDA_ERROR_INVALID_VALUE`.

So **nothing is added to the signature.** `nodePoolOverflow` and `occupiedCells` ride the
`remo::DeviceTimeline*` that is already passed in both variants. The consequence is that
the timeline buffer is no longer profiling-only: its counter block must be zeroed and
read back **unconditionally**, while the mark array stays under `REMO_PROFILE`.
`RemolodPipeline::readTimeline()` (`src/pipelines/RemolodPipeline.cpp:445-491`) currently
early-returns on `!m_phaseTimings`; that early return moves below the counter read.

> Collapsing the 11 arguments into a `remo::ConstructArgs` struct — the pattern
> `CLAUDE.md` prefers, and that `remo::AccumArgs` already follows — is the right
> long-term fix and is **explicitly not in this plan**. It is a one-time arity change
> that deserves its own commit and its own structural-count verification, not a rider on
> an experiment.

### The flag

`--remolod-fixed-depth N`, following `--remolod-phase-timings` exactly (`plans/05` §6):

| file | change |
| --- | --- |
| `src/main.cpp` | help text; parse arm using the existing `takeArg` helper (`:223-230`); reject D < 1 or D > 8 |
| `src/shell/App.h` | `int remolodFixedDepth = 0;` on `AppOptions` (0 = off) |
| `src/shell/App.cpp` | `p->setFixedDepth(m_options.remolodFixedDepth);` in the registry factory, beside the two existing setters |
| `RemolodPipeline::initPrograms` | `desc.defines.push_back("-DREMO_FIXED_DEPTH=" + std::to_string(m_fixedDepth));` for the construct program only |

`defines` is part of the compile cache key (`CudaModularProgram.cpp:272-281`), so every D
caches as its own variant and a sweep recompiles each value once. The default build is
byte-for-byte unaffected.

No GUI control: the depth selects a compiled variant, so it must be fixed before
`initPrograms()`. `guiStats()` shows the active D read-only.

### Readouts

- `PipelineStats::maxPointsPerNode` is declared but **RemoLOD never sets it**, so
  `--dump-frame` prints 0 today. Set it from a max-reduction in the existing closing
  stats pass. It is the direct measure of the shallow-D failure mode, and `--dump-frame`
  already prints it.
- `diagnostics()` lines (`RemolodPipeline.cpp:553-630`), emitted before the accumulator
  early-return as the phase rows already are: `fixed depth`, `occupied cells`,
  `expand iters/batch`, `nodes split/batch`, `spilled pts/batch` (expected 0), and
  `node pool  OVERFLOW` when set.

---

## 7. Staging

**Stage 1 — the device-side node pool check, alone.** Unconditional, both variants, no
flag. Verify the default tree is still 4,137 / 12,742,751. Lands the safety fix that
every later stage depends on, and proves the `DeviceTimeline` counter block can be read
without `REMO_PROFILE`.

**Stage 2 — `expandFixed()` behind `-DREMO_FIXED_DEPTH`, plus the flag.** All four
sub-passes. Acceptance is *structural*, not timing: the arm builds a tree, the point
count matches the baseline's ingested count, `spilled pts/batch` is 0, and
`counter >= numPoints` holds for every leaf.

**Stage 3 — the sweep and the write-up.** D ∈ {5,6,7,8} × 5 repeats, both arms, producing
the curve in §5 and a `wiki/04_FixedDepthFindings.md` in the shape of
`wiki/03_ExpandStageFindings.md`.

**Stage 4 (deferred) — the real density-only baseline.** Replace the compile-time
constant with a cheap per-region density estimate, and pay for the estimate. Only worth
starting if Stage 3 shows a ceiling worth chasing. This is the arm
`plans/01_NoveltyAssessment.md` Recommendation 2 actually asks for.

---

## 8. Measurement protocol

```sh
# baseline arm
./build/remobench --pipeline remolod --strict-timing --remolod-no-accum \
  --remolod-phase-timings --open data/morro_bay_35M/morro_bay_36M.simlod \
  --dump-frame /tmp/base.ppm

# fixed-depth arm, once per D in 5 6 7 8
./build/remobench --pipeline remolod --strict-timing --remolod-no-accum \
  --remolod-phase-timings --remolod-fixed-depth 7 \
  --open data/morro_bay_35M/morro_bay_36M.simlod --dump-frame /tmp/d7.ppm
```

Strict regime throughout, accumulator off so its ~2.5 ms is not folded in, five repeats
per cell because the per-launch time split is noisy while the structural counts are not.

**Report as MP/s, never as construct ms.** `MAX_PROCESSING_TIME = 10.0f`
(`remolod_octree.cu:27`) caps every construct launch, so removing work does not shorten
the launch — it ingests more batches per launch. `plans/05` §9 and
`wiki/03_ExpandStageFindings.md` §6 both record this.

Every row carries `(nodes, voxels, maxPointsPerNode)` beside its timing. A speed number
without them is meaningless, because building less tree is always faster.

**Do not compare frames.** Only `flat` is run-to-run deterministic; RemoLOD samples a
voxel's colour from the first point to reach the cell, so two runs of the same binary
differ. Compare the printed counts.

---

## 9. Verification

There is no test suite — `tests/unit/` is empty and `REMOBENCH_BUILD_TESTS` is OFF — so
these are the acceptance tests.

| check | expected |
| --- | --- |
| `make check-vendored` | 9 vendored kernels byte-identical — nothing outside `kernels/remolod/` is touched |
| `make check-kernels` | all programs compile, 0 failed, **in both variants** |
| **default arm structural counts** | **4,137 nodes / 12,742,751 voxels — unchanged** |
| `simlod` structural counts | 4,137 / 12,742,751 — still identical to the default RemoLOD arm |
| `cudalod` structural counts | 2,252 / 12,742,500 — matches `bench/reference/` |
| fixed arm: points ingested | equals the baseline's for the same batch count — no points lost |
| fixed arm: `spilled pts/batch` | 0 |
| fixed arm: `expand iters/batch` | 1.0 (baseline: 3.46) |
| fixed arm: per-leaf invariant | `counter >= numPoints` for every leaf — else `allocatePointChunks` under-allocates |
| `DeviceTimeline::overflow` | 0 |
| `nodePoolOverflow` | 0 at the chosen D; set, not crashed, at a deliberately too-deep D |
| phase sum vs `remolod.construct` CUevent | within ~5%, as `plans/02` Stage 3's acceptance test requires |

**The structural counts on the default arm are the load-bearing check.** Both the node
pool check and the `#ifdef` restructuring touch code the default arm executes.
`--check-kernels` compiles and links; it does not launch, and the driver only rejects an
argument mismatch at `cuLaunchKernel`. The counts are the only evidence the launch
happened.

---

## 10. Files

**Modified**

| file | change |
| --- | --- |
| `kernels/remolod/remolod_octree.cu` | `expandFixed()` and its four sub-passes; the unconditional node pool check in `doSplitting`; `#ifdef` selection in `addBatch`; `maxPointsPerNode` in the closing stats reduction |
| `include/remo/HostDeviceCommon.h` | six `DeviceTimeline` fields; size assert 4288 → 4328 |
| `src/pipelines/RemolodPipeline.{h,cpp}` | `setFixedDepth()`, the `-DREMO_FIXED_DEPTH=N` define, unconditional counter readback, diagnostics lines, GUI rows |
| `src/main.cpp`, `src/shell/App.{h,cpp}` | `--remolod-fixed-depth N` |
| `wiki/04_FixedDepthFindings.md` | new, at Stage 3 |

**Reused rather than rebuilt:** `doSplitting` (child allocation, chunk recycling, grid
allocate-and-zero); `processRange` from `kernels/shared/remo_prelude.cuh`; `REMO_MARK`
and the existing `kPhaseExpand` mark pair; `GpuProfiler::addSample()`; the
`KernelProgramDesc::defines` route and its cache-key behaviour; `takeArg` in `main.cpp`.

**Untouched:** every vendored kernel, `kernels/{simlod,cudalod,flat}/`, the other three
pipelines, the shared rasteriser, the loader, `TimingScopes::build`, and
`kernel_construct`'s parameter list.

---

## 11. What this is not

- **Not a contribution.** `plans/05` §1's argument stands whatever the number turns out
  to be: a depth rule driven by density is what an octree already is, and the
  detail-aware thesis is depth driven by *geometric complexity*, which is decoupled from
  density. This is the baseline that has to be beaten.
- **Not a quality result.** The arm will build a different tree, and no Chamfer, PCQM or
  PointSSIM number is produced here. `plans/01` §3 names those as required before any
  matched-budget claim. Until one exists, "faster" travels with its counts and nothing
  more is claimed.
- **Not validated past 36M.** `plans/05` §8 could not run the 350M arm — `remolod`
  declares `needsWholeCloudResident` and needs 15.3 GB against a 16.585 GB card. That is
  `plans/06_ComparisonFixes.md`'s work. `expand`'s share should grow with depth, so 46%
  is a floor for larger clouds, but that is untested and so is everything here.

---

## 12. What actually landed

Stages 1-3 are done. The measurements are in
[wiki/04_FixedDepthFindings.md](../wiki/04_FixedDepthFindings.md); this section records
only where the plan was wrong.

**The headline, against §1's question.** At the best depth the arm is **+42% construct
MP/s** (877 vs 617 on morro_bay 36M, RTX 5080, strict, accumulator off,
`--device-budget 9G`, five runs). `expand` falls 24.56 -> 10.94 ms, a saving of 23.2% of
construct, which is the 21.5% this plan put under test. The curve is the U §5 predicted:
D=5 +33%, **D=6 +42%**, D=7 -7%, D=8 -62% and unable to finish the cloud.

### Where the saving actually comes from

**The bound is confirmed and is the largest single term.** Deleting the re-counts is
-9.46 ms, **55%** of the 17.34 ms the arm saves. `plans/05`'s prediction was right about
what it measured; the baseline reproduces it phase for phase, and its structural
counters came out identical (37 batches, 3.46 iters/batch, 233,070 spilled pts/batch,
14.0 nodes split/batch). The one shift is that the re-counts are 16.1% of construct in
this session against 21.5% there -- the bottom edge of `plans/05`'s own 16.4-24.4%
range, and §6 of `wiki/03` already says why the millisecond split moves.

**What the plan got wrong is that the ceiling is higher than the bound, not lower.**
`plans/05` §2 recorded counting iteration 0 as irreducible: "you still must count once
to place points". True, but it does not have to *descend*. The analytic index cut that
pass by 61% (11.20 -> 4.37 ms), a further -6.83 ms -- 39% of the saving, on a pass the
bound had written off. Against that, the pre-split costs +2.67 ms more than the
splitting it replaces. So the ceiling is "the re-counts, plus the descent leaving the
first count, minus what pre-splitting costs".

§4's contamination warning was right and the numbers are reported separately: spilling
vanished entirely (233,070 pts/batch, `spilled pts/batch` is 0 at every D), which shows
up in `voxelSampling` and `insertPoints` outside `expand`, and `insertVoxels` pays part
of it back because a deeper tree holds more voxels.

**§5's named risk did not materialise.** `allocPointChunks` got *cheaper* at every
depth -- 2.89 -> 0.87 ms at D=6 with 2.5x the nodes -- because its cost is the
linked-list walk over each leaf's chunks, and a fixed-depth leaf is small. What turns
the curve instead is `materialise` (O(numNodes x D), 30.48 ms at D=8), `pyramid`
(O(8^D), 7.67 ms at D=8), and the voxel count. At the shallow end D=5's `insertPoints`
is +10.10 ms on its own, because a 145k-point leaf is a 145-chunk list walked from the
head per point.

### Deviations from §3 and §6

- **The cell index is masked, not clamped** (§3.1). A point on `boxMax` quantises to
  `2^MAX_DEPTH`, and the descent absorbs that in its per-level `& 1`, which *wraps* it
  to cell 0. Clamping to the last cell left 30 leaves with `counter < numPoints` and
  `allocatePointChunks` under-allocating them. The index must be bit-for-bit what the
  descent computes, whatever the descent computes.
- **The node pool check needed a second half** (§3.4). `doSplitting`'s grid-zeroing pass
  still dereferenced the refused node's `grid`, which the refusal had left null, so the
  safety check was itself a fault until that pass learned to skip refused nodes. It also
  clamps `numNodes` back to the pool, because every later pass sweeps `[0, numNodes)`
  and the refused `atomicAdd` had already moved it past the end.
- **`DeviceTimeline` is 4336 bytes, not the 4328 §6 predicted.** `maxPointsPerNode` took
  the existing `pad1` and cost nothing; the extra 8 is a seventh counter,
  `counterUnderflows`, plus its padding. §9 asks for `counter >= numPoints` per leaf and
  nothing else could report it: the kernel's only complaint goes through
  `CudaPrint::print()`, which returns on its first line. It caught the masking bug
  above, which is the whole argument for it.
- **Upstream's 200 MB memory safety margin is not enough for this arm.** It is tested
  only between batches and `AllocatorGlobal` has no bounds check, so one fixed-depth
  batch -- 610 splits at D=8, 256 KB of grid each plus a 16 KB chunk per new leaf --
  walks past the end of the store and faults in `allocatePointChunks` with no warning
  printed. The fixed arm uses 1 GB and stops cleanly. Found by D=8, via
  `compute-sanitizer`.
- **The arm cannot be sized at 48 B/pt.** `RemolodPipeline::allocate` gives the fixed arm
  what the shared budget leaves, because the grid ratchet is not a per-point cost. At
  D=7 the baseline's coefficient stopped the build at 58% of the cloud, and a truncated
  run is not comparable to a complete one.
- **§3.3's memory ceiling is the wrong ceiling.** `cellCount` at D=8 is 76.7 MB and fits
  as predicted; what actually stops D=8 is the occupancy-grid ratchet in the persistent
  store -- it truncates at 80.1% of 36M with a 9 GB budget. D=9 is still refused, in the
  kernel's `static_assert` and in the flag parse.
- **`--dump-after 8` is not enough for a slow arm.** D=8's first figures said "55%
  ingested"; that was the frame count, not memory. The protocol in §8 needs
  `--dump-after 40`.
- **`spillingNodes` is bounded.** The materialise pass can nominate more than the
  100,000-entry list holds, so the write is guarded and the overflow reported; a node
  that does not fit stays a leaf above D and pass 4 still gives it the whole subtree's
  count from the pyramid.

### What was verified

The default arm still builds **4,137 nodes / 12,742,751 voxels**, identical across five
runs and still identical to `simlod`; `cudalod` is 2,252 / 12,742,500; `check-vendored`
is 9 byte-identical kernels. Every arm reproduced its counts exactly in all five runs.
The node pool overflow path was exercised by temporarily rebuilding with
`MAX_NODES_CAPACITY = 20'000`: the count pinned at 20,000, `node pool OVERFLOW` was
reported, the leaf-counter invariant held, and nothing faulted.

**Stage 4 remains open** and is unchanged by this: the ceiling is real, but the depth
that reaches it is data-dependent (6 for morro_bay at this budget, and nothing here says
that transfers), and a predictor that costs what the counting cost reaches none of it.

---

## 13. Open — the next plan

- **Where does a real predicted depth come from?** Stage 4. If the estimate costs what
  the counting it replaces cost, the ceiling measured here is unreachable in practice.
- **Should the fixed arm still honour `MAX_POINTS_PER_NODE`?** As specified it does not:
  a depth-D cell holding 200k points stays one leaf. Letting it split past D would
  reintroduce the loop for a small number of nodes and might be the best of both — but it
  changes what the arm measures, so it is a separate question.
- **Grid memory is a one-way ratchet in D.** Every node that becomes inner takes 256 KB
  that is never returned. Collapsing needs a grid pool that does not exist
  (`CLAUDE.md`), so a too-deep sweep value is not recoverable within a session.

## Related

- `plans/05_ExpandMeasure.md` — Stage 0, the bound this plan spends.
- `wiki/03_ExpandStageFindings.md` — the prose form of those measurements.
- `plans/01_NoveltyAssessment.md` §3, Recommendation 2 — why a density-only arm exists.
- `plans/02_ProfilingTools.md` §8 — Layer 3 as built for RemoLOD, the mechanism reused here.
- `plans/06_ComparisonFixes.md` — the 350M blocker that keeps this 36M-only.
