# What a free depth oracle is worth

[03_ExpandStageFindings.md](03_ExpandStageFindings.md) measured that 21.5% of
`kernel_construct` is re-counting: expand iterations 1..N, the passes a known octree
depth would delete. This page spends that bound. It measures an arm that pre-splits to
a **compile-time constant depth**, with no density estimate, no table and no analysis
pass behind it — an oracle, not a method, so what it reports is the **ceiling** on what
any depth predictor could recover.

The answer has three parts:

- **At the best depth the arm is 42% faster** (877 vs 617 MP/s of construct), which is
  more than double the 21.5% that was under test.
- **`03`'s bound is confirmed and is the largest single term** — deleting the re-counts
  is 55% of the saving. The arm beats the bound because of two effects `03` did not
  count: the *descent* leaving the first counting pass, which `03` ruled out (39%), and
  the spill windfall in phases outside `expand` entirely.
- **It is a U-curve, and it turns hard.** One depth deeper than the peak the arm is
  *slower* than the baseline; two deeper it is 62% slower and cannot finish the cloud.

Nothing here is a contribution. It is the baseline that a detail-aware rule has to
beat, and [plans/07_HardCodingExpandStage.md](../plans/07_HardCodingExpandStage.md) §11
says why.

---

## 1. The curve

morro_bay 36M, RTX 5080, strict regime, accumulator off, `--device-budget 9G`, five
runs per arm. `base` is the default iterative `expand()`; D is `--remolod-fixed-depth`.
Construct MP/s is points ingested over the `remolod.construct` CUevent total.

| arm | nodes | voxels | max pts/node | construct ms | **MP/s** | MP/s range | vs base | high water |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 4,137 | 12,742,751 | 49,739 | 59.12 | 617 | 551–700 | — | 0.94 GB |
| D = 5 | 2,553 | 7,436,345 | **145,378** | 44.97 | 822 | 716–1062 | **+33%** | 0.79 GB |
| **D = 6** | 10,169 | 22,929,597 | 63,944 | 41.72 | **877** | 813–1068 | **+42%** | 1.32 GB |
| D = 7 | 40,969 | 50,162,895 | 29,845 | 62.96 | 575 | 548–594 | −7% | 2.92 GB |
| D = 8 | 141,537 † | 67,279,210 † | 11,651 | 125.63 | 231 | 221–256 | −62% | 6.89 GB |

† **D = 8 does not finish.** It stops on `memCapacityReached` at **80.1%** of the cloud
(29.0M of 36.2M points) against a 7.46 GB store, so its row describes a smaller tree
than every other row and its MP/s is a rate on a truncated build. Every other arm
ingested 100%.

The counts are the load-bearing part of this table. **Every arm reproduced its node,
voxel and point counts exactly across all five runs** — 4,137 / 12,742,751 for the
baseline, and 141,537 / 67,279,210 / 29,000,000 even for the truncated D = 8. The
milliseconds move (§6); the tree does not.

`max pts/node` is the shallow-depth failure mode in one number. At D = 5 a leaf holds
145k points against a 50k threshold, and §4 shows where that is paid.

### The counters behind the milliseconds

Identical in all five runs of each arm, like the counts:

| arm | batches | expand iters/batch | spilled pts/batch | nodes split/batch | occupied cells/batch |
|---|---:|---:|---:|---:|---:|
| base | 37 | **3.46** | **233,070** | 14.0 | — |
| D = 5 | 37 | **1.00** | **0** | 8.6 | 74 |
| D = 6 | 37 | **1.00** | **0** | 34.4 | 264 |
| D = 7 | 37 | **1.00** | **0** | 138.4 | 1,066 |
| D = 8 | 29 | **1.00** | **0** | 610.1 | 4,475 |

`expand iters/batch` going 3.46 → 1.00 is the arm's whole mechanism in one number: 71%
of all counting passes in the baseline are re-counts, and here there are none to re-do.

**The sparsity assumption holds.** `plans/07` §2 ruled out a dense pre-split — uniform
depth 6 is 299,593 nodes against a 200,000 pool, and depth 6 grids alone are 9.58 GB —
and argued that a scanned surface would occupy cells as roughly 4^D rather than 8^D. It
does: both occupied cells per batch (74 → 264 → 1,066 → 4,475) and total nodes
(2,553 → 10,169 → 40,969 → 141,537) multiply by **~4 per level, not 8**. A dense D = 6
would have been 29× the node count the sparse pass actually builds.

## 2. The saving is not where the bound said it was

`expand` decomposes cleanly in both arms, and the three parts move in different
directions. Baseline against D = 6, mean ms:

| | baseline | D = 6 | Δ |
|---|---:|---:|---:|
| counting, iteration 0 | 11.20 | 4.37 (bucket + zeroing) | **−6.83** |
| counting, iterations 1..N | 9.46 | 0.00 | **−9.46** |
| splitting | 3.90 | 6.57 (pyramid + materialise + seed) | **+2.67** |
| = all of `expand` | 24.56 | 10.94 | −13.62 |

Two of those three were predicted. The third was not:

- **The re-counts went away, as expected.** 9.46 ms, 16.1% of construct in this
  session — the same quantity `03_ExpandStageFindings.md` measured as 21.5% (§6 on why
  the percentage moves between sessions). This is the thing the plan set out to test.
- **The pre-split costs more than the splitting it replaces**, +2.67 ms. Three
  grid-wide rounds of node sweeps plus the pyramid are not free.
- **The pass that was called irreducible fell by 61%.** `03` §2 recorded iteration 0 as
  "no — you still must count once to place points". True, but it does not have to
  *descend*: at a fixed depth the destination cell is `X >> (MAX_DEPTH - D)`, computable
  from the point's own coordinates. One `atomicAdd` into a flat array replaces a
  root-to-leaf pointer chase, and that is worth 6.83 ms — **72% of what deleting the
  loop was worth, on a pass the bound had written off entirely.**

So the ranking is: the loop is still the biggest single piece, and the descent is close
behind it. Against the 17.34 ms the arm saves overall,

| term | ms | share of the saving | predicted by `03`? |
|---|---:|---:|---|
| re-counts deleted | −9.46 | **55%** | **yes — this is the 21.5%** |
| descent out of counting iteration 0 | −6.83 | 39% | no — called irreducible |
| pre-split structure (pyramid + materialise + seed) | +2.67 | −15% | partly — "once instead of iteratively" |
| spill windfall in `voxelSampling` | −3.89 | 22% | named as a consequence, not costed |
| spill windfall in `insertPoints` | −3.22 | 19% | " |
| `allocPointChunks` + `allocVoxelChunks` | −2.54 | 15% | no — expected to get *worse* |
| extra voxels in `insertVoxels` | +5.94 | −34% | no |

`03` §4 saw the descent coming from the other side — "`voxelSampling` + `insertPoints` +
`insertVoxels` are 49.1% of construct … the descent itself is the larger target". This
arm removes the descent from a fourth pass it did not list, counting, and that alone is
worth nearly as much as the thing the page was written about.

### The baseline reproduces `03`

The comparison above only means something if the baseline arm is the same baseline `03`
measured. It is. Five fresh runs, same cloud and regime, `--device-budget 9G` pinned
(`03` did not pin it) and `--dump-after 40`:

| | `03` mean ms | `03` % | now mean ms | now % | now range % | overlaps `03`'s range? |
|---|---:|---:|---:|---:|---:|---|
| expand | 28.37 | 46.3 | 24.56 | 41.9 | 39.5–47.1 | yes |
| insertPoints | 12.47 | 20.6 | 12.82 | 21.8 | 18.1–24.0 | yes |
| voxelSampling | 11.80 | 19.4 | 12.76 | 21.7 | 20.4–23.2 | yes |
| insertVoxels | 5.52 | 9.1 | 3.79 | 6.5 | 5.4–9.8 | yes |
| allocPointChunks | 1.36 | 2.3 | 2.89 | 4.9 | 2.4–6.5 | yes |
| allocVoxelChunks | 1.29 | 2.2 | 1.83 | 3.1 | 1.6–4.0 | yes |
| batchBegin | 0.04 | 0.1 | 0.04 | 0.1 | 0.1 | yes |
| — counting, iteration 0 | 12.36 | 20.1 | 11.20 | 19.1 | 14.2–26.8 | yes |
| — counting, iterations 1..N | 13.04 | **21.5** | 9.46 | **16.1** | 11.5–21.5 | yes, at the bottom edge |
| — splitting | 2.97 | 4.8 | 3.90 | 6.6 | 3.4–9.6 | yes |
| phase sum | 60.85 | | 58.69 | | | |

**Every phase overlaps**, and the structural counters in `03` §3 came out *identical*,
not merely close: 37 batches, 3.46 expand iterations per batch, 233,070 spilled points
per batch, 14.0 nodes split per batch. That is the confirmation that matters, and it is
exactly what `03` §3 said to reason from.

The one number that lands differently is the headline itself: **the re-counts are 16.1%
of construct here against `03`'s 21.5%**, at the bottom edge of `03`'s own 16.4–24.4%
range. So the bound was real but its central value was optimistic, for the reason `03`
§6 gives — the 10 ms launch cap moves how many batches land in a launch and the
millisecond split moves with it. The arm cleared the bound anyway, on terms the bound
did not include.

## 3. Where the other 29% comes from

The headline is +42% construct throughput; §2 accounts for 13.62 ms of a 17.34 ms
saving. The rest is outside `expand` altogether, and the plan predicted it would be
(§4 there, "what disappears, and why that flatters the arm"). Per-phase deltas against
the baseline, mean ms:

| phase | D = 5 | D = 6 | D = 7 | D = 8 |
|---|---:|---:|---:|---:|
| expand | −15.49 | −13.62 | −8.00 | +25.92 |
| voxelSampling | −7.25 | −3.89 | +1.82 | +12.58 |
| insertPoints | **+10.10** | −3.22 | −6.71 | −8.07 |
| insertVoxels | −0.94 | **+5.94** | **+18.70** | +25.80 |
| allocPointChunks | −0.53 | −2.02 | −2.43 | −1.89 |
| allocVoxelChunks | +0.07 | −0.52 | +0.15 | +1.67 |
| **phase sum** | **−14.05** | **−17.34** | **+3.63** | **+56.00** |

**Spilling vanishes completely — 233,070 points per batch, 23% of a 1M batch.** At a
fixed depth points are only ever stored at depth D, so no node below D ever holds
points to evacuate. `spilled pts/batch` is 0 in every fixed run, and those points were
being re-descended by `voxelSampling` and `insertPoints` as well as by the counting
loop. That is most of the `voxelSampling` and `insertPoints` columns.

**`insertVoxels` pays it back.** A deeper tree writes into more ancestor grids per
point, so it creates more voxels: 22.9M at D = 6 against 12.7M in the baseline, 50.2M
at D = 7. `insertVoxels` grows with them, and by D = 7 it has eaten the whole saving.

So the two numbers travel together and neither stands alone:

| number | value at D = 6 | what it is |
|---|---:|---|
| `remolod.construct.expand` delta | −13.62 ms, −23.2% of construct | the 21.5% claim under test |
| construct MP/s delta | +42% | the arm's end-to-end effect |

The gap between them is the spill windfall net of the extra voxels. Quoting only the
second would make the original prediction uncheckable.

## 4. Why the curve turns

The plan expected the per-batch `O(numNodes)` sweeps to swallow the saving. **They did
not — `allocPointChunks` got *cheaper*, not dearer**, at every depth (2.89 ms → 0.87 ms
at D = 6, 0.46 ms at D = 7) despite 10× the nodes. Its cost is dominated by the
linked-list walk over each leaf's existing chunks, and a fixed-depth leaf is small, so
the walk shortens faster than the node count grows.

What actually turns the curve is visible in the sub-passes, which are timed separately
(`bucketNs`, `pyramidNs`, `materialiseNs`, `seedNs` on `DeviceTimeline`). Mean ms,
summed over the batches in the run:

| | bucket | pyramid | materialise | seed | zeroing † | = `expand` |
|---|---:|---:|---:|---:|---:|---:|
| D = 5 | **6.08** | 0.19 | 2.57 | 0.14 | 0.09 | 9.07 |
| D = 6 | 4.12 | 0.26 | **5.37** | 0.94 | 0.25 | 10.94 |
| D = 7 | 2.81 | 0.53 | **10.37** | 0.20 | 2.65 | 16.56 |
| D = 8 | 1.57 | 7.67 | **30.48** | 0.62 | 10.13 | 50.48 |

† the per-batch `cellCount` clear, which is inside the phase but outside the four timed
sub-passes; it is `expand` minus their sum.

**D = 8's row is 29 batches against 37 for the others** (§1), so read it as a shape, not
as a directly comparable total.

Three things move, in different directions:

- **Only `bucket` gets cheaper with depth** — per batch, 0.164 ms at D = 5, 0.111 at
  D = 6, 0.076 at D = 7, 0.054 at D = 8. It is the one pass proportional to points
  rather than to cells or nodes, and a deeper tree spreads the same batch of atomics
  over 8× as many counters at each level, so they collide less. Everything else in the
  table grows.
- **`materialise` is `O(numNodes × D)`** and dominates everywhere past D = 5 — 2.57 ms
  at D = 5 to 30.48 at D = 8. It is the price of sweeping the whole pool once per level
  per batch, and it is the term that makes pre-splitting cost more than the splitting it
  replaced.
- **Two terms are `O(8^D)` and independent of the point count**: the pyramid (0.26 ms at
  D = 6, 7.67 at D = 8) and the `cellCount` clear (0.25 → 10.13 ms, writing 76.7 MB per
  batch at D = 8). Together they are **17.80 ms at D = 8, 35% of `expand`** — dense
  levels are cheap until suddenly they are not, and they are why D = 9 is refused rather
  than merely expensive.

On top of the sub-passes, **more tree means more voxels**, and `voxelSampling` +
`insertVoxels` scale with them (§3).

And at the shallow end, D = 5's **`insertPoints` is +10.10 ms** — worse than the
baseline by itself. A 145k-point leaf is a 145-chunk linked list, and `insertPoints`
walks it from the head for every point it places. That is the cost of a depth too
shallow for the data, and it is why `max pts/node` travels with every row.

## 5. Reproducing

```sh
# baseline arm
./build/remobench --pipeline remolod --strict-timing --remolod-no-accum \
  --remolod-phase-timings --device-budget 9G --dump-after 40 \
  --open data/morro_bay_35M/morro_bay_36M.simlod --dump-frame /tmp/base.ppm

# fixed-depth arm, once per D in 5 6 7 8
./build/remobench --pipeline remolod --strict-timing --remolod-no-accum \
  --remolod-phase-timings --device-budget 9G --dump-after 40 \
  --remolod-fixed-depth 6 \
  --open data/morro_bay_35M/morro_bay_36M.simlod --dump-frame /tmp/d6.ppm
```

Four things about that command line are not optional:

- **`--device-budget` is pinned.** A progressive pipeline stops at whatever fraction the
  budget holds, and the fixed arm's store is sized from the budget rather than from
  48 B/pt (§7), so two unpinned runs are not comparable.
- **`--dump-after 40`, not the default 8.** A slow arm is still ingesting when the dump
  fires. D = 8's first measurements said "55% ingested"; that was the frame count, not
  memory, and 40 frames takes it to its real ceiling of 80.1%.
- **Strict regime, accumulator off**, as in `03`, so the ~2.5 ms accumulator is not
  folded in.
- **MP/s, never construct ms.** `MAX_PROCESSING_TIME = 10.0f` caps every launch, so
  removing work does not shorten the launch — it ingests more batches per launch.

`--remolod-fixed-depth N` compiles a separate variant of the construct kernel through
`KernelProgramDesc::defines`, which is part of the compile cache key, so each D caches
side by side and a sweep recompiles each value once. There is deliberately **no GUI
control**: the depth has to be fixed before `initPrograms()`, so a checkbox could not
honour it. `guiStats()` shows the active D read-only.

## 6. What moves between runs and what does not

Same as `03` §6, and for the same reason. Construct MP/s ranged 551–700 for the
baseline and 813–1068 at D = 6 across five runs each; the structural counts were
byte-identical within every arm. The 10 ms launch cap moves how many batches land in a
launch, and GPU clocks drift. **The arms do not overlap** — the worst D = 6 run beat the
best baseline run — which is what makes the ordering safe to quote.

The phase sum is 99.0–99.3% of the `remolod.construct` CUevent total in every complete
arm, which is [plans/02](../plans/02_ProfilingTools.md)'s acceptance test for this
layer. D = 8 is 91.3%, because it spends 40 launches on 29 batches and the unmarked
prologue is paid every launch.

## 7. What it cost to build, and what broke

Four things went wrong that the plan did not anticipate. Each one is a silent failure
mode, and each is now checked:

- **The analytic cell index has to be *masked*, not clamped.** A point exactly on
  `boxMax` quantises to `2^MAX_DEPTH`; the descent it replaces absorbs that in its
  per-level `& 1`, which *wraps* it into cell 0 rather than pinning it to the last cell.
  Clamping left 30 leaves on morro_bay with `counter < numPoints`, so
  `allocatePointChunks` under-allocated them. The rule is that the index must be
  bit-for-bit what the descent computes, whatever the descent computes.
- **The node pool check needed a second half.** `doSplitting` refuses a split that would
  run past `MAX_NODES_CAPACITY`, but its grid-zeroing pass then still walked that node's
  `grid` — which the refusal had left null. The safety check was itself a fault until
  the zeroing pass learned to skip refused nodes.
- **Upstream's 200 MB memory safety margin is not enough for this arm.** It is tested
  only between batches, and `AllocatorGlobal` has no bounds check to fall back on. A
  fixed-depth batch splits 610 nodes at D = 8 against 14 in the baseline — 256 KB of
  occupancy grid each, plus a 16 KB chunk per new leaf — so one batch walks past the end
  of the store and faults in `allocatePointChunks`, with no warning printed. The fixed
  arm uses a 1 GB margin and stops cleanly instead.
- **The arm cannot be sized at 48 B/pt.** Every node that becomes inner takes 256 KB of
  occupancy grid that is never returned, so pre-splitting ratchets the store upward with
  D. At D = 7 the baseline's per-point estimate stopped the build at 58% of the cloud.
  The fixed arm therefore takes what the shared budget leaves.

Acceptance, all on the final binary:

| check | result |
|---|---|
| default RemoLOD structural counts | **4,137 / 12,742,751 — unchanged** |
| `simlod` | 4,137 / 12,742,751 — still identical to the default arm |
| `cudalod` | 2,252 / 12,742,500 — matches `bench/reference/` |
| `make check-vendored` | 9 kernels byte-identical |
| `make check-kernels` | 10 programs, 0 failed |
| fixed arm, points ingested | equals the baseline's, 36,200,706, at D = 5,6,7 |
| fixed arm, `spilled pts/batch` | 0 at every D |
| fixed arm, `expand iters/batch` | 1.00 (baseline 3.46) |
| leaf `counter >= numPoints` | ok in every arm, every run |
| `DeviceTimeline::overflow` | 0 in every run |
| `nodePoolOverflow` | 0 at every D here; set and survived at a forced 20k pool |

That last row is the one that needed contriving. The grid ratchet exhausts memory before
the 200k node pool is reached, so the overflow path was exercised by temporarily
rebuilding with `MAX_NODES_CAPACITY = 20'000`: the count pinned at exactly 20,000,
`node pool OVERFLOW` was reported, the leaf-counter invariant still held — leaves left
above D take the whole subtree's count from the pyramid — and nothing faulted.
`max pts/node` read 1,965,877, which is what a tree that stopped splitting looks like.

## 8. What this does not say

- **Not a quality result.** The arms build different trees — a third as many voxels at
  D = 5, four times as many at D = 7 — and no Chamfer, PCQM or PointSSIM number is
  produced here. [plans/01](../plans/01_NoveltyAssessment.md) §3 names those as required
  before any matched-budget claim. "Faster" travels with its counts and nothing more is
  claimed. In particular, D = 6 being 42% faster *while building 2.5× the nodes* is not
  the usual "building less tree is always faster": D = 5 is that, D = 6 is not.
- **Not a method.** The depth is free here. A real predictor pays for its own estimate
  and lands below this ceiling; if the estimate costs what the counting it replaces
  cost, none of this is reachable. That is Stage 4, and
  [plans/01](../plans/01_NoveltyAssessment.md) Recommendation 2 is what asks for it.
- **Not validated past 36M**, for the same memory reason as `03` §9.
- **The depth that wins is data-dependent.** D = 6 is the peak for morro_bay at this
  budget and nothing here says it transfers. A rule that picks 6 for this cloud and 7
  for another is the whole open problem, not a detail.
- **Grid memory is a one-way ratchet in D.** Every node that becomes inner takes 256 KB
  that is never returned, and collapsing would need a grid pool that does not exist. A
  sweep value that is too deep is not recoverable within a session.

## Related

- [03_ExpandStageFindings.md](03_ExpandStageFindings.md) — the 21.5% bound this spends.
- [plans/07_HardCodingExpandStage.md](../plans/07_HardCodingExpandStage.md) — the plan,
  and §12 for where it was wrong.
- [plans/01_NoveltyAssessment.md](../plans/01_NoveltyAssessment.md) §3 — why a
  density-only arm exists at all.
