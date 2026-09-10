# Accumulator Hook: Implementation Plan

> **Superseded in part — read [§10](#10-what-actually-landed-and-where-this-plan-was-wrong) first.**
> The accumulator shipped as a **separate kernel launch in `kernels/remolod/`**, not as a
> subpass of SimLOD's `kernel_construct`. Sections 1–9 are kept as originally written because
> the precision, normalisation and layout analysis in them is still exactly right and still
> describes the code. What changed is *where the pass runs* and *how it finds its points* —
> and, as a consequence, the file paths in §8 and the acceptance test in §7. §10 records all
> of it, including why the original shape broke the SimLOD baseline.

## TL;DR

- **The hook is the first half of the project.** `wiki/02_KernelPipeline.md` lists four kernels; two exist. The Analysis and Refinement kernels cannot be written until per-node geometry statistics exist, and this plan is the one thing that produces them. It computes nothing and decides nothing — it maintains running sums so that Analysis has something to roll up.
- **It goes in as a subpass of step 4 of the update kernel**, after `insertVoxels` in `addBatch` (`kernels/simlod/progressive_octree_voxels.cu:785`). Step 4 is the only place a point comes to rest exactly once, at exactly one node.
- **The accumulator is a side array indexed by node index, not fields on `Node`.** This is the one place this plan departs from the notes it grew out of, and it is not optional: `Node` is vendored, 152 bytes, `static_assert`ed at `structures.cuh:184`, and mirrored host-side as `kNodeBytes` (`simlod_layout.h:20`) to size the 200k-node pool. See §3.1.
- **Two details decide whether the output is usable at all.** Coordinates must be normalised to node-local space before accumulating, and the geometry sums must be `double`. Both come from the same fact: every metric is a covariance, `cov = S2/n - (S1/n)^2`, and the answer the LOD decision depends on is the *smallest* eigenvalue — a subtraction of two nearly equal numbers. §3.2 does the arithmetic.
- **Upstream already wrote the warp-aggregation idiom this needs**, at `progressive_octree_voxels.cu:215-221`: `cg::labeled_partition` over a `cg::coalesced_threads()` group, one atomic per group. Use that rather than raw `__match_any_sync` — it is correct for partially converged warps, which a hand-rolled butterfly reduction is not (§3.4).
- **Two premises in the notes are not satisfied by the current repo.** No reader sorts points into Morton order, so (a) the "most of a warp shares a leaf" argument for warp aggregation is weaker than stated and (b) the Morton watermark is not yet a stream watermark. Both are recorded in §3.5 rather than assumed away.
- **Verification is strong and already available.** The subpass does not mutate the tree, so `--dump-frame` must stay byte-identical across this change; and `Σ count` over leaves must equal `Stats::numPoints`. That pair is the acceptance test, which matters because `tests/unit/` is still empty.

---

## 1. What the hook is

*(This section is the original design note, kept, with call sites attached.)*

Right now, when a point arrives at its leaf, it does one atomic to reserve a slot
(`progressive_octree_voxels.cu:605`):

```cpp
uint32_t pointInNodeIndex = atomicAdd(&leaf->numPoints, 1);
// write point into chunk pointInNodeIndex/1000, offset pointInNodeIndex%1000
```

The hook adds a second thing at that same moment: the point folds its coordinates into a set
of running sums on that leaf. Ten sums — `n`, `Σx Σy Σz`, `Σxx Σxy Σxz Σyy Σyz Σzz` — plus six
more for colour variance. Nothing is computed, nothing is decided. Just totals.

Two extra stamps go alongside: set the leaf's `lastTouchedBatch` to the current batch number,
and max-reduce the point's Morton code into a global watermark.

**And accumulating only at leaves** rather than at every level along the traversal is what
keeps the cost at ten atomics per point rather than ten per point per level. Ancestors get
their statistics by summation later, which is exact because covariance sums are additive.

### 1.1 What this plan explicitly does not do

No scores, no thresholds, no split decisions, no roll-up, no tree mutation, no colour
filtering. Those are the Analysis and Refinement kernels. The separation is load-bearing —
CLAUDE.md: *"Analysis is cheap, unconditional and does not mutate the tree. Refinement is the
only thing that mutates it and the only thing that consumes budget."* This plan sits below
even that line: it is neither, it is the data both read. The `state` and `score` fields are
allocated here and written by nobody, so the Analysis kernel does not have to change the
layout to land.

---

## 2. Where it goes

### 2.1 Step 4, not step 1

*(Original note, kept.)* Step 1's counting sub-pass looks tempting since it is already
tallying, but it runs repeatedly — once per expansion iteration (`expand` loops up to 20
times, `:406`) — with existing don't-count-twice logic you would have to interact with
(`if(leaf->countIteration < countIteration)`, `:204`), on top of the spill buffer re-feeding
points from scratch. Step 4 runs exactly once per point per update, at the single node where
the point comes to rest. One pass, one accumulation, no special cases.

### 2.2 The exact call site

`addBatch` (`:712`) is six phases separated by `grid.sync()`. The subpass becomes a seventh:

| phase | line | wiki step |
|---|---|---|
| `expand` | `:738` | 1 — expand octree |
| `voxelSampling` | `:751` | 2 — voxel sampling |
| `allocatePointChunks` | `:761` | 3 — allocate chunks |
| `allocateVoxelChunks` | `:768` | 3 |
| `insertPoints` | `:775` | 4 — insert points |
| `insertVoxels` | `:785` | 4 — insert voxels |
| **`accumulate`** | **new, after `:787`** | **4 — subpass** |

The subpass reads only the tree topology and the two point arrays, so its true dependency is
`expand` — it could legally run anywhere after `:746`. It goes last for two reasons: the
existing `t_00`..`t_70` phase deltas (`:793-813`) keep their meaning and stay comparable to
runs already captured, and putting it adjacent to `insertPoints` keeps the option of fusing
it into that lambda later (§3.4).

### 2.3 Route through the vendored-code rule

CLAUDE.md permits two ways to hook vendored device code: *"a separate pass that re-traverses
the batch, or a `#ifdef`-guarded macro so the default build emits identical device code."*
This plan takes the first, because the accumulator is not instrumentation — it is the
feature, and a build variant that is off by default is a feature that never runs.

The edit to `progressive_octree_voxels.cu` is therefore deliberately shaped to be reviewable
at a glance: **one `#include`, two file-scope pointers, one call plus `grid.sync()` in
`addBatch`, two parameters on `kernel_construct`, and two counters in the existing stats
pass.** All logic lives in a new non-vendored header. No existing line of upstream algorithm
is touched — in particular `insertPoints` and `doSplitting` are not edited at all (§3.3
explains how clearing-on-split is achieved without touching the split).

The price is one extra descent per point per batch. That is the honest cost of route 1 and it
should be measured, not assumed: see §5 and the fallback in §3.4.

---

## 3. Design

### 3.1 A side array, not fields on `Node`

The original note proposed adding the sums to `Node`. That conflicts with a constraint
CLAUDE.md records as load-bearing, and the constraint wins:

- `Node` is vendored device code (`structures.cuh:108-178`), kept byte-identical so the port
  stays verifiable against `bench/reference/`.
- `sizeof(Node) == 152` is `static_assert`ed at `structures.cuh:184`.
- The host cannot include `structures.cuh` (its methods call `dot()` from `helper_math.h`), so
  the size is mirrored as `simlod::kNodeBytes` at `simlod_layout.h:20` and used to size the
  node pool at `SimlodPipeline.cpp:117`. Widening `Node` means editing upstream code, one
  `static_assert` and one host constant, in step.

So the accumulator is `NodeAccum accums[MAX_NODES_CAPACITY]`, indexed by **node index**
(`node - nodes`), allocated by the host and passed to the kernel. Every place that needs it
already has both the node pointer and the pool base: `addBatch` receives `nodes`, and the
stats pass at `:989` already iterates the pool *by index*.

This is strictly better than the original proposal in three ways, not merely compliant: the
pool stays at exactly `kMaxNodes * 152` so an existing capture's node pool is bit-comparable,
`Node` stays 152 bytes and therefore stays 8-per-cache-line for the render kernel's hot
traversal, and the accumulator can be reallocated or dropped entirely without touching the
tree.

The note's remark that *"nodes come from the same persistent buffer, so nothing about
allocation changes"* was also inaccurate for a different reason: the node pool is its own
`cuMemAlloc` (`SimlodPipeline.cpp:149`), separate from the persistent octree store.

### 3.2 Normalisation, and the precision it forces

*(Original note, kept.)* **Normalise the coordinates first.** Transform the point to
node-local `[0,1]` before accumulating. If you sum raw world coordinates — UTM eastings in
the hundreds of thousands — then `Σxx` in float32 catastrophically loses the small
differences, and the small eigenvalue is precisely what every metric depends on. This is the
single most important detail in the hook.

Two things follow that the note did not carry.

**Where the normalisation constants come from.** A leaf knows its own cell: `level`, `X`, `Y`,
`Z`, set at split time (`:341-344`). With `octreeMin`/`octreeSize` already in hand:

```
local = (p - octreeMin) / octreeSize * 2^level - (X,Y,Z)
```

Compute that in `double` and narrow the result to `float`. The subtraction is a cancellation
at depth — at level 15 the left term is ~32768 and the answer is in `[0,1]`, which float32
resolves to about 2e-3 of a node — and a `double` reciprocal multiply costs single-digit
microseconds over a 36M-point cloud. Do **not** be tempted to reuse the integer `X,Y,Z` at
`MAX_DEPTH` that the traversal already computed: those quantise the point to `2^-20` of the
root cube (1.3 mm on morro_bay), which is a floor on the smallest eigenvalue that shallow
nodes cannot see past.

Clamp the result to `[0,1]`. The float→uint32 truncation that picks the cell (`:567-569`) and
this normalisation are not the same arithmetic, so a point exactly on a cell boundary can
land marginally outside; clamping is one instruction and prevents a garbage term entering a
sum that is never recomputed.

**The geometry sums must be `double`.** The note budgeted 56 bytes, which is float32. Work
the error through for the case that matters — a locally planar leaf, 50k points, node-local
coordinates, and the plane roughly axis-aligned so `z ≈ 0.5`:

- `Σzz ≈ 0.25n = 12500`, and `Σz/n ≈ 0.5`.
- The variance being extracted is `Σzz/n - (Σz/n)^2`, and for a plane 1e-3 of the node thick
  that is about `1e-6`. So `Σzz` is needed to an absolute accuracy of `n * 1e-6 = 0.05`.
- float32 atomic accumulation of 50k terms into a sum of 12500 carries an absolute error on
  the order of `sqrt(n) * eps * Σ ≈ 224 * 6e-8 * 12500 ≈ 0.17`.

The error is three times the signal. In `double` the same estimate is ~1e-11, which is 1e9 of
margin. `atomicAdd(double*)` is native on sm_60+, and the compile arch is queried from the
device (`CudaModularProgram.cpp:252`), so this is not a portability question in practice —
but `--check-kernels` is where it gets confirmed.

Colour is the opposite case: the input is 8 bits per channel and the metric is a filtering
decision, so float32 sums keep the note's 24-byte budget with three orders of magnitude to
spare. The partial sums *inside* a warp reduction can also stay float — at most 32 terms, so
the reduction contributes ~1e-5 absolute against the 0.05 that matters (§3.4).

### 3.3 Clearing on split, without editing the split

*(Original note, kept.)* **Zero the accumulator when a leaf splits.** In step 1, a leaf that
spills becomes an inner node. Its accumulator is now meaningless, because inner-node
statistics are derived by roll-up later, not stored. The spilled points get re-inserted
through step 4 into new leaves and accumulate there naturally, so there is no double-counting
to reason about — the spill buffer handles itself.

That last claim was worth checking against the code, and it holds. `doCounting:265-301` walks
every point in every chunk of each spilling node into `spilledPoints`; `doSplitting:371-372`
then sets `numPoints = 0` and `points = nullptr`. A freshly created child has `numPoints == 0`,
so a child that itself spills during a later `expand` iteration contributes nothing to
`spilledPoints` — each point is spilled at most once per batch, and `insertPoints:638-648`
re-inserts each exactly once. The subpass mirrors that by processing both arrays.

**Where to do the clearing.** Not in `doSplitting`. Instead the subpass opens with a pass over
`stats->numNodes` that clears the entry of any node that is no longer a leaf:

```
if (!node->isLeafFn() && accums[i].count != 0) clear(accums[i]);
```

Three reasons this is better than an edit at the split site. It keeps `doSplitting` untouched,
which keeps §2.3's diff to one call. It is idempotent and self-healing, so a clear that is
somehow missed is corrected on the next batch rather than silently poisoning a roll-up. And it
turns the invariant into something observable: the same pass can count inner nodes that still
hold sums, which is exactly the "must be zero" check in §7. The cost is one pass over at most
200k nodes per batch, against ~1M points of real work.

The pass must run before the accumulation and after `expand`, which §2.2's placement gives
for free.

### 3.4 Warp aggregation — use the idiom already in the file

*(Original note.)* **Reduce within the warp before going to global.** Ten atomicAdds per point
is a real multiplier on the hottest loop. Find the lanes in a warp that landed in the same
leaf, reduce among them, and have one lane issue the atomic for the group.

The note reaches for `__match_any_sync`. Do not hand-roll the reduction on top of it, for a
reason that is easy to get wrong and produces plausible-but-wrong sums rather than a crash:
**a guarded butterfly (`__shfl_xor_sync`) over an arbitrary lane subset does not converge to
the group total.** Take the mask `{lane 0, lane 3}` — at every step each lane's partner is
outside the mask, so both lanes keep only their own value and the total exists nowhere. Any
warp that is not fully converged, including the tail iteration of a grid-stride loop, can
produce such a mask.

Upstream already solved this, in this very file, for exactly this operation
(`progressive_octree_voxels.cu:215-221`):

```cpp
uint64_t leafptr = uint64_t(leaf);
auto warp  = cg::coalesced_threads();
auto group = cg::labeled_partition(warp, leafptr);
if (group.thread_rank() == 0) {
    old = atomicAdd(&leaf->counter, group.num_threads());
}
```

The subpass uses the same construction with `cg::reduce(group, term, cg::plus<float>())` from
`<cooperative_groups/reduce.h>` for each of the 15 sums, `group.num_threads()` for `n`, and
`group.thread_rank() == 0` issuing the atomics. `labeled_partition` is built on
`coalesced_threads()`, so it is correct for whatever subset of the warp is actually
converged — the failure mode above cannot occur — and it matches the code a reader of this
file has already seen. It also removes the need to restructure `processRange` for
convergence, which a hand-rolled version would have required.

Cost shape: 15 `cg::reduce` calls (about five shuffles each) plus 17 atomics per *group*,
against 17 atomics per *point* unaggregated. Add the early out `if (group.num_threads() == 1)`
— issue the atomics directly and skip the reductions — so the all-distinct-leaves case does
not pay 75 shuffles to save nothing.

**Two fallbacks, in order, if measurement says the subpass is too expensive.** Both are in the
note; the ordering is the addition.

1. **Subsample.** Accumulate from every Nth point. Covariance from 3k of 50k points is
   statistically indistinguishable, and the metrics are only steering recursion. This is a
   one-line change and it also removes the extra descent's cost proportionally, which is why
   it comes first.
2. **Fuse into `insertPoints`.** Move the accumulation into the `insertPoint` lambda at `:605`,
   where the leaf and the traversal are already in hand, and delete the re-traversal. This
   costs the clean vendored diff of §2.3 and is the reason it is second, not first.

### 3.5 The Morton watermark, and the premise it is missing

The watermark is the mechanism the Analysis kernel's "advance the watermark / test closure"
step (`wiki/02_KernelPipeline.md:13-14`) is built on: if the stream arrives in Morton order,
then once the watermark passes a node's Morton range, that node can receive no further points
and its statistics are final.

Implement it as specified — `atomicMax` of the point's `MAX_DEPTH` Morton code into one global
— with two constraints:

- **Interleave with X most significant**, i.e. `(part(X) << 2) | (part(Y) << 1) | part(Z)`, to
  match `childIndex = (child_X << 2) | (child_Y << 1) | child_Z` (`:593`). Otherwise Morton
  order is not the octree's own child order and the closure test compares incomparable things.
- **Reduce per group before the atomic.** One `atomicMax` per point on a single global address
  is 36M serialised atomics on one cache line. Fold it into the same `labeled_partition` group
  as everything else, or take the warp max — either way it becomes one atomic per group.

**The premise, recorded rather than assumed: nothing in RemoBench sorts points into Morton
order.** `grep -niE "morton|sort" src/io/` finds nothing relevant; every reader hands points
to the device in file order, which for `.las`/`.laz` is typically acquisition order along
flight lines. So:

- The watermark is well-defined and cheap, but it is **not yet a closure oracle**. Building
  the closure test on it requires either a Morton-ordering stage in the loader or a different
  closure criterion. That decision belongs to the Analysis plan; this plan only guarantees the
  watermark is maintained and correctly oriented.
- The same fact weakens the note's warp-aggregation argument, which assumed Morton-ordered
  input gives a warp shared leaves. There is still real locality — acquisition order is
  spatially coherent, and `processRange` hands each *thread* a contiguous run of points
  (`utils.h.cu:63-73`), so a warp spans a window of ~1000 consecutive input points, small
  against a 50k-point leaf. But the aggregation ratio is an empirical question now, not a
  given, which is why §7 measures it.

---

## 4. Data layout

New host/device shared header, `include/remo/SimlodAccum.h`. It must follow the two rules
`HostDeviceCommon.h:1-25` states, plus a third that is specific to being included by SimLOD's
vendored code:

**No includes at all, and no imported type names.** `progressive_octree_voxels.cu` gets its
fixed-width types from `utils.h.cu:24-31`, where `uint64_t` is `typedef unsigned long long`.
CCCL's `cuda::std::uint64_t` is `unsigned long` on Linux — a *different type* — so bringing
the CCCL names into global scope in that translation unit is a conflicting-typedef error. This
is the same trap `patches/cudalod-linux-port.patch` exists to fix and that
`HostDeviceCommon.h:29-38` warns about. Spelling out `double`, `float`, `unsigned int` and
`unsigned long long` sidesteps the question entirely.

```cpp
enum GeomSum  { kSumX=0, kSumY, kSumZ, kSumXX, kSumXY, kSumXZ, kSumYY, kSumYZ, kSumZZ,
                kNumGeomSums };
enum ColorSum { kSumR=0, kSumG, kSumB, kSumRR, kSumGG, kSumBB, kNumColorSums };

enum AccumState : unsigned int {
    kAccumOpen      = 0,  // still receiving points; sums are partial
    kAccumClosed    = 1,  // watermark has passed; no more points due   (Analysis writes this)
    kAccumFinalized = 2,  // roll-up done, `score` is meaningful        (Analysis writes this)
};

struct NodeAccum {                    // 112 B, one per node in the flat pool
    double       g[kNumGeomSums];     // 72  Sx Sy Sz Sxx Sxy Sxz Syy Syz Szz, node-local [0,1]
    float        c[kNumColorSums];    // 24  Sr Sg Sb Srr Sgg Sbb, channels in [0,1]
    unsigned int count;               //     n
    unsigned int lastTouchedBatch;
    unsigned int state;               //     AccumState
    float        score;               //     Analysis writes this; 0 until then
};

struct AccumGlobals {                 // 32 B, read back once per launch
    unsigned long long mortonWatermark;
    unsigned long long numAccumulated; // verification only, see §7
    unsigned long long sumLeafCounts;  //   "
    unsigned int       innerWithSums;  //   "
    unsigned int       pad0;
};
```

`static_assert` both sizes in the header so the host and device cannot drift — the same
discipline `simlod_layout.h` applies to `Node`.

Arrays rather than named fields for the sums, because the reduction in §3.4 and the roll-up in
the Analysis kernel both want to iterate them, and 15 named fields would be 15 copies of every
loop body. The enums keep the call sites readable.

---

## 5. Cost and memory

| item | figure | note |
|---|---|---|
| accumulator array | **22.4 MB** | `200'000 × 112 B`, sized by `MAX_NODES_CAPACITY` |
| globals | 32 B | |
| atomics per point, unaggregated | 17 | 9 geometry + 6 colour + `count` + `lastTouchedBatch` |
| atomics per *group*, aggregated | 17 | plus ~75 shuffles; break-even at group size ≈ 2 |
| extra work per point | one octree descent | the price of route 1 (§2.3) |

22.4 MB must be added to the `fixed` term of the device-memory check at
`SimlodPipeline.cpp:124` and to `m_stats.bytesAllocated`, or the pipeline will over-commit a
tight budget by exactly the amount it forgot to declare. It is not a memory-wall concern in
the sense CLAUDE.md warns about — that concern is per-*cell* accumulators at 16 MB *per node*;
this is 112 bytes per node, total.

The array is sized to the pool, not to the live node count, because `numNodes` is a bump index
(`:329`) and a side array that grows would need the capacity check the node pool itself does
not have.

---

## 6. Staging

**Stage 1 — layout and plumbing, no accumulation.** `SimlodAccum.h`, host allocation, zeroing
on reset, the two new `kernel_construct` parameters, and the clear-inner-nodes pass. Acceptance:
`--check-kernels` passes, `--dump-frame` is byte-identical to a pre-change capture, and
`AccumGlobals` reads back all zeros. This stage is worth having on its own because it isolates
"did adding parameters to the vendored kernel break anything" from "is the arithmetic right".

**Stage 2 — accumulation, unaggregated.** The descent, the normalisation, the per-point
atomics, `lastTouchedBatch`, the watermark. Deliberately without warp aggregation: it is the
simplest correct version and it is the reference the aggregated version is checked against.
Acceptance: §7's two invariants hold.

**Stage 3 — warp aggregation.** `labeled_partition` + `cg::reduce`, with the group-size-1 early
out. Acceptance: `count` and `Σ` are unchanged from Stage 2 (bit-identical for `count`,
within float tolerance for the sums), and `simlod.construct` gets faster. If it does not get
faster, measure the group-size distribution before reaching for the fallbacks in §3.4 — with
unsorted input (§3.5) the answer may simply be that there is nothing to aggregate.

**Stage 4 — expose it.** A GUI row for the invariants and the watermark, and whatever the
Analysis plan needs to read. Small, but this is where the layer stops being invisible.

Stages 2 and 3 are where the whole risk is, and both are validated by the same two checks.

---

## 7. Verification

There is no test suite (`tests/unit/` is empty, `REMOBENCH_BUILD_TESTS` is OFF), so the checks
have to come from existing tooling. Three of them are strong.

**1. The tree must not change.** The subpass writes only to the side array; it does not touch
`Node`, the chunks, the grids or the allocators. So `--dump-frame` must produce a
**byte-identical** frame before and after, and `Stats` must be unchanged field for field. This
is the same verification the readers get (CLAUDE.md: *"bit-identical trees and byte-identical
frames"*), and it is the single best test available here because it catches an accidental
mutation of vendored state, which is the failure mode that would be hardest to find later.

```sh
# before the change
./build/remobench --pipeline simlod --open data/morro_bay_35M/morro_bay_36M.simlod \
    --dump-frame /tmp/base.ppm
# after
./build/remobench --pipeline simlod --open data/morro_bay_35M/morro_bay_36M.simlod \
    --dump-frame /tmp/accum.ppm
cmp /tmp/base.ppm /tmp/accum.ppm
```

**2. `Σ count` over leaves must equal `Stats::numPoints`, and no inner node may hold sums.**
Both are computed by the stats pass that already walks the pool by index at `:989-1006` — two
more counters in the two branches it already has, written into `AccumGlobals`. `numPoints` is
`Σ node->numPoints` over leaves, so this checks the accumulation against the *insertion* it
shadows, per point, over the whole cloud. `innerWithSums` checks §3.3. Anything that
double-counts, misses the spill buffer, or clears the wrong entry shows up in one of the two.

**3. `--check-kernels`.** Non-negotiable after touching anything under `kernels/` — NVRTC
compiles at runtime, so a kernel error is invisible to `make`. It is also what confirms
`atomicAdd(double*)`, `cg::reduce` and `<cooperative_groups/reduce.h>` are available under the
NVRTC options this repo passes.

Then the arithmetic itself, which the invariants above cannot check:

- **A synthetic sanity case.** A leaf whose points lie in a plane must give a smallest
  eigenvalue near zero and two large ones; a leaf filled uniformly must give three comparable
  ones. Note `--synthetic` currently faults CudaLOD (CLAUDE.md, known traps) but SimLOD is
  fine, so this is reachable from the command line.
- **Normalisation, deliberately adversarially.** Run a UTM cloud, since that is the case §3.2
  exists for, and confirm the eigenvalues of a deep node are not dominated by a constant floor
  — the signature of either un-normalised sums or the `MAX_DEPTH`-quantisation mistake §3.2
  warns against.
- **Aggregation ratio.** `numAccumulated / atomics issued` — worth a counter during Stage 3,
  because §3.5 means the expected value is unknown rather than "most of a warp".

Finally, and per CLAUDE.md's reporting rules: any construct-time number quoted after this
lands names the batch size, the device-side budget, the timing regime, **and** whether the
accumulator was on. The subpass adds unconditional work to `simlod.construct`, so every
SimLOD throughput figure in `bench/reference/` predates it and is not comparable without
saying so.

---

## 8. Files

**New**

| path | contents |
|---|---|
| `include/remo/SimlodAccum.h` | `NodeAccum`, `AccumGlobals`, the enums, the `static_assert`s (§4) |
| `kernels/simlod/remo_accum.cuh` | the subpass: clear-inner pass, descent, normalisation, `labeled_partition` reduction, Morton. **RemoBench's own file, not vendored** — carries no upstream header, unlike its neighbours |

**Modified**

| path | change |
|---|---|
| `kernels/simlod/progressive_octree_voxels.cu` | one `#include`, two file-scope pointers beside `backlog_voxels` (`:41-45`), two `kernel_construct` parameters, the call + `grid.sync()` after `:787`, two counters in the stats pass at `:989`, one extra phase in the `t_*` print |
| `src/pipelines/SimlodPipeline.h` | `m_accums`, `m_accumBytes`, `m_accumGlobals`, and a cached `AccumGlobals` for the GUI |
| `src/pipelines/SimlodPipeline.cpp` | allocate and free the array; add it to the `fixed` budget term (`:124`) and to `bytesAllocated`; `cuMemsetD8` both on reset; two more launch args; read `AccumGlobals` in `readStats()`; invariant rows in `gui()` |

**Deliberately not modified**

- `kernels/simlod/structures.cuh` — `Node` stays 152 bytes (§3.1).
- `kernels/simlod/simlod_layout.h` — `kNodeBytes` unchanged; the host gets the accumulator
  layout from the real struct in `SimlodAccum.h`, so no mirrored constant is needed.
- `kernels/simlod/reset.cu` — the array is zeroed host-side with `cuMemsetD8`. Reset is
  launched with **one block and one thread** (`SimlodPipeline.cpp:268`), so a `processRange`
  over 200k entries there would be a serial loop for no reason.
- `insertPoints`, `doSplitting`, `doCounting` — untouched (§2.3, §3.3).

**Reused rather than rebuilt**

- `cg::labeled_partition` over `cg::coalesced_threads()` (`progressive_octree_voxels.cu:215-221`)
  — the aggregation idiom; do not write a second one.
- `processRange` (`utils.h.cu:79-96`) — and its block-contiguous distribution, which §3.5
  depends on for locality. Do not "fix" it here; CLAUDE.md's shared-path rule and
  `remo_prelude.cuh:38-46` both explain why it is the way it is.
- The stats pass at `:989-1006` — already walks the pool by index; the invariants of §7 are
  two counters inside it, not a new pass.
- `GpuScope` / `simlod.construct` — no new timing scope. The subpass is inside the existing
  megakernel, so it is already inside the scope; intra-kernel attribution for it is Stage 3 of
  `plans/02_ProfilingTools.md`, and the `t_*` marks it adds are where that will pick it up.

---

## 9. Decisions deferred to the Analysis plan

Recorded here because this plan's layout has to leave room for them, not because it answers
them.

- **What closure actually tests**, given that the input is not Morton-ordered (§3.5). Either
  the loader gains an ordering stage, or `lastTouchedBatch` plus a batch-age margin does the
  job without a watermark at all. The `state` field is allocated for either.
- **Where the roll-up result lives.** Inner nodes have a `NodeAccum` entry that this plan keeps
  at zero. Whether the roll-up writes into it, or into a second array, or is recomputed on
  demand, is an Analysis decision — but the entry existing means it costs no layout change.
- **Whether colour sums belong here at all.** They are cheap and CLAUDE.md's colour-averaging
  note says a sparse accumulator keyed off the voxel backlog is what filtering will actually
  need. Six float sums per node may turn out to be the wrong shape; they are 24 bytes of a
  112-byte struct, so carrying them until Refinement says otherwise is the cheap option.
- **Subsampling as policy rather than fallback.** §3.4 lists it as a performance escape hatch,
  but if Analysis only ever uses these sums to steer recursion, a fixed stride may be the
  right default and the full-rate accumulation the special case.

---

## 10. What actually landed, and where this plan was wrong

*Added after implementation. The plan above is kept as written; this section records where it
was superseded, because the reasoning it was superseded by is the useful part.*

### 10.1 The hook became a separate kernel

§2.3 chose "a separate pass that re-traverses the batch" over an `#ifdef`-guarded macro. Both
options in that pair assumed the pass would live **inside** `addBatch`, as a seventh phase of
`kernel_construct`. That is what shipped first, and it was wrong twice over.

**It broke the baseline.** Adding the subpass meant adding two parameters to
`kernel_construct` — a vendored kernel. The host launch was never updated to pass them.
`cuLaunchCooperativeKernel` reads one entry per declared parameter, found ten where the kernel
wanted twelve, and failed **every** launch with `CUDA_ERROR_INVALID_VALUE`. The SimLOD
pipeline built no tree at all — 0 points, 1 node — for a whole commit, while `make` succeeded,
`--check-kernels` passed and `--dump-frame` exited 0 with a valid image of an empty tree.

The rule that came out of it: `simlod` and `cudalod` are **external comparison baselines** and
are never edited. RemoLOD forks what it needs into `kernels/remolod/`.
`bench/check_vendored.sh` asserts it mechanically.

**And the separate launch is simply better.** `kernel_accumulate` runs after
`kernel_construct` returns and reads the finished tree:

| | the hook (§1–§4) | what landed |
| --- | --- | --- |
| finding a point's leaf | descend 20 levels from the root, per point per batch | it is already in the leaf's chunk list |
| atomics | 15 per point, warp-aggregated to survive it | one uncontended read-modify-write per leaf |
| warp aggregation | required (`labeled_partition` over `coalesced_threads`) | not needed; a block owns a whole leaf |
| spill buffer | must be re-traversed explicitly | never seen; split children just walk from 0 |
| `batchIndex` | needed | only for `lastTouchedBatch` |
| timing | buried inside `simlod.construct` | its own `remolod.accumulate` scope |
| baseline | edits vendored code | touches nothing |

§3.4's warp-aggregation analysis is therefore moot, and so is the concern in §3.5 that
aggregation would underperform on non-Morton-ordered input — there is nothing left to
aggregate.

### 10.2 `count` is the watermark

The plan had no incremental mechanism; the subpass folded each batch's points as they were
inserted. The separate pass needs one, and `NodeAccum::count` already is one:
`insertPoints` stores points densely at `[0, numPoints)` and only ever appends, so folding
`[count, node->numPoints)` and setting `count = numPoints` is exact.

This also answers §3.3 more cleanly than the plan does. A split sets the parent to
`numPoints = 0` and hands its points to fresh children, so clearing the parent (which the pass
does anyway, because inner nodes hold no sums) and letting the children walk from `count == 0`
re-accumulates the redistributed points exactly once. No double-counting to reason about, and
a launch that is skipped or interrupted just leaves more to do next time.

### 10.3 The acceptance test lost half of itself

§7's first check — `--dump-frame` byte-identical across the change — **is not available**, and
never was. Only `flat` is run-to-run deterministic. Every octree pipeline colours a voxel from
the first point to reach its cell, so which *thread* wins decides the colour and two runs of
the same binary on the same file differ. This is first-come sampling doing exactly what
`README.md` criticises it for; it is one of the things colour filtering is meant to fix.

What replaces it: **structural counts identical with the pass on and off**, via
`--remolod-no-accum` (a real flag, because a GUI-only toggle would make the test
unscriptable). §7's second check survives intact and is the strong one.

Measured on `morro_bay_36M`:

```
accum sum/points    36200706 / 36200706  ok
accum inner w/ sums 0  ok
accum folded total  37540263
remolod.construct   n=4  med 10.75 ms
remolod.accumulate  n=4  med  2.49 ms
```

Structural counts are identical with the accumulator on and off (36,200,706 points /
12,742,751 voxels / 4,137 nodes), and `remolod.construct` is unchanged between the two
(41.09 ms vs 41.06 ms total) — the pass adds no cost to construction, only its own launch.

`folded total` exceeding `numPoints` by 3–5% is the re-walk after splits, and is the number to
watch if the pass ever looks expensive: it is the only redundant work in it. It varies between
runs (37.5M–37.9M observed) because each time-budgeted construct launch consumes a different
number of batches, so the splits fall in different places. `sum/points` is exact regardless,
which is the point of having both.

### 10.4 `AccumGlobals::numAtomicGroups` became `numLeavesFolded`

§4's departure note added `numAtomicGroups` to measure the warp-aggregation ratio. With no
warp aggregation left there is nothing to measure, so the field is now `numLeavesFolded` plus
`pointsFolded`, both describing the launch that just ran. Same 40-byte layout.

### 10.5 Staging, as executed

§6's Stage 1/2/3 split assumed the risky part was the arithmetic under aggregation. In the end
Stages 2 and 3 collapsed into one — the unaggregated reference implementation *is* the block
reduction, there is no second version to check it against — and the work that mattered was
undoing the vendored edit. Stage 4 (exposing it) landed as `diagnostics()` on `ILodPipeline`,
so `--dump-frame` prints the invariants for any pipeline that has some.

### 10.6 Still open

- §9's list is untouched; all four are still Analysis decisions.
- The Morton watermark is maintained and correctly oriented, and is still **not a closure
  oracle** — nothing sorts points into Morton order. §3.5 stands.
- `lastTouchedBatch` is written but read by nobody, as designed.
