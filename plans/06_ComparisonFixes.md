# Comparison Fixes: the 350M Blocker

## TL;DR

- **`data/morro_bay_350M/` is refused by `remolod` and `simlod`, and the refusal message is wrong by 2.5×.** `PipelineRegistry::fits()` multiplies `bytesPerPointEstimate` (64 B/pt) by 350,360,028 points and reports 22.4 GB. The SimLOD paper measured **9.1 GB** for this exact cloud (Table 5, §6.6 — 26 B/pt). The estimate is a guess that was never checked against the paper it is reproducing.
- **But relaxing the constant would only move the failure from the gate to `cuMemAlloc`,** because RemoBench holds the *whole cloud resident* (5.6 GB) on top of the tree. Upstream does not: it streams through a 50-slot × 1M-point ring and never has more than 800 MB of input on the device. The honest requirement here is ~14.7 GB, not 9.1.
- **The fix is the wrapping ring `PointSource` was designed for and never got.** `wiki/01_Overview.md:184-192` already documents the addressing; `Mode::Stream`, `BatchView::wrapping`, `setPointsConsumed()` and `needsWholeCloudResident` are all present and all unwired. The device side needs *no change at all* — the async-producer handshake is already implemented at `progressive_octree_voxels.cu:875-881`.
- **The documented `Mode::Whole` addressing is not achievable, and that is the 50M cap.** `kernel_construct` hardcodes `batchIndex % BATCH_STREAM_SIZE` (`progressive_octree_voxels.cu:897`), so the kernel *always* wraps. A non-wrapping resident feed is correct only while `k < BATCH_STREAM_SIZE`. `BatchView::wrapping` can never be honoured, and no consumer reads it.
- **RemoLOD's fork constant comes back *down*, not up.** `BATCH_STREAM_SIZE = 8192` (`remolod_structures.cuh:21`) is 131 GB as a real ring; it only makes sense under the current non-wrapping feed. Once the ring exists, 50 is correct for both — which *removes* a divergence and leaves the RemoLOD/SimLOD tree equality as a free check on the ring.
- **Only the gate is fatal; allocation already degrades gracefully.** `m_persistentBytes = std::min(wantPersistent, available - fixed)` clamps to what is left and `memCapacityReached` stops ingest cleanly. `fits()` refuses before that clamp ever runs, because one constant is doing two jobs: the refusal threshold and the allocation coefficient.
- **Honest expected outcome: 350M stops being refused; it completes only on a freed-up GPU.** The full tree needs a **10.48 GB budget**, which needs ~**12.0 GiB free** of this card's 15.92 GiB. At today's ~8.1 GiB free the budget is 6.94 GB and the build reaches **~61%** of points before `memCapacityReached` — against **~8%** on the resident path. That 7× is the real deliverable on this machine; "350M completes" needs ~4 GiB handed back by the desktop. See §2.8.

---

## 1. Why

### 1.1 The abstraction is documented, and it is unachievable

`wiki/01_Overview.md:184-192` states the ingest seam as a "small trick":

```text
Mode::Stream → deviceAddr(k) = ringBase     + (k % numSlots) * slotBytes
Mode::Whole  → deviceAddr(k) = residentBase +  k             * slotBytes
```

The second line cannot hold. `kernel_construct` computes the slot itself, at
`kernels/simlod/progressive_octree_voxels.cu:897-899`:

```cpp
		uint32_t ringSlotIndex = batchIndex % BATCH_STREAM_SIZE;
		uint32_t batchSize = batchSizes[ringSlotIndex];
		Point* sub_points = points + ringSlotIndex * MAX_BATCH_SIZE;
```

The kernel *always* wraps. Feeding it a flat resident cloud agrees with that arithmetic
only while `batchIndex < BATCH_STREAM_SIZE` — which is precisely the 50M ceiling
`kernels/simlod/simlod_layout.h:41-60` documents and `PipelineRegistry.cpp:67-79`
enforces. `BatchView::wrapping` (`include/remo/PointSource.h:41`) defaults to `true`, is
forced to `false` at `src/io/PointSource.cpp:110`, and **is read by no consumer**, because
there is nothing a consumer could do with it: the vendored kernel has no non-wrapping mode
to select.

So the wiki describes one design, the kernel implements another, and the gate hides the
difference. That is the defect underneath the 350M refusal.

### 1.2 The scaffolding is all there and none of it is wired

| artifact | location | state |
|---|---|---|
| `Mode::Stream` | `include/remo/PointSource.h:46` | declared; referenced nowhere in the repo |
| `BatchView::wrapping` | `PointSource.h:41` | defaulted for a ring; forced false; no readers |
| `setPointsConsumed(uint64_t)` | `PointSource.h:62` | no-op body at `PointSource.cpp:119`; zero call sites |
| `numPointsUploaded()` | `PointSource.h:57` | implemented at `PointSource.cpp:114`; zero call sites |
| `rewind()` | `PointSource.h:53` | empty body; called from `PipelineRegistry.cpp:121,138` |
| `needsWholeCloudResident` | `include/remo/ILodPipeline.h:25` | set by all four pipelines; **read by none** |
| `numSlots` vs `numBatchesTotal` | `PointSource.h:39-40` | two fields, set equal — they diverge exactly when a ring exists |

The device half is further along than the host half. The async-producer handshake is
already written, at `progressive_octree_voxels.cu:875-881` — thread 0 reads
`*_numBatchesUploaded_volatile` into a grid-wide global, `grid.sync()`, then every thread
reads it back through `atomicAdd(..., 0)`, with a comment explaining that the value "could
change at any moment by the parallel upload stream". There is no parallel upload stream.
The host publishes the final count once, at `src/io/PointSource.cpp:152`:

```cpp
		if (REMO_CU(cuMemsetD32(m_numBatchesUploaded, numSlots, 1)) != CUDA_SUCCESS) {
```

so from the kernel's point of view the entire cloud arrived on frame 1. **This plan
therefore changes no device code except one constant in RemoLOD's own fork.**

### 1.3 One constant is doing two jobs, and only one of them is fatal

`kBytesPerPointPersistent = 48.0` (`SimlodPipeline.cpp:28`, `RemolodPipeline.cpp:28`) is
both the gate estimate —

```cpp
	info.bytesPerPointEstimate = kBytesPerPointPersistent + 16.0;   // :45
```

— and the actual allocation coefficient:

```cpp
	uint64_t wantPersistent = static_cast<uint64_t>(
		kBytesPerPointPersistent * static_cast<double>(meta.numPoints));
	wantPersistent = std::max(wantPersistent, kMinPersistentBytes);
	...
	m_persistentBytes = std::min(wantPersistent, available - fixed);  // :110
```

The allocation path is already graceful: it takes what it can get, and the device reports
`memCapacityReached` when the persistent allocator runs dry, which stops ingest cleanly
rather than corrupting a tree. The *gate* is the only hard failure, and it tests the
pipeline's appetite rather than its floor. Tuning the constant to fix the gate would
silently shrink every persistent buffer; tuning it to fix allocation would loosen the gate
past what the machine can do. They need to be two numbers.

For scale, the paper's own figure for this cloud (Table 5):

| data set | points | voxels | input | voxels | occupancy grids | total | increase |
|---|---|---|---|---|---|---|---|
| Morro Bay | 350 M | 128 M | 5.8 GB | 2.1 GB | 1.3 GB | **9.1 GB** | ×1.6 |
| Meroe | 684 M | 209 M | 11.3 GB | 3.4 GB | 2.5 GB | 17.2 GB | ×1.5 |
| Endeavor | 796 M | 318 M | 13.2 GB | 5.2 GB | 2.7 GB | 21.1 GB | ×1.6 |

9.1 GB / 350 M = **26 B/pt**, including unused chunk capacity, on an RTX 4090. Note the
5.8 GB "input" row is the octree's *own* copy of the points inside the persistent buffer —
not a resident input cloud, which upstream never has.

### 1.4 The resident cloud is the real cost, and it is RemoBench's own choice

Every budget-aware pipeline subtracts the resident input up front, identically
(`SimlodPipeline.cpp:90-92`, `RemolodPipeline.cpp:93-95`, `CudalodPipeline.cpp:100`):

```cpp
	const uint64_t inputBytes = meta.numPoints * 16ull;
	const uint64_t available =
		budget.bytes > inputBytes ? budget.bytes - inputBytes : 0;
```

`ResidentSource::start()` allocates and uploads the whole cloud in one shot
(`PointSource.cpp:46,60-70`) and then drops the host copy
(`m_points.clear(); m_points.shrink_to_fit();`, `:83-84`) — so nothing is even left to
re-stream from. For 350M that is 5.6 GB of device memory that upstream never spends, on
top of the 9.1 GB tree: **~14.7 GB**, against a card with 15.9 GB and a budget of 9.

This is the line in `wiki/01_Overview.md:223` that reads `residency | whole cloud (for
now)` for `remolod`. The "for now" is this plan.

### 1.5 Bounds correction is a whole-file operation

`loadLasCloud` (`src/io/PointSource.cpp:191-264`) performs two fixups that a streaming
loader cannot afford, both of which require having seen every point:

1. **A full re-read** when the header's declared minimum sits above its own points — the
   `for (int attempt = 0; attempt < 2; ++attempt)` loop at `:208`. Translation is baked
   into `Point` at read time (`:217-219`), so a corrected translation means re-reading.
2. **Growing the box after the fact** when points exceed the header maximum, at `:250`:
   `meta.boxSize[i] = static_cast<float>(bounds[3 + i]);`

Under streaming, (1) means re-uploading and rebuilding from scratch, and (2) is worse:
`boxSize` sizes the octree root cube, so growing it mid-stream invalidates every node
already built. And nothing on the device will complain, for the reason CLAUDE.md records —
CudaLOD clamps the cell index, SimLOD's float→uint32 conversion saturates, so out-of-box
points pile into cell 0 instead of faulting. This is the same class of silent failure that
once left a 1.3 km UTM cloud 169 km off origin with 7.04M points in one leaf.

The box must be final before the first batch is uploaded.

### 1.6 None of this can be verified from a script today

- A gate refusal at startup writes to stderr, leaves a window with **no active pipeline**,
  and exits **0** — `App::init` returns `true` regardless of a load failure.
- `--dump-frame` on a failed activation skips the entire stats block, which is guarded by
  `if (const ILodPipeline* pipeline = m_registry.active())` (`App.cpp:482`). The PPM is
  still written and the process still exits 0.
- The dump prints `bytesHighWater` and `bytesAllocated` (`App.cpp:537-539`) but **never the
  budget**, so a captured record cannot be interpreted after the fact — and the budget is
  the one number that moves with whatever else was on the GPU.
- **No CLI flag anywhere touches memory or the budget.** All 18 flags in `src/main.cpp`
  are unrelated, and `kUsableFraction` / `reserve` are function-local constants inside
  `App::computeBudget()`.

`bench/reference/README.md` has no 350M capture at all — the cloud appears there only as a
reader-determinism cross-check (`:115-117`). That is the gap this plan has to close, and
it cannot be closed with a tool that exits 0 on refusal.

---

## 2. Design

### 2.1 Ring depth is `BATCH_STREAM_SIZE`, and RemoLOD's fork comes back down

The kernel's modulo fixes the contract: the device ring must hold **exactly**
`BATCH_STREAM_SIZE` slots of `MAX_BATCH_SIZE` (1M) points. `reset.cu:78-79` independently
requires the `batchSizes` array to be at least that long, and getting it short is an
out-of-bounds device write presenting as `CUDA_ERROR_LAUNCH_FAILED` with no indication of
where (`simlod_layout.h:56-59`).

SimLOD's 50 is vendored and immutable, so **50 is the depth** — 800 MB of ring. RemoLOD's
`BATCH_STREAM_SIZE = 8192` (`kernels/remolod/remolod_structures.cuh:21`, mirrored at
`remolod_layout.h:10` with a `static_assert`) would be 131 GB as a real ring. That value
exists *only* to make the current non-wrapping resident feed addressable past 50M; once
the ring genuinely wraps it is meaningless, and it drops to 50.

This is worth stating plainly because it inverts the usual direction of a fork edit: the
change **removes** a divergence. CLAUDE.md records that RemoLOD and SimLOD currently build
identical trees — 4,137 nodes / 12,742,751 voxels on morro_bay 36M — because the fork is
line-for-line SimLOD's apart from two constants. After this change it is apart by one, and
the tree equality becomes a free, strong check that the ring is correct.

`PointSource` is shared and the constant is not, so the source cannot hardcode the depth.
Add `ringSlots` to `PipelineInfo` beside `needsWholeCloudResident` and pass it into
`start()`. `PipelineInfo` is already the place the shell learns a pipeline's requirements
without knowing which pipeline it is.

### 2.2 Backpressure, and the silent failure if it is wrong

The producer must not write slot `B % depth` until the consumer has finished batch `B`.
The only consumption signal is `stats->batchletIndex`, a monotonic device counter
(`HostDeviceInterface.h:69`) already read back every frame:

```cpp
	m_batchesConsumed = s.batchletIndex;     // SimlodPipeline.cpp:280
```

So the protocol is: after the stats readback, the pipeline calls
`source.setPointsConsumed(...)` — the hook that already exists and does nothing — and the
source refuses to fill any slot whose batch index is within `depth` of the consumer's
position. The consumer's own rate limits are already in the kernel and must be respected
rather than fought: `numBatches = min(numBatchesUploaded - stats->batchletIndex, 20)`
(`progressive_octree_voxels.cu:888`) caps ingest at 20 batches per launch, and the
device-side `MAX_PROCESSING_TIME` budget can cut a launch shorter still. A producer that
runs ahead stalls, which is correct; a producer that ignores the window overwrites points
the tree has not read yet.

**Name the failure mode, because it is invisible.** Getting this wrong builds a tree from
overwritten points, with no fault, no allocation error and plausible-looking node counts.
That is the same class of defect CLAUDE.md records as having silently taken the SimLOD
pipeline down for a whole commit while `make` succeeded, `--check-kernels` passed and
`--dump-frame` exited 0. The invariant therefore gets an explicit host-side check —
producer index minus consumer index never exceeds `depth` — not just careful code.

### 2.3 Synchronous upload first; the copy stream is its own stage

There is no second CUDA stream in the repo (`FrameContext::stream` is `nullptr` at
`App.cpp:398`) and every pipeline already does a full `cuCtxSynchronize()` after construct
(`SimlodPipeline.cpp:259`, `RemolodPipeline.cpp:327`). Stage 2 uploads synchronously, once
per frame, inside the existing serialisation. That is enough to prove the ring, and it is
the smallest change that makes 350M correct.

It does **not** deliver the paper's loading/generation overlap, which `README.md:219-221`
lists as the actual point of the streaming loader. That needs a real async copy stream and
is deferred to Stage 5 rather than smuggled in here — the two claims are separable and
conflating them would make the first one unverifiable.

### 2.4 The box is final before the first upload

The re-read loop of §1.5 has to go for the streaming path, which means the translation
must be trustworthy before anything is uploaded. Add a **coordinates-only bounds pre-pass**
for the two seekable formats, and reuse the existing threaded block reader rather than
writing a second one: `LasReader.cpp:221-299` already fans out over up to 8 threads in
64K-point blocks, each with its own `ifstream` seeking to
`info.offsetToPointData + p * bpp`, and merges per-thread `Bounds`. It needs a
`(firstPoint, count)` range and a mode that reads coordinates without materialising
`Point`s. `.simlod` is trivially seekable too — 24-byte header, point *i* at `24 + 16i`.

Cost is one extra pass over coordinates: ~3 s on the 350M `.las`. In exchange the box is
final before construction starts, can never grow underneath a live tree, and a point
outside it afterwards is a hard error naming the file rather than a silent pile-up in cell
0.

`.laz` decodes strictly sequentially through laszip with no seek — ~70 s for this cloud —
so a pre-pass would double that. `.laz` stays on the resident path in Stage 3, stated as a
limitation. Parallel `.laz` decode is already its own README item and depends on the
lazperf swap.

### 2.5 Both modes survive; the pipeline selects

`flat` and `cudalod` genuinely require whole residency — both check
`source.isFullyResident()` (`FlatPipeline.cpp:92`, `CudalodPipeline.cpp:184`) and
`FlatPipeline` caches the raw device pointer across frames (`:93-95`), which a wrapping
ring would invalidate immediately. `Mode::Whole` therefore stays exactly as it is.

`needsWholeCloudResident` flips to `false` for `simlod` and `remolod`, which already
consume nothing but `BatchView` and pass `view.slots` straight to the kernel with no host
pointer arithmetic. `App.cpp:162` then reads that field instead of hard-wiring
`Mode::Whole`, and the dead field acquires its first reader.

A GUI switch from a streaming to a resident pipeline mid-load re-enters `switchTo()`, which
already calls `source->rewind()` (`PipelineRegistry.cpp:121`). `rewind()` must become real:
restart ingest from batch 0, reset the producer and consumer indices, and re-enter whatever
mode the incoming pipeline asked for. Switching *into* a resident pipeline from a partial
stream means a full re-read, which is correct and should be visible in the status line
rather than silent.

> **A correction to how this was framed.** When scoping this work the options offered were
> "a streaming ring" versus "a ring but keep resident mode too". Those are not
> alternatives: `flat` and `cudalod` cannot function without residency, so both modes
> coexist in either case. The decision that was actually taken is *which path the
> progressive pipelines take* — and they take the ring.

### 2.6 Split "wants" from "needs" in the gate

Add `minBytesPerPointEstimate` to `PipelineInfo`. `fits()` refuses on that; the existing
`bytesPerPointEstimate` keeps its job as the allocation coefficient and stops being a
refusal threshold. The floor is measured, not guessed: the paper's 26 B/pt, corroborated
by RemoBench's own 36M tree — 36.2M points, 12,742,751 voxels, ~0.93 GB of points + voxels
+ occupancy grids ≈ 26 B/pt, which is the agreement that makes the paper's figure usable
here.

Under streaming the input term in the gate changes shape entirely: from `16 × numPoints`
to a flat `ringSlots × 1M × 16` — 800 MB regardless of cloud size. That is the whole
point of the exercise, and it is what the gate must reflect once Stage 2 lands.

### 2.7 Enough scriptability to verify the rest

Scoped strictly to what verifying this work needs (§1.6):

- Non-zero exit when a pipeline named on the command line is refused, instead of a window
  with no pipeline and exit 0.
- `--dump-frame` prints the refusal reason instead of skipping the stats block.
- The budget appears in the dump, so a capture is self-describing.
- `--device-budget <bytes>` overrides `computeBudget()`, so a benchmark run stops
  depending on what else is on the GPU. Note `src/main.cpp` has no size-suffix parser
  today — only `atoi` and `strtoull` — so either accept plain bytes or add the parser
  deliberately.

The GUI tooltip at `StatsPanel.cpp:112-116` also claims the budget is "Computed once",
which is false: `computeBudget()` runs inside `activateCloud()` (`App.cpp:156`) and so
re-runs on every cloud load. Fix the text while touching the file.

### 2.8 Expected outcome, stated honestly

Per-point coefficient for the persistent store is **26 B/pt** — the paper's 9.1 GB / 350.36M,
corroborated independently by RemoBench's own 36M tree (~0.93 GB / 36.2M = 25.7 B/pt). Two
measurements 1% apart, same `POINTS_PER_CHUNK = 1000`.

Full 350M tree:

| term | bytes |
|---|---|
| tree, 26 B/pt × 350.36M | 9.11 GB |
| ring, 50 × 1M × 16 B | 0.80 GB |
| momentary buffer | 0.54 GB |
| node pool, 200k × 152 B | 0.03 GB |
| **budget required** | **10.48 GB** |
| **free VRAM required** | **≈ 12.0 GiB** of 15.92 GiB |

What that means at real budgets, with persistent clamped to `available - fixed` and ingest
stopping at `memCapacityReached`:

| free VRAM | budget | resident (today) | streaming |
|---|---|---|---|
| 8.1 GiB (measured, desktop up) | 6.94 GB | 8% of points | **61%** |
| 8.7 GiB | 7.48 GB | 14% | 67% |
| 10.0 GiB | 8.67 GB | 27% | 80% |
| 12.0 GiB | 10.48 GB | 47% | **100%** |
| 14.5 GiB | 12.78 GB | 73% | 100% |

So the deliverable is **not** "the 350M cloud builds a complete tree". It is: the cloud is no
longer refused; it ingests as far as the budget allows and stops cleanly on a *valid*
truncated tree; and the fraction it reaches rises ~7× at the same budget, because 5.6 GB of
resident cloud stops crowding out the persistent store. Completion follows on an idle GPU or
a 24 GB card. Writing the stage up as "350M works" would be overselling it; writing it up as
a failure because it truncated would be misreading it.

**This also makes the truncation point a test.** `completionWarning()` (§2.6) predicts the
percentage from 26 B/pt; the dump reports the observed one. Agreement within a few percent
validates the coefficient, and a large disagreement means 26 B/pt is wrong for this build —
which is worth knowing before it is quoted anywhere.

`cudalod` remains out of reach for 350M at 144 B/pt (50 GB) and is out of scope — it is a
vendored batch builder that requires whole residency by construction.

---

## 3. Staging

**Stage 1 — the gate and scriptability.** `minBytesPerPointEstimate`, exit codes, the
budget in the dump, `--device-budget`. No loader changes and no device changes. Delivers an
honest refusal message immediately, and every later stage is measurable only because this
one landed first.

**Stage 2 — the ring.** `Mode::Stream` in `PointSource`, depth from `PipelineInfo`,
incremental `numBatchesUploaded`, backpressure through `setPointsConsumed`, `rewind()` made
real, and RemoLOD's `BATCH_STREAM_SIZE` back to 50. Synchronous upload. Verified at 36M
first, where the answer is already known.

**Stage 3 — the bounds pre-pass.** Removes the re-read loop for the streaming path and
unblocks 350M on `.simlod` and `.las`. `.laz` stays resident.

**Stage 4 — lift the `simlod` cap.** Delete the `kMaxAddressablePoints` refusal
(`PipelineRegistry.cpp:67-79`). **This is the stage not to rush:** that gate is the only
thing standing between a wrong ring and a silently wrong tree, so it comes last, after
Stage 2's invariant has been exercised on a cloud that actually wraps.

**Stage 5 — the async copy stream** (deferred). The paper's loading/generation overlap,
and the first stage whose claim is about throughput rather than correctness.

---

## 4. Verification

There is no test suite — `tests/unit/` is empty and `REMOBENCH_BUILD_TESTS` is OFF — so
these are the acceptance tests.

- **`make check` after every kernel-adjacent change.** `check-vendored` is specifically
  what proves the RemoLOD constant edit did not touch `kernels/simlod/`. Remember that
  `--check-kernels` compiles and links but does not launch, so it cannot see a host/device
  signature mismatch.
- **The tree-equality check.** After RemoLOD's `BATCH_STREAM_SIZE` drops to 50, `remolod`
  and `simlod` must still produce **4,137 nodes / 12,742,751 voxels** on morro_bay 36M. A
  divergence means the ring is wrong, and this is the cheapest possible signal for it.
- **The three-reader invariant.** `.simlod`, `.las` and `.laz` must still give
  bit-identical trees from the same cloud. The bounds pre-pass is exactly the kind of
  change that breaks this, and it is the strongest verification the readers have.
- **Structural counts, not frames,** for anything with an octree — `simlod`, `remolod` and
  `cudalod` all sample a voxel's colour from whichever thread reaches the cell first, so
  two runs of the same binary differ. `--dump-frame` plus `cmp` is for `flat` only.
- **The backpressure invariant**, asserted host-side: producer index minus
  `stats->batchletIndex` never exceeds the ring depth. Per §2.2 this failure is otherwise
  invisible.
- **The 350M acceptance run.** `remolod` and `simlod` on
  `data/morro_bay_350M/morro_bay_350M.simlod` with an explicit `--device-budget`, reaching
  351 of 351 batches with `memCapacityReached` clear. Expect at least 18 launches — the
  kernel takes at most 20 batches each — and ~3.5 s of read at the `.simlod` reader's
  ~100 MP/s.
- **Capture what does not exist yet.** `bench/reference/README.md` has no 350M row and no
  SimLOD capture at all. Stage 4 is the first point at which either is possible; record
  the budget alongside, per §2.7, or the numbers are not interpretable later.

---

## 5. Files

**Modified**

| path | change |
|---|---|
| `include/remo/PointSource.h` | `start()` takes the ring depth; document what `wrapping` actually means |
| `src/io/PointSource.cpp` | the bulk — a `StreamingSource` beside `ResidentSource`, incremental `numBatchesUploaded`, real `rewind()`, real `setPointsConsumed()` |
| `include/remo/ILodPipeline.h` | `PipelineInfo`: `minBytesPerPointEstimate`, `ringSlots` |
| `src/shell/PipelineRegistry.cpp` | `fits()` / `unsupportedReason()` on the new floor; drop the `kMaxAddressablePoints` refusal in Stage 4 |
| `src/shell/App.{h,cpp}` | mode selection from `needsWholeCloudResident`, `--device-budget`, budget in the dump block, exit codes |
| `src/main.cpp` | the new flag, its help text, and a size parser if one is wanted |
| `src/io/LasReader.{h,cpp}` | a `(firstPoint, count)` range and a coordinates-only bounds pass |
| `src/io/RawReader.{h,cpp}` | ranged `.simlod` read — point *i* at `24 + 16i` |
| `src/pipelines/{Simlod,Remolod}Pipeline.cpp` | `needsWholeCloudResident = false`; call `setPointsConsumed` after the stats readback |
| `kernels/remolod/remolod_structures.cuh`, `remolod_layout.h` | `BATCH_STREAM_SIZE` 8192 → 50, both sides of the `static_assert` |
| `src/shell/StatsPanel.cpp` | the "Computed once" tooltip is false (§2.7) |
| `README.md`, `wiki/01_Overview.md` | §3.4's `Mode::Whole` addressing claim is wrong (§1.1); the 50M-cap and streaming-loader items change |

**Reused rather than rebuilt**

- The threaded block reader (`LasReader.cpp:221-299`) — 8 threads, 64K-point blocks,
  per-thread bounds already merged. The pre-pass is a mode on this, not a new reader.
- `readLasHeader` / `LasHeaderInfo` (`LasReader.h:11-35`) — already a cheap, separate,
  incremental entry point carrying `offsetToPointData`, `bytesPerPoint`, `scale`, `offset`.
- `BatchView`, `Mode::Stream`, `setPointsConsumed`, `numPointsUploaded`,
  `needsWholeCloudResident` — all present, all currently dead.
- `deriveTranslation` and `applyTranslation` (`PointSource.cpp:178-189`).
- The device-side producer handshake (`progressive_octree_voxels.cu:875-881`) — **no
  kernel change required**; the host simply starts incrementing what it currently sets once.
- `PipelineStats::memCapacityReached` — the existing clean-stop signal; do not add a second
  exhaustion path.

---

## 6. What actually landed

Stages 1–4. Stage 5 (the async copy stream) is untouched and still deferred: ingest is a
blocking `cuMemcpyHtoD` on the render thread, inside the serialisation every pipeline
already does after construct.

**The headline.** `data/morro_bay_350M/` is no longer refused. It streams through a 50-slot
wrapping ring, stops cleanly on `memCapacityReached`, and `simlod` and `remolod` build the
same tree from it. Peak host RSS for the 350M cloud is **371 MB**, against ~5.6 GB before —
the cloud is no longer resident on the host either, which §2.4 did not ask for but §2.8's
arithmetic quietly assumed was free.

| | before | after |
|---|---|---|
| 350M `remolod` | refused, "needs 22.4 GB" | 209M points ingested, 75,207,427 voxels, 25,345 nodes |
| 350M `simlod` | refused, 50M ceiling | identical to `remolod` |
| input cost, any cloud | 16 B/pt (5.6 GB at 350M) | 0.8 GB, flat |
| host RSS at 350M | ~5.6 GB | 371 MB |
| RemoLOD/SimLOD divergence | two constants | one (`MAX_NODES_CAPACITY`) |
| refusal exit code | 0 | 1 |

At a pinned `--device-budget 6G`, `.simlod` and `.las` both give 169,000,000 points /
59,548,545 voxels / 20,633 nodes. Captured in `bench/reference/README.md`.

### 6.1 The gate refuses on a floor, but not the floor §2.6 described

§2.6 said `fits()` should refuse on `minBytesPerPointEstimate`. Applied literally that
refuses the 350M cloud at 10.5 GB — which is the **correct number**, and the wrong outcome:
§2.8's whole deliverable is that the cloud is accepted and truncates. The two sections
disagree, and §2.8 wins, because a 61% tree is a result and a refusal is not.

So `PipelineInfo` grew a third number, `minStoreBytes`, and the registry two functions:

- `minBytesRequired()` — what holding the **whole** cloud costs. Drives the completion
  warning and the "would need 10.5 GB" line. This is §2.6's number, and it is reported.
- `refusalFloor()` — what running **at all** costs: ring + fixed overhead + the smallest
  store the pipeline works with. Only this refuses.

They are the same for a pipeline that cannot truncate — `cudalod` voxelizes the whole cloud
or overruns its slab — and `minStoreBytes == 0` is how that is spelled.

Two other fields the plan did not list turned out to be load-bearing. `fixedBytesEstimate`
carries the momentary buffer, node pool and side arrays; without it the gate was off by
0.59 GB, and §2.8's own table has those rows. `ringSlots` sits beside
`needsWholeCloudResident` as §2.1 asked.

`bytesPerPointEstimate` also changed meaning: it was 48 + 16, mixing in an input term that
now depends on residency. It is 48 — the persistent coefficient alone — and the input term
is derived from `needsWholeCloudResident` and `ringSlots`.

### 6.2 `BatchView::wrapping` is not the check worth making

§2.1 and the field's own default suggested `wrapping` gates the ring's correctness. It does
not. `kernel_construct` reads batch B from slot `B % BATCH_STREAM_SIZE` **always**, so a
ring *deeper* than `BATCH_STREAM_SIZE` is exactly as wrong as a shallower one — and
`wrapping` is false in that case. The first version of the check had this bug. The real
condition is:

```cpp
numSlots == BATCH_STREAM_SIZE || (numBatchesTotal <= BATCH_STREAM_SIZE &&
                                  numSlots >= numBatchesTotal)
```

`wrapping` survives as documentation of whether slots are reused, nothing more. The source
allocates `min(ringSlots, numBatchesTotal)` slots, which satisfies the condition by
construction and saves 0.8 GB on a cloud smaller than the ring — so a 36M cloud is charged
for 36 slots, not 50, and `inputBytesFor()` in the registry matches.

### 6.3 Interface changes the plan did not anticipate

- **`pump()` is new on `PointSource`.** Something has to drive the upload, and none of the
  listed dead scaffolding did. The shell calls it once per frame before the consumer builds.
- **`setPointsConsumed(uint64_t)` became `setBatchesConsumed(uint32_t)`.** §5 listed the
  points-based hook as reusable. It is not: the consumption signal *is*
  `stats->batchletIndex`, the invariant is stated in batches, and a points→batches division
  is a rounding bug waiting for a short final batch. Reusing a hook whose units do not match
  the signal is worse than renaming it.
- **`overranConsumer()` is new**, per §2.2's insistence that the invariant be checked rather
  than carefully coded. `CloudSource::uploadBatch` refuses out-of-window writes and reports
  once on stderr.
- **`start()` takes `(mode, ringSlots, maxDeviceBytes)` and is re-entrant.** A pipeline
  switch calls it again with the incoming pipeline's shape.
- **`ResidentSource` became `CloudSource`**, one class serving both modes, over a new
  `BatchProducer` seam (`src/io/BatchProducer.h`). `Mode::Whole` is the degenerate ring —
  one slot per batch, nothing reused — which is what makes the two modes one code path.
- **`PipelineRegistry::switchTo()` now starts the source**, before allocating the pipeline.
  §2.5 put mode selection in `App.cpp:162`; that is the wrong place, because the shape is
  the *incoming* pipeline's requirement and a mid-run switch has to re-shape ingest before
  the new pipeline takes its share of the budget. `App::activateCloud` no longer calls
  `start()` at all.

### 6.4 Stage 4 kept the cap instead of deleting it

§3 said to delete the `kMaxAddressablePoints` refusal. It is instead conditional on
`needsWholeCloudResident && ringSlots == 0 && progressive` — the shape that was actually
broken. The ceiling was never SimLOD's; it was a resident feed colliding with the kernel's
modulo, and that combination is still constructible. §3 called this "the stage not to
rush", and the check is the only thing between a wrongly-shaped feed and a silently wrong
tree, so it stays.

### 6.5 Bit-identity was preserved by keeping the header, not by trusting the scan

§2.4 asked for a coordinates-only pre-pass and got one — `readLasBounds`, `readSimlodBounds`,
plus `readLasRange` / `readSimlodRange` for slot fills, all on the existing threaded block
reader as §5 specified. The subtlety the plan does not mention: replacing the header box
with the observed box would have **changed every tree**, because `boxSize` comes from the
header maximum today and a header maximum is usually larger than the observed one.

So the pre-pass feeds the *same* arithmetic the two-attempt read loop did — lower the
minimum only on a slack-gated underflow, grow the maximum whenever a point exceeds it — and
the bounds are accumulated from the same float32 coordinates the points carry, not from the
f64 originals. Both warnings survive verbatim. `.simlod`, `.las` and `.laz` still give
4,137 nodes / 12,742,751 voxels at 36M.

The whole-file `readSimlod` is gone; the ranged reader replaces it.

### 6.6 A reset must rewind the source, which nothing in §2 says

`reset.cu` zeroes `stats->batchletIndex`, so the consumer goes back to batch 0 — and the
producer must too, or it keeps filling slots ahead of a consumer that has restarted. The
GUI "rebuild" button and every kernel hot-reload take this path. Both pipelines now call
`source.rewind()` where they handle `m_needsReset`. §2.5 only covered `switchTo`.

### 6.7 An unrelated defect this work surfaced

`readStats()` copied `Stats::numVisible{Nodes,Points,Voxels}` into `PipelineStats`,
over the values the *render* kernel had just written into `DeviceDiagnostics`. **No kernel
writes those `Stats` fields** — grep finds no assignment in
`progressive_octree_voxels.cu` or `simlod_render.cu` — so the copy always zeroed them.

It never showed, because a build that completes stops calling `readStats()` and the last
render's numbers survived. A build that stops on `memCapacityReached` never completes, so
350M reported `visible nodes 0` for a tree plainly on screen. Removed from both pipelines.

### 6.8 What did not change

- **No device code except one constant.** `kernels/simlod/` is byte-identical
  (`make check-vendored` passes), and RemoLOD's fork changed only `BATCH_STREAM_SIZE`,
  downward, to upstream's 50. §1.2 was right that the device half was already finished: the
  host simply started incrementing what it used to set once.
- **`flat` and `cudalod` are untouched** and stay on `Mode::Whole`.
- **No new GUI-only path.** `--device-budget` is the new control and it is a flag.

### 6.9 Numbers worth carrying forward

- **26 B/pt is confirmed on this build.** Predicted-vs-observed completion agrees within a
  point at every budget tried (48.4/48.2, 60.2/59.7, 60.3/59.7). §2.8 proposed this as a
  test; it passes, and the dump prints both so it keeps being one.
- **§2.8's table is accurate.** The gate independently computes 10.5 GB for the full 350M
  tree against the plan's 10.48. The predicted fractions run ~0.5 points below §2.8's
  column because the kernel's 200 MB `safetyMargin` is now subtracted, which §2.8 did not.
- **350M still does not complete on this machine**, exactly as §2.8 said it would not:
  10.5 GB needed, 9.4 GB of VRAM free with a desktop up. The deliverable was never
  completion.

---

## Related

- `plans/02_ProfilingTools.md` — Stage 2 (`--bench`, NDJSON) is the natural consumer of
  `--device-budget` and of the 350M capture this plan unblocks.
- `bench/reference/README.md` — capture methodology; records no 350M and no SimLOD
  baseline, and documents why an unqualified memory figure is not a claim.
- `references/SimLOD.pdf` §6.6 / Table 5 — the 9.1 GB measurement this plan is calibrated
  against. §3 *Persistent Buffer* says upstream pre-allocates "90% of the available GPU
  memory", but the shipped code uses `availableMem * 0.80`
  (`external/SimLOD/modules/progressive_octree/main_progressive_octree.cpp:572`). Cite the
  code, not the prose.
