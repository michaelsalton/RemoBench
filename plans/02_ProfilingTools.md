# Profiling Tools: Implementation Plan

## TL;DR

- **RemoBench is a measurement instrument whose measurement layer is three doubles.** `ILodPipeline` exposes `buildDeviceMsTotal`, `renderDeviceMsLast` and `buildLaunchCount` (`include/remo/ILodPipeline.h:177-179`), filled inconsistently by the three pipelines, aggregated into running sums, and printed once at exit. Every architectural decision in this repo — one shared shell, one budget handed unchanged to every pipeline, health flags that invalidate a run — exists so that two LOD algorithms can be compared honestly. The timing layer is the one part that has not kept up.
- **The single most important defect: SimLOD's render time is not measured at all.** `SimlodPipeline::render` and `CudalodPipeline::render` record no CUDA events. Only `RemolodPipeline` fills `renderDeviceMsLast`. The GUI and the dump block print `0.00` for the other two, formatted identically to a real measurement.
- **The plan is four layers, staged.** (1) A `GpuProfiler` owned by the shell, handed to pipelines through `FrameContext`, replacing the ad-hoc `CUevent` pairs. (2) Retained per-launch samples with Welford statistics and percentiles, replacing running sums. (3) A device-side `DeviceTimeline` for intra-kernel phase attribution, compiled in only under `-DREMO_PROFILE`. (4) A `--bench` run mode writing NDJSON time series with full provenance.
- **Layer 3 exists because both pipelines are cooperative megakernels.** A `CUevent` pair can only ever say "`kernel_construct` took 8.1 ms". Whether that is expansion, voxel creation, or point insertion is invisible to every host-side timer, and to `ncu` as well. Only `%globaltimer` inside the kernel can see it — and SimLOD's author already instrumented exactly those phases, then wired them to a no-op.
- **The immediate payoff is that `bench/reference/` stops being hand-transcribed.** The existing CudaLOD reference table doubles as an oracle for the new instrument: if `--bench` does not reproduce 4.9 / 4.7 / 20.1 / 60.9 ms per strategy, the profiler is wrong.

---

## 1. Why — an audit of what is measured today

### 1.1 SimLOD and CudaLOD render time is never measured

`SimlodPipeline::render` (`src/pipelines/SimlodPipeline.cpp:370-411`) and `CudalodPipeline::render` (`src/pipelines/CudalodPipeline.cpp:367-406`) launch their render kernels with no `cuEventRecord` on either side. Only `RemolodPipeline` assigns `renderDeviceMsLast` (`src/pipelines/RemolodPipeline.cpp:158-184`).

Consequently `SettingsPanel.cpp:109` (`ImGui::Text("render kernel: %.2f ms", ...)`) and `App.cpp:549` (`printf("  render device ms    %.2f\n", ...)`) render a default-initialised `0.0` in the same format as a genuine measurement. This is worse than a missing number: it is a *plausible* number. The first metric the research needs — SimLOD render time — is the one metric that does not exist.

### 1.2 Only aggregates survive; the distribution is discarded

```cpp
m_lastBuildMs = eventMs(m_buildStart, m_buildEnd);   // SimlodPipeline.cpp:311
buildDeviceMsTotal += m_lastBuildMs;
++buildLaunchCount;
```

`m_lastBuildMs` is overwritten every launch and `buildDeviceMsTotal` is a sum. A progressive build over a 36M-point cloud performs on the order of hundreds of bounded launches, each governed by the device-side `MAX_PROCESSING_TIME` budget in `progressive_octree_voxels.cu`. The interesting result is *the shape of that distribution* — does the budget hold, what is the tail, how does per-launch cost evolve as the octree deepens — and none of it can be recovered from a sum and a count.

The same applies to render: a mean frame time over a camera sweep is nearly meaningless when the visible-node count varies by an order of magnitude across the sweep. What is needed is the per-frame series with the visible-sample count attached.

### 1.3 The finest-grained data in the codebase is computed and thrown away

`kernels/simlod/progressive_octree_voxels.cu:789-812` reads eight `nanotime()` marks across the construction phases, computes seven deltas, and hands them to:

```cpp
cudaprint->print("t_00_70: {:.3f}, t_00_10: {:.3f}, expand: {:.3f}, "
                 "createVoxels: {:.3f}, ... insertPoints: {:.3f}, ...", ...);
```

`CudaPrint` is a no-op on both ends. Its own header says so (`kernels/simlod/CudaPrint.cuh:1-11`): `print()` returns immediately and the host half is entirely commented out upstream. RemoBench passes it a 1 KB dummy allocation precisely so the vendored kernel signature does not have to change (`SimlodPipeline.cpp:156-158`).

Separately, `durationExpandMS` is computed at `progressive_octree_voxels.cu:965` and never read by anything.

So the labels `expand`, `createVoxels`, `insertPoints` — the exact intra-kernel breakdown the research wants — are already being computed on-device, every launch, and discarded.

**Why this cannot be recovered any other way.** Every kernel in both pipelines is a single cooperative launch using `cg::this_grid().sync()`; this is stated as a load-bearing constraint at `include/remo/ILodPipeline.h:13-14`. A `CUevent` pair therefore brackets *the entire algorithm*, not a phase of it. Nsight Compute cannot subdivide it either — its kernel-level replay does not support grid-wide sync, and even with application replay the unit of measurement is still the whole kernel. Intra-kernel attribution in a megakernel is only available from inside the kernel.

### 1.4 Two measurement regimes silently mix

Build synchronises on every launch (`SimlodPipeline.cpp:305-309`, `CudalodPipeline.cpp:280-303`). Render only synchronises when `frame.strictTiming` is set (`SimlodPipeline.cpp:395-401`). So in a default run, build numbers are strict and render numbers — where they exist at all — are read one frame late. `--strict-timing` is documented in `main.cpp:202-204` as the flag that makes timings trustworthy, but nothing in any output records whether it was on. Two runs whose numbers are not comparable are indistinguishable after the fact.

### 1.5 There is no time series

`--dump-frame` writes a PPM and prints a stats block (`src/shell/App.cpp:526-558`). The comment there is honest about it: *"This is the readout the benchmark harness will formalise."* It has not been formalised. `bench/reference/README.md` is a hand-transcribed set of tables, which is why it carries a note that SimLOD was never captured at all — the upstream binary cannot be scripted.

---

## 2. Design

Four layers. Each is independently useful, and the staging in §3 orders them by payoff-per-unit-risk.

### Layer 1 — `GpuProfiler` and `GpuScope`, owned by the shell

**New files:** `include/remo/GpuProfiler.h`, `src/shell/GpuProfiler.cpp`.

The comment at `include/remo/ILodPipeline.h:5` already states the rule: a pipeline does not own "the window, the camera, the loader, the rasterizer, EDL, the GL blit, timing or stats plumbing — all of that is shared, which is the entire point: it is what makes two pipelines comparable." The current per-pipeline `CUevent` pairs violate that rule, and defect 1.1 is the direct consequence — a pipeline that simply forgets to record an event produces a zero that looks like data.

A profiler pointer joins `FrameContext`, next to the `strictTiming` flag that is already there:

```cpp
struct FrameContext {
    SharedUniforms uniforms;
    RenderTargets  targets;
    CUstream       stream = nullptr;
    int            numSMs = 0;
    bool           strictTiming = false;
    GpuProfiler*   profiler = nullptr;   // never null in a normal frame
};
```

Pipelines name their own scopes, so the shell never needs a list of them:

```cpp
// SimlodPipeline::build
{
    GpuScope s(*frame.profiler, "simlod.construct");
    REMO_CU(cuLaunchCooperativeKernel(construct, grid, 1, 1, m_blockSize, 1, 1, 0, 0, args));
}

// SimlodPipeline::render  -- the measurement that does not exist today
{
    GpuScope s(*frame.profiler, "simlod.render");
    REMO_CU(cuLaunchCooperativeKernel(kernel, grid, 1, 1, m_blockSize, 1, 1, 0, 0, kernelArgs));
}
```

**Internals.**

- A pool of `CUevent`s, allocated lazily per scope name and held in a ring four frames deep. Events are recycled, never created per frame — `cuEventCreate` in a render loop is its own measurement artefact.
- Harvest with `cuEventQuery`, never `cuEventSynchronize`. A scope whose events are not yet ready is simply left for the next frame's harvest pass. Profiling therefore costs no stall in the default regime, which matters because a profiler that changes the frame time is measuring itself.
- Under `frame.strictTiming`, harvest synchronously at end of frame so samples are attributed to the frame that produced them rather than arriving three frames late.
- Every sample records which regime produced it. `Regime::Strict` and `Regime::Deferred` samples are accumulated separately and never pooled. This closes defect 1.4 structurally rather than by convention.
- Scopes nest. `simlod.construct` may contain `simlod.construct.reset`; the profiler stores the parent chain so a flame-style breakdown is available later without re-instrumenting.

**What this replaces.** The event members and the duplicated `eventMs()` helper in `SimlodPipeline.cpp:49-53`, `CudalodPipeline.cpp:53-57` and `RemolodPipeline.cpp` all go away, along with `m_buildStart`/`m_buildEnd`, `m_splitStart`/`m_splitEnd`, `m_voxelStart`/`m_voxelEnd`, `m_renderStart`/`m_renderEnd` and their create/destroy bookkeeping in `allocate()` and `release()`. Net line count is roughly flat; the difference is that a pipeline can no longer *fail to measure* something, because the scope is at the launch site.

**Scope names.** Flat, dotted, stable — they become column names in the output, so they are part of the data format:

| scope | pipeline | what it brackets |
|---|---|---|
| `simlod.reset` | SimLOD | the one-block reset kernel |
| `simlod.construct` | SimLOD | one bounded progressive launch |
| `simlod.render` | SimLOD | `kernel_render` |
| `cudalod.split` | CudaLOD | `kernel2`, split + counting sort |
| `cudalod.voxelize` | CudaLOD | `kernel3`, per strategy |
| `cudalod.render` | CudaLOD | `kernel_render` |
| `remolod.render` | RemoLOD | the baseline rasteriser |

`remolod.render` matters more than it looks: `RemolodPipeline` does no LOD at all, so it is the control against which both LOD renderers' cost is interpreted.

### Layer 2 — retain samples, not sums

**Touches:** `include/remo/GpuProfiler.h`, `src/shell/SettingsPanel.cpp`, `include/remo/ILodPipeline.h`.

Per scope name, the profiler keeps:

- A **Welford accumulator** — `n`, `min`, `max`, `mean`, `M2` — giving mean and variance in one pass with no catastrophic cancellation. Cheap enough to update unconditionally.
- A **ring of the last 4096 raw samples**, for median, p95, p99 and histograms. 4096 × 8 B per scope is negligible and covers a 600-frame bench run several times over.

```cpp
struct ScopeStats {
    uint64_t n; double min, max, mean, m2;
    double variance() const;      // m2 / (n - 1)
    double percentile(double) const;  // from the ring, sorted on demand
};
```

**Migration of the three public doubles.** `buildDeviceMsTotal`, `renderDeviceMsLast` and `buildLaunchCount` (`ILodPipeline.h:177-179`) are read by `SettingsPanel.cpp:109`, `SimlodPipeline::gui()` (`SimlodPipeline.cpp:436-443`), `CudalodPipeline::gui()` and `App::dumpFrame` (`App.cpp:547-549`). They stay initially, computed from the profiler as a compatibility facade, so Layer 1 can land without touching every call site. They are retired at the end of Layer 2, once the panel reads the profiler directly.

**Panel changes.** The timing table in `SimlodPipeline::gui()` (`SimlodPipeline.cpp:428-447`) currently shows last / total / launches / throughput. It gains median and p95, and the throughput row is annotated with which of the two it derives from. The `%.1f fps (%.2f ms)` line at `SettingsPanel.cpp:91` — currently a 30-frame mean from `GLRenderer` (`GLRenderer.cpp:216-221`) — gains a p95 alongside it, because a 60 fps mean hiding a 40 ms hitch every twelfth frame is the exact pathology a progressive builder produces.

### Layer 3 — `DeviceTimeline`, replacing the CudaPrint no-op

**Touches:** `include/remo/HostDeviceCommon.h`, `kernels/shared/remo_prelude.cuh`, `kernels/simlod/progressive_octree_voxels.cu`, `kernels/cudalod/kernel.cu`, the two pipelines' readback functions.

This is the layer that gets `expand` / `createVoxels` / `insertPoints` out of the megakernel, and it is the only layer that touches vendored device code.

**The provenance constraint, and how it is respected.** Both `include/remo/ILodPipeline.h:100-104` and `kernels/simlod/HostDeviceInterface.h:5-7` state the rule: rewriting the structs the reference kernels read is how a port silently stops reproducing its published numbers. The CMake preamble states a related rule about never including a file that a patch modifies.

The resolution is a compile-time guard. Marks are written through a macro that expands to *nothing* unless `REMO_PROFILE` is defined:

```cpp
// kernels/shared/remo_prelude.cuh
#ifdef REMO_PROFILE
  #define REMO_MARK(tl, phase)                                  \
      do { if (cg::this_grid().thread_rank() == 0) {            \
             uint32_t i = (tl)->numMarks++;                     \
             if (i < REMO_MAX_MARKS) {                          \
               (tl)->marks[i].phase = (phase);                  \
               (tl)->marks[i].ns    = nanotime();               \
             } } } while (0)
#else
  #define REMO_MARK(tl, phase) ((void)0)
#endif
```

The default build therefore emits byte-identical device code to today's, and the numbers it produces remain directly comparable to the upstream reference. The profiling build is a *declared, separate measurement mode* — the same discipline `FrameContext::strictTiming` already applies on the host side, and for the same reason.

The variant is requested through the existing `KernelProgramDesc::defines` field (`include/remo/CudaModularProgram.h`), which is documented as being part of the compile cache key. Both variants therefore cache side by side on disk and hot reload independently; no cache invalidation work is needed.

**The buffer**, declared next to `DeviceDiagnostics` in `include/remo/HostDeviceCommon.h:173-186`:

```cpp
constexpr uint32_t REMO_MAX_MARKS = 256;

struct DeviceTimeline {
    uint32_t numMarks;
    uint32_t overflow;              // more phases than REMO_MAX_MARKS
    struct { uint32_t phase; uint32_t pad; uint64_t ns; } marks[REMO_MAX_MARKS];
};
```

**Where the marks go.** SimLOD's already exist as local variables — `t_00` through `t_70` at `progressive_octree_voxels.cu:724-789`, and `tStart` / `tStartExpand` / `tEndExpand` in the outer kernel at lines 831-964. The work is to give them a sink, not to find the phase boundaries. CudaLOD's `kernel2` / `kernel3` have no instrumentation and need marks placed from scratch, which is where the majority of Layer 3's effort actually is.

The `cudaprint->print(...)` call at `progressive_octree_voxels.cu:803` is left exactly as it is. It costs nothing, and removing it would be an edit to a vendored kernel for no benefit.

**Readback.** `DeviceTimeline` is copied to the host in `SimlodPipeline::readStats()` (`SimlodPipeline.cpp:325`) and `CudalodPipeline::readResults()` (`CudalodPipeline.cpp:315`) alongside the existing `Stats` / `Results` copies — no extra synchronisation point, since both already run after a `cuCtxSynchronize`. Consecutive marks are differenced into named phase durations and fed to the profiler as samples under scope names like `simlod.construct.expand`, so Layers 2 and 4 handle them with no special cases.

**Two caveats that go in the output header, not just in a comment.**

1. `%globaltimer` (`kernels/simlod/utils.h.cu:322-327`) is a wall clock with coarse resolution, not a cycle counter. It is fine for phases measured in hundreds of microseconds; it is not fine for anything sub-microsecond, and phase deltas near that scale must be reported as such rather than quietly rounded.
2. A mark read by one thread describes *that thread's* view. It is only meaningful as a grid-wide phase boundary when it sits immediately after a `grid.sync()`. SimLOD's existing marks already do; CudaLOD's new ones must be placed to the same discipline, and a mark that cannot be placed after a barrier should not be placed at all.

**The cross-check that validates the layer.** The sum of a launch's phase deltas must equal that launch's `CUevent` measurement from Layer 1, to within a few percent. If it does not, either a phase is unmarked or a mark is in the wrong place. This is a test, not an aspiration — see §5.

### Layer 4 — `--bench` and NDJSON output

**Touches:** `src/main.cpp`, `src/shell/App.h`, `src/shell/App.cpp`, new `src/shell/BenchRun.cpp`.

New options alongside the existing `--strict-timing` / `--dump-*` group in `src/shell/App.h:25-53`:

```
--bench                 run the benchmark protocol and exit
--bench-frames <n>      measured frames (default 600)
--bench-warmup <n>      frames discarded before measuring (default 60)
--bench-out <path>      output file (default bench/runs/<timestamp>_<pipeline>_<dataset>.jsonl)
```

**Behaviour.** Force `strictTiming` on. Seed the camera from the existing `OrbitControls::frameBox()` (`src/shell/OrbitControls.h:91-100`), which `App::activateCloud` already calls (`App.cpp:186-189`). Then drive a deterministic orbit by writing `yaw` directly each frame — the fields are public and `update()` recomputes the world matrix, so no new camera API is required:

```cpp
controls.yaw = kStartYaw + 2.0 * M_PI * double(measuredFrame) / double(benchFrames);
```

A full revolution over the measured frames sweeps the visible-node count across its whole range in one run, which is what makes a render-cost-versus-visible-samples plot possible from a single invocation. Pitch and radius stay at the framed values so the sweep varies one thing.

Discard the warm-up frames — NVRTC compilation, the first cooperative launch, GL interop registration and clock ramp all land there — then write one record per scope per frame.

**Record format**, NDJSON so a run can be appended to and read with `jq` or three lines of pandas:

```json
{"t":"header","gpu":"NVIDIA GeForce RTX 5080","sm":120,"numSMs":84,"cuda":"13.1",
 "driver":"...","clocksLocked":true,"width":1600,"height":900,
 "dataset":"morro_bay_36M.simlod","numPoints":36200706,"pipeline":"cudalod",
 "strategy":"3 WEIGHTED_NEIGHBORHOOD","kernelHash":"...","profileBuild":false,
 "regime":"strict","warmup":60,"frames":600}

{"t":"s","frame":142,"scope":"simlod.construct","ms":8.14,
 "visibleNodes":3204,"visibleSamples":18412330,"pointsIngested":21000000,
 "bytesHighWater":4211081216,"warn":[]}

{"t":"summary","scope":"simlod.construct","n":412,"min":6.02,"median":8.11,
 "mean":8.44,"p95":11.40,"max":19.83,"stddev":1.72}
```

The context fields on each sample are not decoration. A render time without `visibleSamples` is uninterpretable, and a sample taken while `memCapacityReached` or `nodeCapacityReached` is set is not a data point at all — `PipelineStats` (`ILodPipeline.h:126-131`) already models this correctly, and `warn` carries those flags per sample so the analysis can filter rather than the harness silently dropping them.

The header's provenance fields all come from things that already exist: `CudaContext::deviceName()`, `numSMs()`, `ccMajor()`/`ccMinor()`; the dataset from `CloudMeta`; the strategy from `CudalodPipeline`'s own tunable; and `kernelHash` from the source hash `CudaModularProgram` already computes as part of its disk cache key — it needs only to be exposed via an accessor.

**Exit code.** Non-zero if any health flag tripped during the run. A benchmark that reports a suspiciously good number from a silently truncated structure is the specific failure this codebase has already designed against; the harness must honour that rather than reintroduce it.

---

## 3. Staging

**Stage 1 — Layers 1 + 2. DONE.** `GpuProfiler`, `GpuScope`, `FrameContext` wiring, all three pipelines converted, sample retention and percentiles, panel updated. Self-contained, no device-code changes, no new output format. Delivers real SimLOD and CudaLOD render times with distributions on day one — which is the metric the research is missing and the reason this stage is first. See §7 for what it measured and for one defect the audit in §1 understated.

**Stage 2 — Layer 4.** `--bench`, the orbit path, NDJSON. Turns the instrument into data on disk and makes `bench/reference/` reproducible rather than transcribed. Depends on Stage 1 only.

**Stage 3 — Layer 3.** `DeviceTimeline` and the `REMO_PROFILE` build variant. Deepest, highest-risk, and the only stage touching vendored device code — hence last, and hence guarded. By this point Stages 1 and 2 provide the CUevent totals that the phase sums are validated against.

---

## 4. Measurement protocol

The instrument is necessary but not sufficient; the protocol is what makes the numbers publishable. This section belongs in the repo because it must be recorded per run, not remembered.

- **Lock the clocks.** `nvidia-smi -pm 1` and `nvidia-smi -lgc <clock>`, recorded in the run header. This is not pedantry: `bench/reference/README.md` already documents a cold-vs-warm CudaLOD delta of roughly 25% at identical work, which is boost-clock ramp. Any effect smaller than that is unmeasurable without locking, and most interesting effects are.
- **Discard warm-up deliberately.** NVRTC compilation, first-launch cost, and clock ramp all land in the first frames. `--bench-warmup` defaults to 60; report the value used.
- **Report median and IQR over ≥300 post-warm-up frames**, not a mean over whatever the run happened to do. Means over GPU frame times are dominated by the tail, and the tail is exactly what a progressive builder produces.
- **Name the configuration in every claim.** The reference README's own lesson: CudaLOD's `WEIGHTED_NEIGHBORHOOD` voxelises 13× slower than `FIRST_COME` for bit-identical tree structure, so "CudaLOD does N MP/s" without naming the strategy is not a claim. The same will be true of any SimLOD number without the batch size and the device-side time budget.
- **Use `ncu` for what a hand-rolled profiler cannot see** — occupancy, achieved memory throughput, warp-stall reasons. One caveat specific to this codebase: every kernel here is a cooperative launch, and Nsight Compute's default kernel-level replay does not support grid-wide sync. Use `--replay-mode application`.
- **Record the input's provenance** — the file actually read, not just the point count. The original version of this bullet asserted that `.las` versus `.simlod` differ in the last f32 bit of the bounding box and that this is what shifts CudaLOD's voxel grid by 388 voxels. The `.las` reader has since landed and refuted it: all three formats give 12,742,500 voxels and byte-identical frames, because `Metadata::max_x` is a `float` and both headers narrow to the same value (see `bench/reference/README.md`). Provenance still belongs in the record — a run whose input you cannot name is not reproducible — but the +388 needs another explanation, and it is not the box.

---

## 5. Verification

Each stage has a concrete acceptance test, and one of them is an oracle that already exists.

**Stage 1.**
- `remobench --pipeline simlod --open data/morro_bay_35M/morro_bay_36M.simlod --strict-timing` — `simlod.render` must be non-zero and stable. This is the regression that defect 1.1 describes.
- Existing behaviour unchanged: `--dump-frame` still prints the same stats block with the same build totals, since the compatibility facade computes them from the profiler.
- Profiler overhead: frame time with and without scopes must be indistinguishable in the deferred regime. A profiler that costs a measurable fraction of the frame is measuring itself.

**Stage 2 — the oracle.**
- `remobench --pipeline cudalod --open data/morro_bay_35M/morro_bay_36M.simlod --bench`, once per sampling strategy. The `cudalod.split` and `cudalod.voxelize` medians must land on the table already recorded in `bench/reference/README.md`:

  | strategy | split | voxelize |
  |---|---|---|
  | 0 `FIRST_COME` (warm) | 5.1 ms | 4.1 ms |
  | 2 `AVERAGE_SINGLECELL` | 4.9 ms | 20.1 ms |
  | 3 `WEIGHTED_NEIGHBORHOOD` | 4.9 ms | 60.9 ms |

  Those numbers were captured from the upstream binary on this machine and are independent of anything in this plan. If the new instrument does not reproduce them, the instrument is wrong — which is exactly the property a measurement tool needs and exactly what this repo's structural comparisons already do for point and node counts.
- Structural invariants must also survive: 36,200,706 points and 2,252 nodes, as `bench/reference/README.md` records for the port.
- Determinism: two `--bench` runs with locked clocks must agree on medians to within a percent or two.
- Finally, capture the SimLOD baseline that `bench/reference/README.md` currently lists as missing — upstream cannot be scripted, but `--bench` can, which is the point.

**Stage 3.**
- `remobench --check-kernels` must pass both with and without `-DREMO_PROFILE`, confirming both variants compile and link.
- The default build's cached LTOIR must be unchanged from before Stage 3 — the guard's whole purpose.
- **The cross-check:** for the same launch, the sum of `DeviceTimeline` phase deltas must agree with the Layer 1 `CUevent` measurement to within a few percent. This validates both layers simultaneously — a disagreement means either an unmarked phase or a mark that is not sitting on a barrier.
- Sanity against known behaviour: under CudaLOD strategy 3, the voxelisation phase marks must show the ~13× cost over strategy 0 that the reference table attributes to `kernel3`, localised to the sampling phase rather than smeared across the kernel.

---

## 6. Files

**New**

| path | contents |
|---|---|
| `include/remo/GpuProfiler.h` | `GpuProfiler`, `GpuScope`, `ScopeStats`, `Regime` |
| `src/shell/GpuProfiler.cpp` | event pool, ring, harvest, Welford, percentiles |
| `src/shell/BenchRun.cpp` | orbit path, warm-up, NDJSON writer |

**Modified**

| path | change |
|---|---|
| `include/remo/ILodPipeline.h` | `GpuProfiler*` in `FrameContext`; retire the three timing doubles at the end of Stage 1 |
| `include/remo/HostDeviceCommon.h` | `DeviceTimeline`, `REMO_MAX_MARKS` |
| `src/pipelines/SimlodPipeline.{h,cpp}` | scopes replace `m_buildStart`/`m_buildEnd` and the local `eventMs`; add the missing render scope; read `DeviceTimeline` in `readStats()` |
| `src/pipelines/CudalodPipeline.{h,cpp}` | same, for split / voxelize / render; read `DeviceTimeline` in `readResults()` |
| `src/pipelines/RemolodPipeline.cpp` | same, for the baseline render scope |
| `src/shell/App.{h,cpp}` | own the profiler, populate `FrameContext`, `--bench` options and driver |
| `src/shell/SettingsPanel.cpp` | median / p95 in the timing rows |
| `src/main.cpp` | parse and document the `--bench*` flags |
| `kernels/shared/remo_prelude.cuh` | the `REMO_MARK` macro and its `#ifdef` guard |
| `kernels/simlod/progressive_octree_voxels.cu` | guarded marks at the existing `t_00`..`t_70` points |
| `kernels/cudalod/kernel.cu` | guarded marks in `kernel2` / `kernel3` |
| `bench/reference/README.md` | note that the tables are now reproducible via `--bench` |

**Reused rather than rebuilt**

- `OrbitControls::frameBox()` (`src/shell/OrbitControls.h:91-100`) — bench camera seeding.
- `KernelProgramDesc::defines` (`include/remo/CudaModularProgram.h`) — already part of the compile cache key, so the `REMO_PROFILE` variant needs no cache work.
- `CudaContext` accessors — run-header provenance.
- `nanotime()` (`kernels/simlod/utils.h.cu:322-327`) — the device clock read; do not write a second one.
- `PipelineStats` health flags (`include/remo/ILodPipeline.h:126-131`) — per-sample `warn` and the harness exit code.
- `now()`, `formatNumber()`, `writeFile()` from `include/remo/unsuck.hpp`.

---

## 7. Stage 1 as built

Landed as described, with three deviations and one correction to the audit above.

### 7.1 Defect 1.1 was worse than stated

§1.1 says only `RemolodPipeline` filled `renderDeviceMsLast`. It did — but **only under
`--strict-timing`**. In the default regime `RemolodPipeline::render` queried the event pair
it had just re-recorded three lines earlier, so `cuEventElapsedTime` returned
`CUDA_ERROR_NOT_READY` on every frame and the assignment never happened. The comment
there described reading "last frame's numbers", which needs a double-buffered pair; the
code had one pair. So in a default run **no pipeline had a render time**, not two of
three. Measured before the change: `remolod` printed `0.00` by default and `0.18` with
`--strict-timing`, same scene.

The profiler removes the failure mode rather than fixing the arithmetic: events are
pooled and recycled, and a sample whose events are not ready is left for the next
frame's harvest pass instead of being read early and discarded.

### 7.2 Deviations from §2

- **`GpuScope` takes a pointer, not a reference.** `FrameContext::profiler` is
  documented as never null in a normal frame, but a null dereference inside a render
  loop is a worse failure than an unmeasured scope, and a headless path that forgets to
  set it should not crash.
- **`TimingScopes timingScopes()` was added to `ILodPipeline`.** §2 left the shell to
  find a pipeline's scopes; nothing said how. Inferring them from the pipeline id and a
  naming convention would work right up until a pipeline added a phase and quietly
  stopped being counted, which is the same class of silent gap as defect 1.1. Pipelines
  declare their scope names; `buildTotals()` sums them.
- **`gui()` takes the profiler as a parameter** rather than the pipeline stashing one
  during `build()`. Same reason `FrameContext` exists.

The three public doubles are gone, as §2 planned for the end of Layer 2. `BuildTotals`
replaces them and carries a `measured` flag, so "no build scope produced a sample" is
distinguishable from "the build took 0.00 ms" — a distinction the old interface could
not express. `GpuProfiler::find()` returns null for a scope with no samples in the
current regime, and `timingRow()` renders that as `not measured`.

### 7.3 Clearing policy

Samples are dropped exactly when they stop describing the same work:

- **On cloud load** — the whole profiler. A new scene invalidates everything.
- **NOT on a pipeline switch.** The cloud, camera and pixel budget are unchanged, so
  keeping `remolod.render` alongside `simlod.render` is the entire point — the control
  condition and the thing being measured, side by side. Verified: after
  `--switch-to cudalod --switch-after 20`, both scopes are present in one output.
- **On a SimLOD reset**, `simlod.*` only. The tree those samples describe is gone.
- **On a CudaLOD strategy change**, `cudalod.*` only. `WEIGHTED_NEIGHBORHOOD` voxelises
  ~13× slower than `FIRST_COME` for a bit-identical tree, so a median pooled across two
  strategies describes neither. Repeated rebuilds at a *fixed* strategy deliberately do
  accumulate — press rebuild ten times and the median is worth more than any one run.

### 7.4 Acceptance, measured

RTX 5080, 84 SMs, sm_120, clocks not locked.

`--pipeline cudalod --open morro_bay_36M.simlod --strict-timing`, strategy 0, against
the reference table in `bench/reference/README.md`:

| scope | reference | measured | run 2 |
|---|---|---|---|
| `cudalod.split` | 5.1 ms | 4.99 ms | 5.14 ms |
| `cudalod.voxelize` | 4.1 ms | 4.08 ms | 4.06 ms |

Within ~2%, with the structural invariants intact (36,200,706 points, 2,252 nodes). The
oracle §5 asked for holds for strategy 0. **Strategies 2 and 3 are not yet verifiable
from a script**: the sampling strategy is a GUI-only radio button, so `--bench` in Stage
2 has to reach it. That also leaves the strategy-change clearing path in §7.3 exercised
only by hand.

`--pipeline simlod --open morro_bay_36M.simlod --strict-timing`, 400 frames:

```
simlod.construct  n=4    med  10.151  p95  10.547  min 10.142  max 10.547  total 41.04
simlod.render     n=399  med   0.193  p95   0.195  min  0.156  max  0.197
simlod.reset      n=1    med   3.502
```

Two things the running sum could not have shown:

1. **The device-side 10 ms `MAX_PROCESSING_TIME` budget holds**, and the overshoot is
   bounded: p95 is 10.55 ms against a 10 ms budget, max 10.55 ms. That is the shape §1.2
   argued for, and it is now one number rather than an assumption.
2. **`simlod.reset` costs 3.5 ms** — a one-block, one-thread kernel zeroing
   `BATCH_STREAM_SIZE` entries, which was never timed by anything and is 35% of a
   construct launch. It is in the build total now because §5's rule is that "cheap" is a
   measurement, not an assumption.

Render times now exist for all three pipelines in *both* regimes. On the same cloud,
camera and pixel budget: `remolod.render` 0.760 ms median (36.2M points, no selection)
against `cudalod.render` 0.105 ms median (374k visible samples). That comparison was
not previously expressible.

Profiler overhead is not measurable: three runs each of strict and deferred are all
vsync-bound at a 16.6–16.8 ms median, indistinguishable from each other.

### 7.5 Files, as built

New: `include/remo/GpuProfiler.h`, `src/shell/GpuProfiler.cpp`,
`src/shell/TimingUi.{h,cpp}` (one place that decides what an unmeasured scope looks
like, shared by the panel and all three `gui()`s).

Modified: `include/remo/ILodPipeline.h`, all three pipelines, `src/shell/App.{h,cpp}`,
`src/shell/SettingsPanel.cpp`, `CMakeLists.txt`. No device code was touched, so
`--check-kernels` still reports 6 programs from 8 modules, 0 failed.

`--dump-frame` now prints a per-scope table (n, last, median, p95, min, max, total) and
names the regime, which is the readout Stage 2 formalises into NDJSON.
