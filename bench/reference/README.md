# Reference baselines

Upstream SimLOD numbers captured on this machine, via `make simlod`, are the oracle
RemoBench's ported pipeline is validated against — not legacy cruft. Recapture after any
submodule bump or driver change. **No upstream capture exists yet** (see below); what this
file holds today is RemoBench's own structural counts.

## Machine

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5080, 16303 MiB, compute capability 12.0 (`sm_120`), 84 SMs |
| Display | 3840x1600 |
| CUDA | 13.1 |
| Compiler | g++ 15.2.0, CMake 4.2.3 |
| Dataset | `data/morro_bay_35M/morro_bay_36M.las`, 36,200,706 points |

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
