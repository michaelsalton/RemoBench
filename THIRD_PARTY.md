# Third-party code

RemoBench is a research tool built on two prior implementations by Markus Schütz
(TU Wien). It vendors code from both. This file records what came from where,
under what licence, and — the part that vendoring usually gets wrong — *what we
did with it*.

Three provenance categories are used throughout:

| category | meaning |
| -------- | ------- |
| `referenced` | used in place from `external/`, unmodified, at a pinned submodule commit |
| `vendored` | copied into this repo verbatim (or near-verbatim), not to be edited |
| `adapted` | copied into this repo and then changed — ours to maintain from here on |

Every `vendored` or `adapted` file carries a provenance header naming its upstream
path, the submodule commit it came from, and its copyright line.

For **vendored device code the header is exactly five lines** (upstream path, commit,
copyright, a pointer to the sidecar notes, then a blank line), followed by the upstream file
byte for byte. `bench/check_vendored.sh` — `make check-vendored` — asserts this on every
vendored kernel, and explanatory notes live in `kernels/simlod/VENDORED.md` rather than in the
files, so "is this still upstream?" stays a `diff` and not a judgement call.

That matters beyond licensing hygiene: `simlod` and `cudalod` are **external comparison
baselines**, and they are only worth having while they still reproduce their published numbers
against `bench/reference/`. RemoLOD — this project's own pipeline — forks what it needs into
`kernels/remolod/` and changes the fork. `kernels/CudaPrint/` sits at that path, rather than
inside `kernels/simlod/`, so that upstream's `#include "../CudaPrint/CudaPrint.cuh"` resolves
unchanged; mirroring the upstream layout was cheaper than carrying an edited include line.

## Upstream projects

| project | licence | copyright | submodule |
| ------- | ------- | --------- | --------- |
| [SimLOD](https://github.com/m-schuetz/SimLOD) | MIT | 2023 Markus Schütz and Lukas Herzberger | `external/SimLOD` (branch `ubuntu`) |
| [CudaLOD](https://github.com/m-schuetz/CudaLOD) | MIT | 2022 Markus Schütz | `external/CudaLOD` (branch `main`) |

Both are MIT, so copying with attribution is unambiguously permitted. CudaLOD's
`LICENSE.md` additionally notes that some shader files are adapted from
[three.js](https://github.com/mrdoob/three.js) (also MIT); RemoBench does not use
those files.

Neither submodule is copied wholesale. They stay as submodules pinned to exact
commits — referenced, not absorbed — which is the cleanest provenance story
available. `patches/` holds the Linux/CUDA-13 fixes applied to them at build time.

## Libraries

Referenced in place from `external/SimLOD/libs/` (see the provenance rule in
`CMakeLists.txt`: no patch touches `libs/`, so it is a stable reference target).

| library | licence | how | note |
| ------- | ------- | --- | ---- |
| [GLEW](https://github.com/nigels-com/glew) | Modified BSD / MIT | `referenced`, `glew.c` compiled in | `glew.h` includes `<GL/glu.h>` unconditionally → `libglu1-mesa-dev` is a hard dependency |
| [GLM](https://github.com/g-truc/glm) | MIT | `referenced`, header-only | host-side camera math only; device math is our own header |
| [Dear ImGui](https://github.com/ocornut/imgui) | MIT | `referenced`, compiled in | **pinned deliberately** — see below |
| [ImPlot](https://github.com/epezent/implot) | MIT | `referenced`, compiled in | **pinned deliberately** — see below |
| [laszip](https://github.com/LASzip/LASzip) | **LGPL-2.1** | `referenced`, built SHARED | only used for `.laz`; see the licence note below |
| [GLFW](https://github.com/glfw/glfw) | zlib/libpng | own submodule `external/glfw`, pinned to tag **3.4** | see below |
| CUDA Toolkit (driver API, NVRTC, nvJitLink) | NVIDIA proprietary | system dependency | not redistributed |

### Why GLFW is our own submodule rather than referenced

`external/*/libs/glfw` contains headers plus a prebuilt `msvc2017_x64` `.lib` —
useless on Linux. SimLOD therefore `FetchContent`s GLFW 3.3.2 at configure time,
which needs network access on every fresh configure and forces
`-DCMAKE_POLICY_VERSION_MINIMUM=3.5`, because 3.3.2 declares a pre-3.5 minimum
that CMake 4 rejects outright. A submodule pinned to 3.4 avoids both and has a
better Wayland/X11 story. `find_package(glfw3)` is tried first, so a future
`apt install libglfw3-dev` transparently takes over.

### Why ImGui and ImPlot are pinned

SimLOD's `GLRenderer.cpp` calls `ImPlot::SetNextPlotLimitsX` and a 3-argument
`ImPlot::BeginPlot` — both **removed** from modern ImPlot. Referencing the
vendored copies keeps the pin automatic rather than aspirational. Once the shell's
plotting code is rewritten (it has to be, since that code cannot compile against
current ImPlot), these become upgradable.

### laszip and LGPL-2.1

laszip is the one non-permissive dependency. It is built as a **shared** library
and dynamically linked, which keeps the LGPL-2.1 obligation to supplying the
library's own source (available at its upstream, and vendored in the submodule).
It is needed only for `.laz` input, and every call into it lives in
[src/io/LazReader.cpp](src/io/LazReader.cpp) — one translation unit, so the licence
boundary is greppable rather than a matter of trust. Keep it that way; the
uncompressed `.las` path in `LasReader.cpp` deliberately does not use it.
If that obligation ever becomes inconvenient,
[lazperf](https://github.com/hobuinc/laz-perf) (Apache-2.0) is a drop-in
replacement direction and would also be faster, since it supports per-chunk
parallel decode.

## Deliberately NOT used

### `include/utils.h` (Morton encoding) — quarantined

Present and byte-identical in **both** submodules, and licensed
[CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/) (see the
comment at the top of the file, and the same code inlined at
`external/CudaLOD/modules/simlod/SimLOD.h:32-49`).

It must not be copied into RemoBench:

- the **NC** (non-commercial) clause is incompatible with an MIT project;
- the **SA** (share-alike) clause attempts to infect derivatives.

It is also unused by SimLOD's live code path, so nothing is lost. If RemoBench ever
needs Morton codes, write them from the published algorithm (or use
`_pdep_u64`/BMI2 host-side, which is faster anyway) and test against a naive
reference — do not copy these magic constants.

A negative entry is as valuable as a positive one; this is why it is recorded.

### Other upstream code intentionally skipped

| what | why |
| ---- | --- |
| `SimLOD/modules/CudaPrint/**` | a no-op on both host and device, yet threaded through all three kernel signatures |
| `SimLOD/modules/progressive_octree/progressive_octree_mno.cu` | stale; would not compile against the current `Uniforms`. Reference implementation only |
| `CudaLOD/modules/compute/**` | a second, unrelated GL-compute renderer that never feeds the CUDA LOD builder |
| `CudaLOD/modules/simlod/sampling_cuda/**`, `voxel_sampling_gentree/**` | earlier generations, not instantiated upstream |
| `CudaLOD/libs/openvr/**` | Windows-only binary; VR is off by default and the code paths never execute |
| ~9 dead `voxelize_*.cu` variants, `split_countsort`, `split_hashmap` | superseded precursors; upstream marks one "prototyping, dont use" |