// Adapted from SimLOD: include/CudaModularProgram.h
// Upstream: https://github.com/m-schuetz/SimLOD @ fa7891613c138bd41775ca72a47cd89e32a5a647
// Copyright 2023 Markus Schuetz and Lukas Herzberger -- MIT (see THIRD_PARTY.md)
//
// Runtime NVRTC compilation + nvJitLink linking, with hot reload.
//
// The API shape is upstream's and deliberately preserved -- construct with a list
// of .cu modules and a list of kernel names, get CUfunctions back, and every
// module is watched so saving it recompiles and relinks. That design is the single
// best idea in either research codebase and the reason this project can iterate on
// device code without restarting.
//
// The implementation is rewritten. Upstream's is unusable for experimentation:
//
//   - NVJITLINK_SAFE_CALL calls exit(1) on any link error, so one typo in a kernel
//     kills the session and discards a freshly ingested point cloud. Here a failed
//     compile or link is NON-FATAL: the previously loaded module stays live, the
//     error text is retained for the GUI, and you fix the file and save again.
//   - cu_checked only cout'd an error code and carried on.
//   - Leaks on every path: the ltoir buffer per recompile, the cubin per link, and
//     the nvrtcProgram on the compile-error branch.
//   - The target architecture came from an env var (SIMLOD_GPU_ARCH), so compiling
//     a kernel required remembering to set it. Here it is queried from the device,
//     with the env var demoted to an override.
//   - No compile cache, so every launch re-ran NVRTC for every module before the
//     first frame. Compiled LTOIR/PTX is now cached on disk, keyed on the source
//     text and the full option set.
//
// One option is worth calling out because it surprises everyone reading a .cu file
// here: -default-device. It makes NVRTC treat unannotated functions as __device__,
// which is why the kernel sources and .cuh headers carry no __device__ annotations
// and read like plain C++. It is inherited from upstream and kept, because it is
// genuinely nicer to author -- but it means these files will NOT compile under nvcc
// as-is.

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda.h>

namespace remo {

class CudaModularProgram;

// Liveness handle for file watching.
//
// unsuck.hpp's monitorFile() spawns a detached, uncancellable thread that retains the
// callback forever, so a callback capturing a program directly would fire into freed
// memory once that program is destroyed -- which is routine, since switching pipeline
// destroys one. The watcher holds a weak_ptr to this instead and skips dead programs.
struct ReloadToken {
	CudaModularProgram* program = nullptr;
};

// How to get from NVRTC output to a loadable cubin.
//
// LtoIr is the good path: NVRTC emits LTOIR, nvJitLink does link-time optimisation
// across modules. Ptx exists as an escape hatch -- CudaLOD's device sources come
// from a --std=c++17 / driver-JIT era and may not survive -default-device with LTO.
// Having both available from the start is ~40 lines of insurance against having to
// restructure when those kernels get ported.
enum class LinkMode { LtoIr, Ptx };

struct KernelProgramDesc {
	// .cu files to compile. Relative paths resolve against the kernels root.
	std::vector<std::string> modules;
	// extern "C" __global__ entry points to resolve after linking.
	std::vector<std::string> kernels;

	LinkMode linkMode = LinkMode::LtoIr;

	// Extra NVRTC options, e.g. "-DREMO_LOD_PIXELS=1". Part of the cache key.
	std::vector<std::string> defines;

	// Watch each module and recompile on save. Off for one-shot/test compiles.
	bool watch = true;
};

// Returns the kernels root: $REMOBENCH_KERNEL_DIR, else the compiled-in source path,
// else ./kernels.
const std::string& kernelRoot();

class CudaModularProgram {
public:
	explicit CudaModularProgram(KernelProgramDesc desc);
	~CudaModularProgram();

	CudaModularProgram(const CudaModularProgram&) = delete;
	CudaModularProgram& operator=(const CudaModularProgram&) = delete;

	// A loaded, linked module is available and every requested kernel resolved.
	// False after a failed FIRST build; a failed *re*build leaves this true and the
	// previous kernels valid.
	bool ok() const { return m_loaded; }

	// nullptr if unknown or if the first build failed. Callers must check.
	CUfunction kernel(const std::string& name) const;

	// Compiler/linker diagnostics from the most recent attempt. Empty on success.
	// Intended to be shown in the GUI rather than printed and forgotten.
	const std::string& lastError() const { return m_lastError; }

	// True if the most recent rebuild attempt failed while an older module is still
	// live -- i.e. what you see on screen is stale relative to the source on disk.
	bool isStale() const { return m_stale; }

	// Fired after every successful link, including hot reloads. Pipelines use this
	// to re-read kernel handles and to trigger a reset, since a tree built by the
	// previous version of a construct kernel is not necessarily valid input to the
	// new one.
	void onCompile(std::function<void()> callback);

	// Force a full recompile + relink, ignoring the disk cache.
	void rebuild();

	// Called by the watch hub, on the main thread, when a module or a header this program
	// could include has changed on disk. Public because the hub dispatches to it; not
	// intended for callers.
	void onWatchedFileChanged();

private:
	struct Module {
		std::string path;
		std::string name;
		std::vector<char> image;   // LTOIR or PTX, per linkMode
		bool success = false;
	};

	bool compile(Module& mod);
	bool link();
	void unload();

	std::string optionsSignature() const;
	std::vector<std::string> nvrtcOptions(const std::string& moduleDir) const;

	KernelProgramDesc m_desc;
	std::vector<Module> m_modules;

	CUmodule m_module = nullptr;
	bool m_loaded = false;
	bool m_stale = false;
	std::string m_lastError;

	uint32_t m_linkCount = 0;  // >0 means a later link is a hot reload

	// Cleared in the destructor, so watchers holding a weak_ptr to it stop dispatching
	// here. See ReloadToken above.
	std::shared_ptr<ReloadToken> m_token;

	std::string m_arch;        // e.g. "compute_120"
	int m_smArch = 0;          // e.g. 120

	std::unordered_map<std::string, CUfunction> m_kernels;
	std::vector<std::function<void()>> m_callbacks;
};

}  // namespace remo
