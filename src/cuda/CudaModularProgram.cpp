// Adapted from SimLOD: include/CudaModularProgram.h
// Upstream: https://github.com/m-schuetz/SimLOD @ fa7891613c138bd41775ca72a47cd89e32a5a647
// Copyright 2023 Markus Schuetz and Lukas Herzberger -- MIT (see THIRD_PARTY.md)
//
// See include/remo/CudaModularProgram.h for what changed and why.

#include "remo/CudaModularProgram.h"

#include <nvJitLink.h>
#include <nvrtc.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>

#include "remo/CudaCheck.h"
// For monitorFile() / EventQueue -- the hot-reload substrate.
#include "remo/unsuck.hpp"

namespace fs = std::filesystem;

namespace remo {

// Defined below; used by the helpers in the anonymous namespace.
const std::string& kernelRoot();

namespace {

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

std::string envOr(const char* name, const std::string& fallback) {
	const char* v = std::getenv(name);
	return (v && *v) ? std::string(v) : fallback;
}

std::string cudaIncludeDir() {
#ifdef REMOBENCH_CUDA_INCLUDE_DIR
	// Baked in by CMake from CUDAToolkit_INCLUDE_DIRS. Both research repos read
	// CUDA_PATH from the environment instead, which is why they need an env var
	// set just to compile a kernel. CUDA_PATH still wins if explicitly set.
	const std::string builtin = REMOBENCH_CUDA_INCLUDE_DIR;
#else
	const std::string builtin = "/usr/local/cuda/include";
#endif
	const char* cudaPath = std::getenv("CUDA_PATH");
	if (cudaPath && *cudaPath) return std::string(cudaPath) + "/include";
	return builtin;
}

std::string projectIncludeDir() {
#ifdef REMOBENCH_INCLUDE_DIR
	return envOr("REMOBENCH_INCLUDE_DIR", REMOBENCH_INCLUDE_DIR);
#else
	return envOr("REMOBENCH_INCLUDE_DIR", "include");
#endif
}

std::string cacheDir() {
	static const std::string dir = envOr(
		"REMOBENCH_CACHE_DIR",
		(fs::temp_directory_path() / "remobench-kernel-cache").string());
	return dir;
}

// FNV-1a. Only needs to be stable within a run of this binary and collision-
// resistant enough that two different kernel sources do not share a cache entry.
uint64_t hash64(const std::string& s, uint64_t seed = 0xcbf29ce484222325ull) {
	uint64_t h = seed;
	for (unsigned char c : s) {
		h ^= c;
		h *= 0x100000001b3ull;
	}
	return h;
}

std::string readTextOrEmpty(const std::string& path) {
	std::ifstream in(path, std::ios::binary);
	if (!in) return {};
	std::ostringstream ss;
	ss << in.rdbuf();
	return ss.str();
}

// Fingerprint of every header a kernel module could include.
//
// This exists because of a bug worth remembering: keying the compile cache on the
// module's own source text alone means editing a shared .cuh changes nothing the
// key can see, so the cache happily serves LTOIR built from the OLD header. The
// symptom is editing a rasteriser header, saving, and seeing the image not change
// -- which reads as "my edit was wrong" rather than "the cache lied".
//
// Hashing every candidate header is deliberately coarse: touching one header
// invalidates every module. That is the right trade here, since a shared-header edit
// is exactly when you want everything rebuilt, and there are only a handful of them.
uint64_t dependencyFingerprint() {
	std::vector<fs::path> headers;

	std::error_code ec;
	for (fs::recursive_directory_iterator it(kernelRoot(), ec), end; it != end;
	     it.increment(ec)) {
		if (ec) break;
		if (!it->is_regular_file(ec)) continue;
		const std::string ext = it->path().extension().string();
		if (ext == ".cuh" || ext == ".h") headers.push_back(it->path());
	}
	// The one header shared with host code.
	headers.push_back(fs::path(projectIncludeDir()) / "remo" / "HostDeviceCommon.h");

	// Sort so the fingerprint does not depend on directory iteration order.
	std::sort(headers.begin(), headers.end());

	uint64_t h = 0xcbf29ce484222325ull;
	for (const fs::path& p : headers) {
		h = hash64(p.string(), h);
		h = hash64(readTextOrEmpty(p.string()), h);
	}
	return h;
}

bool readCache(const std::string& key, std::vector<char>& out) {
	const fs::path p = fs::path(cacheDir()) / (key + ".bin");
	std::error_code ec;
	const auto size = fs::file_size(p, ec);
	if (ec || size == 0) return false;
	std::ifstream in(p, std::ios::binary);
	if (!in) return false;
	out.resize(static_cast<size_t>(size));
	in.read(out.data(), static_cast<std::streamsize>(size));
	return static_cast<size_t>(in.gcount()) == out.size();
}

void writeCache(const std::string& key, const std::vector<char>& data) {
	std::error_code ec;
	fs::create_directories(cacheDir(), ec);
	if (ec) return;  // caching is best-effort; never fail a build over it
	const fs::path p = fs::path(cacheDir()) / (key + ".bin");
	// Write-then-rename so a killed process cannot leave a truncated entry that
	// would later be loaded as valid device code.
	const fs::path tmp = p.string() + ".tmp";
	{
		std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
		if (!out) return;
		out.write(data.data(), static_cast<std::streamsize>(data.size()));
	}
	fs::rename(tmp, p, ec);
	if (ec) fs::remove(tmp, ec);
}

// ---------------------------------------------------------------------------
// File-watch hub
//
// monitorFile() spawns a DETACHED, INFINITE thread per call, with no way to cancel it,
// and the thread keeps a copy of the callback -- which captures the CudaModularProgram.
// Used naively that is wrong in two compounding ways:
//
//   1. The watcher outlives the program. Switching pipeline destroys a program, and its
//      watchers keep polling with a dangling `this`.
//   2. Every program re-registers every shared header. remolod is 1 module + ~7 headers;
//      cudalod is two programs x (2 modules + 7 headers). One switch left ~35 immortal
//      threads, each constructing an fs::path (a heap allocation) every 20ms forever.
//      The observed crash was a SIGSEGV inside _int_malloc on a *thread* arena, in
//      exactly that allocation.
//
// The hub fixes both: at most ONE OS-level watcher per distinct path for the lifetime of
// the process, and delivery through weak_ptr tokens so a dead program is simply skipped.
// Dispatch still lands on the main thread, because monitorFile routes through
// schedule()/EventQueue.
class WatchHub {
public:
	static WatchHub& instance() {
		static WatchHub hub;
		return hub;
	}

	// Interest in `path` on behalf of `token`. Cheap and idempotent per path.
	void watch(const std::string& path, std::weak_ptr<ReloadToken> token) {
		std::lock_guard<std::mutex> lock(m_mutex);
		auto& entry = m_watched[path];
		entry.push_back(std::move(token));
		if (entry.size() > 1) return;  // already have an OS watcher for this path

		const std::string pathCopy = path;
		monitorFile(path, [this, pathCopy]() { dispatch(pathCopy); });
	}

private:
	void dispatch(const std::string& path) {
		// Runs on the main thread, drained from the EventQueue by the render loop.
		std::vector<std::shared_ptr<ReloadToken>> live;
		{
			std::lock_guard<std::mutex> lock(m_mutex);
			auto it = m_watched.find(path);
			if (it == m_watched.end()) return;

			auto& tokens = it->second;
			// Reap dead tokens while we are here, so a long session that switches
			// pipelines repeatedly does not accumulate them.
			tokens.erase(std::remove_if(tokens.begin(), tokens.end(),
			                            [](const std::weak_ptr<ReloadToken>& w) {
				                            return w.expired();
			                            }),
			             tokens.end());
			for (const std::weak_ptr<ReloadToken>& w : tokens) {
				if (std::shared_ptr<ReloadToken> s = w.lock()) live.push_back(s);
			}
		}
		for (const std::shared_ptr<ReloadToken>& token : live) {
			if (token->program) token->program->onWatchedFileChanged();
		}
	}

	std::mutex m_mutex;
	std::unordered_map<std::string, std::vector<std::weak_ptr<ReloadToken>>> m_watched;
};

}  // namespace

const std::string& kernelRoot() {
	static const std::string root = [] {
#ifdef REMOBENCH_KERNEL_DIR
		return envOr("REMOBENCH_KERNEL_DIR", REMOBENCH_KERNEL_DIR);
#else
		return envOr("REMOBENCH_KERNEL_DIR", "kernels");
#endif
	}();
	return root;
}

// ---------------------------------------------------------------------------

CudaModularProgram::CudaModularProgram(KernelProgramDesc desc)
	: m_desc(std::move(desc)) {

	// Query the target architecture from the device rather than the environment.
	// Note upstream already did this for the *linker* arch but not the *compile*
	// arch, which is how the two could disagree.
	CUdevice dev = 0;
	cuDeviceGet(&dev, 0);
	int major = 0, minor = 0;
	cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
	cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
	m_smArch = major * 10 + minor;

	// REMOBENCH_GPU_ARCH overrides, for cross-compiling or reproducing a bug on
	// another target.
	m_arch = envOr("REMOBENCH_GPU_ARCH", "compute_" + std::to_string(m_smArch));

	for (const std::string& rel : m_desc.modules) {
		Module mod;
		const fs::path p(rel);
		// Absolute wins; then an existing path relative to the working directory
		// (so `remobench --check-kernels external/.../kernel.cu` does what it looks
		// like); then relative to the kernels root, which is how pipelines name
		// their own modules ("remolod/remolod_render.cu").
		std::error_code ec;
		if (p.is_absolute()) {
			mod.path = p.string();
		} else if (fs::exists(p, ec)) {
			mod.path = fs::absolute(p, ec).string();
		} else {
			mod.path = (fs::path(kernelRoot()) / p).string();
		}
		mod.name = p.filename().string();
		m_modules.push_back(std::move(mod));
	}

	for (Module& mod : m_modules) compile(mod);
	link();

	if (m_desc.watch) {
		// All watching goes through the hub: one OS watcher per distinct path for the
		// whole process, delivered via a weak_ptr so a destroyed program is skipped
		// rather than called into. See WatchHub for why the naive form is unsafe.
		m_token = std::make_shared<ReloadToken>();
		m_token->program = this;

		for (const Module& mod : m_modules) {
			WatchHub::instance().watch(mod.path, m_token);
		}

		// Also watch the shared headers. Upstream watches only the listed .cu files, so
		// editing a header they include changes nothing until a restart -- which,
		// combined with a compile cache, is a silently stale image. Since the whole
		// point of these headers is that every pipeline shares them, an edit there must
		// rebuild everything that could include them.
		std::error_code ec;
		for (fs::recursive_directory_iterator it(kernelRoot(), ec), end; it != end;
		     it.increment(ec)) {
			if (ec) break;
			if (!it->is_regular_file(ec)) continue;
			const std::string ext = it->path().extension().string();
			if (ext != ".cuh" && ext != ".h") continue;
			WatchHub::instance().watch(it->path().string(), m_token);
		}
	}
}

CudaModularProgram::~CudaModularProgram() {
	// Sever the watchers BEFORE tearing anything down. The token is what they hold; once
	// it is cleared and released, any in-flight dispatch skips this program.
	if (m_token) {
		m_token->program = nullptr;
		m_token.reset();
	}
	unload();
}

void CudaModularProgram::onWatchedFileChanged() {
	// A watched module or header changed. Recompile everything in this program: NVRTC
	// gives us no include graph, so we cannot tell which modules a given header actually
	// affects, and a shared-header edit is exactly when a full rebuild is wanted anyway.
	for (Module& mod : m_modules) compile(mod);
	link();
}

void CudaModularProgram::unload() {
	if (m_module) {
		cuModuleUnload(m_module);
		m_module = nullptr;
	}
	m_kernels.clear();
	m_loaded = false;
}

CUfunction CudaModularProgram::kernel(const std::string& name) const {
	auto it = m_kernels.find(name);
	return it == m_kernels.end() ? nullptr : it->second;
}

void CudaModularProgram::onCompile(std::function<void()> callback) {
	m_callbacks.push_back(std::move(callback));
}

void CudaModularProgram::rebuild() {
	std::error_code ec;
	fs::remove_all(cacheDir(), ec);
	for (Module& mod : m_modules) compile(mod);
	link();
}

std::string CudaModularProgram::optionsSignature() const {
	std::string sig = m_arch;
	sig += m_desc.linkMode == LinkMode::LtoIr ? "|lto" : "|ptx";
	for (const std::string& d : m_desc.defines) sig += "|" + d;
	int nvrtcMajor = 0, nvrtcMinor = 0;
	nvrtcVersion(&nvrtcMajor, &nvrtcMinor);
	sig += "|nvrtc" + std::to_string(nvrtcMajor) + "." + std::to_string(nvrtcMinor);
	return sig;
}

std::vector<std::string> CudaModularProgram::nvrtcOptions(
	const std::string& moduleDir) const {

	std::vector<std::string> opts = {
		"--gpu-architecture=" + m_arch,
		"--use_fast_math",
		"--extra-device-vectorization",
		"-lineinfo",
		"-I" + cudaIncludeDir(),
		// CUDA 13 relocated the CCCL / libcu++ headers (<cuda/std/*>, pulled in
		// by cooperative_groups) into include/cccl. Harmless on CUDA 12.
		"-I" + cudaIncludeDir() + "/cccl",
		"-I" + moduleDir,
		// So a pipeline can #include "shared/remo_math.cuh".
		"-I" + kernelRoot(),
		// So a kernel can #include "remo/HostDeviceCommon.h" -- the single header
		// shared between host C++ and device code.
		"-I" + projectIncludeDir(),
		"--relocatable-device-code=true",
		// See the header comment: this is why kernel sources carry no __device__.
		"-default-device",
		"--std=c++20",
		"--disable-warnings",
	};

	if (m_desc.linkMode == LinkMode::LtoIr) opts.push_back("-dlto");

	for (const std::string& d : m_desc.defines) opts.push_back(d);
	return opts;
}

bool CudaModularProgram::compile(Module& mod) {
	mod.success = false;

	const std::string source = readTextOrEmpty(mod.path);
	if (source.empty()) {
		m_lastError = "cannot read kernel source: " + mod.path;
		fprintf(stderr, "remobench: %s\n", m_lastError.c_str());
		return false;
	}

	const std::string sig = optionsSignature();
	const uint64_t deps = dependencyFingerprint();
	const std::string key =
		mod.name + "-" +
		std::to_string(hash64(sig + "\0" + source, deps ? deps : 1ull));

	if (readCache(key, mod.image)) {
		mod.success = true;
		return true;
	}

	const std::string dir = fs::path(mod.path).parent_path().string();
	const std::vector<std::string> optStrings = nvrtcOptions(dir);
	std::vector<const char*> opts;
	opts.reserve(optStrings.size());
	for (const std::string& o : optStrings) opts.push_back(o.c_str());

	nvrtcProgram prog = nullptr;
	if (nvrtcCreateProgram(&prog, source.c_str(), mod.name.c_str(), 0, nullptr,
	                       nullptr) != NVRTC_SUCCESS) {
		m_lastError = "nvrtcCreateProgram failed for " + mod.name;
		return false;
	}

	const nvrtcResult res =
		nvrtcCompileProgram(prog, static_cast<int>(opts.size()), opts.data());

	if (res != NVRTC_SUCCESS) {
		size_t logSize = 0;
		nvrtcGetProgramLogSize(prog, &logSize);
		std::string log(logSize ? logSize - 1 : 0, '\0');
		if (logSize) nvrtcGetProgramLog(prog, log.data());

		m_lastError = "compile failed: " + mod.name + "\n" + log;
		fprintf(stderr, "remobench: %s\n", m_lastError.c_str());

		// Upstream leaked the program on this branch.
		nvrtcDestroyProgram(&prog);
		return false;
	}

	size_t imageSize = 0;
	nvrtcResult getRes;
	if (m_desc.linkMode == LinkMode::LtoIr) {
		getRes = nvrtcGetLTOIRSize(prog, &imageSize);
		if (getRes == NVRTC_SUCCESS) {
			mod.image.resize(imageSize);
			getRes = nvrtcGetLTOIR(prog, mod.image.data());
		}
	} else {
		getRes = nvrtcGetPTXSize(prog, &imageSize);
		if (getRes == NVRTC_SUCCESS) {
			mod.image.resize(imageSize);
			getRes = nvrtcGetPTX(prog, mod.image.data());
		}
	}
	nvrtcDestroyProgram(&prog);

	if (getRes != NVRTC_SUCCESS) {
		m_lastError = "could not retrieve compiled image for " + mod.name;
		return false;
	}

	writeCache(key, mod.image);
	mod.success = true;
	return true;
}

bool CudaModularProgram::link() {
	for (const Module& mod : m_modules) {
		if (!mod.success) {
			// A module failed to compile. Keep whatever is currently loaded so the
			// session survives a typo; flag staleness so the GUI can say so.
			m_stale = m_loaded;
			return false;
		}
	}

	std::string linkError;
	void* cubin = nullptr;
	size_t cubinSize = 0;

	if (m_desc.linkMode == LinkMode::LtoIr) {
		const std::string archOpt = "-arch=sm_" + std::to_string(m_smArch);
		const char* lopts[] = {"-dlto", archOpt.c_str()};

		nvJitLinkHandle handle = nullptr;
		if (nvJitLinkCreate(&handle, 2, lopts) != NVJITLINK_SUCCESS) {
			m_lastError = "nvJitLinkCreate failed";
			m_stale = m_loaded;
			return false;
		}

		auto fail = [&](const char* what) {
			size_t logSize = 0;
			std::string log;
			if (nvJitLinkGetErrorLogSize(handle, &logSize) == NVJITLINK_SUCCESS &&
			    logSize > 1) {
				log.resize(logSize - 1);
				nvJitLinkGetErrorLog(handle, log.data());
			}
			linkError = std::string(what) + " failed\n" + log;
			nvJitLinkDestroy(&handle);
		};

		for (const Module& mod : m_modules) {
			if (nvJitLinkAddData(handle, NVJITLINK_INPUT_LTOIR,
			                     static_cast<const void*>(mod.image.data()),
			                     mod.image.size(), mod.name.c_str()) !=
			    NVJITLINK_SUCCESS) {
				fail(("nvJitLinkAddData(" + mod.name + ")").c_str());
				m_lastError = linkError;
				m_stale = m_loaded;
				return false;
			}
		}
		if (nvJitLinkComplete(handle) != NVJITLINK_SUCCESS) {
			fail("nvJitLinkComplete");
			m_lastError = linkError;
			m_stale = m_loaded;
			return false;
		}
		if (nvJitLinkGetLinkedCubinSize(handle, &cubinSize) != NVJITLINK_SUCCESS) {
			fail("nvJitLinkGetLinkedCubinSize");
			m_lastError = linkError;
			m_stale = m_loaded;
			return false;
		}
		std::vector<char> buffer(cubinSize);
		if (nvJitLinkGetLinkedCubin(handle, buffer.data()) != NVJITLINK_SUCCESS) {
			fail("nvJitLinkGetLinkedCubin");
			m_lastError = linkError;
			m_stale = m_loaded;
			return false;
		}
		nvJitLinkDestroy(&handle);

		// Load before unloading the old module, so a load failure is also non-fatal.
		CUmodule loaded = nullptr;
		const CUresult lr = cuModuleLoadData(&loaded, buffer.data());
		if (lr != CUDA_SUCCESS) {
			m_lastError = std::string("cuModuleLoadData failed: ") + cuErrorName(lr);
			m_stale = m_loaded;
			return false;
		}
		unload();
		m_module = loaded;
	} else {
		// Driver JIT path. cuLinkAddData has no LTOIR input kind, so PTX is linked
		// without LTO -- slower device code, but it accepts sources that the LTOIR
		// path rejects.
		CUlinkState state = nullptr;
		const std::string archOpt = std::to_string(m_smArch);
		CUjit_option jitOpts[] = {CU_JIT_TARGET};
		void* jitVals[] = {reinterpret_cast<void*>(
			static_cast<uintptr_t>(m_smArch))};
		(void)archOpt;

		CUresult r = cuLinkCreate(1, jitOpts, jitVals, &state);
		if (r != CUDA_SUCCESS) {
			m_lastError = std::string("cuLinkCreate failed: ") + cuErrorName(r);
			m_stale = m_loaded;
			return false;
		}
		for (const Module& mod : m_modules) {
			// cuLinkAddData wants a NUL-terminated PTX string.
			std::vector<char> ptx = mod.image;
			if (ptx.empty() || ptx.back() != '\0') ptx.push_back('\0');
			r = cuLinkAddData(state, CU_JIT_INPUT_PTX, ptx.data(), ptx.size(),
			                  mod.name.c_str(), 0, nullptr, nullptr);
			if (r != CUDA_SUCCESS) {
				m_lastError = "cuLinkAddData(" + mod.name + ") failed: " +
				              cuErrorName(r);
				cuLinkDestroy(state);
				m_stale = m_loaded;
				return false;
			}
		}
		r = cuLinkComplete(state, &cubin, &cubinSize);
		if (r != CUDA_SUCCESS) {
			m_lastError = std::string("cuLinkComplete failed: ") + cuErrorName(r);
			cuLinkDestroy(state);
			m_stale = m_loaded;
			return false;
		}
		CUmodule loaded = nullptr;
		r = cuModuleLoadData(&loaded, cubin);
		// The cubin is owned by the link state and freed with it, so this must
		// happen after the load and not before.
		cuLinkDestroy(state);
		if (r != CUDA_SUCCESS) {
			m_lastError = std::string("cuModuleLoadData failed: ") + cuErrorName(r);
			m_stale = m_loaded;
			return false;
		}
		unload();
		m_module = loaded;
	}

	// Resolve entry points.
	bool allFound = true;
	for (const std::string& name : m_desc.kernels) {
		CUfunction fn = nullptr;
		const CUresult r = cuModuleGetFunction(&fn, m_module, name.c_str());
		if (r != CUDA_SUCCESS) {
			m_lastError = "kernel not found in linked module: " + name +
			              " (is it extern \"C\" __global__?)";
			fprintf(stderr, "remobench: %s\n", m_lastError.c_str());
			allFound = false;
			continue;
		}
		m_kernels[name] = fn;
	}

	if (!allFound) {
		m_stale = m_loaded;
		return false;
	}

	const bool wasReload = m_linkCount > 0;
	++m_linkCount;

	m_loaded = true;
	m_stale = false;
	m_lastError.clear();

	// Report reloads. The whole value of hot reload is a tight save-and-watch loop,
	// and that loop needs a visible confirmation that the thing on screen is the
	// thing on disk -- otherwise a stale image is indistinguishable from a bad edit.
	if (wasReload) {
		std::string names;
		for (const Module& mod : m_modules) {
			if (!names.empty()) names += ", ";
			names += mod.name;
		}
		printf("remobench: reloaded [%s]\n", names.c_str());
		fflush(stdout);
	}

	for (auto& cb : m_callbacks) cb();
	return true;
}

}  // namespace remo
