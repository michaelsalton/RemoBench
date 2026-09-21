// Adapted from SimLOD: include/CudaModularProgram.h
// Upstream: https://github.com/m-schuetz/SimLOD @ fa7891613c138bd41775ca72a47cd89e32a5a647
// Copyright 2023 Markus Schuetz and Lukas Herzberger -- MIT (see THIRD_PARTY.md)

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
#include "remo/unsuck.hpp"

namespace fs = std::filesystem;

namespace remo {

const std::string& kernelRoot();

namespace {

std::string envOr(const char* name, const std::string& fallback) {
	const char* v = std::getenv(name);
	return (v && *v) ? std::string(v) : fallback;
}

std::string cudaIncludeDir() {
#ifdef REMOBENCH_CUDA_INCLUDE_DIR
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
	headers.push_back(fs::path(projectIncludeDir()) / "remo" / "HostDeviceCommon.h");

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
	if (ec) return;
	const fs::path p = fs::path(cacheDir()) / (key + ".bin");
	const fs::path tmp = p.string() + ".tmp";
	{
		std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
		if (!out) return;
		out.write(data.data(), static_cast<std::streamsize>(data.size()));
	}
	fs::rename(tmp, p, ec);
	if (ec) fs::remove(tmp, ec);
}

class WatchHub {
public:
	static WatchHub& instance() {
		static WatchHub hub;
		return hub;
	}

	void watch(const std::string& path, std::weak_ptr<ReloadToken> token) {
		std::lock_guard<std::mutex> lock(m_mutex);
		auto& entry = m_watched[path];
		entry.push_back(std::move(token));
		if (entry.size() > 1) return;

		const std::string pathCopy = path;
		monitorFile(path, [this, pathCopy]() { dispatch(pathCopy); });
	}

private:
	void dispatch(const std::string& path) {
		std::vector<std::shared_ptr<ReloadToken>> live;
		{
			std::lock_guard<std::mutex> lock(m_mutex);
			auto it = m_watched.find(path);
			if (it == m_watched.end()) return;

			auto& tokens = it->second;
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

}

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

CudaModularProgram::CudaModularProgram(KernelProgramDesc desc)
	: m_desc(std::move(desc)) {

	CUdevice dev = 0;
	cuDeviceGet(&dev, 0);
	int major = 0, minor = 0;
	cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
	cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
	m_smArch = major * 10 + minor;

	m_arch = envOr("REMOBENCH_GPU_ARCH", "compute_" + std::to_string(m_smArch));

	for (const std::string& rel : m_desc.modules) {
		Module mod;
		const fs::path p(rel);
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
		m_token = std::make_shared<ReloadToken>();
		m_token->program = this;

		for (const Module& mod : m_modules) {
			WatchHub::instance().watch(mod.path, m_token);
		}

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
	if (m_token) {
		m_token->program = nullptr;
		m_token.reset();
	}
	unload();
}

void CudaModularProgram::onWatchedFileChanged() {
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
		"-I" + cudaIncludeDir() + "/cccl",
		"-I" + moduleDir,
		"-I" + kernelRoot(),
		"-I" + projectIncludeDir(),
		"--relocatable-device-code=true",
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
		cuLinkDestroy(state);
		if (r != CUDA_SUCCESS) {
			m_lastError = std::string("cuModuleLoadData failed: ") + cuErrorName(r);
			m_stale = m_loaded;
			return false;
		}
		unload();
		m_module = loaded;
	}

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

}
