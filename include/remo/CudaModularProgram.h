// Adapted from SimLOD: include/CudaModularProgram.h
// Upstream: https://github.com/m-schuetz/SimLOD @ fa7891613c138bd41775ca72a47cd89e32a5a647
// Copyright 2023 Markus Schuetz and Lukas Herzberger -- MIT (see THIRD_PARTY.md)

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

struct ReloadToken {
	CudaModularProgram* program = nullptr;
};

enum class LinkMode { LtoIr, Ptx };

struct KernelProgramDesc {
	std::vector<std::string> modules;
	std::vector<std::string> kernels;

	LinkMode linkMode = LinkMode::LtoIr;

	std::vector<std::string> defines;

	bool watch = true;
};

const std::string& kernelRoot();

class CudaModularProgram {
public:
	explicit CudaModularProgram(KernelProgramDesc desc);
	~CudaModularProgram();

	CudaModularProgram(const CudaModularProgram&) = delete;
	CudaModularProgram& operator=(const CudaModularProgram&) = delete;

	bool ok() const { return m_loaded; }

	CUfunction kernel(const std::string& name) const;

	const std::string& lastError() const { return m_lastError; }

	bool isStale() const { return m_stale; }

	void onCompile(std::function<void()> callback);

	void rebuild();

	void onWatchedFileChanged();

private:
	struct Module {
		std::string path;
		std::string name;
		std::vector<char> image;
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

	uint32_t m_linkCount = 0;

	std::shared_ptr<ReloadToken> m_token;

	std::string m_arch;
	int m_smArch = 0;

	std::unordered_map<std::string, CUfunction> m_kernels;
	std::vector<std::function<void()>> m_callbacks;
};

}
