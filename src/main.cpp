#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "remo/CudaContext.h"
#include "remo/CudaModularProgram.h"
#include "shell/App.h"

namespace fs = std::filesystem;

namespace {

int checkKernels(const std::vector<std::string>& explicitPaths, bool usePtx,
                 bool asGroup) {
	remo::CudaContext cuda;
	printf("remobench: %s, sm_%d%d\n", cuda.deviceName().c_str(), cuda.ccMajor(),
	       cuda.ccMinor());

	std::vector<std::vector<std::string>> groups;
	size_t moduleCount = 0;

	if (!explicitPaths.empty()) {
		if (asGroup) {
			groups.push_back(explicitPaths);
		} else {
			for (const std::string& p : explicitPaths) groups.push_back({p});
		}
		moduleCount = explicitPaths.size();
	} else {
		std::vector<fs::path> manifests;
		std::error_code ec;
		for (fs::recursive_directory_iterator it(remo::kernelRoot(), ec), end;
		     it != end; it.increment(ec)) {
			if (ec) break;
			if (it->is_regular_file(ec) && it->path().filename() == "programs.txt") {
				manifests.push_back(it->path());
			}
		}
		std::sort(manifests.begin(), manifests.end());

		for (const fs::path& manifest : manifests) {
			std::ifstream in(manifest);
			std::string line;
			while (std::getline(in, line)) {
				const size_t hash = line.find('#');
				if (hash != std::string::npos) line.resize(hash);
				std::istringstream ls(line);
				std::vector<std::string> modules;
				std::string token;
				while (ls >> token) {
					modules.push_back((manifest.parent_path() / token).string());
				}
				if (modules.empty()) continue;
				moduleCount += modules.size();
				groups.push_back(std::move(modules));
			}
		}

		if (groups.empty()) {
			fprintf(stderr,
			        "remobench: no programs.txt found under %s\n"
			        "         (a pipeline declares its link groups there)\n",
			        remo::kernelRoot().c_str());
			return 1;
		}
	}

	if (groups.empty()) {
		fprintf(stderr, "remobench: nothing to check\n");
		return 1;
	}

	int failures = 0;
	for (const std::vector<std::string>& group : groups) {
		std::string label;
		for (const std::string& p : group) {
			if (!label.empty()) label += " + ";
			label += fs::path(p).filename().string();
		}

		remo::KernelProgramDesc desc;
		desc.modules = group;
		desc.kernels = {};
		desc.linkMode = usePtx ? remo::LinkMode::Ptx : remo::LinkMode::LtoIr;
		desc.watch = false;

		remo::CudaModularProgram program(std::move(desc));
		const bool ok = program.ok();
		printf("%s  %s\n", ok ? "  ok  " : "FAILED", label.c_str());
		if (!ok) {
			const std::string& err = program.lastError();
			if (!err.empty()) printf("        %s\n", err.c_str());
			++failures;
		}
	}

	printf("remobench: %zu program(s) from %zu module(s), %d failed (%s)\n",
	       groups.size(), moduleCount, failures,
	       usePtx ? "PTX + driver JIT" : "LTOIR + nvJitLink");
	return failures == 0 ? 0 : 1;
}

int listDatasets() {
	std::string dir = "data";
	if (const char* env = std::getenv("REMOBENCH_DATA_DIR")) {
		if (*env) dir = env;
	}

	const std::vector<remo::DatasetEntry> entries = remo::scanDatasetDir(dir);
	if (entries.empty()) {
		printf("remobench: no .simlod / .las / .laz found under %s/\n", dir.c_str());
		return 1;
	}

	printf("%-40s %10s %14s  %s\n", "dataset", "size", "points", "status");
	for (const remo::DatasetEntry& e : entries) {
		char points[32] = "-";
		if (e.numPoints > 0) snprintf(points, sizeof(points), "%llu",
		                              static_cast<unsigned long long>(e.numPoints));
		printf("%-40s %8.1f MB %14s  %s\n", e.label.c_str(),
		       static_cast<double>(e.bytes) / (1024.0 * 1024.0), points,
		       e.supported ? "ok" : e.note.c_str());
	}
	return 0;
}

void printUsage() {
	printf(
		"remobench -- point cloud viewer with swappable LOD pipelines\n"
		"\n"
		"usage: remobench [options]\n"
		"\n"
		"  --open <file>       load a .simlod / .las / .laz point cloud\n"
		"  --synthetic <n>     generate n synthetic points instead\n"
		"  --pipeline <id>     start with this pipeline (default: flat)\n"
		"  --size <w> <h>      window size (default: 1600 900)\n"
		"  --dump-frame <ppm>  render, write the frame to a binary PPM, exit.\n"
		"                      Headless verification, and the basis of the\n"
		"                      golden-image comparison.\n"
		"  --dump-after <n>    frames to render before dumping (default 8)\n"
		"  --dump-ui           include the ImGui overlay in the dump\n"
		"  --list-datasets     list the clouds the dataset dropdown will show, and exit\n"
		"  --check-kernels     compile+link every program declared in a\n"
		"                      kernels/*/programs.txt, and exit. No window needed.\n"
		"                      Pass paths to check specific files instead.\n"
		"  --ptx               with --check-kernels: use the PTX + driver-JIT path\n"
		"                      instead of LTOIR + nvJitLink\n"
		"  --as-group          with --check-kernels: link all given files into ONE\n"
		"                      program, for modules that are not standalone\n"
		"  --switch-to <id>    request a runtime pipeline switch (see --switch-after),\n"
		"                      so the switch path is scriptable and not GUI-only\n"
		"  --switch-after <n>  frame at which to switch (default 0)\n"
		"  --show-bounds       draw a wireframe cube per selected node, coloured by\n"
		"                      level -- the LOD cut itself. `flat` has no tree and\n"
		"                      draws none.\n"
		"  --hide-points       do not rasterise samples. With --show-bounds this\n"
		"                      leaves the octree structure alone on screen, which is\n"
		"                      the only way to read it in a dense cloud.\n"
		"  --device-budget <n> pin the device memory budget instead of deriving it\n"
		"                      from what happens to be free. Accepts 9G / 8Gi / bytes.\n"
		"                      Every memory figure is relative to this, so a benchmark\n"
		"                      capture that does not pin it is not comparable.\n"
		"  --strict-timing     synchronise and read CUevents every frame.\n"
		"                      Accurate but slower; required for benchmarking,\n"
		"                      since the default reads timings one frame late.\n"
		"  --remolod-phase-timings\n"
		"                      break RemoLOD's construct kernel down by phase (expand,\n"
		"                      voxelSampling, insertPoints, ...). Compiles the kernel\n"
		"                      with -DREMO_PROFILE; off, the device code is unchanged.\n"
		"  --remolod-fixed-depth <n>\n"
		"                      build RemoLOD's octree by pre-splitting to a fixed\n"
		"                      depth n (1..8) instead of counting and splitting\n"
		"                      iteratively. An oracle arm: the depth is free, so this\n"
		"                      measures the ceiling on what a depth predictor could\n"
		"                      save. Compiles the kernel with -DREMO_FIXED_DEPTH=n.\n"
		"  --remolod-no-accum  build RemoLOD's octree without the per-node accumulator.\n"
		"                      The pass mutates no tree state, so the structural counts\n"
		"                      must be identical with it on and off -- that is its\n"
		"                      acceptance test, and this is what makes it scriptable.\n"
		"  -h, --help          this message\n"
		"\n"
		"Files can also be dropped onto the window.\n");
}

// Plain bytes, or a K/M/G suffix -- decimal by default, binary with a trailing `i`
// (so 10G is 10e9 and 10Gi is 10 * 2^30). Written out rather than reached for with
// strtoull alone because a budget given in bytes is unreadable at these sizes and a
// silently misparsed one would move every memory figure in the capture.
bool parseByteSize(const std::string& text, size_t* out) {
	char* end = nullptr;
	const double value = std::strtod(text.c_str(), &end);
	if (end == text.c_str() || value < 0.0) return false;

	std::string suffix(end);
	suffix.erase(0, suffix.find_first_not_of(" \t"));

	bool binary = false;
	if (!suffix.empty() && (suffix.back() == 'i' || suffix.back() == 'I')) {
		binary = true;
		suffix.pop_back();
	}
	if (!suffix.empty() && (suffix.back() == 'b' || suffix.back() == 'B')) {
		suffix.pop_back();
	}

	double scale = 1.0;
	if (suffix.empty()) {
		if (binary) return false;
	} else if (suffix.size() == 1) {
		switch (std::tolower(static_cast<unsigned char>(suffix[0]))) {
			case 'k': scale = binary ? 1024.0 : 1e3; break;
			case 'm': scale = binary ? 1024.0 * 1024.0 : 1e6; break;
			case 'g': scale = binary ? 1024.0 * 1024.0 * 1024.0 : 1e9; break;
			default: return false;
		}
	} else {
		return false;
	}

	*out = static_cast<size_t>(value * scale);
	return true;
}

bool takeArg(int argc, char** argv, int& i, const char* flag, std::string* out) {
	if (i + 1 >= argc) {
		fprintf(stderr, "remobench: %s needs an argument\n", flag);
		return false;
	}
	*out = argv[++i];
	return true;
}

}

int main(int argc, char** argv) {
	remo::AppOptions options;

	bool checkMode = false;
	bool checkPtx = false;
	bool checkAsGroup = false;
	std::vector<std::string> checkPaths;

	for (int i = 1; i < argc; ++i) {
		const std::string arg = argv[i];

		if (arg == "-h" || arg == "--help") {
			printUsage();
			return 0;
		} else if (arg == "--list-datasets") {
			return listDatasets();
		} else if (arg == "--check-kernels") {
			checkMode = true;
		} else if (arg == "--ptx") {
			checkPtx = true;
		} else if (arg == "--as-group") {
			checkAsGroup = true;
		} else if (arg == "--open") {
			std::string path;
			if (!takeArg(argc, argv, i, "--open", &path)) return 2;
			options.files.push_back(path);
		} else if (arg == "--synthetic") {
			std::string n;
			if (!takeArg(argc, argv, i, "--synthetic", &n)) return 2;
			options.syntheticPoints = std::strtoull(n.c_str(), nullptr, 10);
		} else if (arg == "--pipeline") {
			if (!takeArg(argc, argv, i, "--pipeline", &options.pipeline)) return 2;
		} else if (arg == "--size") {
			std::string w, h;
			if (!takeArg(argc, argv, i, "--size", &w)) return 2;
			if (!takeArg(argc, argv, i, "--size", &h)) return 2;
			options.width = std::atoi(w.c_str());
			options.height = std::atoi(h.c_str());
		} else if (arg == "--dump-frame") {
			if (!takeArg(argc, argv, i, "--dump-frame", &options.dumpFramePath))
				return 2;
		} else if (arg == "--dump-ui") {
			options.dumpIncludeGui = true;
		} else if (arg == "--dump-after") {
			std::string n;
			if (!takeArg(argc, argv, i, "--dump-after", &n)) return 2;
			options.dumpAfterFrames = std::atoi(n.c_str());
		} else if (arg == "--switch-to") {
			if (!takeArg(argc, argv, i, "--switch-to", &options.switchToPipeline))
				return 2;
		} else if (arg == "--switch-after") {
			std::string n;
			if (!takeArg(argc, argv, i, "--switch-after", &n)) return 2;
			options.switchAfterFrames = std::atoi(n.c_str());
		} else if (arg == "--show-bounds") {
			options.showBoundingBox = true;
		} else if (arg == "--hide-points") {
			options.hidePoints = true;
		} else if (arg == "--strict-timing") {
			options.strictTiming = true;
		} else if (arg == "--device-budget") {
			std::string text;
			if (!takeArg(argc, argv, i, "--device-budget", &text)) return 2;
			if (!parseByteSize(text, &options.deviceBudgetBytes)) {
				fprintf(stderr,
				        "remobench: cannot parse --device-budget '%s' "
				        "(try 9G, 8Gi, or a plain byte count)\n",
				        text.c_str());
				return 2;
			}
		} else if (arg == "--remolod-no-accum") {
			options.remolodNoAccum = true;
		} else if (arg == "--remolod-phase-timings") {
			options.remolodPhaseTimings = true;
		} else if (arg == "--remolod-fixed-depth") {
			std::string n;
			if (!takeArg(argc, argv, i, "--remolod-fixed-depth", &n)) return 2;
			options.remolodFixedDepth = std::atoi(n.c_str());
			// 9 is 613 MB of cell counters against a 512 MB momentary buffer, and
			// its occupancy grids exceed the card. The kernel static_asserts the
			// same range; this is so the refusal is a message and not a compile
			// error in a hot-reloaded kernel.
			if (options.remolodFixedDepth < 1 || options.remolodFixedDepth > 8) {
				fprintf(stderr,
				        "remobench: --remolod-fixed-depth takes 1..8, got '%s'\n",
				        n.c_str());
				return 2;
			}
		} else if (!arg.empty() && arg[0] != '-') {
			options.files.push_back(arg);
		} else {
			fprintf(stderr, "remobench: unknown option '%s'\n", arg.c_str());
			printUsage();
			return 2;
		}
	}

	if (checkMode) {
		checkPaths = options.files;
		return checkKernels(checkPaths, checkPtx, checkAsGroup);
	}

	remo::App app;
	std::string err;
	if (!app.init(options, &err)) {
		fprintf(stderr, "remobench: %s\n", err.c_str());
		return 1;
	}

	return app.run();
}
