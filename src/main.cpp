// RemoBench -- a point cloud viewer with swappable LOD generation pipelines.
//
// Note the command line. Both upstream projects load a cloud ONLY by drag-and-drop
// onto the window (SimLOD accepts no arguments at all; CudaLOD hardcodes the
// author's Windows paths and the Linux port had to add an env var). That is a
// liability for anything scripted -- a benchmark runner cannot drag a file -- so
// --open exists from the first commit and drag-and-drop is the convenience, not the
// mechanism.

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

// --check-kernels: NVRTC-compile and link kernel modules without opening a window.
//
// Highest value per line in the project. It turns "did I break a pipeline" into a
// single command, needs no display, and is the thing that will catch a CUDA-toolkit
// upgrade breaking the -default-device / cccl include setup. It is also the
// compile-only spike used to find out whether a foreign kernel tree (CudaLOD's, whose
// sources predate this option set) survives our compile flags at all -- worth knowing
// before writing host code against the wrong assumption.
//
// With no paths, the programs to check are read from each pipeline's programs.txt, which
// declares its link groups. Scanning for .cu files would be wrong: most of CudaLOD's are
// #include fragments that are not independently compilable.
//
// With explicit paths, each is its own program unless asGroup links them all into one.
// That distinction is not academic -- CudaLOD's kernel.cu and cudalod_render.cu reference
// symbols defined in lib.cu, and checked alone they fail with unresolved externs that say
// nothing about whether the code is compatible with our flags.
int checkKernels(const std::vector<std::string>& explicitPaths, bool usePtx,
                 bool asGroup) {
	// A CUDA context is required for nvJitLink to query the device architecture, but
	// no GL context and no window are.
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
		// Discover programs from each pipeline's programs.txt.
		//
		// NOT by scanning for .cu files: most of CudaLOD's .cu files are #include
		// fragments that are not independently compilable, so a blanket scan reports
		// pages of errors about code that is perfectly fine in its intended context.
		// See kernels/cudalod/programs.txt.
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
				// Strip comments and surrounding whitespace.
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
		// No kernel names: this checks that the module COMPILES and LINKS, without
		// assuming what its entry points are called. A module that links but whose
		// entry point is misnamed is caught by the pipeline that uses it.
		desc.kernels = {};
		desc.linkMode = usePtx ? remo::LinkMode::Ptx : remo::LinkMode::LtoIr;
		// No file watching: this is a one-shot check, not a session.
		desc.watch = false;

		remo::CudaModularProgram program(std::move(desc));
		const bool ok = program.ok();
		printf("%s  %s\n", ok ? "  ok  " : "FAILED", label.c_str());
		if (!ok) {
			// The error text is already on stderr from the compile; print it again
			// compactly so a CI log reads top-to-bottom.
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

// Prints exactly what the dataset dropdown will show, without opening a window or a CUDA
// context. Verifies the scan (and the .simlod point counts, which are derived from file
// size) from a script.
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
		"  --strict-timing     synchronise and read CUevents every frame.\n"
		"                      Accurate but slower; required for benchmarking,\n"
		"                      since the default reads timings one frame late.\n"
		"  --remolod-no-accum  build RemoLOD's octree without the per-node accumulator.\n"
		"                      The pass mutates no tree state, so the structural counts\n"
		"                      must be identical with it on and off -- that is its\n"
		"                      acceptance test, and this is what makes it scriptable.\n"
		"  -h, --help          this message\n"
		"\n"
		"Files can also be dropped onto the window.\n");
}

// Returns false if the flag is missing its argument, so a typo is an error rather
// than a silently ignored option.
bool takeArg(int argc, char** argv, int& i, const char* flag, std::string* out) {
	if (i + 1 >= argc) {
		fprintf(stderr, "remobench: %s needs an argument\n", flag);
		return false;
	}
	*out = argv[++i];
	return true;
}

}  // namespace

int main(int argc, char** argv) {
	remo::AppOptions options;

	// --check-kernels short-circuits everything else: no window, no cloud.
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
		} else if (arg == "--remolod-no-accum") {
			options.remolodNoAccum = true;
		} else if (!arg.empty() && arg[0] != '-') {
			// Bare path: a cloud normally, or a kernel to check in --check-kernels
			// mode. Note --check-kernels may appear after the path, so this is sorted
			// out below rather than here.
			options.files.push_back(arg);
		} else {
			fprintf(stderr, "remobench: unknown option '%s'\n", arg.c_str());
			printUsage();
			return 2;
		}
	}

	if (checkMode) {
		// Bare paths and --open paths are both taken as kernels to check here.
		checkPaths = options.files;
		return checkKernels(checkPaths, checkPtx, checkAsGroup);
	}

	remo::App app;
	std::string err;
	if (!app.init(options, &err)) {
		fprintf(stderr, "remobench: %s\n", err.c_str());
		return 1;
	}

	// run() returns normally on window close. SimLOD's loop() calls
	// exit(EXIT_SUCCESS) instead, which is why it has no shutdown path and could
	// never host a headless benchmark or a GPU test.
	return app.run();
}
