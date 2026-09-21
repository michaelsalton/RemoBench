#include "io/RawReader.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <vector>

namespace fs = std::filesystem;

namespace remo {

namespace {
constexpr uint64_t kHeaderBytes = 24;
constexpr uint64_t kPointBytes = 16;

// Points per read when walking the file for bounds. 64K points is 1 MB, the same block
// size LasReader uses, and keeps the scratch off the stack.
constexpr uint64_t kScanPoints = 64 * 1024;

bool openAt(const std::string& path, uint64_t firstPoint, std::ifstream& in,
            std::string* err) {
	in.open(path, std::ios::binary);
	if (!in) {
		if (err) *err = "cannot open " + path;
		return false;
	}
	in.seekg(static_cast<std::streamoff>(kHeaderBytes + firstPoint * kPointBytes));
	if (!in) {
		if (err) *err = "cannot seek into " + path;
		return false;
	}
	return true;
}
}

bool readSimlodHeader(const std::string& path, SimlodInfo& info, std::string* err) {
	std::error_code ec;
	const uint64_t fileSize = fs::file_size(path, ec);
	if (ec) {
		if (err) *err = "cannot stat " + path;
		return false;
	}
	if (fileSize < kHeaderBytes) {
		if (err) *err = "file is smaller than a .simlod header: " + path;
		return false;
	}

	const uint64_t payload = fileSize - kHeaderBytes;
	if (payload % kPointBytes != 0) {
		if (err) {
			*err = "point data is not a multiple of 16 bytes (truncated?): " + path;
		}
		return false;
	}

	std::ifstream in(path, std::ios::binary);
	if (!in) {
		if (err) *err = "cannot open " + path;
		return false;
	}
	in.read(reinterpret_cast<char*>(info.headerBox), sizeof(info.headerBox));
	if (!in) {
		if (err) *err = "failed reading the .simlod header";
		return false;
	}

	info.numPoints = payload / kPointBytes;
	if (info.numPoints == 0) {
		if (err) *err = ".simlod file holds no points: " + path;
		return false;
	}
	return true;
}

bool readSimlodBounds(const std::string& path, const SimlodInfo& info, double bounds[6],
                      std::string* err) {
	std::ifstream in;
	if (!openAt(path, 0, in, err)) return false;

	double lo[3] = {1e300, 1e300, 1e300};
	double hi[3] = {-1e300, -1e300, -1e300};

	std::vector<Point> buf(static_cast<size_t>(kScanPoints));
	for (uint64_t p = 0; p < info.numPoints; p += kScanPoints) {
		const uint64_t n = std::min<uint64_t>(kScanPoints, info.numPoints - p);
		const uint64_t bytes = n * kPointBytes;
		in.read(reinterpret_cast<char*>(buf.data()),
		        static_cast<std::streamsize>(bytes));
		if (static_cast<uint64_t>(in.gcount()) != bytes) {
			if (err) *err = "short read on " + path;
			return false;
		}
		for (uint64_t i = 0; i < n; ++i) {
			const double v[3] = {buf[i].x, buf[i].y, buf[i].z};
			for (int k = 0; k < 3; ++k) {
				if (v[k] < lo[k]) lo[k] = v[k];
				if (v[k] > hi[k]) hi[k] = v[k];
			}
		}
	}

	for (int i = 0; i < 3; ++i) {
		bounds[i] = lo[i];
		bounds[3 + i] = hi[i];
	}
	return true;
}

bool readSimlodRange(const std::string& path, const SimlodInfo& info,
                     const double translation[3], uint64_t firstPoint,
                     uint64_t numPoints, Point* out, std::string* err) {
	if (firstPoint + numPoints > info.numPoints) {
		if (err) *err = ".simlod point range runs past the end of the file: " + path;
		return false;
	}

	std::ifstream in;
	if (!openAt(path, firstPoint, in, err)) return false;

	const uint64_t bytes = numPoints * kPointBytes;
	in.read(reinterpret_cast<char*>(out), static_cast<std::streamsize>(bytes));
	if (static_cast<uint64_t>(in.gcount()) != bytes) {
		if (err) *err = "short read on " + path;
		return false;
	}

	const float tx = static_cast<float>(translation[0]);
	const float ty = static_cast<float>(translation[1]);
	const float tz = static_cast<float>(translation[2]);
	if (tx == 0.0f && ty == 0.0f && tz == 0.0f) return true;
	for (uint64_t i = 0; i < numPoints; ++i) {
		out[i].x += tx;
		out[i].y += ty;
		out[i].z += tz;
	}
	return true;
}


}
