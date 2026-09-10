#include "io/RawReader.h"

#include <cstdio>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

namespace remo {

namespace {
constexpr uint64_t kHeaderBytes = 24;
constexpr uint64_t kPointBytes = 16;
}  // namespace

bool readSimlod(const std::string& path, CloudMeta& meta,
                std::vector<Point>& points, std::string* err) {
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
		// The count is inferred from file size, so this is the only truncation check
		// the format permits. Refuse rather than silently dropping a partial point.
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

	float bbox[6] = {};
	in.read(reinterpret_cast<char*>(bbox), sizeof(bbox));
	if (!in) {
		if (err) *err = "failed reading the .simlod header";
		return false;
	}

	meta.numPoints = payload / kPointBytes;
	for (int i = 0; i < 3; ++i) {
		meta.boxMinOrig[i] = static_cast<double>(bbox[i]);
		meta.boxMaxOrig[i] = static_cast<double>(bbox[3 + i]);
	}
	meta.files = {path};
	meta.hasCompressed = false;

	// One resize, one read. Point is exactly 16 bytes and laid out identically to
	// the on-disk record, which is the whole reason this format is fast -- so read
	// straight into the destination rather than parsing per point.
	static_assert(sizeof(Point) == 16, "Point must match the .simlod record");
	points.resize(meta.numPoints);
	in.read(reinterpret_cast<char*>(points.data()),
	        static_cast<std::streamsize>(payload));
	if (static_cast<uint64_t>(in.gcount()) != payload) {
		if (err) *err = "short read on " + path;
		return false;
	}

	// The header's bbox is float32 and comes from a different code path than the
	// points, so it can disagree with them. Trust the points: a wrong bbox silently
	// mis-frames the camera and, later, mis-sizes an octree root.
	double lo[3] = {1e300, 1e300, 1e300};
	double hi[3] = {-1e300, -1e300, -1e300};
	for (const Point& p : points) {
		const double v[3] = {p.x, p.y, p.z};
		for (int i = 0; i < 3; ++i) {
			if (v[i] < lo[i]) lo[i] = v[i];
			if (v[i] > hi[i]) hi[i] = v[i];
		}
	}
	if (meta.numPoints > 0) {
		for (int i = 0; i < 3; ++i) {
			meta.boxMinOrig[i] = lo[i];
			meta.boxMaxOrig[i] = hi[i];
		}
	}

	return true;
}

}  // namespace remo
