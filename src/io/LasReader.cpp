#include "io/LasReader.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <thread>

namespace fs = std::filesystem;

namespace remo {

namespace {

constexpr uint64_t kHeaderBytesCommon = 227;
constexpr uint64_t kHeaderBytesMax = 375;

constexpr size_t kOffSignature = 0;
constexpr size_t kOffVersionMajor = 24;
constexpr size_t kOffVersionMinor = 25;
constexpr size_t kOffHeaderSize = 94;
constexpr size_t kOffPointDataOffset = 96;
constexpr size_t kOffPointFormat = 104;
constexpr size_t kOffPointRecordLength = 105;
constexpr size_t kOffLegacyNumPoints = 107;
constexpr size_t kOffScaleX = 131;
constexpr size_t kOffOffsetX = 155;
constexpr size_t kOffMaxX = 179;
constexpr size_t kOffMinX = 187;
constexpr size_t kOffExtendedNumPoints = 247;

template <typename T>
T readField(const std::vector<char>& buf, size_t offset) {
	T v{};
	std::memcpy(&v, buf.data() + offset, sizeof(T));
	return v;
}

struct FormatGeometry {
	uint32_t standardBytes;
	uint32_t rgbOffset;
};

bool formatGeometry(uint32_t format, FormatGeometry& out) {
	switch (format) {
		case 0:  out = {20, 0};  return true;
		case 1:  out = {28, 0};  return true;
		case 2:  out = {26, 20}; return true;
		case 3:  out = {34, 28}; return true;
		case 4:  out = {57, 0};  return true;
		case 5:  out = {63, 28}; return true;
		case 6:  out = {30, 0};  return true;
		case 7:  out = {36, 30}; return true;
		case 8:  out = {38, 30}; return true;
		case 9:  out = {59, 0};  return true;
		case 10: out = {67, 30}; return true;
		default: return false;
	}
}

constexpr uint64_t kBlockPoints = 64 * 1024;

constexpr unsigned kMaxLoaderThreads = 8;

struct Bounds {
	double lo[3] = {1e300, 1e300, 1e300};
	double hi[3] = {-1e300, -1e300, -1e300};

	void add(float x, float y, float z) {
		const double v[3] = {x, y, z};
		for (int i = 0; i < 3; ++i) {
			if (v[i] < lo[i]) lo[i] = v[i];
			if (v[i] > hi[i]) hi[i] = v[i];
		}
	}

	void merge(const Bounds& o) {
		for (int i = 0; i < 3; ++i) {
			if (o.lo[i] < lo[i]) lo[i] = o.lo[i];
			if (o.hi[i] > hi[i]) hi[i] = o.hi[i];
		}
	}
};

}

bool readLasHeader(const std::string& path, LasHeaderInfo& info, std::string* err) {
	std::error_code ec;
	const uint64_t fileSize = fs::file_size(path, ec);
	if (ec) {
		if (err) *err = "cannot stat " + path;
		return false;
	}
	if (fileSize < kHeaderBytesCommon) {
		if (err) *err = "file is smaller than a LAS public header block: " + path;
		return false;
	}

	std::ifstream in(path, std::ios::binary);
	if (!in) {
		if (err) *err = "cannot open " + path;
		return false;
	}

	const size_t want = static_cast<size_t>(std::min(fileSize, kHeaderBytesMax));
	std::vector<char> buf(want, 0);
	in.read(buf.data(), static_cast<std::streamsize>(want));
	if (static_cast<size_t>(in.gcount()) != want) {
		if (err) *err = "short read on the LAS header: " + path;
		return false;
	}

	if (std::memcmp(buf.data() + kOffSignature, "LASF", 4) != 0) {
		if (err) {
			*err = "not a LAS/LAZ file (missing the LASF signature): " + path;
		}
		return false;
	}

	info.versionMajor = readField<uint8_t>(buf, kOffVersionMajor);
	info.versionMinor = readField<uint8_t>(buf, kOffVersionMinor);

	const uint32_t headerSize = readField<uint16_t>(buf, kOffHeaderSize);
	info.offsetToPointData = readField<uint32_t>(buf, kOffPointDataOffset);

	const uint8_t rawFormat = readField<uint8_t>(buf, kOffPointFormat);
	info.format = rawFormat & 0x3F;
	info.compressed = (rawFormat & 0xC0) != 0;

	std::string ext = fs::path(path).extension().string();
	std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
		return static_cast<char>(std::tolower(c));
	});
	if (ext == ".laz") info.compressed = true;

	info.bytesPerPoint = readField<uint16_t>(buf, kOffPointRecordLength);

	if (info.versionMajor == 1 && info.versionMinor >= 4 &&
	    want >= kOffExtendedNumPoints + 8) {
		info.numPoints = readField<uint64_t>(buf, kOffExtendedNumPoints);
	}
	if (info.numPoints == 0) {
		info.numPoints = readField<uint32_t>(buf, kOffLegacyNumPoints);
	}

	for (int i = 0; i < 3; ++i) {
		info.scale[i] = readField<double>(buf, kOffScaleX + 8 * i);
		info.offset[i] = readField<double>(buf, kOffOffsetX + 8 * i);
		info.max[i] = readField<double>(buf, kOffMaxX + 16 * i);
		info.min[i] = readField<double>(buf, kOffMinX + 16 * i);
	}

	FormatGeometry geom{};
	if (!formatGeometry(info.format, geom)) {
		if (err) {
			*err = "unsupported LAS point data record format " +
			       std::to_string(info.format);
		}
		return false;
	}
	info.rgbOffset = geom.rgbOffset;

	if (info.bytesPerPoint < geom.standardBytes) {
		if (err) {
			*err = "point record length " + std::to_string(info.bytesPerPoint) +
			       " is shorter than format " + std::to_string(info.format) +
			       " requires (" + std::to_string(geom.standardBytes) + ")";
		}
		return false;
	}
	if (info.offsetToPointData < headerSize) {
		if (err) *err = "offset to point data lies inside the header: " + path;
		return false;
	}
	if (info.scale[0] == 0.0 || info.scale[1] == 0.0 || info.scale[2] == 0.0) {
		if (err) *err = "LAS header declares a zero scale factor: " + path;
		return false;
	}

	if (!info.compressed) {
		const uint64_t need =
			info.offsetToPointData +
			static_cast<uint64_t>(info.bytesPerPoint) * info.numPoints;
		if (need > fileSize) {
			if (err) {
				*err = "LAS header claims " + std::to_string(info.numPoints) +
				       " points but the file holds only " +
				       std::to_string((fileSize - info.offsetToPointData) /
				                      info.bytesPerPoint) +
				       " (truncated?): " + path;
			}
			return false;
		}
	}

	return true;
}

bool readLasPoints(const std::string& path, const LasHeaderInfo& info,
                   const double translation[3], Point* out,
                   double translatedBounds[6], std::string* err) {
	if (info.compressed) {
		if (err) *err = "compressed input must go through readLazPoints";
		return false;
	}
	if (info.numPoints == 0) {
		if (err) *err = "LAS header declares zero points: " + path;
		return false;
	}

	const uint32_t bpp = info.bytesPerPoint;
	const uint32_t rgbOffset = info.rgbOffset;
	const double sx = info.scale[0], sy = info.scale[1], sz = info.scale[2];
	const double ox = info.offset[0] + translation[0];
	const double oy = info.offset[1] + translation[1];
	const double oz = info.offset[2] + translation[2];

	const uint64_t numBlocks = (info.numPoints + kBlockPoints - 1) / kBlockPoints;
	const unsigned hw = std::thread::hardware_concurrency();
	unsigned numThreads = std::min<unsigned>(hw ? hw : 4u, kMaxLoaderThreads);
	numThreads = static_cast<unsigned>(
		std::min<uint64_t>(numThreads, std::max<uint64_t>(numBlocks, 1)));

	const uint64_t blocksPerThread = (numBlocks + numThreads - 1) / numThreads;

	std::atomic<bool> failed{false};
	std::mutex errMutex;
	std::string firstError;
	std::vector<Bounds> perThread(numThreads);

	const auto fail = [&](const std::string& message) {
		std::lock_guard<std::mutex> lock(errMutex);
		if (firstError.empty()) firstError = message;
		failed.store(true, std::memory_order_relaxed);
	};

	const auto worker = [&](unsigned t) {
		const uint64_t firstPoint = t * blocksPerThread * kBlockPoints;
		if (firstPoint >= info.numPoints) return;
		const uint64_t lastPoint = std::min(
			info.numPoints, firstPoint + blocksPerThread * kBlockPoints);

		std::ifstream in(path, std::ios::binary);
		if (!in) {
			fail("cannot open " + path);
			return;
		}

		std::vector<uint8_t> buf(static_cast<size_t>(kBlockPoints) * bpp);
		Bounds bounds;

		for (uint64_t p = firstPoint; p < lastPoint; p += kBlockPoints) {
			if (failed.load(std::memory_order_relaxed)) return;

			const uint64_t n = std::min<uint64_t>(kBlockPoints, lastPoint - p);
			const uint64_t bytes = n * bpp;

			in.seekg(static_cast<std::streamoff>(info.offsetToPointData + p * bpp));
			in.read(reinterpret_cast<char*>(buf.data()),
			        static_cast<std::streamsize>(bytes));
			if (static_cast<uint64_t>(in.gcount()) != bytes) {
				fail("short read on " + path);
				return;
			}

			for (uint64_t i = 0; i < n; ++i) {
				const uint8_t* record = buf.data() + i * bpp;

				int32_t xyz[3];
				std::memcpy(xyz, record, 12);

				Point point;
				point.x = static_cast<float>(static_cast<double>(xyz[0]) * sx + ox);
				point.y = static_cast<float>(static_cast<double>(xyz[1]) * sy + oy);
				point.z = static_cast<float>(static_cast<double>(xyz[2]) * sz + oz);

				if (rgbOffset != 0) {
					uint16_t rgb[3];
					std::memcpy(rgb, record + rgbOffset, 6);
					point.color = packLasColor(rgb);
				} else {
					point.color = kLasNoColor;
				}

				out[p + i] = point;
				bounds.add(point.x, point.y, point.z);
			}
		}

		perThread[t] = bounds;
	};

	std::vector<std::thread> threads;
	threads.reserve(numThreads);
	for (unsigned t = 0; t < numThreads; ++t) threads.emplace_back(worker, t);
	for (std::thread& thread : threads) thread.join();

	if (failed.load(std::memory_order_relaxed)) {
		if (err) *err = firstError;
		return false;
	}

	Bounds total;
	for (const Bounds& b : perThread) total.merge(b);
	for (int i = 0; i < 3; ++i) {
		translatedBounds[i] = total.lo[i];
		translatedBounds[3 + i] = total.hi[i];
	}
	return true;
}

}
