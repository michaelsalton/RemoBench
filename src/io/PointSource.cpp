#include "remo/PointSource.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <random>

#include "remo/CudaCheck.h"
#include "remo/CudaContext.h"
#include "io/LasReader.h"
#include "io/RawReader.h"

namespace fs = std::filesystem;

namespace remo {

double CloudMeta::worstQuantisationError() const {
	const double extent = std::max({std::fabs(static_cast<double>(boxSize[0])),
	                                std::fabs(static_cast<double>(boxSize[1])),
	                                std::fabs(static_cast<double>(boxSize[2]))});
	return extent * std::pow(2.0, -23.0);
}

namespace {

class ResidentSource final : public PointSource {
public:
	ResidentSource(CudaContext& cuda, CloudMeta meta, std::vector<Point> points)
		: m_cuda(cuda), m_meta(std::move(meta)), m_points(std::move(points)) {}

	~ResidentSource() override { stop(); }

	const CloudMeta& meta() const override { return m_meta; }

	bool start(Mode mode, size_t maxDeviceBytes, std::string* err) override {
		if (mode != Mode::Whole) {
			if (err) {
				*err = "streaming ingest is not implemented yet; this source is "
				       "whole-cloud only";
			}
			return false;
		}
		if (m_devicePoints) return true;

		const size_t bytes = m_points.size() * sizeof(Point);
		if (maxDeviceBytes != 0 && bytes > maxDeviceBytes) {
			if (err) {
				*err = "point data needs " + std::to_string(bytes / (1024 * 1024)) +
				       " MB but the budget allows " +
				       std::to_string(maxDeviceBytes / (1024 * 1024)) + " MB";
			}
			return false;
		}
		if (bytes == 0) {
			if (err) *err = "point cloud is empty";
			return false;
		}

		if (REMO_CU(cuMemAlloc(&m_devicePoints, bytes)) != CUDA_SUCCESS) {
			if (err) *err = "cuMemAlloc failed for the point buffer";
			m_devicePoints = 0;
			return false;
		}
		if (REMO_CU(cuMemcpyHtoD(m_devicePoints, m_points.data(), bytes)) !=
		    CUDA_SUCCESS) {
			if (err) *err = "uploading points failed";
			stop();
			return false;
		}

		m_uploaded = m_points.size();

		const uint32_t slotCapacity = 1'000'000;
		const uint32_t numSlots = static_cast<uint32_t>(
			(m_meta.numPoints + slotCapacity - 1) / slotCapacity);

		if (!allocBatchMetadata(numSlots, slotCapacity, err)) {
			stop();
			return false;
		}

		m_points.clear();
		m_points.shrink_to_fit();
		return true;
	}

	void rewind() override {
	}

	void stop() override {
		for (CUdeviceptr* p : {&m_devicePoints, &m_batchSizes, &m_numBatchesUploaded}) {
			if (*p) {
				REMO_CU(cuMemFree(*p));
				*p = 0;
			}
		}
		m_uploaded = 0;
		m_numSlots = 0;
	}

	BatchView view() const override {
		BatchView v;
		v.slots = m_devicePoints;
		v.batchSizes = m_batchSizes;
		v.numBatchesUploaded = m_numBatchesUploaded;
		v.slotCapacity = m_slotCapacity;
		v.numSlots = m_numSlots;
		v.numBatchesTotal = m_numSlots;
		v.wrapping = false;
		return v;
	}

	uint64_t numPointsUploaded() const override { return m_uploaded; }
	bool isFullyResident() const override {
		return m_devicePoints != 0 && m_uploaded == m_meta.numPoints;
	}
	CUdeviceptr residentPoints() const override { return m_devicePoints; }
	void setPointsConsumed(uint64_t) override {}

private:
	bool allocBatchMetadata(uint32_t numSlots, uint32_t slotCapacity,
	                        std::string* err) {
		m_numSlots = numSlots;
		m_slotCapacity = slotCapacity;

		std::vector<uint32_t> sizes(numSlots, slotCapacity);
		if (numSlots > 0) {
			const uint64_t remainder = m_meta.numPoints % slotCapacity;
			if (remainder != 0) {
				sizes.back() = static_cast<uint32_t>(remainder);
			}
		}

		const size_t sizesBytes = sizes.size() * sizeof(uint32_t);
		if (REMO_CU(cuMemAlloc(&m_batchSizes, sizesBytes ? sizesBytes : 4)) !=
		    CUDA_SUCCESS) {
			if (err) *err = "cuMemAlloc failed for batchSizes";
			return false;
		}
		if (sizesBytes &&
		    REMO_CU(cuMemcpyHtoD(m_batchSizes, sizes.data(), sizesBytes)) !=
		        CUDA_SUCCESS) {
			if (err) *err = "uploading batchSizes failed";
			return false;
		}

		if (REMO_CU(cuMemAlloc(&m_numBatchesUploaded, 4)) != CUDA_SUCCESS) {
			if (err) *err = "cuMemAlloc failed for numBatchesUploaded";
			return false;
		}
		if (REMO_CU(cuMemsetD32(m_numBatchesUploaded, numSlots, 1)) != CUDA_SUCCESS) {
			if (err) *err = "could not publish numBatchesUploaded";
			return false;
		}
		return true;
	}

	CudaContext& m_cuda;
	CloudMeta m_meta;
	std::vector<Point> m_points;
	CUdeviceptr m_devicePoints = 0;
	CUdeviceptr m_batchSizes = 0;
	CUdeviceptr m_numBatchesUploaded = 0;
	uint32_t m_numSlots = 0;
	uint32_t m_slotCapacity = 1'000'000;
	uint64_t m_uploaded = 0;
};

void deriveTranslation(CloudMeta& meta) {
	for (int i = 0; i < 3; ++i) {
		meta.translation[i] = -meta.boxMinOrig[i];
		meta.boxSize[i] = static_cast<float>(meta.boxMaxOrig[i] +
		                                    meta.translation[i]);
	}
}

void applyTranslation(CloudMeta& meta, std::vector<Point>& points) {
	deriveTranslation(meta);
	const float tx = static_cast<float>(meta.translation[0]);
	const float ty = static_cast<float>(meta.translation[1]);
	const float tz = static_cast<float>(meta.translation[2]);
	if (tx == 0.0f && ty == 0.0f && tz == 0.0f) return;
	for (Point& p : points) {
		p.x += tx;
		p.y += ty;
		p.z += tz;
	}
}

bool loadLasCloud(const std::string& path, CloudMeta& meta,
                  std::vector<Point>& points, std::string* err) {
	LasHeaderInfo info;
	if (!readLasHeader(path, info, err)) return false;

	meta.numPoints = info.numPoints;
	for (int i = 0; i < 3; ++i) {
		meta.boxMinOrig[i] = info.min[i];
		meta.boxMaxOrig[i] = info.max[i];
	}
	meta.files = {path};
	meta.hasCompressed = info.compressed;
	deriveTranslation(meta);

	points.resize(info.numPoints);

	double bounds[6] = {};
	for (int attempt = 0; attempt < 2; ++attempt) {
		const bool ok =
			info.compressed
				? readLazPoints(path, info, meta.translation, points.data(), bounds, err)
				: readLasPoints(path, info, meta.translation, points.data(), bounds, err);
		if (!ok) return false;

		bool underflow = false;
		for (int i = 0; i < 3; ++i) {
			const double slack = info.scale[i] +
			                     static_cast<double>(meta.boxSize[i]) * 0x1p-23;
			if (bounds[i] < -slack) underflow = true;
		}
		if (!underflow) break;

		if (attempt == 1) {
			if (err) {
				*err = "LAS bounding box does not converge; is the file being written "
				       "underneath us? " + path;
			}
			return false;
		}

		printf("remobench: WARNING -- %s declares a minimum above its own points; "
		       "re-reading against the observed box. Structural counts will not match "
		       "a reference that trusted the header.\n",
		       path.c_str());
		for (int i = 0; i < 3; ++i) {
			meta.boxMinOrig[i] =
				std::min(meta.boxMinOrig[i], bounds[i] - meta.translation[i]);
			meta.boxMaxOrig[i] =
				std::max(meta.boxMaxOrig[i], bounds[3 + i] - meta.translation[i]);
		}
		deriveTranslation(meta);
	}

	bool grewMaterially = false;
	for (int i = 0; i < 3; ++i) {
		if (bounds[3 + i] > meta.boxSize[i]) {
			const double slack =
				info.scale[i] + static_cast<double>(meta.boxSize[i]) * 0x1p-23;
			if (bounds[3 + i] - meta.boxSize[i] > slack) grewMaterially = true;
			meta.boxSize[i] = static_cast<float>(bounds[3 + i]);
			meta.boxMaxOrig[i] = bounds[3 + i] - meta.translation[i];
		}
	}
	if (grewMaterially) {
		printf("remobench: WARNING -- %s declares a bounding box smaller than its "
		       "points; grown to [%.3f %.3f %.3f]. Structural counts will not match "
		       "a reference that trusted the header.\n",
		       path.c_str(), static_cast<double>(meta.boxSize[0]),
		       static_cast<double>(meta.boxSize[1]),
		       static_cast<double>(meta.boxSize[2]));
	}

	return true;
}

}

std::unique_ptr<PointSource> makeSyntheticSource(CudaContext& cuda,
                                                 uint64_t numPoints,
                                                 uint32_t seed) {
	std::mt19937 rng(seed);
	std::uniform_real_distribution<float> unit(0.0f, 1.0f);

	std::vector<Point> points;
	points.reserve(numPoints);

	const uint64_t helixCount = numPoints / 8;
	const uint64_t shellCount = numPoints - helixCount;

	constexpr float kShellThickness = 0.02f;

	for (uint64_t i = 0; i < shellCount; ++i) {
		float xyz[3] = {unit(rng), unit(rng), unit(rng)};
		const int face = static_cast<int>(unit(rng) * 6.0f) % 6;
		const float depth = unit(rng) * kShellThickness;
		xyz[face / 2] = (face % 2) ? 1.0f - depth : depth;

		const uint32_t r = static_cast<uint32_t>(xyz[0] * 255.0f);
		const uint32_t g = static_cast<uint32_t>(xyz[1] * 255.0f);
		const uint32_t b = static_cast<uint32_t>(xyz[2] * 255.0f);
		points.push_back({xyz[0] * 100.0f, xyz[1] * 100.0f, xyz[2] * 100.0f,
		                  r | (g << 8) | (b << 16) | (255u << 24)});
	}

	std::uniform_real_distribution<float> jitter(-0.6f, 0.6f);
	for (uint64_t i = 0; i < helixCount; ++i) {
		const float t = static_cast<float>(i) / static_cast<float>(helixCount ? helixCount : 1);
		const float angle = t * 6.2831853f * 6.0f;
		const float radius = 35.0f + jitter(rng);
		points.push_back({50.0f + std::cos(angle) * radius + jitter(rng),
		                  50.0f + std::sin(angle) * radius + jitter(rng),
		                  t * 100.0f + jitter(rng), 0xFF20E0FFu});
	}

	CloudMeta meta;
	meta.numPoints = points.size();
	for (int i = 0; i < 3; ++i) {
		meta.boxMinOrig[i] = 0.0;
		meta.boxMaxOrig[i] = 100.0;
	}
	meta.files = {"<synthetic>"};
	meta.isSyntheticFixture = true;
	applyTranslation(meta, points);

	return std::make_unique<ResidentSource>(cuda, std::move(meta),
	                                        std::move(points));
}

std::unique_ptr<PointSource> openPointSource(
	CudaContext& cuda, const std::vector<std::string>& files, std::string* err) {

	if (files.empty()) {
		if (err) *err = "no input files";
		return nullptr;
	}
	if (files.size() > 1) {
		if (err) *err = "multi-file input is not implemented yet";
		return nullptr;
	}

	const std::string& path = files.front();
	if (!fs::exists(path)) {
		if (err) *err = "no such file: " + path;
		return nullptr;
	}

	std::string ext = fs::path(path).extension().string();
	std::transform(ext.begin(), ext.end(), ext.begin(),
	               [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

	CloudMeta meta;
	std::vector<Point> points;

	if (ext == ".simlod") {
		if (!readSimlod(path, meta, points, err)) return nullptr;
	} else if (ext == ".las" || ext == ".laz") {
		if (!loadLasCloud(path, meta, points, err)) return nullptr;
		return std::make_unique<ResidentSource>(cuda, std::move(meta),
		                                        std::move(points));
	} else {
		if (err) *err = "unrecognised extension: " + ext;
		return nullptr;
	}

	applyTranslation(meta, points);
	return std::make_unique<ResidentSource>(cuda, std::move(meta),
	                                        std::move(points));
}

}
