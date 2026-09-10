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
	// float32 has a 24-bit significand, so the spacing between representable values
	// at magnitude m is about m * 2^-23. The far corner of the (translated) box is
	// the worst case.
	const double extent = std::max({std::fabs(static_cast<double>(boxSize[0])),
	                                std::fabs(static_cast<double>(boxSize[1])),
	                                std::fabs(static_cast<double>(boxSize[2]))});
	return extent * std::pow(2.0, -23.0);
}

namespace {


// ---------------------------------------------------------------------------
// A source whose points all live in device memory at once.
//
// This is the Mode::Whole half of the design in PointSource.h. The streaming ring
// arrives with the progressive pipeline that needs it; until then, keeping this one
// path means the window, camera, GL interop and rasteriser can be brought up and
// validated without also debugging a multi-threaded loader.
//
// Note it does NOT repeat CudaLOD's mistake of building a host vector<Point> with
// no reserve() while the raw file blob is still resident -- callers hand over an
// already-sized vector and it is released as soon as the upload completes.
// ---------------------------------------------------------------------------
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
		if (m_devicePoints) return true;  // idempotent

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

		// Publish the batch view a progressive consumer needs.
		//
		// This is the "whole cloud is a ring with non-wrapping slot addresses" idea from
		// PointSource.h, made concrete. SimLOD's kernel_construct does not care that the
		// points are already resident -- it walks batches, reading batchSizes[slot] and
		// numBatchesUploaded. So all that is needed is to describe the resident array as a
		// sequence of full slots and declare every one of them already uploaded.
		//
		// Its slot addressing is `(batchIndex % BATCH_STREAM_SIZE) * MAX_BATCH_SIZE`, which
		// only matches a non-wrapping array while the index stays below BATCH_STREAM_SIZE
		// -- hence the raised value in kernels/simlod/structures.cuh.
		const uint32_t slotCapacity = 1'000'000;
		const uint32_t numSlots = static_cast<uint32_t>(
			(m_meta.numPoints + slotCapacity - 1) / slotCapacity);

		if (!allocBatchMetadata(numSlots, slotCapacity, err)) {
			stop();
			return false;
		}

		// Release the host copy: for the 350M cloud this is 5.6GB that nothing needs
		// once the upload is done.
		m_points.clear();
		m_points.shrink_to_fit();
		return true;
	}

	void rewind() override {
		// Nothing to do: all points are already resident and immutable. A streaming
		// source resets its cursor here.
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
		v.wrapping = false;  // the defining difference from Mode::Stream
		return v;
	}

	uint64_t numPointsUploaded() const override { return m_uploaded; }
	bool isFullyResident() const override {
		return m_devicePoints != 0 && m_uploaded == m_meta.numPoints;
	}
	CUdeviceptr residentPoints() const override { return m_devicePoints; }
	void setPointsConsumed(uint64_t) override {}

private:
	// batchSizes[] and numBatchesUploaded, as a progressive consumer expects them: every
	// slot full except the last, and everything already uploaded.
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
		// All of it, immediately: nothing is streaming here. A progressive pipeline still
		// paces itself, because it bounds how many batches it consumes per launch and has
		// its own device-side time budget.
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

// Derive the translation and the post-translation size from original-CRS bounds.
//
// The translation is exactly -boxMin, which is what makes "boxMin is the origin by
// construction" (HostDeviceCommon.h) true rather than aspirational.
//
// It used to snap down to a power of two, on the theory that a power of two is exactly
// representable in f32 and therefore stable across runs. That reasoning only ever held
// up on already-translated input. A UTM cloud has min_x = 693414.98, whose power of two
// below is 524288 -- so a 1.3 km cloud kept a 169 km offset, the octree root cube came
// out 170424 units wide, and every point landed in one corner of it: 36M points, 29
// nodes, one leaf holding 7.04M of them. Only `.simlod` hid this, because SimLOD's own
// converter has already subtracted the minimum, so min is exactly 0 and both rules
// agree on a translation of 0.
//
// What is given up is that the translation is no longer guaranteed exact in f32. That
// costs nothing where it is applied: readers holding f64 coordinates (LAS/LAZ) apply it
// in f64 before narrowing, and a reader whose coordinates are already f32 has an f32
// minimum, so -min is exact there anyway.
//
// Split from applyTranslation because an f64 reader needs the translation BEFORE it
// produces points -- folding it into the parse is what keeps the low bits that a
// post-pass over f32 points has already discarded.
void deriveTranslation(CloudMeta& meta) {
	for (int i = 0; i < 3; ++i) {
		meta.translation[i] = -meta.boxMinOrig[i];
		meta.boxSize[i] = static_cast<float>(meta.boxMaxOrig[i] +
		                                    meta.translation[i]);
	}
}

// Finish a CloudMeta for a reader whose points are already float32 in original
// coordinates (.simlod, synthetic): derive the translation, then shift in f32.
void applyTranslation(CloudMeta& meta, std::vector<Point>& points) {
	deriveTranslation(meta);
	// Points arrive in original coordinates; shift them once, here, so device code
	// only ever sees origin-relative float32.
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

// Load a LAS/LAZ cloud whole. Returns points already translated -- the caller must
// NOT call applyTranslation on them.
//
// The order here is the interesting part. The bounding box comes from the header
// (float64) and the translation is derived from it BEFORE a single point is parsed,
// so the shift can be folded into the parse and applied at full precision. That is
// also why the box is not recomputed from the points afterwards, the way the .simlod
// reader has to: for LAS the header box IS the input. CudaLOD and SimLOD both size
// the octree root cube from it, so recomputing it would silently produce a different
// tree from the one the reference produced on the same file -- which is precisely the
// provenance question in bench/reference/README.md that this reader exists to settle.
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

	// A header bbox that does not contain its own points is a real and common defect,
	// and it is not cosmetic here: the octree root cube IS the box, so a point below it
	// is out of the grid. Neither pipeline faults on that -- CudaLOD clamps the cell
	// index and SimLOD's float-to-uint32 conversion saturates, both landing on cell 0 --
	// which is worse than a fault, because the result is a silent pile of points in one
	// corner cell. That is also the shape that walks into the unchecked capacities
	// makeSyntheticSource documents below, so it can turn into a fault somewhere with no
	// visible connection to the input.
	//
	// The low side cannot be fixed after the fact -- the translation was derived from
	// it, and every point has already been shifted by it -- so the recovery is to take
	// the minimum we actually observed and parse again. Twice at most: the second parse
	// is measured against a box that provably contains the points.
	double bounds[6] = {};
	for (int attempt = 0; attempt < 2; ++attempt) {
		const bool ok =
			info.compressed
				? readLazPoints(path, info, meta.translation, points.data(), bounds, err)
				: readLasPoints(path, info, meta.translation, points.data(), bounds, err);
		if (!ok) return false;

		// Slack for the low side: LAS stores coordinates as integers times `scale`, and
		// writers exist that compute the header bbox from the unquantised values, so a
		// declared minimum can sit a fraction of a quantisation step above the smallest
		// point actually stored. A negative that small is harmless -- it truncates to
		// grid index 0 -- and re-parsing every such file would be a lot of I/O to move a
		// point by less than the file's own precision.
		bool underflow = false;
		for (int i = 0; i < 3; ++i) {
			const double slack = info.scale[i] +
			                     static_cast<double>(meta.boxSize[i]) * 0x1p-23;
			if (bounds[i] < -slack) underflow = true;
		}
		if (!underflow) break;

		if (attempt == 1) {
			// Cannot happen from a static file: the box was just derived from these very
			// points. Refuse rather than loop.
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

	// The high side needs no re-parse: growing the box leaves the translation, and so
	// every point, exactly where it is.
	//
	// Only worth a warning when the excess is bigger than the low side's slack. A
	// header maximum a fraction of a quantisation step short of its own points is the
	// same unquantised-bbox sloppiness, it is silently correct to absorb, and warning
	// about it on every load of every such file would train the reader to ignore the
	// message that matters.
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
		// Loud, because it means the tree built from this file is NOT the tree another
		// reader of the same file builds.
		printf("remobench: WARNING -- %s declares a bounding box smaller than its "
		       "points; grown to [%.3f %.3f %.3f]. Structural counts will not match "
		       "a reference that trusted the header.\n",
		       path.c_str(), static_cast<double>(meta.boxSize[0]),
		       static_cast<double>(meta.boxSize[1]),
		       static_cast<double>(meta.boxSize[2]));
	}

	return true;
}

}  // namespace

// ---------------------------------------------------------------------------

std::unique_ptr<PointSource> makeSyntheticSource(CudaContext& cuda,
                                                 uint64_t numPoints,
                                                 uint32_t seed) {
	// A cube shell of finite thickness, plus a diagonal helix. Chosen so orientation,
	// depth ordering and the Z-up flip are all immediately visible by eye -- a uniform
	// random cube looks identical upside down and would hide a wrong flip matrix.
	//
	// NOTE THE JITTER, AND DO NOT REMOVE IT.
	//
	// The first version of this pinned a face coordinate to exactly 0.0 or 1.0, giving
	// perfectly coplanar points with exactly duplicated coordinates. That is a
	// degenerate distribution, and it made CudaLOD's build kernels fault with
	// CUDA_ERROR_ILLEGAL_ADDRESS on every run, while real scans loaded fine.
	//
	// The reason is upstream's, not ours: split_countsort_blockwise calls split_node
	// exactly ONCE, at depth 8 plus depth-4 subgrids. Its own comment concedes that
	// leaves can then exceed MAX_POINTS_PER_NODE and that split_node "must be called
	// again" in that case -- which it never is. Several of its capacities
	// (LARGE_CELLS_CAPACITY at 100'000, MAX_NODES at 200'000) are likewise unchecked on
	// the device. A dense plane concentrates points into few cells and walks straight
	// into that.
	//
	// We deliberately do NOT patch upstream's algorithm -- reproducing the published
	// behaviour is the point, and the stats panel surfaces maxPointsPerNode so an
	// oversized leaf is visible. But a *test fixture* has no business being degenerate:
	// real scanners do not produce exactly coplanar duplicate points, so neither should
	// this.
	std::mt19937 rng(seed);
	std::uniform_real_distribution<float> unit(0.0f, 1.0f);

	std::vector<Point> points;
	points.reserve(numPoints);

	const uint64_t helixCount = numPoints / 8;
	const uint64_t shellCount = numPoints - helixCount;

	// Shell thickness as a fraction of the cube, comfortably more than one cell of a
	// 256^3 grid (1/256 ~= 0.004) so a face spreads across several cells in depth.
	constexpr float kShellThickness = 0.02f;

	for (uint64_t i = 0; i < shellCount; ++i) {
		float xyz[3] = {unit(rng), unit(rng), unit(rng)};
		// Pull one axis towards a face, but keep it a slab rather than a plane.
		const int face = static_cast<int>(unit(rng) * 6.0f) % 6;
		const float depth = unit(rng) * kShellThickness;
		xyz[face / 2] = (face % 2) ? 1.0f - depth : depth;

		const uint32_t r = static_cast<uint32_t>(xyz[0] * 255.0f);
		const uint32_t g = static_cast<uint32_t>(xyz[1] * 255.0f);
		const uint32_t b = static_cast<uint32_t>(xyz[2] * 255.0f);
		points.push_back({xyz[0] * 100.0f, xyz[1] * 100.0f, xyz[2] * 100.0f,
		                  r | (g << 8) | (b << 16) | (255u << 24)});
	}

	// The helix is a curve, so give it a tube's worth of thickness for the same reason.
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
		// Multi-file loading needs a global bbox pass before any point is
		// translated, since the translation must be shared. Deferred until the
		// streaming loader lands.
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
		// Returns already-translated points: the shift is folded into the parse so it
		// happens in f64. Do not add applyTranslation to this branch.
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

}  // namespace remo
