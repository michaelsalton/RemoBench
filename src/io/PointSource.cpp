#include "remo/PointSource.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <random>

#include "remo/CudaCheck.h"
#include "remo/CudaContext.h"
#include "io/BatchProducer.h"
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

uint32_t batchCount(uint64_t numPoints) {
	if (numPoints == 0) return 0;
	return static_cast<uint32_t>((numPoints + kSlotCapacity - 1) / kSlotCapacity);
}

// A producer that reads batch k from the file when the ring asks for it, holding nothing
// but a path and a translation. This is what keeps a 350M cloud off the host as well as
// off the device: the seekable formats never materialise more than one slot at a time.
class SimlodFileProducer final : public BatchProducer {
public:
	SimlodFileProducer(std::string path, SimlodInfo info, const double translation[3])
		: m_path(std::move(path)), m_info(info) {
		for (int i = 0; i < 3; ++i) m_translation[i] = translation[i];
	}

	bool fill(uint64_t first, uint32_t count, Point* out, std::string* err) override {
		return readSimlodRange(m_path, m_info, m_translation, first, count, out, err);
	}

private:
	std::string m_path;
	SimlodInfo m_info;
	double m_translation[3] = {0, 0, 0};
};

class LasFileProducer final : public BatchProducer {
public:
	LasFileProducer(std::string path, LasHeaderInfo info, const double translation[3])
		: m_path(std::move(path)), m_info(std::move(info)) {
		for (int i = 0; i < 3; ++i) m_translation[i] = translation[i];
	}

	bool fill(uint64_t first, uint32_t count, Point* out, std::string* err) override {
		return readLasRange(m_path, m_info, m_translation, first, count, out, err);
	}

private:
	std::string m_path;
	LasHeaderInfo m_info;
	double m_translation[3] = {0, 0, 0};
};

class MemoryProducer final : public BatchProducer {
public:
	explicit MemoryProducer(std::vector<Point> points) : m_points(std::move(points)) {}

	const Point* peek(uint64_t first, uint32_t count) override {
		if (first + count > m_points.size()) return nullptr;
		return m_points.data() + first;
	}

	bool fill(uint64_t first, uint32_t count, Point* out, std::string* err) override {
		if (first + count > m_points.size()) {
			if (err) *err = "batch range lies past the end of the cloud";
			return false;
		}
		std::copy_n(m_points.data() + first, count, out);
		return true;
	}

private:
	std::vector<Point> m_points;
};

// One source, two shapes.
//
// `Mode::Whole` uploads the cloud once and hands out a pointer to it; `Mode::Stream`
// keeps a fixed-depth ring and refills it behind the consumer. They differ only in how
// much device memory the input costs and in who writes the slots -- the per-slot
// `batchSizes[]` and the `numBatchesUploaded` handshake are identical, which is why the
// device code needs no change at all to go from one to the other.
class CloudSource final : public PointSource {
public:
	CloudSource(CudaContext& cuda, CloudMeta meta,
	            std::unique_ptr<BatchProducer> producer)
		: m_cuda(cuda), m_meta(std::move(meta)), m_producer(std::move(producer)) {}

	~CloudSource() override { stop(); }

	const CloudMeta& meta() const override { return m_meta; }

	bool start(Mode mode, uint32_t ringSlots, size_t maxDeviceBytes,
	           std::string* err) override {
		if (m_meta.numPoints == 0) {
			if (err) *err = "point cloud is empty";
			return false;
		}

		const uint32_t total = batchCount(m_meta.numPoints);

		// A ring deeper than the cloud is pointless, and a ring SHALLOWER than the
		// consumer's BATCH_STREAM_SIZE is only safe while nothing wraps -- which is
		// exactly the case min() leaves: either slots == ringSlots and the kernel's
		// modulo lands inside the buffer, or total <= ringSlots and batch B maps to
		// slot B with no wrapping at all.
		uint32_t slots = (mode == Mode::Whole) ? total : std::min(ringSlots, total);
		if (mode == Mode::Stream && ringSlots == 0) {
			if (err) *err = "streaming ingest needs a ring depth; none was given";
			return false;
		}

		if (m_slots && m_mode == mode && m_numSlots == slots) {
			rewind();
			return true;
		}
		stop();

		const uint64_t bytes =
			static_cast<uint64_t>(slots) * kSlotCapacity * sizeof(Point);
		if (maxDeviceBytes != 0 && bytes > maxDeviceBytes) {
			if (err) {
				*err = "point data needs " + std::to_string(bytes / (1024 * 1024)) +
				       " MB but the budget allows " +
				       std::to_string(maxDeviceBytes / (1024 * 1024)) + " MB";
			}
			return false;
		}

		if (REMO_CU(cuMemAlloc(&m_slots, bytes)) != CUDA_SUCCESS) {
			m_slots = 0;
			if (err) {
				*err = "cuMemAlloc failed for a " +
				       std::to_string(bytes / (1024 * 1024)) + " MB point buffer";
			}
			return false;
		}
		if (REMO_CU(cuMemAlloc(&m_batchSizes, uint64_t(slots) * 4)) != CUDA_SUCCESS) {
			if (err) *err = "cuMemAlloc failed for batchSizes";
			stop();
			return false;
		}
		if (REMO_CU(cuMemAlloc(&m_numBatchesUploaded, 4)) != CUDA_SUCCESS) {
			if (err) *err = "cuMemAlloc failed for numBatchesUploaded";
			stop();
			return false;
		}

		m_mode = mode;
		m_numSlots = slots;
		m_numBatchesTotal = total;
		m_produced = 0;
		m_consumed = 0;
		m_uploadedPoints = 0;
		m_overran = false;

		if (!publishUploaded(err)) {
			stop();
			return false;
		}
		if (REMO_CU(cuMemsetD8(m_batchSizes, 0, uint64_t(slots) * 4)) != CUDA_SUCCESS) {
			if (err) *err = "could not clear batchSizes";
			stop();
			return false;
		}

		if (mode == Mode::Whole) {
			// Nothing wraps here, so the window is the whole cloud and one pump fills
			// it. Consumers that keep a pointer across frames depend on that.
			pump();
			if (m_produced != m_numBatchesTotal) {
				if (err) *err = m_uploadError.empty() ? "uploading points failed"
				                                      : m_uploadError;
				stop();
				return false;
			}
		}
		return true;
	}

	void rewind() override {
		if (!m_slots) return;
		m_produced = 0;
		m_consumed = 0;
		m_uploadedPoints = 0;
		m_overran = false;
		m_uploadError.clear();
		publishUploaded(nullptr);
		REMO_CU(cuMemsetD8(m_batchSizes, 0, uint64_t(m_numSlots) * 4));
		if (m_mode == Mode::Whole) pump();
	}

	void stop() override {
		for (CUdeviceptr* p : {&m_slots, &m_batchSizes, &m_numBatchesUploaded}) {
			if (*p) {
				REMO_CU(cuMemFree(*p));
				*p = 0;
			}
		}
		m_numSlots = 0;
		m_numBatchesTotal = 0;
		m_produced = 0;
		m_consumed = 0;
		m_uploadedPoints = 0;
	}

	void pump() override {
		if (!m_slots || !m_producer) return;

		// The window. Slot B % numSlots still holds batch B - numSlots, so writing it
		// before the consumer has finished that batch overwrites points the tree has
		// not read. Whole mode has numSlots == numBatchesTotal, so the window is the
		// whole cloud and this loop runs once.
		const uint64_t limit = std::min<uint64_t>(
			m_numBatchesTotal, static_cast<uint64_t>(m_consumed) + m_numSlots);

		while (m_produced < limit) {
			if (!uploadBatch(m_produced)) return;
			++m_produced;
		}

		publishUploaded(nullptr);
	}

	BatchView view() const override {
		BatchView v;
		v.slots = m_slots;
		v.batchSizes = m_batchSizes;
		v.numBatchesUploaded = m_numBatchesUploaded;
		v.slotCapacity = kSlotCapacity;
		v.numSlots = m_numSlots;
		v.numBatchesTotal = m_numBatchesTotal;
		v.wrapping = m_numBatchesTotal > m_numSlots;
		return v;
	}

	uint64_t numPointsUploaded() const override { return m_uploadedPoints; }

	bool isFullyResident() const override {
		return m_mode == Mode::Whole && m_slots != 0 &&
		       m_uploadedPoints == m_meta.numPoints;
	}

	CUdeviceptr residentPoints() const override {
		return m_mode == Mode::Whole ? m_slots : 0;
	}

	void setBatchesConsumed(uint32_t numBatches) override {
		if (numBatches > m_numBatchesTotal) numBatches = m_numBatchesTotal;
		// The consumer only ever moves forward. Going backwards without a rewind()
		// would widen the window rather than narrow it, which is the wrong direction
		// to be wrong in.
		if (numBatches > m_consumed) m_consumed = numBatches;
	}

	bool overranConsumer() const override { return m_overran; }

private:
	bool publishUploaded(std::string* err) {
		if (!m_numBatchesUploaded) return true;
		if (REMO_CU(cuMemsetD32(m_numBatchesUploaded,
		                        static_cast<unsigned>(m_produced), 1)) != CUDA_SUCCESS) {
			if (err) *err = "could not publish numBatchesUploaded";
			return false;
		}
		return true;
	}

	bool uploadBatch(uint64_t batchIndex) {
		// The invariant, checked rather than trusted. Its failure has no device-side
		// symptom: the tree builds from overwritten points, with no fault, no
		// allocation error and plausible node counts.
		if (batchIndex >= static_cast<uint64_t>(m_consumed) + m_numSlots) {
			if (!m_overran) {
				m_overran = true;
				fprintf(stderr,
				        "remobench: ring overrun -- batch %llu would overwrite a slot "
				        "the consumer has not read (consumed %u, depth %u). Refusing; "
				        "the tree would have been built from the wrong points.\n",
				        static_cast<unsigned long long>(batchIndex), m_consumed,
				        m_numSlots);
			}
			return false;
		}

		const uint64_t first = batchIndex * kSlotCapacity;
		const uint32_t count = static_cast<uint32_t>(
			std::min<uint64_t>(kSlotCapacity, m_meta.numPoints - first));
		const uint32_t slot = static_cast<uint32_t>(batchIndex % m_numSlots);

		const CUdeviceptr dst =
			m_slots + static_cast<uint64_t>(slot) * kSlotCapacity * sizeof(Point);

		std::string err;
		if (const Point* direct = m_producer->peek(first, count)) {
			if (REMO_CU(cuMemcpyHtoD(dst, direct, uint64_t(count) * sizeof(Point))) !=
			    CUDA_SUCCESS) {
				m_uploadError = "uploading a batch failed";
				return false;
			}
		} else {
			if (m_staging.size() < kSlotCapacity) m_staging.resize(kSlotCapacity);
			if (!m_producer->fill(first, count, m_staging.data(), &err)) {
				m_uploadError = err;
				fprintf(stderr, "remobench: %s\n", err.c_str());
				return false;
			}
			if (REMO_CU(cuMemcpyHtoD(dst, m_staging.data(),
			                         uint64_t(count) * sizeof(Point))) != CUDA_SUCCESS) {
				m_uploadError = "uploading a batch failed";
				return false;
			}
		}

		if (REMO_CU(cuMemsetD32(m_batchSizes + uint64_t(slot) * 4, count, 1)) !=
		    CUDA_SUCCESS) {
			m_uploadError = "could not publish a batch size";
			return false;
		}

		m_uploadedPoints += count;
		return true;
	}

	CudaContext& m_cuda;
	CloudMeta m_meta;
	std::unique_ptr<BatchProducer> m_producer;
	std::vector<Point> m_staging;

	Mode m_mode = Mode::Whole;
	CUdeviceptr m_slots = 0;
	CUdeviceptr m_batchSizes = 0;
	CUdeviceptr m_numBatchesUploaded = 0;

	uint32_t m_numSlots = 0;
	uint32_t m_numBatchesTotal = 0;
	uint64_t m_produced = 0;
	uint32_t m_consumed = 0;
	uint64_t m_uploadedPoints = 0;
	bool m_overran = false;
	std::string m_uploadError;
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

// Apply the observed bounds to a header-derived box, exactly as the old two-attempt read
// loop did -- the header's minimum is lowered only when points sit materially below it,
// and the maximum is grown whenever any point exceeds it.
//
// Both fixups used to require having read every point, and the first one required
// reading them twice, because the translation is baked into Point at read time. Against
// a coordinate pass they are just arithmetic, and the box is final before anything is
// uploaded. That matters more than the saved read: boxSize sizes the octree root cube,
// so a box that grows mid-stream invalidates every node already built, and nothing on
// the device will complain -- CudaLOD clamps the cell index and SimLOD's float->uint32
// conversion saturates, so out-of-box points pile into cell 0 instead of faulting.
//
enum class BoundsFix { Settled, Rescan, Failed };

BoundsFix applyObservedBounds(const std::string& path, CloudMeta& meta,
                              const LasHeaderInfo& info, const double bounds[6],
                              bool lastAttempt, std::string* err) {
	bool underflow = false;
	for (int i = 0; i < 3; ++i) {
		const double slack =
			info.scale[i] + static_cast<double>(meta.boxSize[i]) * 0x1p-23;
		if (bounds[i] < -slack) underflow = true;
	}

	if (underflow) {
		if (lastAttempt) {
			if (err) {
				*err = "LAS bounding box does not converge; is the file being written "
				       "underneath us? " + path;
			}
			return BoundsFix::Failed;
		}
		printf("remobench: WARNING -- %s declares a minimum above its own points; "
		       "re-scanning against the observed box. Structural counts will not match "
		       "a reference that trusted the header.\n",
		       path.c_str());
		for (int i = 0; i < 3; ++i) {
			meta.boxMinOrig[i] =
				std::min(meta.boxMinOrig[i], bounds[i] - meta.translation[i]);
			meta.boxMaxOrig[i] =
				std::max(meta.boxMaxOrig[i], bounds[3 + i] - meta.translation[i]);
		}
		deriveTranslation(meta);
		return BoundsFix::Rescan;
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
	return BoundsFix::Settled;
}

void seedBoxFromHeader(CloudMeta& meta, const std::string& path,
                       const LasHeaderInfo& info) {
	meta.numPoints = info.numPoints;
	for (int i = 0; i < 3; ++i) {
		meta.boxMinOrig[i] = info.min[i];
		meta.boxMaxOrig[i] = info.max[i];
	}
	meta.files = {path};
	meta.hasCompressed = info.compressed;
	deriveTranslation(meta);
}

// The compressed path, unchanged in shape: laszip decodes strictly sequentially with no
// seek, so there is no cheap coordinate pass to run and no way to serve one ring slot at
// a time. A pre-pass would double a ~70 s decode for the 350M cloud. `.laz` therefore
// stays whole-cloud resident, which is a stated limitation rather than an oversight;
// lifting it depends on the parallel-decode work the README already lists.
bool loadLazCloud(const std::string& path, CloudMeta& meta,
                  std::vector<Point>& points, std::string* err) {
	LasHeaderInfo info;
	if (!readLasHeader(path, info, err)) return false;
	seedBoxFromHeader(meta, path, info);

	points.resize(info.numPoints);

	double bounds[6] = {};
	for (int attempt = 0; attempt < 2; ++attempt) {
		const bool ok =
			info.compressed
				? readLazPoints(path, info, meta.translation, points.data(), bounds, err)
				: readLasPoints(path, info, meta.translation, points.data(), bounds, err);
		if (!ok) return false;

		const BoundsFix fix =
			applyObservedBounds(path, meta, info, bounds, attempt == 1, err);
		if (fix == BoundsFix::Failed) return false;
		if (fix == BoundsFix::Settled) return true;
		// Rescan: the translation moved, and it is baked into Point at read time.
	}
	return true;
}

// The seekable path. One coordinate pass settles the box, then nothing else is read
// until the ring asks for a slot.
std::unique_ptr<BatchProducer> openLasStreaming(const std::string& path,
                                                CloudMeta& meta, std::string* err) {
	LasHeaderInfo info;
	if (!readLasHeader(path, info, err)) return nullptr;
	seedBoxFromHeader(meta, path, info);

	double bounds[6] = {};
	for (int attempt = 0; attempt < 2; ++attempt) {
		if (!readLasBounds(path, info, meta.translation, bounds, err)) return nullptr;

		const BoundsFix fix =
			applyObservedBounds(path, meta, info, bounds, attempt == 1, err);
		if (fix == BoundsFix::Failed) return nullptr;
		if (fix == BoundsFix::Settled) break;
	}

	return std::make_unique<LasFileProducer>(path, std::move(info), meta.translation);
}

std::unique_ptr<BatchProducer> openSimlodStreaming(const std::string& path,
                                                   CloudMeta& meta, std::string* err) {
	SimlodInfo info;
	if (!readSimlodHeader(path, info, err)) return nullptr;

	// The header carries a box, and the loader has never trusted it: the observed one
	// has always won. Keeping that is what preserves the three-reader invariant --
	// .simlod, .las and .laz must still give bit-identical trees from the same cloud.
	double bounds[6] = {};
	if (!readSimlodBounds(path, info, bounds, err)) return nullptr;

	meta.numPoints = info.numPoints;
	meta.files = {path};
	meta.hasCompressed = false;
	for (int i = 0; i < 3; ++i) {
		meta.boxMinOrig[i] = bounds[i];
		meta.boxMaxOrig[i] = bounds[3 + i];
	}
	deriveTranslation(meta);

	return std::make_unique<SimlodFileProducer>(path, info, meta.translation);
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

	return std::make_unique<CloudSource>(
		cuda, std::move(meta), std::make_unique<MemoryProducer>(std::move(points)));
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
	std::unique_ptr<BatchProducer> producer;

	if (ext == ".simlod") {
		producer = openSimlodStreaming(path, meta, err);
		if (!producer) return nullptr;
	} else if (ext == ".las" || ext == ".laz") {
		LasHeaderInfo probe;
		if (!readLasHeader(path, probe, err)) return nullptr;

		if (probe.compressed) {
			std::vector<Point> points;
			if (!loadLazCloud(path, meta, points, err)) return nullptr;
			producer = std::make_unique<MemoryProducer>(std::move(points));
		} else {
			producer = openLasStreaming(path, meta, err);
			if (!producer) return nullptr;
		}
	} else {
		if (err) *err = "unrecognised extension: " + ext;
		return nullptr;
	}

	return std::make_unique<CloudSource>(cuda, std::move(meta), std::move(producer));
}

}
