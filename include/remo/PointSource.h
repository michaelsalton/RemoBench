// Point cloud ingest.
//
// One loader serves both consumer shapes. The trick that makes that cheap: a
// whole-cloud consumer is the streaming ring with NON-WRAPPING slot addresses.
//
//   Mode::Stream -> deviceAddr(k) = ringBase     + (k % numSlots) * slotBytes
//   Mode::Whole  -> deviceAddr(k) = residentBase +  k             * slotBytes
//
// Everything else -- the header scan, the loader threads, the pinned pool, the
// single uploader thread, the per-slot batchSizes[] and numBatchesUploaded signals
// -- is byte-for-byte identical. So CudaLOD inherits the fast multithreaded loader
// for free, instead of its own loadLas() that reads the entire point blob into a
// host Buffer and then push_back's into a vector<Point> with no reserve(). Peak
// host RAM there is bytesPerPoint*N + 16*N simultaneously: about 18GB for the 350M
// file already sitting in data/, which would simply fail.
//
// Two experiments this shape unlocks that neither upstream can run:
//   - SimLOD in Whole mode, isolating ring/streaming overhead from construction.
//   - CudaLOD rebuilding from the first K batches each frame, which produces a
//     progressive-comparable time-to-quality curve for a batch pipeline.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <cuda.h>

#include "remo/HostDeviceCommon.h"

namespace remo {

class CudaContext;

struct CloudMeta {
	uint64_t numPoints = 0;

	// Original coordinates, in the file's own CRS. Kept in f64 -- the translation
	// below is applied host-side and an f32 origin is never stored.
	double boxMinOrig[3] = {0, 0, 0};
	double boxMaxOrig[3] = {0, 0, 0};

	// Subtracted from every point on the host, so device coordinates start at the
	// origin and fit float32. Exactly -boxMinOrig: the octree root cube is sized from
	// boxSize with its minimum ASSUMED to be the origin, so any slack left here is
	// slack the whole tree is built on. See deriveTranslation for what a coarser rule
	// did to a UTM cloud.
	//
	// A reader holding f64 coordinates must apply this in f64 and narrow afterwards,
	// not add it to points it has already rounded to f32 -- at UTM magnitudes an f32
	// coordinate has ~4 cm of resolution, and the translation is what buys the
	// sub-millimetre resolution worstQuantisationError() reports.
	double translation[3] = {0, 0, 0};

	float boxSize[3] = {0, 0, 0};  // post-translation extent

	std::vector<std::string> files;
	bool hasCompressed = false;  // any .laz -> throughput warning

	// True for makeSyntheticSource's cloud. Not cosmetic: its distribution (dense
	// slabs plus a 1D helix) makes CudaLOD's build kernels fault, so pipelines can be
	// gated on it rather than letting a GUI click kill the session. See
	// PipelineRegistry::unsupportedReason.
	bool isSyntheticFixture = false;

	// Worst-case f32 quantisation error at the far corner, in world units. Morro
	// Bay is sub-millimetre; a national-scale cloud is centimetres; ECEF
	// coordinates are ~0.5m and unusable. Surfaced in the GUI so crossing that line
	// is visible rather than being discovered as visual noise.
	double worstQuantisationError() const;
};

// What ingest guarantees to a pipeline: a device pointer to tightly packed 16-byte
// Points, pre-translated so boxMin is the origin, plus the two signals a device
// kernel needs to know how much has landed.
struct BatchView {
	CUdeviceptr slots = 0;               // base of the ring / resident array
	CUdeviceptr batchSizes = 0;          // uint32_t[numSlots]
	CUdeviceptr numBatchesUploaded = 0;  // uint32_t, written by the uploader,
	                                     // read racily by the construct kernel
	uint32_t slotCapacity = 1'000'000;   // points per slot
	uint32_t numSlots = 0;
	uint32_t numBatchesTotal = 0;
	bool wrapping = true;
};

class PointSource {
public:
	enum class Mode { Stream, Whole };

	virtual ~PointSource() = default;

	virtual const CloudMeta& meta() const = 0;

	// Begin (or restart) ingest. rewind() is required for pipeline switching, since
	// switching frees the old structure and rebuilds from scratch.
	virtual bool start(Mode mode, size_t maxDeviceBytes, std::string* err) = 0;
	virtual void rewind() = 0;
	virtual void stop() = 0;

	virtual BatchView view() const = 0;
	virtual uint64_t numPointsUploaded() const = 0;

	virtual bool isFullyResident() const = 0;
	virtual CUdeviceptr residentPoints() const = 0;  // Whole mode only, else 0

	// Consumers publish progress so loaders can apply backpressure. This is what
	// keeps the source from needing to know what a pipeline is -- upstream's loader
	// threads reach directly into the global stats struct instead.
	virtual void setPointsConsumed(uint64_t n) = 0;
};

// A deterministic synthetic cloud. Not a toy: it is the fixture that lets the
// window, camera, GL interop and rasterizer be brought up and tested before any
// file parsing exists, and it gives the octree-invariant tests a structure whose
// correct answer is known in advance.
std::unique_ptr<PointSource> makeSyntheticSource(CudaContext& cuda,
                                                 uint64_t numPoints,
                                                 uint32_t seed = 1);

// Dispatches on extension: .simlod / .las / .laz. Header scan only; no point data
// is read until start().
std::unique_ptr<PointSource> openPointSource(CudaContext& cuda,
                                             const std::vector<std::string>& files,
                                             std::string* err);

}  // namespace remo
