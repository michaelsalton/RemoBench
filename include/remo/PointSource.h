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

	double boxMinOrig[3] = {0, 0, 0};
	double boxMaxOrig[3] = {0, 0, 0};

	double translation[3] = {0, 0, 0};

	float boxSize[3] = {0, 0, 0};

	std::vector<std::string> files;
	bool hasCompressed = false;

	bool isSyntheticFixture = false;

	double worstQuantisationError() const;
};

// Points per batch slot, on the host and on the device alike.
//
// This is kernel_construct's MAX_BATCH_SIZE, which is a function-local constexpr there
// (progressive_octree_voxels.cu) and so cannot be asserted against from any other
// translation unit. It is the ring's stride whatever the actual batch sizes are: slot s
// begins at base + s * kSlotCapacity points, and batchSizes[s] says how many of them
// are real.
inline constexpr uint32_t kSlotCapacity = 1'000'000;

struct BatchView {
	CUdeviceptr slots = 0;
	CUdeviceptr batchSizes = 0;
	CUdeviceptr numBatchesUploaded = 0;
	uint32_t slotCapacity = kSlotCapacity;

	// Slots actually allocated on the device.
	uint32_t numSlots = 0;

	// Batches in the whole cloud. Equal to numSlots exactly when nothing wraps.
	uint32_t numBatchesTotal = 0;

	// True when numBatchesTotal > numSlots, so batch B lands in slot B % numSlots and
	// the slot is reused.
	//
	// The wiki used to describe this as a choice -- Mode::Stream wrapping, Mode::Whole
	// not -- but kernel_construct computes `batchIndex % BATCH_STREAM_SIZE` itself and
	// has no non-wrapping mode to select. So a consumer cannot turn wrapping off; it
	// can only check that the ring it has been handed is the one the kernel assumes.
	// That is the check worth making, because getting it wrong is an out-of-bounds
	// device read that produces a plausible-looking tree from the wrong points.
	// A non-wrapping feed is correct only while numBatchesTotal <= BATCH_STREAM_SIZE,
	// which is where SimLOD's 50M ceiling comes from.
	bool wrapping = false;
};

class PointSource {
public:
	// Whole: the entire cloud sits in device memory and batch B is at base + B * slot.
	//        Required by any consumer that keeps a pointer into the cloud across
	//        frames (flat) or reads it all at once (cudalod).
	// Stream: a fixed-depth ring the host refills behind the consumer. The input cost
	//        stops scaling with the cloud, which is what lets a cloud larger than the
	//        card's memory be built at all.
	enum class Mode { Stream, Whole };

	virtual ~PointSource() = default;

	virtual const CloudMeta& meta() const = 0;

	// ringSlots is the consumer's ring depth, from PipelineInfo, and is ignored in
	// Mode::Whole. Re-entrant: calling it with a different mode restarts ingest in that
	// mode, which is what a pipeline switch does.
	virtual bool start(Mode mode, uint32_t ringSlots, size_t maxDeviceBytes,
	                   std::string* err) = 0;

	// Restart ingest from batch 0 in the current mode. Any consumer that zeroes its own
	// batch counter (a reset, a rebuild) must call this, or the producer keeps filling
	// slots ahead of a consumer that has gone back to the beginning.
	virtual void rewind() = 0;
	virtual void stop() = 0;

	// Fill whatever the consumer's window allows. No-op in Mode::Whole. Called once per
	// frame by the shell, before the consumer builds.
	virtual void pump() = 0;

	virtual BatchView view() const = 0;
	virtual uint64_t numPointsUploaded() const = 0;

	virtual bool isFullyResident() const = 0;
	virtual CUdeviceptr residentPoints() const = 0;

	// The consumption signal, in batches -- `stats->batchletIndex` for the progressive
	// pipelines. Batches, not points, because the backpressure invariant is stated in
	// batches (producer index minus this never exceeds the ring depth) and converting
	// through a point count would round.
	virtual void setBatchesConsumed(uint32_t numBatches) = 0;

	// Set once if the producer ever ran ahead of the consumer window. There is no
	// device-side symptom for that -- the tree builds from overwritten points with no
	// fault and plausible counts -- so it is checked here and reported loudly.
	virtual bool overranConsumer() const = 0;
};

std::unique_ptr<PointSource> makeSyntheticSource(CudaContext& cuda,
                                                 uint64_t numPoints,
                                                 uint32_t seed = 1);

std::unique_ptr<PointSource> openPointSource(CudaContext& cuda,
                                             const std::vector<std::string>& files,
                                             std::string* err);

}
