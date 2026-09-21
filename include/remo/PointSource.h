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

struct BatchView {
	CUdeviceptr slots = 0;
	CUdeviceptr batchSizes = 0;
	CUdeviceptr numBatchesUploaded = 0;
	uint32_t slotCapacity = 1'000'000;
	uint32_t numSlots = 0;
	uint32_t numBatchesTotal = 0;
	bool wrapping = true;
};

class PointSource {
public:
	enum class Mode { Stream, Whole };

	virtual ~PointSource() = default;

	virtual const CloudMeta& meta() const = 0;

	virtual bool start(Mode mode, size_t maxDeviceBytes, std::string* err) = 0;
	virtual void rewind() = 0;
	virtual void stop() = 0;

	virtual BatchView view() const = 0;
	virtual uint64_t numPointsUploaded() const = 0;

	virtual bool isFullyResident() const = 0;
	virtual CUdeviceptr residentPoints() const = 0;

	virtual void setPointsConsumed(uint64_t n) = 0;
};

std::unique_ptr<PointSource> makeSyntheticSource(CudaContext& cuda,
                                                 uint64_t numPoints,
                                                 uint32_t seed = 1);

std::unique_ptr<PointSource> openPointSource(CudaContext& cuda,
                                             const std::vector<std::string>& files,
                                             std::string* err);

}
