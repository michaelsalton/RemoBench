#pragma once

#include <cstdint>
#include <string>

#include "remo/PointSource.h"

namespace remo {

// A .simlod file is a 24-byte header followed by our own Point record, so point i is at
// 24 + 16i and any range of it can be read directly. That is what makes the streaming
// path cheap here: no parsing, and no need to hold the cloud on the host.
struct SimlodInfo {
	uint64_t numPoints = 0;
	float headerBox[6] = {};
};

bool readSimlodHeader(const std::string& path, SimlodInfo& info, std::string* err);

// Coordinates only. The header carries a box, but it has never been trusted -- the
// loader has always used the observed one -- and under streaming the box has to be final
// before the first upload, so the observed one has to be known up front.
bool readSimlodBounds(const std::string& path, const SimlodInfo& info, double bounds[6],
                      std::string* err);

// Points [firstPoint, firstPoint + numPoints) into out[0..numPoints), translated.
bool readSimlodRange(const std::string& path, const SimlodInfo& info,
                     const double translation[3], uint64_t firstPoint,
                     uint64_t numPoints, Point* out, std::string* err);

}
