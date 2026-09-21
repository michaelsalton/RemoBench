#pragma once

#include <cstdint>
#include <string>

#include "remo/PointSource.h"

namespace remo {

// Where the host gets batch k's points from.
//
// The split exists because the ring made "read the whole cloud, then upload it" and
// "upload a slot at a time from wherever the points live" two different questions.
// CloudSource owns the device ring and the backpressure window and knows nothing about
// file formats; a producer knows the format and nothing about the ring.
//
// Coordinates are already translated when they reach here -- the translation is fixed
// before the first upload, because the octree root cube is sized from it and growing the
// box under a live tree invalidates every node already built.
class BatchProducer {
public:
	virtual ~BatchProducer() = default;

	// A pointer to points [first, first+count) if this producer already holds them
	// contiguously, else nullptr and the caller stages through fill(). Not const: a
	// producer may have to read to answer.
	virtual const Point* peek(uint64_t first, uint32_t count) {
		(void)first;
		(void)count;
		return nullptr;
	}

	virtual bool fill(uint64_t first, uint32_t count, Point* out, std::string* err) = 0;
};

}
