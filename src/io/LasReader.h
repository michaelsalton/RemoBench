#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "remo/PointSource.h"

namespace remo {

struct LasHeaderInfo {
	int versionMajor = 0;
	int versionMinor = 0;
	uint64_t numPoints = 0;
	uint64_t offsetToPointData = 0;
	uint32_t format = 0;
	uint32_t bytesPerPoint = 0;
	uint32_t rgbOffset = 0;
	double scale[3] = {0, 0, 0};
	double offset[3] = {0, 0, 0};
	double min[3] = {0, 0, 0};
	double max[3] = {0, 0, 0};
	bool compressed = false;
};

inline uint32_t packLasColor(const uint16_t rgb[3]) {
	const uint32_t r = rgb[0] > 255 ? rgb[0] / 256u : rgb[0];
	const uint32_t g = rgb[1] > 255 ? rgb[1] / 256u : rgb[1];
	const uint32_t b = rgb[2] > 255 ? rgb[2] / 256u : rgb[2];
	return r | (g << 8) | (b << 16) | 0xFF000000u;
}

constexpr uint32_t kLasNoColor = 0xFFFFFFFFu;

bool readLasHeader(const std::string& path, LasHeaderInfo& info, std::string* err);

bool readLasPoints(const std::string& path, const LasHeaderInfo& info,
                   const double translation[3], Point* out,
                   double translatedBounds[6], std::string* err);

bool readLazPoints(const std::string& path, const LasHeaderInfo& info,
                   const double translation[3], Point* out,
                   double translatedBounds[6], std::string* err);

}
