#include "io/LasReader.h"

#include <cstdio>
#include <string>

#include "laszip_api.h"

namespace remo {

namespace {

constexpr uint64_t kProgressStride = 10'000'000;

std::string laszipError(laszip_POINTER reader) {
	laszip_CHAR* message = nullptr;
	if (reader != nullptr && laszip_get_error(reader, &message) == 0 &&
	    message != nullptr && *message != '\0') {
		return message;
	}
	return "laszip reported no detail";
}

}

bool readLazPoints(const std::string& path, const LasHeaderInfo& info,
                   const double translation[3], Point* out,
                   double translatedBounds[6], std::string* err) {
	if (info.numPoints == 0) {
		if (err) *err = "LAZ header declares zero points: " + path;
		return false;
	}

	laszip_POINTER reader = nullptr;
	if (laszip_create(&reader) != 0) {
		if (err) *err = "laszip_create failed";
		return false;
	}

	const auto bail = [&](const std::string& what) {
		if (err) *err = what + " (" + laszipError(reader) + "): " + path;
		laszip_close_reader(reader);
		laszip_destroy(reader);
		return false;
	};

	laszip_decompress_selective(
		reader, laszip_DECOMPRESS_SELECTIVE_Z | laszip_DECOMPRESS_SELECTIVE_RGB);

	laszip_BOOL isCompressed = 0;
	if (laszip_open_reader(reader, path.c_str(), &isCompressed) != 0) {
		return bail("laszip_open_reader failed");
	}

	laszip_point* point = nullptr;
	if (laszip_get_point_pointer(reader, &point) != 0) {
		return bail("laszip_get_point_pointer failed");
	}

	const double sx = info.scale[0], sy = info.scale[1], sz = info.scale[2];
	const double ox = info.offset[0] + translation[0];
	const double oy = info.offset[1] + translation[1];
	const double oz = info.offset[2] + translation[2];
	const bool hasRgb = info.rgbOffset != 0;

	double lo[3] = {1e300, 1e300, 1e300};
	double hi[3] = {-1e300, -1e300, -1e300};
	uint64_t nextReport = kProgressStride;

	for (uint64_t i = 0; i < info.numPoints; ++i) {
		if (laszip_read_point(reader) != 0) {
			return bail("laszip_read_point failed at point " + std::to_string(i));
		}

		Point p;
		p.x = static_cast<float>(static_cast<double>(point->X) * sx + ox);
		p.y = static_cast<float>(static_cast<double>(point->Y) * sy + oy);
		p.z = static_cast<float>(static_cast<double>(point->Z) * sz + oz);

		if (hasRgb) {
			const uint16_t rgb[3] = {point->rgb[0], point->rgb[1], point->rgb[2]};
			p.color = packLasColor(rgb);
		} else {
			p.color = kLasNoColor;
		}

		out[i] = p;

		const double v[3] = {p.x, p.y, p.z};
		for (int a = 0; a < 3; ++a) {
			if (v[a] < lo[a]) lo[a] = v[a];
			if (v[a] > hi[a]) hi[a] = v[a];
		}

		if (i >= nextReport) {
			printf("\rremobench: decoding %s ... %3.0f%%",
			       path.c_str(),
			       100.0 * static_cast<double>(i) /
			           static_cast<double>(info.numPoints));
			fflush(stdout);
			nextReport += kProgressStride;
		}
	}

	if (nextReport > kProgressStride) printf("\r%80s\r", "");

	laszip_close_reader(reader);
	laszip_destroy(reader);

	for (int i = 0; i < 3; ++i) {
		translatedBounds[i] = lo[i];
		translatedBounds[3 + i] = hi[i];
	}
	return true;
}

}
