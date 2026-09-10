// Reader for ASPRS LAS (uncompressed) and LAZ (laszip-compressed) point clouds.
//
// Why this exists at all, given that `.simlod` loads ~20x faster: the bounding box.
// A `.simlod` header stores the box in float32, LAS stores it in float64, and both
// CudaLOD and SimLOD derive the octree root cube from it -- so the same points read
// through the two formats produce trees that differ in the last bits of a cell
// boundary. bench/reference/README.md records exactly that as an open question
// (+388 voxels, 0.003%). Reading the LAS the reference read is the only way to close
// it. It also unlocks the datasets that only ship as LAS/LAZ.
//
// Two deliberate departures from upstream's LasLoader, which this otherwise follows:
//
//   - The translation is applied in float64, inside the parse, before the coordinate
//     is narrowed to float32. Upstream does the same; the important part is that our
//     `.simlod` path CANNOT (its points are already f32 on disk), so this reader is
//     the only one that achieves the precision CloudMeta::worstQuantisationError
//     advertises. Do not "simplify" it into a post-pass over f32 points -- at UTM
//     magnitudes that throws away ~5 bits before the translation can rescue them.
//   - Alpha is set to 255. Upstream leaves `point.a` uninitialised, so its alpha is
//     stack garbage; our Point packs RGBA into one uint32 and the rasteriser reads
//     the whole word.
//
// The 16-bit-colour heuristic IS upstream's, per point and per component
// (`v > 255 ? v / 256 : v`), and that is on purpose: it is also what CudaLOD and
// PotreeConverter do, so colours stay bit-identical to the references we validate
// against. A single per-file decision would be more self-consistent and would also
// silently diverge from them.
//
// Point formats without RGB (0, 1, 4, 6, 9) load as white rather than as mapped
// intensity. Intensity needs a normalisation choice of its own -- it is uncalibrated
// and per-sensor -- and guessing one here would put an invented value in the colour
// channel that every quality comparison then reads.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "remo/PointSource.h"

namespace remo {

// The subset of the LAS public header block this project needs. Parsed from the
// first 375 bytes; no point data is touched.
struct LasHeaderInfo {
	int versionMajor = 0;
	int versionMinor = 0;
	uint64_t numPoints = 0;
	uint64_t offsetToPointData = 0;
	uint32_t format = 0;         // point data record format, compression bits masked off
	uint32_t bytesPerPoint = 0;  // point data record length, incl. any extra bytes
	uint32_t rgbOffset = 0;      // byte offset of the RGB triple, 0 when the format has none
	double scale[3] = {0, 0, 0};
	double offset[3] = {0, 0, 0};
	double min[3] = {0, 0, 0};  // float64, unlike .simlod's float32 -- the whole point
	double max[3] = {0, 0, 0};
	bool compressed = false;  // laszip; the record length above is the DECOMPRESSED one
};

// Upstream's per-component "8-bit stored in a 16-bit field" heuristic. Single-sourced
// so the .las and .laz paths cannot drift apart -- a colour difference between the two
// readings of the same cloud would be invisible in every structural count and would
// only ever show up as unexplained noise in an image comparison.
inline uint32_t packLasColor(const uint16_t rgb[3]) {
	const uint32_t r = rgb[0] > 255 ? rgb[0] / 256u : rgb[0];
	const uint32_t g = rgb[1] > 255 ? rgb[1] / 256u : rgb[1];
	const uint32_t b = rgb[2] > 255 ? rgb[2] / 256u : rgb[2];
	return r | (g << 8) | (b << 16) | 0xFF000000u;
}

// What a point format without RGB gets. See the header comment on why this is not
// derived from intensity.
constexpr uint32_t kLasNoColor = 0xFFFFFFFFu;

// Header only. Cheap enough for the dataset scan to call it on every file it lists.
bool readLasHeader(const std::string& path, LasHeaderInfo& info, std::string* err);

// Parses `info.numPoints` records into `out`, which must have room for all of them.
//
// `translation` is added in float64 before the narrowing to float32, so pass the same
// value that ends up in CloudMeta::translation.
//
// `translatedBounds` returns the bounds actually observed, {min[3], max[3]}, in
// post-translation coordinates -- i.e. directly comparable against {0,0,0} and
// CloudMeta::boxSize. It is not diagnostics: a LAS header whose bbox does not contain
// its own points is a real and common defect, and the octree root cube is that box, so
// the points outside it are silently clamped into one corner cell rather than faulting.
// The caller checks it and re-reads if it has to.
bool readLasPoints(const std::string& path, const LasHeaderInfo& info,
                   const double translation[3], Point* out,
                   double translatedBounds[6], std::string* err);

// laszip-backed decode, in LazReader.cpp. Split out so that laszip -- the one
// LGPL-2.1 dependency (see THIRD_PARTY.md) -- is confined to a single translation
// unit, and so that `using namespace std;` at the bottom of laszip_api.h cannot leak
// into ours.
bool readLazPoints(const std::string& path, const LasHeaderInfo& info,
                   const double translation[3], Point* out,
                   double translatedBounds[6], std::string* err);

}  // namespace remo
