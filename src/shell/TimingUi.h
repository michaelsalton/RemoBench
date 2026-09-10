// One presentation of a GpuProfiler scope, shared by the settings panel and by every
// pipeline's gui().
//
// It exists so there is exactly one place that decides what an UNMEASURED scope looks
// like. That is the whole point of the layer: the previous code printed a
// default-initialised double as "0.00 ms" in the same format as a real measurement, so
// the reader could not tell a fast kernel from a kernel nobody timed. Here, absent is
// rendered as absent.

#pragma once

#include <string>

namespace remo {

class GpuProfiler;

// A row of "label | last  med  p95  n", or "label | not measured".
void timingRow(const GpuProfiler& profiler, const char* label,
               const std::string& scope);

// Same, for a statistic that is not a profiler scope (host frame time, for one).
class ScopeStats;
void timingRow(const char* label, const ScopeStats& stats);

}  // namespace remo
