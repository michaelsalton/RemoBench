#pragma once

#include <string>

namespace remo {

class GpuProfiler;

void timingRow(const GpuProfiler& profiler, const char* label,
               const std::string& scope);

class ScopeStats;
void timingRow(const char* label, const ScopeStats& stats);

}
