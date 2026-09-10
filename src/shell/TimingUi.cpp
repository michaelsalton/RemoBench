#include "shell/TimingUi.h"

#include <imgui.h>

#include "remo/GpuProfiler.h"

namespace remo {
namespace {

void emit(const char* label, const ScopeStats* stats) {
	ImGui::TableNextRow();
	ImGui::TableNextColumn();
	ImGui::TextUnformatted(label);
	ImGui::TableNextColumn();

	if (!stats || stats->empty()) {
		// Deliberately not "0.00 ms". A scope with no samples means nothing measured it,
		// which is a different fact from a kernel that took no time, and conflating the
		// two is the defect this whole layer was written to remove.
		ImGui::TextDisabled("not measured");
		return;
	}

	ImGui::Text("%.2f   med %.2f   p95 %.2f   n=%llu", stats->last(),
	            stats->median(), stats->percentile(0.95),
	            static_cast<unsigned long long>(stats->count()));
}

}  // namespace

void timingRow(const GpuProfiler& profiler, const char* label,
               const std::string& scope) {
	emit(label, profiler.find(scope));
}

void timingRow(const char* label, const ScopeStats& stats) { emit(label, &stats); }

}  // namespace remo
