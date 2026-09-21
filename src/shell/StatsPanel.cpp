#include <imgui.h>
#include <implot.h>

#include <algorithm>
#include <cinttypes>

#include "remo/GpuProfiler.h"
#include "remo/unsuck.hpp"
#include "shell/App.h"
#include "shell/GuiWidgets.h"
#include "shell/TimingUi.h"

namespace remo {

namespace {

void frameTimePlot(const FrameHistory& history) {
	if (history.count == 0) return;

	float window[FrameHistory::kCapacity];
	std::copy(history.ms, history.ms + history.count, window);
	std::sort(window, window + history.count);

	const double peak = window[history.count - 1];
	const double p95 = window[std::min(history.count - 1,
	                                   int(history.count * 0.95))];
	const double yMax = std::max(20.0, p95 * 1.4);

	ImPlot::SetNextPlotLimits(0, FrameHistory::kCapacity, 0, yMax, ImGuiCond_Always);
	if (ImPlot::BeginPlot("##frametime", nullptr, nullptr, ImVec2(-1, 88),
	                      ImPlotFlags_CanvasOnly, ImPlotAxisFlags_NoDecorations,
	                      ImPlotAxisFlags_NoGridLines | ImPlotAxisFlags_NoTickMarks)) {
		const int offset = history.count == FrameHistory::kCapacity ? history.head : 0;

		ImPlot::SetNextFillStyle(ImVec4(0.35f, 0.62f, 1.0f, 1.0f), 0.25f);
		ImPlot::PlotShaded("frame ms", history.ms, history.count, 0.0, 1.0, 0.0, offset);
		ImPlot::SetNextLineStyle(ImVec4(0.55f, 0.75f, 1.0f, 1.0f), 1.5f);
		ImPlot::PlotLine("frame ms", history.ms, history.count, 1.0, 0.0, offset);

		const double sixtyFps = 1000.0 / 60.0;
		ImPlot::SetNextLineStyle(ImVec4(1.0f, 1.0f, 1.0f, 0.25f), 1.0f);
		ImPlot::PlotHLines("60 fps", &sixtyFps, 1);

		ImPlot::EndPlot();
	}
	hint("last %d frames, 0 - %.0f ms   peak %.0f   (line: 60 fps)", history.count,
	     yMax, peak);
}

}

void App::drawStatsPanel() {
	const ImGuiIO& io = ImGui::GetIO();
	const float x =
		std::max(kPanelMargin, io.DisplaySize.x - kStatsPanelWidth - kPanelMargin);
	const float height = std::max(320.0f, io.DisplaySize.y - 2.0f * kPanelMargin);

	ImGui::SetNextWindowPos(ImVec2(x, kPanelMargin), ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowSize(ImVec2(kStatsPanelWidth, height), ImGuiCond_FirstUseEver);
	ImGui::Begin("Statistics");

	ILodPipeline* pipeline = m_registry.active();

	sectionHeader("Frame");

	ImGui::Text("%.1f fps", m_renderer.fps());
	ImGui::SameLine();
	ImGui::TextDisabled("%.2f ms   med %.2f   p95 %.2f", m_renderer.frameMs(),
	                    m_frameTimeStats.median(),
	                    m_frameTimeStats.percentile(0.95));

	frameTimePlot(m_frameHistory);

	if (pipeline) {
		const TimingScopes scopes = pipeline->timingScopes();
		if (ImGui::BeginTable("frame_timing", 2, ImGuiTableFlags_SizingStretchProp)) {
			timingRow(m_profiler, "render kernel (ms)", scopes.render);
			ImGui::EndTable();
		}
	}

	hint("timing regime: %s%s", regimeName(m_profiler.regime()),
	     m_profiler.regime() == Regime::Strict
	         ? ""
	         : "  (--strict-timing for per-frame attribution)");
	if (m_profiler.droppedScopes() > 0) {
		ImGui::TextColored(ImVec4(1, 0.6f, 0.2f, 1), "%llu timing sample(s) dropped",
		                   static_cast<unsigned long long>(m_profiler.droppedScopes()));
	}

	sectionHeader("Cloud");
	if (ImGui::BeginTable("cloud", 2, ImGuiTableFlags_SizingStretchProp)) {
		statRow("points", m_meta.numPoints);
		statRowF("extent x", "%.1f", m_meta.boxSize[0]);
		statRowF("extent y", "%.1f", m_meta.boxSize[1]);
		statRowF("extent z", "%.1f", m_meta.boxSize[2]);
		ImGui::EndTable();
	}

	const double quantError = m_meta.worstQuantisationError();
	if (quantError > 0.01) {
		ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f),
		                   "float32 precision at the far corner: %.3f m", quantError);
	} else {
		hint("float32 precision at the far corner: %.4f m", quantError);
	}

	sectionHeader("Device memory");
	ImGui::Text("budget %.2f GB of %.2f GB total", double(m_budget.bytes) / 1e9,
	            double(m_budget.vramTotal) / 1e9);
	if (ImGui::IsItemHovered()) {
		ImGui::SetTooltip(
			"Computed once and handed unchanged to every pipeline, so memory\n"
			"figures are comparable. Upstream instead grabs 80%% of whatever\n"
			"happens to be free, which makes runs depend on what else was on\n"
			"the GPU at the time.");
	}

	if (!pipeline) {
		ImGui::TextColored(ImVec4(1, 0.35f, 0.25f, 1), "no active pipeline");
		ImGui::End();
		return;
	}

	const PipelineStats& s = pipeline->stats();

	sectionHeader("Pipeline stats");
	if (ImGui::BeginTable("stats", 2, ImGuiTableFlags_SizingStretchProp)) {
		statRow("points", s.numPoints);
		statRow("voxels", s.numVoxels);
		statRow("nodes", s.numNodes);
		statRow("visible nodes", s.numVisibleNodes);
		statRow("visible samples", s.numVisiblePoints + s.numVisibleVoxels);
		statRowF("scratch high water (MB)", "%.1f",
		         double(s.bytesHighWater) / (1024.0 * 1024.0));
		ImGui::EndTable();
	}

	if (s.allocOverflow) {
		ImGui::TextColored(ImVec4(1, 0.25f, 0.2f, 1),
		                   "ALLOCATOR OVERFLOW -- results are invalid");
	}
	if (s.nodeCapacityReached) {
		ImGui::TextColored(ImVec4(1, 0.25f, 0.2f, 1),
		                   "NODE POOL EXHAUSTED -- tree was truncated");
	}
	if (s.memCapacityReached) {
		ImGui::TextColored(ImVec4(1, 0.6f, 0.2f, 1),
		                   "device memory budget reached -- ingest stopped");
	}

	sectionHeader(("Pipeline: " + m_registry.activeId()).c_str());
	pipeline->guiStats(m_profiler);

	ImGui::End();
}

}
