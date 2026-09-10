// App::drawGui -- the shared settings panel and stats readout.
//
// Kept out of GLRenderer on purpose. CudaLOD bakes its GUI INTO the renderer
// (src/Renderer.cpp:546-609 hardcodes the four sampling-strategy buttons and a LOD
// slider), which means its renderer knows about one pipeline's tunables. That has to
// be undone before anything is swappable.
//
// The split here is the one that keeps comparisons honest: this panel owns knobs
// that apply to EVERY pipeline identically, and delegates to pipeline->gui() for
// tunables a pipeline owns. Nothing pipeline-specific is allowed above that line.

#include <imgui.h>
// For PushItemFlag / ImGuiItemFlags_Disabled -- see the shims below.
#include <imgui_internal.h>
#include <implot.h>

#include <cinttypes>

#include "remo/GpuProfiler.h"
#include "remo/unsuck.hpp"
#include "shell/App.h"
#include "shell/TimingUi.h"

namespace remo {

namespace {

// --- ImGui 1.81 compatibility shims ---------------------------------------
//
// The vendored ImGui is 1.81 (2021) and is PINNED, because SimLOD's plotting code
// requires a matching ImPlot -- it calls ImPlot::SetNextPlotLimitsX and a 3-argument
// BeginPlot, both since removed. See THIRD_PARTY.md.
//
// SeparatorText arrived in 1.89.5 and BeginDisabled in 1.88, so both are open-coded
// here. Keeping the shims in one place means the call sites read normally and
// unpinning later is a matter of deleting this block.

void sectionHeader(const char* label) {
	ImGui::Spacing();
	ImGui::Separator();
	ImGui::TextColored(ImVec4(0.55f, 0.75f, 1.0f, 1.0f), "%s", label);
	ImGui::Spacing();
}

void beginDisabled(bool disabled) {
	if (!disabled) return;
	ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
	ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
}

void endDisabled(bool disabled) {
	if (!disabled) return;
	ImGui::PopStyleVar();
	ImGui::PopItemFlag();
}

void statRow(const char* label, uint64_t value) {
	ImGui::TableNextRow();
	ImGui::TableNextColumn();
	ImGui::TextUnformatted(label);
	ImGui::TableNextColumn();
	ImGui::TextUnformatted(formatNumber(static_cast<double>(value)).c_str());
}

void statRowF(const char* label, const char* fmt, double value) {
	ImGui::TableNextRow();
	ImGui::TableNextColumn();
	ImGui::TextUnformatted(label);
	ImGui::TableNextColumn();
	ImGui::Text(fmt, value);
}

}  // namespace

void App::drawGui() {
	ILodPipeline* pipeline = m_registry.active();

	ImGui::SetNextWindowPos(ImVec2(12, 12), ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowSize(ImVec2(400, 700), ImGuiCond_FirstUseEver);
	ImGui::Begin("RemoBench");

	// --- status ------------------------------------------------------------
	if (!m_status.empty()) {
		if (m_statusIsError) {
			ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.25f, 1.0f), "%s",
			                   m_status.c_str());
		} else {
			ImGui::TextDisabled("%s", m_status.c_str());
		}
		ImGui::Separator();
	}

	// Mean fps with the tail beside it. A mean alone is the wrong summary for a
	// progressive builder: the whole point of the construction budget is that it bounds
	// the hitch, and a hitch is invisible in an average.
	ImGui::Text("%.1f fps  (%.2f ms, med %.2f, p95 %.2f)", m_renderer.fps(),
	            m_renderer.frameMs(), m_frameTimeStats.median(),
	            m_frameTimeStats.percentile(0.95));

	// Exit on the same row, right-aligned. Escape already closes the window, but a
	// keyboard-only shutdown is undiscoverable, and the loop is caller-owned
	// (GLRenderer.h) precisely so there IS a clean shutdown path -- so expose it.
	{
		constexpr float kExitWidth = 56.0f;
		ImGui::SameLine(ImGui::GetWindowContentRegionWidth() - kExitWidth);
		ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.45f, 0.13f, 0.11f, 1.0f));
		ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.65f, 0.18f, 0.15f, 1.0f));
		ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.80f, 0.22f, 0.18f, 1.0f));
		const bool exitClicked = ImGui::Button("Exit", ImVec2(kExitWidth, 0.0f));
		ImGui::PopStyleColor(3);
		if (ImGui::IsItemHovered()) ImGui::SetTooltip("quit RemoBench (or press Escape)");
		if (exitClicked) m_renderer.requestClose();
	}

	if (pipeline) {
		const TimingScopes scopes = pipeline->timingScopes();
		if (ImGui::BeginTable("frame_timing", 2, ImGuiTableFlags_SizingStretchProp)) {
			timingRow(m_profiler, "render kernel (ms)", scopes.render);
			ImGui::EndTable();
		}
		ImGui::TextDisabled("timing regime: %s%s", regimeName(m_profiler.regime()),
		                    m_profiler.regime() == Regime::Strict
		                        ? ""
		                        : "  (--strict-timing for per-frame attribution)");
		if (m_profiler.droppedScopes() > 0) {
			ImGui::TextColored(ImVec4(1, 0.6f, 0.2f, 1), "%llu timing sample(s) dropped",
			                   static_cast<unsigned long long>(
			                       m_profiler.droppedScopes()));
		}
	}

	// --- dataset ------------------------------------------------------------
	sectionHeader("Dataset");
	{
		// Scanned lazily and cached: a recursive walk of data/ every frame would be
		// pointless IO, and the directory holds multi-GB files.
		if (!m_datasetsScanned) scanDatasets();

		std::string preview;
		if (m_selectedDataset >= 0 &&
		    m_selectedDataset < static_cast<int>(m_datasets.size())) {
			preview = m_datasets[m_selectedDataset].label;
		} else if (m_datasets.empty()) {
			preview = "(nothing found in " + m_datasetDir + "/)";
		} else {
			preview = "(select a dataset)";
		}

		if (ImGui::BeginCombo("dataset", preview.c_str())) {
			for (int i = 0; i < static_cast<int>(m_datasets.size()); ++i) {
				const DatasetEntry& entry = m_datasets[i];

				// Show size, and for .simlod the exact point count -- derivable from the
				// file size without reading it, so the dropdown can be sanity-checked
				// against the reference before loading anything.
				std::string label = entry.label;
				label += "   " + formatNumber(double(entry.bytes) / (1024.0 * 1024.0)) +
				         " MB";
				if (entry.numPoints > 0) {
					label += "   " +
					         formatNumber(static_cast<double>(entry.numPoints)) + " pts";
				}
				if (!entry.supported) label += "   [" + entry.note + "]";

				beginDisabled(!entry.supported);
				if (ImGui::Selectable(label.c_str(), i == m_selectedDataset) &&
				    entry.supported) {
					m_selectedDataset = i;
					requestLoad(entry.path, entry.label);
				}
				endDisabled(!entry.supported);

				if (ImGui::IsItemHovered()) {
					ImGui::SetTooltip("%s", entry.path.c_str());
				}
			}
			ImGui::EndCombo();
		}

		ImGui::SameLine();
		if (ImGui::Button("rescan")) m_datasetsScanned = false;
		if (ImGui::IsItemHovered()) {
			ImGui::SetTooltip("re-read %s/ (override with REMOBENCH_DATA_DIR)",
			                  m_datasetDir.c_str());
		}

		// Ingest is synchronous, so a multi-GB file stalls the window. Say so rather
		// than letting it look like a hang.
		ImGui::TextDisabled("loading blocks the window; 5 GB takes a few seconds");

		// The synthetic fixture, for bringing things up without touching a file. Note
		// CudaLOD is refused for it -- see the tooltip on the pipeline entry.
		ImGui::TextUnformatted("synthetic:");
		const struct {
			const char* label;
			uint64_t count;
		} kSynthetic[] = {{"1M", 1'000'000}, {"5M", 5'000'000}, {"20M", 20'000'000}};
		for (const auto& s : kSynthetic) {
			ImGui::SameLine();
			if (ImGui::Button(s.label)) {
				m_selectedDataset = -1;
				requestLoad("synthetic:" + std::to_string(s.count),
				            std::string(s.label) + " synthetic points");
			}
		}

		// Files can also be dropped on the window; that goes through the same path.
	}

	// --- pipeline selection ------------------------------------------------
	sectionHeader("Pipeline");
	{
		const std::vector<PipelineInfo>& infos = m_registry.list();
		for (const PipelineInfo& info : infos) {
			const bool isActive = info.id == m_registry.activeId();
			const std::string reason =
				m_registry.unsupportedReason(info, m_meta, m_budget);
			const bool fits = reason.empty();

			// A pipeline that cannot fit the loaded cloud is disabled rather than
			// allowed to crash. On a 16GB card this is a real case, not a hypothetical.
			beginDisabled(!fits);
			if (ImGui::RadioButton(info.displayName.c_str(), isActive) && !isActive) {
				// REQUEST the switch; App applies it at the top of the next frame.
				//
				// Switching here would destroy the pipeline that `pipeline` above
				// points at, and this function goes on to read its stats and call
				// pipeline->gui() -- a use-after-free that crashes on the first click.
				m_pendingPipeline = info.id;
			}
			endDisabled(!fits);
			if (!fits && ImGui::IsItemHovered()) {
				ImGui::SetTooltip("%s", reason.c_str());
			}
		}
	}

	// --- shared settings ---------------------------------------------------
	// Everything here applies identically to every pipeline. That is what makes an
	// A/B attributable to the LOD algorithm rather than to one of them having been
	// handed a different point size.
	sectionHeader("Shared settings");

	ImGui::SliderFloat("LOD budget (px)", &m_settings.lodPixelBudget, 8.0f, 512.0f,
	                   "%.0f");
	if (ImGui::IsItemHovered()) {
		ImGui::SetTooltip(
			"Projected node extent, in pixels, above which a node is subdivided.\n"
			"A single shared metric so that 'both at the same LOD' means the same\n"
			"cut: SimLOD's native test is in world units (dataset-dependent) and\n"
			"CudaLOD's is angular but not viewport-calibrated.");
	}

	ImGui::SliderInt("point size", &m_settings.pointSize, 1, 8);
	ImGui::Checkbox("update visibility", &m_settings.doUpdateVisibility);
	if (ImGui::IsItemHovered()) {
		ImGui::SetTooltip(
			"Off freezes the LOD cut while the camera keeps moving,\n"
			"so you can fly around and inspect where the pipeline chose\n"
			"its boundaries.");
	}

	ImGui::Checkbox("EDL", &m_settings.enableEDL);
	ImGui::SameLine();
	beginDisabled(!m_settings.enableEDL);
	ImGui::SliderFloat("strength", &m_settings.edlStrength, 0.0f, 2.0f, "%.2f");
	endDisabled(!m_settings.enableEDL);

	const char* colorModes[] = {"RGB", "by node", "by LOD", "white"};
	ImGui::Combo("colour", &m_settings.colorMode, colorModes,
	             IM_ARRAYSIZE(colorModes));

	// --- octree wireframe --------------------------------------------------
	ImGui::Checkbox("node boxes", &m_settings.showBoundingBox);
	if (ImGui::IsItemHovered()) {
		ImGui::SetTooltip(
			"Draw a cube per node the selection pass emitted, coloured by\n"
			"level -- the same colours 'by LOD' gives the samples, so a box\n"
			"and its contents match.\n"
			"\n"
			"This is the LOD cut itself rather than an inference from it:\n"
			"SimLOD emits a disjoint frontier, so its boxes tile; CudaLOD\n"
			"marks parents and children both visible, so its boxes nest.\n"
			"`flat` has no tree and draws none.");
	}

	ImGui::SameLine();
	ImGui::Checkbox("points", &m_settings.showPoints);
	if (ImGui::IsItemHovered()) {
		ImGui::SetTooltip(
			"Off leaves the wireframe on its own, which is the only way to\n"
			"read it in a dense cloud.\n"
			"\n"
			"A view toggle only: selection still runs, so the visible-sample\n"
			"counts below continue to report what the pipeline CHOSE and a\n"
			"hidden frame is not mistaken for a cheap one.");
	}

	// --- cloud -------------------------------------------------------------
	sectionHeader("Cloud");
	if (ImGui::BeginTable("cloud", 2, ImGuiTableFlags_SizingStretchProp)) {
		statRow("points", m_meta.numPoints);
		statRowF("extent x", "%.1f", m_meta.boxSize[0]);
		statRowF("extent y", "%.1f", m_meta.boxSize[1]);
		statRowF("extent z", "%.1f", m_meta.boxSize[2]);
		ImGui::EndTable();
	}

	// Surface the f32 precision limit rather than letting it be discovered as
	// visual noise. Sub-mm for a city block, centimetres nationally, ~0.5m for
	// ECEF coordinates -- at which point float32 device coordinates are unusable.
	const double quantError = m_meta.worstQuantisationError();
	if (quantError > 0.01) {
		ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f),
		                   "float32 precision at the far corner: %.3f m", quantError);
	} else {
		ImGui::TextDisabled("float32 precision at the far corner: %.4f m",
		                    quantError);
	}

	// --- memory ------------------------------------------------------------
	sectionHeader("Device memory");
	ImGui::Text("budget %.2f GB of %.2f GB total",
	            double(m_budget.bytes) / 1e9, double(m_budget.vramTotal) / 1e9);
	if (ImGui::IsItemHovered()) {
		ImGui::SetTooltip(
			"Computed once and handed unchanged to every pipeline, so memory\n"
			"figures are comparable. Upstream instead grabs 80%% of whatever\n"
			"happens to be free, which makes runs depend on what else was on\n"
			"the GPU at the time.");
	}

	// --- pipeline stats ----------------------------------------------------
	if (pipeline) {
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

		// Health flags. Any of these means the structure was silently truncated, so
		// the run is not a valid data point -- say so loudly rather than reporting a
		// suspiciously good number.
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
		pipeline->gui(m_profiler);
	} else {
		ImGui::TextColored(ImVec4(1, 0.35f, 0.25f, 1), "no active pipeline");
	}

	ImGui::End();
}

}  // namespace remo
