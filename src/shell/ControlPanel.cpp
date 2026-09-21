// The control panel: everything that changes what the program does. Read-outs
// live in StatsPanel.cpp, so a screenshot of one panel answers "what was set"
// and the other "what happened".

#include <imgui.h>

#include "remo/unsuck.hpp"
#include "shell/App.h"
#include "shell/GuiWidgets.h"

namespace remo {

void App::drawGui() {
	drawControlPanel();
	drawStatsPanel();
}

void App::drawControlPanel() {
	ImGui::SetNextWindowPos(ImVec2(kPanelMargin, kPanelMargin), ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowSize(ImVec2(kControlPanelWidth, 660), ImGuiCond_FirstUseEver);
	ImGui::Begin("Controls");

	if (!m_status.empty()) {
		if (m_statusIsError) {
			ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.25f, 1.0f), "%s", m_status.c_str());
		} else {
			ImGui::TextDisabled("%s", m_status.c_str());
		}
		ImGui::Separator();
	}

	{
		constexpr float kExitWidth = 56.0f;
		ImGui::TextDisabled("RemoBench");
		ImGui::SameLine(ImGui::GetWindowContentRegionWidth() - kExitWidth);
		ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.45f, 0.13f, 0.11f, 1.0f));
		ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.65f, 0.18f, 0.15f, 1.0f));
		ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.80f, 0.22f, 0.18f, 1.0f));
		const bool exitClicked = ImGui::Button("Exit", ImVec2(kExitWidth, 0.0f));
		ImGui::PopStyleColor(3);
		if (ImGui::IsItemHovered()) ImGui::SetTooltip("quit RemoBench (or press Escape)");
		if (exitClicked) m_renderer.requestClose();
	}

	sectionHeader("Dataset");
	{
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

		ImGui::TextDisabled("loading blocks the window; 5 GB takes a few seconds");

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
	}

	sectionHeader("Pipeline");
	{
		const std::vector<PipelineInfo>& infos = m_registry.list();
		for (const PipelineInfo& info : infos) {
			const bool isActive = info.id == m_registry.activeId();
			const std::string reason =
				m_registry.unsupportedReason(info, m_meta, m_budget);
			const bool fits = reason.empty();

			beginDisabled(!fits);
			if (ImGui::RadioButton(info.displayName.c_str(), isActive) && !isActive) {
				m_pendingPipeline = info.id;
			}
			endDisabled(!fits);
			if (!fits && ImGui::IsItemHovered()) {
				ImGui::SetTooltip("%s", reason.c_str());
			}
		}
	}

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
			"counts in the dashboard continue to report what the pipeline\n"
			"CHOSE and a hidden frame is not mistaken for a cheap one.");
	}

	if (ILodPipeline* pipeline = m_registry.active()) {
		sectionHeader(("Pipeline: " + m_registry.activeId()).c_str());
		pipeline->guiControls();
	} else {
		sectionHeader("Pipeline");
		ImGui::TextColored(ImVec4(1, 0.35f, 0.25f, 1), "no active pipeline");
	}

	ImGui::End();
}

}
