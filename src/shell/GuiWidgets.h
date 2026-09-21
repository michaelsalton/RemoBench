#pragma once

// Widgets shared by the two top-level panels (ControlPanel.cpp, StatsPanel.cpp).
// Header-only because ImGui 1.81 predates ImGui::BeginDisabled(), so every one of
// these is a two-line wrapper that wants to inline.

#include <cstdint>

#include <imgui.h>
#include <imgui_internal.h>

#include "remo/unsuck.hpp"

namespace remo {

// The control panel is anchored top-left and the dashboard top-right, so neither
// sits over the middle of the viewport where the cloud is. Both are ordinary
// movable, resizable windows once placed.
inline constexpr float kPanelMargin = 12.0f;
inline constexpr float kControlPanelWidth = 400.0f;
inline constexpr float kStatsPanelWidth = 420.0f;

inline void sectionHeader(const char* label) {
	ImGui::Spacing();
	ImGui::Separator();
	ImGui::TextColored(ImVec4(0.55f, 0.75f, 1.0f, 1.0f), "%s", label);
	ImGui::Spacing();
}

inline void beginDisabled(bool disabled) {
	if (!disabled) return;
	ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
	ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
}

inline void endDisabled(bool disabled) {
	if (!disabled) return;
	ImGui::PopStyleVar();
	ImGui::PopItemFlag();
}

inline void statRow(const char* label, uint64_t value) {
	ImGui::TableNextRow();
	ImGui::TableNextColumn();
	ImGui::TextUnformatted(label);
	ImGui::TableNextColumn();
	ImGui::TextUnformatted(formatNumber(static_cast<double>(value)).c_str());
}

inline void statRowF(const char* label, const char* fmt, double value) {
	ImGui::TableNextRow();
	ImGui::TableNextColumn();
	ImGui::TextUnformatted(label);
	ImGui::TableNextColumn();
	ImGui::Text(fmt, value);
}

inline void statRowText(const char* label, const char* value, bool bad = false) {
	ImGui::TableNextRow();
	ImGui::TableNextColumn();
	ImGui::TextUnformatted(label);
	ImGui::TableNextColumn();
	if (bad) {
		ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.2f, 1.0f), "%s", value);
	} else {
		ImGui::TextUnformatted(value);
	}
}

}
