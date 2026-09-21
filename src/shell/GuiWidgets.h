#pragma once

#include <cstdarg>
#include <cstdint>

#include <imgui.h>
#include <imgui_internal.h>

#include "remo/unsuck.hpp"

namespace remo {

inline constexpr float kPanelMargin = 12.0f;
inline constexpr float kControlPanelWidth = 440.0f;
inline constexpr float kStatsPanelWidth = 500.0f;

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

inline void paragraph(const char* text) { ImGui::TextWrapped("%s", text); }

inline void hint(const char* fmt, ...) IM_FMTARGS(1);
inline void hint(const char* fmt, ...) {
	va_list args;
	va_start(args, fmt);
	ImGui::PushStyleColor(ImGuiCol_Text,
	                      ImGui::GetStyle().Colors[ImGuiCol_TextDisabled]);
	ImGui::TextWrappedV(fmt, args);
	ImGui::PopStyleColor();
	va_end(args);
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
