// Window input state.
//
// Replaces SimLOD's Runtime.h, which was a 65536-entry static global key array
// plus an unused GuiItem struct and an unused singleton. Roughly five lines of it
// were load-bearing.
//
// Passed by reference rather than being a global, because the benchmark runner may
// create and destroy windows, and because OrbitControls reaching into a global for
// one modifier key is the sort of coupling that makes a class untestable.

#pragma once

namespace remo {

struct Input {
	// GLFW_KEY_LAST is 348; round up.
	static constexpr int kNumKeys = 512;
	static constexpr int kNumButtons = 8;

	bool keys[kNumKeys] = {};
	bool buttons[kNumButtons] = {};

	double mouseX = 0.0;
	double mouseY = 0.0;

	// True while ImGui wants the mouse, so camera controls stand down instead of
	// orbiting the scene every time a slider is dragged.
	bool guiCapturedMouse = false;
	bool guiCapturedKeyboard = false;

	bool key(int code) const {
		return code >= 0 && code < kNumKeys && keys[code];
	}
	bool button(int code) const {
		return code >= 0 && code < kNumButtons && buttons[code];
	}
};

}  // namespace remo
