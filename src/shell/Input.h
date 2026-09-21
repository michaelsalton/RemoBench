#pragma once

namespace remo {

struct Input {
	static constexpr int kNumKeys = 512;
	static constexpr int kNumButtons = 8;

	bool keys[kNumKeys] = {};
	bool buttons[kNumButtons] = {};

	double mouseX = 0.0;
	double mouseY = 0.0;

	bool guiCapturedMouse = false;
	bool guiCapturedKeyboard = false;

	bool key(int code) const {
		return code >= 0 && code < kNumKeys && keys[code];
	}
	bool button(int code) const {
		return code >= 0 && code < kNumButtons && buttons[code];
	}
};

}
