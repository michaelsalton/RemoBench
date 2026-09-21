// Adapted from SimLOD: include/OrbitControls.h
// Upstream: https://github.com/m-schuetz/SimLOD @ fa7891613c138bd41775ca72a47cd89e32a5a647
// Copyright 2023 Markus Schuetz and Lukas Herzberger -- MIT (see THIRD_PARTY.md)

#pragma once

#include <algorithm>
#include <cmath>

#include <glm/common.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/matrix.hpp>

#include "shell/Input.h"

namespace remo {

class OrbitControls {
public:
	double yaw = 0.0;
	double pitch = 0.0;
	double radius = 2.0;
	glm::dvec3 target = {0.0, 0.0, 0.0};

	glm::dmat4 world = glm::dmat4(1.0);

	glm::dvec3 direction() const {
		return glm::dvec3(rotation() * glm::dvec4(0, 1, 0, 1.0));
	}

	glm::dvec3 position() const { return target - radius * direction(); }

	glm::dmat4 rotation() const {
		const glm::dvec3 up = {0, 0, 1};
		const glm::dvec3 right = {1, 0, 0};
		return glm::rotate(pitch, right) * glm::rotate(yaw, up);
	}

	void onMouseButton(int button, bool down) {
		if (button == 0) m_leftDown = down;
		if (button == 1) m_rightDown = down;
	}

	void onMouseMove(const Input& input, double x, double y) {
		const glm::dvec2 pos = {x, y};
		const glm::dvec2 diff = pos - m_mousePos;
		m_mousePos = pos;

		if (input.key(340) || input.key(342)) return;
		if (input.guiCapturedMouse) return;

		if (m_leftDown) {
			yaw -= diff.x / 400.0;
			pitch -= diff.y / 400.0;
			constexpr double kLimit = 1.5707;
			pitch = std::clamp(pitch, -kLimit, kLimit);
		} else if (m_rightDown) {
			panLocal(-diff.x / 1000.0 * radius, diff.y / 1000.0 * radius);
		}
	}

	void onMouseScroll(const Input& input, double yoffset) {
		if (input.guiCapturedMouse) return;
		radius = yoffset < 0.0 ? radius * 1.1 : radius / 1.1;
		radius = std::max(radius, 1e-6);
	}

	void frameBox(const glm::dvec3& boxMin, const glm::dvec3& boxMax,
	              double fovyRad) {
		target = (boxMin + boxMax) * 0.5;
		const double extent = glm::length(boxMax - boxMin);
		radius = (extent * 0.5) / std::tan(std::max(fovyRad, 1e-3) * 0.5) * 1.1;
		yaw = 0.35;
		pitch = -0.6;
	}

	void update() {
		const glm::dvec3 up = {0, 0, 1};
		const glm::dvec3 right = {1, 0, 0};

		const auto translateRadius =
			glm::translate(glm::dmat4(1.0), glm::dvec3(0.0, 0.0, radius));
		const auto translateTarget = glm::translate(glm::dmat4(1.0), target);
		const auto rotYaw = glm::rotate(yaw, up);
		const auto rotPitch = glm::rotate(pitch, right);

		const auto flip = glm::dmat4(1.0, 0.0, 0.0, 0.0,
		                             0.0, 0.0, 1.0, 0.0,
		                             0.0, -1.0, 0.0, 0.0,
		                             0.0, 0.0, 0.0, 1.0);

		world = translateTarget * rotYaw * rotPitch * flip * translateRadius;
	}

private:
	void panLocal(double x, double y) {
		const auto origin = glm::dvec3(world * glm::dvec4(0, 0, 0, 1));
		const auto right = glm::dvec3(world * glm::dvec4(1, 0, 0, 1));
		const auto forward = glm::dvec3(world * glm::dvec4(0, 1, 0, 1));
		target += glm::normalize(right - origin) * x +
		          glm::normalize(forward - origin) * y;
	}

	bool m_leftDown = false;
	bool m_rightDown = false;
	glm::dvec2 m_mousePos = {0.0, 0.0};
};

}
