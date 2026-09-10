// Adapted from SimLOD: include/OrbitControls.h
// Upstream: https://github.com/m-schuetz/SimLOD @ fa7891613c138bd41775ca72a47cd89e32a5a647
// Copyright 2023 Markus Schuetz and Lukas Herzberger -- MIT (see THIRD_PARTY.md)
//
// Yaw/pitch/radius orbit camera in f64, producing an f64 world matrix.
//
// The best-reuse-value file in either upstream repo: pure glm, no framework
// coupling beyond one modifier-key lookup, and it already gets the awkward part
// right. Changes from upstream:
//   - Runtime::keyStates[342] -> an Input& parameter, so this is testable and not
//     tied to a process-wide global.
//   - glm::dmat4() default-construct replaced with glm::dmat4(1.0). GLM's default
//     constructor is uninitialised under GLM_FORCE_CTOR_INIT-less builds; upstream
//     relies on it behaving as identity, which is not guaranteed.
//   - frameBox(), so a freshly loaded cloud can be framed without the caller
//     open-coding the same trigonometry (SimLOD does it inline in its drop handler).
//
// KEEP THE Z-UP FLIP. Point clouds are Z-up; the flip matrix in update() is what
// reconciles that with a Y-up view convention. Getting it wrong costs an hour of
// confusion over a scene that is merely lying on its side.

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

		// Left-shift is the "select" modifier; don't orbit while it is held.
		if (input.key(340) || input.key(342)) return;
		if (input.guiCapturedMouse) return;

		if (m_leftDown) {
			yaw -= diff.x / 400.0;
			pitch -= diff.y / 400.0;
			// Clamp pitch instead of letting the camera roll through the pole.
			// Upstream lets it wrap, which flips the horizon mid-drag.
			constexpr double kLimit = 1.5707;  // just under pi/2
			pitch = std::clamp(pitch, -kLimit, kLimit);
		} else if (m_rightDown) {
			panLocal(-diff.x / 1000.0 * radius, diff.y / 1000.0 * radius);
		}
	}

	void onMouseScroll(const Input& input, double yoffset) {
		if (input.guiCapturedMouse) return;
		radius = yoffset < 0.0 ? radius * 1.1 : radius / 1.1;
		// A radius of zero cannot be recovered from by scrolling, since both
		// branches are multiplicative.
		radius = std::max(radius, 1e-6);
	}

	// Position the camera to view an axis-aligned box in full.
	void frameBox(const glm::dvec3& boxMin, const glm::dvec3& boxMax,
	              double fovyRad) {
		target = (boxMin + boxMax) * 0.5;
		const double extent = glm::length(boxMax - boxMin);
		// Half the diagonal over tan(fovy/2) frames the bounding sphere; the extra
		// factor leaves a little margin so the cloud is not flush to the edges.
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

		// Z-up (point cloud) -> Y-up (view). See the header note.
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

}  // namespace remo
