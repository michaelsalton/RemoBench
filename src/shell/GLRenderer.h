// Window, GL context, ImGui bootstrap, offscreen render target, and the frame loop.
//
// Rewritten rather than adapted from SimLOD's GLRenderer. Its decomposition is
// good and is kept (Camera / Framebuffer-with-lazy-resize / Texture / a
// loop(update, render) shape), but three defects are architectural rather than
// cosmetic and all three block goals of this project:
//
//   1. `auto _controls = make_shared<OrbitControls>()` at FILE SCOPE
//      (GLRenderer.cpp:9), shared by every GLRenderer instance. Single window,
//      single camera, by construction -- which rules out the side-by-side A/B view
//      that is the whole point of a comparison tool.
//   2. `static GLRenderer* ref = this` inside init() for the GLFW drop callback.
//      Same problem. Here the window user pointer carries the instance.
//   3. loop() calls exit(EXIT_SUCCESS) and never returns (GLRenderer.cpp:375).
//      No shutdown path at all, which means no headless benchmark run (it would
//      truncate its own results file) and no GPU test harness. runFrame() returns
//      a bool and the caller owns the loop.
//
// Also note: SimLOD's performance panel calls ImPlot::SetNextPlotLimitsX and a
// 3-argument ImPlot::BeginPlot, both REMOVED from current ImPlot. That code
// physically cannot be carried forward, which is why plotting is rewritten here
// rather than copied.

#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#include "shell/Input.h"
#include "shell/OrbitControls.h"

struct GLFWwindow;

namespace remo {

// Lazily-resized offscreen colour+depth target. CUDA renders into the colour
// attachment via a registered surface; GL then blits it to the backbuffer.
class Framebuffer {
public:
	~Framebuffer();

	// No-op when the size is unchanged, so this is safe to call every frame.
	// Returns true if the attachments were actually recreated, which is the signal
	// GLInterop needs to re-register its surface.
	bool setSize(int width, int height);

	unsigned int fbo() const { return m_fbo; }
	unsigned int colorTexture() const { return m_colorTexture; }
	int width() const { return m_width; }
	int height() const { return m_height; }

private:
	void destroy();

	unsigned int m_fbo = 0;
	unsigned int m_colorTexture = 0;
	unsigned int m_depthTexture = 0;
	int m_width = 0;
	int m_height = 0;
};

class Camera {
public:
	void setSize(int width, int height);
	void update();

	glm::dmat4 world = glm::dmat4(1.0);
	glm::dmat4 view = glm::dmat4(1.0);
	glm::dmat4 proj = glm::dmat4(1.0);

	double fovy = 60.0;   // degrees
	double near = 0.1;
	double far = 2'000'000.0;

	int width = 1280;
	int height = 720;

	double fovyRad() const;
};

class GLRenderer {
public:
	GLRenderer();
	~GLRenderer();

	GLRenderer(const GLRenderer&) = delete;
	GLRenderer& operator=(const GLRenderer&) = delete;

	bool init(const std::string& title, int width, int height, std::string* err);

	// One frame: poll input, drain the hot-reload event queue, update the camera,
	// run the caller's callbacks, blit and swap.
	//
	// Returns false when the window should close -- the caller owns the loop, so
	// there IS a shutdown path.
	bool runFrame(const std::function<void()>& update,
	              const std::function<void()>& render);

	void requestClose();

	Framebuffer& framebuffer() { return m_framebuffer; }
	const Framebuffer& framebuffer() const { return m_framebuffer; }
	Camera& camera() { return m_camera; }
	const Camera& camera() const { return m_camera; }
	OrbitControls& controls() { return m_controls; }
	const Input& input() const { return m_input; }

	int width() const { return m_width; }
	int height() const { return m_height; }
	double fps() const { return m_fps; }
	double frameMs() const { return m_frameMs; }

	// Files dropped on the window. Not the only load path -- see --open in main.cpp.
	// Drag-and-drop is upstream's ONLY way to load a cloud, which is a liability for
	// anything scripted.
	void onFileDrop(std::function<void(const std::vector<std::string>&)> callback);

private:
	static void keyCallback(GLFWwindow*, int key, int scancode, int action, int mods);
	static void mouseButtonCallback(GLFWwindow*, int button, int action, int mods);
	static void cursorPosCallback(GLFWwindow*, double x, double y);
	static void scrollCallback(GLFWwindow*, double dx, double dy);
	static void dropCallback(GLFWwindow*, int count, const char** paths);
	static GLRenderer* from(GLFWwindow* window);

	void blitToBackbuffer();

	GLFWwindow* m_window = nullptr;
	Framebuffer m_framebuffer;
	Camera m_camera;
	OrbitControls m_controls;
	Input m_input;

	int m_width = 0;
	int m_height = 0;

	double m_fps = 0.0;
	double m_frameMs = 0.0;
	double m_lastFrameTime = 0.0;
	double m_fpsWindowStart = 0.0;
	int m_fpsFrames = 0;

	bool m_imguiReady = false;

	std::vector<std::function<void(const std::vector<std::string>&)>> m_dropCallbacks;
};

}  // namespace remo
