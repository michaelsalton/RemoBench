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

class Framebuffer {
public:
	~Framebuffer();

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

	double fovy = 60.0;
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

}
