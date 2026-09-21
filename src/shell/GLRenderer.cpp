#include "shell/GLRenderer.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cstdio>

#include <glm/gtc/matrix_transform.hpp>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <implot.h>

#include "remo/unsuck.hpp"

namespace remo {
namespace {

void glfwErrorCallback(int code, const char* description) {
	fprintf(stderr, "remobench: glfw error %d: %s\n", code, description);
}

void APIENTRY glDebugCallback(GLenum, GLenum type, GLuint,
                              GLenum severity, GLsizei,
                              const GLchar* message, const void*) {
	if (severity == GL_DEBUG_SEVERITY_NOTIFICATION) return;
	fprintf(stderr, "remobench: GL %s: %s\n",
	        type == GL_DEBUG_TYPE_ERROR ? "error" : "message", message);
}

}

Framebuffer::~Framebuffer() { destroy(); }

void Framebuffer::destroy() {
	if (m_colorTexture) glDeleteTextures(1, &m_colorTexture);
	if (m_depthTexture) glDeleteTextures(1, &m_depthTexture);
	if (m_fbo) glDeleteFramebuffers(1, &m_fbo);
	m_colorTexture = m_depthTexture = m_fbo = 0;
}

bool Framebuffer::setSize(int width, int height) {
	width = width < 1 ? 1 : width;
	height = height < 1 ? 1 : height;
	if (width == m_width && height == m_height && m_fbo != 0) return false;

	destroy();
	m_width = width;
	m_height = height;

	glCreateFramebuffers(1, &m_fbo);

	glCreateTextures(GL_TEXTURE_2D, 1, &m_colorTexture);
	glTextureStorage2D(m_colorTexture, 1, GL_RGBA8, width, height);
	glTextureParameteri(m_colorTexture, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTextureParameteri(m_colorTexture, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glCreateTextures(GL_TEXTURE_2D, 1, &m_depthTexture);
	glTextureStorage2D(m_depthTexture, 1, GL_DEPTH_COMPONENT32F, width, height);

	glNamedFramebufferTexture(m_fbo, GL_COLOR_ATTACHMENT0, m_colorTexture, 0);
	glNamedFramebufferTexture(m_fbo, GL_DEPTH_ATTACHMENT, m_depthTexture, 0);

	return true;
}

double Camera::fovyRad() const { return glm::radians(fovy); }

void Camera::setSize(int w, int h) {
	width = w < 1 ? 1 : w;
	height = h < 1 ? 1 : h;
}

void Camera::update() {
	view = glm::inverse(world);
	const double aspect = double(width) / double(height);
	proj = glm::perspective(fovyRad(), aspect, near, far);
}

GLRenderer::GLRenderer() = default;

GLRenderer::~GLRenderer() {
	if (m_imguiReady) {
		ImGui_ImplOpenGL3_Shutdown();
		ImGui_ImplGlfw_Shutdown();
		ImPlot::DestroyContext();
		ImGui::DestroyContext();
	}
	if (m_window) glfwDestroyWindow(m_window);
	glfwTerminate();
}

GLRenderer* GLRenderer::from(GLFWwindow* window) {
	return static_cast<GLRenderer*>(glfwGetWindowUserPointer(window));
}

bool GLRenderer::init(const std::string& title, int width, int height,
                      std::string* err) {
	glfwSetErrorCallback(glfwErrorCallback);

	if (!glfwInit()) {
		if (err) *err = "glfwInit failed (no display?)";
		return false;
	}

#ifdef GLFW_PLATFORM
	const char* platform = std::getenv("REMOBENCH_GLFW_PLATFORM");
	if (!platform || std::string(platform) == "x11") {
		if (glfwPlatformSupported(GLFW_PLATFORM_X11)) {
			glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_X11);
		}
	}
#endif

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);

	m_window = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
	if (!m_window) {
		if (err) *err = "glfwCreateWindow failed (need GL 4.5 core)";
		return false;
	}

	glfwSetWindowUserPointer(m_window, this);
	glfwMakeContextCurrent(m_window);
	glfwSwapInterval(1);

	glewExperimental = GL_TRUE;
	const GLenum glewStatus = glewInit();
	if (glewStatus != GLEW_OK) {
		if (glewStatus != 4) {
			if (err) {
				*err = std::string("glewInit failed: ") +
				       reinterpret_cast<const char*>(glewGetErrorString(glewStatus));
			}
			return false;
		}
	}
	glGetError();

	if (glDebugMessageCallback) {
		glEnable(GL_DEBUG_OUTPUT);
		glDebugMessageCallback(glDebugCallback, nullptr);
	}

	glfwSetKeyCallback(m_window, keyCallback);
	glfwSetMouseButtonCallback(m_window, mouseButtonCallback);
	glfwSetCursorPosCallback(m_window, cursorPosCallback);
	glfwSetScrollCallback(m_window, scrollCallback);
	glfwSetDropCallback(m_window, dropCallback);

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImPlot::CreateContext();
	ImGui::StyleColorsDark();
	ImGui_ImplGlfw_InitForOpenGL(m_window, true);
	ImGui_ImplOpenGL3_Init("#version 450");
	m_imguiReady = true;

	glfwGetFramebufferSize(m_window, &m_width, &m_height);
	m_camera.setSize(m_width, m_height);
	m_framebuffer.setSize(m_width, m_height);

	m_lastFrameTime = now();
	m_fpsWindowStart = m_lastFrameTime;

	printf("remobench: GL %s on %s\n", glGetString(GL_VERSION),
	       glGetString(GL_RENDERER));
	return true;
}

void GLRenderer::requestClose() {
	if (m_window) glfwSetWindowShouldClose(m_window, GLFW_TRUE);
}

void GLRenderer::onFileDrop(
	std::function<void(const std::vector<std::string>&)> callback) {
	m_dropCallbacks.push_back(std::move(callback));
}

bool GLRenderer::runFrame(const std::function<void()>& update,
                          const std::function<void()>& render) {
	if (!m_window || glfwWindowShouldClose(m_window)) return false;

	glfwPollEvents();

	EventQueue::instance->process();

	const double t = now();
	m_frameMs = (t - m_lastFrameTime) * 1000.0;
	m_lastFrameTime = t;
	if (++m_fpsFrames >= 30) {
		m_fps = m_fpsFrames / (t - m_fpsWindowStart);
		m_fpsWindowStart = t;
		m_fpsFrames = 0;
	}

	glfwGetFramebufferSize(m_window, &m_width, &m_height);
	if (m_width == 0 || m_height == 0) {
		return true;
	}
	m_camera.setSize(m_width, m_height);
	m_framebuffer.setSize(m_width, m_height);

	const ImGuiIO& io = ImGui::GetIO();
	m_input.guiCapturedMouse = io.WantCaptureMouse;
	m_input.guiCapturedKeyboard = io.WantCaptureKeyboard;

	m_controls.update();
	m_camera.world = m_controls.world;
	m_camera.update();

	if (update) update();

	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	if (render) render();

	ImGui::Render();

	blitToBackbuffer();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

	glfwSwapBuffers(m_window);
	return true;
}

void GLRenderer::blitToBackbuffer() {
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glViewport(0, 0, m_width, m_height);
	glClearColor(0.05f, 0.05f, 0.07f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	if (m_framebuffer.fbo()) {
		glBlitNamedFramebuffer(m_framebuffer.fbo(), 0,
		                       0, 0, m_framebuffer.width(), m_framebuffer.height(),
		                       0, 0, m_width, m_height,
		                       GL_COLOR_BUFFER_BIT, GL_NEAREST);
	}
}

void GLRenderer::keyCallback(GLFWwindow* window, int key, int,
                             int action, int) {
	GLRenderer* self = from(window);
	if (!self) return;
	if (key >= 0 && key < Input::kNumKeys) {
		if (action == GLFW_PRESS) self->m_input.keys[key] = true;
		if (action == GLFW_RELEASE) self->m_input.keys[key] = false;
	}
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) self->requestClose();
}

void GLRenderer::mouseButtonCallback(GLFWwindow* window, int button, int action,
                                     int) {
	GLRenderer* self = from(window);
	if (!self) return;
	const bool down = action == GLFW_PRESS;
	if (button >= 0 && button < Input::kNumButtons) {
		self->m_input.buttons[button] = down;
	}
	if (self->m_input.guiCapturedMouse && down) return;
	self->m_controls.onMouseButton(button, down);
}

void GLRenderer::cursorPosCallback(GLFWwindow* window, double x, double y) {
	GLRenderer* self = from(window);
	if (!self) return;
	self->m_input.mouseX = x;
	self->m_input.mouseY = y;
	self->m_controls.onMouseMove(self->m_input, x, y);
}

void GLRenderer::scrollCallback(GLFWwindow* window, double, double dy) {
	GLRenderer* self = from(window);
	if (!self) return;
	self->m_controls.onMouseScroll(self->m_input, dy);
}

void GLRenderer::dropCallback(GLFWwindow* window, int count,
                              const char** paths) {
	GLRenderer* self = from(window);
	if (!self) return;
	std::vector<std::string> files;
	files.reserve(static_cast<size_t>(count));
	for (int i = 0; i < count; ++i) files.emplace_back(paths[i]);
	for (auto& cb : self->m_dropCallbacks) cb(files);
}

}
