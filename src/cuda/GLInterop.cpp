#include "remo/GLInterop.h"

#include <GL/glew.h>
#include <cudaGL.h>

#include "remo/CudaCheck.h"

namespace remo {

GLInterop::~GLInterop() { unregister(); }

void GLInterop::unregister() {
	if (m_mapped) {
		if (m_surface) cuSurfObjectDestroy(m_surface);
		m_surface = 0;
		cuGraphicsUnmapResources(1, &m_resource, nullptr);
		m_mapped = false;
	}
	if (m_resource) {
		REMO_CU(cuGraphicsUnregisterResource(m_resource));
		m_resource = nullptr;
	}
	m_glTexture = 0;
	m_width = m_height = 0;
}

bool GLInterop::bind(unsigned int glTexture, int width, int height,
                     std::string* err) {
	if (glTexture == m_glTexture && width == m_width && height == m_height &&
	    m_resource != nullptr) {
		return true;
	}

	unregister();

	if (glTexture == 0 || width <= 0 || height <= 0) {
		if (err) *err = "GLInterop::bind called with an invalid texture";
		return false;
	}

	const CUresult r = REMO_CU(cuGraphicsGLRegisterImage(
		&m_resource, glTexture, GL_TEXTURE_2D,
		CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD));
	if (r != CUDA_SUCCESS) {
		if (err) {
			*err = std::string("cuGraphicsGLRegisterImage failed: ") + cuErrorName(r);
		}
		m_resource = nullptr;
		return false;
	}

	m_glTexture = glTexture;
	m_width = width;
	m_height = height;
	return true;
}

bool GLInterop::map(CUstream stream, std::string* err) {
	if (!m_resource) {
		if (err) *err = "GLInterop::map before a successful bind";
		return false;
	}
	if (m_mapped) return true;

	if (REMO_CU(cuGraphicsMapResources(1, &m_resource, stream)) != CUDA_SUCCESS) {
		if (err) *err = "cuGraphicsMapResources failed";
		return false;
	}

	CUarray array = nullptr;
	if (REMO_CU(cuGraphicsSubResourceGetMappedArray(&array, m_resource, 0, 0)) !=
	    CUDA_SUCCESS) {
		cuGraphicsUnmapResources(1, &m_resource, stream);
		if (err) *err = "cuGraphicsSubResourceGetMappedArray failed";
		return false;
	}

	CUDA_RESOURCE_DESC desc = {};
	desc.resType = CU_RESOURCE_TYPE_ARRAY;
	desc.res.array.hArray = array;

	CUsurfObject surface = 0;
	if (REMO_CU(cuSurfObjectCreate(&surface, &desc)) != CUDA_SUCCESS) {
		cuGraphicsUnmapResources(1, &m_resource, stream);
		if (err) *err = "cuSurfObjectCreate failed";
		return false;
	}

	m_surface = surface;
	m_mapped = true;
	return true;
}

void GLInterop::unmap(CUstream stream) {
	if (!m_mapped) return;
	if (m_surface) {
		REMO_CU(cuSurfObjectDestroy(m_surface));
		m_surface = 0;
	}
	REMO_CU(cuGraphicsUnmapResources(1, &m_resource, stream));
	m_mapped = false;
}

}
