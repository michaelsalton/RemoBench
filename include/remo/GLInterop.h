#pragma once

#include <cstdint>
#include <string>

#include <cuda.h>

namespace remo {

class GLInterop {
public:
	~GLInterop();

	GLInterop() = default;
	GLInterop(const GLInterop&) = delete;
	GLInterop& operator=(const GLInterop&) = delete;

	bool bind(unsigned int glTexture, int width, int height, std::string* err);

	bool map(CUstream stream, std::string* err);

	void unmap(CUstream stream);

	uint64_t surface() const { return m_surface; }

	int width() const { return m_width; }
	int height() const { return m_height; }

private:
	void unregister();

	CUgraphicsResource m_resource = nullptr;
	CUsurfObject m_surface = 0;
	unsigned int m_glTexture = 0;
	int m_width = 0;
	int m_height = 0;
	bool m_mapped = false;
};

}
