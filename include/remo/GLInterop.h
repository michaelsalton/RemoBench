// GL colour texture <-> CUDA surface bridge.
//
// The render kernels do not issue a single GL draw call. They software-rasterise
// into a packed uint64 framebuffer and then surf2Dwrite the result into a GL
// texture, which GL blits to the backbuffer. This class owns that one piece of
// plumbing.
//
// It registers the texture ONCE per resize. Upstream calls
// cuGraphicsGLRegisterImage and cuGraphicsUnregisterResource on EVERY frame
// (main_progressive_octree.cpp:460-464, :533), which is a per-frame driver
// synchronisation for no reason, and leaves two dead `static bool registered`
// locals behind as evidence that someone meant to fix it.

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

	// Point at a GL texture. Cheap and idempotent when the handle and size are
	// unchanged, so it is safe to call once per frame; pass the texture whose
	// dimensions changed after a window resize and it re-registers.
	bool bind(unsigned int glTexture, int width, int height, std::string* err);

	// Map for CUDA access and create the surface object. Call once per frame,
	// before launching a render kernel.
	bool map(CUstream stream, std::string* err);

	// Unmap. GL must not touch the texture between map() and unmap().
	void unmap(CUstream stream);

	// Valid only between map() and unmap(). Passed to kernels as an integer, since
	// HostDeviceCommon.h cannot include CUDA headers.
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

}  // namespace remo
