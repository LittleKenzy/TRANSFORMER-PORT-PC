#pragma once

#include "../../Recompiler/Common/Types.h"
#include <string>

namespace tf::ps2 {

enum class AspectRatioMode {
    Native4x3,
    Widescreen16x9,
    StretchToWindow
};

struct GraphicsSettings {
    u32 internal_resolution_scale = 1; // 1 = native PS2 640x448
    bool vsync = true;
    bool bilinear_filtering = true;
    AspectRatioMode aspect_ratio = AspectRatioMode::Native4x3;
};

// Platform-independent native graphics presenter interface for GS framebuffer output.
class IGraphicsBackend {
public:
    virtual ~IGraphicsBackend() = default;

    virtual const char* backend_name() const = 0;
    virtual bool initialize(void* native_window, u32 width, u32 height, const GraphicsSettings& settings) = 0;
    virtual void present_rgba(const u32* rgba_pixels, u32 fb_width, u32 fb_height) = 0;
    virtual void resize(u32 width, u32 height) = 0;
    virtual void shutdown() = 0;
};

} // namespace tf::ps2
