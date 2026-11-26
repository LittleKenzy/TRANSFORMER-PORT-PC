#pragma once

#include "../Common/GraphicsBackend.h"
#include <atomic>
#include <mutex>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <GL/gl.h>

struct ID3D11Device;
struct ID3D11DeviceContext;
struct IDXGISwapChain1;
struct ID3D11RenderTargetView;
struct ID3D11Texture2D;
struct ID3D11ShaderResourceView;
struct ID3D11Buffer;
struct ID3D11VertexShader;
struct ID3D11PixelShader;
struct ID3D11SamplerState;
struct ID3D11RasterizerState;
struct ID3D11BlendState;
#endif

namespace tf::ps2 {

// Native Windows ultra-low-latency Direct3D 11 Flip-Model presenter
// (DXGI_SWAP_EFFECT_FLIP_DISCARD + FRAME_LATENCY_WAITABLE_OBJECT + ALLOW_TEARING)
// with automatic OpenGL/WGL and GDI DIB fallbacks.
class Win32Presenter final : public IGraphicsBackend {
public:
    Win32Presenter() = default;
    ~Win32Presenter() override { shutdown(); }

    const char* backend_name() const override {
#ifdef _WIN32
        if (m_d3d_ready) return "D3D11 Flip";
        if (m_gl_ready)  return "OpenGL";
        return "GDI";
#else
        return "Win32 Native Framebuffer Presenter";
#endif
    }
    bool initialize(void* native_window, u32 width, u32 height, const GraphicsSettings& settings) override;
    void present_rgba(const u32* rgba_pixels, u32 fb_width, u32 fb_height) override;
    void resize(u32 width, u32 height) override;
    void shutdown() override;

    void update_settings(const GraphicsSettings& settings);
    void clear_black();
    bool is_hardware_accelerated() const {
#ifdef _WIN32
        return m_d3d_ready || m_gl_ready;
#else
        return false;
#endif
    }
    bool supports_tearing() const {
#ifdef _WIN32
        return m_d3d_ready && m_d3d_allow_tearing;
#else
        return false;
#endif
    }
#ifdef _WIN32
    void present_rgba_to_dc(HDC hdc, const u32* rgba_pixels, u32 fb_width, u32 fb_height);

    // Shared D3D11 device/context accessors for zero-copy D3D11GsRenderer integration.
    // Callers sharing the immediate context MUST lock d3d11_context_mutex() around context ops
    // and yield when d3d11_ui_waiters() > 0 so the Win32 UI thread is never starved.
    ID3D11Device* d3d11_device() const { return m_d3d_ready ? m_d3d_device : nullptr; }
    ID3D11DeviceContext* d3d11_context() const { return m_d3d_ready ? m_d3d_ctx : nullptr; }
    std::mutex* d3d11_context_mutex() { return &m_d3d_ctx_mtx; }
    std::atomic<u32>* d3d11_ui_waiters() { return &m_d3d_ui_waiters; }

    // Notify presenter on WM_ACTIVATEAPP / WM_SETFOCUS so occluded probe clears immediately.
    void on_window_activated();

    // Present a GPU-resident D3D11 ShaderResourceView directly to the Flip-Model swapchain
    // using the specified normalized UV crop rectangle [u0,v0]..[u1,v1].
    // Non-blocking: returns false immediately if minimized, occluded, or busy (>2ms).
    bool present_d3d11_srv(ID3D11ShaderResourceView* srv,
                           float u0, float v0, float u1, float v1,
                           u32 src_w, u32 src_h);
#endif

private:
#ifdef _WIN32
    bool init_d3d11();
    void shutdown_d3d11();
    bool create_d3d11_rtv();
    bool ensure_d3d11_texture(u32 fb_width, u32 fb_height);
    bool resize_d3d11_if_needed(u32 cw, u32 ch);
    bool is_window_minimized() const;
    bool check_d3d11_occluded_and_ready();
    bool present_d3d11(const u32* rgba_pixels, u32 fb_width, u32 fb_height);
    void clear_black_d3d11();

    bool init_wgl();
    bool ensure_wgl_current();
    void apply_wgl_vsync();

    HWND  m_hwnd = nullptr;

    // Direct3D 11 Flip-Model state (primary ultra-low-latency path)
    std::mutex                m_d3d_ctx_mtx;
    std::atomic<u32>          m_d3d_ui_waiters{0};
    std::atomic<bool>         m_d3d_device_lost{false};
    bool                      m_d3d_occluded = false;
    bool                      m_d3d_latency_acquired = false;
    u64                       m_d3d_last_occluded_probe_ms = 0;
    ID3D11Device*             m_d3d_device = nullptr;
    ID3D11DeviceContext*      m_d3d_ctx = nullptr;
    IDXGISwapChain1*          m_d3d_swapchain = nullptr;
    ID3D11RenderTargetView*   m_d3d_rtv = nullptr;
    ID3D11Texture2D*          m_d3d_tex = nullptr;
    ID3D11ShaderResourceView* m_d3d_srv = nullptr;
    ID3D11Buffer*             m_d3d_uv_cb = nullptr;
    ID3D11VertexShader*       m_d3d_vs = nullptr;
    ID3D11PixelShader*        m_d3d_ps = nullptr;
    ID3D11SamplerState*       m_d3d_sampler_linear = nullptr;
    ID3D11SamplerState*       m_d3d_sampler_point = nullptr;
    ID3D11RasterizerState*    m_d3d_rs = nullptr;
    ID3D11BlendState*         m_d3d_bs = nullptr;
    HANDLE                    m_d3d_latency_waitable = nullptr;
    u32   m_d3d_bb_w = 0;
    u32   m_d3d_bb_h = 0;
    u32   m_d3d_tex_w = 0;
    u32   m_d3d_tex_h = 0;
    UINT  m_d3d_sc_flags = 0;
    bool  m_d3d_allow_tearing = false;
    bool  m_d3d_ready = false;

    // OpenGL/WGL fallback state
    HDC   m_gl_dc = nullptr;
    HGLRC m_gl_rc = nullptr;
    GLuint m_gl_tex = 0;
    u32   m_gl_tex_w = 0;
    u32   m_gl_tex_h = 0;
    bool  m_gl_ready = false;
    bool  m_gl_last_bilinear = true;
    bool  m_gl_last_vsync = true;
    bool  m_gl_has_swap_ctrl = false;
    void* m_wgl_swap_interval = nullptr;
#endif
    u32 m_win_width = 1280;
    u32 m_win_height = 720;
    GraphicsSettings m_settings{};
    std::vector<u32> m_bgra_scratch;
};

} // namespace tf::ps2
