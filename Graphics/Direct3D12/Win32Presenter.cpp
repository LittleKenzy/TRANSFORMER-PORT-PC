#include "Win32Presenter.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

#ifdef _WIN32
#include <d3d10.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <dxgi1_4.h>
#include <dxgi1_5.h>
#endif

namespace tf::ps2 {

#ifdef _WIN32
namespace {

constexpr GLenum kGlClampToEdge = 0x812Fu;
constexpr GLenum kGlRgba8       = 0x8058u;
using WglSwapIntervalExtFn = BOOL (WINAPI *)(int interval);

template <typename T>
void safe_release(T*& ptr) {
    if (ptr) {
        ptr->Release();
        ptr = nullptr;
    }
}

void compute_aspect_rect(int win_w, int win_h, AspectRatioMode mode, int& dst_x, int& dst_y, int& dst_w, int& dst_h) {
    dst_x = 0;
    dst_y = 0;
    dst_w = win_w;
    dst_h = win_h;
    if (mode == AspectRatioMode::StretchToWindow || win_w <= 0 || win_h <= 0) return;

    const double target_aspect = (mode == AspectRatioMode::Widescreen16x9) ? (16.0 / 9.0) : (4.0 / 3.0);
    const double win_aspect = static_cast<double>(win_w) / static_cast<double>(win_h);
    if (win_aspect > target_aspect) {
        dst_h = win_h;
        dst_w = std::clamp(static_cast<int>(std::lround(static_cast<double>(win_h) * target_aspect)), 1, win_w);
        dst_x = (win_w - dst_w) / 2;
    } else {
        dst_w = win_w;
        dst_h = std::clamp(static_cast<int>(std::lround(static_cast<double>(win_w) / target_aspect)), 1, win_h);
        dst_y = (win_h - dst_h) / 2;
    }
}

constexpr const char* kD3d11VsSource = R"(
cbuffer UvRect : register(b0) {
    float4 uv_rect; // xy = offset (u0, v0), zw = scale (u1 - u0, v1 - v0)
};
struct VSOut {
    float4 pos : SV_POSITION;
    float2 uv  : TEXCOORD0;
};
VSOut main(uint id : SV_VertexID) {
    VSOut o;
    float2 uv = float2((id << 1) & 2, id & 2);
    o.uv = uv_rect.xy + uv * uv_rect.zw;
    o.pos = float4(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f, 0.0f, 1.0f);
    return o;
}
)";

constexpr const char* kD3d11PsSource = R"(
Texture2D<float4> g_tex : register(t0);
SamplerState g_samp : register(s0);
float4 main(float4 pos : SV_POSITION, float2 uv : TEXCOORD0) : SV_TARGET {
    float4 c = g_tex.Sample(g_samp, uv);
    return float4(c.rgb, 1.0f);
}
)";

} // namespace

bool Win32Presenter::create_d3d11_rtv() {
    safe_release(m_d3d_rtv);
    if (!m_d3d_device || !m_d3d_swapchain) return false;
    ID3D11Texture2D* backbuf = nullptr;
    HRESULT hr = m_d3d_swapchain->GetBuffer(0, __uuidof(ID3D11Texture2D), reinterpret_cast<void**>(&backbuf));
    if (FAILED(hr) || !backbuf) return false;
    hr = m_d3d_device->CreateRenderTargetView(backbuf, nullptr, &m_d3d_rtv);
    backbuf->Release();
    return SUCCEEDED(hr) && m_d3d_rtv != nullptr;
}

bool Win32Presenter::ensure_d3d11_texture(u32 fb_width, u32 fb_height) {
    if (m_d3d_tex && m_d3d_srv && m_d3d_tex_w == fb_width && m_d3d_tex_h == fb_height) {
        return true;
    }
    safe_release(m_d3d_srv);
    safe_release(m_d3d_tex);
    m_d3d_tex_w = 0;
    m_d3d_tex_h = 0;
    if (!m_d3d_device || fb_width == 0 || fb_height == 0) return false;

    D3D11_TEXTURE2D_DESC td{};
    td.Width = fb_width;
    td.Height = fb_height;
    td.MipLevels = 1;
    td.ArraySize = 1;
    td.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    td.SampleDesc.Count = 1;
    td.Usage = D3D11_USAGE_DYNAMIC;
    td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    td.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

    if (FAILED(m_d3d_device->CreateTexture2D(&td, nullptr, &m_d3d_tex)) || !m_d3d_tex) {
        return false;
    }
    if (FAILED(m_d3d_device->CreateShaderResourceView(m_d3d_tex, nullptr, &m_d3d_srv)) || !m_d3d_srv) {
        safe_release(m_d3d_tex);
        return false;
    }
    m_d3d_tex_w = fb_width;
    m_d3d_tex_h = fb_height;
    return true;
}

bool Win32Presenter::resize_d3d11_if_needed(u32 cw, u32 ch) {
    if (!m_d3d_swapchain || !m_d3d_ctx || cw == 0 || ch == 0) return false;
    if (cw == m_d3d_bb_w && ch == m_d3d_bb_h && m_d3d_rtv != nullptr) return true;

    m_d3d_ctx->OMSetRenderTargets(0, nullptr, nullptr);
    safe_release(m_d3d_rtv);

    HRESULT hr = m_d3d_swapchain->ResizeBuffers(2, cw, ch, DXGI_FORMAT_R8G8B8A8_UNORM, m_d3d_sc_flags);
    if (FAILED(hr)) {
        return false;
    }
    m_d3d_bb_w = cw;
    m_d3d_bb_h = ch;
    return create_d3d11_rtv();
}

bool Win32Presenter::init_d3d11() {
    if (!m_hwnd) return false;

    u32 cw = std::max<u32>(1u, m_win_width);
    u32 ch = std::max<u32>(1u, m_win_height);
    RECT rc{};
    if (GetClientRect(m_hwnd, &rc)) {
        if (rc.right > rc.left) cw = static_cast<u32>(rc.right - rc.left);
        if (rc.bottom > rc.top) ch = static_cast<u32>(rc.bottom - rc.top);
    }

    static constexpr D3D_FEATURE_LEVEL kLevels[] = {
        D3D_FEATURE_LEVEL_11_1,
        D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_10_1,
        D3D_FEATURE_LEVEL_10_0
    };
    D3D_FEATURE_LEVEL chosen = D3D_FEATURE_LEVEL_11_0;
    HRESULT hr = D3D11CreateDevice(
        nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
        D3D11_CREATE_DEVICE_BGRA_SUPPORT,
        kLevels, static_cast<UINT>(std::size(kLevels)),
        D3D11_SDK_VERSION,
        &m_d3d_device, &chosen, &m_d3d_ctx);
    if (FAILED(hr) || !m_d3d_device || !m_d3d_ctx) {
        // Retry without 11_1 on older Windows runtimes
        hr = D3D11CreateDevice(
            nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
            D3D11_CREATE_DEVICE_BGRA_SUPPORT,
            kLevels + 1, static_cast<UINT>(std::size(kLevels) - 1),
            D3D11_SDK_VERSION,
            &m_d3d_device, &chosen, &m_d3d_ctx);
        if (FAILED(hr) || !m_d3d_device || !m_d3d_ctx) {
            shutdown_d3d11();
            return false;
        }
    }

    // Enable D3D11/DXGI internal immediate-context multithread protection so unlocked
    // IDXGISwapChain::Present() / Present(TEST) on the UI thread NEVER races concurrent
    // D3D11GsRenderer ID3D11DeviceContext calls on the EE thread inside the GPU UMD driver.
    {
        ID3D10Multithread* mt = nullptr;
        if (FAILED(m_d3d_ctx->QueryInterface(__uuidof(ID3D10Multithread), reinterpret_cast<void**>(&mt))) || !mt) {
            m_d3d_device->QueryInterface(__uuidof(ID3D10Multithread), reinterpret_cast<void**>(&mt));
        }
        if (mt) {
            mt->SetMultithreadProtected(TRUE);
            mt->Release();
        }
    }
    m_d3d_device_lost.store(false, std::memory_order_release);

    IDXGIDevice1* dxgi_dev = nullptr;
    if (FAILED(m_d3d_device->QueryInterface(__uuidof(IDXGIDevice1), reinterpret_cast<void**>(&dxgi_dev))) || !dxgi_dev) {
        shutdown_d3d11();
        return false;
    }
    dxgi_dev->SetMaximumFrameLatency(1);

    IDXGIAdapter* adapter = nullptr;
    hr = dxgi_dev->GetAdapter(&adapter);
    dxgi_dev->Release();
    if (FAILED(hr) || !adapter) {
        shutdown_d3d11();
        return false;
    }

    IDXGIFactory2* factory2 = nullptr;
    hr = adapter->GetParent(__uuidof(IDXGIFactory2), reinterpret_cast<void**>(&factory2));
    adapter->Release();
    if (FAILED(hr) || !factory2) {
        shutdown_d3d11();
        return false;
    }

    m_d3d_allow_tearing = false;
    IDXGIFactory5* factory5 = nullptr;
    if (SUCCEEDED(factory2->QueryInterface(__uuidof(IDXGIFactory5), reinterpret_cast<void**>(&factory5))) && factory5) {
        BOOL tearing = FALSE;
        if (SUCCEEDED(factory5->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, &tearing, sizeof(tearing))) && tearing) {
            m_d3d_allow_tearing = true;
        }
        factory5->Release();
    }

    DXGI_SWAP_CHAIN_DESC1 sc{};
    sc.Width = cw;
    sc.Height = ch;
    sc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sc.SampleDesc.Count = 1;
    sc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sc.BufferCount = 2;
    sc.Scaling = DXGI_SCALING_STRETCH;
    sc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    sc.AlphaMode = DXGI_ALPHA_MODE_IGNORE;
    sc.Flags = static_cast<UINT>(DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT) |
               (m_d3d_allow_tearing ? static_cast<UINT>(DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING) : 0u);

    hr = factory2->CreateSwapChainForHwnd(m_d3d_device, m_hwnd, &sc, nullptr, nullptr, &m_d3d_swapchain);
    if (FAILED(hr) || !m_d3d_swapchain) {
        // Fallback: FLIP_DISCARD without ALLOW_TEARING
        m_d3d_allow_tearing = false;
        sc.Flags = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT;
        hr = factory2->CreateSwapChainForHwnd(m_d3d_device, m_hwnd, &sc, nullptr, nullptr, &m_d3d_swapchain);
    }
    if (FAILED(hr) || !m_d3d_swapchain) {
        // Fallback: FLIP_SEQUENTIAL without waitable object
        sc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;
        sc.Flags = 0;
        hr = factory2->CreateSwapChainForHwnd(m_d3d_device, m_hwnd, &sc, nullptr, nullptr, &m_d3d_swapchain);
    }
    factory2->MakeWindowAssociation(m_hwnd, DXGI_MWA_NO_ALT_ENTER);
    factory2->Release();

    if (FAILED(hr) || !m_d3d_swapchain) {
        shutdown_d3d11();
        return false;
    }
    m_d3d_sc_flags = sc.Flags;
    m_d3d_bb_w = cw;
    m_d3d_bb_h = ch;

    if ((m_d3d_sc_flags & DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT) != 0) {
        IDXGISwapChain2* sc2 = nullptr;
        if (SUCCEEDED(m_d3d_swapchain->QueryInterface(__uuidof(IDXGISwapChain2), reinterpret_cast<void**>(&sc2))) && sc2) {
            sc2->SetMaximumFrameLatency(1);
            m_d3d_latency_waitable = sc2->GetFrameLatencyWaitableObject();
            sc2->Release();
        }
    }

    if (!create_d3d11_rtv()) {
        shutdown_d3d11();
        return false;
    }

    // Compile minimal fullscreen-triangle VS & PS
    ID3DBlob* vs_blob = nullptr;
    ID3DBlob* ps_blob = nullptr;
    ID3DBlob* err_blob = nullptr;
    hr = D3DCompile(kD3d11VsSource, std::strlen(kD3d11VsSource), nullptr, nullptr, nullptr,
                    "main", "vs_4_0", 0, 0, &vs_blob, &err_blob);
    safe_release(err_blob);
    if (FAILED(hr) || !vs_blob) {
        shutdown_d3d11();
        return false;
    }
    hr = D3DCompile(kD3d11PsSource, std::strlen(kD3d11PsSource), nullptr, nullptr, nullptr,
                    "main", "ps_4_0", 0, 0, &ps_blob, &err_blob);
    safe_release(err_blob);
    if (FAILED(hr) || !ps_blob) {
        safe_release(vs_blob);
        shutdown_d3d11();
        return false;
    }

    const HRESULT hr_vs = m_d3d_device->CreateVertexShader(vs_blob->GetBufferPointer(), vs_blob->GetBufferSize(), nullptr, &m_d3d_vs);
    const HRESULT hr_ps = m_d3d_device->CreatePixelShader(ps_blob->GetBufferPointer(), ps_blob->GetBufferSize(), nullptr, &m_d3d_ps);
    safe_release(vs_blob);
    safe_release(ps_blob);
    if (FAILED(hr_vs) || FAILED(hr_ps) || !m_d3d_vs || !m_d3d_ps) {
        shutdown_d3d11();
        return false;
    }

    D3D11_SAMPLER_DESC sd{};
    sd.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
    sd.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
    sd.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    sd.ComparisonFunc = D3D11_COMPARISON_NEVER;
    sd.MaxLOD = D3D11_FLOAT32_MAX;
    sd.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    if (FAILED(m_d3d_device->CreateSamplerState(&sd, &m_d3d_sampler_linear))) {
        shutdown_d3d11();
        return false;
    }
    sd.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
    if (FAILED(m_d3d_device->CreateSamplerState(&sd, &m_d3d_sampler_point))) {
        shutdown_d3d11();
        return false;
    }

    D3D11_RASTERIZER_DESC rd{};
    rd.FillMode = D3D11_FILL_SOLID;
    rd.CullMode = D3D11_CULL_NONE;
    rd.DepthClipEnable = FALSE;
    if (FAILED(m_d3d_device->CreateRasterizerState(&rd, &m_d3d_rs))) {
        shutdown_d3d11();
        return false;
    }

    D3D11_BLEND_DESC bd{};
    bd.RenderTarget[0].BlendEnable = FALSE;
    bd.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
    if (FAILED(m_d3d_device->CreateBlendState(&bd, &m_d3d_bs))) {
        shutdown_d3d11();
        return false;
    }

    D3D11_BUFFER_DESC cbd{};
    cbd.ByteWidth = 16u;
    cbd.Usage = D3D11_USAGE_DYNAMIC;
    cbd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    cbd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    static constexpr float kDefaultUv[4] = {0.0f, 0.0f, 1.0f, 1.0f};
    D3D11_SUBRESOURCE_DATA init_cb{};
    init_cb.pSysMem = kDefaultUv;
    if (FAILED(m_d3d_device->CreateBuffer(&cbd, &init_cb, &m_d3d_uv_cb)) || !m_d3d_uv_cb) {
        shutdown_d3d11();
        return false;
    }

    m_d3d_ready = true;
    return true;
}

void Win32Presenter::shutdown_d3d11() {
    std::lock_guard<std::mutex> lk(m_d3d_ctx_mtx);
    if (m_d3d_ctx) {
        m_d3d_ctx->ClearState();
        m_d3d_ctx->Flush();
    }
    if (m_d3d_latency_waitable) {
        CloseHandle(m_d3d_latency_waitable);
        m_d3d_latency_waitable = nullptr;
    }
    safe_release(m_d3d_uv_cb);
    safe_release(m_d3d_bs);
    safe_release(m_d3d_rs);
    safe_release(m_d3d_sampler_point);
    safe_release(m_d3d_sampler_linear);
    safe_release(m_d3d_ps);
    safe_release(m_d3d_vs);
    safe_release(m_d3d_srv);
    safe_release(m_d3d_tex);
    safe_release(m_d3d_rtv);
    safe_release(m_d3d_swapchain);
    safe_release(m_d3d_ctx);
    safe_release(m_d3d_device);
    m_d3d_bb_w = 0;
    m_d3d_bb_h = 0;
    m_d3d_tex_w = 0;
    m_d3d_tex_h = 0;
    m_d3d_sc_flags = 0;
    m_d3d_allow_tearing = false;
    m_d3d_occluded = false;
    m_d3d_latency_acquired = false;
    m_d3d_last_occluded_probe_ms = 0;
    m_d3d_device_lost.store(false, std::memory_order_release);
    m_d3d_ready = false;
}

void Win32Presenter::on_window_activated() {
    m_d3d_occluded = false;
    m_d3d_last_occluded_probe_ms = 0;
}

bool Win32Presenter::is_window_minimized() const {
    if (!m_hwnd || !IsWindow(m_hwnd)) return true;
    HWND root = GetAncestor(m_hwnd, GA_ROOT);
    if (root && IsIconic(root)) return true;
    return IsIconic(m_hwnd) != FALSE;
}

bool Win32Presenter::check_d3d11_occluded_and_ready() {
    if (!m_d3d_ready || m_d3d_device_lost.load(std::memory_order_acquire) || !m_d3d_swapchain || is_window_minimized()) {
        return false;
    }
    if (!m_d3d_occluded) return true;

    const u64 now_ms = static_cast<u64>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
    if (now_ms >= m_d3d_last_occluded_probe_ms && (now_ms - m_d3d_last_occluded_probe_ms) < 50u) {
        return false;
    }
    m_d3d_last_occluded_probe_ms = now_ms;

    const HRESULT hr = m_d3d_swapchain->Present(0, DXGI_PRESENT_TEST);
    if (hr == DXGI_STATUS_OCCLUDED) {
        return false;
    }
    if (hr == DXGI_ERROR_DEVICE_REMOVED || hr == DXGI_ERROR_DEVICE_RESET) {
        // Mark device lost without freeing shared ID3D11Device/Context while EE thread may still reference them
        m_d3d_device_lost.store(true, std::memory_order_release);
        return false;
    }
    m_d3d_occluded = false;
    return true;
}

namespace {
struct UiCtxTryLock {
    std::mutex& mtx;
    std::atomic<u32>& waiters;
    bool locked = false;

    UiCtxTryLock(std::mutex& m, std::atomic<u32>& w, u32 max_wait_ms) : mtx(m), waiters(w) {
        waiters.fetch_add(1u, std::memory_order_release);
        if (mtx.try_lock()) {
            locked = true;
            return;
        }
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(max_wait_ms);
        while (std::chrono::steady_clock::now() < deadline) {
            SwitchToThread();
            if (mtx.try_lock()) {
                locked = true;
                return;
            }
        }
    }
    ~UiCtxTryLock() {
        if (locked) {
            mtx.unlock();
        }
        waiters.fetch_sub(1u, std::memory_order_release);
    }
};
} // namespace

void Win32Presenter::clear_black_d3d11() {
    if (!check_d3d11_occluded_and_ready()) return;
    u32 cw = std::max<u32>(1u, m_win_width);
    u32 ch = std::max<u32>(1u, m_win_height);
    RECT rc{};
    if (GetClientRect(m_hwnd, &rc)) {
        if (rc.right > rc.left) cw = static_cast<u32>(rc.right - rc.left);
        if (rc.bottom > rc.top) ch = static_cast<u32>(rc.bottom - rc.top);
    }
    if (cw == 0 || ch == 0) return;

    // NEVER hold m_d3d_ctx_mtx across WaitForSingleObjectEx or Present!
    if (m_d3d_latency_waitable && !m_d3d_latency_acquired) {
        if (WaitForSingleObjectEx(m_d3d_latency_waitable, 4, TRUE) != WAIT_OBJECT_0) {
            return;
        }
        m_d3d_latency_acquired = true;
    }
    {
        UiCtxTryLock lk(m_d3d_ctx_mtx, m_d3d_ui_waiters, 4u);
        if (!lk.locked || !m_d3d_ready) return;
        if (!resize_d3d11_if_needed(cw, ch)) return;
        static constexpr float kBlack[4] = {0.0f, 0.0f, 0.0f, 1.0f};
        m_d3d_ctx->OMSetRenderTargets(1, &m_d3d_rtv, nullptr);
        m_d3d_ctx->ClearRenderTargetView(m_d3d_rtv, kBlack);
        m_d3d_ctx->OMSetRenderTargets(0, nullptr, nullptr);
    }
    const UINT sync = m_settings.vsync ? 1u : 0u;
    const UINT flags = (!m_settings.vsync && m_d3d_allow_tearing) ? DXGI_PRESENT_ALLOW_TEARING : 0u;
    const HRESULT hr = m_d3d_swapchain->Present(sync, flags);
    m_d3d_latency_acquired = false;
    if (hr == DXGI_STATUS_OCCLUDED) {
        m_d3d_occluded = true;
        m_d3d_last_occluded_probe_ms = static_cast<u64>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
    } else if (hr == DXGI_ERROR_DEVICE_REMOVED || hr == DXGI_ERROR_DEVICE_RESET) {
        m_d3d_device_lost.store(true, std::memory_order_release);
    }
}

bool Win32Presenter::present_d3d11_srv(ID3D11ShaderResourceView* srv,
                                       float u0, float v0, float u1, float v1,
                                       u32 /*src_w*/, u32 /*src_h*/) {
    if (!srv || !check_d3d11_occluded_and_ready()) return false;

    int win_w = static_cast<int>(m_win_width);
    int win_h = static_cast<int>(m_win_height);
    RECT client_rc{};
    if (GetClientRect(m_hwnd, &client_rc)) {
        const int cw = client_rc.right - client_rc.left;
        const int ch = client_rc.bottom - client_rc.top;
        if (cw <= 0 || ch <= 0) return false;
        win_w = cw;
        win_h = ch;
    }
    if (win_w <= 0 || win_h <= 0) return false;

    // Hardware-enforced 1-frame maximum queue: bounded 2ms wait BEFORE locking m_d3d_ctx_mtx.
    // Track m_d3d_latency_acquired so a try_lock bailout never leaks the waitable semaphore token!
    if (m_d3d_latency_waitable && !m_d3d_latency_acquired) {
        if (WaitForSingleObjectEx(m_d3d_latency_waitable, 2, TRUE) != WAIT_OBJECT_0) {
            return false;
        }
        m_d3d_latency_acquired = true;
    }

    int dst_x = 0, dst_y = 0, dst_w = win_w, dst_h = win_h;
    compute_aspect_rect(win_w, win_h, m_settings.aspect_ratio, dst_x, dst_y, dst_w, dst_h);

    {
        UiCtxTryLock lk(m_d3d_ctx_mtx, m_d3d_ui_waiters, 2u);
        if (!lk.locked || !m_d3d_ready || !m_d3d_ctx) return false;
        if (!resize_d3d11_if_needed(static_cast<u32>(win_w), static_cast<u32>(win_h))) return false;

        D3D11_MAPPED_SUBRESOURCE cb_map{};
        if (SUCCEEDED(m_d3d_ctx->Map(m_d3d_uv_cb, 0, D3D11_MAP_WRITE_DISCARD, 0, &cb_map))) {
            float uv_rect[4] = {u0, v0, u1 - u0, v1 - v0};
            std::memcpy(cb_map.pData, uv_rect, sizeof(uv_rect));
            m_d3d_ctx->Unmap(m_d3d_uv_cb, 0);
        }

        m_d3d_ctx->OMSetRenderTargets(1, &m_d3d_rtv, nullptr);
        m_d3d_ctx->OMSetDepthStencilState(nullptr, 0);
        if (dst_x > 0 || dst_y > 0) {
            static constexpr float kBlack[4] = {0.0f, 0.0f, 0.0f, 1.0f};
            m_d3d_ctx->ClearRenderTargetView(m_d3d_rtv, kBlack);
        }

        D3D11_VIEWPORT vp{};
        vp.TopLeftX = static_cast<float>(dst_x);
        vp.TopLeftY = static_cast<float>(dst_y);
        vp.Width = static_cast<float>(dst_w);
        vp.Height = static_cast<float>(dst_h);
        vp.MinDepth = 0.0f;
        vp.MaxDepth = 1.0f;
        m_d3d_ctx->RSSetViewports(1, &vp);
        m_d3d_ctx->RSSetState(m_d3d_rs);
        static constexpr float kBlendFactor[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        m_d3d_ctx->OMSetBlendState(m_d3d_bs, kBlendFactor, 0xFFFFFFFFu);

        m_d3d_ctx->IASetInputLayout(nullptr);
        m_d3d_ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        m_d3d_ctx->VSSetShader(m_d3d_vs, nullptr, 0);
        m_d3d_ctx->VSSetConstantBuffers(0, 1, &m_d3d_uv_cb);
        m_d3d_ctx->PSSetShader(m_d3d_ps, nullptr, 0);
        m_d3d_ctx->PSSetShaderResources(0, 1, &srv);
        ID3D11SamplerState* samp = m_settings.bilinear_filtering ? m_d3d_sampler_linear : m_d3d_sampler_point;
        m_d3d_ctx->PSSetSamplers(0, 1, &samp);
        m_d3d_ctx->Draw(3, 0);

        // Unbind SRV & RTV cleanly so shared D3D11GsRenderer can bind textures/RTVs with zero hazard
        ID3D11ShaderResourceView* null_srv = nullptr;
        m_d3d_ctx->PSSetShaderResources(0, 1, &null_srv);
        m_d3d_ctx->OMSetRenderTargets(0, nullptr, nullptr);
    }

    // NEVER hold m_d3d_ctx_mtx across Present (VSync / DWM wait)!
    const UINT sync_interval = m_settings.vsync ? 1u : 0u;
    const UINT present_flags = (!m_settings.vsync && m_d3d_allow_tearing) ? DXGI_PRESENT_ALLOW_TEARING : 0u;
    const HRESULT hr = m_d3d_swapchain->Present(sync_interval, present_flags);
    m_d3d_latency_acquired = false;
    if (hr == DXGI_STATUS_OCCLUDED) {
        m_d3d_occluded = true;
        m_d3d_last_occluded_probe_ms = static_cast<u64>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
        return false;
    }
    if (hr == DXGI_ERROR_DEVICE_REMOVED || hr == DXGI_ERROR_DEVICE_RESET) {
        m_d3d_device_lost.store(true, std::memory_order_release);
        return false;
    }
    return SUCCEEDED(hr);
}

bool Win32Presenter::present_d3d11(const u32* rgba_pixels, u32 fb_width, u32 fb_height) {
    if (!check_d3d11_occluded_and_ready()) return false;

    int win_w = static_cast<int>(m_win_width);
    int win_h = static_cast<int>(m_win_height);
    RECT client_rc{};
    if (GetClientRect(m_hwnd, &client_rc)) {
        const int cw = client_rc.right - client_rc.left;
        const int ch = client_rc.bottom - client_rc.top;
        if (cw <= 0 || ch <= 0) return false;
        win_w = cw;
        win_h = ch;
    }
    if (win_w <= 0 || win_h <= 0) return false;

    // Hardware-enforced 1-frame maximum queue: bounded 2ms wait BEFORE locking m_d3d_ctx_mtx
    if (m_d3d_latency_waitable && !m_d3d_latency_acquired) {
        if (WaitForSingleObjectEx(m_d3d_latency_waitable, 2, TRUE) != WAIT_OBJECT_0) {
            return false;
        }
        m_d3d_latency_acquired = true;
    }

    int dst_x = 0, dst_y = 0, dst_w = win_w, dst_h = win_h;
    compute_aspect_rect(win_w, win_h, m_settings.aspect_ratio, dst_x, dst_y, dst_w, dst_h);

    {
        UiCtxTryLock lk(m_d3d_ctx_mtx, m_d3d_ui_waiters, 2u);
        if (!lk.locked || !m_d3d_ready || !m_d3d_ctx) return false;
        if (!resize_d3d11_if_needed(static_cast<u32>(win_w), static_cast<u32>(win_h))) return false;
        if (!ensure_d3d11_texture(fb_width, fb_height)) return false;

        D3D11_MAPPED_SUBRESOURCE mapped{};
        if (SUCCEEDED(m_d3d_ctx->Map(m_d3d_tex, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped))) {
            const std::size_t row_bytes = static_cast<std::size_t>(fb_width) * 4u;
            if (mapped.RowPitch == row_bytes) {
                std::memcpy(mapped.pData, rgba_pixels, row_bytes * fb_height);
            } else {
                u8* dst = static_cast<u8*>(mapped.pData);
                const u8* src = reinterpret_cast<const u8*>(rgba_pixels);
                for (u32 y = 0; y < fb_height; ++y) {
                    std::memcpy(dst + static_cast<std::size_t>(y) * mapped.RowPitch, src + y * row_bytes, row_bytes);
                }
            }
            m_d3d_ctx->Unmap(m_d3d_tex, 0);
        }

        D3D11_MAPPED_SUBRESOURCE cb_map{};
        if (SUCCEEDED(m_d3d_ctx->Map(m_d3d_uv_cb, 0, D3D11_MAP_WRITE_DISCARD, 0, &cb_map))) {
            static constexpr float kFullUv[4] = {0.0f, 0.0f, 1.0f, 1.0f};
            std::memcpy(cb_map.pData, kFullUv, sizeof(kFullUv));
            m_d3d_ctx->Unmap(m_d3d_uv_cb, 0);
        }

        m_d3d_ctx->OMSetRenderTargets(1, &m_d3d_rtv, nullptr);
        m_d3d_ctx->OMSetDepthStencilState(nullptr, 0);
        if (dst_x > 0 || dst_y > 0) {
            static constexpr float kBlack[4] = {0.0f, 0.0f, 0.0f, 1.0f};
            m_d3d_ctx->ClearRenderTargetView(m_d3d_rtv, kBlack);
        }

        D3D11_VIEWPORT vp{};
        vp.TopLeftX = static_cast<float>(dst_x);
        vp.TopLeftY = static_cast<float>(dst_y);
        vp.Width = static_cast<float>(dst_w);
        vp.Height = static_cast<float>(dst_h);
        vp.MinDepth = 0.0f;
        vp.MaxDepth = 1.0f;
        m_d3d_ctx->RSSetViewports(1, &vp);
        m_d3d_ctx->RSSetState(m_d3d_rs);
        static constexpr float kBlendFactor[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        m_d3d_ctx->OMSetBlendState(m_d3d_bs, kBlendFactor, 0xFFFFFFFFu);

        m_d3d_ctx->IASetInputLayout(nullptr);
        m_d3d_ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        m_d3d_ctx->VSSetShader(m_d3d_vs, nullptr, 0);
        m_d3d_ctx->VSSetConstantBuffers(0, 1, &m_d3d_uv_cb);
        m_d3d_ctx->PSSetShader(m_d3d_ps, nullptr, 0);
        m_d3d_ctx->PSSetShaderResources(0, 1, &m_d3d_srv);
        ID3D11SamplerState* samp = m_settings.bilinear_filtering ? m_d3d_sampler_linear : m_d3d_sampler_point;
        m_d3d_ctx->PSSetSamplers(0, 1, &samp);
        m_d3d_ctx->Draw(3, 0);

        // Unbind SRV & RTV cleanly
        ID3D11ShaderResourceView* null_srv = nullptr;
        m_d3d_ctx->PSSetShaderResources(0, 1, &null_srv);
        m_d3d_ctx->OMSetRenderTargets(0, nullptr, nullptr);
    }

    const UINT sync_interval = m_settings.vsync ? 1u : 0u;
    const UINT present_flags = (!m_settings.vsync && m_d3d_allow_tearing) ? DXGI_PRESENT_ALLOW_TEARING : 0u;
    const HRESULT hr = m_d3d_swapchain->Present(sync_interval, present_flags);
    m_d3d_latency_acquired = false;
    if (hr == DXGI_STATUS_OCCLUDED) {
        m_d3d_occluded = true;
        m_d3d_last_occluded_probe_ms = static_cast<u64>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
        return false;
    }
    if (hr == DXGI_ERROR_DEVICE_REMOVED || hr == DXGI_ERROR_DEVICE_RESET) {
        // Mark device lost safely without freeing shared ID3D11Device/Context under the EE thread
        m_d3d_device_lost.store(true, std::memory_order_release);
        return false;
    }
    return SUCCEEDED(hr);
}

bool Win32Presenter::init_wgl() {
    if (!m_hwnd) return false;
    m_gl_dc = GetDC(m_hwnd);
    if (!m_gl_dc) return false;

    PIXELFORMATDESCRIPTOR pfd{};
    pfd.nSize = sizeof(pfd);
    pfd.nVersion = 1;
    pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
    pfd.iPixelType = PFD_TYPE_RGBA;
    pfd.cColorBits = 32;
    pfd.cAlphaBits = 8;
    pfd.iLayerType = PFD_MAIN_PLANE;

    int pf = ChoosePixelFormat(m_gl_dc, &pfd);
    if (pf == 0 || !SetPixelFormat(m_gl_dc, pf, &pfd)) {
        ReleaseDC(m_hwnd, m_gl_dc);
        m_gl_dc = nullptr;
        return false;
    }

    m_gl_rc = wglCreateContext(m_gl_dc);
    if (!m_gl_rc || !wglMakeCurrent(m_gl_dc, m_gl_rc)) {
        if (m_gl_rc) { wglDeleteContext(m_gl_rc); m_gl_rc = nullptr; }
        ReleaseDC(m_hwnd, m_gl_dc);
        m_gl_dc = nullptr;
        return false;
    }

    m_wgl_swap_interval = reinterpret_cast<void*>(wglGetProcAddress("wglSwapIntervalEXT"));
    m_gl_has_swap_ctrl = (m_wgl_swap_interval != nullptr);

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glDisable(GL_BLEND);
    glDisable(GL_LIGHTING);
    glEnable(GL_TEXTURE_2D);

    glGenTextures(1, &m_gl_tex);
    glBindTexture(GL_TEXTURE_2D, m_gl_tex);
    m_gl_last_bilinear = m_settings.bilinear_filtering;
    const GLint filter = m_gl_last_bilinear ? GL_LINEAR : GL_NEAREST;
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, kGlClampToEdge);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, kGlClampToEdge);
    glBindTexture(GL_TEXTURE_2D, 0);

    m_gl_tex_w = 0;
    m_gl_tex_h = 0;
    m_gl_ready = true;
    apply_wgl_vsync();
    return true;
}

void Win32Presenter::apply_wgl_vsync() {
    if (!m_gl_ready || !m_wgl_swap_interval) return;
    auto fn = reinterpret_cast<WglSwapIntervalExtFn>(m_wgl_swap_interval);
    fn(m_settings.vsync ? 1 : 0);
    m_gl_last_vsync = m_settings.vsync;
}

bool Win32Presenter::ensure_wgl_current() {
    if (!m_gl_ready || !m_gl_dc || !m_gl_rc) return false;
    if (wglGetCurrentContext() == m_gl_rc && wglGetCurrentDC() == m_gl_dc) {
        return true;
    }
    return wglMakeCurrent(m_gl_dc, m_gl_rc) != FALSE;
}
#endif

bool Win32Presenter::initialize(void* native_window, u32 width, u32 height, const GraphicsSettings& settings) {
    shutdown();
#ifdef _WIN32
    m_hwnd = static_cast<HWND>(native_window);
#else
    (void)native_window;
#endif
    m_win_width = width;
    m_win_height = height;
    m_settings = settings;
#ifdef _WIN32
    if (!init_d3d11()) {
        init_wgl();
    }
#endif
    return true;
}

void Win32Presenter::resize(u32 width, u32 height) {
    // Deferred resize: NEVER lock m_d3d_ctx_mtx or call IDXGISwapChain::ResizeBuffers inside WM_SIZE!
    // resize_d3d11_if_needed() applies the new dimensions safely in the main-loop present path.
    if (width > 0 && height > 0) {
        m_win_width = width;
        m_win_height = height;
    }
}

void Win32Presenter::update_settings(const GraphicsSettings& settings) {
    m_settings = settings;
#ifdef _WIN32
    if (!m_d3d_ready && m_gl_ready && m_settings.vsync != m_gl_last_vsync) {
        if (ensure_wgl_current()) {
            apply_wgl_vsync();
        }
    }
#endif
}

void Win32Presenter::clear_black() {
#ifdef _WIN32
    if (!m_hwnd) return;
    if (m_d3d_ready) {
        clear_black_d3d11();
        return;
    }
    if (ensure_wgl_current()) {
        RECT client_rc{};
        int cw = static_cast<int>(m_win_width);
        int ch = static_cast<int>(m_win_height);
        if (GetClientRect(m_hwnd, &client_rc)) {
            if (client_rc.right > client_rc.left) cw = client_rc.right - client_rc.left;
            if (client_rc.bottom > client_rc.top) ch = client_rc.bottom - client_rc.top;
        }
        glViewport(0, 0, cw, ch);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        SwapBuffers(m_gl_dc);
        glFinish();
        return;
    }
    RECT rc{};
    if (!GetClientRect(m_hwnd, &rc)) return;
    HDC hdc = GetDC(m_hwnd);
    if (hdc) {
        FillRect(hdc, &rc, static_cast<HBRUSH>(GetStockObject(BLACK_BRUSH)));
        ReleaseDC(m_hwnd, hdc);
    }
#endif
}

#ifdef _WIN32
void Win32Presenter::present_rgba_to_dc(HDC hdc, const u32* rgba_pixels, u32 fb_width, u32 fb_height) {
    if (!hdc || !rgba_pixels || fb_width == 0 || fb_height == 0) return;

    int win_w = static_cast<int>(m_win_width);
    int win_h = static_cast<int>(m_win_height);
    if (m_hwnd) {
        RECT client_rc{};
        if (GetClientRect(m_hwnd, &client_rc)) {
            const int cw = client_rc.right - client_rc.left;
            const int ch = client_rc.bottom - client_rc.top;
            if (cw > 0 && ch > 0) {
                win_w = cw;
                win_h = ch;
            }
        }
    }
    if (win_w <= 0 || win_h <= 0) return;

    const std::size_t pixel_count = static_cast<std::size_t>(fb_width) * fb_height;
    if (m_bgra_scratch.size() != pixel_count) {
        m_bgra_scratch.resize(pixel_count);
    }
    for (std::size_t i = 0; i < pixel_count; ++i) {
        const u32 px = rgba_pixels[i];
        const u32 r = px & 0xFFu;
        const u32 g = (px >> 8) & 0xFFu;
        const u32 b = (px >> 16) & 0xFFu;
        m_bgra_scratch[i] = b | (g << 8) | (r << 16) | 0xFF000000u;
    }

    int dst_x = 0, dst_y = 0, dst_w = win_w, dst_h = win_h;
    compute_aspect_rect(win_w, win_h, m_settings.aspect_ratio, dst_x, dst_y, dst_w, dst_h);

    BITMAPINFO bmi{};
    bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmi.bmiHeader.biWidth = static_cast<LONG>(fb_width);
    bmi.bmiHeader.biHeight = -static_cast<LONG>(fb_height);
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = 32;
    bmi.bmiHeader.biCompression = BI_RGB;

    HBRUSH black = static_cast<HBRUSH>(GetStockObject(BLACK_BRUSH));
    if (dst_x > 0) {
        RECT left_bar{0, 0, dst_x, win_h};
        RECT right_bar{dst_x + dst_w, 0, win_w, win_h};
        FillRect(hdc, &left_bar, black);
        FillRect(hdc, &right_bar, black);
    }
    if (dst_y > 0) {
        RECT top_bar{0, 0, win_w, dst_y};
        RECT bottom_bar{0, dst_y + dst_h, win_w, win_h};
        FillRect(hdc, &top_bar, black);
        FillRect(hdc, &bottom_bar, black);
    }

    if (m_settings.bilinear_filtering) {
        SetStretchBltMode(hdc, HALFTONE);
        SetBrushOrgEx(hdc, 0, 0, nullptr);
    } else {
        SetStretchBltMode(hdc, COLORONCOLOR);
    }

    StretchDIBits(
        hdc,
        dst_x, dst_y, dst_w, dst_h,
        0, 0, static_cast<int>(fb_width), static_cast<int>(fb_height),
        m_bgra_scratch.data(), &bmi, DIB_RGB_COLORS, SRCCOPY
    );
}
#endif

void Win32Presenter::present_rgba(const u32* rgba_pixels, u32 fb_width, u32 fb_height) {
#ifdef _WIN32
    if (!m_hwnd || !rgba_pixels || fb_width == 0 || fb_height == 0) return;
    if (m_d3d_ready) {
        present_d3d11(rgba_pixels, fb_width, fb_height);
        return;
    }
    if (ensure_wgl_current()) {
        int win_w = static_cast<int>(m_win_width);
        int win_h = static_cast<int>(m_win_height);
        RECT client_rc{};
        if (GetClientRect(m_hwnd, &client_rc)) {
            const int cw = client_rc.right - client_rc.left;
            const int ch = client_rc.bottom - client_rc.top;
            if (cw > 0 && ch > 0) {
                win_w = cw;
                win_h = ch;
            }
        }
        if (win_w <= 0 || win_h <= 0) return;

        if (m_settings.vsync != m_gl_last_vsync) {
            apply_wgl_vsync();
        }

        int dst_x = 0, dst_y = 0, dst_w = win_w, dst_h = win_h;
        compute_aspect_rect(win_w, win_h, m_settings.aspect_ratio, dst_x, dst_y, dst_w, dst_h);

        if (dst_x > 0 || dst_y > 0) {
            glViewport(0, 0, win_w, win_h);
            glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);
        }

        glBindTexture(GL_TEXTURE_2D, m_gl_tex);
        if (m_settings.bilinear_filtering != m_gl_last_bilinear) {
            m_gl_last_bilinear = m_settings.bilinear_filtering;
            const GLint filter = m_gl_last_bilinear ? GL_LINEAR : GL_NEAREST;
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter);
        }
        glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
        if (fb_width != m_gl_tex_w || fb_height != m_gl_tex_h) {
            glTexImage2D(GL_TEXTURE_2D, 0, kGlRgba8,
                         static_cast<GLsizei>(fb_width), static_cast<GLsizei>(fb_height),
                         0, GL_RGBA, GL_UNSIGNED_BYTE, rgba_pixels);
            m_gl_tex_w = fb_width;
            m_gl_tex_h = fb_height;
        } else {
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                            static_cast<GLsizei>(fb_width), static_cast<GLsizei>(fb_height),
                            GL_RGBA, GL_UNSIGNED_BYTE, rgba_pixels);
        }

        glViewport(dst_x, win_h - dst_y - dst_h, dst_w, dst_h);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
        glBegin(GL_QUADS);
            glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f, -1.0f);
            glTexCoord2f(1.0f, 1.0f); glVertex2f( 1.0f, -1.0f);
            glTexCoord2f(1.0f, 0.0f); glVertex2f( 1.0f,  1.0f);
            glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f,  1.0f);
        glEnd();
        glBindTexture(GL_TEXTURE_2D, 0);

        SwapBuffers(m_gl_dc);
        // Prevent OpenGL ICD drivers from queuing 2-3 pre-rendered frames ahead
        glFinish();
        return;
    }
    HDC hdc = GetDC(m_hwnd);
    if (hdc) {
        present_rgba_to_dc(hdc, rgba_pixels, fb_width, fb_height);
        ReleaseDC(m_hwnd, hdc);
    }
#else
    (void)rgba_pixels; (void)fb_width; (void)fb_height;
#endif
}

void Win32Presenter::shutdown() {
#ifdef _WIN32
    shutdown_d3d11();
    if (m_gl_ready && m_gl_dc && m_gl_rc) {
        if (wglMakeCurrent(m_gl_dc, m_gl_rc)) {
            if (m_gl_tex != 0) {
                glDeleteTextures(1, &m_gl_tex);
                m_gl_tex = 0;
            }
            wglMakeCurrent(nullptr, nullptr);
        }
    }
    if (m_gl_rc) {
        wglDeleteContext(m_gl_rc);
        m_gl_rc = nullptr;
    }
    if (m_gl_dc && m_hwnd) {
        ReleaseDC(m_hwnd, m_gl_dc);
    }
    m_gl_dc = nullptr;
    m_gl_ready = false;
    m_gl_tex_w = 0;
    m_gl_tex_h = 0;
    m_gl_has_swap_ctrl = false;
    m_wgl_swap_interval = nullptr;
    m_hwnd = nullptr;
#endif
    m_bgra_scratch.clear();
}

} // namespace tf::ps2
