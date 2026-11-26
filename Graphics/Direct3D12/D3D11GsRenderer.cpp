#include "D3D11GsRenderer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <d3d11.h>
#include <d3dcompiler.h>
#endif

namespace tf::ps2 {

#ifdef _WIN32
namespace {

template <typename T>
void safe_release(T*& ptr) {
    if (ptr) {
        ptr->Release();
        ptr = nullptr;
    }
}

inline u32 psm_page_height_hw(u32 psm) {
    switch (psm) {
        case 0x02u: case 0x0Au: case 0x32u: case 0x3Au: case 0x13u:
            return 64u;
        case 0x14u:
            return 128u;
        default:
            return 32u;
    }
}

inline bool ranges_overlap_vram_hw(u32 a_start, u32 a_len, u32 b_start, u32 b_len) {
    if (a_len == 0u || b_len == 0u) return false;
    if (a_len >= 16384u || b_len >= 16384u) return true;
    const u32 a0 = a_start & 0x3FFFu;
    const u32 b0 = b_start & 0x3FFFu;
    const u32 a1 = a0 + a_len;
    const u32 b1 = b0 + b_len;
    if (a1 <= 16384u && b1 <= 16384u) {
        return (a0 < b1) && (b0 < a1);
    }
    auto check_seg = [](u32 s0, u32 e0, u32 s1, u32 e1) {
        return (s0 < e1) && (s1 < e0);
    };
    const u32 a_hi = std::min<u32>(a1, 16384u);
    const u32 a_wrap = (a1 > 16384u) ? (a1 - 16384u) : 0u;
    const u32 b_hi = std::min<u32>(b1, 16384u);
    const u32 b_wrap = (b1 > 16384u) ? (b1 - 16384u) : 0u;
    if (check_seg(a0, a_hi, b0, b_hi)) return true;
    if (a_wrap > 0u && check_seg(0u, a_wrap, b0, b_hi)) return true;
    if (b_wrap > 0u && check_seg(a0, a_hi, 0u, b_wrap)) return true;
    if (a_wrap > 0u && b_wrap > 0u) return true;
    return false;
}

struct alignas(16) GsDrawConstantBuffer {
    float rt_inv_size[4];   // (2/rt_w, 2/rt_h, rt_w, rt_h)
    float tex_params[4];    // (tw, th, 1/src_tex_w, 1/src_tex_h)
    float clamp_bounds[4];  // (minu, maxu, minv, maxv)
    float misc[4];          // (aref_u8, fix/128.0, texa_ta0/255.0, texa_aem ? 1 : 0)
    u32   flags0[4];        // x=tme, y=tfx, z=tcc, w=bilinear
    u32   flags1[4];        // x=wms, y=wmt, z=eff_atst (1=always/off), w=afail_invert (0=keep_pass, 1=keep_fail)
    u32   flags2[4];        // x=alpha_c_mode (0=As,1=Ad,2=Fix,3=0), y=fba, z=psm24_a, w=date_mode (0=off,1=datm0,2=datm1)
    u32   flags3[4];        // x=premul_c, y=scanmsk, z=tex_is_rt, w=pabe
    float extra[4];         // x=inv_scale (1/scale), y=z_max_f, z=0, w=0
    u32   flags4[4];        // x=tex_mode (0=RGBA,1=PSMT8H,2=PSMT4HL,3=PSMT4HH,4=Z32_RGBA,5=Z16_LO,6=Z16_HI), y/z/w=0
};

constexpr const char* kGsVsSource = R"(
cbuffer GsDrawCB : register(b0) {
    float4 u_rt_inv_size;
    float4 u_tex_params;
    float4 u_clamp_bounds;
    float4 u_misc;
    uint4  u_flags0;
    uint4  u_flags1;
    uint4  u_flags2;
    uint4  u_flags3;
    float4 u_extra;
    uint4  u_flags4;
};

struct VSIn {
    float2 pos   : POSITION;
    float  z     : TEXCOORD0;
    float4 color : COLOR0;
    float3 stq   : TEXCOORD1;
};

struct VSOut {
    float4 pos   : SV_POSITION;
    float4 color : COLOR0;
    noperspective float3 stq : TEXCOORD1;
};

VSOut main(VSIn v) {
    VSOut o;
    float ndc_x = v.pos.x * u_rt_inv_size.x - 1.0f;
    float ndc_y = 1.0f - v.pos.y * u_rt_inv_size.y;
    o.pos = float4(ndc_x, ndc_y, saturate(v.z), 1.0f);
    o.color = v.color;
    o.stq = v.stq;
    return o;
}
)";

constexpr const char* kGsPsCommonBody = R"(
cbuffer GsDrawCB : register(b0) {
    float4 u_rt_inv_size;
    float4 u_tex_params;
    float4 u_clamp_bounds;
    float4 u_misc;
    uint4  u_flags0;
    uint4  u_flags1;
    uint4  u_flags2;
    uint4  u_flags3;
    float4 u_extra;
    uint4  u_flags4;
};

Texture2D<float4> g_tex    : register(t0);
Texture2D<float4> g_dsttex : register(t1);
Texture2D<float4> g_clut   : register(t2);
SamplerState      g_samp   : register(s0);

struct VSOut {
    float4 pos   : SV_POSITION;
    float4 color : COLOR0;
    noperspective float3 stq : TEXCOORD1;
};

struct PSOut {
    float4 color : SV_TARGET0;
    float4 blend : SV_TARGET1;
};

float4 sample_ps2_texture(float3 stq) {
    float inv_q = (abs(stq.z) > 1e-6f) ? (1.0f / stq.z) : 1.0f;
    float2 uv_norm = stq.xy * inv_q;
    float2 texel = uv_norm * u_tex_params.xy;

    uint wms = u_flags1.x;
    uint wmt = u_flags1.y;
    bool use_bilinear = (u_flags0.w != 0u);
    bool tex_is_rt = (u_flags3.z != 0u);
    uint tex_mode = u_flags4.x;

    float4 raw;
    if (wms <= 1u && wmt <= 1u && !tex_is_rt && tex_mode == 0u) {
        return g_tex.Sample(g_samp, uv_norm);
    }

    float tw = u_tex_params.x;
    float th = u_tex_params.y;
    if (wms == 0u) {
        texel.x = texel.x - floor(texel.x / tw) * tw;
    } else if (wms == 1u) {
        texel.x = clamp(texel.x, 0.0f, tw - 0.001f);
    } else if (wms == 2u) {
        float lo = min(u_clamp_bounds.x, min(u_clamp_bounds.y, tw - 1.0f));
        float hi = min(u_clamp_bounds.y, tw - 1.0f);
        if (use_bilinear) texel.x = clamp(texel.x, lo + 0.5f, hi + 0.5f);
        else              texel.x = clamp(floor(texel.x), lo, hi) + 0.5f;
    } else {
        uint ix = ((uint)max(0, (int)floor(texel.x)) & (uint)u_clamp_bounds.x) | (uint)u_clamp_bounds.y;
        texel.x = (float)(ix & ((uint)tw - 1u)) + 0.5f;
    }

    if (wmt == 0u) {
        texel.y = texel.y - floor(texel.y / th) * th;
    } else if (wmt == 1u) {
        texel.y = clamp(texel.y, 0.0f, th - 0.001f);
    } else if (wmt == 2u) {
        float lo = min(u_clamp_bounds.z, min(u_clamp_bounds.w, th - 1.0f));
        float hi = min(u_clamp_bounds.w, th - 1.0f);
        if (use_bilinear) texel.y = clamp(texel.y, lo + 0.5f, hi + 0.5f);
        else              texel.y = clamp(floor(texel.y), lo, hi) + 0.5f;
    } else {
        uint iy = ((uint)max(0, (int)floor(texel.y)) & (uint)u_clamp_bounds.z) | (uint)u_clamp_bounds.w;
        texel.y = (float)(iy & ((uint)th - 1u)) + 0.5f;
    }

    float2 final_uv = texel * u_tex_params.zw;
    if (tex_is_rt || tex_mode != 0u) {
        final_uv = saturate(final_uv);
    }
    raw = g_tex.Sample(g_samp, final_uv);
    if (tex_mode == 0u) return raw;
    if (tex_mode == 1u) { // PSMT8H on GPU RT: alpha byte -> 256-entry CLUT
        uint idx = (uint)floor(saturate(raw.a) * 255.0f + 0.5f) & 255u;
        return g_clut.Load(int3((int)idx, 0, 0));
    }
    if (tex_mode == 2u) { // PSMT4HL on GPU RT
        uint idx = (uint)floor(saturate(raw.a) * 255.0f + 0.5f) & 15u;
        return g_clut.Load(int3((int)idx, 0, 0));
    }
    if (tex_mode == 3u) { // PSMT4HH on GPU RT
        uint idx = ((uint)floor(saturate(raw.a) * 255.0f + 0.5f) >> 4u) & 15u;
        return g_clut.Load(int3((int)idx, 0, 0));
    }
    // GPU Z-buffer read (tex_mode 4=PSMZ32/24 RGBA, 5=PSMZ16 low16, 6=PSMZ16 high16)
    float d = saturate(raw.r);
    uint z_int = (d >= 1.0f && u_extra.y > 16777216.0f)
               ? 0xFFFFFFFFu
               : (uint)floor(d * u_extra.y + 0.5f);
    float b0 = (float)(z_int & 255u) / 255.0f;
    float b1 = (float)((z_int >> 8u) & 255u) / 255.0f;
    float b2 = (float)((z_int >> 16u) & 255u) / 255.0f;
    float b3 = (float)((z_int >> 24u) & 255u) / 255.0f;
    if (tex_mode == 4u) return float4(b0, b1, b2, b3);
    if (tex_mode == 5u) return float4(b0, b1, b0, b1);
    return float4(b2, b3, b2, b3);
}

float4 shade_ps2_color(VSOut inp) {
    float4 col = inp.color;
    if (u_flags0.x != 0u) {
        float4 tex = sample_ps2_texture(inp.stq);
        if (u_flags2.z != 0u) {
            bool rgb_zero = (tex.r < (0.5f / 255.0f) && tex.g < (0.5f / 255.0f) && tex.b < (0.5f / 255.0f));
            tex.a = (u_misc.w > 0.5f && rgb_zero) ? 0.0f : u_misc.z;
        }
        uint tfx = u_flags0.y;
        uint tcc = u_flags0.z;
        if (tfx == 0u) {
            col.rgb = saturate(tex.rgb * inp.color.rgb * (255.0f / 128.0f));
            col.a   = (tcc != 0u) ? saturate(tex.a * inp.color.a * (255.0f / 128.0f)) : inp.color.a;
        } else if (tfx == 1u) {
            col.rgb = tex.rgb;
            col.a   = (tcc != 0u) ? tex.a : inp.color.a;
        } else if (tfx == 2u) {
            col.rgb = saturate(tex.rgb * inp.color.rgb * (255.0f / 128.0f) + inp.color.aaa);
            col.a   = (tcc != 0u) ? saturate(tex.a + inp.color.a) : inp.color.a;
        } else {
            col.rgb = saturate(tex.rgb * inp.color.rgb * (255.0f / 128.0f) + inp.color.aaa);
            col.a   = (tcc != 0u) ? tex.a : inp.color.a;
        }
    }
    return col;
}

PSOut pack_ps2_output(float4 col) {
    float cf = 0.0f;
    uint cmode = u_flags2.x;
    if (cmode == 0u)      cf = col.a * (255.0f / 128.0f);
    else if (cmode == 2u) cf = u_misc.y;
    else                  cf = 0.0f;

    // PABE: if enabled and bit 7 of source alpha is 0 (< 128), suppress alpha blending (effectively cf = 0 or opaque)
    if (u_flags3.w != 0u && (col.a * 255.0f) < 127.5f) {
        // Fallback: no blend modulation when PABE rejects blending
        cf = (u_flags3.x != 0u) ? 1.0f : 1.0f;
    }

    float out_a = col.a;
    if (u_flags2.y != 0u) { // FBA: force MSB of alpha (|= 0x80)
        float a_u8 = floor(out_a * 255.0f + 0.5f);
        if (a_u8 < 128.0f) out_a = (a_u8 + 128.0f) / 255.0f;
    }

    PSOut o;
    if (u_flags3.x != 0u) {
        // Pre-multiply RGB by full PS2 [0..2.0] factor for additive/subtractive modes
        o.color = float4(saturate(col.rgb * cf), out_a);
        o.blend = float4(1.0f, 1.0f, 1.0f, 1.0f);
    } else {
        o.color = float4(col.rgb, out_a);
        float c_sat = saturate(cf);
        o.blend = float4(c_sat, c_sat, c_sat, c_sat);
    }
    return o;
}
)";

constexpr const char* kGsPsOpaqueMain = R"(
PSOut main(VSOut inp) {
    float4 col = shade_ps2_color(inp);
    return pack_ps2_output(col);
}
)";

constexpr const char* kGsPsAlphaMain = R"(
bool eval_alpha_test(float a_norm, uint atst, uint aref) {
    uint a = (uint)floor(a_norm * 255.0f + 0.5f);
    if (atst == 0u) return false;
    if (atst == 1u) return true;
    if (atst == 2u) return a <  aref;
    if (atst == 3u) return a <= aref;
    if (atst == 4u) return a == aref;
    if (atst == 5u) return a >= aref;
    if (atst == 6u) return a >  aref;
    return a != aref;
}

PSOut main(VSOut inp) {
    if (u_flags3.y != 0u) {
        uint py = (uint)floor(inp.pos.y * u_extra.x);
        if (u_flags3.y == 2u && (py & 1u) == 0u) discard;
        if (u_flags3.y == 3u && (py & 1u) != 0u) discard;
    }
    if (u_flags2.w != 0u) {
        float4 dst = g_dsttex.Load(int3((int)inp.pos.x, (int)inp.pos.y, 0));
        uint dst_msb = (dst.a >= (127.5f / 255.0f)) ? 1u : 0u;
        uint want = (u_flags2.w == 2u) ? 1u : 0u;
        if (dst_msb != want) discard;
    }

    float4 col = shade_ps2_color(inp);
    uint atst = u_flags1.z;
    if (atst != 1u) {
        uint aref = (uint)floor(u_misc.x + 0.5f);
        bool apass = eval_alpha_test(col.a, atst, aref);
        if (u_flags1.w != 0u) apass = !apass;
        if (!apass) discard;
    }
    return pack_ps2_output(col);
}
)";

constexpr const char* kSeedVsSource = R"(
struct VSOut {
    float4 pos : SV_POSITION;
    float2 uv  : TEXCOORD0;
};
VSOut main(uint id : SV_VertexID) {
    VSOut o;
    float2 uv = float2((id << 1) & 2, id & 2);
    o.uv = uv;
    o.pos = float4(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f, 0.0f, 1.0f);
    return o;
}
)";

constexpr const char* kSeedPsSource = R"(
Texture2D<float4> g_tex : register(t0);
SamplerState g_samp : register(s0);
float4 main(float4 pos : SV_POSITION, float2 uv : TEXCOORD0) : SV_TARGET {
    return g_tex.Sample(g_samp, uv);
}
)";

} // namespace

bool D3D11GsRenderer::init(ID3D11Device* device, ID3D11DeviceContext* ctx, std::mutex* ctx_mtx,
                           std::atomic<u32>* ui_waiters) {
    shutdown();
    if (!device || !ctx || !ctx_mtx) return false;

    m_device = device;
    m_ctx = ctx;
    m_ctx_mtx = ctx_mtx;
    m_ui_waiters = ui_waiters;
    m_device->AddRef();
    m_ctx->AddRef();

    std::lock_guard<std::mutex> lk(*m_ctx_mtx);

    auto compile_shader = [](const std::string& src, const char* target, ID3DBlob*& out_blob) -> bool {
        ID3DBlob* err = nullptr;
        const HRESULT hr = D3DCompile(src.data(), src.size(), nullptr, nullptr, nullptr,
                                      "main", target, 0, 0, &out_blob, &err);
        safe_release(err);
        return SUCCEEDED(hr) && out_blob != nullptr;
    };

    ID3DBlob* vs_blob = nullptr;
    ID3DBlob* ps_op_blob = nullptr;
    ID3DBlob* ps_al_blob = nullptr;
    ID3DBlob* seed_vs_blob = nullptr;
    ID3DBlob* seed_ps_blob = nullptr;

    const std::string ps_opaque_src = std::string(kGsPsCommonBody) + kGsPsOpaqueMain;
    const std::string ps_alpha_src  = std::string(kGsPsCommonBody) + kGsPsAlphaMain;

    if (!compile_shader(kGsVsSource, "vs_4_0", vs_blob) ||
        !compile_shader(ps_opaque_src, "ps_4_0", ps_op_blob) ||
        !compile_shader(ps_alpha_src, "ps_4_0", ps_al_blob) ||
        !compile_shader(kSeedVsSource, "vs_4_0", seed_vs_blob) ||
        !compile_shader(kSeedPsSource, "ps_4_0", seed_ps_blob)) {
        safe_release(vs_blob);
        safe_release(ps_op_blob);
        safe_release(ps_al_blob);
        safe_release(seed_vs_blob);
        safe_release(seed_ps_blob);
        shutdown();
        return false;
    }

    const HRESULT h1 = m_device->CreateVertexShader(vs_blob->GetBufferPointer(), vs_blob->GetBufferSize(), nullptr, &m_vs);
    const HRESULT h2 = m_device->CreatePixelShader(ps_op_blob->GetBufferPointer(), ps_op_blob->GetBufferSize(), nullptr, &m_ps_opaque);
    const HRESULT h3 = m_device->CreatePixelShader(ps_al_blob->GetBufferPointer(), ps_al_blob->GetBufferSize(), nullptr, &m_ps_alpha);
    const HRESULT h4 = m_device->CreateVertexShader(seed_vs_blob->GetBufferPointer(), seed_vs_blob->GetBufferSize(), nullptr, &m_seed_vs);
    const HRESULT h5 = m_device->CreatePixelShader(seed_ps_blob->GetBufferPointer(), seed_ps_blob->GetBufferSize(), nullptr, &m_seed_ps);

    static constexpr D3D11_INPUT_ELEMENT_DESC kLayout[] = {
        { "POSITION", 0, DXGI_FORMAT_R32G32_FLOAT,    0,  0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "TEXCOORD", 0, DXGI_FORMAT_R32_FLOAT,       0,  8, D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "COLOR",    0, DXGI_FORMAT_R8G8B8A8_UNORM,  0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "TEXCOORD", 1, DXGI_FORMAT_R32G32B32_FLOAT, 0, 16, D3D11_INPUT_PER_VERTEX_DATA, 0 }
    };
    const HRESULT h6 = m_device->CreateInputLayout(
        kLayout, static_cast<UINT>(std::size(kLayout)),
        vs_blob->GetBufferPointer(), vs_blob->GetBufferSize(), &m_input_layout);

    safe_release(vs_blob);
    safe_release(ps_op_blob);
    safe_release(ps_al_blob);
    safe_release(seed_vs_blob);
    safe_release(seed_ps_blob);

    if (FAILED(h1) || FAILED(h2) || FAILED(h3) || FAILED(h4) || FAILED(h5) || FAILED(h6)) {
        shutdown();
        return false;
    }

    D3D11_BUFFER_DESC vbd{};
    vbd.ByteWidth = m_vb_capacity * sizeof(GsHwVertex);
    vbd.Usage = D3D11_USAGE_DYNAMIC;
    vbd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    vbd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    if (FAILED(m_device->CreateBuffer(&vbd, nullptr, &m_vb))) {
        shutdown();
        return false;
    }

    D3D11_BUFFER_DESC cbd{};
    cbd.ByteWidth = sizeof(GsDrawConstantBuffer);
    cbd.Usage = D3D11_USAGE_DYNAMIC;
    cbd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    cbd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    if (FAILED(m_device->CreateBuffer(&cbd, nullptr, &m_cb))) {
        shutdown();
        return false;
    }

    // Create 8 sampler combos: [bilinear][clampU][clampV]
    for (int i = 0; i < 8; ++i) {
        D3D11_SAMPLER_DESC sd{};
        sd.Filter = (i & 4) ? D3D11_FILTER_MIN_MAG_MIP_LINEAR : D3D11_FILTER_MIN_MAG_MIP_POINT;
        sd.AddressU = (i & 2) ? D3D11_TEXTURE_ADDRESS_CLAMP : D3D11_TEXTURE_ADDRESS_WRAP;
        sd.AddressV = (i & 1) ? D3D11_TEXTURE_ADDRESS_CLAMP : D3D11_TEXTURE_ADDRESS_WRAP;
        sd.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
        sd.ComparisonFunc = D3D11_COMPARISON_NEVER;
        sd.MaxLOD = D3D11_FLOAT32_MAX;
        if (FAILED(m_device->CreateSamplerState(&sd, &m_samplers[static_cast<std::size_t>(i)]))) {
            shutdown();
            return false;
        }
    }

    D3D11_RASTERIZER_DESC rd{};
    rd.FillMode = D3D11_FILL_SOLID;
    rd.CullMode = D3D11_CULL_NONE;
    rd.DepthClipEnable = FALSE;
    rd.ScissorEnable = TRUE;
    if (FAILED(m_device->CreateRasterizerState(&rd, &m_rs_scissor))) {
        shutdown();
        return false;
    }
    rd.ScissorEnable = FALSE;
    if (FAILED(m_device->CreateRasterizerState(&rd, &m_rs_noscissor))) {
        shutdown();
        return false;
    }

    D3D11_BLEND_DESC bd{};
    bd.RenderTarget[0].BlendEnable = FALSE;
    bd.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
    if (FAILED(m_device->CreateBlendState(&bd, &m_bs_opaque_all))) {
        shutdown();
        return false;
    }

    D3D11_DEPTH_STENCIL_DESC dsd{};
    dsd.DepthEnable = FALSE;
    dsd.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO;
    dsd.DepthFunc = D3D11_COMPARISON_ALWAYS;
    if (FAILED(m_device->CreateDepthStencilState(&dsd, &m_ds_disabled))) {
        shutdown();
        return false;
    }

    D3D11_QUERY_DESC qd{};
    qd.Query = D3D11_QUERY_EVENT;
    for (auto& q : m_frame_queries) {
        q = nullptr;
        m_device->CreateQuery(&qd, &q);
    }
    m_query_pending = {false, false, false};
    m_query_cursor = 0;
    m_draws_since_flush = 0;
    m_last_publish_us = 0;
    m_last_publish_fbp = 0xFFFFFFFFu;
    m_last_publish_w = m_last_publish_h = 0;
    m_last_publish_dbx = m_last_publish_dby = 0;

    m_active_scale = m_res_scale.load(std::memory_order_acquire);
    m_vb_offset = 0;
    m_ready = true;
    return true;
}

void D3D11GsRenderer::release_all_targets_locked() {
    for (auto& rt : m_rts) {
        safe_release(rt.srv);
        safe_release(rt.rtv);
        safe_release(rt.tex);
        rt.logical_w = rt.logical_h = rt.phys_w = rt.phys_h = rt.fbw = rt.psm = 0;
        rt.valid = false;
    }
    for (auto& zb : m_zbs) {
        safe_release(zb.srv);
        safe_release(zb.dsv);
        safe_release(zb.tex);
        zb.logical_w = zb.logical_h = zb.phys_w = zb.phys_h = zb.z_psm = 0;
        zb.valid = false;
    }
    safe_release(m_self_blit_srv);
    safe_release(m_self_blit_tex);
    m_self_blit_w = m_self_blit_h = 0;

    safe_release(m_date_srv);
    safe_release(m_date_tex);
    m_date_w = m_date_h = 0;

    safe_release(m_zb_copy_srv);
    safe_release(m_zb_copy_tex);
    m_zb_copy_w = m_zb_copy_h = 0;
}

void D3D11GsRenderer::shutdown() {
    m_ready = false;

    ID3D11ShaderResourceView* detached_presented_srv = nullptr;
    {
        std::lock_guard<std::mutex> pmlk(m_present_meta_mtx);
        m_presented_is_gpu = false;
        m_presented_seq = 0;
        m_pres_src_w = m_pres_src_h = 0;
        detached_presented_srv = m_presented_srv;
        m_presented_srv = nullptr;
    }

    if (m_ctx_mtx) {
        std::lock_guard<std::mutex> lk(*m_ctx_mtx);
        release_all_targets_locked();
        for (auto& ts : m_tex_slots) {
            safe_release(ts.srv);
            safe_release(ts.tex);
            ts.w = ts.h = 0;
            ts.uploaded_gen = 0;
        }
        for (auto& q : m_frame_queries) {
            safe_release(q);
        }
        m_query_pending = {false, false, false};
        m_query_cursor = 0;
        m_draws_since_flush = 0;
        m_last_publish_us = 0;
        m_last_publish_fbp = 0xFFFFFFFFu;
        m_last_publish_w = m_last_publish_h = 0;
        m_last_publish_dbx = m_last_publish_dby = 0;

        safe_release(m_seed_srv);
        safe_release(m_seed_tex);
        m_seed_w = m_seed_h = 0;
        safe_release(m_clut_srv);
        safe_release(m_clut_tex);
        m_clut_uploaded_hash = 0;
        safe_release(m_staging_tex);
        m_staging_w = m_staging_h = 0;
        safe_release(detached_presented_srv);
        safe_release(m_presented_tex);
        m_presented_tex_w = m_presented_tex_h = 0;

        for (auto& kv : m_blend_cache) safe_release(kv.second);
        m_blend_cache.clear();
        for (auto& kv : m_depth_cache) safe_release(kv.second);
        m_depth_cache.clear();

        safe_release(m_ds_disabled);
        safe_release(m_bs_opaque_all);
        safe_release(m_rs_noscissor);
        safe_release(m_rs_scissor);
        for (auto& s : m_samplers) safe_release(s);
        safe_release(m_cb);
        safe_release(m_vb);
        safe_release(m_input_layout);
        safe_release(m_seed_ps);
        safe_release(m_seed_vs);
        safe_release(m_ps_alpha);
        safe_release(m_ps_opaque);
        safe_release(m_vs);
    } else {
        safe_release(detached_presented_srv);
    }

    safe_release(m_ctx);
    safe_release(m_device);
    m_ctx_mtx = nullptr;
    m_ui_waiters = nullptr;
}

void D3D11GsRenderer::set_resolution_scale(u32 scale) {
    const u32 clamped = std::clamp<u32>(scale, 1u, 4u);
    m_res_scale.store(clamped, std::memory_order_release);
}

void D3D11GsRenderer::yield_for_ui_waiter() const {
    if (!m_ui_waiters) return;
    for (int i = 0; i < 64 && m_ui_waiters->load(std::memory_order_acquire) > 0u; ++i) {
        if (i < 16) {
            SwitchToThread();
        } else {
            Sleep(1);
        }
    }
}

std::unique_lock<std::mutex> D3D11GsRenderer::acquire_ctx_lock() const {
    yield_for_ui_waiter();
    return std::unique_lock<std::mutex>(*m_ctx_mtx);
}

void D3D11GsRenderer::throttle_gpu_frame_queue() {
    if (!m_ready || !m_ctx_mtx || !m_ui_waiters ||
        !m_frame_queries[0] || !m_frame_queries[1] || !m_frame_queries[2]) {
        return;
    }

    const u32 cur = m_query_cursor % 3u;
    const u32 oldest = (cur + 1u) % 3u;
    bool need_wait_oldest = false;

    {
        auto lk = acquire_ctx_lock();
        if (!m_ready || !m_ctx) return;
        m_ctx->End(m_frame_queries[cur]);
        m_ctx->Flush();
        m_query_pending[cur] = true;
        m_draws_since_flush = 0;
        if (m_query_pending[oldest]) {
            BOOL done = FALSE;
            const HRESULT hr = m_ctx->GetData(
                m_frame_queries[oldest], &done, sizeof(done), D3D11_ASYNC_GETDATA_DONOTFLUSH);
            if (hr == S_OK) {
                m_query_pending[oldest] = false;
            } else if (hr == S_FALSE) {
                need_wait_oldest = true;
            } else {
                m_query_pending[oldest] = false;
            }
        }
    }

    // Wait strictly OUTSIDE *m_ctx_mtx so the Win32 UI thread can freely present/resize/Alt-Tab!
    if (need_wait_oldest) {
        const auto wait_start = std::chrono::steady_clock::now();
        int poll_iter = 0;
        while (m_ready) {
            if (poll_iter < 2) {
                SwitchToThread();
            } else {
                Sleep(1);
            }
            ++poll_iter;
            BOOL done = FALSE;
            HRESULT hr = S_FALSE;
            {
                auto lk = acquire_ctx_lock();
                if (!m_ready || !m_ctx) return;
                hr = m_ctx->GetData(
                    m_frame_queries[oldest], &done, sizeof(done), D3D11_ASYNC_GETDATA_DONOTFLUSH);
                if (hr == S_FALSE && m_device && FAILED(m_device->GetDeviceRemovedReason())) {
                    m_query_pending[oldest] = false;
                    break;
                }
            }
            if (hr != S_FALSE) {
                m_query_pending[oldest] = false;
                break;
            }
            const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - wait_start).count();
            if (elapsed_ms >= 500) {
                m_query_pending[oldest] = false;
                break;
            }
        }
    }

    m_query_cursor = oldest;
}

void D3D11GsRenderer::on_reset() {
    if (!m_ready || !m_ctx_mtx) return;
    {
        auto lk = acquire_ctx_lock();
        for (auto& rt : m_rts) rt.valid = false;
        for (auto& zb : m_zbs) zb.valid = false;
        for (auto& ts : m_tex_slots) ts.uploaded_gen = 0;
        m_clut_uploaded_hash = 0;
        m_vb_offset = 0;
        m_query_pending = {false, false, false};
        m_query_cursor = 0;
        m_draws_since_flush = 0;
        m_last_publish_us = 0;
        m_last_publish_fbp = 0xFFFFFFFFu;
        m_last_publish_w = m_last_publish_h = 0;
        m_last_publish_dbx = m_last_publish_dby = 0;
    }
    {
        std::lock_guard<std::mutex> pmlk(m_present_meta_mtx);
        m_presented_is_gpu = false;
        m_presented_seq = 0;
        m_pres_src_w = m_pres_src_h = 0;
    }
}

void D3D11GsRenderer::on_vram_write(u32 start_block, u32 num_blocks) {
    // EE-thread-only RT validity tracking — never touches ID3D11DeviceContext*, zero UI lock contention!
    if (!m_ready) return;
    for (u32 fbp = 0; fbp < 512u; ++fbp) {
        GpuRt& rt = m_rts[fbp];
        if (!rt.valid || rt.fbw == 0u) continue;
        const u32 ph = psm_page_height_hw(rt.psm);
        const u32 pages_y = std::max<u32>(1u, (rt.logical_h + ph - 1u) / ph);
        const u32 rt_len = pages_y * rt.fbw * 32u;
        if (ranges_overlap_vram_hw(start_block, num_blocks, fbp * 32u, rt_len)) {
            rt.valid = false;
        }
    }
}

bool D3D11GsRenderer::is_rt_resident(u32 fbp) const {
    // EE-thread-only metadata read — zero *m_ctx_mtx contention!
    if (!m_ready) return false;
    return m_rts[fbp & 0x1FFu].valid;
}

bool D3D11GsRenderer::is_zb_resident(u32 zbp) const {
    if (!m_ready) return false;
    return m_zbs[zbp & 0x1FFu].valid;
}

void D3D11GsRenderer::blit_seed_to_rt_locked(GpuRt& rt, const u32* rgba, u32 stride, u32 h) {
    if (!rgba || stride == 0u || h == 0u || !rt.rtv) return;

    if (!m_seed_tex || m_seed_w != stride || m_seed_h != h) {
        safe_release(m_seed_srv);
        safe_release(m_seed_tex);
        D3D11_TEXTURE2D_DESC td{};
        td.Width = stride;
        td.Height = h;
        td.MipLevels = 1;
        td.ArraySize = 1;
        td.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        td.SampleDesc.Count = 1;
        td.Usage = D3D11_USAGE_DYNAMIC;
        td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        td.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        if (FAILED(m_device->CreateTexture2D(&td, nullptr, &m_seed_tex))) return;
        if (FAILED(m_device->CreateShaderResourceView(m_seed_tex, nullptr, &m_seed_srv))) {
            safe_release(m_seed_tex);
            return;
        }
        m_seed_w = stride;
        m_seed_h = h;
    }

    D3D11_MAPPED_SUBRESOURCE mapped{};
    if (SUCCEEDED(m_ctx->Map(m_seed_tex, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped))) {
        if (mapped.pData != nullptr) {
            const std::size_t row_bytes = static_cast<std::size_t>(stride) * 4u;
            if (mapped.RowPitch == row_bytes) {
                std::memcpy(mapped.pData, rgba, row_bytes * h);
            } else {
                u8* dst = static_cast<u8*>(mapped.pData);
                const u8* src = reinterpret_cast<const u8*>(rgba);
                for (u32 y = 0; y < h; ++y) {
                    std::memcpy(dst + static_cast<std::size_t>(y) * mapped.RowPitch, src + y * row_bytes, row_bytes);
                }
            }
        }
        m_ctx->Unmap(m_seed_tex, 0);
    }

    D3D11_VIEWPORT vp{};
    vp.Width = static_cast<float>(rt.phys_w);
    vp.Height = static_cast<float>(rt.phys_h);
    vp.MinDepth = 0.0f;
    vp.MaxDepth = 1.0f;
    static constexpr float kZeroBF[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    m_ctx->OMSetRenderTargets(1, &rt.rtv, nullptr);
    m_ctx->RSSetViewports(1, &vp);
    m_ctx->RSSetState(m_rs_noscissor);
    m_ctx->OMSetBlendState(m_bs_opaque_all, kZeroBF, 0xFFFFFFFFu);
    m_ctx->OMSetDepthStencilState(m_ds_disabled, 0);
    m_ctx->IASetInputLayout(nullptr);
    m_ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    m_ctx->VSSetShader(m_seed_vs, nullptr, 0);
    m_ctx->PSSetShader(m_seed_ps, nullptr, 0);
    m_ctx->PSSetShaderResources(0, 1, &m_seed_srv);
    m_ctx->PSSetSamplers(0, 1, &m_samplers[3]); // point clamp
    m_ctx->Draw(3, 0);
    ID3D11ShaderResourceView* null_srv = nullptr;
    m_ctx->PSSetShaderResources(0, 1, &null_srv);
}

D3D11GsRenderer::GpuRt* D3D11GsRenderer::ensure_rt_locked(
    u32 fbp, u32 fbw, u32 psm, u32 logical_h, u32 scale,
    const u32* seed_rgba, u32 seed_stride, u32 seed_h, bool will_overwrite_all) {
    fbp &= 0x1FFu;
    const u32 eff_fbw = std::max<u32>(1u, fbw);
    const u32 logical_w = eff_fbw * 64u;
    const u32 eff_h = std::clamp<u32>(logical_h, 32u, 1024u);

    GpuRt& rt = m_rts[fbp];
    const u32 target_h = (rt.valid && rt.fbw == eff_fbw && rt.logical_h > eff_h) ? rt.logical_h : eff_h;
    const u32 phys_w = logical_w * scale;
    const u32 phys_h = target_h * scale;

    const bool need_alloc = (!rt.tex || rt.phys_w != phys_w || rt.phys_h != phys_h);
    const bool was_valid = rt.valid && !need_alloc && rt.fbw == eff_fbw;

    if (need_alloc) {
        safe_release(rt.srv);
        safe_release(rt.rtv);
        safe_release(rt.tex);

        D3D11_TEXTURE2D_DESC td{};
        td.Width = phys_w;
        td.Height = phys_h;
        td.MipLevels = 1;
        td.ArraySize = 1;
        td.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        td.SampleDesc.Count = 1;
        td.Usage = D3D11_USAGE_DEFAULT;
        td.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
        if (FAILED(m_device->CreateTexture2D(&td, nullptr, &rt.tex)) || !rt.tex) {
            rt.valid = false;
            return nullptr;
        }
        if (FAILED(m_device->CreateRenderTargetView(rt.tex, nullptr, &rt.rtv)) ||
            FAILED(m_device->CreateShaderResourceView(rt.tex, nullptr, &rt.srv))) {
            safe_release(rt.srv);
            safe_release(rt.rtv);
            safe_release(rt.tex);
            rt.valid = false;
            return nullptr;
        }
        rt.phys_w = phys_w;
        rt.phys_h = phys_h;
    }

    rt.logical_w = logical_w;
    rt.logical_h = target_h;
    rt.fbw = eff_fbw;
    rt.psm = psm;

    if (!was_valid) {
        if (!will_overwrite_all) {
            if (seed_rgba && seed_stride > 0u && seed_h > 0u) {
                blit_seed_to_rt_locked(rt, seed_rgba, seed_stride, seed_h);
            } else {
                static constexpr float kZero[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                m_ctx->ClearRenderTargetView(rt.rtv, kZero);
            }
        }
        rt.valid = true;
    }
    return &rt;
}

D3D11GsRenderer::GpuZb* D3D11GsRenderer::ensure_zb_locked(u32 zbp, u32 logical_w, u32 logical_h, u32 scale) {
    zbp &= 0x1FFu;
    GpuZb& zb = m_zbs[zbp];
    const u32 target_w = std::max<u32>(zb.valid ? zb.logical_w : 0u, logical_w);
    const u32 target_h = std::max<u32>(zb.valid ? zb.logical_h : 0u, logical_h);
    const u32 phys_w = target_w * scale;
    const u32 phys_h = target_h * scale;

    if (!zb.tex || zb.phys_w != phys_w || zb.phys_h != phys_h) {
        safe_release(zb.srv);
        safe_release(zb.dsv);
        safe_release(zb.tex);

        D3D11_TEXTURE2D_DESC td{};
        td.Width = phys_w;
        td.Height = phys_h;
        td.MipLevels = 1;
        td.ArraySize = 1;
        td.Format = DXGI_FORMAT_R32_TYPELESS;
        td.SampleDesc.Count = 1;
        td.Usage = D3D11_USAGE_DEFAULT;
        td.BindFlags = D3D11_BIND_DEPTH_STENCIL | D3D11_BIND_SHADER_RESOURCE;
        if (FAILED(m_device->CreateTexture2D(&td, nullptr, &zb.tex)) || !zb.tex) {
            zb.valid = false;
            return nullptr;
        }
        D3D11_DEPTH_STENCIL_VIEW_DESC dsvd{};
        dsvd.Format = DXGI_FORMAT_D32_FLOAT;
        dsvd.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
        D3D11_SHADER_RESOURCE_VIEW_DESC srvd{};
        srvd.Format = DXGI_FORMAT_R32_FLOAT;
        srvd.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
        srvd.Texture2D.MipLevels = 1;
        if (FAILED(m_device->CreateDepthStencilView(zb.tex, &dsvd, &zb.dsv)) ||
            FAILED(m_device->CreateShaderResourceView(zb.tex, &srvd, &zb.srv))) {
            safe_release(zb.srv);
            safe_release(zb.dsv);
            safe_release(zb.tex);
            zb.valid = false;
            return nullptr;
        }
        zb.logical_w = target_w;
        zb.logical_h = target_h;
        zb.phys_w = phys_w;
        zb.phys_h = phys_h;
        zb.valid = false;
    }
    if (!zb.valid) {
        m_ctx->ClearDepthStencilView(zb.dsv, D3D11_CLEAR_DEPTH, 0.0f, 0);
        zb.valid = true;
    }
    return &zb;
}

ID3D11ShaderResourceView* D3D11GsRenderer::ensure_tex_slot_locked(
    u32 slot, u64 gen, u32 tw, u32 th, const u32* rgba) {
    if (!rgba || tw == 0u || th == 0u || slot >= kGpuTexSlots) return nullptr;
    GpuTexSlot& ts = m_tex_slots[slot];
    if (ts.tex && ts.srv && ts.w == tw && ts.h == th && ts.uploaded_gen == gen) {
        return ts.srv;
    }
    if (!ts.tex || ts.w != tw || ts.h != th) {
        safe_release(ts.srv);
        safe_release(ts.tex);
        D3D11_TEXTURE2D_DESC td{};
        td.Width = tw;
        td.Height = th;
        td.MipLevels = 1;
        td.ArraySize = 1;
        td.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        td.SampleDesc.Count = 1;
        td.Usage = D3D11_USAGE_DEFAULT;
        td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        if (FAILED(m_device->CreateTexture2D(&td, nullptr, &ts.tex)) || !ts.tex) return nullptr;
        if (FAILED(m_device->CreateShaderResourceView(ts.tex, nullptr, &ts.srv))) {
            safe_release(ts.tex);
            return nullptr;
        }
        ts.w = tw;
        ts.h = th;
    }
    m_ctx->UpdateSubresource(ts.tex, 0, nullptr, rgba, tw * 4u, 0);
    ts.uploaded_gen = gen;
    return ts.srv;
}

ID3D11ShaderResourceView* D3D11GsRenderer::ensure_clut_lut_locked(const u32* clut_rgba, u64 clut_hash) {
    if (!clut_rgba) return nullptr;
    if (!m_clut_tex || !m_clut_srv) {
        safe_release(m_clut_srv);
        safe_release(m_clut_tex);
        D3D11_TEXTURE2D_DESC td{};
        td.Width = 256;
        td.Height = 1;
        td.MipLevels = 1;
        td.ArraySize = 1;
        td.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        td.SampleDesc.Count = 1;
        td.Usage = D3D11_USAGE_DEFAULT;
        td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        if (FAILED(m_device->CreateTexture2D(&td, nullptr, &m_clut_tex)) || !m_clut_tex) return nullptr;
        if (FAILED(m_device->CreateShaderResourceView(m_clut_tex, nullptr, &m_clut_srv))) {
            safe_release(m_clut_tex);
            return nullptr;
        }
        m_clut_uploaded_hash = 0;
    }
    if (m_clut_uploaded_hash != clut_hash || clut_hash == 0u) {
        m_ctx->UpdateSubresource(m_clut_tex, 0, nullptr, clut_rgba, 256u * 4u, 0);
        m_clut_uploaded_hash = clut_hash;
    }
    return m_clut_srv;
}

ID3D11ShaderResourceView* D3D11GsRenderer::copy_zb_to_scratch_locked(const GpuZb& zb) {
    if (!zb.tex || zb.phys_w == 0u || zb.phys_h == 0u) return nullptr;
    if (!m_zb_copy_tex || m_zb_copy_w != zb.phys_w || m_zb_copy_h != zb.phys_h) {
        safe_release(m_zb_copy_srv);
        safe_release(m_zb_copy_tex);
        D3D11_TEXTURE2D_DESC td{};
        td.Width = zb.phys_w;
        td.Height = zb.phys_h;
        td.MipLevels = 1;
        td.ArraySize = 1;
        td.Format = DXGI_FORMAT_R32_TYPELESS;
        td.SampleDesc.Count = 1;
        td.Usage = D3D11_USAGE_DEFAULT;
        td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        if (FAILED(m_device->CreateTexture2D(&td, nullptr, &m_zb_copy_tex)) || !m_zb_copy_tex) return nullptr;
        D3D11_SHADER_RESOURCE_VIEW_DESC srvd{};
        srvd.Format = DXGI_FORMAT_R32_FLOAT;
        srvd.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
        srvd.Texture2D.MipLevels = 1;
        if (FAILED(m_device->CreateShaderResourceView(m_zb_copy_tex, &srvd, &m_zb_copy_srv))) {
            safe_release(m_zb_copy_tex);
            return nullptr;
        }
        m_zb_copy_w = zb.phys_w;
        m_zb_copy_h = zb.phys_h;
    }
    m_ctx->CopyResource(m_zb_copy_tex, zb.tex);
    return m_zb_copy_srv;
}

ID3D11ShaderResourceView* D3D11GsRenderer::copy_rt_to_scratch_locked(
    const GpuRt& rt, ID3D11Texture2D*& dst_tex, ID3D11ShaderResourceView*& dst_srv, u32& dst_w, u32& dst_h) {
    if (!rt.tex || rt.phys_w == 0u || rt.phys_h == 0u) return nullptr;
    if (!dst_tex || dst_w != rt.phys_w || dst_h != rt.phys_h) {
        safe_release(dst_srv);
        safe_release(dst_tex);
        D3D11_TEXTURE2D_DESC td{};
        td.Width = rt.phys_w;
        td.Height = rt.phys_h;
        td.MipLevels = 1;
        td.ArraySize = 1;
        td.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        td.SampleDesc.Count = 1;
        td.Usage = D3D11_USAGE_DEFAULT;
        td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        if (FAILED(m_device->CreateTexture2D(&td, nullptr, &dst_tex)) || !dst_tex) return nullptr;
        if (FAILED(m_device->CreateShaderResourceView(dst_tex, nullptr, &dst_srv))) {
            safe_release(dst_tex);
            return nullptr;
        }
        dst_w = rt.phys_w;
        dst_h = rt.phys_h;
    }
    m_ctx->CopyResource(dst_tex, rt.tex);
    return dst_srv;
}

ID3D11BlendState* D3D11GsRenderer::get_blend_state_locked(
    bool abe, u32 spec_a, u32 spec_b, u32 spec_c, u32 spec_d,
    u32 fbmsk, bool write_color, bool write_alpha, bool& out_premul_c) {
    out_premul_c = false;

    UINT8 mask = 0;
    if (write_color) {
        if ((fbmsk & 0x000000FFu) != 0x000000FFu) mask |= D3D11_COLOR_WRITE_ENABLE_RED;
        if ((fbmsk & 0x0000FF00u) != 0x0000FF00u) mask |= D3D11_COLOR_WRITE_ENABLE_GREEN;
        if ((fbmsk & 0x00FF0000u) != 0x00FF0000u) mask |= D3D11_COLOR_WRITE_ENABLE_BLUE;
        if (write_alpha && (fbmsk & 0xFF000000u) != 0xFF000000u) mask |= D3D11_COLOR_WRITE_ENABLE_ALPHA;
    }

    if (!abe && mask == D3D11_COLOR_WRITE_ENABLE_ALL) {
        return m_bs_opaque_all;
    }

    // Determine whether to pre-multiply RGB by full PS2 [0..2.0] blend factor in PS
    if (abe && spec_c != 1u) {
        if ((spec_a == 0u && spec_b == 2u && (spec_d == 1u || spec_d == 2u)) ||
            (spec_a == 2u && spec_b == 0u && spec_d == 1u)) {
            out_premul_c = true;
        }
    }

    const u64 key = (static_cast<u64>(abe ? 1u : 0u)) |
                    (static_cast<u64>(spec_a & 3u) << 1) |
                    (static_cast<u64>(spec_b & 3u) << 3) |
                    (static_cast<u64>(spec_c & 3u) << 5) |
                    (static_cast<u64>(spec_d & 3u) << 7) |
                    (static_cast<u64>(mask) << 9) |
                    (static_cast<u64>(out_premul_c ? 1u : 0u) << 13);

    if (auto it = m_blend_cache.find(key); it != m_blend_cache.end()) {
        return it->second;
    }

    D3D11_BLEND_DESC bd{};
    auto& rt = bd.RenderTarget[0];
    rt.RenderTargetWriteMask = mask;
    rt.BlendEnable = abe ? TRUE : FALSE;
    rt.SrcBlendAlpha = D3D11_BLEND_ONE;
    rt.DestBlendAlpha = D3D11_BLEND_ZERO;
    rt.BlendOpAlpha = D3D11_BLEND_OP_ADD;

    if (abe) {
        const D3D11_BLEND c_src = (spec_c == 1u) ? D3D11_BLEND_DEST_ALPHA : D3D11_BLEND_SRC1_ALPHA;
        const D3D11_BLEND inv_c = (spec_c == 1u) ? D3D11_BLEND_INV_DEST_ALPHA : D3D11_BLEND_INV_SRC1_ALPHA;

        // Exact PS2 ((A - B) * C) >> 7 + D mapping
        if (spec_a == spec_b) {
            rt.BlendOp = D3D11_BLEND_OP_ADD;
            rt.SrcBlend = (spec_d == 0u) ? D3D11_BLEND_ONE : D3D11_BLEND_ZERO;
            rt.DestBlend = (spec_d == 1u) ? D3D11_BLEND_ONE : D3D11_BLEND_ZERO;
        } else if (spec_a == 0u && spec_b == 1u && spec_d == 1u) {
            // Cs * C + Cd * (1 - C) — standard alpha blend
            rt.BlendOp = D3D11_BLEND_OP_ADD;
            rt.SrcBlend = c_src;
            rt.DestBlend = inv_c;
        } else if (spec_a == 0u && spec_b == 2u && spec_d == 1u) {
            // Cs * C + Cd — additive glow/bloom
            rt.BlendOp = D3D11_BLEND_OP_ADD;
            rt.SrcBlend = out_premul_c ? D3D11_BLEND_ONE : c_src;
            rt.DestBlend = D3D11_BLEND_ONE;
        } else if (spec_a == 0u && spec_b == 2u && spec_d == 2u) {
            // Cs * C
            rt.BlendOp = D3D11_BLEND_OP_ADD;
            rt.SrcBlend = out_premul_c ? D3D11_BLEND_ONE : c_src;
            rt.DestBlend = D3D11_BLEND_ZERO;
        } else if (spec_a == 2u && spec_b == 0u && spec_d == 1u) {
            // Cd - Cs * C — subtractive
            rt.BlendOp = D3D11_BLEND_OP_REV_SUBTRACT;
            rt.SrcBlend = out_premul_c ? D3D11_BLEND_ONE : c_src;
            rt.DestBlend = D3D11_BLEND_ONE;
        } else if (spec_a == 1u && spec_b == 0u && spec_d == 1u) {
            // Cd * (1 + C) - Cs * C -> closest hardware op: Cd - Cs * C
            rt.BlendOp = D3D11_BLEND_OP_REV_SUBTRACT;
            rt.SrcBlend = c_src;
            rt.DestBlend = D3D11_BLEND_ONE;
        } else if (spec_a == 1u && spec_b == 2u && spec_d == 0u) {
            // Cs + Cd * C
            rt.BlendOp = D3D11_BLEND_OP_ADD;
            rt.SrcBlend = D3D11_BLEND_ONE;
            rt.DestBlend = c_src;
        } else if (spec_a == 1u && spec_b == 2u && spec_d == 2u) {
            // Cd * C
            rt.BlendOp = D3D11_BLEND_OP_ADD;
            rt.SrcBlend = D3D11_BLEND_ZERO;
            rt.DestBlend = c_src;
        } else if (spec_a == 2u && spec_b == 1u && spec_d == 1u) {
            // Cd * (1 - C)
            rt.BlendOp = D3D11_BLEND_OP_ADD;
            rt.SrcBlend = D3D11_BLEND_ZERO;
            rt.DestBlend = inv_c;
        } else if (spec_a == 1u && spec_b == 0u && spec_d == 0u) {
            // Cs * (1 - C) + Cd * C
            rt.BlendOp = D3D11_BLEND_OP_ADD;
            rt.SrcBlend = inv_c;
            rt.DestBlend = c_src;
        } else if (spec_a == 0u && spec_b == 1u && spec_d == 2u) {
            // (Cs - Cd) * C
            rt.BlendOp = D3D11_BLEND_OP_SUBTRACT;
            rt.SrcBlend = c_src;
            rt.DestBlend = c_src;
        } else if (spec_a == 1u && spec_b == 0u && spec_d == 2u) {
            // (Cd - Cs) * C
            rt.BlendOp = D3D11_BLEND_OP_REV_SUBTRACT;
            rt.SrcBlend = c_src;
            rt.DestBlend = c_src;
        } else {
            // Fallback to standard alpha blend
            rt.BlendOp = D3D11_BLEND_OP_ADD;
            rt.SrcBlend = c_src;
            rt.DestBlend = inv_c;
        }
    }

    ID3D11BlendState* bs = nullptr;
    if (SUCCEEDED(m_device->CreateBlendState(&bd, &bs)) && bs) {
        m_blend_cache[key] = bs;
        return bs;
    }
    return m_bs_opaque_all;
}

ID3D11DepthStencilState* D3D11GsRenderer::get_depth_state_locked(bool zte, u32 ztst, bool write_zb) {
    if (!zte) return m_ds_disabled;
    const u32 key = 1u | ((ztst & 3u) << 1) | (write_zb ? 8u : 0u);
    if (auto it = m_depth_cache.find(key); it != m_depth_cache.end()) {
        return it->second;
    }

    D3D11_DEPTH_STENCIL_DESC dsd{};
    dsd.DepthEnable = TRUE;
    dsd.DepthWriteMask = write_zb ? D3D11_DEPTH_WRITE_MASK_ALL : D3D11_DEPTH_WRITE_MASK_ZERO;
    switch (ztst & 3u) {
        case 0u: dsd.DepthFunc = D3D11_COMPARISON_NEVER; break;
        case 1u: dsd.DepthFunc = D3D11_COMPARISON_ALWAYS; break;
        case 2u: dsd.DepthFunc = D3D11_COMPARISON_GREATER_EQUAL; break;
        default: dsd.DepthFunc = D3D11_COMPARISON_GREATER; break;
    }
    ID3D11DepthStencilState* ds = nullptr;
    if (SUCCEEDED(m_device->CreateDepthStencilState(&dsd, &ds)) && ds) {
        m_depth_cache[key] = ds;
        return ds;
    }
    return m_ds_disabled;
}

void D3D11GsRenderer::upload_and_draw_verts_locked(const GsHwVertex* verts, u32 vertex_count) {
    if (!verts || vertex_count == 0u || !m_vb) return;

    u32 offset = 0;
    while (offset < vertex_count) {
        const u32 batch = std::min<u32>(vertex_count - offset, m_vb_capacity);
        D3D11_MAP map_type = D3D11_MAP_WRITE_NO_OVERWRITE;
        if (m_vb_offset + batch > m_vb_capacity) {
            m_vb_offset = 0;
            map_type = D3D11_MAP_WRITE_DISCARD;
        }
        D3D11_MAPPED_SUBRESOURCE mapped{};
        if (FAILED(m_ctx->Map(m_vb, 0, map_type, 0, &mapped))) break;
        if (mapped.pData == nullptr) {
            m_ctx->Unmap(m_vb, 0);
            break;
        }
        std::memcpy(static_cast<u8*>(mapped.pData) + static_cast<std::size_t>(m_vb_offset) * sizeof(GsHwVertex),
                    verts + offset,
                    static_cast<std::size_t>(batch) * sizeof(GsHwVertex));
        m_ctx->Unmap(m_vb, 0);

        const UINT stride = sizeof(GsHwVertex);
        const UINT vb_byte_off = 0;
        m_ctx->IASetVertexBuffers(0, 1, &m_vb, &stride, &vb_byte_off);
        m_ctx->Draw(batch, m_vb_offset);

        m_vb_offset += batch;
        offset += batch;
    }
}

void D3D11GsRenderer::clear_depth(u32 zbp, u32 width, u32 height, f32 z_norm,
                                  s32 sc_x0, s32 sc_y0, s32 sc_x1, s32 sc_y1) {
    if (!m_ready || !m_ctx_mtx) return;
    auto lk = acquire_ctx_lock();

    const u32 scale = m_res_scale.load(std::memory_order_acquire);
    if (scale != m_active_scale) {
        release_all_targets_locked();
        m_active_scale = scale;
    }

    GpuZb* zb = ensure_zb_locked(zbp, std::max<u32>(64u, width), std::max<u32>(32u, height), scale);
    if (!zb || !zb->dsv) return;

    const bool full_clear = (sc_x0 <= 0 && sc_y0 <= 0 &&
                             sc_x1 + 1 >= static_cast<s32>(zb->logical_w) &&
                             sc_y1 + 1 >= static_cast<s32>(zb->logical_h));
    const float clamped_z = std::clamp(z_norm, 0.0f, 1.0f);
    if (full_clear) {
        m_ctx->ClearDepthStencilView(zb->dsv, D3D11_CLEAR_DEPTH, clamped_z, 0);
        return;
    }

    // Partial scissored depth clear via a single depth-only quad
    GsDrawConstantBuffer cb{};
    cb.rt_inv_size[0] = 2.0f / static_cast<float>(zb->logical_w);
    cb.rt_inv_size[1] = 2.0f / static_cast<float>(zb->logical_h);
    cb.rt_inv_size[2] = static_cast<float>(zb->logical_w);
    cb.rt_inv_size[3] = static_cast<float>(zb->logical_h);
    cb.flags1[2] = 1u; // ATE always

    D3D11_MAPPED_SUBRESOURCE mapped{};
    if (SUCCEEDED(m_ctx->Map(m_cb, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped))) {
        if (mapped.pData != nullptr) {
            std::memcpy(mapped.pData, &cb, sizeof(cb));
        }
        m_ctx->Unmap(m_cb, 0);
    }

    D3D11_VIEWPORT vp{};
    vp.Width = static_cast<float>(zb->phys_w);
    vp.Height = static_cast<float>(zb->phys_h);
    vp.MinDepth = 0.0f;
    vp.MaxDepth = 1.0f;
    D3D11_RECT sc{};
    sc.left   = std::clamp<LONG>(static_cast<LONG>(sc_x0 * static_cast<s32>(scale)), 0, static_cast<LONG>(zb->phys_w));
    sc.top    = std::clamp<LONG>(static_cast<LONG>(sc_y0 * static_cast<s32>(scale)), 0, static_cast<LONG>(zb->phys_h));
    sc.right  = std::clamp<LONG>(static_cast<LONG>((sc_x1 + 1) * static_cast<s32>(scale)), sc.left, static_cast<LONG>(zb->phys_w));
    sc.bottom = std::clamp<LONG>(static_cast<LONG>((sc_y1 + 1) * static_cast<s32>(scale)), sc.top, static_cast<LONG>(zb->phys_h));
    if (sc.left >= sc.right || sc.top >= sc.bottom) return;

    m_ctx->OMSetRenderTargets(0, nullptr, zb->dsv);
    m_ctx->RSSetViewports(1, &vp);
    m_ctx->RSSetScissorRects(1, &sc);
    m_ctx->RSSetState(m_rs_scissor);
    m_ctx->OMSetDepthStencilState(get_depth_state_locked(true, 1u, true), 0);
    m_ctx->IASetInputLayout(m_input_layout);
    m_ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    m_ctx->VSSetShader(m_vs, nullptr, 0);
    m_ctx->VSSetConstantBuffers(0, 1, &m_cb);
    m_ctx->PSSetShader(nullptr, nullptr, 0);

    const float x0 = static_cast<float>(sc_x0);
    const float y0 = static_cast<float>(sc_y0);
    const float x1 = static_cast<float>(sc_x1 + 1);
    const float y1 = static_cast<float>(sc_y1 + 1);
    const GsHwVertex quad[6] = {
        { x0, y0, clamped_z, 0, 0, 0, 1 },
        { x1, y0, clamped_z, 0, 0, 0, 1 },
        { x0, y1, clamped_z, 0, 0, 0, 1 },
        { x1, y0, clamped_z, 0, 0, 0, 1 },
        { x1, y1, clamped_z, 0, 0, 0, 1 },
        { x0, y1, clamped_z, 0, 0, 0, 1 }
    };
    upload_and_draw_verts_locked(quad, 6);
}

void D3D11GsRenderer::draw_triangles(const GsHwDrawDesc& desc, const GsHwVertex* verts, u32 vertex_count) {
    if (!m_ready || !m_ctx_mtx || !verts || vertex_count < 3u) return;

    bool should_throttle = false;
    {
    auto lk = acquire_ctx_lock();

    const u32 scale = m_res_scale.load(std::memory_order_acquire);
    if (scale != m_active_scale) {
        release_all_targets_locked();
        m_active_scale = scale;
    }

    const bool tme = (desc.prim_attrs & (1u << 4)) != 0u;
    const bool abe = (desc.prim_attrs & (1u << 6)) != 0u;

    const bool ate     = (desc.test_reg & 1u) != 0u;
    const u32 atst     = static_cast<u32>((desc.test_reg >> 1) & 7u);
    const u32 aref     = static_cast<u32>((desc.test_reg >> 4) & 0xFFu);
    const u32 afail    = static_cast<u32>((desc.test_reg >> 12) & 3u);
    const bool eff_ate = ate && (atst != 1u);
    const bool date    = ((desc.test_reg >> 14) & 1u) != 0u;
    const u32 datm     = static_cast<u32>((desc.test_reg >> 15) & 1u);
    const bool zte     = ((desc.test_reg >> 16) & 1u) != 0u;
    const u32 ztst     = static_cast<u32>((desc.test_reg >> 17) & 3u);

    if ((zte && ztst == 0u) || (ate && atst == 0u && afail == 0u)) {
        return;
    }

    const u32 tex_psm = static_cast<u32>((desc.tex0_reg >> 20) & 0x3Fu);
    const u32 tw = 1u << std::min<u32>(10u, static_cast<u32>((desc.tex0_reg >> 26) & 0xFu));
    const u32 th = 1u << std::min<u32>(10u, static_cast<u32>((desc.tex0_reg >> 30) & 0xFu));

    const GpuZb* zb_src = (tme && desc.tex_is_zb) ? &m_zbs[desc.tex_zb_zbp & 0x1FFu] : nullptr;
    if (zb_src && (!zb_src->valid || !zb_src->srv)) {
        zb_src = nullptr;
    }

    // Detect PS2 16-bit Z-buffer channel shuffle (PSMZ16 -> PSMCT16 double-height 8-px strips into a 32-bit RT)
    // used for PSMT8H distance-fog / DoF readback: map directly onto 32-bit (zb_src->logical_h) RT channels!
    const bool is_z16_shuffle =
        (zb_src != nullptr && zb_src->z_psm <= 1u &&
         (tex_psm == 0x32u || tex_psm == 0x3Au) &&
         (desc.fb_psm == 0x02u || desc.fb_psm == 0x0Au) &&
         vertex_count == 6u && desc.fb_height > zb_src->logical_h);

    u32 eff_fb_psm = desc.fb_psm;
    u32 eff_fb_h   = desc.fb_height;
    u32 eff_fbmsk  = desc.fbmsk;
    s32 eff_sc_x0  = desc.scissor_x0;
    s32 eff_sc_y0  = desc.scissor_y0;
    s32 eff_sc_x1  = desc.scissor_x1;
    s32 eff_sc_y1  = desc.scissor_y1;
    u32 eff_th     = th;
    bool z16_src_hi = false;
    GsHwVertex z16_quad[6]{};
    const GsHwVertex* draw_verts = verts;

    if (is_z16_shuffle) {
        f32 min_x = verts[0].x;
        f32 min_y = verts[0].y, max_y = verts[0].y;
        f32 min_s = verts[0].s;
        for (u32 i = 1; i < 6u; ++i) {
            min_x = std::min(min_x, verts[i].x);
            min_y = std::min(min_y, verts[i].y);
            max_y = std::max(max_y, verts[i].y);
            min_s = std::min(min_s, verts[i].s);
        }
        const f32 min_u = min_s * static_cast<f32>(tw);
        z16_src_hi = ((static_cast<int>(std::floor(min_u + 0.5f))) & 8) != 0;
        const bool z16_dst_hi = ((static_cast<int>(std::floor(min_x + 0.5f))) & 8) != 0;

        eff_fb_psm = 0u;
        eff_fb_h   = zb_src->logical_h;
        eff_fbmsk  = desc.fbmsk | (z16_dst_hi ? 0x0000FFFFu : 0xFFFF0000u);
        eff_sc_y0  = desc.scissor_y0 >> 1;
        eff_sc_y1  = desc.scissor_y1 >> 1;
        eff_th     = std::max<u32>(1u, th >> 1u);

        const f32 col_x0 = std::floor((min_x + 0.5f) / 16.0f) * 16.0f;
        const f32 col_x1 = col_x0 + 16.0f;
        const f32 col_y0 = (min_y + 0.5f) * 0.5f;
        const f32 col_y1 = (max_y + 0.5f) * 0.5f;
        const f32 src_u0 = std::floor((min_u + 0.5f) / 16.0f) * 16.0f;
        const f32 src_u1 = src_u0 + 16.0f;
        const f32 s_lo = src_u0 / static_cast<f32>(tw);
        const f32 s_hi = src_u1 / static_cast<f32>(tw);
        const f32 t_lo = col_y0 / static_cast<f32>(eff_th);
        const f32 t_hi = col_y1 / static_cast<f32>(eff_th);
        const f32 vz   = verts[0].z;
        const u32 vcol = verts[0].rgba;

        z16_quad[0] = { col_x0, col_y0, vz, vcol, s_lo, t_lo, 1.0f };
        z16_quad[1] = { col_x1, col_y0, vz, vcol, s_hi, t_lo, 1.0f };
        z16_quad[2] = { col_x0, col_y1, vz, vcol, s_lo, t_hi, 1.0f };
        z16_quad[3] = { col_x1, col_y0, vz, vcol, s_hi, t_lo, 1.0f };
        z16_quad[4] = { col_x1, col_y1, vz, vcol, s_hi, t_hi, 1.0f };
        z16_quad[5] = { col_x0, col_y1, vz, vcol, s_lo, t_hi, 1.0f };
        draw_verts = z16_quad;
    }

    GpuRt* rt = ensure_rt_locked(desc.fbp, desc.fbw, eff_fb_psm, eff_fb_h, scale,
                                 desc.seed_rt_rgba, desc.seed_rt_stride, desc.seed_rt_height,
                                 desc.will_overwrite_all);
    if (!rt || !rt->rtv) return;

    const bool need_zb = zte && (ztst != 1u || desc.z_write_en);
    GpuZb* zb = need_zb ? ensure_zb_locked(desc.zbp, rt->logical_w, rt->logical_h, scale) : nullptr;
    if (zb) {
        zb->z_psm = desc.z_psm;
    }

    // Resolve texture SRV (GPU Z-buffer, GPU RT-as-texture, GPU RT+CLUT, or 512-slot decoded texture cache)
    ID3D11ShaderResourceView* tex_srv = nullptr;
    ID3D11ShaderResourceView* clut_srv = nullptr;
    u32 src_tex_w = std::max<u32>(1u, desc.tex_w);
    u32 src_tex_h = std::max<u32>(1u, desc.tex_h);
    bool tex_is_rt_actual = false;
    u32 tex_mode = 0u;
    f32 z_max_f = 16777215.0f;

    if (tme) {
        if (zb_src != nullptr) {
            tex_is_rt_actual = true;
            src_tex_w = zb_src->logical_w;
            src_tex_h = zb_src->logical_h;
            tex_srv = (zb == zb_src) ? copy_zb_to_scratch_locked(*zb_src) : zb_src->srv;
            tex_mode = is_z16_shuffle ? (z16_src_hi ? 6u : 5u) : 4u;
            z_max_f = (zb_src->z_psm == 0x01u) ? 16777215.0f
                    : (zb_src->z_psm == 0x02u || zb_src->z_psm == 0x0Au) ? 65535.0f
                    : 4294967040.0f;
        } else if (desc.tex_is_rt) {
            const u32 src_fbp = desc.tex_rt_fbp & 0x1FFu;
            GpuRt& src_rt = m_rts[src_fbp];
            if (src_rt.valid && src_rt.srv) {
                tex_is_rt_actual = true;
                src_tex_w = src_rt.logical_w;
                src_tex_h = src_rt.logical_h;
                if (src_fbp == (desc.fbp & 0x1FFu)) {
                    // Self-blit (e.g., bloom blur pass): GPU-to-GPU CopyResource to avoid RTV/SRV hazard
                    tex_srv = copy_rt_to_scratch_locked(src_rt, m_self_blit_tex, m_self_blit_srv,
                                                         m_self_blit_w, m_self_blit_h);
                } else {
                    tex_srv = src_rt.srv;
                }
                if (desc.tex_rt_clut_mode != 0u && desc.tex_clut_rgba != nullptr) {
                    clut_srv = ensure_clut_lut_locked(desc.tex_clut_rgba, desc.tex_clut_hash);
                    if (clut_srv) {
                        tex_mode = desc.tex_rt_clut_mode;
                    }
                }
            } else if (desc.tex_rgba != nullptr && desc.tex_rt_fbw > 0u && desc.tex_rt_height > 0u) {
                // Fallback: CPU-only RT sampled as texture
                src_tex_w = desc.tex_rt_fbw * 64u;
                src_tex_h = desc.tex_rt_height;
                tex_is_rt_actual = true;
                tex_srv = ensure_tex_slot_locked(desc.tex_cache_slot, desc.tex_cache_gen,
                                                 src_tex_w, src_tex_h, desc.tex_rgba);
            }
        } else if (desc.tex_rgba != nullptr) {
            tex_srv = ensure_tex_slot_locked(desc.tex_cache_slot, desc.tex_cache_gen,
                                             desc.tex_w, desc.tex_h, desc.tex_rgba);
        }
    }

    ID3D11ShaderResourceView* date_srv = nullptr;
    if (date) {
        date_srv = copy_rt_to_scratch_locked(*rt, m_date_tex, m_date_srv, m_date_w, m_date_h);
    }

    const u32 tcc = static_cast<u32>((desc.tex0_reg >> 34) & 1u);
    const u32 tfx = static_cast<u32>((desc.tex0_reg >> 35) & 3u);
    const u32 wms = static_cast<u32>(desc.clamp_reg & 3u);
    const u32 wmt = static_cast<u32>((desc.clamp_reg >> 2) & 3u);
    const u32 minu = static_cast<u32>((desc.clamp_reg >> 4) & 0x3FFu);
    const u32 maxu = static_cast<u32>((desc.clamp_reg >> 14) & 0x3FFu);
    const u32 minv = static_cast<u32>((desc.clamp_reg >> 24) & 0x3FFu);
    const u32 maxv = static_cast<u32>((desc.clamp_reg >> 34) & 0x3FFu);

    const u32 spec_a = static_cast<u32>(desc.alpha_reg & 3u);
    const u32 spec_b = static_cast<u32>((desc.alpha_reg >> 2) & 3u);
    const u32 spec_c = static_cast<u32>((desc.alpha_reg >> 4) & 3u);
    const u32 spec_d = static_cast<u32>((desc.alpha_reg >> 6) & 3u);
    const u32 fix    = static_cast<u32>((desc.alpha_reg >> 32) & 0xFFu);

    D3D11_VIEWPORT vp{};
    vp.TopLeftX = 0.0f;
    vp.TopLeftY = 0.0f;
    vp.Width = static_cast<float>(rt->phys_w);
    vp.Height = static_cast<float>(rt->phys_h);
    vp.MinDepth = 0.0f;
    vp.MaxDepth = 1.0f;

    D3D11_RECT sc{};
    sc.left   = std::clamp<LONG>(static_cast<LONG>(eff_sc_x0 * static_cast<s32>(scale)), 0, static_cast<LONG>(rt->phys_w));
    sc.top    = std::clamp<LONG>(static_cast<LONG>(eff_sc_y0 * static_cast<s32>(scale)), 0, static_cast<LONG>(rt->phys_h));
    sc.right  = std::clamp<LONG>(static_cast<LONG>((eff_sc_x1 + 1) * static_cast<s32>(scale)), sc.left, static_cast<LONG>(rt->phys_w));
    sc.bottom = std::clamp<LONG>(static_cast<LONG>((eff_sc_y1 + 1) * static_cast<s32>(scale)), sc.top, static_cast<LONG>(rt->phys_h));
    if (sc.left >= sc.right || sc.top >= sc.bottom) return;

    const bool eff_bilinear = desc.use_bilinear && (tex_mode == 0u);
    const int samp_idx = (eff_bilinear ? 4 : 0) | ((wms == 1u || tex_is_rt_actual) ? 2 : 0) | ((wmt == 1u || tex_is_rt_actual) ? 1 : 0);
    ID3D11SamplerState* samp = m_samplers[static_cast<std::size_t>(samp_idx)];

    const bool need_alpha_ps = eff_ate || date || (desc.scanmsk >= 2u);
    ID3D11PixelShader* ps = need_alpha_ps ? m_ps_alpha : m_ps_opaque;

    auto issue_pass = [&](u32 afail_invert, bool write_color, bool write_alpha, bool write_zb) {
        bool premul_c = false;
        ID3D11BlendState* bs = get_blend_state_locked(
            abe, spec_a, spec_b, spec_c, spec_d, eff_fbmsk, write_color, write_alpha, premul_c);
        ID3D11DepthStencilState* ds = get_depth_state_locked(zte && zb != nullptr, ztst, write_zb);

        GsDrawConstantBuffer cb{};
        cb.rt_inv_size[0] = 2.0f / static_cast<float>(rt->logical_w);
        cb.rt_inv_size[1] = 2.0f / static_cast<float>(rt->logical_h);
        cb.rt_inv_size[2] = static_cast<float>(rt->logical_w);
        cb.rt_inv_size[3] = static_cast<float>(rt->logical_h);
        cb.tex_params[0]  = static_cast<float>(tw);
        cb.tex_params[1]  = static_cast<float>(eff_th);
        cb.tex_params[2]  = 1.0f / static_cast<float>(src_tex_w);
        cb.tex_params[3]  = 1.0f / static_cast<float>(src_tex_h);
        cb.clamp_bounds[0] = static_cast<float>(minu);
        cb.clamp_bounds[1] = static_cast<float>(maxu);
        cb.clamp_bounds[2] = static_cast<float>(minv);
        cb.clamp_bounds[3] = static_cast<float>(maxv);
        cb.misc[0] = static_cast<float>(aref);
        cb.misc[1] = static_cast<float>(fix) / 128.0f;
        cb.misc[2] = static_cast<float>(desc.texa_ta0) / 255.0f;
        cb.misc[3] = desc.texa_aem ? 1.0f : 0.0f;
        cb.flags0[0] = (tme && tex_srv != nullptr) ? 1u : 0u;
        cb.flags0[1] = tfx;
        cb.flags0[2] = tcc;
        cb.flags0[3] = eff_bilinear ? 1u : 0u;
        cb.flags1[0] = wms;
        cb.flags1[1] = wmt;
        cb.flags1[2] = eff_ate ? atst : 1u;
        cb.flags1[3] = afail_invert;
        cb.flags2[0] = spec_c;
        cb.flags2[1] = desc.fba ? 1u : 0u;
        cb.flags2[2] = (tex_mode == 0u && desc.tex_psm24_alpha) ? 1u : 0u;
        cb.flags2[3] = (date && date_srv != nullptr) ? (datm + 1u) : 0u;
        cb.flags3[0] = premul_c ? 1u : 0u;
        cb.flags3[1] = desc.scanmsk;
        cb.flags3[2] = tex_is_rt_actual ? 1u : 0u;
        cb.flags3[3] = desc.pabe ? 1u : 0u;
        cb.extra[0]  = 1.0f / static_cast<float>(scale);
        cb.extra[1]  = z_max_f;
        cb.flags4[0] = tex_mode;

        D3D11_MAPPED_SUBRESOURCE mapped{};
        if (SUCCEEDED(m_ctx->Map(m_cb, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped))) {
            if (mapped.pData != nullptr) {
                std::memcpy(mapped.pData, &cb, sizeof(cb));
            }
            m_ctx->Unmap(m_cb, 0);
        }

        static constexpr float kZeroBF[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        m_ctx->OMSetRenderTargets(1, &rt->rtv, (zte && zb) ? zb->dsv : nullptr);
        m_ctx->RSSetViewports(1, &vp);
        m_ctx->RSSetScissorRects(1, &sc);
        m_ctx->RSSetState(m_rs_scissor);
        m_ctx->OMSetBlendState(bs, kZeroBF, 0xFFFFFFFFu);
        m_ctx->OMSetDepthStencilState(ds, 0);
        m_ctx->IASetInputLayout(m_input_layout);
        m_ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        m_ctx->VSSetShader(m_vs, nullptr, 0);
        m_ctx->VSSetConstantBuffers(0, 1, &m_cb);
        m_ctx->PSSetShader(ps, nullptr, 0);
        m_ctx->PSSetConstantBuffers(0, 1, &m_cb);
        ID3D11ShaderResourceView* srvs[3] = { tex_srv, date_srv, clut_srv };
        m_ctx->PSSetShaderResources(0, 3, srvs);
        m_ctx->PSSetSamplers(0, 1, &samp);

        upload_and_draw_verts_locked(draw_verts, vertex_count);
    };

    // Pass 1: pixels passing alpha test (normal FB + ZB write)
    if (!eff_ate || atst != 0u) {
        issue_pass(0u, true, true, desc.z_write_en);
    }
    // Pass 2: AFAIL != KEEP (FB_ONLY=1, ZB_ONLY=2, RGB_ONLY=3)
    if (eff_ate && afail != 0u) {
        if (afail == 1u)      issue_pass(1u, true, true, false);
        else if (afail == 2u) issue_pass(1u, false, false, desc.z_write_en);
        else                  issue_pass(1u, true, false, false);
    }

    // Unbind SRVs so subsequent draws can freely bind any RT/ZB without D3D11 hazard
    ID3D11ShaderResourceView* null_srvs[3] = { nullptr, nullptr, nullptr };
    m_ctx->PSSetShaderResources(0, 3, null_srvs);
    m_ctx->OMSetRenderTargets(0, nullptr, nullptr);
    if (m_ui_waiters != nullptr && ++m_draws_since_flush >= 2048u) {
        should_throttle = true;
    }
    }
    if (should_throttle) {
        throttle_gpu_frame_queue();
    }
}

bool D3D11GsRenderer::present_dispfb(u32 fbp, u32 fbw, u32 psm, u32 dbx, u32 dby, u32 src_w, u32 src_h,
                                     u64 presented_seq) {
    if (!m_ready || !m_ctx_mtx) return false;
    fbp &= 0x1FFu;

    const u64 now_us = static_cast<u64>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
    const bool interactive_coalesce = (m_ui_waiters != nullptr);
    bool did_publish = false;

    {
        auto lk = acquire_ctx_lock();
        const GpuRt& rt = m_rts[fbp];
        const bool compat = rt.valid && rt.tex && (fbw == 0u || rt.fbw == fbw) &&
                            (rt.psm == psm || (rt.psm <= 1u && psm <= 1u));
        if (!compat) {
            std::lock_guard<std::mutex> pmlk(m_present_meta_mtx);
            m_presented_is_gpu = false;
            m_presented_seq = presented_seq;
            return false;
        }

        const u32 scale = m_active_scale;
        const u32 copy_x = std::min<u32>(dbx * scale, rt.phys_w > 0u ? rt.phys_w - 1u : 0u);
        const u32 copy_y = std::min<u32>(dby * scale, rt.phys_h > 0u ? rt.phys_h - 1u : 0u);
        const u32 eff_w  = std::max<u32>(1u, std::min<u32>(src_w * scale, rt.phys_w - copy_x));
        const u32 eff_h  = std::max<u32>(1u, std::min<u32>(src_h * scale, rt.phys_h - copy_y));

        const bool same_surface = (m_presented_tex != nullptr && m_presented_srv != nullptr &&
                                   m_presented_tex_w == eff_w && m_presented_tex_h == eff_h &&
                                   m_last_publish_fbp == fbp && m_last_publish_w == eff_w &&
                                   m_last_publish_h == eff_h && m_last_publish_dbx == copy_x &&
                                   m_last_publish_dby == copy_y);
        // Coalesce redundant GPU-to-GPU copies during >125 Hz uncapped loading (zero delay on 60 Hz gameplay!)
        const bool skip_copy = interactive_coalesce && same_surface &&
                               m_last_publish_us != 0u && (now_us - m_last_publish_us) < 8000u;

        if (!skip_copy) {
            if (!m_presented_tex || m_presented_tex_w != eff_w || m_presented_tex_h != eff_h) {
                ID3D11Texture2D* new_tex = nullptr;
                ID3D11ShaderResourceView* new_srv = nullptr;
                D3D11_TEXTURE2D_DESC td{};
                td.Width = eff_w;
                td.Height = eff_h;
                td.MipLevels = 1;
                td.ArraySize = 1;
                td.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
                td.SampleDesc.Count = 1;
                td.Usage = D3D11_USAGE_DEFAULT;
                td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
                if (FAILED(m_device->CreateTexture2D(&td, nullptr, &new_tex)) || !new_tex) {
                    return false;
                }
                if (FAILED(m_device->CreateShaderResourceView(new_tex, nullptr, &new_srv)) || !new_srv) {
                    safe_release(new_tex);
                    return false;
                }

                // Swap m_presented_srv under m_present_meta_mtx BEFORE releasing old_srv (prevents UAF!)
                ID3D11ShaderResourceView* old_srv = nullptr;
                {
                    std::lock_guard<std::mutex> pmlk(m_present_meta_mtx);
                    old_srv = m_presented_srv;
                    m_presented_srv = new_srv;
                }
                safe_release(old_srv);
                safe_release(m_presented_tex);
                m_presented_tex = new_tex;
                m_presented_tex_w = eff_w;
                m_presented_tex_h = eff_h;
            }

            D3D11_BOX box{};
            box.left = copy_x;
            box.top = copy_y;
            box.front = 0;
            box.right = copy_x + eff_w;
            box.bottom = copy_y + eff_h;
            box.back = 1;
            m_ctx->CopySubresourceRegion(m_presented_tex, 0, 0, 0, 0, rt.tex, 0, &box);
            m_last_publish_us = now_us;
            m_last_publish_fbp = fbp;
            m_last_publish_w = eff_w;
            m_last_publish_h = eff_h;
            m_last_publish_dbx = copy_x;
            m_last_publish_dby = copy_y;
            did_publish = true;
        }

        {
            std::lock_guard<std::mutex> pmlk(m_present_meta_mtx);
            m_presented_is_gpu = true;
            m_presented_seq = presented_seq;
            m_pres_u0 = 0.0f;
            m_pres_v0 = 0.0f;
            m_pres_u1 = 1.0f;
            m_pres_v1 = 1.0f;
            m_pres_src_w = eff_w;
            m_pres_src_h = eff_h;
        }
    }

    // Cap GPU command queue at <=3 frames in flight in interactive mode (strictly outside *m_ctx_mtx!)
    if (interactive_coalesce && did_publish) {
        throttle_gpu_frame_queue();
    }
    return true;
}

bool D3D11GsRenderer::get_presented_srv(u64& in_out_seq, ID3D11ShaderResourceView*& out_srv,
                                        float& u0, float& v0, float& u1, float& v1,
                                        u32& out_w, u32& out_h) const {
    out_srv = nullptr;
    if (!m_ready) return false;
    std::lock_guard<std::mutex> pmlk(m_present_meta_mtx);
    if (!m_presented_is_gpu || !m_presented_srv || m_presented_seq == 0u || m_presented_seq == in_out_seq) {
        return false;
    }
    // AddRef while holding m_present_meta_mtx so concurrent resolution resize cannot free the SRV in flight!
    m_presented_srv->AddRef();
    out_srv = m_presented_srv;
    u0 = m_pres_u0;
    v0 = m_pres_v0;
    u1 = m_pres_u1;
    v1 = m_pres_v1;
    out_w = m_pres_src_w;
    out_h = m_pres_src_h;
    in_out_seq = m_presented_seq;
    return true;
}

bool D3D11GsRenderer::readback_presented_rgba(std::vector<u32>& out_rgba, u32& out_w, u32& out_h) {
    if (!m_ready || !m_ctx_mtx) return false;
    auto lk = acquire_ctx_lock();
    if (!m_presented_tex || m_presented_tex_w == 0u || m_presented_tex_h == 0u) return false;

    const u32 w = m_presented_tex_w;
    const u32 h = m_presented_tex_h;
    if (!m_staging_tex || m_staging_w != w || m_staging_h != h) {
        safe_release(m_staging_tex);
        D3D11_TEXTURE2D_DESC td{};
        td.Width = w;
        td.Height = h;
        td.MipLevels = 1;
        td.ArraySize = 1;
        td.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        td.SampleDesc.Count = 1;
        td.Usage = D3D11_USAGE_STAGING;
        td.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        if (FAILED(m_device->CreateTexture2D(&td, nullptr, &m_staging_tex)) || !m_staging_tex) return false;
        m_staging_w = w;
        m_staging_h = h;
    }

    m_ctx->CopyResource(m_staging_tex, m_presented_tex);
    D3D11_MAPPED_SUBRESOURCE mapped{};
    if (FAILED(m_ctx->Map(m_staging_tex, 0, D3D11_MAP_READ, 0, &mapped))) return false;
    if (mapped.pData == nullptr) {
        m_ctx->Unmap(m_staging_tex, 0);
        return false;
    }

    out_rgba.resize(static_cast<std::size_t>(w) * h);
    const std::size_t row_bytes = static_cast<std::size_t>(w) * 4u;
    const u8* src = static_cast<const u8*>(mapped.pData);
    for (u32 y = 0; y < h; ++y) {
        u32* dst_row = out_rgba.data() + static_cast<std::size_t>(y) * w;
        std::memcpy(dst_row, src + static_cast<std::size_t>(y) * mapped.RowPitch, row_bytes);
        for (u32 x = 0; x < w; ++x) dst_row[x] |= 0xFF000000u;
    }
    m_ctx->Unmap(m_staging_tex, 0);
    out_w = w;
    out_h = h;
    return true;
}

#else // !_WIN32

void D3D11GsRenderer::shutdown() { m_ready = false; }
void D3D11GsRenderer::set_resolution_scale(u32 scale) {
    m_res_scale.store(std::clamp<u32>(scale, 1u, 4u), std::memory_order_release);
}
bool D3D11GsRenderer::readback_presented_rgba(std::vector<u32>&, u32&, u32&) { return false; }
void D3D11GsRenderer::on_reset() {}
void D3D11GsRenderer::on_vram_write(u32, u32) {}
bool D3D11GsRenderer::is_rt_resident(u32) const { return false; }
bool D3D11GsRenderer::is_zb_resident(u32) const { return false; }
void D3D11GsRenderer::clear_depth(u32, u32, u32, f32, s32, s32, s32, s32) {}
void D3D11GsRenderer::draw_triangles(const GsHwDrawDesc&, const GsHwVertex*, u32) {}
bool D3D11GsRenderer::present_dispfb(u32, u32, u32, u32, u32, u32, u32, u64) { return false; }

#endif

} // namespace tf::ps2
