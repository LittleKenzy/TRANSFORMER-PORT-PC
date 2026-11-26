#pragma once

#include "../../Runtime/GS/IGsHwBackend.h"
#include <array>
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
struct ID3D11Device;
struct ID3D11DeviceContext;
struct ID3D11Texture2D;
struct ID3D11RenderTargetView;
struct ID3D11DepthStencilView;
struct ID3D11ShaderResourceView;
struct ID3D11Buffer;
struct ID3D11InputLayout;
struct ID3D11VertexShader;
struct ID3D11PixelShader;
struct ID3D11SamplerState;
struct ID3D11RasterizerState;
struct ID3D11BlendState;
struct ID3D11DepthStencilState;
struct ID3D11Query;
#endif

namespace tf::ps2 {

// Direct3D 11 Hardware GS Renderer sharing Win32Presenter's ID3D11Device/Context.
// Renders PS2 3D scenes, bloom/RT-to-texture passes, and UI directly on GPU at 1x..4x internal resolution,
// presenting straight from GPU VRAM to the D3D11 Flip-Model swapchain with zero CPU framebuffer upload.
class D3D11GsRenderer final : public IGsHwBackend {
public:
    D3D11GsRenderer() = default;
    ~D3D11GsRenderer() override { shutdown(); }

    D3D11GsRenderer(const D3D11GsRenderer&) = delete;
    D3D11GsRenderer& operator=(const D3D11GsRenderer&) = delete;

#ifdef _WIN32
    bool init(ID3D11Device* device, ID3D11DeviceContext* ctx, std::mutex* ctx_mtx,
              std::atomic<u32>* ui_waiters = nullptr);
    // Returns a COM-AddRef'd SRV when returning true; caller MUST call out_srv->Release().
    bool get_presented_srv(u64& in_out_seq, ID3D11ShaderResourceView*& out_srv,
                           float& u0, float& v0, float& u1, float& v1,
                           u32& out_w, u32& out_h) const;
#endif
    void shutdown();
    void set_resolution_scale(u32 scale);
    u32 resolution_scale() const { return m_res_scale.load(std::memory_order_acquire); }
    bool is_ready() const { return m_ready; }

    // Staging-texture readback of the latest GPU-presented frame (for headless GPU visual verification/tests)
    bool readback_presented_rgba(std::vector<u32>& out_rgba, u32& out_w, u32& out_h);

    // IGsHwBackend implementation
    void on_reset() override;
    void on_vram_write(u32 start_block, u32 num_blocks) override;
    bool is_rt_resident(u32 fbp) const override;
    bool is_zb_resident(u32 zbp) const override;
    void clear_depth(u32 zbp, u32 width, u32 height, f32 z_norm,
                     s32 sc_x0, s32 sc_y0, s32 sc_x1, s32 sc_y1) override;
    void draw_triangles(const GsHwDrawDesc& desc, const GsHwVertex* verts, u32 vertex_count) override;
    bool present_dispfb(u32 fbp, u32 fbw, u32 psm, u32 dbx, u32 dby, u32 src_w, u32 src_h,
                        u64 presented_seq) override;

private:
#ifdef _WIN32
    struct GpuRt {
        ID3D11Texture2D* tex = nullptr;
        ID3D11RenderTargetView* rtv = nullptr;
        ID3D11ShaderResourceView* srv = nullptr;
        u32 logical_w = 0;
        u32 logical_h = 0;
        u32 phys_w = 0;
        u32 phys_h = 0;
        u32 fbw = 0;
        u32 psm = 0;
        bool valid = false;
    };

    struct GpuZb {
        ID3D11Texture2D* tex = nullptr;
        ID3D11DepthStencilView* dsv = nullptr;
        ID3D11ShaderResourceView* srv = nullptr;
        u32 logical_w = 0;
        u32 logical_h = 0;
        u32 phys_w = 0;
        u32 phys_h = 0;
        u32 z_psm = 0;
        bool valid = false;
    };

    struct GpuTexSlot {
        ID3D11Texture2D* tex = nullptr;
        ID3D11ShaderResourceView* srv = nullptr;
        u32 w = 0;
        u32 h = 0;
        u64 uploaded_gen = 0;
    };

    GpuRt* ensure_rt_locked(u32 fbp, u32 fbw, u32 psm, u32 logical_h, u32 scale,
                            const u32* seed_rgba, u32 seed_stride, u32 seed_h, bool will_overwrite_all);
    GpuZb* ensure_zb_locked(u32 zbp, u32 logical_w, u32 logical_h, u32 scale);
    ID3D11ShaderResourceView* ensure_tex_slot_locked(u32 slot, u64 gen, u32 tw, u32 th, const u32* rgba);
    ID3D11ShaderResourceView* ensure_clut_lut_locked(const u32* clut_rgba, u64 clut_hash);
    ID3D11ShaderResourceView* copy_zb_to_scratch_locked(const GpuZb& zb);
    ID3D11ShaderResourceView* copy_rt_to_scratch_locked(const GpuRt& rt, ID3D11Texture2D*& dst_tex,
                                                        ID3D11ShaderResourceView*& dst_srv,
                                                        u32& dst_w, u32& dst_h);
    void blit_seed_to_rt_locked(GpuRt& rt, const u32* rgba, u32 stride, u32 h);
    void release_all_targets_locked();
    ID3D11BlendState* get_blend_state_locked(bool abe, u32 spec_a, u32 spec_b, u32 spec_c, u32 spec_d,
                                             u32 fbmsk, bool write_color, bool write_alpha, bool& out_premul_c);
    ID3D11DepthStencilState* get_depth_state_locked(bool zte, u32 ztst, bool write_zb);
    void upload_and_draw_verts_locked(const GsHwVertex* verts, u32 vertex_count);
    void yield_for_ui_waiter() const;
    std::unique_lock<std::mutex> acquire_ctx_lock() const;
    void throttle_gpu_frame_queue();

    ID3D11Device*        m_device = nullptr;
    ID3D11DeviceContext* m_ctx = nullptr;
    std::mutex*          m_ctx_mtx = nullptr;
    std::atomic<u32>*    m_ui_waiters = nullptr;

    // 3-frame GPU completion fence ring (prevents unbounded WDDM queue buildup when backgrounded/loading
    // without causing CPU-GPU lockstep stalls at 3x HD 60-120 FPS)
    std::array<ID3D11Query*, 3> m_frame_queries{};
    std::array<bool, 3>         m_query_pending{};
    u32                         m_query_cursor = 0;
    u32                         m_draws_since_flush = 0;
    u64                         m_last_publish_us = 0;
    u32                         m_last_publish_fbp = 0xFFFFFFFFu;
    u32                         m_last_publish_w = 0;
    u32                         m_last_publish_h = 0;
    u32                         m_last_publish_dbx = 0;
    u32                         m_last_publish_dby = 0;

    ID3D11VertexShader*  m_vs = nullptr;
    ID3D11PixelShader*   m_ps_opaque = nullptr;
    ID3D11PixelShader*   m_ps_alpha = nullptr;
    ID3D11VertexShader*  m_seed_vs = nullptr;
    ID3D11PixelShader*   m_seed_ps = nullptr;
    ID3D11InputLayout*   m_input_layout = nullptr;
    ID3D11Buffer*        m_vb = nullptr;
    ID3D11Buffer*        m_cb = nullptr;
    u32                  m_vb_capacity = 65536;
    u32                  m_vb_offset = 0;

    // 8 samplers: [bilinear 0..1][clampU 0..1][clampV 0..1]
    std::array<ID3D11SamplerState*, 8> m_samplers{};
    ID3D11RasterizerState* m_rs_scissor = nullptr;
    ID3D11RasterizerState* m_rs_noscissor = nullptr;
    ID3D11BlendState*      m_bs_opaque_all = nullptr;
    ID3D11DepthStencilState* m_ds_disabled = nullptr;

    std::unordered_map<u64, ID3D11BlendState*> m_blend_cache;
    std::unordered_map<u32, ID3D11DepthStencilState*> m_depth_cache;

    std::array<GpuRt, 512> m_rts{};
    std::array<GpuZb, 512> m_zbs{};
    static constexpr std::size_t kGpuTexSlots = 513; // 0..511 match GsTexCache, 512 = CPU-RT fallback
    std::array<GpuTexSlot, kGpuTexSlots> m_tex_slots{};

    // Scratch textures for RT self-blit, DATE copy, CPU RT seed upload, and readback
    ID3D11Texture2D*          m_self_blit_tex = nullptr;
    ID3D11ShaderResourceView* m_self_blit_srv = nullptr;
    u32 m_self_blit_w = 0;
    u32 m_self_blit_h = 0;

    ID3D11Texture2D*          m_date_tex = nullptr;
    ID3D11ShaderResourceView* m_date_srv = nullptr;
    u32 m_date_w = 0;
    u32 m_date_h = 0;

    ID3D11Texture2D*          m_zb_copy_tex = nullptr;
    ID3D11ShaderResourceView* m_zb_copy_srv = nullptr;
    u32 m_zb_copy_w = 0;
    u32 m_zb_copy_h = 0;

    ID3D11Texture2D*          m_clut_tex = nullptr;
    ID3D11ShaderResourceView* m_clut_srv = nullptr;
    u64 m_clut_uploaded_hash = 0;

    ID3D11Texture2D*          m_seed_tex = nullptr;
    ID3D11ShaderResourceView* m_seed_srv = nullptr;
    u32 m_seed_w = 0;
    u32 m_seed_h = 0;

    ID3D11Texture2D*          m_staging_tex = nullptr;
    u32 m_staging_w = 0;
    u32 m_staging_h = 0;

    // Dedicated GPU-to-GPU presented surface handed directly to Win32Presenter::present_d3d11_srv
    ID3D11Texture2D*          m_presented_tex = nullptr;
    ID3D11ShaderResourceView* m_presented_srv = nullptr;
    u32 m_presented_tex_w = 0;
    u32 m_presented_tex_h = 0;

    mutable std::mutex m_present_meta_mtx;
    bool  m_presented_is_gpu = false;
    u64   m_presented_seq = 0;
    float m_pres_u0 = 0.0f;
    float m_pres_v0 = 0.0f;
    float m_pres_u1 = 1.0f;
    float m_pres_v1 = 1.0f;
    u32   m_pres_src_w = 0;
    u32   m_pres_src_h = 0;
#endif
    std::atomic<u32> m_res_scale{1};
    u32 m_active_scale = 1;
    bool m_ready = false;
};

} // namespace tf::ps2
