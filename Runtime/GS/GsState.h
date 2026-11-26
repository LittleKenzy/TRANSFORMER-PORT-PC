#pragma once

#include "../../Recompiler/Common/Types.h"
#include "IGsHwBackend.h"
#include <array>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace tf::ps2 {

constexpr u32 kGsVramSize = 4U * 1024U * 1024U; // 4 MiB GS Local Memory

struct GsVertex {
    s32 x = 0;  // Integer pixel coordinate (x4 >> 4)
    s32 y = 0;  // Integer pixel coordinate (y4 >> 4)
    s32 x4 = 0; // Signed 12.4 fixed-point relative to XYOFFSET
    s32 y4 = 0;
    u32 z = 0;
    u8 r = 0, g = 0, b = 0, a = 255;
    f32 s = 0.0f, t = 0.0f, q = 1.0f;
    f32 u = 0.0f, v = 0.0f; // Fixed-point 10.4 UV converted to texel coordinates
};

struct GsSurface {
    u32 width = 0;
    u32 height = 0;
    u32 psm = 0;
    // For PSMCT32/24/16: RGBA8888 pixels. For PSMT8/4: palette index in low byte.
    std::vector<u32> pixels;
};

struct GsContext {
    s32 xyoffset_x = 0;
    s32 xyoffset_y = 0;
    s32 scissor_x0 = 0;
    s32 scissor_x1 = 639;
    s32 scissor_y0 = 0;
    s32 scissor_y1 = 447;
    u64 frame_reg = 0;
    u64 zbuf_reg = 0;
    u64 test_reg = 0;
    u64 alpha_reg = 0;
    u64 tex0_reg = 0;
    u64 tex1_reg = 0;
    u64 clamp_reg = 0;
    u32 active_draw_fbp = 0;
    bool fba = false;
};

struct GsState {
    GsState();
    ~GsState();
    GsState(const GsState&) = delete;
    GsState& operator=(const GsState&) = delete;
    void reset();

    // Privileged registers (0x12000000 .. 0x12001080)
    u64 read_priv_reg(u32 paddr) const;
    void write_priv_reg(u32 paddr, u64 val);
    void write_priv_reg32(u32 addr, u32 val);

    // Internal GS registers written via GIF
    void write_internal_reg(u8 reg, u64 val);
    void write_hwreg_qwords(const u8* qwords, std::size_t qwc);
    void set_packed_q(u32 qbits);

    // RGBA8888 output framebuffer for native presentation (Direct3D/Vulkan/OpenGL ES/Win32)
    u32 fb_width() const { return m_fb_width; }
    u32 fb_height() const { return m_fb_height; }
    const std::vector<u32>& framebuffer_rgba() const {
        return m_has_presented_single_thread ? m_triple_fb[m_last_presented_slot] : m_present_fb;
    }
    const std::vector<u32>& last_nonblack_rgba() const { return m_last_nonblack_fb; }
    u64 presented_seq() const { return m_presented_seq.load(std::memory_order_acquire); }
    bool copy_presented_framebuffer(std::vector<u32>& out_rgba, u32& out_w, u32& out_h) const;
    bool copy_presented_framebuffer_if_newer(u64& in_out_seq, std::vector<u32>& out_rgba, u32& out_w, u32& out_h) const;
    void set_present_callback(std::function<void()> cb);
    void set_frame_pacing_callback(std::function<void()> cb);
    void set_hw_backend(IGsHwBackend* hw) { m_hw = hw; }
    IGsHwBackend* hw_backend() const { return m_hw; }
    void publish_direct_framebuffer_if_active();
    void present_fbp(u32 fbp);
    const std::vector<u8>& vram() const { return m_vram; }

    u64 primitives_drawn = 0;
    u64 textured_primitives = 0;
    u64 hwreg_uploads = 0;
    u64 pixels_drawn = 0;
    u64 frames_presented = 0;
    std::vector<std::string> debug_log;

private:
    struct GsFbTarget {
        std::vector<u32> pixels;
        u32 fbw = 0;
        u32 height = 0;
        u32 psm = 0;
        u32 dirty_min_y = 0xFFFFFFFFu;
        u32 dirty_max_y = 0;
        bool valid = false;
    };

    struct GsZTarget {
        std::vector<u32> z;
        u32 stride = 0;
        u32 height = 0;
    };

    // Decoded linear RGBA8888 texture cache entry (eliminates per-pixel VRAM swizzle + CLUT decode)
    struct GsTexCacheEntry {
        u64 key = 0;
        u32 texa_sig = 0;
        u64 clut_hash = 0;
        u32 last_vram_gen = 0;
        u32 first_page = 0;
        u32 num_pages = 0;
        u64 page_gen_hash = 0;
        u32 tw = 0;
        u32 th = 0;
        u64 last_use = 0;
        u64 gpu_gen = 0;
        bool valid = false;
        std::vector<u32> rgba;
    };

    struct LinearTexView {
        const u32* pixels = nullptr;
        u32 stride = 0;
        u32 height = 0;
        bool apply_psm24_alpha = false;
        bool is_rt = false;
        u32 cache_slot = 0;
        u64 cache_gen = 0;
        u32 rt_fbp = 0;
        u32 rt_fbw = 0;
        u32 rt_clut_mode = 0; // 0=RGBA, 1=PSMT8H, 2=PSMT4HL, 3=PSMT4HH
        bool is_zb = false;
        u32 zb_zbp = 0;
    };

    struct TriBatchItem {
        GsVertex v0;
        GsVertex v1;
        GsVertex v2;
        s32 min_x = 0;
        s32 max_x = 0;
        s32 min_y = 0;
        s32 max_y = 0;
    };

    void on_dispfb_written(int circuit);
    void present_dispfb(u64 dispfb, u64 display);
    void publish_present_fb_to_shared();
    void commit_back_buffer_to_ready(u32 w, u32 h);
    void notify_present_callbacks();
    bool has_color_upload_pages(u32 fbp, u32 fbw, u32 psm, u32 height) const;
    void mark_vram_pages_color_upload(u32 start_block, u32 num_blocks, u8 dbw_val);
    void populate_hw_draw_desc(const GsContext& c, u32 attrs, u32 eff_fbw, u32 eff_psm, u32 req_h,
                               bool use_bilinear, const LinearTexView& tex_view, const GsFbTarget* rt_src,
                               bool will_overwrite_all, GsHwDrawDesc& out_desc);

    void kick_vertex(u64 xyz_val, bool drawing_kick);
    void rasterize_line(const GsVertex& v0, const GsVertex& v1);
    void rasterize_sprite(const GsVertex& v0, const GsVertex& v1);
    void rasterize_triangle(const GsVertex& v0, const GsVertex& v1, const GsVertex& v2);
    void enqueue_triangle(const GsVertex& v0, const GsVertex& v1, const GsVertex& v2);
    void flush_tri_batch();
    struct GsTriPerfStats {
        u64 fast_tex_px = 0;
        u64 fast_flat_px = 0;
        u64 general_px = 0;
        u64 fast_tex_cnt = 0;
        u64 fast_flat_cnt = 0;
        u64 general_cnt = 0;
        u64 gen_date_cnt = 0;
        u64 gen_pabe_cnt = 0;
        u64 gen_afail_cnt = 0;
    };

    void rasterize_triangle_range(const GsContext& c, u32 attrs, GsFbTarget& rt, GsZTarget* zt,
                                  u32 z_mask, const LinearTexView& tex_view, const GsFbTarget* rt_src,
                                  GsZTarget* clear_zt, u32 clear_z_mask, bool mirror_pres, u32* pres_fb,
                                  const GsVertex& v0, const GsVertex& v1, const GsVertex& v2,
                                  s32 band_y0, s32 band_y1, u64& out_px,
                                  GsTriPerfStats* perf_stats = nullptr) const;
    void plot_pixel(const GsContext& c, GsFbTarget& rt, GsZTarget* zt, u32 z_mask,
                    s32 x, s32 y, u32 z, u8 r, u8 g, u8 b, u8 a);
    void handle_hwreg_write(u64 val);
    void exec_local_to_local_trx();

    // Persistent dynamic multi-band horizontal slice worker pool (up to 8 bands on >=12-thread CPUs)
    void ensure_workers_started();
    void dispatch_bands(void (*fn)(void*, int, int), void* ctx);

    // Texture cache & sampling
    void bump_vram_pages(u32 start_block, u32 num_blocks);
    u64 compute_page_gen_hash(u32 first_page, u32 num_pages) const;
    LinearTexView resolve_linear_texture(u64 tex0, const GsFbTarget* rt_src);
    u32 sample_texel_at(const GsContext& c, const GsFbTarget* rt_src, u32 ix, u32 iy) const;
    u32 sample_texture(const GsContext& c, const GsFbTarget* rt_src, f32 u_tex, f32 v_tex, bool bilinear) const;
    u32 apply_tex_func(const GsContext& c, u32 tex, u32 vr, u32 vg, u32 vb, u32 va) const;

    // CLUT cache (CSM1/CSM2, CLD 0..5, CSA 4-bit & 8-bit)
    void handle_tex0_clut_load(u64 tex0);
    void load_clut_from_vram(u32 cbp, u32 cpsm, u32 csm, u32 csa, bool is_8bit);
    void refresh_clut_rgba_from_raw();
    void update_clut_block_hash(u32 block_idx);
    u64  compute_active_clut_hash(u32 psm, u32 cpsm, u32 csa) const;
    u32 unpack_clut_raw(u32 raw, u32 cpsm) const;
    u32 unpack_psm16_texa(u16 c) const;

    // VRAM pixel helpers
    void write_vram_pixel(u32 bp, u32 bw, u32 psm, u32 x, u32 y, u32 raw_val);
    u32  read_vram_texel_raw(u32 bp, u32 bw, u32 psm, u32 x, u32 y) const;
    u32  read_vram_rgba(u32 bp, u32 bw, u32 psm, u32 x, u32 y) const;

    // Dirty-tracked linear framebuffer cache <-> 4 MiB VRAM coherence
    void invalidate_draw_cache();
    GsFbTarget* ensure_fb_target(u32 fbp, u32 fbw, u32 psm, u32 min_height, bool will_overwrite_all = false);
    GsZTarget*  ensure_z_target(u32 zbp, u32 stride, u32 height);
    const GsFbTarget* prepare_texture_source(u64 tex0);
    void mark_fb_dirty(u32 fbp, u32 y0, u32 y1_excl);
    void flush_fb_target(u32 fbp);
    void flush_overlapping_fbs(u32 start_block, u32 num_blocks, u32 exclude_fbp = 0xFFFFFFFFu);
    void invalidate_overlapping_fbs(u32 start_block, u32 num_blocks);

    // Internal 512-entry CLUT cache (256x32-bit or 512x16-bit)
    std::array<u32, 512> m_clut_raw{};
    std::array<u32, 512> m_clut_rgba{};
    std::array<u8, 512>  m_clut_slot_cpsm{};
    bool m_clut_explicit_cld = false;
    bool m_clut_fallback_valid = false;
    u32  m_clut_fb_cbp = 0;
    u32  m_clut_fb_cpsm = 0;
    u32  m_clut_fb_csm = 0;
    u32  m_clut_fb_csa = 0;
    bool m_clut_fb_is8 = false;
    u32  m_cbp0 = 0xFFFFFFFFu;
    u32  m_cbp1 = 0xFFFFFFFFu;
    u32  m_clut_gen = 1;
    std::array<u64, 32> m_clut_block_hash{};

    // VRAM page generation counters + 512-slot (128 sets x 4-way LRU) decoded texture cache
    static constexpr std::size_t kTexCacheSets = 128;
    static constexpr std::size_t kTexCacheWays = 4;
    static constexpr std::size_t kTexCacheSlots = kTexCacheSets * kTexCacheWays;
    std::array<u32, 512> m_vram_page_gen{};
    std::array<u8, 512>  m_vram_page_color_dbw{};
    u32 m_vram_write_gen = 1;
    u64 m_tex_cache_tick = 1;
    std::array<GsTexCacheEntry, kTexCacheSlots> m_tex_cache{};

    // Persistent scratch buffer for overlapping self-blits (avoids per-draw heap allocations)
    std::vector<u32> m_self_blit_scratch;

    // Fast draw-target cache across consecutive primitives in the same mesh
    GsFbTarget* m_cached_rt = nullptr;
    GsZTarget*  m_cached_zt = nullptr;
    u32 m_cached_rt_fbp = 0xFFFFFFFFu;
    u32 m_cached_rt_fbw = 0;
    u32 m_cached_rt_psm = 0;
    u32 m_cached_rt_min_h = 0;
    u32 m_cached_zt_zbp = 0xFFFFFFFFu;

    // TEXCLUT register (0x1C) for CSM2
    u32 m_texclut_cbw = 1;
    u32 m_texclut_cou = 0;
    u32 m_texclut_cov = 0;

    // Cached horizontal display scaling LUT for present_dispfb
    u32 m_scale_disp_w = 0;
    u32 m_scale_dbx = 0;
    u32 m_scale_buf_w = 0;
    std::vector<u16> m_scale_sx0;
    std::vector<u16> m_scale_sx1;
    std::vector<u16> m_scale_frac;

    // Privileged regs
    u64 m_pmode = 0;
    u64 m_smode2 = 0;
    u64 m_dispfb1 = 0;
    u64 m_display1 = 0;
    u64 m_dispfb2 = 0;
    u64 m_display2 = 0;
    u64 m_bgcolor = 0;
    u64 m_csr = 0;
    u64 m_imr = 0;

    // Internal drawing state
    u32 m_prim_reg = 0;
    u32 m_prmode_reg = 0;
    bool m_prmodecont = true; // 1=use PRIM attrs (default), 0=use PRMODE attrs
    u32 m_prim_type = 0;
    u32 active_prim_attrs() const { return m_prmodecont ? m_prim_reg : m_prmode_reg; }
    const GsContext& active_ctx() const { return m_ctx[(active_prim_attrs() >> 9) & 1u]; }
    u8  m_cur_r = 0, m_cur_g = 0, m_cur_b = 0, m_cur_a = 255;
    f32 m_cur_s = 0.0f, m_cur_t = 0.0f, m_cur_q = 1.0f;
    f32 m_cur_u = 0.0f, m_cur_v = 0.0f;
    std::array<GsContext, 2> m_ctx{};
    u8  m_texa_ta0 = 0x80;
    u8  m_texa_ta1 = 0x80;
    bool m_texa_aem = false;
    bool m_pabe = false;
    u32 m_scanmsk = 0;

    // Host <-> Local VRAM transfer state (BITBLTBUF / TRXPOS / TRXREG / TRXDIR / HWREG)
    u32 m_bitblt_sbp = 0;
    u32 m_bitblt_sbw = 1;
    u32 m_bitblt_spsm = 0;
    u32 m_bitblt_dbp = 0;
    u32 m_bitblt_dbw = 1;
    u32 m_bitblt_dpsm = 0;
    u32 m_trx_ssax = 0;
    u32 m_trx_ssay = 0;
    u32 m_trx_dsax = 0;
    u32 m_trx_dsay = 0;
    u32 m_trx_dir  = 0;
    u32 m_trx_rrw = 0;
    u32 m_trx_rrh = 0;
    u32 m_trx_x = 0;
    u32 m_trx_y = 0;
    bool m_trx_active = false;
    u8  m_rgb24_buf[3]{};
    u32 m_rgb24_cnt = 0;

    std::vector<GsVertex> m_vtx_queue;
    std::vector<u8> m_vram;

    u32 m_fb_width = 640;
    u32 m_fb_height = 448;
    std::array<GsFbTarget, 512> m_fb_targets{};
    std::vector<u32> m_valid_fbps;
    std::vector<u32> m_dirty_fbps;
    std::array<GsZTarget, 512> m_z_targets{};
    bool m_drew_to_disp_fbp = false;
    u32  m_last_presented_fbp = 0;
    u32  m_vblanks_since_present = 0;

    // Same-state triangle batch for 4-band lock-free parallel rasterization
    std::vector<TriBatchItem> m_tri_batch;
    s32 m_tri_batch_min_y = 0x7FFFFFFF;
    s32 m_tri_batch_max_y = -1;

    // Persistent dynamic multi-band thread pool (up to 8 bands; lazy-started on first parallel draw)
    static constexpr int kGsMaxBands = 8;
    int m_num_bands = 1;
    std::vector<std::thread> m_workers;
    bool m_workers_init_attempted = false;
    std::mutex m_pool_mtx;
    std::condition_variable m_pool_cv;
    std::atomic<bool> m_pool_stop{false};
    std::atomic<u64> m_job_gen{0};
    std::atomic<u32> m_workers_done{0};
    std::atomic<u32> m_workers_sleeping{0};
    void (*m_job_fn)(void*, int, int) = nullptr;
    void* m_job_ctx = nullptr;

    // Instant present wakeup & post-present frame pacing callbacks (invoked outside m_fb_mutex)
    mutable std::mutex m_cb_mutex;
    std::function<void()> m_present_cb;
    std::function<void()> m_pacing_cb;
    std::atomic<bool> m_has_callbacks{false};
    u64 m_last_present_cb_us = 0;

    // Optional hardware GS backend (e.g. D3D11GsRenderer; nullptr = verified CPU software path)
    IGsHwBackend* m_hw = nullptr;
    u64 m_hw_tex_gen = 1;
    std::vector<GsHwVertex> m_hw_verts;
    std::array<u32, 256> m_hw_clut_lut{};

    std::vector<u32> m_present_fb;
    std::vector<u32> m_last_nonblack_fb;

    // Triple-buffered presentation (5ns index swap under m_fb_mutex, zero memcpy while holding lock)
    std::array<std::vector<u32>, 3> m_triple_fb;
    int m_back_idx = 0;
    mutable int m_ready_idx = 1;
    mutable int m_front_idx = 2;
    int m_last_presented_slot = 0;
    bool m_has_presented_single_thread = false;
    mutable std::mutex m_reader_mutex;
    mutable std::mutex m_fb_mutex;
    mutable u32 m_ready_w = 0;
    mutable u32 m_ready_h = 0;
    mutable u64 m_ready_seq = 0;
    mutable u32 m_front_w = 0;
    mutable u32 m_front_h = 0;
    mutable u64 m_front_seq = 0;
    mutable bool m_has_new_ready = false;
    bool m_has_presented = false;
    std::atomic<u64> m_presented_seq{0};
};

} // namespace tf::ps2
