#pragma once

#include "../../Recompiler/Common/Types.h"
#include <vector>

namespace tf::ps2 {

// Platform-agnostic hardware GS vertex (exact PS2 12.4 screen-space coords + perspective STQ)
struct GsHwVertex {
    f32 x = 0.0f;  // x4 / 16.0f
    f32 y = 0.0f;  // y4 / 16.0f
    f32 z = 0.0f;  // Normalized [0.0 .. 1.0] = (z & z_mask) / double(z_mask)
    u32 rgba = 0;  // Packed RGBA8888 (R in low byte, A in high byte)
    f32 s = 0.0f;  // Perspective S (or u / tw when FST=1)
    f32 t = 0.0f;  // Perspective T (or v / th when FST=1)
    f32 q = 1.0f;  // Perspective Q (1.0f when FST=1)
};

// Complete PS2 GS draw-state snapshot for a hardware triangle/sprite/line batch
struct GsHwDrawDesc {
    // Destination Render Target & Z-Buffer
    u32 fbp = 0;
    u32 fbw = 10;       // in 64-pixel pages (logical RT width = fbw * 64)
    u32 fb_psm = 0;     // 0=PSMCT32, 1=PSMCT24, 2=PSMCT16, 0xA=PSMCT16S
    u32 fb_height = 448;
    u32 fbmsk = 0;
    bool fba = false;

    u32 zbp = 0;
    u32 z_psm = 0;
    bool z_write_en = true; // !(ZBUF.ZMSK)

    // Scissor (inclusive PS2 pixel coordinates)
    s32 scissor_x0 = 0;
    s32 scissor_y0 = 0;
    s32 scissor_x1 = 639;
    s32 scissor_y1 = 447;

    // GS state registers
    u32 prim_attrs = 0; // IIP(3), TME(4), FGE(5), ABE(6), AA1(7), FST(8)
    u64 test_reg = 0;   // ATE/ATST/AREF/AFAIL/DATE/DATM/ZTE/ZTST
    u64 alpha_reg = 0;  // A/B/C/D/FIX
    u64 tex0_reg = 0;
    u64 tex1_reg = 0;
    u64 clamp_reg = 0;
    bool pabe = false;
    u32 scanmsk = 0;
    bool use_bilinear = false;

    // Texture source (valid when TME=1)
    bool tex_is_rt = false;
    u32  tex_rt_fbp = 0;
    u32  tex_rt_fbw = 0;
    u32  tex_rt_height = 0;
    u32  tex_rt_clut_mode = 0; // 0=RGBA, 1=PSMT8H, 2=PSMT4HL, 3=PSMT4HH
    const u32* tex_clut_rgba = nullptr;
    u64  tex_clut_hash = 0;
    bool tex_is_zb = false;
    u32  tex_zb_zbp = 0;
    bool tex_psm24_alpha = false;
    u8   texa_ta0 = 0x80;
    bool texa_aem = false;

    // Decoded linear GsTexCache slot (valid when TME=1 && !tex_is_rt)
    u32  tex_cache_slot = 0;
    u64  tex_cache_gen = 0;
    u32  tex_w = 0;
    u32  tex_h = 0;
    const u32* tex_rgba = nullptr;

    // Optional CPU framebuffer seed when a GPU RT is first created/revalidated after CPU VRAM writes
    const u32* seed_rt_rgba = nullptr;
    u32 seed_rt_stride = 0;
    u32 seed_rt_height = 0;
    bool will_overwrite_all = false;
};

// Pure platform-agnostic interface for GPU-accelerated PS2 GS rendering (e.g. D3D11GsRenderer on Windows).
// Keeps Runtime/GS/ free of OS/GPU headers so Android ARM64, unit tests, and --headless builds stay clean.
class IGsHwBackend {
public:
    virtual ~IGsHwBackend() = default;

    virtual void on_reset() = 0;
    virtual void on_vram_write(u32 start_block, u32 num_blocks) = 0;
    virtual bool is_rt_resident(u32 fbp) const = 0;
    virtual bool is_zb_resident(u32 zbp) const { (void)zbp; return false; }

    // Clear GPU depth buffer when PS2 game clears Z via flat sprite to FRAME.FBP == ZBUF.ZBP
    virtual void clear_depth(u32 zbp, u32 width, u32 height, f32 z_norm,
                             s32 sc_x0, s32 sc_y0, s32 sc_x1, s32 sc_y1) = 0;

    // Submit a batch of triangles (sprites & lines are pre-expanded to triangle lists by GsState)
    virtual void draw_triangles(const GsHwDrawDesc& desc, const GsHwVertex* verts, u32 vertex_count) = 0;

    // GPU-to-GPU copy of displayed DISPFB render target into presented SRV.
    // Returns true if fbp was GPU-resident and published; false if GsState should fall back to CPU present.
    virtual bool present_dispfb(u32 fbp, u32 fbw, u32 psm, u32 dbx, u32 dby, u32 src_w, u32 src_h,
                                u64 presented_seq) = 0;
};

} // namespace tf::ps2
