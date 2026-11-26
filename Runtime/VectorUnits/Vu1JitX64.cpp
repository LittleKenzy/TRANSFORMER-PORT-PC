#include "Vu1JitX64.h"
#include <algorithm>
#include <cstring>

#if defined(_WIN32) && (defined(__x86_64__) || defined(_M_X64))
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#define TF_VU1_JIT_X64_ENABLED 1
#else
#define TF_VU1_JIT_X64_ENABLED 0
#endif

namespace tf::ps2 {

#if TF_VU1_JIT_X64_ENABLED
namespace {

constexpr std::size_t kCodeBufSize = 4U * 1024U * 1024U; // 4 MiB RWX persistent JIT buffer
constexpr u32 kMaxBlockInsns = 64u;

alignas(16) constexpr u32 kJitDestMask4[16][4] = {
    {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x00000000u, 0x00000000u, 0x00000000u, 0xFFFFFFFFu},
    {0x00000000u, 0x00000000u, 0xFFFFFFFFu, 0x00000000u},
    {0x00000000u, 0x00000000u, 0xFFFFFFFFu, 0xFFFFFFFFu},
    {0x00000000u, 0xFFFFFFFFu, 0x00000000u, 0x00000000u},
    {0x00000000u, 0xFFFFFFFFu, 0x00000000u, 0xFFFFFFFFu},
    {0x00000000u, 0xFFFFFFFFu, 0xFFFFFFFFu, 0x00000000u},
    {0x00000000u, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu},
    {0xFFFFFFFFu, 0x00000000u, 0x00000000u, 0x00000000u},
    {0xFFFFFFFFu, 0x00000000u, 0x00000000u, 0xFFFFFFFFu},
    {0xFFFFFFFFu, 0x00000000u, 0xFFFFFFFFu, 0x00000000u},
    {0xFFFFFFFFu, 0x00000000u, 0xFFFFFFFFu, 0xFFFFFFFFu},
    {0xFFFFFFFFu, 0xFFFFFFFFu, 0x00000000u, 0x00000000u},
    {0xFFFFFFFFu, 0xFFFFFFFFu, 0x00000000u, 0xFFFFFFFFu},
    {0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0x00000000u},
    {0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu},
};

alignas(16) constexpr u32 kJitAbsMask[4] = {0x7FFFFFFFu, 0x7FFFFFFFu, 0x7FFFFFFFu, 0x7FFFFFFFu};
alignas(16) constexpr f32 kJitItofScale[4][4] = {
    {1.0f, 1.0f, 1.0f, 1.0f},
    {1.0f / 16.0f, 1.0f / 16.0f, 1.0f / 16.0f, 1.0f / 16.0f},
    {1.0f / 4096.0f, 1.0f / 4096.0f, 1.0f / 4096.0f, 1.0f / 4096.0f},
    {1.0f / 32768.0f, 1.0f / 32768.0f, 1.0f / 32768.0f, 1.0f / 32768.0f},
};

// Tiny x86-64/SSE2 machine code emitter
struct Emitter {
    u8* base = nullptr;
    std::size_t pos = 0;
    std::size_t cap = 0;
    bool overflow = false;

    void u8b(u8 v) {
        if (pos + 1 > cap) { overflow = true; return; }
        base[pos++] = v;
    }
    void u16b(u16 v) {
        if (pos + 2 > cap) { overflow = true; return; }
        std::memcpy(base + pos, &v, 2);
        pos += 2;
    }
    void u32b(u32 v) {
        if (pos + 4 > cap) { overflow = true; return; }
        std::memcpy(base + pos, &v, 4);
        pos += 4;
    }
    void s32b(s32 v) { u32b(static_cast<u32>(v)); }
    void u64b(u64 v) {
        if (pos + 8 > cap) { overflow = true; return; }
        std::memcpy(base + pos, &v, 8);
        pos += 8;
    }

    // Patch a 32-bit relative jump offset located at `patch_pos` to target `target_pos`
    void patch_rel32(std::size_t patch_pos, std::size_t target_pos) {
        if (overflow || patch_pos + 4 > cap) return;
        const s32 rel = static_cast<s32>(static_cast<std::ptrdiff_t>(target_pos) - static_cast<std::ptrdiff_t>(patch_pos + 4));
        std::memcpy(base + patch_pos, &rel, 4);
    }

    // SSE2 helpers: xmm0..xmm5 only (caller-saved on Windows x64!)
    void movups_xmm_rbx(u8 xmm, u32 disp) { u8b(0x0F); u8b(0x10); u8b(static_cast<u8>(0x83u | (xmm << 3))); u32b(disp); }
    void movups_rbx_xmm(u32 disp, u8 xmm) { u8b(0x0F); u8b(0x11); u8b(static_cast<u8>(0x83u | (xmm << 3))); u32b(disp); }
    void movups_xmm_rsi_rax(u8 xmm)       { u8b(0x0F); u8b(0x10); u8b(static_cast<u8>(0x04u | (xmm << 3))); u8b(0x06); }
    void movups_rsi_rax_xmm(u8 xmm)       { u8b(0x0F); u8b(0x11); u8b(static_cast<u8>(0x04u | (xmm << 3))); u8b(0x06); }
    void movaps_xmm_rsp(u8 xmm, u8 disp8) { u8b(0x0F); u8b(0x28); u8b(static_cast<u8>(0x44u | (xmm << 3))); u8b(0x24); u8b(disp8); }
    void movaps_rsp_xmm(u8 disp8, u8 xmm) { u8b(0x0F); u8b(0x29); u8b(static_cast<u8>(0x44u | (xmm << 3))); u8b(0x24); u8b(disp8); }
    void movaps_xmm_r13(u8 xmm, u32 disp) { u8b(0x41); u8b(0x0F); u8b(0x28); u8b(static_cast<u8>(0x85u | (xmm << 3))); u32b(disp); }
    void movaps_xmm_rax(u8 xmm)           { u8b(0x0F); u8b(0x28); u8b(static_cast<u8>(0x00u | (xmm << 3))); }
    void movss_xmm_rbx(u8 xmm, u32 disp)  { u8b(0xF3); u8b(0x0F); u8b(0x10); u8b(static_cast<u8>(0x83u | (xmm << 3))); u32b(disp); }
    void movaps_rr(u8 dst, u8 src)        { u8b(0x0F); u8b(0x28); u8b(static_cast<u8>(0xC0u | (dst << 3) | src)); }
    void shufps(u8 dst, u8 src, u8 imm8)  { u8b(0x0F); u8b(0xC6); u8b(static_cast<u8>(0xC0u | (dst << 3) | src)); u8b(imm8); }
    void addps(u8 dst, u8 src)            { u8b(0x0F); u8b(0x58); u8b(static_cast<u8>(0xC0u | (dst << 3) | src)); }
    void subps(u8 dst, u8 src)            { u8b(0x0F); u8b(0x5C); u8b(static_cast<u8>(0xC0u | (dst << 3) | src)); }
    void mulps(u8 dst, u8 src)            { u8b(0x0F); u8b(0x59); u8b(static_cast<u8>(0xC0u | (dst << 3) | src)); }
    void maxps(u8 dst, u8 src)            { u8b(0x0F); u8b(0x5F); u8b(static_cast<u8>(0xC0u | (dst << 3) | src)); }
    void minps(u8 dst, u8 src)            { u8b(0x0F); u8b(0x5D); u8b(static_cast<u8>(0xC0u | (dst << 3) | src)); }
    void andps(u8 dst, u8 src)            { u8b(0x0F); u8b(0x54); u8b(static_cast<u8>(0xC0u | (dst << 3) | src)); }
    void andnps(u8 dst, u8 src)           { u8b(0x0F); u8b(0x55); u8b(static_cast<u8>(0xC0u | (dst << 3) | src)); }
    void orps(u8 dst, u8 src)             { u8b(0x0F); u8b(0x56); u8b(static_cast<u8>(0xC0u | (dst << 3) | src)); }
    void cvtdq2ps(u8 dst, u8 src)         { u8b(0x0F); u8b(0x5B); u8b(static_cast<u8>(0xC0u | (dst << 3) | src)); }

    // Emit masked write of xmm0 into [rbx + dst_off] using xmm1, xmm2, xmm3 (clobbers xmm0..xmm3)
    void store_vec4_rbx_masked(u32 dst_off, u8 dest) {
        if (dest == 0u) return;
        if (dest == 0x0Fu) {
            movups_rbx_xmm(dst_off, 0);
            return;
        }
        movups_xmm_rbx(1, dst_off);             // xmm1 = old
        movaps_xmm_r13(2, static_cast<u32>(dest) * 16u); // xmm2 = mask
        movaps_rr(3, 2);                        // xmm3 = mask
        andps(0, 2);                            // xmm0 = new & mask
        andnps(3, 1);                           // xmm3 = (~mask) & old
        orps(0, 3);                             // xmm0 = merged
        movups_rbx_xmm(dst_off, 0);
    }

    // Emit masked write of xmm0 into [rsi + rax] (VU1 data memory) using xmm1..xmm3
    void store_vec4_mem_masked(u8 dest) {
        if (dest == 0u) return;
        if (dest == 0x0Fu) {
            movups_rsi_rax_xmm(0);
            return;
        }
        movups_xmm_rsi_rax(1);                  // xmm1 = old mem
        movaps_xmm_r13(2, static_cast<u32>(dest) * 16u);
        movaps_rr(3, 2);
        andps(0, 2);
        andnps(3, 1);
        orps(0, 3);
        movups_rsi_rax_xmm(0);
    }

    // Call absolute C++ function pointer via rax (caller must have set rcx/rdx/r8/r9)
    void call_abs(const void* fn) {
        u8b(0x48); u8b(0xB8); u64b(reinterpret_cast<u64>(fn)); // mov rax, imm64
        u8b(0xFF); u8b(0xD0);                                  // call rax
    }

    // Load/store 16-bit VI register
    void movzx_eax_vi(u32 vi_base, u8 reg) {
        u8b(0x0F); u8b(0xB7); u8b(0x83); u32b(vi_base + static_cast<u32>(reg & 15u) * 2u);
    }
    void movzx_ecx_vi(u32 vi_base, u8 reg) {
        u8b(0x0F); u8b(0xB7); u8b(0x8B); u32b(vi_base + static_cast<u32>(reg & 15u) * 2u);
    }
    void mov_vi_ax(u32 vi_base, u8 reg) {
        if ((reg & 15u) == 0u) return;
        u8b(0x66); u8b(0x89); u8b(0x83); u32b(vi_base + static_cast<u32>(reg & 15u) * 2u);
    }
    void mov_vi_imm16(u32 vi_base, u8 reg, u16 imm) {
        if ((reg & 15u) == 0u) return;
        u8b(0x66); u8b(0xC7); u8b(0x83); u32b(vi_base + static_cast<u32>(reg & 15u) * 2u); u16b(imm);
    }

    // Load/store 32-bit fields on VU1Unit (rbx)
    void mov_eax_rbx32(u32 off)        { u8b(0x8B); u8b(0x83); u32b(off); }
    void mov_rbx32_eax(u32 off)        { u8b(0x89); u8b(0x83); u32b(off); }
    void mov_rbx32_imm(u32 off, u32 v) { u8b(0xC7); u8b(0x83); u32b(off); u32b(v); }

    // Load/store scratch u32 on stack [rsp + disp8]
    void mov_rsp32_eax(u8 disp8)        { u8b(0x89); u8b(0x44); u8b(0x24); u8b(disp8); }
    void mov_rsp32_imm(u8 disp8, u32 v) { u8b(0xC7); u8b(0x44); u8b(0x24); u8b(disp8); u32b(v); }
    void mov_eax_rsp32(u8 disp8)        { u8b(0x8B); u8b(0x44); u8b(0x24); u8b(disp8); }
};

struct RawVu1Pair {
    u32 lower = 0;
    u32 upper = 0;
    u32 pc = 0;
    bool i_bit = false;
    bool e_bit = false;
    bool upper_nop = false;
    bool lower_nop = false;
    bool is_branch = false;
    bool is_cond_branch = false;
    bool is_vi_alu = false;
    u8   vi_alu_dst = 0;
    bool uses_pipes_lower = false; // DIV/SQRT/RSQRT/WAITQ/EFU/WAITP/FC*/FS*/FM*
};

RawVu1Pair inspect_pair(const u8* code_mem, u32 idx) {
    RawVu1Pair p{};
    p.pc = (idx & 0x7FFu) * 8u;
    p.lower = read_le32(code_mem + p.pc);
    p.upper = read_le32(code_mem + p.pc + 4u);
    p.i_bit = (p.upper & 0x80000000u) != 0;
    p.e_bit = (p.upper & 0x40000000u) != 0;
    p.upper_nop = ((p.upper & 0x07FFFFFFu) == 0x000002FFu);
    p.lower_nop = (!p.i_bit && p.lower == 0x8000033Cu);

    if (!p.i_bit && !p.lower_nop) {
        const u8 op = static_cast<u8>((p.lower >> 25) & 0x7Fu);
        const u8 it = static_cast<u8>((p.lower >> 16) & 0x1Fu);
        const u8 id = static_cast<u8>((p.lower >> 6) & 0x1Fu);
        if (op == 0x40) {
            const u8 fn = static_cast<u8>(p.lower & 0x3Fu);
            if (fn < 0x3C) {
                if (fn == 0x30 || fn == 0x31 || fn == 0x34 || fn == 0x35) {
                    p.is_vi_alu = true;
                    p.vi_alu_dst = id & 15u;
                } else if (fn == 0x32) {
                    p.is_vi_alu = true;
                    p.vi_alu_dst = it & 15u;
                }
            } else {
                const u32 sub = (((p.lower >> 6) & 0x1Fu) << 2) | (p.lower & 3u);
                if (sub == 0x34 || sub == 0x36) {
                    // LQI / LQD modify VI[is] via the integer unit and carry the branch VI-delay hazard.
                    p.is_vi_alu = true;
                    p.vi_alu_dst = static_cast<u8>((p.lower >> 11) & 15u);
                } else if (sub == 0x35 || sub == 0x37) {
                    // SQI / SQD modify VI[it] via the integer unit and carry the branch VI-delay hazard.
                    p.is_vi_alu = true;
                    p.vi_alu_dst = it & 15u;
                } else if (sub == 0x38 || sub == 0x39 || sub == 0x3A || sub == 0x3B || sub == 0x6C || sub >= 0x70) {
                    p.uses_pipes_lower = true;
                }
            }
        } else {
            if (op == 0x08 || op == 0x09) {
                p.is_vi_alu = true;
                p.vi_alu_dst = it & 15u;
            } else if (op >= 0x10 && op <= 0x1C) {
                p.uses_pipes_lower = true;
            } else if (op == 0x20 || op == 0x21 || op == 0x24 || op == 0x25) {
                p.is_branch = true;
            } else if (op == 0x28 || op == 0x29 || (op >= 0x2C && op <= 0x2F)) {
                p.is_branch = true;
                p.is_cond_branch = true;
            }
        }
    }
    return p;
}

// Returns true if upper writes a vf register (out_vf = 1..31), or acc (out_acc = true), or clipw (out_clip = true)
void classify_upper(u32 upper, u8& out_vf, bool& out_acc, bool& out_mac, bool& out_clip, bool& out_needs_helper) {
    out_vf = 0;
    out_acc = false;
    out_mac = false;
    out_clip = false;
    out_needs_helper = false;
    if ((upper & 0x07FFFFFFu) == 0x000002FFu) return;

    const u8 ft = static_cast<u8>((upper >> 16) & 0x1Fu);
    const u8 fd = static_cast<u8>((upper >> 6) & 0x1Fu);
    const u8 fn = static_cast<u8>(upper & 0x3Fu);

    if (fn >= 0x3C) {
        const u32 sub = (((upper >> 6) & 0x1Fu) << 2) | (upper & 3u);
        switch (sub) {
            case 0x00: case 0x01: case 0x02: case 0x03:
            case 0x04: case 0x05: case 0x06: case 0x07:
            case 0x08: case 0x09: case 0x0A: case 0x0B:
            case 0x0C: case 0x0D: case 0x0E: case 0x0F:
            case 0x18: case 0x19: case 0x1A: case 0x1B:
            case 0x1C: case 0x1E:
            case 0x20: case 0x21: case 0x22: case 0x23:
            case 0x24: case 0x25: case 0x26: case 0x27:
            case 0x28: case 0x29: case 0x2A: case 0x2C: case 0x2D:
                out_acc = true;
                out_mac = true;
                return;
            case 0x2E: // OPMULA
                out_acc = true;
                out_mac = true;
                out_needs_helper = true;
                return;
            case 0x10: case 0x11: case 0x12: case 0x13: // ITOF*
            case 0x1D: // ABS
                out_vf = ft;
                return;
            case 0x14: case 0x15: case 0x16: case 0x17: // FTOI*
                out_vf = ft;
                out_needs_helper = true;
                return;
            case 0x1F: // CLIPw
                out_clip = true;
                return;
            default:
                return;
        }
    }

    switch (fn) {
        case 0x00: case 0x01: case 0x02: case 0x03:
        case 0x04: case 0x05: case 0x06: case 0x07:
        case 0x08: case 0x09: case 0x0A: case 0x0B:
        case 0x0C: case 0x0D: case 0x0E: case 0x0F:
        case 0x18: case 0x19: case 0x1A: case 0x1B:
        case 0x1C: case 0x1E:
        case 0x20: case 0x21: case 0x22: case 0x23:
        case 0x24: case 0x25: case 0x26: case 0x27:
        case 0x28: case 0x29: case 0x2A: case 0x2C: case 0x2D:
            out_vf = fd;
            out_mac = true;
            return;
        case 0x2E: // OPMSUB
            out_vf = fd;
            out_mac = true;
            out_needs_helper = true;
            return;
        case 0x10: case 0x11: case 0x12: case 0x13:
        case 0x14: case 0x15: case 0x16: case 0x17:
        case 0x1D: case 0x1F: case 0x2B: case 0x2F:
            out_vf = fd;
            return;
        default:
            return;
    }
}

// Conservative check: could `lower` read or write `vf[target_vf]`?
bool lower_touches_vf(const RawVu1Pair& pair, u8 target_vf) {
    if (target_vf == 0u || pair.i_bit || pair.lower_nop) return false;
    const u8 op = static_cast<u8>((pair.lower >> 25) & 0x7Fu);
    const u8 it = static_cast<u8>((pair.lower >> 16) & 0x1Fu);
    const u8 is = static_cast<u8>((pair.lower >> 11) & 0x1Fu);
    if (op == 0x40) {
        const u8 fn = static_cast<u8>(pair.lower & 0x3Fu);
        if (fn < 0x3C) return false; // integer ALU only touches vi
        const u32 sub = (((pair.lower >> 6) & 0x1Fu) << 2) | (pair.lower & 3u);
        switch (sub) {
            case 0x30: case 0x31: // MOVE/MR32: reads is, writes it
                return (is == target_vf || it == target_vf);
            case 0x34: case 0x36: // LQI/LQD: writes it
            case 0x3D: case 0x40: case 0x41: case 0x64: // MFIR/RNEXT/RGET/MFP: writes it
                return (it == target_vf);
            case 0x35: case 0x37: // SQI/SQD: reads is
            case 0x3C: case 0x42: case 0x43: // MTIR/RINIT/RXOR: reads is
                return (is == target_vf);
            case 0x38: case 0x3A: // DIV/RSQRT: reads is, it
                return (is == target_vf || it == target_vf);
            case 0x39: // SQRT: reads it
                return (it == target_vf);
            case 0x70: case 0x71: case 0x72: case 0x73:
            case 0x74: case 0x75: case 0x76:
            case 0x78: case 0x79: case 0x7A: case 0x7B: case 0x7C: case 0x7D: case 0x7E:
                return (is == target_vf);
            default:
                return false;
        }
    }
    if (op == 0x00) return (it == target_vf); // LQ writes it
    if (op == 0x01) return (is == target_vf); // SQ reads is
    return false;
}

// Emits SSE2 computation of upper into xmm0 (does NOT write vf/acc yet).
// Returns false if upper is NOP or CLIPw (no xmm0 output).
bool emit_upper_compute_xmm0(Emitter& e, const Vu1JitOffsets& off, const Vu1JitHelpers& hlp, u32 upper, u8& out_dest) {
    if ((upper & 0x07FFFFFFu) == 0x000002FFu) return false;
    const u8 dest = static_cast<u8>((upper >> 21) & 0x0Fu);
    const u8 ft   = static_cast<u8>((upper >> 16) & 0x1Fu);
    const u8 fs   = static_cast<u8>((upper >> 11) & 0x1Fu);
    const u8 fn   = static_cast<u8>(upper & 0x3Fu);
    out_dest = dest;

    const u32 off_s = off.vf + static_cast<u32>(fs) * 16u;
    const u32 off_t = off.vf + static_cast<u32>(ft) * 16u;

    auto load_bc = [&](u8 xmm, u8 bc) {
        e.movss_xmm_rbx(xmm, off_t + static_cast<u32>(bc) * 4u);
        e.shufps(xmm, xmm, 0x00);
    };
    auto load_scalar_field = [&](u8 xmm, u32 field_off) {
        e.movss_xmm_rbx(xmm, field_off);
        e.shufps(xmm, xmm, 0x00);
    };

    if (fn >= 0x3C) {
        const u32 sub = (((upper >> 6) & 0x1Fu) << 2) | (upper & 3u);
        switch (sub) {
            case 0x00: case 0x01: case 0x02: case 0x03: // ADDAbc
                e.movups_xmm_rbx(0, off_s); load_bc(1, sub & 3u); e.addps(0, 1); return true;
            case 0x04: case 0x05: case 0x06: case 0x07: // SUBAbc
                e.movups_xmm_rbx(0, off_s); load_bc(1, sub & 3u); e.subps(0, 1); return true;
            case 0x08: case 0x09: case 0x0A: case 0x0B: // MADDAbc
                e.movups_xmm_rbx(0, off_s); load_bc(1, sub & 3u); e.mulps(0, 1);
                e.movups_xmm_rbx(1, off.acc); e.addps(0, 1); return true;
            case 0x0C: case 0x0D: case 0x0E: case 0x0F: // MSUBAbc
                e.movups_xmm_rbx(1, off_s); load_bc(2, sub & 3u); e.mulps(1, 2);
                e.movups_xmm_rbx(0, off.acc); e.subps(0, 1); return true;
            case 0x10: case 0x11: case 0x12: case 0x13: // ITOF0/4/12/15
                e.movups_xmm_rbx(0, off_s);
                e.cvtdq2ps(0, 0);
                if (sub != 0x10) {
                    e.u8b(0x48); e.u8b(0xB8); e.u64b(reinterpret_cast<u64>(&kJitItofScale[sub - 0x10][0]));
                    e.movaps_xmm_rax(1);
                    e.mulps(0, 1);
                }
                return true;
            case 0x14: case 0x15: case 0x16: case 0x17: // FTOI* -> helper writes to [rsp+0x20]
            case 0x2E: { // OPMULA -> helper writes to [rsp+0x20]
                // exec_upper_to_tmp(vu, upper, &tmp_vec, &tmp_dest)
                e.u8b(0x48); e.u8b(0x89); e.u8b(0xD9);             // mov rcx, rbx
                e.u8b(0xBA); e.u32b(upper);                        // mov edx, upper
                e.u8b(0x4C); e.u8b(0x8D); e.u8b(0x44); e.u8b(0x24); e.u8b(0x20); // lea r8, [rsp+0x20]
                e.u8b(0x4C); e.u8b(0x8D); e.u8b(0x4C); e.u8b(0x24); e.u8b(0x38); // lea r9, [rsp+0x38]
                e.call_abs(reinterpret_cast<const void*>(hlp.exec_upper_to_tmp));
                e.movaps_xmm_rsp(0, 0x20);
                if (sub == 0x2E) out_dest = 0xEu;
                return true;
            }
            case 0x18: case 0x19: case 0x1A: case 0x1B: // MULAbc
                e.movups_xmm_rbx(0, off_s); load_bc(1, sub & 3u); e.mulps(0, 1); return true;
            case 0x1C: // MULAq
                e.movups_xmm_rbx(0, off_s); load_scalar_field(1, off.q); e.mulps(0, 1); return true;
            case 0x1D: // ABS
                e.movups_xmm_rbx(0, off_s);
                e.u8b(0x48); e.u8b(0xB8); e.u64b(reinterpret_cast<u64>(&kJitAbsMask[0]));
                e.movaps_xmm_rax(1);
                e.andps(0, 1);
                return true;
            case 0x1E: // MULAi
                e.movups_xmm_rbx(0, off_s); load_scalar_field(1, off.i_reg); e.mulps(0, 1); return true;
            case 0x1F: // CLIPw (handled in post-step)
                return false;
            case 0x20: // ADDAq
                e.movups_xmm_rbx(0, off_s); load_scalar_field(1, off.q); e.addps(0, 1); return true;
            case 0x21: // MADDAq
                e.movups_xmm_rbx(0, off_s); load_scalar_field(1, off.q); e.mulps(0, 1);
                e.movups_xmm_rbx(1, off.acc); e.addps(0, 1); return true;
            case 0x22: // ADDAi
                e.movups_xmm_rbx(0, off_s); load_scalar_field(1, off.i_reg); e.addps(0, 1); return true;
            case 0x23: // MADDAi
                e.movups_xmm_rbx(0, off_s); load_scalar_field(1, off.i_reg); e.mulps(0, 1);
                e.movups_xmm_rbx(1, off.acc); e.addps(0, 1); return true;
            case 0x24: // SUBAq
                e.movups_xmm_rbx(0, off_s); load_scalar_field(1, off.q); e.subps(0, 1); return true;
            case 0x25: // MSUBAq
                e.movups_xmm_rbx(1, off_s); load_scalar_field(2, off.q); e.mulps(1, 2);
                e.movups_xmm_rbx(0, off.acc); e.subps(0, 1); return true;
            case 0x26: // SUBAi
                e.movups_xmm_rbx(0, off_s); load_scalar_field(1, off.i_reg); e.subps(0, 1); return true;
            case 0x27: // MSUBAi
                e.movups_xmm_rbx(1, off_s); load_scalar_field(2, off.i_reg); e.mulps(1, 2);
                e.movups_xmm_rbx(0, off.acc); e.subps(0, 1); return true;
            case 0x28: // ADDA
                e.movups_xmm_rbx(0, off_s); e.movups_xmm_rbx(1, off_t); e.addps(0, 1); return true;
            case 0x29: // MADDA
                e.movups_xmm_rbx(0, off_s); e.movups_xmm_rbx(1, off_t); e.mulps(0, 1);
                e.movups_xmm_rbx(1, off.acc); e.addps(0, 1); return true;
            case 0x2A: // MULA
                e.movups_xmm_rbx(0, off_s); e.movups_xmm_rbx(1, off_t); e.mulps(0, 1); return true;
            case 0x2C: // SUBA
                e.movups_xmm_rbx(0, off_s); e.movups_xmm_rbx(1, off_t); e.subps(0, 1); return true;
            case 0x2D: // MSUBA
                e.movups_xmm_rbx(1, off_s); e.movups_xmm_rbx(2, off_t); e.mulps(1, 2);
                e.movups_xmm_rbx(0, off.acc); e.subps(0, 1); return true;
            default:
                return false;
        }
    }

    switch (fn) {
        case 0x00: case 0x01: case 0x02: case 0x03: // ADDbc
            e.movups_xmm_rbx(0, off_s); load_bc(1, fn & 3u); e.addps(0, 1); return true;
        case 0x04: case 0x05: case 0x06: case 0x07: // SUBbc
            e.movups_xmm_rbx(0, off_s); load_bc(1, fn & 3u); e.subps(0, 1); return true;
        case 0x08: case 0x09: case 0x0A: case 0x0B: // MADDbc
            e.movups_xmm_rbx(0, off_s); load_bc(1, fn & 3u); e.mulps(0, 1);
            e.movups_xmm_rbx(1, off.acc); e.addps(0, 1); return true;
        case 0x0C: case 0x0D: case 0x0E: case 0x0F: // MSUBbc
            e.movups_xmm_rbx(1, off_s); load_bc(2, fn & 3u); e.mulps(1, 2);
            e.movups_xmm_rbx(0, off.acc); e.subps(0, 1); return true;
        case 0x10: case 0x11: case 0x12: case 0x13: // MAXbc
            e.movups_xmm_rbx(0, off_s); load_bc(1, fn & 3u); e.maxps(0, 1); return true;
        case 0x14: case 0x15: case 0x16: case 0x17: // MINIbc
            e.movups_xmm_rbx(0, off_s); load_bc(1, fn & 3u); e.minps(0, 1); return true;
        case 0x18: case 0x19: case 0x1A: case 0x1B: // MULbc
            e.movups_xmm_rbx(0, off_s); load_bc(1, fn & 3u); e.mulps(0, 1); return true;
        case 0x1C: // MULq
            e.movups_xmm_rbx(0, off_s); load_scalar_field(1, off.q); e.mulps(0, 1); return true;
        case 0x1D: // MAXi
            e.movups_xmm_rbx(0, off_s); load_scalar_field(1, off.i_reg); e.maxps(0, 1); return true;
        case 0x1E: // MULi
            e.movups_xmm_rbx(0, off_s); load_scalar_field(1, off.i_reg); e.mulps(0, 1); return true;
        case 0x1F: // MINIi
            e.movups_xmm_rbx(0, off_s); load_scalar_field(1, off.i_reg); e.minps(0, 1); return true;
        case 0x20: // ADDq
            e.movups_xmm_rbx(0, off_s); load_scalar_field(1, off.q); e.addps(0, 1); return true;
        case 0x21: // MADDq
            e.movups_xmm_rbx(0, off_s); load_scalar_field(1, off.q); e.mulps(0, 1);
            e.movups_xmm_rbx(1, off.acc); e.addps(0, 1); return true;
        case 0x22: // ADDi
            e.movups_xmm_rbx(0, off_s); load_scalar_field(1, off.i_reg); e.addps(0, 1); return true;
        case 0x23: // MADDi
            e.movups_xmm_rbx(0, off_s); load_scalar_field(1, off.i_reg); e.mulps(0, 1);
            e.movups_xmm_rbx(1, off.acc); e.addps(0, 1); return true;
        case 0x24: // SUBq
            e.movups_xmm_rbx(0, off_s); load_scalar_field(1, off.q); e.subps(0, 1); return true;
        case 0x25: // MSUBq
            e.movups_xmm_rbx(1, off_s); load_scalar_field(2, off.q); e.mulps(1, 2);
            e.movups_xmm_rbx(0, off.acc); e.subps(0, 1); return true;
        case 0x26: // SUBi
            e.movups_xmm_rbx(0, off_s); load_scalar_field(1, off.i_reg); e.subps(0, 1); return true;
        case 0x27: // MSUBi
            e.movups_xmm_rbx(1, off_s); load_scalar_field(2, off.i_reg); e.mulps(1, 2);
            e.movups_xmm_rbx(0, off.acc); e.subps(0, 1); return true;
        case 0x28: // ADD
            e.movups_xmm_rbx(0, off_s); e.movups_xmm_rbx(1, off_t); e.addps(0, 1); return true;
        case 0x29: // MADD
            e.movups_xmm_rbx(0, off_s); e.movups_xmm_rbx(1, off_t); e.mulps(0, 1);
            e.movups_xmm_rbx(1, off.acc); e.addps(0, 1); return true;
        case 0x2A: // MUL
            e.movups_xmm_rbx(0, off_s); e.movups_xmm_rbx(1, off_t); e.mulps(0, 1); return true;
        case 0x2B: // MAX
            e.movups_xmm_rbx(0, off_s); e.movups_xmm_rbx(1, off_t); e.maxps(0, 1); return true;
        case 0x2C: // SUB
            e.movups_xmm_rbx(0, off_s); e.movups_xmm_rbx(1, off_t); e.subps(0, 1); return true;
        case 0x2D: // MSUB
            e.movups_xmm_rbx(1, off_s); e.movups_xmm_rbx(2, off_t); e.mulps(1, 2);
            e.movups_xmm_rbx(0, off.acc); e.subps(0, 1); return true;
        case 0x2E: { // OPMSUB
            e.u8b(0x48); e.u8b(0x89); e.u8b(0xD9);
            e.u8b(0xBA); e.u32b(upper);
            e.u8b(0x4C); e.u8b(0x8D); e.u8b(0x44); e.u8b(0x24); e.u8b(0x20);
            e.u8b(0x4C); e.u8b(0x8D); e.u8b(0x4C); e.u8b(0x24); e.u8b(0x38);
            e.call_abs(reinterpret_cast<const void*>(hlp.exec_upper_to_tmp));
            e.movaps_xmm_rsp(0, 0x20);
            return true;
        }
        case 0x2F: // MINI
            e.movups_xmm_rbx(0, off_s); e.movups_xmm_rbx(1, off_t); e.minps(0, 1); return true;
        default:
            return false;
    }
}

void emit_lower_exec(
    Emitter& e,
    const Vu1JitOffsets& off,
    const Vu1JitHelpers& hlp,
    const RawVu1Pair& pair,
    bool save_old_vi_for_next_branch,
    u8 prev_alu_reg
) {
    if (pair.i_bit) {
        e.mov_rbx32_imm(off.i_reg, pair.lower);
        return;
    }
    if (pair.lower_nop) return;

    const u32 lower = pair.lower;
    const u8 op   = static_cast<u8>((lower >> 25) & 0x7Fu);
    const u8 dest = static_cast<u8>((lower >> 21) & 0x0Fu);
    const u8 it   = static_cast<u8>((lower >> 16) & 0x1Fu);
    const u8 is   = static_cast<u8>((lower >> 11) & 0x1Fu);
    const u8 id   = static_cast<u8>((lower >> 6)  & 0x1Fu);
    const s16 imm11 = static_cast<s16>((lower & 0x7FFu) | ((lower & 0x400u) ? 0xF800u : 0u));
    const u16 imm15 = static_cast<u16>((lower & 0x7FFu) | (((lower >> 21) & 0x0Fu) << 11));

    auto maybe_save_old_vi = [&](u8 dst_vi) {
        if (save_old_vi_for_next_branch && (dst_vi & 15u) != 0u) {
            e.movzx_eax_vi(off.vi, dst_vi);
            e.mov_rsp32_eax(0x34); // save old VI at [rsp+0x34]
        }
    };

    auto load_branch_vi_eax = [&](u8 reg) {
        const u8 r = reg & 15u;
        if (r != 0u && r == prev_alu_reg) {
            e.mov_eax_rsp32(0x34);
        } else {
            e.movzx_eax_vi(off.vi, r);
        }
    };
    auto load_branch_vi_ecx = [&](u8 reg) {
        const u8 r = reg & 15u;
        if (r != 0u && r == prev_alu_reg) {
            e.u8b(0x8B); e.u8b(0x4C); e.u8b(0x24); e.u8b(0x34); // mov ecx, [rsp+0x34]
        } else {
            e.movzx_ecx_vi(off.vi, r);
        }
    };

    auto emit_lower_fallback = [&]() {
        e.u8b(0x48); e.u8b(0x89); e.u8b(0xD9); // mov rcx, rbx
        e.u8b(0xBA); e.u32b(lower);            // mov edx, lower
        e.u8b(0x41); e.u8b(0xB8); e.u32b(pair.pc); // mov r8d, pc
        e.call_abs(reinterpret_cast<const void*>(hlp.exec_lower_fallback));
    };

    if (op == 0x40) {
        const u8 fn = static_cast<u8>(lower & 0x3Fu);
        if (fn < 0x3C) {
            switch (fn) {
                case 0x30: // IADD
                    maybe_save_old_vi(id);
                    e.movzx_eax_vi(off.vi, is); e.movzx_ecx_vi(off.vi, it);
                    e.u8b(0x01); e.u8b(0xC8); // add eax, ecx
                    e.mov_vi_ax(off.vi, id);
                    return;
                case 0x31: // ISUB
                    maybe_save_old_vi(id);
                    e.movzx_eax_vi(off.vi, is); e.movzx_ecx_vi(off.vi, it);
                    e.u8b(0x29); e.u8b(0xC8); // sub eax, ecx
                    e.mov_vi_ax(off.vi, id);
                    return;
                case 0x32: { // IADDI
                    maybe_save_old_vi(it);
                    const s16 imm5 = static_cast<s16>(id | ((id & 0x10u) ? 0xFFE0u : 0u));
                    e.movzx_eax_vi(off.vi, is);
                    e.u8b(0x05); e.s32b(static_cast<s32>(imm5)); // add eax, imm5
                    e.mov_vi_ax(off.vi, it);
                    return;
                }
                case 0x34: // IAND
                    maybe_save_old_vi(id);
                    e.movzx_eax_vi(off.vi, is); e.movzx_ecx_vi(off.vi, it);
                    e.u8b(0x21); e.u8b(0xC8); // and eax, ecx
                    e.mov_vi_ax(off.vi, id);
                    return;
                case 0x35: // IOR
                    maybe_save_old_vi(id);
                    e.movzx_eax_vi(off.vi, is); e.movzx_ecx_vi(off.vi, it);
                    e.u8b(0x09); e.u8b(0xC8); // or eax, ecx
                    e.mov_vi_ax(off.vi, id);
                    return;
                default:
                    return;
            }
        }

        const u32 sub = (((lower >> 6) & 0x1Fu) << 2) | (lower & 3u);
        switch (sub) {
            case 0x30: // MOVE ft, fs
                if (it != 0u) {
                    e.movups_xmm_rbx(0, off.vf + static_cast<u32>(is) * 16u);
                    e.store_vec4_rbx_masked(off.vf + static_cast<u32>(it) * 16u, dest);
                }
                return;
            case 0x31: // MR32 ft, fs
                if (it != 0u) {
                    e.movups_xmm_rbx(0, off.vf + static_cast<u32>(is) * 16u);
                    e.shufps(0, 0, 0x39); // (y, z, w, x)
                    e.store_vec4_rbx_masked(off.vf + static_cast<u32>(it) * 16u, dest);
                }
                return;
            case 0x34: // LQI ft, (is++)
                maybe_save_old_vi(is);
                e.movzx_eax_vi(off.vi, is);
                e.u8b(0x89); e.u8b(0xC1); // mov ecx, eax
                e.u8b(0x25); e.u32b(0x3FFu); // and eax, 0x3FF
                e.u8b(0xC1); e.u8b(0xE0); e.u8b(4); // shl eax, 4
                if (it != 0u) {
                    e.movups_xmm_rsi_rax(0);
                    e.store_vec4_rbx_masked(off.vf + static_cast<u32>(it) * 16u, dest);
                }
                if ((is & 15u) != 0u) {
                    e.u8b(0x8D); e.u8b(0x41); e.u8b(1); // lea eax, [rcx+1]
                    e.mov_vi_ax(off.vi, is);
                }
                return;
            case 0x35: // SQI fs, (it++)
                maybe_save_old_vi(it);
                e.movzx_eax_vi(off.vi, it);
                e.u8b(0x89); e.u8b(0xC1); // mov ecx, eax
                e.u8b(0x25); e.u32b(0x3FFu);
                e.u8b(0xC1); e.u8b(0xE0); e.u8b(4);
                e.movups_xmm_rbx(0, off.vf + static_cast<u32>(is) * 16u);
                e.store_vec4_mem_masked(dest);
                if ((it & 15u) != 0u) {
                    e.u8b(0x8D); e.u8b(0x41); e.u8b(1);
                    e.mov_vi_ax(off.vi, it);
                }
                return;
            case 0x36: // LQD ft, (--is)
                maybe_save_old_vi(is);
                e.movzx_eax_vi(off.vi, is);
                if ((is & 15u) != 0u) {
                    e.u8b(0xFF); e.u8b(0xC8); // dec eax
                    e.mov_vi_ax(off.vi, is);
                }
                e.u8b(0x25); e.u32b(0x3FFu);
                e.u8b(0xC1); e.u8b(0xE0); e.u8b(4);
                if (it != 0u) {
                    e.movups_xmm_rsi_rax(0);
                    e.store_vec4_rbx_masked(off.vf + static_cast<u32>(it) * 16u, dest);
                }
                return;
            case 0x37: // SQD fs, (--it)
                maybe_save_old_vi(it);
                e.movzx_eax_vi(off.vi, it);
                if ((it & 15u) != 0u) {
                    e.u8b(0xFF); e.u8b(0xC8); // dec eax
                    e.mov_vi_ax(off.vi, it);
                }
                e.u8b(0x25); e.u32b(0x3FFu);
                e.u8b(0xC1); e.u8b(0xE0); e.u8b(4);
                e.movups_xmm_rbx(0, off.vf + static_cast<u32>(is) * 16u);
                e.store_vec4_mem_masked(dest);
                return;
            case 0x3B: // WAITQ
            case 0x7B: { // WAITP: fast-skip C++ call when no hardware pipe is active
                e.mov_eax_rbx32(off.pipe_busy);
                e.u8b(0x85); e.u8b(0xC0); // test eax, eax
                e.u8b(0x74);              // jz skip
                const std::size_t jz_pos = e.pos; e.u8b(0);
                emit_lower_fallback();
                e.base[jz_pos] = static_cast<u8>(e.pos - (jz_pos + 1));
                return;
            }
            case 0x64: // MFP ft, P
                if (it != 0u) {
                    e.movss_xmm_rbx(0, off.p);
                    e.shufps(0, 0, 0x00);
                    e.store_vec4_rbx_masked(off.vf + static_cast<u32>(it) * 16u, dest);
                }
                return;
            case 0x68: // XTOP it
                if ((it & 15u) != 0u) {
                    e.u8b(0x0F); e.u8b(0xB7); e.u8b(0x83); e.u32b(off.xtop_val);
                    e.mov_vi_ax(off.vi, it);
                }
                return;
            case 0x69: // XITOP it
                if ((it & 15u) != 0u) {
                    e.u8b(0x0F); e.u8b(0xB7); e.u8b(0x83); e.u32b(off.xitop_val);
                    e.mov_vi_ax(off.vi, it);
                }
                return;
            case 0x6C: // XGKICK is
                e.u8b(0x48); e.u8b(0x89); e.u8b(0xD9); // mov rcx, rbx
                e.u8b(0xBA); e.u32b(static_cast<u32>(is & 15u)); // mov edx, is&15
                e.call_abs(reinterpret_cast<const void*>(hlp.exec_xgkick));
                return;
            default:
                emit_lower_fallback();
                return;
        }
    }

    switch (op) {
        case 0x00: // LQ ft, imm11(is)
            if (it != 0u) {
                e.movzx_eax_vi(off.vi, is);
                if (imm11 != 0) { e.u8b(0x05); e.s32b(static_cast<s32>(imm11)); }
                e.u8b(0x25); e.u32b(0x3FFu);
                e.u8b(0xC1); e.u8b(0xE0); e.u8b(4);
                e.movups_xmm_rsi_rax(0);
                e.store_vec4_rbx_masked(off.vf + static_cast<u32>(it) * 16u, dest);
            }
            return;
        case 0x01: // SQ fs, imm11(it)
            e.movzx_eax_vi(off.vi, it);
            if (imm11 != 0) { e.u8b(0x05); e.s32b(static_cast<s32>(imm11)); }
            e.u8b(0x25); e.u32b(0x3FFu);
            e.u8b(0xC1); e.u8b(0xE0); e.u8b(4);
            e.movups_xmm_rbx(0, off.vf + static_cast<u32>(is) * 16u);
            e.store_vec4_mem_masked(dest);
            return;
        case 0x08: // IADDIU
            maybe_save_old_vi(it);
            e.movzx_eax_vi(off.vi, is);
            e.u8b(0x05); e.u32b(static_cast<u32>(imm15));
            e.mov_vi_ax(off.vi, it);
            return;
        case 0x09: // ISUBIU
            maybe_save_old_vi(it);
            e.movzx_eax_vi(off.vi, is);
            e.u8b(0x2D); e.u32b(static_cast<u32>(imm15));
            e.mov_vi_ax(off.vi, it);
            return;
        case 0x20: { // B imm11
            const u32 target = (pair.pc + 8u + static_cast<u32>(imm11 * 8)) & 0x3FF8u;
            e.u8b(0x41); e.u8b(0xBE); e.u32b(1u); // mov r14d, 1
            e.mov_rsp32_imm(0x30, target);
            return;
        }
        case 0x21: { // BAL it, imm11
            const u32 target = (pair.pc + 8u + static_cast<u32>(imm11 * 8)) & 0x3FF8u;
            const u16 link_val = static_cast<u16>(((pair.pc + 16u) >> 3) & 0x7FFu);
            e.mov_vi_imm16(off.vi, it, link_val);
            e.u8b(0x41); e.u8b(0xBE); e.u32b(1u);
            e.mov_rsp32_imm(0x30, target);
            return;
        }
        case 0x24: // JR is
            e.movzx_eax_vi(off.vi, is);
            e.u8b(0xC1); e.u8b(0xE0); e.u8b(3); // shl eax, 3
            e.u8b(0x25); e.u32b(0x3FF8u);       // and eax, 0x3FF8
            e.mov_rsp32_eax(0x30);
            e.u8b(0x41); e.u8b(0xBE); e.u32b(1u);
            return;
        case 0x25: { // JALR it, is (save target from vi[is] BEFORE writing vi[it])
            const u16 link_val = static_cast<u16>(((pair.pc + 16u) >> 3) & 0x7FFu);
            e.movzx_eax_vi(off.vi, is);
            e.u8b(0xC1); e.u8b(0xE0); e.u8b(3);
            e.u8b(0x25); e.u32b(0x3FF8u);
            e.mov_rsp32_eax(0x30);
            e.mov_vi_imm16(off.vi, it, link_val);
            e.u8b(0x41); e.u8b(0xBE); e.u32b(1u);
            return;
        }
        case 0x28: case 0x29: { // IBEQ / IBNE
            const u32 target = (pair.pc + 8u + static_cast<u32>(imm11 * 8)) & 0x3FF8u;
            e.u8b(0x45); e.u8b(0x31); e.u8b(0xF6); // xor r14d, r14d
            load_branch_vi_eax(it);
            load_branch_vi_ecx(is);
            e.u8b(0x66); e.u8b(0x39); e.u8b(0xC8); // cmp ax, cx
            e.u8b(0x0F); e.u8b(op == 0x28 ? 0x85 : 0x84); // jne / je skip
            const std::size_t skip_patch = e.pos; e.s32b(0);
            e.u8b(0x41); e.u8b(0xBE); e.u32b(1u);
            e.mov_rsp32_imm(0x30, target);
            e.patch_rel32(skip_patch, e.pos);
            return;
        }
        case 0x2C: case 0x2D: case 0x2E: case 0x2F: { // IBLTZ / IBGTZ / IBLEZ / IBGEZ
            const u32 target = (pair.pc + 8u + static_cast<u32>(imm11 * 8)) & 0x3FF8u;
            e.u8b(0x45); e.u8b(0x31); e.u8b(0xF6); // xor r14d, r14d
            load_branch_vi_eax(is);
            e.u8b(0x0F); e.u8b(0xBF); e.u8b(0xC0); // movsx eax, ax
            e.u8b(0x85); e.u8b(0xC0);              // test eax, eax
            // Jump to skip when condition is FALSE
            const u8 jcc_skip = (op == 0x2C) ? 0x8D : // jge (not <0)
                                (op == 0x2D) ? 0x8E : // jle (not >0)
                                (op == 0x2E) ? 0x8F : // jg  (not <=0)
                                               0x8C;  // jl  (not >=0)
            e.u8b(0x0F); e.u8b(jcc_skip);
            const std::size_t skip_patch = e.pos; e.s32b(0);
            e.u8b(0x41); e.u8b(0xBE); e.u32b(1u);
            e.mov_rsp32_imm(0x30, target);
            e.patch_rel32(skip_patch, e.pos);
            return;
        }
        default:
            emit_lower_fallback();
            return;
    }
}

} // namespace
#endif // TF_VU1_JIT_X64_ENABLED

Vu1JitX64::Vu1JitX64() {
#if TF_VU1_JIT_X64_ENABLED
    m_code_buf = static_cast<u8*>(VirtualAlloc(nullptr, kCodeBufSize, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE));
    if (m_code_buf) {
        m_code_cap = kCodeBufSize;
    }
    m_l2.resize(kL2Cap);
#endif
    invalidate_all();
}

Vu1JitX64::~Vu1JitX64() {
#if TF_VU1_JIT_X64_ENABLED
    if (m_code_buf) {
        VirtualFree(m_code_buf, 0, MEM_RELEASE);
        m_code_buf = nullptr;
    }
#endif
    m_code_cap = 0;
    m_write_pos = 0;
}

void Vu1JitX64::invalidate_all() {
    m_write_pos = 0;
    m_blocks.fill(nullptr);
    m_slot_gen.fill(0u);
    m_cur_gen = 1u;
    if (!m_l2.empty()) {
        std::fill(m_l2.begin(), m_l2.end(), L2Entry{});
    }
    m_l2_count = 0;
}

Vu1JitBlockFn Vu1JitX64::get_or_compile(u32 start_idx, const u8* code_mem, bool has_mac_readers, bool has_clip_readers) {
#if TF_VU1_JIT_X64_ENABLED
    const u32 idx = start_idx & 0x7FFu;
    if (m_slot_gen[idx] == m_cur_gen) {
        return m_blocks[idx];
    }
    m_slot_gen[idx] = m_cur_gen;
    if (!m_code_buf || !m_configured || !code_mem) {
        m_blocks[idx] = nullptr;
        return nullptr;
    }

    Vu1JitBlockFn fn = compile_block(idx, code_mem, has_mac_readers, has_clip_readers);
    m_blocks[idx] = fn;
    return fn;
#else
    (void)start_idx; (void)code_mem; (void)has_mac_readers; (void)has_clip_readers;
    return nullptr;
#endif
}

Vu1JitBlockFn Vu1JitX64::compile_block(u32 start_idx, const u8* code_mem, bool has_mac_readers, bool has_clip_readers) {
#if TF_VU1_JIT_X64_ENABLED
    // 1. Scan basic block starting at `start_idx` up to branch+delay, E-bit+delay, or kMaxBlockInsns
    RawVu1Pair pairs[kMaxBlockInsns + 2];
    u32 count = 0;
    bool ends_with_ebit = false;
    bool ends_with_branch = false;
    bool block_has_pipes = has_mac_readers || has_clip_readers;
    bool can_self_loop = false;

    auto check_pair_clip_and_fcset = [&](const RawVu1Pair& pr) -> bool {
        u8 tvf = 0; bool tacc = false, tmac = false, tclip = false, thlp = false;
        classify_upper(pr.upper, tvf, tacc, tmac, tclip, thlp);
        if (tclip) {
            block_has_pipes = true;
            // Rare edge case: CLIPw paired with FCSET in the same cycle -> fall back to C++ interpreter
            if (!pr.i_bit && ((pr.lower >> 25) & 0x7Fu) == 0x11u) return false;
        }
        return true;
    };

    u32 cur_idx = start_idx & 0x7FFu;
    while (count < kMaxBlockInsns) {
        RawVu1Pair p = inspect_pair(code_mem, cur_idx);
        if (p.uses_pipes_lower) block_has_pipes = true;
        if (!check_pair_clip_and_fcset(p)) return nullptr;
        pairs[count++] = p;
        cur_idx = (cur_idx + 1u) & 0x7FFu;

        if (p.e_bit) {
            // Do not JIT weird edge case of simultaneous branch + E-bit
            if (p.is_branch) return nullptr;
            RawVu1Pair ds = inspect_pair(code_mem, cur_idx);
            if (ds.is_branch || ds.e_bit) return nullptr;
            if (ds.uses_pipes_lower) block_has_pipes = true;
            if (!check_pair_clip_and_fcset(ds)) return nullptr;
            pairs[count++] = ds;
            ends_with_ebit = true;
            break;
        }
        if (p.is_branch) {
            RawVu1Pair ds = inspect_pair(code_mem, cur_idx);
            if (ds.is_branch || ds.e_bit) return nullptr;
            if (ds.uses_pipes_lower) block_has_pipes = true;
            if (!check_pair_clip_and_fcset(ds)) return nullptr;
            pairs[count++] = ds;
            ends_with_branch = true;
            // Check if this conditional branch can loop straight back to start_idx!
            if (p.is_cond_branch) {
                const s16 imm11 = static_cast<s16>((p.lower & 0x7FFu) | ((p.lower & 0x400u) ? 0xF800u : 0u));
                const u32 target = (p.pc + 8u + static_cast<u32>(imm11 * 8)) & 0x3FF8u;
                if (target == (start_idx & 0x7FFu) * 8u) {
                    can_self_loop = true;
                }
            }
            break;
        }
    }

    // Never leave an unresolved VI ALU hazard (or same-reg chain) dangling across the end of a non-branch block
    while (!ends_with_branch && !ends_with_ebit && count > 0 && pairs[count - 1].is_vi_alu) {
        count--;
    }
    if (count == 0) return nullptr;
    // Also guard cross-block VI ALU hazard if a conditional branch's delay slot is VI ALU
    // and the fallthrough (or self-loop target) instruction is a conditional branch or continues the same-reg chain.
    if (ends_with_branch && count >= 2 && pairs[count - 2].is_cond_branch && pairs[count - 1].is_vi_alu) {
        const u8 ds_dst = pairs[count - 1].vi_alu_dst;
        RawVu1Pair next_fallthrough = inspect_pair(code_mem, (start_idx + count) & 0x7FFu);
        if (next_fallthrough.is_cond_branch ||
            (ds_dst != 0u && next_fallthrough.is_vi_alu && next_fallthrough.vi_alu_dst == ds_dst)) {
            return nullptr;
        }
        if (can_self_loop && (pairs[0].is_cond_branch ||
            (ds_dst != 0u && pairs[0].is_vi_alu && pairs[0].vi_alu_dst == ds_dst))) {
            return nullptr;
        }
    }

    // Compute 128-bit persistent L2 key so rotating between VU1 overlays reuses compiled x64 code with ZERO recompile!
    u64 h1 = 14695981039346656037ULL;
    u64 h2 = 0x9E3779B97F4A7C15ULL;
    auto mix_u64 = [&](u64 v) {
        h1 = (h1 ^ v) * 1099511628211ULL;
        u64 z = h2 ^ (v + 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        h2 = z ^ (z >> 31);
    };
    const u64 hdr = (static_cast<u64>(start_idx & 0x7FFu) << 32) |
                    (static_cast<u64>(count) << 16) |
                    (has_mac_readers ? 1ULL : 0ULL) |
                    (has_clip_readers ? 2ULL : 0ULL) |
                    (ends_with_ebit ? 4ULL : 0ULL) |
                    (ends_with_branch ? 8ULL : 0ULL);
    mix_u64(hdr);
    for (u32 i = 0; i < count; ++i) {
        const u64 raw = static_cast<u64>(pairs[i].lower) | (static_cast<u64>(pairs[i].upper) << 32);
        mix_u64(raw);
    }
    if ((h1 | h2) == 0ULL) h1 = 1ULL;

    if (!m_l2.empty()) {
        u32 slot = static_cast<u32>(h1) & kL2Mask;
        for (u32 probe = 0; probe < 64u; ++probe) {
            const L2Entry& ent = m_l2[slot];
            if (ent.fn == nullptr) break;
            if (ent.key_lo == h1 && ent.key_hi == h2) {
                return ent.fn;
            }
            slot = (slot + 1u) & kL2Mask;
        }
    }

    // Align write cursor to 16 bytes; flush arena + L2 only if the 4 MiB buffer or L2 table is full
    std::size_t aligned_pos = (m_write_pos + 15u) & ~std::size_t(15u);
    if (aligned_pos + 32768u > m_code_cap || m_l2_count * 2u >= kL2Cap) {
        invalidate_all();
        m_slot_gen[start_idx & 0x7FFu] = m_cur_gen;
        aligned_pos = 0;
    }

    Emitter e{m_code_buf, aligned_pos, m_code_cap, false};
    const std::size_t fn_entry = e.pos;

    // Prologue (Windows x64 ABI):
    //   push rbx, rsi, rdi, r12, r13, r14  (6*8 = 48 bytes; RSP%16 == 8)
    //   sub rsp, 0x48                      (72 bytes; RSP%16 == 0!)
    e.u8b(0x53);                   // push rbx
    e.u8b(0x56);                   // push rsi
    e.u8b(0x57);                   // push rdi
    e.u8b(0x41); e.u8b(0x54);      // push r12
    e.u8b(0x41); e.u8b(0x55);      // push r13
    e.u8b(0x41); e.u8b(0x56);      // push r14
    e.u8b(0x48); e.u8b(0x83); e.u8b(0xEC); e.u8b(0x48); // sub rsp, 0x48

    e.u8b(0x48); e.u8b(0x89); e.u8b(0xCB);              // mov rbx, rcx (VU1Unit*)
    e.u8b(0x89); e.u8b(0xD7);                           // mov edi, edx (max_steps)
    e.u8b(0x45); e.u8b(0x31); e.u8b(0xE4);              // xor r12d, r12d (total_steps = 0)
    e.u8b(0x49); e.u8b(0xBD); e.u64b(reinterpret_cast<u64>(&kJitDestMask4[0][0])); // mov r13, kJitDestMask4
    e.u8b(0x48); e.u8b(0x8B); e.u8b(0xB3); e.u32b(m_offsets.data_mem); // mov rsi, [rbx + data_mem]

    // Loop head (also entry check for rem_steps >= count)
    const std::size_t loop_head = e.pos;
    e.u8b(0x81); e.u8b(0xFF); e.u32b(count);            // cmp edi, count
    e.u8b(0x0F); e.u8b(0x82);                           // jb epilogue_no_ebit
    const std::size_t budget_exit_patch = e.pos; e.s32b(0);

    // If this block omits per-instruction pipe checks, ensure no in-flight Q/P/MAC/CLIP pipe
    // from a prior block is active on entry; if one is active, return 0 so C++ drains it cycle-accurately.
    std::size_t pipe_exit_patch = 0;
    if (!block_has_pipes) {
        e.mov_eax_rbx32(m_offsets.pipe_busy);
        e.u8b(0x85); e.u8b(0xC0);                       // test eax, eax
        e.u8b(0x0F); e.u8b(0x85);                       // jnz epilogue_no_ebit
        pipe_exit_patch = e.pos; e.s32b(0);
    }

    // If no FM*/FS* readers exist in code_mem, we skip MAC flags across all hot loops,
    // but still commit the final MAC-writing upper op in an E-bit terminating block so
    // post-MSCAL mac_flags/status_flags remain observable at negligible cost (~1x/MSCAL).
    int last_ebit_mac_idx = -1;
    if (ends_with_ebit && !has_mac_readers) {
        for (u32 i = 0; i < count; ++i) {
            u8 tvf = 0; bool tacc = false, tmac = false, tclip = false, thlp = false;
            classify_upper(pairs[i].upper, tvf, tacc, tmac, tclip, thlp);
            if (tmac) last_ebit_mac_idx = static_cast<int>(i);
        }
    }

    // Emit each VU1 instruction pair in the block
    for (u32 i = 0; i < count; ++i) {
        const RawVu1Pair& pair = pairs[i];

        // 1. Advance hardware pipelines if active
        if (block_has_pipes) {
            e.mov_eax_rbx32(m_offsets.pipe_busy);
            e.u8b(0x85); e.u8b(0xC0); // test eax, eax
            e.u8b(0x74);              // jz skip
            const std::size_t jz_pos = e.pos; e.u8b(0);
            e.u8b(0x48); e.u8b(0x89); e.u8b(0xD9);
            e.call_abs(reinterpret_cast<const void*>(m_helpers.advance_pipes));
            e.base[jz_pos] = static_cast<u8>(e.pos - (jz_pos + 1));
        }

        // 2. Classify upper instruction
        u8 u_vf = 0;
        bool u_acc = false;
        bool u_mac = false;
        bool u_clip = false;
        bool u_needs_helper = false;
        classify_upper(pair.upper, u_vf, u_acc, u_mac, u_clip, u_needs_helper);

        const bool want_mac = (has_mac_readers || static_cast<int>(i) == last_ebit_mac_idx) && u_mac;
        const bool want_clip = u_clip; // always commit CLIPw since clip_flags persists across MSCNT
        const bool defer_vf = (u_vf != 0u) && lower_touches_vf(pair, u_vf);
        const bool need_upper_compute = (u_vf != 0u) || u_acc || want_mac;

        u8 u_dest = static_cast<u8>((pair.upper >> 21) & 0x0Fu);
        bool computed_xmm0 = false;
        if (need_upper_compute) {
            computed_xmm0 = emit_upper_compute_xmm0(e, m_offsets, m_helpers, pair.upper, u_dest);
            if (computed_xmm0) {
                if (defer_vf || want_mac) {
                    // Save upper result in 16-byte aligned [rsp+0x20] across lower execution
                    e.movaps_rsp_xmm(0x20, 0);
                }
                if (!defer_vf) {
                    if (u_vf != 0u) {
                        e.store_vec4_rbx_masked(m_offsets.vf + static_cast<u32>(u_vf) * 16u, u_dest);
                    } else if (u_acc) {
                        e.store_vec4_rbx_masked(m_offsets.acc, u_dest);
                    }
                }
            }
        }
        // Compute & enqueue CLIPw BEFORE lower runs so lower overwriting vf[fs]/vf[ft] sees pre-lower values!
        if (want_clip) {
            const u8 fs = static_cast<u8>((pair.upper >> 11) & 0x1Fu);
            const u8 ft = static_cast<u8>((pair.upper >> 16) & 0x1Fu);
            e.u8b(0x48); e.u8b(0x89); e.u8b(0xD9);
            e.u8b(0xBA); e.u32b(static_cast<u32>(fs));
            e.u8b(0x41); e.u8b(0xB8); e.u32b(static_cast<u32>(ft));
            e.call_abs(reinterpret_cast<const void*>(m_helpers.commit_clip));
        }

        // 3. Execute lower instruction (with exact branch VI-delay hazard + consecutive-write chaining!)
        bool save_vi_old = false;
        if (pair.is_vi_alu && pair.vi_alu_dst != 0u) {
            const u8 dst_r = pair.vi_alu_dst;
            const bool continued_from_prev = (i > 0u && pairs[i - 1u].is_vi_alu && pairs[i - 1u].vi_alu_dst == dst_r);
            if (!continued_from_prev) {
                u32 k = i + 1u;
                while (k < count && pairs[k].is_vi_alu && pairs[k].vi_alu_dst == dst_r) {
                    ++k;
                }
                if (k < count && pairs[k].is_cond_branch) {
                    save_vi_old = true;
                }
            }
        }
        const u8 prev_alu_reg = (i > 0u && pairs[i - 1u].is_vi_alu) ? pairs[i - 1u].vi_alu_dst : 0u;
        emit_lower_exec(e, m_offsets, m_helpers, pair, save_vi_old, prev_alu_reg);

        // 4. Commit deferred upper write & FMAC MAC pipeline write
        if (computed_xmm0 && defer_vf) {
            e.movaps_xmm_rsp(0, 0x20);
            e.store_vec4_rbx_masked(m_offsets.vf + static_cast<u32>(u_vf) * 16u, u_dest);
        }
        if (computed_xmm0 && want_mac) {
            e.u8b(0x48); e.u8b(0x89); e.u8b(0xD9);                         // mov rcx, rbx
            e.u8b(0x48); e.u8b(0x8D); e.u8b(0x54); e.u8b(0x24); e.u8b(0x20); // lea rdx, [rsp+0x20]
            e.u8b(0x41); e.u8b(0xB8); e.u32b(static_cast<u32>(u_dest));    // mov r8d, u_dest
            e.call_abs(reinterpret_cast<const void*>(m_helpers.commit_mac));
        }
    }

    // Account for executed steps in this iteration
    e.u8b(0x81); e.u8b(0xEF); e.u32b(count);            // sub edi, count
    e.u8b(0x41); e.u8b(0x81); e.u8b(0xC4); e.u32b(count); // add r12d, count

    const u32 fallthrough_pc = ((start_idx + count) & 0x7FFu) * 8u;

    if (ends_with_branch) {
        // If branch was taken (r14d != 0)
        e.u8b(0x45); e.u8b(0x85); e.u8b(0xF6); // test r14d, r14d
        e.u8b(0x0F); e.u8b(0x84);              // je not_taken
        const std::size_t not_taken_patch = e.pos; e.s32b(0);

        if (can_self_loop) {
            // Taken self-loop: update PC to loop start (in case budget check exits!) and jump to loop_head
            e.mov_rbx32_imm(m_offsets.pc, (start_idx & 0x7FFu) * 8u);
            e.u8b(0xE9);
            const std::size_t jmp_back = e.pos; e.s32b(0);
            e.patch_rel32(jmp_back, loop_head);
        } else {
            e.mov_eax_rsp32(0x30);
            e.mov_rbx32_eax(m_offsets.pc);
            e.u8b(0xE9);
            const std::size_t jmp_ep = e.pos; e.s32b(0);
            e.patch_rel32(not_taken_patch, e.pos);
            e.mov_rbx32_imm(m_offsets.pc, fallthrough_pc);
            e.patch_rel32(jmp_ep, e.pos);
            goto emit_epilogue;
        }

        e.patch_rel32(not_taken_patch, e.pos);
        e.mov_rbx32_imm(m_offsets.pc, fallthrough_pc);
    } else {
        e.mov_rbx32_imm(m_offsets.pc, fallthrough_pc);
    }

emit_epilogue:
    if (ends_with_ebit) {
        // Normal E-bit exit: return total_steps | 0x80000000
        e.u8b(0x44); e.u8b(0x89); e.u8b(0xE0);       // mov eax, r12d
        e.u8b(0x0D); e.u32b(0x80000000u);            // or eax, 0x80000000
        e.u8b(0xE9);
        const std::size_t jmp_ret = e.pos; e.s32b(0);
        e.patch_rel32(budget_exit_patch, e.pos);
        if (pipe_exit_patch != 0) e.patch_rel32(pipe_exit_patch, e.pos);
        e.u8b(0x44); e.u8b(0x89); e.u8b(0xE0);       // mov eax, r12d
        e.patch_rel32(jmp_ret, e.pos);
    } else {
        e.patch_rel32(budget_exit_patch, e.pos);
        if (pipe_exit_patch != 0) e.patch_rel32(pipe_exit_patch, e.pos);
        e.u8b(0x44); e.u8b(0x89); e.u8b(0xE0);       // mov eax, r12d
    }

    // Epilogue
    e.u8b(0x48); e.u8b(0x83); e.u8b(0xC4); e.u8b(0x48); // add rsp, 0x48
    e.u8b(0x41); e.u8b(0x5E);                           // pop r14
    e.u8b(0x41); e.u8b(0x5D);                           // pop r13
    e.u8b(0x41); e.u8b(0x5C);                           // pop r12
    e.u8b(0x5F);                                        // pop rdi
    e.u8b(0x5E);                                        // pop rsi
    e.u8b(0x5B);                                        // pop rbx
    e.u8b(0xC3);                                        // ret

    if (e.overflow) return nullptr;
    m_write_pos = e.pos;
    m_blocks_compiled++;
    Vu1JitBlockFn fn = reinterpret_cast<Vu1JitBlockFn>(m_code_buf + fn_entry);
    if (!m_l2.empty()) {
        u32 slot = static_cast<u32>(h1) & kL2Mask;
        for (u32 probe = 0; probe < 64u; ++probe) {
            if (m_l2[slot].fn == nullptr) {
                m_l2[slot] = L2Entry{h1, h2, fn};
                m_l2_count++;
                break;
            }
            slot = (slot + 1u) & kL2Mask;
        }
    }
    return fn;
#else
    (void)start_idx; (void)code_mem; (void)has_mac_readers; (void)has_clip_readers;
    return nullptr;
#endif
}

} // namespace tf::ps2
