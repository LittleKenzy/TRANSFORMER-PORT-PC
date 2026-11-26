#include "VU0.h"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace tf::ps2 {

namespace {

inline f32 get_q(const EEContext& ctx) {
    f32 q = 0.0f;
    std::memcpy(&q, &ctx.vi[22], 4); // VI[22] = Q register
    return q;
}
inline void set_q(EEContext& ctx, f32 q) {
    std::memcpy(&ctx.vi[22], &q, 4);
}
inline f32 get_i(const EEContext& ctx) {
    f32 iv = 0.0f;
    std::memcpy(&iv, &ctx.vi[21], 4); // VI[21] = I register
    return iv;
}

} // namespace

void VU0Unit::reset(EEContext& ctx) {
    ctx.vf.fill(Vec4f{});
    ctx.vf[0] = Vec4f{0.0f, 0.0f, 0.0f, 1.0f};
    ctx.vi.fill(0);
    ctx.vu0_acc = Vec4f{};
    m_executed_ops = 0;
    m_unsupported_ops = 0;
}

void VU0Unit::update_mac_flags(EEContext& ctx, const Vec4f& v, u8 dest_mask) {
    // MAC flag bits: [0..3]=Z(w,z,y,x), [4..7]=S(w,z,y,x)
    u32 mac = 0;
    // dest_mask bit 3=x(0), 2=y(1), 1=z(2), 0=w(3)
    for (int lane = 0; lane < 4; ++lane) {
        if (dest_mask & (1u << (3 - lane))) {
            f32 val = v[lane];
            if (val == 0.0f) mac |= (1u << (3 - lane));
            if (std::signbit(val)) mac |= (1u << (7 - lane));
        }
    }
    ctx.vi[17] = mac; // VI[17] = MAC flag
}

void VU0Unit::write_vf(EEContext& ctx, u8 fd, const Vec4f& v, u8 dest_mask) {
    if (fd == 0) return;
    for (int lane = 0; lane < 4; ++lane) {
        if (dest_mask & (1u << (3 - lane))) {
            ctx.vf[fd][lane] = v[lane];
        }
    }
}

void VU0Unit::execute_macro(EEContext& ctx, u32 raw) {
    m_executed_ops++;
    ctx.vf[0] = Vec4f{0.0f, 0.0f, 0.0f, 1.0f};

    u8 dest = static_cast<u8>((raw >> 21) & 0x0Fu); // x=8, y=4, z=2, w=1
    u8 ft   = static_cast<u8>((raw >> 16) & 0x1Fu);
    u8 fs   = static_cast<u8>((raw >> 11) & 0x1Fu);
    u8 fd   = static_cast<u8>((raw >> 6)  & 0x1Fu);
    u8 fn   = static_cast<u8>(raw & 0x3Fu);

    const Vec4f& S = ctx.vf[fs];
    const Vec4f& T = ctx.vf[ft];
    Vec4f R{};

    // Special2 table (fn >= 0x3C)
    if (fn >= 0x3C) {
        u16 fn2 = static_cast<u16>(((raw >> 4) & 0x7Cu) | (raw & 0x03u));
        switch (fn2) {
            case 0x00: case 0x01: case 0x02: case 0x03: { // VADDAx/y/z/w
                f32 bc = T[fn2 & 3];
                for (int i = 0; i < 4; ++i) R[i] = S[i] + bc;
                for (int i = 0; i < 4; ++i) if (dest & (1u << (3 - i))) ctx.vu0_acc[i] = R[i];
                update_mac_flags(ctx, R, dest);
                return;
            }
            case 0x04: case 0x05: case 0x06: case 0x07: { // VSUBAx/y/z/w
                f32 bc = T[fn2 & 3];
                for (int i = 0; i < 4; ++i) R[i] = S[i] - bc;
                for (int i = 0; i < 4; ++i) if (dest & (1u << (3 - i))) ctx.vu0_acc[i] = R[i];
                update_mac_flags(ctx, R, dest);
                return;
            }
            case 0x08: case 0x09: case 0x0A: case 0x0B: { // VMADDAx/y/z/w
                f32 bc = T[fn2 & 3];
                for (int i = 0; i < 4; ++i) R[i] = ctx.vu0_acc[i] + S[i] * bc;
                for (int i = 0; i < 4; ++i) if (dest & (1u << (3 - i))) ctx.vu0_acc[i] = R[i];
                update_mac_flags(ctx, R, dest);
                return;
            }
            case 0x0C: case 0x0D: case 0x0E: case 0x0F: { // VMSUBAx/y/z/w
                f32 bc = T[fn2 & 3];
                for (int i = 0; i < 4; ++i) R[i] = ctx.vu0_acc[i] - S[i] * bc;
                for (int i = 0; i < 4; ++i) if (dest & (1u << (3 - i))) ctx.vu0_acc[i] = R[i];
                update_mac_flags(ctx, R, dest);
                return;
            }
            case 0x10: // VITOF0
                for (int i = 0; i < 4; ++i) { s32 iv = 0; std::memcpy(&iv, &S[i], 4); R[i] = static_cast<f32>(iv); }
                write_vf(ctx, ft, R, dest);
                return;
            case 0x11: // VITOF4
                for (int i = 0; i < 4; ++i) { s32 iv = 0; std::memcpy(&iv, &S[i], 4); R[i] = static_cast<f32>(iv) / 16.0f; }
                write_vf(ctx, ft, R, dest);
                return;
            case 0x12: // VITOF12
                for (int i = 0; i < 4; ++i) { s32 iv = 0; std::memcpy(&iv, &S[i], 4); R[i] = static_cast<f32>(iv) / 4096.0f; }
                write_vf(ctx, ft, R, dest);
                return;
            case 0x13: // VITOF15
                for (int i = 0; i < 4; ++i) { s32 iv = 0; std::memcpy(&iv, &S[i], 4); R[i] = static_cast<f32>(iv) / 32768.0f; }
                write_vf(ctx, ft, R, dest);
                return;
            case 0x14: // VFTOI0
                for (int i = 0; i < 4; ++i) { s32 iv = static_cast<s32>(S[i]); std::memcpy(&R[i], &iv, 4); }
                write_vf(ctx, ft, R, dest);
                return;
            case 0x15: // VFTOI4
                for (int i = 0; i < 4; ++i) { s32 iv = static_cast<s32>(S[i] * 16.0f); std::memcpy(&R[i], &iv, 4); }
                write_vf(ctx, ft, R, dest);
                return;
            case 0x16: // VFTOI12
                for (int i = 0; i < 4; ++i) { s32 iv = static_cast<s32>(S[i] * 4096.0f); std::memcpy(&R[i], &iv, 4); }
                write_vf(ctx, ft, R, dest);
                return;
            case 0x17: // VFTOI15
                for (int i = 0; i < 4; ++i) { s32 iv = static_cast<s32>(S[i] * 32768.0f); std::memcpy(&R[i], &iv, 4); }
                write_vf(ctx, ft, R, dest);
                return;
            case 0x18: case 0x19: case 0x1A: case 0x1B: { // VMULAx/y/z/w
                f32 bc = T[fn2 & 3];
                for (int i = 0; i < 4; ++i) R[i] = S[i] * bc;
                for (int i = 0; i < 4; ++i) if (dest & (1u << (3 - i))) ctx.vu0_acc[i] = R[i];
                update_mac_flags(ctx, R, dest);
                return;
            }
            case 0x1C: // VMULAq
                for (int i = 0; i < 4; ++i) R[i] = S[i] * get_q(ctx);
                for (int i = 0; i < 4; ++i) if (dest & (1u << (3 - i))) ctx.vu0_acc[i] = R[i];
                update_mac_flags(ctx, R, dest);
                return;
            case 0x1D: // VABS
                for (int i = 0; i < 4; ++i) R[i] = std::fabs(S[i]);
                write_vf(ctx, ft, R, dest);
                return;
            case 0x1E: // VMULAi
                for (int i = 0; i < 4; ++i) R[i] = S[i] * get_i(ctx);
                for (int i = 0; i < 4; ++i) if (dest & (1u << (3 - i))) ctx.vu0_acc[i] = R[i];
                update_mac_flags(ctx, R, dest);
                return;
            case 0x1F: { // VCLIPw.xyz
                f32 w = std::fabs(T.w);
                u32 clip = (ctx.vi[18] << 6) & 0xFFFFFFu;
                if (S.x >  w) clip |= 0x01u;
                if (S.x < -w) clip |= 0x02u;
                if (S.y >  w) clip |= 0x04u;
                if (S.y < -w) clip |= 0x08u;
                if (S.z >  w) clip |= 0x10u;
                if (S.z < -w) clip |= 0x20u;
                ctx.vi[18] = clip;
                return;
            }
            case 0x20: // VADDAq
                for (int i = 0; i < 4; ++i) R[i] = S[i] + get_q(ctx);
                for (int i = 0; i < 4; ++i) if (dest & (1u << (3 - i))) ctx.vu0_acc[i] = R[i];
                update_mac_flags(ctx, R, dest);
                return;
            case 0x21: // VMADDAq
                for (int i = 0; i < 4; ++i) R[i] = ctx.vu0_acc[i] + S[i] * get_q(ctx);
                for (int i = 0; i < 4; ++i) if (dest & (1u << (3 - i))) ctx.vu0_acc[i] = R[i];
                update_mac_flags(ctx, R, dest);
                return;
            case 0x22: // VADDAi
                for (int i = 0; i < 4; ++i) R[i] = S[i] + get_i(ctx);
                for (int i = 0; i < 4; ++i) if (dest & (1u << (3 - i))) ctx.vu0_acc[i] = R[i];
                update_mac_flags(ctx, R, dest);
                return;
            case 0x23: // VMADDAi
                for (int i = 0; i < 4; ++i) R[i] = ctx.vu0_acc[i] + S[i] * get_i(ctx);
                for (int i = 0; i < 4; ++i) if (dest & (1u << (3 - i))) ctx.vu0_acc[i] = R[i];
                update_mac_flags(ctx, R, dest);
                return;
            case 0x24: // VSUBAq
                for (int i = 0; i < 4; ++i) R[i] = S[i] - get_q(ctx);
                for (int i = 0; i < 4; ++i) if (dest & (1u << (3 - i))) ctx.vu0_acc[i] = R[i];
                update_mac_flags(ctx, R, dest);
                return;
            case 0x25: // VMSUBAq
                for (int i = 0; i < 4; ++i) R[i] = ctx.vu0_acc[i] - S[i] * get_q(ctx);
                for (int i = 0; i < 4; ++i) if (dest & (1u << (3 - i))) ctx.vu0_acc[i] = R[i];
                update_mac_flags(ctx, R, dest);
                return;
            case 0x26: // VSUBAi
                for (int i = 0; i < 4; ++i) R[i] = S[i] - get_i(ctx);
                for (int i = 0; i < 4; ++i) if (dest & (1u << (3 - i))) ctx.vu0_acc[i] = R[i];
                update_mac_flags(ctx, R, dest);
                return;
            case 0x27: // VMSUBAi
                for (int i = 0; i < 4; ++i) R[i] = ctx.vu0_acc[i] - S[i] * get_i(ctx);
                for (int i = 0; i < 4; ++i) if (dest & (1u << (3 - i))) ctx.vu0_acc[i] = R[i];
                update_mac_flags(ctx, R, dest);
                return;
            case 0x28: // VADDA
                for (int i = 0; i < 4; ++i) R[i] = S[i] + T[i];
                for (int i = 0; i < 4; ++i) if (dest & (1u << (3 - i))) ctx.vu0_acc[i] = R[i];
                update_mac_flags(ctx, R, dest);
                return;
            case 0x29: // VMADDA
                for (int i = 0; i < 4; ++i) R[i] = ctx.vu0_acc[i] + S[i] * T[i];
                for (int i = 0; i < 4; ++i) if (dest & (1u << (3 - i))) ctx.vu0_acc[i] = R[i];
                update_mac_flags(ctx, R, dest);
                return;
            case 0x2A: // VMULA
                for (int i = 0; i < 4; ++i) R[i] = S[i] * T[i];
                for (int i = 0; i < 4; ++i) if (dest & (1u << (3 - i))) ctx.vu0_acc[i] = R[i];
                update_mac_flags(ctx, R, dest);
                return;
            case 0x2C: // VSUBA
                for (int i = 0; i < 4; ++i) R[i] = S[i] - T[i];
                for (int i = 0; i < 4; ++i) if (dest & (1u << (3 - i))) ctx.vu0_acc[i] = R[i];
                update_mac_flags(ctx, R, dest);
                return;
            case 0x2D: // VMSUBA
                for (int i = 0; i < 4; ++i) R[i] = ctx.vu0_acc[i] - S[i] * T[i];
                for (int i = 0; i < 4; ++i) if (dest & (1u << (3 - i))) ctx.vu0_acc[i] = R[i];
                update_mac_flags(ctx, R, dest);
                return;
            case 0x2E: { // VOPMULA.xyz
                ctx.vu0_acc.x = S.y * T.z;
                ctx.vu0_acc.y = S.z * T.x;
                ctx.vu0_acc.z = S.x * T.y;
                update_mac_flags(ctx, ctx.vu0_acc, 0x0Eu);
                return;
            }
            case 0x2F: // VNOP
                return;
            case 0x30: // VMOVE
                write_vf(ctx, ft, S, dest);
                return;
            case 0x31: // VMR32
                R = Vec4f{S.y, S.z, S.w, S.x};
                write_vf(ctx, ft, R, dest);
                return;
            case 0x38: { // VDIV
                u8 fsf = (raw >> 21) & 3u;
                u8 ftf = (raw >> 23) & 3u;
                f32 denom = T[ftf];
                set_q(ctx, denom == 0.0f ? 0.0f : (S[fsf] / denom));
                return;
            }
            case 0x39: { // VSQRT
                u8 ftf = (raw >> 23) & 3u;
                set_q(ctx, std::sqrt(std::fabs(T[ftf])));
                return;
            }
            case 0x3A: { // VRSQRT
                u8 fsf = (raw >> 21) & 3u;
                u8 ftf = (raw >> 23) & 3u;
                f32 denom = std::sqrt(std::fabs(T[ftf]));
                set_q(ctx, denom == 0.0f ? 0.0f : (S[fsf] / denom));
                return;
            }
            case 0x3B: // VWAITQ
                return;
            default:
                m_unsupported_ops++;
                return;
        }
    }

    // Special1 table (fn < 0x3C)
    switch (fn) {
        case 0x00: case 0x01: case 0x02: case 0x03: { // VADDx/y/z/w
            f32 bc = T[fn & 3];
            for (int i = 0; i < 4; ++i) R[i] = S[i] + bc;
            write_vf(ctx, fd, R, dest);
            update_mac_flags(ctx, R, dest);
            break;
        }
        case 0x04: case 0x05: case 0x06: case 0x07: { // VSUBx/y/z/w
            f32 bc = T[fn & 3];
            for (int i = 0; i < 4; ++i) R[i] = S[i] - bc;
            write_vf(ctx, fd, R, dest);
            update_mac_flags(ctx, R, dest);
            break;
        }
        case 0x08: case 0x09: case 0x0A: case 0x0B: { // VMADDx/y/z/w
            f32 bc = T[fn & 3];
            for (int i = 0; i < 4; ++i) R[i] = ctx.vu0_acc[i] + S[i] * bc;
            write_vf(ctx, fd, R, dest);
            update_mac_flags(ctx, R, dest);
            break;
        }
        case 0x0C: case 0x0D: case 0x0E: case 0x0F: { // VMSUBx/y/z/w
            f32 bc = T[fn & 3];
            for (int i = 0; i < 4; ++i) R[i] = ctx.vu0_acc[i] - S[i] * bc;
            write_vf(ctx, fd, R, dest);
            update_mac_flags(ctx, R, dest);
            break;
        }
        case 0x10: case 0x11: case 0x12: case 0x13: { // VMAXx/y/z/w
            f32 bc = T[fn & 3];
            for (int i = 0; i < 4; ++i) R[i] = std::max(S[i], bc);
            write_vf(ctx, fd, R, dest);
            break;
        }
        case 0x14: case 0x15: case 0x16: case 0x17: { // VMINIx/y/z/w
            f32 bc = T[fn & 3];
            for (int i = 0; i < 4; ++i) R[i] = std::min(S[i], bc);
            write_vf(ctx, fd, R, dest);
            break;
        }
        case 0x18: case 0x19: case 0x1A: case 0x1B: { // VMULx/y/z/w
            f32 bc = T[fn & 3];
            for (int i = 0; i < 4; ++i) R[i] = S[i] * bc;
            write_vf(ctx, fd, R, dest);
            update_mac_flags(ctx, R, dest);
            break;
        }
        case 0x1C: // VMULq
            for (int i = 0; i < 4; ++i) R[i] = S[i] * get_q(ctx);
            write_vf(ctx, fd, R, dest);
            update_mac_flags(ctx, R, dest);
            break;
        case 0x1D: // VMAXi
            for (int i = 0; i < 4; ++i) R[i] = std::max(S[i], get_i(ctx));
            write_vf(ctx, fd, R, dest);
            break;
        case 0x1E: // VMULi
            for (int i = 0; i < 4; ++i) R[i] = S[i] * get_i(ctx);
            write_vf(ctx, fd, R, dest);
            update_mac_flags(ctx, R, dest);
            break;
        case 0x1F: // VMINIi
            for (int i = 0; i < 4; ++i) R[i] = std::min(S[i], get_i(ctx));
            write_vf(ctx, fd, R, dest);
            break;
        case 0x20: // VADDq
            for (int i = 0; i < 4; ++i) R[i] = S[i] + get_q(ctx);
            write_vf(ctx, fd, R, dest);
            update_mac_flags(ctx, R, dest);
            break;
        case 0x21: // VMADDq
            for (int i = 0; i < 4; ++i) R[i] = ctx.vu0_acc[i] + S[i] * get_q(ctx);
            write_vf(ctx, fd, R, dest);
            update_mac_flags(ctx, R, dest);
            break;
        case 0x22: // VADDi
            for (int i = 0; i < 4; ++i) R[i] = S[i] + get_i(ctx);
            write_vf(ctx, fd, R, dest);
            update_mac_flags(ctx, R, dest);
            break;
        case 0x23: // VMADDi
            for (int i = 0; i < 4; ++i) R[i] = ctx.vu0_acc[i] + S[i] * get_i(ctx);
            write_vf(ctx, fd, R, dest);
            update_mac_flags(ctx, R, dest);
            break;
        case 0x24: // VSUBq
            for (int i = 0; i < 4; ++i) R[i] = S[i] - get_q(ctx);
            write_vf(ctx, fd, R, dest);
            update_mac_flags(ctx, R, dest);
            break;
        case 0x25: // VMSUBq
            for (int i = 0; i < 4; ++i) R[i] = ctx.vu0_acc[i] - S[i] * get_q(ctx);
            write_vf(ctx, fd, R, dest);
            update_mac_flags(ctx, R, dest);
            break;
        case 0x26: // VSUBi
            for (int i = 0; i < 4; ++i) R[i] = S[i] - get_i(ctx);
            write_vf(ctx, fd, R, dest);
            update_mac_flags(ctx, R, dest);
            break;
        case 0x27: // VMSUBi
            for (int i = 0; i < 4; ++i) R[i] = ctx.vu0_acc[i] - S[i] * get_i(ctx);
            write_vf(ctx, fd, R, dest);
            update_mac_flags(ctx, R, dest);
            break;
        case 0x28: // VADD
            for (int i = 0; i < 4; ++i) R[i] = S[i] + T[i];
            write_vf(ctx, fd, R, dest);
            update_mac_flags(ctx, R, dest);
            break;
        case 0x29: // VMADD
            for (int i = 0; i < 4; ++i) R[i] = ctx.vu0_acc[i] + S[i] * T[i];
            write_vf(ctx, fd, R, dest);
            update_mac_flags(ctx, R, dest);
            break;
        case 0x2A: // VMUL
            for (int i = 0; i < 4; ++i) R[i] = S[i] * T[i];
            write_vf(ctx, fd, R, dest);
            update_mac_flags(ctx, R, dest);
            break;
        case 0x2B: // VMAX
            for (int i = 0; i < 4; ++i) R[i] = std::max(S[i], T[i]);
            write_vf(ctx, fd, R, dest);
            break;
        case 0x2C: // VSUB
            for (int i = 0; i < 4; ++i) R[i] = S[i] - T[i];
            write_vf(ctx, fd, R, dest);
            update_mac_flags(ctx, R, dest);
            break;
        case 0x2D: // VMSUB
            for (int i = 0; i < 4; ++i) R[i] = ctx.vu0_acc[i] - S[i] * T[i];
            write_vf(ctx, fd, R, dest);
            update_mac_flags(ctx, R, dest);
            break;
        case 0x2E: { // VOPMSUB.xyz
            R.x = ctx.vu0_acc.x - S.y * T.z;
            R.y = ctx.vu0_acc.y - S.z * T.x;
            R.z = ctx.vu0_acc.z - S.x * T.y;
            write_vf(ctx, fd, R, 0x0Eu);
            update_mac_flags(ctx, R, 0x0Eu);
            break;
        }
        case 0x2F: // VMINI
            for (int i = 0; i < 4; ++i) R[i] = std::min(S[i], T[i]);
            write_vf(ctx, fd, R, dest);
            break;
        default:
            m_unsupported_ops++;
            break;
    }
}

} // namespace tf::ps2
