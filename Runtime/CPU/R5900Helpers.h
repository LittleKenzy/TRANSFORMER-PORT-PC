#pragma once

#include "EEContext.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace tf::ps2::helpers {

// --- Unaligned MIPS Loads & Stores (Little-Endian) ---

inline u32 lwl(const GuestMemory& mem, u32 addr, u32 cur_rt) {
    u32 shift = (addr & 3u) * 8u;
    u32 word = mem.read32(addr & ~3u);
    u32 mask = 0x00FFFFFFu >> shift;
    return (cur_rt & mask) | (word << (24u - shift));
}

inline u32 lwr(const GuestMemory& mem, u32 addr, u32 cur_rt) {
    u32 shift = (addr & 3u) * 8u;
    u32 word = mem.read32(addr & ~3u);
    u32 mask = (shift == 0) ? 0u : (0xFFFFFF00u << (24u - shift));
    return (cur_rt & mask) | (word >> shift);
}

inline void swl(GuestMemory& mem, u32 addr, u32 val) {
    u32 shift = (addr & 3u) * 8u;
    u32 aligned = addr & ~3u;
    u32 orig = mem.read32(aligned);
    u32 mask = (shift == 24u) ? 0u : (0xFFFFFF00u << shift);
    mem.write32(aligned, (orig & mask) | (val >> (24u - shift)));
}

inline void swr(GuestMemory& mem, u32 addr, u32 val) {
    u32 shift = (addr & 3u) * 8u;
    u32 aligned = addr & ~3u;
    u32 orig = mem.read32(aligned);
    u32 mask = (shift == 0) ? 0u : (0x00FFFFFFu >> (24u - shift));
    mem.write32(aligned, (orig & mask) | (val << shift));
}

inline u64 ldl(const GuestMemory& mem, u32 addr, u64 cur_rt) {
    u32 shift = (addr & 7u) * 8u;
    u64 dw = mem.read64(addr & ~7u);
    u64 mask = (shift == 56u) ? 0ULL : (0x00FFFFFFFFFFFFFFULL >> shift);
    return (cur_rt & mask) | (dw << (56u - shift));
}

inline u64 ldr(const GuestMemory& mem, u32 addr, u64 cur_rt) {
    u32 shift = (addr & 7u) * 8u;
    u64 dw = mem.read64(addr & ~7u);
    u64 mask = (shift == 0) ? 0ULL : (0xFFFFFFFFFFFFFF00ULL << (56u - shift));
    return (cur_rt & mask) | (dw >> shift);
}

inline void sdl(GuestMemory& mem, u32 addr, u64 val) {
    u32 shift = (addr & 7u) * 8u;
    u32 aligned = addr & ~7u;
    u64 orig = mem.read64(aligned);
    u64 mask = (shift == 56u) ? 0ULL : (0xFFFFFFFFFFFFFF00ULL << shift);
    mem.write64(aligned, (orig & mask) | (val >> (56u - shift)));
}

inline void sdr(GuestMemory& mem, u32 addr, u64 val) {
    u32 shift = (addr & 7u) * 8u;
    u32 aligned = addr & ~7u;
    u64 orig = mem.read64(aligned);
    u64 mask = (shift == 0) ? 0ULL : (0x00FFFFFFFFFFFFFFULL >> (56u - shift));
    mem.write64(aligned, (orig & mask) | (val << shift));
}

// --- Multiply / Divide (Pipeline 0 & 1) ---

inline void mult_op(EEContext& ctx, u8 rd, u8 rs, u8 rt, bool pipe1) {
    s64 prod = static_cast<s64>(ctx.gpr_s32(rs)) * static_cast<s64>(ctx.gpr_s32(rt));
    s64 lo_sext = static_cast<s64>(static_cast<s32>(prod & 0xFFFFFFFFLL));
    s64 hi_sext = static_cast<s64>(static_cast<s32>((prod >> 32) & 0xFFFFFFFFLL));
    if (pipe1) {
        ctx.lo1 = static_cast<u64>(lo_sext);
        ctx.hi1 = static_cast<u64>(hi_sext);
    } else {
        ctx.lo = static_cast<u64>(lo_sext);
        ctx.hi = static_cast<u64>(hi_sext);
    }
    if (rd != 0) ctx.set_gpr_s64(rd, lo_sext);
}

inline void multu_op(EEContext& ctx, u8 rd, u8 rs, u8 rt, bool pipe1) {
    u64 prod = static_cast<u64>(ctx.gpr_u32(rs)) * static_cast<u64>(ctx.gpr_u32(rt));
    s64 lo_sext = static_cast<s64>(static_cast<s32>(prod & 0xFFFFFFFFULL));
    s64 hi_sext = static_cast<s64>(static_cast<s32>((prod >> 32) & 0xFFFFFFFFULL));
    if (pipe1) {
        ctx.lo1 = static_cast<u64>(lo_sext);
        ctx.hi1 = static_cast<u64>(hi_sext);
    } else {
        ctx.lo = static_cast<u64>(lo_sext);
        ctx.hi = static_cast<u64>(hi_sext);
    }
    if (rd != 0) ctx.set_gpr_s64(rd, lo_sext);
}

inline void madd_op(EEContext& ctx, u8 rd, u8 rs, u8 rt, bool pipe1) {
    u64 cur_lo = pipe1 ? ctx.lo1 : ctx.lo;
    u64 cur_hi = pipe1 ? ctx.hi1 : ctx.hi;
    s64 acc = static_cast<s64>((cur_hi << 32) | (cur_lo & 0xFFFFFFFFULL));
    s64 prod = static_cast<s64>(ctx.gpr_s32(rs)) * static_cast<s64>(ctx.gpr_s32(rt));
    s64 sum = acc + prod;
    s64 lo_sext = static_cast<s64>(static_cast<s32>(sum & 0xFFFFFFFFLL));
    s64 hi_sext = static_cast<s64>(static_cast<s32>((sum >> 32) & 0xFFFFFFFFLL));
    if (pipe1) {
        ctx.lo1 = static_cast<u64>(lo_sext);
        ctx.hi1 = static_cast<u64>(hi_sext);
    } else {
        ctx.lo = static_cast<u64>(lo_sext);
        ctx.hi = static_cast<u64>(hi_sext);
    }
    if (rd != 0) ctx.set_gpr_s64(rd, lo_sext);
}

inline void maddu_op(EEContext& ctx, u8 rd, u8 rs, u8 rt, bool pipe1) {
    u64 cur_lo = pipe1 ? ctx.lo1 : ctx.lo;
    u64 cur_hi = pipe1 ? ctx.hi1 : ctx.hi;
    u64 acc = (cur_hi << 32) | (cur_lo & 0xFFFFFFFFULL);
    u64 prod = static_cast<u64>(ctx.gpr_u32(rs)) * static_cast<u64>(ctx.gpr_u32(rt));
    u64 sum = acc + prod;
    s64 lo_sext = static_cast<s64>(static_cast<s32>(sum & 0xFFFFFFFFULL));
    s64 hi_sext = static_cast<s64>(static_cast<s32>((sum >> 32) & 0xFFFFFFFFULL));
    if (pipe1) {
        ctx.lo1 = static_cast<u64>(lo_sext);
        ctx.hi1 = static_cast<u64>(hi_sext);
    } else {
        ctx.lo = static_cast<u64>(lo_sext);
        ctx.hi = static_cast<u64>(hi_sext);
    }
    if (rd != 0) ctx.set_gpr_s64(rd, lo_sext);
}

inline void div_op(EEContext& ctx, u8 rs, u8 rt, bool pipe1) {
    s32 a = ctx.gpr_s32(rs);
    s32 b = ctx.gpr_s32(rt);
    s32 q = 0, r = 0;
    if (b == 0) {
        q = (a < 0) ? 1 : -1;
        r = a;
    } else if (a == std::numeric_limits<s32>::min() && b == -1) {
        q = std::numeric_limits<s32>::min();
        r = 0;
    } else {
        q = a / b;
        r = a % b;
    }
    if (pipe1) {
        ctx.lo1 = static_cast<u64>(static_cast<s64>(q));
        ctx.hi1 = static_cast<u64>(static_cast<s64>(r));
    } else {
        ctx.lo = static_cast<u64>(static_cast<s64>(q));
        ctx.hi = static_cast<u64>(static_cast<s64>(r));
    }
}

inline void divu_op(EEContext& ctx, u8 rs, u8 rt, bool pipe1) {
    u32 a = ctx.gpr_u32(rs);
    u32 b = ctx.gpr_u32(rt);
    u32 q = 0, r = 0;
    if (b == 0) {
        q = 0xFFFFFFFFu;
        r = a;
    } else {
        q = a / b;
        r = a % b;
    }
    if (pipe1) {
        ctx.lo1 = static_cast<u64>(static_cast<s64>(static_cast<s32>(q)));
        ctx.hi1 = static_cast<u64>(static_cast<s64>(static_cast<s32>(r)));
    } else {
        ctx.lo = static_cast<u64>(static_cast<s64>(static_cast<s32>(q)));
        ctx.hi = static_cast<u64>(static_cast<s64>(static_cast<s32>(r)));
    }
}

// --- Saturating & SIMD MMI Helpers ---

template <typename T>
inline T clamp_sat(s64 v) {
    if (v > static_cast<s64>(std::numeric_limits<T>::max())) return std::numeric_limits<T>::max();
    if (v < static_cast<s64>(std::numeric_limits<T>::min())) return std::numeric_limits<T>::min();
    return static_cast<T>(v);
}

inline u32 plzcw_word(u32 v) {
    u32 sign = (v >> 31) & 1u;
    u32 count = 0;
    for (int bit = 30; bit >= 0; --bit) {
        if (((v >> bit) & 1u) == sign) count++;
        else break;
    }
    return count;
}

inline u128 mmi_paddw(u128 a, u128 b) { u128 r{}; for (int i = 0; i < 4; ++i) r.set_uw(i, a.uw(i) + b.uw(i)); return r; }
inline u128 mmi_psubw(u128 a, u128 b) { u128 r{}; for (int i = 0; i < 4; ++i) r.set_uw(i, a.uw(i) - b.uw(i)); return r; }
inline u128 mmi_pcgtw(u128 a, u128 b) { u128 r{}; for (int i = 0; i < 4; ++i) r.set_uw(i, a.sw(i) > b.sw(i) ? 0xFFFFFFFFu : 0u); return r; }
inline u128 mmi_pceqw(u128 a, u128 b) { u128 r{}; for (int i = 0; i < 4; ++i) r.set_uw(i, a.uw(i) == b.uw(i) ? 0xFFFFFFFFu : 0u); return r; }
inline u128 mmi_pmaxw(u128 a, u128 b) { u128 r{}; for (int i = 0; i < 4; ++i) r.set_uw(i, static_cast<u32>(std::max(a.sw(i), b.sw(i)))); return r; }
inline u128 mmi_pminw(u128 a, u128 b) { u128 r{}; for (int i = 0; i < 4; ++i) r.set_uw(i, static_cast<u32>(std::min(a.sw(i), b.sw(i)))); return r; }
inline u128 mmi_paddsw(u128 a, u128 b){ u128 r{}; for (int i = 0; i < 4; ++i) r.set_uw(i, static_cast<u32>(clamp_sat<s32>(static_cast<s64>(a.sw(i)) + b.sw(i)))); return r; }
inline u128 mmi_psubsw(u128 a, u128 b){ u128 r{}; for (int i = 0; i < 4; ++i) r.set_uw(i, static_cast<u32>(clamp_sat<s32>(static_cast<s64>(a.sw(i)) - b.sw(i)))); return r; }
inline u128 mmi_padduw(u128 a, u128 b){ u128 r{}; for (int i = 0; i < 4; ++i) { u64 s = static_cast<u64>(a.uw(i)) + b.uw(i); r.set_uw(i, s > 0xFFFFFFFFULL ? 0xFFFFFFFFu : static_cast<u32>(s)); } return r; }
inline u128 mmi_psubuw(u128 a, u128 b){ u128 r{}; for (int i = 0; i < 4; ++i) r.set_uw(i, a.uw(i) > b.uw(i) ? (a.uw(i) - b.uw(i)) : 0u); return r; }
inline u128 mmi_pabsw(u128 b) { u128 r{}; for (int i = 0; i < 4; ++i) { s32 v = b.sw(i); r.set_uw(i, v == std::numeric_limits<s32>::min() ? 0x7FFFFFFFu : static_cast<u32>(v < 0 ? -v : v)); } return r; }

inline u128 mmi_paddh(u128 a, u128 b) { u128 r{}; for (int i = 0; i < 8; ++i) r.set_uh(i, static_cast<u16>(a.uh(i) + b.uh(i))); return r; }
inline u128 mmi_psubh(u128 a, u128 b) { u128 r{}; for (int i = 0; i < 8; ++i) r.set_uh(i, static_cast<u16>(a.uh(i) - b.uh(i))); return r; }
inline u128 mmi_pcgth(u128 a, u128 b) { u128 r{}; for (int i = 0; i < 8; ++i) r.set_uh(i, a.sh(i) > b.sh(i) ? 0xFFFFu : 0u); return r; }
inline u128 mmi_pceqh(u128 a, u128 b) { u128 r{}; for (int i = 0; i < 8; ++i) r.set_uh(i, a.uh(i) == b.uh(i) ? 0xFFFFu : 0u); return r; }
inline u128 mmi_pmaxh(u128 a, u128 b) { u128 r{}; for (int i = 0; i < 8; ++i) r.set_uh(i, static_cast<u16>(std::max(a.sh(i), b.sh(i)))); return r; }
inline u128 mmi_pminh(u128 a, u128 b) { u128 r{}; for (int i = 0; i < 8; ++i) r.set_uh(i, static_cast<u16>(std::min(a.sh(i), b.sh(i)))); return r; }
inline u128 mmi_paddsh(u128 a, u128 b){ u128 r{}; for (int i = 0; i < 8; ++i) r.set_uh(i, static_cast<u16>(clamp_sat<s16>(static_cast<s64>(a.sh(i)) + b.sh(i)))); return r; }
inline u128 mmi_psubsh(u128 a, u128 b){ u128 r{}; for (int i = 0; i < 8; ++i) r.set_uh(i, static_cast<u16>(clamp_sat<s16>(static_cast<s64>(a.sh(i)) - b.sh(i)))); return r; }
inline u128 mmi_padduh(u128 a, u128 b){ u128 r{}; for (int i = 0; i < 8; ++i) { u32 s = static_cast<u32>(a.uh(i)) + b.uh(i); r.set_uh(i, s > 0xFFFFu ? 0xFFFFu : static_cast<u16>(s)); } return r; }
inline u128 mmi_psubuh(u128 a, u128 b){ u128 r{}; for (int i = 0; i < 8; ++i) r.set_uh(i, a.uh(i) > b.uh(i) ? static_cast<u16>(a.uh(i) - b.uh(i)) : 0u); return r; }
inline u128 mmi_pabsh(u128 b) { u128 r{}; for (int i = 0; i < 8; ++i) { s16 v = b.sh(i); r.set_uh(i, v == std::numeric_limits<s16>::min() ? 0x7FFFu : static_cast<u16>(v < 0 ? -v : v)); } return r; }

inline u128 mmi_paddb(u128 a, u128 b) { u128 r{}; for (int i = 0; i < 16; ++i) r.set_ub(i, static_cast<u8>(a.ub(i) + b.ub(i))); return r; }
inline u128 mmi_psubb(u128 a, u128 b) { u128 r{}; for (int i = 0; i < 16; ++i) r.set_ub(i, static_cast<u8>(a.ub(i) - b.ub(i))); return r; }
inline u128 mmi_pcgtb(u128 a, u128 b) { u128 r{}; for (int i = 0; i < 16; ++i) r.set_ub(i, a.sb(i) > b.sb(i) ? 0xFFu : 0u); return r; }
inline u128 mmi_pceqb(u128 a, u128 b) { u128 r{}; for (int i = 0; i < 16; ++i) r.set_ub(i, a.ub(i) == b.ub(i) ? 0xFFu : 0u); return r; }
inline u128 mmi_paddsb(u128 a, u128 b){ u128 r{}; for (int i = 0; i < 16; ++i) r.set_ub(i, static_cast<u8>(clamp_sat<s8>(static_cast<s64>(a.sb(i)) + b.sb(i)))); return r; }
inline u128 mmi_psubsb(u128 a, u128 b){ u128 r{}; for (int i = 0; i < 16; ++i) r.set_ub(i, static_cast<u8>(clamp_sat<s8>(static_cast<s64>(a.sb(i)) - b.sb(i)))); return r; }
inline u128 mmi_paddub(u128 a, u128 b){ u128 r{}; for (int i = 0; i < 16; ++i) { u32 s = static_cast<u32>(a.ub(i)) + b.ub(i); r.set_ub(i, s > 0xFFu ? 0xFFu : static_cast<u8>(s)); } return r; }
inline u128 mmi_psubub(u128 a, u128 b){ u128 r{}; for (int i = 0; i < 16; ++i) r.set_ub(i, a.ub(i) > b.ub(i) ? static_cast<u8>(a.ub(i) - b.ub(i)) : 0u); return r; }

inline u128 mmi_pextlw(u128 a, u128 b) { u128 r{}; r.set_uw(0, b.uw(0)); r.set_uw(1, a.uw(0)); r.set_uw(2, b.uw(1)); r.set_uw(3, a.uw(1)); return r; }
inline u128 mmi_pextuw(u128 a, u128 b) { u128 r{}; r.set_uw(0, b.uw(2)); r.set_uw(1, a.uw(2)); r.set_uw(2, b.uw(3)); r.set_uw(3, a.uw(3)); return r; }
inline u128 mmi_ppacw(u128 a, u128 b)  { u128 r{}; r.set_uw(0, b.uw(0)); r.set_uw(1, b.uw(2)); r.set_uw(2, a.uw(0)); r.set_uw(3, a.uw(2)); return r; }
inline u128 mmi_pextlh(u128 a, u128 b) { u128 r{}; for (int i = 0; i < 4; ++i) { r.set_uh(i * 2, b.uh(i)); r.set_uh(i * 2 + 1, a.uh(i)); } return r; }
inline u128 mmi_pextuh(u128 a, u128 b) { u128 r{}; for (int i = 0; i < 4; ++i) { r.set_uh(i * 2, b.uh(i + 4)); r.set_uh(i * 2 + 1, a.uh(i + 4)); } return r; }
inline u128 mmi_ppach(u128 a, u128 b)  { u128 r{}; for (int i = 0; i < 4; ++i) { r.set_uh(i, b.uh(i * 2)); r.set_uh(i + 4, a.uh(i * 2)); } return r; }
inline u128 mmi_pextlb(u128 a, u128 b) { u128 r{}; for (int i = 0; i < 8; ++i) { r.set_ub(i * 2, b.ub(i)); r.set_ub(i * 2 + 1, a.ub(i)); } return r; }
inline u128 mmi_pextub(u128 a, u128 b) { u128 r{}; for (int i = 0; i < 8; ++i) { r.set_ub(i * 2, b.ub(i + 8)); r.set_ub(i * 2 + 1, a.ub(i + 8)); } return r; }
inline u128 mmi_ppacb(u128 a, u128 b)  { u128 r{}; for (int i = 0; i < 8; ++i) { r.set_ub(i, b.ub(i * 2)); r.set_ub(i + 8, a.ub(i * 2)); } return r; }
inline u128 mmi_pcpyld(u128 a, u128 b) { return u128(b.lo, a.lo); }
inline u128 mmi_pcpyud(u128 a, u128 b) { return u128(a.hi, b.hi); }
inline u128 mmi_pcpyh(u128 b) {
    u128 r{};
    u16 h0 = b.uh(0), h4 = b.uh(4);
    for (int i = 0; i < 4; ++i) { r.set_uh(i, h0); r.set_uh(i + 4, h4); }
    return r;
}
inline u128 mmi_prot3w(u128 b) {
    u128 r{}; r.set_uw(0, b.uw(1)); r.set_uw(1, b.uw(2)); r.set_uw(2, b.uw(0)); r.set_uw(3, b.uw(3)); return r;
}
inline u128 mmi_pexew(u128 b) {
    u128 r{}; r.set_uw(0, b.uw(2)); r.set_uw(1, b.uw(1)); r.set_uw(2, b.uw(0)); r.set_uw(3, b.uw(3)); return r;
}
inline u128 mmi_pexcw(u128 b) {
    u128 r{}; r.set_uw(0, b.uw(0)); r.set_uw(1, b.uw(2)); r.set_uw(2, b.uw(1)); r.set_uw(3, b.uw(3)); return r;
}
inline u128 mmi_qfsrv(u128 a, u128 b, u32 sa) {
    u32 bit_shift = (sa & 0xFu) * 8u;
    if (bit_shift == 0) return b;
    u128 r{};
    for (int i = 0; i < 16; ++i) {
        u32 src_byte = static_cast<u32>(i) + (sa & 0xFu);
        r.set_ub(i, src_byte < 16 ? b.ub(src_byte) : a.ub(src_byte - 16));
    }
    return r;
}
inline u128 mmi_psllh(u128 b, u8 sa) { u128 r{}; u8 s = sa & 0xFu; for (int i = 0; i < 8; ++i) r.set_uh(i, static_cast<u16>(b.uh(i) << s)); return r; }
inline u128 mmi_psrlh(u128 b, u8 sa) { u128 r{}; u8 s = sa & 0xFu; for (int i = 0; i < 8; ++i) r.set_uh(i, static_cast<u16>(b.uh(i) >> s)); return r; }
inline u128 mmi_psrah(u128 b, u8 sa) { u128 r{}; u8 s = sa & 0xFu; for (int i = 0; i < 8; ++i) r.set_uh(i, static_cast<u16>(b.sh(i) >> s)); return r; }
inline u128 mmi_psllw(u128 b, u8 sa) { u128 r{}; u8 s = sa & 0x1Fu; for (int i = 0; i < 4; ++i) r.set_uw(i, b.uw(i) << s); return r; }
inline u128 mmi_psrlw(u128 b, u8 sa) { u128 r{}; u8 s = sa & 0x1Fu; for (int i = 0; i < 4; ++i) r.set_uw(i, b.uw(i) >> s); return r; }
inline u128 mmi_psraw(u128 b, u8 sa) { u128 r{}; u8 s = sa & 0x1Fu; for (int i = 0; i < 4; ++i) r.set_uw(i, static_cast<u32>(b.sw(i) >> s)); return r; }

inline u128 mmi_pmfhl(const EEContext& ctx, u8 fmt) {
    u128 r{};
    switch (fmt) {
        case 0: // LW: 4x low 32-bit words of LO, HI, LO1, HI1
            r.set_uw(0, static_cast<u32>(ctx.lo));
            r.set_uw(1, static_cast<u32>(ctx.hi));
            r.set_uw(2, static_cast<u32>(ctx.lo1));
            r.set_uw(3, static_cast<u32>(ctx.hi1));
            break;
        case 1: // UW: 4x high 32-bit words of LO, HI, LO1, HI1
            r.set_uw(0, static_cast<u32>(ctx.lo >> 32));
            r.set_uw(1, static_cast<u32>(ctx.hi >> 32));
            r.set_uw(2, static_cast<u32>(ctx.lo1 >> 32));
            r.set_uw(3, static_cast<u32>(ctx.hi1 >> 32));
            break;
        case 2: { // SLW: signed 32-bit saturated from 64-bit (HI:LO) and (HI1:LO1), sign-extended to 64 bits
            s64 v0 = static_cast<s64>((static_cast<u64>(static_cast<u32>(ctx.hi)) << 32) | static_cast<u32>(ctx.lo));
            s64 v1 = static_cast<s64>((static_cast<u64>(static_cast<u32>(ctx.hi1)) << 32) | static_cast<u32>(ctx.lo1));
            r.lo = static_cast<u64>(static_cast<s64>(clamp_sat<s32>(v0)));
            r.hi = static_cast<u64>(static_cast<s64>(clamp_sat<s32>(v1)));
            break;
        }
        case 3: // LH: 8x low 16-bit halfwords of the 8 32-bit lanes in LO, HI, LO1, HI1
            r.set_uh(0, static_cast<u16>(ctx.lo & 0xFFFFu));
            r.set_uh(1, static_cast<u16>((ctx.lo >> 32) & 0xFFFFu));
            r.set_uh(2, static_cast<u16>(ctx.hi & 0xFFFFu));
            r.set_uh(3, static_cast<u16>((ctx.hi >> 32) & 0xFFFFu));
            r.set_uh(4, static_cast<u16>(ctx.lo1 & 0xFFFFu));
            r.set_uh(5, static_cast<u16>((ctx.lo1 >> 32) & 0xFFFFu));
            r.set_uh(6, static_cast<u16>(ctx.hi1 & 0xFFFFu));
            r.set_uh(7, static_cast<u16>((ctx.hi1 >> 32) & 0xFFFFu));
            break;
        case 4: // SH: 8x signed 16-bit saturated halfwords of the 8 32-bit lanes in LO, HI, LO1, HI1
            r.set_uh(0, static_cast<u16>(clamp_sat<s16>(static_cast<s32>(ctx.lo & 0xFFFFFFFFULL))));
            r.set_uh(1, static_cast<u16>(clamp_sat<s16>(static_cast<s32>(ctx.lo >> 32))));
            r.set_uh(2, static_cast<u16>(clamp_sat<s16>(static_cast<s32>(ctx.hi & 0xFFFFFFFFULL))));
            r.set_uh(3, static_cast<u16>(clamp_sat<s16>(static_cast<s32>(ctx.hi >> 32))));
            r.set_uh(4, static_cast<u16>(clamp_sat<s16>(static_cast<s32>(ctx.lo1 & 0xFFFFFFFFULL))));
            r.set_uh(5, static_cast<u16>(clamp_sat<s16>(static_cast<s32>(ctx.lo1 >> 32))));
            r.set_uh(6, static_cast<u16>(clamp_sat<s16>(static_cast<s32>(ctx.hi1 & 0xFFFFFFFFULL))));
            r.set_uh(7, static_cast<u16>(clamp_sat<s16>(static_cast<s32>(ctx.hi1 >> 32))));
            break;
        default:
            r.set_uw(0, static_cast<u32>(ctx.lo));
            r.set_uw(1, static_cast<u32>(ctx.hi));
            r.set_uw(2, static_cast<u32>(ctx.lo1));
            r.set_uw(3, static_cast<u32>(ctx.hi1));
            break;
    }
    return r;
}

inline void mmi_pmultw(EEContext& ctx, u8 rd, u128 a, u128 b) {
    s64 p0 = static_cast<s64>(a.sw(0)) * static_cast<s64>(b.sw(0));
    s64 p2 = static_cast<s64>(a.sw(2)) * static_cast<s64>(b.sw(2));
    ctx.lo  = static_cast<u64>(static_cast<s64>(static_cast<s32>(p0 & 0xFFFFFFFFLL)));
    ctx.hi  = static_cast<u64>(static_cast<s64>(static_cast<s32>((p0 >> 32) & 0xFFFFFFFFLL)));
    ctx.lo1 = static_cast<u64>(static_cast<s64>(static_cast<s32>(p2 & 0xFFFFFFFFLL)));
    ctx.hi1 = static_cast<u64>(static_cast<s64>(static_cast<s32>((p2 >> 32) & 0xFFFFFFFFLL)));
    if (rd != 0) ctx.set_gpr_u128(rd, u128(static_cast<u64>(p0), static_cast<u64>(p2)));
}

inline void mmi_pmultuw(EEContext& ctx, u8 rd, u128 a, u128 b) {
    u64 p0 = static_cast<u64>(a.uw(0)) * static_cast<u64>(b.uw(0));
    u64 p2 = static_cast<u64>(a.uw(2)) * static_cast<u64>(b.uw(2));
    ctx.lo  = static_cast<u64>(static_cast<s64>(static_cast<s32>(p0 & 0xFFFFFFFFULL)));
    ctx.hi  = static_cast<u64>(static_cast<s64>(static_cast<s32>((p0 >> 32) & 0xFFFFFFFFULL)));
    ctx.lo1 = static_cast<u64>(static_cast<s64>(static_cast<s32>(p2 & 0xFFFFFFFFULL)));
    ctx.hi1 = static_cast<u64>(static_cast<s64>(static_cast<s32>((p2 >> 32) & 0xFFFFFFFFULL)));
    if (rd != 0) ctx.set_gpr_u128(rd, u128(p0, p2));
}

inline u128 mmi_pext5(u128 b) {
    u128 r{};
    for (int i = 0; i < 4; ++i) {
        u32 w = b.uw(i);
        u32 red   = ((w >> 0) & 0x1Fu) << 3;
        u32 green = ((w >> 5) & 0x1Fu) << 3;
        u32 blue  = ((w >> 10) & 0x1Fu) << 3;
        u32 alpha = ((w >> 15) & 0x1u) << 7;
        r.set_uw(i, red | (green << 8) | (blue << 16) | (alpha << 24));
    }
    return r;
}

inline u128 mmi_ppac5(u128 b) {
    u128 r{};
    for (int i = 0; i < 4; ++i) {
        u32 w = b.uw(i);
        u32 red   = (w >> 3) & 0x1Fu;
        u32 green = (w >> 11) & 0x1Fu;
        u32 blue  = (w >> 19) & 0x1Fu;
        u32 alpha = (w >> 31) & 0x1u;
        r.set_uw(i, red | (green << 5) | (blue << 10) | (alpha << 15));
    }
    return r;
}

inline u128 mmi_pinth(u128 a, u128 b) {
    u128 r{};
    for (int i = 0; i < 4; ++i) {
        r.set_uh(i * 2, b.uh(i));
        r.set_uh(i * 2 + 1, a.uh(i + 4));
    }
    return r;
}

inline u128 mmi_pinteh(u128 a, u128 b) {
    u128 r{};
    for (int i = 0; i < 4; ++i) {
        r.set_uh(i * 2, b.uh(i * 2));
        r.set_uh(i * 2 + 1, a.uh(i * 2));
    }
    return r;
}

inline u128 mmi_prevh(u128 b) {
    u128 r{};
    for (int i = 0; i < 4; ++i) {
        r.set_uh(i, b.uh(3 - i));
        r.set_uh(i + 4, b.uh(7 - i));
    }
    return r;
}

inline u128 mmi_pexeh(u128 b) {
    u128 r{};
    r.set_uh(0, b.uh(2)); r.set_uh(1, b.uh(1)); r.set_uh(2, b.uh(0)); r.set_uh(3, b.uh(3));
    r.set_uh(4, b.uh(6)); r.set_uh(5, b.uh(5)); r.set_uh(6, b.uh(4)); r.set_uh(7, b.uh(7));
    return r;
}

inline u128 mmi_pexch(u128 b) {
    u128 r{};
    r.set_uh(0, b.uh(0)); r.set_uh(1, b.uh(2)); r.set_uh(2, b.uh(1)); r.set_uh(3, b.uh(3));
    r.set_uh(4, b.uh(4)); r.set_uh(5, b.uh(6)); r.set_uh(6, b.uh(5)); r.set_uh(7, b.uh(7));
    return r;
}

inline u128 mmi_padsbh(u128 a, u128 b) {
    u128 r{};
    for (int i = 0; i < 4; ++i) r.set_uh(i, static_cast<u16>(a.uh(i) - b.uh(i)));
    for (int i = 4; i < 8; ++i) r.set_uh(i, static_cast<u16>(a.uh(i) + b.uh(i)));
    return r;
}

inline void mmi_pmulth(EEContext& ctx, u8 rd, u128 a, u128 b) {
    // R5900 PMULTH: multiplies 8 pairs of signed 16-bit lanes -> 8 32-bit products distributed across LO, HI, LO1, HI1 and rd
    s32 p[8];
    for (int i = 0; i < 8; ++i) p[i] = static_cast<s32>(a.sh(i)) * static_cast<s32>(b.sh(i));
    ctx.lo  = static_cast<u64>(static_cast<u32>(p[0])) | (static_cast<u64>(static_cast<u32>(p[1])) << 32);
    ctx.hi  = static_cast<u64>(static_cast<u32>(p[2])) | (static_cast<u64>(static_cast<u32>(p[3])) << 32);
    ctx.lo1 = static_cast<u64>(static_cast<u32>(p[4])) | (static_cast<u64>(static_cast<u32>(p[5])) << 32);
    ctx.hi1 = static_cast<u64>(static_cast<u32>(p[6])) | (static_cast<u64>(static_cast<u32>(p[7])) << 32);
    if (rd != 0) {
        u128 r{};
        r.set_uw(0, static_cast<u32>(p[0]));
        r.set_uw(1, static_cast<u32>(p[2]));
        r.set_uw(2, static_cast<u32>(p[4]));
        r.set_uw(3, static_cast<u32>(p[6]));
        ctx.set_gpr_u128(rd, r);
    }
}

inline void mmi_pmaddh(EEContext& ctx, u8 rd, u128 a, u128 b) {
    // R5900 PMADDH: multiplies 8 pairs of signed 16-bit lanes and accumulates into LO/HI/LO1/HI1 32-bit lanes
    s32 acc[8] = {
        static_cast<s32>(ctx.lo & 0xFFFFFFFFULL),  static_cast<s32>(ctx.lo >> 32),
        static_cast<s32>(ctx.hi & 0xFFFFFFFFULL),  static_cast<s32>(ctx.hi >> 32),
        static_cast<s32>(ctx.lo1 & 0xFFFFFFFFULL), static_cast<s32>(ctx.lo1 >> 32),
        static_cast<s32>(ctx.hi1 & 0xFFFFFFFFULL), static_cast<s32>(ctx.hi1 >> 32)
    };
    for (int i = 0; i < 8; ++i) acc[i] += static_cast<s32>(a.sh(i)) * static_cast<s32>(b.sh(i));
    ctx.lo  = static_cast<u64>(static_cast<u32>(acc[0])) | (static_cast<u64>(static_cast<u32>(acc[1])) << 32);
    ctx.hi  = static_cast<u64>(static_cast<u32>(acc[2])) | (static_cast<u64>(static_cast<u32>(acc[3])) << 32);
    ctx.lo1 = static_cast<u64>(static_cast<u32>(acc[4])) | (static_cast<u64>(static_cast<u32>(acc[5])) << 32);
    ctx.hi1 = static_cast<u64>(static_cast<u32>(acc[6])) | (static_cast<u64>(static_cast<u32>(acc[7])) << 32);
    if (rd != 0) {
        u128 r{};
        r.set_uw(0, static_cast<u32>(acc[0]));
        r.set_uw(1, static_cast<u32>(acc[2]));
        r.set_uw(2, static_cast<u32>(acc[4]));
        r.set_uw(3, static_cast<u32>(acc[6]));
        ctx.set_gpr_u128(rd, r);
    }
}

} // namespace tf::ps2::helpers
