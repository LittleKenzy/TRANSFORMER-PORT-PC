#include "VU1.h"
#include "../PerfCounters.h"
#include <algorithm>
#include <cmath>
#include <cstring>

#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#include <emmintrin.h>
#define TF_VU1_SSE2 1
#else
#define TF_VU1_SSE2 0
#endif

namespace tf::ps2 {

namespace {

enum Vu1UpperOp : u8 {
    U_NOP = 0,
    U_ADDbc, U_SUBbc, U_MADDbc, U_MSUBbc,
    U_MAXbc, U_MINIbc, U_MULbc,
    U_MULq, U_MAXi, U_MULi, U_MINIi,
    U_ADDq, U_MADDq, U_ADDi, U_MADDi,
    U_SUBq, U_MSUBq, U_SUBi, U_MSUBi,
    U_ADD, U_MADD, U_MUL, U_MAX,
    U_SUB, U_MSUB, U_OPMSUB, U_MINI,
    U_ADDAbc, U_SUBAbc, U_MADDAbc, U_MSUBAbc,
    U_ITOF0, U_ITOF4, U_ITOF12, U_ITOF15,
    U_FTOI0, U_FTOI4, U_FTOI12, U_FTOI15,
    U_MULAbc, U_MULAq, U_ABS, U_MULAi, U_CLIPw,
    U_ADDAq, U_MADDAq, U_ADDAi, U_MADDAi,
    U_SUBAq, U_MSUBAq, U_SUBAi, U_MSUBAi,
    U_ADDA, U_MADDA, U_MULA, U_SUBA, U_MSUBA, U_OPMULA,
};

enum Vu1LowerOp : u8 {
    L_NOP = 0,
    L_LOI,
    L_IADD, L_ISUB, L_IADDI, L_IAND, L_IOR,
    L_MOVE, L_MR32, L_LQI, L_SQI, L_LQD, L_SQD,
    L_DIV, L_SQRT, L_RSQRT, L_WAITQ,
    L_MTIR, L_MFIR, L_ILWR, L_ISWR,
    L_RNEXT, L_RGET, L_RINIT, L_RXOR,
    L_MFP, L_XTOP, L_XITOP, L_XGKICK,
    L_ESADD, L_ERSADD, L_ELENG, L_ERLENG,
    L_EATANxy, L_EATANxz, L_ESUM,
    L_ERCPR, L_ESQRT, L_ERSQRT, L_ESIN, L_EATAN, L_EEXP, L_WAITP,
    L_LQ, L_SQ, L_ILW, L_ISW, L_IADDIU, L_ISUBIU,
    L_FCEQ, L_FCSET, L_FCAND, L_FCOR,
    L_FSEQ, L_FSSET, L_FSAND, L_FSOR,
    L_FMEQ, L_FMAND, L_FMOR, L_FCGET,
    L_B, L_BAL, L_JR, L_JALR,
    L_IBEQ, L_IBNE, L_IBLTZ, L_IBGTZ, L_IBLEZ, L_IBGEZ,
};

inline s32 f32_to_s32_clamp(f32 v) {
    if (std::isnan(v)) return 0;
    if (v >= 2147483647.0f) return 2147483647;
    if (v <= -2147483648.0f) return static_cast<s32>(0x80000000u);
    return static_cast<s32>(v);
}

inline f32 bits_to_f32(u32 b) {
    f32 f;
    std::memcpy(&f, &b, 4);
    return f;
}

inline u32 f32_to_bits(f32 f) {
    u32 b;
    std::memcpy(&b, &f, 4);
    return b;
}

inline u8 field_from_dest(u8 d) {
    if (d & 8u) return 0;
    if (d & 4u) return 1;
    if (d & 2u) return 2;
    return 3;
}

#if TF_VU1_SSE2
alignas(16) constexpr u32 kDestMask4[16][4] = {
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

constexpr u8 kRev4[16] = {
    0x0u, 0x8u, 0x4u, 0xCu, 0x2u, 0xAu, 0x6u, 0xEu,
    0x1u, 0x9u, 0x5u, 0xDu, 0x3u, 0xBu, 0x7u, 0xFu
};

inline u32 compute_mac_sse2(__m128 v, u8 mask) {
    const __m128i vi = _mm_castps_si128(v);
    const __m128i abs_i = _mm_and_si128(vi, _mm_set1_epi32(0x7FFFFFFF));
    const __m128i eq_zero = _mm_cmpeq_epi32(abs_i, _mm_setzero_si128());
    const __m128i exp_bits = _mm_and_si128(vi, _mm_set1_epi32(0x7F800000));
    const __m128i eq_ovf = _mm_cmpeq_epi32(exp_bits, _mm_set1_epi32(0x7F800000));
    const u32 z = static_cast<u32>(kRev4[_mm_movemask_ps(_mm_castsi128_ps(eq_zero))]) & mask;
    const u32 s = static_cast<u32>(kRev4[_mm_movemask_ps(v)]) & mask;
    const u32 o = static_cast<u32>(kRev4[_mm_movemask_ps(_mm_castsi128_ps(eq_ovf))]) & mask;
    return z | (s << 4) | (o << 12);
}

inline void write_vec4_masked(Vec4f& dst, const Vec4f& val, u8 dest) {
    if (dest == 0x0Fu) {
        dst = val;
        return;
    }
    const __m128 m = _mm_load_ps(reinterpret_cast<const float*>(kDestMask4[dest & 0x0Fu]));
    const __m128 cur = _mm_loadu_ps(&dst.x);
    const __m128 nv = _mm_loadu_ps(&val.x);
    _mm_storeu_ps(&dst.x, _mm_or_ps(_mm_and_ps(m, nv), _mm_andnot_ps(m, cur)));
}
#else
inline u32 compute_mac_scalar(const Vec4f& val, u8 mask) {
    u32 m = 0;
    for (int k = 0; k < 4; ++k) {
        if (!(mask & (1u << (3 - k)))) continue;
        const u32 bits = f32_to_bits(val[k]);
        if ((bits & 0x7FFFFFFFu) == 0u) m |= (1u << (3 - k));
        if ((bits & 0x80000000u) != 0u) m |= (1u << (7 - k));
        if ((bits & 0x7F800000u) == 0x7F800000u) m |= (1u << (15 - k));
    }
    return m;
}

inline void write_vec4_masked(Vec4f& dst, const Vec4f& val, u8 dest) {
    if (dest == 0x0Fu) {
        dst = val;
        return;
    }
    if (dest & 8u) dst.x = val.x;
    if (dest & 4u) dst.y = val.y;
    if (dest & 2u) dst.z = val.z;
    if (dest & 1u) dst.w = val.w;
}
#endif

} // namespace

VU1Unit::VU1Unit() {
    const u8* base = reinterpret_cast<const u8*>(this);
    Vu1JitOffsets off{};
    off.vf        = static_cast<u32>(reinterpret_cast<const u8*>(&vf[0]) - base);
    off.vi        = static_cast<u32>(reinterpret_cast<const u8*>(&vi[0]) - base);
    off.acc       = static_cast<u32>(reinterpret_cast<const u8*>(&acc) - base);
    off.q         = static_cast<u32>(reinterpret_cast<const u8*>(&q) - base);
    off.p         = static_cast<u32>(reinterpret_cast<const u8*>(&p) - base);
    off.i_reg     = static_cast<u32>(reinterpret_cast<const u8*>(&i_reg) - base);
    off.pc        = static_cast<u32>(reinterpret_cast<const u8*>(&pc) - base);
    off.xtop_val  = static_cast<u32>(reinterpret_cast<const u8*>(&xtop_val) - base);
    off.xitop_val = static_cast<u32>(reinterpret_cast<const u8*>(&xitop_val) - base);
    off.data_mem  = static_cast<u32>(reinterpret_cast<const u8*>(&m_data_mem) - base);
    off.pipe_busy = static_cast<u32>(reinterpret_cast<const u8*>(&m_pipe_busy) - base);

    Vu1JitHelpers hlp{};
    hlp.advance_pipes       = &VU1Unit::jit_advance_pipes;
    hlp.exec_upper_to_tmp   = &VU1Unit::jit_exec_upper_to_tmp;
    hlp.exec_lower_fallback = &VU1Unit::jit_exec_lower_fallback;
    hlp.commit_mac          = &VU1Unit::jit_commit_mac;
    hlp.commit_clip         = &VU1Unit::jit_commit_clip;
    hlp.exec_xgkick         = &VU1Unit::jit_exec_xgkick;
    m_jit.configure(off, hlp);

    reset();
}

void VU1Unit::invalidate_code_cache() {
    for (auto& s : m_decode_cache) {
        s.raw = 0xFFFFFFFFFFFFFFFFULL;
    }
    m_code_dirty = true;
    m_analysis_initialized = false;
    m_jit_bindings_dirty = false;
    m_has_mac_readers = true;
    m_has_clip_readers = true;
    m_mac_reader_count = 0;
    m_clip_reader_count = 0;
    m_slot_reader_flags.fill(0);
    m_jit.invalidate_all();
}

void VU1Unit::mark_code_dirty() {
    m_code_dirty = true;
    m_jit_bindings_dirty = true;
}

void VU1Unit::on_code_written(u32 start_byte, u32 byte_len) {
    if (!m_code_mem || byte_len == 0) return;
    if (!m_analysis_initialized || m_code_dirty) {
        m_code_dirty = true;
        m_jit_bindings_dirty = true;
        return;
    }
    const u32 slot_start = (start_byte >> 3) & 0x7FFu;
    const u32 slot_count = std::min<u32>(2048u, (byte_len + 7u) >> 3);
    for (u32 n = 0; n < slot_count; ++n) {
        const u32 idx = (slot_start + n) & 0x7FFu;
        const u64 raw = read_le64(m_code_mem + idx * 8u);
        DecodedVu1Slot& slot = m_decode_cache[idx];
        if (slot.raw == raw) continue;

        const u8 old_f = m_slot_reader_flags[idx];
        if (old_f & 1u) m_mac_reader_count--;
        if (old_f & 2u) m_clip_reader_count--;

        decode_slot(slot, raw);
        u8 new_f = 0;
        switch (slot.l_op) {
            case L_FSEQ: case L_FSSET: case L_FSAND: case L_FSOR:
            case L_FMEQ: case L_FMAND: case L_FMOR:
                new_f |= 1u;
                m_mac_reader_count++;
                break;
            case L_FCEQ: case L_FCSET: case L_FCAND: case L_FCOR: case L_FCGET:
                new_f |= 2u;
                m_clip_reader_count++;
                break;
            default:
                break;
        }
        m_slot_reader_flags[idx] = new_f;
    }
    m_has_mac_readers = (m_mac_reader_count > 0);
    m_has_clip_readers = (m_clip_reader_count > 0);
    m_jit_bindings_dirty = true;
}

void VU1Unit::refresh_code_analysis() {
    m_jit.clear_active_bindings();
    m_jit_bindings_dirty = false;
    m_mac_reader_count = 0;
    m_clip_reader_count = 0;
    m_slot_reader_flags.fill(0);
    if (!m_code_mem) {
        m_has_mac_readers = false;
        m_has_clip_readers = false;
        m_code_dirty = false;
        m_analysis_initialized = false;
        return;
    }
    for (u32 i = 0; i < 2048u; ++i) {
        const u64 raw = read_le64(m_code_mem + i * 8u);
        DecodedVu1Slot& slot = m_decode_cache[i];
        if (slot.raw != raw) {
            decode_slot(slot, raw);
        }
        u8 f = 0;
        switch (slot.l_op) {
            case L_FSEQ: case L_FSSET: case L_FSAND: case L_FSOR:
            case L_FMEQ: case L_FMAND: case L_FMOR:
                f |= 1u;
                m_mac_reader_count++;
                break;
            case L_FCEQ: case L_FCSET: case L_FCAND: case L_FCOR: case L_FCGET:
                f |= 2u;
                m_clip_reader_count++;
                break;
            default:
                break;
        }
        m_slot_reader_flags[i] = f;
    }
    m_has_mac_readers = (m_mac_reader_count > 0);
    m_has_clip_readers = (m_clip_reader_count > 0);
    m_code_dirty = false;
    m_analysis_initialized = true;
}

void VU1Unit::reset() {
    vf.fill(Vec4f{});
    vf[0] = Vec4f{0.0f, 0.0f, 0.0f, 1.0f};
    vi.fill(0);
    acc = Vec4f{};
    q = 1.0f;
    p = 0.0f;
    i_reg = 0.0f;
    r_reg = 0x3F800000u;
    pc = 0;
    mac_flags = 0;
    status_flags = 0;
    clip_flags = 0;
    xtop_val = 0;
    xitop_val = 0;
    micro_steps_executed = 0;
    xgkick_count = 0;
    m_q_pending = 1.0f;
    m_q_delay = 0;
    m_q_status_pending = 0;
    m_p_pending = 0.0f;
    m_p_delay = 0;
    m_xgkick_addr = 0;
    m_xgkick_delay = 0;
    m_mac_pipe.fill(0);
    m_clip_pipe.fill(0);
    m_mac_valid_mask = 0;
    m_clip_valid_mask = 0;
    m_pipe_busy = 0;
    m_vi_alu_reg = 0;
    m_vi_alu_old = 0;
    invalidate_code_cache();
    debug_log.clear();
}

void VU1Unit::decode_slot(DecodedVu1Slot& slot, u64 raw) {
    slot.raw = raw;
    const u32 lower = static_cast<u32>(raw & 0xFFFFFFFFULL);
    const u32 upper = static_cast<u32>(raw >> 32);

    const bool i_bit = (upper & 0x80000000u) != 0;
    const bool e_bit = (upper & 0x40000000u) != 0;
    slot.flags = (i_bit ? 1u : 0u) | (e_bit ? 2u : 0u);

    // Decode upper instruction
    slot.u_dest = static_cast<u8>((upper >> 21) & 0x0Fu);
    slot.u_ft   = static_cast<u8>((upper >> 16) & 0x1Fu);
    slot.u_fs   = static_cast<u8>((upper >> 11) & 0x1Fu);
    slot.u_fd   = static_cast<u8>((upper >> 6)  & 0x1Fu);
    slot.u_bc   = static_cast<u8>(upper & 3u);

    if ((upper & 0x07FFFFFFu) == 0x000002FFu) {
        slot.u_op = U_NOP;
    } else {
        const u8 fn = static_cast<u8>(upper & 0x3Fu);
        if (fn >= 0x3C) {
            const u32 sub = (((upper >> 6) & 0x1Fu) << 2) | (upper & 3u);
            switch (sub) {
                case 0x00: case 0x01: case 0x02: case 0x03: slot.u_op = U_ADDAbc; break;
                case 0x04: case 0x05: case 0x06: case 0x07: slot.u_op = U_SUBAbc; break;
                case 0x08: case 0x09: case 0x0A: case 0x0B: slot.u_op = U_MADDAbc; break;
                case 0x0C: case 0x0D: case 0x0E: case 0x0F: slot.u_op = U_MSUBAbc; break;
                case 0x10: slot.u_op = U_ITOF0; break;
                case 0x11: slot.u_op = U_ITOF4; break;
                case 0x12: slot.u_op = U_ITOF12; break;
                case 0x13: slot.u_op = U_ITOF15; break;
                case 0x14: slot.u_op = U_FTOI0; break;
                case 0x15: slot.u_op = U_FTOI4; break;
                case 0x16: slot.u_op = U_FTOI12; break;
                case 0x17: slot.u_op = U_FTOI15; break;
                case 0x18: case 0x19: case 0x1A: case 0x1B: slot.u_op = U_MULAbc; break;
                case 0x1C: slot.u_op = U_MULAq; break;
                case 0x1D: slot.u_op = U_ABS; break;
                case 0x1E: slot.u_op = U_MULAi; break;
                case 0x1F: slot.u_op = U_CLIPw; break;
                case 0x20: slot.u_op = U_ADDAq; break;
                case 0x21: slot.u_op = U_MADDAq; break;
                case 0x22: slot.u_op = U_ADDAi; break;
                case 0x23: slot.u_op = U_MADDAi; break;
                case 0x24: slot.u_op = U_SUBAq; break;
                case 0x25: slot.u_op = U_MSUBAq; break;
                case 0x26: slot.u_op = U_SUBAi; break;
                case 0x27: slot.u_op = U_MSUBAi; break;
                case 0x28: slot.u_op = U_ADDA; break;
                case 0x29: slot.u_op = U_MADDA; break;
                case 0x2A: slot.u_op = U_MULA; break;
                case 0x2C: slot.u_op = U_SUBA; break;
                case 0x2D: slot.u_op = U_MSUBA; break;
                case 0x2E: slot.u_op = U_OPMULA; break;
                default:   slot.u_op = U_NOP; break;
            }
        } else {
            switch (fn) {
                case 0x00: case 0x01: case 0x02: case 0x03: slot.u_op = U_ADDbc; break;
                case 0x04: case 0x05: case 0x06: case 0x07: slot.u_op = U_SUBbc; break;
                case 0x08: case 0x09: case 0x0A: case 0x0B: slot.u_op = U_MADDbc; break;
                case 0x0C: case 0x0D: case 0x0E: case 0x0F: slot.u_op = U_MSUBbc; break;
                case 0x10: case 0x11: case 0x12: case 0x13: slot.u_op = U_MAXbc; break;
                case 0x14: case 0x15: case 0x16: case 0x17: slot.u_op = U_MINIbc; break;
                case 0x18: case 0x19: case 0x1A: case 0x1B: slot.u_op = U_MULbc; break;
                case 0x1C: slot.u_op = U_MULq; break;
                case 0x1D: slot.u_op = U_MAXi; break;
                case 0x1E: slot.u_op = U_MULi; break;
                case 0x1F: slot.u_op = U_MINIi; break;
                case 0x20: slot.u_op = U_ADDq; break;
                case 0x21: slot.u_op = U_MADDq; break;
                case 0x22: slot.u_op = U_ADDi; break;
                case 0x23: slot.u_op = U_MADDi; break;
                case 0x24: slot.u_op = U_SUBq; break;
                case 0x25: slot.u_op = U_MSUBq; break;
                case 0x26: slot.u_op = U_SUBi; break;
                case 0x27: slot.u_op = U_MSUBi; break;
                case 0x28: slot.u_op = U_ADD; break;
                case 0x29: slot.u_op = U_MADD; break;
                case 0x2A: slot.u_op = U_MUL; break;
                case 0x2B: slot.u_op = U_MAX; break;
                case 0x2C: slot.u_op = U_SUB; break;
                case 0x2D: slot.u_op = U_MSUB; break;
                case 0x2E: slot.u_op = U_OPMSUB; break;
                case 0x2F: slot.u_op = U_MINI; break;
                default:   slot.u_op = U_NOP; break;
            }
        }
    }

    // Decode lower instruction
    slot.l_dest  = static_cast<u8>((lower >> 21) & 0x0Fu);
    slot.l_it    = static_cast<u8>((lower >> 16) & 0x1Fu);
    slot.l_is    = static_cast<u8>((lower >> 11) & 0x1Fu);
    slot.l_id    = static_cast<u8>((lower >> 6)  & 0x1Fu);
    slot.l_imm11 = static_cast<s16>((lower & 0x7FFu) | ((lower & 0x400u) ? 0xF800u : 0u));
    slot.l_imm15 = static_cast<u16>((lower & 0x7FFu) | (((lower >> 21) & 0x0Fu) << 11));
    slot.l_imm24 = i_bit ? lower : (lower & 0xFFFFFFu);
    const u8 fsf = static_cast<u8>((lower >> 21) & 3u);
    const u8 ftf = static_cast<u8>((lower >> 23) & 3u);
    slot.l_aux   = static_cast<u8>(fsf | (ftf << 2) | (field_from_dest(slot.l_dest) << 4));

    if (i_bit) {
        slot.l_op = L_LOI;
        return;
    }
    if (lower == 0x8000033Cu) {
        slot.l_op = L_NOP;
        return;
    }

    const u8 op = static_cast<u8>((lower >> 25) & 0x7Fu);
    if (op == 0x40) {
        const u8 fn = static_cast<u8>(lower & 0x3Fu);
        if (fn < 0x3C) {
            switch (fn) {
                case 0x30: slot.l_op = L_IADD; break;
                case 0x31: slot.l_op = L_ISUB; break;
                case 0x32:
                    slot.l_op = L_IADDI;
                    slot.l_imm11 = static_cast<s16>(slot.l_id | ((slot.l_id & 0x10u) ? 0xFFE0u : 0u));
                    break;
                case 0x34: slot.l_op = L_IAND; break;
                case 0x35: slot.l_op = L_IOR; break;
                default:   slot.l_op = L_NOP; break;
            }
            return;
        }
        const u32 sub = (((lower >> 6) & 0x1Fu) << 2) | (lower & 3u);
        switch (sub) {
            case 0x30: slot.l_op = L_MOVE; break;
            case 0x31: slot.l_op = L_MR32; break;
            case 0x34: slot.l_op = L_LQI; break;
            case 0x35: slot.l_op = L_SQI; break;
            case 0x36: slot.l_op = L_LQD; break;
            case 0x37: slot.l_op = L_SQD; break;
            case 0x38: slot.l_op = L_DIV; break;
            case 0x39: slot.l_op = L_SQRT; break;
            case 0x3A: slot.l_op = L_RSQRT; break;
            case 0x3B: slot.l_op = L_WAITQ; break;
            case 0x3C: slot.l_op = L_MTIR; break;
            case 0x3D: slot.l_op = L_MFIR; break;
            case 0x3E: slot.l_op = L_ILWR; break;
            case 0x3F: slot.l_op = L_ISWR; break;
            case 0x40: slot.l_op = L_RNEXT; break;
            case 0x41: slot.l_op = L_RGET; break;
            case 0x42: slot.l_op = L_RINIT; break;
            case 0x43: slot.l_op = L_RXOR; break;
            case 0x64: slot.l_op = L_MFP; break;
            case 0x68: slot.l_op = L_XTOP; break;
            case 0x69: slot.l_op = L_XITOP; break;
            case 0x6C: slot.l_op = L_XGKICK; break;
            case 0x70: slot.l_op = L_ESADD; break;
            case 0x71: slot.l_op = L_ERSADD; break;
            case 0x72: slot.l_op = L_ELENG; break;
            case 0x73: slot.l_op = L_ERLENG; break;
            case 0x74: slot.l_op = L_EATANxy; break;
            case 0x75: slot.l_op = L_EATANxz; break;
            case 0x76: slot.l_op = L_ESUM; break;
            case 0x78: slot.l_op = L_ERCPR; break;
            case 0x79: slot.l_op = L_ESQRT; break;
            case 0x7A: slot.l_op = L_ERSQRT; break;
            case 0x7B: slot.l_op = L_WAITP; break;
            case 0x7C: slot.l_op = L_ESIN; break;
            case 0x7D: slot.l_op = L_EATAN; break;
            case 0x7E: slot.l_op = L_EEXP; break;
            default:   slot.l_op = L_NOP; break;
        }
        return;
    }

    switch (op) {
        case 0x00: slot.l_op = L_LQ; break;
        case 0x01: slot.l_op = L_SQ; break;
        case 0x04: slot.l_op = L_ILW; break;
        case 0x05: slot.l_op = L_ISW; break;
        case 0x08: slot.l_op = L_IADDIU; break;
        case 0x09: slot.l_op = L_ISUBIU; break;
        case 0x10: slot.l_op = L_FCEQ; break;
        case 0x11: slot.l_op = L_FCSET; break;
        case 0x12: slot.l_op = L_FCAND; break;
        case 0x13: slot.l_op = L_FCOR; break;
        case 0x14: slot.l_op = L_FSEQ; break;
        case 0x15: slot.l_op = L_FSSET; break;
        case 0x16: slot.l_op = L_FSAND; break;
        case 0x17: slot.l_op = L_FSOR; break;
        case 0x18: slot.l_op = L_FMEQ; break;
        case 0x1A: slot.l_op = L_FMAND; break;
        case 0x1B: slot.l_op = L_FMOR; break;
        case 0x1C: slot.l_op = L_FCGET; break;
        case 0x20: slot.l_op = L_B; break;
        case 0x21: slot.l_op = L_BAL; break;
        case 0x24: slot.l_op = L_JR; break;
        case 0x25: slot.l_op = L_JALR; break;
        case 0x28: slot.l_op = L_IBEQ; break;
        case 0x29: slot.l_op = L_IBNE; break;
        case 0x2C: slot.l_op = L_IBLTZ; break;
        case 0x2D: slot.l_op = L_IBGTZ; break;
        case 0x2E: slot.l_op = L_IBLEZ; break;
        case 0x2F: slot.l_op = L_IBGEZ; break;
        default:   slot.l_op = L_NOP; break;
    }
}

void VU1Unit::flush_q() {
    if (m_q_delay > 0) {
        q = m_q_pending;
        status_flags = (status_flags & ~0x30u) | m_q_status_pending;
        m_q_delay = 0;
        update_pipe_busy();
    }
}

void VU1Unit::flush_p() {
    if (m_p_delay > 0) {
        p = m_p_pending;
        m_p_delay = 0;
        update_pipe_busy();
    }
}

void VU1Unit::flush_xgkick() {
    if (m_xgkick_delay > 0) {
        const u32 addr = m_xgkick_addr;
        m_xgkick_delay = 0;
        update_pipe_busy();
        exec_xgkick_addr(addr);
    }
}

void VU1Unit::advance_pipes() {
    if (m_q_delay > 0 && --m_q_delay == 0) {
        q = m_q_pending;
        status_flags = (status_flags & ~0x30u) | m_q_status_pending;
    }
    if (m_p_delay > 0 && --m_p_delay == 0) {
        p = m_p_pending;
    }
    if (m_mac_valid_mask) {
        if (m_mac_valid_mask & 1u) {
            const u32 m = m_mac_pipe[0];
            mac_flags = m;
            u32 st_cur = 0;
            if (m & 0x000Fu) st_cur |= 0x01u; // Z
            if (m & 0x00F0u) st_cur |= 0x02u; // S
            if (m & 0x0F00u) st_cur |= 0x04u; // U
            if (m & 0xF000u) st_cur |= 0x08u; // O
            status_flags = (status_flags & 0xFF0u) | st_cur | (st_cur << 6);
        }
        m_mac_pipe[0] = m_mac_pipe[1];
        m_mac_pipe[1] = m_mac_pipe[2];
        m_mac_pipe[2] = m_mac_pipe[3];
        m_mac_valid_mask >>= 1;
    }
    if (m_clip_valid_mask) {
        if (m_clip_valid_mask & 1u) {
            clip_flags = ((clip_flags << 6) | static_cast<u32>(m_clip_pipe[0])) & 0xFFFFFFu;
        }
        m_clip_pipe[0] = m_clip_pipe[1];
        m_clip_pipe[1] = m_clip_pipe[2];
        m_clip_pipe[2] = m_clip_pipe[3];
        m_clip_valid_mask >>= 1;
    }
    bool fire_xgkick = false;
    u32 xg_addr = 0;
    if (m_xgkick_delay > 0 && --m_xgkick_delay == 0) {
        fire_xgkick = true;
        xg_addr = m_xgkick_addr;
    }
    update_pipe_busy();
    if (fire_xgkick) {
        exec_xgkick_addr(xg_addr);
    }
}

void VU1Unit::flush_all_pipes() {
    flush_q();
    flush_p();
    flush_xgkick();
    while (m_mac_valid_mask || m_clip_valid_mask) {
        advance_pipes();
    }
    m_pipe_busy = 0;
    m_vi_alu_reg = 0;
}

void VU1Unit::exec_xgkick(u8 is_reg) {
    // PS2 VU1 hardware: XGKICK has a 1-instruction delay (cycle N+1 executes before GIF reads VU1 mem).
    // If a previous XGKICK is still pending its delay slot, complete it first before latching the new addr.
    flush_xgkick();
    m_xgkick_addr = static_cast<u32>(vi[is_reg & 15u] & 0x3FFu) * 16u;
    m_xgkick_delay = 2u;
    m_pipe_busy = 1u;
}

void VU1Unit::exec_xgkick_addr(u32 addr) {
    if (!m_xgkick_cb || !m_data_mem) return;
    xgkick_count++;
    bool needs_wrap = false;
    if (addr > 0) {
        u32 qpos = addr >> 4;
        while (qpos < 1024u) {
            const u64 lo = read_le64(m_data_mem + (qpos << 4));
            const u32 nloop = static_cast<u32>(lo & 0x7FFFu);
            const bool eop  = ((lo >> 15) & 1u) != 0;
            const u32 flg   = static_cast<u32>((lo >> 58) & 0x3u);
            u32 nreg        = static_cast<u32>((lo >> 60) & 0xFu);
            if (nreg == 0) nreg = 16u;
            const u32 payload_qwc = (flg == 0) ? (nloop * nreg)
                                  : (flg == 1) ? ((nloop * nreg + 1u) >> 1)
                                               : nloop;
            qpos += 1u + payload_qwc;
            if (qpos > 1024u) { needs_wrap = true; break; }
            if (eop) break;
            if (qpos == 1024u) { needs_wrap = true; break; }
        }
    }
    if (needs_wrap) {
        const u32 tail = 16384u - addr;
        std::memcpy(m_xgkick_wrap_buf.data(), m_data_mem + addr, tail);
        std::memcpy(m_xgkick_wrap_buf.data() + tail, m_data_mem, addr);
        PerfScope perf_xg(PerfCat::GifParse);
        m_xgkick_cb(m_xgkick_wrap_buf.data(), 16384u);
    } else {
        PerfScope perf_xg(PerfCat::GifParse);
        m_xgkick_cb(m_data_mem + addr, 16384u - addr);
    }
}

void VU1Unit::jit_advance_pipes(VU1Unit* vu) {
    vu->advance_pipes();
}

void VU1Unit::jit_exec_upper_to_tmp(VU1Unit* vu, u32 upper_word, Vec4f* out_val, u8* out_dest) {
    const u8 dest = static_cast<u8>((upper_word >> 21) & 0x0Fu);
    const u8 ft   = static_cast<u8>((upper_word >> 16) & 0x1Fu);
    const u8 fs   = static_cast<u8>((upper_word >> 11) & 0x1Fu);
    const u8 fn   = static_cast<u8>(upper_word & 0x3Fu);
    const Vec4f& S = vu->vf[fs];
    const Vec4f& T = vu->vf[ft];
    *out_dest = dest;
    if (fn >= 0x3C) {
        const u32 sub = (((upper_word >> 6) & 0x1Fu) << 2) | (upper_word & 3u);
        switch (sub) {
            case 0x14: // FTOI0
                *out_val = Vec4f{bits_to_f32(static_cast<u32>(f32_to_s32_clamp(S.x))),
                                 bits_to_f32(static_cast<u32>(f32_to_s32_clamp(S.y))),
                                 bits_to_f32(static_cast<u32>(f32_to_s32_clamp(S.z))),
                                 bits_to_f32(static_cast<u32>(f32_to_s32_clamp(S.w)))};
                return;
            case 0x15: // FTOI4
                *out_val = Vec4f{bits_to_f32(static_cast<u32>(f32_to_s32_clamp(S.x * 16.0f))),
                                 bits_to_f32(static_cast<u32>(f32_to_s32_clamp(S.y * 16.0f))),
                                 bits_to_f32(static_cast<u32>(f32_to_s32_clamp(S.z * 16.0f))),
                                 bits_to_f32(static_cast<u32>(f32_to_s32_clamp(S.w * 16.0f)))};
                return;
            case 0x16: // FTOI12
                *out_val = Vec4f{bits_to_f32(static_cast<u32>(f32_to_s32_clamp(S.x * 4096.0f))),
                                 bits_to_f32(static_cast<u32>(f32_to_s32_clamp(S.y * 4096.0f))),
                                 bits_to_f32(static_cast<u32>(f32_to_s32_clamp(S.z * 4096.0f))),
                                 bits_to_f32(static_cast<u32>(f32_to_s32_clamp(S.w * 4096.0f)))};
                return;
            case 0x17: // FTOI15
                *out_val = Vec4f{bits_to_f32(static_cast<u32>(f32_to_s32_clamp(S.x * 32768.0f))),
                                 bits_to_f32(static_cast<u32>(f32_to_s32_clamp(S.y * 32768.0f))),
                                 bits_to_f32(static_cast<u32>(f32_to_s32_clamp(S.z * 32768.0f))),
                                 bits_to_f32(static_cast<u32>(f32_to_s32_clamp(S.w * 32768.0f)))};
                return;
            case 0x2E: // OPMULA
                *out_val = Vec4f{S.y * T.z, S.z * T.x, S.x * T.y, vu->acc.w};
                *out_dest = 0xEu;
                return;
            default:
                *out_val = Vec4f{};
                return;
        }
    }
    if (fn == 0x2E) { // OPMSUB
        *out_val = Vec4f{vu->acc.x - S.y * T.z, vu->acc.y - S.z * T.x, vu->acc.z - S.x * T.y, 0.0f};
        return;
    }
    *out_val = Vec4f{};
}

void VU1Unit::jit_exec_lower_fallback(VU1Unit* vu, u32 lower_word, u32 cur_pc) {
    vu->pc = cur_pc;
    bool dummy_branched = false;
    u32 dummy_target = 0;
    const DecodedVu1Slot& cached = vu->m_decode_cache[(cur_pc >> 3) & 0x7FFu];
    if (static_cast<u32>(cached.raw) == lower_word && (cached.flags & 1u) == 0) {
        vu->exec_lower_decoded(cached, dummy_branched, dummy_target, 0, 0);
        return;
    }
    DecodedVu1Slot slot{};
    decode_slot(slot, static_cast<u64>(lower_word) | (0x000002FFULL << 32));
    vu->exec_lower_decoded(slot, dummy_branched, dummy_target, 0, 0);
}

void VU1Unit::jit_commit_mac(VU1Unit* vu, const Vec4f* val, u32 dest_mask) {
#if TF_VU1_SSE2
    const u32 m = compute_mac_sse2(_mm_loadu_ps(&val->x), static_cast<u8>(dest_mask & 0x0Fu));
#else
    const u32 m = compute_mac_scalar(*val, static_cast<u8>(dest_mask & 0x0Fu));
#endif
    vu->m_mac_pipe[3] = m;
    vu->m_mac_valid_mask |= 8u;
    vu->m_pipe_busy = 1u;
}

void VU1Unit::jit_commit_clip(VU1Unit* vu, u32 fs, u32 ft) {
    const Vec4f& S = vu->vf[fs & 31u];
    const f32 w = std::fabs(vu->vf[ft & 31u].w);
    u8 c = 0;
    if (S.x >  w) c |= 0x01u;
    if (S.x < -w) c |= 0x02u;
    if (S.y >  w) c |= 0x04u;
    if (S.y < -w) c |= 0x08u;
    if (S.z >  w) c |= 0x10u;
    if (S.z < -w) c |= 0x20u;
    vu->m_clip_pipe[3] = c;
    vu->m_clip_valid_mask |= 8u;
    vu->m_pipe_busy = 1u;
}

void VU1Unit::jit_exec_xgkick(VU1Unit* vu, u32 is_reg) {
    vu->exec_xgkick(static_cast<u8>(is_reg & 15u));
}

void VU1Unit::exec_upper_decoded(const DecodedVu1Slot& slot, UpperWrite& uw) {
    const u8 dest = slot.u_dest;
    const Vec4f& S = vf[slot.u_fs];
    const Vec4f& T = vf[slot.u_ft];

#if TF_VU1_SSE2
    auto store_fd_v = [&](u8 dst_reg, __m128 res) {
        if (m_has_mac_readers || !enable_jit) {
            uw.write_mac = true;
            uw.mac = compute_mac_sse2(res, dest);
        }
        if (dst_reg == 0) return;
        uw.write_vf = true;
        uw.reg = dst_reg;
        uw.dest = dest;
        _mm_storeu_ps(&uw.val.x, res);
    };
    auto store_nomac_v = [&](u8 dst_reg, __m128 res) {
        if (dst_reg == 0) return;
        uw.write_vf = true;
        uw.reg = dst_reg;
        uw.dest = dest;
        _mm_storeu_ps(&uw.val.x, res);
    };
    auto store_acc_v = [&](__m128 res) {
        if (m_has_mac_readers || !enable_jit) {
            uw.write_mac = true;
            uw.mac = compute_mac_sse2(res, dest);
        }
        uw.write_acc = true;
        uw.dest = dest;
        _mm_storeu_ps(&uw.val.x, res);
    };

    const __m128 vs = _mm_loadu_ps(&S.x);
    switch (slot.u_op) {
        case U_ADDbc:   store_fd_v(slot.u_fd, _mm_add_ps(vs, _mm_set1_ps(T[slot.u_bc]))); break;
        case U_SUBbc:   store_fd_v(slot.u_fd, _mm_sub_ps(vs, _mm_set1_ps(T[slot.u_bc]))); break;
        case U_MADDbc:  store_fd_v(slot.u_fd, _mm_add_ps(_mm_loadu_ps(&acc.x), _mm_mul_ps(vs, _mm_set1_ps(T[slot.u_bc])))); break;
        case U_MSUBbc:  store_fd_v(slot.u_fd, _mm_sub_ps(_mm_loadu_ps(&acc.x), _mm_mul_ps(vs, _mm_set1_ps(T[slot.u_bc])))); break;
        case U_MAXbc:   store_nomac_v(slot.u_fd, _mm_max_ps(vs, _mm_set1_ps(T[slot.u_bc]))); break;
        case U_MINIbc:  store_nomac_v(slot.u_fd, _mm_min_ps(vs, _mm_set1_ps(T[slot.u_bc]))); break;
        case U_MULbc:   store_fd_v(slot.u_fd, _mm_mul_ps(vs, _mm_set1_ps(T[slot.u_bc]))); break;
        case U_MULq:    store_fd_v(slot.u_fd, _mm_mul_ps(vs, _mm_set1_ps(q))); break;
        case U_MAXi:    store_nomac_v(slot.u_fd, _mm_max_ps(vs, _mm_set1_ps(i_reg))); break;
        case U_MULi:    store_fd_v(slot.u_fd, _mm_mul_ps(vs, _mm_set1_ps(i_reg))); break;
        case U_MINIi:   store_nomac_v(slot.u_fd, _mm_min_ps(vs, _mm_set1_ps(i_reg))); break;
        case U_ADDq:    store_fd_v(slot.u_fd, _mm_add_ps(vs, _mm_set1_ps(q))); break;
        case U_MADDq:   store_fd_v(slot.u_fd, _mm_add_ps(_mm_loadu_ps(&acc.x), _mm_mul_ps(vs, _mm_set1_ps(q)))); break;
        case U_ADDi:    store_fd_v(slot.u_fd, _mm_add_ps(vs, _mm_set1_ps(i_reg))); break;
        case U_MADDi:   store_fd_v(slot.u_fd, _mm_add_ps(_mm_loadu_ps(&acc.x), _mm_mul_ps(vs, _mm_set1_ps(i_reg)))); break;
        case U_SUBq:    store_fd_v(slot.u_fd, _mm_sub_ps(vs, _mm_set1_ps(q))); break;
        case U_MSUBq:   store_fd_v(slot.u_fd, _mm_sub_ps(_mm_loadu_ps(&acc.x), _mm_mul_ps(vs, _mm_set1_ps(q)))); break;
        case U_SUBi:    store_fd_v(slot.u_fd, _mm_sub_ps(vs, _mm_set1_ps(i_reg))); break;
        case U_MSUBi:   store_fd_v(slot.u_fd, _mm_sub_ps(_mm_loadu_ps(&acc.x), _mm_mul_ps(vs, _mm_set1_ps(i_reg)))); break;
        case U_ADD:     store_fd_v(slot.u_fd, _mm_add_ps(vs, _mm_loadu_ps(&T.x))); break;
        case U_MADD:    store_fd_v(slot.u_fd, _mm_add_ps(_mm_loadu_ps(&acc.x), _mm_mul_ps(vs, _mm_loadu_ps(&T.x)))); break;
        case U_MUL:     store_fd_v(slot.u_fd, _mm_mul_ps(vs, _mm_loadu_ps(&T.x))); break;
        case U_MAX:     store_nomac_v(slot.u_fd, _mm_max_ps(vs, _mm_loadu_ps(&T.x))); break;
        case U_SUB:     store_fd_v(slot.u_fd, _mm_sub_ps(vs, _mm_loadu_ps(&T.x))); break;
        case U_MSUB:    store_fd_v(slot.u_fd, _mm_sub_ps(_mm_loadu_ps(&acc.x), _mm_mul_ps(vs, _mm_loadu_ps(&T.x)))); break;
        case U_OPMSUB: {
            const Vec4f r{acc.x - S.y * T.z, acc.y - S.z * T.x, acc.z - S.x * T.y, 0.0f};
            store_fd_v(slot.u_fd, _mm_loadu_ps(&r.x));
            break;
        }
        case U_MINI:    store_nomac_v(slot.u_fd, _mm_min_ps(vs, _mm_loadu_ps(&T.x))); break;
        case U_ADDAbc:  store_acc_v(_mm_add_ps(vs, _mm_set1_ps(T[slot.u_bc]))); break;
        case U_SUBAbc:  store_acc_v(_mm_sub_ps(vs, _mm_set1_ps(T[slot.u_bc]))); break;
        case U_MADDAbc: store_acc_v(_mm_add_ps(_mm_loadu_ps(&acc.x), _mm_mul_ps(vs, _mm_set1_ps(T[slot.u_bc])))); break;
        case U_MSUBAbc: store_acc_v(_mm_sub_ps(_mm_loadu_ps(&acc.x), _mm_mul_ps(vs, _mm_set1_ps(T[slot.u_bc])))); break;
        case U_ITOF0:   store_nomac_v(slot.u_ft, _mm_cvtepi32_ps(_mm_castps_si128(vs))); break;
        case U_ITOF4:   store_nomac_v(slot.u_ft, _mm_mul_ps(_mm_cvtepi32_ps(_mm_castps_si128(vs)), _mm_set1_ps(1.0f / 16.0f))); break;
        case U_ITOF12:  store_nomac_v(slot.u_ft, _mm_mul_ps(_mm_cvtepi32_ps(_mm_castps_si128(vs)), _mm_set1_ps(1.0f / 4096.0f))); break;
        case U_ITOF15:  store_nomac_v(slot.u_ft, _mm_mul_ps(_mm_cvtepi32_ps(_mm_castps_si128(vs)), _mm_set1_ps(1.0f / 32768.0f))); break;
        case U_FTOI0: {
            Vec4f r{bits_to_f32(static_cast<u32>(f32_to_s32_clamp(S.x))),
                    bits_to_f32(static_cast<u32>(f32_to_s32_clamp(S.y))),
                    bits_to_f32(static_cast<u32>(f32_to_s32_clamp(S.z))),
                    bits_to_f32(static_cast<u32>(f32_to_s32_clamp(S.w)))};
            store_nomac_v(slot.u_ft, _mm_loadu_ps(&r.x));
            break;
        }
        case U_FTOI4: {
            Vec4f r{bits_to_f32(static_cast<u32>(f32_to_s32_clamp(S.x * 16.0f))),
                    bits_to_f32(static_cast<u32>(f32_to_s32_clamp(S.y * 16.0f))),
                    bits_to_f32(static_cast<u32>(f32_to_s32_clamp(S.z * 16.0f))),
                    bits_to_f32(static_cast<u32>(f32_to_s32_clamp(S.w * 16.0f)))};
            store_nomac_v(slot.u_ft, _mm_loadu_ps(&r.x));
            break;
        }
        case U_FTOI12: {
            Vec4f r{bits_to_f32(static_cast<u32>(f32_to_s32_clamp(S.x * 4096.0f))),
                    bits_to_f32(static_cast<u32>(f32_to_s32_clamp(S.y * 4096.0f))),
                    bits_to_f32(static_cast<u32>(f32_to_s32_clamp(S.z * 4096.0f))),
                    bits_to_f32(static_cast<u32>(f32_to_s32_clamp(S.w * 4096.0f)))};
            store_nomac_v(slot.u_ft, _mm_loadu_ps(&r.x));
            break;
        }
        case U_FTOI15: {
            Vec4f r{bits_to_f32(static_cast<u32>(f32_to_s32_clamp(S.x * 32768.0f))),
                    bits_to_f32(static_cast<u32>(f32_to_s32_clamp(S.y * 32768.0f))),
                    bits_to_f32(static_cast<u32>(f32_to_s32_clamp(S.z * 32768.0f))),
                    bits_to_f32(static_cast<u32>(f32_to_s32_clamp(S.w * 32768.0f)))};
            store_nomac_v(slot.u_ft, _mm_loadu_ps(&r.x));
            break;
        }
        case U_MULAbc:  store_acc_v(_mm_mul_ps(vs, _mm_set1_ps(T[slot.u_bc]))); break;
        case U_MULAq:   store_acc_v(_mm_mul_ps(vs, _mm_set1_ps(q))); break;
        case U_ABS:     store_nomac_v(slot.u_ft, _mm_and_ps(vs, _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF)))); break;
        case U_MULAi:   store_acc_v(_mm_mul_ps(vs, _mm_set1_ps(i_reg))); break;
        case U_CLIPw: {
            const f32 w = std::fabs(T.w);
            u8 c = 0;
            if (S.x >  w) c |= 0x01u;
            if (S.x < -w) c |= 0x02u;
            if (S.y >  w) c |= 0x04u;
            if (S.y < -w) c |= 0x08u;
            if (S.z >  w) c |= 0x10u;
            if (S.z < -w) c |= 0x20u;
            uw.write_clip = true;
            uw.clip = c;
            break;
        }
        case U_ADDAq:   store_acc_v(_mm_add_ps(vs, _mm_set1_ps(q))); break;
        case U_MADDAq:  store_acc_v(_mm_add_ps(_mm_loadu_ps(&acc.x), _mm_mul_ps(vs, _mm_set1_ps(q)))); break;
        case U_ADDAi:   store_acc_v(_mm_add_ps(vs, _mm_set1_ps(i_reg))); break;
        case U_MADDAi:  store_acc_v(_mm_add_ps(_mm_loadu_ps(&acc.x), _mm_mul_ps(vs, _mm_set1_ps(i_reg)))); break;
        case U_SUBAq:   store_acc_v(_mm_sub_ps(vs, _mm_set1_ps(q))); break;
        case U_MSUBAq:  store_acc_v(_mm_sub_ps(_mm_loadu_ps(&acc.x), _mm_mul_ps(vs, _mm_set1_ps(q)))); break;
        case U_SUBAi:   store_acc_v(_mm_sub_ps(vs, _mm_set1_ps(i_reg))); break;
        case U_MSUBAi:  store_acc_v(_mm_sub_ps(_mm_loadu_ps(&acc.x), _mm_mul_ps(vs, _mm_set1_ps(i_reg)))); break;
        case U_ADDA:    store_acc_v(_mm_add_ps(vs, _mm_loadu_ps(&T.x))); break;
        case U_MADDA:   store_acc_v(_mm_add_ps(_mm_loadu_ps(&acc.x), _mm_mul_ps(vs, _mm_loadu_ps(&T.x)))); break;
        case U_MULA:    store_acc_v(_mm_mul_ps(vs, _mm_loadu_ps(&T.x))); break;
        case U_SUBA:    store_acc_v(_mm_sub_ps(vs, _mm_loadu_ps(&T.x))); break;
        case U_MSUBA:   store_acc_v(_mm_sub_ps(_mm_loadu_ps(&acc.x), _mm_mul_ps(vs, _mm_loadu_ps(&T.x)))); break;
        case U_OPMULA: {
            const Vec4f r{S.y * T.z, S.z * T.x, S.x * T.y, acc.w};
            if (m_has_mac_readers || !enable_jit) {
                const __m128 vr = _mm_loadu_ps(&r.x);
                uw.write_mac = true;
                uw.mac = compute_mac_sse2(vr, 0xEu);
            }
            uw.write_acc = true;
            uw.dest = 0xEu;
            uw.val = r;
            break;
        }
        default: break;
    }
#else
    Vec4f R{};
    auto store_fd = [&](u8 dst_reg, const Vec4f& val) {
        if (m_has_mac_readers || !enable_jit) {
            uw.write_mac = true;
            uw.mac = compute_mac_scalar(val, dest);
        }
        if (dst_reg == 0) return;
        uw.write_vf = true;
        uw.reg = dst_reg;
        uw.dest = dest;
        uw.val = val;
    };
    auto store_nomac = [&](u8 dst_reg, const Vec4f& val) {
        if (dst_reg == 0) return;
        uw.write_vf = true;
        uw.reg = dst_reg;
        uw.dest = dest;
        uw.val = val;
    };
    auto store_acc = [&](const Vec4f& val) {
        if (m_has_mac_readers || !enable_jit) {
            uw.write_mac = true;
            uw.mac = compute_mac_scalar(val, dest);
        }
        uw.write_acc = true;
        uw.dest = dest;
        uw.val = val;
    };

    switch (slot.u_op) {
        case U_ADDbc:   { f32 bc = T[slot.u_bc]; for (int k = 0; k < 4; ++k) R[k] = S[k] + bc; store_fd(slot.u_fd, R); break; }
        case U_SUBbc:   { f32 bc = T[slot.u_bc]; for (int k = 0; k < 4; ++k) R[k] = S[k] - bc; store_fd(slot.u_fd, R); break; }
        case U_MADDbc:  { f32 bc = T[slot.u_bc]; for (int k = 0; k < 4; ++k) R[k] = acc[k] + S[k] * bc; store_fd(slot.u_fd, R); break; }
        case U_MSUBbc:  { f32 bc = T[slot.u_bc]; for (int k = 0; k < 4; ++k) R[k] = acc[k] - S[k] * bc; store_fd(slot.u_fd, R); break; }
        case U_MAXbc:   { f32 bc = T[slot.u_bc]; for (int k = 0; k < 4; ++k) R[k] = std::max(S[k], bc); store_nomac(slot.u_fd, R); break; }
        case U_MINIbc:  { f32 bc = T[slot.u_bc]; for (int k = 0; k < 4; ++k) R[k] = std::min(S[k], bc); store_nomac(slot.u_fd, R); break; }
        case U_MULbc:   { f32 bc = T[slot.u_bc]; for (int k = 0; k < 4; ++k) R[k] = S[k] * bc; store_fd(slot.u_fd, R); break; }
        case U_MULq:    for (int k = 0; k < 4; ++k) R[k] = S[k] * q; store_fd(slot.u_fd, R); break;
        case U_MAXi:    for (int k = 0; k < 4; ++k) R[k] = std::max(S[k], i_reg); store_nomac(slot.u_fd, R); break;
        case U_MULi:    for (int k = 0; k < 4; ++k) R[k] = S[k] * i_reg; store_fd(slot.u_fd, R); break;
        case U_MINIi:   for (int k = 0; k < 4; ++k) R[k] = std::min(S[k], i_reg); store_nomac(slot.u_fd, R); break;
        case U_ADDq:    for (int k = 0; k < 4; ++k) R[k] = S[k] + q; store_fd(slot.u_fd, R); break;
        case U_MADDq:   for (int k = 0; k < 4; ++k) R[k] = acc[k] + S[k] * q; store_fd(slot.u_fd, R); break;
        case U_ADDi:    for (int k = 0; k < 4; ++k) R[k] = S[k] + i_reg; store_fd(slot.u_fd, R); break;
        case U_MADDi:   for (int k = 0; k < 4; ++k) R[k] = acc[k] + S[k] * i_reg; store_fd(slot.u_fd, R); break;
        case U_SUBq:    for (int k = 0; k < 4; ++k) R[k] = S[k] - q; store_fd(slot.u_fd, R); break;
        case U_MSUBq:   for (int k = 0; k < 4; ++k) R[k] = acc[k] - S[k] * q; store_fd(slot.u_fd, R); break;
        case U_SUBi:    for (int k = 0; k < 4; ++k) R[k] = S[k] - i_reg; store_fd(slot.u_fd, R); break;
        case U_MSUBi:   for (int k = 0; k < 4; ++k) R[k] = acc[k] - S[k] * i_reg; store_fd(slot.u_fd, R); break;
        case U_ADD:     for (int k = 0; k < 4; ++k) R[k] = S[k] + T[k]; store_fd(slot.u_fd, R); break;
        case U_MADD:    for (int k = 0; k < 4; ++k) R[k] = acc[k] + S[k] * T[k]; store_fd(slot.u_fd, R); break;
        case U_MUL:     for (int k = 0; k < 4; ++k) R[k] = S[k] * T[k]; store_fd(slot.u_fd, R); break;
        case U_MAX:     for (int k = 0; k < 4; ++k) R[k] = std::max(S[k], T[k]); store_nomac(slot.u_fd, R); break;
        case U_SUB:     for (int k = 0; k < 4; ++k) R[k] = S[k] - T[k]; store_fd(slot.u_fd, R); break;
        case U_MSUB:    for (int k = 0; k < 4; ++k) R[k] = acc[k] - S[k] * T[k]; store_fd(slot.u_fd, R); break;
        case U_OPMSUB:  R = Vec4f{acc.x - S.y * T.z, acc.y - S.z * T.x, acc.z - S.x * T.y, 0.0f}; store_fd(slot.u_fd, R); break;
        case U_MINI:    for (int k = 0; k < 4; ++k) R[k] = std::min(S[k], T[k]); store_nomac(slot.u_fd, R); break;
        case U_ADDAbc:  { f32 bc = T[slot.u_bc]; for (int k = 0; k < 4; ++k) R[k] = S[k] + bc; store_acc(R); break; }
        case U_SUBAbc:  { f32 bc = T[slot.u_bc]; for (int k = 0; k < 4; ++k) R[k] = S[k] - bc; store_acc(R); break; }
        case U_MADDAbc: { f32 bc = T[slot.u_bc]; for (int k = 0; k < 4; ++k) R[k] = acc[k] + S[k] * bc; store_acc(R); break; }
        case U_MSUBAbc: { f32 bc = T[slot.u_bc]; for (int k = 0; k < 4; ++k) R[k] = acc[k] - S[k] * bc; store_acc(R); break; }
        case U_ITOF0:   for (int k = 0; k < 4; ++k) R[k] = static_cast<f32>(static_cast<s32>(f32_to_bits(S[k]))); store_nomac(slot.u_ft, R); break;
        case U_ITOF4:   for (int k = 0; k < 4; ++k) R[k] = static_cast<f32>(static_cast<s32>(f32_to_bits(S[k]))) * (1.0f / 16.0f); store_nomac(slot.u_ft, R); break;
        case U_ITOF12:  for (int k = 0; k < 4; ++k) R[k] = static_cast<f32>(static_cast<s32>(f32_to_bits(S[k]))) * (1.0f / 4096.0f); store_nomac(slot.u_ft, R); break;
        case U_ITOF15:  for (int k = 0; k < 4; ++k) R[k] = static_cast<f32>(static_cast<s32>(f32_to_bits(S[k]))) * (1.0f / 32768.0f); store_nomac(slot.u_ft, R); break;
        case U_FTOI0:   for (int k = 0; k < 4; ++k) R[k] = bits_to_f32(static_cast<u32>(f32_to_s32_clamp(S[k]))); store_nomac(slot.u_ft, R); break;
        case U_FTOI4:   for (int k = 0; k < 4; ++k) R[k] = bits_to_f32(static_cast<u32>(f32_to_s32_clamp(S[k] * 16.0f))); store_nomac(slot.u_ft, R); break;
        case U_FTOI12:  for (int k = 0; k < 4; ++k) R[k] = bits_to_f32(static_cast<u32>(f32_to_s32_clamp(S[k] * 4096.0f))); store_nomac(slot.u_ft, R); break;
        case U_FTOI15:  for (int k = 0; k < 4; ++k) R[k] = bits_to_f32(static_cast<u32>(f32_to_s32_clamp(S[k] * 32768.0f))); store_nomac(slot.u_ft, R); break;
        case U_MULAbc:  { f32 bc = T[slot.u_bc]; for (int k = 0; k < 4; ++k) R[k] = S[k] * bc; store_acc(R); break; }
        case U_MULAq:   for (int k = 0; k < 4; ++k) R[k] = S[k] * q; store_acc(R); break;
        case U_ABS:     for (int k = 0; k < 4; ++k) R[k] = std::fabs(S[k]); store_nomac(slot.u_ft, R); break;
        case U_MULAi:   for (int k = 0; k < 4; ++k) R[k] = S[k] * i_reg; store_acc(R); break;
        case U_CLIPw: {
            f32 w = std::fabs(T.w);
            u8 c = 0;
            if (S.x >  w) c |= 0x01u;
            if (S.x < -w) c |= 0x02u;
            if (S.y >  w) c |= 0x04u;
            if (S.y < -w) c |= 0x08u;
            if (S.z >  w) c |= 0x10u;
            if (S.z < -w) c |= 0x20u;
            uw.write_clip = true;
            uw.clip = c;
            break;
        }
        case U_ADDAq:   for (int k = 0; k < 4; ++k) R[k] = S[k] + q; store_acc(R); break;
        case U_MADDAq:  for (int k = 0; k < 4; ++k) R[k] = acc[k] + S[k] * q; store_acc(R); break;
        case U_ADDAi:   for (int k = 0; k < 4; ++k) R[k] = S[k] + i_reg; store_acc(R); break;
        case U_MADDAi:  for (int k = 0; k < 4; ++k) R[k] = acc[k] + S[k] * i_reg; store_acc(R); break;
        case U_SUBAq:   for (int k = 0; k < 4; ++k) R[k] = S[k] - q; store_acc(R); break;
        case U_MSUBAq:  for (int k = 0; k < 4; ++k) R[k] = acc[k] - S[k] * q; store_acc(R); break;
        case U_SUBAi:   for (int k = 0; k < 4; ++k) R[k] = S[k] - i_reg; store_acc(R); break;
        case U_MSUBAi:  for (int k = 0; k < 4; ++k) R[k] = acc[k] - S[k] * i_reg; store_acc(R); break;
        case U_ADDA:    for (int k = 0; k < 4; ++k) R[k] = S[k] + T[k]; store_acc(R); break;
        case U_MADDA:   for (int k = 0; k < 4; ++k) R[k] = acc[k] + S[k] * T[k]; store_acc(R); break;
        case U_MULA:    for (int k = 0; k < 4; ++k) R[k] = S[k] * T[k]; store_acc(R); break;
        case U_SUBA:    for (int k = 0; k < 4; ++k) R[k] = S[k] - T[k]; store_acc(R); break;
        case U_MSUBA:   for (int k = 0; k < 4; ++k) R[k] = acc[k] - S[k] * T[k]; store_acc(R); break;
        case U_OPMULA:
            R = Vec4f{S.y * T.z, S.z * T.x, S.x * T.y, acc.w};
            if (m_has_mac_readers || !enable_jit) {
                uw.write_mac = true;
                uw.mac = compute_mac_scalar(R, 0xEu);
            }
            uw.write_acc = true;
            uw.dest = 0xEu;
            uw.val = R;
            break;
        default: break;
    }
#endif
}

void VU1Unit::exec_lower_decoded(const DecodedVu1Slot& slot, bool& branched, u32& branch_target, u8 prev_alu_reg, u16 prev_alu_old) {
    branched = false;
    const u8 dest = slot.l_dest;
    const u8 it   = slot.l_it;
    const u8 is   = slot.l_is;
    const u8 id   = slot.l_id;
    const u8 fsf  = slot.l_aux & 3u;
    const u8 ftf  = (slot.l_aux >> 2) & 3u;
    const u8 fld  = (slot.l_aux >> 4) & 3u;

    auto set_vi = [&](u8 reg, u16 val) {
        if ((reg & 15u) != 0) vi[reg & 15u] = val;
    };
    auto set_vi_alu = [&](u8 reg, u16 val) {
        const u8 r = reg & 15u;
        if (r != 0) {
            // PS2 VU1 hardware branch VI-delay hazard (with consecutive-write chaining):
            // If consecutive integer ALU / LQI / SQI / LQD / SQD ops modify the same VI[r],
            // a following conditional branch observes the value from before the chain started.
            m_vi_alu_reg = r;
            m_vi_alu_old = (r == prev_alu_reg) ? prev_alu_old : vi[r];
            vi[r] = val;
        }
    };
    auto read_branch_vi = [&](u8 reg) -> u16 {
        const u8 r = reg & 15u;
        return (r != 0 && r == prev_alu_reg) ? prev_alu_old : vi[r];
    };

    switch (slot.l_op) {
        case L_LOI:
            std::memcpy(&i_reg, &slot.l_imm24, 4);
            break;
        case L_IADD:   set_vi_alu(id, static_cast<u16>(vi[is & 15] + vi[it & 15])); break;
        case L_ISUB:   set_vi_alu(id, static_cast<u16>(vi[is & 15] - vi[it & 15])); break;
        case L_IADDI:  set_vi_alu(it, static_cast<u16>(vi[is & 15] + slot.l_imm11)); break;
        case L_IAND:   set_vi_alu(id, static_cast<u16>(vi[is & 15] & vi[it & 15])); break;
        case L_IOR:    set_vi_alu(id, static_cast<u16>(vi[is & 15] | vi[it & 15])); break;
        case L_MOVE:
            if (it != 0) write_vec4_masked(vf[it], vf[is], dest);
            break;
        case L_MR32:
            if (it != 0) {
                const Vec4f& s = vf[is];
                write_vec4_masked(vf[it], Vec4f{s.y, s.z, s.w, s.x}, dest);
            }
            break;
        case L_LQI:
            if (m_data_mem) {
                const u32 addr = static_cast<u32>(vi[is & 15] & 0x3FFu) * 16u;
                Vec4f v{}; std::memcpy(&v, m_data_mem + addr, 16);
                if (it != 0) write_vec4_masked(vf[it], v, dest);
                set_vi_alu(is, static_cast<u16>(vi[is & 15] + 1));
            }
            break;
        case L_SQI:
            if (m_data_mem) {
                const u32 addr = static_cast<u32>(vi[it & 15] & 0x3FFu) * 16u;
                if (dest == 0x0Fu) {
                    std::memcpy(m_data_mem + addr, &vf[is], 16);
                } else {
                    Vec4f cur{}; std::memcpy(&cur, m_data_mem + addr, 16);
                    write_vec4_masked(cur, vf[is], dest);
                    std::memcpy(m_data_mem + addr, &cur, 16);
                }
                set_vi_alu(it, static_cast<u16>(vi[it & 15] + 1));
            }
            break;
        case L_LQD:
            if (m_data_mem) {
                set_vi_alu(is, static_cast<u16>(vi[is & 15] - 1));
                const u32 addr = static_cast<u32>(vi[is & 15] & 0x3FFu) * 16u;
                Vec4f v{}; std::memcpy(&v, m_data_mem + addr, 16);
                if (it != 0) write_vec4_masked(vf[it], v, dest);
            }
            break;
        case L_SQD:
            if (m_data_mem) {
                set_vi_alu(it, static_cast<u16>(vi[it & 15] - 1));
                const u32 addr = static_cast<u32>(vi[it & 15] & 0x3FFu) * 16u;
                if (dest == 0x0Fu) {
                    std::memcpy(m_data_mem + addr, &vf[is], 16);
                } else {
                    Vec4f cur{}; std::memcpy(&cur, m_data_mem + addr, 16);
                    write_vec4_masked(cur, vf[is], dest);
                    std::memcpy(m_data_mem + addr, &cur, 16);
                }
            }
            break;
        case L_DIV: {
            flush_q();
            const f32 num = vf[is][fsf];
            const f32 den = vf[it][ftf];
            m_q_status_pending = 0;
            if (den == 0.0f) {
                m_q_status_pending = (num == 0.0f) ? 0x410u : 0x820u;
                m_q_pending = (num >= 0.0f ? 3.4e38f : -3.4e38f);
            } else {
                m_q_pending = num / den;
            }
            // PS2 VU1 FDIV latency is 7 cycles: advance_pipes() runs at the start of T+1..T+6,
            // so m_q_delay=6 keeps OLD Q visible through T+5 and commits NEW Q at T+6.
            m_q_delay = 6;
            m_pipe_busy = 1u;
            break;
        }
        case L_SQRT: {
            flush_q();
            const f32 v = vf[it][ftf];
            m_q_status_pending = (v < 0.0f) ? 0x410u : 0u;
            m_q_pending = std::sqrt(std::fabs(v));
            m_q_delay = 6;
            m_pipe_busy = 1u;
            break;
        }
        case L_RSQRT: {
            flush_q();
            const f32 num = vf[is][fsf];
            const f32 raw_den = vf[it][ftf];
            const f32 den = std::sqrt(std::fabs(raw_den));
            m_q_status_pending = (raw_den < 0.0f) ? 0x410u : 0u;
            if (den == 0.0f) {
                m_q_status_pending |= (num == 0.0f) ? 0x410u : 0x820u;
                m_q_pending = (num >= 0.0f ? 3.4e38f : -3.4e38f);
            } else {
                m_q_pending = num / den;
            }
            // PS2 VU1 RSQRT latency is 13 cycles -> m_q_delay=12 (commits at T+12).
            m_q_delay = 12;
            m_pipe_busy = 1u;
            break;
        }
        case L_WAITQ:
            flush_q();
            break;
        case L_MTIR:
            set_vi(it, static_cast<u16>(f32_to_bits(vf[is][fsf]) & 0xFFFFu));
            break;
        case L_MFIR:
            if (it != 0) {
                const f32 v = bits_to_f32(static_cast<u32>(static_cast<s32>(static_cast<s16>(vi[is & 15]))));
                write_vec4_masked(vf[it], Vec4f{v, v, v, v}, dest);
            }
            break;
        case L_ILWR:
            if (m_data_mem) {
                const u32 addr = static_cast<u32>(vi[is & 15] & 0x3FFu) * 16u + static_cast<u32>(fld) * 4u;
                set_vi(it, static_cast<u16>(read_le32(m_data_mem + addr) & 0xFFFFu));
            }
            break;
        case L_ISWR:
            if (m_data_mem) {
                // PS2 VU1 ISA §ISWR: ISWR.dest it, (is) -> Mem[VI[is]].dest = VI[it]
                const u32 addr = static_cast<u32>(vi[is & 15] & 0x3FFu) * 16u;
                const u32 val = vi[it & 15];
                if (dest & 8u) write_le32(m_data_mem + addr + 0, val);
                if (dest & 4u) write_le32(m_data_mem + addr + 4, val);
                if (dest & 2u) write_le32(m_data_mem + addr + 8, val);
                if (dest & 1u) write_le32(m_data_mem + addr + 12, val);
            }
            break;
        case L_RNEXT:
            r_reg = (r_reg * 1664525u + 1013904223u);
            r_reg = 0x3F800000u | (r_reg & 0x007FFFFFu);
            if (it != 0) {
                const f32 rv = bits_to_f32(r_reg);
                write_vec4_masked(vf[it], Vec4f{rv, rv, rv, rv}, dest);
            }
            break;
        case L_RGET:
            if (it != 0) {
                const f32 rv = bits_to_f32(r_reg);
                write_vec4_masked(vf[it], Vec4f{rv, rv, rv, rv}, dest);
            }
            break;
        case L_RINIT:
            r_reg = 0x3F800000u | (f32_to_bits(vf[is][fsf]) & 0x007FFFFFu);
            break;
        case L_RXOR:
            r_reg = 0x3F800000u | ((r_reg ^ f32_to_bits(vf[is][fsf])) & 0x007FFFFFu);
            break;
        case L_MFP:
            if (it != 0) write_vec4_masked(vf[it], Vec4f{p, p, p, p}, dest);
            break;
        case L_XTOP:  set_vi(it, xtop_val); break;
        case L_XITOP: set_vi(it, xitop_val); break;
        case L_XGKICK:
            exec_xgkick(is);
            break;
        case L_ESADD: {
            flush_p();
            const Vec4f& s = vf[is];
            m_p_pending = s.x * s.x + s.y * s.y + s.z * s.z;
            m_p_delay = 11;
            m_pipe_busy = 1u;
            break;
        }
        case L_ERSADD: {
            flush_p();
            const Vec4f& s = vf[is];
            const f32 v = s.x * s.x + s.y * s.y + s.z * s.z;
            m_p_pending = (v == 0.0f) ? 3.4e38f : (1.0f / v);
            m_p_delay = 18;
            m_pipe_busy = 1u;
            break;
        }
        case L_ELENG: {
            flush_p();
            const Vec4f& s = vf[is];
            m_p_pending = std::sqrt(s.x * s.x + s.y * s.y + s.z * s.z);
            m_p_delay = 18;
            m_pipe_busy = 1u;
            break;
        }
        case L_ERLENG: {
            flush_p();
            const Vec4f& s = vf[is];
            const f32 v = std::sqrt(s.x * s.x + s.y * s.y + s.z * s.z);
            m_p_pending = (v == 0.0f) ? 3.4e38f : (1.0f / v);
            m_p_delay = 24;
            m_pipe_busy = 1u;
            break;
        }
        case L_EATANxy: flush_p(); m_p_pending = std::atan2(vf[is].y, vf[is].x); m_p_delay = 54; m_pipe_busy = 1u; break;
        case L_EATANxz: flush_p(); m_p_pending = std::atan2(vf[is].z, vf[is].x); m_p_delay = 54; m_pipe_busy = 1u; break;
        case L_ESUM:    flush_p(); m_p_pending = vf[is].x + vf[is].y + vf[is].z + vf[is].w; m_p_delay = 12; m_pipe_busy = 1u; break;
        case L_ERCPR:   flush_p(); m_p_pending = (vf[is][fsf] == 0.0f) ? 3.4e38f : (1.0f / vf[is][fsf]); m_p_delay = 12; m_pipe_busy = 1u; break;
        case L_ESQRT:   flush_p(); m_p_pending = std::sqrt(std::fabs(vf[is][fsf])); m_p_delay = 12; m_pipe_busy = 1u; break;
        case L_ERSQRT: {
            flush_p();
            const f32 v = std::sqrt(std::fabs(vf[is][fsf]));
            m_p_pending = (v == 0.0f) ? 3.4e38f : (1.0f / v);
            m_p_delay = 18;
            m_pipe_busy = 1u;
            break;
        }
        case L_ESIN:    flush_p(); m_p_pending = std::sin(vf[is][fsf]); m_p_delay = 29; m_pipe_busy = 1u; break;
        case L_EATAN:   flush_p(); m_p_pending = std::atan(vf[is][fsf]); m_p_delay = 54; m_pipe_busy = 1u; break;
        case L_EEXP:    flush_p(); m_p_pending = std::exp(-vf[is][fsf]); m_p_delay = 44; m_pipe_busy = 1u; break;
        case L_WAITP:   flush_p(); break;
        case L_LQ:
            if (m_data_mem) {
                const u32 addr = static_cast<u32>((static_cast<s32>(vi[is & 15]) + slot.l_imm11) & 0x3FF) * 16u;
                Vec4f v{}; std::memcpy(&v, m_data_mem + addr, 16);
                if (it != 0) write_vec4_masked(vf[it], v, dest);
            }
            break;
        case L_SQ:
            if (m_data_mem) {
                const u32 addr = static_cast<u32>((static_cast<s32>(vi[it & 15]) + slot.l_imm11) & 0x3FF) * 16u;
                if (dest == 0x0Fu) {
                    std::memcpy(m_data_mem + addr, &vf[is], 16);
                } else {
                    Vec4f cur{}; std::memcpy(&cur, m_data_mem + addr, 16);
                    write_vec4_masked(cur, vf[is], dest);
                    std::memcpy(m_data_mem + addr, &cur, 16);
                }
            }
            break;
        case L_ILW:
            if (m_data_mem) {
                const u32 addr = static_cast<u32>((static_cast<s32>(vi[is & 15]) + slot.l_imm11) & 0x3FF) * 16u
                               + static_cast<u32>(fld) * 4u;
                set_vi(it, static_cast<u16>(read_le32(m_data_mem + addr) & 0xFFFFu));
            }
            break;
        case L_ISW:
            if (m_data_mem) {
                // PS2 VU1 ISA §ISW: ISW.dest it, imm11(is) -> Mem[VI[is] + imm11].dest = VI[it]
                const u32 addr = static_cast<u32>((static_cast<s32>(vi[is & 15]) + slot.l_imm11) & 0x3FF) * 16u;
                const u32 val = vi[it & 15];
                if (dest & 8u) write_le32(m_data_mem + addr + 0, val);
                if (dest & 4u) write_le32(m_data_mem + addr + 4, val);
                if (dest & 2u) write_le32(m_data_mem + addr + 8, val);
                if (dest & 1u) write_le32(m_data_mem + addr + 12, val);
            }
            break;
        case L_IADDIU: set_vi_alu(it, static_cast<u16>(vi[is & 15] + slot.l_imm15)); break;
        case L_ISUBIU: set_vi_alu(it, static_cast<u16>(vi[is & 15] - slot.l_imm15)); break;
        case L_FCEQ:   set_vi(1, static_cast<u16>((clip_flags & 0xFFFFFFu) == slot.l_imm24 ? 1u : 0u)); break;
        case L_FCSET:  clip_flags = slot.l_imm24 & 0xFFFFFFu; m_clip_valid_mask = 0; update_pipe_busy(); break;
        case L_FCAND:  set_vi(1, static_cast<u16>((clip_flags & slot.l_imm24) != 0 ? 1u : 0u)); break;
        case L_FCOR:   set_vi(1, static_cast<u16>(((clip_flags | slot.l_imm24) & 0xFFFFFFu) == 0xFFFFFFu ? 1u : 0u)); break;
        case L_FSEQ:   set_vi(it, static_cast<u16>((status_flags & 0xFFFu) == (slot.l_imm15 & 0xFFFu) ? 1u : 0u)); break;
        case L_FSSET:  status_flags = (status_flags & 0x03Fu) | (slot.l_imm15 & 0xFC0u); break;
        case L_FSAND:  set_vi(it, static_cast<u16>(status_flags & (slot.l_imm15 & 0xFFFu))); break;
        case L_FSOR:   set_vi(it, static_cast<u16>((status_flags | slot.l_imm15) & 0xFFFu)); break;
        case L_FMEQ:   set_vi(it, static_cast<u16>((mac_flags & 0xFFFFu) == vi[is & 15] ? 1u : 0u)); break;
        case L_FMAND:  set_vi(it, static_cast<u16>(mac_flags & vi[is & 15])); break;
        case L_FMOR:   set_vi(it, static_cast<u16>((mac_flags | vi[is & 15]) & 0xFFFFu)); break;
        case L_FCGET:  set_vi(it, static_cast<u16>(clip_flags & 0xFFFu)); break;
        case L_B:
            branched = true;
            branch_target = (pc + 8u + static_cast<u32>(slot.l_imm11 * 8)) & 0x3FF8u;
            break;
        case L_BAL:
            set_vi(it, static_cast<u16>(((pc + 16u) >> 3) & 0x7FFu));
            branched = true;
            branch_target = (pc + 8u + static_cast<u32>(slot.l_imm11 * 8)) & 0x3FF8u;
            break;
        case L_JR:
            branched = true;
            branch_target = (static_cast<u32>(vi[is & 15]) * 8u) & 0x3FF8u;
            break;
        case L_JALR: {
            const u32 target = (static_cast<u32>(vi[is & 15]) * 8u) & 0x3FF8u;
            set_vi(it, static_cast<u16>(((pc + 16u) >> 3) & 0x7FFu));
            branched = true;
            branch_target = target;
            break;
        }
        case L_IBEQ:
            if (read_branch_vi(it) == read_branch_vi(is)) {
                branched = true;
                branch_target = (pc + 8u + static_cast<u32>(slot.l_imm11 * 8)) & 0x3FF8u;
            }
            break;
        case L_IBNE:
            if (read_branch_vi(it) != read_branch_vi(is)) {
                branched = true;
                branch_target = (pc + 8u + static_cast<u32>(slot.l_imm11 * 8)) & 0x3FF8u;
            }
            break;
        case L_IBLTZ:
            if (static_cast<s16>(read_branch_vi(is)) < 0) {
                branched = true;
                branch_target = (pc + 8u + static_cast<u32>(slot.l_imm11 * 8)) & 0x3FF8u;
            }
            break;
        case L_IBGTZ:
            if (static_cast<s16>(read_branch_vi(is)) > 0) {
                branched = true;
                branch_target = (pc + 8u + static_cast<u32>(slot.l_imm11 * 8)) & 0x3FF8u;
            }
            break;
        case L_IBLEZ:
            if (static_cast<s16>(read_branch_vi(is)) <= 0) {
                branched = true;
                branch_target = (pc + 8u + static_cast<u32>(slot.l_imm11 * 8)) & 0x3FF8u;
            }
            break;
        case L_IBGEZ:
            if (static_cast<s16>(read_branch_vi(is)) >= 0) {
                branched = true;
                branch_target = (pc + 8u + static_cast<u32>(slot.l_imm11 * 8)) & 0x3FF8u;
            }
            break;
        default:
            break;
    }
}

u32 VU1Unit::run_microprogram(u32 start_addr_bytes, u32 max_steps) {
    const u8* const code_mem = m_code_mem;
    if (!code_mem) return 0;
    PerfScope perf_vu1(PerfCat::Vu1);
    pc = start_addr_bytes & 0x3FF8u;
    if (m_code_dirty || m_decode_cache[pc >> 3].raw != read_le64(code_mem + pc)) {
        refresh_code_analysis();
    } else if (m_jit_bindings_dirty) {
        m_jit.clear_active_bindings();
        m_jit_bindings_dirty = false;
    }
    m_vi_alu_reg = 0;
    u32 steps = 0;
    bool end_pending = false;
    bool branch_pending = false;
    u32 pending_target = 0;

    while (steps < max_steps) {
        if (enable_jit && m_jit.is_supported() && m_data_mem &&
            !end_pending && !branch_pending && m_vi_alu_reg == 0) {
            Vu1JitBlockFn blk = m_jit.get_or_compile(pc >> 3, code_mem, m_has_mac_readers, m_has_clip_readers);
            if (blk) {
                const u32 res = blk(this, max_steps - steps);
                const u32 ran = res & 0x7FFFFFFFu;
                steps += ran;
                micro_steps_executed += ran;
                if (res & 0x80000000u) break;
                if (ran > 0) continue;
            }
        }

        if (m_pipe_busy) {
            advance_pipes();
        }

        const DecodedVu1Slot& slot = m_decode_cache[pc >> 3];

        UpperWrite uw{};
        if (slot.u_op != U_NOP) {
            exec_upper_decoded(slot, uw);
        }

        const u8 prev_alu_reg = m_vi_alu_reg;
        const u16 prev_alu_old = m_vi_alu_old;
        m_vi_alu_reg = 0;

        bool branched = false;
        u32 branch_target = 0;
        if (slot.l_op != L_NOP) {
            exec_lower_decoded(slot, branched, branch_target, prev_alu_reg, prev_alu_old);
        }

        // Commit upper-pipe writes after lower-pipe reads (PS2 VU1 parallel upper/lower execution)
        if (slot.u_op != U_NOP) {
            if (uw.write_mac) {
                m_mac_pipe[3] = uw.mac;
                m_mac_valid_mask |= 8u;
                m_pipe_busy = 1u;
            }
            if (uw.write_clip) {
                m_clip_pipe[3] = uw.clip;
                m_clip_valid_mask |= 8u;
                m_pipe_busy = 1u;
            }
            if (uw.write_vf && uw.reg != 0) {
                write_vec4_masked(vf[uw.reg], uw.val, uw.dest);
            }
            if (uw.write_acc) {
                write_vec4_masked(acc, uw.val, uw.dest);
            }
        }

        steps++;
        micro_steps_executed++;

        // Advance PC properly BEFORE breaking on E-bit delay slot so MSCNT resumes right after!
        if (branch_pending) {
            pc = pending_target;
            branch_pending = false;
            m_vi_alu_reg = 0;
        } else {
            pc = (pc + 8u) & 0x3FF8u;
            if (branched) {
                branch_pending = true;
                pending_target = branch_target;
            }
        }

        if (end_pending) break;
        if (slot.flags & 2u) end_pending = true;
    }

    flush_all_pipes();
    return steps;
}

} // namespace tf::ps2
