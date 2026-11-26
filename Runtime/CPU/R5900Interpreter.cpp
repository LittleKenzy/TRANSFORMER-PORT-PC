#include "R5900Interpreter.h"
#include "R5900Helpers.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace tf::ps2 {

namespace {

struct CachedR5900Ins {
    u32 address = 0xFFFFFFFFu;
    u32 raw = 0;
    u32 target = 0;
    R5900Opcode op = R5900Opcode::Unknown;
    s16 simm = 0;
    u16 uimm = 0;
    u8 rs = 0, rt = 0, rd = 0, sa = 0;
    u8 fs = 0, ft = 0, fd = 0;
    u8 flags = 0; // bit 0: has_delay_slot, bit 1: is_branch_likely

    bool has_delay_slot() const { return (flags & 1u) != 0; }
    bool is_branch_likely() const { return (flags & 2u) != 0; }
};

constexpr u32 kDecodeCacheSize = 65536u;
constexpr u32 kDecodeCacheMask = kDecodeCacheSize - 1u;

inline const CachedR5900Ins& fetch_cached_ins(
    std::vector<CachedR5900Ins>& cache,
    const u8* ram,
    const GuestMemory* mem,
    u32 pc
) {
    const u32 raw = (pc <= kEeRamSize - 4u) ? read_le32(ram + pc) : mem->read32(pc);
    CachedR5900Ins& e = cache[(pc >> 2) & kDecodeCacheMask];
    if (e.address != pc || e.raw != raw) {
        DecodedInstruction d = R5900Decoder::decode(pc, raw);
        e.address = pc;
        e.raw     = raw;
        e.target  = d.target;
        e.op      = d.op;
        e.simm    = d.simm;
        e.uimm    = d.uimm;
        e.rs      = d.rs;
        e.rt      = d.rt;
        e.rd      = d.rd;
        e.sa      = d.sa;
        e.fs      = d.fs;
        e.ft      = d.ft;
        e.fd      = d.fd;
        e.flags   = (d.has_delay_slot ? 1u : 0u) | (d.is_branch_likely ? 2u : 0u);
    }
    return e;
}

template <typename InsT>
inline void exec_single_impl(EEContext& ctx, const InsT& ins) {
    if (ins.raw == 0) return;

    u8 rs = ins.rs, rt = ins.rt, rd = ins.rd, sa = ins.sa;
    u8 fs = ins.fs, ft = ins.ft, fd = ins.fd;
    s16 simm = ins.simm;
    u16 uimm = ins.uimm;

    switch (ins.op) {
        case R5900Opcode::SLL:    ctx.set_gpr_u32_sext(rd, ctx.gpr_u32(rt) << sa); break;
        case R5900Opcode::SRL:    ctx.set_gpr_u32_sext(rd, ctx.gpr_u32(rt) >> sa); break;
        case R5900Opcode::SRA:    ctx.set_gpr_s32(rd, ctx.gpr_s32(rt) >> sa); break;
        case R5900Opcode::SLLV:   ctx.set_gpr_u32_sext(rd, ctx.gpr_u32(rt) << (ctx.gpr_u32(rs) & 31u)); break;
        case R5900Opcode::SRLV:   ctx.set_gpr_u32_sext(rd, ctx.gpr_u32(rt) >> (ctx.gpr_u32(rs) & 31u)); break;
        case R5900Opcode::SRAV:   ctx.set_gpr_s32(rd, ctx.gpr_s32(rt) >> (ctx.gpr_u32(rs) & 31u)); break;
        case R5900Opcode::DSLL:   ctx.set_gpr_u64(rd, ctx.gpr_u64(rt) << sa); break;
        case R5900Opcode::DSRL:   ctx.set_gpr_u64(rd, ctx.gpr_u64(rt) >> sa); break;
        case R5900Opcode::DSRA:   ctx.set_gpr_s64(rd, ctx.gpr_s64(rt) >> sa); break;
        case R5900Opcode::DSLL32: ctx.set_gpr_u64(rd, ctx.gpr_u64(rt) << (sa + 32)); break;
        case R5900Opcode::DSRL32: ctx.set_gpr_u64(rd, ctx.gpr_u64(rt) >> (sa + 32)); break;
        case R5900Opcode::DSRA32: ctx.set_gpr_s64(rd, ctx.gpr_s64(rt) >> (sa + 32)); break;
        case R5900Opcode::DSLLV:  ctx.set_gpr_u64(rd, ctx.gpr_u64(rt) << (ctx.gpr_u32(rs) & 63u)); break;
        case R5900Opcode::DSRLV:  ctx.set_gpr_u64(rd, ctx.gpr_u64(rt) >> (ctx.gpr_u32(rs) & 63u)); break;
        case R5900Opcode::DSRAV:  ctx.set_gpr_s64(rd, ctx.gpr_s64(rt) >> (ctx.gpr_u32(rs) & 63u)); break;

        case R5900Opcode::ADD:
        case R5900Opcode::ADDU:   ctx.set_gpr_u32_sext(rd, ctx.gpr_u32(rs) + ctx.gpr_u32(rt)); break;
        case R5900Opcode::SUB:
        case R5900Opcode::SUBU:   ctx.set_gpr_u32_sext(rd, ctx.gpr_u32(rs) - ctx.gpr_u32(rt)); break;
        case R5900Opcode::DADD:
        case R5900Opcode::DADDU:  ctx.set_gpr_u64(rd, ctx.gpr_u64(rs) + ctx.gpr_u64(rt)); break;
        case R5900Opcode::DSUB:
        case R5900Opcode::DSUBU:  ctx.set_gpr_u64(rd, ctx.gpr_u64(rs) - ctx.gpr_u64(rt)); break;
        case R5900Opcode::ADDI:
        case R5900Opcode::ADDIU:  ctx.set_gpr_u32_sext(rt, ctx.gpr_u32(rs) + static_cast<u32>(static_cast<s32>(simm))); break;
        case R5900Opcode::DADDI:
        case R5900Opcode::DADDIU: ctx.set_gpr_u64(rt, ctx.gpr_u64(rs) + static_cast<u64>(static_cast<s64>(simm))); break;
        case R5900Opcode::AND:    ctx.set_gpr_u64(rd, ctx.gpr_u64(rs) & ctx.gpr_u64(rt)); break;
        case R5900Opcode::OR:     ctx.set_gpr_u64(rd, ctx.gpr_u64(rs) | ctx.gpr_u64(rt)); break;
        case R5900Opcode::XOR:    ctx.set_gpr_u64(rd, ctx.gpr_u64(rs) ^ ctx.gpr_u64(rt)); break;
        case R5900Opcode::NOR:    ctx.set_gpr_u64(rd, ~(ctx.gpr_u64(rs) | ctx.gpr_u64(rt))); break;
        case R5900Opcode::ANDI:   ctx.set_gpr_u64(rt, ctx.gpr_u64(rs) & uimm); break;
        case R5900Opcode::ORI:    ctx.set_gpr_u64(rt, ctx.gpr_u64(rs) | uimm); break;
        case R5900Opcode::XORI:   ctx.set_gpr_u64(rt, ctx.gpr_u64(rs) ^ uimm); break;
        case R5900Opcode::LUI:    ctx.set_gpr_u32_sext(rt, static_cast<u32>(uimm) << 16); break;
        case R5900Opcode::SLT:    ctx.set_gpr_u64(rd, ctx.gpr_s64(rs) < ctx.gpr_s64(rt) ? 1ULL : 0ULL); break;
        case R5900Opcode::SLTU:   ctx.set_gpr_u64(rd, ctx.gpr_u64(rs) < ctx.gpr_u64(rt) ? 1ULL : 0ULL); break;
        case R5900Opcode::SLTI:   ctx.set_gpr_u64(rt, ctx.gpr_s64(rs) < static_cast<s64>(simm) ? 1ULL : 0ULL); break;
        case R5900Opcode::SLTIU:  ctx.set_gpr_u64(rt, ctx.gpr_u64(rs) < static_cast<u64>(static_cast<s64>(simm)) ? 1ULL : 0ULL); break;
        case R5900Opcode::MOVZ:   if (ctx.gpr_u64(rt) == 0) ctx.set_gpr_u64(rd, ctx.gpr_u64(rs)); break;
        case R5900Opcode::MOVN:   if (ctx.gpr_u64(rt) != 0) ctx.set_gpr_u64(rd, ctx.gpr_u64(rs)); break;

        case R5900Opcode::MFHI:   ctx.set_gpr_u64(rd, ctx.hi); break;
        case R5900Opcode::MTHI:   ctx.hi = ctx.gpr_u64(rs); break;
        case R5900Opcode::MFLO:   ctx.set_gpr_u64(rd, ctx.lo); break;
        case R5900Opcode::MTLO:   ctx.lo = ctx.gpr_u64(rs); break;
        case R5900Opcode::MFHI1:  ctx.set_gpr_u64(rd, ctx.hi1); break;
        case R5900Opcode::MTHI1:  ctx.hi1 = ctx.gpr_u64(rs); break;
        case R5900Opcode::MFLO1:  ctx.set_gpr_u64(rd, ctx.lo1); break;
        case R5900Opcode::MTLO1:  ctx.lo1 = ctx.gpr_u64(rs); break;
        case R5900Opcode::MFSA:   ctx.set_gpr_u64(rd, ctx.sa); break;
        case R5900Opcode::MTSA:   ctx.sa = ctx.gpr_u32(rs) & 0xFu; break;
        case R5900Opcode::MTSAB:  ctx.sa = (ctx.gpr_u32(rs) ^ uimm) & 0xFu; break;
        case R5900Opcode::MTSAH:  ctx.sa = ((ctx.gpr_u32(rs) ^ uimm) & 0x7u) << 1; break;
        case R5900Opcode::MULT:   helpers::mult_op(ctx, rd, rs, rt, false); break;
        case R5900Opcode::MULTU:  helpers::multu_op(ctx, rd, rs, rt, false); break;
        case R5900Opcode::DIV:    helpers::div_op(ctx, rs, rt, false); break;
        case R5900Opcode::DIVU:   helpers::divu_op(ctx, rs, rt, false); break;
        case R5900Opcode::MULT1:  helpers::mult_op(ctx, rd, rs, rt, true); break;
        case R5900Opcode::MULTU1: helpers::multu_op(ctx, rd, rs, rt, true); break;
        case R5900Opcode::DIV1:   helpers::div_op(ctx, rs, rt, true); break;
        case R5900Opcode::DIVU1:  helpers::divu_op(ctx, rs, rt, true); break;
        case R5900Opcode::MADD:   helpers::madd_op(ctx, rd, rs, rt, false); break;
        case R5900Opcode::MADDU:  helpers::maddu_op(ctx, rd, rs, rt, false); break;
        case R5900Opcode::MADD1:  helpers::madd_op(ctx, rd, rs, rt, true); break;
        case R5900Opcode::MADDU1: helpers::maddu_op(ctx, rd, rs, rt, true); break;

        case R5900Opcode::LB:   ctx.set_gpr_s64(rt, static_cast<s8>(ctx.mem->read8(ctx.gpr_u32(rs) + simm))); break;
        case R5900Opcode::LBU:  ctx.set_gpr_u64(rt, ctx.mem->read8(ctx.gpr_u32(rs) + simm)); break;
        case R5900Opcode::LH:   ctx.set_gpr_s64(rt, static_cast<s16>(ctx.mem->read16(ctx.gpr_u32(rs) + simm))); break;
        case R5900Opcode::LHU:  ctx.set_gpr_u64(rt, ctx.mem->read16(ctx.gpr_u32(rs) + simm)); break;
        case R5900Opcode::LW:   ctx.set_gpr_u32_sext(rt, ctx.mem->read32(ctx.gpr_u32(rs) + simm)); break;
        case R5900Opcode::LWU:  ctx.set_gpr_u64(rt, ctx.mem->read32(ctx.gpr_u32(rs) + simm)); break;
        case R5900Opcode::LD:   ctx.set_gpr_u64(rt, ctx.mem->read64(ctx.gpr_u32(rs) + simm)); break;
        case R5900Opcode::LQ:   ctx.set_gpr_u128(rt, ctx.mem->read128(ctx.gpr_u32(rs) + simm)); break;
        case R5900Opcode::LWL:  ctx.set_gpr_u32_sext(rt, helpers::lwl(*ctx.mem, ctx.gpr_u32(rs) + simm, ctx.gpr_u32(rt))); break;
        case R5900Opcode::LWR:  ctx.set_gpr_u32_sext(rt, helpers::lwr(*ctx.mem, ctx.gpr_u32(rs) + simm, ctx.gpr_u32(rt))); break;
        case R5900Opcode::LDL:  ctx.set_gpr_u64(rt, helpers::ldl(*ctx.mem, ctx.gpr_u32(rs) + simm, ctx.gpr_u64(rt))); break;
        case R5900Opcode::LDR:  ctx.set_gpr_u64(rt, helpers::ldr(*ctx.mem, ctx.gpr_u32(rs) + simm, ctx.gpr_u64(rt))); break;
        case R5900Opcode::SB:   ctx.mem->write8(ctx.gpr_u32(rs) + simm, static_cast<u8>(ctx.gpr_u32(rt))); break;
        case R5900Opcode::SH:   ctx.mem->write16(ctx.gpr_u32(rs) + simm, static_cast<u16>(ctx.gpr_u32(rt))); break;
        case R5900Opcode::SW:   ctx.mem->write32(ctx.gpr_u32(rs) + simm, ctx.gpr_u32(rt)); break;
        case R5900Opcode::SD:   ctx.mem->write64(ctx.gpr_u32(rs) + simm, ctx.gpr_u64(rt)); break;
        case R5900Opcode::SQ:   ctx.mem->write128(ctx.gpr_u32(rs) + simm, ctx.gpr_u128(rt)); break;
        case R5900Opcode::SWL:  helpers::swl(*ctx.mem, ctx.gpr_u32(rs) + simm, ctx.gpr_u32(rt)); break;
        case R5900Opcode::SWR:  helpers::swr(*ctx.mem, ctx.gpr_u32(rs) + simm, ctx.gpr_u32(rt)); break;
        case R5900Opcode::SDL:  helpers::sdl(*ctx.mem, ctx.gpr_u32(rs) + simm, ctx.gpr_u64(rt)); break;
        case R5900Opcode::SDR:  helpers::sdr(*ctx.mem, ctx.gpr_u32(rs) + simm, ctx.gpr_u64(rt)); break;
        case R5900Opcode::LWC1: ctx.set_fpr_bits(ft, ctx.mem->read32(ctx.gpr_u32(rs) + simm)); break;
        case R5900Opcode::SWC1: ctx.mem->write32(ctx.gpr_u32(rs) + simm, ctx.fpr_bits(ft)); break;
        case R5900Opcode::LQC2: { u128 q = ctx.mem->read128(ctx.gpr_u32(rs) + simm); if (ft != 0) std::memcpy(static_cast<void*>(&ctx.vf[ft]), &q, 16); break; }
        case R5900Opcode::SQC2: { u128 q{}; std::memcpy(static_cast<void*>(&q), &ctx.vf[ft], 16); ctx.mem->write128(ctx.gpr_u32(rs) + simm, q); break; }
        case R5900Opcode::PREF: case R5900Opcode::CACHE: case R5900Opcode::SYNC: break;

        case R5900Opcode::SYSCALL: ctx.pc = ins.address; ctx.do_syscall((ins.raw >> 6) & 0xFFFFFu); break;
        case R5900Opcode::MFC0: ctx.set_gpr_u32_sext(rt, ctx.cop0[rd]); break;
        case R5900Opcode::MTC0: ctx.cop0[rd] = ctx.gpr_u32(rt); break;
        case R5900Opcode::EI:   ctx.interrupts_enabled = true; ctx.cop0[12] |= 0x00010000u; break;
        case R5900Opcode::DI:   ctx.interrupts_enabled = false; ctx.cop0[12] &= ~0x00010000u; break;
        case R5900Opcode::TLBR:
        case R5900Opcode::TLBWI:
        case R5900Opcode::TLBWR:
        case R5900Opcode::TLBP: break;

        case R5900Opcode::MFC1:    ctx.set_gpr_u32_sext(rt, ctx.fpr_bits(fs)); break;
        case R5900Opcode::MTC1:    ctx.set_fpr_bits(fs, ctx.gpr_u32(rt)); break;
        case R5900Opcode::CFC1:    ctx.set_gpr_u32_sext(rt, fs == 0 ? ctx.fcr0 : ctx.fcr31); break;
        case R5900Opcode::CTC1:    if (fs == 31) ctx.fcr31 = ctx.gpr_u32(rt); break;
        case R5900Opcode::ADD_S:   ctx.fpr[fd] = ctx.fpr[fs] + ctx.fpr[ft]; break;
        case R5900Opcode::SUB_S:   ctx.fpr[fd] = ctx.fpr[fs] - ctx.fpr[ft]; break;
        case R5900Opcode::MUL_S:   ctx.fpr[fd] = ctx.fpr[fs] * ctx.fpr[ft]; break;
        case R5900Opcode::DIV_S:   ctx.fpr[fd] = ctx.fpr[fs] / ctx.fpr[ft]; break;
        case R5900Opcode::SQRT_S:  ctx.fpr[fd] = std::sqrt(std::fabs(ctx.fpr[ft])); break;
        case R5900Opcode::RSQRT_S: ctx.fpr[fd] = ctx.fpr[fs] / std::sqrt(std::fabs(ctx.fpr[ft])); break;
        case R5900Opcode::ABS_S:   ctx.fpr[fd] = std::fabs(ctx.fpr[fs]); break;
        case R5900Opcode::MOV_S:   ctx.fpr[fd] = ctx.fpr[fs]; break;
        case R5900Opcode::NEG_S:   ctx.fpr[fd] = -ctx.fpr[fs]; break;
        case R5900Opcode::ADDA_S:  ctx.fpu_acc = ctx.fpr[fs] + ctx.fpr[ft]; break;
        case R5900Opcode::SUBA_S:  ctx.fpu_acc = ctx.fpr[fs] - ctx.fpr[ft]; break;
        case R5900Opcode::MULA_S:  ctx.fpu_acc = ctx.fpr[fs] * ctx.fpr[ft]; break;
        case R5900Opcode::MADD_S:  ctx.fpr[fd] = ctx.fpu_acc + ctx.fpr[fs] * ctx.fpr[ft]; break;
        case R5900Opcode::MSUB_S:  ctx.fpr[fd] = ctx.fpu_acc - ctx.fpr[fs] * ctx.fpr[ft]; break;
        case R5900Opcode::MADDA_S: ctx.fpu_acc += ctx.fpr[fs] * ctx.fpr[ft]; break;
        case R5900Opcode::MSUBA_S: ctx.fpu_acc -= ctx.fpr[fs] * ctx.fpr[ft]; break;
        case R5900Opcode::CVT_W_S: ctx.set_fpr_bits(fd, static_cast<u32>(static_cast<s32>(ctx.fpr[fs]))); break;
        case R5900Opcode::CVT_S_W: ctx.fpr[fd] = static_cast<f32>(static_cast<s32>(ctx.fpr_bits(fs))); break;
        case R5900Opcode::MAX_S:   ctx.fpr[fd] = std::max(ctx.fpr[fs], ctx.fpr[ft]); break;
        case R5900Opcode::MIN_S:   ctx.fpr[fd] = std::min(ctx.fpr[fs], ctx.fpr[ft]); break;
        case R5900Opcode::C_F_S:   ctx.set_fpu_cond(false); break;
        case R5900Opcode::C_EQ_S:  ctx.set_fpu_cond(ctx.fpr[fs] == ctx.fpr[ft]); break;
        case R5900Opcode::C_LT_S:  ctx.set_fpu_cond(ctx.fpr[fs] < ctx.fpr[ft]); break;
        case R5900Opcode::C_LE_S:  ctx.set_fpu_cond(ctx.fpr[fs] <= ctx.fpr[ft]); break;

        case R5900Opcode::PAND:    { auto a=ctx.gpr_u128(rs), b=ctx.gpr_u128(rt); ctx.set_gpr_u128(rd, u128(a.lo & b.lo, a.hi & b.hi)); break; }
        case R5900Opcode::POR:     { auto a=ctx.gpr_u128(rs), b=ctx.gpr_u128(rt); ctx.set_gpr_u128(rd, u128(a.lo | b.lo, a.hi | b.hi)); break; }
        case R5900Opcode::PXOR:    { auto a=ctx.gpr_u128(rs), b=ctx.gpr_u128(rt); ctx.set_gpr_u128(rd, u128(a.lo ^ b.lo, a.hi ^ b.hi)); break; }
        case R5900Opcode::PNOR:    { auto a=ctx.gpr_u128(rs), b=ctx.gpr_u128(rt); ctx.set_gpr_u128(rd, u128(~(a.lo | b.lo), ~(a.hi | b.hi))); break; }
        case R5900Opcode::PADDW:   ctx.set_gpr_u128(rd, helpers::mmi_paddw(ctx.gpr_u128(rs), ctx.gpr_u128(rt))); break;
        case R5900Opcode::PSUBW:   ctx.set_gpr_u128(rd, helpers::mmi_psubw(ctx.gpr_u128(rs), ctx.gpr_u128(rt))); break;
        case R5900Opcode::PCGTW:   ctx.set_gpr_u128(rd, helpers::mmi_pcgtw(ctx.gpr_u128(rs), ctx.gpr_u128(rt))); break;
        case R5900Opcode::PCEQW:   ctx.set_gpr_u128(rd, helpers::mmi_pceqw(ctx.gpr_u128(rs), ctx.gpr_u128(rt))); break;
        case R5900Opcode::PMAXW:   ctx.set_gpr_u128(rd, helpers::mmi_pmaxw(ctx.gpr_u128(rs), ctx.gpr_u128(rt))); break;
        case R5900Opcode::PMINW:   ctx.set_gpr_u128(rd, helpers::mmi_pminw(ctx.gpr_u128(rs), ctx.gpr_u128(rt))); break;
        case R5900Opcode::PADDSW:  ctx.set_gpr_u128(rd, helpers::mmi_paddsw(ctx.gpr_u128(rs), ctx.gpr_u128(rt))); break;
        case R5900Opcode::PSUBSW:  ctx.set_gpr_u128(rd, helpers::mmi_psubsw(ctx.gpr_u128(rs), ctx.gpr_u128(rt))); break;
        case R5900Opcode::PADDUW:  ctx.set_gpr_u128(rd, helpers::mmi_padduw(ctx.gpr_u128(rs), ctx.gpr_u128(rt))); break;
        case R5900Opcode::PSUBUW:  ctx.set_gpr_u128(rd, helpers::mmi_psubuw(ctx.gpr_u128(rs), ctx.gpr_u128(rt))); break;
        case R5900Opcode::PABSW:   ctx.set_gpr_u128(rd, helpers::mmi_pabsw(ctx.gpr_u128(rt))); break;
        case R5900Opcode::PADDH:   ctx.set_gpr_u128(rd, helpers::mmi_paddh(ctx.gpr_u128(rs), ctx.gpr_u128(rt))); break;
        case R5900Opcode::PSUBH:   ctx.set_gpr_u128(rd, helpers::mmi_psubh(ctx.gpr_u128(rs), ctx.gpr_u128(rt))); break;
        case R5900Opcode::PCGTH:   ctx.set_gpr_u128(rd, helpers::mmi_pcgth(ctx.gpr_u128(rs), ctx.gpr_u128(rt))); break;
        case R5900Opcode::PCEQH:   ctx.set_gpr_u128(rd, helpers::mmi_pceqh(ctx.gpr_u128(rs), ctx.gpr_u128(rt))); break;
        case R5900Opcode::PMAXH:   ctx.set_gpr_u128(rd, helpers::mmi_pmaxh(ctx.gpr_u128(rs), ctx.gpr_u128(rt))); break;
        case R5900Opcode::PMINH:   ctx.set_gpr_u128(rd, helpers::mmi_pminh(ctx.gpr_u128(rs), ctx.gpr_u128(rt))); break;
        case R5900Opcode::PADDSH:  ctx.set_gpr_u128(rd, helpers::mmi_paddsh(ctx.gpr_u128(rs), ctx.gpr_u128(rt))); break;
        case R5900Opcode::PSUBSH:  ctx.set_gpr_u128(rd, helpers::mmi_psubsh(ctx.gpr_u128(rs), ctx.gpr_u128(rt))); break;
        case R5900Opcode::PADDUH:  ctx.set_gpr_u128(rd, helpers::mmi_padduh(ctx.gpr_u128(rs), ctx.gpr_u128(rt))); break;
        case R5900Opcode::PSUBUH:  ctx.set_gpr_u128(rd, helpers::mmi_psubuh(ctx.gpr_u128(rs), ctx.gpr_u128(rt))); break;
        case R5900Opcode::PABSH:   ctx.set_gpr_u128(rd, helpers::mmi_pabsh(ctx.gpr_u128(rt))); break;
        case R5900Opcode::PADDB:   ctx.set_gpr_u128(rd, helpers::mmi_paddb(ctx.gpr_u128(rs), ctx.gpr_u128(rt))); break;
        case R5900Opcode::PSUBB:   ctx.set_gpr_u128(rd, helpers::mmi_psubb(ctx.gpr_u128(rs), ctx.gpr_u128(rt))); break;
        case R5900Opcode::PCGTB:   ctx.set_gpr_u128(rd, helpers::mmi_pcgtb(ctx.gpr_u128(rs), ctx.gpr_u128(rt))); break;
        case R5900Opcode::PCEQB:   ctx.set_gpr_u128(rd, helpers::mmi_pceqb(ctx.gpr_u128(rs), ctx.gpr_u128(rt))); break;
        case R5900Opcode::PADDSB:  ctx.set_gpr_u128(rd, helpers::mmi_paddsb(ctx.gpr_u128(rs), ctx.gpr_u128(rt))); break;
        case R5900Opcode::PSUBSB:  ctx.set_gpr_u128(rd, helpers::mmi_psubsb(ctx.gpr_u128(rs), ctx.gpr_u128(rt))); break;
        case R5900Opcode::PADDUB:  ctx.set_gpr_u128(rd, helpers::mmi_paddub(ctx.gpr_u128(rs), ctx.gpr_u128(rt))); break;
        case R5900Opcode::PSUBUB:  ctx.set_gpr_u128(rd, helpers::mmi_psubub(ctx.gpr_u128(rs), ctx.gpr_u128(rt))); break;
        case R5900Opcode::PEXTLW:  ctx.set_gpr_u128(rd, helpers::mmi_pextlw(ctx.gpr_u128(rs), ctx.gpr_u128(rt))); break;
        case R5900Opcode::PEXTUW:  ctx.set_gpr_u128(rd, helpers::mmi_pextuw(ctx.gpr_u128(rs), ctx.gpr_u128(rt))); break;
        case R5900Opcode::PPACW:   ctx.set_gpr_u128(rd, helpers::mmi_ppacw(ctx.gpr_u128(rs), ctx.gpr_u128(rt))); break;
        case R5900Opcode::PEXTLH:  ctx.set_gpr_u128(rd, helpers::mmi_pextlh(ctx.gpr_u128(rs), ctx.gpr_u128(rt))); break;
        case R5900Opcode::PEXTUH:  ctx.set_gpr_u128(rd, helpers::mmi_pextuh(ctx.gpr_u128(rs), ctx.gpr_u128(rt))); break;
        case R5900Opcode::PPACH:   ctx.set_gpr_u128(rd, helpers::mmi_ppach(ctx.gpr_u128(rs), ctx.gpr_u128(rt))); break;
        case R5900Opcode::PEXTLB:  ctx.set_gpr_u128(rd, helpers::mmi_pextlb(ctx.gpr_u128(rs), ctx.gpr_u128(rt))); break;
        case R5900Opcode::PEXTUB:  ctx.set_gpr_u128(rd, helpers::mmi_pextub(ctx.gpr_u128(rs), ctx.gpr_u128(rt))); break;
        case R5900Opcode::PPACB:   ctx.set_gpr_u128(rd, helpers::mmi_ppacb(ctx.gpr_u128(rs), ctx.gpr_u128(rt))); break;
        case R5900Opcode::PCPYLD:  ctx.set_gpr_u128(rd, helpers::mmi_pcpyld(ctx.gpr_u128(rs), ctx.gpr_u128(rt))); break;
        case R5900Opcode::PCPYUD:  ctx.set_gpr_u128(rd, helpers::mmi_pcpyud(ctx.gpr_u128(rs), ctx.gpr_u128(rt))); break;
        case R5900Opcode::PCPYH:   ctx.set_gpr_u128(rd, helpers::mmi_pcpyh(ctx.gpr_u128(rt))); break;
        case R5900Opcode::PROT3W:  ctx.set_gpr_u128(rd, helpers::mmi_prot3w(ctx.gpr_u128(rt))); break;
        case R5900Opcode::PEXEW:   ctx.set_gpr_u128(rd, helpers::mmi_pexew(ctx.gpr_u128(rt))); break;
        case R5900Opcode::PEXCW:   ctx.set_gpr_u128(rd, helpers::mmi_pexcw(ctx.gpr_u128(rt))); break;
        case R5900Opcode::PEXEH:   ctx.set_gpr_u128(rd, helpers::mmi_pexeh(ctx.gpr_u128(rt))); break;
        case R5900Opcode::PEXCH:   ctx.set_gpr_u128(rd, helpers::mmi_pexch(ctx.gpr_u128(rt))); break;
        case R5900Opcode::PREVH:   ctx.set_gpr_u128(rd, helpers::mmi_prevh(ctx.gpr_u128(rt))); break;
        case R5900Opcode::PINTH:   ctx.set_gpr_u128(rd, helpers::mmi_pinth(ctx.gpr_u128(rs), ctx.gpr_u128(rt))); break;
        case R5900Opcode::PINTEH:  ctx.set_gpr_u128(rd, helpers::mmi_pinteh(ctx.gpr_u128(rs), ctx.gpr_u128(rt))); break;
        case R5900Opcode::PADSBH:  ctx.set_gpr_u128(rd, helpers::mmi_padsbh(ctx.gpr_u128(rs), ctx.gpr_u128(rt))); break;
        case R5900Opcode::PEXT5:   ctx.set_gpr_u128(rd, helpers::mmi_pext5(ctx.gpr_u128(rt))); break;
        case R5900Opcode::PPAC5:   ctx.set_gpr_u128(rd, helpers::mmi_ppac5(ctx.gpr_u128(rt))); break;
        case R5900Opcode::QFSRV:   ctx.set_gpr_u128(rd, helpers::mmi_qfsrv(ctx.gpr_u128(rs), ctx.gpr_u128(rt), ctx.sa)); break;
        case R5900Opcode::PSLLH:   ctx.set_gpr_u128(rd, helpers::mmi_psllh(ctx.gpr_u128(rt), sa)); break;
        case R5900Opcode::PSRLH:   ctx.set_gpr_u128(rd, helpers::mmi_psrlh(ctx.gpr_u128(rt), sa)); break;
        case R5900Opcode::PSRAH:   ctx.set_gpr_u128(rd, helpers::mmi_psrah(ctx.gpr_u128(rt), sa)); break;
        case R5900Opcode::PSLLW:   ctx.set_gpr_u128(rd, helpers::mmi_psllw(ctx.gpr_u128(rt), sa)); break;
        case R5900Opcode::PSRLW:   ctx.set_gpr_u128(rd, helpers::mmi_psrlw(ctx.gpr_u128(rt), sa)); break;
        case R5900Opcode::PSRAW:   ctx.set_gpr_u128(rd, helpers::mmi_psraw(ctx.gpr_u128(rt), sa)); break;
        case R5900Opcode::PSLLVW:  { auto a=ctx.gpr_u128(rs), b=ctx.gpr_u128(rt); u128 r{}; r.lo = static_cast<u64>(static_cast<s64>(static_cast<s32>(b.uw(0) << (a.uw(0) & 31u)))); r.hi = static_cast<u64>(static_cast<s64>(static_cast<s32>(b.uw(2) << (a.uw(2) & 31u)))); ctx.set_gpr_u128(rd, r); break; }
        case R5900Opcode::PSRLVW:  { auto a=ctx.gpr_u128(rs), b=ctx.gpr_u128(rt); u128 r{}; r.lo = static_cast<u64>(static_cast<s64>(static_cast<s32>(b.uw(0) >> (a.uw(0) & 31u)))); r.hi = static_cast<u64>(static_cast<s64>(static_cast<s32>(b.uw(2) >> (a.uw(2) & 31u)))); ctx.set_gpr_u128(rd, r); break; }
        case R5900Opcode::PSRAVW:  { auto a=ctx.gpr_u128(rs), b=ctx.gpr_u128(rt); u128 r{}; r.lo = static_cast<u64>(static_cast<s64>(b.sw(0) >> (a.uw(0) & 31u))); r.hi = static_cast<u64>(static_cast<s64>(b.sw(2) >> (a.uw(2) & 31u))); ctx.set_gpr_u128(rd, r); break; }
        case R5900Opcode::PMFHL:   ctx.set_gpr_u128(rd, helpers::mmi_pmfhl(ctx, sa)); break;
        case R5900Opcode::PMTHL:   { auto a=ctx.gpr_u128(rs); ctx.lo=a.uw(0); ctx.hi=a.uw(1); ctx.lo1=a.uw(2); ctx.hi1=a.uw(3); break; }
        case R5900Opcode::PMFHI:   ctx.set_gpr_u128(rd, u128(ctx.hi, ctx.hi1)); break;
        case R5900Opcode::PMFLO:   ctx.set_gpr_u128(rd, u128(ctx.lo, ctx.lo1)); break;
        case R5900Opcode::PMTHI:   { auto a=ctx.gpr_u128(rs); ctx.hi=a.lo; ctx.hi1=a.hi; break; }
        case R5900Opcode::PMTLO:   { auto a=ctx.gpr_u128(rs); ctx.lo=a.lo; ctx.lo1=a.hi; break; }
        case R5900Opcode::PMULTW:  helpers::mmi_pmultw(ctx, rd, ctx.gpr_u128(rs), ctx.gpr_u128(rt)); break;
        case R5900Opcode::PMULTUW: helpers::mmi_pmultuw(ctx, rd, ctx.gpr_u128(rs), ctx.gpr_u128(rt)); break;
        case R5900Opcode::PMULTH:  helpers::mmi_pmulth(ctx, rd, ctx.gpr_u128(rs), ctx.gpr_u128(rt)); break;
        case R5900Opcode::PMADDH:  helpers::mmi_pmaddh(ctx, rd, ctx.gpr_u128(rs), ctx.gpr_u128(rt)); break;
        case R5900Opcode::PLZCW:   { u32 a=ctx.gpr_u32(rs); ctx.set_gpr_u32_sext(rd, helpers::plzcw_word(a)); break; }
        case R5900Opcode::QMFC2:   { u128 q{}; std::memcpy(static_cast<void*>(&q), &ctx.vf[rd], 16); ctx.set_gpr_u128(rt, q); break; }
        case R5900Opcode::QMTC2:   { u128 q = ctx.gpr_u128(rt); if (rd != 0) std::memcpy(static_cast<void*>(&ctx.vf[rd]), &q, 16); break; }
        case R5900Opcode::CFC2:    ctx.set_gpr_u32_sext(rt, ctx.vi[rd]); break;
        case R5900Opcode::CTC2:    if (rd != 0) ctx.vi[rd] = ctx.gpr_u32(rt); break;
        case R5900Opcode::VU0_MACRO_OP: ctx.do_vu0_macro(ins.raw); break;

        default:
            ctx.trap_unsupported(ins.address, ins.raw, R5900Decoder::opcode_name(ins.op));
            break;
    }
}

} // namespace

void R5900FallbackInterpreter::exec_single(EEContext& ctx, const DecodedInstruction& ins) {
    exec_single_impl(ctx, ins);
}

u64 R5900FallbackInterpreter::run_until_return_or_handoff(
    EEContext& ctx,
    u32 start_pc,
    const std::unordered_map<u32, RecompiledFn>& static_table,
    u64 max_steps,
    const RecompiledFn* fast_table,
    u32 fast_base,
    u32 fast_end,
    void (*cooperative_yield_fn)(void*),
    void* cooperative_yield_ctx
) {
    thread_local std::vector<CachedR5900Ins> t_decode_cache;
    if (t_decode_cache.empty()) {
        t_decode_cache.resize(kDecodeCacheSize);
    }

    auto lookup_static = [&](u32 target) -> RecompiledFn {
        if (fast_table && target >= fast_base && target < fast_end && (target & 3u) == 0) {
            return fast_table[(target - fast_base) >> 2];
        }
        auto it = static_table.find(target);
        return (it != static_table.end()) ? it->second : nullptr;
    };

    const GuestMemory* mem = ctx.mem;
    const u8* ram = mem ? mem->ram() : nullptr;
    u32 pc = start_pc;
    u64 steps = 0;
    u64 slice_steps = 0;
    u32 backward_branches = 0;
    constexpr u64 kCooperativeYieldStepInterval = 4096u;
    constexpr u32 kCooperativeYieldBackBranchInterval = 64u;

    while (!ctx.halted && (cooperative_yield_fn != nullptr || steps < max_steps)) {
        if (cooperative_yield_fn != nullptr &&
            (slice_steps >= kCooperativeYieldStepInterval ||
             backward_branches >= kCooperativeYieldBackBranchInterval)) {
            slice_steps = 0;
            backward_branches = 0;
            ctx.pc = pc;
            cooperative_yield_fn(cooperative_yield_ctx);
            if (ctx.halted) {
                break;
            }
        }
        if (pc == 0 || (pc & 3u) != 0) {
            ctx.halted = true;
            ctx.trap_message = "Fallback interpreter jumped to invalid PC " + hex32(pc);
            break;
        }
        const CachedR5900Ins ins = fetch_cached_ins(t_decode_cache, ram, mem, pc);
        steps++;
        slice_steps++;
        ctx.cycles++;

        if (!ins.has_delay_slot()) {
            if (ins.op == R5900Opcode::ERET) {
                ctx.pc = ctx.cop0[14];
                return steps;
            }
            exec_single_impl(ctx, ins);
            pc += 4;
            continue;
        }

        // Branch / jump with delay slot at pc + 4
        const CachedR5900Ins ds = fetch_cached_ins(t_decode_cache, ram, mem, pc + 4);

        if (ins.op == R5900Opcode::JR) {
            u32 target = ctx.gpr_u32(ins.rs);
            exec_single_impl(ctx, ds);
            steps++;
            slice_steps++;
            if (ins.rs == 31) {
                ctx.pc = target;
                return steps;
            }
            if (RecompiledFn fn = lookup_static(target)) {
                ctx.pc = target;
                fn(ctx);
                return steps;
            }
            if (target <= pc) backward_branches++;
            pc = target;
            continue;
        }
        if (ins.op == R5900Opcode::JALR) {
            u32 target = ctx.gpr_u32(ins.rs);
            ctx.set_gpr_u32_sext(ins.rd, pc + 8);
            exec_single_impl(ctx, ds);
            steps++;
            slice_steps++;
            ctx.dispatch(target);
            pc += 8;
            continue;
        }
        if (ins.op == R5900Opcode::JAL) {
            ctx.set_gpr_u32_sext(31, pc + 8);
            exec_single_impl(ctx, ds);
            steps++;
            slice_steps++;
            ctx.dispatch(ins.target);
            pc += 8;
            continue;
        }
        if (ins.op == R5900Opcode::J) {
            exec_single_impl(ctx, ds);
            steps++;
            slice_steps++;
            if (RecompiledFn fn = lookup_static(ins.target)) {
                ctx.pc = ins.target;
                fn(ctx);
                return steps;
            }
            if (ins.target <= pc) backward_branches++;
            pc = ins.target;
            continue;
        }

        // Conditional branches
        bool cond = false;
        switch (ins.op) {
            case R5900Opcode::BEQ: case R5900Opcode::BEQL: cond = (ctx.gpr_u64(ins.rs) == ctx.gpr_u64(ins.rt)); break;
            case R5900Opcode::BNE: case R5900Opcode::BNEL: cond = (ctx.gpr_u64(ins.rs) != ctx.gpr_u64(ins.rt)); break;
            case R5900Opcode::BLEZ: case R5900Opcode::BLEZL: cond = (ctx.gpr_s64(ins.rs) <= 0); break;
            case R5900Opcode::BGTZ: case R5900Opcode::BGTZL: cond = (ctx.gpr_s64(ins.rs) > 0); break;
            case R5900Opcode::BLTZ: case R5900Opcode::BLTZL:
            case R5900Opcode::BLTZAL: case R5900Opcode::BLTZALL: cond = (ctx.gpr_s64(ins.rs) < 0); break;
            case R5900Opcode::BGEZ: case R5900Opcode::BGEZL:
            case R5900Opcode::BGEZAL: case R5900Opcode::BGEZALL: cond = (ctx.gpr_s64(ins.rs) >= 0); break;
            case R5900Opcode::BC1F: case R5900Opcode::BC1FL: cond = !ctx.fpu_cond(); break;
            case R5900Opcode::BC1T: case R5900Opcode::BC1TL: cond = ctx.fpu_cond(); break;
            case R5900Opcode::BC2F: case R5900Opcode::BC2FL: cond = (ctx.vi[16] & 0x100u) == 0; break;
            case R5900Opcode::BC2T: case R5900Opcode::BC2TL: cond = (ctx.vi[16] & 0x100u) != 0; break;
            default: break;
        }

        if (ins.is_branch_likely()) {
            if (cond) {
                exec_single_impl(ctx, ds);
                steps++;
                slice_steps++;
                if (ins.target <= pc) backward_branches++;
                pc = ins.target;
            } else {
                pc += 8;
            }
        } else {
            exec_single_impl(ctx, ds);
            steps++;
            slice_steps++;
            if (cond) {
                if (ins.target <= pc) backward_branches++;
                pc = ins.target;
            } else {
                pc += 8;
            }
        }
    }

    if (cooperative_yield_fn == nullptr && steps >= max_steps && !ctx.halted) {
        ctx.halted = true;
        ctx.trap_message = "Fallback interpreter step budget reached at PC " + hex32(pc);
    }
    ctx.pc = pc;
    return steps;
}

} // namespace tf::ps2
