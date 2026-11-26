#include "CppCodeGenerator.h"

#include <cctype>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>

namespace tf::ps2 {

namespace {

std::string label_for(u32 addr) {
    std::ostringstream oss;
    oss << "L_" << std::hex << std::setw(8) << std::setfill('0') << addr;
    return oss.str();
}

void emit_non_branch_instruction(
    std::ostringstream& out,
    const DecodedInstruction& ins,
    const char* indent,
    CodegenResult& stats
) {
    if (ins.raw == 0) {
        stats.instructions_translated++;
        return; // Architectural NOP
    }

    auto rs = static_cast<unsigned>(ins.rs);
    auto rt = static_cast<unsigned>(ins.rt);
    auto rd = static_cast<unsigned>(ins.rd);
    auto sa = static_cast<unsigned>(ins.sa);
    auto fs = static_cast<unsigned>(ins.fs);
    auto ft = static_cast<unsigned>(ins.ft);
    auto fd = static_cast<unsigned>(ins.fd);
    auto simm = static_cast<int>(ins.simm);
    auto uimm = static_cast<unsigned>(ins.uimm);

    switch (ins.op) {
        // --- Shifts ---
        case R5900Opcode::SLL:
            out << indent << "ctx.set_gpr_u32_sext(" << rd << ", ctx.gpr_u32(" << rt << ") << " << sa << "u);\n";
            break;
        case R5900Opcode::SRL:
            out << indent << "ctx.set_gpr_u32_sext(" << rd << ", ctx.gpr_u32(" << rt << ") >> " << sa << "u);\n";
            break;
        case R5900Opcode::SRA:
            out << indent << "ctx.set_gpr_s32(" << rd << ", ctx.gpr_s32(" << rt << ") >> " << sa << "u);\n";
            break;
        case R5900Opcode::SLLV:
            out << indent << "ctx.set_gpr_u32_sext(" << rd << ", ctx.gpr_u32(" << rt << ") << (ctx.gpr_u32(" << rs << ") & 31u));\n";
            break;
        case R5900Opcode::SRLV:
            out << indent << "ctx.set_gpr_u32_sext(" << rd << ", ctx.gpr_u32(" << rt << ") >> (ctx.gpr_u32(" << rs << ") & 31u));\n";
            break;
        case R5900Opcode::SRAV:
            out << indent << "ctx.set_gpr_s32(" << rd << ", ctx.gpr_s32(" << rt << ") >> (ctx.gpr_u32(" << rs << ") & 31u));\n";
            break;
        case R5900Opcode::DSLL:
            out << indent << "ctx.set_gpr_u64(" << rd << ", ctx.gpr_u64(" << rt << ") << " << sa << "u);\n";
            break;
        case R5900Opcode::DSRL:
            out << indent << "ctx.set_gpr_u64(" << rd << ", ctx.gpr_u64(" << rt << ") >> " << sa << "u);\n";
            break;
        case R5900Opcode::DSRA:
            out << indent << "ctx.set_gpr_s64(" << rd << ", ctx.gpr_s64(" << rt << ") >> " << sa << "u);\n";
            break;
        case R5900Opcode::DSLL32:
            out << indent << "ctx.set_gpr_u64(" << rd << ", ctx.gpr_u64(" << rt << ") << " << (sa + 32) << "u);\n";
            break;
        case R5900Opcode::DSRL32:
            out << indent << "ctx.set_gpr_u64(" << rd << ", ctx.gpr_u64(" << rt << ") >> " << (sa + 32) << "u);\n";
            break;
        case R5900Opcode::DSRA32:
            out << indent << "ctx.set_gpr_s64(" << rd << ", ctx.gpr_s64(" << rt << ") >> " << (sa + 32) << "u);\n";
            break;
        case R5900Opcode::DSLLV:
            out << indent << "ctx.set_gpr_u64(" << rd << ", ctx.gpr_u64(" << rt << ") << (ctx.gpr_u32(" << rs << ") & 63u));\n";
            break;
        case R5900Opcode::DSRLV:
            out << indent << "ctx.set_gpr_u64(" << rd << ", ctx.gpr_u64(" << rt << ") >> (ctx.gpr_u32(" << rs << ") & 63u));\n";
            break;
        case R5900Opcode::DSRAV:
            out << indent << "ctx.set_gpr_s64(" << rd << ", ctx.gpr_s64(" << rt << ") >> (ctx.gpr_u32(" << rs << ") & 63u));\n";
            break;

        // --- Integer ALU ---
        case R5900Opcode::ADD:
        case R5900Opcode::ADDU:
            out << indent << "ctx.set_gpr_u32_sext(" << rd << ", ctx.gpr_u32(" << rs << ") + ctx.gpr_u32(" << rt << "));\n";
            break;
        case R5900Opcode::SUB:
        case R5900Opcode::SUBU:
            out << indent << "ctx.set_gpr_u32_sext(" << rd << ", ctx.gpr_u32(" << rs << ") - ctx.gpr_u32(" << rt << "));\n";
            break;
        case R5900Opcode::DADD:
        case R5900Opcode::DADDU:
            out << indent << "ctx.set_gpr_u64(" << rd << ", ctx.gpr_u64(" << rs << ") + ctx.gpr_u64(" << rt << "));\n";
            break;
        case R5900Opcode::DSUB:
        case R5900Opcode::DSUBU:
            out << indent << "ctx.set_gpr_u64(" << rd << ", ctx.gpr_u64(" << rs << ") - ctx.gpr_u64(" << rt << "));\n";
            break;
        case R5900Opcode::ADDI:
        case R5900Opcode::ADDIU:
            out << indent << "ctx.set_gpr_u32_sext(" << rt << ", ctx.gpr_u32(" << rs << ") + static_cast<u32>(" << simm << "));\n";
            break;
        case R5900Opcode::DADDI:
        case R5900Opcode::DADDIU:
            out << indent << "ctx.set_gpr_u64(" << rt << ", ctx.gpr_u64(" << rs << ") + static_cast<u64>(static_cast<s64>(" << simm << ")));\n";
            break;
        case R5900Opcode::AND:
            out << indent << "ctx.set_gpr_u64(" << rd << ", ctx.gpr_u64(" << rs << ") & ctx.gpr_u64(" << rt << "));\n";
            break;
        case R5900Opcode::OR:
            out << indent << "ctx.set_gpr_u64(" << rd << ", ctx.gpr_u64(" << rs << ") | ctx.gpr_u64(" << rt << "));\n";
            break;
        case R5900Opcode::XOR:
            out << indent << "ctx.set_gpr_u64(" << rd << ", ctx.gpr_u64(" << rs << ") ^ ctx.gpr_u64(" << rt << "));\n";
            break;
        case R5900Opcode::NOR:
            out << indent << "ctx.set_gpr_u64(" << rd << ", ~(ctx.gpr_u64(" << rs << ") | ctx.gpr_u64(" << rt << ")));\n";
            break;
        case R5900Opcode::ANDI:
            out << indent << "ctx.set_gpr_u64(" << rt << ", ctx.gpr_u64(" << rs << ") & " << uimm << "ULL);\n";
            break;
        case R5900Opcode::ORI:
            out << indent << "ctx.set_gpr_u64(" << rt << ", ctx.gpr_u64(" << rs << ") | " << uimm << "ULL);\n";
            break;
        case R5900Opcode::XORI:
            out << indent << "ctx.set_gpr_u64(" << rt << ", ctx.gpr_u64(" << rs << ") ^ " << uimm << "ULL);\n";
            break;
        case R5900Opcode::LUI:
            out << indent << "ctx.set_gpr_u32_sext(" << rt << ", static_cast<u32>(" << uimm << "u) << 16);\n";
            break;
        case R5900Opcode::SLT:
            out << indent << "ctx.set_gpr_u64(" << rd << ", (ctx.gpr_s64(" << rs << ") < ctx.gpr_s64(" << rt << ")) ? 1ULL : 0ULL);\n";
            break;
        case R5900Opcode::SLTU:
            out << indent << "ctx.set_gpr_u64(" << rd << ", (ctx.gpr_u64(" << rs << ") < ctx.gpr_u64(" << rt << ")) ? 1ULL : 0ULL);\n";
            break;
        case R5900Opcode::SLTI:
            out << indent << "ctx.set_gpr_u64(" << rt << ", (ctx.gpr_s64(" << rs << ") < static_cast<s64>(" << simm << ")) ? 1ULL : 0ULL);\n";
            break;
        case R5900Opcode::SLTIU:
            out << indent << "ctx.set_gpr_u64(" << rt << ", (ctx.gpr_u64(" << rs << ") < static_cast<u64>(static_cast<s64>(" << simm << "))) ? 1ULL : 0ULL);\n";
            break;
        case R5900Opcode::MOVZ:
            out << indent << "if (ctx.gpr_u64(" << rt << ") == 0) ctx.set_gpr_u64(" << rd << ", ctx.gpr_u64(" << rs << "));\n";
            break;
        case R5900Opcode::MOVN:
            out << indent << "if (ctx.gpr_u64(" << rt << ") != 0) ctx.set_gpr_u64(" << rd << ", ctx.gpr_u64(" << rs << "));\n";
            break;

        // --- Multiply / Divide / HI-LO / SA ---
        case R5900Opcode::MFHI:  out << indent << "ctx.set_gpr_u64(" << rd << ", ctx.hi);\n"; break;
        case R5900Opcode::MTHI:  out << indent << "ctx.hi = ctx.gpr_u64(" << rs << ");\n"; break;
        case R5900Opcode::MFLO:  out << indent << "ctx.set_gpr_u64(" << rd << ", ctx.lo);\n"; break;
        case R5900Opcode::MTLO:  out << indent << "ctx.lo = ctx.gpr_u64(" << rs << ");\n"; break;
        case R5900Opcode::MFHI1: out << indent << "ctx.set_gpr_u64(" << rd << ", ctx.hi1);\n"; break;
        case R5900Opcode::MTHI1: out << indent << "ctx.hi1 = ctx.gpr_u64(" << rs << ");\n"; break;
        case R5900Opcode::MFLO1: out << indent << "ctx.set_gpr_u64(" << rd << ", ctx.lo1);\n"; break;
        case R5900Opcode::MTLO1: out << indent << "ctx.lo1 = ctx.gpr_u64(" << rs << ");\n"; break;
        case R5900Opcode::MFSA:  out << indent << "ctx.set_gpr_u64(" << rd << ", ctx.sa);\n"; break;
        case R5900Opcode::MTSA:  out << indent << "ctx.sa = ctx.gpr_u32(" << rs << ") & 0xFu;\n"; break;
        case R5900Opcode::MTSAB: out << indent << "ctx.sa = (ctx.gpr_u32(" << rs << ") ^ " << uimm << "u) & 0xFu;\n"; break;
        case R5900Opcode::MTSAH: out << indent << "ctx.sa = ((ctx.gpr_u32(" << rs << ") ^ " << uimm << "u) & 0x7u) << 1;\n"; break;
        case R5900Opcode::MULT:   out << indent << "helpers::mult_op(ctx, " << rd << ", " << rs << ", " << rt << ", false);\n"; break;
        case R5900Opcode::MULTU:  out << indent << "helpers::multu_op(ctx, " << rd << ", " << rs << ", " << rt << ", false);\n"; break;
        case R5900Opcode::DIV:    out << indent << "helpers::div_op(ctx, " << rs << ", " << rt << ", false);\n"; break;
        case R5900Opcode::DIVU:   out << indent << "helpers::divu_op(ctx, " << rs << ", " << rt << ", false);\n"; break;
        case R5900Opcode::MULT1:  out << indent << "helpers::mult_op(ctx, " << rd << ", " << rs << ", " << rt << ", true);\n"; break;
        case R5900Opcode::MULTU1: out << indent << "helpers::multu_op(ctx, " << rd << ", " << rs << ", " << rt << ", true);\n"; break;
        case R5900Opcode::DIV1:   out << indent << "helpers::div_op(ctx, " << rs << ", " << rt << ", true);\n"; break;
        case R5900Opcode::DIVU1:  out << indent << "helpers::divu_op(ctx, " << rs << ", " << rt << ", true);\n"; break;
        case R5900Opcode::MADD:   out << indent << "helpers::madd_op(ctx, " << rd << ", " << rs << ", " << rt << ", false);\n"; break;
        case R5900Opcode::MADDU:  out << indent << "helpers::maddu_op(ctx, " << rd << ", " << rs << ", " << rt << ", false);\n"; break;
        case R5900Opcode::MADD1:  out << indent << "helpers::madd_op(ctx, " << rd << ", " << rs << ", " << rt << ", true);\n"; break;
        case R5900Opcode::MADDU1: out << indent << "helpers::maddu_op(ctx, " << rd << ", " << rs << ", " << rt << ", true);\n"; break;

        // --- Loads & Stores ---
        case R5900Opcode::LB:
            out << indent << "ctx.set_gpr_s64(" << rt << ", static_cast<s8>(ctx.mem->read8(ctx.gpr_u32(" << rs << ") + static_cast<u32>(" << simm << "))));\n";
            break;
        case R5900Opcode::LBU:
            out << indent << "ctx.set_gpr_u64(" << rt << ", ctx.mem->read8(ctx.gpr_u32(" << rs << ") + static_cast<u32>(" << simm << ")));\n";
            break;
        case R5900Opcode::LH:
            out << indent << "ctx.set_gpr_s64(" << rt << ", static_cast<s16>(ctx.mem->read16(ctx.gpr_u32(" << rs << ") + static_cast<u32>(" << simm << "))));\n";
            break;
        case R5900Opcode::LHU:
            out << indent << "ctx.set_gpr_u64(" << rt << ", ctx.mem->read16(ctx.gpr_u32(" << rs << ") + static_cast<u32>(" << simm << ")));\n";
            break;
        case R5900Opcode::LW:
            out << indent << "ctx.set_gpr_u32_sext(" << rt << ", ctx.mem->read32(ctx.gpr_u32(" << rs << ") + static_cast<u32>(" << simm << ")));\n";
            break;
        case R5900Opcode::LWU:
            out << indent << "ctx.set_gpr_u64(" << rt << ", ctx.mem->read32(ctx.gpr_u32(" << rs << ") + static_cast<u32>(" << simm << ")));\n";
            break;
        case R5900Opcode::LD:
            out << indent << "ctx.set_gpr_u64(" << rt << ", ctx.mem->read64(ctx.gpr_u32(" << rs << ") + static_cast<u32>(" << simm << ")));\n";
            break;
        case R5900Opcode::LQ:
            out << indent << "ctx.set_gpr_u128(" << rt << ", ctx.mem->read128(ctx.gpr_u32(" << rs << ") + static_cast<u32>(" << simm << ")));\n";
            break;
        case R5900Opcode::LWL:
            out << indent << "ctx.set_gpr_u32_sext(" << rt << ", helpers::lwl(*ctx.mem, ctx.gpr_u32(" << rs << ") + static_cast<u32>(" << simm << "), ctx.gpr_u32(" << rt << ")));\n";
            break;
        case R5900Opcode::LWR:
            out << indent << "ctx.set_gpr_u32_sext(" << rt << ", helpers::lwr(*ctx.mem, ctx.gpr_u32(" << rs << ") + static_cast<u32>(" << simm << "), ctx.gpr_u32(" << rt << ")));\n";
            break;
        case R5900Opcode::LDL:
            out << indent << "ctx.set_gpr_u64(" << rt << ", helpers::ldl(*ctx.mem, ctx.gpr_u32(" << rs << ") + static_cast<u32>(" << simm << "), ctx.gpr_u64(" << rt << ")));\n";
            break;
        case R5900Opcode::LDR:
            out << indent << "ctx.set_gpr_u64(" << rt << ", helpers::ldr(*ctx.mem, ctx.gpr_u32(" << rs << ") + static_cast<u32>(" << simm << "), ctx.gpr_u64(" << rt << ")));\n";
            break;
        case R5900Opcode::SB:
            out << indent << "ctx.mem->write8(ctx.gpr_u32(" << rs << ") + static_cast<u32>(" << simm << "), static_cast<u8>(ctx.gpr_u32(" << rt << ")));\n";
            break;
        case R5900Opcode::SH:
            out << indent << "ctx.mem->write16(ctx.gpr_u32(" << rs << ") + static_cast<u32>(" << simm << "), static_cast<u16>(ctx.gpr_u32(" << rt << ")));\n";
            break;
        case R5900Opcode::SW:
            out << indent << "ctx.mem->write32(ctx.gpr_u32(" << rs << ") + static_cast<u32>(" << simm << "), ctx.gpr_u32(" << rt << "));\n";
            break;
        case R5900Opcode::SD:
            out << indent << "ctx.mem->write64(ctx.gpr_u32(" << rs << ") + static_cast<u32>(" << simm << "), ctx.gpr_u64(" << rt << "));\n";
            break;
        case R5900Opcode::SQ:
            out << indent << "ctx.mem->write128(ctx.gpr_u32(" << rs << ") + static_cast<u32>(" << simm << "), ctx.gpr_u128(" << rt << "));\n";
            break;
        case R5900Opcode::SWL:
            out << indent << "helpers::swl(*ctx.mem, ctx.gpr_u32(" << rs << ") + static_cast<u32>(" << simm << "), ctx.gpr_u32(" << rt << "));\n";
            break;
        case R5900Opcode::SWR:
            out << indent << "helpers::swr(*ctx.mem, ctx.gpr_u32(" << rs << ") + static_cast<u32>(" << simm << "), ctx.gpr_u32(" << rt << "));\n";
            break;
        case R5900Opcode::SDL:
            out << indent << "helpers::sdl(*ctx.mem, ctx.gpr_u32(" << rs << ") + static_cast<u32>(" << simm << "), ctx.gpr_u64(" << rt << "));\n";
            break;
        case R5900Opcode::SDR:
            out << indent << "helpers::sdr(*ctx.mem, ctx.gpr_u32(" << rs << ") + static_cast<u32>(" << simm << "), ctx.gpr_u64(" << rt << "));\n";
            break;
        case R5900Opcode::LWC1:
            out << indent << "ctx.set_fpr_bits(" << ft << ", ctx.mem->read32(ctx.gpr_u32(" << rs << ") + static_cast<u32>(" << simm << ")));\n";
            break;
        case R5900Opcode::SWC1:
            out << indent << "ctx.mem->write32(ctx.gpr_u32(" << rs << ") + static_cast<u32>(" << simm << "), ctx.fpr_bits(" << ft << "));\n";
            break;
        case R5900Opcode::LQC2:
            out << indent << "{ u128 _q = ctx.mem->read128(ctx.gpr_u32(" << rs << ") + static_cast<u32>(" << simm << ")); if (" << ft << " != 0) std::memcpy(&ctx.vf[" << ft << "], &_q, 16); }\n";
            break;
        case R5900Opcode::SQC2:
            out << indent << "{ u128 _q{}; std::memcpy(&_q, &ctx.vf[" << ft << "], 16); ctx.mem->write128(ctx.gpr_u32(" << rs << ") + static_cast<u32>(" << simm << "), _q); }\n";
            break;
        case R5900Opcode::PREF:
        case R5900Opcode::CACHE:
        case R5900Opcode::SYNC:
            // Memory hint / cache maintenance barrier (coherent on host)
            break;

        // --- System / COP0 ---
        case R5900Opcode::SYSCALL:
            out << indent << "ctx.pc = " << hex32(ins.address) << "u; ctx.do_syscall(" << ((ins.raw >> 6) & 0xFFFFFu) << "u);\n";
            break;
        case R5900Opcode::BREAK:
            out << indent << "ctx.trap_unsupported(" << hex32(ins.address) << "u, " << hex32(ins.raw) << "u, \"BREAK exception\"); return;\n";
            break;
        case R5900Opcode::MFC0:
            out << indent << "ctx.set_gpr_u32_sext(" << rt << ", ctx.cop0[" << rd << "]);\n";
            break;
        case R5900Opcode::MTC0:
            out << indent << "ctx.cop0[" << rd << "] = ctx.gpr_u32(" << rt << ");\n";
            break;
        case R5900Opcode::EI:
            out << indent << "ctx.interrupts_enabled = true; ctx.cop0[12] |= 0x00010000u;\n";
            break;
        case R5900Opcode::DI:
            out << indent << "ctx.interrupts_enabled = false; ctx.cop0[12] &= ~0x00010000u;\n";
            break;
        case R5900Opcode::ERET:
            out << indent << "ctx.pc = ctx.cop0[14]; return;\n";
            break;
        case R5900Opcode::TLBR:
        case R5900Opcode::TLBWI:
        case R5900Opcode::TLBWR:
        case R5900Opcode::TLBP:
            break;
        case R5900Opcode::TGE:  out << indent << "if (ctx.gpr_s64(" << rs << ") >= ctx.gpr_s64(" << rt << ")) { ctx.trap_unsupported(" << hex32(ins.address) << "u, " << hex32(ins.raw) << "u, \"TGE trap\"); return; }\n"; break;
        case R5900Opcode::TGEU: out << indent << "if (ctx.gpr_u64(" << rs << ") >= ctx.gpr_u64(" << rt << ")) { ctx.trap_unsupported(" << hex32(ins.address) << "u, " << hex32(ins.raw) << "u, \"TGEU trap\"); return; }\n"; break;
        case R5900Opcode::TLT:  out << indent << "if (ctx.gpr_s64(" << rs << ") < ctx.gpr_s64(" << rt << ")) { ctx.trap_unsupported(" << hex32(ins.address) << "u, " << hex32(ins.raw) << "u, \"TLT trap\"); return; }\n"; break;
        case R5900Opcode::TLTU: out << indent << "if (ctx.gpr_u64(" << rs << ") < ctx.gpr_u64(" << rt << ")) { ctx.trap_unsupported(" << hex32(ins.address) << "u, " << hex32(ins.raw) << "u, \"TLTU trap\"); return; }\n"; break;
        case R5900Opcode::TEQ:  out << indent << "if (ctx.gpr_u64(" << rs << ") == ctx.gpr_u64(" << rt << ")) { ctx.trap_unsupported(" << hex32(ins.address) << "u, " << hex32(ins.raw) << "u, \"TEQ trap\"); return; }\n"; break;
        case R5900Opcode::TNE:  out << indent << "if (ctx.gpr_u64(" << rs << ") != ctx.gpr_u64(" << rt << ")) { ctx.trap_unsupported(" << hex32(ins.address) << "u, " << hex32(ins.raw) << "u, \"TNE trap\"); return; }\n"; break;
        case R5900Opcode::TGEI: out << indent << "if (ctx.gpr_s64(" << rs << ") >= " << simm << ") { ctx.trap_unsupported(" << hex32(ins.address) << "u, " << hex32(ins.raw) << "u, \"TGEI trap\"); return; }\n"; break;
        case R5900Opcode::TGEIU:out << indent << "if (ctx.gpr_u64(" << rs << ") >= static_cast<u64>(static_cast<s64>(" << simm << "))) { ctx.trap_unsupported(" << hex32(ins.address) << "u, " << hex32(ins.raw) << "u, \"TGEIU trap\"); return; }\n"; break;
        case R5900Opcode::TLTI: out << indent << "if (ctx.gpr_s64(" << rs << ") < " << simm << ") { ctx.trap_unsupported(" << hex32(ins.address) << "u, " << hex32(ins.raw) << "u, \"TLTI trap\"); return; }\n"; break;
        case R5900Opcode::TLTIU:out << indent << "if (ctx.gpr_u64(" << rs << ") < static_cast<u64>(static_cast<s64>(" << simm << "))) { ctx.trap_unsupported(" << hex32(ins.address) << "u, " << hex32(ins.raw) << "u, \"TLTIU trap\"); return; }\n"; break;
        case R5900Opcode::TEQI: out << indent << "if (ctx.gpr_s64(" << rs << ") == " << simm << ") { ctx.trap_unsupported(" << hex32(ins.address) << "u, " << hex32(ins.raw) << "u, \"TEQI trap\"); return; }\n"; break;
        case R5900Opcode::TNEI: out << indent << "if (ctx.gpr_s64(" << rs << ") != " << simm << ") { ctx.trap_unsupported(" << hex32(ins.address) << "u, " << hex32(ins.raw) << "u, \"TNEI trap\"); return; }\n"; break;

        // --- COP1 (FPU Single Precision) ---
        case R5900Opcode::MFC1:    out << indent << "ctx.set_gpr_u32_sext(" << rt << ", ctx.fpr_bits(" << fs << "));\n"; break;
        case R5900Opcode::MTC1:    out << indent << "ctx.set_fpr_bits(" << fs << ", ctx.gpr_u32(" << rt << "));\n"; break;
        case R5900Opcode::CFC1:    out << indent << "ctx.set_gpr_u32_sext(" << rt << ", (" << fs << " == 0) ? ctx.fcr0 : ctx.fcr31);\n"; break;
        case R5900Opcode::CTC1:    out << indent << "if (" << fs << " == 31) ctx.fcr31 = ctx.gpr_u32(" << rt << ");\n"; break;
        case R5900Opcode::ADD_S:   out << indent << "ctx.fpr[" << fd << "] = ctx.fpr[" << fs << "] + ctx.fpr[" << ft << "];\n"; break;
        case R5900Opcode::SUB_S:   out << indent << "ctx.fpr[" << fd << "] = ctx.fpr[" << fs << "] - ctx.fpr[" << ft << "];\n"; break;
        case R5900Opcode::MUL_S:   out << indent << "ctx.fpr[" << fd << "] = ctx.fpr[" << fs << "] * ctx.fpr[" << ft << "];\n"; break;
        case R5900Opcode::DIV_S:   out << indent << "ctx.fpr[" << fd << "] = ctx.fpr[" << fs << "] / ctx.fpr[" << ft << "];\n"; break;
        case R5900Opcode::SQRT_S:  out << indent << "ctx.fpr[" << fd << "] = std::sqrt(std::fabs(ctx.fpr[" << ft << "]));\n"; break;
        case R5900Opcode::RSQRT_S: out << indent << "ctx.fpr[" << fd << "] = ctx.fpr[" << fs << "] / std::sqrt(std::fabs(ctx.fpr[" << ft << "]));\n"; break;
        case R5900Opcode::ABS_S:   out << indent << "ctx.fpr[" << fd << "] = std::fabs(ctx.fpr[" << fs << "]);\n"; break;
        case R5900Opcode::MOV_S:   out << indent << "ctx.fpr[" << fd << "] = ctx.fpr[" << fs << "];\n"; break;
        case R5900Opcode::NEG_S:   out << indent << "ctx.fpr[" << fd << "] = -ctx.fpr[" << fs << "];\n"; break;
        case R5900Opcode::ADDA_S:  out << indent << "ctx.fpu_acc = ctx.fpr[" << fs << "] + ctx.fpr[" << ft << "];\n"; break;
        case R5900Opcode::SUBA_S:  out << indent << "ctx.fpu_acc = ctx.fpr[" << fs << "] - ctx.fpr[" << ft << "];\n"; break;
        case R5900Opcode::MULA_S:  out << indent << "ctx.fpu_acc = ctx.fpr[" << fs << "] * ctx.fpr[" << ft << "];\n"; break;
        case R5900Opcode::MADD_S:  out << indent << "ctx.fpr[" << fd << "] = ctx.fpu_acc + ctx.fpr[" << fs << "] * ctx.fpr[" << ft << "];\n"; break;
        case R5900Opcode::MSUB_S:  out << indent << "ctx.fpr[" << fd << "] = ctx.fpu_acc - ctx.fpr[" << fs << "] * ctx.fpr[" << ft << "];\n"; break;
        case R5900Opcode::MADDA_S: out << indent << "ctx.fpu_acc = ctx.fpu_acc + ctx.fpr[" << fs << "] * ctx.fpr[" << ft << "];\n"; break;
        case R5900Opcode::MSUBA_S: out << indent << "ctx.fpu_acc = ctx.fpu_acc - ctx.fpr[" << fs << "] * ctx.fpr[" << ft << "];\n"; break;
        case R5900Opcode::CVT_W_S: out << indent << "ctx.set_fpr_bits(" << fd << ", static_cast<u32>(static_cast<s32>(ctx.fpr[" << fs << "])));\n"; break;
        case R5900Opcode::CVT_S_W: out << indent << "ctx.fpr[" << fd << "] = static_cast<f32>(static_cast<s32>(ctx.fpr_bits(" << fs << ")));\n"; break;
        case R5900Opcode::MAX_S:   out << indent << "ctx.fpr[" << fd << "] = std::max(ctx.fpr[" << fs << "], ctx.fpr[" << ft << "]);\n"; break;
        case R5900Opcode::MIN_S:   out << indent << "ctx.fpr[" << fd << "] = std::min(ctx.fpr[" << fs << "], ctx.fpr[" << ft << "]);\n"; break;
        case R5900Opcode::C_F_S:   out << indent << "ctx.set_fpu_cond(false);\n"; break;
        case R5900Opcode::C_EQ_S:  out << indent << "ctx.set_fpu_cond(ctx.fpr[" << fs << "] == ctx.fpr[" << ft << "]);\n"; break;
        case R5900Opcode::C_LT_S:  out << indent << "ctx.set_fpu_cond(ctx.fpr[" << fs << "] < ctx.fpr[" << ft << "]);\n"; break;
        case R5900Opcode::C_LE_S:  out << indent << "ctx.set_fpu_cond(ctx.fpr[" << fs << "] <= ctx.fpr[" << ft << "]);\n"; break;

        // --- MMI 128-bit SIMD ---
        case R5900Opcode::PAND:    out << indent << "{ auto a=ctx.gpr_u128(" << rs << "), b=ctx.gpr_u128(" << rt << "); ctx.set_gpr_u128(" << rd << ", u128(a.lo & b.lo, a.hi & b.hi)); }\n"; break;
        case R5900Opcode::POR:     out << indent << "{ auto a=ctx.gpr_u128(" << rs << "), b=ctx.gpr_u128(" << rt << "); ctx.set_gpr_u128(" << rd << ", u128(a.lo | b.lo, a.hi | b.hi)); }\n"; break;
        case R5900Opcode::PXOR:    out << indent << "{ auto a=ctx.gpr_u128(" << rs << "), b=ctx.gpr_u128(" << rt << "); ctx.set_gpr_u128(" << rd << ", u128(a.lo ^ b.lo, a.hi ^ b.hi)); }\n"; break;
        case R5900Opcode::PNOR:    out << indent << "{ auto a=ctx.gpr_u128(" << rs << "), b=ctx.gpr_u128(" << rt << "); ctx.set_gpr_u128(" << rd << ", u128(~(a.lo | b.lo), ~(a.hi | b.hi))); }\n"; break;
        case R5900Opcode::PADDW:   out << indent << "ctx.set_gpr_u128(" << rd << ", helpers::mmi_paddw(ctx.gpr_u128(" << rs << "), ctx.gpr_u128(" << rt << ")));\n"; break;
        case R5900Opcode::PSUBW:   out << indent << "ctx.set_gpr_u128(" << rd << ", helpers::mmi_psubw(ctx.gpr_u128(" << rs << "), ctx.gpr_u128(" << rt << ")));\n"; break;
        case R5900Opcode::PCGTW:   out << indent << "ctx.set_gpr_u128(" << rd << ", helpers::mmi_pcgtw(ctx.gpr_u128(" << rs << "), ctx.gpr_u128(" << rt << ")));\n"; break;
        case R5900Opcode::PCEQW:   out << indent << "ctx.set_gpr_u128(" << rd << ", helpers::mmi_pceqw(ctx.gpr_u128(" << rs << "), ctx.gpr_u128(" << rt << ")));\n"; break;
        case R5900Opcode::PMAXW:   out << indent << "ctx.set_gpr_u128(" << rd << ", helpers::mmi_pmaxw(ctx.gpr_u128(" << rs << "), ctx.gpr_u128(" << rt << ")));\n"; break;
        case R5900Opcode::PMINW:   out << indent << "ctx.set_gpr_u128(" << rd << ", helpers::mmi_pminw(ctx.gpr_u128(" << rs << "), ctx.gpr_u128(" << rt << ")));\n"; break;
        case R5900Opcode::PADDSW:  out << indent << "ctx.set_gpr_u128(" << rd << ", helpers::mmi_paddsw(ctx.gpr_u128(" << rs << "), ctx.gpr_u128(" << rt << ")));\n"; break;
        case R5900Opcode::PSUBSW:  out << indent << "ctx.set_gpr_u128(" << rd << ", helpers::mmi_psubsw(ctx.gpr_u128(" << rs << "), ctx.gpr_u128(" << rt << ")));\n"; break;
        case R5900Opcode::PADDUW:  out << indent << "ctx.set_gpr_u128(" << rd << ", helpers::mmi_padduw(ctx.gpr_u128(" << rs << "), ctx.gpr_u128(" << rt << ")));\n"; break;
        case R5900Opcode::PSUBUW:  out << indent << "ctx.set_gpr_u128(" << rd << ", helpers::mmi_psubuw(ctx.gpr_u128(" << rs << "), ctx.gpr_u128(" << rt << ")));\n"; break;
        case R5900Opcode::PABSW:   out << indent << "ctx.set_gpr_u128(" << rd << ", helpers::mmi_pabsw(ctx.gpr_u128(" << rt << ")));\n"; break;
        case R5900Opcode::PADDH:   out << indent << "ctx.set_gpr_u128(" << rd << ", helpers::mmi_paddh(ctx.gpr_u128(" << rs << "), ctx.gpr_u128(" << rt << ")));\n"; break;
        case R5900Opcode::PSUBH:   out << indent << "ctx.set_gpr_u128(" << rd << ", helpers::mmi_psubh(ctx.gpr_u128(" << rs << "), ctx.gpr_u128(" << rt << ")));\n"; break;
        case R5900Opcode::PCGTH:   out << indent << "ctx.set_gpr_u128(" << rd << ", helpers::mmi_pcgth(ctx.gpr_u128(" << rs << "), ctx.gpr_u128(" << rt << ")));\n"; break;
        case R5900Opcode::PCEQH:   out << indent << "ctx.set_gpr_u128(" << rd << ", helpers::mmi_pceqh(ctx.gpr_u128(" << rs << "), ctx.gpr_u128(" << rt << ")));\n"; break;
        case R5900Opcode::PMAXH:   out << indent << "ctx.set_gpr_u128(" << rd << ", helpers::mmi_pmaxh(ctx.gpr_u128(" << rs << "), ctx.gpr_u128(" << rt << ")));\n"; break;
        case R5900Opcode::PMINH:   out << indent << "ctx.set_gpr_u128(" << rd << ", helpers::mmi_pminh(ctx.gpr_u128(" << rs << "), ctx.gpr_u128(" << rt << ")));\n"; break;
        case R5900Opcode::PADDSH:  out << indent << "ctx.set_gpr_u128(" << rd << ", helpers::mmi_paddsh(ctx.gpr_u128(" << rs << "), ctx.gpr_u128(" << rt << ")));\n"; break;
        case R5900Opcode::PSUBSH:  out << indent << "ctx.set_gpr_u128(" << rd << ", helpers::mmi_psubsh(ctx.gpr_u128(" << rs << "), ctx.gpr_u128(" << rt << ")));\n"; break;
        case R5900Opcode::PADDUH:  out << indent << "ctx.set_gpr_u128(" << rd << ", helpers::mmi_padduh(ctx.gpr_u128(" << rs << "), ctx.gpr_u128(" << rt << ")));\n"; break;
        case R5900Opcode::PSUBUH:  out << indent << "ctx.set_gpr_u128(" << rd << ", helpers::mmi_psubuh(ctx.gpr_u128(" << rs << "), ctx.gpr_u128(" << rt << ")));\n"; break;
        case R5900Opcode::PABSH:   out << indent << "ctx.set_gpr_u128(" << rd << ", helpers::mmi_pabsh(ctx.gpr_u128(" << rt << ")));\n"; break;
        case R5900Opcode::PADDB:   out << indent << "ctx.set_gpr_u128(" << rd << ", helpers::mmi_paddb(ctx.gpr_u128(" << rs << "), ctx.gpr_u128(" << rt << ")));\n"; break;
        case R5900Opcode::PSUBB:   out << indent << "ctx.set_gpr_u128(" << rd << ", helpers::mmi_psubb(ctx.gpr_u128(" << rs << "), ctx.gpr_u128(" << rt << ")));\n"; break;
        case R5900Opcode::PCGTB:   out << indent << "ctx.set_gpr_u128(" << rd << ", helpers::mmi_pcgtb(ctx.gpr_u128(" << rs << "), ctx.gpr_u128(" << rt << ")));\n"; break;
        case R5900Opcode::PCEQB:   out << indent << "ctx.set_gpr_u128(" << rd << ", helpers::mmi_pceqb(ctx.gpr_u128(" << rs << "), ctx.gpr_u128(" << rt << ")));\n"; break;
        case R5900Opcode::PADDSB:  out << indent << "ctx.set_gpr_u128(" << rd << ", helpers::mmi_paddsb(ctx.gpr_u128(" << rs << "), ctx.gpr_u128(" << rt << ")));\n"; break;
        case R5900Opcode::PSUBSB:  out << indent << "ctx.set_gpr_u128(" << rd << ", helpers::mmi_psubsb(ctx.gpr_u128(" << rs << "), ctx.gpr_u128(" << rt << ")));\n"; break;
        case R5900Opcode::PADDUB:  out << indent << "ctx.set_gpr_u128(" << rd << ", helpers::mmi_paddub(ctx.gpr_u128(" << rs << "), ctx.gpr_u128(" << rt << ")));\n"; break;
        case R5900Opcode::PSUBUB:  out << indent << "ctx.set_gpr_u128(" << rd << ", helpers::mmi_psubub(ctx.gpr_u128(" << rs << "), ctx.gpr_u128(" << rt << ")));\n"; break;
        case R5900Opcode::PEXTLW:  out << indent << "ctx.set_gpr_u128(" << rd << ", helpers::mmi_pextlw(ctx.gpr_u128(" << rs << "), ctx.gpr_u128(" << rt << ")));\n"; break;
        case R5900Opcode::PEXTUW:  out << indent << "ctx.set_gpr_u128(" << rd << ", helpers::mmi_pextuw(ctx.gpr_u128(" << rs << "), ctx.gpr_u128(" << rt << ")));\n"; break;
        case R5900Opcode::PPACW:   out << indent << "ctx.set_gpr_u128(" << rd << ", helpers::mmi_ppacw(ctx.gpr_u128(" << rs << "), ctx.gpr_u128(" << rt << ")));\n"; break;
        case R5900Opcode::PEXTLH:  out << indent << "ctx.set_gpr_u128(" << rd << ", helpers::mmi_pextlh(ctx.gpr_u128(" << rs << "), ctx.gpr_u128(" << rt << ")));\n"; break;
        case R5900Opcode::PEXTUH:  out << indent << "ctx.set_gpr_u128(" << rd << ", helpers::mmi_pextuh(ctx.gpr_u128(" << rs << "), ctx.gpr_u128(" << rt << ")));\n"; break;
        case R5900Opcode::PPACH:   out << indent << "ctx.set_gpr_u128(" << rd << ", helpers::mmi_ppach(ctx.gpr_u128(" << rs << "), ctx.gpr_u128(" << rt << ")));\n"; break;
        case R5900Opcode::PEXTLB:  out << indent << "ctx.set_gpr_u128(" << rd << ", helpers::mmi_pextlb(ctx.gpr_u128(" << rs << "), ctx.gpr_u128(" << rt << ")));\n"; break;
        case R5900Opcode::PEXTUB:  out << indent << "ctx.set_gpr_u128(" << rd << ", helpers::mmi_pextub(ctx.gpr_u128(" << rs << "), ctx.gpr_u128(" << rt << ")));\n"; break;
        case R5900Opcode::PPACB:   out << indent << "ctx.set_gpr_u128(" << rd << ", helpers::mmi_ppacb(ctx.gpr_u128(" << rs << "), ctx.gpr_u128(" << rt << ")));\n"; break;
        case R5900Opcode::PCPYLD:  out << indent << "ctx.set_gpr_u128(" << rd << ", helpers::mmi_pcpyld(ctx.gpr_u128(" << rs << "), ctx.gpr_u128(" << rt << ")));\n"; break;
        case R5900Opcode::PCPYUD:  out << indent << "ctx.set_gpr_u128(" << rd << ", helpers::mmi_pcpyud(ctx.gpr_u128(" << rs << "), ctx.gpr_u128(" << rt << ")));\n"; break;
        case R5900Opcode::PCPYH:   out << indent << "ctx.set_gpr_u128(" << rd << ", helpers::mmi_pcpyh(ctx.gpr_u128(" << rt << ")));\n"; break;
        case R5900Opcode::PROT3W:  out << indent << "ctx.set_gpr_u128(" << rd << ", helpers::mmi_prot3w(ctx.gpr_u128(" << rt << ")));\n"; break;
        case R5900Opcode::PEXEW:   out << indent << "ctx.set_gpr_u128(" << rd << ", helpers::mmi_pexew(ctx.gpr_u128(" << rt << ")));\n"; break;
        case R5900Opcode::PEXCW:   out << indent << "ctx.set_gpr_u128(" << rd << ", helpers::mmi_pexcw(ctx.gpr_u128(" << rt << ")));\n"; break;
        case R5900Opcode::PEXEH:   out << indent << "ctx.set_gpr_u128(" << rd << ", helpers::mmi_pexeh(ctx.gpr_u128(" << rt << ")));\n"; break;
        case R5900Opcode::PEXCH:   out << indent << "ctx.set_gpr_u128(" << rd << ", helpers::mmi_pexch(ctx.gpr_u128(" << rt << ")));\n"; break;
        case R5900Opcode::PREVH:   out << indent << "ctx.set_gpr_u128(" << rd << ", helpers::mmi_prevh(ctx.gpr_u128(" << rt << ")));\n"; break;
        case R5900Opcode::PINTH:   out << indent << "ctx.set_gpr_u128(" << rd << ", helpers::mmi_pinth(ctx.gpr_u128(" << rs << "), ctx.gpr_u128(" << rt << ")));\n"; break;
        case R5900Opcode::PINTEH:  out << indent << "ctx.set_gpr_u128(" << rd << ", helpers::mmi_pinteh(ctx.gpr_u128(" << rs << "), ctx.gpr_u128(" << rt << ")));\n"; break;
        case R5900Opcode::PADSBH:  out << indent << "ctx.set_gpr_u128(" << rd << ", helpers::mmi_padsbh(ctx.gpr_u128(" << rs << "), ctx.gpr_u128(" << rt << ")));\n"; break;
        case R5900Opcode::PEXT5:   out << indent << "ctx.set_gpr_u128(" << rd << ", helpers::mmi_pext5(ctx.gpr_u128(" << rt << ")));\n"; break;
        case R5900Opcode::PPAC5:   out << indent << "ctx.set_gpr_u128(" << rd << ", helpers::mmi_ppac5(ctx.gpr_u128(" << rt << ")));\n"; break;
        case R5900Opcode::QFSRV:   out << indent << "ctx.set_gpr_u128(" << rd << ", helpers::mmi_qfsrv(ctx.gpr_u128(" << rs << "), ctx.gpr_u128(" << rt << "), ctx.sa));\n"; break;
        case R5900Opcode::PLZCW:   out << indent << "{ auto a=ctx.gpr_u128(" << rs << "); u128 _r{}; _r.set_uw(0, helpers::plzcw_word(a.uw(0))); _r.set_uw(1, helpers::plzcw_word(a.uw(1))); ctx.set_gpr_u64(" << rd << ", _r.lo); }\n"; break;
        case R5900Opcode::PMFHI:   out << indent << "ctx.set_gpr_u128(" << rd << ", u128(ctx.hi, ctx.hi1));\n"; break;
        case R5900Opcode::PMFLO:   out << indent << "ctx.set_gpr_u128(" << rd << ", u128(ctx.lo, ctx.lo1));\n"; break;
        case R5900Opcode::PMTHI:   out << indent << "{ auto a=ctx.gpr_u128(" << rs << "); ctx.hi = a.lo; ctx.hi1 = a.hi; }\n"; break;
        case R5900Opcode::PMTLO:   out << indent << "{ auto a=ctx.gpr_u128(" << rs << "); ctx.lo = a.lo; ctx.lo1 = a.hi; }\n"; break;
        case R5900Opcode::PMFHL:   out << indent << "ctx.set_gpr_u128(" << rd << ", helpers::mmi_pmfhl(ctx, " << sa << "u));\n"; break;
        case R5900Opcode::PMTHL:   out << indent << "{ auto a=ctx.gpr_u128(" << rs << "); ctx.lo = static_cast<s64>(a.sw(0)); ctx.hi = static_cast<s64>(a.sw(1)); ctx.lo1 = static_cast<s64>(a.sw(2)); ctx.hi1 = static_cast<s64>(a.sw(3)); }\n"; break;
        case R5900Opcode::PMULTW:  out << indent << "helpers::mmi_pmultw(ctx, " << rd << ", ctx.gpr_u128(" << rs << "), ctx.gpr_u128(" << rt << "));\n"; break;
        case R5900Opcode::PMULTUW: out << indent << "helpers::mmi_pmultuw(ctx, " << rd << ", ctx.gpr_u128(" << rs << "), ctx.gpr_u128(" << rt << "));\n"; break;
        case R5900Opcode::PMULTH:  out << indent << "helpers::mmi_pmulth(ctx, " << rd << ", ctx.gpr_u128(" << rs << "), ctx.gpr_u128(" << rt << "));\n"; break;
        case R5900Opcode::PMADDH:  out << indent << "helpers::mmi_pmaddh(ctx, " << rd << ", ctx.gpr_u128(" << rs << "), ctx.gpr_u128(" << rt << "));\n"; break;
        case R5900Opcode::PSLLH:   out << indent << "ctx.set_gpr_u128(" << rd << ", helpers::mmi_psllh(ctx.gpr_u128(" << rt << "), " << sa << "u));\n"; break;
        case R5900Opcode::PSRLH:   out << indent << "ctx.set_gpr_u128(" << rd << ", helpers::mmi_psrlh(ctx.gpr_u128(" << rt << "), " << sa << "u));\n"; break;
        case R5900Opcode::PSRAH:   out << indent << "ctx.set_gpr_u128(" << rd << ", helpers::mmi_psrah(ctx.gpr_u128(" << rt << "), " << sa << "u));\n"; break;
        case R5900Opcode::PSLLW:   out << indent << "ctx.set_gpr_u128(" << rd << ", helpers::mmi_psllw(ctx.gpr_u128(" << rt << "), " << sa << "u));\n"; break;
        case R5900Opcode::PSRLW:   out << indent << "ctx.set_gpr_u128(" << rd << ", helpers::mmi_psrlw(ctx.gpr_u128(" << rt << "), " << sa << "u));\n"; break;
        case R5900Opcode::PSRAW:   out << indent << "ctx.set_gpr_u128(" << rd << ", helpers::mmi_psraw(ctx.gpr_u128(" << rt << "), " << sa << "u));\n"; break;

        // --- COP2 / VU0 Macro Mode ---
        case R5900Opcode::QMFC2:
            out << indent << "{ u128 _q{}; std::memcpy(&_q, &ctx.vf[" << rd << "], 16); ctx.set_gpr_u128(" << rt << ", _q); }\n";
            break;
        case R5900Opcode::QMTC2:
            out << indent << "{ u128 _q = ctx.gpr_u128(" << rt << "); if (" << rd << " != 0) std::memcpy(&ctx.vf[" << rd << "], &_q, 16); }\n";
            break;
        case R5900Opcode::CFC2:
            out << indent << "ctx.set_gpr_u32_sext(" << rt << ", ctx.vi[" << rd << "]);\n";
            break;
        case R5900Opcode::CTC2:
            out << indent << "if (" << rd << " != 0) ctx.vi[" << rd << "] = ctx.gpr_u32(" << rt << ");\n";
            break;
        case R5900Opcode::VU0_MACRO_OP:
            out << indent << "ctx.do_vu0_macro(" << hex32(ins.raw) << "u);\n";
            break;

        default: {
            std::string reason = (ins.op == R5900Opcode::Unknown)
                ? ins.unknown_reason
                : (std::string("untranslated opcode: ") + R5900Decoder::opcode_name(ins.op));
            out << indent << "ctx.trap_unsupported(" << hex32(ins.address) << "u, " << hex32(ins.raw)
                << "u, \"" << reason << "\"); return;\n";
            stats.instructions_trapped_unsupported++;
            stats.unsupported_reasons[reason]++;
            return;
        }
    }

    stats.instructions_translated++;
}

std::string branch_condition_expr(const DecodedInstruction& ins) {
    auto rs = static_cast<unsigned>(ins.rs);
    auto rt = static_cast<unsigned>(ins.rt);
    switch (ins.op) {
        case R5900Opcode::BEQ:
        case R5900Opcode::BEQL:    return "ctx.gpr_u64(" + std::to_string(rs) + ") == ctx.gpr_u64(" + std::to_string(rt) + ")";
        case R5900Opcode::BNE:
        case R5900Opcode::BNEL:    return "ctx.gpr_u64(" + std::to_string(rs) + ") != ctx.gpr_u64(" + std::to_string(rt) + ")";
        case R5900Opcode::BLEZ:
        case R5900Opcode::BLEZL:   return "ctx.gpr_s64(" + std::to_string(rs) + ") <= 0";
        case R5900Opcode::BGTZ:
        case R5900Opcode::BGTZL:   return "ctx.gpr_s64(" + std::to_string(rs) + ") > 0";
        case R5900Opcode::BLTZ:
        case R5900Opcode::BLTZL:
        case R5900Opcode::BLTZAL:
        case R5900Opcode::BLTZALL: return "ctx.gpr_s64(" + std::to_string(rs) + ") < 0";
        case R5900Opcode::BGEZ:
        case R5900Opcode::BGEZL:
        case R5900Opcode::BGEZAL:
        case R5900Opcode::BGEZALL: return "ctx.gpr_s64(" + std::to_string(rs) + ") >= 0";
        case R5900Opcode::BC1F:
        case R5900Opcode::BC1FL:   return "!ctx.fpu_cond()";
        case R5900Opcode::BC1T:
        case R5900Opcode::BC1TL:   return "ctx.fpu_cond()";
        case R5900Opcode::BC2F:
        case R5900Opcode::BC2FL:   return "(ctx.vi[16] & 0x100u) == 0";
        case R5900Opcode::BC2T:
        case R5900Opcode::BC2TL:   return "(ctx.vi[16] & 0x100u) != 0";
        case R5900Opcode::BC0F:
        case R5900Opcode::BC0FL:   return "true";
        case R5900Opcode::BC0T:
        case R5900Opcode::BC0TL:   return "false";
        default:                   return "false";
    }
}

} // namespace

std::string CppCodeGenerator::sanitize_identifier(const std::string& name, u32 addr) {
    std::string out;
    out.reserve(name.size() + 16);
    for (char c : name) {
        if (std::isalnum(static_cast<unsigned char>(c)) || c == '_') {
            out.push_back(c);
        } else {
            out.push_back('_');
        }
    }
    if (out.empty() || std::isdigit(static_cast<unsigned char>(out[0]))) {
        out = "fn_" + out;
    }
    // Append address suffix so duplicate/stripped names never collide
    std::ostringstream oss;
    oss << out << "_" << std::hex << std::setw(8) << std::setfill('0') << addr;
    return oss.str();
}

std::string CppCodeGenerator::emit_function(
    const DiscoveredFunction& fn,
    const std::map<u32, std::string>& all_func_names,
    const CodegenOptions& opts,
    CodegenResult& stats
) {
    std::ostringstream out;
    std::string fn_ident = all_func_names.at(fn.start_addr);

    // Collect every label we need inside this function
    std::set<u32> needed_labels;
    for (std::size_t i = 0; i < fn.instructions.size(); ++i) {
        const auto& ins = fn.instructions[i];
        if ((ins.is_branch || ins.op == R5900Opcode::J) &&
            ins.target >= fn.start_addr && ins.target < fn.end_addr) {
            needed_labels.insert(ins.target);
        }
    }

    out << "void " << fn_ident << "(EEContext& ctx) {\n";
    out << "    (void)ctx;\n";

    const std::size_t n = fn.instructions.size();
    for (std::size_t i = 0; i < n; ++i) {
        const auto& ins = fn.instructions[i];

        if (needed_labels.count(ins.address)) {
            out << label_for(ins.address) << ":\n";
        }
        if (opts.emit_disassembly_comments) {
            out << "    // " << hex32(ins.address) << ": " << R5900Decoder::disassemble(ins) << "\n";
        }

        if (!ins.has_delay_slot) {
            emit_non_branch_instruction(out, ins, "    ", stats);
            continue;
        }

        // Handle branch / jump + delay slot at i + 1
        bool has_ds = (i + 1 < n);
        const DecodedInstruction* ds = has_ds ? &fn.instructions[i + 1] : nullptr;

        auto emit_target_transfer = [&](u32 target, const char* indent) {
            if (target >= fn.start_addr && target < fn.end_addr) {
                out << indent << "goto " << label_for(target) << ";\n";
            } else if (auto it = all_func_names.find(target); it != all_func_names.end()) {
                out << indent << it->second << "(ctx); return;\n";
            } else {
                out << indent << "ctx.dispatch(" << hex32(target) << "u); return;\n";
            }
        };

        if (ins.op == R5900Opcode::JR) {
            out << "    {\n";
            out << "        u32 _target = ctx.gpr_u32(" << static_cast<unsigned>(ins.rs) << ");\n";
            if (ds) emit_non_branch_instruction(out, *ds, "        ", stats);
            if (ins.rs == 31) {
                out << "        ctx.pc = _target;\n";
                out << "        return;\n";
            } else {
                out << "        ctx.dispatch(_target);\n";
                out << "        return;\n";
            }
            out << "    }\n";
            stats.instructions_translated++;
        } else if (ins.op == R5900Opcode::JALR) {
            out << "    {\n";
            out << "        u32 _target = ctx.gpr_u32(" << static_cast<unsigned>(ins.rs) << ");\n";
            out << "        ctx.set_gpr_u32_sext(" << static_cast<unsigned>(ins.rd) << ", " << hex32(ins.address + 8) << "u);\n";
            if (ds) emit_non_branch_instruction(out, *ds, "        ", stats);
            out << "        ctx.dispatch(_target);\n";
            out << "    }\n";
            stats.instructions_translated++;
        } else if (ins.op == R5900Opcode::JAL) {
            out << "    {\n";
            out << "        ctx.set_gpr_u32_sext(31, " << hex32(ins.address + 8) << "u);\n";
            if (ds) emit_non_branch_instruction(out, *ds, "        ", stats);
            if (auto it = all_func_names.find(ins.target); it != all_func_names.end()) {
                out << "        " << it->second << "(ctx);\n";
            } else {
                out << "        ctx.dispatch(" << hex32(ins.target) << "u);\n";
            }
            out << "    }\n";
            stats.instructions_translated++;
        } else if (ins.op == R5900Opcode::J) {
            out << "    {\n";
            if (ds) emit_non_branch_instruction(out, *ds, "        ", stats);
            emit_target_transfer(ins.target, "        ");
            out << "    }\n";
            stats.instructions_translated++;
        } else if (ins.is_branch) {
            std::string cond = branch_condition_expr(ins);
            bool link = (ins.op == R5900Opcode::BLTZAL || ins.op == R5900Opcode::BGEZAL ||
                         ins.op == R5900Opcode::BLTZALL || ins.op == R5900Opcode::BGEZALL);
            out << "    {\n";
            out << "        bool _cond = (" << cond << ");\n";
            if (link) {
                out << "        ctx.set_gpr_u32_sext(31, " << hex32(ins.address + 8) << "u);\n";
            }
            if (ins.is_branch_likely) {
                out << "        if (_cond) {\n";
                if (ds) emit_non_branch_instruction(out, *ds, "            ", stats);
                if (link) {
                    if (auto it = all_func_names.find(ins.target); it != all_func_names.end()) {
                        out << "            " << it->second << "(ctx);\n";
                    } else {
                        out << "            ctx.dispatch(" << hex32(ins.target) << "u);\n";
                    }
                } else {
                    emit_target_transfer(ins.target, "            ");
                }
                out << "        }\n";
            } else {
                if (ds) emit_non_branch_instruction(out, *ds, "        ", stats);
                out << "        if (_cond) {\n";
                if (link) {
                    if (auto it = all_func_names.find(ins.target); it != all_func_names.end()) {
                        out << "            " << it->second << "(ctx);\n";
                    } else {
                        out << "            ctx.dispatch(" << hex32(ins.target) << "u);\n";
                    }
                } else {
                    emit_target_transfer(ins.target, "            ");
                }
                out << "        }\n";
            }
            out << "    }\n";
            stats.instructions_translated++;
        }

        // Advance past the delay slot; if another branch targets the delay slot directly,
        // emit an unreachable fallthrough jump around its standalone label.
        if (has_ds) {
            if (needed_labels.count(ds->address)) {
                u32 after_ds_addr = ds->address + 4;
                if (i + 2 < n) {
                    needed_labels.insert(after_ds_addr);
                    out << "    goto " << label_for(after_ds_addr) << ";\n";
                } else {
                    out << "    return;\n";
                }
                out << label_for(ds->address) << ":\n";
                emit_non_branch_instruction(out, *ds, "    ", stats);
            }
            ++i; // Delay slot already handled
        }
    }

    // If the last instruction in this partition does not unconditionally terminate control flow
    // (e.g., prologue heuristic split a function mid-stream or two entry points share a body),
    // fall through cleanly into the adjacent function at fn.end_addr.
    bool ends_unconditionally = false;
    if (n >= 2) {
        const auto& prev = fn.instructions[n - 2];
        if (prev.op == R5900Opcode::JR || prev.op == R5900Opcode::J) {
            ends_unconditionally = true;
        }
    }
    if (!ends_unconditionally) {
        if (auto it = all_func_names.find(fn.end_addr); it != all_func_names.end()) {
            out << "    " << it->second << "(ctx);\n";
        }
    }

    out << "}\n";
    stats.functions_emitted++;
    return out.str();
}

CodegenResult CppCodeGenerator::generate_project(
    const AnalysisReport& analysis,
    const std::string& output_dir,
    const CodegenOptions& opts
) {
    CodegenResult res;
    namespace fs = std::filesystem;

    std::error_code ec;
    fs::create_directories(output_dir, ec);
    if (ec) {
        res.error = "Failed to create output directory " + output_dir + ": " + ec.message();
        return res;
    }

    std::size_t emit_count = analysis.functions.size();
    if (opts.max_functions_to_emit > 0 && opts.max_functions_to_emit < emit_count) {
        emit_count = opts.max_functions_to_emit;
    }

    // Build deterministic unique C++ identifiers for every emitted function
    std::map<u32, std::string> func_names;
    for (std::size_t i = 0; i < emit_count; ++i) {
        const auto& fn = analysis.functions[i];
        func_names[fn.start_addr] = sanitize_identifier(fn.name, fn.start_addr);
    }

    // 1. Emit master header: recomp_module.h
    fs::path hdr_path = fs::path(output_dir) / "recomp_module.h";
    {
        std::ofstream hdr(hdr_path);
        if (!hdr) {
            res.error = "Failed to write " + hdr_path.string();
            return res;
        }
        hdr << "#pragma once\n";
        hdr << "#include \"" << opts.runtime_include_prefix << "EEContext.h\"\n";
        hdr << "#include \"" << opts.runtime_include_prefix << "R5900Helpers.h\"\n";
        hdr << "#include <unordered_map>\n\n";
        hdr << "namespace " << opts.namespace_name << " {\n";
        hdr << "using namespace tf::ps2;\n\n";
        hdr << "constexpr u32 kEntryPoint = " << hex32(analysis.entry_point) << "u;\n";
        hdr << "void register_recompiled_functions(std::unordered_map<u32, RecompiledFn>& table);\n\n";
        for (const auto& [addr, ident] : func_names) {
            hdr << "void " << ident << "(EEContext& ctx);\n";
        }
        hdr << "\n} // namespace " << opts.namespace_name << "\n";
    }
    res.generated_files.push_back(hdr_path.string());

    // 2. Emit sharded .cpp translation units
    std::size_t shard_size = std::max<std::size_t>(1, opts.max_functions_per_shard);
    std::size_t shard_idx = 0;
    for (std::size_t start = 0; start < emit_count; start += shard_size, ++shard_idx) {
        std::size_t end = std::min(emit_count, start + shard_size);
        fs::path shard_path = fs::path(output_dir) / ("recomp_shard_" + std::to_string(shard_idx) + ".cpp");
        std::ofstream sf(shard_path);
        if (!sf) {
            res.error = "Failed to write shard file " + shard_path.string();
            return res;
        }
        sf << "#include \"recomp_module.h\"\n";
        sf << "#include <algorithm>\n#include <cmath>\n#include <cstring>\n\n";
        sf << "namespace " << opts.namespace_name << " {\n\n";
        for (std::size_t i = start; i < end; ++i) {
            sf << emit_function(analysis.functions[i], func_names, opts, res) << "\n";
        }
        sf << "} // namespace " << opts.namespace_name << "\n";
        res.generated_files.push_back(shard_path.string());
    }

    // 3. Emit dispatch table registration unit: recomp_dispatch.cpp
    fs::path disp_path = fs::path(output_dir) / "recomp_dispatch.cpp";
    {
        std::ofstream df(disp_path);
        if (!df) {
            res.error = "Failed to write " + disp_path.string();
            return res;
        }
        df << "#include \"recomp_module.h\"\n\n";
        df << "namespace " << opts.namespace_name << " {\n\n";
        df << "void register_recompiled_functions(std::unordered_map<u32, RecompiledFn>& table) {\n";
        df << "    table.reserve(" << func_names.size() << ");\n";
        for (const auto& [addr, ident] : func_names) {
            df << "    table[" << hex32(addr) << "u] = &" << ident << ";\n";
        }
        df << "}\n\n";
        df << "} // namespace " << opts.namespace_name << "\n";
    }
    res.generated_files.push_back(disp_path.string());

    res.success = true;
    return res;
}

} // namespace tf::ps2
