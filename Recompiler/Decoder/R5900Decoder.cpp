#include "R5900Decoder.h"
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <vector>

namespace tf::ps2 {

namespace {

inline u32 branch_target(u32 addr, s16 simm) {
    return static_cast<u32>(static_cast<s64>(addr + 4) + (static_cast<s64>(simm) << 2));
}

inline u32 jump_target(u32 addr, u32 raw) {
    return ((addr + 4) & 0xF0000000u) | ((raw & 0x03FFFFFFu) << 2);
}

void mark_unknown(DecodedInstruction& ins, const std::string& reason) {
    ins.op = R5900Opcode::Unknown;
    ins.unknown_reason = reason;
}

} // namespace

DecodedInstruction R5900Decoder::decode(u32 address, u32 word) {
    DecodedInstruction ins;
    ins.address = address;
    ins.raw = word;

    const u8 primary = static_cast<u8>((word >> 26) & 0x3Fu);
    ins.rs   = static_cast<u8>((word >> 21) & 0x1Fu);
    ins.rt   = static_cast<u8>((word >> 16) & 0x1Fu);
    ins.rd   = static_cast<u8>((word >> 11) & 0x1Fu);
    ins.sa   = static_cast<u8>((word >> 6) & 0x1Fu);
    ins.fs   = ins.rd;
    ins.ft   = ins.rt;
    ins.fd   = ins.sa;
    ins.uimm = static_cast<u16>(word & 0xFFFFu);
    ins.simm = static_cast<s16>(ins.uimm);
    const u8 funct = static_cast<u8>(word & 0x3Fu);

    switch (primary) {
        case 0x00: { // SPECIAL
            switch (funct) {
                case 0x00: ins.op = R5900Opcode::SLL; break;
                case 0x02: ins.op = R5900Opcode::SRL; break;
                case 0x03: ins.op = R5900Opcode::SRA; break;
                case 0x04: ins.op = R5900Opcode::SLLV; break;
                case 0x06: ins.op = R5900Opcode::SRLV; break;
                case 0x07: ins.op = R5900Opcode::SRAV; break;
                case 0x08:
                    ins.op = R5900Opcode::JR;
                    ins.is_jump = true;
                    ins.is_indirect_jump = true;
                    ins.has_delay_slot = true;
                    ins.is_return = (ins.rs == 31);
                    break;
                case 0x09:
                    ins.op = R5900Opcode::JALR;
                    ins.is_jump = true;
                    ins.is_call = true;
                    ins.is_indirect_jump = true;
                    ins.has_delay_slot = true;
                    break;
                case 0x0A: ins.op = R5900Opcode::MOVZ; break;
                case 0x0B: ins.op = R5900Opcode::MOVN; break;
                case 0x0C: ins.op = R5900Opcode::SYSCALL; break;
                case 0x0D: ins.op = R5900Opcode::BREAK; break;
                case 0x0F: ins.op = R5900Opcode::SYNC; break;
                case 0x10: ins.op = R5900Opcode::MFHI; break;
                case 0x11: ins.op = R5900Opcode::MTHI; break;
                case 0x12: ins.op = R5900Opcode::MFLO; break;
                case 0x13: ins.op = R5900Opcode::MTLO; break;
                case 0x14: ins.op = R5900Opcode::DSLLV; break;
                case 0x16: ins.op = R5900Opcode::DSRLV; break;
                case 0x17: ins.op = R5900Opcode::DSRAV; break;
                case 0x18: ins.op = R5900Opcode::MULT; break;
                case 0x19: ins.op = R5900Opcode::MULTU; break;
                case 0x1A: ins.op = R5900Opcode::DIV; break;
                case 0x1B: ins.op = R5900Opcode::DIVU; break;
                case 0x20: ins.op = R5900Opcode::ADD; break;
                case 0x21: ins.op = R5900Opcode::ADDU; break;
                case 0x22: ins.op = R5900Opcode::SUB; break;
                case 0x23: ins.op = R5900Opcode::SUBU; break;
                case 0x24: ins.op = R5900Opcode::AND; break;
                case 0x25: ins.op = R5900Opcode::OR; break;
                case 0x26: ins.op = R5900Opcode::XOR; break;
                case 0x27: ins.op = R5900Opcode::NOR; break;
                case 0x28: ins.op = R5900Opcode::MFSA; break;
                case 0x29: ins.op = R5900Opcode::MTSA; break;
                case 0x2A: ins.op = R5900Opcode::SLT; break;
                case 0x2B: ins.op = R5900Opcode::SLTU; break;
                case 0x2C: ins.op = R5900Opcode::DADD; break;
                case 0x2D: ins.op = R5900Opcode::DADDU; break;
                case 0x2E: ins.op = R5900Opcode::DSUB; break;
                case 0x2F: ins.op = R5900Opcode::DSUBU; break;
                case 0x30: ins.op = R5900Opcode::TGE; break;
                case 0x31: ins.op = R5900Opcode::TGEU; break;
                case 0x32: ins.op = R5900Opcode::TLT; break;
                case 0x33: ins.op = R5900Opcode::TLTU; break;
                case 0x34: ins.op = R5900Opcode::TEQ; break;
                case 0x36: ins.op = R5900Opcode::TNE; break;
                case 0x38: ins.op = R5900Opcode::DSLL; break;
                case 0x3A: ins.op = R5900Opcode::DSRL; break;
                case 0x3B: ins.op = R5900Opcode::DSRA; break;
                case 0x3C: ins.op = R5900Opcode::DSLL32; break;
                case 0x3E: ins.op = R5900Opcode::DSRL32; break;
                case 0x3F: ins.op = R5900Opcode::DSRA32; break;
                default:
                    mark_unknown(ins, "SPECIAL funct=" + hex32(funct));
                    break;
            }
            break;
        }
        case 0x01: { // REGIMM
            switch (ins.rt) {
                case 0x00: ins.op = R5900Opcode::BLTZ;   ins.is_branch = true; ins.has_delay_slot = true; ins.target = branch_target(address, ins.simm); break;
                case 0x01: ins.op = R5900Opcode::BGEZ;   ins.is_branch = true; ins.has_delay_slot = true; ins.target = branch_target(address, ins.simm); break;
                case 0x02: ins.op = R5900Opcode::BLTZL;  ins.is_branch = true; ins.is_branch_likely = true; ins.has_delay_slot = true; ins.target = branch_target(address, ins.simm); break;
                case 0x03: ins.op = R5900Opcode::BGEZL;  ins.is_branch = true; ins.is_branch_likely = true; ins.has_delay_slot = true; ins.target = branch_target(address, ins.simm); break;
                case 0x08: ins.op = R5900Opcode::TGEI;   break;
                case 0x09: ins.op = R5900Opcode::TGEIU;  break;
                case 0x0A: ins.op = R5900Opcode::TLTI;   break;
                case 0x0B: ins.op = R5900Opcode::TLTIU;  break;
                case 0x0C: ins.op = R5900Opcode::TEQI;   break;
                case 0x0E: ins.op = R5900Opcode::TNEI;   break;
                case 0x10: ins.op = R5900Opcode::BLTZAL; ins.is_branch = true; ins.is_call = true; ins.has_delay_slot = true; ins.target = branch_target(address, ins.simm); break;
                case 0x11: ins.op = R5900Opcode::BGEZAL; ins.is_branch = true; ins.is_call = true; ins.has_delay_slot = true; ins.target = branch_target(address, ins.simm); break;
                case 0x12: ins.op = R5900Opcode::BLTZALL; ins.is_branch = true; ins.is_branch_likely = true; ins.is_call = true; ins.has_delay_slot = true; ins.target = branch_target(address, ins.simm); break;
                case 0x13: ins.op = R5900Opcode::BGEZALL; ins.is_branch = true; ins.is_branch_likely = true; ins.is_call = true; ins.has_delay_slot = true; ins.target = branch_target(address, ins.simm); break;
                case 0x18: ins.op = R5900Opcode::MTSAB;  break;
                case 0x19: ins.op = R5900Opcode::MTSAH;  break;
                default:
                    mark_unknown(ins, "REGIMM rt=" + hex32(ins.rt));
                    break;
            }
            break;
        }
        case 0x02: ins.op = R5900Opcode::J;      ins.is_jump = true; ins.has_delay_slot = true; ins.target = jump_target(address, word); break;
        case 0x03: ins.op = R5900Opcode::JAL;    ins.is_jump = true; ins.is_call = true; ins.has_delay_slot = true; ins.target = jump_target(address, word); break;
        case 0x04: ins.op = R5900Opcode::BEQ;    ins.is_branch = true; ins.has_delay_slot = true; ins.target = branch_target(address, ins.simm); break;
        case 0x05: ins.op = R5900Opcode::BNE;    ins.is_branch = true; ins.has_delay_slot = true; ins.target = branch_target(address, ins.simm); break;
        case 0x06: ins.op = R5900Opcode::BLEZ;   ins.is_branch = true; ins.has_delay_slot = true; ins.target = branch_target(address, ins.simm); break;
        case 0x07: ins.op = R5900Opcode::BGTZ;   ins.is_branch = true; ins.has_delay_slot = true; ins.target = branch_target(address, ins.simm); break;
        case 0x08: ins.op = R5900Opcode::ADDI;   break;
        case 0x09: ins.op = R5900Opcode::ADDIU;  break;
        case 0x0A: ins.op = R5900Opcode::SLTI;   break;
        case 0x0B: ins.op = R5900Opcode::SLTIU;  break;
        case 0x0C: ins.op = R5900Opcode::ANDI;   break;
        case 0x0D: ins.op = R5900Opcode::ORI;    break;
        case 0x0E: ins.op = R5900Opcode::XORI;   break;
        case 0x0F: ins.op = R5900Opcode::LUI;    break;

        case 0x10: { // COP0
            switch (ins.rs) {
                case 0x00: ins.op = R5900Opcode::MFC0; break;
                case 0x04: ins.op = R5900Opcode::MTC0; break;
                case 0x08: {
                    switch (ins.rt) {
                        case 0x00: ins.op = R5900Opcode::BC0F;  ins.is_branch = true; ins.has_delay_slot = true; ins.target = branch_target(address, ins.simm); break;
                        case 0x01: ins.op = R5900Opcode::BC0T;  ins.is_branch = true; ins.has_delay_slot = true; ins.target = branch_target(address, ins.simm); break;
                        case 0x02: ins.op = R5900Opcode::BC0FL; ins.is_branch = true; ins.is_branch_likely = true; ins.has_delay_slot = true; ins.target = branch_target(address, ins.simm); break;
                        case 0x03: ins.op = R5900Opcode::BC0TL; ins.is_branch = true; ins.is_branch_likely = true; ins.has_delay_slot = true; ins.target = branch_target(address, ins.simm); break;
                        default: mark_unknown(ins, "COP0 BC0 rt=" + hex32(ins.rt)); break;
                    }
                    break;
                }
                case 0x10: {
                    switch (funct) {
                        case 0x01: ins.op = R5900Opcode::TLBR; break;
                        case 0x02: ins.op = R5900Opcode::TLBWI; break;
                        case 0x06: ins.op = R5900Opcode::TLBWR; break;
                        case 0x08: ins.op = R5900Opcode::TLBP; break;
                        case 0x18: ins.op = R5900Opcode::ERET; break;
                        case 0x38: ins.op = R5900Opcode::EI; break;
                        case 0x39: ins.op = R5900Opcode::DI; break;
                        default: mark_unknown(ins, "COP0 C0 funct=" + hex32(funct)); break;
                    }
                    break;
                }
                default:
                    mark_unknown(ins, "COP0 rs=" + hex32(ins.rs));
                    break;
            }
            break;
        }

        case 0x11: { // COP1 (FPU)
            switch (ins.rs) {
                case 0x00: ins.op = R5900Opcode::MFC1; break;
                case 0x02: ins.op = R5900Opcode::CFC1; break;
                case 0x04: ins.op = R5900Opcode::MTC1; break;
                case 0x06: ins.op = R5900Opcode::CTC1; break;
                case 0x08: {
                    switch (ins.rt) {
                        case 0x00: ins.op = R5900Opcode::BC1F;  ins.is_branch = true; ins.has_delay_slot = true; ins.target = branch_target(address, ins.simm); break;
                        case 0x01: ins.op = R5900Opcode::BC1T;  ins.is_branch = true; ins.has_delay_slot = true; ins.target = branch_target(address, ins.simm); break;
                        case 0x02: ins.op = R5900Opcode::BC1FL; ins.is_branch = true; ins.is_branch_likely = true; ins.has_delay_slot = true; ins.target = branch_target(address, ins.simm); break;
                        case 0x03: ins.op = R5900Opcode::BC1TL; ins.is_branch = true; ins.is_branch_likely = true; ins.has_delay_slot = true; ins.target = branch_target(address, ins.simm); break;
                        default: mark_unknown(ins, "COP1 BC1 rt=" + hex32(ins.rt)); break;
                    }
                    break;
                }
                case 0x10: { // Single precision (S)
                    switch (funct) {
                        case 0x00: ins.op = R5900Opcode::ADD_S;   break;
                        case 0x01: ins.op = R5900Opcode::SUB_S;   break;
                        case 0x02: ins.op = R5900Opcode::MUL_S;   break;
                        case 0x03: ins.op = R5900Opcode::DIV_S;   break;
                        case 0x04: ins.op = R5900Opcode::SQRT_S;  break;
                        case 0x05: ins.op = R5900Opcode::ABS_S;   break;
                        case 0x06: ins.op = R5900Opcode::MOV_S;   break;
                        case 0x07: ins.op = R5900Opcode::NEG_S;   break;
                        case 0x16: ins.op = R5900Opcode::RSQRT_S; break;
                        case 0x18: ins.op = R5900Opcode::ADDA_S;  break;
                        case 0x19: ins.op = R5900Opcode::SUBA_S;  break;
                        case 0x1A: ins.op = R5900Opcode::MULA_S;  break;
                        case 0x1C: ins.op = R5900Opcode::MADD_S;  break;
                        case 0x1D: ins.op = R5900Opcode::MSUB_S;  break;
                        case 0x1E: ins.op = R5900Opcode::MADDA_S; break;
                        case 0x1F: ins.op = R5900Opcode::MSUBA_S; break;
                        case 0x24: ins.op = R5900Opcode::CVT_W_S; break;
                        case 0x28: ins.op = R5900Opcode::MAX_S;   break;
                        case 0x29: ins.op = R5900Opcode::MIN_S;   break;
                        case 0x30: ins.op = R5900Opcode::C_F_S;   break;
                        case 0x32: ins.op = R5900Opcode::C_EQ_S;  break;
                        case 0x34: ins.op = R5900Opcode::C_LT_S;  break;
                        case 0x36: ins.op = R5900Opcode::C_LE_S;  break;
                        default: mark_unknown(ins, "COP1.S funct=" + hex32(funct)); break;
                    }
                    break;
                }
                case 0x14: { // Word (W)
                    if (funct == 0x20) {
                        ins.op = R5900Opcode::CVT_S_W;
                    } else {
                        mark_unknown(ins, "COP1.W funct=" + hex32(funct));
                    }
                    break;
                }
                default:
                    mark_unknown(ins, "COP1 rs=" + hex32(ins.rs));
                    break;
            }
            break;
        }

        case 0x12: { // COP2 (VU0 Macro Mode)
            if ((ins.rs & 0x10u) != 0) {
                ins.op = R5900Opcode::VU0_MACRO_OP;
            } else {
                switch (ins.rs) {
                    case 0x01: ins.op = R5900Opcode::QMFC2; break;
                    case 0x02: ins.op = R5900Opcode::CFC2;  break;
                    case 0x05: ins.op = R5900Opcode::QMTC2; break;
                    case 0x06: ins.op = R5900Opcode::CTC2;  break;
                    case 0x08: {
                        switch (ins.rt) {
                            case 0x00: ins.op = R5900Opcode::BC2F;  ins.is_branch = true; ins.has_delay_slot = true; ins.target = branch_target(address, ins.simm); break;
                            case 0x01: ins.op = R5900Opcode::BC2T;  ins.is_branch = true; ins.has_delay_slot = true; ins.target = branch_target(address, ins.simm); break;
                            case 0x02: ins.op = R5900Opcode::BC2FL; ins.is_branch = true; ins.is_branch_likely = true; ins.has_delay_slot = true; ins.target = branch_target(address, ins.simm); break;
                            case 0x03: ins.op = R5900Opcode::BC2TL; ins.is_branch = true; ins.is_branch_likely = true; ins.has_delay_slot = true; ins.target = branch_target(address, ins.simm); break;
                            default: mark_unknown(ins, "COP2 BC2 rt=" + hex32(ins.rt)); break;
                        }
                        break;
                    }
                    default:
                        mark_unknown(ins, "COP2 rs=" + hex32(ins.rs));
                        break;
                }
            }
            break;
        }

        case 0x14: ins.op = R5900Opcode::BEQL;   ins.is_branch = true; ins.is_branch_likely = true; ins.has_delay_slot = true; ins.target = branch_target(address, ins.simm); break;
        case 0x15: ins.op = R5900Opcode::BNEL;   ins.is_branch = true; ins.is_branch_likely = true; ins.has_delay_slot = true; ins.target = branch_target(address, ins.simm); break;
        case 0x16: ins.op = R5900Opcode::BLEZL;  ins.is_branch = true; ins.is_branch_likely = true; ins.has_delay_slot = true; ins.target = branch_target(address, ins.simm); break;
        case 0x17: ins.op = R5900Opcode::BGTZL;  ins.is_branch = true; ins.is_branch_likely = true; ins.has_delay_slot = true; ins.target = branch_target(address, ins.simm); break;
        case 0x18: ins.op = R5900Opcode::DADDI;  break;
        case 0x19: ins.op = R5900Opcode::DADDIU; break;
        case 0x1A: ins.op = R5900Opcode::LDL;    break;
        case 0x1B: ins.op = R5900Opcode::LDR;    break;

        case 0x1C: { // MMI (Emotion Engine MultiMedia Instructions)
            switch (funct) {
                case 0x00: ins.op = R5900Opcode::MADD;   break;
                case 0x01: ins.op = R5900Opcode::MADDU;  break;
                case 0x04: ins.op = R5900Opcode::PLZCW;  break;
                case 0x08: { // MMI0
                    switch (ins.sa) {
                        case 0x00: ins.op = R5900Opcode::PADDW;  break;
                        case 0x01: ins.op = R5900Opcode::PSUBW;  break;
                        case 0x02: ins.op = R5900Opcode::PCGTW;  break;
                        case 0x03: ins.op = R5900Opcode::PMAXW;  break;
                        case 0x04: ins.op = R5900Opcode::PADDH;  break;
                        case 0x05: ins.op = R5900Opcode::PSUBH;  break;
                        case 0x06: ins.op = R5900Opcode::PCGTH;  break;
                        case 0x07: ins.op = R5900Opcode::PMAXH;  break;
                        case 0x08: ins.op = R5900Opcode::PADDB;  break;
                        case 0x09: ins.op = R5900Opcode::PSUBB;  break;
                        case 0x0A: ins.op = R5900Opcode::PCGTB;  break;
                        case 0x10: ins.op = R5900Opcode::PADDSW; break;
                        case 0x11: ins.op = R5900Opcode::PSUBSW; break;
                        case 0x12: ins.op = R5900Opcode::PEXTLW; break;
                        case 0x13: ins.op = R5900Opcode::PPACW;  break;
                        case 0x14: ins.op = R5900Opcode::PADDSH; break;
                        case 0x15: ins.op = R5900Opcode::PSUBSH; break;
                        case 0x16: ins.op = R5900Opcode::PEXTLH; break;
                        case 0x17: ins.op = R5900Opcode::PPACH;  break;
                        case 0x18: ins.op = R5900Opcode::PADDSB; break;
                        case 0x19: ins.op = R5900Opcode::PSUBSB; break;
                        case 0x1A: ins.op = R5900Opcode::PEXTLB; break;
                        case 0x1B: ins.op = R5900Opcode::PPACB;  break;
                        case 0x1E: ins.op = R5900Opcode::PEXT5;  break;
                        case 0x1F: ins.op = R5900Opcode::PPAC5;  break;
                        default: mark_unknown(ins, "MMI0 sa=" + hex32(ins.sa)); break;
                    }
                    break;
                }
                case 0x09: { // MMI2
                    switch (ins.sa) {
                        case 0x00: ins.op = R5900Opcode::PMADDW; break;
                        case 0x02: ins.op = R5900Opcode::PSLLVW; break;
                        case 0x03: ins.op = R5900Opcode::PSRLVW; break;
                        case 0x04: ins.op = R5900Opcode::PMSUBW; break;
                        case 0x08: ins.op = R5900Opcode::PMFHI;  break;
                        case 0x09: ins.op = R5900Opcode::PMFLO;  break;
                        case 0x0A: ins.op = R5900Opcode::PINTH;  break;
                        case 0x0C: ins.op = R5900Opcode::PMULTW; break;
                        case 0x0D: ins.op = R5900Opcode::PDIVW;  break;
                        case 0x0E: ins.op = R5900Opcode::PCPYLD; break;
                        case 0x10: ins.op = R5900Opcode::PMADDH; break;
                        case 0x11: ins.op = R5900Opcode::PHMADH; break;
                        case 0x12: ins.op = R5900Opcode::PAND;   break;
                        case 0x13: ins.op = R5900Opcode::PXOR;   break;
                        case 0x14: ins.op = R5900Opcode::PMSUBH; break;
                        case 0x15: ins.op = R5900Opcode::PHMSBH; break;
                        case 0x1A: ins.op = R5900Opcode::PEXEH;  break;
                        case 0x1B: ins.op = R5900Opcode::PREVH;  break;
                        case 0x1C: ins.op = R5900Opcode::PMULTH; break;
                        case 0x1D: ins.op = R5900Opcode::PDIVBW; break;
                        case 0x1E: ins.op = R5900Opcode::PEXEW;  break;
                        case 0x1F: ins.op = R5900Opcode::PROT3W; break;
                        default: mark_unknown(ins, "MMI2 sa=" + hex32(ins.sa)); break;
                    }
                    break;
                }
                case 0x10: ins.op = R5900Opcode::MFHI1;  break;
                case 0x11: ins.op = R5900Opcode::MTHI1;  break;
                case 0x12: ins.op = R5900Opcode::MFLO1;  break;
                case 0x13: ins.op = R5900Opcode::MTLO1;  break;
                case 0x18: ins.op = R5900Opcode::MULT1;  break;
                case 0x19: ins.op = R5900Opcode::MULTU1; break;
                case 0x1A: ins.op = R5900Opcode::DIV1;   break;
                case 0x1B: ins.op = R5900Opcode::DIVU1;  break;
                case 0x20: ins.op = R5900Opcode::MADD1;  break;
                case 0x21: ins.op = R5900Opcode::MADDU1; break;
                case 0x28: { // MMI1
                    switch (ins.sa) {
                        case 0x01: ins.op = R5900Opcode::PABSW;  break;
                        case 0x02: ins.op = R5900Opcode::PCEQW;  break;
                        case 0x03: ins.op = R5900Opcode::PMINW;  break;
                        case 0x04: ins.op = R5900Opcode::PADSBH; break;
                        case 0x05: ins.op = R5900Opcode::PABSH;  break;
                        case 0x06: ins.op = R5900Opcode::PCEQH;  break;
                        case 0x07: ins.op = R5900Opcode::PMINH;  break;
                        case 0x0A: ins.op = R5900Opcode::PCEQB;  break;
                        case 0x10: ins.op = R5900Opcode::PADDUW; break;
                        case 0x11: ins.op = R5900Opcode::PSUBUW; break;
                        case 0x12: ins.op = R5900Opcode::PEXTUW; break;
                        case 0x14: ins.op = R5900Opcode::PADDUH; break;
                        case 0x15: ins.op = R5900Opcode::PSUBUH; break;
                        case 0x16: ins.op = R5900Opcode::PEXTUH; break;
                        case 0x18: ins.op = R5900Opcode::PADDUB; break;
                        case 0x19: ins.op = R5900Opcode::PSUBUB; break;
                        case 0x1A: ins.op = R5900Opcode::PEXTUB; break;
                        case 0x1B: ins.op = R5900Opcode::QFSRV;  break;
                        default: mark_unknown(ins, "MMI1 sa=" + hex32(ins.sa)); break;
                    }
                    break;
                }
                case 0x29: { // MMI3
                    switch (ins.sa) {
                        case 0x00: ins.op = R5900Opcode::PMADDUW; break;
                        case 0x03: ins.op = R5900Opcode::PSRAVW;  break;
                        case 0x08: ins.op = R5900Opcode::PMTHI;   break;
                        case 0x09: ins.op = R5900Opcode::PMTLO;   break;
                        case 0x0A: ins.op = R5900Opcode::PINTEH;  break;
                        case 0x0C: ins.op = R5900Opcode::PMULTUW; break;
                        case 0x0D: ins.op = R5900Opcode::PDIVUW;  break;
                        case 0x0E: ins.op = R5900Opcode::PCPYUD;  break;
                        case 0x12: ins.op = R5900Opcode::POR;     break;
                        case 0x13: ins.op = R5900Opcode::PNOR;    break;
                        case 0x1A: ins.op = R5900Opcode::PEXCH;   break;
                        case 0x1B: ins.op = R5900Opcode::PCPYH;   break;
                        case 0x1E: ins.op = R5900Opcode::PEXCW;   break;
                        default: mark_unknown(ins, "MMI3 sa=" + hex32(ins.sa)); break;
                    }
                    break;
                }
                case 0x30: ins.op = R5900Opcode::PMFHL; break;
                case 0x31: ins.op = R5900Opcode::PMTHL; break;
                case 0x34: ins.op = R5900Opcode::PSLLH; break;
                case 0x36: ins.op = R5900Opcode::PSRLH; break;
                case 0x37: ins.op = R5900Opcode::PSRAH; break;
                case 0x3C: ins.op = R5900Opcode::PSLLW; break;
                case 0x3E: ins.op = R5900Opcode::PSRLW; break;
                case 0x3F: ins.op = R5900Opcode::PSRAW; break;
                default:
                    mark_unknown(ins, "MMI funct=" + hex32(funct));
                    break;
            }
            break;
        }

        case 0x1E: ins.op = R5900Opcode::LQ;    break;
        case 0x1F: ins.op = R5900Opcode::SQ;    break;
        case 0x20: ins.op = R5900Opcode::LB;    break;
        case 0x21: ins.op = R5900Opcode::LH;    break;
        case 0x22: ins.op = R5900Opcode::LWL;   break;
        case 0x23: ins.op = R5900Opcode::LW;    break;
        case 0x24: ins.op = R5900Opcode::LBU;   break;
        case 0x25: ins.op = R5900Opcode::LHU;   break;
        case 0x26: ins.op = R5900Opcode::LWR;   break;
        case 0x27: ins.op = R5900Opcode::LWU;   break;
        case 0x28: ins.op = R5900Opcode::SB;    break;
        case 0x29: ins.op = R5900Opcode::SH;    break;
        case 0x2A: ins.op = R5900Opcode::SWL;   break;
        case 0x2B: ins.op = R5900Opcode::SW;    break;
        case 0x2C: ins.op = R5900Opcode::SDL;   break;
        case 0x2D: ins.op = R5900Opcode::SDR;   break;
        case 0x2E: ins.op = R5900Opcode::SWR;   break;
        case 0x2F: ins.op = R5900Opcode::CACHE; break;
        case 0x31: ins.op = R5900Opcode::LWC1;  break;
        case 0x33: ins.op = R5900Opcode::PREF;  break;
        case 0x36: ins.op = R5900Opcode::LQC2;  break;
        case 0x37: ins.op = R5900Opcode::LD;    break;
        case 0x39: ins.op = R5900Opcode::SWC1;  break;
        case 0x3E: ins.op = R5900Opcode::SQC2;  break;
        case 0x3F: ins.op = R5900Opcode::SD;    break;

        default:
            mark_unknown(ins, "PRIMARY op=" + hex32(primary));
            break;
    }

    return ins;
}

const char* R5900Decoder::gpr_name(u8 reg) {
    static const char* kNames[32] = {
        "zero", "at", "v0", "v1", "a0", "a1", "a2", "a3",
        "t0",   "t1", "t2", "t3", "t4", "t5", "t6", "t7",
        "s0",   "s1", "s2", "s3", "s4", "s5", "s6", "s7",
        "t8",   "t9", "k0", "k1", "gp", "sp", "fp", "ra"
    };
    return kNames[reg & 31];
}

const char* R5900Decoder::opcode_name(R5900Opcode op) {
    switch (op) {
        case R5900Opcode::Unknown: return "UNKNOWN";
        case R5900Opcode::SLL: return "sll"; case R5900Opcode::SRL: return "srl"; case R5900Opcode::SRA: return "sra";
        case R5900Opcode::SLLV: return "sllv"; case R5900Opcode::SRLV: return "srlv"; case R5900Opcode::SRAV: return "srav";
        case R5900Opcode::DSLL: return "dsll"; case R5900Opcode::DSRL: return "dsrl"; case R5900Opcode::DSRA: return "dsra";
        case R5900Opcode::DSLL32: return "dsll32"; case R5900Opcode::DSRL32: return "dsrl32"; case R5900Opcode::DSRA32: return "dsra32";
        case R5900Opcode::DSLLV: return "dsllv"; case R5900Opcode::DSRLV: return "dsrlv"; case R5900Opcode::DSRAV: return "dsrav";
        case R5900Opcode::ADD: return "add"; case R5900Opcode::ADDU: return "addu"; case R5900Opcode::SUB: return "sub"; case R5900Opcode::SUBU: return "subu";
        case R5900Opcode::DADD: return "dadd"; case R5900Opcode::DADDU: return "daddu"; case R5900Opcode::DSUB: return "dsub"; case R5900Opcode::DSUBU: return "dsubu";
        case R5900Opcode::ADDI: return "addi"; case R5900Opcode::ADDIU: return "addiu"; case R5900Opcode::DADDI: return "daddi"; case R5900Opcode::DADDIU: return "daddiu";
        case R5900Opcode::AND: return "and"; case R5900Opcode::OR: return "or"; case R5900Opcode::XOR: return "xor"; case R5900Opcode::NOR: return "nor";
        case R5900Opcode::ANDI: return "andi"; case R5900Opcode::ORI: return "ori"; case R5900Opcode::XORI: return "xori"; case R5900Opcode::LUI: return "lui";
        case R5900Opcode::SLT: return "slt"; case R5900Opcode::SLTU: return "sltu"; case R5900Opcode::SLTI: return "slti"; case R5900Opcode::SLTIU: return "sltiu";
        case R5900Opcode::MOVZ: return "movz"; case R5900Opcode::MOVN: return "movn";
        case R5900Opcode::MFHI: return "mfhi"; case R5900Opcode::MTHI: return "mthi"; case R5900Opcode::MFLO: return "mflo"; case R5900Opcode::MTLO: return "mtlo";
        case R5900Opcode::MFHI1: return "mfhi1"; case R5900Opcode::MTHI1: return "mthi1"; case R5900Opcode::MFLO1: return "mflo1"; case R5900Opcode::MTLO1: return "mtlo1";
        case R5900Opcode::MFSA: return "mfsa"; case R5900Opcode::MTSA: return "mtsa"; case R5900Opcode::MTSAB: return "mtsab"; case R5900Opcode::MTSAH: return "mtsah";
        case R5900Opcode::MULT: return "mult"; case R5900Opcode::MULTU: return "multu"; case R5900Opcode::DIV: return "div"; case R5900Opcode::DIVU: return "divu";
        case R5900Opcode::MULT1: return "mult1"; case R5900Opcode::MULTU1: return "multu1"; case R5900Opcode::DIV1: return "div1"; case R5900Opcode::DIVU1: return "divu1";
        case R5900Opcode::MADD: return "madd"; case R5900Opcode::MADDU: return "maddu"; case R5900Opcode::MADD1: return "madd1"; case R5900Opcode::MADDU1: return "maddu1";
        case R5900Opcode::J: return "j"; case R5900Opcode::JAL: return "jal"; case R5900Opcode::JR: return "jr"; case R5900Opcode::JALR: return "jalr";
        case R5900Opcode::BEQ: return "beq"; case R5900Opcode::BNE: return "bne"; case R5900Opcode::BLEZ: return "blez"; case R5900Opcode::BGTZ: return "bgtz";
        case R5900Opcode::BLTZ: return "bltz"; case R5900Opcode::BGEZ: return "bgez"; case R5900Opcode::BLTZAL: return "bltzal"; case R5900Opcode::BGEZAL: return "bgezal";
        case R5900Opcode::BEQL: return "beql"; case R5900Opcode::BNEL: return "bnel"; case R5900Opcode::BLEZL: return "blezl"; case R5900Opcode::BGTZL: return "bgtzl";
        case R5900Opcode::BLTZL: return "bltzl"; case R5900Opcode::BGEZL: return "bgezl"; case R5900Opcode::BLTZALL: return "bltzall"; case R5900Opcode::BGEZALL: return "bgezall";
        case R5900Opcode::LB: return "lb"; case R5900Opcode::LH: return "lh"; case R5900Opcode::LWL: return "lwl"; case R5900Opcode::LW: return "lw";
        case R5900Opcode::LBU: return "lbu"; case R5900Opcode::LHU: return "lhu"; case R5900Opcode::LWR: return "lwr"; case R5900Opcode::LWU: return "lwu";
        case R5900Opcode::LD: return "ld"; case R5900Opcode::LDL: return "ldl"; case R5900Opcode::LDR: return "ldr"; case R5900Opcode::LQ: return "lq";
        case R5900Opcode::SB: return "sb"; case R5900Opcode::SH: return "sh"; case R5900Opcode::SWL: return "swl"; case R5900Opcode::SW: return "sw"; case R5900Opcode::SWR: return "swr";
        case R5900Opcode::SD: return "sd"; case R5900Opcode::SDL: return "sdl"; case R5900Opcode::SDR: return "sdr"; case R5900Opcode::SQ: return "sq";
        case R5900Opcode::LWC1: return "lwc1"; case R5900Opcode::SWC1: return "swc1"; case R5900Opcode::LQC2: return "lqc2"; case R5900Opcode::SQC2: return "sqc2";
        case R5900Opcode::PREF: return "pref"; case R5900Opcode::CACHE: return "cache"; case R5900Opcode::SYNC: return "sync";
        case R5900Opcode::SYSCALL: return "syscall"; case R5900Opcode::BREAK: return "break";
        case R5900Opcode::TGE: return "tge"; case R5900Opcode::TGEU: return "tgeu"; case R5900Opcode::TLT: return "tlt"; case R5900Opcode::TLTU: return "tltu"; case R5900Opcode::TEQ: return "teq"; case R5900Opcode::TNE: return "tne";
        case R5900Opcode::TGEI: return "tgei"; case R5900Opcode::TGEIU: return "tgeiu"; case R5900Opcode::TLTI: return "tlti"; case R5900Opcode::TLTIU: return "tltiu"; case R5900Opcode::TEQI: return "teqi"; case R5900Opcode::TNEI: return "tnei";
        case R5900Opcode::MFC0: return "mfc0"; case R5900Opcode::MTC0: return "mtc0";
        case R5900Opcode::BC0F: return "bc0f"; case R5900Opcode::BC0T: return "bc0t"; case R5900Opcode::BC0FL: return "bc0fl"; case R5900Opcode::BC0TL: return "bc0tl";
        case R5900Opcode::TLBR: return "tlbr"; case R5900Opcode::TLBWI: return "tlbwi"; case R5900Opcode::TLBWR: return "tlbwr"; case R5900Opcode::TLBP: return "tlbp";
        case R5900Opcode::ERET: return "eret"; case R5900Opcode::EI: return "ei"; case R5900Opcode::DI: return "di";
        case R5900Opcode::MFC1: return "mfc1"; case R5900Opcode::CFC1: return "cfc1"; case R5900Opcode::MTC1: return "mtc1"; case R5900Opcode::CTC1: return "ctc1";
        case R5900Opcode::BC1F: return "bc1f"; case R5900Opcode::BC1T: return "bc1t"; case R5900Opcode::BC1FL: return "bc1fl"; case R5900Opcode::BC1TL: return "bc1tl";
        case R5900Opcode::ADD_S: return "add.s"; case R5900Opcode::SUB_S: return "sub.s"; case R5900Opcode::MUL_S: return "mul.s"; case R5900Opcode::DIV_S: return "div.s";
        case R5900Opcode::SQRT_S: return "sqrt.s"; case R5900Opcode::ABS_S: return "abs.s"; case R5900Opcode::MOV_S: return "mov.s"; case R5900Opcode::NEG_S: return "neg.s";
        case R5900Opcode::RSQRT_S: return "rsqrt.s"; case R5900Opcode::ADDA_S: return "adda.s"; case R5900Opcode::SUBA_S: return "suba.s"; case R5900Opcode::MULA_S: return "mula.s";
        case R5900Opcode::MADD_S: return "madd.s"; case R5900Opcode::MSUB_S: return "msub.s"; case R5900Opcode::MADDA_S: return "madda.s"; case R5900Opcode::MSUBA_S: return "msuba.s";
        case R5900Opcode::CVT_W_S: return "cvt.w.s"; case R5900Opcode::CVT_S_W: return "cvt.s.w"; case R5900Opcode::MAX_S: return "max.s"; case R5900Opcode::MIN_S: return "min.s";
        case R5900Opcode::C_F_S: return "c.f.s"; case R5900Opcode::C_EQ_S: return "c.eq.s"; case R5900Opcode::C_LT_S: return "c.lt.s"; case R5900Opcode::C_LE_S: return "c.le.s";
        case R5900Opcode::PADDW: return "paddw"; case R5900Opcode::PSUBW: return "psubw"; case R5900Opcode::PCGTW: return "pcgtw"; case R5900Opcode::PMAXW: return "pmaxw";
        case R5900Opcode::PADDH: return "paddh"; case R5900Opcode::PSUBH: return "psubh"; case R5900Opcode::PCGTH: return "pcgth"; case R5900Opcode::PMAXH: return "pmaxh";
        case R5900Opcode::PADDB: return "paddb"; case R5900Opcode::PSUBB: return "psubb"; case R5900Opcode::PCGTB: return "pcgtb";
        case R5900Opcode::PADDSW: return "paddsw"; case R5900Opcode::PSUBSW: return "psubsw"; case R5900Opcode::PEXTLW: return "pextlw"; case R5900Opcode::PPACW: return "ppacw";
        case R5900Opcode::PADDSH: return "paddsh"; case R5900Opcode::PSUBSH: return "psubsh"; case R5900Opcode::PEXTLH: return "pextlh"; case R5900Opcode::PPACH: return "ppach";
        case R5900Opcode::PADDSB: return "paddsb"; case R5900Opcode::PSUBSB: return "psubsb"; case R5900Opcode::PEXTLB: return "pextlb"; case R5900Opcode::PPACB: return "ppacb";
        case R5900Opcode::PEXT5: return "pext5"; case R5900Opcode::PPAC5: return "ppac5";
        case R5900Opcode::PABSW: return "pabsw"; case R5900Opcode::PCEQW: return "pceqw"; case R5900Opcode::PMINW: return "pminw"; case R5900Opcode::PADSBH: return "padsbh";
        case R5900Opcode::PABSH: return "pabsh"; case R5900Opcode::PCEQH: return "pceqh"; case R5900Opcode::PMINH: return "pminh"; case R5900Opcode::PCEQB: return "pceqb";
        case R5900Opcode::PADDUW: return "padduw"; case R5900Opcode::PSUBUW: return "psubuw"; case R5900Opcode::PEXTUW: return "pextuw";
        case R5900Opcode::PADDUH: return "padduh"; case R5900Opcode::PSUBUH: return "psubuh"; case R5900Opcode::PEXTUH: return "pextuh";
        case R5900Opcode::PADDUB: return "paddub"; case R5900Opcode::PSUBUB: return "psubub"; case R5900Opcode::PEXTUB: return "pextub"; case R5900Opcode::QFSRV: return "qfsrv";
        case R5900Opcode::PMADDW: return "pmaddw"; case R5900Opcode::PSLLVW: return "psllvw"; case R5900Opcode::PSRLVW: return "psrlvw"; case R5900Opcode::PMSUBW: return "pmsubw";
        case R5900Opcode::PMFHI: return "pmfhi"; case R5900Opcode::PMFLO: return "pmflo"; case R5900Opcode::PINTH: return "pinth";
        case R5900Opcode::PMULTW: return "pmultw"; case R5900Opcode::PDIVW: return "pdivw"; case R5900Opcode::PCPYLD: return "pcpyld";
        case R5900Opcode::PMADDH: return "pmaddh"; case R5900Opcode::PHMADH: return "phmadh"; case R5900Opcode::PAND: return "pand"; case R5900Opcode::PXOR: return "pxor";
        case R5900Opcode::PMSUBH: return "pmsubh"; case R5900Opcode::PHMSBH: return "phmsbh";
        case R5900Opcode::PEXEH: return "pexeh"; case R5900Opcode::PREVH: return "prevh"; case R5900Opcode::PMULTH: return "pmulth";
        case R5900Opcode::PDIVBW: return "pdivbw"; case R5900Opcode::PEXEW: return "pexew"; case R5900Opcode::PROT3W: return "prot3w";
        case R5900Opcode::PMADDUW: return "pmadduw"; case R5900Opcode::PSRAVW: return "psravw";
        case R5900Opcode::PMTHI: return "pmthi"; case R5900Opcode::PMTLO: return "pmtlo"; case R5900Opcode::PINTEH: return "pinteh";
        case R5900Opcode::PMULTUW: return "pmultuw"; case R5900Opcode::PDIVUW: return "pdivuw"; case R5900Opcode::PCPYUD: return "pcpyud";
        case R5900Opcode::POR: return "por"; case R5900Opcode::PNOR: return "pnor";
        case R5900Opcode::PEXCH: return "pexch"; case R5900Opcode::PCPYH: return "pcpyh"; case R5900Opcode::PEXCW: return "pexcw";
        case R5900Opcode::PLZCW: return "plzcw"; case R5900Opcode::PMFHL: return "pmfhl"; case R5900Opcode::PMTHL: return "pmthl";
        case R5900Opcode::PSLLH: return "psllh"; case R5900Opcode::PSRLH: return "psrlh"; case R5900Opcode::PSRAH: return "psrah";
        case R5900Opcode::PSLLW: return "psllw"; case R5900Opcode::PSRLW: return "psrlw"; case R5900Opcode::PSRAW: return "psraw";
        case R5900Opcode::QMFC2: return "qmfc2"; case R5900Opcode::CFC2: return "cfc2"; case R5900Opcode::QMTC2: return "qmtc2"; case R5900Opcode::CTC2: return "ctc2";
        case R5900Opcode::BC2F: return "bc2f"; case R5900Opcode::BC2T: return "bc2t"; case R5900Opcode::BC2FL: return "bc2fl"; case R5900Opcode::BC2TL: return "bc2tl";
        case R5900Opcode::VU0_MACRO_OP: return "vu0.macro";
    }
    return "UNKNOWN";
}

std::string R5900Decoder::disassemble(const DecodedInstruction& ins) {
    if (ins.raw == 0) return "nop";
    std::ostringstream oss;
    oss << opcode_name(ins.op);
    if (ins.op == R5900Opcode::Unknown) {
        oss << " (" << hex32(ins.raw) << ": " << ins.unknown_reason << ")";
        return oss.str();
    }
    if (ins.is_jump && !ins.is_indirect_jump) {
        oss << " " << hex32(ins.target);
    } else if (ins.is_branch) {
        oss << " $" << gpr_name(ins.rs) << ", $" << gpr_name(ins.rt) << ", " << hex32(ins.target);
    } else if (ins.op == R5900Opcode::JR) {
        oss << " $" << gpr_name(ins.rs);
    } else {
        oss << " $" << gpr_name(ins.rd) << ", $" << gpr_name(ins.rs) << ", $" << gpr_name(ins.rt)
            << " [imm=" << ins.simm << "]";
    }
    return oss.str();
}

std::string DecodeCoverageReport::format_report() const {
    std::ostringstream oss;
    oss << "=== MIPS R5900 DECODE COVERAGE REPORT ===\n";
    oss << "Total Instructions   : " << total_instructions << "\n";
    oss << "Decoded Instructions : " << decoded_instructions << " ("
        << std::fixed << std::setprecision(4) << coverage_percent() << "%)\n";
    oss << "Unknown Instructions : " << unknown_instructions << "\n";
    oss << "Distinct Opcodes Used: " << opcode_histogram.size() << "\n";

    // Top 15 opcodes
    std::vector<std::pair<R5900Opcode, u64>> sorted(opcode_histogram.begin(), opcode_histogram.end());
    std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    oss << "\n--- Top 15 Opcodes ---\n";
    for (std::size_t i = 0; i < std::min<std::size_t>(sorted.size(), 15); ++i) {
        oss << "  " << std::left << std::setw(12) << R5900Decoder::opcode_name(sorted[i].first)
            << " : " << sorted[i].second << "\n";
    }

    if (!unsupported_by_key.empty()) {
        oss << "\n--- Unrecognized / Non-Code Words (" << unsupported_by_key.size() << " distinct patterns) ---\n";
        std::size_t shown = 0;
        for (const auto& [key, rec] : unsupported_by_key) {
            if (++shown > 25) {
                oss << "  ... (" << (unsupported_by_key.size() - 25) << " more)\n";
                break;
            }
            oss << "  " << key << " | count=" << rec.count
                << " | sample=" << hex32(rec.sample_address) << " raw=" << hex32(rec.sample_raw) << "\n";
        }
    }
    return oss.str();
}

} // namespace tf::ps2
