#pragma once

#include "../Common/Types.h"
#include <string>

namespace tf::ps2 {

enum class R5900Opcode : u16 {
    Unknown = 0,

    // Core MIPS / R5900 Integer ALU & Shifts
    SLL, SRL, SRA, SLLV, SRLV, SRAV,
    DSLL, DSRL, DSRA, DSLL32, DSRL32, DSRA32, DSLLV, DSRLV, DSRAV,
    ADD, ADDU, SUB, SUBU, DADD, DADDU, DSUB, DSUBU,
    ADDI, ADDIU, DADDI, DADDIU,
    AND, OR, XOR, NOR,
    ANDI, ORI, XORI, LUI,
    SLT, SLTU, SLTI, SLTIU,
    MOVZ, MOVN,

    // Multiply / Divide / HI-LO / SA
    MFHI, MTHI, MFLO, MTLO,
    MFHI1, MTHI1, MFLO1, MTLO1,
    MFSA, MTSA, MTSAB, MTSAH,
    MULT, MULTU, DIV, DIVU,
    MULT1, MULTU1, DIV1, DIVU1,
    MADD, MADDU, MADD1, MADDU1,

    // Jumps & Branches
    J, JAL, JR, JALR,
    BEQ, BNE, BLEZ, BGTZ,
    BLTZ, BGEZ, BLTZAL, BGEZAL,
    BEQL, BNEL, BLEZL, BGTZL,
    BLTZL, BGEZL, BLTZALL, BGEZALL,

    // Loads & Stores (8/16/32/64/128-bit + unaligned + FPU + VU0)
    LB, LH, LWL, LW, LBU, LHU, LWR, LWU,
    LD, LDL, LDR, LQ,
    SB, SH, SWL, SW, SWR,
    SD, SDL, SDR, SQ,
    LWC1, SWC1,
    LQC2, SQC2,
    PREF, CACHE, SYNC,

    // System & Traps
    SYSCALL, BREAK,
    TGE, TGEU, TLT, TLTU, TEQ, TNE,
    TGEI, TGEIU, TLTI, TLTIU, TEQI, TNEI,

    // COP0 (System Control)
    MFC0, MTC0,
    BC0F, BC0T, BC0FL, BC0TL,
    TLBR, TLBWI, TLBWR, TLBP, ERET, EI, DI,

    // COP1 (FPU Single-Precision + EE extensions)
    MFC1, CFC1, MTC1, CTC1,
    BC1F, BC1T, BC1FL, BC1TL,
    ADD_S, SUB_S, MUL_S, DIV_S, SQRT_S, ABS_S, MOV_S, NEG_S,
    RSQRT_S, ADDA_S, SUBA_S, MULA_S, MADD_S, MSUB_S, MADDA_S, MSUBA_S,
    CVT_W_S, CVT_S_W, MAX_S, MIN_S,
    C_F_S, C_EQ_S, C_LT_S, C_LE_S,

    // MMI (R5900 128-bit Multimedia Instructions)
    // MMI0
    PADDW, PSUBW, PCGTW, PMAXW,
    PADDH, PSUBH, PCGTH, PMAXH,
    PADDB, PSUBB, PCGTB,
    PADDSW, PSUBSW, PEXTLW, PPACW,
    PADDSH, PSUBSH, PEXTLH, PPACH,
    PADDSB, PSUBSB, PEXTLB, PPACB,
    PEXT5, PPAC5,
    // MMI1
    PABSW, PCEQW, PMINW, PADSBH,
    PABSH, PCEQH, PMINH,
    PCEQB,
    PADDUW, PSUBUW, PEXTUW,
    PADDUH, PSUBUH, PEXTUH,
    PADDUB, PSUBUB, PEXTUB, QFSRV,
    // MMI2
    PMADDW, PSLLVW, PSRLVW, PMSUBW,
    PMFHI, PMFLO, PINTH,
    PMULTW, PDIVW, PCPYLD,
    PMADDH, PHMADH, PAND, PXOR,
    PMSUBH, PHMSBH,
    PEXEH, PREVH, PMULTH, PDIVBW, PEXEW, PROT3W,
    // MMI3
    PMADDUW, PSRAVW,
    PMTHI, PMTLO, PINTEH,
    PMULTUW, PDIVUW, PCPYUD,
    POR, PNOR,
    PEXCH, PCPYH, PEXCW,
    // MMI direct
    PLZCW, PMFHL, PMTHL, PSLLH, PSRLH, PSRAH, PSLLW, PSRLW, PSRAW,

    // COP2 / VU0 Macro Mode
    QMFC2, CFC2, QMTC2, CTC2,
    BC2F, BC2T, BC2FL, BC2TL,
    VU0_MACRO_OP, // Dispatched to VU0 macro executor with full raw 32-bit word
};

struct DecodedInstruction {
    u32 address = 0;
    u32 raw = 0;
    R5900Opcode op = R5900Opcode::Unknown;

    u8 rs = 0;
    u8 rt = 0;
    u8 rd = 0;
    u8 sa = 0;
    u8 fs = 0;
    u8 ft = 0;
    u8 fd = 0;
    s16 simm = 0;
    u16 uimm = 0;
    u32 target = 0; // Computed jump/branch target address

    bool has_delay_slot = false;
    bool is_branch_likely = false;
    bool is_branch = false;
    bool is_jump = false;
    bool is_call = false;
    bool is_return = false;
    bool is_indirect_jump = false;

    std::string unknown_reason;
};

} // namespace tf::ps2
