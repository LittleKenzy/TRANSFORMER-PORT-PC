#include "TestFramework.h"
#include "../Recompiler/Decoder/R5900Decoder.h"

using namespace tf::ps2;

TF_TEST(DecoderTests, CoreIntegerBranchesAndMmi) {
    // ADDIU $sp, $sp, -32 -> 0x27BDFFE0
    auto ins = R5900Decoder::decode(0x00100000u, 0x27BDFFE0u);
    TF_EXPECT(ins.op == R5900Opcode::ADDIU);
    TF_EXPECT_EQ(ins.rs, u8{29});
    TF_EXPECT_EQ(ins.rt, u8{29});
    TF_EXPECT_EQ(ins.simm, s16{-32});

    // JAL 0x00100200 -> opcode=3, target=0x00100200 >> 2 = 0x00040080
    ins = R5900Decoder::decode(0x00100004u, (3u << 26) | 0x00040080u);
    TF_EXPECT(ins.op == R5900Opcode::JAL);
    TF_EXPECT(ins.is_call);
    TF_EXPECT(ins.has_delay_slot);
    TF_EXPECT_EQ(ins.target, 0x00100200u);

    // JR $ra -> 0x03E00008
    ins = R5900Decoder::decode(0x00100008u, 0x03E00008u);
    TF_EXPECT(ins.op == R5900Opcode::JR);
    TF_EXPECT(ins.is_return);
    TF_EXPECT(ins.has_delay_slot);

    // PADDW $v0, $a0, $a1 -> primary=0x1C, rs=4, rt=5, rd=2, sa=0(PADDW), funct=0x08(MMI0)
    u32 paddw = (0x1Cu << 26) | (4u << 21) | (5u << 16) | (2u << 11) | (0u << 6) | 0x08u;
    ins = R5900Decoder::decode(0x0010000Cu, paddw);
    TF_EXPECT(ins.op == R5900Opcode::PADDW);

    // ADD.S $f0, $f1, $f2 -> primary=0x11, rs=0x10, ft=2, fs=1, fd=0, funct=0x00
    u32 adds = (0x11u << 26) | (0x10u << 21) | (2u << 16) | (1u << 11) | (0u << 6) | 0x00u;
    ins = R5900Decoder::decode(0x00100010u, adds);
    TF_EXPECT(ins.op == R5900Opcode::ADD_S);

    // VU0 VADD.xyzw -> primary=0x12, rs=0x1F (bit4=1, dest=0xF), ft=2, fs=1, fd=3, funct=0x28
    u32 vadd = (0x12u << 26) | (0x1Fu << 21) | (2u << 16) | (1u << 11) | (3u << 6) | 0x28u;
    ins = R5900Decoder::decode(0x00100014u, vadd);
    TF_EXPECT(ins.op == R5900Opcode::VU0_MACRO_OP);

    // Unknown primary opcode 0x3D reports explicit diagnostic reason
    ins = R5900Decoder::decode(0x00100018u, (0x3Du << 26));
    TF_EXPECT(ins.op == R5900Opcode::Unknown);
    TF_EXPECT(!ins.unknown_reason.empty());
}
