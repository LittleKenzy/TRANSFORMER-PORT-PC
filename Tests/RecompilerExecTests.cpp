#include "TestFramework.h"
#include "../Recompiler/ELF/ElfParser.h"
#include "../Recompiler/Analyzer/FunctionAnalyzer.h"
#include "../Recompiler/CodeGenerator/CppCodeGenerator.h"
#include "../Runtime/RuntimeHost.h"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace {

using namespace tf::ps2;

// MIPS R5900 instruction encoders for exact end-to-end static recompilation verification
constexpr u32 enc_r(u8 op, u8 rs, u8 rt, u8 rd, u8 sa, u8 fn) {
    return (u32(op) << 26) | (u32(rs) << 21) | (u32(rt) << 16) | (u32(rd) << 11) | (u32(sa) << 6) | u32(fn);
}
constexpr u32 enc_i(u8 op, u8 rs, u8 rt, s16 imm) {
    return (u32(op) << 26) | (u32(rs) << 21) | (u32(rt) << 16) | (u32(u16(imm)));
}
constexpr u32 enc_j(u8 op, u32 target) {
    return (u32(op) << 26) | ((target >> 2) & 0x03FFFFFFu);
}

std::vector<u8> build_test_elf(u32 base_addr, const std::vector<u32>& words) {
    u32 code_size = static_cast<u32>(words.size() * 4);
    u32 code_off = 128;
    std::vector<u8> elf(code_off + code_size, 0);

    elf[0] = 0x7F; elf[1] = 'E'; elf[2] = 'L'; elf[3] = 'F';
    elf[4] = 1; elf[5] = 1; elf[6] = 1;
    write_le16(elf.data() + 16, 2); // ET_EXEC
    write_le16(elf.data() + 18, 8); // EM_MIPS
    write_le32(elf.data() + 20, 1);
    write_le32(elf.data() + 24, base_addr);
    write_le32(elf.data() + 28, 52);
    write_le16(elf.data() + 40, 52);
    write_le16(elf.data() + 42, 32);
    write_le16(elf.data() + 44, 1);

    write_le32(elf.data() + 52 + 0, 1); // PT_LOAD
    write_le32(elf.data() + 52 + 4, code_off);
    write_le32(elf.data() + 52 + 8, base_addr);
    write_le32(elf.data() + 52 + 12, base_addr);
    write_le32(elf.data() + 52 + 16, code_size);
    write_le32(elf.data() + 52 + 20, code_size);
    write_le32(elf.data() + 52 + 24, 5); // RX

    for (std::size_t i = 0; i < words.size(); ++i) {
        write_le32(elf.data() + code_off + i * 4, words[i]);
    }
    return elf;
}

} // namespace

TF_TEST(RecompilerExecTests, StaticRecompileCompileAndExecuteNativeCode) {
#ifdef _WIN32
    constexpr u32 kBase = 0x00100000u;
    constexpr u32 kHelperAddr = kBase + 20 * 4;

    // Function 0 at 0x00100000:
    //   Computes fib(a0) using branch delay slots, calls helper via JAL (with delay slot arg setup),
    //   tests BEQL branch-likely nullification, tests MMI PADDW, tests FPU ADD.S, and returns in $v0.
    //
    //   0: ADDIU $t0, $zero, 0      ; a = 0
    //   1: ADDIU $t1, $zero, 1      ; b = 1
    //   2: ADDIU $t2, $a0, 0        ; i = n (expect n=10 -> fib(10)=55)
    //   3: BEQ   $t2, $zero, +5     ; loop_end if i == 0 (target = ins 9)
    //   4: NOP
    //   5: ADDU  $t3, $t0, $t1      ; next = a + b
    //   6: ADDU  $t0, $t1, $zero    ; a = b
    //   7: BNE   $t2, $t4_one, -3   ; wait, let's decrement i cleanly:
    std::vector<u32> code = {
        // 0..2: setup fib(a0): t0=0, t1=1, t2=a0
        enc_i(0x09, 0, 8, 0),            // 0: addiu $t0, $zero, 0
        enc_i(0x09, 0, 9, 1),            // 1: addiu $t1, $zero, 1
        enc_i(0x09, 4, 10, 0),           // 2: addiu $t2, $a0, 0
        // 3..8: loop
        enc_i(0x04, 10, 0, 5),           // 3: beq $t2, $zero, -> 9
        0x00000000u,                     // 4: nop (delay slot)
        enc_r(0x00, 8, 9, 11, 0, 0x21),  // 5: addu $t3, $t0, $t1
        enc_r(0x00, 9, 0, 8, 0, 0x21),   // 6: addu $t0, $t1, $zero
        enc_i(0x05, 10, 0, -5),          // 7: bne $t2, $zero -> 3 (Wait: decrements in delay slot! Let's check:
                                         //    if $t2 starts at 10, BNE checks $t2 != 0 BEFORE delay slot, then delay slot does $t2 -= 1!)
        enc_i(0x09, 10, 10, -1),         // 8: delay slot: addiu $t2, $t2, -1  -- wait, $t1 wasn't updated! Fix order below:
    };

    // Clean, exact instruction sequence:
    code.clear();
    // 0: addiu $t0, $zero, 0     (a = 0)
    code.push_back(enc_i(0x09, 0, 8, 0));
    // 1: addiu $t1, $zero, 1     (b = 1)
    code.push_back(enc_i(0x09, 0, 9, 1));
    // 2: addiu $t2, $a0, 0       (count = a0)
    code.push_back(enc_i(0x09, 4, 10, 0));
    // 3: loop: beq $t2, $zero, +5 (-> index 9)
    code.push_back(enc_i(0x04, 10, 0, 5));
    // 4: nop
    code.push_back(0x00000000u);
    // 5: addu $t3, $t0, $t1      (next = a + b)
    code.push_back(enc_r(0x00, 8, 9, 11, 0, 0x21));
    // 6: addu $t0, $t1, $zero    (a = b)
    code.push_back(enc_r(0x00, 9, 0, 8, 0, 0x21));
    // 7: bne $t2, $zero, -5      (target = (7+1) - 5 = 3; delay slot updates b and count? Let's put b=next in 7 and dec in delay slot)
    code.pop_back(); // redo 6..8
    // 6: addu $t0, $t1, $zero
    code.push_back(enc_r(0x00, 9, 0, 8, 0, 0x21));
    // 7: addu $t1, $t3, $zero
    code.push_back(enc_r(0x00, 11, 0, 9, 0, 0x21));
    // 8: addiu $t2, $t2, -1
    code.push_back(enc_i(0x09, 10, 10, -1));
    // 9: bne $t2, $zero, -5  (target = (9+1)-5 = 5!)
    code.push_back(enc_i(0x05, 10, 0, -5));
    // 10: nop
    code.push_back(0x00000000u);
    // Patch index 3 beq offset: from 3 -> 11 is 11 - 4 = +7
    code[3] = enc_i(0x04, 10, 0, 7);

    // 11: Test BEQL (branch-likely NOT taken -> MUST nullify delay slot!)
    //     $t4 = 100; beql $t0, $zero, +2 (not taken since fib(10)=55 != 0); delay slot addiu $t4,$t4,999 (MUST BE SKIPPED!)
    code.push_back(enc_i(0x09, 0, 12, 100));          // 11: addiu $t4, $zero, 100
    code.push_back(enc_i(0x14, 8, 0, 2));             // 12: beql $t0, $zero, +2
    code.push_back(enc_i(0x09, 12, 12, 999));         // 13: delay slot (must NOT run!)

    // 14: Test JAL to helper at index 20 with delay slot setting $a0 = $t0
    code.push_back(enc_j(0x03, kHelperAddr));         // 14: jal helper
    code.push_back(enc_r(0x00, 8, 0, 4, 0, 0x21));    // 15: delay slot: addu $a0, $t0, $zero

    // 16: Test MMI PADDW $s0, $s1, $s2 (lanes set by test harness before call)
    code.push_back(enc_r(0x1C, 17, 18, 16, 0, 0x08)); // 16: paddw $s0, $s1, $s2

    // 17: Test FPU ADD.S $f3, $f1, $f2
    code.push_back(enc_r(0x11, 0x10, 2, 1, 3, 0x00)); // 17: add.s $f3, $f1, $f2

    // 18: jr $ra
    code.push_back(enc_r(0x00, 31, 0, 0, 0, 0x08));   // 18: jr $ra
    // 19: delay slot: addu $v0, $v0, $t4  (helper returned 55*2=110; + $t4(100) = 210!)
    code.push_back(enc_r(0x00, 2, 12, 2, 0, 0x21));   // 19: addu $v0, $v0, $t4

    // 20 (kHelperAddr): helper(a0) -> returns a0 * 2 in $v0
    code.push_back(enc_r(0x00, 4, 4, 2, 0, 0x21));    // 20: addu $v0, $a0, $a0
    code.push_back(enc_r(0x00, 31, 0, 0, 0, 0x08));   // 21: jr $ra
    code.push_back(0x00000000u);                      // 22: nop

    auto elf_bytes = build_test_elf(kBase, code);
    Ps2ElfImage elf;
    std::string err;
    TF_EXPECT(ElfParser::parse(elf_bytes, elf, err));

    auto analysis = FunctionAnalyzer::analyze(elf);
    TF_EXPECT_EQ(analysis.functions.size(), std::size_t{2});
    TF_EXPECT_EQ(analysis.coverage.unknown_instructions, u64{0});

    namespace fs = std::filesystem;
    fs::path out_dir = fs::temp_directory_path() / "tf_recomp_exec_test";
    fs::remove_all(out_dir);

    CodegenOptions cg_opts;
    cg_opts.emit_disassembly_comments = true;
    auto cg = CppCodeGenerator::generate_project(analysis, out_dir.string(), cg_opts);
    TF_EXPECT(cg.success);
    TF_EXPECT_EQ(cg.functions_emitted, std::size_t{2});
    TF_EXPECT_EQ(cg.instructions_trapped_unsupported, u64{0});

    // Write shared DLL export shim
    fs::path shim_path = out_dir / "test_shim.cpp";
    {
        std::ofstream sf(shim_path);
        sf << "#include \"recomp_module.h\"\n"
           << "extern \"C\" __declspec(dllexport) void register_fns(void* m) {\n"
           << "    auto* tbl = static_cast<std::unordered_map<tf::ps2::u32, tf::ps2::RecompiledFn>*>(m);\n"
           << "    tf::recomp::register_recompiled_functions(*tbl);\n"
           << "}\n";
    }

    // Compile the generated C++20 code into a REAL native x64 DLL with g++!
    fs::path dll_path = out_dir / "recomp_exec_test.dll";
    std::ostringstream cmd;
    cmd << "g++ -std=c++20 -O2 -shared -I\"" << TF_SOURCE_ROOT << "\" "
        << "\"" << (out_dir / "recomp_shard_0.cpp").string() << "\" "
        << "\"" << (out_dir / "recomp_dispatch.cpp").string() << "\" "
        << "\"" << shim_path.string() << "\" "
        << "\"" << (fs::path(TF_SOURCE_ROOT) / "Runtime/Memory/GuestMemory.cpp").string() << "\" "
        << "-o \"" << dll_path.string() << "\"";

    int rc = std::system(cmd.str().c_str());
    TF_EXPECT_EQ(rc, 0);
    TF_EXPECT(fs::exists(dll_path));

    // Load the natively compiled DLL and EXECUTE the statically recompiled R5900 code!
    HMODULE mod = LoadLibraryA(dll_path.string().c_str());
    TF_EXPECT(mod != nullptr);
    auto reg_fn = reinterpret_cast<void(*)(void*)>(reinterpret_cast<void*>(GetProcAddress(mod, "register_fns")));
    TF_EXPECT(reg_fn != nullptr);

    RuntimeHost rt;
    TF_EXPECT(rt.load_elf(elf, err));
    reg_fn(&rt.function_table());

    // Set inputs: $a0 = 10 (fib(10)=55), $s1 = {1,2,3,4}, $s2 = {10,20,30,40}, $f1 = 1.5f, $f2 = 2.25f
    rt.cpu.set_gpr_s32(4, 10);
    u128 v1{}, v2{};
    v1.set_uw(0, 1); v1.set_uw(1, 2); v1.set_uw(2, 3); v1.set_uw(3, 4);
    v2.set_uw(0, 10); v2.set_uw(1, 20); v2.set_uw(2, 30); v2.set_uw(3, 40);
    rt.cpu.set_gpr_u128(17, v1);
    rt.cpu.set_gpr_u128(18, v2);
    rt.cpu.fpr[1] = 1.5f;
    rt.cpu.fpr[2] = 2.25f;

    rt.call_guest(kBase);

    // Verify exact results:
    // fib(10) = 55 -> helper(55) = 110 -> + $t4 (100, proving BEQL nullified +999!) = 210
    TF_EXPECT(!rt.cpu.halted);
    TF_EXPECT_EQ(rt.cpu.gpr_s32(2), 210);
    TF_EXPECT_EQ(rt.cpu.gpr_s32(12), 100);
    // MMI PADDW result in $s0 ($16)
    u128 s0 = rt.cpu.gpr_u128(16);
    TF_EXPECT_EQ(s0.uw(0), 11u);
    TF_EXPECT_EQ(s0.uw(1), 22u);
    TF_EXPECT_EQ(s0.uw(2), 33u);
    TF_EXPECT_EQ(s0.uw(3), 44u);
    // FPU ADD.S result in $f3
    TF_EXPECT(std::fabs(rt.cpu.fpr[3] - 3.75f) < 1e-6f);

    FreeLibrary(mod);
#endif
}

TF_TEST(RecompilerExecTests, UnsupportedOpcodeGeneratesExplicitTrap) {
    DiscoveredFunction fn;
    fn.start_addr = 0x00100000u;
    fn.end_addr = 0x00100004u;
    fn.name = "bad_op_fn";
    fn.instructions.push_back(R5900Decoder::decode(0x00100000u, 0x3Du << 26));

    std::map<u32, std::string> names{{0x00100000u, "bad_op_fn_00100000"}};
    CodegenOptions opts;
    CodegenResult stats;
    std::string code = CppCodeGenerator::emit_function(fn, names, opts, stats);

    TF_EXPECT_EQ(stats.instructions_trapped_unsupported, u64{1});
    TF_EXPECT(code.find("ctx.trap_unsupported") != std::string::npos);
}
