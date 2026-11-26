#pragma once

#include "../../Recompiler/Common/Types.h"
#include "../Memory/GuestMemory.h"
#include <array>
#include <cstring>
#include <functional>
#include <string>

namespace tf::ps2 {

struct alignas(16) Vec4f {
    f32 x = 0.0f;
    f32 y = 0.0f;
    f32 z = 0.0f;
    f32 w = 0.0f;

    f32& operator[](std::size_t i) { return (&x)[i & 3]; }
    const f32& operator[](std::size_t i) const { return (&x)[i & 3]; }
};

struct EEContext {
    // 32 128-bit General Purpose Registers (r[0] is always 0)
    std::array<u128, 32> r{};

    // Multiply/Divide accumulators (pipeline 0 and R5900 pipeline 1)
    u64 hi = 0;
    u64 lo = 0;
    u64 hi1 = 0;
    u64 lo1 = 0;
    u32 sa = 0; // Shift amount register (QFSRV / MTSA / MTSAB / MTSAH)

    // COP0 system control registers
    std::array<u32, 32> cop0{};
    bool interrupts_enabled = true;

    // COP1 single-precision FPU
    std::array<f32, 32> fpr{};
    f32 fpu_acc = 0.0f;
    u32 fcr0 = 0x00002E00u; // EE FPU Implementation/Revision
    u32 fcr31 = 0;          // Bit 23 = Condition flag

    // COP2 / VU0 register file (macro mode)
    std::array<Vec4f, 32> vf{};
    std::array<u32, 32> vi{}; // vi[0..15] integer regs, vi[16..31] control regs (STATUS, MAC, CLIP, R, I, Q, P...)
    Vec4f vu0_acc{};

    // Execution control
    u32 pc = 0;
    u64 cycles = 0;
    bool halted = false;
    std::string trap_message;
    u64 unhandled_opcodes = 0;

    // Attached runtime services
    GuestMemory* mem = nullptr;
    void* runtime_user_data = nullptr;

    std::function<void(EEContext&, u32 target_pc)> dispatch_fn;
    std::function<void(EEContext&, u32 code)> syscall_fn;
    std::function<void(EEContext&, u32 raw_word)> vu0_macro_fn;

    void reset() {
        r.fill(u128{});
        hi = lo = hi1 = lo1 = 0;
        sa = 0;
        cop0.fill(0);
        cop0[12] = 0x70030C01u; // Status: CU0/1/2, EDI, EIE, IM2/3, IE (PS2 EE post-BIOS default)
        cop0[15] = 0x00002E20u; // PRId: R5900 Emotion Engine
        interrupts_enabled = true;
        fpr.fill(0.0f);
        fpu_acc = 0.0f;
        fcr0 = 0x00002E00u;
        fcr31 = 0;
        vf.fill(Vec4f{});
        vf[0] = Vec4f{0.0f, 0.0f, 0.0f, 1.0f}; // VF0 is constant (0,0,0,1)
        vi.fill(0);
        vu0_acc = Vec4f{};
        pc = 0;
        cycles = 0;
        halted = false;
        trap_message.clear();
        unhandled_opcodes = 0;
    }

    // GPR helpers preserving $zero == 0
    u32 gpr_u32(u8 reg) const { return (reg == 0) ? 0u : static_cast<u32>(r[reg].lo); }
    s32 gpr_s32(u8 reg) const { return static_cast<s32>(gpr_u32(reg)); }
    u64 gpr_u64(u8 reg) const { return (reg == 0) ? 0ULL : r[reg].lo; }
    s64 gpr_s64(u8 reg) const { return static_cast<s64>(gpr_u64(reg)); }
    u128 gpr_u128(u8 reg) const { return (reg == 0) ? u128{} : r[reg]; }

    void set_gpr_s32(u8 reg, s32 v) {
        if (reg == 0) return;
        r[reg].lo = static_cast<u64>(static_cast<s64>(v)); // MIPS 32-bit ops sign-extend into 64 bits
    }
    void set_gpr_u32_sext(u8 reg, u32 v) { set_gpr_s32(reg, static_cast<s32>(v)); }
    void set_gpr_u64(u8 reg, u64 v) {
        if (reg == 0) return;
        r[reg].lo = v;
    }
    void set_gpr_s64(u8 reg, s64 v) { set_gpr_u64(reg, static_cast<u64>(v)); }
    void set_gpr_u128(u8 reg, u128 v) {
        if (reg == 0) return;
        r[reg] = v;
    }

    u32 fpr_bits(u8 reg) const {
        u32 bits = 0;
        std::memcpy(&bits, &fpr[reg & 31], sizeof(u32));
        return bits;
    }
    void set_fpr_bits(u8 reg, u32 bits) {
        std::memcpy(&fpr[reg & 31], &bits, sizeof(u32));
    }

    bool fpu_cond() const { return (fcr31 & (1u << 23)) != 0; }
    void set_fpu_cond(bool c) {
        if (c) fcr31 |= (1u << 23);
        else   fcr31 &= ~(1u << 23);
    }

    void dispatch(u32 target) {
        pc = target;
        if (dispatch_fn) {
            dispatch_fn(*this, target);
        } else {
            halted = true;
            trap_message = "Indirect dispatch with no handler to " + hex32(target);
        }
    }

    void do_syscall(u32 code) {
        if (syscall_fn) {
            syscall_fn(*this, code);
        } else {
            halted = true;
            trap_message = "SYSCALL invoked with no dispatcher (v1=" + hex32(gpr_u32(3)) + ")";
        }
    }

    void do_vu0_macro(u32 raw_word) {
        if (vu0_macro_fn) {
            vu0_macro_fn(*this, raw_word);
        } else {
            unhandled_opcodes++;
            trap_message = "VU0 macro op invoked with no VU0 handler: " + hex32(raw_word);
        }
    }

    void trap_unsupported(u32 addr, u32 raw, const char* reason) {
        unhandled_opcodes++;
        halted = true;
        trap_message = "UNSUPPORTED R5900 INSTRUCTION at " + hex32(addr) +
                       " raw=" + hex32(raw) + " (" + (reason ? reason : "unknown") + ")";
    }
};

using RecompiledFn = void(*)(EEContext&);

} // namespace tf::ps2
