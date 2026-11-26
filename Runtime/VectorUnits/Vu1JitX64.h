#pragma once

#include "../../Recompiler/Common/Types.h"
#include "../CPU/EEContext.h"
#include <array>
#include <cstddef>
#include <vector>

namespace tf::ps2 {

class VU1Unit;

// Return value of a compiled VU1 basic block:
//   bits [30:0] = number of VU1 cycles executed
//   bit  [31]   = 1 if block terminated on an E-bit delay slot
using Vu1JitBlockFn = u32 (*)(VU1Unit* vu, u32 max_steps);

struct Vu1JitOffsets {
    u32 vf = 0;
    u32 vi = 0;
    u32 acc = 0;
    u32 q = 0;
    u32 p = 0;
    u32 i_reg = 0;
    u32 r_reg = 0;
    u32 pc = 0;
    u32 xtop_val = 0;
    u32 xitop_val = 0;
    u32 data_mem = 0;
    u32 pipe_busy = 0; // u32 non-zero iff any Q/P/MAC/CLIP pipeline is active
};

struct Vu1JitHelpers {
    void (*advance_pipes)(VU1Unit* vu) = nullptr;
    void (*exec_upper_to_tmp)(VU1Unit* vu, u32 upper, Vec4f* out_val, u8* out_dest) = nullptr;
    void (*exec_lower_fallback)(VU1Unit* vu, u32 lower, u32 pc) = nullptr;
    void (*commit_mac)(VU1Unit* vu, const Vec4f* val, u32 dest) = nullptr;
    void (*commit_clip)(VU1Unit* vu, u32 fs, u32 ft) = nullptr;
    void (*exec_xgkick)(VU1Unit* vu, u32 is_reg) = nullptr;
};

// Lightweight x86-64/SSE2 Basic-Block JIT compiler for PS2 VU1 microprograms (Windows x64).
// Falls back cleanly to nullptr (C++ interpreter) on non-x64 / Android / unsupported blocks.
class Vu1JitX64 {
public:
    Vu1JitX64();
    ~Vu1JitX64();

    Vu1JitX64(const Vu1JitX64&) = delete;
    Vu1JitX64& operator=(const Vu1JitX64&) = delete;

    bool is_supported() const { return m_code_buf != nullptr; }

    void configure(const Vu1JitOffsets& offsets, const Vu1JitHelpers& helpers) {
        m_offsets = offsets;
        m_helpers = helpers;
        m_configured = true;
    }

    void invalidate_all();
    // O(1) L1 active-binding flush on VIF1 MPG overlay swaps; preserves the compiled L2 code cache!
    void clear_active_bindings() {
        if (++m_cur_gen == 0u) {
            m_slot_gen.fill(0u);
            m_cur_gen = 1u;
        }
    }

    // Returns a compiled block for VU1 instruction index `start_idx` (pc >> 3),
    // reusing cached x64 blocks across overlay swaps or compiling on first miss.
    // Returns nullptr if the block should run on the C++ interpreter.
    Vu1JitBlockFn get_or_compile(u32 start_idx, const u8* code_mem, bool has_mac_readers, bool has_clip_readers);

    u64 blocks_compiled() const { return m_blocks_compiled; }

private:
    struct L2Entry {
        u64 key_lo = 0;
        u64 key_hi = 0;
        Vu1JitBlockFn fn = nullptr;
    };
    static constexpr u32 kL2Cap = 16384u;
    static constexpr u32 kL2Mask = kL2Cap - 1u;

    Vu1JitBlockFn compile_block(u32 start_idx, const u8* code_mem, bool has_mac_readers, bool has_clip_readers);

    u8* m_code_buf = nullptr;
    std::size_t m_code_cap = 0;
    std::size_t m_write_pos = 0;
    bool m_configured = false;
    Vu1JitOffsets m_offsets{};
    Vu1JitHelpers m_helpers{};
    std::array<Vu1JitBlockFn, 2048> m_blocks{};
    std::array<u32, 2048> m_slot_gen{};
    u32 m_cur_gen = 1u;
    std::vector<L2Entry> m_l2;
    u32 m_l2_count = 0;
    u64 m_blocks_compiled = 0;
};

} // namespace tf::ps2
