#pragma once

#include "../CPU/EEContext.h"
#include "Vu1JitX64.h"
#include <array>
#include <functional>
#include <string>
#include <vector>

namespace tf::ps2 {

// Vector Unit 1 (VU1): Complete PS2 micromode interpreter executing original VU1 microprograms
// (.vutext / .DVP.overlay.* uploaded via VIF1 MPG) with XGKICK output to GIF.
class VU1Unit {
public:
    VU1Unit();
    void reset();

    void attach_memory(u8* code_mem_16k, u8* data_mem_16k) {
        m_code_mem = code_mem_16k;
        m_data_mem = data_mem_16k;
        mark_code_dirty();
    }

    void set_xgkick_callback(std::function<void(const u8* data, std::size_t max_bytes)> cb) {
        m_xgkick_cb = std::move(cb);
    }

    void set_vif_regs(u16 top_val, u16 itop_val) {
        xtop_val = top_val & 0x3FFu;
        xitop_val = itop_val & 0x3FFu;
    }

    // Executes VU1 microprogram starting at byte offset `start_addr_bytes` until E-bit + delay slot.
    u32 run_microprogram(u32 start_addr_bytes, u32 max_steps = 65536);
    void invalidate_code_cache();
    void mark_code_dirty();
    void on_code_written(u32 start_byte, u32 byte_len);
    u64  jit_blocks_compiled() const { return m_jit.blocks_compiled(); }
    bool enable_jit = true;

    std::array<Vec4f, 32> vf{};
    std::array<u16, 16>   vi{};
    Vec4f acc{};
    f32 q = 1.0f;
    f32 p = 0.0f;
    f32 i_reg = 0.0f;
    u32 r_reg = 0x3F800000u;
    u32 pc = 0;
    u32 mac_flags = 0;
    u32 status_flags = 0;
    u32 clip_flags = 0;
    u16 xtop_val = 0;
    u16 xitop_val = 0;
    u64 micro_steps_executed = 0;
    u64 xgkick_count = 0;
    std::vector<std::string> debug_log;

private:
    struct DecodedVu1Slot {
        u64 raw = 0xFFFFFFFFFFFFFFFFULL;
        u8  u_op = 0;
        u8  u_dest = 0;
        u8  u_fs = 0;
        u8  u_ft = 0;
        u8  u_fd = 0;
        u8  u_bc = 0;
        u8  flags = 0; // bit0: i_bit, bit1: e_bit
        u8  l_op = 0;
        u8  l_dest = 0;
        u8  l_is = 0;
        u8  l_it = 0;
        u8  l_id = 0;
        u8  l_aux = 0; // bits 0..1: fsf, bits 2..3: ftf, bits 4..5: field_from_dest
        s16 l_imm11 = 0;
        u16 l_imm15 = 0;
        u32 l_imm24 = 0;
    };

    struct UpperWrite {
        bool write_vf = false;
        bool write_acc = false;
        bool write_mac = false;
        bool write_clip = false;
        u8 reg = 0;
        u8 dest = 0;
        u32 mac = 0;
        u8 clip = 0;
        Vec4f val{};
    };
    static void decode_slot(DecodedVu1Slot& slot, u64 raw);
    void refresh_code_analysis();
    void exec_xgkick(u8 is_reg);
    void exec_xgkick_addr(u32 addr_bytes);
    void exec_upper_decoded(const DecodedVu1Slot& slot, UpperWrite& uw);
    void exec_lower_decoded(const DecodedVu1Slot& slot, bool& branched, u32& branch_target, u8 prev_alu_reg, u16 prev_alu_old);
    void flush_q();
    void flush_p();
    void flush_xgkick();
    void advance_pipes();
    void flush_all_pipes();
    inline void update_pipe_busy() {
        m_pipe_busy = (m_mac_valid_mask | m_clip_valid_mask | m_q_delay | m_p_delay | m_xgkick_delay) ? 1u : 0u;
    }

    static void jit_advance_pipes(VU1Unit* vu);
    static void jit_exec_upper_to_tmp(VU1Unit* vu, u32 upper_word, Vec4f* out_val, u8* out_dest);
    static void jit_exec_lower_fallback(VU1Unit* vu, u32 lower_word, u32 cur_pc);
    static void jit_commit_mac(VU1Unit* vu, const Vec4f* val, u32 dest_mask);
    static void jit_commit_clip(VU1Unit* vu, u32 fs, u32 ft);
    static void jit_exec_xgkick(VU1Unit* vu, u32 is_reg);

    // Hardware pipeline latency state: FDIV/EFU (Q/P), 4-stage FMAC (MAC/STATUS/CLIP), and 1-cycle XGKICK delay
    f32 m_q_pending = 1.0f;
    u32 m_q_delay = 0;
    u32 m_q_status_pending = 0;
    f32 m_p_pending = 0.0f;
    u32 m_p_delay = 0;
    u32 m_xgkick_addr = 0;
    u32 m_xgkick_delay = 0;
    std::array<u32, 4> m_mac_pipe{};
    std::array<u8, 4>  m_clip_pipe{};
    u8  m_mac_valid_mask = 0;
    u8  m_clip_valid_mask = 0;
    u32 m_pipe_busy = 0;
    u8  m_vi_alu_reg = 0;
    u16 m_vi_alu_old = 0;

    bool m_code_dirty = true;
    bool m_analysis_initialized = false;
    bool m_jit_bindings_dirty = false;
    bool m_has_mac_readers = true;
    bool m_has_clip_readers = true;
    u32  m_mac_reader_count = 0;
    u32  m_clip_reader_count = 0;
    std::array<u8, 2048> m_slot_reader_flags{};

    u8* m_code_mem = nullptr;
    u8* m_data_mem = nullptr;
    Vu1JitX64 m_jit;
    std::array<DecodedVu1Slot, 2048> m_decode_cache{};
    std::array<u8, 16384> m_xgkick_wrap_buf{};
    std::function<void(const u8*, std::size_t)> m_xgkick_cb;
};

} // namespace tf::ps2
