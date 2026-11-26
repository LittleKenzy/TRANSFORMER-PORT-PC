#pragma once

#include "EEContext.h"
#include <functional>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace tf::ps2 {

struct Ps2Thread {
    s32 id = 0;
    u32 entry = 0;
    u32 stack = 0;
    u32 stack_size = 0;
    u32 gp_reg = 0;
    s32 priority = 0;
    u32 arg = 0;
    u32 status = 0x10; // THS_DORMANT (0x10) until StartThread
    bool running = false;
    bool suspended = false;
    bool sleeping = false;
    s32 wakeup_count = 0;
};

struct Ps2Sema {
    s32 id = 0;
    s32 count = 0;
    s32 max_count = 1;
    s32 init_count = 0;
};

struct Ps2IrqHandler {
    s32 id = 0;
    u32 cause = 0;
    u32 handler_addr = 0;
    u32 arg = 0;
};

class SyscallDispatcher {
public:
    SyscallDispatcher();
    void reset();

    void dispatch(EEContext& ctx, u32 syscall_imm);

    void set_sif_dma_callback(std::function<u32(u32 transfer_ptr, u32 count)> cb) {
        m_sif_dma_cb = std::move(cb);
    }
    void set_start_thread_callback(std::function<void(const Ps2Thread&)> cb) {
        m_start_thread_cb = std::move(cb);
    }
    void set_suspend_thread_callback(std::function<void(s32 tid)> cb) {
        m_suspend_thread_cb = std::move(cb);
    }
    void set_resume_thread_callback(std::function<void(s32 tid)> cb) {
        m_resume_thread_cb = std::move(cb);
    }
    void set_yield_callback(std::function<void(s32 current_tid)> cb) {
        m_yield_cb = std::move(cb);
    }
    void set_current_thread_id(s32 tid) { m_current_thread_id = tid; }
    s32 current_thread_id() const { return m_current_thread_id; }

    const std::unordered_map<u32, u64>& syscall_counts() const { return m_call_counts; }
    const std::unordered_map<u32, u64>& unhandled_syscalls() const { return m_unhandled_counts; }
    u32 gs_imr() const { return m_gs_imr; }
    u32 intc_mask() const { return m_intc_mask; }
    u32 dmac_mask() const { return m_dmac_mask; }
    void pulse_vsync(GuestMemory& mem);
    void fire_intc_handlers(EEContext& ctx, u32 cause);
    void fire_dmac_handlers(EEContext& ctx, u32 channel);

    u64 intc_isrs_fired() const { return m_intc_isrs_fired; }
    u64 dmac_isrs_fired() const { return m_dmac_isrs_fired; }

private:
    s32 m_next_thread_id = 1;
    s32 m_next_sema_id = 1;
    s32 m_next_handler_id = 1;
    s32 m_current_thread_id = 1;
    u32 m_heap_end = 0x01FEF000u; // 32MB EE RAM minus 64KB stack
    u32 m_stack_bottom = 0x01FEF000u;
    u32 m_gs_imr = 0xFF00u;
    u32 m_intc_mask = 0;
    u32 m_dmac_mask = 0;
    u32 m_sif_transfer_id = 1;
    u32 m_vsync_flag_addr = 0;
    u32 m_vsync_count_addr = 0;

    std::map<s32, Ps2Thread> m_threads;
    std::map<s32, Ps2Sema> m_semas;
    std::map<s32, Ps2IrqHandler> m_intc_handlers;
    std::map<s32, Ps2IrqHandler> m_dmac_handlers;

    std::unordered_map<u32, u64> m_call_counts;
    std::unordered_map<u32, u64> m_unhandled_counts;
    std::unordered_map<u32, u32> m_custom_syscalls;
    std::unordered_map<u32, u32> m_sif_regs;
    std::array<u8, 64> m_osd_config{};
    std::function<u32(u32, u32)> m_sif_dma_cb;
    std::function<void(const Ps2Thread&)> m_start_thread_cb;
    std::function<void(s32)> m_suspend_thread_cb;
    std::function<void(s32)> m_resume_thread_cb;
    std::function<void(s32)> m_yield_cb;
    bool m_in_isr = false;
    u32 m_pending_dmac_irqs = 0;
    u64 m_intc_isrs_fired = 0;
    u64 m_dmac_isrs_fired = 0;
};

} // namespace tf::ps2
