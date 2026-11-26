#include "SyscallDispatcher.h"

namespace tf::ps2 {

SyscallDispatcher::SyscallDispatcher() {
    reset();
}

void SyscallDispatcher::reset() {
    m_next_thread_id = 2;
    m_next_sema_id = 1;
    m_next_handler_id = 1;
    m_current_thread_id = 1;
    m_heap_end = kEeRamSize - 0x11000u;
    m_stack_bottom = kEeRamSize - 0x11000u;
    m_gs_imr = 0xFF00u;
    m_intc_mask = 0;
    m_dmac_mask = 0;
    m_sif_transfer_id = 1;
    m_vsync_flag_addr = 0;
    m_vsync_count_addr = 0;
    m_threads.clear();
    m_semas.clear();
    m_intc_handlers.clear();
    m_dmac_handlers.clear();
    m_call_counts.clear();
    m_unhandled_counts.clear();
    m_custom_syscalls.clear();
    m_sif_regs.clear();
    m_osd_config.fill(0);
    m_in_isr = false;
    m_pending_dmac_irqs = 0;
    m_intc_isrs_fired = 0;
    m_dmac_isrs_fired = 0;
    // Standard PS2 NTSC USA English 4:3 OSD config
    m_osd_config[0] = 0x01; // English
    // Standard PS2 SIF ready registers (IOP booted, SIFINIT + CMDINIT + BOOTEND)
    m_sif_regs[4]           = 0x00070000u; // SIF_REG_SMFLAG
    m_sif_regs[0x80000002u] = 0x00000001u; // SIF_SYSREG_CMDINIT
    m_sif_regs[0x80000004u] = 0x00070000u; // SIF_REG_SMFLAG (system alias)

    Ps2Thread main_thread;
    main_thread.id = 1;
    main_thread.priority = 0;
    main_thread.status = 0x01; // THS_RUN
    main_thread.running = true;
    m_threads[1] = main_thread;
}

void SyscallDispatcher::dispatch(EEContext& ctx, u32 /*syscall_imm*/) {
    // PS2 EE kernel convention: syscall number is in $v1 ($3); signed negative for ISR variants
    s32 raw_num = ctx.gpr_s32(3);
    u32 num = static_cast<u32>(raw_num < 0 ? -raw_num : raw_num) & 0xFFu;
    m_call_counts[num]++;

    u32 a0 = ctx.gpr_u32(4);
    u32 a1 = ctx.gpr_u32(5);
    u32 a2 = ctx.gpr_u32(6);
    u32 a3 = ctx.gpr_u32(7);

    switch (num) {
        case 0x01: // ResetEE
        case 0x02: // SetGsCrt(interlace, mode, ffmd)
            ctx.set_gpr_s32(2, 0);
            break;

        case 0x04: // Exit(code)
            ctx.halted = true;
            ctx.trap_message = "Guest exited cleanly via Exit(" + std::to_string(static_cast<s32>(a0)) +
                               ") from ra=" + hex32(ctx.gpr_u32(31));
            break;

        case 0x10: { // AddIntcHandler(cause, handler, next, arg)
            s32 id = m_next_handler_id++;
            m_intc_handlers[id] = Ps2IrqHandler{id, a0, a1, a3};
            ctx.set_gpr_s32(2, id);
            break;
        }
        case 0x11: { // RemoveIntcHandler(cause, handler_id)
            m_intc_handlers.erase(static_cast<s32>(a1));
            ctx.set_gpr_s32(2, 0);
            break;
        }
        case 0x12: { // AddDmacHandler(channel, handler, next, arg)
            s32 id = m_next_handler_id++;
            m_dmac_handlers[id] = Ps2IrqHandler{id, a0, a1, a3};
            ctx.set_gpr_s32(2, id);
            break;
        }
        case 0x13: { // RemoveDmacHandler(channel, handler_id)
            m_dmac_handlers.erase(static_cast<s32>(a1));
            ctx.set_gpr_s32(2, 0);
            break;
        }
        case 0x14: { // _EnableIntc(cause)
            bool prev = (m_intc_mask & (1u << (a0 & 31u))) != 0;
            m_intc_mask |= (1u << (a0 & 31u));
            ctx.set_gpr_s32(2, prev ? 0 : 1);
            break;
        }
        case 0x15: { // _DisableIntc(cause)
            bool prev = (m_intc_mask & (1u << (a0 & 31u))) != 0;
            m_intc_mask &= ~(1u << (a0 & 31u));
            ctx.set_gpr_s32(2, prev ? 1 : 0);
            break;
        }
        case 0x16: { // _EnableDmac(channel)
            bool prev = (m_dmac_mask & (1u << (a0 & 31u))) != 0;
            m_dmac_mask |= (1u << (a0 & 31u));
            ctx.set_gpr_s32(2, prev ? 0 : 1);
            break;
        }
        case 0x17: { // _DisableDmac(channel)
            bool prev = (m_dmac_mask & (1u << (a0 & 31u))) != 0;
            m_dmac_mask &= ~(1u << (a0 & 31u));
            ctx.set_gpr_s32(2, prev ? 1 : 0);
            break;
        }
        case 0x20: { // CreateThread(ee_thread_t* param)
            s32 id = m_next_thread_id++;
            Ps2Thread t;
            t.id = id;
            if (ctx.mem && a0 != 0) {
                t.entry      = ctx.mem->read32(a0 + 4);
                t.stack      = ctx.mem->read32(a0 + 8);
                t.stack_size = ctx.mem->read32(a0 + 12);
                t.gp_reg     = ctx.mem->read32(a0 + 16);
                t.priority   = static_cast<s32>(ctx.mem->read32(a0 + 20));
            }
            m_threads[id] = t;
            ctx.set_gpr_s32(2, id);
            break;
        }
        case 0x21: { // DeleteThread(tid)
            m_threads.erase(static_cast<s32>(a0));
            ctx.set_gpr_s32(2, 0);
            break;
        }
        case 0x22: { // StartThread(tid, arg)
            auto it = m_threads.find(static_cast<s32>(a0));
            if (it != m_threads.end()) {
                it->second.arg = a1;
                it->second.status = 0x01; // THS_RUN
                it->second.running = true;
                it->second.suspended = false;
                if (m_start_thread_cb) m_start_thread_cb(it->second);
                ctx.set_gpr_s32(2, 0);
            } else {
                ctx.set_gpr_s32(2, -1);
            }
            break;
        }
        case 0x23: // ExitThread
        case 0x24: // ExitDeleteThread
            if (m_current_thread_id != 1) {
                auto it = m_threads.find(m_current_thread_id);
                if (it != m_threads.end()) {
                    it->second.running = false;
                    it->second.status = 0x10;
                }
                ctx.halted = true;
            }
            ctx.set_gpr_s32(2, 0);
            break;
        case 0x29: { // ChangeThreadPriority(tid, prio)
            auto it = m_threads.find(static_cast<s32>(a0));
            if (it != m_threads.end()) {
                s32 old_p = it->second.priority;
                it->second.priority = static_cast<s32>(a1);
                ctx.set_gpr_s32(2, old_p);
            } else {
                ctx.set_gpr_s32(2, -1);
            }
            break;
        }
        case 0x2B: // RotateThreadReadyQueue(prio)
            if (m_yield_cb) m_yield_cb(m_current_thread_id);
            ctx.set_gpr_s32(2, 0);
            break;
        case 0x2F: // GetThreadId
            ctx.set_gpr_s32(2, m_current_thread_id);
            break;
        case 0x30: { // ReferThreadStatus(tid, ee_thread_status_t* info)
            s32 tid = (a0 == 0) ? m_current_thread_id : static_cast<s32>(a0);
            auto it = m_threads.find(tid);
            if (it != m_threads.end()) {
                const auto& t = it->second;
                if (ctx.mem && a1 != 0) {
                    ctx.mem->write32(a1 + 0, t.status);
                    ctx.mem->write32(a1 + 4, t.entry);
                    ctx.mem->write32(a1 + 8, t.stack);
                    ctx.mem->write32(a1 + 12, t.stack_size);
                    ctx.mem->write32(a1 + 16, t.gp_reg);
                    ctx.mem->write32(a1 + 20, static_cast<u32>(t.priority));
                    ctx.mem->write32(a1 + 24, static_cast<u32>(t.priority));
                    ctx.mem->write32(a1 + 28, 0);
                    ctx.mem->write32(a1 + 32, 0);
                    ctx.mem->write32(a1 + 36, 0);
                    ctx.mem->write32(a1 + 40, 0);
                    ctx.mem->write32(a1 + 44, static_cast<u32>(t.wakeup_count));
                }
                ctx.set_gpr_s32(2, static_cast<s32>(t.status));
            } else {
                ctx.set_gpr_s32(2, -1);
            }
            break;
        }
        case 0x32: { // SleepThread
            auto& t = m_threads[m_current_thread_id];
            if (t.wakeup_count > 0) t.wakeup_count--;
            if (m_yield_cb) m_yield_cb(m_current_thread_id);
            ctx.set_gpr_s32(2, 0);
            break;
        }
        case 0x33: // WakeupThread(tid)
        case 0x34: { // iWakeupThread(tid)
            auto it = m_threads.find(static_cast<s32>(a0));
            if (it != m_threads.end()) {
                it->second.wakeup_count++;
                ctx.set_gpr_s32(2, it->second.wakeup_count);
            } else {
                ctx.set_gpr_s32(2, -1);
            }
            break;
        }
        case 0x37: { // SuspendThread(tid)
            s32 tid = static_cast<s32>(a0);
            auto it = m_threads.find(tid);
            if (it != m_threads.end()) {
                it->second.suspended = true;
                it->second.status = 0x08; // THS_SUSPEND
                if (m_suspend_thread_cb) m_suspend_thread_cb(tid);
            }
            ctx.set_gpr_s32(2, 0);
            break;
        }
        case 0x39: { // ResumeThread(tid)
            s32 tid = static_cast<s32>(a0);
            auto it = m_threads.find(tid);
            if (it != m_threads.end()) {
                it->second.suspended = false;
                it->second.status = 0x01; // THS_RUN
                if (m_resume_thread_cb) m_resume_thread_cb(tid);
            }
            ctx.set_gpr_s32(2, 0);
            break;
        }
        case 0x3C: { // InitMainThread(gp, stack, stack_size, args, root) -> returns stack top
            u32 stack = a1;
            u32 stack_size = (a2 == 0 || a2 == 0xFFFFFFFFu) ? 0x10000u : a2;
            u32 sp_top = (stack == 0xFFFFFFFFu || stack == 0) ? (kEeRamSize - 0x1000u) : (stack + stack_size);
            m_stack_bottom = (sp_top > stack_size) ? (sp_top - stack_size) : 0x01F00000u;
            ctx.set_gpr_u32_sext(2, sp_top);
            break;
        }
        case 0x3D: { // InitHeap(heap, heap_size) -> returns heap end
            if (a1 == 0xFFFFFFFFu) {
                m_heap_end = m_stack_bottom;
            } else {
                m_heap_end = a0 + a1;
            }
            ctx.set_gpr_u32_sext(2, m_heap_end);
            break;
        }
        case 0x3E: // EndOfHeap
            ctx.set_gpr_u32_sext(2, m_heap_end);
            break;
        case 0x40: { // CreateSema(ee_sema_t* param)
            s32 id = m_next_sema_id++;
            Ps2Sema s;
            s.id = id;
            if (ctx.mem && a0 != 0) {
                s.count      = static_cast<s32>(ctx.mem->read32(a0 + 0));
                s.max_count  = static_cast<s32>(ctx.mem->read32(a0 + 4));
                s.init_count = static_cast<s32>(ctx.mem->read32(a0 + 8));
                if (s.max_count <= 0) s.max_count = 1;
                s.count = s.init_count;
            }
            m_semas[id] = s;
            ctx.set_gpr_s32(2, id);
            break;
        }
        case 0x41: { // DeleteSema(sid)
            m_semas.erase(static_cast<s32>(a0));
            ctx.set_gpr_s32(2, 0);
            break;
        }
        case 0x42: // SignalSema(sid)
        case 0x43: { // iSignalSema(sid)
            auto it = m_semas.find(static_cast<s32>(a0));
            if (it != m_semas.end()) {
                if (it->second.count < it->second.max_count) it->second.count++;
                ctx.set_gpr_s32(2, it->second.id);
            } else {
                ctx.set_gpr_s32(2, -1);
            }
            break;
        }
        case 0x44: { // WaitSema(sid)
            if (m_yield_cb) m_yield_cb(m_current_thread_id);
            auto it = m_semas.find(static_cast<s32>(a0));
            if (it != m_semas.end()) {
                if (it->second.count > 0) it->second.count--;
                ctx.set_gpr_s32(2, it->second.id);
            } else {
                ctx.set_gpr_s32(2, -1);
            }
            break;
        }
        case 0x45: // PollSema(sid)
        case 0x46: { // iPollSema(sid)
            auto it = m_semas.find(static_cast<s32>(a0));
            if (it != m_semas.end() && it->second.count > 0) {
                it->second.count--;
                ctx.set_gpr_s32(2, it->second.id);
            } else {
                ctx.set_gpr_s32(2, -1);
            }
            break;
        }
        case 0x4A: { // SetOsdConfigParam(void* param)
            if (ctx.mem && a0 != 0) {
                ctx.mem->read_block(a0, m_osd_config.data(), 4);
            }
            ctx.set_gpr_s32(2, 0);
            break;
        }
        case 0x4B: { // GetOsdConfigParam(void* param)
            if (ctx.mem && a0 != 0) {
                ctx.mem->write_block(a0, m_osd_config.data(), 4);
            }
            ctx.set_gpr_s32(2, 0);
            break;
        }
        case 0x4C: { // GetGsHParam
            ctx.set_gpr_s32(2, 0);
            break;
        }
        case 0x4D: { // GetGsVParam
            ctx.set_gpr_s32(2, 0);
            break;
        }
        case 0x4E: { // SetOsdConfigParam2(void* param, int size, int offset)
            if (ctx.mem && a0 != 0 && a1 > 0 && static_cast<std::size_t>(a1) + a2 <= m_osd_config.size()) {
                ctx.mem->read_block(a0, m_osd_config.data() + a2, static_cast<std::size_t>(a1));
            }
            ctx.set_gpr_s32(2, 0);
            break;
        }
        case 0x4F: { // GetOsdConfigParam2(void* param, int size, int offset)
            if (ctx.mem && a0 != 0 && a1 > 0 && static_cast<std::size_t>(a1) + a2 <= m_osd_config.size()) {
                ctx.mem->write_block(a0, m_osd_config.data() + a2, static_cast<std::size_t>(a1));
            }
            ctx.set_gpr_s32(2, 0);
            break;
        }
        case 0x64: // FlushCache(mode)
            ctx.set_gpr_s32(2, 0);
            break;
        case 0x6B: // SetupFunc2 / SifCheckInit
            ctx.set_gpr_s32(2, 1);
            break;
        case 0x70: // GsGetIMR
            ctx.set_gpr_u32_sext(2, m_gs_imr);
            break;
        case 0x71: { // GsPutIMR(imr)
            u32 prev = m_gs_imr;
            m_gs_imr = a0;
            ctx.set_gpr_u32_sext(2, prev);
            break;
        }
        case 0x73: // SetVSyncFlag(flag_ptr, count_ptr)
            m_vsync_flag_addr = a0;
            m_vsync_count_addr = a1;
            ctx.set_gpr_s32(2, 0);
            break;
        case 0x74: { // SetSyscall(num, handler_addr)
            u32 slot = a0 & 0xFFu;
            // Write into real EE Kernel Syscall Table in GuestMemory (0x800002F0 + slot*4)
            // so Sony SDK _InitSys FindAddress(0x80000000..0x80080000) locates the syscall table cleanly.
            constexpr u32 kKernelSyscallTableBase = 0x000002F0u;
            if (ctx.mem) {
                ctx.mem->write32(kKernelSyscallTableBase + slot * 4u, a1);
            }
            if (a1 == 0) {
                m_custom_syscalls.erase(slot);
            } else {
                m_custom_syscalls[slot] = a1 & 0x1FFFFFFFu;
            }
            ctx.set_gpr_s32(2, 0);
            break;
        }
        case 0x75: // SifStopDma
            ctx.set_gpr_s32(2, 0);
            break;
        case 0x76: // SifDmaStat(id) -> 0 means transfer completed
            ctx.set_gpr_s32(2, 0);
            break;
        case 0x77: { // SifSetDma(sdd, len)
            u32 tid = m_sif_dma_cb ? m_sif_dma_cb(a0, a1) : m_sif_transfer_id++;
            ctx.set_gpr_u32_sext(2, tid);
            break;
        }
        case 0x78: // SifSetDChain
            ctx.set_gpr_s32(2, 0);
            break;
        case 0x79: { // SifSetReg(reg, val)
            if (a0 == 4u || a0 == 0x80000004u) {
                // On real PS2, EE clears SMFLAG before sceSifRebootIop and IOP immediately
                // sets SIF_STAT_SIFINIT|CMDINIT|BOOTEND (0x70000) when IOPRP300.IMG finishes booting.
                m_sif_regs[4u] = 0x00070000u;
                m_sif_regs[0x80000004u] = 0x00070000u;
            } else {
                m_sif_regs[a0] = a1;
            }
            ctx.set_gpr_s32(2, 1);
            break;
        }
        case 0x7A: { // SifGetReg(reg)
            if (a0 == 4u || a0 == 0x80000004u) {
                ctx.set_gpr_u32_sext(2, 0x00070000u);
                break;
            }
            if (a0 == 0x80000002u) {
                ctx.set_gpr_u32_sext(2, 1u); // SIF_SYSREG_CMDINIT always ready after IOP boot
                break;
            }
            auto it = m_sif_regs.find(a0);
            u32 v = (it != m_sif_regs.end()) ? it->second : 0u;
            ctx.set_gpr_u32_sext(2, v);
            break;
        }
        case 0x7C: // Deci2Call: returns -1 on retail PS2 (no DevKit DECI2 host)
            ctx.set_gpr_s32(2, -1);
            break;
        case 0x7F: // GetMemorySize
            ctx.set_gpr_u32_sext(2, kEeRamSize);
            break;

        default: {
            if (auto it = m_custom_syscalls.find(num); it != m_custom_syscalls.end() && it->second != 0) {
                // Execute game-installed kernel-mode custom syscall handler natively!
                u32 saved_ra = ctx.gpr_u32(31);
                ctx.dispatch(it->second);
                ctx.set_gpr_u32_sext(31, saved_ra);
                break;
            }
            m_unhandled_counts[num]++;
            ctx.set_gpr_s32(2, 0);
            break;
        }
    }
}

void SyscallDispatcher::pulse_vsync(GuestMemory& mem) {
    if (m_vsync_flag_addr != 0) {
        mem.write32(m_vsync_flag_addr, 1);
    }
    if (m_vsync_count_addr != 0) {
        u64 cnt = mem.read64(m_vsync_count_addr);
        mem.write64(m_vsync_count_addr, cnt + 1);
    }
}

namespace {

struct SavedIsrRegs {
    std::array<u128, 32> r;
    u64 hi, lo, hi1, lo1;
    u32 sa;
    std::array<f32, 32> fpr;
    f32 fpu_acc;
    u32 fcr31;
    std::array<Vec4f, 32> vf;
    std::array<u32, 32> vi;
    Vec4f vu0_acc;
    u32 pc;

    explicit SavedIsrRegs(const EEContext& ctx)
        : r(ctx.r),
          hi(ctx.hi), lo(ctx.lo), hi1(ctx.hi1), lo1(ctx.lo1),
          sa(ctx.sa),
          fpr(ctx.fpr), fpu_acc(ctx.fpu_acc), fcr31(ctx.fcr31),
          vf(ctx.vf), vi(ctx.vi), vu0_acc(ctx.vu0_acc),
          pc(ctx.pc) {}

    void restore(EEContext& ctx) const {
        ctx.r = r;
        ctx.hi = hi; ctx.lo = lo; ctx.hi1 = hi1; ctx.lo1 = lo1;
        ctx.sa = sa;
        ctx.fpr = fpr; ctx.fpu_acc = fpu_acc; ctx.fcr31 = fcr31;
        ctx.vf = vf; ctx.vi = vi; ctx.vu0_acc = vu0_acc;
        ctx.pc = pc;
    }
};

} // namespace

void SyscallDispatcher::fire_intc_handlers(EEContext& ctx, u32 cause) {
    if (m_in_isr || ctx.halted) return;
    if ((m_intc_mask & (1u << (cause & 31u))) == 0) return;
    m_in_isr = true;
    for (const auto& [id, h] : m_intc_handlers) {
        if (h.cause == cause && h.handler_addr != 0) {
            SavedIsrRegs saved(ctx);
            ctx.set_gpr_u32_sext(29, ctx.gpr_u32(29) - 256u);
            ctx.set_gpr_u32_sext(4, cause);
            ctx.set_gpr_u32_sext(5, h.arg);
            ctx.dispatch(h.handler_addr);
            saved.restore(ctx);
            m_intc_isrs_fired++;
        }
    }
    m_in_isr = false;

    // Drain any DMAC completion IRQs queued while inside the INTC ISR
    u32 drain_guard = 0;
    while (m_pending_dmac_irqs != 0 && !ctx.halted && drain_guard++ < 64u) {
        const u32 pending = m_pending_dmac_irqs;
        m_pending_dmac_irqs = 0;
        for (u32 ch = 0; ch < 16u && !ctx.halted; ++ch) {
            if ((pending & (1u << ch)) != 0) {
                fire_dmac_handlers(ctx, ch);
            }
        }
    }
}

void SyscallDispatcher::fire_dmac_handlers(EEContext& ctx, u32 channel) {
    if (ctx.halted) return;
    const u32 ch_bit = 1u << (channel & 31u);
    if ((m_dmac_mask & ch_bit) == 0) return;
    if (m_in_isr) {
        m_pending_dmac_irqs |= ch_bit;
        return;
    }
    m_in_isr = true;
    u32 current_mask = ch_bit;
    u32 drain_guard = 0;
    while (current_mask != 0 && !ctx.halted && drain_guard++ < 64u) {
        for (u32 ch = 0; ch < 16u && !ctx.halted; ++ch) {
            if ((current_mask & (1u << ch)) == 0) continue;
            if ((m_dmac_mask & (1u << ch)) == 0) continue;
            for (const auto& [id, h] : m_dmac_handlers) {
                if (h.cause == ch && h.handler_addr != 0) {
                    SavedIsrRegs saved(ctx);
                    ctx.set_gpr_u32_sext(29, ctx.gpr_u32(29) - 256u);
                    ctx.set_gpr_u32_sext(4, ch);
                    ctx.set_gpr_u32_sext(5, h.arg);
                    ctx.dispatch(h.handler_addr);
                    saved.restore(ctx);
                    m_dmac_isrs_fired++;
                }
            }
        }
        current_mask = m_pending_dmac_irqs;
        m_pending_dmac_irqs = 0;
    }
    m_in_isr = false;
}

} // namespace tf::ps2
