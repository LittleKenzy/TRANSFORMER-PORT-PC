#include "RuntimeHost.h"
#include "PerfCounters.h"
#include "CPU/R5900Interpreter.h"
#include <algorithm>
#include <chrono>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#ifndef CREATE_WAITABLE_TIMER_HIGH_RESOLUTION
#define CREATE_WAITABLE_TIMER_HIGH_RESOLUTION 0x00000002
#endif
#endif

namespace tf::ps2 {

namespace {

struct GuestHaltSignal {};

thread_local bool t_in_guest_execution = false;

struct GuestExecScope {
    bool prev;
    GuestExecScope() : prev(t_in_guest_execution) { t_in_guest_execution = true; }
    ~GuestExecScope() { t_in_guest_execution = prev; }
};

struct CallDepthGuard {
    u32& depth;
    u32& mirror;
    CallDepthGuard(u32& d, u32& m) : depth(d), mirror(m) {
        ++depth;
        mirror = depth;
    }
    ~CallDepthGuard() {
        --depth;
        mirror = depth;
    }
};

} // namespace

RuntimeHost::RuntimeHost() {
#ifdef _WIN32
    m_pacer_timer = CreateWaitableTimerExW(
        nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
#endif
    reset();
    m_iop_ticker_stop.store(false, std::memory_order_release);
    m_iop_ticker_paused.store(false, std::memory_order_release);
    m_iop_ticker = std::thread([this]() {
#ifdef _WIN32
        HANDLE hi_timer = CreateWaitableTimerExW(
            nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
#endif
        while (!m_iop_ticker_stop.load(std::memory_order_acquire)) {
            if (m_iop_ticker_paused.load(std::memory_order_acquire)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            m_iop_ticker_active.store(true, std::memory_order_release);
            bool did_tick = false;
            if (!m_iop_ticker_paused.load(std::memory_order_acquire) && iop.has_heartbeat()) {
                iop.tick_heartbeat();
                did_tick = true;
            }
            m_iop_ticker_active.store(false, std::memory_order_release);
            if (did_tick) {
                // Fast-path during initial 0.15s IOP/NuSound3M boot handshake (before first 10 GS frames);
                // switch to quiet 500us high-res timer once video starts so steady-state CPU usage is <0.2%.
                if (gs.frames_presented < 10u) {
                    std::this_thread::yield();
                } else {
#ifdef _WIN32
                    if (hi_timer) {
                        LARGE_INTEGER due{};
                        due.QuadPart = -5000LL; // 500 us (2,000 Hz)
                        if (SetWaitableTimer(hi_timer, &due, 0, nullptr, nullptr, FALSE)) {
                            WaitForSingleObject(hi_timer, 2);
                        } else {
                            std::this_thread::sleep_for(std::chrono::milliseconds(1));
                        }
                    } else {
                        std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    }
#else
                    std::this_thread::sleep_for(std::chrono::microseconds(500));
#endif
                }
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
        }
#ifdef _WIN32
        if (hi_timer) {
            CloseHandle(hi_timer);
        }
#endif
    });
}

RuntimeHost::~RuntimeHost() {
    gs.set_present_callback(nullptr);
    gs.set_frame_pacing_callback(nullptr);
    set_jit_input_callback(nullptr);
    stop_workers();
    m_iop_ticker_paused.store(true, std::memory_order_release);
    m_iop_ticker_stop.store(true, std::memory_order_release);
    if (m_iop_ticker.joinable()) {
        m_iop_ticker.join();
    }
#ifdef _WIN32
    if (m_pacer_timer) {
        CloseHandle(static_cast<HANDLE>(m_pacer_timer));
        m_pacer_timer = nullptr;
    }
#endif
}

void RuntimeHost::set_jit_input_callback(std::function<void()> cb) {
    std::lock_guard<std::mutex> lk(m_jit_input_mtx);
    m_jit_input_cb = std::move(cb);
}

void RuntimeHost::invoke_jit_input() {
    std::lock_guard<std::mutex> lk(m_jit_input_mtx);
    if (m_jit_input_cb) {
        m_jit_input_cb();
    }
}

void RuntimeHost::pace_to_60hz() {
    using clock = std::chrono::steady_clock;
    constexpr auto kFramePeriod = std::chrono::microseconds(16667); // 60.0 Hz NTSC
    const auto now = clock::now();

    if (m_vblank_pacer_init) {
        const auto work_us = std::chrono::duration_cast<std::chrono::microseconds>(now - m_frame_work_start_tp).count();
        if (work_us >= 0 && work_us < 1000000) {
            m_last_frame_work_us.store(static_cast<u32>(work_us), std::memory_order_relaxed);
        }
    }

    if (!m_vblank_pacer_init) {
        m_next_vblank_tp = now + kFramePeriod;
        m_frame_work_start_tp = now;
        m_vblank_pacer_init = true;
        return;
    }

    if (now < m_next_vblank_tp) {
        const auto rem_us = std::chrono::duration_cast<std::chrono::microseconds>(m_next_vblank_tp - now).count();
        if (rem_us > 700) {
#ifdef _WIN32
            if (m_pacer_timer) {
                LARGE_INTEGER due{};
                due.QuadPart = -static_cast<LONGLONG>((rem_us - 600) * 10); // 100ns units; wake 600us early for tight spin
                if (SetWaitableTimer(static_cast<HANDLE>(m_pacer_timer), &due, 0, nullptr, nullptr, FALSE)) {
                    WaitForSingleObject(static_cast<HANDLE>(m_pacer_timer), 25);
                } else {
                    std::this_thread::sleep_for(std::chrono::microseconds(rem_us - 600));
                }
            } else {
                std::this_thread::sleep_for(std::chrono::microseconds(rem_us - 600));
            }
#else
            std::this_thread::sleep_for(std::chrono::microseconds(rem_us - 600));
#endif
        }
        while (!cpu.halted && clock::now() < m_next_vblank_tp) {
            std::this_thread::yield();
        }
    }

    const auto woke = clock::now();
    m_frame_work_start_tp = woke;
    m_next_vblank_tp += kFramePeriod;
    if (m_next_vblank_tp < woke - std::chrono::milliseconds(50)) {
        m_next_vblank_tp = woke + kFramePeriod;
    }
}

void RuntimeHost::request_halt() {
    cpu.halted = true;
    {
        std::lock_guard<std::mutex> lk(m_sched_mtx);
        m_sched_shutdown = true;
        for (auto& w : m_workers) {
            w->ctx.halted = true;
        }
    }
    m_sched_cv.notify_all();
}

void RuntimeHost::stop_workers() {
    {
        std::lock_guard<std::mutex> lk(m_sched_mtx);
        m_sched_shutdown = true;
        for (auto& w : m_workers) {
            w->ctx.halted = true;
        }
    }
    m_sched_cv.notify_all();
    for (auto& w : m_workers) {
        if (w->host_thread.joinable()) {
            w->host_thread.join();
        }
    }
    m_workers.clear();
    m_rr_worker_idx = 0;
    m_active_tid = 1;
    m_active_ctx = &cpu;
    m_sched_shutdown = false;
}

void RuntimeHost::reset() {
    m_in_reset.store(true, std::memory_order_release);
    m_iop_ticker_paused.store(true, std::memory_order_release);
    while (m_iop_ticker_active.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    std::lock_guard<std::mutex> iop_lk(m_iop_tick_mtx);
    stop_workers();
    memory.reset();
    cpu.reset();
    m_active_ctx = &cpu;
    m_active_call_depth = &m_main_call_depth;
    syscalls.reset();
    vu0.reset(cpu);
    vu1.reset();
    dmac.reset();
    vif0.reset();
    vif1.reset();
    gif.reset();
    gs.reset();
    iop.reset();
    spu2.reset();
    pad0.reset();
    m_mmio_reads.clear();
    m_fn_call_counts.clear();
    m_fast_fn_synced_size = 0;
    m_bound_sids.clear();
    m_rpc_trace.clear();
    m_server_to_sid.clear();
    m_next_sif_dma_id = 1;
    m_timers_count.fill(0);
    m_timers_mode.fill(0);
    m_intc_stat = 0;
    m_intc_mask = 0;
    m_gif_mode = 0;
    m_vblank_pulsed_this_frame = false;
    m_vblank_pacer_init = false;
    m_paced_this_frame = false;
    m_worker_saturated_slice = false;
    m_last_loading_input_tp = {};
    m_last_frame_work_us.store(0, std::memory_order_relaxed);
    indirect_calls_resolved = 0;
    indirect_calls_missed = 0;
    fallback_steps_executed = 0;
    total_static_calls = 0;
    call_depth = 0;
    m_main_call_depth = 0;
    last_missed_target = 0;

    wire_subsystems();
    m_in_reset.store(false, std::memory_order_release);
    m_iop_ticker_paused.store(false, std::memory_order_release);
}

void RuntimeHost::wire_subsystems() {
    cpu.mem = &memory;
    cpu.runtime_user_data = this;

    // Wire CPU callbacks
    cpu.dispatch_fn = [this](EEContext& c, u32 target_pc) {
        if (c.halted || cpu.halted) {
            if (t_in_guest_execution) throw GuestHaltSignal{};
            return;
        }
        call_guest_on(c, target_pc);
    };
    cpu.syscall_fn = [this](EEContext& c, u32 code) {
        if (c.halted || cpu.halted) {
            if (t_in_guest_execution) throw GuestHaltSignal{};
            return;
        }
        syscalls.dispatch(c, code);
        if (c.halted || cpu.halted) {
            if (t_in_guest_execution) throw GuestHaltSignal{};
        }
    };
    cpu.vu0_macro_fn = [this](EEContext& c, u32 raw) {
        vu0.execute_macro(c, raw);
    };

    // Wire cooperative PS2 EE multi-threading callbacks
    syscalls.set_start_thread_callback([this](const Ps2Thread& t) {
        start_worker_thread(t);
    });
    syscalls.set_suspend_thread_callback([this](s32 tid) {
        std::lock_guard<std::mutex> lk(m_sched_mtx);
        for (auto& w : m_workers) {
            if (w->tid == tid) w->suspended = true;
        }
    });
    syscalls.set_resume_thread_callback([this](s32 tid) {
        std::lock_guard<std::mutex> lk(m_sched_mtx);
        for (auto& w : m_workers) {
            if (w->tid == tid) w->suspended = false;
        }
    });
    syscalls.set_yield_callback([this](s32 current_tid) {
        yield_from_thread(current_tid);
    });

    // Wire GIF -> GS + Optimal Frame Pacing callback (paces Frame N+1 *after* Frame N wakes D3D11,
    // then immediately samples JIT input at t=0.0ms of Frame N+1!)
    gif.attach_gs(&gs);
    gs.set_frame_pacing_callback([this]() {
        if (!enable_frame_pacing || cpu.halted || m_in_reset.load(std::memory_order_acquire)) return;
        if (m_worker_saturated_slice) {
            m_vblank_pacer_init = false;
            m_paced_this_frame = false;
            return;
        }
        pace_to_60hz();
        m_paced_this_frame = true;
        invoke_jit_input();
        iop.sync_pad0_now(memory);
    });

    // Wire VU1 -> memory + XGKICK -> GIF PATH1 (independent stream, stop on EOP)
    vu1.attach_memory(memory.vu1_code(), memory.vu1_data());
    vu1.set_xgkick_callback([this](const u8* data, std::size_t max_bytes) {
        gif.feed_path(0, data, max_bytes / 16u, gs);
    });

    // Wire VIF0 & VIF1 -> VU memory, GIF PATH2 DIRECT, and VU1 MSCAL (with XTOP/XITOP)
    vif0.attach_vu_memory(memory.vu0_code(), 4096, memory.vu0_data(), 4096);
    vif1.attach_vu_memory(memory.vu1_code(), 16384, memory.vu1_data(), 16384);
    vif1.set_gif_direct_callback([this](const u8* qwords, std::size_t qwc) {
        gif.feed_path(1, qwords, qwc, gs);
    });
    vif1.set_mscal_callback([this](u32 micro_addr, bool is_mscnt) {
        u32 start_pc = is_mscnt ? vu1.pc : micro_addr;
        vu1.set_vif_regs(static_cast<u16>(vif1.top), static_cast<u16>(vif1.itop));
        vu1.run_microprogram(start_pc);
    });
    vif1.set_mpg_callback([this](u32 start_byte, u32 byte_len) {
        vu1.on_code_written(start_byte, byte_len);
    });

    // Wire DMAC channels -> VIF0 / VIF1 / GIF PATH3 + exact 64-bit TTE delivery + DMAC completion ISR
    dmac.set_channel_sink(DmaChannelId::VIF0, [this](const u8* q, std::size_t n) {
        if (t_in_guest_execution && cpu.halted) throw GuestHaltSignal{};
        vif0.process_packet(q, n);
    });
    dmac.set_channel_sink(DmaChannelId::VIF1, [this](const u8* q, std::size_t n) {
        if (t_in_guest_execution && cpu.halted) throw GuestHaltSignal{};
        vif1.process_packet(q, n);
    });
    dmac.set_channel_sink(DmaChannelId::GIF,  [this](const u8* q, std::size_t n) {
        if (t_in_guest_execution && cpu.halted) throw GuestHaltSignal{};
        gif.feed_path(2, q, n, gs);
    });
    dmac.set_channel_tte_sink(DmaChannelId::VIF0, [this](u64 hi) { u8 b[8]; write_le64(b, hi); vif0.process_words(b, 2); });
    dmac.set_channel_tte_sink(DmaChannelId::VIF1, [this](u64 hi) { u8 b[8]; write_le64(b, hi); vif1.process_words(b, 2); });
    dmac.set_completion_callback([this](u32 ch) {
        if (t_in_guest_execution && cpu.halted) throw GuestHaltSignal{};
        EEContext& act = m_active_ctx ? *m_active_ctx : cpu;
        syscalls.fire_dmac_handlers(act, ch);
    });

    // Wire IOP -> CDVD, GuestMemory, Pad0, MemCard0 & SIF DMA RPC Bridge
    iop.attach_cdvd(&cdvd);
    iop.attach_memory(&memory);
    iop.attach_pad0(&pad0);
    iop.attach_pad_mutex(&m_pad_mtx);
    iop.attach_memcard0(&memcard0);
    syscalls.set_sif_dma_callback([this](u32 sdd_ptr, u32 count) -> u32 {
        // SifDmaTransfer { u32 src, dst, size, attr; } (16 bytes each)
        std::vector<u8> last_send_payload;
        for (u32 i = 0; i < count && i < 16; ++i) {
            u32 t_addr = sdd_ptr + i * 16u;
            u32 src  = memory.read32(t_addr + 0);
            u32 size = memory.read32(t_addr + 8);
            if (src == 0 || size == 0 || size > 4U * 1024U * 1024U) continue;

            // SifCmdHeader: +0=(psize|dsize<<8), +4=dest, +8=cid, +12=opt
            u32 cid = (size >= 16) ? memory.read32(src + 8) : 0;
            if (cid == 0x80000009u && size >= 36) {
                // SIF_CMD_BIND: bind EE client to IOP RPC server
                u32 pkt_addr   = memory.read32(src + 20);
                u32 client_ptr = memory.read32(src + 28);
                u32 sid        = memory.read32(src + 32);
                u32 handle     = 0x00100000u | (static_cast<u32>(m_bound_sids.size() + 1) << 4);
                m_bound_sids.push_back(sid);
                m_server_to_sid[handle] = sid;
                if (client_ptr != 0) {
                    memory.write32(client_ptr + 36, handle); // client->server = handle
                    memory.write32(client_ptr + 0, 0);       // clear busy paddr
                }
                if (pkt_addr != 0) {
                    // Emulate SIF_CMD_REND packet pool release (sub_00461e90)
                    memory.write32(pkt_addr + 24, 0);
                    memory.write32(pkt_addr + 16, memory.read32(pkt_addr + 16) & ~1u);
                }
            } else if (cid == 0x8000000Au && size >= 56) {
                // SIF_CMD_CALL: invoke IOP RPC server & write reply back to EE RAM
                u32 pkt_addr   = memory.read32(src + 20);
                u32 client_ptr = memory.read32(src + 28);
                u32 rpc_num    = memory.read32(src + 32);
                u32 recv_ptr   = memory.read32(src + 40);
                u32 recv_size  = memory.read32(src + 44);
                u32 server_h   = memory.read32(src + 52);
                u32 sid = 0;
                if (auto it = m_server_to_sid.find(server_h); it != m_server_to_sid.end()) {
                    sid = it->second;
                }
                std::vector<u8> recv_buf(std::min<u32>(recv_size, 65536u), 0);
                if (m_rpc_trace.size() < 96) {
                    std::string tag = hex32(sid) + ":fn=" + std::to_string(rpc_num) +
                                      "(send=" + std::to_string(last_send_payload.size()) +
                                      ",recv=" + std::to_string(recv_size) + ")";
                    m_rpc_trace.push_back(tag);
                }
                {
                    std::lock_guard<std::mutex> iop_lk(m_iop_tick_mtx);
                    iop.call_rpc(
                        sid, rpc_num,
                        last_send_payload.empty() ? nullptr : last_send_payload.data(),
                        static_cast<u32>(last_send_payload.size()),
                        recv_buf.empty() ? nullptr : recv_buf.data(),
                        static_cast<u32>(recv_buf.size())
                    );
                }
                if (recv_ptr != 0 && !recv_buf.empty()) {
                    memory.write_block(recv_ptr, recv_buf.data(), recv_buf.size());
                }
                u32 end_func  = 0;
                u32 end_param = 0;
                u32 end_gp    = 0;
                if (client_ptr != 0) {
                    end_gp    = memory.read32(client_ptr + 24);
                    end_func  = memory.read32(client_ptr + 28);
                    end_param = memory.read32(client_ptr + 32);
                    memory.write32(client_ptr + 0, 0); // clear busy paddr
                }
                if (pkt_addr != 0) {
                    // Emulate SIF_CMD_REND packet pool release (sub_00461e90)
                    memory.write32(pkt_addr + 24, 0);
                    memory.write32(pkt_addr + 16, memory.read32(pkt_addr + 16) & ~1u);
                }
                if (end_func != 0) {
                    // Emulate SIF_CMD_REND async completion callback (client->end_func(client->end_param))
                    EEContext& act = m_active_ctx ? *m_active_ctx : cpu;
                    auto saved_r = act.r;
                    u32 saved_pc = act.pc;
                    u64 saved_hi = act.hi, saved_lo = act.lo;
                    u64 saved_hi1 = act.hi1, saved_lo1 = act.lo1;
                    u32 saved_sa = act.sa;
                    auto saved_fpr = act.fpr;
                    f32 saved_fpu_acc = act.fpu_acc;
                    u32 saved_fcr31 = act.fcr31;
                    auto saved_vf = act.vf;
                    auto saved_vi = act.vi;
                    Vec4f saved_vu0_acc = act.vu0_acc;
                    act.set_gpr_u32_sext(4, end_param);
                    if (end_gp != 0) act.set_gpr_u32_sext(28, end_gp);
                    act.set_gpr_u32_sext(31, 0);
                    call_guest_on(act, end_func);
                    act.r = saved_r;
                    act.pc = saved_pc;
                    act.hi = saved_hi; act.lo = saved_lo;
                    act.hi1 = saved_hi1; act.lo1 = saved_lo1;
                    act.sa = saved_sa;
                    act.fpr = saved_fpr;
                    act.fpu_acc = saved_fpu_acc;
                    act.fcr31 = saved_fcr31;
                    act.vf = saved_vf;
                    act.vi = saved_vi;
                    act.vu0_acc = saved_vu0_acc;
                }
            } else if (cid == 0x80000003u) {
                // SIF_CMD_RESET (IOP reboot e.g. IOPRP300.IMG) — keep SIF ready
            } else {
                // Capture preceding payload transfer (sent right before SIF_CMD_CALL)
                u32 cap = std::min<u32>(size, 65536u);
                last_send_payload.resize(cap);
                memory.read_block(src, last_send_payload.data(), cap);
            }
        }
        return m_next_sif_dma_id++;
    });

    // Wire MMIO routing for EE Timers, GIF, VIF0/1, INTC, DMAC, SIF, and GS privileged registers
    MmioHandler mmio;
    mmio.read32 = [this](u32 paddr) -> u32 {
        if (t_in_guest_execution && cpu.halted) throw GuestHaltSignal{};
        if (enable_profiling) m_mmio_reads[paddr]++;
        if (paddr >= 0x10000000u && paddr < 0x10002000u) {
            u32 idx = (paddr >> 11) & 3u;
            u32 reg = paddr & 0x3Fu;
            if (reg == 0x00) {
                m_timers_count[idx] = (m_timers_count[idx] + 1u) & 0xFFFFu;
                return m_timers_count[idx];
            }
            if (reg == 0x10) return m_timers_mode[idx];
            return 0;
        }
        if (paddr >= 0x10003000u && paddr < 0x10003100u) {
            if (paddr == 0x10003010u) return m_gif_mode; // GIF_MODE
            if (paddr == 0x10003020u) return 0u;         // GIF_STAT (idle, FIFO empty)
            return 0;
        }
        if (paddr >= 0x10003800u && paddr < 0x10003900u) {
            return vif0.read_reg(paddr - 0x10003800u);
        }
        if (paddr >= 0x10003C00u && paddr < 0x10003D00u) {
            return vif1.read_reg(paddr - 0x10003C00u);
        }
        if ((paddr >= 0x10008000u && paddr < 0x1000F000u) || paddr == 0x1000F520u) {
            return dmac.read_reg(paddr);
        }
        if (paddr == 0x1000F000u) {
            // INTC_STAT: handle VBlank Start (0x4) and VBlank End (0x8) distinctly so
            // sceGsSyncV polling both edges fires exactly 1 VSync pulse + cause-2 then cause-3.
            const bool need_vb_start = (m_intc_stat & 0x04u) == 0u;
            const bool need_vb_end   = (m_intc_stat & 0x08u) == 0u;
            if (need_vb_start || need_vb_end) {
                auto do_frame_tick = [this]() {
                    m_worker_saturated_slice = false;
                    if (m_active_tid == 1) {
                        yield_from_thread(1);
                    }
                    gs.publish_direct_framebuffer_if_active();

                    // Optimal Frame Pacing: if DISPFB already paced & JIT-sampled after presenting Frame N,
                    // skip double-sleeping here; otherwise (single-buffered screen) pace FIRST, then JIT-sample!
                    if (enable_frame_pacing && !cpu.halted) {
                        if (m_worker_saturated_slice) {
                            m_vblank_pacer_init = false;
                        } else if (!m_paced_this_frame) {
                            pace_to_60hz();
                        }
                    }
                    m_paced_this_frame = false;
                    if (!m_worker_saturated_slice) {
                        invoke_jit_input();
                    } else {
                        // During uncapped 550-975 FPS loading, throttle input polling to <=60 Hz
                        // so GetAsyncKeyState / XInput / pad_mutex are not hammered 975x/sec.
                        const auto now_in = std::chrono::steady_clock::now();
                        if (m_last_loading_input_tp.time_since_epoch().count() == 0 ||
                            now_in - m_last_loading_input_tp >= std::chrono::milliseconds(16)) {
                            m_last_loading_input_tp = now_in;
                            invoke_jit_input();
                        }
                    }
                    iop.tick_vblank();
                    syscalls.pulse_vsync(memory);
                };
                if (need_vb_start) {
                    do_frame_tick();
                    m_intc_stat |= 0x04u;
                    m_vblank_pulsed_this_frame = true;
                    EEContext& act = m_active_ctx ? *m_active_ctx : cpu;
                    syscalls.fire_intc_handlers(act, 2);
                }
                if (need_vb_end) {
                    if (!m_vblank_pulsed_this_frame) {
                        do_frame_tick();
                    }
                    m_vblank_pulsed_this_frame = false;
                    m_intc_stat |= 0x08u;
                    EEContext& act = m_active_ctx ? *m_active_ctx : cpu;
                    syscalls.fire_intc_handlers(act, 3);
                }
            }
            return m_intc_stat;
        }
        if (paddr == 0x1000F010u) return m_intc_mask;
        if (paddr >= 0x1000F200u && paddr < 0x1000F270u) {
            return iop.read_sif_reg(paddr);
        }
        if (paddr >= 0x12000000u && paddr < 0x12002000u) {
            u64 v = gs.read_priv_reg(paddr & ~7u);
            return static_cast<u32>((paddr & 4u) ? (v >> 32) : v);
        }
        return 0;
    };
    mmio.write32 = [this](u32 paddr, u32 val) {
        if (t_in_guest_execution && cpu.halted) throw GuestHaltSignal{};
        if (paddr >= 0x10000000u && paddr < 0x10002000u) {
            u32 idx = (paddr >> 11) & 3u;
            u32 reg = paddr & 0x3Fu;
            if (reg == 0x00) m_timers_count[idx] = val & 0xFFFFu;
            else if (reg == 0x10) m_timers_mode[idx] = val;
            return;
        }
        if (paddr >= 0x10003000u && paddr < 0x10003100u) {
            if (paddr == 0x10003000u && (val & 1u) != 0) gif.reset(); // GIF_CTRL RST
            else if (paddr == 0x10003010u) m_gif_mode = val;         // GIF_MODE
            return;
        }
        if (paddr >= 0x10003800u && paddr < 0x10003900u) {
            vif0.write_reg(paddr - 0x10003800u, val);
            return;
        }
        if (paddr >= 0x10003C00u && paddr < 0x10003D00u) {
            vif1.write_reg(paddr - 0x10003C00u, val);
            return;
        }
        if ((paddr >= 0x10008000u && paddr < 0x1000F000u) || paddr == 0x1000F590u) {
            dmac.write_reg(paddr, val, memory);
            return;
        }
        if (paddr == 0x1000F000u) { m_intc_stat &= ~val; return; }
        if (paddr == 0x1000F010u) { m_intc_mask ^= val; return; }
        if (paddr >= 0x1000F200u && paddr < 0x1000F270u) {
            iop.write_sif_reg(paddr, val);
            return;
        }
        if (paddr >= 0x12000000u && paddr < 0x12002000u) {
            gs.write_priv_reg32(paddr, val);
            return;
        }
    };
    mmio.read64 = [this](u32 paddr) -> u64 {
        if (t_in_guest_execution && cpu.halted) throw GuestHaltSignal{};
        if (paddr >= 0x12000000u && paddr < 0x12002000u) {
            return gs.read_priv_reg(paddr);
        }
        return 0;
    };
    mmio.write64 = [this](u32 paddr, u64 val) {
        if (t_in_guest_execution && cpu.halted) throw GuestHaltSignal{};
        if (paddr >= 0x12000000u && paddr < 0x12002000u) {
            gs.write_priv_reg(paddr, val);
        }
    };
    mmio.write128 = [this](u32 paddr, u128 val) {
        if (t_in_guest_execution && cpu.halted) throw GuestHaltSignal{};
        u8 buf[16];
        write_le64(buf, val.lo);
        write_le64(buf + 8, val.hi);
        if (paddr == 0x10004000u) {
            vif0.process_packet(buf, 1);
        } else if (paddr == 0x10005000u) {
            vif1.process_packet(buf, 1);
        } else if (paddr == 0x10006000u) {
            gif.feed_path(2, buf, 1, gs);
        }
    };
    memory.set_mmio_handler(std::move(mmio));
}

bool RuntimeHost::load_elf(const Ps2ElfImage& elf, std::string& out_error) {
    if (!elf.valid) {
        out_error = "Cannot load invalid ELF image";
        return false;
    }

    for (const auto& ph : elf.program_headers) {
        if (!ph.is_load() || ph.memsz == 0) continue;
        if (!ph.data.empty()) {
            if (!memory.write_block(ph.vaddr, ph.data.data(), ph.data.size())) {
                out_error = "Failed writing ELF PT_LOAD segment at " + hex32(ph.vaddr);
                return false;
            }
        }
        if (ph.memsz > ph.data.size()) {
            if (!memory.zero_block(ph.vaddr + static_cast<u32>(ph.data.size()), ph.memsz - ph.data.size())) {
                out_error = "Failed zeroing BSS tail of segment at " + hex32(ph.vaddr);
                return false;
            }
        }
    }

    cpu.pc = elf.e_entry;
    // Standard PS2 initial stack near top of 32MB EE RAM (16-byte aligned)
    cpu.set_gpr_u32_sext(29, kEeRamSize - 0x1000u);
    return true;
}

bool RuntimeHost::mount_iso(const std::string& iso_path, std::string& out_error) {
    return cdvd.mount_iso(iso_path, out_error);
}

void RuntimeHost::register_function(u32 guest_addr, RecompiledFn fn) {
    m_fn_table[guest_addr] = fn;
    if (!m_fast_fn_table.empty() && guest_addr >= kFastFnBase && guest_addr < kFastFnEnd && (guest_addr & 3u) == 0) {
        m_fast_fn_table[(guest_addr - kFastFnBase) >> 2] = fn;
        m_fast_fn_synced_size = m_fn_table.size();
    }
}

void RuntimeHost::sync_fast_fn_table() {
    const std::size_t n = m_fn_table.size();
    if (n == m_fast_fn_synced_size && !m_fast_fn_table.empty()) return;
    if (m_fast_fn_table.empty()) {
        m_fast_fn_table.assign((kFastFnEnd - kFastFnBase) >> 2, nullptr);
    } else {
        std::fill(m_fast_fn_table.begin(), m_fast_fn_table.end(), nullptr);
    }
    for (const auto& [addr, fn] : m_fn_table) {
        if (fn && addr >= kFastFnBase && addr < kFastFnEnd && (addr & 3u) == 0) {
            m_fast_fn_table[(addr - kFastFnBase) >> 2] = fn;
        }
    }
    m_fast_fn_synced_size = n;
}

void RuntimeHost::start_worker_thread(const Ps2Thread& t) {
    if (t.id <= 1 || t.entry == 0) return;

    // Reap any previously exited worker for the same TID to avoid duplicate entries
    std::vector<std::unique_ptr<WorkerThread>> exited_same_tid;
    {
        std::lock_guard<std::mutex> lk(m_sched_mtx);
        for (auto it = m_workers.begin(); it != m_workers.end(); ) {
            if ((*it)->tid == t.id && (*it)->exited) {
                exited_same_tid.push_back(std::move(*it));
                it = m_workers.erase(it);
            } else {
                ++it;
            }
        }
    }
    for (auto& old_w : exited_same_tid) {
        if (old_w->host_thread.joinable()) old_w->host_thread.join();
    }

    auto w = std::make_unique<WorkerThread>();
    w->tid = t.id;
    w->entry = t.entry;
    w->arg = t.arg;
    w->suspended = t.suspended;
    w->exited = false;
    w->ctx.reset();
    w->ctx.mem = &memory;
    w->ctx.runtime_user_data = this;
    w->ctx.set_gpr_u32_sext(4, t.arg);
    w->ctx.set_gpr_u32_sext(28, t.gp_reg);
    u32 sp_top = (t.stack != 0 && t.stack_size >= 256)
        ? ((t.stack + t.stack_size - 64u) & ~15u)
        : (kEeRamSize - 0x20000u);
    w->ctx.set_gpr_u32_sext(29, sp_top);
    w->ctx.set_gpr_u32_sext(31, 0);

    WorkerThread* wp = w.get();
    w->ctx.dispatch_fn = [this, wp](EEContext& c, u32 target_pc) {
        if (c.halted || cpu.halted) {
            if (t_in_guest_execution) throw GuestHaltSignal{};
            return;
        }
        if (++wp->slice_ops >= 1048576 || wp->suspended) {
            if (wp->slice_ops >= 1048576) m_worker_saturated_slice = true;
            yield_from_thread(wp->tid);
        }
        if (c.halted || cpu.halted) {
            if (t_in_guest_execution) throw GuestHaltSignal{};
            return;
        }
        call_guest_on(c, target_pc);
    };
    w->ctx.syscall_fn = [this, wp](EEContext& c, u32 code) {
        if (c.halted || cpu.halted) {
            if (t_in_guest_execution) throw GuestHaltSignal{};
            return;
        }
        syscalls.dispatch(c, code);
        if (c.halted || cpu.halted) {
            if (t_in_guest_execution) throw GuestHaltSignal{};
            return;
        }
        if (++wp->slice_ops >= 1048576 || wp->suspended) {
            if (wp->slice_ops >= 1048576) m_worker_saturated_slice = true;
            yield_from_thread(wp->tid);
        }
    };
    w->ctx.vu0_macro_fn = [this](EEContext& c, u32 raw) {
        vu0.execute_macro(c, raw);
    };

    w->host_thread = std::thread([this, wp]() {
        {
            std::unique_lock<std::mutex> lk(m_sched_mtx);
            while (m_active_tid != wp->tid && !m_sched_shutdown && !cpu.halted && !wp->ctx.halted) {
                m_sched_cv.wait_for(lk, std::chrono::milliseconds(5));
            }
            if (m_sched_shutdown || cpu.halted || wp->ctx.halted) {
                wp->exited = true;
                return;
            }
        }
        m_active_ctx = &wp->ctx;
        m_active_call_depth = &wp->call_depth;
        syscalls.set_current_thread_id(wp->tid);
        try {
            GuestExecScope scope;
            PerfScope perf_worker(PerfCat::EeRecomp);
            call_guest_on(wp->ctx, wp->entry);
        } catch (const GuestHaltSignal&) {
            wp->ctx.halted = true;
        }
        {
            std::lock_guard<std::mutex> lk(m_sched_mtx);
            wp->exited = true;
            if (m_active_tid == wp->tid) {
                m_active_tid = 1;
                m_active_ctx = &cpu;
                m_active_call_depth = &m_main_call_depth;
                syscalls.set_current_thread_id(1);
            }
        }
        m_sched_cv.notify_all();
    });

    std::lock_guard<std::mutex> lk(m_sched_mtx);
    m_workers.push_back(std::move(w));
}

void RuntimeHost::interpreter_cooperative_yield_thunk(void* user_data) {
    auto* host = static_cast<RuntimeHost*>(user_data);
    if (!host) return;
    const s32 tid = host->m_active_tid;
    if (tid > 1) {
        host->yield_from_thread(tid);
    }
}

void RuntimeHost::yield_from_thread(s32 current_tid) {
    if (cpu.halted || m_sched_shutdown) {
        if (m_active_ctx) m_active_ctx->halted = true;
        if (t_in_guest_execution) throw GuestHaltSignal{};
        return;
    }
    if (iop.has_heartbeat()) {
        iop.tick_heartbeat();
    }
    if (current_tid == 1) {
        // Main thread yielding round-robin to any runnable (non-suspended, non-exited) worker thread
        s32 next_tid = 0;
        {
            std::lock_guard<std::mutex> lk(m_sched_mtx);
            std::size_t n = m_workers.size();
            for (std::size_t k = 0; k < n; ++k) {
                std::size_t idx = (m_rr_worker_idx + k) % n;
                auto& w = m_workers[idx];
                if (!w->exited && !w->suspended && !w->ctx.halted) {
                    w->slice_ops = 0;
                    next_tid = w->tid;
                    m_active_tid = next_tid;
                    m_rr_worker_idx = (idx + 1) % n;
                    break;
                }
            }
        }
        if (next_tid == 0) return;
        m_sched_cv.notify_all();
        std::unique_lock<std::mutex> lk(m_sched_mtx);
        {
            PerfScope perf_idle(PerfCat::IdleYield);
            while (m_active_tid != 1 && !m_sched_shutdown && !cpu.halted) {
                m_sched_cv.wait_for(lk, std::chrono::milliseconds(5));
            }
        }
        m_active_ctx = &cpu;
        m_active_call_depth = &m_main_call_depth;
        syscalls.set_current_thread_id(1);
        if (m_sched_shutdown || cpu.halted) {
            if (t_in_guest_execution) throw GuestHaltSignal{};
        }
    } else {
        // Worker thread yielding back to Main thread
        WorkerThread* self = nullptr;
        bool should_halt = false;
        {
            std::lock_guard<std::mutex> lk(m_sched_mtx);
            for (auto& w : m_workers) {
                if (w->tid == current_tid) { self = w.get(); break; }
            }
            if (m_sched_shutdown || cpu.halted || (self && self->ctx.halted)) {
                if (self) self->ctx.halted = true;
                should_halt = true;
            } else {
                m_active_tid = 1;
                m_active_ctx = &cpu;
                m_active_call_depth = &m_main_call_depth;
                syscalls.set_current_thread_id(1);
            }
        }
        if (should_halt) {
            if (t_in_guest_execution) throw GuestHaltSignal{};
            return;
        }
        m_sched_cv.notify_all();
        std::unique_lock<std::mutex> lk(m_sched_mtx);
        {
            PerfScope perf_idle(PerfCat::IdleYield);
            while (m_active_tid != current_tid && !m_sched_shutdown && !cpu.halted &&
                   (!self || !self->ctx.halted)) {
                m_sched_cv.wait_for(lk, std::chrono::milliseconds(5));
            }
        }
        if (m_sched_shutdown || cpu.halted || (self && self->ctx.halted)) {
            if (self) self->ctx.halted = true;
            if (t_in_guest_execution) throw GuestHaltSignal{};
            return;
        }
        if (self) {
            self->slice_ops = 0;
            m_active_ctx = &self->ctx;
            m_active_call_depth = &self->call_depth;
        }
        syscalls.set_current_thread_id(current_tid);
    }
}

void RuntimeHost::call_guest(u32 guest_addr) {
    sync_fast_fn_table();
    m_active_ctx = &cpu;
    m_active_call_depth = &m_main_call_depth;
    syscalls.set_current_thread_id(1);
    try {
        GuestExecScope scope;
        PerfScope perf_root(PerfCat::EeRecomp);
        call_guest_on(cpu, guest_addr);
    } catch (const GuestHaltSignal&) {
        cpu.halted = true;
    }
    m_main_call_depth = 0;
    call_depth = 0;
    stop_workers();
}

u32& RuntimeHost::call_depth_for(EEContext& ctx) {
    if (&ctx == &cpu) return m_main_call_depth;
    for (auto& w : m_workers) {
        if (&w->ctx == &ctx) return w->call_depth;
    }
    return m_main_call_depth;
}

void RuntimeHost::call_guest_on(EEContext& ctx, u32 guest_addr) {
    if (ctx.halted || cpu.halted) {
        if (t_in_guest_execution) throw GuestHaltSignal{};
        return;
    }
    u32& depth = (&ctx == m_active_ctx && m_active_call_depth) ? *m_active_call_depth : call_depth_for(ctx);
    if (depth >= max_call_depth) {
        ctx.halted = true;
        ctx.trap_message = "Call depth limit reached at " + hex32(guest_addr);
        return;
    }
    if (max_static_calls > 0 && total_static_calls >= max_static_calls) {
        ctx.halted = true;
        cpu.halted = true;
        ctx.trap_message = "Static call budget reached at " + hex32(guest_addr);
        if (t_in_guest_execution) throw GuestHaltSignal{};
        return;
    }

    RecompiledFn fn = nullptr;
    if (!m_fast_fn_table.empty() && guest_addr >= kFastFnBase && guest_addr < kFastFnEnd && (guest_addr & 3u) == 0) {
        fn = m_fast_fn_table[(guest_addr - kFastFnBase) >> 2];
    } else {
        auto it = m_fn_table.find(guest_addr);
        if (it != m_fn_table.end()) fn = it->second;
    }
    if (fn != nullptr) {
        indirect_calls_resolved++;
        total_static_calls++;
        if (enable_profiling) m_fn_call_counts[guest_addr]++;
        CallDepthGuard depth_guard(depth, call_depth);
        PerfScope perf_recomp(PerfCat::EeRecomp);
        ctx.pc = guest_addr;
        fn(ctx);
        return;
    }

    indirect_calls_missed++;
    last_missed_target = guest_addr;

    if (enable_fallback_interpreter && memory.translate_ptr(guest_addr, 4) != nullptr) {
        CallDepthGuard depth_guard(depth, call_depth);
        if (g_perf.enabled) g_perf.ee_interp_calls.fetch_add(1, std::memory_order_relaxed);
        PerfScope perf_interp(PerfCat::EeInterp);
        const bool is_worker = (&ctx != &cpu) && (m_active_tid > 1);
        fallback_steps_executed += R5900FallbackInterpreter::run_until_return_or_handoff(
            ctx, guest_addr, m_fn_table, 100000000ULL,
            m_fast_fn_table.empty() ? nullptr : m_fast_fn_table.data(),
            kFastFnBase, kFastFnEnd,
            is_worker ? &RuntimeHost::interpreter_cooperative_yield_thunk : nullptr,
            is_worker ? this : nullptr);
        return;
    }

    ctx.halted = true;
    ctx.trap_message = "Dynamic/uncompiled guest target " + hex32(guest_addr) +
                       " is not present in static function table";
}

} // namespace tf::ps2
