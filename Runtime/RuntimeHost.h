#pragma once

#include "CPU/EEContext.h"
#include "CPU/SyscallDispatcher.h"
#include "Memory/GuestMemory.h"
#include "VectorUnits/VU0.h"
#include "VectorUnits/VU1.h"
#include "DMA/DmaController.h"
#include "VIF/VifUnit.h"
#include "GIF/GifUnit.h"
#include "GS/GsState.h"
#include "IOP/IopHle.h"
#include "CDVD/CdvdDrive.h"
#include "Audio/Spu2.h"
#include "Input/DualShock2.h"
#include "MemoryCard/VirtualMemoryCard.h"
#include "../Recompiler/ELF/ElfParser.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace tf::ps2 {

// Complete PS2 Hardware Compatibility Runtime connecting static recompiled code
// to EE Memory, Syscalls, VU0/VU1, DMAC, VIF0/1, GIF, GS, IOP HLE, CDVD, SPU2, Input, and MemoryCard.
class RuntimeHost {
public:
    RuntimeHost();
    ~RuntimeHost();
    void reset();

    // Loads ELF PT_LOAD segments into GuestMemory and initializes SP/PC.
    bool load_elf(const Ps2ElfImage& elf, std::string& out_error);

    // Mounts the read-only PS2 ISO into the CDVD subsystem.
    bool mount_iso(const std::string& iso_path, std::string& out_error);

    // Registers statically recompiled C++ functions keyed by guest PS2 virtual address.
    void register_function(u32 guest_addr, RecompiledFn fn);
    std::unordered_map<u32, RecompiledFn>& function_table() { return m_fn_table; }
    const std::unordered_map<u32, RecompiledFn>& function_table() const { return m_fn_table; }

    // Invokes a guest address: calls statically recompiled native function if present,
    // otherwise logs/traps or runs fallback diagnostics.
    void call_guest(u32 guest_addr);
    void call_guest_on(EEContext& ctx, u32 guest_addr);
    void request_halt();
    void stop_workers();

    // Subsystems
    EEContext         cpu;
    GuestMemory       memory;
    SyscallDispatcher syscalls;
    VU0Unit           vu0;
    VU1Unit           vu1;
    DmaController     dmac;
    VifUnit           vif0{0};
    VifUnit           vif1{1};
    GifUnit           gif;
    GsState           gs;
    CdvdDrive         cdvd;
    IopHle            iop;
    Spu2Unit          spu2;
    DualShock2State   pad0;
    InputConfig       input_config;
    VirtualMemoryCard memcard0;

    u64 indirect_calls_resolved = 0;
    u64 indirect_calls_missed = 0;
    u64 fallback_steps_executed = 0;
    u32 last_missed_target = 0;
    bool enable_fallback_interpreter = true;
    u32 call_depth = 0;
    u32 max_call_depth = 1024;
    u64 total_static_calls = 0;
    u64 max_static_calls = 0; // 0 = unlimited
    bool enable_profiling = false; // Gate per-call / per-MMIO hash-map telemetry off in hot path
    bool enable_frame_pacing = false; // Pace VBlanks to smooth 60.0 Hz in interactive desktop mode

    const std::unordered_map<u32, u64>& mmio_read_counts() const { return m_mmio_reads; }
    const std::unordered_map<u32, u64>& fn_call_counts() const { return m_fn_call_counts; }
    const std::vector<u32>& bound_sif_sids() const { return m_bound_sids; }
    const std::vector<std::string>& sif_rpc_trace() const { return m_rpc_trace; }
    std::mutex& pad_mutex() { return m_pad_mtx; }

    // Registers a lightweight callback invoked at t=0.0ms of each frame (right after pacer wakeup,
    // immediately before pad0 DMA write) for Just-In-Time ultra-low-latency input sampling.
    void set_jit_input_callback(std::function<void()> cb);
    u32 last_frame_work_us() const { return m_last_frame_work_us.load(std::memory_order_relaxed); }

private:
    struct WorkerThread {
        s32 tid = 0;
        u32 entry = 0;
        u32 arg = 0;
        bool suspended = false;
        bool exited = false;
        u32 slice_ops = 0;
        u32 call_depth = 0;
        EEContext ctx;
        std::thread host_thread;
    };

    void wire_subsystems();
    void start_worker_thread(const Ps2Thread& t);
    void yield_from_thread(s32 current_tid);
    static void interpreter_cooperative_yield_thunk(void* user_data);
    u32& call_depth_for(EEContext& ctx);
    void sync_fast_fn_table();
    void pace_to_60hz();
    void invoke_jit_input();

    static constexpr u32 kFastFnBase = 0x00100000u;
    static constexpr u32 kFastFnEnd  = 0x00600000u;

    std::unordered_map<u32, RecompiledFn> m_fn_table;
    std::vector<RecompiledFn> m_fast_fn_table;
    std::size_t m_fast_fn_synced_size = 0;
    std::unordered_map<u32, u64> m_mmio_reads;
    std::unordered_map<u32, u64> m_fn_call_counts;
    std::vector<u32> m_bound_sids;
    std::vector<std::string> m_rpc_trace;
    std::unordered_map<u32, u32> m_server_to_sid;
    u32 m_next_sif_dma_id = 1;
    std::array<u32, 4> m_timers_count{};
    std::array<u32, 4> m_timers_mode{};
    u32 m_intc_stat = 0;
    u32 m_intc_mask = 0;
    u32 m_gif_mode = 0;
    bool m_vblank_pulsed_this_frame = false;
    u32 m_main_call_depth = 0;
    u32* m_active_call_depth = nullptr;
    std::atomic<bool> m_iop_ticker_stop{false};
    std::atomic<bool> m_iop_ticker_paused{false};
    std::atomic<bool> m_iop_ticker_active{false};
    std::mutex m_iop_tick_mtx;
    std::mutex m_pad_mtx;
    std::mutex m_jit_input_mtx;
    std::function<void()> m_jit_input_cb;
    std::thread m_iop_ticker;
    std::chrono::steady_clock::time_point m_next_vblank_tp{};
    std::chrono::steady_clock::time_point m_frame_work_start_tp{};
    std::chrono::steady_clock::time_point m_last_loading_input_tp{};
    void* m_pacer_timer = nullptr;
    std::atomic<u32> m_last_frame_work_us{0};
    std::atomic<bool> m_in_reset{false};
    bool m_vblank_pacer_init = false;
    bool m_paced_this_frame = false;
    bool m_worker_saturated_slice = false;

    // Cooperative PS2 EE multi-threading (one active EE thread at a time, zero data races)
    EEContext* m_active_ctx = nullptr;
    std::mutex m_sched_mtx;
    std::condition_variable m_sched_cv;
    s32 m_active_tid = 1;
    std::size_t m_rr_worker_idx = 0;
    bool m_sched_shutdown = false;
    std::vector<std::unique_ptr<WorkerThread>> m_workers;
};

} // namespace tf::ps2
