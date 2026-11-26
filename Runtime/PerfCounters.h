#pragma once

#include "../Recompiler/Common/Types.h"
#include <array>
#include <atomic>
#include <chrono>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace tf::ps2 {

enum class PerfCat : u8 {
    EeRecomp = 0,
    EeInterp,
    VifDma,
    VifUnpack,
    Vu1,
    GifParse,
    GsTri,
    GsSprite,
    GsTex,
    GsVram,
    GsPresent,
    IdleYield,
    Count
};

inline constexpr std::size_t kPerfCatCount = static_cast<std::size_t>(PerfCat::Count);

inline u64 perf_now_ticks() noexcept {
#ifdef _WIN32
    LARGE_INTEGER li{};
    QueryPerformanceCounter(&li);
    return static_cast<u64>(li.QuadPart);
#else
    return static_cast<u64>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
#endif
}

inline double perf_ticks_to_ms(u64 ticks) noexcept {
#ifdef _WIN32
    static const double s_inv_freq_ms = []() {
        LARGE_INTEGER freq{};
        QueryPerformanceFrequency(&freq);
        return (freq.QuadPart > 0) ? (1000.0 / static_cast<double>(freq.QuadPart)) : 0.0;
    }();
    return static_cast<double>(ticks) * s_inv_freq_ms;
#else
    return static_cast<double>(ticks) * 1e-6;
#endif
}

struct PerfSnapshot {
    std::array<u64, kPerfCatCount> ticks{};
    u64 tri_parallel_dispatches = 0;
    u64 tri_single_draws = 0;
    u64 tri_batches = 0;
    u64 tri_batch_tris = 0;
    u64 tri_fast_tex_px = 0;
    u64 tri_fast_flat_px = 0;
    u64 tri_general_px = 0;
    u64 tri_fast_tex_cnt = 0;
    u64 tri_fast_flat_cnt = 0;
    u64 tri_general_cnt = 0;
    u64 tri_gen_date_cnt = 0;
    u64 tri_gen_pabe_cnt = 0;
    u64 tri_gen_afail_cnt = 0;
    u64 sprite_draws = 0;
    u64 sprite_parallel_dispatches = 0;
    u64 sprite_fast_clear = 0;
    u64 sprite_fast_copy = 0;
    u64 sprite_fast_simd = 0;
    u64 sprite_general = 0;
    u64 tex_rt_hits = 0;
    u64 tex_cache_hits = 0;
    u64 tex_cache_misses = 0;
    u64 tex_cache_redecodes = 0;
    u64 fb_flushes = 0;
    u64 fb_reloads = 0;
    u64 ee_interp_calls = 0;
};

struct PerfCounters {
    bool enabled = false;

    std::array<std::atomic<u64>, kPerfCatCount> ticks{};
    std::atomic<u64> tri_parallel_dispatches{0};
    std::atomic<u64> tri_single_draws{0};
    std::atomic<u64> tri_batches{0};
    std::atomic<u64> tri_batch_tris{0};
    std::atomic<u64> tri_fast_tex_px{0};
    std::atomic<u64> tri_fast_flat_px{0};
    std::atomic<u64> tri_general_px{0};
    std::atomic<u64> tri_fast_tex_cnt{0};
    std::atomic<u64> tri_fast_flat_cnt{0};
    std::atomic<u64> tri_general_cnt{0};
    std::atomic<u64> tri_gen_date_cnt{0};
    std::atomic<u64> tri_gen_pabe_cnt{0};
    std::atomic<u64> tri_gen_afail_cnt{0};
    std::atomic<u64> sprite_draws{0};
    std::atomic<u64> sprite_parallel_dispatches{0};
    std::atomic<u64> sprite_fast_clear{0};
    std::atomic<u64> sprite_fast_copy{0};
    std::atomic<u64> sprite_fast_simd{0};
    std::atomic<u64> sprite_general{0};
    std::atomic<u64> tex_rt_hits{0};
    std::atomic<u64> tex_cache_hits{0};
    std::atomic<u64> tex_cache_misses{0};
    std::atomic<u64> tex_cache_redecodes{0};
    std::atomic<u64> fb_flushes{0};
    std::atomic<u64> fb_reloads{0};
    std::atomic<u64> ee_interp_calls{0};

    void reset() {
        for (auto& t : ticks) t.store(0, std::memory_order_relaxed);
        tri_parallel_dispatches.store(0, std::memory_order_relaxed);
        tri_single_draws.store(0, std::memory_order_relaxed);
        tri_batches.store(0, std::memory_order_relaxed);
        tri_batch_tris.store(0, std::memory_order_relaxed);
        tri_fast_tex_px.store(0, std::memory_order_relaxed);
        tri_fast_flat_px.store(0, std::memory_order_relaxed);
        tri_general_px.store(0, std::memory_order_relaxed);
        tri_fast_tex_cnt.store(0, std::memory_order_relaxed);
        tri_fast_flat_cnt.store(0, std::memory_order_relaxed);
        tri_general_cnt.store(0, std::memory_order_relaxed);
        tri_gen_date_cnt.store(0, std::memory_order_relaxed);
        tri_gen_pabe_cnt.store(0, std::memory_order_relaxed);
        tri_gen_afail_cnt.store(0, std::memory_order_relaxed);
        sprite_draws.store(0, std::memory_order_relaxed);
        sprite_parallel_dispatches.store(0, std::memory_order_relaxed);
        sprite_fast_clear.store(0, std::memory_order_relaxed);
        sprite_fast_copy.store(0, std::memory_order_relaxed);
        sprite_fast_simd.store(0, std::memory_order_relaxed);
        sprite_general.store(0, std::memory_order_relaxed);
        tex_rt_hits.store(0, std::memory_order_relaxed);
        tex_cache_hits.store(0, std::memory_order_relaxed);
        tex_cache_misses.store(0, std::memory_order_relaxed);
        tex_cache_redecodes.store(0, std::memory_order_relaxed);
        fb_flushes.store(0, std::memory_order_relaxed);
        fb_reloads.store(0, std::memory_order_relaxed);
        ee_interp_calls.store(0, std::memory_order_relaxed);
    }

    PerfSnapshot snapshot() const {
        PerfSnapshot s;
        for (std::size_t i = 0; i < kPerfCatCount; ++i) {
            s.ticks[i] = ticks[i].load(std::memory_order_relaxed);
        }
        s.tri_parallel_dispatches = tri_parallel_dispatches.load(std::memory_order_relaxed);
        s.tri_single_draws = tri_single_draws.load(std::memory_order_relaxed);
        s.tri_batches = tri_batches.load(std::memory_order_relaxed);
        s.tri_batch_tris = tri_batch_tris.load(std::memory_order_relaxed);
        s.tri_fast_tex_px = tri_fast_tex_px.load(std::memory_order_relaxed);
        s.tri_fast_flat_px = tri_fast_flat_px.load(std::memory_order_relaxed);
        s.tri_general_px = tri_general_px.load(std::memory_order_relaxed);
        s.tri_fast_tex_cnt = tri_fast_tex_cnt.load(std::memory_order_relaxed);
        s.tri_fast_flat_cnt = tri_fast_flat_cnt.load(std::memory_order_relaxed);
        s.tri_general_cnt = tri_general_cnt.load(std::memory_order_relaxed);
        s.tri_gen_date_cnt = tri_gen_date_cnt.load(std::memory_order_relaxed);
        s.tri_gen_pabe_cnt = tri_gen_pabe_cnt.load(std::memory_order_relaxed);
        s.tri_gen_afail_cnt = tri_gen_afail_cnt.load(std::memory_order_relaxed);
        s.sprite_draws = sprite_draws.load(std::memory_order_relaxed);
        s.sprite_parallel_dispatches = sprite_parallel_dispatches.load(std::memory_order_relaxed);
        s.sprite_fast_clear = sprite_fast_clear.load(std::memory_order_relaxed);
        s.sprite_fast_copy = sprite_fast_copy.load(std::memory_order_relaxed);
        s.sprite_fast_simd = sprite_fast_simd.load(std::memory_order_relaxed);
        s.sprite_general = sprite_general.load(std::memory_order_relaxed);
        s.tex_rt_hits = tex_rt_hits.load(std::memory_order_relaxed);
        s.tex_cache_hits = tex_cache_hits.load(std::memory_order_relaxed);
        s.tex_cache_misses = tex_cache_misses.load(std::memory_order_relaxed);
        s.tex_cache_redecodes = tex_cache_redecodes.load(std::memory_order_relaxed);
        s.fb_flushes = fb_flushes.load(std::memory_order_relaxed);
        s.fb_reloads = fb_reloads.load(std::memory_order_relaxed);
        s.ee_interp_calls = ee_interp_calls.load(std::memory_order_relaxed);
        return s;
    }
};

inline PerfCounters g_perf{};

struct PerfThreadState {
    bool has_active = false;
    PerfCat cur_cat = PerfCat::EeRecomp;
    u64 start_ticks = 0;
};

inline thread_local PerfThreadState t_perf_state{};

// Exclusive hierarchical RAII scope timer: zero overhead when g_perf.enabled == false,
// and zero QPC overhead on same-category re-entry (e.g. EeRecomp -> EeRecomp).
class PerfScope {
public:
    explicit PerfScope(PerfCat cat) noexcept {
        if (!g_perf.enabled) return;
        PerfThreadState& st = t_perf_state;
        if (st.has_active && st.cur_cat == cat) return;
        const u64 now = perf_now_ticks();
        m_prev_has = st.has_active;
        m_prev_cat = st.cur_cat;
        if (m_prev_has) {
            g_perf.ticks[static_cast<std::size_t>(m_prev_cat)].fetch_add(
                now - st.start_ticks, std::memory_order_relaxed);
        }
        st.has_active = true;
        st.cur_cat = cat;
        st.start_ticks = now;
        m_cat = cat;
        m_active = true;
    }

    ~PerfScope() noexcept {
        if (!m_active) return;
        const u64 now = perf_now_ticks();
        PerfThreadState& st = t_perf_state;
        g_perf.ticks[static_cast<std::size_t>(m_cat)].fetch_add(
            now - st.start_ticks, std::memory_order_relaxed);
        st.has_active = m_prev_has;
        st.cur_cat = m_prev_cat;
        st.start_ticks = now;
    }

    PerfScope(const PerfScope&) = delete;
    PerfScope& operator=(const PerfScope&) = delete;

private:
    bool m_active = false;
    bool m_prev_has = false;
    PerfCat m_prev_cat = PerfCat::EeRecomp;
    PerfCat m_cat = PerfCat::EeRecomp;
};

} // namespace tf::ps2
