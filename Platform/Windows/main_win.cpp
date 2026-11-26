#include "../../Runtime/RuntimeHost.h"
#include "../../Runtime/PerfCounters.h"
#include "../../Graphics/Direct3D12/D3D11GsRenderer.h"
#include "../../Graphics/Direct3D12/Win32Presenter.h"
#ifdef _WIN32
#include "DesktopUi.h"
#include <windows.h>
#include <d3d11.h>
#include <mmsystem.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>

namespace {

using EntryPointFn    = tf::ps2::u32 (*)();
using FingerprintFn   = const char* (*)();
using RegisterFuncsFn = void (*)(void*);

void print_usage() {
    std::cout
        << "Transformers.exe — Native Windows x64 PS2 Static Recompilation Desktop App\n"
        << "Legal Notice: Requires your own legally obtained Transformers PS2 ISO.\n\n"
        << "Usage:\n"
        << "  Transformers.exe                         (Opens Transformers Desktop Launcher & Settings UI)\n"
        << "  Transformers.exe --iso <path.iso> [--module <recomp.dll>] [--headless] [--max-calls <N>]\n";
}

} // namespace

int main(int argc, char** argv) {
    std::string iso_path;
    std::string module_path;
    std::string memcard_dir = "MemoryCard0";
    std::string dump_frame_path;
    std::string dump_dir_path;
    bool headless = false;
    bool auto_press = false;
    bool profile = false;
    bool use_hw_gs = false;
    tf::ps2::u32 res_scale = 1u;
    int run_ms = 4000;
    tf::ps2::u64 max_calls = 200000;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage();
            return 0;
        } else if (arg == "--iso" && i + 1 < argc) {
            iso_path = argv[++i];
        } else if ((arg == "--module" || arg == "--dll") && i + 1 < argc) {
            module_path = argv[++i];
        } else if (arg == "--memcard" && i + 1 < argc) {
            memcard_dir = argv[++i];
        } else if (arg == "--dump-frame" && i + 1 < argc) {
            dump_frame_path = argv[++i];
        } else if (arg == "--dump-dir" && i + 1 < argc) {
            dump_dir_path = argv[++i];
        } else if (arg == "--max-calls" && i + 1 < argc) {
            max_calls = std::stoull(argv[++i]);
        } else if (arg == "--ms" && i + 1 < argc) {
            run_ms = std::stoi(argv[++i]);
        } else if (arg == "--auto-press") {
            auto_press = true;
        } else if (arg == "--profile") {
            profile = true;
        } else if (arg == "--hw-gs") {
            use_hw_gs = true;
        } else if (arg == "--res-scale" && i + 1 < argc) {
            res_scale = std::clamp<tf::ps2::u32>(static_cast<tf::ps2::u32>(std::stoul(argv[++i])), 1u, 4u);
        } else if (arg == "--headless") {
            headless = true;
        }
    }

    // Default interactive desktop GUI mode (no command line needed!)
    if (!headless) {
#ifdef _WIN32
        struct WinMmTimerGuard {
            WinMmTimerGuard() { timeBeginPeriod(1); }
            ~WinMmTimerGuard() { timeEndPeriod(1); }
        } timer_guard;
        tf::ps2::TransformersDesktopApp app;
        return app.run(iso_path, module_path);
#endif
    }

    if (iso_path.empty()) {
        print_usage();
        return 1;
    }

    tf::ps2::RuntimeHost runtime;
    runtime.max_static_calls = max_calls;
    std::string err;

    if (!runtime.mount_iso(iso_path, err)) {
        std::cerr << "[Runtime] Failed to mount ISO: " << err << "\n";
        return 2;
    }

    auto iso_rep = runtime.cdvd.iso().inspect();
    if (!iso_rep.valid) {
        std::cerr << "[Runtime] Invalid PS2 ISO: " << iso_rep.error << "\n";
        return 3;
    }

    std::cout << "[Runtime] Mounted ISO : " << iso_rep.system_cnf.serial
              << " (" << iso_rep.system_cnf.region << ")\n";
    std::cout << "[Runtime] Fingerprint : " << iso_rep.volume_fingerprint_sha256 << "\n";
    runtime.memcard0.mount_directory(memcard_dir, err);

    std::vector<tf::ps2::u8> elf_bytes;
    if (!runtime.cdvd.iso().read_file(iso_rep.boot_executable_entry, elf_bytes)) {
        std::cerr << "[Runtime] Failed reading BOOT2 executable from ISO\n";
        return 4;
    }
    tf::ps2::Ps2ElfImage elf;
    if (!tf::ps2::ElfParser::parse(elf_bytes, elf, err) || !runtime.load_elf(elf, err)) {
        std::cerr << "[Runtime] Failed loading PS2 ELF: " << err << "\n";
        return 5;
    }
    std::cout << "[Runtime] Loaded ELF into 32 MiB EE RAM (entry=" << tf::ps2::hex32(elf.e_entry) << ")\n";

    tf::ps2::u32 entry_addr = elf.e_entry;
#ifdef _WIN32
    HMODULE mod = nullptr;
    if (!module_path.empty()) {
        mod = LoadLibraryA(module_path.c_str());
        if (!mod) {
            std::cerr << "[Runtime] Failed loading recompiled native module: " << module_path << "\n";
            return 7;
        }
        auto fp_fn = reinterpret_cast<FingerprintFn>(reinterpret_cast<void*>(GetProcAddress(mod, "tf_recomp_iso_fingerprint")));
        auto ep_fn = reinterpret_cast<EntryPointFn>(reinterpret_cast<void*>(GetProcAddress(mod, "tf_recomp_entry_point")));
        auto reg_fn = reinterpret_cast<RegisterFuncsFn>(reinterpret_cast<void*>(GetProcAddress(mod, "tf_recomp_register")));
        if (!fp_fn || !ep_fn || !reg_fn) {
            std::cerr << "[Runtime] Recompiled module missing required exports\n";
            FreeLibrary(mod);
            return 8;
        }
        if (std::string(fp_fn()) != iso_rep.volume_fingerprint_sha256) {
            std::cerr << "[Runtime] ISO fingerprint mismatch!\n";
            FreeLibrary(mod);
            return 9;
        }
        entry_addr = ep_fn();
        reg_fn(&runtime.function_table());
        std::cout << "[Runtime] Registered " << runtime.function_table().size()
                  << " statically recompiled native x64 functions\n";
    }
#endif

#ifdef _WIN32
    ID3D11Device* headless_d3d_dev = nullptr;
    ID3D11DeviceContext* headless_d3d_ctx = nullptr;
    std::mutex headless_d3d_mtx;
    tf::ps2::D3D11GsRenderer headless_hw_gs;
    bool headless_hw_gs_ready = false;
    if (use_hw_gs) {
        static constexpr D3D_FEATURE_LEVEL kLevels[] = {
            D3D_FEATURE_LEVEL_11_1,
            D3D_FEATURE_LEVEL_11_0,
            D3D_FEATURE_LEVEL_10_1,
            D3D_FEATURE_LEVEL_10_0
        };
        D3D_FEATURE_LEVEL chosen = D3D_FEATURE_LEVEL_11_0;
        HRESULT hr = D3D11CreateDevice(
            nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
            D3D11_CREATE_DEVICE_BGRA_SUPPORT,
            kLevels, static_cast<UINT>(std::size(kLevels)),
            D3D11_SDK_VERSION,
            &headless_d3d_dev, &chosen, &headless_d3d_ctx);
        if (FAILED(hr) || !headless_d3d_dev || !headless_d3d_ctx) {
            hr = D3D11CreateDevice(
                nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
                D3D11_CREATE_DEVICE_BGRA_SUPPORT,
                kLevels + 1, static_cast<UINT>(std::size(kLevels) - 1),
                D3D11_SDK_VERSION,
                &headless_d3d_dev, &chosen, &headless_d3d_ctx);
        }
        if (SUCCEEDED(hr) && headless_d3d_dev && headless_d3d_ctx) {
            headless_hw_gs_ready = headless_hw_gs.init(headless_d3d_dev, headless_d3d_ctx, &headless_d3d_mtx);
            if (headless_hw_gs_ready) {
                headless_hw_gs.set_resolution_scale(res_scale);
                runtime.gs.set_hw_backend(&headless_hw_gs);
                std::cout << "[Runtime] D3D11 Hardware GS Renderer enabled (res_scale=" << res_scale << "x)\n";
            } else {
                std::cerr << "[Runtime] WARNING: D3D11GsRenderer::init() failed; falling back to CPU software GS\n";
            }
        } else {
            std::cerr << "[Runtime] WARNING: Headless D3D11CreateDevice failed; falling back to CPU software GS\n";
        }
    }
#else
    (void)use_hw_gs;
    (void)res_scale;
#endif

    auto read_presented_frame = [&](std::vector<tf::ps2::u32>& out_rgba, tf::ps2::u32& out_w, tf::ps2::u32& out_h) -> bool {
#ifdef _WIN32
        if (headless_hw_gs_ready && headless_hw_gs.readback_presented_rgba(out_rgba, out_w, out_h)) {
            return true;
        }
#endif
        return runtime.gs.copy_presented_framebuffer(out_rgba, out_w, out_h);
    };

    std::cout << "[Runtime] Calling recompiled entry point " << tf::ps2::hex32(entry_addr) << "..." << std::endl;
    if (profile) {
        tf::ps2::g_perf.reset();
        tf::ps2::g_perf.enabled = true;
    }
    std::atomic<bool> done{false};
    std::thread runner([&]() {
        runtime.call_guest(entry_addr);
        done = true;
    });
    auto save_bmp = [](const std::string& path, const std::vector<tf::ps2::u32>& px_buf, tf::ps2::u32 w, tf::ps2::u32 h) {
        if (px_buf.size() < static_cast<std::size_t>(w) * h) return;
        tf::ps2::u32 img_sz = w * h * 3u;
        tf::ps2::u32 file_sz = 54u + img_sz;
        tf::ps2::u8 hdr[54]{};
        hdr[0] = 'B'; hdr[1] = 'M';
        tf::ps2::write_le32(hdr + 2, file_sz);
        tf::ps2::write_le32(hdr + 10, 54u);
        tf::ps2::write_le32(hdr + 14, 40u);
        tf::ps2::write_le32(hdr + 18, w);
        tf::ps2::write_le32(hdr + 22, h);
        tf::ps2::write_le16(hdr + 26, 1u);
        tf::ps2::write_le16(hdr + 28, 24u);
        tf::ps2::write_le32(hdr + 34, img_sz);
        std::ofstream ofs(path, std::ios::binary);
        ofs.write(reinterpret_cast<const char*>(hdr), 54);
        std::vector<tf::ps2::u8> row(w * 3u);
        for (int y = static_cast<int>(h) - 1; y >= 0; --y) {
            for (tf::ps2::u32 x = 0; x < w; ++x) {
                tf::ps2::u32 px = px_buf[static_cast<std::size_t>(y) * w + x];
                row[x * 3 + 0] = static_cast<tf::ps2::u8>((px >> 16) & 0xFFu);
                row[x * 3 + 1] = static_cast<tf::ps2::u8>((px >> 8) & 0xFFu);
                row[x * 3 + 2] = static_cast<tf::ps2::u8>(px & 0xFFu);
            }
            ofs.write(reinterpret_cast<const char*>(row.data()), static_cast<std::streamsize>(row.size()));
        }
    };

    int snap_idx = 0;
    tf::ps2::u64 last_hash = 0;
    tf::ps2::u64 last_state_key = ~0ULL;
    auto prev_prof_tp = std::chrono::steady_clock::now();
    tf::ps2::PerfSnapshot prev_prof = tf::ps2::g_perf.snapshot();
    tf::ps2::u64 prev_prof_frames = 0;
    tf::ps2::u64 prev_prof_vu1_steps = 0;
    tf::ps2::u64 prev_prof_mscal = 0;
    tf::ps2::u64 prev_prof_xgkick = 0;
    tf::ps2::u64 prev_prof_fb_steps = 0;
    tf::ps2::u64 prev_prof_prims = 0;
    tf::ps2::u64 prev_prof_px = 0;
    auto dump_profile_window = [&](int cur_ms, tf::ps2::u32 cur_boot, tf::ps2::u32 cur_tscr, const char* tag) {
        if (!profile) return;
        const auto now_tp = std::chrono::steady_clock::now();
        const double wall_ms = std::chrono::duration<double, std::milli>(now_tp - prev_prof_tp).count();
        const tf::ps2::PerfSnapshot cur_prof = tf::ps2::g_perf.snapshot();
        const tf::ps2::u64 cur_frames = runtime.gs.frames_presented;
        const tf::ps2::u64 df = (cur_frames >= prev_prof_frames) ? (cur_frames - prev_prof_frames) : 0;
        const tf::ps2::u64 cur_vu1_steps = runtime.vu1.micro_steps_executed;
        const tf::ps2::u64 cur_mscal = runtime.vif1.mscal_count;
        const tf::ps2::u64 cur_xgkick = runtime.vu1.xgkick_count;
        const tf::ps2::u64 cur_fb_steps = runtime.fallback_steps_executed;
        const tf::ps2::u64 cur_prims = runtime.gs.primitives_drawn;
        const tf::ps2::u64 cur_px = runtime.gs.pixels_drawn;
        if (df > 0 && wall_ms > 1.0) {
            auto cat_ms = [&](tf::ps2::PerfCat c) -> double {
                const std::size_t idx = static_cast<std::size_t>(c);
                const tf::ps2::u64 dt = (cur_prof.ticks[idx] >= prev_prof.ticks[idx]) ? (cur_prof.ticks[idx] - prev_prof.ticks[idx]) : 0;
                return tf::ps2::perf_ticks_to_ms(dt) / static_cast<double>(df);
            };
            const double m_vu1   = cat_ms(tf::ps2::PerfCat::Vu1);
            const double m_dma   = cat_ms(tf::ps2::PerfCat::VifDma);
            const double m_unp   = cat_ms(tf::ps2::PerfCat::VifUnpack);
            const double m_gif   = cat_ms(tf::ps2::PerfCat::GifParse);
            const double m_tri   = cat_ms(tf::ps2::PerfCat::GsTri);
            const double m_spr   = cat_ms(tf::ps2::PerfCat::GsSprite);
            const double m_tex   = cat_ms(tf::ps2::PerfCat::GsTex);
            const double m_vram  = cat_ms(tf::ps2::PerfCat::GsVram);
            const double m_pres  = cat_ms(tf::ps2::PerfCat::GsPresent);
            const double m_int   = cat_ms(tf::ps2::PerfCat::EeInterp);
            const double m_rec   = cat_ms(tf::ps2::PerfCat::EeRecomp);
            const double m_sum   = m_vu1 + m_dma + m_unp + m_gif + m_tri + m_spr + m_tex + m_vram + m_pres + m_int + m_rec;
            const double m_wall  = wall_ms / static_cast<double>(df);
            const double fps_wall = static_cast<double>(df) * 1000.0 / wall_ms;
            std::cout << std::fixed << std::setprecision(2)
                      << "[PROFILE " << tag << " t=" << cur_ms << "ms boot=" << cur_boot << " scr=" << cur_tscr
                      << " df=" << df << " wall_fps=" << fps_wall << " wall_ms=" << m_wall << " sum_ms=" << m_sum << "]\n"
                      << "  ms/frm: vu1=" << m_vu1
                      << " vif_dma=" << (m_dma + m_unp) << "(dma=" << m_dma << "+unp=" << m_unp << ")"
                      << " gif=" << m_gif
                      << " gs_tri=" << m_tri
                      << " gs_spr=" << m_spr
                      << " gs_tex=" << m_tex
                      << " gs_vram=" << m_vram
                      << " gs_pres=" << m_pres
                      << " ee_int=" << m_int
                      << " ee_rec=" << m_rec << "\n"
                      << "  tri/frm: batches=" << (cur_prof.tri_batches - prev_prof.tri_batches) / static_cast<double>(df)
                      << " tris=" << (cur_prof.tri_batch_tris - prev_prof.tri_batch_tris) / static_cast<double>(df)
                      << " par_disp=" << (cur_prof.tri_parallel_dispatches - prev_prof.tri_parallel_dispatches) / static_cast<double>(df)
                      << " single=" << (cur_prof.tri_single_draws - prev_prof.tri_single_draws) / static_cast<double>(df)
                      << " | px/frm(K): tex_simd=" << (cur_prof.tri_fast_tex_px - prev_prof.tri_fast_tex_px) / (1000.0 * df)
                      << " flat_simd=" << (cur_prof.tri_fast_flat_px - prev_prof.tri_fast_flat_px) / (1000.0 * df)
                      << " gen_fb=" << (cur_prof.tri_general_px - prev_prof.tri_general_px) / (1000.0 * df)
                      << " (gen_tris=" << (cur_prof.tri_general_cnt - prev_prof.tri_general_cnt) / static_cast<double>(df)
                      << " date=" << (cur_prof.tri_gen_date_cnt - prev_prof.tri_gen_date_cnt) / static_cast<double>(df)
                      << " pabe=" << (cur_prof.tri_gen_pabe_cnt - prev_prof.tri_gen_pabe_cnt) / static_cast<double>(df)
                      << " afail=" << (cur_prof.tri_gen_afail_cnt - prev_prof.tri_gen_afail_cnt) / static_cast<double>(df) << ")\n"
                      << "  spr/frm: tot=" << (cur_prof.sprite_draws - prev_prof.sprite_draws) / static_cast<double>(df)
                      << " clr=" << (cur_prof.sprite_fast_clear - prev_prof.sprite_fast_clear) / static_cast<double>(df)
                      << " cpy=" << (cur_prof.sprite_fast_copy - prev_prof.sprite_fast_copy) / static_cast<double>(df)
                      << " simd=" << (cur_prof.sprite_fast_simd - prev_prof.sprite_fast_simd) / static_cast<double>(df)
                      << "(par=" << (cur_prof.sprite_parallel_dispatches - prev_prof.sprite_parallel_dispatches) / static_cast<double>(df) << ")"
                      << " gen=" << (cur_prof.sprite_general - prev_prof.sprite_general) / static_cast<double>(df)
                      << " | tex/frm: rt=" << (cur_prof.tex_rt_hits - prev_prof.tex_rt_hits) / static_cast<double>(df)
                      << " hit=" << (cur_prof.tex_cache_hits - prev_prof.tex_cache_hits) / static_cast<double>(df)
                      << " miss=" << (cur_prof.tex_cache_misses - prev_prof.tex_cache_misses) / static_cast<double>(df)
                      << " redec=" << (cur_prof.tex_cache_redecodes - prev_prof.tex_cache_redecodes) / static_cast<double>(df)
                      << " | fb_fl=" << (cur_prof.fb_flushes - prev_prof.fb_flushes) / static_cast<double>(df)
                      << " fb_rl=" << (cur_prof.fb_reloads - prev_prof.fb_reloads) / static_cast<double>(df) << "\n"
                      << "  vu1/frm: steps=" << (cur_vu1_steps - prev_prof_vu1_steps) / static_cast<double>(df)
                      << " mscal=" << (cur_mscal - prev_prof_mscal) / static_cast<double>(df)
                      << " xgkick=" << (cur_xgkick - prev_prof_xgkick) / static_cast<double>(df)
                      << " | ee_int: calls=" << (cur_prof.ee_interp_calls - prev_prof.ee_interp_calls) / static_cast<double>(df)
                      << " steps=" << (cur_fb_steps - prev_prof_fb_steps) / static_cast<double>(df)
                      << " | prims=" << (cur_prims - prev_prof_prims) / static_cast<double>(df)
                      << " tot_px(K)=" << (cur_px - prev_prof_px) / (1000.0 * df)
                      << std::defaultfloat << std::endl;
        }
        prev_prof_tp = now_tp;
        prev_prof = cur_prof;
        prev_prof_frames = cur_frames;
        prev_prof_vu1_steps = cur_vu1_steps;
        prev_prof_mscal = cur_mscal;
        prev_prof_xgkick = cur_xgkick;
        prev_prof_fb_steps = cur_fb_steps;
        prev_prof_prims = cur_prims;
        prev_prof_px = cur_px;
    };
    for (int ms = 0; ms < run_ms && !done.load(); ms += 25) {
        if (auto_press && ms >= 3500) {
            // Pulse Start and Cross alternatingly to advance past splash/dialogs in headless testing
            int phase = (ms / 350) % 4;
            std::lock_guard<std::mutex> pad_lk(runtime.pad_mutex());
            runtime.pad0.set_button(tf::ps2::Ps2Button::Start, phase == 0);
            runtime.pad0.set_button(tf::ps2::Ps2Button::Cross, phase == 2);
        }
        tf::ps2::u32 cur_ucnt = runtime.memory.read32(0x004B2EA4u);
        tf::ps2::u32 cur_tscr = 0;
        if (cur_ucnt > 0 && cur_ucnt < 16) {
            tf::ps2::u32 stk = 0x0060C3D8u + cur_ucnt * 224u;
            tf::ps2::s16 sidx_d = static_cast<tf::ps2::s16>(runtime.memory.read16(stk + 20u));
            if (sidx_d >= 0 && sidx_d < 256) {
                cur_tscr = runtime.memory.read32(0x004B30C8u + static_cast<tf::ps2::u32>(sidx_d) * 28u);
            }
        }
        tf::ps2::u32 cur_boot  = runtime.memory.read32(0x00503B68u + 48u);
        tf::ps2::u32 cur_ui_en = runtime.memory.read32(0x004B2ECCu);
        tf::ps2::u32 cur_s1019 = runtime.memory.read32(0x004B2EC4u);
        tf::ps2::u32 cur_bg    = runtime.memory.read32(0x005580E8u);
        tf::ps2::u64 state_key = (static_cast<tf::ps2::u64>(cur_boot) << 48) |
                                 (static_cast<tf::ps2::u64>(cur_tscr & 0xFFFFu) << 32) |
                                 (static_cast<tf::ps2::u64>(cur_ui_en & 0xFFu) << 24) |
                                 (static_cast<tf::ps2::u64>(cur_s1019 & 0xFFu) << 16) |
                                 (static_cast<tf::ps2::u64>(cur_bg & 0xFFu) << 8) |
                                 static_cast<tf::ps2::u64>(cur_ucnt & 0xFFu);
        bool state_changed = (state_key != last_state_key);
        if (state_changed) last_state_key = state_key;

        if (ms > 0 && ((ms % 5000 == 0) || state_changed)) {
            std::vector<tf::ps2::u32> cur;
            tf::ps2::u32 cw = 0, ch = 0;
            tf::ps2::u64 nb = 0;
            if (read_presented_frame(cur, cw, ch)) {
                for (std::size_t i = 0; i < cur.size(); i += 13) if (cur[i] & 0x00FFFFFFu) nb++;
            }
            tf::ps2::u32 prog = runtime.memory.read32(0x004A4B1Cu);
            tf::ps2::u32 strp = runtime.memory.read32(0x005573B0u);
            tf::ps2::u32 log_n = runtime.memory.read32(0x00503B68u + 1024u);
            std::string stage;
            if (strp >= 0x00100000u && strp < tf::ps2::kEeRamSize - 64u) {
                for (tf::ps2::u32 k = 0; k < 48; ++k) {
                    char ch = static_cast<char>(runtime.memory.read8(strp + k));
                    if (!ch) break;
                    if (ch >= 32 && ch < 127) stage.push_back(ch);
                }
            }
            std::cout << "[t=" << ms << "ms] frm=" << runtime.gs.frames_presented
                      << " boot=" << cur_boot
                      << " ui=" << cur_ucnt << " scr=" << cur_tscr
                      << " en=" << cur_ui_en << " s19=" << cur_s1019
                      << " bg=" << cur_bg << " prog=" << prog << " logs=" << log_n << " stg=\"" << stage << "\""
                      << " rd=" << runtime.cdvd.read_requests()
                      << " vb=" << runtime.memory.read32(0x004B1754u)
                      << " nb=" << nb << std::endl;
            if (profile && ((ms % 5000 == 0) || state_changed)) {
                dump_profile_window(ms, cur_boot, cur_tscr, (ms % 5000 == 0) ? "5s" : "state");
            }
        }
        bool want_snap = !dump_dir_path.empty() && ms > 0 &&
                         ((ms <= 12000 && (ms % 1000 == 0)) ||
                          (cur_ui_en != 0 && (ms % 1000 == 0)) ||
                          (cur_bg != 0 && (ms % 1000 == 0)) ||
                          state_changed ||
                          (ms % 10000 == 0));
        if (want_snap) {
            std::vector<tf::ps2::u32> cur;
            tf::ps2::u32 cw = 0, ch = 0;
            if (read_presented_frame(cur, cw, ch)) {
                tf::ps2::u64 non_b = 0, h = 1469598103934665603ULL;
                for (std::size_t i = 0; i < cur.size(); i += 7) {
                    tf::ps2::u32 rgb = cur[i] & 0x00FFFFFFu;
                    if (rgb) non_b++;
                    h = (h ^ rgb) * 1099511628211ULL;
                }
                if (non_b > 100 && h != last_hash) {
                    last_hash = h;
                    save_bmp(dump_dir_path + "/snap_" + std::to_string(ms) + "ms_scr" + std::to_string(cur_tscr) + ".bmp",
                             cur, cw, ch);
                    snap_idx++;
                }
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }
    bool timed_out = !done.load();
    if (timed_out) {
        runtime.request_halt();
        runtime.cpu.trap_message = "Live trace capture complete";
        for (int wait_ms = 0; wait_ms < 2000 && !done.load(); wait_ms += 10) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    bool runner_joined = false;
    if (done.load() && runner.joinable()) {
        runner.join();
        runner_joined = true;
    }
    std::cout << "[Runtime] Trace Summary:\n"
              << "  PC              : " << tf::ps2::hex32(runtime.cpu.pc) << " (ra=" << tf::ps2::hex32(runtime.cpu.gpr_u32(31)) << ")\n"
              << "  Halted          : " << (runtime.cpu.halted ? "yes" : "no") << "\n"
              << "  Static Calls    : " << runtime.total_static_calls << "\n"
              << "  Fallback Steps  : " << runtime.fallback_steps_executed << "\n"
              << "  Indirect Missed : " << runtime.indirect_calls_missed << " (last=" << tf::ps2::hex32(runtime.last_missed_target) << ")\n"
              << "  Invalid Mem Ops : " << runtime.memory.invalid_access_count() << " (last=" << tf::ps2::hex32(runtime.memory.last_invalid_address()) << ")\n"
              << "  GS Primitives   : " << runtime.gs.primitives_drawn
              << " (textured=" << runtime.gs.textured_primitives
              << ", hwreg_uploads=" << runtime.gs.hwreg_uploads
              << ", " << runtime.gs.pixels_drawn << " px)\n"
              << "  Message         : " << runtime.cpu.trap_message << "\n";

    std::cout << "  Syscalls Called : ";
    for (const auto& [num, cnt] : runtime.syscalls.syscall_counts()) {
        std::cout << "0x" << std::hex << num << std::dec << "(" << cnt << ") ";
    }
    std::cout << "\n";
    if (!runtime.syscalls.unhandled_syscalls().empty()) {
        std::cout << "  Unhandled Syscalls: ";
        for (const auto& [num, cnt] : runtime.syscalls.unhandled_syscalls()) {
            std::cout << "0x" << std::hex << num << std::dec << "(" << cnt << ") ";
        }
        std::cout << "\n";
    }
    if (!runtime.mmio_read_counts().empty()) {
        std::cout << "  MMIO Reads      : ";
        for (const auto& [addr, cnt] : runtime.mmio_read_counts()) {
            std::cout << tf::ps2::hex32(addr) << "(" << cnt << ") ";
        }
        std::cout << "\n";
    }
    if (!runtime.bound_sif_sids().empty()) {
        std::cout << "  Bound SIF SIDs  : ";
        for (tf::ps2::u32 sid : runtime.bound_sif_sids()) {
            std::cout << tf::ps2::hex32(sid) << " ";
        }
        std::cout << "\n";
    }
    if (!runtime.iop.loaded_irx_modules().empty()) {
        std::cout << "  Loaded IRX Mods : ";
        for (const auto& m : runtime.iop.loaded_irx_modules()) {
            std::cout << m << " ";
        }
        std::cout << "\n";
    }
    if (!runtime.fn_call_counts().empty()) {
        std::vector<std::pair<tf::ps2::u64, tf::ps2::u32>> top;
        for (const auto& [addr, cnt] : runtime.fn_call_counts()) top.push_back({cnt, addr});
        std::sort(top.begin(), top.end(), std::greater<>());
        std::cout << "  Top Hot Fns     : ";
        for (std::size_t i = 0; i < std::min<std::size_t>(12, top.size()); ++i) {
            std::cout << tf::ps2::hex32(top[i].second) << "(" << top[i].first << ") ";
        }
        std::cout << "\n";
    }
    if (!runtime.iop.file_access_log().empty()) {
        std::cout << "  ISO File Access :\n";
        for (const auto& f : runtime.iop.file_access_log()) {
            std::cout << "    " << f << "\n";
        }
    }
    if (!runtime.sif_rpc_trace().empty()) {
        std::cout << "  SIF RPC Calls   :\n";
        for (const auto& t : runtime.sif_rpc_trace()) {
            std::cout << "    " << t << "\n";
        }
    }
    tf::ps2::u32 pad0_obj = runtime.memory.read32(0x004B5928u);
    tf::ps2::u32 pad0_st = (pad0_obj >= 0x00100000u && pad0_obj < tf::ps2::kEeRamSize - 1344u) ? runtime.memory.read32(pad0_obj + 1288u) : 0u;
    tf::ps2::u32 pad0_flg = (pad0_obj >= 0x00100000u && pad0_obj < tf::ps2::kEeRamSize - 1344u) ? runtime.memory.read32(pad0_obj + 1312u) : 0u;
    tf::ps2::u32 pad_conn = runtime.memory.read8(0x00636260u);
    tf::ps2::u32 ui_cnt = runtime.memory.read32(0x004B2EA4u);
    tf::ps2::u32 top_scr_id = 0;
    if (ui_cnt > 0 && ui_cnt < 16) {
        tf::ps2::u32 stk = 0x0060C3D8u + ui_cnt * 224u;
        tf::ps2::s16 sidx = static_cast<tf::ps2::s16>(runtime.memory.read16(stk + 20u));
        if (sidx >= 0 && sidx < 256) {
            top_scr_id = runtime.memory.read32(0x004B30C8u + static_cast<tf::ps2::u32>(sidx) * 28u);
        }
    }
    std::cout << "  Game State Regs : boot_st=" << runtime.memory.read32(0x00503B68u + 48u)
              << " boot_flg=" << tf::ps2::hex32(runtime.memory.read32(0x00503B68u + 16u))
              << " pad_st=" << runtime.memory.read32(0x00503B68u + 52u)
              << " pad_conn=" << pad_conn
              << " pad0_fsm=" << pad0_st
              << " pad0_flg=" << tf::ps2::hex32(pad0_flg)
              << " ui_cnt=" << ui_cnt
              << " top_scr=" << top_scr_id
              << " mc_st=" << runtime.memory.read32(0x004B1DECu)
              << " scr1019_st=" << runtime.memory.read32(0x004B2EC4u)
              << " ui_en=" << runtime.memory.read32(0x004B2ECCu)
              << " snaps=" << snap_idx << "\n";
    std::cout << "  DMA/VIF Sync    : d1_chcr=" << tf::ps2::hex32(runtime.dmac.channel(tf::ps2::DmaChannelId::VIF1).chcr)
              << " d1_madr=" << tf::ps2::hex32(runtime.dmac.channel(tf::ps2::DmaChannelId::VIF1).madr)
              << " d1_tadr=" << tf::ps2::hex32(runtime.dmac.channel(tf::ps2::DmaChannelId::VIF1).tadr)
              << " d2_chcr=" << tf::ps2::hex32(runtime.dmac.channel(tf::ps2::DmaChannelId::GIF).chcr)
              << " d2_madr=" << tf::ps2::hex32(runtime.dmac.channel(tf::ps2::DmaChannelId::GIF).madr)
              << " d2_tadr=" << tf::ps2::hex32(runtime.dmac.channel(tf::ps2::DmaChannelId::GIF).tadr)
              << " vif_busy=" << runtime.memory.read32(0x004B1754u)
              << " vif_done=" << runtime.memory.read32(0x004B177Cu)
              << " vif_w_gif=" << runtime.memory.read32(0x004B1598u)
              << " gif_w_vif=" << runtime.memory.read32(0x004B159Cu)
              << " gif_busy=" << runtime.memory.read32(0x004B1750u) << "\n";
    {
        tf::ps2::u32 ln = runtime.memory.read32(0x00503B68u + 1024u);
        std::cout << "  Load Stages (" << ln << "): ";
        for (tf::ps2::u32 i = 0; i < ln && i < 32; ++i) {
            tf::ps2::u32 base = 0x00503B68u + 1032u + i * 28u;
            std::string nm;
            for (tf::ps2::u32 k = 0; k < 16; ++k) {
                char c = static_cast<char>(runtime.memory.read8(base + k));
                if (!c) break;
                if (c >= 32 && c < 127) nm.push_back(c);
            }
            std::cout << "[" << i << ":" << nm << "(" << (runtime.memory.read32(base + 16u) >> 10) << "K)] ";
        }
        std::cout << "\n";
    }
    std::cout << "  CDVD Sectors Rd : " << runtime.cdvd.sectors_read()
              << " (" << runtime.cdvd.read_requests() << " reqs)\n";
    {
        std::vector<tf::ps2::u32> safe_fb;
        tf::ps2::u32 safe_w = 0, safe_h = 0;
        const bool have_safe = read_presented_frame(safe_fb, safe_w, safe_h);
        const std::vector<tf::ps2::u32>& fb = have_safe ? safe_fb : (runner_joined ? runtime.gs.framebuffer_rgba() : safe_fb);
        const std::vector<tf::ps2::u32>& dump_fb =
            have_safe
                ? safe_fb
                : ((runner_joined && !runtime.gs.last_nonblack_rgba().empty())
                       ? runtime.gs.last_nonblack_rgba()
                       : fb);
        tf::ps2::u64 non_black = 0;
        for (tf::ps2::u32 px : fb) if ((px & 0x00FFFFFFu) != 0) non_black++;
        tf::ps2::u64 peak_non_black = 0;
        for (tf::ps2::u32 px : dump_fb) if ((px & 0x00FFFFFFu) != 0) peak_non_black++;
        const tf::ps2::u32 fb_w = have_safe ? safe_w : runtime.gs.fb_width();
        const tf::ps2::u32 fb_h = have_safe ? safe_h : runtime.gs.fb_height();
        std::cout << "  GS Framebuffer  : " << fb_w << "x" << fb_h
                  << " (non-black px=" << non_black << ", peak=" << peak_non_black
                  << ", vif1_direct_qwc=" << runtime.vif1.direct_qwords_forwarded
                  << ", vif1_mpg=" << runtime.vif1.mpg_uploads
                  << ", vif1_unpack=" << runtime.vif1.unpacks_processed
                  << ", vif1_mscal=" << runtime.vif1.mscal_count
                  << ", vu1_steps=" << runtime.vu1.micro_steps_executed
                  << ", vu1_jit_blks=" << runtime.vu1.jit_blocks_compiled()
                  << ", xgkick=" << runtime.vu1.xgkick_count << ")\n";
        for (const auto& s : runtime.gs.debug_log) {
            std::cout << "    [GS] " << s << "\n";
        }
        for (const auto& s : runtime.vu1.debug_log) {
            std::cout << "    [VU1] " << s << "\n";
        }
        if (!dump_frame_path.empty() && dump_fb.size() >= static_cast<std::size_t>(fb_w) * fb_h && fb_w > 0 && fb_h > 0) {
            save_bmp(dump_frame_path, dump_fb, fb_w, fb_h);
            std::cout << "  Dumped Frame    : " << dump_frame_path << "\n";
        }
    }
    std::cout << std::flush;

    if (runner.joinable()) {
        runner.detach();
        std::_Exit(0);
    }

#ifdef _WIN32
    runtime.gs.set_hw_backend(nullptr);
    headless_hw_gs.shutdown();
    if (headless_d3d_ctx) { headless_d3d_ctx->Release(); headless_d3d_ctx = nullptr; }
    if (headless_d3d_dev) { headless_d3d_dev->Release(); headless_d3d_dev = nullptr; }
    if (mod) FreeLibrary(mod);
#endif
    return 0;
}
