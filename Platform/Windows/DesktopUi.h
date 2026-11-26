#pragma once

#ifdef _WIN32

#include "../../Runtime/RuntimeHost.h"
#include "../../Runtime/Config/AppSettings.h"
#include "../../Graphics/Direct3D12/D3D11GsRenderer.h"
#include "../../Graphics/Direct3D12/Win32Presenter.h"

#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace tf::ps2 {

// Native Windows x64 Desktop Application & Settings UI for Transformers: The Game PS2 Port.
class TransformersDesktopApp {
public:
    TransformersDesktopApp();
    ~TransformersDesktopApp();

    int run(const std::string& initial_iso = "", const std::string& initial_module = "");

private:
    static LRESULT CALLBACK main_wnd_proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp);
    static LRESULT CALLBACK viewport_wnd_proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp);
    static LRESULT CALLBACK settings_wnd_proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp);
    static DWORD WINAPI guest_thread_proc(LPVOID param);
    static LONG WINAPI unhandled_exception_filter(EXCEPTION_POINTERS* ep);

    LRESULT handle_main_msg(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp);
    LRESULT handle_viewport_msg(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp);
    LRESULT handle_settings_msg(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp);

    void create_main_controls(HWND hwnd);
    void layout_controls(int width, int height);
    void open_settings_dialog();
    void apply_settings_from_dialog(HWND dlg);
    void on_browse_memory_card(HWND dlg);
    void toggle_fullscreen(bool enable);

    void on_select_iso();
    void on_recompile_iso();
    void on_launch_game();
    void on_stop_game();

    bool load_iso_and_auto_detect_module(const std::string& iso_path, bool preserve_active_module = false);
    std::string find_recompiled_module_for_serial(const std::string& serial) const;
    bool verify_module_matches_iso(const std::string& dll_path, const std::string& expected_fp, std::string* out_reason = nullptr) const;
    void update_iso_info_label();
    void refresh_active_module_ready();
    void update_button_states();
    void append_log(const std::string& line);
    void refresh_status_ui();
    void update_true_fps(u64 now_ms);

    void init_xinput();
    void shutdown_xinput();
    void set_xinput_vibration(u16 left_motor, u16 right_motor);
    void record_kb_event(WPARAM vk, bool pressed);
    void sample_jit_input();
    void poll_input();
    bool present_current_frame();
    bool redraw_last_frame();

    HWND m_hwnd = nullptr;
    HWND m_viewport = nullptr;
    HWND m_btn_iso = nullptr;
    HWND m_btn_recomp = nullptr;
    HWND m_btn_play = nullptr;
    HWND m_btn_stop = nullptr;
    HWND m_btn_settings = nullptr;
    HWND m_lbl_title = nullptr;
    HWND m_lbl_subtitle = nullptr;
    HWND m_lbl_iso_info = nullptr;
    HWND m_lbl_telemetry = nullptr;
    HWND m_lbl_controls = nullptr;
    HWND m_edit_log = nullptr;

    HBRUSH m_brush_bg = nullptr;
    HBRUSH m_brush_panel = nullptr;
    HBRUSH m_brush_edit = nullptr;
    HFONT  m_font_title = nullptr;
    HFONT  m_font_sub = nullptr;
    HFONT  m_font_ui = nullptr;
    HFONT  m_font_mono = nullptr;

    // Settings dialog controls
    HWND m_dlg_settings = nullptr;
    std::atomic<HWND> m_dlg_settings_atomic{nullptr};
    HWND m_lbl_sec_gfx = nullptr;
    HWND m_lbl_sec_ctrl = nullptr;
    HWND m_lbl_sec_audio = nullptr;
    HWND m_cb_res = nullptr;
    HWND m_cb_aspect = nullptr;
    HWND m_chk_vsync = nullptr;
    HWND m_chk_bilinear = nullptr;
    HWND m_chk_fullscreen = nullptr;
    HWND m_cb_deadzone = nullptr;
    HWND m_cb_sens = nullptr;
    HWND m_chk_inv_x = nullptr;
    HWND m_chk_inv_y = nullptr;
    HWND m_chk_vibrate = nullptr;
    HWND m_chk_audio = nullptr;
    HWND m_cb_vol = nullptr;
    HWND m_edit_memcard = nullptr;
    HWND m_btn_memcard_browse = nullptr;

    // Dynamic XInput loader (xinput1_4.dll / xinput1_3.dll / xinput9_1_0.dll)
    HMODULE m_xinput_dll = nullptr;
    void*   m_xinput_get_state = nullptr;
    void*   m_xinput_set_state = nullptr;
    std::atomic<bool> m_xinput_connected{false};
    std::atomic<bool> m_vibration_enabled{true};
    u16     m_last_vib_left = 0;
    u16     m_last_vib_right = 0;
    std::atomic<u32> m_kb_pressed_mask{0};
    std::atomic<u32> m_kb_edge_latch{0};

    // Fullscreen state
    bool  m_is_fullscreen = false;
    DWORD m_saved_style = 0;
    DWORD m_saved_ex_style = 0;
    RECT  m_saved_rect{};

    AppSettings m_settings;
    RuntimeHost m_runtime;
    Win32Presenter m_presenter;
    D3D11GsRenderer m_hw_gs;
    bool m_hw_gs_active = false;
    u64 m_last_hw_presented_seq = 0;
    bool m_last_frame_was_hw = false;
    bool m_force_redraw = false;
    bool m_app_active = true;
    IsoInspectionReport m_iso_report;
    Ps2ElfImage m_elf_image;
    bool m_iso_loaded = false;
    std::string m_active_module_path;
    bool m_active_module_ready = false;
    HMODULE m_loaded_dll = nullptr;

    // Thread-safe UI copy of the latest presented PS2 GS framebuffer
    std::vector<u32> m_frame_copy;
    u32 m_frame_w = 0;
    u32 m_frame_h = 0;
    bool m_has_frame = false;
    u64 m_last_presented_seq = 0;

    std::atomic<bool> m_game_running{false};
    std::atomic<bool> m_recompiling{false};
    HANDLE m_worker_handle = nullptr;
    HANDLE m_frame_event = nullptr;
    u32 m_launch_entry = 0;
    LPTOP_LEVEL_EXCEPTION_FILTER m_prev_exception_filter = nullptr;

    std::thread m_recomp_thread;
    std::mutex m_recomp_mutex;
    std::string m_pending_recomp_module;
    bool m_recomp_completed_flag = false;

    std::mutex m_log_mutex;
    std::vector<std::string> m_pending_logs;

    u32 m_display_fps = 0;
    u32 m_last_present_us = 0;
    u64 m_fps_last_gs_frames = 0;
    u64 m_fps_window_start_ms = 0;
    u64 m_last_ui_refresh_ms = 0;
    std::atomic<u64> m_last_xinput_probe_ms{0};
};

} // namespace tf::ps2

#endif // _WIN32
