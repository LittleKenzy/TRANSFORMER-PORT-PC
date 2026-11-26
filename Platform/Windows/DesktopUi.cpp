#ifdef _WIN32

#include "DesktopUi.h"
#include "../../Builder/BuildPipeline/RecompPipeline.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <commdlg.h>
#include <d3d11.h>
#include <exception>
#include <filesystem>
#include <fstream>
#include <mmsystem.h>
#include <shlobj.h>
#include <sstream>

namespace tf::ps2 {

namespace fs = std::filesystem;

namespace {

constexpr int IDC_BTN_SELECT_ISO = 101;
constexpr int IDC_BTN_RECOMPILE  = 102;
constexpr int IDC_BTN_PLAY       = 103;
constexpr int IDC_BTN_STOP       = 104;
constexpr int IDC_BTN_SETTINGS   = 105;
constexpr int IDC_EDIT_LOG       = 106;

constexpr int IDC_SET_RES            = 201;
constexpr int IDC_SET_ASPECT         = 202;
constexpr int IDC_SET_VSYNC          = 203;
constexpr int IDC_SET_BILINEAR       = 204;
constexpr int IDC_SET_FULLSCREEN     = 205;
constexpr int IDC_SET_DEADZONE       = 206;
constexpr int IDC_SET_SENS           = 207;
constexpr int IDC_SET_INVX           = 208;
constexpr int IDC_SET_INVY           = 209;
constexpr int IDC_SET_VIBRATE        = 210;
constexpr int IDC_SET_AUDIO          = 211;
constexpr int IDC_SET_VOL            = 212;
constexpr int IDC_SET_MEMCARD_EDIT   = 213;
constexpr int IDC_SET_MEMCARD_BROWSE = 214;
constexpr int IDC_SET_SAVE           = 215;
constexpr int IDC_SET_CANCEL         = 216;

constexpr UINT_PTR kUiTimerId = 1;

constexpr COLORREF kColBg     = RGB(13, 17, 26);   // Deep Cybertronian Obsidian
constexpr COLORREF kColPanel  = RGB(22, 29, 43);   // Dark Metallic Slate
constexpr COLORREF kColCyan   = RGB(0, 212, 255);  // Energon Cyan
constexpr COLORREF kColGold   = RGB(245, 183, 0);  // Autobot Gold
constexpr COLORREF kColText   = RGB(230, 236, 245);
constexpr COLORREF kColMuted  = RGB(150, 165, 190);

constexpr f32 kDeadzones[4] = {0.10f, 0.15f, 0.20f, 0.25f};
constexpr f32 kSensitivities[4] = {0.75f, 1.00f, 1.25f, 1.50f};
constexpr u32 kVolumes[5] = {100u, 75u, 50u, 25u, 0u};

using EntryPointFn    = u32 (*)();
using FingerprintFn   = const char* (*)();
using RegisterFuncsFn = void (*)(void*);

// Binary-compatible XInput structures (loaded dynamically via LoadLibraryA so no extra link libs needed)
struct TfXInputGamepad {
    WORD  wButtons;
    BYTE  bLeftTrigger;
    BYTE  bRightTrigger;
    SHORT sThumbLX;
    SHORT sThumbLY;
    SHORT sThumbRX;
    SHORT sThumbRY;
};

struct TfXInputState {
    DWORD           dwPacketNumber;
    TfXInputGamepad Gamepad;
};

struct TfXInputVibration {
    WORD wLeftMotorSpeed;
    WORD wRightMotorSpeed;
};

using XInputGetStateFn = DWORD (WINAPI *)(DWORD dwUserIndex, TfXInputState* pState);
using XInputSetStateFn = DWORD (WINAPI *)(DWORD dwUserIndex, TfXInputVibration* pVibration);

constexpr WORD kXInputDpadUp        = 0x0001;
constexpr WORD kXInputDpadDown      = 0x0002;
constexpr WORD kXInputDpadLeft      = 0x0004;
constexpr WORD kXInputDpadRight     = 0x0008;
constexpr WORD kXInputStart         = 0x0010;
constexpr WORD kXInputBack          = 0x0020;
constexpr WORD kXInputLeftThumb     = 0x0040;
constexpr WORD kXInputRightThumb    = 0x0080;
constexpr WORD kXInputLeftShoulder  = 0x0100;
constexpr WORD kXInputRightShoulder = 0x0200;
constexpr WORD kXInputA             = 0x1000;
constexpr WORD kXInputB             = 0x2000;
constexpr WORD kXInputX             = 0x4000;
constexpr WORD kXInputY             = 0x8000;
constexpr BYTE kXInputTriggerThresh = 30;

using SHBrowseForFolderAFn   = PIDLIST_ABSOLUTE (WINAPI *)(LPBROWSEINFOA lpbi);
using SHGetPathFromIDListAFn = BOOL (WINAPI *)(PCIDLIST_ABSOLUTE pidl, LPSTR pszPath);
using CoTaskMemFreeFn        = void (WINAPI *)(LPVOID pv);

std::atomic<TransformersDesktopApp*> g_active_desktop_app{nullptr};
u64 g_last_present_ms = 0;

struct SrvReleaseGuard {
    ID3D11ShaderResourceView* ptr = nullptr;
    ~SrvReleaseGuard() {
        if (ptr) {
            ptr->Release();
        }
    }
};

std::string exe_directory() {
    char buf[MAX_PATH] = {};
    GetModuleFileNameA(nullptr, buf, MAX_PATH);
    return fs::path(buf).parent_path().string();
}

bool key_down(int vk) {
    return (GetAsyncKeyState(vk) & 0x8000) != 0;
}

// Compact 32-bit keyboard bitmask for instant WM_KEYDOWN/WM_KEYUP capture & sub-frame tap latching
constexpr u32 kKbUp       = 1u << 0;
constexpr u32 kKbDown     = 1u << 1;
constexpr u32 kKbLeft     = 1u << 2;
constexpr u32 kKbRight    = 1u << 3;
constexpr u32 kKbStart    = 1u << 4;
constexpr u32 kKbSelect   = 1u << 5;
constexpr u32 kKbCross    = 1u << 6;
constexpr u32 kKbCircle   = 1u << 7;
constexpr u32 kKbSquare   = 1u << 8;
constexpr u32 kKbTriangle = 1u << 9;
constexpr u32 kKbL1       = 1u << 10;
constexpr u32 kKbR1       = 1u << 11;
constexpr u32 kKbL2       = 1u << 12;
constexpr u32 kKbR2       = 1u << 13;
constexpr u32 kKbLsLeft   = 1u << 14;
constexpr u32 kKbLsRight  = 1u << 15;
constexpr u32 kKbLsUp     = 1u << 16;
constexpr u32 kKbLsDown   = 1u << 17;
constexpr u32 kKbRsLeft   = 1u << 18;
constexpr u32 kKbRsRight  = 1u << 19;
constexpr u32 kKbRsUp     = 1u << 20;
constexpr u32 kKbRsDown   = 1u << 21;

u32 vk_to_kb_bit(WPARAM vk) {
    switch (vk) {
        case VK_UP:      return kKbUp;
        case VK_DOWN:    return kKbDown;
        case VK_LEFT:    return kKbLeft;
        case VK_RIGHT:   return kKbRight;
        case VK_RETURN:  return kKbStart;
        case VK_BACK:
        case VK_RSHIFT:  return kKbSelect;
        case VK_SPACE:
        case 'Z':
        case 'N':        return kKbCross;
        case 'X':
        case 'M':        return kKbCircle;
        case 'C':
        case 'B':        return kKbSquare;
        case 'V':
        case 'F':        return kKbTriangle;
        case 'Q':        return kKbL1;
        case 'E':        return kKbR1;
        case '1':        return kKbL2;
        case '3':        return kKbR2;
        case 'A':        return kKbLsLeft;
        case 'D':        return kKbLsRight;
        case 'W':        return kKbLsUp;
        case 'S':        return kKbLsDown;
        case 'J':
        case VK_NUMPAD4: return kKbRsLeft;
        case 'L':
        case VK_NUMPAD6: return kKbRsRight;
        case 'I':
        case VK_NUMPAD8: return kKbRsUp;
        case 'K':
        case VK_NUMPAD5: return kKbRsDown;
        default:         return 0u;
    }
}

template <typename T, std::size_t N>
int nearest_index(const T (&arr)[N], T val) {
    int best = 0;
    double best_diff = std::fabs(static_cast<double>(arr[0]) - static_cast<double>(val));
    for (std::size_t i = 1; i < N; ++i) {
        double d = std::fabs(static_cast<double>(arr[i]) - static_cast<double>(val));
        if (d < best_diff) {
            best_diff = d;
            best = static_cast<int>(i);
        }
    }
    return best;
}

std::string trim_copy(const std::string& s) {
    std::size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return {};
    std::size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

} // namespace

TransformersDesktopApp::TransformersDesktopApp() {
    m_settings.load();
    if (m_settings.memory_card_dir.empty()) {
        m_settings.memory_card_dir = (fs::path(exe_directory()) / "MemoryCard0").string();
    }
    m_vibration_enabled.store(m_settings.input.vibration_enabled, std::memory_order_relaxed);
    m_frame_event = CreateEventA(nullptr, FALSE, FALSE, nullptr);
    init_xinput();
    g_active_desktop_app.store(this, std::memory_order_release);
    m_prev_exception_filter = SetUnhandledExceptionFilter(&TransformersDesktopApp::unhandled_exception_filter);
}

TransformersDesktopApp::~TransformersDesktopApp() {
    SetUnhandledExceptionFilter(m_prev_exception_filter);
    g_active_desktop_app.store(nullptr, std::memory_order_release);

    on_stop_game();
    if (m_recomp_thread.joinable()) {
        m_recomp_thread.join();
    }
    m_hw_gs.shutdown();
    m_hw_gs_active = false;
    m_presenter.shutdown();
    if (m_frame_event) {
        CloseHandle(m_frame_event);
        m_frame_event = nullptr;
    }
    m_runtime.function_table().clear();
    if (m_loaded_dll) {
        FreeLibrary(m_loaded_dll);
        m_loaded_dll = nullptr;
    }
    shutdown_xinput();
    if (m_brush_bg) DeleteObject(m_brush_bg);
    if (m_brush_panel) DeleteObject(m_brush_panel);
    if (m_brush_edit) DeleteObject(m_brush_edit);
    if (m_font_title) DeleteObject(m_font_title);
    if (m_font_sub) DeleteObject(m_font_sub);
    if (m_font_ui) DeleteObject(m_font_ui);
    if (m_font_mono) DeleteObject(m_font_mono);
}

LONG WINAPI TransformersDesktopApp::unhandled_exception_filter(EXCEPTION_POINTERS* ep) {
    TransformersDesktopApp* app = g_active_desktop_app.load(std::memory_order_acquire);
    const DWORD code = (ep && ep->ExceptionRecord) ? ep->ExceptionRecord->ExceptionCode : 0u;
    const void* addr = (ep && ep->ExceptionRecord) ? ep->ExceptionRecord->ExceptionAddress : nullptr;

    std::ostringstream oss;
    oss << "[CRASH] Unhandled Win32 exception code=0x" << std::hex << code
        << " addr=" << addr;
    if (addr) {
        HMODULE fault_mod = nullptr;
        if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               static_cast<LPCSTR>(addr), &fault_mod) && fault_mod) {
            char mod_path[MAX_PATH] = {};
            if (GetModuleFileNameA(fault_mod, mod_path, MAX_PATH) > 0) {
                const auto rva = reinterpret_cast<std::uintptr_t>(addr) - reinterpret_cast<std::uintptr_t>(fault_mod);
                oss << " mod=" << fs::path(mod_path).filename().string() << "+0x" << std::hex << rva;
            }
        }
    }
    oss << " tid=" << std::dec << GetCurrentThreadId();
    if (app) {
        oss << " guest_pc=" << hex32(app->m_runtime.cpu.pc)
            << " static_calls=" << std::dec << app->m_runtime.total_static_calls
            << " gs_frames=" << app->m_runtime.gs.frames_presented;
    }
    const std::string line = oss.str();
    OutputDebugStringA((line + "\n").c_str());

    // Persist crash diagnostic next to settings.ini for post-mortem analysis
    std::error_code ec;
    fs::path crash_path = fs::path(AppSettings::default_config_path()).parent_path() / "crash.log";
    fs::create_directories(crash_path.parent_path(), ec);
    std::ofstream ofs(crash_path, std::ios::app);
    if (ofs) {
        ofs << line << "\n";
    }
    if (app) {
        app->append_log(line);
    }
    return EXCEPTION_EXECUTE_HANDLER;
}

void TransformersDesktopApp::init_xinput() {
    const char* dlls[] = {"xinput1_4.dll", "xinput1_3.dll", "xinput9_1_0.dll"};
    for (const char* name : dlls) {
        m_xinput_dll = LoadLibraryA(name);
        if (m_xinput_dll) {
            m_xinput_get_state = reinterpret_cast<void*>(GetProcAddress(m_xinput_dll, "XInputGetState"));
            m_xinput_set_state = reinterpret_cast<void*>(GetProcAddress(m_xinput_dll, "XInputSetState"));
            if (m_xinput_get_state) {
                break;
            }
            FreeLibrary(m_xinput_dll);
            m_xinput_dll = nullptr;
            m_xinput_get_state = nullptr;
            m_xinput_set_state = nullptr;
        }
    }
}

void TransformersDesktopApp::shutdown_xinput() {
    set_xinput_vibration(0, 0);
    if (m_xinput_dll) {
        FreeLibrary(m_xinput_dll);
        m_xinput_dll = nullptr;
    }
    m_xinput_get_state = nullptr;
    m_xinput_set_state = nullptr;
    m_xinput_connected = false;
}

void TransformersDesktopApp::set_xinput_vibration(u16 left_motor, u16 right_motor) {
    if (!m_xinput_set_state) return;
    if (left_motor == m_last_vib_left && right_motor == m_last_vib_right) return;
    auto set_fn = reinterpret_cast<XInputSetStateFn>(m_xinput_set_state);
    TfXInputVibration vib{};
    vib.wLeftMotorSpeed = left_motor;
    vib.wRightMotorSpeed = right_motor;
    set_fn(0, &vib);
    m_last_vib_left = left_motor;
    m_last_vib_right = right_motor;
}

int TransformersDesktopApp::run(const std::string& initial_iso, const std::string& initial_module) {
    struct WinMmTimerGuard {
        WinMmTimerGuard() { timeBeginPeriod(1); }
        ~WinMmTimerGuard() { timeEndPeriod(1); }
    } timer_precision_guard;

    m_brush_bg    = CreateSolidBrush(kColBg);
    m_brush_panel = CreateSolidBrush(kColPanel);
    m_brush_edit  = CreateSolidBrush(RGB(10, 13, 20));

    m_font_title = CreateFontA(26, 0, 0, 0, FW_BOLD, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
                               OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, DEFAULT_PITCH, "Segoe UI");
    m_font_sub   = CreateFontA(15, 0, 0, 0, FW_SEMIBOLD, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
                               OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, DEFAULT_PITCH, "Segoe UI");
    m_font_ui    = CreateFontA(15, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
                               OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, DEFAULT_PITCH, "Segoe UI");
    m_font_mono  = CreateFontA(13, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
                               OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, FIXED_PITCH, "Consolas");

    HINSTANCE hinst = GetModuleHandleA(nullptr);

    WNDCLASSA wc{};
    wc.lpfnWndProc = &TransformersDesktopApp::main_wnd_proc;
    wc.hInstance = hinst;
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.hbrBackground = m_brush_bg;
    wc.lpszClassName = "TransformersDesktopMainWnd";
    RegisterClassA(&wc);

    // Custom viewport child window class: WM_ERASEBKGND returns 1, WM_PAINT blits latest framebuffer
    WNDCLASSA vp_wc{};
    vp_wc.style = CS_OWNDC;
    vp_wc.lpfnWndProc = &TransformersDesktopApp::viewport_wnd_proc;
    vp_wc.hInstance = hinst;
    vp_wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    vp_wc.hbrBackground = nullptr;
    vp_wc.lpszClassName = "TransformersViewportWnd";
    RegisterClassA(&vp_wc);

    WNDCLASSA swc{};
    swc.lpfnWndProc = &TransformersDesktopApp::settings_wnd_proc;
    swc.hInstance = hinst;
    swc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    swc.hbrBackground = m_brush_bg;
    swc.lpszClassName = "TransformersDesktopSettingsWnd";
    RegisterClassA(&swc);

    m_hwnd = CreateWindowExA(
        0, wc.lpszClassName,
        "TRANSFORMERS: THE GAME - Native PS2 Recompilation Port (Windows x64)",
        WS_OVERLAPPEDWINDOW | WS_VISIBLE | WS_CLIPCHILDREN,
        CW_USEDEFAULT, CW_USEDEFAULT, 1240, 780,
        nullptr, nullptr, hinst, this
    );

    if (!initial_module.empty()) {
        m_active_module_path = initial_module;
        refresh_active_module_ready();
    }

    std::string iso_to_load = !initial_iso.empty() ? initial_iso : m_settings.last_iso_path;
    std::error_code ec;
    if (!iso_to_load.empty() && fs::exists(iso_to_load, ec)) {
        load_iso_and_auto_detect_module(iso_to_load, !initial_module.empty());
    } else {
        append_log("Welcome to TRANSFORMERS: THE GAME (PS2 Native Static Recompilation Port).");
        append_log("Click '1. Select PS2 ISO...' to load your legally obtained PS2 ISO image.");
    }

    if (m_settings.start_fullscreen) {
        toggle_fullscreen(true);
    }

    refresh_active_module_ready();
    update_button_states();

    // Backup timer ensures live presentation & UI refresh continue even during Win32 window drag/resize modal loops
    SetTimer(m_hwnd, kUiTimerId, 33, nullptr);

    using clock = std::chrono::steady_clock;
    auto steady_ms = []() -> u64 {
        return static_cast<u64>(
            std::chrono::duration_cast<std::chrono::milliseconds>(clock::now().time_since_epoch()).count());
    };
    m_fps_window_start_ms = steady_ms();
    m_fps_last_gs_frames = 0;
    m_last_ui_refresh_ms = m_fps_window_start_ms;
    m_last_xinput_probe_ms = 0;

    MSG msg{};
    bool quit = false;
    while (!quit) {
        while (PeekMessageA(&msg, nullptr, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) {
                quit = true;
                break;
            }
            TranslateMessage(&msg);
            DispatchMessageA(&msg);
        }
        if (quit) break;

        const u64 now_ms = steady_ms();
        if (m_game_running.load(std::memory_order_relaxed)) {
            const bool minimized = (m_hwnd != nullptr && IsIconic(m_hwnd) != FALSE);
            const u64 cur_seq = m_runtime.gs.presented_seq();
            const bool has_new_frame = (cur_seq != m_last_presented_seq);
            const u64 since_present = (now_ms >= g_last_present_ms) ? (now_ms - g_last_present_ms) : 0u;

            if (!minimized) {
                if (m_force_redraw) {
                    bool ok = false;
                    if (has_new_frame) {
                        ok = present_current_frame();
                    }
                    if (!ok) {
                        ok = redraw_last_frame();
                    }
                    if (ok) {
                        m_force_redraw = false;
                    }
                } else if (has_new_frame && since_present >= 8u) {
                    present_current_frame();
                }
            }

            if (now_ms - m_last_ui_refresh_ms >= 250u) {
                update_true_fps(now_ms);
                refresh_status_ui();
                m_last_ui_refresh_ms = now_ms;
            }

            // Instant event-driven wakeup (<0.05ms) at 60 FPS gameplay, while coalescing
            // uncapped 550-975 FPS loading & minimized states so Win32 messages never starve!
            const u64 after_ms = steady_ms();
            const u64 post_since = (after_ms >= g_last_present_ms) ? (after_ms - g_last_present_ms) : 0u;
            if (minimized) {
                MsgWaitForMultipleObjects(0, nullptr, FALSE, 33, QS_ALLINPUT);
            } else if (m_runtime.gs.presented_seq() == m_last_presented_seq && !m_force_redraw) {
                if (m_frame_event) {
                    MsgWaitForMultipleObjects(1, &m_frame_event, FALSE, 16, QS_ALLINPUT);
                } else {
                    MsgWaitForMultipleObjects(0, nullptr, FALSE, 2, QS_ALLINPUT);
                }
            } else if (post_since < 8u) {
                const DWORD wait_ms = static_cast<DWORD>(std::max<u64>(1u, 8u - post_since));
                MsgWaitForMultipleObjects(0, nullptr, FALSE, wait_ms, QS_ALLINPUT);
            }
        } else {
            if (m_force_redraw && m_hwnd && !IsIconic(m_hwnd)) {
                m_force_redraw = false;
                if (m_presenter.is_hardware_accelerated()) {
                    m_presenter.clear_black();
                }
            }
            if (now_ms - m_last_ui_refresh_ms >= 100u) {
                refresh_status_ui();
                m_last_ui_refresh_ms = now_ms;
            }
            MsgWaitForMultipleObjects(0, nullptr, FALSE, 33, QS_ALLINPUT);
        }
    }

    KillTimer(m_hwnd, kUiTimerId);
    on_stop_game();
    return 0;
}

LRESULT CALLBACK TransformersDesktopApp::main_wnd_proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    TransformersDesktopApp* self = nullptr;
    if (msg == WM_NCCREATE) {
        auto* cs = reinterpret_cast<CREATESTRUCTA*>(lp);
        self = static_cast<TransformersDesktopApp*>(cs->lpCreateParams);
        SetWindowLongPtrA(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(self));
    } else {
        self = reinterpret_cast<TransformersDesktopApp*>(GetWindowLongPtrA(hwnd, GWLP_USERDATA));
    }
    return self ? self->handle_main_msg(hwnd, msg, wp, lp) : DefWindowProcA(hwnd, msg, wp, lp);
}

LRESULT CALLBACK TransformersDesktopApp::viewport_wnd_proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    TransformersDesktopApp* self = nullptr;
    if (msg == WM_NCCREATE) {
        auto* cs = reinterpret_cast<CREATESTRUCTA*>(lp);
        self = static_cast<TransformersDesktopApp*>(cs->lpCreateParams);
        SetWindowLongPtrA(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(self));
    } else {
        self = reinterpret_cast<TransformersDesktopApp*>(GetWindowLongPtrA(hwnd, GWLP_USERDATA));
    }
    return self ? self->handle_viewport_msg(hwnd, msg, wp, lp) : DefWindowProcA(hwnd, msg, wp, lp);
}

LRESULT TransformersDesktopApp::handle_viewport_msg(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
        case WM_ERASEBKGND:
            return 1;
        case WM_PAINT: {
            PAINTSTRUCT ps{};
            HDC hdc = BeginPaint(hwnd, &ps);
            if (hdc) {
                if (m_presenter.is_hardware_accelerated()) {
                    // NEVER call D3D11 Present() / ResizeBuffers() / mutex locks inside WM_PAINT!
                    // DWM retains the Flip-Model surface; defer fresh redraw to the main loop.
                    m_force_redraw = true;
                    if (m_frame_event) SetEvent(m_frame_event);
                } else if (m_has_frame && m_frame_w > 0 && m_frame_h > 0 &&
                           m_frame_copy.size() >= static_cast<std::size_t>(m_frame_w) * m_frame_h) {
                    m_presenter.present_rgba_to_dc(hdc, m_frame_copy.data(), m_frame_w, m_frame_h);
                } else {
                    RECT rc{};
                    if (GetClientRect(hwnd, &rc)) {
                        FillRect(hdc, &rc, static_cast<HBRUSH>(GetStockObject(BLACK_BRUSH)));
                    }
                }
                EndPaint(hwnd, &ps);
            }
            return 0;
        }
        case WM_PRINTCLIENT: {
            HDC hdc = reinterpret_cast<HDC>(wp);
            if (hdc) {
                if (m_has_frame && m_frame_w > 0 && m_frame_h > 0 &&
                    m_frame_copy.size() >= static_cast<std::size_t>(m_frame_w) * m_frame_h) {
                    m_presenter.present_rgba_to_dc(hdc, m_frame_copy.data(), m_frame_w, m_frame_h);
                } else {
                    RECT rc{};
                    if (GetClientRect(hwnd, &rc)) {
                        FillRect(hdc, &rc, static_cast<HBRUSH>(GetStockObject(BLACK_BRUSH)));
                    }
                }
            }
            return 0;
        }
        case WM_LBUTTONDOWN:
            if (m_hwnd) {
                SetFocus(m_hwnd);
            }
            return 0;
        default:
            return DefWindowProcA(hwnd, msg, wp, lp);
    }
}

void TransformersDesktopApp::create_main_controls(HWND hwnd) {
    HINSTANCE hinst = GetModuleHandleA(nullptr);

    m_lbl_title = CreateWindowA("STATIC", "TRANSFORMERS: THE GAME", WS_CHILD | WS_VISIBLE,
                                20, 14, 520, 30, hwnd, nullptr, hinst, nullptr);
    SendMessageA(m_lbl_title, WM_SETFONT, reinterpret_cast<WPARAM>(m_font_title), TRUE);

    m_lbl_subtitle = CreateWindowA("STATIC", "PS2 ISO Static Recompilation Native Runtime  |  Windows x64  |  F11 Fullscreen",
                                   WS_CHILD | WS_VISIBLE, 20, 44, 680, 20, hwnd, nullptr, hinst, nullptr);
    SendMessageA(m_lbl_subtitle, WM_SETFONT, reinterpret_cast<WPARAM>(m_font_sub), TRUE);

    m_lbl_iso_info = CreateWindowA("STATIC", "No PS2 ISO loaded yet.", WS_CHILD | WS_VISIBLE | SS_LEFT,
                                   20, 76, 290, 96, hwnd, nullptr, hinst, nullptr);
    SendMessageA(m_lbl_iso_info, WM_SETFONT, reinterpret_cast<WPARAM>(m_font_ui), TRUE);

    m_btn_iso = CreateWindowA("BUTTON", "1. Select PS2 ISO...", WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
                              20, 180, 280, 36, hwnd, reinterpret_cast<HMENU>(IDC_BTN_SELECT_ISO), hinst, nullptr);
    m_btn_recomp = CreateWindowA("BUTTON", "2. Recompile ISO (Static AOT)", WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
                                 20, 224, 280, 36, hwnd, reinterpret_cast<HMENU>(IDC_BTN_RECOMPILE), hinst, nullptr);
    m_btn_play = CreateWindowA("BUTTON", "3. PLAY / LAUNCH NATIVE", WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
                               20, 268, 280, 42, hwnd, reinterpret_cast<HMENU>(IDC_BTN_PLAY), hinst, nullptr);
    m_btn_stop = CreateWindowA("BUTTON", "Stop Execution", WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
                               20, 318, 134, 34, hwnd, reinterpret_cast<HMENU>(IDC_BTN_STOP), hinst, nullptr);
    m_btn_settings = CreateWindowA("BUTTON", "Settings...", WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
                                   166, 318, 134, 34, hwnd, reinterpret_cast<HMENU>(IDC_BTN_SETTINGS), hinst, nullptr);

    for (HWND b : {m_btn_iso, m_btn_recomp, m_btn_play, m_btn_stop, m_btn_settings}) {
        SendMessageA(b, WM_SETFONT, reinterpret_cast<WPARAM>(m_font_sub), TRUE);
    }

    m_lbl_telemetry = CreateWindowA("STATIC", "Telemetry: Idle", WS_CHILD | WS_VISIBLE | SS_LEFT,
                                    20, 366, 285, 166, hwnd, nullptr, hinst, nullptr);
    SendMessageA(m_lbl_telemetry, WM_SETFONT, reinterpret_cast<WPARAM>(m_font_mono), TRUE);

    const char* controls_txt =
        "CONTROLS (Keyboard + XInput Pad 0)\r\n"
        "  WASD        : Left Analog Stick\r\n"
        "  Arrows      : D-Pad\r\n"
        "  IJKL/Num8456: Right Analog Stick\r\n"
        "  Enter       : Start | Bksp/RSh: Sel\r\n"
        "  Space/Z/N   : Cross | X/M : Circle\r\n"
        "  C/B         : Square| V/F : Triangle\r\n"
        "  Q/E         : L1/R1 | 1/3 : L2/R2";
    m_lbl_controls = CreateWindowA("STATIC", controls_txt, WS_CHILD | WS_VISIBLE | SS_LEFT,
                                   20, 540, 285, 175, hwnd, nullptr, hinst, nullptr);
    SendMessageA(m_lbl_controls, WM_SETFONT, reinterpret_cast<WPARAM>(m_font_mono), TRUE);

    m_viewport = CreateWindowExA(WS_EX_CLIENTEDGE, "TransformersViewportWnd", "", WS_CHILD | WS_VISIBLE,
                                 320, 76, 880, 480, hwnd, nullptr, hinst, this);
    m_presenter.initialize(m_viewport, 880, 480, m_settings.graphics);
    if (m_presenter.d3d11_device() && m_presenter.d3d11_context()) {
        m_hw_gs_active = m_hw_gs.init(
            m_presenter.d3d11_device(),
            m_presenter.d3d11_context(),
            m_presenter.d3d11_context_mutex(),
            m_presenter.d3d11_ui_waiters());
        if (m_hw_gs_active) {
            m_hw_gs.set_resolution_scale(m_settings.graphics.internal_resolution_scale);
        }
    }

    m_edit_log = CreateWindowExA(WS_EX_CLIENTEDGE, "EDIT", "",
                                 WS_CHILD | WS_VISIBLE | ES_MULTILINE | ES_AUTOVSCROLL | ES_READONLY | WS_VSCROLL,
                                 320, 566, 880, 155, hwnd, reinterpret_cast<HMENU>(IDC_EDIT_LOG), hinst, nullptr);
    SendMessageA(m_edit_log, WM_SETFONT, reinterpret_cast<WPARAM>(m_font_mono), TRUE);
}

void TransformersDesktopApp::layout_controls(int w, int h) {
    if (!m_viewport || !m_edit_log) return;

    if (m_is_fullscreen) {
        const int show = SW_HIDE;
        for (HWND c : {m_lbl_title, m_lbl_subtitle, m_lbl_iso_info, m_btn_iso, m_btn_recomp,
                       m_btn_play, m_btn_stop, m_btn_settings, m_lbl_telemetry, m_lbl_controls, m_edit_log}) {
            if (c) ShowWindow(c, show);
        }
        MoveWindow(m_viewport, 0, 0, std::max(1, w), std::max(1, h), TRUE);
        m_presenter.resize(static_cast<u32>(std::max(1, w)), static_cast<u32>(std::max(1, h)));
        InvalidateRect(m_viewport, nullptr, FALSE);
        return;
    }

    for (HWND c : {m_lbl_title, m_lbl_subtitle, m_lbl_iso_info, m_btn_iso, m_btn_recomp,
                   m_btn_play, m_btn_stop, m_btn_settings, m_lbl_telemetry, m_lbl_controls, m_edit_log}) {
        if (c) ShowWindow(c, SW_SHOW);
    }

    int left_w = 310;
    int right_x = left_w + 14;
    int right_w = std::max(320, w - right_x - 16);
    int log_h = 155;
    int vp_h = std::max(240, h - 76 - log_h - 24);
    MoveWindow(m_viewport, right_x, 76, right_w, vp_h, TRUE);
    MoveWindow(m_edit_log, right_x, 76 + vp_h + 10, right_w, log_h, TRUE);
    m_presenter.resize(static_cast<u32>(right_w), static_cast<u32>(vp_h));
    InvalidateRect(m_viewport, nullptr, FALSE);
}

void TransformersDesktopApp::toggle_fullscreen(bool enable) {
    if (!m_hwnd || m_is_fullscreen == enable) return;
    if (enable) {
        m_saved_style = static_cast<DWORD>(GetWindowLongA(m_hwnd, GWL_STYLE));
        m_saved_ex_style = static_cast<DWORD>(GetWindowLongA(m_hwnd, GWL_EXSTYLE));
        GetWindowRect(m_hwnd, &m_saved_rect);

        MONITORINFO mi{};
        mi.cbSize = sizeof(mi);
        if (GetMonitorInfoA(MonitorFromWindow(m_hwnd, MONITOR_DEFAULTTONEAREST), &mi)) {
            m_is_fullscreen = true;
            SetWindowLongA(m_hwnd, GWL_STYLE, static_cast<LONG>(m_saved_style & ~(WS_CAPTION | WS_THICKFRAME)));
            SetWindowLongA(m_hwnd, GWL_EXSTYLE, static_cast<LONG>(m_saved_ex_style & ~(WS_EX_DLGMODALFRAME | WS_EX_WINDOWEDGE | WS_EX_CLIENTEDGE | WS_EX_STATICEDGE)));
            SetWindowPos(
                m_hwnd, HWND_TOP,
                mi.rcMonitor.left, mi.rcMonitor.top,
                mi.rcMonitor.right - mi.rcMonitor.left,
                mi.rcMonitor.bottom - mi.rcMonitor.top,
                SWP_NOOWNERZORDER | SWP_FRAMECHANGED
            );
        }
    } else {
        m_is_fullscreen = false;
        SetWindowLongA(m_hwnd, GWL_STYLE, static_cast<LONG>(m_saved_style));
        SetWindowLongA(m_hwnd, GWL_EXSTYLE, static_cast<LONG>(m_saved_ex_style));
        SetWindowPos(
            m_hwnd, nullptr,
            m_saved_rect.left, m_saved_rect.top,
            m_saved_rect.right - m_saved_rect.left,
            m_saved_rect.bottom - m_saved_rect.top,
            SWP_NOZORDER | SWP_NOOWNERZORDER | SWP_FRAMECHANGED
        );
    }
    RECT rc{};
    if (GetClientRect(m_hwnd, &rc)) {
        layout_controls(rc.right - rc.left, rc.bottom - rc.top);
    }
    if (m_game_running.load(std::memory_order_relaxed)) {
        m_force_redraw = true;
        if (m_frame_event) SetEvent(m_frame_event);
    }
}

LRESULT TransformersDesktopApp::handle_main_msg(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
        case WM_CREATE:
            create_main_controls(hwnd);
            return 0;
        case WM_SIZE:
            if (wp == SIZE_MINIMIZED || LOWORD(lp) == 0 || HIWORD(lp) == 0) {
                return 0;
            }
            layout_controls(LOWORD(lp), HIWORD(lp));
            m_force_redraw = true;
            if (m_frame_event) SetEvent(m_frame_event);
            return 0;
        case WM_ACTIVATEAPP: {
            const bool active = (wp != FALSE);
            m_app_active = active;
            if (!active) {
                m_kb_pressed_mask.store(0, std::memory_order_relaxed);
                m_kb_edge_latch.store(0, std::memory_order_relaxed);
            } else {
                m_presenter.on_window_activated();
                m_force_redraw = true;
                if (m_frame_event) SetEvent(m_frame_event);
            }
            return DefWindowProcA(hwnd, msg, wp, lp);
        }
        case WM_SETFOCUS:
            m_presenter.on_window_activated();
            m_force_redraw = true;
            if (m_frame_event) SetEvent(m_frame_event);
            return DefWindowProcA(hwnd, msg, wp, lp);
        case WM_KEYDOWN:
        case WM_SYSKEYDOWN:
            if (wp == VK_F11) {
                toggle_fullscreen(!m_is_fullscreen);
                return 0;
            }
            record_kb_event(wp, true);
            return DefWindowProcA(hwnd, msg, wp, lp);
        case WM_KEYUP:
        case WM_SYSKEYUP:
            record_kb_event(wp, false);
            return DefWindowProcA(hwnd, msg, wp, lp);
        case WM_KILLFOCUS:
            m_kb_pressed_mask.store(0, std::memory_order_relaxed);
            m_kb_edge_latch.store(0, std::memory_order_relaxed);
            return DefWindowProcA(hwnd, msg, wp, lp);
        case WM_LBUTTONDOWN:
            if (m_game_running.load(std::memory_order_relaxed)) {
                SetFocus(hwnd);
            }
            return DefWindowProcA(hwnd, msg, wp, lp);
        case WM_CTLCOLORSTATIC: {
            HDC hdc = reinterpret_cast<HDC>(wp);
            HWND ctrl = reinterpret_cast<HWND>(lp);
            SetBkColor(hdc, kColBg);
            if (ctrl == m_lbl_title) SetTextColor(hdc, kColGold);
            else if (ctrl == m_lbl_subtitle) SetTextColor(hdc, kColCyan);
            else if (ctrl == m_lbl_telemetry || ctrl == m_lbl_controls) SetTextColor(hdc, kColMuted);
            else SetTextColor(hdc, kColText);
            return reinterpret_cast<LRESULT>(m_brush_bg);
        }
        case WM_CTLCOLOREDIT: {
            HDC hdc = reinterpret_cast<HDC>(wp);
            SetBkColor(hdc, RGB(10, 13, 20));
            SetTextColor(hdc, RGB(150, 235, 255));
            return reinterpret_cast<LRESULT>(m_brush_edit);
        }
        case WM_COMMAND: {
            int id = LOWORD(wp);
            if (id == IDC_BTN_SELECT_ISO) on_select_iso();
            else if (id == IDC_BTN_RECOMPILE) on_recompile_iso();
            else if (id == IDC_BTN_PLAY) on_launch_game();
            else if (id == IDC_BTN_STOP) on_stop_game();
            else if (id == IDC_BTN_SETTINGS) open_settings_dialog();
            return 0;
        }
        case WM_TIMER:
            if (wp == kUiTimerId) {
                const u64 now_ms = static_cast<u64>(
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now().time_since_epoch()).count());
                if (m_game_running.load(std::memory_order_relaxed)) {
                    // Keep WM_TIMER 100% non-blocking (never call D3D11 Present or cross-thread poll_input in WndProc)
                    if (now_ms - m_last_ui_refresh_ms >= 250u) {
                        update_true_fps(now_ms);
                        refresh_status_ui();
                        m_last_ui_refresh_ms = now_ms;
                    }
                } else if (now_ms - m_last_ui_refresh_ms >= 100u) {
                    refresh_status_ui();
                    m_last_ui_refresh_ms = now_ms;
                }
            }
            return 0;
        case WM_CLOSE:
            if (m_dlg_settings && IsWindow(m_dlg_settings)) {
                DestroyWindow(m_dlg_settings);
                m_dlg_settings = nullptr;
                m_dlg_settings_atomic.store(nullptr, std::memory_order_release);
            }
            on_stop_game();
            DestroyWindow(hwnd);
            return 0;
        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;
        default:
            return DefWindowProcA(hwnd, msg, wp, lp);
    }
}

void TransformersDesktopApp::open_settings_dialog() {
    if (m_dlg_settings && IsWindow(m_dlg_settings)) {
        SetForegroundWindow(m_dlg_settings);
        return;
    }
    HINSTANCE hinst = GetModuleHandleA(nullptr);
    m_dlg_settings = CreateWindowExA(
        WS_EX_DLGMODALFRAME,
        "TransformersDesktopSettingsWnd",
        "TRANSFORMERS - Settings (Graphics, Controls, Audio, Memory Card)",
        WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_VISIBLE,
        CW_USEDEFAULT, CW_USEDEFAULT, 540, 600,
        m_hwnd, nullptr, hinst, this
    );
    m_dlg_settings_atomic.store(m_dlg_settings, std::memory_order_release);
}

LRESULT CALLBACK TransformersDesktopApp::settings_wnd_proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    TransformersDesktopApp* self = nullptr;
    if (msg == WM_NCCREATE) {
        auto* cs = reinterpret_cast<CREATESTRUCTA*>(lp);
        self = static_cast<TransformersDesktopApp*>(cs->lpCreateParams);
        SetWindowLongPtrA(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(self));
    } else {
        self = reinterpret_cast<TransformersDesktopApp*>(GetWindowLongPtrA(hwnd, GWLP_USERDATA));
    }
    return self ? self->handle_settings_msg(hwnd, msg, wp, lp) : DefWindowProcA(hwnd, msg, wp, lp);
}

LRESULT TransformersDesktopApp::handle_settings_msg(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
        case WM_CREATE: {
            HINSTANCE hinst = GetModuleHandleA(nullptr);
            auto add_lbl = [&](const char* txt, int x, int y, int w, int h, HFONT f) {
                HWND s = CreateWindowA("STATIC", txt, WS_CHILD | WS_VISIBLE, x, y, w, h, hwnd, nullptr, hinst, nullptr);
                SendMessageA(s, WM_SETFONT, reinterpret_cast<WPARAM>(f), TRUE);
                return s;
            };

            m_lbl_sec_gfx = add_lbl("GRAPHICS SETTINGS", 20, 14, 340, 22, m_font_sub);
            add_lbl("Internal Resolution:", 20, 44, 170, 22, m_font_ui);
            m_cb_res = CreateWindowA("COMBOBOX", "", WS_CHILD | WS_VISIBLE | CBS_DROPDOWNLIST,
                                     200, 40, 295, 140, hwnd, reinterpret_cast<HMENU>(IDC_SET_RES), hinst, nullptr);
            SendMessageA(m_cb_res, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>("1x Native PS2 (640x448)"));
            SendMessageA(m_cb_res, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>("2x HD (1280x896)"));
            SendMessageA(m_cb_res, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>("3x Full HD (1920x1344)"));
            u32 res_scale = std::clamp<u32>(m_settings.graphics.internal_resolution_scale, 1u, 3u);
            SendMessageA(m_cb_res, CB_SETCURSEL, static_cast<WPARAM>(res_scale - 1u), 0);

            add_lbl("Aspect Ratio:", 20, 78, 170, 22, m_font_ui);
            m_cb_aspect = CreateWindowA("COMBOBOX", "", WS_CHILD | WS_VISIBLE | CBS_DROPDOWNLIST,
                                        200, 74, 295, 140, hwnd, reinterpret_cast<HMENU>(IDC_SET_ASPECT), hinst, nullptr);
            SendMessageA(m_cb_aspect, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>("4:3 Native PS2 (Pillarbox)"));
            SendMessageA(m_cb_aspect, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>("16:9 Widescreen (Letterbox)"));
            SendMessageA(m_cb_aspect, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>("Stretch to Window"));
            int aspect_idx = std::clamp<int>(static_cast<int>(m_settings.graphics.aspect_ratio), 0, 2);
            SendMessageA(m_cb_aspect, CB_SETCURSEL, static_cast<WPARAM>(aspect_idx), 0);

            m_chk_vsync = CreateWindowA("BUTTON", "Enable VSync (60 Hz Pacing)", WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX,
                                        20, 112, 235, 24, hwnd, reinterpret_cast<HMENU>(IDC_SET_VSYNC), hinst, nullptr);
            SendMessageA(m_chk_vsync, BM_SETCHECK, m_settings.graphics.vsync ? BST_CHECKED : BST_UNCHECKED, 0);

            m_chk_bilinear = CreateWindowA("BUTTON", "Bilinear Texture Filtering", WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX,
                                           270, 112, 225, 24, hwnd, reinterpret_cast<HMENU>(IDC_SET_BILINEAR), hinst, nullptr);
            SendMessageA(m_chk_bilinear, BM_SETCHECK, m_settings.graphics.bilinear_filtering ? BST_CHECKED : BST_UNCHECKED, 0);

            m_chk_fullscreen = CreateWindowA("BUTTON", "Start in Fullscreen Mode (F11 to toggle)", WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX,
                                             20, 142, 320, 24, hwnd, reinterpret_cast<HMENU>(IDC_SET_FULLSCREEN), hinst, nullptr);
            SendMessageA(m_chk_fullscreen, BM_SETCHECK, m_settings.start_fullscreen ? BST_CHECKED : BST_UNCHECKED, 0);

            m_lbl_sec_ctrl = add_lbl("CONTROLS (DUALSHOCK 2 / XINPUT / KEYBOARD)", 20, 184, 460, 22, m_font_sub);
            add_lbl("Analog Deadzone:", 20, 214, 170, 22, m_font_ui);
            m_cb_deadzone = CreateWindowA("COMBOBOX", "", WS_CHILD | WS_VISIBLE | CBS_DROPDOWNLIST,
                                          200, 210, 295, 140, hwnd, reinterpret_cast<HMENU>(IDC_SET_DEADZONE), hinst, nullptr);
            SendMessageA(m_cb_deadzone, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>("10% (0.10)"));
            SendMessageA(m_cb_deadzone, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>("15% (0.15 Default)"));
            SendMessageA(m_cb_deadzone, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>("20% (0.20)"));
            SendMessageA(m_cb_deadzone, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>("25% (0.25)"));
            SendMessageA(m_cb_deadzone, CB_SETCURSEL, static_cast<WPARAM>(nearest_index(kDeadzones, m_settings.input.deadzone)), 0);

            add_lbl("Camera Sensitivity:", 20, 248, 170, 22, m_font_ui);
            m_cb_sens = CreateWindowA("COMBOBOX", "", WS_CHILD | WS_VISIBLE | CBS_DROPDOWNLIST,
                                      200, 244, 295, 140, hwnd, reinterpret_cast<HMENU>(IDC_SET_SENS), hinst, nullptr);
            SendMessageA(m_cb_sens, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>("0.75x"));
            SendMessageA(m_cb_sens, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>("1.00x (Default)"));
            SendMessageA(m_cb_sens, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>("1.25x"));
            SendMessageA(m_cb_sens, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>("1.50x"));
            SendMessageA(m_cb_sens, CB_SETCURSEL, static_cast<WPARAM>(nearest_index(kSensitivities, m_settings.input.sensitivity)), 0);

            m_chk_inv_x = CreateWindowA("BUTTON", "Invert Camera X", WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX,
                                        20, 282, 160, 24, hwnd, reinterpret_cast<HMENU>(IDC_SET_INVX), hinst, nullptr);
            SendMessageA(m_chk_inv_x, BM_SETCHECK, m_settings.input.invert_camera_x ? BST_CHECKED : BST_UNCHECKED, 0);

            m_chk_inv_y = CreateWindowA("BUTTON", "Invert Camera Y", WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX,
                                        195, 282, 160, 24, hwnd, reinterpret_cast<HMENU>(IDC_SET_INVY), hinst, nullptr);
            SendMessageA(m_chk_inv_y, BM_SETCHECK, m_settings.input.invert_camera_y ? BST_CHECKED : BST_UNCHECKED, 0);

            m_chk_vibrate = CreateWindowA("BUTTON", "Vibration", WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX,
                                          370, 282, 125, 24, hwnd, reinterpret_cast<HMENU>(IDC_SET_VIBRATE), hinst, nullptr);
            SendMessageA(m_chk_vibrate, BM_SETCHECK, m_settings.input.vibration_enabled ? BST_CHECKED : BST_UNCHECKED, 0);

            m_lbl_sec_audio = add_lbl("AUDIO & VIRTUAL MEMORY CARD", 20, 326, 400, 22, m_font_sub);
            m_chk_audio = CreateWindowA("BUTTON", "Enable SPU2 Native Audio (Unmuted)", WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX,
                                        20, 356, 300, 24, hwnd, reinterpret_cast<HMENU>(IDC_SET_AUDIO), hinst, nullptr);
            SendMessageA(m_chk_audio, BM_SETCHECK, m_settings.audio_enabled ? BST_CHECKED : BST_UNCHECKED, 0);

            add_lbl("Master Volume:", 20, 392, 170, 22, m_font_ui);
            m_cb_vol = CreateWindowA("COMBOBOX", "", WS_CHILD | WS_VISIBLE | CBS_DROPDOWNLIST,
                                     200, 388, 295, 160, hwnd, reinterpret_cast<HMENU>(IDC_SET_VOL), hinst, nullptr);
            SendMessageA(m_cb_vol, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>("100%"));
            SendMessageA(m_cb_vol, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>("75%"));
            SendMessageA(m_cb_vol, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>("50%"));
            SendMessageA(m_cb_vol, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>("25%"));
            SendMessageA(m_cb_vol, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>("0% (Muted)"));
            SendMessageA(m_cb_vol, CB_SETCURSEL, static_cast<WPARAM>(nearest_index(kVolumes, m_settings.audio_volume_percent)), 0);

            add_lbl("Memory Card Folder:", 20, 428, 170, 22, m_font_ui);
            m_edit_memcard = CreateWindowExA(WS_EX_CLIENTEDGE, "EDIT", m_settings.memory_card_dir.c_str(),
                                             WS_CHILD | WS_VISIBLE | ES_AUTOHSCROLL,
                                             20, 452, 380, 26, hwnd, reinterpret_cast<HMENU>(IDC_SET_MEMCARD_EDIT), hinst, nullptr);
            m_btn_memcard_browse = CreateWindowA("BUTTON", "Browse...", WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
                                                 408, 452, 87, 26, hwnd, reinterpret_cast<HMENU>(IDC_SET_MEMCARD_BROWSE), hinst, nullptr);

            HWND btn_save = CreateWindowA("BUTTON", "Save Settings", WS_CHILD | WS_VISIBLE | BS_DEFPUSHBUTTON,
                                          265, 504, 115, 34, hwnd, reinterpret_cast<HMENU>(IDC_SET_SAVE), hinst, nullptr);
            HWND btn_cancel = CreateWindowA("BUTTON", "Cancel", WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
                                            390, 504, 105, 34, hwnd, reinterpret_cast<HMENU>(IDC_SET_CANCEL), hinst, nullptr);

            for (HWND c : {m_cb_res, m_cb_aspect, m_chk_vsync, m_chk_bilinear, m_chk_fullscreen,
                           m_cb_deadzone, m_cb_sens, m_chk_inv_x, m_chk_inv_y, m_chk_vibrate,
                           m_chk_audio, m_cb_vol, m_btn_memcard_browse, btn_save, btn_cancel}) {
                SendMessageA(c, WM_SETFONT, reinterpret_cast<WPARAM>(m_font_ui), TRUE);
            }
            SendMessageA(m_edit_memcard, WM_SETFONT, reinterpret_cast<WPARAM>(m_font_mono), TRUE);
            return 0;
        }
        case WM_CTLCOLORSTATIC: {
            HDC hdc = reinterpret_cast<HDC>(wp);
            HWND ctrl = reinterpret_cast<HWND>(lp);
            SetBkColor(hdc, kColBg);
            if (ctrl == m_lbl_sec_gfx || ctrl == m_lbl_sec_ctrl || ctrl == m_lbl_sec_audio) {
                SetTextColor(hdc, kColGold);
            } else {
                SetTextColor(hdc, kColText);
            }
            return reinterpret_cast<LRESULT>(m_brush_bg);
        }
        case WM_CTLCOLOREDIT: {
            HDC hdc = reinterpret_cast<HDC>(wp);
            SetBkColor(hdc, RGB(10, 13, 20));
            SetTextColor(hdc, kColCyan);
            return reinterpret_cast<LRESULT>(m_brush_edit);
        }
        case WM_COMMAND: {
            int id = LOWORD(wp);
            if (id == IDC_SET_MEMCARD_BROWSE) {
                on_browse_memory_card(hwnd);
            } else if (id == IDC_SET_SAVE) {
                apply_settings_from_dialog(hwnd);
                DestroyWindow(hwnd);
                m_dlg_settings = nullptr;
                m_dlg_settings_atomic.store(nullptr, std::memory_order_release);
            } else if (id == IDC_SET_CANCEL) {
                DestroyWindow(hwnd);
                m_dlg_settings = nullptr;
                m_dlg_settings_atomic.store(nullptr, std::memory_order_release);
            }
            return 0;
        }
        case WM_CLOSE:
            DestroyWindow(hwnd);
            m_dlg_settings = nullptr;
            m_dlg_settings_atomic.store(nullptr, std::memory_order_release);
            return 0;
        default:
            return DefWindowProcA(hwnd, msg, wp, lp);
    }
}

void TransformersDesktopApp::on_browse_memory_card(HWND dlg) {
    HMODULE shell32 = LoadLibraryA("shell32.dll");
    HMODULE ole32 = LoadLibraryA("ole32.dll");
    if (shell32) {
        auto browse_fn = reinterpret_cast<SHBrowseForFolderAFn>(
            reinterpret_cast<void*>(GetProcAddress(shell32, "SHBrowseForFolderA")));
        auto get_path_fn = reinterpret_cast<SHGetPathFromIDListAFn>(
            reinterpret_cast<void*>(GetProcAddress(shell32, "SHGetPathFromIDListA")));
        auto free_fn = ole32 ? reinterpret_cast<CoTaskMemFreeFn>(
            reinterpret_cast<void*>(GetProcAddress(ole32, "CoTaskMemFree"))) : nullptr;

        if (browse_fn && get_path_fn) {
            BROWSEINFOA bi{};
            bi.hwndOwner = dlg;
            bi.lpszTitle = "Select Virtual PS2 Memory Card Folder";
            bi.ulFlags = BIF_RETURNONLYFSDIRS | BIF_NEWDIALOGSTYLE;
            PIDLIST_ABSOLUTE pidl = browse_fn(&bi);
            if (pidl) {
                char folder[MAX_PATH] = {};
                if (get_path_fn(pidl, folder) && folder[0] != '\0') {
                    SetWindowTextA(m_edit_memcard, folder);
                }
                if (free_fn) {
                    free_fn(pidl);
                }
            }
        }
        FreeLibrary(shell32);
    }
    if (ole32) {
        FreeLibrary(ole32);
    }
}

void TransformersDesktopApp::apply_settings_from_dialog(HWND /*dlg*/) {
    int res_idx = static_cast<int>(SendMessageA(m_cb_res, CB_GETCURSEL, 0, 0));
    m_settings.graphics.internal_resolution_scale = static_cast<u32>(std::clamp(res_idx, 0, 2) + 1);
    int aspect_idx = std::clamp(static_cast<int>(SendMessageA(m_cb_aspect, CB_GETCURSEL, 0, 0)), 0, 2);
    m_settings.graphics.aspect_ratio = static_cast<AspectRatioMode>(aspect_idx);
    m_settings.graphics.vsync = (SendMessageA(m_chk_vsync, BM_GETCHECK, 0, 0) == BST_CHECKED);
    m_settings.graphics.bilinear_filtering = (SendMessageA(m_chk_bilinear, BM_GETCHECK, 0, 0) == BST_CHECKED);
    m_settings.start_fullscreen = (SendMessageA(m_chk_fullscreen, BM_GETCHECK, 0, 0) == BST_CHECKED);

    int dz_idx = std::clamp(static_cast<int>(SendMessageA(m_cb_deadzone, CB_GETCURSEL, 0, 0)), 0, 3);
    int sens_idx = std::clamp(static_cast<int>(SendMessageA(m_cb_sens, CB_GETCURSEL, 0, 0)), 0, 3);
    int vol_idx = std::clamp(static_cast<int>(SendMessageA(m_cb_vol, CB_GETCURSEL, 0, 0)), 0, 4);

    m_settings.input.deadzone = kDeadzones[dz_idx];
    m_settings.input.sensitivity = kSensitivities[sens_idx];
    m_settings.input.invert_camera_x = (SendMessageA(m_chk_inv_x, BM_GETCHECK, 0, 0) == BST_CHECKED);
    m_settings.input.invert_camera_y = (SendMessageA(m_chk_inv_y, BM_GETCHECK, 0, 0) == BST_CHECKED);
    m_settings.input.vibration_enabled = (SendMessageA(m_chk_vibrate, BM_GETCHECK, 0, 0) == BST_CHECKED);
    m_settings.audio_enabled = (SendMessageA(m_chk_audio, BM_GETCHECK, 0, 0) == BST_CHECKED);
    m_settings.audio_volume_percent = kVolumes[vol_idx];

    char mc_buf[MAX_PATH] = {};
    GetWindowTextA(m_edit_memcard, mc_buf, MAX_PATH);
    std::string mc_dir = trim_copy(mc_buf);
    if (!mc_dir.empty()) {
        m_settings.memory_card_dir = mc_dir;
    }

    {
        std::lock_guard<std::mutex> pad_lk(m_runtime.pad_mutex());
        m_runtime.input_config = m_settings.input;
    }
    m_vibration_enabled.store(m_settings.input.vibration_enabled, std::memory_order_relaxed);
    m_presenter.update_settings(m_settings.graphics);
    if (m_hw_gs_active) {
        m_hw_gs.set_resolution_scale(m_settings.graphics.internal_resolution_scale);
    }
    if (!m_settings.input.vibration_enabled) {
        set_xinput_vibration(0, 0);
    }
    if (m_game_running.load(std::memory_order_relaxed)) {
        m_force_redraw = true;
        if (m_frame_event) SetEvent(m_frame_event);
    } else if (m_viewport) {
        InvalidateRect(m_viewport, nullptr, FALSE);
    }

    m_settings.save();
    append_log("Settings saved to " + AppSettings::default_config_path());
}

void TransformersDesktopApp::on_select_iso() {
    char filename[MAX_PATH] = {};
    OPENFILENAMEA ofn{};
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = m_hwnd;
    ofn.lpstrFilter = "PlayStation 2 ISO Images (*.iso)\0*.iso\0All Files (*.*)\0*.*\0";
    ofn.lpstrFile = filename;
    ofn.nMaxFile = MAX_PATH;
    ofn.lpstrTitle = "Select Legally Obtained Transformers PS2 ISO";
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST;
    if (GetOpenFileNameA(&ofn)) {
        load_iso_and_auto_detect_module(filename, false);
    }
}

std::string TransformersDesktopApp::find_recompiled_module_for_serial(const std::string& serial) const {
    const std::string dll_name = serial + "_recomp.dll";
    const fs::path exe_dir = fs::path(exe_directory());
    std::vector<fs::path> candidates = {
        exe_dir / dll_name,
        exe_dir / ".." / "dist" / dll_name,
        fs::path("dist") / dll_name,
        exe_dir / "dist" / dll_name,
        exe_dir / "recomp_full_dll" / dll_name,
        exe_dir / ".." / "build-win64" / "recomp_full_dll" / dll_name,
        exe_dir / ".." / "build-win64" / "recomp_boot_dll" / dll_name,
    };
    std::error_code ec;
    for (const auto& c : candidates) {
        if (fs::exists(c, ec)) {
            auto canon = fs::weakly_canonical(c, ec);
            return ec ? c.string() : canon.string();
        }
    }
    if (!m_settings.last_module_path.empty() && fs::exists(m_settings.last_module_path, ec)) {
        return m_settings.last_module_path;
    }
    return {};
}

bool TransformersDesktopApp::verify_module_matches_iso(
    const std::string& dll_path,
    const std::string& expected_fp,
    std::string* out_reason
) const {
    std::error_code ec;
    if (dll_path.empty() || !fs::exists(dll_path, ec)) {
        if (out_reason) *out_reason = "DLL file does not exist";
        return false;
    }
    HMODULE mod = LoadLibraryA(dll_path.c_str());
    if (!mod) {
        if (out_reason) *out_reason = "LoadLibraryA failed";
        return false;
    }
    auto fp_fn  = reinterpret_cast<FingerprintFn>(reinterpret_cast<void*>(GetProcAddress(mod, "tf_recomp_iso_fingerprint")));
    auto ep_fn  = reinterpret_cast<EntryPointFn>(reinterpret_cast<void*>(GetProcAddress(mod, "tf_recomp_entry_point")));
    auto reg_fn = reinterpret_cast<RegisterFuncsFn>(reinterpret_cast<void*>(GetProcAddress(mod, "tf_recomp_register")));
    bool ok = false;
    if (!fp_fn || !ep_fn || !reg_fn) {
        if (out_reason) *out_reason = "Missing tf_recomp_* exports";
    } else {
        const char* mod_fp = fp_fn();
        if (!mod_fp || std::string(mod_fp) != expected_fp) {
            if (out_reason) *out_reason = "ISO fingerprint mismatch";
        } else {
            ok = true;
        }
    }
    FreeLibrary(mod);
    return ok;
}

void TransformersDesktopApp::refresh_active_module_ready() {
    std::error_code ec;
    m_active_module_ready = !m_active_module_path.empty() && fs::exists(m_active_module_path, ec);
}

void TransformersDesktopApp::update_iso_info_label() {
    if (!m_lbl_iso_info) return;
    if (!m_iso_loaded) {
        SetWindowTextA(m_lbl_iso_info, "No PS2 ISO loaded yet.");
        return;
    }
    refresh_active_module_ready();
    std::ostringstream info;
    info << "Serial: " << m_iso_report.system_cnf.serial << " (" << m_iso_report.system_cnf.region << ")\r\n"
         << "ELF: " << m_iso_report.boot_executable_entry.path << " (entry " << hex32(m_elf_image.e_entry) << ")\r\n"
         << "Native DLL: " << (m_active_module_ready
                                   ? (fs::path(m_active_module_path).filename().string() + " [READY]")
                                   : "Not built yet (click Step 2)");
    SetWindowTextA(m_lbl_iso_info, info.str().c_str());
}

void TransformersDesktopApp::update_button_states() {
    const bool running = m_game_running.load(std::memory_order_relaxed);
    const bool recomp = m_recompiling.load(std::memory_order_relaxed);
    const bool have_dll = m_iso_loaded && m_active_module_ready;

    if (m_btn_iso)    EnableWindow(m_btn_iso, (!running && !recomp) ? TRUE : FALSE);
    if (m_btn_recomp) {
        EnableWindow(m_btn_recomp, (m_iso_loaded && !running && !recomp) ? TRUE : FALSE);
        const char* want_recomp = have_dll ? "2. Verify / Recompile (DLL Ready)" : "2. Recompile ISO (Static AOT)";
        char cur_buf[128] = {};
        GetWindowTextA(m_btn_recomp, cur_buf, sizeof(cur_buf));
        if (std::string(cur_buf) != want_recomp) {
            SetWindowTextA(m_btn_recomp, want_recomp);
        }
    }
    if (m_btn_play) {
        EnableWindow(m_btn_play, (m_iso_loaded && !running && !recomp) ? TRUE : FALSE);
        const char* want_play = have_dll ? "3. PLAY / LAUNCH NATIVE [READY]" : "3. PLAY / LAUNCH NATIVE";
        char cur_buf[128] = {};
        GetWindowTextA(m_btn_play, cur_buf, sizeof(cur_buf));
        if (std::string(cur_buf) != want_play) {
            SetWindowTextA(m_btn_play, want_play);
        }
    }
    if (m_btn_stop) EnableWindow(m_btn_stop, running ? TRUE : FALSE);
}

bool TransformersDesktopApp::load_iso_and_auto_detect_module(const std::string& iso_path, bool preserve_active_module) {
    on_stop_game();
    std::string err;
    if (!m_runtime.mount_iso(iso_path, err)) {
        append_log("ERROR mounting ISO: " + err);
        return false;
    }
    m_iso_report = m_runtime.cdvd.iso().inspect();
    if (!m_iso_report.valid) {
        append_log("ERROR invalid PS2 ISO: " + m_iso_report.error);
        return false;
    }

    std::vector<u8> elf_bytes;
    if (!m_runtime.cdvd.iso().read_file(m_iso_report.boot_executable_entry, elf_bytes) ||
        !ElfParser::parse(elf_bytes, m_elf_image, err)) {
        append_log("ERROR parsing BOOT2 ELF: " + err);
        return false;
    }

    m_iso_loaded = true;
    m_settings.last_iso_path = iso_path;

    std::error_code ec;
    if (!preserve_active_module || m_active_module_path.empty() || !fs::exists(m_active_module_path, ec)) {
        m_active_module_path = find_recompiled_module_for_serial(m_iso_report.system_cnf.serial);
    }
    if (!m_active_module_path.empty() && fs::exists(m_active_module_path, ec)) {
        m_settings.last_module_path = m_active_module_path;
    }
    m_settings.save();

    update_iso_info_label();
    update_button_states();

    append_log("Mounted ISO: " + m_iso_report.system_cnf.serial + " [" + m_iso_report.system_cnf.region + "]");
    append_log("ISO Fingerprint: " + m_iso_report.volume_fingerprint_sha256);
    if (!m_active_module_path.empty() && fs::exists(m_active_module_path, ec)) {
        append_log("Auto-detected native module [READY TO PLAY]: " + m_active_module_path);
        append_log("You can click '3. PLAY / LAUNCH NATIVE [READY]' immediately!");
    } else {
        append_log("No prebuilt DLL found for " + m_iso_report.system_cnf.serial + ". Click '2. Recompile ISO (Static AOT)' to build it.");
    }
    return true;
}

void TransformersDesktopApp::on_recompile_iso() {
    if (!m_iso_loaded) {
        append_log("Select a PS2 ISO first before recompiling.");
        return;
    }
    if (m_game_running.load(std::memory_order_relaxed)) {
        append_log("Stop the running game before recompiling.");
        return;
    }

    // If a valid matching <SERIAL>_recomp.dll already exists and matches this ISO's fingerprint,
    // verify and keep it intact instead of clearing or needlessly overwriting it!
    std::string existing_dll = m_active_module_path;
    std::error_code ec;
    if (existing_dll.empty() || !fs::exists(existing_dll, ec)) {
        existing_dll = find_recompiled_module_for_serial(m_iso_report.system_cnf.serial);
    }
    if (!existing_dll.empty() && verify_module_matches_iso(existing_dll, m_iso_report.volume_fingerprint_sha256)) {
        m_active_module_path = existing_dll;
        m_settings.last_module_path = existing_dll;
        m_settings.save();
        update_iso_info_label();
        update_button_states();
        append_log("Verified matching native DLL already ready: " + existing_dll);
        append_log("No recompile needed - click '3. PLAY / LAUNCH NATIVE [READY]'!");
        return;
    }

    if (m_recompiling.exchange(true)) return;
    if (m_recomp_thread.joinable()) {
        m_recomp_thread.join();
    }

    update_button_states();
    append_log("Starting full static recompilation + parallel native DLL compilation...");
    std::string iso_p = m_settings.last_iso_path;
    std::string serial = m_iso_report.system_cnf.serial;
    std::string runtime_root = (fs::path(exe_directory()) / "..").string();
    if (!fs::exists(fs::path(runtime_root) / "Runtime/CPU/EEContext.h", ec)) {
        runtime_root = exe_directory();
    }

    m_recomp_thread = std::thread([this, iso_p, serial, runtime_root]() {
        PipelineOptions opts;
        opts.iso_path = iso_p;
        opts.runtime_source_root = runtime_root;
        opts.compile_native_module = true;
        auto res = RecompPipeline::run(opts);
        std::error_code local_ec;
        if (res.success && !res.compiled_module_path.empty() && fs::exists(res.compiled_module_path, local_ec)) {
            // Also copy next to Transformers.exe if writable so future sessions auto-detect immediately
            fs::path local_copy = fs::path(exe_directory()) / (serial + "_recomp.dll");
            std::error_code cp_ec;
            if (fs::weakly_canonical(res.compiled_module_path, cp_ec) != fs::weakly_canonical(local_copy, cp_ec)) {
                fs::copy_file(res.compiled_module_path, local_copy, fs::copy_options::overwrite_existing, cp_ec);
            }
            {
                std::lock_guard<std::mutex> lk(m_recomp_mutex);
                m_pending_recomp_module = res.compiled_module_path;
                m_recomp_completed_flag = true;
            }
            append_log("Recompilation complete! Native DLL: " + res.compiled_module_path);
        } else {
            std::string err_msg = res.error.empty()
                ? "Recompilation finished without producing a native DLL."
                : res.error;
            append_log("Recompilation failed: " + err_msg);
        }
        m_recompiling = false;
    });
}

DWORD WINAPI TransformersDesktopApp::guest_thread_proc(LPVOID param) {
    auto* self = static_cast<TransformersDesktopApp*>(param);
    if (!self) return 1;

    try {
        self->m_runtime.call_guest(self->m_launch_entry);
    } catch (const std::exception& ex) {
        self->append_log(std::string("RUNTIME EXCEPTION at PC=") + hex32(self->m_runtime.cpu.pc) + ": " + ex.what());
    } catch (...) {
        self->append_log(std::string("UNKNOWN RUNTIME EXCEPTION at PC=") + hex32(self->m_runtime.cpu.pc));
    }

    std::ostringstream oss;
    oss << "Execution stopped at PC=" << hex32(self->m_runtime.cpu.pc)
        << " | static_calls=" << self->m_runtime.total_static_calls
        << " | fallback_steps=" << self->m_runtime.fallback_steps_executed
        << " | syscalls=" << self->m_runtime.syscalls.syscall_counts().size()
        << " | gs_frames=" << self->m_runtime.gs.frames_presented
        << " | gs_prims=" << self->m_runtime.gs.primitives_drawn;
    if (!self->m_runtime.cpu.trap_message.empty()) {
        oss << " | note: " << self->m_runtime.cpu.trap_message;
    }
    self->append_log(oss.str());
    self->m_game_running = false;
    return 0;
}

void TransformersDesktopApp::on_launch_game() {
    if (!m_iso_loaded) {
        append_log("Please select your PS2 ISO first.");
        return;
    }
    if (m_game_running.load(std::memory_order_relaxed)) return;

    // Cleanly close any previously finished guest worker thread handle
    if (m_worker_handle) {
        WaitForSingleObject(m_worker_handle, INFINITE);
        CloseHandle(m_worker_handle);
        m_worker_handle = nullptr;
    }

    std::string err;
    m_runtime.reset();
    m_runtime.total_static_calls = 0;
    m_runtime.fallback_steps_executed = 0;
    m_runtime.max_static_calls = 0; // Unlimited continuous native execution
    m_runtime.enable_frame_pacing = true; // Smooth real-time 60.0 Hz pacing in interactive desktop mode
    m_runtime.input_config = m_settings.input;

    if (!m_runtime.mount_iso(m_settings.last_iso_path, err)) {
        append_log("Failed mounting ISO: " + err);
        return;
    }
    if (!m_runtime.memcard0.mount_directory(m_settings.memory_card_dir, err)) {
        append_log("WARNING mounting Memory Card (" + m_settings.memory_card_dir + "): " + err);
    }
    if (!m_runtime.load_elf(m_elf_image, err)) {
        append_log("Failed loading ELF into EE RAM: " + err);
        return;
    }

    std::error_code ec;
    if (m_active_module_path.empty() || !fs::exists(m_active_module_path, ec)) {
        m_active_module_path = find_recompiled_module_for_serial(m_iso_report.system_cnf.serial);
        update_iso_info_label();
    }

    u32 entry = m_elf_image.e_entry;
    m_runtime.function_table().clear();
    if (!m_active_module_path.empty() && fs::exists(m_active_module_path, ec)) {
        if (m_loaded_dll) {
            FreeLibrary(m_loaded_dll);
            m_loaded_dll = nullptr;
        }
        m_loaded_dll = LoadLibraryA(m_active_module_path.c_str());
        if (m_loaded_dll) {
            auto fp_fn = reinterpret_cast<FingerprintFn>(reinterpret_cast<void*>(GetProcAddress(m_loaded_dll, "tf_recomp_iso_fingerprint")));
            auto ep_fn = reinterpret_cast<EntryPointFn>(reinterpret_cast<void*>(GetProcAddress(m_loaded_dll, "tf_recomp_entry_point")));
            auto reg_fn = reinterpret_cast<RegisterFuncsFn>(reinterpret_cast<void*>(GetProcAddress(m_loaded_dll, "tf_recomp_register")));
            if (fp_fn && ep_fn && reg_fn) {
                if (std::string(fp_fn()) != m_iso_report.volume_fingerprint_sha256) {
                    append_log("WARNING: Module ISO fingerprint mismatch! Recompile recommended.");
                }
                entry = ep_fn();
                reg_fn(&m_runtime.function_table());
                append_log("Loaded " + std::to_string(m_runtime.function_table().size()) +
                           " native x64 recompiled functions from " + fs::path(m_active_module_path).filename().string());
            } else {
                append_log("WARNING: Recompiled DLL missing expected tf_recomp_* exports.");
            }
        } else {
            append_log("WARNING: Failed loading DLL " + m_active_module_path);
        }
    } else {
        append_log("No compiled DLL found; running with R5900 fallback interpreter.");
    }

    m_has_frame = false;
    m_frame_w = 0;
    m_frame_h = 0;
    m_last_presented_seq = 0;
    m_last_hw_presented_seq = 0;
    m_last_frame_was_hw = false;
    m_last_present_us = 0;
    if (m_hw_gs_active) {
        m_hw_gs.set_resolution_scale(m_settings.graphics.internal_resolution_scale);
        m_runtime.gs.set_hw_backend(&m_hw_gs);
    } else {
        m_runtime.gs.set_hw_backend(nullptr);
    }
    m_kb_pressed_mask.store(0, std::memory_order_relaxed);
    m_kb_edge_latch.store(0, std::memory_order_relaxed);
    m_vibration_enabled.store(m_settings.input.vibration_enabled, std::memory_order_relaxed);
    m_presenter.update_settings(m_settings.graphics);
    m_presenter.clear_black();
    m_display_fps = 0;
    m_fps_last_gs_frames = 0;
    m_fps_window_start_ms = static_cast<u64>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());

    if (m_frame_event) {
        ResetEvent(m_frame_event);
        m_runtime.gs.set_present_callback([this]() {
            SetEvent(m_frame_event);
        });
    }
    m_runtime.set_jit_input_callback([this]() {
        sample_jit_input();
    });

    m_launch_entry = entry;
    m_game_running = true;
    update_button_states();

    // Move keyboard focus off launcher buttons so Space/Enter go to gameplay instead of clicking buttons
    if (m_hwnd) {
        SetFocus(m_hwnd);
    }

    append_log(std::string("Launching native PS2 entry point ") + hex32(entry) +
               " (" + (m_hw_gs_active ? "D3D11 HW GS + Flip" : m_presenter.backend_name()) +
               ", Optimal 60Hz Pacing + JIT Input)...");

    // Use Win32 CreateThread with 16 MiB reserved stack so deep recompiled PS2 call chains never stack-overflow
    m_worker_handle = CreateThread(
        nullptr,
        16u * 1024u * 1024u,
        &TransformersDesktopApp::guest_thread_proc,
        this,
        STACK_SIZE_PARAM_IS_A_RESERVATION,
        nullptr
    );
    if (!m_worker_handle) {
        m_game_running = false;
        update_button_states();
        append_log("ERROR: CreateThread failed for PS2 guest worker (Win32 error " + std::to_string(GetLastError()) + ")");
    }
}

void TransformersDesktopApp::on_stop_game() {
    m_runtime.gs.set_present_callback(nullptr);
    m_runtime.set_jit_input_callback(nullptr);
    m_runtime.request_halt();
    if (m_worker_handle) {
        DWORD wait_res = WaitForSingleObject(m_worker_handle, 3000);
        if (wait_res == WAIT_TIMEOUT) {
            m_runtime.request_halt();
            wait_res = WaitForSingleObject(m_worker_handle, 2000);
        }
        if (wait_res == WAIT_OBJECT_0) {
            CloseHandle(m_worker_handle);
            m_worker_handle = nullptr;
        } else {
            append_log("WARNING: Guest worker did not halt within 5s; forcing thread cleanup.");
            TerminateThread(m_worker_handle, 1);
            WaitForSingleObject(m_worker_handle, 500);
            CloseHandle(m_worker_handle);
            m_worker_handle = nullptr;
        }
    }
    m_runtime.gs.set_hw_backend(nullptr);
    m_game_running = false;
    m_has_frame = false;
    m_frame_w = 0;
    m_frame_h = 0;
    m_last_presented_seq = 0;
    m_last_hw_presented_seq = 0;
    m_last_frame_was_hw = false;
    m_last_present_us = 0;
    m_kb_pressed_mask.store(0, std::memory_order_relaxed);
    m_kb_edge_latch.store(0, std::memory_order_relaxed);
    m_runtime.pad0.reset();
    set_xinput_vibration(0, 0);
    m_display_fps = 0;
    m_fps_last_gs_frames = 0;
    if (m_viewport && IsWindow(m_viewport)) {
        m_presenter.clear_black();
        InvalidateRect(m_viewport, nullptr, FALSE);
    }
    if (m_hwnd && IsWindow(m_hwnd)) {
        update_button_states();
    }
}

void TransformersDesktopApp::append_log(const std::string& line) {
    std::lock_guard<std::mutex> lk(m_log_mutex);
    m_pending_logs.push_back(line);
}

void TransformersDesktopApp::refresh_status_ui() {
    // Integrate completed recompilation result safely on the UI thread (NEVER overwrite with empty path!)
    {
        std::string completed_mod;
        bool has_completed = false;
        {
            std::lock_guard<std::mutex> lk(m_recomp_mutex);
            if (m_recomp_completed_flag) {
                completed_mod = m_pending_recomp_module;
                m_pending_recomp_module.clear();
                m_recomp_completed_flag = false;
                has_completed = true;
            }
        }
        std::error_code ec;
        if (has_completed && !completed_mod.empty() && fs::exists(completed_mod, ec)) {
            m_active_module_path = completed_mod;
            m_settings.last_module_path = completed_mod;
            m_settings.save();
            update_iso_info_label();
        }
    }

    update_button_states();

    std::vector<std::string> batch;
    {
        std::lock_guard<std::mutex> lk(m_log_mutex);
        batch.swap(m_pending_logs);
    }
    if (!batch.empty() && m_edit_log) {
        std::ostringstream oss;
        for (const auto& l : batch) oss << l << "\r\n";
        int len = GetWindowTextLengthA(m_edit_log);
        SendMessageA(m_edit_log, EM_SETSEL, len, len);
        std::string s = oss.str();
        SendMessageA(m_edit_log, EM_REPLACESEL, FALSE, reinterpret_cast<LPARAM>(s.c_str()));
    }

    if (m_lbl_telemetry) {
        const bool running = m_game_running.load(std::memory_order_relaxed);
        const char* state_str = running
            ? "RUNNING"
            : (m_recompiling.load(std::memory_order_relaxed) ? "RECOMPILING" : "READY");
        const u32 boot_st = running ? m_runtime.memory.read32(0x00503B68u + 48u) : 0u;
        const u32 work_us = running ? m_runtime.last_frame_work_us() : 0u;
        std::ostringstream t;
        t << "State     : " << state_str << " (" << m_display_fps << " FPS";
        if (running && work_us > 0) {
            t << " | " << (work_us / 1000u) << "." << ((work_us % 1000u) / 100u) << "ms";
        }
        t << ") b=" << boot_st << "\r\n"
          << "Display   : " << (m_hw_gs_active ? "D3D11 HW GS + Flip" : m_presenter.backend_name())
          << (m_settings.graphics.vsync ? " [VSync]" : (m_presenter.supports_tearing() ? " [0ms Immediate]" : " [NoVSync]"))
          << " (" << (m_last_present_us / 1000u) << "." << ((m_last_present_us % 1000u) / 100u) << "ms)\r\n"
          << "Guest PC  : " << hex32(m_runtime.cpu.pc) << "\r\n"
          << "Native Fns: " << m_runtime.function_table().size() << "\r\n"
          << "Calls/Step: " << m_runtime.total_static_calls << " / " << m_runtime.fallback_steps_executed << "\r\n"
          << "GS Frames : " << m_runtime.gs.frames_presented << " | Prims: " << m_runtime.gs.primitives_drawn << "\r\n"
          << "VU1 XGKick: " << m_runtime.vu1.xgkick_count << "\r\n"
          << "Pad 0     : " << (m_xinput_connected.load(std::memory_order_relaxed) ? "XInput + Keyboard" : "Keyboard (JIT)") << "\r\n"
          << "Res/Audio : " << m_settings.graphics.internal_resolution_scale << "x | SPU2 "
          << (m_settings.audio_enabled ? (std::to_string(m_settings.audio_volume_percent) + "%") : "MUTED");
        SetWindowTextA(m_lbl_telemetry, t.str().c_str());
    }
}

void TransformersDesktopApp::update_true_fps(u64 now_ms) {
    if (now_ms - m_fps_window_start_ms >= 500u) {
        const u64 cur_frames = m_runtime.gs.frames_presented;
        const u64 delta_frames = (cur_frames >= m_fps_last_gs_frames) ? (cur_frames - m_fps_last_gs_frames) : 0u;
        const u64 dt_ms = std::max<u64>(1u, now_ms - m_fps_window_start_ms);
        m_display_fps = static_cast<u32>((delta_frames * 1000u + (dt_ms / 2u)) / dt_ms);
        m_fps_last_gs_frames = cur_frames;
        m_fps_window_start_ms = now_ms;
    }
}

bool TransformersDesktopApp::redraw_last_frame() {
    if (m_hwnd && IsIconic(m_hwnd)) return false;
    if (m_hw_gs_active && m_last_frame_was_hw) {
        u64 seq = 0;
        ID3D11ShaderResourceView* srv = nullptr;
        float u0 = 0.0f, v0 = 0.0f, u1 = 1.0f, v1 = 1.0f;
        u32 w = 0, h = 0;
        if (m_hw_gs.get_presented_srv(seq, srv, u0, v0, u1, v1, w, h) && srv != nullptr) {
            SrvReleaseGuard srv_guard{srv};
            const bool ok = m_presenter.present_d3d11_srv(srv, u0, v0, u1, v1, w, h);
            g_last_present_ms = static_cast<u64>(
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()).count());
            return ok;
        }
    }
    if (m_has_frame && m_frame_w > 0 && m_frame_h > 0 &&
        m_frame_copy.size() >= static_cast<std::size_t>(m_frame_w) * m_frame_h) {
        m_presenter.present_rgba(m_frame_copy.data(), m_frame_w, m_frame_h);
        g_last_present_ms = static_cast<u64>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
        return true;
    }
    m_presenter.clear_black();
    return true;
}

bool TransformersDesktopApp::present_current_frame() {
    if (m_hwnd && IsIconic(m_hwnd)) return false;

    if (m_hw_gs_active) {
        u64 seq_probe = m_last_hw_presented_seq;
        ID3D11ShaderResourceView* srv = nullptr;
        float u0 = 0.0f, v0 = 0.0f, u1 = 1.0f, v1 = 1.0f;
        u32 w = 0, h = 0;
        if (m_hw_gs.get_presented_srv(seq_probe, srv, u0, v0, u1, v1, w, h) && srv != nullptr) {
            SrvReleaseGuard srv_guard{srv};
            const auto t0 = std::chrono::steady_clock::now();
            const bool presented = m_presenter.present_d3d11_srv(srv, u0, v0, u1, v1, w, h);
            const auto t1 = std::chrono::steady_clock::now();
            g_last_present_ms = static_cast<u64>(
                std::chrono::duration_cast<std::chrono::milliseconds>(t1.time_since_epoch()).count());
            if (presented) {
                m_last_hw_presented_seq = seq_probe;
                m_last_presented_seq = m_runtime.gs.presented_seq();
                m_last_frame_was_hw = true;
                m_last_present_us = static_cast<u32>(
                    std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
                return true;
            }
            return false;
        }
    }

    u32 w = 0;
    u32 h = 0;
    if (m_runtime.gs.copy_presented_framebuffer_if_newer(m_last_presented_seq, m_frame_copy, w, h) &&
        w > 0 && h > 0 &&
        m_frame_copy.size() >= static_cast<std::size_t>(w) * h) {
        m_frame_w = w;
        m_frame_h = h;
        m_has_frame = true;
        m_last_frame_was_hw = false;
        const auto t0 = std::chrono::steady_clock::now();
        m_presenter.present_rgba(m_frame_copy.data(), w, h);
        const auto t1 = std::chrono::steady_clock::now();
        m_last_present_us = static_cast<u32>(
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
        g_last_present_ms = static_cast<u64>(
            std::chrono::duration_cast<std::chrono::milliseconds>(t1.time_since_epoch()).count());
        return true;
    }
    return false;
}

void TransformersDesktopApp::record_kb_event(WPARAM vk, bool pressed) {
    const u32 bit = vk_to_kb_bit(vk);
    if (bit == 0u) return;
    if (pressed) {
        m_kb_pressed_mask.fetch_or(bit, std::memory_order_relaxed);
        m_kb_edge_latch.fetch_or(bit, std::memory_order_release);
    } else {
        m_kb_pressed_mask.fetch_and(~bit, std::memory_order_relaxed);
    }
}

void TransformersDesktopApp::poll_input() {
    sample_jit_input();
}

void TransformersDesktopApp::sample_jit_input() {
    if (!m_game_running.load(std::memory_order_relaxed)) return;

    // Keyboard active whenever main window (or any child such as viewport/controls) is foreground,
    // and NOT while the settings dialog is active. Thread-safe for JIT sampling on the EE thread.
    HWND fg = GetForegroundWindow();
    HWND dlg = m_dlg_settings_atomic.load(std::memory_order_acquire);
    const bool settings_active = (dlg != nullptr && IsWindow(dlg) &&
                                  (fg == dlg || IsChild(dlg, fg)));
    const bool kb_active = !settings_active && m_hwnd != nullptr &&
                           (fg == m_hwnd || IsChild(m_hwnd, fg));

    // Consume sub-frame edge-latched keypresses captured instantly via WM_KEYDOWN so quick taps never drop,
    // combined with live kernel GetAsyncKeyState() at exact t=0.0ms of the new frame!
    const u32 latched = kb_active ? m_kb_edge_latch.exchange(0u, std::memory_order_acq_rel)
                                  : (m_kb_edge_latch.store(0u, std::memory_order_relaxed), 0u);
    auto kb_bit = [&](u32 mask, bool live) -> bool {
        return kb_active && (live || ((latched & mask) != 0));
    };

    bool dpad_up    = kb_bit(kKbUp,    key_down(VK_UP));
    bool dpad_down  = kb_bit(kKbDown,  key_down(VK_DOWN));
    bool dpad_left  = kb_bit(kKbLeft,  key_down(VK_LEFT));
    bool dpad_right = kb_bit(kKbRight, key_down(VK_RIGHT));

    bool btn_start  = kb_bit(kKbStart,  key_down(VK_RETURN));
    bool btn_select = kb_bit(kKbSelect, key_down(VK_BACK) || key_down(VK_RSHIFT));

    bool btn_cross    = kb_bit(kKbCross,    key_down(VK_SPACE) || key_down('Z') || key_down('N'));
    bool btn_circle   = kb_bit(kKbCircle,   key_down('X') || key_down('M'));
    bool btn_square   = kb_bit(kKbSquare,   key_down('C') || key_down('B'));
    bool btn_triangle = kb_bit(kKbTriangle, key_down('V') || key_down('F'));

    bool btn_l1 = kb_bit(kKbL1, key_down('Q'));
    bool btn_r1 = kb_bit(kKbR1, key_down('E'));
    bool btn_l2 = kb_bit(kKbL2, key_down('1'));
    bool btn_r2 = kb_bit(kKbR2, key_down('3'));
    bool btn_l3 = false;
    bool btn_r3 = false;

    // Keyboard Left Stick (WASD): -1 = Left/Up, +1 = Right/Down (matches DualShock2State)
    f32 kb_lx = 0.0f;
    f32 kb_ly = 0.0f;
    if (kb_active) {
        if (kb_bit(kKbLsLeft,  key_down('A'))) kb_lx -= 1.0f;
        if (kb_bit(kKbLsRight, key_down('D'))) kb_lx += 1.0f;
        if (kb_bit(kKbLsUp,    key_down('W'))) kb_ly -= 1.0f;
        if (kb_bit(kKbLsDown,  key_down('S'))) kb_ly += 1.0f;
    }

    // Keyboard Right Stick (IJKL or Numpad 8/4/5/6)
    f32 kb_rx = 0.0f;
    f32 kb_ry = 0.0f;
    if (kb_active) {
        if (kb_bit(kKbRsLeft,  key_down('J') || key_down(VK_NUMPAD4))) kb_rx -= 1.0f;
        if (kb_bit(kKbRsRight, key_down('L') || key_down(VK_NUMPAD6))) kb_rx += 1.0f;
        if (kb_bit(kKbRsUp,    key_down('I') || key_down(VK_NUMPAD8))) kb_ry -= 1.0f;
        if (kb_bit(kKbRsDown,  key_down('K') || key_down(VK_NUMPAD5))) kb_ry += 1.0f;
    }

    // XInput Controller 0 (dynamically loaded from xinput1_4.dll / xinput1_3.dll / xinput9_1_0.dll)
    f32 xi_lx = 0.0f;
    f32 xi_ly = 0.0f;
    f32 xi_rx = 0.0f;
    f32 xi_ry = 0.0f;
    u8  xi_l2_pressure = 0;
    u8  xi_r2_pressure = 0;

    if (m_xinput_get_state) {
        const u64 now_ms = static_cast<u64>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
        // Avoid Windows HID enumeration stall: probe unconnected XInput slot at most once per 1.5s;
        // once connected, poll immediately on every frame right before pad DMA!
        const bool connected = m_xinput_connected.load(std::memory_order_relaxed);
        const u64 last_probe = m_last_xinput_probe_ms.load(std::memory_order_relaxed);
        if (connected || (now_ms - last_probe >= 1500u)) {
            m_last_xinput_probe_ms.store(now_ms, std::memory_order_relaxed);
            auto get_fn = reinterpret_cast<XInputGetStateFn>(m_xinput_get_state);
            TfXInputState xstate{};
            if (get_fn(0, &xstate) == ERROR_SUCCESS) {
                m_xinput_connected.store(true, std::memory_order_relaxed);
                const WORD b = xstate.Gamepad.wButtons;

                dpad_up      = dpad_up      || ((b & kXInputDpadUp) != 0);
                dpad_down    = dpad_down    || ((b & kXInputDpadDown) != 0);
                dpad_left    = dpad_left    || ((b & kXInputDpadLeft) != 0);
                dpad_right   = dpad_right   || ((b & kXInputDpadRight) != 0);
                btn_start    = btn_start    || ((b & kXInputStart) != 0);
                btn_select   = btn_select   || ((b & kXInputBack) != 0);
                btn_l3       = btn_l3       || ((b & kXInputLeftThumb) != 0);
                btn_r3       = btn_r3       || ((b & kXInputRightThumb) != 0);
                btn_l1       = btn_l1       || ((b & kXInputLeftShoulder) != 0);
                btn_r1       = btn_r1       || ((b & kXInputRightShoulder) != 0);
                btn_cross    = btn_cross    || ((b & kXInputA) != 0);
                btn_circle   = btn_circle   || ((b & kXInputB) != 0);
                btn_square   = btn_square   || ((b & kXInputX) != 0);
                btn_triangle = btn_triangle || ((b & kXInputY) != 0);

                if (xstate.Gamepad.bLeftTrigger > kXInputTriggerThresh) {
                    btn_l2 = true;
                    xi_l2_pressure = xstate.Gamepad.bLeftTrigger;
                }
                if (xstate.Gamepad.bRightTrigger > kXInputTriggerThresh) {
                    btn_r2 = true;
                    xi_r2_pressure = xstate.Gamepad.bRightTrigger;
                }

                // XInput Y axis is +32767 up / -32768 down; negate Y for PS2 DualShock 2 (-1 up, +1 down)
                xi_lx = std::clamp(static_cast<f32>(xstate.Gamepad.sThumbLX) / 32767.0f, -1.0f, 1.0f);
                xi_ly = std::clamp(-static_cast<f32>(xstate.Gamepad.sThumbLY) / 32767.0f, -1.0f, 1.0f);
                xi_rx = std::clamp(static_cast<f32>(xstate.Gamepad.sThumbRX) / 32767.0f, -1.0f, 1.0f);
                xi_ry = std::clamp(-static_cast<f32>(xstate.Gamepad.sThumbRY) / 32767.0f, -1.0f, 1.0f);

                // Sync DualShock 2 rumble motors -> XInput vibration
                if (m_vibration_enabled.load(std::memory_order_relaxed)) {
                    const u16 left_vib = static_cast<u16>(m_runtime.pad0.large_motor) * 257u;
                    const u16 right_vib = static_cast<u16>(m_runtime.pad0.small_motor) * 257u;
                    set_xinput_vibration(left_vib, right_vib);
                } else {
                    set_xinput_vibration(0, 0);
                }
            } else {
                m_xinput_connected.store(false, std::memory_order_relaxed);
            }
        }
    }

    {
        std::lock_guard<std::mutex> pad_lk(m_runtime.pad_mutex());
        auto& pad = m_runtime.pad0;
        pad.set_button(Ps2Button::DpadUp,    dpad_up);
        pad.set_button(Ps2Button::DpadDown,  dpad_down);
        pad.set_button(Ps2Button::DpadLeft,  dpad_left);
        pad.set_button(Ps2Button::DpadRight, dpad_right);
        pad.set_button(Ps2Button::Start,     btn_start);
        pad.set_button(Ps2Button::Select,    btn_select);
        pad.set_button(Ps2Button::L1,        btn_l1);
        pad.set_button(Ps2Button::R1,        btn_r1);
        pad.set_button(Ps2Button::L2,        btn_l2);
        pad.set_button(Ps2Button::R2,        btn_r2);
        pad.set_button(Ps2Button::L3,        btn_l3);
        pad.set_button(Ps2Button::R3,        btn_r3);
        pad.set_button(Ps2Button::Cross,     btn_cross);
        pad.set_button(Ps2Button::Circle,    btn_circle);
        pad.set_button(Ps2Button::Square,    btn_square);
        pad.set_button(Ps2Button::Triangle,  btn_triangle);

        // Preserve analog trigger pressure from XInput when triggers are active
        if (xi_l2_pressure > 0) pad.pressure[10] = xi_l2_pressure;
        if (xi_r2_pressure > 0) pad.pressure[11] = xi_r2_pressure;

        // Prefer non-zero keyboard stick input so controller drift never overrides WASD/IJKL
        const f32 final_lx = (kb_lx != 0.0f) ? kb_lx : xi_lx;
        const f32 final_ly = (kb_ly != 0.0f) ? kb_ly : xi_ly;
        const f32 final_rx = (kb_rx != 0.0f) ? kb_rx : xi_rx;
        const f32 final_ry = (kb_ry != 0.0f) ? kb_ry : xi_ry;

        pad.set_left_stick(final_lx, final_ly, m_runtime.input_config);
        pad.set_right_stick(final_rx, final_ry, m_runtime.input_config);
    }
}

} // namespace tf::ps2

#endif // _WIN32
