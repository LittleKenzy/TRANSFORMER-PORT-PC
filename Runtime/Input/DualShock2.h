#pragma once

#include "../../Recompiler/Common/Types.h"
#include <array>

namespace tf::ps2 {

// Standard PS2 DualShock 2 button bitmasks (active-low in raw PS2 pad buffer)
enum class Ps2Button : u16 {
    Select   = 1u << 0,
    L3       = 1u << 1,
    R3       = 1u << 2,
    Start    = 1u << 3,
    DpadUp   = 1u << 4,
    DpadRight= 1u << 5,
    DpadDown = 1u << 6,
    DpadLeft = 1u << 7,
    L2       = 1u << 8,
    R2       = 1u << 9,
    L1       = 1u << 10,
    R1       = 1u << 11,
    Triangle = 1u << 12,
    Circle   = 1u << 13,
    Cross    = 1u << 14,
    Square   = 1u << 15,
};

struct InputConfig {
    f32 deadzone = 0.15f;
    f32 sensitivity = 1.0f;
    bool invert_camera_x = false;
    bool invert_camera_y = false;
    bool vibration_enabled = true;
};

struct DualShock2State {
    u16 buttons_pressed = 0; // Active-high host convenience bitmask
    u8 rx = 0x80, ry = 0x80; // Right analog stick (0x00..0xFF, center 0x80)
    u8 lx = 0x80, ly = 0x80; // Left analog stick (0x00..0xFF, center 0x80)
    std::array<u8, 12> pressure{}; // DualShock 2 analog button pressures
    u8 small_motor = 0;
    u8 large_motor = 0;

    void reset();
    void set_button(Ps2Button btn, bool pressed);
    void set_left_stick(f32 x, f32 y, const InputConfig& cfg);
    void set_right_stick(f32 x, f32 y, const InputConfig& cfg);

    // Encodes standard 32-byte libpad DMA report packet (active-low buttons)
    void pack_ps2_pad_buffer(u8 out_buf[32]) const;
};

} // namespace tf::ps2
