#include "DualShock2.h"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace tf::ps2 {

namespace {

u8 axis_to_u8(f32 v, const InputConfig& cfg, bool invert) {
    if (std::fabs(v) < cfg.deadzone) {
        v = 0.0f;
    } else {
        f32 sign = v < 0.0f ? -1.0f : 1.0f;
        v = sign * ((std::fabs(v) - cfg.deadzone) / std::max(0.01f, 1.0f - cfg.deadzone));
    }
    v = std::clamp(v * cfg.sensitivity, -1.0f, 1.0f);
    if (invert) v = -v;
    int scaled = static_cast<int>(std::lround((v * 0.5f + 0.5f) * 255.0f));
    return static_cast<u8>(std::clamp(scaled, 0, 255));
}

} // namespace

void DualShock2State::reset() {
    buttons_pressed = 0;
    rx = ry = lx = ly = 0x80;
    pressure.fill(0);
    small_motor = 0;
    large_motor = 0;
}

void DualShock2State::set_button(Ps2Button btn, bool pressed) {
    u16 bit = static_cast<u16>(btn);
    if (pressed) buttons_pressed |= bit;
    else         buttons_pressed &= ~bit;
    const u8 p = pressed ? 0xFFu : 0x00u;
    switch (btn) {
        case Ps2Button::DpadRight: pressure[0]  = p; break;
        case Ps2Button::DpadLeft:  pressure[1]  = p; break;
        case Ps2Button::DpadUp:    pressure[2]  = p; break;
        case Ps2Button::DpadDown:  pressure[3]  = p; break;
        case Ps2Button::Triangle:  pressure[4]  = p; break;
        case Ps2Button::Circle:    pressure[5]  = p; break;
        case Ps2Button::Cross:     pressure[6]  = p; break;
        case Ps2Button::Square:    pressure[7]  = p; break;
        case Ps2Button::L1:        pressure[8]  = p; break;
        case Ps2Button::R1:        pressure[9]  = p; break;
        case Ps2Button::L2:        pressure[10] = p; break;
        case Ps2Button::R2:        pressure[11] = p; break;
        default: break;
    }
}

void DualShock2State::set_left_stick(f32 x, f32 y, const InputConfig& cfg) {
    lx = axis_to_u8(x, cfg, false);
    ly = axis_to_u8(y, cfg, false);
}

void DualShock2State::set_right_stick(f32 x, f32 y, const InputConfig& cfg) {
    rx = axis_to_u8(x, cfg, cfg.invert_camera_x);
    ry = axis_to_u8(y, cfg, cfg.invert_camera_y);
}

void DualShock2State::pack_ps2_pad_buffer(u8 out_buf[32]) const {
    std::memset(out_buf, 0, 32);
    out_buf[0] = 0x00; // Valid
    out_buf[1] = 0x79; // DualShock 2 Analog + Pressure mode
    u16 active_low = static_cast<u16>(~buttons_pressed);
    out_buf[2] = static_cast<u8>(active_low & 0xFFu);
    out_buf[3] = static_cast<u8>((active_low >> 8) & 0xFFu);
    out_buf[4] = rx;
    out_buf[5] = ry;
    out_buf[6] = lx;
    out_buf[7] = ly;
    std::memcpy(out_buf + 8, pressure.data(), pressure.size());
}

} // namespace tf::ps2
