#pragma once

#include <array>
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace tf::ps2 {

using u8  = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using s8  = std::int8_t;
using s16 = std::int16_t;
using s32 = std::int32_t;
using s64 = std::int64_t;
using f32 = float;
using f64 = double;

struct alignas(16) u128 {
    u64 lo = 0;
    u64 hi = 0;

    constexpr u128() = default;
    constexpr u128(u64 low, u64 high = 0) : lo(low), hi(high) {}

    u32 uw(std::size_t idx) const {
        if (idx == 0) return static_cast<u32>(lo & 0xFFFFFFFFULL);
        if (idx == 1) return static_cast<u32>(lo >> 32);
        if (idx == 2) return static_cast<u32>(hi & 0xFFFFFFFFULL);
        return static_cast<u32>(hi >> 32);
    }

    s32 sw(std::size_t idx) const { return static_cast<s32>(uw(idx)); }

    void set_uw(std::size_t idx, u32 val) {
        switch (idx & 3) {
            case 0: lo = (lo & 0xFFFFFFFF00000000ULL) | val; break;
            case 1: lo = (lo & 0x00000000FFFFFFFFULL) | (static_cast<u64>(val) << 32); break;
            case 2: hi = (hi & 0xFFFFFFFF00000000ULL) | val; break;
            case 3: hi = (hi & 0x00000000FFFFFFFFULL) | (static_cast<u64>(val) << 32); break;
        }
    }

    u16 uh(std::size_t idx) const {
        u32 w = uw(idx >> 1);
        return static_cast<u16>((idx & 1) ? (w >> 16) : (w & 0xFFFFU));
    }

    s16 sh(std::size_t idx) const { return static_cast<s16>(uh(idx)); }

    void set_uh(std::size_t idx, u16 val) {
        std::size_t widx = (idx >> 1) & 3;
        u32 w = uw(widx);
        if (idx & 1) {
            w = (w & 0x0000FFFFU) | (static_cast<u32>(val) << 16);
        } else {
            w = (w & 0xFFFF0000U) | val;
        }
        set_uw(widx, w);
    }

    u8 ub(std::size_t idx) const {
        u64 lane = (idx < 8) ? lo : hi;
        return static_cast<u8>((lane >> ((idx & 7) * 8)) & 0xFFU);
    }

    s8 sb(std::size_t idx) const { return static_cast<s8>(ub(idx)); }

    void set_ub(std::size_t idx, u8 val) {
        u64 shift = (idx & 7) * 8;
        u64 mask = ~(0xFFULL << shift);
        if (idx < 8) {
            lo = (lo & mask) | (static_cast<u64>(val) << shift);
        } else {
            hi = (hi & mask) | (static_cast<u64>(val) << shift);
        }
    }

    u64 ud(std::size_t idx) const { return (idx & 1) ? hi : lo; }
    s64 sd(std::size_t idx) const { return static_cast<s64>(ud(idx)); }

    void set_ud(std::size_t idx, u64 val) {
        if (idx & 1) hi = val;
        else lo = val;
    }

    bool operator==(const u128& o) const { return lo == o.lo && hi == o.hi; }
    bool operator!=(const u128& o) const { return !(*this == o); }
};

static_assert(sizeof(u128) == 16, "u128 must be 16 bytes");

inline u16 read_le16(const u8* p) {
    return static_cast<u16>(p[0]) | (static_cast<u16>(p[1]) << 8);
}

inline u32 read_le32(const u8* p) {
    return static_cast<u32>(p[0]) |
           (static_cast<u32>(p[1]) << 8) |
           (static_cast<u32>(p[2]) << 16) |
           (static_cast<u32>(p[3]) << 24);
}

inline u64 read_le64(const u8* p) {
    return static_cast<u64>(read_le32(p)) |
           (static_cast<u64>(read_le32(p + 4)) << 32);
}

inline void write_le16(u8* p, u16 v) {
    p[0] = static_cast<u8>(v & 0xFF);
    p[1] = static_cast<u8>((v >> 8) & 0xFF);
}

inline void write_le32(u8* p, u32 v) {
    p[0] = static_cast<u8>(v & 0xFF);
    p[1] = static_cast<u8>((v >> 8) & 0xFF);
    p[2] = static_cast<u8>((v >> 16) & 0xFF);
    p[3] = static_cast<u8>((v >> 24) & 0xFF);
}

inline void write_le64(u8* p, u64 v) {
    write_le32(p, static_cast<u32>(v & 0xFFFFFFFFULL));
    write_le32(p + 4, static_cast<u32>(v >> 32));
}

inline std::string hex32(u32 v) {
    std::ostringstream oss;
    oss << "0x" << std::hex << std::setw(8) << std::setfill('0') << v;
    return oss.str();
}

inline std::string hex64(u64 v) {
    std::ostringstream oss;
    oss << "0x" << std::hex << std::setw(16) << std::setfill('0') << v;
    return oss.str();
}

constexpr const char* kRecompilerVersion = "0.1.0-r5900-aot";
constexpr const char* kRuntimeVersion    = "0.1.0-ps2rt";

} // namespace tf::ps2
