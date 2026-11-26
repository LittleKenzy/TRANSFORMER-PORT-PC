#pragma once

#include "Types.h"
#include <array>
#include <string>

namespace tf::ps2 {

class Sha256 {
public:
    Sha256();
    void update(const u8* data, std::size_t len);
    void update(std::string_view text);
    std::array<u8, 32> finalize();

    static std::string to_hex(const std::array<u8, 32>& digest);
    static std::string hash_bytes(const u8* data, std::size_t len);
    static std::string hash_string(std::string_view text);

private:
    void transform(const u8 block[64]);

    u32 m_state[8];
    u64 m_bit_count = 0;
    u8  m_buffer[64];
    std::size_t m_buffer_len = 0;
    bool m_finalized = false;
};

} // namespace tf::ps2
