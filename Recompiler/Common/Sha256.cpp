#include "Sha256.h"
#include <cstring>
#include <iomanip>
#include <sstream>

namespace tf::ps2 {

namespace {

constexpr u32 kRoundConstants[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
};

inline u32 rotr(u32 x, u32 n) { return (x >> n) | (x << (32 - n)); }

} // namespace

Sha256::Sha256() {
    m_state[0] = 0x6a09e667u;
    m_state[1] = 0xbb67ae85u;
    m_state[2] = 0x3c6ef372u;
    m_state[3] = 0xa54ff53au;
    m_state[4] = 0x510e527fu;
    m_state[5] = 0x9b05688cu;
    m_state[6] = 0x1f83d9abu;
    m_state[7] = 0x5be0cd19u;
    std::memset(m_buffer, 0, sizeof(m_buffer));
}

void Sha256::transform(const u8 block[64]) {
    u32 w[64];
    for (int i = 0; i < 16; ++i) {
        w[i] = (static_cast<u32>(block[i * 4]) << 24) |
               (static_cast<u32>(block[i * 4 + 1]) << 16) |
               (static_cast<u32>(block[i * 4 + 2]) << 8) |
               (static_cast<u32>(block[i * 4 + 3]));
    }
    for (int i = 16; i < 64; ++i) {
        u32 s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        u32 s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    u32 a = m_state[0], b = m_state[1], c = m_state[2], d = m_state[3];
    u32 e = m_state[4], f = m_state[5], g = m_state[6], h = m_state[7];

    for (int i = 0; i < 64; ++i) {
        u32 S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        u32 ch = (e & f) ^ ((~e) & g);
        u32 temp1 = h + S1 + ch + kRoundConstants[i] + w[i];
        u32 S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        u32 maj = (a & b) ^ (a & c) ^ (b & c);
        u32 temp2 = S0 + maj;

        h = g; g = f; f = e; e = d + temp1;
        d = c; c = b; b = a; a = temp1 + temp2;
    }

    m_state[0] += a; m_state[1] += b; m_state[2] += c; m_state[3] += d;
    m_state[4] += e; m_state[5] += f; m_state[6] += g; m_state[7] += h;
}

void Sha256::update(const u8* data, std::size_t len) {
    if (m_finalized || len == 0) return;
    m_bit_count += static_cast<u64>(len) * 8ULL;

    std::size_t offset = 0;
    if (m_buffer_len > 0) {
        std::size_t take = std::min(len, 64 - m_buffer_len);
        std::memcpy(m_buffer + m_buffer_len, data, take);
        m_buffer_len += take;
        offset += take;
        if (m_buffer_len == 64) {
            transform(m_buffer);
            m_buffer_len = 0;
        }
    }

    while (offset + 64 <= len) {
        transform(data + offset);
        offset += 64;
    }

    if (offset < len) {
        m_buffer_len = len - offset;
        std::memcpy(m_buffer, data + offset, m_buffer_len);
    }
}

void Sha256::update(std::string_view text) {
    update(reinterpret_cast<const u8*>(text.data()), text.size());
}

std::array<u8, 32> Sha256::finalize() {
    if (!m_finalized) {
        m_buffer[m_buffer_len++] = 0x80;
        if (m_buffer_len > 56) {
            std::memset(m_buffer + m_buffer_len, 0, 64 - m_buffer_len);
            transform(m_buffer);
            m_buffer_len = 0;
        }
        std::memset(m_buffer + m_buffer_len, 0, 56 - m_buffer_len);
        for (int i = 0; i < 8; ++i) {
            m_buffer[56 + i] = static_cast<u8>((m_bit_count >> ((7 - i) * 8)) & 0xFFU);
        }
        transform(m_buffer);
        m_finalized = true;
    }

    std::array<u8, 32> out{};
    for (int i = 0; i < 8; ++i) {
        out[i * 4 + 0] = static_cast<u8>((m_state[i] >> 24) & 0xFFU);
        out[i * 4 + 1] = static_cast<u8>((m_state[i] >> 16) & 0xFFU);
        out[i * 4 + 2] = static_cast<u8>((m_state[i] >> 8) & 0xFFU);
        out[i * 4 + 3] = static_cast<u8>(m_state[i] & 0xFFU);
    }
    return out;
}

std::string Sha256::to_hex(const std::array<u8, 32>& digest) {
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for (u8 b : digest) {
        oss << std::setw(2) << static_cast<unsigned>(b);
    }
    return oss.str();
}

std::string Sha256::hash_bytes(const u8* data, std::size_t len) {
    Sha256 s;
    s.update(data, len);
    return to_hex(s.finalize());
}

std::string Sha256::hash_string(std::string_view text) {
    return hash_bytes(reinterpret_cast<const u8*>(text.data()), text.size());
}

} // namespace tf::ps2
