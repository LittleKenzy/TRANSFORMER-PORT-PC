#include "Spu2.h"
#include <algorithm>
#include <cstring>

namespace tf::ps2 {

namespace {

constexpr s32 kAdpcmF0[5] = { 0,  60, 115,  98, 122 };
constexpr s32 kAdpcmF1[5] = { 0,   0, -52, -55, -60 };

} // namespace

Spu2Unit::Spu2Unit() : m_ram(kSpu2RamSize, 0) {
    reset();
}

void Spu2Unit::reset() {
    std::fill(m_ram.begin(), m_ram.end(), u8{0});
    m_voices.fill(Spu2Voice{});
}

void Spu2Unit::write_ram(u32 addr, const u8* data, std::size_t len) {
    if (!data || len == 0) return;
    u32 offset = addr % kSpu2RamSize;
    std::size_t avail = std::min<std::size_t>(len, kSpu2RamSize - offset);
    std::memcpy(m_ram.data() + offset, data, avail);
}

void Spu2Unit::key_on(u32 voice_idx, u32 start_addr_bytes, u16 pitch) {
    if (voice_idx >= m_voices.size()) return;
    auto& v = m_voices[voice_idx];
    v.start_addr = start_addr_bytes & (kSpu2RamSize - 16u);
    v.current_addr = v.start_addr;
    v.loop_addr = v.start_addr;
    v.pitch = pitch;
    v.hist1 = 0;
    v.hist2 = 0;
    v.sample_pos = 28;
    v.key_on = true;
}

void Spu2Unit::key_off(u32 voice_idx) {
    if (voice_idx >= m_voices.size()) return;
    m_voices[voice_idx].key_on = false;
}

void Spu2Unit::decode_adpcm_block(const u8 block[16], s16 out_samples[28], s32& hist1, s32& hist2, u8& out_flags) {
    u8 shift = block[0] & 0x0Fu;
    if (shift > 12) shift = 9;
    u8 filter = (block[0] >> 4) & 0x07u;
    if (filter > 4) filter = 0;
    out_flags = block[1];

    s32 f0 = kAdpcmF0[filter];
    s32 f1 = kAdpcmF1[filter];

    for (int i = 0; i < 14; ++i) {
        u8 byte = block[2 + i];
        s32 nibbles[2] = {
            static_cast<s32>(static_cast<s16>((byte & 0x0Fu) << 12)),
            static_cast<s32>(static_cast<s16>((byte & 0xF0u) << 8))
        };
        for (int n = 0; n < 2; ++n) {
            s32 sample = (nibbles[n] >> shift) + ((hist1 * f0 + hist2 * f1 + 32) >> 6);
            sample = std::clamp(sample, -32768, 32767);
            out_samples[i * 2 + n] = static_cast<s16>(sample);
            hist2 = hist1;
            hist1 = sample;
        }
    }
}

void Spu2Unit::mix_stereo_samples(s16* out_interleaved_lr, std::size_t frame_count) {
    if (!out_interleaved_lr || frame_count == 0) return;

    for (std::size_t f = 0; f < frame_count; ++f) {
        s32 mix_l = 0;
        s32 mix_r = 0;

        for (auto& v : m_voices) {
            if (!v.key_on) continue;
            if (v.sample_pos >= 28) {
                u8 flags = 0;
                u32 addr = v.current_addr & (kSpu2RamSize - 16u);
                decode_adpcm_block(m_ram.data() + addr, v.decoded_block.data(), v.hist1, v.hist2, flags);
                if (flags & 0x04u) v.loop_addr = addr;
                v.current_addr = (addr + 16u) & (kSpu2RamSize - 16u);
                if (flags & 0x01u) { // Loop end
                    if (flags & 0x02u) {
                        v.current_addr = v.loop_addr;
                    } else {
                        v.key_on = false;
                    }
                }
                v.sample_pos = 0;
            }
            s32 s = v.decoded_block[v.sample_pos++];
            mix_l += (s * v.vol_left) >> 15;
            mix_r += (s * v.vol_right) >> 15;
        }

        out_interleaved_lr[f * 2 + 0] = static_cast<s16>(std::clamp(mix_l, -32768, 32767));
        out_interleaved_lr[f * 2 + 1] = static_cast<s16>(std::clamp(mix_r, -32768, 32767));
    }
}

} // namespace tf::ps2
