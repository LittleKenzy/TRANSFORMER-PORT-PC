#pragma once

#include "../../Recompiler/Common/Types.h"
#include <array>
#include <vector>

namespace tf::ps2 {

constexpr u32 kSpu2RamSize = 2U * 1024U * 1024U; // 2 MiB SPU2 Sound RAM

struct Spu2Voice {
    u32 start_addr = 0;
    u32 current_addr = 0;
    u32 loop_addr = 0;
    u16 pitch = 0x1000; // 48kHz nominal
    s16 vol_left = 0x3FFF;
    s16 vol_right = 0x3FFF;
    s32 hist1 = 0;
    s32 hist2 = 0;
    bool key_on = false;
    std::array<s16, 28> decoded_block{};
    u32 sample_pos = 28;
};

// SPU2 Sound Processor: 2MB Sound RAM, 48 PS-ADPCM voices (2 cores x 24), stereo 48kHz mixer.
class Spu2Unit {
public:
    Spu2Unit();
    void reset();

    u8* sound_ram() { return m_ram.data(); }
    const u8* sound_ram() const { return m_ram.data(); }

    void write_ram(u32 addr, const u8* data, std::size_t len);
    void key_on(u32 voice_idx, u32 start_addr_bytes, u16 pitch = 0x1000);
    void key_off(u32 voice_idx);

    // Decodes one 16-byte PS-ADPCM block into 28 signed 16-bit PCM samples.
    static void decode_adpcm_block(const u8 block[16], s16 out_samples[28], s32& hist1, s32& hist2, u8& out_flags);

    // Mixes N stereo frames (interleaved L, R s16) at 48 kHz.
    void mix_stereo_samples(s16* out_interleaved_lr, std::size_t frame_count);

private:
    std::vector<u8> m_ram;
    std::array<Spu2Voice, 48> m_voices{};
};

} // namespace tf::ps2
