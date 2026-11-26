#pragma once

#include "../Memory/GuestMemory.h"
#include <array>
#include <functional>

namespace tf::ps2 {

enum class DmaChannelId : u8 {
    VIF0 = 0,
    VIF1 = 1,
    GIF  = 2,
    FromIPU = 3,
    ToIPU   = 4,
    SIF0 = 5,
    SIF1 = 6,
    SIF2 = 7,
    FromSPR = 8,
    ToSPR   = 9,
    Count   = 10
};

struct DmaChannelState {
    u32 chcr = 0; // Channel Control
    u32 madr = 0; // Memory Address
    u32 qwc  = 0; // Quadword Count
    u32 tadr = 0; // Tag Address
    u32 asr0 = 0; // Address Stack 0
    u32 asr1 = 0; // Address Stack 1
    u32 sadr = 0; // Scratchpad Address
    u64 transfers_completed = 0;
    u64 quadwords_transferred = 0;
};

using DmaSinkCallback = std::function<void(const u8* qwords, std::size_t qwc)>;
using DmaTteCallback  = std::function<void(u64 tag_hi)>;

class DmaController {
public:
    DmaController();
    void reset();

    void set_channel_sink(DmaChannelId ch, DmaSinkCallback cb) {
        m_sinks[static_cast<std::size_t>(ch)] = std::move(cb);
    }

    void set_channel_tte_sink(DmaChannelId ch, DmaTteCallback cb) {
        m_tte_sinks[static_cast<std::size_t>(ch)] = std::move(cb);
    }

    void set_completion_callback(std::function<void(u32 channel)> cb) {
        m_completion_cb = std::move(cb);
    }

    // MMIO register read/write (0x10008000 .. 0x1000EFFF)
    u32 read_reg(u32 paddr) const;
    void write_reg(u32 paddr, u32 val, GuestMemory& mem);

    // Executes a channel transfer directly (Normal or Chain mode)
    bool execute_channel(DmaChannelId ch, GuestMemory& mem);

    const DmaChannelState& channel(DmaChannelId ch) const {
        return m_channels[static_cast<std::size_t>(ch)];
    }

private:
    static int channel_from_addr(u32 paddr);
    static u32 resolve_dma_addr(u32 addr);
    void deliver_qwords(DmaChannelId ch, GuestMemory& mem, u32 addr, u32 qwc);
    void transfer_spr_qwords(DmaChannelId ch, GuestMemory& mem, u32 ram_addr, u32& sadr, u32 qwc);

    std::array<DmaChannelState, static_cast<std::size_t>(DmaChannelId::Count)> m_channels{};
    std::array<DmaSinkCallback, static_cast<std::size_t>(DmaChannelId::Count)> m_sinks{};
    std::array<DmaTteCallback, static_cast<std::size_t>(DmaChannelId::Count)> m_tte_sinks{};
    std::function<void(u32)> m_completion_cb;
    u32 m_ctrl = 0;
    u32 m_stat = 0;
    u32 m_pcr  = 0;
    u32 m_sqwc = 0;
    u32 m_enable_hold = 0;
    bool m_pumping = false;
    std::array<u8, kEeScratchpadSize> m_spr_scratch{};
    void pump_pending_channels(GuestMemory& mem);
};

} // namespace tf::ps2
