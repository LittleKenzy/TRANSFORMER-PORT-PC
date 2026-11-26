#include "DmaController.h"
#include "../PerfCounters.h"
#include <algorithm>
#include <cstring>
#include <vector>

namespace tf::ps2 {

DmaController::DmaController() {
    reset();
}

void DmaController::reset() {
    m_channels.fill(DmaChannelState{});
    m_ctrl = 0;
    m_stat = 0;
    m_pcr = 0;
    m_sqwc = 0;
    m_enable_hold = 0;
    m_pumping = false;
}

u32 DmaController::resolve_dma_addr(u32 addr) {
    if ((addr & 0x80000000u) != 0) {
        return 0x70000000u + (addr & 0x3FF0u);
    }
    return addr & 0x7FFFFFF0u;
}

int DmaController::channel_from_addr(u32 paddr) {
    u32 base = paddr & 0xFFFFFC00u;
    switch (base) {
        case 0x10008000u: return static_cast<int>(DmaChannelId::VIF0);
        case 0x10009000u: return static_cast<int>(DmaChannelId::VIF1);
        case 0x1000A000u: return static_cast<int>(DmaChannelId::GIF);
        case 0x1000B000u: return static_cast<int>(DmaChannelId::FromIPU);
        case 0x1000B400u: return static_cast<int>(DmaChannelId::ToIPU);
        case 0x1000C000u: return static_cast<int>(DmaChannelId::SIF0);
        case 0x1000C400u: return static_cast<int>(DmaChannelId::SIF1);
        case 0x1000C800u: return static_cast<int>(DmaChannelId::SIF2);
        case 0x1000D000u: return static_cast<int>(DmaChannelId::FromSPR);
        case 0x1000D400u: return static_cast<int>(DmaChannelId::ToSPR);
        default: return -1;
    }
}

u32 DmaController::read_reg(u32 paddr) const {
    if (paddr == 0x1000E000u) return m_ctrl;
    if (paddr == 0x1000E010u) return m_stat;
    if (paddr == 0x1000E020u) return m_pcr;
    if (paddr == 0x1000E030u) return m_sqwc;
    if (paddr == 0x1000E040u) return m_sqwc;
    if (paddr == 0x1000F520u) return m_enable_hold; // D_ENABLER

    int ch = channel_from_addr(paddr);
    if (ch < 0) return 0;
    const auto& c = m_channels[static_cast<std::size_t>(ch)];
    switch (paddr & 0xFFu) {
        case 0x00: return c.chcr;
        case 0x10: return c.madr;
        case 0x20: return c.qwc;
        case 0x30: return c.tadr;
        case 0x40: return c.asr0;
        case 0x50: return c.asr1;
        case 0x80: return c.sadr;
        default:   return 0;
    }
}

void DmaController::pump_pending_channels(GuestMemory& mem) {
    if (m_pumping || (m_enable_hold & 0x10000u) != 0) return;
    PerfScope perf_dma(PerfCat::VifDma);
    struct PumpGuard {
        bool& flag;
        explicit PumpGuard(bool& f) : flag(f) { flag = true; }
        ~PumpGuard() { flag = false; }
    } guard{m_pumping};
    u32 rounds = 0;
    u32 same_state_streak = 0;
    u64 last_sig = ~0ULL;
    while (rounds++ < 1048576u && (m_enable_hold & 0x10000u) == 0) {
        bool any = false;
        for (std::size_t i = 0; i < m_channels.size(); ++i) {
            if ((m_channels[i].chcr & 0x100u) != 0) {
                any = true;
                u64 sig = (static_cast<u64>(i) << 56) ^
                          (static_cast<u64>(m_channels[i].tadr) << 24) ^
                          (static_cast<u64>(m_channels[i].madr) << 4) ^
                          static_cast<u64>(m_channels[i].qwc);
                if (sig == last_sig) {
                    if (++same_state_streak > 1024u) {
                        return;
                    }
                } else {
                    last_sig = sig;
                    same_state_streak = 0;
                }
                execute_channel(static_cast<DmaChannelId>(i), mem);
                if ((m_enable_hold & 0x10000u) != 0) break;
            }
        }
        if (!any) break;
    }
}

void DmaController::write_reg(u32 paddr, u32 val, GuestMemory& mem) {
    if (paddr == 0x1000E000u) { m_ctrl = val; return; }
    if (paddr == 0x1000E010u) {
        // Low 16 bits clear status flags on write-1; high 16 bits toggle masks on write-1
        m_stat &= ~(val & 0xFFFFu);
        m_stat ^= (val & 0xFFFF0000u);
        return;
    }
    if (paddr == 0x1000E020u) { m_pcr = val; return; }
    if (paddr == 0x1000E030u || paddr == 0x1000E040u) { m_sqwc = val; return; }
    if (paddr == 0x1000F590u) { // D_ENABLEW (bit 16 = CPND/hold DMA)
        m_enable_hold = val & 0x10000u;
        if ((m_enable_hold & 0x10000u) == 0) {
            pump_pending_channels(mem);
        }
        return;
    }

    int ch = channel_from_addr(paddr);
    if (ch < 0) return;
    auto& c = m_channels[static_cast<std::size_t>(ch)];
    switch (paddr & 0xFFu) {
        case 0x00:
            c.chcr = val;
            if ((val & 0x100u) != 0) { // STR bit = start transfer
                pump_pending_channels(mem);
            }
            break;
        case 0x10: c.madr = val; break;
        case 0x20: c.qwc  = val & 0xFFFFu; break;
        case 0x30: c.tadr = val; break;
        case 0x40: c.asr0 = val; break;
        case 0x50: c.asr1 = val; break;
        case 0x80: c.sadr = val; break;
        default: break;
    }
}

void DmaController::transfer_spr_qwords(DmaChannelId ch, GuestMemory& mem, u32 ram_addr, u32& sadr, u32 qwc) {
    if (qwc == 0) return;
    auto& c = m_channels[static_cast<std::size_t>(ch)];
    const u32 total_bytes = qwc * 16u;
    const u32 ram_base = ram_addr & 0x7FFFFFF0u;
    u32 spr_off = sadr & 0x3FF0u;
    u32 done = 0;
    while (done < total_bytes) {
        const u32 chunk = std::min<u32>(total_bytes - done, kEeScratchpadSize - spr_off);
        u8* tmp = m_spr_scratch.data();
        if (ch == DmaChannelId::ToSPR) {
            if (mem.read_block(ram_base + done, tmp, chunk)) {
                mem.write_block(0x70000000u + spr_off, tmp, chunk);
            }
        } else {
            if (mem.read_block(0x70000000u + spr_off, tmp, chunk)) {
                mem.write_block(ram_base + done, tmp, chunk);
            }
        }
        done += chunk;
        spr_off = (spr_off + chunk) & (kEeScratchpadSize - 1u);
    }
    sadr = spr_off;
    c.quadwords_transferred += qwc;
}

void DmaController::deliver_qwords(DmaChannelId ch, GuestMemory& mem, u32 addr, u32 qwc) {
    if (qwc == 0) return;
    auto& c = m_channels[static_cast<std::size_t>(ch)];
    if (ch == DmaChannelId::ToSPR || ch == DmaChannelId::FromSPR) {
        transfer_spr_qwords(ch, mem, addr, c.sadr, qwc);
        return;
    }
    auto& sink = m_sinks[static_cast<std::size_t>(ch)];
    const u32 byte_len = qwc * 16u;
    const u32 resolved = resolve_dma_addr(addr);
    const u8* ptr = mem.translate_ptr(resolved, byte_len);
    if (ptr && sink) {
        sink(ptr, qwc);
    } else if (sink && resolved >= 0x70000000u && resolved < 0x70000000u + kEeScratchpadSize) {
        // Scratchpad transfer wrapping across 16 KiB boundary
        std::vector<u8> heap_tmp;
        u8* buf = m_spr_scratch.data();
        if (byte_len > kEeScratchpadSize) {
            heap_tmp.resize(byte_len);
            buf = heap_tmp.data();
        }
        u32 spr_off = resolved - 0x70000000u;
        u32 done = 0;
        while (done < byte_len) {
            const u32 chunk = std::min<u32>(byte_len - done, kEeScratchpadSize - spr_off);
            mem.read_block(0x70000000u + spr_off, buf + done, chunk);
            done += chunk;
            spr_off = (spr_off + chunk) & (kEeScratchpadSize - 1u);
        }
        sink(buf, qwc);
    }
    c.quadwords_transferred += qwc;
}

bool DmaController::execute_channel(DmaChannelId ch, GuestMemory& mem) {
    auto& c = m_channels[static_cast<std::size_t>(ch)];
    u32 mode = (c.chcr >> 2) & 3u; // 0=Normal, 1=Chain, 2=Interleave
    bool tte = (c.chcr & (1u << 6)) != 0; // Tag Transfer Enable

    if (mode == 0) {
        // Normal mode
        deliver_qwords(ch, mem, c.madr, c.qwc);
        c.madr += c.qwc * 16u;
        c.qwc = 0;
    } else if (mode == 2) {
        // Interleave mode (D_SQWC: bits 0..7 = SQWC skip, bits 16..23 = TQWC transfer)
        const u32 sqwc = m_sqwc & 0xFFu;
        const u32 tqwc = (m_sqwc >> 16) & 0xFFu;
        if (tqwc == 0) {
            deliver_qwords(ch, mem, c.madr, c.qwc);
            c.madr += c.qwc * 16u;
            c.qwc = 0;
        } else {
            while (c.qwc > 0) {
                const u32 slice = std::min<u32>(c.qwc, tqwc);
                deliver_qwords(ch, mem, c.madr, slice);
                c.madr += slice * 16u;
                c.qwc -= slice;
                if (c.qwc > 0) {
                    c.madr += sqwc * 16u;
                }
            }
        }
    } else if (mode == 1) {
        bool tie = (c.chcr & (1u << 7)) != 0;

        // Honor non-zero initial QWC before walking chain tags (e.g. resuming after TIE/IRQ stop)
        if (c.qwc > 0) {
            deliver_qwords(ch, mem, c.madr, c.qwc);
            c.madr += c.qwc * 16u;
            c.qwc = 0;
        }

        if (ch == DmaChannelId::FromSPR) {
            // Destination Chain mode: tags live in Scratchpad at SADR, data follows tag -> RAM at tag.ADDR
            u32 safety = 0;
            u32 probe_sadr = 0xFFFFFFFFu;
            u32 next_power = 2048;
            while (safety++ < 262144u) {
                const u32 spr_tag_off = c.sadr & 0x3FF0u;
                if (safety == next_power) {
                    probe_sadr = spr_tag_off;
                    next_power <<= 1;
                } else if (safety > 2048 && spr_tag_off == probe_sadr) {
                    break;
                }

                const u128 tag = mem.read128(0x70000000u + spr_tag_off);
                const u32 tag_qwc = static_cast<u32>(tag.lo & 0xFFFFu);
                const u32 tag_id  = static_cast<u32>((tag.lo >> 28) & 0x7u);
                const bool tag_irq = ((tag.lo >> 31) & 1u) != 0;
                const u32 dst_addr = static_cast<u32>((tag.lo >> 32) & 0x7FFFFFF0u);

                c.chcr = (c.chcr & 0x0000FFFFu) | (static_cast<u32>(tag.lo) & 0xFFFF0000u);
                c.sadr = (spr_tag_off + 16u) & 0x3FF0u;
                c.madr = dst_addr;
                if (tag_qwc > 0) {
                    transfer_spr_qwords(DmaChannelId::FromSPR, mem, c.madr, c.sadr, tag_qwc);
                    c.madr += tag_qwc * 16u;
                }

                bool end = (tag_id == 7u); // END
                if (tie && tag_irq) end = true;
                if (end) break;
            }
            c.qwc = 0;
        } else {
            // Source Chain mode (standard PS2 DMAtag walker with CHCR.TAG mirror, MADR updates, and TIE/IRQ stop)
            int asp = static_cast<int>((c.chcr >> 4) & 3u);
            u32 safety = 0;
            u32 cycle_probe_tadr = 0;
            int cycle_probe_asp = -1;
            u32 cycle_probe_asr0 = 0;
            u32 cycle_probe_asr1 = 0;
            u32 next_power = 2048;
            while (safety++ < 262144u) {
                u32 tag_addr = resolve_dma_addr(c.tadr);

                if (safety == next_power) {
                    cycle_probe_tadr = tag_addr;
                    cycle_probe_asp = asp;
                    cycle_probe_asr0 = c.asr0;
                    cycle_probe_asr1 = c.asr1;
                    next_power <<= 1;
                } else if (safety > 2048 && tag_addr == cycle_probe_tadr && asp == cycle_probe_asp &&
                           c.asr0 == cycle_probe_asr0 && c.asr1 == cycle_probe_asr1) {
                    break; // Exact DMA chain state cycle detected; terminate safely
                }

                u128 tag = mem.read128(tag_addr);
                u32 tag_qwc = static_cast<u32>(tag.lo & 0xFFFFu);
                u32 tag_id  = static_cast<u32>((tag.lo >> 28) & 0x7u);
                bool tag_irq = ((tag.lo >> 31) & 1u) != 0;
                u32 next_addr = static_cast<u32>((tag.lo >> 32) & 0x7FFFFFF0u);
                bool next_spr = ((tag.lo >> 63) & 1u) != 0;
                if (next_spr) next_addr = 0x70000000u + (next_addr & 0x3FF0u);

                // Mirror upper 16 bits of DMAtag[31:16] into CHCR[31:16] (ID, IRQ, PCE, user tag bits)
                c.chcr = (c.chcr & 0x0000FFFFu) | (static_cast<u32>(tag.lo) & 0xFFFF0000u);

                if (tte) {
                    auto& tte_sink = m_tte_sinks[static_cast<std::size_t>(ch)];
                    if (tte_sink) {
                        tte_sink(tag.hi);
                    } else if (m_sinks[static_cast<std::size_t>(ch)]) {
                        u8 upper_tag[16] = {};
                        write_le64(upper_tag, tag.hi);
                        m_sinks[static_cast<std::size_t>(ch)](upper_tag, 1);
                    }
                }

                bool end = false;
                switch (tag_id) {
                    case 0: // REFE
                        c.madr = next_addr;
                        deliver_qwords(ch, mem, next_addr, tag_qwc);
                        c.madr += tag_qwc * 16u;
                        c.tadr = tag_addr + 16;
                        end = true;
                        break;
                    case 1: // CNT
                        c.madr = tag_addr + 16;
                        deliver_qwords(ch, mem, tag_addr + 16, tag_qwc);
                        c.madr += tag_qwc * 16u;
                        c.tadr = tag_addr + 16 + tag_qwc * 16;
                        break;
                    case 2: // NEXT
                        c.madr = tag_addr + 16;
                        deliver_qwords(ch, mem, tag_addr + 16, tag_qwc);
                        c.madr += tag_qwc * 16u;
                        c.tadr = next_addr;
                        break;
                    case 3: // REF
                    case 4: // REFS
                        c.madr = next_addr;
                        deliver_qwords(ch, mem, next_addr, tag_qwc);
                        c.madr += tag_qwc * 16u;
                        c.tadr = tag_addr + 16;
                        break;
                    case 5: // CALL
                        c.madr = tag_addr + 16;
                        deliver_qwords(ch, mem, tag_addr + 16, tag_qwc);
                        c.madr += tag_qwc * 16u;
                        if (asp == 0) c.asr0 = tag_addr + 16 + tag_qwc * 16;
                        else if (asp == 1) c.asr1 = tag_addr + 16 + tag_qwc * 16;
                        if (asp < 2) asp++;
                        c.tadr = next_addr;
                        break;
                    case 6: // RET
                        c.madr = tag_addr + 16;
                        deliver_qwords(ch, mem, tag_addr + 16, tag_qwc);
                        c.madr += tag_qwc * 16u;
                        if (asp == 2) { asp--; c.tadr = c.asr1; }
                        else if (asp == 1) { asp--; c.tadr = c.asr0; }
                        else { end = true; }
                        break;
                    case 7: // END
                        c.madr = tag_addr + 16;
                        deliver_qwords(ch, mem, tag_addr + 16, tag_qwc);
                        c.madr += tag_qwc * 16u;
                        c.tadr = tag_addr + 16 + tag_qwc * 16;
                        end = true;
                        break;
                }
                // Update ASP bits in CHCR[5:4]
                c.chcr = (c.chcr & ~0x30u) | (static_cast<u32>(asp & 3) << 4);
                if (tie && tag_irq) {
                    end = true;
                }
                if (end) break;
            }
            c.qwc = 0;
        }
    }

    c.chcr &= ~0x100u; // Clear STR
    c.transfers_completed++;
    m_stat |= (1u << static_cast<u32>(ch));
    if (m_completion_cb) {
        m_completion_cb(static_cast<u32>(ch));
    }
    return true;
}

} // namespace tf::ps2
