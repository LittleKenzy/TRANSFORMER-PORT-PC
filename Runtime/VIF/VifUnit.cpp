#include "VifUnit.h"
#include "../PerfCounters.h"
#include <algorithm>
#include <cstring>

namespace tf::ps2 {

VifUnit::VifUnit(int index) : m_index(index) {
    reset();
}

void VifUnit::reset() {
    cycle_cl = 1;
    cycle_wl = 1;
    offset = 0;
    base = 0;
    tops = 0;
    top = 0;
    itop = 0;
    mode = 0;
    mark = 0;
    mark_detected = false;
    mask = 0;
    std::memset(row, 0, sizeof(row));
    std::memset(col, 0, sizeof(col));
    commands_processed = 0;
    unpacks_processed = 0;
    direct_qwords_forwarded = 0;
    mpg_uploads = 0;
    mscal_count = 0;
    m_state = State::Idle;
    m_words_rem = 0;
    m_param_idx = 0;
    m_direct_buf.clear();
    m_unpack_buf.clear();
    m_mpg_dst = 0;
    m_mpg_start = 0;
    m_mpg_total_bytes = 0;
    m_mpg_modified = false;
}

void VifUnit::finish_unpack(const u8* src_bytes, std::size_t src_sz) {
    if (!m_vu_data || m_vu_data_size == 0) return;
    PerfScope perf_unpack(PerfCat::VifUnpack);
    const u32 vn = (m_unpack_cmd >> 2) & 3u;
    const u32 vl = m_unpack_cmd & 3u;
    const bool use_mask = (m_unpack_cmd & 0x10u) != 0;
    const u32 count = (m_unpack_num == 0) ? 256u : m_unpack_num;
    const u32 comps = (vl == 3) ? 1u : (vn + 1u);

    u32 dest_qword = m_unpack_imm & 0x3FFu;
    if ((m_unpack_imm & 0x8000u) != 0) dest_qword = (dest_qword + tops) & 0x3FFu;
    const bool is_unsigned = (m_unpack_imm & 0x4000u) != 0;

    const u32 cl = (cycle_cl == 0) ? 1u : cycle_cl;
    const u32 wl = (cycle_wl == 0) ? 1u : cycle_wl;

    // Fast paths for the overwhelmingly common unmasked linear uploads (cl == wl, mode == 0)
    if (!use_mask && mode == 0 && cl == wl && (m_vu_data_size & 15u) == 0) {
        const u32 max_qwords = static_cast<u32>(m_vu_data_size >> 4);
        const u32 qmask = max_qwords - 1u;
        const bool pow2 = (max_qwords > 0) && ((max_qwords & qmask) == 0);
        if (pow2) {
            if (vn == 3 && vl == 0 && src_sz >= static_cast<std::size_t>(count) * 16u) {
                // V4-32: direct contiguous memcpy (split only if wrapping 16 KiB boundary)
                const u32 start_q = dest_qword & qmask;
                const u32 first_q = std::min<u32>(count, max_qwords - start_q);
                std::memcpy(m_vu_data + start_q * 16u, src_bytes, static_cast<std::size_t>(first_q) * 16u);
                if (first_q < count) {
                    const u32 rem_q = count - first_q;
                    std::memcpy(m_vu_data, src_bytes + static_cast<std::size_t>(first_q) * 16u, static_cast<std::size_t>(rem_q) * 16u);
                }
                unpacks_processed++;
                return;
            }
            if (vn == 2 && vl == 0 && src_sz >= static_cast<std::size_t>(count) * 12u) {
                // V3-32: 12-byte copy + zero w
                u32 q = dest_qword & qmask;
                const u8* sp = src_bytes;
                for (u32 i = 0; i < count; ++i) {
                    u8* dp = m_vu_data + (q << 4);
                    std::memcpy(dp, sp, 12);
                    write_le32(dp + 12, 0u);
                    sp += 12;
                    q = (q + 1u) & qmask;
                }
                unpacks_processed++;
                return;
            }
            if (vn == 1 && vl == 0 && src_sz >= static_cast<std::size_t>(count) * 8u) {
                // V2-32: 8-byte copy + zero zw
                u32 q = dest_qword & qmask;
                const u8* sp = src_bytes;
                for (u32 i = 0; i < count; ++i) {
                    u8* dp = m_vu_data + (q << 4);
                    std::memcpy(dp, sp, 8);
                    write_le64(dp + 8, 0ULL);
                    sp += 8;
                    q = (q + 1u) & qmask;
                }
                unpacks_processed++;
                return;
            }
            if (vn == 3 && vl == 1 && src_sz >= static_cast<std::size_t>(count) * 8u) {
                // V4-16: 4x 16-bit -> 4x 32-bit
                u32 q = dest_qword & qmask;
                const u8* sp = src_bytes;
                if (is_unsigned) {
                    for (u32 i = 0; i < count; ++i) {
                        u8* dp = m_vu_data + (q << 4);
                        write_le32(dp + 0,  read_le16(sp + 0));
                        write_le32(dp + 4,  read_le16(sp + 2));
                        write_le32(dp + 8,  read_le16(sp + 4));
                        write_le32(dp + 12, read_le16(sp + 6));
                        sp += 8;
                        q = (q + 1u) & qmask;
                    }
                } else {
                    for (u32 i = 0; i < count; ++i) {
                        u8* dp = m_vu_data + (q << 4);
                        write_le32(dp + 0,  static_cast<u32>(static_cast<s32>(static_cast<s16>(read_le16(sp + 0)))));
                        write_le32(dp + 4,  static_cast<u32>(static_cast<s32>(static_cast<s16>(read_le16(sp + 2)))));
                        write_le32(dp + 8,  static_cast<u32>(static_cast<s32>(static_cast<s16>(read_le16(sp + 4)))));
                        write_le32(dp + 12, static_cast<u32>(static_cast<s32>(static_cast<s16>(read_le16(sp + 6)))));
                        sp += 8;
                        q = (q + 1u) & qmask;
                    }
                }
                unpacks_processed++;
                return;
            }
            if (vn == 3 && vl == 2 && src_sz >= static_cast<std::size_t>(count) * 4u) {
                // V4-8: 4x 8-bit -> 4x 32-bit
                u32 q = dest_qword & qmask;
                const u8* sp = src_bytes;
                if (is_unsigned) {
                    for (u32 i = 0; i < count; ++i) {
                        u8* dp = m_vu_data + (q << 4);
                        write_le32(dp + 0,  sp[0]);
                        write_le32(dp + 4,  sp[1]);
                        write_le32(dp + 8,  sp[2]);
                        write_le32(dp + 12, sp[3]);
                        sp += 4;
                        q = (q + 1u) & qmask;
                    }
                } else {
                    for (u32 i = 0; i < count; ++i) {
                        u8* dp = m_vu_data + (q << 4);
                        write_le32(dp + 0,  static_cast<u32>(static_cast<s32>(static_cast<s8>(sp[0]))));
                        write_le32(dp + 4,  static_cast<u32>(static_cast<s32>(static_cast<s8>(sp[1]))));
                        write_le32(dp + 8,  static_cast<u32>(static_cast<s32>(static_cast<s8>(sp[2]))));
                        write_le32(dp + 12, static_cast<u32>(static_cast<s32>(static_cast<s8>(sp[3]))));
                        sp += 4;
                        q = (q + 1u) & qmask;
                    }
                }
                unpacks_processed++;
                return;
            }
        }
    }

    std::size_t byte_off = 0;

    auto read_one_vector = [&](u32 out_w[4]) {
        out_w[0] = out_w[1] = out_w[2] = out_w[3] = 0;
        if (vl == 0) {
            for (u32 c = 0; c < comps && byte_off + 4 <= src_sz; ++c) {
                out_w[c] = read_le32(src_bytes + byte_off);
                byte_off += 4;
            }
        } else if (vl == 1) {
            for (u32 c = 0; c < comps && byte_off + 2 <= src_sz; ++c) {
                u16 v = read_le16(src_bytes + byte_off);
                out_w[c] = is_unsigned ? static_cast<u32>(v) : static_cast<u32>(static_cast<s32>(static_cast<s16>(v)));
                byte_off += 2;
            }
        } else if (vl == 2) {
            for (u32 c = 0; c < comps && byte_off + 1 <= src_sz; ++c) {
                u8 v = src_bytes[byte_off++];
                out_w[c] = is_unsigned ? static_cast<u32>(v) : static_cast<u32>(static_cast<s32>(static_cast<s8>(v)));
            }
        } else if (byte_off + 2 <= src_sz) {
            u16 v = read_le16(src_bytes + byte_off);
            byte_off += 2;
            out_w[0] = (v & 0x1Fu) << 3;
            out_w[1] = ((v >> 5) & 0x1Fu) << 3;
            out_w[2] = ((v >> 10) & 0x1Fu) << 3;
            out_w[3] = ((v >> 15) & 0x1u) << 7;
        }
        // S-formats (vn == 0) broadcast x to all 4 components
        if (vn == 0 && vl != 3) {
            out_w[1] = out_w[2] = out_w[3] = out_w[0];
        }
    };

    auto store_masked = [&](u32 qword_addr, u32 cycle_idx, const u32 in_w[4], bool is_fill) {
        u32 dst_off = (qword_addr * 16u) % m_vu_data_size;
        if (dst_off + 16 > m_vu_data_size) return;
        u32 n = std::min<u32>(cycle_idx, 3u);
        for (u32 c = 0; c < 4; ++c) {
            u32 m = use_mask ? ((mask >> (n * 8u + c * 2u)) & 3u) : 0u;
            u32 val = in_w[c];
            if (is_fill) {
                if (m == 0) val = row[c];
                else if (m == 1) val = row[c];
                else if (m == 2) val = col[n];
                else continue;
            } else {
                if (m == 0) {
                    if (vl != 3) {
                        if (mode == 1) val += row[c];
                        else if (mode == 2) { row[c] += val; val = row[c]; }
                    }
                } else if (m == 1) {
                    val = row[c];
                } else if (m == 2) {
                    val = col[n];
                } else {
                    continue; // write-protect
                }
            }
            write_le32(m_vu_data + dst_off + c * 4u, val);
        }
    };

    u32 cur_qword = dest_qword;
    if (wl <= cl) {
        // Skipping write: NUM input vectors; after every WL writes, skip (CL - WL) qwords
        u32 in_cycle = 0;
        for (u32 i = 0; i < count; ++i) {
            u32 out_w[4];
            read_one_vector(out_w);
            store_masked(cur_qword, in_cycle, out_w, false);
            cur_qword = (cur_qword + 1u) & 0x3FFu;
            in_cycle++;
            if (in_cycle >= wl) {
                cur_qword = (cur_qword + (cl - wl)) & 0x3FFu;
                in_cycle = 0;
            }
        }
    } else {
        // Filling write: NUM destination qwords; first CL from stream, rest filled with ROW/COL
        for (u32 i = 0; i < count; ++i) {
            u32 in_cycle = i % wl;
            u32 out_w[4] = {0, 0, 0, 0};
            bool is_fill = (in_cycle >= cl);
            if (!is_fill) read_one_vector(out_w);
            store_masked(cur_qword, in_cycle, out_w, is_fill);
            cur_qword = (cur_qword + 1u) & 0x3FFu;
        }
    }
    unpacks_processed++;
}

void VifUnit::process_packet(const u8* qwords, std::size_t qwc) {
    if (!qwords || qwc == 0) return;
    process_words(qwords, qwc * 4u);
}

void VifUnit::process_words(const u8* bytes, std::size_t total_words) {
    if (!bytes || total_words == 0) return;
    PerfScope perf_vif(PerfCat::VifDma);
    std::size_t w = 0;

    while (w < total_words) {
        if (m_state != State::Idle) {
            std::size_t avail = total_words - w;
            std::size_t take = std::min<std::size_t>(avail, m_words_rem);

            if (m_state == State::Direct) {
                const u8* p = bytes + w * 4;
                if (m_direct_buf.empty() && (take & 3u) == 0) {
                    std::size_t full_qwc = take >> 2;
                    if (m_gif_direct_cb) {
                        m_gif_direct_cb(p, full_qwc);
                    }
                    direct_qwords_forwarded += full_qwc;
                } else {
                    m_direct_buf.insert(m_direct_buf.end(), p, p + take * 4);
                    std::size_t full_qwc = m_direct_buf.size() / 16u;
                    if (full_qwc > 0) {
                        if (m_gif_direct_cb) {
                            m_gif_direct_cb(m_direct_buf.data(), full_qwc);
                        }
                        direct_qwords_forwarded += full_qwc;
                        m_direct_buf.erase(m_direct_buf.begin(), m_direct_buf.begin() + static_cast<std::ptrdiff_t>(full_qwc * 16u));
                    }
                }
            } else if (m_state == State::Mpg) {
                if (m_vu_code && m_vu_code_size > 0) {
                    for (std::size_t i = 0; i < take; ++i) {
                        const u32 dst = m_mpg_dst % m_vu_code_size;
                        if (dst + 4 <= m_vu_code_size) {
                            const u8* src = bytes + (w + i) * 4;
                            if (std::memcmp(m_vu_code + dst, src, 4) != 0) {
                                std::memcpy(m_vu_code + dst, src, 4);
                                m_mpg_modified = true;
                            }
                        }
                        m_mpg_dst += 4;
                    }
                }
            } else if (m_state == State::Unpack) {
                const u8* p = bytes + w * 4;
                if (m_unpack_buf.empty() && take == m_words_rem) {
                    finish_unpack(p, take * 4u);
                    w += take;
                    m_words_rem = 0;
                    m_state = State::Idle;
                    continue;
                }
                m_unpack_buf.insert(m_unpack_buf.end(), p, p + take * 4);
            } else if (m_state == State::StMask) {
                mask = read_le32(bytes + w * 4);
            } else if (m_state == State::StRow) {
                for (std::size_t i = 0; i < take && m_param_idx < 4; ++i) {
                    row[m_param_idx++] = read_le32(bytes + (w + i) * 4);
                }
            } else if (m_state == State::StCol) {
                for (std::size_t i = 0; i < take && m_param_idx < 4; ++i) {
                    col[m_param_idx++] = read_le32(bytes + (w + i) * 4);
                }
            }

            w += take;
            m_words_rem -= static_cast<u32>(take);
            if (m_words_rem == 0) {
                if (m_state == State::Mpg) {
                    mpg_uploads++;
                    if (m_mpg_modified && m_mpg_cb) {
                        m_mpg_cb(m_mpg_start, m_mpg_total_bytes);
                    }
                    m_mpg_modified = false;
                } else if (m_state == State::Unpack) {
                    finish_unpack(m_unpack_buf.data(), m_unpack_buf.size());
                    m_unpack_buf.clear();
                }
                m_state = State::Idle;
            }
            continue;
        }

        u32 vifcode = read_le32(bytes + w * 4);
        w++;
        commands_processed++;

        u8 cmd = static_cast<u8>((vifcode >> 24) & 0x7Fu);
        u8 num = static_cast<u8>((vifcode >> 16) & 0xFFu);
        u16 imm = static_cast<u16>(vifcode & 0xFFFFu);

        if ((cmd & 0x60u) == 0x60u) {
            u32 vn = (cmd >> 2) & 3u;
            u32 vl = cmd & 3u;
            u32 count = (num == 0) ? 256u : num;
            u32 cl = (cycle_cl == 0) ? 1u : cycle_cl;
            u32 wl = (cycle_wl == 0) ? 1u : cycle_wl;
            u32 input_vecs = (wl > cl) ? ((count / wl) * cl + std::min<u32>(count % wl, cl)) : count;
            u32 bits_per_comp = (vl == 0) ? 32u : (vl == 1) ? 16u : (vl == 2) ? 8u : 16u;
            u32 comps = (vl == 3) ? 1u : (vn + 1u);
            u32 data_words = (input_vecs * comps * bits_per_comp + 31u) / 32u;
            m_unpack_cmd = cmd;
            m_unpack_num = num;
            m_unpack_imm = imm;
            m_unpack_buf.clear();
            if (data_words > 0) {
                m_state = State::Unpack;
                m_words_rem = data_words;
            } else {
                finish_unpack(nullptr, 0);
            }
            continue;
        }

        switch (cmd) {
            case 0x00: break; // NOP
            case 0x01: cycle_cl = imm & 0xFFu; cycle_wl = (imm >> 8) & 0xFFu; break; // STCYCL
            case 0x02: offset = imm & 0x3FFu; break; // OFFSET
            case 0x03: base = imm & 0x3FFu; tops = base; break; // BASE (also resets TOPS=BASE)
            case 0x04: itop = imm & 0x3FFu; break;   // ITOP
            case 0x05: mode = imm & 3u; break;       // STMOD
            case 0x07: mark = imm; mark_detected = true; break; // MARK
            case 0x10: case 0x11: case 0x13: break;  // FLUSHE / FLUSH / FLUSHA
            case 0x14: case 0x15: // MSCAL / MSCALF
                top = tops;
                tops = (tops == base) ? ((base + offset) & 0x3FFu) : base;
                mscal_count++;
                if (m_mscal_cb) m_mscal_cb(static_cast<u32>(imm) * 8u, false);
                break;
            case 0x17: // MSCNT (continue from current VU1 PC)
                top = tops;
                tops = (tops == base) ? ((base + offset) & 0x3FFu) : base;
                mscal_count++;
                if (m_mscal_cb) m_mscal_cb(0, true);
                break;
            case 0x20: // STMASK
                m_state = State::StMask;
                m_words_rem = 1;
                break;
            case 0x30: // STROW
                m_state = State::StRow;
                m_words_rem = 4;
                m_param_idx = 0;
                break;
            case 0x31: // STCOL
                m_state = State::StCol;
                m_words_rem = 4;
                m_param_idx = 0;
                break;
            case 0x4A: { // MPG
                u32 count = (num == 0) ? 256u : num;
                m_state = State::Mpg;
                m_words_rem = count * 2u;
                m_mpg_dst = static_cast<u32>(imm) * 8u;
                m_mpg_start = m_mpg_dst;
                m_mpg_total_bytes = count * 8u;
                m_mpg_modified = false;
                break;
            }
            case 0x50: // DIRECT
            case 0x51: { // DIRECTHL
                u32 q = (imm == 0) ? 65536u : imm;
                m_state = State::Direct;
                m_words_rem = q * 4u;
                m_direct_buf.clear();
                break;
            }
            default:
                break;
        }
    }
}

u32 VifUnit::read_reg(u32 reg_offset) const {
    switch (reg_offset & 0xFFu) {
        case 0x00: { // VIFn_STAT
            u32 stat = 0;
            if (mark_detected) stat |= (1u << 6); // MRK
            if (m_index == 1 && tops != base) stat |= (1u << 7); // DBF
            return stat;
        }
        case 0x30: return mark; // VIFn_MARK
        case 0x40: return (cycle_cl & 0xFFu) | ((cycle_wl & 0xFFu) << 8); // VIFn_CYCLE
        case 0x50: return mode; // VIFn_MODE
        case 0x80: return mask; // VIFn_MASK
        case 0xA0: return itop; // VIFn_ITOPS
        case 0xB0: return base; // VIF1_BASE
        case 0xC0: return offset; // VIF1_OFST
        case 0xD0: return tops; // VIF1_TOPS
        case 0xE0: return itop; // VIFn_ITOP
        case 0xF0: return top;  // VIF1_TOP
        default:   return 0;
    }
}

void VifUnit::write_reg(u32 reg_offset, u32 val) {
    switch (reg_offset & 0xFFu) {
        case 0x10: // VIFn_FBRST
            if ((val & 1u) != 0) reset();
            if ((val & 8u) != 0) mark_detected = false; // STC: clear stall/mark flags
            break;
        case 0x30: // VIFn_MARK
            mark = val;
            mark_detected = false;
            break;
        default:
            break;
    }
}

} // namespace tf::ps2
