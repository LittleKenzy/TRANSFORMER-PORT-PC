#include "GifUnit.h"
#include "../PerfCounters.h"

namespace tf::ps2 {

void GifUnit::reset() {
    tags_processed = 0;
    registers_written = 0;
    for (auto& p : m_paths) {
        p = PathState{};
    }
}

void GifUnit::process_qwords(const u8* qwords, std::size_t qwc, bool stop_on_eop) {
    if (!m_gs) return;
    feed_path(kPath3, qwords, qwc, *m_gs, stop_on_eop);
}

void GifUnit::feed_path(int path, const u8* data, std::size_t qwords, GsState& gs) {
    // PATH1 (VU1 XGKICK) always terminates on GIFtag EOP=1; PATH2/PATH3 consume full transfer QWC.
    feed_path(path, data, qwords, gs, path == kPath1);
}

void GifUnit::feed_path(int path, const u8* qwords, std::size_t qwc, GsState& gs, bool stop_on_eop) {
    if (!qwords || qwc == 0) return;
    PerfScope perf_gif(PerfCat::GifParse);
    const int idx = (path >= 0 && path < 3) ? path : kPath3;
    PathState& st = m_paths[static_cast<std::size_t>(idx)];

    // Each PATH1 XGKICK begins at a fresh GIFtag in VU1 data memory.
    if (idx == kPath1 && stop_on_eop) {
        st = PathState{};
    }

    std::size_t q = 0;
    while (q < qwc) {
        if (st.nloop_rem == 0 && st.reglist_rem == 0) {
            const u64 lo = read_le64(qwords + q * 16);
            const u64 hi = read_le64(qwords + q * 16 + 8);
            q++;
            tags_processed++;

            const u32 nloop = static_cast<u32>(lo & 0x7FFFu);
            st.eop          = ((lo >> 15) & 1u) != 0;
            const bool pre  = ((lo >> 46) & 1u) != 0;
            const u32 prim  = static_cast<u32>((lo >> 47) & 0x7FFu);
            const u32 flg   = static_cast<u32>((lo >> 58) & 0x3u); // 0=PACKED, 1=REGLIST, 2/3=IMAGE
            u32 nreg        = static_cast<u32>((lo >> 60) & 0xFu);
            if (nreg == 0) nreg = 16;

            // PS2 EE Manual §7.2.2: PRE applies in PACKED (flg==0) and REGLIST (flg==1); ignored in IMAGE (flg>=2).
            if (pre && flg <= 1) {
                gs.write_internal_reg(0x00, prim);
                registers_written++;
            }

            if (nloop == 0) {
                if (stop_on_eop && st.eop) {
                    st = PathState{};
                    return;
                }
                continue;
            }

            st.flg = flg;
            st.nreg = nreg;
            st.regs = hi;
            st.cur_reg = 0;
            st.packed_q = 0x3F800000u; // PS2 GIFtag resets internal Q to 1.0f
            if (flg == 1) {
                st.reglist_rem = nloop * nreg;
                st.nloop_rem = 0;
            } else {
                st.nloop_rem = nloop;
                st.reglist_rem = 0;
            }
            continue;
        }

        if (st.flg == 0) {
            const u8 reg_desc = static_cast<u8>((st.regs >> (st.cur_reg * 4)) & 0xFu);
            const u64 dlo = read_le64(qwords + q * 16);
            const u64 dhi = read_le64(qwords + q * 16 + 8);
            q++;

            switch (reg_desc) {
                case 0x00: // PRIM
                    gs.write_internal_reg(0x00, dlo & 0x7FFu);
                    registers_written++;
                    break;
                case 0x01: { // RGBAQ
                    const u64 rgbaq = (dlo & 0xFFu) |
                                      (((dlo >> 32) & 0xFFu) << 8) |
                                      ((dhi & 0xFFu) << 16) |
                                      (((dhi >> 32) & 0xFFu) << 24) |
                                      (static_cast<u64>(st.packed_q) << 32);
                    gs.write_internal_reg(0x01, rgbaq);
                    registers_written++;
                    break;
                }
                case 0x02: // ST (updates GIF internal Q for subsequent RGBAQ)
                    st.packed_q = static_cast<u32>(dhi & 0xFFFFFFFFu);
                    gs.set_packed_q(st.packed_q);
                    gs.write_internal_reg(0x02, dlo);
                    registers_written++;
                    break;
                case 0x03: { // UV
                    const u64 uv = (dlo & 0x3FFFu) | (((dlo >> 32) & 0x3FFFu) << 16);
                    gs.write_internal_reg(0x03, uv);
                    registers_written++;
                    break;
                }
                case 0x04: { // XYZF2 (with ADC -> XYZF3)
                    const u64 x = dlo & 0xFFFFu;
                    const u64 y = (dlo >> 32) & 0xFFFFu;
                    const u64 z = (dhi >> 4) & 0xFFFFFFu;
                    const u64 f = (dhi >> 36) & 0xFFu;
                    const bool adc = ((dhi >> 47) & 1u) != 0;
                    gs.write_internal_reg(adc ? 0x0C : 0x04, x | (y << 16) | (z << 32) | (f << 56));
                    registers_written++;
                    break;
                }
                case 0x05: { // XYZ2 (with ADC -> XYZ3)
                    const u64 x = dlo & 0xFFFFu;
                    const u64 y = (dlo >> 32) & 0xFFFFu;
                    const u64 z = dhi & 0xFFFFFFFFu;
                    const bool adc = ((dhi >> 47) & 1u) != 0;
                    gs.write_internal_reg(adc ? 0x0D : 0x05, x | (y << 16) | (z << 32));
                    registers_written++;
                    break;
                }
                case 0x06: // TEX0_1
                case 0x07: // TEX0_2
                case 0x08: // CLAMP_1
                case 0x09: // CLAMP_2
                    gs.write_internal_reg(reg_desc, dlo);
                    registers_written++;
                    break;
                case 0x0A: // FOG
                    gs.write_internal_reg(0x0A, ((dhi >> 36) & 0xFFu) << 56);
                    registers_written++;
                    break;
                case 0x0C: { // XYZF3 (no kick)
                    const u64 x = dlo & 0xFFFFu;
                    const u64 y = (dlo >> 32) & 0xFFFFu;
                    const u64 z = (dhi >> 4) & 0xFFFFFFu;
                    const u64 f = (dhi >> 36) & 0xFFu;
                    gs.write_internal_reg(0x0C, x | (y << 16) | (z << 32) | (f << 56));
                    registers_written++;
                    break;
                }
                case 0x0D: { // XYZ3 (no kick)
                    const u64 x = dlo & 0xFFFFu;
                    const u64 y = (dlo >> 32) & 0xFFFFu;
                    const u64 z = dhi & 0xFFFFFFFFu;
                    gs.write_internal_reg(0x0D, x | (y << 16) | (z << 32));
                    registers_written++;
                    break;
                }
                case 0x0E: { // A+D
                    const u8 gs_reg = static_cast<u8>(dhi & 0xFFu);
                    gs.write_internal_reg(gs_reg, dlo);
                    registers_written++;
                    break;
                }
                default: // 0x0B / 0x0F (NOP)
                    break;
            }

            st.cur_reg++;
            if (st.cur_reg >= st.nreg) {
                st.cur_reg = 0;
                st.nloop_rem--;
                if (st.nloop_rem == 0 && stop_on_eop && st.eop) {
                    st = PathState{};
                    return;
                }
            }
        } else if (st.flg == 1) {
            for (int half = 0; half < 2 && st.reglist_rem > 0; ++half) {
                const u8 reg_desc = static_cast<u8>((st.regs >> (st.cur_reg * 4)) & 0xFu);
                const u64 val = read_le64(qwords + q * 16 + half * 8);
                if (reg_desc != 0x0E && reg_desc != 0x0F) {
                    gs.write_internal_reg(reg_desc, val);
                    registers_written++;
                }
                st.cur_reg++;
                if (st.cur_reg >= st.nreg) st.cur_reg = 0;
                st.reglist_rem--;
            }
            q++;
            if (st.reglist_rem == 0 && stop_on_eop && st.eop) {
                st = PathState{};
                return;
            }
        } else {
            PerfScope perf_vram(PerfCat::GsVram);
            const std::size_t avail = std::min<std::size_t>(qwc - q, st.nloop_rem);
            gs.write_hwreg_qwords(qwords + q * 16, avail);
            q += avail;
            registers_written += avail * 2;
            st.nloop_rem -= avail;
            if (st.nloop_rem == 0 && stop_on_eop && st.eop) {
                st = PathState{};
                return;
            }
        }
    }

    if (idx == kPath1 && stop_on_eop) {
        st = PathState{};
    }
}

} // namespace tf::ps2
