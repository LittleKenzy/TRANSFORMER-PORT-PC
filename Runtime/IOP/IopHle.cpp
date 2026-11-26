#include "IopHle.h"
#include <algorithm>
#include <atomic>
#include <cstring>

namespace tf::ps2 {

IopHle::IopHle() {
    reset();
}

void IopHle::reset() {
    m_servers.clear();
    m_loaded_irx.clear();
    m_file_log.clear();
    m_open_files.clear();
    m_mc_files.clear();
    m_mc_cwd = "/";
    m_next_mc_fd = 1;
    m_mc_first_getinfo = true;
    m_next_fd = 3;
    m_next_mod_id = 1;
    m_iop_heap_cursor = 0x00080000u;
    m_nusnd_heartbeat_addr.store(0, std::memory_order_release);
    m_nusnd_heartbeat_val.store(0, std::memory_order_release);
    m_nusnd_seq = 1;
    m_nusnd_last_token = 0;
    m_nusnd_last_size = 0;
    m_nusnd_last_lba = 0;
    m_nusnd_cached_path.clear();
    m_nusnd_cached_lba = 0;
    m_nusnd_cached_size = 0;
    m_nusnd_cached_valid = false;
    m_nusnd_read_pending = false;
    m_pad0_dma_buf = 0;
    m_pad_frame = 1;
    m_mscom = 0;
    m_smcom = 0;
    m_msflg = 0;
    m_smflg = 0x00020000u;
    m_ctrl = 0;
    m_rpc_handled = 0;
    m_rpc_unhandled = 0;

    // Built-in HLE SIF RPC servers backed by live ISO & GuestMemory
    register_rpc_server(kSifSidCdvdInit, [](u32 /*fn*/, const u8*, u32, u8* recv, u32 recv_sz) -> s32 {
        if (recv && recv_sz >= 4) write_le32(recv, 1);
        return 0;
    });

    register_rpc_server(kSifSidCdvdScmd, [](u32 fn, const u8*, u32, u8* recv, u32 recv_sz) -> s32 {
        if (recv && recv_sz >= 4) {
            if (fn == 0x05) write_le32(recv, 0x14);      // SCECdPS2DVD
            else if (fn == 0x06) write_le32(recv, 0x00); // SCECdErNO
            else if (fn == 0x0C) write_le32(recv, 0x02); // SCECdStatPause (ready)
            else write_le32(recv, 1);
        }
        return 0;
    });

    register_rpc_server(kSifSidCdvdSearch, [this](u32 /*fn*/, const u8* send, u32 send_sz, u8* recv, u32 recv_sz) -> s32 {
        if (!m_cdvd || !m_cdvd->is_mounted() || !send) {
            if (recv && recv_sz >= 4) write_le32(recv, 0);
            return 0;
        }
        // Sony SDK 3.00 sceCdSearchFile: send[36..291]=path, send[292]=ee_dest_ptr (36-byte sceCdlFILE)
        std::string path;
        if (send_sz > 36 && send[36] != 0) {
            path = std::string(reinterpret_cast<const char*>(send + 36), strnlen(reinterpret_cast<const char*>(send + 36), send_sz - 36));
        } else {
            path = std::string(reinterpret_cast<const char*>(send), strnlen(reinterpret_cast<const char*>(send), send_sz));
        }
        const IsoFileEntry* e = m_cdvd->locate_file(path);
        if (m_file_log.size() < 64) {
            m_file_log.push_back(std::string("SEARCH ") + path + (e ? (" -> FOUND lba=" + std::to_string(e->lba) + " sz=" + std::to_string(e->size)) : " -> NOT_FOUND"));
        }
        if (e) {
            u8 file_rec[36]{};
            write_le32(file_rec + 0, e->lba);
            write_le32(file_rec + 4, static_cast<u32>(e->size));
            std::memcpy(file_rec + 8, e->name.data(), std::min<std::size_t>(15, e->name.size()));
            write_le32(file_rec + 32, 0);
            if (send_sz >= 296 && m_mem) {
                u32 ee_dest = read_le32(send + 292);
                if (ee_dest != 0) m_mem->write_block(ee_dest, file_rec, sizeof(file_rec));
            }
            if (recv && recv_sz >= 32) {
                std::memcpy(recv, file_rec, std::min<u32>(recv_sz, 36u));
                if (recv_sz >= 36) write_le32(recv + 32, 1);
            } else if (recv && recv_sz >= 4) {
                write_le32(recv, 1); // 1 = file found
            }
            return 0;
        }
        if (recv && recv_sz >= 4) std::memset(recv, 0, recv_sz);
        return 0;
    });

    register_rpc_server(kSifSidCdvdNcmd, [this](u32 fn, const u8* send, u32 send_sz, u8* recv, u32 recv_sz) -> s32 {
        // fn==1: sceCdRead(lsn, sectors, ee_buf, mode)
        if ((fn == 1 || fn == 0x0E) && m_cdvd && m_cdvd->is_mounted() && m_mem && send && send_sz >= 12) {
            u32 lsn = read_le32(send + 0);
            u32 sectors = read_le32(send + 4);
            u32 ee_buf = read_le32(send + 8);
            if (m_file_log.size() < 160) {
                m_file_log.push_back("CDVD_READ fn=" + std::to_string(fn) + " lsn=" + std::to_string(lsn) +
                    " secs=" + std::to_string(sectors) + " buf=" + hex32(ee_buf));
            }
            if (sectors > 0 && sectors <= 2048 && ee_buf != 0) {
                std::vector<u8> tmp(static_cast<std::size_t>(sectors) * kIsoSectorSize);
                if (m_cdvd->read_sectors(lsn, sectors, tmp.data())) {
                    m_mem->write_block(ee_buf, tmp.data(), tmp.size());
                }
            }
        } else if (m_file_log.size() < 160) {
            m_file_log.push_back("CDVD_NCMD fn=" + std::to_string(fn) + " send=" + std::to_string(send_sz));
        }
        if (recv && recv_sz >= 4) write_le32(recv, 1);
        return 0;
    });

    register_rpc_server(kSifSidLoadFile, [this](u32 fn, const u8* send, u32 send_sz, u8* recv, u32 recv_sz) -> s32 {
        if (fn == 255u) {
            // SIF_LF_F_GET_VERSION: Sony SDK 3.00 expects 4-byte ASCII "3000"
            if (recv && recv_sz >= 4) std::memcpy(recv, "3000", 4);
            return 0;
        }
        if (send && send_sz > 8) {
            const char* p = reinterpret_cast<const char*>(send + 8);
            std::string mod(p, strnlen(p, send_sz - 8));
            if (!mod.empty()) record_irx_load(mod);
        }
        s32 mod_id = m_next_mod_id++;
        if (recv && recv_sz >= 4) write_le32(recv + 0, static_cast<u32>(mod_id)); // valid module ID > 0
        if (recv && recv_sz >= 8) write_le32(recv + 4, 0u);                        // modres = 0 (RESIDENT_END)
        return 0;
    });

    register_rpc_server(kSifSidFileIo, [this](u32 fn, const u8* send, u32 send_sz, u8* recv, u32 recv_sz) -> s32 {
        if (fn == 255u) {
            // FILEIO version check: Sony SDK 3.00 expects "3000"
            if (recv && recv_sz >= 4) std::memcpy(recv, "3000", 4);
            return 0;
        }
        // Sony SDK 3.00 FILEIO async/sync header:
        //   +0: sema_id, +4: ee_result_ptr, +8: result_size (4)
        //   recv[0..3] = 1 (request accepted); actual return value written to *ee_result_ptr
        u32 ee_res_ptr = (send && send_sz >= 12) ? read_le32(send + 4) : 0;
        auto write_result = [&](s32 val) {
            if (ee_res_ptr != 0 && m_mem) m_mem->write32(ee_res_ptr, static_cast<u32>(val));
            if (recv && recv_sz >= 4) write_le32(recv, 1u);
        };

        if (fn == 0u && send && send_sz > 20 && m_cdvd && m_cdvd->is_mounted()) {
            // sceOpen: +12=flags, +16=mode, +20=path[1024]
            const char* p = reinterpret_cast<const char*>(send + 20);
            std::string path(p, strnlen(p, send_sz - 20));
            if (path == "rom0:ROMVER" || path == "ROM0:ROMVER") {
                // Clean-room standard NTSC-U ROMVER metadata string (zero copyrighted BIOS code)
                static constexpr char kRomVer[] = "0160AC20010704\n";
                s32 fd = m_next_fd++;
                OpenIsoFile f;
                f.path = path;
                f.size = sizeof(kRomVer) - 1;
                f.synthetic_data.assign(reinterpret_cast<const u8*>(kRomVer), reinterpret_cast<const u8*>(kRomVer) + f.size);
                m_open_files[fd] = std::move(f);
                if (m_file_log.size() < 64) m_file_log.push_back("OPEN rom0:ROMVER -> fd=" + std::to_string(fd) + " (synthetic NTSC-U)");
                write_result(fd);
                return 0;
            }
            const IsoFileEntry* e = m_cdvd->locate_file(path);
            if (m_file_log.size() < 240) {
                m_file_log.push_back(std::string("OPEN ") + path + (e ? (" -> fd=" + std::to_string(m_next_fd) + " sz=" + std::to_string(e->size)) : " -> ENOENT"));
            }
            if (e) {
                s32 fd = m_next_fd++;
                m_open_files[fd] = OpenIsoFile{e->lba, e->size, 0, path, {}};
                write_result(fd);
                return 0;
            }
            write_result(-2); // ENOENT
            return 0;
        }
        if (fn == 1u && send && send_sz >= 16) {
            // sceClose: +12=fd
            s32 fd = static_cast<s32>(read_le32(send + 12));
            m_open_files.erase(fd);
            write_result(0);
            return 0;
        }
        if (fn == 2u && send && send_sz >= 24 && m_cdvd && m_mem) {
            // sceRead: +12=fd, +16=ee_buf, +20=size, +24=read_data_ptr (optional unaligned fixup struct)
            s32 fd = static_cast<s32>(read_le32(send + 12));
            u32 ee_buf = read_le32(send + 16);
            u32 req_sz = read_le32(send + 20);
            if (send_sz >= 28) {
                u32 rd_ptr = read_le32(send + 24);
                if (rd_ptr != 0) {
                    m_mem->write32(rd_ptr + 0, 0); // size1 = 0
                    m_mem->write32(rd_ptr + 4, 0); // size2 = 0
                }
            }
            auto it = m_open_files.find(fd);
            if (it != m_open_files.end() && ee_buf != 0) {
                auto& f = it->second;
                u64 avail = (f.offset < f.size) ? (f.size - f.offset) : 0;
                u32 to_read = static_cast<u32>(std::min<u64>(req_sz, avail));
                if (to_read > 0) {
                    if (!f.synthetic_data.empty()) {
                        m_mem->write_block(ee_buf, f.synthetic_data.data() + f.offset, to_read);
                        f.offset += to_read;
                        write_result(static_cast<s32>(to_read));
                        return 0;
                    }
                    u32 start_sec = f.lba + static_cast<u32>(f.offset / kIsoSectorSize);
                    u32 sec_off   = static_cast<u32>(f.offset % kIsoSectorSize);
                    u32 num_secs  = (sec_off + to_read + kIsoSectorSize - 1u) / kIsoSectorSize;
                    std::vector<u8> tmp(static_cast<std::size_t>(num_secs) * kIsoSectorSize);
                    if (m_cdvd->read_sectors(start_sec, num_secs, tmp.data())) {
                        m_mem->write_block(ee_buf, tmp.data() + sec_off, to_read);
                        if (m_file_log.size() < 64) {
                            m_file_log.push_back("READ fd=" + std::to_string(fd) + " dst=" + hex32(ee_buf) +
                                " sz=" + std::to_string(to_read) + " off=" + std::to_string(f.offset));
                        }
                        f.offset += to_read;
                        write_result(static_cast<s32>(to_read));
                        return 0;
                    }
                }
                write_result(0);
                return 0;
            }
            write_result(-9); // EBADF
            return 0;
        }
        if (fn == 4u && send && send_sz >= 24) {
            // sceLseek: +12=fd, +16=offset, +20=whence
            s32 fd = static_cast<s32>(read_le32(send + 12));
            s32 off = static_cast<s32>(read_le32(send + 16));
            u32 whence = read_le32(send + 20);
            auto it = m_open_files.find(fd);
            if (it != m_open_files.end()) {
                s64 base = 0;
                if (whence == 1) base = static_cast<s64>(it->second.offset);
                else if (whence == 2) base = static_cast<s64>(it->second.size);
                s64 next = std::clamp<s64>(base + off, 0, static_cast<s64>(it->second.size));
                it->second.offset = static_cast<u64>(next);
                write_result(static_cast<s32>(next));
                return 0;
            }
            write_result(-9);
            return 0;
        }
        write_result(0);
        return 0;
    });

    register_rpc_server(kSifSidPadMan, [this](u32 fn, const u8* send, u32 send_sz, u8* recv, u32 recv_sz) -> s32 {
        u32 cmd = (send && send_sz >= 4) ? read_le32(send + 0) : 0;
        if (send && send_sz >= 24) {
            u32 w1 = read_le32(send + 4);
            u32 w2 = read_le32(send + 8);
            u32 w4 = read_le32(send + 16);
            if (m_file_log.size() < 64) {
                m_file_log.push_back("PADMAN fn=" + std::to_string(fn) +
                    " cmd=" + hex32(cmd) + " p=" + std::to_string(w1) + " s=" + std::to_string(w2) +
                    " buf=" + hex32(w4));
            }
            if (cmd == 1u && w1 == 0 && w2 == 0 && w4 >= 0x00100000u && w4 < kEeRamSize) {
                m_pad0_dma_buf = w4;
                write_pad0_dma();
            }
        }
        if (recv && recv_sz >= 4) write_le32(recv + 0, 1u);
        if (recv && recv_sz >= 16) {
            if (cmd == 0x12u) {
                // PAD_RPCCMD_GET_MODVER: Sony SDK 3.00 libpad requires (ver >> 8) == 4!
                write_le32(recv + 12, 0x00000400u);
            } else {
                write_le32(recv + 12, 1u); // scePadInit (cmd=16) & scePadPortOpen (cmd=1) success
            }
        }
        if (recv && recv_sz >= 20) {
            write_le32(recv + 16, 1u);     // scePadEnterPressMode / SetButtonInfo (cmd=10) reads recv[16]!
        }
        if (recv && recv_sz >= 24) {
            if (cmd == 1u) {
                write_le32(recv + 20, 0x00090000u); // IOP pad DMA buffer address
            } else {
                write_le32(recv + 20, 1u);          // scePadSetMainMode (cmd=6) / scePadSetActAlign (cmd=8) success!
            }
        }
        return 0;
    });
    register_rpc_server(kSifSidPadMan2, [this](u32 fn, const u8* send, u32 send_sz, u8* recv, u32 recv_sz) -> s32 {
        (void)fn;
        if (send && send_sz >= 24) {
            u32 port = read_le32(send + 4);
            u32 ee_buf = read_le32(send + 16);
            if (port == 0 && ee_buf >= 0x00100000u && ee_buf < kEeRamSize) {
                m_pad0_dma_buf = ee_buf;
                write_pad0_dma();
            }
        }
        if (recv && recv_sz >= 4) write_le32(recv + 0, 1u);
        if (recv && recv_sz >= 16) write_le32(recv + 12, 1u);
        if (recv && recv_sz >= 20) write_le32(recv + 16, 1u);
        if (recv && recv_sz >= 24) write_le32(recv + 20, 1u);
        return 0;
    });

    register_rpc_server(0x8000059Cu, [](u32, const u8*, u32, u8* recv, u32 recv_sz) -> s32 {
        if (recv && recv_sz >= 4) write_le32(recv, 1);
        return 0;
    });

    // Sony SDK 3.00 libmc / MCMAN + MCSERV (0x80000400) backed by VirtualMemoryCard
    register_rpc_server(kSifSidMcMan, [this](u32 fn, const u8* send, u32 send_sz, u8* recv, u32 recv_sz) -> s32 {
        auto normalize_mc_rel = [&](const std::string& raw) -> std::string {
            std::string full = raw;
            if (!full.empty() && full[0] != '/') {
                full = (m_mc_cwd == "/" ? "/" : (m_mc_cwd + "/")) + full;
            }
            while (!full.empty() && full[0] == '/') full.erase(full.begin());
            while (!full.empty() && full.back() == '/') full.pop_back();
            return full;
        };

        if (m_file_log.size() < 96) {
            m_file_log.push_back("MCMAN fn=" + std::to_string(fn) + " send=" + std::to_string(send_sz) + " recv=" + std::to_string(recv_sz));
        }

        if (fn == 254u) {
            // MC_RPCCMD_INIT: Sony SDK 3.00 requires mcman >= 522 (0x20A) and mcserv >= 526 (0x20E)
            if (recv && recv_sz >= 4)  write_le32(recv + 0, 0u);
            if (recv && recv_sz >= 8)  write_le32(recv + 4, 0x00000210u); // mcman 2.16 (528)
            if (recv && recv_sz >= 12) write_le32(recv + 8, 0x00000210u); // mcserv 2.16 (528)
            return 0;
        }
        if (fn == 21u) {
            // sceMcGetSlotMax
            if (recv && recv_sz >= 4) write_le32(recv + 0, 1u);
            return 0;
        }
        if (fn == 1u && send && send_sz >= 32) {
            // sceMcGetInfo: +4=port, +8=slot, +28=ee_enddata_ptr (+0=type, +4=free, +144=format)
            u32 port = read_le32(send + 4);
            u32 slot = read_le32(send + 8);
            u32 enddata = read_le32(send + 28) & 0x1FFFFFFFu;
            bool ok = (port == 0 && slot == 0 && m_memcard0 && m_memcard0->is_mounted());
            if (enddata != 0 && m_mem) {
                m_mem->write32(enddata + 0,   ok ? 2u : 0u);    // 2 = sceMcTypePS2
                m_mem->write32(enddata + 4,   ok ? 7999u : 0u); // ~8 MiB free KB
                m_mem->write32(enddata + 144, ok ? 1u : 0u);    // 1 = formatted
            }
            u32 res = static_cast<u32>(-10);
            if (ok) {
                res = m_mc_first_getinfo ? static_cast<u32>(-1) : 0u; // -1 = sceMcResChangedCard on first detection
                m_mc_first_getinfo = false;
            }
            if (m_file_log.size() < 96) {
                m_file_log.push_back("  MC_GETINFO p=" + std::to_string(port) + " s=" + std::to_string(slot) +
                    " end=" + hex32(enddata) + " -> " + std::to_string(static_cast<s32>(res)));
            }
            if (recv && recv_sz >= 4) write_le32(recv + 0, res);
            return 0;
        }
        if (fn == 2u && send && send_sz > 20) {
            // sceMcOpen / sceMcMkdir (flags == 0x40)
            u32 port  = read_le32(send + 0);
            u32 slot  = read_le32(send + 4);
            u32 flags = read_le32(send + 8);
            const char* p = reinterpret_cast<const char*>(send + 20);
            std::string raw_path(p, strnlen(p, send_sz - 20));
            std::string rel = normalize_mc_rel(raw_path);
            if (port != 0 || slot != 0 || !m_memcard0 || !m_memcard0->is_mounted()) {
                if (recv && recv_sz >= 4) write_le32(recv + 0, static_cast<u32>(-10));
                return 0;
            }
            if (flags == 0x40u) {
                // sceMcMkdir: create marker so directory exists
                std::string dummy_err;
                if (!rel.empty()) {
                    m_memcard0->write_save_file(rel + "/.mcdir", {}, dummy_err);
                }
                if (recv && recv_sz >= 4) write_le32(recv + 0, 0u);
                return 0;
            }
            bool create = (flags & 0x200u) != 0;
            bool writable = (flags & 0x02u) != 0 || create;
            std::vector<u8> existing;
            std::string err;
            bool exists = m_memcard0->file_exists(rel) && m_memcard0->read_save_file(rel, existing, err);
            if (!exists && !create) {
                if (recv && recv_sz >= 4) write_le32(recv + 0, static_cast<u32>(-4)); // sceMcResNoEntry
                return 0;
            }
            s32 fd = m_next_mc_fd++;
            m_mc_files[fd] = OpenMcFile{rel, std::move(existing), 0u, writable, create && !exists};
            if (create && !exists) {
                m_memcard0->write_save_file(rel, {}, err);
            }
            if (recv && recv_sz >= 4) write_le32(recv + 0, static_cast<u32>(fd));
            return 0;
        }
        if (fn == 3u && send && send_sz >= 4) {
            // sceMcClose
            s32 fd = static_cast<s32>(read_le32(send + 0));
            auto it = m_mc_files.find(fd);
            if (it != m_mc_files.end()) {
                if (it->second.dirty && m_memcard0) {
                    std::string err;
                    m_memcard0->write_save_file(it->second.rel_path, it->second.data, err);
                }
                m_mc_files.erase(it);
                if (recv && recv_sz >= 4) write_le32(recv + 0, 0u);
            } else {
                if (recv && recv_sz >= 4) write_le32(recv + 0, static_cast<u32>(-5));
            }
            return 0;
        }
        if (fn == 4u && send && send_sz >= 24) {
            // sceMcSeek
            s32 fd = static_cast<s32>(read_le32(send + 0));
            s32 off = static_cast<s32>(read_le32(send + 16));
            u32 org = read_le32(send + 20);
            auto it = m_mc_files.find(fd);
            if (it != m_mc_files.end()) {
                s64 base = (org == 1u) ? it->second.offset : (org == 2u) ? static_cast<s64>(it->second.data.size()) : 0;
                s64 next = std::max<s64>(0, base + off);
                it->second.offset = static_cast<u32>(next);
                if (recv && recv_sz >= 4) write_le32(recv + 0, it->second.offset);
            } else {
                if (recv && recv_sz >= 4) write_le32(recv + 0, static_cast<u32>(-5));
            }
            return 0;
        }
        if (fn == 5u && send && send_sz >= 32) {
            // sceMcRead: +0=fd, +12=size, +24=ee_buf, +28=ee_enddata
            s32 fd = static_cast<s32>(read_le32(send + 0));
            u32 sz = read_le32(send + 12);
            u32 ee_buf = read_le32(send + 24) & 0x1FFFFFFFu;
            u32 enddata = read_le32(send + 28) & 0x1FFFFFFFu;
            if (enddata != 0 && m_mem) m_mem->zero_block(enddata, 192u);
            auto it = m_mc_files.find(fd);
            if (it != m_mc_files.end() && m_mem) {
                u32 avail = (it->second.offset < it->second.data.size())
                    ? static_cast<u32>(it->second.data.size() - it->second.offset) : 0u;
                u32 to_read = std::min(sz, avail);
                if (to_read > 0 && ee_buf != 0) {
                    m_mem->write_block(ee_buf, it->second.data.data() + it->second.offset, to_read);
                    it->second.offset += to_read;
                }
                if (recv && recv_sz >= 4) write_le32(recv + 0, to_read);
            } else {
                if (recv && recv_sz >= 4) write_le32(recv + 0, static_cast<u32>(-5));
            }
            return 0;
        }
        if (fn == 6u && send && send_sz >= 32) {
            // sceMcWrite: +0=fd, +12=body_sz, +20=head_sz(at +32), +24=ee_body_buf
            s32 fd = static_cast<s32>(read_le32(send + 0));
            u32 body_sz = read_le32(send + 12);
            u32 head_sz = std::min<u32>(read_le32(send + 20), (send_sz > 32) ? (send_sz - 32) : 0u);
            u32 ee_buf  = read_le32(send + 24) & 0x1FFFFFFFu;
            auto it = m_mc_files.find(fd);
            if (it != m_mc_files.end() && m_mem) {
                u32 total = head_sz + body_sz;
                if (it->second.offset + total > it->second.data.size()) {
                    it->second.data.resize(it->second.offset + total, 0u);
                }
                if (head_sz > 0) {
                    std::memcpy(it->second.data.data() + it->second.offset, send + 32, head_sz);
                }
                if (body_sz > 0 && ee_buf != 0) {
                    m_mem->read_block(ee_buf, it->second.data.data() + it->second.offset + head_sz, body_sz);
                }
                it->second.offset += total;
                it->second.dirty = true;
                if (m_memcard0) {
                    std::string err;
                    m_memcard0->write_save_file(it->second.rel_path, it->second.data, err);
                }
                if (recv && recv_sz >= 4) write_le32(recv + 0, total);
            } else {
                if (recv && recv_sz >= 4) write_le32(recv + 0, static_cast<u32>(-5));
            }
            return 0;
        }
        if (fn == 12u && send && send_sz > 20) {
            // sceMcChdir: +0=port, +4=slot, +16=ee_pwd_buf, +20=new_dir
            u32 port = read_le32(send + 0);
            u32 pwd_buf = read_le32(send + 16) & 0x1FFFFFFFu;
            const char* p = reinterpret_cast<const char*>(send + 20);
            std::string req(p, strnlen(p, send_sz - 20));
            if (port != 0 || !m_memcard0 || !m_memcard0->is_mounted()) {
                if (recv && recv_sz >= 4) write_le32(recv + 0, static_cast<u32>(-10));
                return 0;
            }
            if (pwd_buf != 0 && m_mem) {
                m_mem->write_block(pwd_buf, reinterpret_cast<const u8*>(m_mc_cwd.c_str()), m_mc_cwd.size() + 1);
            }
            if (req == "/" || req.empty()) {
                m_mc_cwd = "/";
                if (recv && recv_sz >= 4) write_le32(recv + 0, 0u);
                return 0;
            }
            if (req == "..") {
                auto slash = m_mc_cwd.find_last_of('/');
                m_mc_cwd = (slash == std::string::npos || slash == 0) ? "/" : m_mc_cwd.substr(0, slash);
                if (recv && recv_sz >= 4) write_le32(recv + 0, 0u);
                return 0;
            }
            std::string rel = normalize_mc_rel(req);
            bool dir_exists = m_memcard0->file_exists(rel + "/.mcdir");
            if (!dir_exists) {
                for (const auto& f : m_memcard0->list_files()) {
                    if (f.rfind(rel + "/", 0) == 0) { dir_exists = true; break; }
                }
            }
            if (dir_exists) {
                m_mc_cwd = "/" + rel;
                if (recv && recv_sz >= 4) write_le32(recv + 0, 0u);
            } else {
                if (recv && recv_sz >= 4) write_le32(recv + 0, static_cast<u32>(-4)); // sceMcResNoEntry
            }
            return 0;
        }
        if (fn == 13u && send && send_sz > 20) {
            // sceMcGetDir: +0=port, +4=slot, +8=flags, +12=maxent, +16=ee_table_ptr, +20=mask
            u32 port = read_le32(send + 0);
            s32 maxent = static_cast<s32>(read_le32(send + 12));
            u32 tbl_ptr = read_le32(send + 16) & 0x1FFFFFFFu;
            const char* p = reinterpret_cast<const char*>(send + 20);
            std::string mask(p, strnlen(p, send_sz - 20));
            if (port != 0 || !m_memcard0 || !m_memcard0->is_mounted()) {
                if (recv && recv_sz >= 4) write_le32(recv + 0, static_cast<u32>(-10));
                return 0;
            }
            // Strip trailing '*' wildcard
            std::string dir_part = mask;
            auto star = dir_part.find('*');
            if (star != std::string::npos) dir_part.resize(star);
            std::string prefix = normalize_mc_rel(dir_part);
            std::vector<std::pair<std::string, std::pair<bool, u32>>> entries;
            for (const auto& f : m_memcard0->list_files()) {
                std::string sub = f;
                if (!prefix.empty()) {
                    if (f.rfind(prefix, 0) != 0) continue;
                    if (f.size() > prefix.size() && f[prefix.size()] == '/') {
                        sub = f.substr(prefix.size() + 1);
                    }
                }
                if (sub == ".mcdir") continue;
                auto sl = sub.find('/');
                std::string name = (sl == std::string::npos) ? sub : sub.substr(0, sl);
                bool is_dir = (sl != std::string::npos);
                bool seen = false;
                for (const auto& ex : entries) if (ex.first == name) { seen = true; break; }
                if (!seen && !name.empty()) {
                    u32 fsz = 0;
                    if (!is_dir) {
                        std::vector<u8> tmp; std::string err;
                        if (m_memcard0->read_save_file(f, tmp, err)) fsz = static_cast<u32>(tmp.size());
                    }
                    entries.push_back({name, {is_dir, fsz}});
                }
            }
            u32 out_cnt = (maxent < 0) ? static_cast<u32>(entries.size())
                                       : std::min<u32>(static_cast<u32>(entries.size()), static_cast<u32>(maxent));
            if (tbl_ptr != 0 && m_mem && maxent > 0) {
                for (u32 i = 0; i < out_cnt; ++i) {
                    u8 rec[64]{};
                    write_le32(rec + 24, entries[i].second.second);
                    write_le16(rec + 28, entries[i].second.first ? 0x8427u : 0x8417u);
                    std::memcpy(rec + 32, entries[i].first.c_str(), std::min<std::size_t>(31, entries[i].first.size()));
                    m_mem->write_block(tbl_ptr + i * 64u, rec, 64u);
                }
            }
            if (recv && recv_sz >= 4) write_le32(recv + 0, out_cnt);
            return 0;
        }
        if (recv && recv_sz >= 4) write_le32(recv + 0, 0u);
        return 0;
    });

    register_rpc_server(kSifSidLibSd, [](u32, const u8*, u32, u8* recv, u32 recv_sz) -> s32 {
        if (recv && recv_sz >= 4) write_le32(recv, 0);
        return 0;
    });

    // SIF_SID_IOPHEAP (0x80000003): sceSifAllocIopHeap / sceSifFreeIopHeap
    register_rpc_server(0x80000003u, [this](u32 fn, const u8* send, u32 send_sz, u8* recv, u32 recv_sz) -> s32 {
        if (fn == 1u) {
            // AllocIopHeap(size)
            u32 sz = (send && send_sz >= 4) ? read_le32(send) : 4096u;
            sz = (sz + 63u) & ~63u;
            u32 ptr = m_iop_heap_cursor;
            m_iop_heap_cursor += sz;
            if (m_iop_heap_cursor > 0x001F0000u) m_iop_heap_cursor = 0x00080000u;
            if (recv && recv_sz >= 4) write_le32(recv, ptr);
            return 0;
        }
        if (recv && recv_sz >= 4) write_le32(recv, 0);
        return 0;
    });

    // Traveller's Tales NuSound3M (NUSND3M.IRX, SID 0x00012345)
    register_rpc_server(0x00012345u, [this](u32, const u8* send, u32 send_sz, u8* recv, u32 recv_sz) -> s32 {
        auto read_u32_be16 = [&](u32 byte_pos) -> u32 {
            u16 hi = read_le16(send + byte_pos);
            u16 lo = read_le16(send + byte_pos + 2);
            return (static_cast<u32>(hi) << 16) | static_cast<u32>(lo);
        };
        if (send && send_sz >= 4) {
            u16 magic = read_le16(send + 0);
            u16 num_cmds = read_le16(send + 2);
            if (magic == 1354u) {
                u32 pos = 4;
                for (u16 c = 0; c < num_cmds && pos + 4 <= send_sz; ++c) {
                    u16 cmd_id = read_le16(send + pos);
                    u16 argc   = read_le16(send + pos + 2);
                    pos += 4;
                    u32 arg_bytes = static_cast<u32>(argc) * 2u;
                    if (pos + arg_bytes > send_sz) break;
                    if (cmd_id == 73u && argc >= 2) {
                        // Command 73: register EE address for IOP audio heartbeat DMA
                        m_nusnd_heartbeat_addr.store(read_u32_be16(pos), std::memory_order_release);
                    } else if (cmd_id == 32u && argc >= 2) {
                        // Command 32: NuSound3M sync token
                        m_nusnd_last_token = read_u32_be16(pos);
                    } else if (cmd_id == 31u && argc >= 5) {
                        // Command 31: CDVD Open/Locate file on ISO (args: token:2, flags:2, ascii_path...)
                        m_nusnd_last_token = read_u32_be16(pos);
                        const char* p = reinterpret_cast<const char*>(send + pos + 8);
                        std::string path(p, strnlen(p, arg_bytes - 8));
                        if (m_nusnd_cached_valid && path == m_nusnd_cached_path) {
                            m_nusnd_last_size = m_nusnd_cached_size;
                            m_nusnd_last_lba  = m_nusnd_cached_lba;
                        } else {
                            const IsoFileEntry* e = (m_cdvd && m_cdvd->is_mounted()) ? m_cdvd->locate_file(path) : nullptr;
                            if (m_file_log.size() < 96) {
                                m_file_log.push_back(std::string("NUSND_OPEN ") + path +
                                    (e ? (" -> lba=" + std::to_string(e->lba) + " sz=" + std::to_string(e->size)) : " -> NOT_FOUND"));
                            }
                            if (e) {
                                m_nusnd_last_size = static_cast<u32>(e->size);
                                m_nusnd_last_lba  = e->lba;
                                m_nusnd_cached_path = path;
                                m_nusnd_cached_lba  = e->lba;
                                m_nusnd_cached_size = static_cast<u32>(e->size);
                                m_nusnd_cached_valid = true;
                            } else {
                                m_nusnd_last_size = 0;
                                m_nusnd_last_lba  = 0;
                            }
                        }
                    } else if (cmd_id == 30u && argc >= 9) {
                        // Command 30: CDVD Stream Read setup (token:2, lba:2, end_off:2, mode:1, start_off:2)
                        m_nusnd_last_token   = read_u32_be16(pos + 0);
                        m_nusnd_read_lba     = read_u32_be16(pos + 4);
                        m_nusnd_read_end     = read_u32_be16(pos + 8);
                        m_nusnd_read_start   = read_u32_be16(pos + 14);
                        m_nusnd_read_pending = true;
                    } else if (cmd_id == 49u && argc >= 2) {
                        // Command 49: Set EE destination buffer & execute pending CDVD DMA read
                        u32 ee_dst = read_u32_be16(pos);
                        if (m_nusnd_read_pending && m_cdvd && m_cdvd->is_mounted() && m_mem && ee_dst != 0 && m_nusnd_read_end > m_nusnd_read_start) {
                            u32 to_read = m_nusnd_read_end - m_nusnd_read_start;
                            if (to_read <= 16U * 1024U * 1024U) {
                                u32 start_sec = m_nusnd_read_lba + (m_nusnd_read_start / kIsoSectorSize);
                                u32 sec_off   = m_nusnd_read_start % kIsoSectorSize;
                                u32 num_secs  = (sec_off + to_read + kIsoSectorSize - 1u) / kIsoSectorSize;
                                std::vector<u8> tmp(static_cast<std::size_t>(num_secs) * kIsoSectorSize);
                                if (m_cdvd->read_sectors(start_sec, num_secs, tmp.data())) {
                                    m_mem->write_block(ee_dst, tmp.data() + sec_off, to_read);
                                }
                            }
                        }
                        m_nusnd_read_pending = false;
                    }
                    pos += arg_bytes;
                }
            }
        }
        tick_heartbeat();
        if (recv && recv_sz > 0) {
            std::memset(recv, 0, recv_sz);
            if (recv_sz >= 8) {
                write_le32(recv + 4, m_nusnd_seq++);
            }
            if (recv_sz >= 216) {
                write_le32(recv + 204, (m_nusnd_last_token << 8));
                write_le32(recv + 208, m_nusnd_last_size);
                write_le32(recv + 212, m_nusnd_last_lba);
            }
        }
        return 0;
    });
}

void IopHle::write_pad0_dma(GuestMemory* mem_override) {
    GuestMemory* mem = mem_override ? mem_override : m_mem;
    if (m_pad0_dma_buf == 0 || !mem) return;
    u8 report[32]{};
    if (m_pad0) {
        if (m_pad_mtx) {
            std::lock_guard<std::mutex> pad_lk(*m_pad_mtx);
            m_pad0->pack_ps2_pad_buffer(report);
        } else {
            m_pad0->pack_ps2_pad_buffer(report);
        }
    } else {
        report[0] = 0x00; report[1] = 0x79; report[2] = 0xFF; report[3] = 0xFF;
        report[4] = 0x80; report[5] = 0x80; report[6] = 0x80; report[7] = 0x80;
    }
    // Exact Sony SDK 3.00 libpad 128-byte pad_data_t layout (verified against SLUS_216.02):
    //   +0..31: raw DS2 report, +48..55: actuator info[2] (4 bytes each, scePadInfoAct),
    //   +80..86: u16 modeTable[4] ({4, 7}), +88: frame, +96: len(32),
    //   +100: modeConfig(2), +101: modeCurId(0x79), +102: buttonInfo(0x03=pressure),
    //   +103: ok(1), +104: modeTableNum(2), +105: modeCurOffs(1), +106: numActuators(2),
    //   +112: state(6=Stable), +113: reqState(0=Complete), +114: active(1)
    static constexpr u8 kActInfo[8] = {
        0x01, 0x01, 0x00, 0x1E, // Actuator 0 (small motor: func=1, sub=1, size=0, curr=30mA)
        0x01, 0x02, 0x01, 0x3C  // Actuator 1 (large motor: func=1, sub=2, size=1, curr=60mA)
    };
    for (u32 slot = 0; slot < 2; ++slot) {
        u32 base = m_pad0_dma_buf + slot * 128u;
        mem->write_block(base + 0, report, 32);
        mem->write_block(base + 48, kActInfo, sizeof(kActInfo));
        mem->write32(base + 60, m_pad_frame);
        mem->write16(base + 80, 4u);          // modeTable[0] = 4 (Digital)
        mem->write16(base + 82, 7u);          // modeTable[1] = 7 (DualShock 2)
        mem->write32(base + 88, m_pad_frame);
        mem->write32(base + 96, 32u);         // len = 32 bytes for scePadRead!
        mem->write8(base + 100, 2u);          // modeConfig = 2 (DualShock 2)
        mem->write8(base + 101, 0x79u);       // modeCurId = 0x79 (DualShock 2 Analog + Pressure)
        mem->write8(base + 102, 0x03u);       // buttonInfo bit 1 (0x02) => scePadInfoPressMode() == 1
        mem->write8(base + 103, 1u);          // ok = 1
        mem->write8(base + 104, 2u);          // modeTableNum = 2
        mem->write8(base + 105, 1u);          // modeCurOffs = 1 (DualShock 2)
        mem->write8(base + 106, 2u);          // numActuators = 2 (small + large motor for scePadInfoAct)
        mem->write8(base + 112, 6u);          // state = scePadStateStable (6)
        mem->write8(base + 113, 0u);          // reqState = scePadReqStateComplete (0)
        mem->write8(base + 114, 1u);          // active/connected flag for scePadInfoMode/InfoAct/InfoPressMode
        mem->write32(base + 120, 1u);
    }
    m_pad_frame++;
}

void IopHle::sync_pad0_now(GuestMemory& mem) {
    write_pad0_dma(&mem);
}

void IopHle::sync_pad0_now() {
    write_pad0_dma(nullptr);
}

void IopHle::tick_heartbeat() {
    const u32 addr = m_nusnd_heartbeat_addr.load(std::memory_order_acquire);
    if (addr == 0 || !m_mem) return;
    const u32 next_val = m_nusnd_heartbeat_val.fetch_add(1u, std::memory_order_relaxed) + 1u;
    if ((addr & 3u) == 0) {
        if (u8* ptr = m_mem->translate_ptr(addr, 4)) {
            std::atomic_ref<u32>(*reinterpret_cast<u32*>(ptr)).store(next_val, std::memory_order_release);
            return;
        }
    }
    m_mem->write32(addr, next_val);
}

void IopHle::tick_vblank() {
    tick_heartbeat();
    write_pad0_dma();
}

void IopHle::register_rpc_server(u32 sid, SifRpcHandler handler) {
    m_servers[sid] = std::move(handler);
}

s32 IopHle::call_rpc(u32 sid, u32 rpc_num, const u8* send_buf, u32 send_size, u8* recv_buf, u32 recv_size) {
    auto it = m_servers.find(sid);
    if (it == m_servers.end()) {
        m_rpc_unhandled++;
        return -1;
    }
    m_rpc_handled++;
    const s32 res = it->second(rpc_num, send_buf, send_size, recv_buf, recv_size);
    if (has_heartbeat()) {
        tick_heartbeat();
    }
    return res;
}

void IopHle::record_irx_load(const std::string& path) {
    m_loaded_irx.push_back(path);
}

u32 IopHle::read_sif_reg(u32 paddr) const {
    switch (paddr) {
        case 0x1000F200u: return m_mscom;
        case 0x1000F210u: return m_smcom;
        case 0x1000F220u: return m_msflg;
        case 0x1000F230u: return m_smflg;
        case 0x1000F240u: return m_ctrl;
        default: return 0;
    }
}

void IopHle::write_sif_reg(u32 paddr, u32 val) {
    switch (paddr) {
        case 0x1000F200u: m_mscom = val; break;
        case 0x1000F210u: m_smcom = val; break;
        case 0x1000F220u: m_msflg |= val; break;
        case 0x1000F230u: m_smflg &= ~val; break;
        case 0x1000F240u: m_ctrl = val; break;
        default: break;
    }
}

} // namespace tf::ps2
