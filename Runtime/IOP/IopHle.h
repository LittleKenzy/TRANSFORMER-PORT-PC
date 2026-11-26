#pragma once

#include "../Memory/GuestMemory.h"
#include "../CDVD/CdvdDrive.h"
#include "../Input/DualShock2.h"
#include "../MemoryCard/VirtualMemoryCard.h"
#include <atomic>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace tf::ps2 {

constexpr u32 kSifSidFileIo        = 0x80000001u;
constexpr u32 kSifSidLoadFile      = 0x80000006u;
constexpr u32 kSifSidCdvdInit      = 0x80000592u;
constexpr u32 kSifSidCdvdScmd      = 0x80000593u;
constexpr u32 kSifSidCdvdNcmd      = 0x80000595u;
constexpr u32 kSifSidCdvdSearch    = 0x80000597u;
constexpr u32 kSifSidPadMan        = 0x80000100u;
constexpr u32 kSifSidPadMan2       = 0x80000101u;
constexpr u32 kSifSidMcMan         = 0x80000400u;
constexpr u32 kSifSidLibSd         = 0x80000701u;

using SifRpcHandler = std::function<s32(u32 rpc_num, const u8* send_buf, u32 send_size, u8* recv_buf, u32 recv_size)>;

// IOP R3000A High-Level Compatibility Layer & SIF RPC Dispatcher.
class IopHle {
public:
    IopHle();
    void reset();

    void attach_cdvd(CdvdDrive* cdvd) { m_cdvd = cdvd; }
    void attach_memory(GuestMemory* mem) { m_mem = mem; }
    void attach_pad0(DualShock2State* pad) { m_pad0 = pad; }
    void attach_pad_mutex(std::mutex* mtx) { m_pad_mtx = mtx; }
    void attach_memcard0(VirtualMemoryCard* mc) { m_memcard0 = mc; }
    void register_rpc_server(u32 sid, SifRpcHandler handler);

    // Invokes a registered SIF RPC server (returns 0 on success, -1 if unimplemented SID)
    s32 call_rpc(u32 sid, u32 rpc_num, const u8* send_buf, u32 send_size, u8* recv_buf, u32 recv_size);

    // Track loaded IRX modules requested by the game
    void record_irx_load(const std::string& path);
    const std::vector<std::string>& loaded_irx_modules() const { return m_loaded_irx; }
    const std::vector<std::string>& file_access_log() const { return m_file_log; }

    // SIF register MMIO (0x1000F200 .. 0x1000F260)
    u32 read_sif_reg(u32 paddr) const;
    void write_sif_reg(u32 paddr, u32 val);

    u64 rpc_calls_handled() const { return m_rpc_handled; }
    u64 rpc_calls_unhandled() const { return m_rpc_unhandled; }
    void tick_heartbeat();
    void tick_vblank();
    void sync_pad0_now(GuestMemory& mem);
    void sync_pad0_now();
    bool has_heartbeat() const { return m_nusnd_heartbeat_addr.load(std::memory_order_acquire) != 0; }

private:
    void write_pad0_dma(GuestMemory* mem_override = nullptr);
    struct OpenIsoFile {
        u32 lba = 0;
        u64 size = 0;
        u64 offset = 0;
        std::string path;
        std::vector<u8> synthetic_data;
    };

    struct OpenMcFile {
        std::string rel_path;
        std::vector<u8> data;
        u32 offset = 0;
        bool writable = false;
        bool dirty = false;
    };

    CdvdDrive* m_cdvd = nullptr;
    GuestMemory* m_mem = nullptr;
    DualShock2State* m_pad0 = nullptr;
    std::mutex* m_pad_mtx = nullptr;
    VirtualMemoryCard* m_memcard0 = nullptr;
    std::unordered_map<s32, OpenMcFile> m_mc_files;
    std::string m_mc_cwd = "/";
    s32 m_next_mc_fd = 1;
    bool m_mc_first_getinfo = true;
    u32 m_pad0_dma_buf = 0;
    u32 m_pad_frame = 1;
    std::unordered_map<u32, SifRpcHandler> m_servers;
    std::vector<std::string> m_loaded_irx;
    std::vector<std::string> m_file_log;
    std::unordered_map<s32, OpenIsoFile> m_open_files;
    s32 m_next_fd = 3;
    s32 m_next_mod_id = 1;
    u32 m_iop_heap_cursor = 0x00080000u;
    std::atomic<u32> m_nusnd_heartbeat_addr{0};
    std::atomic<u32> m_nusnd_heartbeat_val{0};
    u32 m_nusnd_seq = 1;
    u32 m_nusnd_last_token = 0;
    u32 m_nusnd_last_size = 0;
    u32 m_nusnd_last_lba = 0;
    std::string m_nusnd_cached_path;
    u32 m_nusnd_cached_lba = 0;
    u32 m_nusnd_cached_size = 0;
    bool m_nusnd_cached_valid = false;
    // Pending NuSound3M CDVD read (cmd 30 followed by cmd 49)
    bool m_nusnd_read_pending = false;
    u32 m_nusnd_read_lba = 0;
    u32 m_nusnd_read_start = 0;
    u32 m_nusnd_read_end = 0;

    // SIF hardware registers (MSCOM, SMCOM, MSFLG, SMFLG, CTRL)
    u32 m_mscom = 0;
    u32 m_smcom = 0;
    u32 m_msflg = 0;
    u32 m_smflg = 0x00020000u; // IOP SIF initialized bit set
    u32 m_ctrl  = 0;

    u64 m_rpc_handled = 0;
    u64 m_rpc_unhandled = 0;
};

} // namespace tf::ps2
