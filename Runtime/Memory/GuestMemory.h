#pragma once

#include "../../Recompiler/Common/Types.h"
#include <array>
#include <functional>
#include <vector>

namespace tf::ps2 {

constexpr u32 kEeRamSize        = 32U * 1024U * 1024U; // 32 MiB Main EE RAM
constexpr u32 kEeScratchpadSize = 16U * 1024U;         // 16 KiB Scratchpad (0x70000000)
constexpr u32 kEeBiosSize       = 4U * 1024U * 1024U;  // 4 MiB Kernel/BIOS mirror region

struct MmioHandler {
    std::function<u32(u32 addr)> read32;
    std::function<void(u32 addr, u32 val)> write32;
    std::function<u64(u32 addr)> read64;
    std::function<void(u32 addr, u64 val)> write64;
    std::function<u128(u32 addr)> read128;
    std::function<void(u32 addr, u128 val)> write128;
};

class GuestMemory {
public:
    GuestMemory();

    void reset();

    // Direct accessors to backing memory
    u8* ram() { return m_ram.data(); }
    const u8* ram() const { return m_ram.data(); }
    u8* scratchpad() { return m_scratchpad.data(); }
    const u8* scratchpad() const { return m_scratchpad.data(); }
    u8* vu0_code() { return m_vu0_code.data(); }
    u8* vu0_data() { return m_vu0_data.data(); }
    u8* vu1_code() { return m_vu1_code.data(); }
    u8* vu1_data() { return m_vu1_data.data(); }

    void set_mmio_handler(MmioHandler handler) { m_mmio = std::move(handler); }

    // Translate PS2 EE virtual address (KUSEG / KSEG0 / KSEG1 / uncached / accelerated mirrors)
    static u32 mask_physical(u32 vaddr);
    u8* translate_ptr(u32 vaddr, u32 access_size);
    const u8* translate_ptr(u32 vaddr, u32 access_size) const;

    // Load a block into guest RAM (e.g. ELF PT_LOAD segments)
    bool write_block(u32 vaddr, const u8* src, std::size_t len);
    bool zero_block(u32 vaddr, std::size_t len);
    bool read_block(u32 vaddr, u8* dst, std::size_t len) const;

    // Standard memory reads/writes (fast-path inlined for 32 MiB main EE RAM; layout ABI-preserved)
    inline u8 read8(u32 vaddr) const {
        if (vaddr < kEeRamSize) return m_ram[vaddr];
        return read8_slow(vaddr);
    }
    inline u16 read16(u32 vaddr) const {
        if (vaddr <= kEeRamSize - 2u) return read_le16(m_ram.data() + vaddr);
        return read16_slow(vaddr);
    }
    inline u32 read32(u32 vaddr) const {
        if (vaddr <= kEeRamSize - 4u) return read_le32(m_ram.data() + vaddr);
        return read32_slow(vaddr);
    }
    inline u64 read64(u32 vaddr) const {
        if (vaddr <= kEeRamSize - 8u) return read_le64(m_ram.data() + vaddr);
        return read64_slow(vaddr);
    }
    inline u128 read128(u32 vaddr) const {
        const u32 aligned = vaddr & ~0xFu;
        if (aligned < kEeRamSize) {
            const u8* p = m_ram.data() + aligned;
            return u128(read_le64(p), read_le64(p + 8));
        }
        return read128_slow(vaddr);
    }

    inline void write8(u32 vaddr, u8 val) {
        if (vaddr < kEeRamSize) { m_ram[vaddr] = val; return; }
        write8_slow(vaddr, val);
    }
    inline void write16(u32 vaddr, u16 val) {
        if (vaddr <= kEeRamSize - 2u) { write_le16(m_ram.data() + vaddr, val); return; }
        write16_slow(vaddr, val);
    }
    inline void write32(u32 vaddr, u32 val) {
        if (vaddr <= kEeRamSize - 4u) { write_le32(m_ram.data() + vaddr, val); return; }
        write32_slow(vaddr, val);
    }
    inline void write64(u32 vaddr, u64 val) {
        if (vaddr <= kEeRamSize - 8u) { write_le64(m_ram.data() + vaddr, val); return; }
        write64_slow(vaddr, val);
    }
    inline void write128(u32 vaddr, u128 val) {
        const u32 aligned = vaddr & ~0xFu;
        if (aligned < kEeRamSize) {
            u8* p = m_ram.data() + aligned;
            write_le64(p, val.lo);
            write_le64(p + 8, val.hi);
            return;
        }
        write128_slow(vaddr, val);
    }

    u64 invalid_access_count() const { return m_invalid_accesses; }
    u32 last_invalid_address() const { return m_last_invalid_addr; }

private:
    bool is_io_address(u32 paddr) const;
    u8   read8_slow(u32 vaddr) const;
    u16  read16_slow(u32 vaddr) const;
    u32  read32_slow(u32 vaddr) const;
    u64  read64_slow(u32 vaddr) const;
    u128 read128_slow(u32 vaddr) const;
    void write8_slow(u32 vaddr, u8 val);
    void write16_slow(u32 vaddr, u16 val);
    void write32_slow(u32 vaddr, u32 val);
    void write64_slow(u32 vaddr, u64 val);
    void write128_slow(u32 vaddr, u128 val);

    std::vector<u8> m_ram;
    std::array<u8, kEeScratchpadSize> m_scratchpad{};
    std::array<u8, 4096> m_vu0_code{};
    std::array<u8, 4096> m_vu0_data{};
    std::array<u8, 16384> m_vu1_code{};
    std::array<u8, 16384> m_vu1_data{};
    std::vector<u8> m_bios_region;
    MmioHandler m_mmio;

    mutable u64 m_invalid_accesses = 0;
    mutable u32 m_last_invalid_addr = 0;
};

} // namespace tf::ps2
