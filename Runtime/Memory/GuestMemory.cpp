#include "GuestMemory.h"
#include <algorithm>
#include <cstring>

namespace tf::ps2 {

GuestMemory::GuestMemory()
    : m_ram(kEeRamSize, 0),
      m_bios_region(kEeBiosSize, 0) {
    reset();
}

void GuestMemory::reset() {
    std::fill(m_ram.begin(), m_ram.end(), u8{0});
    m_scratchpad.fill(0);
    m_vu0_code.fill(0);
    m_vu0_data.fill(0);
    m_vu1_code.fill(0);
    m_vu1_data.fill(0);
    std::fill(m_bios_region.begin(), m_bios_region.end(), u8{0});
    m_invalid_accesses = 0;
    m_last_invalid_addr = 0;
}

u32 GuestMemory::mask_physical(u32 vaddr) {
    // Scratchpad virtual address (0x70000000 - 0x70003FFF) is preserved distinctly
    if ((vaddr & 0xFFFFC000u) == 0x70000000u) {
        return vaddr;
    }
    // Uncached (0x20000000) and Uncached Accelerated (0x30000000) mirrors of main RAM
    if (vaddr >= 0x20000000u && vaddr < 0x22000000u) {
        return vaddr - 0x20000000u;
    }
    if (vaddr >= 0x30000000u && vaddr < 0x32000000u) {
        return vaddr - 0x30000000u;
    }
    // KSEG0 (0x80000000..0x9FFFFFFF) and KSEG1 (0xA0000000..0xBFFFFFFF)
    if (vaddr >= 0x80000000u && vaddr < 0xC0000000u) {
        return vaddr & 0x1FFFFFFFu;
    }
    return vaddr;
}

bool GuestMemory::is_io_address(u32 paddr) const {
    return (paddr >= 0x10000000u && paddr < 0x11000000u) ||
           (paddr >= 0x12000000u && paddr < 0x12002000u);
}

u8* GuestMemory::translate_ptr(u32 vaddr, u32 access_size) {
    u32 paddr = mask_physical(vaddr);
    if (paddr < kEeRamSize && (kEeRamSize - paddr) >= access_size) {
        return m_ram.data() + paddr;
    }
    if (paddr >= 0x70000000u && paddr < 0x70000000u + kEeScratchpadSize) {
        u32 off = paddr - 0x70000000u;
        if (kEeScratchpadSize - off >= access_size) return m_scratchpad.data() + off;
    }
    if (paddr >= 0x11000000u && paddr < 0x11001000u && (0x11001000u - paddr) >= access_size) {
        return m_vu0_code.data() + (paddr - 0x11000000u);
    }
    if (paddr >= 0x11004000u && paddr < 0x11005000u && (0x11005000u - paddr) >= access_size) {
        return m_vu0_data.data() + (paddr - 0x11004000u);
    }
    if (paddr >= 0x11008000u && paddr < 0x1100C000u && (0x1100C000u - paddr) >= access_size) {
        return m_vu1_code.data() + (paddr - 0x11008000u);
    }
    if (paddr >= 0x1100C000u && paddr < 0x11010000u && (0x11010000u - paddr) >= access_size) {
        return m_vu1_data.data() + (paddr - 0x1100C000u);
    }
    if (paddr >= 0x1FC00000u && paddr < 0x1FC00000u + kEeBiosSize) {
        u32 off = paddr - 0x1FC00000u;
        if (kEeBiosSize - off >= access_size) return m_bios_region.data() + off;
    }
    return nullptr;
}

const u8* GuestMemory::translate_ptr(u32 vaddr, u32 access_size) const {
    return const_cast<GuestMemory*>(this)->translate_ptr(vaddr, access_size);
}

bool GuestMemory::write_block(u32 vaddr, const u8* src, std::size_t len) {
    if (!src || len == 0) return true;
    u8* dst = translate_ptr(vaddr, static_cast<u32>(len));
    if (!dst) return false;
    std::memcpy(dst, src, len);
    return true;
}

bool GuestMemory::zero_block(u32 vaddr, std::size_t len) {
    if (len == 0) return true;
    u8* dst = translate_ptr(vaddr, static_cast<u32>(len));
    if (!dst) return false;
    std::memset(dst, 0, len);
    return true;
}

bool GuestMemory::read_block(u32 vaddr, u8* dst, std::size_t len) const {
    if (!dst || len == 0) return true;
    const u8* src = translate_ptr(vaddr, static_cast<u32>(len));
    if (!src) return false;
    std::memcpy(dst, src, len);
    return true;
}

u8 GuestMemory::read8_slow(u32 vaddr) const {
    const u8* p = translate_ptr(vaddr, 1);
    if (p) return *p;
    u32 paddr = mask_physical(vaddr);
    if (is_io_address(paddr) && m_mmio.read32) {
        u32 w = m_mmio.read32(paddr & ~3u);
        return static_cast<u8>((w >> ((paddr & 3u) * 8u)) & 0xFFu);
    }
    m_invalid_accesses++;
    m_last_invalid_addr = vaddr;
    return 0;
}

u16 GuestMemory::read16_slow(u32 vaddr) const {
    const u8* p = translate_ptr(vaddr, 2);
    if (p) return read_le16(p);
    u32 paddr = mask_physical(vaddr);
    if (is_io_address(paddr) && m_mmio.read32) {
        u32 w = m_mmio.read32(paddr & ~3u);
        return static_cast<u16>((w >> ((paddr & 2u) * 8u)) & 0xFFFFu);
    }
    m_invalid_accesses++;
    m_last_invalid_addr = vaddr;
    return 0;
}

u32 GuestMemory::read32_slow(u32 vaddr) const {
    const u8* p = translate_ptr(vaddr, 4);
    if (p) return read_le32(p);
    u32 paddr = mask_physical(vaddr);
    if (is_io_address(paddr) && m_mmio.read32) {
        return m_mmio.read32(paddr);
    }
    m_invalid_accesses++;
    m_last_invalid_addr = vaddr;
    return 0;
}

u64 GuestMemory::read64_slow(u32 vaddr) const {
    const u8* p = translate_ptr(vaddr, 8);
    if (p) return read_le64(p);
    u32 paddr = mask_physical(vaddr);
    if (is_io_address(paddr)) {
        if (m_mmio.read64) return m_mmio.read64(paddr);
        if (m_mmio.read32) {
            return static_cast<u64>(m_mmio.read32(paddr)) |
                   (static_cast<u64>(m_mmio.read32(paddr + 4)) << 32);
        }
    }
    m_invalid_accesses++;
    m_last_invalid_addr = vaddr;
    return 0;
}

u128 GuestMemory::read128_slow(u32 vaddr) const {
    u32 aligned = vaddr & ~0xFu; // R5900 LQ/LQC2 masks low 4 bits
    const u8* p = translate_ptr(aligned, 16);
    if (p) return u128(read_le64(p), read_le64(p + 8));
    u32 paddr = mask_physical(aligned);
    if (is_io_address(paddr) && m_mmio.read128) {
        return m_mmio.read128(paddr);
    }
    m_invalid_accesses++;
    m_last_invalid_addr = vaddr;
    return u128{};
}

void GuestMemory::write8_slow(u32 vaddr, u8 val) {
    u8* p = translate_ptr(vaddr, 1);
    if (p) {
        *p = val;
        return;
    }
    u32 paddr = mask_physical(vaddr);
    if (is_io_address(paddr) && m_mmio.write32) {
        m_mmio.write32(paddr & ~3u, val);
        return;
    }
    m_invalid_accesses++;
    m_last_invalid_addr = vaddr;
}

void GuestMemory::write16_slow(u32 vaddr, u16 val) {
    u8* p = translate_ptr(vaddr, 2);
    if (p) {
        write_le16(p, val);
        return;
    }
    u32 paddr = mask_physical(vaddr);
    if (is_io_address(paddr) && m_mmio.write32) {
        m_mmio.write32(paddr & ~3u, val);
        return;
    }
    m_invalid_accesses++;
    m_last_invalid_addr = vaddr;
}

void GuestMemory::write32_slow(u32 vaddr, u32 val) {
    u8* p = translate_ptr(vaddr, 4);
    if (p) {
        write_le32(p, val);
        return;
    }
    u32 paddr = mask_physical(vaddr);
    if (is_io_address(paddr) && m_mmio.write32) {
        m_mmio.write32(paddr, val);
        return;
    }
    m_invalid_accesses++;
    m_last_invalid_addr = vaddr;
}

void GuestMemory::write64_slow(u32 vaddr, u64 val) {
    u8* p = translate_ptr(vaddr, 8);
    if (p) {
        write_le64(p, val);
        return;
    }
    u32 paddr = mask_physical(vaddr);
    if (is_io_address(paddr)) {
        if (m_mmio.write64) {
            m_mmio.write64(paddr, val);
            return;
        }
        if (m_mmio.write32) {
            m_mmio.write32(paddr, static_cast<u32>(val & 0xFFFFFFFFULL));
            m_mmio.write32(paddr + 4, static_cast<u32>(val >> 32));
            return;
        }
    }
    m_invalid_accesses++;
    m_last_invalid_addr = vaddr;
}

void GuestMemory::write128_slow(u32 vaddr, u128 val) {
    u32 aligned = vaddr & ~0xFu; // R5900 SQ/SQC2 masks low 4 bits
    u8* p = translate_ptr(aligned, 16);
    if (p) {
        write_le64(p, val.lo);
        write_le64(p + 8, val.hi);
        return;
    }
    u32 paddr = mask_physical(aligned);
    if (is_io_address(paddr) && m_mmio.write128) {
        m_mmio.write128(paddr, val);
        return;
    }
    m_invalid_accesses++;
    m_last_invalid_addr = vaddr;
}

} // namespace tf::ps2
