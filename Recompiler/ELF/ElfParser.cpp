#include "ElfParser.h"
#include "../Common/Sha256.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <sstream>

namespace tf::ps2 {

namespace {

constexpr u32 SHT_SYMTAB = 2;
constexpr u32 SHT_RELA   = 4;
constexpr u32 SHT_REL    = 9;
constexpr u32 SHT_DYNSYM = 11;

std::string decode_mips_flags(u32 flags) {
    std::ostringstream oss;
    u32 arch = (flags >> 28) & 0xF;
    switch (arch) {
        case 0: oss << "MIPS-I"; break;
        case 1: oss << "MIPS-II"; break;
        case 2: oss << "MIPS-III"; break;
        case 3: oss << "MIPS-IV"; break;
        case 4: oss << "MIPS-V"; break;
        case 5: oss << "MIPS32"; break;
        case 6: oss << "MIPS64"; break;
        default: oss << "MIPS(arch=" << arch << ")"; break;
    }
    u32 mach = (flags >> 16) & 0xFF;
    if (mach == 0x92) {
        oss << " / Sony R5900 (Emotion Engine)";
    } else if (mach != 0) {
        oss << " / mach=" << hex32(mach);
    }
    if (flags & 0x00001000u) oss << " [EABI32]";
    if (flags & 0x00002000u) oss << " [EABI64]";
    if (flags & 0x00000004u) oss << " [CPIC]";
    if (flags & 0x00000002u) oss << " [PIC]";
    return oss.str();
}

} // namespace

bool Ps2ElfImage::read_vaddr_bytes(u32 vaddr, u8* dst, std::size_t len) const {
    if (!dst || len == 0) return false;
    for (const auto& ph : program_headers) {
        if (!ph.is_load() || ph.memsz == 0) continue;
        if (vaddr >= ph.vaddr && vaddr < ph.vaddr + ph.memsz) {
            u32 offset = vaddr - ph.vaddr;
            if (offset + len > ph.memsz) return false;
            std::size_t file_avail = (offset < ph.data.size()) ? std::min<std::size_t>(len, ph.data.size() - offset) : 0;
            if (file_avail > 0) {
                std::memcpy(dst, ph.data.data() + offset, file_avail);
            }
            if (file_avail < len) {
                std::memset(dst + file_avail, 0, len - file_avail);
            }
            return true;
        }
    }
    return false;
}

bool Ps2ElfImage::read_vaddr_u32(u32 vaddr, u32& out_word) const {
    u8 buf[4];
    if (!read_vaddr_bytes(vaddr, buf, 4)) return false;
    out_word = read_le32(buf);
    return true;
}

bool Ps2ElfImage::is_executable_vaddr(u32 vaddr) const {
    for (const auto& r : executable_regions) {
        if (vaddr >= r.start_vaddr && vaddr + 4 <= r.start_vaddr + r.size) {
            return true;
        }
    }
    return false;
}

bool ElfParser::parse(const std::vector<u8>& bytes, Ps2ElfImage& out_elf, std::string& out_error) {
    out_elf = Ps2ElfImage{};

    if (bytes.size() < 52) {
        out_error = "ELF binary too small (< 52 bytes)";
        out_elf.error = out_error;
        return false;
    }

    if (bytes[0] != 0x7F || bytes[1] != 'E' || bytes[2] != 'L' || bytes[3] != 'F') {
        out_error = "Invalid ELF magic header";
        out_elf.error = out_error;
        return false;
    }

    if (bytes[4] != 1) {
        out_error = "Unsupported ELF class (expected ELFCLASS32=1, got " + std::to_string(bytes[4]) + ")";
        out_elf.error = out_error;
        return false;
    }

    if (bytes[5] != 1) {
        out_error = "Unsupported ELF endianness (expected ELFDATA2LSB=1, got " + std::to_string(bytes[5]) + ")";
        out_elf.error = out_error;
        return false;
    }

    out_elf.e_type      = read_le16(bytes.data() + 16);
    out_elf.e_machine   = read_le16(bytes.data() + 18);
    out_elf.e_version   = read_le32(bytes.data() + 20);
    out_elf.e_entry     = read_le32(bytes.data() + 24);
    out_elf.e_phoff     = read_le32(bytes.data() + 28);
    out_elf.e_shoff     = read_le32(bytes.data() + 32);
    out_elf.e_flags     = read_le32(bytes.data() + 36);
    out_elf.e_ehsize    = read_le16(bytes.data() + 40);
    out_elf.e_phentsize = read_le16(bytes.data() + 42);
    out_elf.e_phnum     = read_le16(bytes.data() + 44);
    out_elf.e_shentsize = read_le16(bytes.data() + 46);
    out_elf.e_shnum     = read_le16(bytes.data() + 48);
    out_elf.e_shstrndx  = read_le16(bytes.data() + 50);

    if (out_elf.e_machine != 8) { // EM_MIPS
        out_error = "Unsupported ELF machine (expected EM_MIPS=8, got " + std::to_string(out_elf.e_machine) + ")";
        out_elf.error = out_error;
        return false;
    }

    out_elf.sha256 = Sha256::hash_bytes(bytes.data(), bytes.size());
    out_elf.mips_arch_summary = decode_mips_flags(out_elf.e_flags);

    // Parse program headers
    u32 lo_va = std::numeric_limits<u32>::max();
    u32 hi_va = 0;

    if (out_elf.e_phnum > 0) {
        if (out_elf.e_phentsize < 32) {
            out_error = "Invalid program header entry size: " + std::to_string(out_elf.e_phentsize);
            out_elf.error = out_error;
            return false;
        }
        u64 ph_end = static_cast<u64>(out_elf.e_phoff) + static_cast<u64>(out_elf.e_phnum) * out_elf.e_phentsize;
        if (ph_end > bytes.size()) {
            out_error = "Program header table extends beyond ELF file size";
            out_elf.error = out_error;
            return false;
        }

        out_elf.program_headers.reserve(out_elf.e_phnum);
        for (u16 i = 0; i < out_elf.e_phnum; ++i) {
            const u8* p = bytes.data() + out_elf.e_phoff + i * out_elf.e_phentsize;
            ElfProgramHeader ph;
            ph.type   = read_le32(p + 0);
            ph.offset = read_le32(p + 4);
            ph.vaddr  = read_le32(p + 8);
            ph.paddr  = read_le32(p + 12);
            ph.filesz = read_le32(p + 16);
            ph.memsz  = read_le32(p + 20);
            ph.flags  = read_le32(p + 24);
            ph.align  = read_le32(p + 28);

            if (ph.is_load()) {
                if (static_cast<u64>(ph.offset) + ph.filesz > bytes.size()) {
                    out_error = "PT_LOAD segment " + std::to_string(i) + " extends past EOF";
                    out_elf.error = out_error;
                    return false;
                }
                ph.data.assign(bytes.begin() + ph.offset, bytes.begin() + ph.offset + ph.filesz);
                if (ph.memsz > 0) {
                    lo_va = std::min(lo_va, ph.vaddr);
                    hi_va = std::max(hi_va, ph.vaddr + ph.memsz);
                }
            }
            out_elf.program_headers.push_back(std::move(ph));
        }
    }

    if (lo_va != std::numeric_limits<u32>::max()) {
        out_elf.lowest_vaddr = lo_va;
        out_elf.highest_vaddr = hi_va;
    }

    // Parse section headers if present (note: many retail PS2 ELFs strip section headers or only keep shnum==0)
    if (out_elf.e_shnum > 0 && out_elf.e_shoff > 0 && out_elf.e_shentsize >= 40) {
        u64 sh_end = static_cast<u64>(out_elf.e_shoff) + static_cast<u64>(out_elf.e_shnum) * out_elf.e_shentsize;
        if (sh_end <= bytes.size()) {
            out_elf.section_headers.reserve(out_elf.e_shnum);
            for (u16 i = 0; i < out_elf.e_shnum; ++i) {
                const u8* s = bytes.data() + out_elf.e_shoff + i * out_elf.e_shentsize;
                ElfSectionHeader sh;
                sh.name_idx  = read_le32(s + 0);
                sh.type      = read_le32(s + 4);
                sh.flags     = read_le32(s + 8);
                sh.addr      = read_le32(s + 12);
                sh.offset    = read_le32(s + 16);
                sh.size      = read_le32(s + 20);
                sh.link      = read_le32(s + 24);
                sh.info      = read_le32(s + 28);
                sh.addralign = read_le32(s + 32);
                sh.entsize   = read_le32(s + 36);
                out_elf.section_headers.push_back(std::move(sh));
            }

            // Resolve section names via shstrtab
            if (out_elf.e_shstrndx < out_elf.section_headers.size()) {
                const auto& strsec = out_elf.section_headers[out_elf.e_shstrndx];
                if (static_cast<u64>(strsec.offset) + strsec.size <= bytes.size()) {
                    const char* strtab = reinterpret_cast<const char*>(bytes.data() + strsec.offset);
                    for (auto& sh : out_elf.section_headers) {
                        if (sh.name_idx < strsec.size) {
                            const char* p = strtab + sh.name_idx;
                            std::size_t maxlen = strsec.size - sh.name_idx;
                            sh.name = std::string(p, strnlen(p, maxlen));
                        }
                    }
                }
            }

            // Parse symbol tables and relocations
            for (const auto& sh : out_elf.section_headers) {
                if ((sh.type == SHT_SYMTAB || sh.type == SHT_DYNSYM) && sh.entsize >= 16) {
                    if (static_cast<u64>(sh.offset) + sh.size <= bytes.size() &&
                        sh.link < out_elf.section_headers.size()) {
                        const auto& strsec = out_elf.section_headers[sh.link];
                        bool has_str = (static_cast<u64>(strsec.offset) + strsec.size <= bytes.size());
                        const char* strtab = has_str ? reinterpret_cast<const char*>(bytes.data() + strsec.offset) : nullptr;

                        u32 count = sh.size / sh.entsize;
                        for (u32 i = 0; i < count; ++i) {
                            const u8* e = bytes.data() + sh.offset + i * sh.entsize;
                            ElfSymbol sym;
                            u32 st_name = read_le32(e + 0);
                            sym.value = read_le32(e + 4);
                            sym.size  = read_le32(e + 8);
                            sym.info  = e[12];
                            sym.other = e[13];
                            sym.shndx = read_le16(e + 14);
                            if (strtab && st_name < strsec.size) {
                                sym.name = std::string(strtab + st_name, strnlen(strtab + st_name, strsec.size - st_name));
                            }
                            out_elf.symbols.push_back(std::move(sym));
                        }
                        if (!out_elf.symbols.empty()) {
                            out_elf.is_stripped = false;
                        }
                    }
                } else if (sh.type == SHT_REL && sh.entsize >= 8) {
                    if (static_cast<u64>(sh.offset) + sh.size <= bytes.size()) {
                        u32 count = sh.size / sh.entsize;
                        for (u32 i = 0; i < count; ++i) {
                            const u8* r = bytes.data() + sh.offset + i * sh.entsize;
                            ElfRelocation rel;
                            rel.offset = read_le32(r + 0);
                            rel.info   = read_le32(r + 4);
                            rel.has_addend = false;
                            out_elf.relocations.push_back(rel);
                        }
                    }
                } else if (sh.type == SHT_RELA && sh.entsize >= 12) {
                    if (static_cast<u64>(sh.offset) + sh.size <= bytes.size()) {
                        u32 count = sh.size / sh.entsize;
                        for (u32 i = 0; i < count; ++i) {
                            const u8* r = bytes.data() + sh.offset + i * sh.entsize;
                            ElfRelocation rel;
                            rel.offset = read_le32(r + 0);
                            rel.info   = read_le32(r + 4);
                            rel.addend = static_cast<s32>(read_le32(r + 8));
                            rel.has_addend = true;
                            out_elf.relocations.push_back(rel);
                        }
                    }
                }
            }
        }
    }

    // Separate EE R5900 executable sections (.text) from Sony DVP Vector Unit microcode (.vutext, .DVP.overlay.*)
    for (const auto& sh : out_elf.section_headers) {
        if (sh.is_exec() && sh.size > 0 && static_cast<u64>(sh.offset) + sh.size <= bytes.size()) {
            bool is_vu_sec = (sh.name == ".vutext") ||
                             (sh.name.rfind(".vutext", 0) == 0) ||
                             (sh.name.rfind(".DVP.overlay", 0) == 0);
            MemoryRegion reg;
            reg.start_vaddr = sh.addr;
            reg.size = sh.size;
            reg.executable = true;
            reg.writable = (sh.flags & 1u) != 0;
            reg.name = sh.name.empty() ? ".text" : sh.name;
            reg.bytes.assign(bytes.begin() + sh.offset, bytes.begin() + sh.offset + sh.size);
            if (is_vu_sec) {
                out_elf.vu_microcode_regions.push_back(std::move(reg));
            } else if (sh.is_alloc()) {
                out_elf.executable_regions.push_back(std::move(reg));
            }
        }
    }

    if (out_elf.executable_regions.empty()) {
        for (std::size_t i = 0; i < out_elf.program_headers.size(); ++i) {
            const auto& ph = out_elf.program_headers[i];
            if (ph.is_load() && ph.is_exec() && !ph.data.empty()) {
                MemoryRegion reg;
                reg.start_vaddr = ph.vaddr;
                reg.size = static_cast<u32>(ph.data.size());
                reg.executable = true;
                reg.writable = ph.is_write();
                reg.name = "PT_LOAD_" + std::to_string(i);
                reg.bytes = ph.data;
                out_elf.executable_regions.push_back(std::move(reg));
            }
        }
    }

    out_elf.valid = true;
    return true;
}

std::string Ps2ElfImage::format_summary() const {
    std::ostringstream oss;
    oss << "=== PS2 ELF32 ANALYSIS SUMMARY ===\n";
    oss << "Valid               : " << (valid ? "YES" : ("NO (" + error + ")")) << "\n";
    oss << "ELF SHA-256         : " << sha256 << "\n";
    oss << "Entry Point         : " << hex32(e_entry) << "\n";
    oss << "Architecture        : " << mips_arch_summary << " (flags=" << hex32(e_flags) << ")\n";
    oss << "Virtual Span        : " << hex32(lowest_vaddr) << " .. " << hex32(highest_vaddr) << "\n";
    oss << "Symbols             : " << symbols.size() << (is_stripped ? " (STRIPPED retail ELF)" : " (symbols present)") << "\n";
    oss << "Relocations         : " << relocations.size() << "\n";

    oss << "\n--- Program Headers (" << program_headers.size() << ") ---\n";
    for (std::size_t i = 0; i < program_headers.size(); ++i) {
        const auto& ph = program_headers[i];
        oss << "  [" << i << "] type=" << ph.type
            << " vaddr=" << hex32(ph.vaddr)
            << " filesz=" << hex32(ph.filesz)
            << " memsz=" << hex32(ph.memsz)
            << " flags=" << (ph.is_read() ? "R" : "-") << (ph.is_write() ? "W" : "-") << (ph.is_exec() ? "X" : "-")
            << "\n";
    }

    if (!section_headers.empty()) {
        oss << "\n--- Section Headers (" << section_headers.size() << ") ---\n";
        for (std::size_t i = 0; i < section_headers.size(); ++i) {
            const auto& sh = section_headers[i];
            if (sh.size == 0 && sh.name.empty()) continue;
            oss << "  [" << i << "] " << sh.name
                << " addr=" << hex32(sh.addr)
                << " size=" << hex32(sh.size)
                << " flags=" << (sh.is_alloc() ? "A" : "-") << ((sh.flags & 1u) ? "W" : "-") << (sh.is_exec() ? "X" : "-")
                << "\n";
        }
    }

    oss << "\n--- EE Executable Regions (" << executable_regions.size() << ") ---\n";
    for (const auto& r : executable_regions) {
        oss << "  " << r.name << " : " << hex32(r.start_vaddr) << " .. " << hex32(r.start_vaddr + r.size)
            << " (" << r.size << " bytes, " << (r.size / 4) << " R5900 instructions)\n";
    }
    if (!vu_microcode_regions.empty()) {
        u64 total_vu_bytes = 0;
        for (const auto& r : vu_microcode_regions) total_vu_bytes += r.size;
        oss << "\n--- VU Microcode & DVP Overlay Sections (" << vu_microcode_regions.size()
            << " sections, " << total_vu_bytes << " bytes total) ---\n";
        for (std::size_t i = 0; i < std::min<std::size_t>(vu_microcode_regions.size(), 5); ++i) {
            const auto& r = vu_microcode_regions[i];
            oss << "  " << r.name << " : addr=" << hex32(r.start_vaddr) << " size=" << r.size << " bytes\n";
        }
        if (vu_microcode_regions.size() > 5) {
            oss << "  ... (" << (vu_microcode_regions.size() - 5) << " more VU/DVP overlay sections)\n";
        }
    }

    return oss.str();
}

} // namespace tf::ps2
