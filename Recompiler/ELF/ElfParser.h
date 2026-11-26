#pragma once

#include "../Common/Types.h"
#include <string>
#include <vector>

namespace tf::ps2 {

struct ElfProgramHeader {
    u32 type = 0;      // PT_LOAD = 1
    u32 offset = 0;    // File offset
    u32 vaddr = 0;     // Virtual address in EE RAM
    u32 paddr = 0;
    u32 filesz = 0;
    u32 memsz = 0;
    u32 flags = 0;     // PF_X=1, PF_W=2, PF_R=4
    u32 align = 0;
    std::vector<u8> data; // Populated for PT_LOAD segments (size == filesz)

    bool is_load() const { return type == 1; }
    bool is_exec() const { return (flags & 1u) != 0; }
    bool is_write() const { return (flags & 2u) != 0; }
    bool is_read() const { return (flags & 4u) != 0; }
};

struct ElfSectionHeader {
    std::string name;
    u32 name_idx = 0;
    u32 type = 0;
    u32 flags = 0;     // SHF_WRITE=1, SHF_ALLOC=2, SHF_EXECINSTR=4
    u32 addr = 0;
    u32 offset = 0;
    u32 size = 0;
    u32 link = 0;
    u32 info = 0;
    u32 addralign = 0;
    u32 entsize = 0;

    bool is_exec() const { return (flags & 4u) != 0; }
    bool is_alloc() const { return (flags & 2u) != 0; }
};

struct ElfSymbol {
    std::string name;
    u32 value = 0;
    u32 size = 0;
    u8 info = 0;
    u8 other = 0;
    u16 shndx = 0;

    u8 bind() const { return info >> 4; }
    u8 type() const { return info & 0x0F; }
    bool is_function() const { return type() == 2; } // STT_FUNC
};

struct ElfRelocation {
    u32 offset = 0;
    u32 info = 0;
    s32 addend = 0;
    bool has_addend = false;

    u32 sym_index() const { return info >> 8; }
    u8 rel_type() const { return static_cast<u8>(info & 0xFF); }
};

struct MemoryRegion {
    u32 start_vaddr = 0;
    u32 size = 0;
    bool executable = false;
    bool writable = false;
    std::string name;
    std::vector<u8> bytes; // File-backed portion (memsz may be >= bytes.size())
};

struct Ps2ElfImage {
    bool valid = false;
    std::string error;

    u16 e_type = 0;
    u16 e_machine = 0;
    u32 e_version = 0;
    u32 e_entry = 0;
    u32 e_phoff = 0;
    u32 e_shoff = 0;
    u32 e_flags = 0;
    u16 e_ehsize = 0;
    u16 e_phentsize = 0;
    u16 e_phnum = 0;
    u16 e_shentsize = 0;
    u16 e_shnum = 0;
    u16 e_shstrndx = 0;

    bool is_stripped = true;
    std::string mips_arch_summary;
    std::string sha256;
    u32 lowest_vaddr = 0;
    u32 highest_vaddr = 0;

    std::vector<ElfProgramHeader> program_headers;
    std::vector<ElfSectionHeader> section_headers;
    std::vector<ElfSymbol> symbols;
    std::vector<ElfRelocation> relocations;
    std::vector<MemoryRegion> executable_regions;    // EE R5900 executable sections (.text)
    std::vector<MemoryRegion> vu_microcode_regions;  // VU0/VU1 microprogram sections (.vutext, .DVP.overlay.*)

    // Reads a 32-bit instruction or data word at a virtual address in loaded segments.
    bool read_vaddr_u32(u32 vaddr, u32& out_word) const;
    bool read_vaddr_bytes(u32 vaddr, u8* dst, std::size_t len) const;
    bool is_executable_vaddr(u32 vaddr) const;

    std::string format_summary() const;
};

class ElfParser {
public:
    static bool parse(const std::vector<u8>& bytes, Ps2ElfImage& out_elf, std::string& out_error);
};

} // namespace tf::ps2
