#pragma once

#include "../Common/Types.h"
#include <cstdio>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace tf::ps2 {

constexpr u32 kIsoSectorSize = 2048;

struct IsoFileEntry {
    std::string path;          // Normalized uppercase path e.g. "/SYSTEM.CNF" or "/SLUS_215.61"
    std::string name;          // Base filename without ;1 version suffix
    u32 lba = 0;               // Logical Block Address
    u64 size = 0;              // File size in bytes
    bool is_directory = false;
    u8 flags = 0;
};

struct SystemCnfInfo {
    std::string raw_text;
    std::string boot2_raw;      // e.g. "cdrom0:\SLUS_215.61;1"
    std::string boot2_iso_path; // Normalized ISO path e.g. "/SLUS_215.61"
    std::string version;        // VER = ...
    std::string vmode;          // VMODE = NTSC / PAL
    std::string serial;         // Extracted disc serial e.g. "SLUS_215.61"
    std::string region;         // Detected region (USA / Europe / Japan / Unknown)
    std::unordered_map<std::string, std::string> entries;
};

enum class ExecutableFormat {
    Unknown,
    Ps2Elf32Mips,
    PackedPs2Packer,
    PackedUpxOrCustom,
    CompressedGzip,
    CompressedZlib,
    InvalidOrEmpty
};

struct IsoInspectionReport {
    bool valid = false;
    std::string error;

    std::string iso_path;
    u64 iso_file_size = 0;
    u32 sector_size = kIsoSectorSize;
    u32 volume_space_sectors = 0;

    std::string system_identifier;
    std::string volume_identifier;
    std::string publisher_identifier;
    std::string application_identifier;
    std::string volume_fingerprint_sha256; // Hash of PVD + SYSTEM.CNF + root dir

    bool has_system_cnf = false;
    SystemCnfInfo system_cnf;

    bool has_boot_executable = false;
    IsoFileEntry boot_executable_entry;
    ExecutableFormat executable_format = ExecutableFormat::Unknown;
    std::string executable_format_details;
    std::string executable_sha256;

    std::size_t total_files = 0;
    std::size_t total_directories = 0;
    std::vector<std::string> warnings;
    std::vector<IsoFileEntry> notable_files; // IRX, IOPRP, PSS, BIN, PAK, etc.

    std::string format_report() const;
};

// Streaming read-only file handle backed by an ISO9660 file entry.
class IsoFileStream {
public:
    IsoFileStream(class IsoReader* reader, IsoFileEntry entry);

    bool valid() const { return m_reader != nullptr && !m_entry.is_directory; }
    u64 size() const { return m_entry.size; }
    u64 tell() const { return m_pos; }
    bool seek(u64 offset);
    std::size_t read(u8* dst, std::size_t bytes);

private:
    class IsoReader* m_reader = nullptr;
    IsoFileEntry m_entry;
    u64 m_pos = 0;
};

class IsoReader {
public:
    IsoReader();
    ~IsoReader();

    IsoReader(const IsoReader&) = delete;
    IsoReader& operator=(const IsoReader&) = delete;

    // Opens an ISO file read-only without loading the whole image into RAM.
    bool open(const std::string& path, std::string& out_error);
    void close();
    bool is_open() const { return m_file != nullptr; }

    // 64-bit safe sector / byte reading directly from the underlying ISO.
    bool read_bytes_at(u64 offset, u8* dst, std::size_t count) const;
    bool read_sectors(u32 lba, u32 sector_count, u8* dst) const;

    // Filesystem queries
    const std::vector<IsoFileEntry>& all_entries() const { return m_entries; }
    const IsoFileEntry* find_file(const std::string& path) const;
    bool read_file(const IsoFileEntry& entry, std::vector<u8>& out_data) const;
    bool read_file_slice(const IsoFileEntry& entry, u64 offset, u8* dst, std::size_t count, std::size_t* out_read = nullptr) const;
    std::unique_ptr<IsoFileStream> open_stream(const std::string& path);

    // Produces the full inspection report including SYSTEM.CNF, BOOT2, executable detection, and fingerprint.
    IsoInspectionReport inspect() const;

    u64 file_size() const { return m_file_size; }
    u32 sector_size() const { return m_sector_size; }
    const std::string& path() const { return m_path; }

    static std::string normalize_iso_path(std::string_view raw);
    static bool parse_system_cnf(std::string_view content, SystemCnfInfo& out_info, std::string& out_error);
    static ExecutableFormat detect_executable_format(const std::vector<u8>& data, std::string& out_details);
    static const char* executable_format_name(ExecutableFormat fmt);

private:
    bool parse_pvd(std::string& out_error);
    bool traverse_directory(u32 lba, u32 size, const std::string& parent_path, int depth, std::vector<std::string>& warnings);

    mutable std::FILE* m_file = nullptr;
    std::string m_path;
    u64 m_file_size = 0;
    u32 m_sector_size = kIsoSectorSize;
    u32 m_volume_space_sectors = 0;

    std::string m_system_id;
    std::string m_volume_id;
    std::string m_publisher_id;
    std::string m_application_id;
    std::array<u8, kIsoSectorSize> m_pvd_raw{};

    u32 m_root_lba = 0;
    u32 m_root_size = 0;

    std::vector<IsoFileEntry> m_entries;
    std::unordered_map<std::string, std::size_t> m_path_index;
    std::vector<std::string> m_parse_warnings;
};

} // namespace tf::ps2
