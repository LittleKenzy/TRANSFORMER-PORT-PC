#include "IsoReader.h"
#include "../Common/Sha256.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <sstream>

#ifdef _WIN32
#include <io.h>
#endif

namespace tf::ps2 {

namespace {

int seek64(std::FILE* f, u64 offset) {
#ifdef _WIN32
    return _fseeki64(f, static_cast<__int64>(offset), SEEK_SET);
#else
    return fseeko(f, static_cast<off_t>(offset), SEEK_SET);
#endif
}

u64 get_file_size64(std::FILE* f) {
#ifdef _WIN32
    if (_fseeki64(f, 0, SEEK_END) != 0) return 0;
    __int64 pos = _ftelli64(f);
    _fseeki64(f, 0, SEEK_SET);
    return pos > 0 ? static_cast<u64>(pos) : 0;
#else
    if (fseeko(f, 0, SEEK_END) != 0) return 0;
    off_t pos = ftello(f);
    fseeko(f, 0, SEEK_SET);
    return pos > 0 ? static_cast<u64>(pos) : 0;
#endif
}

std::string trim_spaces(const u8* buf, std::size_t len) {
    std::string s(reinterpret_cast<const char*>(buf), len);
    while (!s.empty() && (s.back() == ' ' || s.back() == '\0' || s.back() == '\r' || s.back() == '\n')) {
        s.pop_back();
    }
    std::size_t start = 0;
    while (start < s.size() && (s[start] == ' ' || s[start] == '\t')) ++start;
    return s.substr(start);
}

std::string trim_str(std::string_view sv) {
    std::size_t a = 0;
    while (a < sv.size() && std::isspace(static_cast<unsigned char>(sv[a]))) ++a;
    std::size_t b = sv.size();
    while (b > a && std::isspace(static_cast<unsigned char>(sv[b - 1]))) --b;
    return std::string(sv.substr(a, b - a));
}

std::string to_upper(std::string_view sv) {
    std::string out(sv);
    for (char& c : out) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return out;
}

std::string detect_region_from_serial(const std::string& serial, const std::string& vmode) {
    std::string u = to_upper(serial);
    if (u.rfind("SLUS", 0) == 0 || u.rfind("SCUS", 0) == 0) return "USA (NTSC-U)";
    if (u.rfind("SLES", 0) == 0 || u.rfind("SCES", 0) == 0) return "Europe (PAL)";
    if (u.rfind("SLPS", 0) == 0 || u.rfind("SLPM", 0) == 0 || u.rfind("SCPS", 0) == 0) return "Japan (NTSC-J)";
    if (u.rfind("SLAJ", 0) == 0 || u.rfind("SCAJ", 0) == 0) return "Asia";
    if (u.rfind("SLKA", 0) == 0 || u.rfind("SCKA", 0) == 0) return "Korea";
    if (to_upper(vmode) == "PAL") return "PAL (Region from VMODE)";
    if (to_upper(vmode) == "NTSC") return "NTSC (Region from VMODE)";
    return "Unknown";
}

bool is_notable_extension(const std::string& upper_name) {
    static const char* kExts[] = {
        ".IRX", ".IMG", ".ELF", ".CNF", ".PSS", ".SFD", ".BIK",
        ".PAK", ".BIN", ".DAT", ".WAD", ".BIG", ".HOG", ".AFS", ".OVL"
    };
    for (const char* ext : kExts) {
        std::size_t elen = std::strlen(ext);
        if (upper_name.size() >= elen &&
            upper_name.compare(upper_name.size() - elen, elen, ext) == 0) {
            return true;
        }
    }
    return upper_name.find("IOPRP") != std::string::npos ||
           upper_name.find("SLUS_") != std::string::npos ||
           upper_name.find("SLES_") != std::string::npos;
}

} // namespace

IsoFileStream::IsoFileStream(IsoReader* reader, IsoFileEntry entry)
    : m_reader(reader), m_entry(std::move(entry)), m_pos(0) {}

bool IsoFileStream::seek(u64 offset) {
    if (!valid() || offset > m_entry.size) return false;
    m_pos = offset;
    return true;
}

std::size_t IsoFileStream::read(u8* dst, std::size_t bytes) {
    if (!valid() || !dst || bytes == 0 || m_pos >= m_entry.size) return 0;
    std::size_t actual = 0;
    if (!m_reader->read_file_slice(m_entry, m_pos, dst, bytes, &actual)) return 0;
    m_pos += actual;
    return actual;
}

IsoReader::IsoReader() = default;

IsoReader::~IsoReader() {
    close();
}

void IsoReader::close() {
    if (m_file) {
        std::fclose(m_file);
        m_file = nullptr;
    }
    m_path.clear();
    m_file_size = 0;
    m_entries.clear();
    m_path_index.clear();
    m_parse_warnings.clear();
}

bool IsoReader::open(const std::string& path, std::string& out_error) {
    close();
    m_path = path;
    m_file = std::fopen(path.c_str(), "rb");
    if (!m_file) {
        out_error = "Failed to open ISO file read-only: " + path;
        return false;
    }

    m_file_size = get_file_size64(m_file);
    if (m_file_size < static_cast<u64>(kIsoSectorSize) * 17ULL) {
        out_error = "ISO image is too small to contain an ISO9660 Primary Volume Descriptor (size=" +
                    std::to_string(m_file_size) + " bytes)";
        close();
        return false;
    }

    if (!parse_pvd(out_error)) {
        close();
        return false;
    }

    if (!traverse_directory(m_root_lba, m_root_size, "", 0, m_parse_warnings)) {
        out_error = "Failed to parse root ISO9660 directory";
        close();
        return false;
    }

    return true;
}

bool IsoReader::read_bytes_at(u64 offset, u8* dst, std::size_t count) const {
    if (!m_file || !dst) return false;
    if (count == 0) return true;
    if (offset > m_file_size || count > m_file_size - offset) return false;
    if (seek64(m_file, offset) != 0) return false;
    return std::fread(dst, 1, count, m_file) == count;
}

bool IsoReader::read_sectors(u32 lba, u32 sector_count, u8* dst) const {
    u64 offset = static_cast<u64>(lba) * static_cast<u64>(m_sector_size);
    u64 bytes = static_cast<u64>(sector_count) * static_cast<u64>(m_sector_size);
    if (bytes > static_cast<u64>(SIZE_MAX)) return false;
    return read_bytes_at(offset, dst, static_cast<std::size_t>(bytes));
}

bool IsoReader::parse_pvd(std::string& out_error) {
    // Scan volume descriptors starting at LBA 16 up to LBA 32
    bool found_pvd = false;
    u8 sector[kIsoSectorSize];
    for (u32 lba = 16; lba < 32; ++lba) {
        if (!read_bytes_at(static_cast<u64>(lba) * kIsoSectorSize, sector, kIsoSectorSize)) {
            break;
        }
        if (std::memcmp(sector + 1, "CD001", 5) != 0) {
            if (lba == 16) {
                out_error = "Missing ISO9660 'CD001' signature at LBA 16";
                return false;
            }
            break;
        }
        u8 vd_type = sector[0];
        if (vd_type == 1) {
            std::memcpy(m_pvd_raw.data(), sector, kIsoSectorSize);
            found_pvd = true;
            break;
        }
        if (vd_type == 255) {
            break;
        }
    }

    if (!found_pvd) {
        out_error = "ISO9660 Primary Volume Descriptor (type 1) not found";
        return false;
    }

    m_system_id = trim_spaces(m_pvd_raw.data() + 8, 32);
    m_volume_id = trim_spaces(m_pvd_raw.data() + 40, 32);
    m_volume_space_sectors = read_le32(m_pvd_raw.data() + 80);
    m_sector_size = read_le16(m_pvd_raw.data() + 128);
    if (m_sector_size == 0) {
        m_parse_warnings.push_back("PVD logical block size was 0; defaulting to 2048");
        m_sector_size = kIsoSectorSize;
    } else if (m_sector_size != kIsoSectorSize) {
        out_error = "Unsupported ISO9660 logical block size: " + std::to_string(m_sector_size);
        return false;
    }

    m_publisher_id = trim_spaces(m_pvd_raw.data() + 318, 128);
    m_application_id = trim_spaces(m_pvd_raw.data() + 574, 128);

    // Root directory record sits at PVD offset 156 (34 bytes minimum)
    const u8* root_rec = m_pvd_raw.data() + 156;
    u8 root_len = root_rec[0];
    if (root_len < 34) {
        out_error = "Invalid root directory record length in PVD: " + std::to_string(root_len);
        return false;
    }

    m_root_lba = read_le32(root_rec + 2);
    m_root_size = read_le32(root_rec + 10);
    if (m_root_size == 0 || static_cast<u64>(m_root_lba) * m_sector_size >= m_file_size) {
        out_error = "Invalid root directory extent in PVD (LBA=" + std::to_string(m_root_lba) +
                    ", size=" + std::to_string(m_root_size) + ")";
        return false;
    }

    return true;
}

bool IsoReader::traverse_directory(u32 lba, u32 size, const std::string& parent_path, int depth, std::vector<std::string>& warnings) {
    if (depth > 16) {
        warnings.push_back("Directory recursion depth limit exceeded at " + parent_path);
        return true;
    }
    if (size == 0) return true;
    // Cap directory buffer sanity at 16MB to protect against corrupted records
    if (size > 16U * 1024U * 1024U) {
        warnings.push_back("Directory extent abnormally large (" + std::to_string(size) + " bytes) at " + parent_path);
        return false;
    }

    u64 offset = static_cast<u64>(lba) * m_sector_size;
    if (offset >= m_file_size) {
        warnings.push_back("Directory LBA out of bounds: " + std::to_string(lba));
        return false;
    }
    std::size_t to_read = static_cast<std::size_t>(std::min<u64>(size, m_file_size - offset));
    std::vector<u8> dir_buf(to_read);
    if (!read_bytes_at(offset, dir_buf.data(), to_read)) {
        warnings.push_back("Failed reading directory bytes at LBA " + std::to_string(lba));
        return false;
    }

    std::size_t pos = 0;
    std::vector<IsoFileEntry> subdirs;

    while (pos < dir_buf.size()) {
        u8 rec_len = dir_buf[pos];
        if (rec_len == 0) {
            // Records never cross sector boundaries; skip to next sector
            std::size_t next_sector = ((pos / m_sector_size) + 1) * m_sector_size;
            if (next_sector <= pos) break;
            pos = next_sector;
            continue;
        }
        if (rec_len < 33 || pos + rec_len > dir_buf.size()) {
            warnings.push_back("Malformed ISO9660 directory record at offset " + std::to_string(pos) +
                               " in " + (parent_path.empty() ? "/" : parent_path) +
                               " (rec_len=" + std::to_string(rec_len) + ")");
            // Advance safely to next sector boundary
            pos = ((pos / m_sector_size) + 1) * m_sector_size;
            continue;
        }

        const u8* rec = dir_buf.data() + pos;
        u32 child_lba = read_le32(rec + 2);
        u32 child_size = read_le32(rec + 10);
        u8 flags = rec[25];
        u8 name_len = rec[32];

        if (33 + name_len > rec_len) {
            warnings.push_back("Directory record filename length exceeds record length in " +
                               (parent_path.empty() ? "/" : parent_path));
            pos += rec_len;
            continue;
        }

        // Skip '.' (0x00) and '..' (0x01)
        bool is_dot = (name_len == 1 && (rec[33] == 0x00 || rec[33] == 0x01));
        if (!is_dot && name_len > 0) {
            std::string raw_name(reinterpret_cast<const char*>(rec + 33), name_len);
            // Strip ISO9660 version suffix ';1'
            auto semi = raw_name.find(';');
            if (semi != std::string::npos) {
                raw_name.resize(semi);
            }
            // Strip trailing dot if present for extensionless files
            if (!raw_name.empty() && raw_name.back() == '.') {
                raw_name.pop_back();
            }

            std::string clean_name = to_upper(raw_name);
            std::string full_path = parent_path + "/" + clean_name;

            IsoFileEntry entry;
            entry.path = full_path;
            entry.name = clean_name;
            entry.lba = child_lba;
            entry.size = child_size;
            entry.is_directory = (flags & 0x02) != 0;
            entry.flags = flags;

            // Validate extent against ISO file size
            u64 child_end = static_cast<u64>(child_lba) * m_sector_size + child_size;
            if (child_end > m_file_size) {
                warnings.push_back("Entry extent exceeds ISO file size: " + full_path +
                                   " (LBA=" + std::to_string(child_lba) + ", size=" + std::to_string(child_size) + ")");
            }

            // Handle multi-extent or duplicate records cleanly
            auto it = m_path_index.find(full_path);
            if (it == m_path_index.end()) {
                m_path_index[full_path] = m_entries.size();
                m_entries.push_back(entry);
                if (entry.is_directory && child_lba != lba) {
                    subdirs.push_back(entry);
                }
            }
        }

        pos += rec_len;
    }

    for (const auto& dir : subdirs) {
        traverse_directory(dir.lba, static_cast<u32>(dir.size), dir.path, depth + 1, warnings);
    }

    return true;
}

std::string IsoReader::normalize_iso_path(std::string_view raw) {
    std::string s = trim_str(raw);
    // Strip cdrom0: or cdrom: prefix if present
    std::string upper = to_upper(s);
    if (upper.rfind("CDROM0:", 0) == 0) {
        s = s.substr(7);
    } else if (upper.rfind("CDROM:", 0) == 0) {
        s = s.substr(6);
    }
    // Strip version suffix ';1'
    auto semi = s.find(';');
    if (semi != std::string::npos) {
        s.resize(semi);
    }
    for (char& c : s) {
        if (c == '\\') c = '/';
        c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    }
    if (s.empty() || s[0] != '/') {
        s.insert(s.begin(), '/');
    }
    // Collapse double slashes
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        if (c == '/' && !out.empty() && out.back() == '/') continue;
        out.push_back(c);
    }
    return out;
}

const IsoFileEntry* IsoReader::find_file(const std::string& path) const {
    std::string norm = normalize_iso_path(path);
    auto it = m_path_index.find(norm);
    if (it == m_path_index.end()) return nullptr;
    return &m_entries[it->second];
}

bool IsoReader::read_file(const IsoFileEntry& entry, std::vector<u8>& out_data) const {
    if (entry.is_directory) return false;
    if (entry.size > 512ULL * 1024ULL * 1024ULL) return false; // Safety guard: stream large assets instead
    out_data.resize(static_cast<std::size_t>(entry.size));
    if (entry.size == 0) return true;
    u64 offset = static_cast<u64>(entry.lba) * m_sector_size;
    return read_bytes_at(offset, out_data.data(), out_data.size());
}

bool IsoReader::read_file_slice(const IsoFileEntry& entry, u64 offset, u8* dst, std::size_t count, std::size_t* out_read) const {
    if (out_read) *out_read = 0;
    if (entry.is_directory || !dst) return false;
    if (offset >= entry.size) return true;
    std::size_t avail = static_cast<std::size_t>(std::min<u64>(count, entry.size - offset));
    u64 abs_offset = static_cast<u64>(entry.lba) * m_sector_size + offset;
    if (!read_bytes_at(abs_offset, dst, avail)) return false;
    if (out_read) *out_read = avail;
    return true;
}

std::unique_ptr<IsoFileStream> IsoReader::open_stream(const std::string& path) {
    const IsoFileEntry* e = find_file(path);
    if (!e || e->is_directory) return nullptr;
    return std::make_unique<IsoFileStream>(this, *e);
}

bool IsoReader::parse_system_cnf(std::string_view content, SystemCnfInfo& out_info, std::string& out_error) {
    out_info = SystemCnfInfo{};
    out_info.raw_text = std::string(content);

    std::istringstream iss(out_info.raw_text);
    std::string line;
    while (std::getline(iss, line)) {
        std::string trimmed = trim_str(line);
        if (trimmed.empty() || trimmed[0] == '#') continue;
        auto eq = trimmed.find('=');
        if (eq == std::string::npos) continue;
        std::string key = to_upper(trim_str(trimmed.substr(0, eq)));
        std::string val = trim_str(trimmed.substr(eq + 1));
        out_info.entries[key] = val;
    }

    auto it_boot = out_info.entries.find("BOOT2");
    if (it_boot == out_info.entries.end()) {
        out_error = "SYSTEM.CNF does not contain a BOOT2 entry";
        return false;
    }

    out_info.boot2_raw = it_boot->second;
    out_info.boot2_iso_path = normalize_iso_path(out_info.boot2_raw);

    if (auto it = out_info.entries.find("VER"); it != out_info.entries.end()) {
        out_info.version = it->second;
    }
    if (auto it = out_info.entries.find("VMODE"); it != out_info.entries.end()) {
        out_info.vmode = it->second;
    }

    // Extract serial from BOOT2 basename (e.g. "/SLUS_215.61" -> "SLUS_215.61")
    auto slash = out_info.boot2_iso_path.rfind('/');
    out_info.serial = (slash != std::string::npos)
                          ? out_info.boot2_iso_path.substr(slash + 1)
                          : out_info.boot2_iso_path;
    out_info.region = detect_region_from_serial(out_info.serial, out_info.vmode);
    return true;
}

ExecutableFormat IsoReader::detect_executable_format(const std::vector<u8>& data, std::string& out_details) {
    if (data.size() < 16) {
        out_details = "Executable is empty or smaller than 16 bytes";
        return ExecutableFormat::InvalidOrEmpty;
    }

    // Check standard ELF magic: 0x7F 'E' 'L' 'F'
    if (data[0] == 0x7F && data[1] == 'E' && data[2] == 'L' && data[3] == 'F') {
        if (data.size() < 52) {
            out_details = "Truncated ELF header";
            return ExecutableFormat::InvalidOrEmpty;
        }
        u8 ei_class = data[4];
        u8 ei_data = data[5];
        u16 e_machine = read_le16(data.data() + 18);
        u32 e_entry = read_le32(data.data() + 24);
        u16 e_phnum = read_le16(data.data() + 44);
        u16 e_shnum = read_le16(data.data() + 48);

        if (ei_class == 1 && ei_data == 1 && e_machine == 8) {
            // Check for known PS2 packers (ps2-packer signature or suspicious tiny/packed stubs)
            std::string_view sv(reinterpret_cast<const char*>(data.data()), std::min<std::size_t>(data.size(), 4096));
            if (sv.find("PS2-PACKER") != std::string_view::npos || sv.find("ps2-packer") != std::string_view::npos) {
                out_details = "PS2 ELF32 MIPS packed with ps2-packer";
                return ExecutableFormat::PackedPs2Packer;
            }
            if (sv.find("UPX!") != std::string_view::npos) {
                out_details = "Packed ELF containing UPX signature";
                return ExecutableFormat::PackedUpxOrCustom;
            }

            std::ostringstream oss;
            oss << "Uncompressed PS2 ELF32 Little-Endian MIPS (entry=" << hex32(e_entry)
                << ", phnum=" << e_phnum << ", shnum=" << e_shnum << ")";
            out_details = oss.str();
            return ExecutableFormat::Ps2Elf32Mips;
        }

        std::ostringstream oss;
        oss << "Non-PS2 ELF header (class=" << static_cast<int>(ei_class)
            << ", data=" << static_cast<int>(ei_data) << ", machine=" << e_machine << ")";
        out_details = oss.str();
        return ExecutableFormat::Unknown;
    }

    // Check gzip magic 1F 8B
    if (data[0] == 0x1F && data[1] == 0x8B) {
        out_details = "GZIP compressed stream (not raw ELF)";
        return ExecutableFormat::CompressedGzip;
    }

    // Check zlib magic 78 9C / 78 DA / 78 01
    if (data[0] == 0x78 && (data[1] == 0x9C || data[1] == 0xDA || data[1] == 0x01)) {
        out_details = "ZLIB compressed stream (not raw ELF)";
        return ExecutableFormat::CompressedZlib;
    }

    out_details = "Unrecognized executable header magic: " + hex32(read_le32(data.data()));
    return ExecutableFormat::Unknown;
}

const char* IsoReader::executable_format_name(ExecutableFormat fmt) {
    switch (fmt) {
        case ExecutableFormat::Ps2Elf32Mips:      return "PS2 ELF32 (MIPS R5900)";
        case ExecutableFormat::PackedPs2Packer:   return "Packed PS2 ELF (ps2-packer)";
        case ExecutableFormat::PackedUpxOrCustom: return "Packed/Obfuscated ELF";
        case ExecutableFormat::CompressedGzip:    return "GZIP Compressed";
        case ExecutableFormat::CompressedZlib:    return "ZLIB Compressed";
        case ExecutableFormat::InvalidOrEmpty:    return "Invalid/Empty";
        default:                                  return "Unknown";
    }
}

IsoInspectionReport IsoReader::inspect() const {
    IsoInspectionReport rep;
    rep.iso_path = m_path;
    rep.iso_file_size = m_file_size;
    rep.sector_size = m_sector_size;
    rep.volume_space_sectors = m_volume_space_sectors;
    rep.system_identifier = m_system_id;
    rep.volume_identifier = m_volume_id;
    rep.publisher_identifier = m_publisher_id;
    rep.application_identifier = m_application_id;
    rep.warnings = m_parse_warnings;

    if (!is_open()) {
        rep.valid = false;
        rep.error = "ISO is not open";
        return rep;
    }

    for (const auto& e : m_entries) {
        if (e.is_directory) {
            rep.total_directories++;
        } else {
            rep.total_files++;
            if (is_notable_extension(e.name)) {
                rep.notable_files.push_back(e);
            }
        }
    }

    Sha256 fp;
    fp.update(m_pvd_raw.data(), m_pvd_raw.size());
    fp.update(std::to_string(m_file_size));

    const IsoFileEntry* sys_cnf = find_file("/SYSTEM.CNF");
    if (!sys_cnf || sys_cnf->is_directory) {
        rep.valid = false;
        rep.has_system_cnf = false;
        rep.error = "SYSTEM.CNF not found in ISO root directory";
        rep.volume_fingerprint_sha256 = Sha256::to_hex(fp.finalize());
        return rep;
    }

    rep.has_system_cnf = true;
    std::vector<u8> cnf_bytes;
    if (!read_file(*sys_cnf, cnf_bytes)) {
        rep.valid = false;
        rep.error = "Failed to read SYSTEM.CNF from ISO";
        rep.volume_fingerprint_sha256 = Sha256::to_hex(fp.finalize());
        return rep;
    }

    fp.update(cnf_bytes.data(), cnf_bytes.size());
    std::string cnf_text(reinterpret_cast<const char*>(cnf_bytes.data()), cnf_bytes.size());
    std::string cnf_err;
    if (!parse_system_cnf(cnf_text, rep.system_cnf, cnf_err)) {
        rep.valid = false;
        rep.error = cnf_err;
        rep.volume_fingerprint_sha256 = Sha256::to_hex(fp.finalize());
        return rep;
    }

    const IsoFileEntry* boot_entry = find_file(rep.system_cnf.boot2_iso_path);
    if (!boot_entry || boot_entry->is_directory) {
        rep.valid = false;
        rep.has_boot_executable = false;
        rep.error = "BOOT2 executable referenced by SYSTEM.CNF not found in ISO: " + rep.system_cnf.boot2_iso_path;
        rep.volume_fingerprint_sha256 = Sha256::to_hex(fp.finalize());
        return rep;
    }

    rep.has_boot_executable = true;
    rep.boot_executable_entry = *boot_entry;

    std::vector<u8> elf_bytes;
    if (!read_file(*boot_entry, elf_bytes)) {
        rep.valid = false;
        rep.error = "Failed to read BOOT2 executable bytes from ISO: " + boot_entry->path;
        rep.volume_fingerprint_sha256 = Sha256::to_hex(fp.finalize());
        return rep;
    }

    rep.executable_sha256 = Sha256::hash_bytes(elf_bytes.data(), elf_bytes.size());
    fp.update(rep.executable_sha256);
    rep.volume_fingerprint_sha256 = Sha256::to_hex(fp.finalize());

    rep.executable_format = detect_executable_format(elf_bytes, rep.executable_format_details);
    if (rep.executable_format != ExecutableFormat::Ps2Elf32Mips) {
        rep.warnings.push_back("BOOT2 executable is not a standard uncompressed PS2 ELF32 MIPS binary: " +
                               rep.executable_format_details);
    }

    rep.valid = true;
    return rep;
}

std::string IsoInspectionReport::format_report() const {
    std::ostringstream oss;
    oss << "=== PS2 ISO INSPECTION REPORT ===\n";
    oss << "Status              : " << (valid ? "VALID" : ("INVALID (" + error + ")")) << "\n";
    oss << "ISO Path            : " << iso_path << "\n";
    oss << "ISO Size            : " << iso_file_size << " bytes ("
        << (iso_file_size / (1024 * 1024)) << " MiB)\n";
    oss << "Sector Size         : " << sector_size << " bytes (" << volume_space_sectors << " sectors)\n";
    oss << "System Identifier   : " << system_identifier << "\n";
    oss << "Volume Identifier   : " << volume_identifier << "\n";
    oss << "Publisher           : " << publisher_identifier << "\n";
    oss << "Application         : " << application_identifier << "\n";
    oss << "ISO Fingerprint     : " << volume_fingerprint_sha256 << "\n";
    oss << "Directory Tree      : " << total_files << " files, " << total_directories << " directories\n";

    if (has_system_cnf) {
        oss << "\n--- SYSTEM.CNF ---\n";
        oss << "Disc Serial         : " << system_cnf.serial << "\n";
        oss << "Detected Region     : " << system_cnf.region << "\n";
        oss << "BOOT2 Raw           : " << system_cnf.boot2_raw << "\n";
        oss << "BOOT2 ISO Path      : " << system_cnf.boot2_iso_path << "\n";
        oss << "VER                 : " << system_cnf.version << "\n";
        oss << "VMODE               : " << system_cnf.vmode << "\n";
    }

    if (has_boot_executable) {
        oss << "\n--- BOOT2 EXECUTABLE ---\n";
        oss << "Path                : " << boot_executable_entry.path << "\n";
        oss << "LBA                 : " << boot_executable_entry.lba << "\n";
        oss << "Size                : " << boot_executable_entry.size << " bytes\n";
        oss << "Format              : " << IsoReader::executable_format_name(executable_format) << "\n";
        oss << "Details             : " << executable_format_details << "\n";
        oss << "ELF SHA-256         : " << executable_sha256 << "\n";
    }

    if (!notable_files.empty()) {
        oss << "\n--- NOTABLE FILES (" << notable_files.size() << ") ---\n";
        std::size_t limit = std::min<std::size_t>(notable_files.size(), 40);
        for (std::size_t i = 0; i < limit; ++i) {
            const auto& f = notable_files[i];
            oss << "  " << f.path << "  [LBA=" << f.lba << ", size=" << f.size << "]\n";
        }
        if (notable_files.size() > limit) {
            oss << "  ... (" << (notable_files.size() - limit) << " more)\n";
        }
    }

    if (!warnings.empty()) {
        oss << "\n--- WARNINGS (" << warnings.size() << ") ---\n";
        for (const auto& w : warnings) {
            oss << "  [WARN] " << w << "\n";
        }
    }

    return oss.str();
}

} // namespace tf::ps2
