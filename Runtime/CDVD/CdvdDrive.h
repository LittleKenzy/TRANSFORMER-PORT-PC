#pragma once

#include "../../Recompiler/ISO/IsoReader.h"
#include <string>
#include <vector>

namespace tf::ps2 {

// ISO-backed PS2 CDVD disc subsystem: reads sectors and files directly from the user's read-only ISO.
class CdvdDrive {
public:
    CdvdDrive() = default;

    bool mount_iso(const std::string& iso_path, std::string& out_error);
    void unmount();
    bool is_mounted() const { return m_iso.is_open(); }

    bool read_sectors(u32 lba, u32 sector_count, u8* dst);
    const IsoFileEntry* locate_file(const std::string& path) const;
    bool read_file_bytes(const std::string& path, std::vector<u8>& out_data) const;
    std::unique_ptr<IsoFileStream> open_stream(const std::string& path);

    const IsoReader& iso() const { return m_iso; }
    u64 sectors_read() const { return m_sectors_read; }
    u64 read_requests() const { return m_read_requests; }

private:
    IsoReader m_iso;
    u64 m_sectors_read = 0;
    u64 m_read_requests = 0;
};

} // namespace tf::ps2
