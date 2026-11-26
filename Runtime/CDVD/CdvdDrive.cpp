#include "CdvdDrive.h"

namespace tf::ps2 {

bool CdvdDrive::mount_iso(const std::string& iso_path, std::string& out_error) {
    m_sectors_read = 0;
    m_read_requests = 0;
    return m_iso.open(iso_path, out_error);
}

void CdvdDrive::unmount() {
    m_iso.close();
}

bool CdvdDrive::read_sectors(u32 lba, u32 sector_count, u8* dst) {
    if (!m_iso.is_open() || !dst) return false;
    if (!m_iso.read_sectors(lba, sector_count, dst)) return false;
    m_read_requests++;
    m_sectors_read += sector_count;
    return true;
}

const IsoFileEntry* CdvdDrive::locate_file(const std::string& path) const {
    return m_iso.is_open() ? m_iso.find_file(path) : nullptr;
}

bool CdvdDrive::read_file_bytes(const std::string& path, std::vector<u8>& out_data) const {
    const IsoFileEntry* e = locate_file(path);
    if (!e) return false;
    return m_iso.read_file(*e, out_data);
}

std::unique_ptr<IsoFileStream> CdvdDrive::open_stream(const std::string& path) {
    return m_iso.is_open() ? m_iso.open_stream(path) : nullptr;
}

} // namespace tf::ps2
