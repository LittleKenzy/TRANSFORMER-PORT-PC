#pragma once

#include "../../Recompiler/Common/Types.h"
#include <string>
#include <vector>

namespace tf::ps2 {

// Persistent directory-backed Virtual PS2 Memory Card (8 MiB logical quota).
class VirtualMemoryCard {
public:
    VirtualMemoryCard() = default;

    bool mount_directory(const std::string& root_dir, std::string& out_error);
    bool is_mounted() const { return !m_root_dir.empty(); }

    bool write_save_file(const std::string& relative_path, const std::vector<u8>& data, std::string& out_error);
    bool read_save_file(const std::string& relative_path, std::vector<u8>& out_data, std::string& out_error) const;
    bool file_exists(const std::string& relative_path) const;
    bool delete_save_file(const std::string& relative_path);
    std::vector<std::string> list_files() const;

    const std::string& root_dir() const { return m_root_dir; }

private:
    std::string resolve_safe_path(const std::string& relative_path) const;
    std::string m_root_dir;
};

} // namespace tf::ps2
