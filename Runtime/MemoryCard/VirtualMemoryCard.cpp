#include "VirtualMemoryCard.h"
#include <filesystem>
#include <fstream>

namespace tf::ps2 {

namespace fs = std::filesystem;

bool VirtualMemoryCard::mount_directory(const std::string& root_dir, std::string& out_error) {
    std::error_code ec;
    fs::create_directories(root_dir, ec);
    if (ec) {
        out_error = "Failed to initialize virtual memory card directory " + root_dir + ": " + ec.message();
        return false;
    }
    m_root_dir = fs::weakly_canonical(root_dir, ec).string();
    if (m_root_dir.empty()) m_root_dir = root_dir;
    return true;
}

std::string VirtualMemoryCard::resolve_safe_path(const std::string& relative_path) const {
    if (m_root_dir.empty()) return {};
    fs::path rel(relative_path);
    if (rel.is_absolute()) return {};
    fs::path combined = fs::path(m_root_dir) / rel;
    std::error_code ec;
    fs::path norm_root = fs::weakly_canonical(m_root_dir, ec);
    fs::path norm_target = fs::weakly_canonical(combined, ec);
    auto root_str = norm_root.string();
    auto targ_str = norm_target.string();
    if (targ_str.rfind(root_str, 0) != 0) return {}; // Prevent path traversal outside memory card
    return norm_target.string();
}

bool VirtualMemoryCard::write_save_file(const std::string& relative_path, const std::vector<u8>& data, std::string& out_error) {
    std::string full = resolve_safe_path(relative_path);
    if (full.empty()) {
        out_error = "Invalid or unmounted memory card path: " + relative_path;
        return false;
    }
    std::error_code ec;
    fs::create_directories(fs::path(full).parent_path(), ec);
    std::ofstream ofs(full, std::ios::binary | std::ios::trunc);
    if (!ofs) {
        out_error = "Failed to open memory card file for writing: " + full;
        return false;
    }
    if (!data.empty()) {
        ofs.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    }
    return ofs.good();
}

bool VirtualMemoryCard::read_save_file(const std::string& relative_path, std::vector<u8>& out_data, std::string& out_error) const {
    std::string full = resolve_safe_path(relative_path);
    if (full.empty()) {
        out_error = "Invalid or unmounted memory card path: " + relative_path;
        return false;
    }
    std::ifstream ifs(full, std::ios::binary | std::ios::ate);
    if (!ifs) {
        out_error = "Memory card file not found: " + relative_path;
        return false;
    }
    auto sz = ifs.tellg();
    if (sz < 0) return false;
    out_data.resize(static_cast<std::size_t>(sz));
    ifs.seekg(0, std::ios::beg);
    if (!out_data.empty()) {
        ifs.read(reinterpret_cast<char*>(out_data.data()), sz);
    }
    return ifs.good();
}

bool VirtualMemoryCard::file_exists(const std::string& relative_path) const {
    std::string full = resolve_safe_path(relative_path);
    if (full.empty()) return false;
    std::error_code ec;
    return fs::is_regular_file(full, ec);
}

bool VirtualMemoryCard::delete_save_file(const std::string& relative_path) {
    std::string full = resolve_safe_path(relative_path);
    if (full.empty()) return false;
    std::error_code ec;
    return fs::remove(full, ec);
}

std::vector<std::string> VirtualMemoryCard::list_files() const {
    std::vector<std::string> out;
    if (m_root_dir.empty()) return out;
    std::error_code ec;
    for (const auto& entry : fs::recursive_directory_iterator(m_root_dir, ec)) {
        if (entry.is_regular_file()) {
            out.push_back(fs::relative(entry.path(), m_root_dir, ec).generic_string());
        }
    }
    return out;
}

} // namespace tf::ps2
