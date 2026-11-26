#include "BuildCache.h"
#include "../../Recompiler/Common/Sha256.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace tf::ps2 {

namespace fs = std::filesystem;

std::string CacheIdentity::compute_key() const {
    std::ostringstream oss;
    oss << disc_serial << "|" << elf_sha256 << "|" << iso_fingerprint_sha256 << "|"
        << recompiler_version << "|" << runtime_version << "|"
        << target_arch << "|" << build_config;
    return Sha256::hash_string(oss.str()).substr(0, 32);
}

std::string BuildCache::default_cache_root() {
#ifdef _WIN32
    if (const char* local = std::getenv("LOCALAPPDATA")) {
        return (fs::path(local) / "TransformersNative" / "Cache").string();
    }
#endif
    return (fs::temp_directory_path() / "TransformersNative" / "Cache").string();
}

BuildCache::BuildCache(std::string cache_root)
    : m_cache_root(cache_root.empty() ? default_cache_root() : std::move(cache_root)) {}

std::string BuildCache::entry_dir(const CacheIdentity& id) const {
    return (fs::path(m_cache_root) / (id.disc_serial + "_" + id.target_arch + "_" + id.compute_key())).string();
}

bool BuildCache::lookup(const CacheIdentity& id, CacheManifest& out_manifest) const {
    out_manifest = CacheManifest{};
    std::string dir = entry_dir(id);
    fs::path manifest_path = fs::path(dir) / "cache_manifest.txt";
    std::ifstream ifs(manifest_path);
    if (!ifs) return false;

    std::string line;
    CacheManifest m;
    while (std::getline(ifs, line)) {
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string k = line.substr(0, eq);
        std::string v = line.substr(eq + 1);
        if (k == "cache_key") m.cache_key = v;
        else if (k == "disc_serial") m.identity.disc_serial = v;
        else if (k == "elf_sha256") m.identity.elf_sha256 = v;
        else if (k == "iso_fingerprint_sha256") m.identity.iso_fingerprint_sha256 = v;
        else if (k == "recompiler_version") m.identity.recompiler_version = v;
        else if (k == "runtime_version") m.identity.runtime_version = v;
        else if (k == "target_arch") m.identity.target_arch = v;
        else if (k == "build_config") m.identity.build_config = v;
        else if (k == "generated_dir") m.generated_dir = v;
        else if (k == "binary_output_path") m.binary_output_path = v;
        else if (k == "functions_emitted") m.functions_emitted = std::stoull(v);
        else if (k == "instructions_translated") m.instructions_translated = std::stoull(v);
    }

    if (m.cache_key != id.compute_key()) return false;
    if (!m.binary_output_path.empty() && !fs::exists(m.binary_output_path)) return false;

    m.valid = true;
    out_manifest = std::move(m);
    return true;
}

bool BuildCache::store(const CacheManifest& manifest, std::string& out_error) const {
    std::string dir = entry_dir(manifest.identity);
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) {
        out_error = "Failed to create build cache directory " + dir + ": " + ec.message();
        return false;
    }

    fs::path manifest_path = fs::path(dir) / "cache_manifest.txt";
    std::ofstream ofs(manifest_path, std::ios::trunc);
    if (!ofs) {
        out_error = "Failed to write build cache manifest: " + manifest_path.string();
        return false;
    }

    ofs << "cache_key=" << manifest.identity.compute_key() << "\n";
    ofs << "disc_serial=" << manifest.identity.disc_serial << "\n";
    ofs << "elf_sha256=" << manifest.identity.elf_sha256 << "\n";
    ofs << "iso_fingerprint_sha256=" << manifest.identity.iso_fingerprint_sha256 << "\n";
    ofs << "recompiler_version=" << manifest.identity.recompiler_version << "\n";
    ofs << "runtime_version=" << manifest.identity.runtime_version << "\n";
    ofs << "target_arch=" << manifest.identity.target_arch << "\n";
    ofs << "build_config=" << manifest.identity.build_config << "\n";
    ofs << "generated_dir=" << manifest.generated_dir << "\n";
    ofs << "binary_output_path=" << manifest.binary_output_path << "\n";
    ofs << "functions_emitted=" << manifest.functions_emitted << "\n";
    ofs << "instructions_translated=" << manifest.instructions_translated << "\n";
    return ofs.good();
}

bool BuildCache::invalidate(const CacheIdentity& id) const {
    std::error_code ec;
    return fs::remove_all(entry_dir(id), ec) > 0;
}

} // namespace tf::ps2
