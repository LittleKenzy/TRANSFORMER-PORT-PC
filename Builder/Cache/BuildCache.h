#pragma once

#include "../../Recompiler/Common/Types.h"
#include <string>

namespace tf::ps2 {

struct CacheIdentity {
    std::string disc_serial;
    std::string elf_sha256;
    std::string iso_fingerprint_sha256;
    std::string recompiler_version = kRecompilerVersion;
    std::string runtime_version    = kRuntimeVersion;
    std::string target_arch;        // "win64-x86_64" or "android-arm64-v8a"
    std::string build_config;       // "Release" / "Debug"

    std::string compute_key() const;
};

struct CacheManifest {
    bool valid = false;
    std::string cache_key;
    CacheIdentity identity;
    std::string generated_dir;
    std::string binary_output_path;
    u64 functions_emitted = 0;
    u64 instructions_translated = 0;
};

class BuildCache {
public:
    explicit BuildCache(std::string cache_root = "");

    static std::string default_cache_root();
    const std::string& cache_root() const { return m_cache_root; }

    std::string entry_dir(const CacheIdentity& id) const;
    bool lookup(const CacheIdentity& id, CacheManifest& out_manifest) const;
    bool store(const CacheManifest& manifest, std::string& out_error) const;
    bool invalidate(const CacheIdentity& id) const;

private:
    std::string m_cache_root;
};

} // namespace tf::ps2
