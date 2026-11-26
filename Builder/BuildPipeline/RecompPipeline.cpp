#include "RecompPipeline.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace tf::ps2 {

namespace fs = std::filesystem;

namespace {

bool is_valid_runtime_root(const fs::path& p) {
    std::error_code ec;
    return !p.empty() &&
           fs::exists(p / "Runtime" / "CPU" / "EEContext.h", ec) &&
           fs::exists(p / "Runtime" / "Memory" / "GuestMemory.cpp", ec);
}

std::string resolve_runtime_root(const std::string& requested) {
    std::vector<fs::path> candidates;
    if (!requested.empty()) {
        candidates.emplace_back(requested);
    }
    std::error_code ec;
#ifdef _WIN32
    char exe_buf[MAX_PATH] = {};
    if (GetModuleFileNameA(nullptr, exe_buf, MAX_PATH) > 0) {
        fs::path exe_dir = fs::path(exe_buf).parent_path();
        candidates.push_back(exe_dir);
        candidates.push_back(exe_dir / "..");
        candidates.push_back(exe_dir / ".." / "TransformersNative");
    }
#endif
    fs::path cwd = fs::current_path(ec);
    if (!ec) {
        candidates.push_back(cwd);
        candidates.push_back(cwd / "TransformersNative");
        candidates.push_back(cwd / "..");
    }
    for (const auto& c : candidates) {
        if (is_valid_runtime_root(c)) {
            auto canon = fs::weakly_canonical(c, ec);
            return ec ? c.string() : canon.string();
        }
    }
    return {};
}

#ifdef _WIN32
std::string find_tool_exe(const char* exe_name, const std::vector<fs::path>& extra_dirs) {
    char found[MAX_PATH] = {};
    if (SearchPathA(nullptr, exe_name, ".exe", MAX_PATH, found, nullptr) > 0) {
        return std::string(found);
    }
    std::error_code ec;
    std::string filename = std::string(exe_name) + ".exe";
    for (const auto& dir : extra_dirs) {
        fs::path p = dir / filename;
        if (fs::exists(p, ec)) {
            return p.string();
        }
    }
    return {};
}

int run_process_capture(const std::string& cmd_line, const std::string& work_dir, std::string& out_output) {
    SECURITY_ATTRIBUTES sa{};
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;

    HANDLE read_pipe = nullptr;
    HANDLE write_pipe = nullptr;
    if (!CreatePipe(&read_pipe, &write_pipe, &sa, 0)) {
        out_output = "CreatePipe failed";
        return -1;
    }
    SetHandleInformation(read_pipe, HANDLE_FLAG_INHERIT, 0);

    STARTUPINFOA si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES | STARTF_USESHOWWINDOW;
    si.wShowWindow = SW_HIDE;
    si.hStdOutput = write_pipe;
    si.hStdError = write_pipe;
    si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);

    PROCESS_INFORMATION pi{};
    std::vector<char> mutable_cmd(cmd_line.begin(), cmd_line.end());
    mutable_cmd.push_back('\0');

    BOOL ok = CreateProcessA(
        nullptr,
        mutable_cmd.data(),
        nullptr,
        nullptr,
        TRUE,
        CREATE_NO_WINDOW,
        nullptr,
        work_dir.empty() ? nullptr : work_dir.c_str(),
        &si,
        &pi
    );

    CloseHandle(write_pipe);
    if (!ok) {
        DWORD err_code = GetLastError();
        CloseHandle(read_pipe);
        out_output = "CreateProcessA failed (Win32 error " + std::to_string(err_code) + ")";
        return -1;
    }

    char buf[4096];
    DWORD bytes_read = 0;
    while (ReadFile(read_pipe, buf, sizeof(buf) - 1, &bytes_read, nullptr) && bytes_read > 0) {
        buf[bytes_read] = '\0';
        out_output.append(buf, bytes_read);
        // Cap captured output to keep error messages concise
        if (out_output.size() > 16384) {
            out_output.erase(0, out_output.size() - 8192);
        }
    }
    CloseHandle(read_pipe);

    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD exit_code = 1;
    GetExitCodeProcess(pi.hProcess, &exit_code);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    return static_cast<int>(exit_code);
}
#endif

} // namespace

PipelineResult RecompPipeline::run(const PipelineOptions& opts) {
    PipelineResult res;

    // 1. Open and inspect the PS2 ISO (read-only)
    IsoReader iso;
    std::string err;
    if (!iso.open(opts.iso_path, err)) {
        res.error = err;
        return res;
    }

    res.iso_report = iso.inspect();
    if (!res.iso_report.valid) {
        res.error = "ISO validation failed: " + res.iso_report.error;
        return res;
    }

    if (res.iso_report.executable_format != ExecutableFormat::Ps2Elf32Mips) {
        res.error = "Unsupported BOOT2 executable format: " +
                    std::string(IsoReader::executable_format_name(res.iso_report.executable_format)) +
                    " (" + res.iso_report.executable_format_details + ")";
        return res;
    }

    // 2. Check persistent BuildCache
    BuildCache cache(opts.cache_dir);
    CacheIdentity cache_id;
    cache_id.disc_serial = res.iso_report.system_cnf.serial;
    cache_id.elf_sha256 = res.iso_report.executable_sha256;
    cache_id.iso_fingerprint_sha256 = res.iso_report.volume_fingerprint_sha256;
    cache_id.target_arch = opts.target_arch;
    cache_id.build_config = (opts.max_functions_to_emit == 0)
                                ? "full"
                                : ("subset_" + std::to_string(opts.max_functions_to_emit));

    if (!opts.force_rebuild && opts.emit_code) {
        CacheManifest cached;
        if (cache.lookup(cache_id, cached)) {
            std::error_code cache_ec;
            const bool have_valid_dll =
                !cached.binary_output_path.empty() && fs::exists(cached.binary_output_path, cache_ec);
            const bool need_dll_missing = opts.compile_native_module && !have_valid_dll;
            if (!need_dll_missing) {
                res.cache_hit = true;
                res.cache_manifest = cached;
                res.compiled_module_path = cached.binary_output_path;
                res.success = true;
                return res;
            }
        }
    }

    // 3. Extract and parse the original PS2 ELF executable
    std::vector<u8> elf_bytes;
    if (!iso.read_file(res.iso_report.boot_executable_entry, elf_bytes)) {
        res.error = "Failed reading BOOT2 executable bytes from ISO";
        return res;
    }

    if (!ElfParser::parse(elf_bytes, res.elf_image, err)) {
        res.error = "ELF parsing failed: " + err;
        return res;
    }

    // 4. Optional Ghidra CSV function import + static ELF analysis & R5900 decode coverage
    std::map<u32, std::string> external_syms;
    if (!opts.ghidra_csv_path.empty()) {
        FunctionAnalyzer::import_ghidra_csv(opts.ghidra_csv_path, external_syms, err);
    }

    res.analysis_report = FunctionAnalyzer::analyze(
        res.elf_image,
        external_syms.empty() ? nullptr : &external_syms
    );

    std::string out_dir = opts.output_dir.empty() ? cache.entry_dir(cache_id) : opts.output_dir;
    std::error_code ec;
    fs::create_directories(out_dir, ec);

    // Save reports & Ghidra-compatible function CSV
    {
        std::ofstream(fs::path(out_dir) / "iso_inspection_report.txt") << res.iso_report.format_report();
        std::ofstream(fs::path(out_dir) / "elf_analysis_report.txt")
            << res.elf_image.format_summary() << "\n"
            << res.analysis_report.format_summary();
        res.analysis_report.export_ghidra_csv((fs::path(out_dir) / "functions_ghidra.csv").string(), err);
    }

    if (!opts.emit_code) {
        res.success = true;
        return res;
    }

    // 5. Static Recompilation -> C++20 translation units
    fs::path gen_dir = fs::path(out_dir) / "Generated";
    CodegenOptions cg_opts;
    cg_opts.max_functions_to_emit = opts.max_functions_to_emit;
    cg_opts.max_functions_per_shard = 250;
    res.codegen_result = CppCodeGenerator::generate_project(res.analysis_report, gen_dir.string(), cg_opts);
    if (!res.codegen_result.success) {
        res.error = "C++ code generation failed: " + res.codegen_result.error;
        return res;
    }

    // Emit dynamic module entry wrapper (recomp_shared_entry.cpp) so Transformers.exe can load it directly
    fs::path shared_entry = gen_dir / "recomp_shared_entry.cpp";
    {
        std::ofstream se(shared_entry);
        se << "#include \"recomp_module.h\"\n";
        se << "#include <unordered_map>\n\n";
        se << "#ifdef _WIN32\n#define TF_EXPORT extern \"C\" __declspec(dllexport)\n";
        se << "#else\n#define TF_EXPORT extern \"C\" __attribute__((visibility(\"default\")))\n#endif\n\n";
        se << "TF_EXPORT tf::ps2::u32 tf_recomp_entry_point() { return tf::recomp::kEntryPoint; }\n";
        se << "TF_EXPORT const char* tf_recomp_iso_fingerprint() { return \"" << res.iso_report.volume_fingerprint_sha256 << "\"; }\n";
        se << "TF_EXPORT const char* tf_recomp_disc_serial() { return \"" << res.iso_report.system_cnf.serial << "\"; }\n";
        se << "TF_EXPORT void tf_recomp_register(void* map_ptr) {\n";
        se << "    auto* tbl = static_cast<std::unordered_map<tf::ps2::u32, tf::ps2::RecompiledFn>*>(map_ptr);\n";
        se << "    tf::recomp::register_recompiled_functions(*tbl);\n";
        se << "}\n";
    }
    res.codegen_result.generated_files.push_back(shared_entry.string());

    // 6. Optionally compile the generated C++20 code in parallel via CMake+Ninja into a native Windows x64 DLL
    if (opts.compile_native_module) {
#ifdef _WIN32
        const std::string resolved_root = resolve_runtime_root(opts.runtime_source_root);
        if (resolved_root.empty()) {
            res.error = "Cannot compile native DLL: Runtime source root (Runtime/CPU/EEContext.h & Runtime/Memory/GuestMemory.cpp) not found.";
            return res;
        }

        const std::vector<fs::path> extra_tool_dirs = {
            "C:\\msys64\\ucrt64\\bin",
            "C:\\msys64\\mingw64\\bin",
            "C:\\Program Files\\CMake\\bin",
            "C:\\Program Files (x86)\\CMake\\bin",
            "C:\\Strawberry\\c\\bin",
        };
        const std::string cmake_exe = find_tool_exe("cmake", extra_tool_dirs);
        if (cmake_exe.empty()) {
            res.error = "Cannot compile native DLL: cmake.exe not found on PATH or standard install directories.";
            return res;
        }
        const std::string ninja_exe = find_tool_exe("ninja", extra_tool_dirs);
        const std::string gpp_exe = find_tool_exe("g++", extra_tool_dirs);

        const std::string dll_name = res.iso_report.system_cnf.serial + "_recomp.dll";
        const fs::path dll_out = fs::path(out_dir) / dll_name;
        const fs::path cmake_file = gen_dir / "CMakeLists.txt";
        const std::string norm_root = fs::path(resolved_root).generic_string();
        {
            std::ofstream cm(cmake_file);
            cm << "cmake_minimum_required(VERSION 3.20)\n";
            cm << "project(RecompModule LANGUAGES CXX)\n";
            cm << "set(CMAKE_CXX_STANDARD 20)\n";
            cm << "set(CMAKE_CXX_STANDARD_REQUIRED ON)\n";
            cm << "add_library(recomp_mod SHARED\n";
            for (const auto& f : res.codegen_result.generated_files) {
                if (f.size() >= 4 && f.compare(f.size() - 4, 4, ".cpp") == 0) {
                    cm << "    \"" << fs::path(f).generic_string() << "\"\n";
                }
            }
            cm << "    \"" << norm_root << "/Runtime/Memory/GuestMemory.cpp\"\n";
            cm << ")\n";
            cm << "target_include_directories(recomp_mod PRIVATE \"" << norm_root << "\")\n";
            cm << "if(CMAKE_CXX_COMPILER_ID MATCHES \"GNU|Clang\")\n";
            cm << "    target_compile_options(recomp_mod PRIVATE -O2 -march=x86-64-v3 -Wno-unused-label -Wno-unused-variable)\n";
            cm << "endif()\n";
            cm << "if(MINGW)\n";
            cm << "    target_link_options(recomp_mod PRIVATE -static -static-libgcc -static-libstdc++)\n";
            cm << "endif()\n";
            cm << "set_target_properties(recomp_mod PROPERTIES PREFIX \"\" OUTPUT_NAME \""
               << res.iso_report.system_cnf.serial << "_recomp\")\n";
        }

        const fs::path ninja_build = fs::path(out_dir) / "ninja_build";
        fs::create_directories(ninja_build, ec);

        std::ostringstream cfg_cmd;
        cfg_cmd << "\"" << cmake_exe << "\" -S \"" << gen_dir.string() << "\" -B \"" << ninja_build.string() << "\"";
        if (!ninja_exe.empty()) {
            cfg_cmd << " -G Ninja \"-DCMAKE_MAKE_PROGRAM=" << fs::path(ninja_exe).generic_string() << "\"";
        } else {
            const std::string mingw_make = find_tool_exe("mingw32-make", extra_tool_dirs);
            if (!mingw_make.empty()) {
                cfg_cmd << " -G \"MinGW Makefiles\" \"-DCMAKE_MAKE_PROGRAM=" << fs::path(mingw_make).generic_string() << "\"";
            }
        }
        if (!gpp_exe.empty()) {
            cfg_cmd << " \"-DCMAKE_CXX_COMPILER=" << fs::path(gpp_exe).generic_string() << "\"";
        }
        cfg_cmd << " -DCMAKE_BUILD_TYPE=Release";

        std::string cfg_out;
        int rc = run_process_capture(cfg_cmd.str(), out_dir, cfg_out);
        if (rc != 0) {
            res.error = "CMake configuration failed (exit " + std::to_string(rc) + "): " + cfg_out;
            return res;
        }

        std::string bld_cmd = "\"" + cmake_exe + "\" --build \"" + ninja_build.string() + "\" --config Release";
        std::string bld_out;
        rc = run_process_capture(bld_cmd, out_dir, bld_out);
        if (rc != 0) {
            res.error = "Parallel native DLL compilation failed (exit " + std::to_string(rc) + "): " + bld_out;
            return res;
        }

        fs::path built_dll = ninja_build / dll_name;
        if (!fs::exists(built_dll, ec)) {
            fs::path rel_dll = ninja_build / "Release" / dll_name;
            if (fs::exists(rel_dll, ec)) {
                built_dll = rel_dll;
            }
        }
        if (!fs::exists(built_dll, ec)) {
            res.error = "Build succeeded but output DLL was not found at " + built_dll.string();
            return res;
        }

        std::error_code copy_ec;
        fs::copy_file(built_dll, dll_out, fs::copy_options::overwrite_existing, copy_ec);
        if (copy_ec || !fs::exists(dll_out, ec)) {
            res.error = "Failed copying compiled DLL to " + dll_out.string() + ": " + copy_ec.message();
            return res;
        }

        // Best-effort sync to <runtime_root>/dist/<SERIAL>_recomp.dll so dist/ stays up to date
        fs::path dist_dir = fs::path(resolved_root) / "dist";
        if (fs::exists(dist_dir, ec) && fs::is_directory(dist_dir, ec)) {
            std::error_code dist_ec;
            fs::copy_file(dll_out, dist_dir / dll_name, fs::copy_options::overwrite_existing, dist_ec);
        }

        res.compiled_module_path = dll_out.string();
#else
        res.error = "Native Windows DLL compilation is only supported on Windows hosts.";
        return res;
#endif
    }

    if (opts.compile_native_module) {
        if (res.compiled_module_path.empty() || !fs::exists(res.compiled_module_path, ec)) {
            res.error = "Native DLL compilation requested but no compiled DLL was produced.";
            return res;
        }
    }

    // 7. Store persistent cache manifest
    res.cache_manifest.valid = true;
    res.cache_manifest.cache_key = cache_id.compute_key();
    res.cache_manifest.identity = cache_id;
    res.cache_manifest.generated_dir = gen_dir.string();
    res.cache_manifest.binary_output_path = res.compiled_module_path;
    res.cache_manifest.functions_emitted = res.codegen_result.functions_emitted;
    res.cache_manifest.instructions_translated = res.codegen_result.instructions_translated;
    cache.store(res.cache_manifest, err);

    res.success = true;
    return res;
}

} // namespace tf::ps2
