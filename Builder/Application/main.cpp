#include "../BuildPipeline/RecompPipeline.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <commdlg.h>
#endif

namespace {

void print_usage() {
    std::cout
        << "TransformersRecompBuilder — PS2 ISO-Based Static Recompilation Builder\n"
        << "Legal Notice: You must supply your own legally obtained PS2 ISO image.\n\n"
        << "Usage:\n"
        << "  TransformersRecompBuilder.exe --iso <path.iso> [options]\n\n"
        << "Options:\n"
        << "  --select               Open native Windows file dialog to pick a PS2 ISO\n"
        << "  --inspect-only         Inspect ISO, SYSTEM.CNF, ELF & R5900 coverage without codegen\n"
        << "  --out <dir>            Output directory for reports & generated C++20 sources\n"
        << "  --cache <dir>          Override persistent build cache directory\n"
        << "  --ghidra-csv <file>    Import additional function addresses from Ghidra CSV\n"
        << "  --max-funcs <N>        Limit number of emitted functions (0 = all)\n"
        << "  --compile-dll          Invoke native C++20 compiler to build Windows x64 module DLL\n"
        << "  --runtime-root <dir>   Path to TransformersNative source tree (for --compile-dll)\n"
        << "  --target <arch>        Target architecture (win64-x86_64 | android-arm64-v8a)\n"
        << "  --force                Bypass persistent build cache\n";
}

#ifdef _WIN32
std::string open_iso_dialog() {
    char filename[MAX_PATH] = {};
    OPENFILENAMEA ofn{};
    ofn.lStructSize = sizeof(ofn);
    ofn.lpstrFilter = "PlayStation 2 ISO Images (*.iso)\0*.iso\0All Files (*.*)\0*.*\0";
    ofn.lpstrFile = filename;
    ofn.nMaxFile = MAX_PATH;
    ofn.lpstrTitle = "Select Legally Obtained Transformers PS2 ISO";
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST;
    if (GetOpenFileNameA(&ofn)) {
        return std::string(filename);
    }
    return {};
}
#endif

} // namespace

int main(int argc, char** argv) {
    tf::ps2::PipelineOptions opts;
    bool select_dialog = (argc == 1);
    bool interactive_pause = (argc == 1);

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage();
            return 0;
        } else if (arg == "--select") {
            select_dialog = true;
        } else if (arg == "--iso" && i + 1 < argc) {
            opts.iso_path = argv[++i];
        } else if (arg == "--out" && i + 1 < argc) {
            opts.output_dir = argv[++i];
        } else if (arg == "--cache" && i + 1 < argc) {
            opts.cache_dir = argv[++i];
        } else if (arg == "--ghidra-csv" && i + 1 < argc) {
            opts.ghidra_csv_path = argv[++i];
        } else if (arg == "--runtime-root" && i + 1 < argc) {
            opts.runtime_source_root = argv[++i];
        } else if (arg == "--target" && i + 1 < argc) {
            opts.target_arch = argv[++i];
        } else if (arg == "--max-funcs" && i + 1 < argc) {
            opts.max_functions_to_emit = static_cast<std::size_t>(std::stoull(argv[++i]));
        } else if (arg == "--inspect-only") {
            opts.emit_code = false;
        } else if (arg == "--compile-dll") {
            opts.compile_native_module = true;
        } else if (arg == "--force") {
            opts.force_rebuild = true;
        }
    }

#ifdef _WIN32
    if (select_dialog && opts.iso_path.empty()) {
        opts.iso_path = open_iso_dialog();
    }
#endif

    if (opts.iso_path.empty()) {
        print_usage();
        return 1;
    }

    std::cout << "[Builder] Starting PS2 Static Recompilation Pipeline\n";
    std::cout << "[Builder] ISO: " << opts.iso_path << "\n";

    auto result = tf::ps2::RecompPipeline::run(opts);
    if (!result.success) {
        std::cerr << "[Builder] ERROR: " << result.error << "\n";
        return 2;
    }

    if (result.cache_hit) {
        std::cout << "[Builder] CACHE HIT (" << result.cache_manifest.cache_key << ")\n";
        std::cout << "  Generated Dir : " << result.cache_manifest.generated_dir << "\n";
        std::cout << "  Functions     : " << result.cache_manifest.functions_emitted << "\n";
        std::cout << "  Instructions  : " << result.cache_manifest.instructions_translated << "\n";
        if (!result.compiled_module_path.empty()) {
            std::cout << "  Compiled Mod  : " << result.compiled_module_path << "\n";
        }
        return 0;
    }

    std::cout << "\n" << result.iso_report.format_report() << "\n";
    std::cout << result.elf_image.format_summary() << "\n";
    std::cout << result.analysis_report.format_summary() << "\n";

    if (opts.emit_code) {
        std::cout << "=== STATIC RECOMPILATION CODEGEN SUMMARY ===\n";
        std::cout << "Functions Emitted    : " << result.codegen_result.functions_emitted << "\n";
        std::cout << "Instructions Lowered : " << result.codegen_result.instructions_translated << "\n";
        std::cout << "Unsupported Trapped  : " << result.codegen_result.instructions_trapped_unsupported << "\n";
        std::cout << "Generated Files      : " << result.codegen_result.generated_files.size() << "\n";
        std::cout << "Cache Key            : " << result.cache_manifest.cache_key << "\n";
        if (!result.compiled_module_path.empty()) {
            std::cout << "Compiled Native DLL  : " << result.compiled_module_path << "\n";
        }
    }

    if (interactive_pause) {
        std::cout << "\nPress Enter to exit...";
        std::cin.get();
    }
    return 0;
}
