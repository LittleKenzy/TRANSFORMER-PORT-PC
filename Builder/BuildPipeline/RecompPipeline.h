#pragma once

#include "../../Recompiler/ISO/IsoReader.h"
#include "../../Recompiler/ELF/ElfParser.h"
#include "../../Recompiler/Analyzer/FunctionAnalyzer.h"
#include "../../Recompiler/CodeGenerator/CppCodeGenerator.h"
#include "../Cache/BuildCache.h"
#include <string>

namespace tf::ps2 {

struct PipelineOptions {
    std::string iso_path;
    std::string output_dir;
    std::string cache_dir;
    std::string ghidra_csv_path;
    std::string runtime_source_root; // Root of TransformersNative for invoking native compiler
    std::string target_arch = "win64-x86_64";
    std::size_t max_functions_to_emit = 0; // 0 = all discovered functions
    bool emit_code = true;
    bool compile_native_module = false;
    bool force_rebuild = false;
};

struct PipelineResult {
    bool success = false;
    bool cache_hit = false;
    std::string error;

    IsoInspectionReport iso_report;
    Ps2ElfImage elf_image;
    AnalysisReport analysis_report;
    CodegenResult codegen_result;
    CacheManifest cache_manifest;
    std::string compiled_module_path;
};

class RecompPipeline {
public:
    static PipelineResult run(const PipelineOptions& opts);
};

} // namespace tf::ps2
