#pragma once

#include "../Analyzer/FunctionAnalyzer.h"
#include <string>
#include <vector>

namespace tf::ps2 {

struct CodegenOptions {
    std::string namespace_name = "tf::recomp";
    std::size_t max_functions_per_shard = 250;
    std::size_t max_functions_to_emit = 0; // 0 = all discovered functions
    bool emit_disassembly_comments = false;
    std::string runtime_include_prefix = "Runtime/CPU/";
};

struct CodegenResult {
    bool success = false;
    std::string error;
    std::vector<std::string> generated_files;
    std::size_t functions_emitted = 0;
    u64 instructions_translated = 0;
    u64 instructions_trapped_unsupported = 0;
    std::map<std::string, u64> unsupported_reasons;
};

class CppCodeGenerator {
public:
    // Generates C++20 code for a single function into a string (useful for unit tests and sharding).
    static std::string emit_function(
        const DiscoveredFunction& fn,
        const std::map<u32, std::string>& all_func_names,
        const CodegenOptions& opts,
        CodegenResult& stats
    );

    // Emits a complete compilable C++20 project (header + sharded .cpp files + dispatch table).
    static CodegenResult generate_project(
        const AnalysisReport& analysis,
        const std::string& output_dir,
        const CodegenOptions& opts = CodegenOptions{}
    );

    static std::string sanitize_identifier(const std::string& name, u32 addr);
};

} // namespace tf::ps2
