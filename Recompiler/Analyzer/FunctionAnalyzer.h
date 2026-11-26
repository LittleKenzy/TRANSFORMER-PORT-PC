#pragma once

#include "../ELF/ElfParser.h"
#include "../Decoder/R5900Decoder.h"
#include <map>
#include <set>
#include <string>
#include <vector>

namespace tf::ps2 {

struct DiscoveredFunction {
    u32 start_addr = 0;
    u32 end_addr = 0; // Exclusive
    std::string name;
    std::string source; // "entry", "symbol", "jal_target", "prologue", "ghidra_import"
    bool has_indirect_jump = false;
    bool has_syscall = false;
    bool has_unknown_instruction = false;
    std::set<u32> call_targets;
    std::set<u32> internal_branch_targets;
    std::vector<u32> jump_table_sites;
    std::vector<DecodedInstruction> instructions;
};

struct AnalysisReport {
    u32 entry_point = 0;
    std::vector<DiscoveredFunction> functions;
    std::map<u32, std::size_t> addr_to_func_index;
    DecodeCoverageReport coverage;
    u64 total_jump_table_sites = 0;
    u64 total_syscalls = 0;

    std::string format_summary() const;
    // Exports Ghidra-compatible CSV function map (name,address,size,source)
    bool export_ghidra_csv(const std::string& path, std::string& out_error) const;
};

class FunctionAnalyzer {
public:
    // Optional Ghidra CSV function list (name,hex_address[,size]) to seed additional function starts.
    static bool import_ghidra_csv(const std::string& path, std::map<u32, std::string>& out_funcs, std::string& out_error);

    static AnalysisReport analyze(
        const Ps2ElfImage& elf,
        const std::map<u32, std::string>* external_symbols = nullptr
    );
};

} // namespace tf::ps2
