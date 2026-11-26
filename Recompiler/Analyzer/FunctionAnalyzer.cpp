#include "FunctionAnalyzer.h"

#include <algorithm>
#include <fstream>
#include <sstream>

namespace tf::ps2 {

bool FunctionAnalyzer::import_ghidra_csv(const std::string& path, std::map<u32, std::string>& out_funcs, std::string& out_error) {
    std::ifstream ifs(path);
    if (!ifs) {
        out_error = "Unable to open Ghidra function CSV: " + path;
        return false;
    }
    std::string line;
    bool first = true;
    while (std::getline(ifs, line)) {
        if (line.empty()) continue;
        if (first && line.find("address") != std::string::npos) {
            first = false;
            continue;
        }
        first = false;
        std::istringstream iss(line);
        std::string name, addr_str;
        if (!std::getline(iss, name, ',')) continue;
        if (!std::getline(iss, addr_str, ',')) continue;
        try {
            u32 addr = static_cast<u32>(std::stoul(addr_str, nullptr, 16));
            if (addr != 0 && (addr & 3u) == 0) {
                out_funcs[addr] = name;
            }
        } catch (...) {
            continue;
        }
    }
    return true;
}

AnalysisReport FunctionAnalyzer::analyze(
    const Ps2ElfImage& elf,
    const std::map<u32, std::string>* external_symbols
) {
    AnalysisReport report;
    report.entry_point = elf.e_entry;

    // Map of function start address -> (name, source)
    std::map<u32, std::pair<std::string, std::string>> func_starts;

    auto register_start = [&](u32 addr, const std::string& name, const std::string& source, bool overwrite_name) {
        if ((addr & 3u) != 0 || !elf.is_executable_vaddr(addr)) return;
        auto it = func_starts.find(addr);
        if (it == func_starts.end()) {
            func_starts[addr] = {name, source};
        } else if (overwrite_name && !name.empty()) {
            it->second = {name, source};
        }
    };

    // 1. Entry point
    register_start(elf.e_entry, "_start", "entry", true);

    // 2. ELF symbol table functions
    for (const auto& sym : elf.symbols) {
        if (sym.is_function() && sym.value != 0) {
            std::string nm = sym.name.empty() ? ("sub_" + hex32(sym.value).substr(2)) : sym.name;
            register_start(sym.value, nm, "symbol", true);
        }
    }

    // 3. External Ghidra symbol map if supplied
    if (external_symbols) {
        for (const auto& [addr, nm] : *external_symbols) {
            register_start(addr, nm, "ghidra_import", true);
        }
    }

    // 4. Linear sweep across every executable region:
    //    - Record exact instruction decode coverage
    //    - Discover JAL targets and stack-frame prologues (ADDIU $sp, $sp, -N)
    struct RegionDecoded {
        const MemoryRegion* region = nullptr;
        std::vector<DecodedInstruction> insns;
    };
    std::vector<RegionDecoded> decoded_regions;
    decoded_regions.reserve(elf.executable_regions.size());

    for (const auto& reg : elf.executable_regions) {
        RegionDecoded rd;
        rd.region = &reg;
        u32 word_count = static_cast<u32>(reg.bytes.size() / 4);
        rd.insns.reserve(word_count);

        for (u32 i = 0; i < word_count; ++i) {
            u32 addr = reg.start_vaddr + i * 4;
            u32 word = read_le32(reg.bytes.data() + i * 4);
            DecodedInstruction ins = R5900Decoder::decode(addr, word);

            report.coverage.total_instructions++;
            if (ins.op == R5900Opcode::Unknown) {
                report.coverage.unknown_instructions++;
                auto& rec = report.coverage.unsupported_by_key[ins.unknown_reason];
                if (rec.count == 0) {
                    rec.sample_address = addr;
                    rec.sample_raw = word;
                    rec.reason = ins.unknown_reason;
                }
                rec.count++;
            } else {
                report.coverage.decoded_instructions++;
                report.coverage.opcode_histogram[ins.op]++;
            }

            // Direct JAL targets are strong function boundaries
            if (ins.op == R5900Opcode::JAL && elf.is_executable_vaddr(ins.target)) {
                register_start(ins.target, "sub_" + hex32(ins.target).substr(2), "jal_target", false);
            }

            // Heuristic stack-frame prologue detection: ADDIU $sp, $sp, -imm (followed within 4 ops by SD/SW $ra)
            if (ins.op == R5900Opcode::ADDIU && ins.rs == 29 && ins.rt == 29 && ins.simm < 0 && ins.simm >= -4096) {
                bool saves_ra = false;
                for (u32 k = 1; k <= 6 && (i + k) < word_count; ++k) {
                    u32 w2 = read_le32(reg.bytes.data() + (i + k) * 4);
                    DecodedInstruction next = R5900Decoder::decode(addr + k * 4, w2);
                    if ((next.op == R5900Opcode::SD || next.op == R5900Opcode::SW || next.op == R5900Opcode::SQ) &&
                        next.rs == 29 && next.rt == 31) {
                        saves_ra = true;
                        break;
                    }
                }
                if (saves_ra) {
                    register_start(addr, "sub_" + hex32(addr).substr(2), "prologue", false);
                }
            }

            if (ins.op == R5900Opcode::SYSCALL) {
                report.total_syscalls++;
            }

            rd.insns.push_back(std::move(ins));
        }

        decoded_regions.push_back(std::move(rd));
    }

    // 5. Partition executable regions into DiscoveredFunction blocks using sorted start addresses.
    for (const auto& rd : decoded_regions) {
        const auto& reg = *rd.region;
        u32 reg_start = reg.start_vaddr;
        u32 reg_end = reg.start_vaddr + static_cast<u32>(rd.insns.size() * 4);

        std::vector<u32> starts;
        for (const auto& [addr, info] : func_starts) {
            if (addr >= reg_start && addr < reg_end) {
                starts.push_back(addr);
            }
        }
        if (starts.empty() && !rd.insns.empty()) {
            starts.push_back(reg_start);
            func_starts[reg_start] = {"region_" + hex32(reg_start).substr(2), "region_start"};
        }

        for (std::size_t idx = 0; idx < starts.size(); ++idx) {
            u32 f_start = starts[idx];
            u32 f_end = (idx + 1 < starts.size()) ? starts[idx + 1] : reg_end;

            DiscoveredFunction fn;
            fn.start_addr = f_start;
            fn.end_addr = f_end;
            fn.name = func_starts[f_start].first;
            fn.source = func_starts[f_start].second;

            u32 first_i = (f_start - reg_start) / 4;
            u32 last_i = (f_end - reg_start) / 4;
            fn.instructions.reserve(last_i - first_i);

            for (u32 i = first_i; i < last_i; ++i) {
                const auto& ins = rd.insns[i];
                if (ins.op == R5900Opcode::Unknown) fn.has_unknown_instruction = true;
                if (ins.op == R5900Opcode::SYSCALL) fn.has_syscall = true;
                if (ins.is_call && !ins.is_indirect_jump) {
                    fn.call_targets.insert(ins.target);
                }
                if (ins.is_branch && ins.target >= f_start && ins.target < f_end) {
                    fn.internal_branch_targets.insert(ins.target);
                }
                if (ins.op == R5900Opcode::JR && ins.rs != 31) {
                    fn.has_indirect_jump = true;
                    fn.jump_table_sites.push_back(ins.address);
                    report.total_jump_table_sites++;
                }
                fn.instructions.push_back(ins);
            }

            report.addr_to_func_index[fn.start_addr] = report.functions.size();
            report.functions.push_back(std::move(fn));
        }
    }

    return report;
}

bool AnalysisReport::export_ghidra_csv(const std::string& path, std::string& out_error) const {
    std::ofstream ofs(path);
    if (!ofs) {
        out_error = "Failed to open CSV output path: " + path;
        return false;
    }
    ofs << "name,address,size,source,calls,indirect_jumps\n";
    for (const auto& fn : functions) {
        ofs << fn.name << ","
            << hex32(fn.start_addr) << ","
            << (fn.end_addr - fn.start_addr) << ","
            << fn.source << ","
            << fn.call_targets.size() << ","
            << fn.jump_table_sites.size() << "\n";
    }
    return true;
}

std::string AnalysisReport::format_summary() const {
    std::ostringstream oss;
    oss << "=== PS2 ELF FUNCTION ANALYSIS REPORT ===\n";
    oss << "Entry Point         : " << hex32(entry_point) << "\n";
    oss << "Discovered Functions: " << functions.size() << "\n";
    oss << "Indirect Jump Sites : " << total_jump_table_sites << "\n";
    oss << "SYSCALL Sites       : " << total_syscalls << "\n\n";
    oss << coverage.format_report();
    return oss.str();
}

} // namespace tf::ps2
