#pragma once

#include "R5900Instruction.h"
#include <map>
#include <string>
#include <vector>

namespace tf::ps2 {

struct UnsupportedOpcodeRecord {
    u32 sample_address = 0;
    u32 sample_raw = 0;
    std::string reason;
    u64 count = 0;
};

struct DecodeCoverageReport {
    u64 total_instructions = 0;
    u64 decoded_instructions = 0;
    u64 unknown_instructions = 0;
    std::map<std::string, UnsupportedOpcodeRecord> unsupported_by_key;
    std::map<R5900Opcode, u64> opcode_histogram;

    double coverage_percent() const {
        if (total_instructions == 0) return 100.0;
        return (100.0 * static_cast<double>(decoded_instructions)) / static_cast<double>(total_instructions);
    }

    std::string format_report() const;
};

class R5900Decoder {
public:
    static DecodedInstruction decode(u32 address, u32 word);
    static const char* opcode_name(R5900Opcode op);
    static const char* gpr_name(u8 reg);
    static std::string disassemble(const DecodedInstruction& ins);
};

} // namespace tf::ps2
