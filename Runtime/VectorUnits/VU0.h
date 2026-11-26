#pragma once

#include "../CPU/EEContext.h"

namespace tf::ps2 {

// VU0 Macro-Mode (COP2) execution unit sharing EEContext::vf/vi/vu0_acc.
class VU0Unit {
public:
    VU0Unit() = default;
    void reset(EEContext& ctx);

    // Executes a COP2 VU0 macro-mode instruction word (primary opcode 0x12, rs bit 4 == 1).
    void execute_macro(EEContext& ctx, u32 raw_word);

    u64 executed_ops() const { return m_executed_ops; }
    u64 unsupported_ops() const { return m_unsupported_ops; }

private:
    static void update_mac_flags(EEContext& ctx, const Vec4f& v, u8 dest_mask);
    static void write_vf(EEContext& ctx, u8 fd, const Vec4f& v, u8 dest_mask);

    u64 m_executed_ops = 0;
    u64 m_unsupported_ops = 0;
};

} // namespace tf::ps2
