#pragma once

#include "EEContext.h"
#include "../../Recompiler/Decoder/R5900Decoder.h"
#include <unordered_map>

namespace tf::ps2 {

// Compatible dynamic fallback interpreter for R5900 code not present in the static
// function table (e.g., intra-function switch jump tables, dynamic overlays, or self-modifying code).
class R5900FallbackInterpreter {
public:
    // Executes starting at start_pc until a return (JR $ra), a static function handoff,
    // a halt/trap, or max_steps instructions are executed.
    // When cooperative_yield_fn is provided (worker EE threads), the interpreter yields
    // periodically on step/backward-branch intervals so spin-waits hand off cleanly to
    // the main thread instead of burning max_steps and aborting mid-function.
    static u64 run_until_return_or_handoff(
        EEContext& ctx,
        u32 start_pc,
        const std::unordered_map<u32, RecompiledFn>& static_table,
        u64 max_steps = 200000,
        const RecompiledFn* fast_table = nullptr,
        u32 fast_base = 0,
        u32 fast_end = 0,
        void (*cooperative_yield_fn)(void*) = nullptr,
        void* cooperative_yield_ctx = nullptr
    );

    // Executes a single decoded non-branch R5900 instruction.
    static void exec_single(EEContext& ctx, const DecodedInstruction& ins);
};

} // namespace tf::ps2
