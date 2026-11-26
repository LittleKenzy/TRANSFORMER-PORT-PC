#pragma once

#include "../../Recompiler/Common/Types.h"
#include "../GS/GsState.h"
#include <array>

namespace tf::ps2 {

// Stateful Graphics Interface (GIF) FIFO: decodes GIFtags (PACKED / REGLIST / IMAGE)
// with independent stream states for PATH1 (VU1 XGKICK), PATH2 (VIF1 DIRECT), and
// PATH3 (GIF DMA / FIFO), seamlessly handling split DMA Source-Chain tags.
class GifUnit {
public:
    static constexpr int kPath1 = 0; // VU1 XGKICK
    static constexpr int kPath2 = 1; // VIF1 DIRECT
    static constexpr int kPath3 = 2; // GIF DMA / FIFO

    explicit GifUnit(GsState* gs = nullptr) : m_gs(gs) {}
    void attach_gs(GsState* gs) { m_gs = gs; }
    void reset();

    // Multi-path entry points (0=PATH1 VU1 XGKICK, 1=PATH2 VIF1 DIRECT, 2=PATH3 GIF DMA)
    void feed_path(int path, const u8* data, std::size_t qwords, GsState& gs);
    void feed_path(int path, const u8* data, std::size_t qwords, GsState& gs, bool stop_on_eop);

    // Backwards-compatible PATH3 entry point using attached GsState
    void process_qwords(const u8* qwords, std::size_t qwc, bool stop_on_eop = false);

    u64 tags_processed = 0;
    u64 registers_written = 0;

private:
    struct PathState {
        u32 nloop_rem = 0;
        u32 flg = 0;
        u32 nreg = 1;
        u32 cur_reg = 0;
        u64 regs = 0;
        u32 reglist_rem = 0;
        u32 packed_q = 0x3F800000u;
        bool eop = false;
    };

    GsState* m_gs = nullptr;
    std::array<PathState, 3> m_paths{};
};

} // namespace tf::ps2
