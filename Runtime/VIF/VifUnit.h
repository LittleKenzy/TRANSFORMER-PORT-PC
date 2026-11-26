#pragma once

#include "../../Recompiler/Common/Types.h"
#include <functional>
#include <vector>

namespace tf::ps2 {

// Stateful VIF0 / VIF1 packet processor: handles split DMA tags (including TTE=1),
// UNPACK into VU memory, MPG microprogram uploads, MSCAL/MSCNT, and DIRECT/DIRECTHL to GIF.
class VifUnit {
public:
    explicit VifUnit(int index = 1);
    void reset();

    void attach_vu_memory(u8* vu_code, std::size_t code_size, u8* vu_data, std::size_t data_size) {
        m_vu_code = vu_code;
        m_vu_code_size = code_size;
        m_vu_data = vu_data;
        m_vu_data_size = data_size;
    }

    void set_gif_direct_callback(std::function<void(const u8* qwords, std::size_t qwc)> cb) {
        m_gif_direct_cb = std::move(cb);
    }

    void set_mscal_callback(std::function<void(u32 micro_addr_bytes, bool is_mscnt)> cb) {
        m_mscal_cb = std::move(cb);
    }

    void set_mpg_callback(std::function<void(u32 start_byte, u32 byte_len)> cb) {
        m_mpg_cb = std::move(cb);
    }

    void process_packet(const u8* qwords, std::size_t qwc);
    void process_words(const u8* bytes, std::size_t num_words);
    u32 read_reg(u32 offset) const;
    void write_reg(u32 offset, u32 val);

    u32 cycle_cl = 1;
    u32 cycle_wl = 1;
    u32 offset = 0;
    u32 base = 0;
    u32 tops = 0;
    u32 top = 0;
    u32 itop = 0;
    u32 mode = 0;
    u32 mark = 0;
    bool mark_detected = false;
    u32 mask = 0;
    u32 row[4] = {};
    u32 col[4] = {};

    u64 commands_processed = 0;
    u64 unpacks_processed = 0;
    u64 direct_qwords_forwarded = 0;
    u64 mpg_uploads = 0;
    u64 mscal_count = 0;

    int index() const { return m_index; }

private:
    enum class State { Idle, Direct, Mpg, Unpack, StMask, StRow, StCol };
    void finish_unpack(const u8* src_bytes, std::size_t src_sz);

    int m_index = 1;
    u8* m_vu_code = nullptr;
    std::size_t m_vu_code_size = 0;
    u8* m_vu_data = nullptr;
    std::size_t m_vu_data_size = 0;
    std::function<void(const u8*, std::size_t)> m_gif_direct_cb;
    std::function<void(u32, bool)> m_mscal_cb;
    std::function<void(u32, u32)> m_mpg_cb;

    // Persistent FIFO streaming state across split DMA chunks
    State m_state = State::Idle;
    u32 m_words_rem = 0;
    u32 m_param_idx = 0;
    // DIRECT accumulation buffer
    std::vector<u8> m_direct_buf;
    // MPG destination cursor & dirty tracking
    u32 m_mpg_dst = 0;
    u32 m_mpg_start = 0;
    u32 m_mpg_total_bytes = 0;
    bool m_mpg_modified = false;
    // UNPACK accumulation
    u8  m_unpack_cmd = 0;
    u8  m_unpack_num = 0;
    u16 m_unpack_imm = 0;
    std::vector<u8> m_unpack_buf;
};

} // namespace tf::ps2
