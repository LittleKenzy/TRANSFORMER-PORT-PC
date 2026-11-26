#include "GsState.h"
#include "../PerfCounters.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(__AVX2__)
#include <immintrin.h>
#define TF_GS_HAS_AVX2 1
#define TF_GS_HAS_SSE2 1
#elif defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#include <emmintrin.h>
#define TF_GS_HAS_AVX2 0
#define TF_GS_HAS_SSE2 1
#else
#define TF_GS_HAS_AVX2 0
#define TF_GS_HAS_SSE2 0
#endif

namespace tf::ps2 {

namespace {

// Exact PlayStation 2 GS Local Memory (4 MiB VRAM) Page / Block / Column tables
constexpr u8 kBlock32[4][8] = {
    {  0,  1,  4,  5, 16, 17, 20, 21 },
    {  2,  3,  6,  7, 18, 19, 22, 23 },
    {  8,  9, 12, 13, 24, 25, 28, 29 },
    { 10, 11, 14, 15, 26, 27, 30, 31 }
};

constexpr u8 kBlock16[8][4] = {
    {  0,  2,  8, 10 },
    {  1,  3,  9, 11 },
    {  4,  6, 12, 14 },
    {  5,  7, 13, 15 },
    { 16, 18, 24, 26 },
    { 17, 19, 25, 27 },
    { 20, 22, 28, 30 },
    { 21, 23, 29, 31 }
};

constexpr u8 kBlock16S[8][4] = {
    {  0,  2, 16, 18 },
    {  1,  3, 17, 19 },
    {  8, 10, 24, 26 },
    {  9, 11, 25, 27 },
    {  4,  6, 20, 22 },
    {  5,  7, 21, 23 },
    { 12, 14, 28, 30 },
    { 13, 15, 29, 31 }
};

constexpr u8 kCol32[2][8] = {
    {  0,  1,  4,  5,  8,  9, 12, 13 },
    {  2,  3,  6,  7, 10, 11, 14, 15 }
};

// Exact PS2 GS VRAM byte addressing: bp is added directly in 64-word (256-byte) block units.
inline u32 vram_addr32(u32 bp, u32 bw, u32 x, u32 y) {
    const u32 bw_pages = std::max<u32>(1u, bw);
    const u32 page_off = (y >> 5u) * bw_pages + (x >> 6u);
    const u32 blk = (bp + page_off * 32u + kBlock32[(y >> 3u) & 3u][(x >> 3u) & 7u]) & 0x3FFFu;
    const u32 word = ((y >> 1u) & 3u) * 16u + kCol32[y & 1u][x & 7u];
    return ((blk * 64u + word) * 4u) & (kGsVramSize - 4u);
}

inline u32 vram_addr16(u32 bp, u32 bw, u32 x, u32 y, bool is_16s) {
    const u32 bw_pages = std::max<u32>(1u, bw);
    const u32 page_off = (y >> 6u) * bw_pages + (x >> 6u);
    const u32 b_idx = is_16s ? kBlock16S[(y >> 3u) & 7u][(x >> 4u) & 3u]
                             : kBlock16[(y >> 3u) & 7u][(x >> 4u) & 3u];
    const u32 blk = (bp + page_off * 32u + b_idx) & 0x3FFFu;
    // PS2 GS PSMCT16 column layout (16x2 pixels = 32 halfwords per column)
    const u32 hw = ((y >> 1u) & 3u) * 32u + ((x & 8u) << 1u) + kCol32[y & 1u][x & 7u];
    return ((blk * 128u + hw) * 2u) & (kGsVramSize - 2u);
}

inline u32 vram_addr8(u32 bp, u32 bw, u32 x, u32 y) {
    const u32 bw_pages = std::max<u32>(1u, bw >> 1u);
    const u32 page_off = (y >> 6u) * bw_pages + (x >> 7u);
    const u32 blk = (bp + page_off * 32u + kBlock32[(y >> 4u) & 3u][(x >> 4u) & 7u]) & 0x3FFFu;
    const u32 cx = x & 15u, cy = y & 15u;
    const u32 swap = (((cy + 2u) >> 2u) & 1u) * 4u;
    const u32 py   = (((cy & ~3u) >> 1u) + (cy & 1u)) & 7u;
    const u32 px   = (cx + swap) & 7u;
    const u32 word = (py >> 1u) * 16u + kCol32[py & 1u][px];
    const u32 byte_lane = ((cy >> 1u) & 1u) + ((cx >> 2u) & 2u);
    return ((blk * 64u + word) * 4u + byte_lane) & (kGsVramSize - 1u);
}

inline u32 vram_addr4(u32 bp, u32 bw, u32 x, u32 y, u32& out_shift) {
    const u32 bw_pages = std::max<u32>(1u, bw >> 1u);
    const u32 page_off = (y >> 7u) * bw_pages + (x >> 7u);
    const u32 blk = (bp + page_off * 32u + kBlock16[(y >> 4u) & 7u][(x >> 5u) & 3u]) & 0x3FFFu;
    const u32 cx = x & 31u, cy = y & 15u;
    const u32 swap = (((cy + 2u) >> 2u) & 1u) * 4u;
    const u32 py   = (((cy & ~3u) >> 1u) + (cy & 1u)) & 7u;
    const u32 px   = (cx ^ swap) & 7u;
    const u32 word = (py >> 1u) * 16u + kCol32[py & 1u][px];
    // PS2 GS PSMT4 column layout: x groups of 8 map to byte lanes 0..3; y&2 selects low/high nibble.
    const u32 byte_lane = (cx >> 3u) & 3u;
    out_shift = ((cy >> 1u) & 1u) * 4u;
    return ((blk * 64u + word) * 4u + byte_lane) & (kGsVramSize - 1u);
}

inline u32 psm_page_width(u32 psm) {
    return (psm == 0x13u || psm == 0x14u) ? 128u : 64u;
}

inline u32 psm_page_height(u32 psm) {
    switch (psm) {
        case 0x02u: case 0x0Au: case 0x32u: case 0x3Au: case 0x13u:
            return 64u;
        case 0x14u:
            return 128u;
        default:
            return 32u;
    }
}

inline u32 vram_block_span(u32 bw, u32 psm, u32 max_x_excl, u32 max_y_excl) {
    if (max_x_excl == 0u || max_y_excl == 0u) return 32u;
    const u32 pw = psm_page_width(psm);
    const u32 ph = psm_page_height(psm);
    const u32 bw_pages = (pw == 128u) ? std::max<u32>(1u, bw >> 1u) : std::max<u32>(1u, bw);
    const u32 pages_x = std::max<u32>(1u, (max_x_excl + pw - 1u) / pw);
    const u32 pages_y = std::max<u32>(1u, (max_y_excl + ph - 1u) / ph);
    return ((pages_y - 1u) * bw_pages + pages_x) * 32u;
}

inline bool ranges_overlap_vram(u32 a_start, u32 a_len, u32 b_start, u32 b_len) {
    if (a_len == 0u || b_len == 0u) return false;
    if (a_len >= 16384u || b_len >= 16384u) return true;
    const u32 a0 = a_start & 0x3FFFu;
    const u32 b0 = b_start & 0x3FFFu;
    const u32 a1 = a0 + a_len;
    const u32 b1 = b0 + b_len;
    if (a1 <= 16384u && b1 <= 16384u) {
        return (a0 < b1) && (b0 < a1);
    }
    auto check_seg = [](u32 s0, u32 e0, u32 s1, u32 e1) {
        return (s0 < e1) && (s1 < e0);
    };
    const u32 a_hi = std::min<u32>(a1, 16384u);
    const u32 a_wrap = (a1 > 16384u) ? (a1 - 16384u) : 0u;
    const u32 b_hi = std::min<u32>(b1, 16384u);
    const u32 b_wrap = (b1 > 16384u) ? (b1 - 16384u) : 0u;
    if (check_seg(a0, a_hi, b0, b_hi)) return true;
    if (a_wrap > 0u && check_seg(0u, a_wrap, b0, b_hi)) return true;
    if (b_wrap > 0u && check_seg(a0, a_hi, 0u, b_wrap)) return true;
    if (a_wrap > 0u && b_wrap > 0u) return true;
    return false;
}

inline u16 pack_rgba_to_psm16(u32 rgba) {
    const u16 r = static_cast<u16>((rgba >> 3) & 0x1Fu);
    const u16 g = static_cast<u16>((rgba >> 11) & 0x1Fu);
    const u16 b = static_cast<u16>((rgba >> 19) & 0x1Fu);
    const u16 a = ((rgba >> 24) >= 0x80u) ? 0x8000u : 0u;
    return static_cast<u16>(r | (g << 5) | (b << 10) | a);
}

inline u32 zbuf_psm_mask(u64 zbuf_reg) {
    const u32 zpsm = static_cast<u32>((zbuf_reg >> 24) & 0xFu);
    switch (zpsm) {
        case 0x01u: return 0x00FFFFFFu; // PSMZ24
        case 0x02u:
        case 0x0Au: return 0x0000FFFFu; // PSMZ16 / PSMZ16S
        default:    return 0xFFFFFFFFu; // PSMZ32
    }
}

inline bool is_paletted_psm(u32 psm) {
    return psm == 0x13u || psm == 0x14u || psm == 0x1Bu || psm == 0x24u || psm == 0x2Cu;
}

inline bool is_8bit_paletted_psm(u32 psm) {
    return psm == 0x13u || psm == 0x1Bu;
}

inline s32 fast_floor(f32 x) {
    // Clamp away NaN/Inf/extreme magnitudes so int cast is always well-defined
    if (!(x >= -65536.0f && x <= 65536.0f)) {
        if (x > 65536.0f) return 65536;
        if (x < -65536.0f) return -65536;
        return 0;
    }
    const s32 i = static_cast<s32>(x);
    return i - static_cast<s32>(x < static_cast<f32>(i));
}

inline u32 apply_clamp_axis(s32 coord, u32 tex_dim, u32 wm, s32 min_v, s32 max_v) {
    if (wm == 0u) {
        return static_cast<u32>(coord) & (tex_dim - 1u);
    }
    if (wm == 1u) {
        return static_cast<u32>(std::clamp<s32>(coord, 0, static_cast<s32>(tex_dim) - 1));
    }
    if (wm == 2u) {
        const s32 hi = std::min<s32>(max_v, static_cast<s32>(tex_dim) - 1);
        const s32 lo = std::min<s32>(min_v, hi);
        return static_cast<u32>(std::clamp<s32>(coord, lo, hi));
    }
    return ((static_cast<u32>(coord) & static_cast<u32>(min_v)) | static_cast<u32>(max_v)) & (tex_dim - 1u);
}

inline u32 bilerp_rgba(u32 c00, u32 c10, u32 c01, u32 c11, u32 fx, u32 fy) {
#if TF_GS_HAS_SSE2
    const __m128i zero = _mm_setzero_si128();
    const __m128i v0 = _mm_unpacklo_epi8(_mm_set_epi32(0, 0, static_cast<int>(c01), static_cast<int>(c00)), zero);
    const __m128i v1 = _mm_unpacklo_epi8(_mm_set_epi32(0, 0, static_cast<int>(c11), static_cast<int>(c10)), zero);
    const __m128i v_fx  = _mm_set1_epi16(static_cast<short>(fx));
    const __m128i v_ifx = _mm_set1_epi16(static_cast<short>(256u - fx));
    const __m128i h = _mm_srli_epi16(_mm_add_epi16(_mm_mullo_epi16(v0, v_ifx), _mm_mullo_epi16(v1, v_fx)), 8);
    const __m128i t0 = h;
    const __m128i t1 = _mm_srli_si128(h, 8);
    const __m128i v_fy  = _mm_set1_epi16(static_cast<short>(fy));
    const __m128i v_ify = _mm_set1_epi16(static_cast<short>(256u - fy));
    const __m128i v = _mm_srli_epi16(_mm_add_epi16(_mm_mullo_epi16(t0, v_ify), _mm_mullo_epi16(t1, v_fy)), 8);
    return static_cast<u32>(_mm_cvtsi128_si32(_mm_packus_epi16(v, zero)));
#else
    const u32 ifx = 256u - fx;
    const u32 ify = 256u - fy;
    const u32 w00 = ifx * ify;
    const u32 w10 = fx  * ify;
    const u32 w01 = ifx * fy;
    const u32 w11 = fx  * fy;
    const u32 r = ((c00 & 0xFFu) * w00 + (c10 & 0xFFu) * w10 + (c01 & 0xFFu) * w01 + (c11 & 0xFFu) * w11) >> 16u;
    const u32 g = (((c00 >> 8) & 0xFFu) * w00 + ((c10 >> 8) & 0xFFu) * w10 + ((c01 >> 8) & 0xFFu) * w01 + ((c11 >> 8) & 0xFFu) * w11) >> 16u;
    const u32 b = (((c00 >> 16) & 0xFFu) * w00 + ((c10 >> 16) & 0xFFu) * w10 + ((c01 >> 16) & 0xFFu) * w01 + ((c11 >> 16) & 0xFFu) * w11) >> 16u;
    const u32 a = (((c00 >> 24) & 0xFFu) * w00 + ((c10 >> 24) & 0xFFu) * w10 + ((c01 >> 24) & 0xFFu) * w01 + ((c11 >> 24) & 0xFFu) * w11) >> 16u;
    return r | (g << 8) | (b << 16) | (a << 24);
#endif
}

inline f32 sanitize_f32(f32 v, f32 fallback = 0.0f) {
    return std::isfinite(v) ? v : fallback;
}

inline s64 fast_f32_to_fp8(f32 x) {
    const f32 c = (x < -65536.0f) ? -65536.0f : ((x > 65536.0f) ? 65536.0f : x);
    return static_cast<s64>(c * 256.0f);
}

inline s32 fast_f32_to_fp16(f32 x) {
    const f32 c = (x < -16384.0f) ? -16384.0f : ((x > 16384.0f) ? 16384.0f : x);
    return static_cast<s32>(c * 65536.0f);
}

inline s64 fast_f64_to_fp16(double x) {
    const double c = (x < -4294967295.0) ? -4294967295.0 : ((x > 8589934590.0) ? 8589934590.0 : x);
    return static_cast<s64>(c * 65536.0);
}

inline bool alpha_test_pass(u32 a, u32 atst, u32 aref) {
    switch (atst) {
        case 0u: return false;
        case 1u: return true;
        case 2u: return a <  aref;
        case 3u: return a <= aref;
        case 4u: return a == aref;
        case 5u: return a >= aref;
        case 6u: return a >  aref;
        case 7u: return a != aref;
        default: return true;
    }
}

inline u32 clamp_fp16_u8(s32 fp) {
    if (fp <= 0) return 0u;
    if (fp >= (255 << 16)) return 255u;
    return static_cast<u32>(fp >> 16);
}

inline u32 clamp_z_fp(s64 z_fp, u32 z_mask) {
    const s64 zi = z_fp >> 16;
    if (zi <= 0) return 0u;
    if (zi >= 0xFFFFFFFFLL) return 0xFFFFFFFFu & z_mask;
    return static_cast<u32>(zi) & z_mask;
}

inline u32 clamp_axis_fast(s32 coord, u32 tex_dim, u32 wm, s32 min_v, s32 max_v) {
    if (wm == 0u) return static_cast<u32>(coord) & (tex_dim - 1u);
    if (wm == 1u) {
        if (coord <= 0) return 0u;
        const s32 hi = static_cast<s32>(tex_dim) - 1;
        return static_cast<u32>(coord >= hi ? hi : coord);
    }
    return apply_clamp_axis(coord, tex_dim, wm, min_v, max_v);
}

inline u32 z_pass_mask_4px(const u32 pz[4], const u32* cur_z, u32 ztst) {
    if (ztst == 1u) return 0xFu;
#if TF_GS_HAS_SSE2
    const __m128i sign_bit = _mm_set1_epi32(static_cast<int>(0x80000000u));
    const __m128i vp = _mm_xor_si128(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pz)), sign_bit);
    const __m128i vc = _mm_xor_si128(_mm_loadu_si128(reinterpret_cast<const __m128i*>(cur_z)), sign_bit);
    const __m128i pass = (ztst == 2u)
        ? _mm_andnot_si128(_mm_cmpgt_epi32(vc, vp), _mm_set1_epi32(-1))
        : _mm_cmpgt_epi32(vp, vc);
    return static_cast<u32>(_mm_movemask_ps(_mm_castsi128_ps(pass)));
#else
    u32 mask = 0u;
    if (ztst == 2u) {
        for (int i = 0; i < 4; ++i) if (pz[i] >= cur_z[i]) mask |= (1u << i);
    } else {
        for (int i = 0; i < 4; ++i) if (pz[i] > cur_z[i]) mask |= (1u << i);
    }
    return mask;
#endif
}

inline u32 modulate_px_tfx0(u32 tex, u32 vcol, bool tcc) {
#if TF_GS_HAS_SSE2
    const __m128i zero = _mm_setzero_si128();
    const __m128i t16 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(static_cast<int>(tex)), zero);
    const __m128i v16 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(static_cast<int>(vcol)), zero);
    const __m128i prod = _mm_srli_epi16(_mm_mullo_epi16(t16, v16), 7);
    const u32 res = static_cast<u32>(_mm_cvtsi128_si32(_mm_packus_epi16(prod, zero)));
    return tcc ? res : ((res & 0x00FFFFFFu) | (vcol & 0xFF000000u));
#else
    const u32 r = std::min<u32>(255u, ((tex & 0xFFu) * (vcol & 0xFFu)) >> 7);
    const u32 g = std::min<u32>(255u, (((tex >> 8) & 0xFFu) * ((vcol >> 8) & 0xFFu)) >> 7);
    const u32 b = std::min<u32>(255u, (((tex >> 16) & 0xFFu) * ((vcol >> 16) & 0xFFu)) >> 7);
    const u32 va = (vcol >> 24) & 0xFFu;
    const u32 a = tcc ? std::min<u32>(255u, (((tex >> 24) & 0xFFu) * va) >> 7) : va;
    return r | (g << 8) | (b << 16) | (a << 24);
#endif
}

inline void modulate_4px_tfx0(const u32 tex4[4], const u32 vcol4[4], u32 out4[4], bool tcc) {
#if TF_GS_HAS_AVX2
    const __m128i t128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(tex4));
    const __m128i v128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(vcol4));
    const __m256i t256 = _mm256_cvtepu8_epi16(t128);
    const __m256i v256 = _mm256_cvtepu8_epi16(v128);
    const __m256i prod = _mm256_srli_epi16(_mm256_mullo_epi16(t256, v256), 7);
    __m128i res = _mm_packus_epi16(_mm256_castsi256_si128(prod), _mm256_extracti128_si256(prod, 1));
    if (!tcc) {
        const __m128i mask_rgb = _mm_set1_epi32(0x00FFFFFF);
        const __m128i mask_a   = _mm_set1_epi32(static_cast<int>(0xFF000000u));
        res = _mm_or_si128(_mm_and_si128(res, mask_rgb), _mm_and_si128(v128, mask_a));
    }
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out4), res);
#elif TF_GS_HAS_SSE2
    const __m128i zero = _mm_setzero_si128();
    const __m128i t128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(tex4));
    const __m128i v128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(vcol4));
    const __m128i plo = _mm_srli_epi16(_mm_mullo_epi16(_mm_unpacklo_epi8(t128, zero), _mm_unpacklo_epi8(v128, zero)), 7);
    const __m128i phi = _mm_srli_epi16(_mm_mullo_epi16(_mm_unpackhi_epi8(t128, zero), _mm_unpackhi_epi8(v128, zero)), 7);
    __m128i res = _mm_packus_epi16(plo, phi);
    if (!tcc) {
        const __m128i mask_rgb = _mm_set1_epi32(0x00FFFFFF);
        const __m128i mask_a   = _mm_set1_epi32(static_cast<int>(0xFF000000u));
        res = _mm_or_si128(_mm_and_si128(res, mask_rgb), _mm_and_si128(v128, mask_a));
    }
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out4), res);
#else
    for (int i = 0; i < 4; ++i) {
        out4[i] = modulate_px_tfx0(tex4[i], vcol4[i], tcc);
    }
#endif
}

inline u32 blend_std_rgba_px(u32 src_px, u32 dst_px, s32 C) {
#if TF_GS_HAS_SSE2
    const __m128i zero = _mm_setzero_si128();
    const __m128i s16 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(static_cast<int>(src_px)), zero);
    const __m128i d16 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(static_cast<int>(dst_px)), zero);
    const __m128i s32v = _mm_unpacklo_epi16(s16, zero);
    const __m128i d32v = _mm_unpacklo_epi16(d16, zero);
    const __m128i diff = _mm_sub_epi16(s32v, d32v); // [sr-dr, 0, sg-dg, 0, sb-db, 0, sa-da, 0]
    const __m128i prod = _mm_madd_epi16(diff, _mm_set1_epi32(C & 0xFFFF));
    const __m128i res32 = _mm_add_epi32(_mm_srai_epi32(prod, 7), d32v);
    const __m128i res8 = _mm_packus_epi16(_mm_packs_epi32(res32, zero), zero);
    return (static_cast<u32>(_mm_cvtsi128_si32(res8)) & 0x00FFFFFFu) | (src_px & 0xFF000000u);
#else
    const s32 dr = static_cast<s32>(dst_px & 0xFFu);
    const s32 dg = static_cast<s32>((dst_px >> 8) & 0xFFu);
    const s32 db = static_cast<s32>((dst_px >> 16) & 0xFFu);
    const u32 r = static_cast<u32>(std::clamp((((static_cast<s32>(src_px & 0xFFu) - dr) * C) >> 7) + dr, 0, 255));
    const u32 g = static_cast<u32>(std::clamp((((static_cast<s32>((src_px >> 8) & 0xFFu) - dg) * C) >> 7) + dg, 0, 255));
    const u32 b = static_cast<u32>(std::clamp((((static_cast<s32>((src_px >> 16) & 0xFFu) - db) * C) >> 7) + db, 0, 255));
    return r | (g << 8) | (b << 16) | (src_px & 0xFF000000u);
#endif
}

inline void blend_std_2px(u32 s0, u32 s1, u32 d0, u32 d1, s32 c0, s32 c1, u32& o0, u32& o1) {
#if TF_GS_HAS_AVX2
    const __m128i s64 = _mm_set_epi32(0, 0, static_cast<int>(s1), static_cast<int>(s0));
    const __m128i d64 = _mm_set_epi32(0, 0, static_cast<int>(d1), static_cast<int>(d0));
    const __m256i s32v = _mm256_cvtepu8_epi32(s64);
    const __m256i d32v = _mm256_cvtepu8_epi32(d64);
    const __m256i vc32 = _mm256_set_epi32(0, c1, c1, c1, 0, c0, c0, c0);
    const __m256i prod = _mm256_mullo_epi32(_mm256_sub_epi32(s32v, d32v), vc32);
    const __m256i res32 = _mm256_add_epi32(_mm256_srai_epi32(prod, 7), d32v);
    const __m128i p16 = _mm_packs_epi32(_mm256_castsi256_si128(res32), _mm256_extracti128_si256(res32, 1));
    const __m128i p8 = _mm_packus_epi16(p16, p16);
    alignas(16) u32 tmp[4];
    _mm_storeu_si128(reinterpret_cast<__m128i*>(tmp), p8);
    o0 = (tmp[0] & 0x00FFFFFFu) | (s0 & 0xFF000000u);
    o1 = (tmp[1] & 0x00FFFFFFu) | (s1 & 0xFF000000u);
#else
    o0 = blend_std_rgba_px(s0, d0, c0);
    o1 = blend_std_rgba_px(s1, d1, c1);
#endif
}

inline u32 blend_add_rgba_px(u32 src_px, u32 dst_px, s32 C) {
#if TF_GS_HAS_SSE2
    const __m128i zero = _mm_setzero_si128();
    const __m128i s16 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(static_cast<int>(src_px)), zero);
    const __m128i d16 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(static_cast<int>(dst_px)), zero);
    const __m128i scaled = _mm_srli_epi16(_mm_mullo_epi16(s16, _mm_set1_epi16(static_cast<short>(C))), 7);
    const __m128i res8 = _mm_packus_epi16(_mm_add_epi16(scaled, d16), zero);
    return (static_cast<u32>(_mm_cvtsi128_si32(res8)) & 0x00FFFFFFu) | (src_px & 0xFF000000u);
#else
    const u32 r = std::min<u32>(255u, (((src_px & 0xFFu) * static_cast<u32>(C)) >> 7) + (dst_px & 0xFFu));
    const u32 g = std::min<u32>(255u, ((((src_px >> 8) & 0xFFu) * static_cast<u32>(C)) >> 7) + ((dst_px >> 8) & 0xFFu));
    const u32 b = std::min<u32>(255u, ((((src_px >> 16) & 0xFFu) * static_cast<u32>(C)) >> 7) + ((dst_px >> 16) & 0xFFu));
    return r | (g << 8) | (b << 16) | (src_px & 0xFF000000u);
#endif
}

inline void blend_add_2px(u32 s0, u32 s1, u32 d0, u32 d1, s32 c0, s32 c1, u32& o0, u32& o1) {
#if TF_GS_HAS_SSE2
    const __m128i zero = _mm_setzero_si128();
    const __m128i s16 = _mm_unpacklo_epi8(_mm_set_epi32(0, 0, static_cast<int>(s1), static_cast<int>(s0)), zero);
    const __m128i d16 = _mm_unpacklo_epi8(_mm_set_epi32(0, 0, static_cast<int>(d1), static_cast<int>(d0)), zero);
    const __m128i vc16 = _mm_set_epi16(static_cast<short>(c1), static_cast<short>(c1), static_cast<short>(c1), static_cast<short>(c1),
                                       static_cast<short>(c0), static_cast<short>(c0), static_cast<short>(c0), static_cast<short>(c0));
    const __m128i res8 = _mm_packus_epi16(_mm_add_epi16(_mm_srli_epi16(_mm_mullo_epi16(s16, vc16), 7), d16), zero);
    alignas(16) u32 tmp[4];
    _mm_storeu_si128(reinterpret_cast<__m128i*>(tmp), res8);
    o0 = (tmp[0] & 0x00FFFFFFu) | (s0 & 0xFF000000u);
    o1 = (tmp[1] & 0x00FFFFFFu) | (s1 & 0xFF000000u);
#else
    o0 = blend_add_rgba_px(s0, d0, c0);
    o1 = blend_add_rgba_px(s1, d1, c1);
#endif
}

inline u32 lerp_rgb_opaque_px(u32 p0, u32 p1, u32 frac) {
#if TF_GS_HAS_SSE2
    const __m128i zero = _mm_setzero_si128();
    const __m128i v0 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(static_cast<int>(p0)), zero);
    const __m128i v1 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(static_cast<int>(p1)), zero);
    const __m128i vf  = _mm_set1_epi16(static_cast<short>(frac));
    const __m128i vif = _mm_set1_epi16(static_cast<short>(256u - frac));
    const __m128i h = _mm_srli_epi16(_mm_add_epi16(_mm_mullo_epi16(v0, vif), _mm_mullo_epi16(v1, vf)), 8);
    return static_cast<u32>(_mm_cvtsi128_si32(_mm_packus_epi16(h, zero))) | 0xFF000000u;
#else
    const u32 invf = 256u - frac;
    const u32 r = ((p0 & 0xFFu) * invf + (p1 & 0xFFu) * frac) >> 8u;
    const u32 g = (((p0 >> 8) & 0xFFu) * invf + ((p1 >> 8) & 0xFFu) * frac) >> 8u;
    const u32 b = (((p0 >> 16) & 0xFFu) * invf + ((p1 >> 16) & 0xFFu) * frac) >> 8u;
    return r | (g << 8) | (b << 16) | 0xFF000000u;
#endif
}

inline void copy_row_opaque_simd(u32* dst, const u32* src, u32 count) {
    u32 x = 0;
#if TF_GS_HAS_AVX2
    const __m256i alpha256 = _mm256_set1_epi32(static_cast<int>(0xFF000000u));
    for (; x + 8u <= count; x += 8u) {
        const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + x));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + x), _mm256_or_si256(v, alpha256));
    }
#endif
#if TF_GS_HAS_SSE2
    const __m128i alpha128 = _mm_set1_epi32(static_cast<int>(0xFF000000u));
    for (; x + 4u <= count; x += 4u) {
        const __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + x));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + x), _mm_or_si128(v, alpha128));
    }
#endif
    for (; x < count; ++x) {
        dst[x] = src[x] | 0xFF000000u;
    }
}

inline void scale_row_opaque_simd(u32* drow, const u32* srow,
                                  const u16* sx0_lut, const u16* sx1_lut, const u16* frac_lut,
                                  u32 width) {
    u32 x = 0;
#if TF_GS_HAS_SSE2
    const __m128i zero = _mm_setzero_si128();
    const __m128i alpha128 = _mm_set1_epi32(static_cast<int>(0xFF000000u));
    const __m128i c256 = _mm_set1_epi16(256);
    for (; x + 2u <= width; x += 2u) {
        const u32 p0a = srow[sx0_lut[x]],     p1a = srow[sx1_lut[x]];
        const u32 p0b = srow[sx0_lut[x + 1]], p1b = srow[sx1_lut[x + 1]];
        const short fa = static_cast<short>(frac_lut[x]);
        const short fb = static_cast<short>(frac_lut[x + 1]);
        const __m128i v0 = _mm_unpacklo_epi8(_mm_set_epi32(0, 0, static_cast<int>(p0b), static_cast<int>(p0a)), zero);
        const __m128i v1 = _mm_unpacklo_epi8(_mm_set_epi32(0, 0, static_cast<int>(p1b), static_cast<int>(p1a)), zero);
        const __m128i vf = _mm_set_epi16(fb, fb, fb, fb, fa, fa, fa, fa);
        const __m128i vif = _mm_sub_epi16(c256, vf);
        const __m128i h = _mm_srli_epi16(_mm_add_epi16(_mm_mullo_epi16(v0, vif), _mm_mullo_epi16(v1, vf)), 8);
        const __m128i packed = _mm_or_si128(_mm_packus_epi16(h, zero), alpha128);
        _mm_storel_epi64(reinterpret_cast<__m128i*>(drow + x), packed);
    }
#endif
    for (; x < width; ++x) {
        drow[x] = lerp_rgb_opaque_px(srow[sx0_lut[x]], srow[sx1_lut[x]], frac_lut[x]);
    }
}

} // namespace

GsState::GsState()
    : m_vram(kGsVramSize, 0),
      m_present_fb(640u * 448u, 0xFF000000u) {
    for (auto& buf : m_triple_fb) {
        buf.assign(640u * 448u, 0xFF000000u);
    }
    m_tri_batch.reserve(256);
    reset();
}

GsState::~GsState() {
    if (!m_workers.empty()) {
        {
            std::lock_guard<std::mutex> lk(m_pool_mtx);
            m_pool_stop.store(true, std::memory_order_release);
        }
        m_pool_cv.notify_all();
        for (auto& t : m_workers) {
            if (t.joinable()) t.join();
        }
        m_workers.clear();
    }
}

void GsState::ensure_workers_started() {
    if (m_workers_init_attempted) return;
    m_workers_init_attempted = true;
    const unsigned hw = std::thread::hardware_concurrency();
    if (hw >= 12u) m_num_bands = 8;
    else if (hw >= 8u) m_num_bands = 6;
    else if (hw >= 4u) m_num_bands = 4;
    else {
        m_num_bands = 1;
        return;
    }

    const int worker_count = m_num_bands - 1;
    m_pool_stop.store(false, std::memory_order_release);
    m_workers.reserve(static_cast<std::size_t>(worker_count));
    for (int w = 0; w < worker_count; ++w) {
        const int band_idx = w + 1;
        const int num_bands = m_num_bands;
        m_workers.emplace_back([this, band_idx, num_bands]() {
            u64 my_gen = 0;
            while (true) {
                u64 cur = m_job_gen.load(std::memory_order_acquire);
                for (int spin = 0; spin < 16384 && cur == my_gen && !m_pool_stop.load(std::memory_order_acquire); ++spin) {
#if TF_GS_HAS_SSE2
                    _mm_pause();
#endif
                    cur = m_job_gen.load(std::memory_order_acquire);
                }
                if (cur == my_gen && !m_pool_stop.load(std::memory_order_acquire)) {
                    m_workers_sleeping.fetch_add(1u, std::memory_order_seq_cst);
                    {
                        std::unique_lock<std::mutex> lk(m_pool_mtx);
                        m_pool_cv.wait(lk, [&]() {
                            return m_job_gen.load(std::memory_order_seq_cst) != my_gen ||
                                   m_pool_stop.load(std::memory_order_seq_cst);
                        });
                    }
                    m_workers_sleeping.fetch_sub(1u, std::memory_order_seq_cst);
                    cur = m_job_gen.load(std::memory_order_acquire);
                }
                if (m_pool_stop.load(std::memory_order_acquire)) break;
                if (cur != my_gen) {
                    my_gen = cur;
                    m_job_fn(m_job_ctx, band_idx, num_bands);
                    m_workers_done.fetch_add(1u, std::memory_order_release);
                }
            }
        });
    }
}

void GsState::dispatch_bands(void (*fn)(void*, int, int), void* ctx) {
    ensure_workers_started();
    if (m_workers.empty()) {
        fn(ctx, 0, 1);
        return;
    }

    m_job_fn = fn;
    m_job_ctx = ctx;
    m_workers_done.store(0u, std::memory_order_relaxed);
    m_job_gen.fetch_add(1u, std::memory_order_seq_cst);
    if (m_workers_sleeping.load(std::memory_order_seq_cst) > 0u) {
        std::lock_guard<std::mutex> lk(m_pool_mtx);
        m_pool_cv.notify_all();
    }

    // Main thread executes Band 0 immediately while workers execute Bands 1..(m_num_bands-1)
    fn(ctx, 0, m_num_bands);

    const u32 target = static_cast<u32>(m_workers.size());
    while (m_workers_done.load(std::memory_order_acquire) < target) {
#if TF_GS_HAS_SSE2
        _mm_pause();
#endif
    }
}

u32 GsState::unpack_psm16_texa(u16 c) const {
    const u32 r = (c & 0x1Fu) << 3;
    const u32 g = ((c >> 5) & 0x1Fu) << 3;
    const u32 b = ((c >> 10) & 0x1Fu) << 3;
    const u32 a = (c & 0x8000u) ? m_texa_ta1 : ((m_texa_aem && (c & 0x7FFFu) == 0u) ? 0u : m_texa_ta0);
    return r | (g << 8) | (b << 16) | (a << 24);
}

u32 GsState::unpack_clut_raw(u32 raw, u32 cpsm) const {
    switch (cpsm) {
        case 0x00u:
            return raw;
        case 0x02u:
        case 0x0Au:
            return unpack_psm16_texa(static_cast<u16>(raw & 0xFFFFu));
        default:
            return raw;
    }
}

void GsState::invalidate_draw_cache() {
    m_cached_rt = nullptr;
    m_cached_zt = nullptr;
    m_cached_rt_fbp = 0xFFFFFFFFu;
    m_cached_zt_zbp = 0xFFFFFFFFu;
}

void GsState::bump_vram_pages(u32 start_block, u32 num_blocks) {
    m_vram_write_gen++;
    if (m_vram_write_gen == 0u) m_vram_write_gen = 1u;
    const u32 first_page = (start_block >> 5u) & 511u;
    const u32 total_blocks = num_blocks + (start_block & 31u);
    const u32 num_pages = std::min<u32>(512u, std::max<u32>(1u, (total_blocks + 31u) >> 5u));
    for (u32 p = 0; p < num_pages; ++p) {
        u32& g = m_vram_page_gen[(first_page + p) & 511u];
        g++;
        if (g == 0u) g = 1u;
    }
}

u64 GsState::compute_page_gen_hash(u32 first_page, u32 num_pages) const {
    u64 h = 14695981039346656037ULL;
    const u32 n = std::min<u32>(512u, num_pages);
    for (u32 p = 0; p < n; ++p) {
        h ^= static_cast<u64>(m_vram_page_gen[(first_page + p) & 511u]);
        h *= 1099511628211ULL;
    }
    return h;
}

void GsState::reset() {
    m_tri_batch.clear();
    m_tri_batch_min_y = 0x7FFFFFFF;
    m_tri_batch_max_y = -1;

    m_pmode = m_smode2 = m_dispfb1 = m_display1 = m_dispfb2 = m_display2 = 0;
    m_bgcolor = 0;
    m_csr = 0x551B4000ULL; // GS Revision ID + idle status
    m_imr = 0xFF00ULL;
    m_prim_reg = 0;
    m_prmode_reg = 0;
    m_prmodecont = true;
    m_prim_type = 0;
    m_cur_r = m_cur_g = m_cur_b = 0;
    m_cur_a = 128;
    m_cur_s = m_cur_t = 0.0f;
    m_cur_q = 1.0f;
    m_cur_u = m_cur_v = 0.0f;
    for (auto& c : m_ctx) {
        c = GsContext{};
        c.scissor_x0 = 0; c.scissor_x1 = static_cast<s32>(m_fb_width - 1);
        c.scissor_y0 = 0; c.scissor_y1 = static_cast<s32>(m_fb_height - 1);
    }
    m_texa_ta0 = 0x80;
    m_texa_ta1 = 0x80;
    m_texa_aem = false;
    m_pabe = false;
    m_scanmsk = 0;
    m_texclut_cbw = 1;
    m_texclut_cou = 0;
    m_texclut_cov = 0;
    m_bitblt_sbp = 0; m_bitblt_sbw = 1; m_bitblt_spsm = 0;
    m_bitblt_dbp = 0; m_bitblt_dbw = 1; m_bitblt_dpsm = 0;
    m_trx_ssax = m_trx_ssay = 0;
    m_trx_dsax = m_trx_dsay = m_trx_dir = 0;
    m_trx_rrw = m_trx_rrh = 0;
    m_trx_x = m_trx_y = 0;
    m_trx_active = false;
    m_rgb24_cnt = 0;
    m_vtx_queue.clear();

    m_clut_raw.fill(0);
    m_clut_rgba.fill(0);
    m_clut_slot_cpsm.fill(0);
    m_clut_explicit_cld = false;
    m_clut_fallback_valid = false;
    m_cbp0 = 0xFFFFFFFFu;
    m_cbp1 = 0xFFFFFFFFu;
    m_clut_gen = 1;
    for (u32 b = 0; b < 32u; ++b) update_clut_block_hash(b);

    m_vram_page_gen.fill(0);
    m_vram_page_color_dbw.fill(0);
    m_vram_write_gen = 1;
    m_tex_cache_tick = 1;
    m_hw_tex_gen = 1;
    for (auto& e : m_tex_cache) {
        e.valid = false;
        e.key = 0;
        e.last_use = 0;
        e.gpu_gen = 0;
    }
    m_self_blit_scratch.clear();
    m_hw_verts.clear();
    if (m_hw) m_hw->on_reset();
    invalidate_draw_cache();

    m_scale_disp_w = 0;
    m_scale_dbx = 0;
    m_scale_buf_w = 0;
    for (auto& rt : m_fb_targets) {
        rt.pixels.clear();
        rt.fbw = 0;
        rt.height = 0;
        rt.psm = 0;
        rt.dirty_min_y = 0xFFFFFFFFu;
        rt.dirty_max_y = 0;
        rt.valid = false;
    }
    m_valid_fbps.clear();
    m_dirty_fbps.clear();
    for (auto& zt : m_z_targets) {
        zt.z.clear();
        zt.stride = 0;
        zt.height = 0;
    }
    m_drew_to_disp_fbp = false;
    m_last_presented_fbp = 0;
    m_vblanks_since_present = 0;
    m_last_present_cb_us = 0;

    std::fill(m_vram.begin(), m_vram.end(), u8{0});
    std::fill(m_present_fb.begin(), m_present_fb.end(), 0xFF000000u);
    m_last_nonblack_fb.clear();
    m_has_presented_single_thread = false;
    {
        std::lock_guard<std::mutex> lk(m_fb_mutex);
        m_has_new_ready = false;
        m_has_presented = false;
        m_ready_w = 0;
        m_ready_h = 0;
        m_ready_seq = 0;
        m_presented_seq.store(0u, std::memory_order_release);
    }
    primitives_drawn = 0;
    textured_primitives = 0;
    hwreg_uploads = 0;
    pixels_drawn = 0;
    frames_presented = 0;
    debug_log.clear();
}

u64 GsState::read_priv_reg(u32 paddr) const {
    const u32 base = paddr & ~7u;
    const bool hi = (paddr & 4u) != 0u;
    u64 val = 0;
    switch (base) {
        case 0x12000000u: val = m_pmode; break;
        case 0x12000020u: val = m_smode2; break;
        case 0x12000070u: val = m_dispfb1; break;
        case 0x12000080u: val = m_display1; break;
        case 0x12000090u: val = m_dispfb2; break;
        case 0x120000A0u: val = m_display2; break;
        case 0x120000E0u: val = m_bgcolor; break;
        case 0x12001000u: val = m_csr | 0x08ULL; break; // VSINT ready
        case 0x12001010u: val = m_imr; break;
        default: return 0;
    }
    return hi ? (val >> 32) : val;
}

void GsState::on_dispfb_written(int circuit) {
    const bool en1 = (m_pmode & 1u) != 0u;
    const bool en2 = (m_pmode & 2u) != 0u;

    if (!en1 && !en2) {
        if (circuit == 1) {
            present_dispfb(m_dispfb1, m_display1);
        } else if (m_dispfb1 == 0u) {
            present_dispfb(m_dispfb2, m_display2 != 0u ? m_display2 : m_display1);
        }
        return;
    }
    if (en1 && !en2) {
        if (circuit == 1) {
            present_dispfb(m_dispfb1, m_display1 != 0u ? m_display1 : m_display2);
        }
        return;
    }
    if (!en1 && en2) {
        if (circuit == 2) {
            present_dispfb(m_dispfb2, m_display2 != 0u ? m_display2 : m_display1);
        }
        return;
    }
    // Both circuits enabled: present once on DISPFB2 (written second in sceGsPutDispEnv),
    // or fallback to DISPFB1 only if DISPFB2 has never been configured.
    if (circuit == 2) {
        present_dispfb(m_dispfb2, m_display2 != 0u ? m_display2 : m_display1);
    } else if (m_dispfb2 == 0u) {
        present_dispfb(m_dispfb1, m_display1 != 0u ? m_display1 : m_display2);
    }
}

void GsState::write_priv_reg(u32 paddr, u64 val) {
    switch (paddr) {
        case 0x12000000u: m_pmode = val; break;
        case 0x12000020u: m_smode2 = val; break;
        case 0x12000070u: m_dispfb1 = val; on_dispfb_written(1); break;
        case 0x12000080u: m_display1 = val; break;
        case 0x12000090u: m_dispfb2 = val; on_dispfb_written(2); break;
        case 0x120000A0u: m_display2 = val; break;
        case 0x120000E0u: m_bgcolor = val; break;
        case 0x12001000u:
            if (val & (1ULL << 9)) reset();
            else m_csr &= ~(val & 0x1Fu);
            break;
        case 0x12001010u: m_imr = val; break;
        default: break;
    }
}

void GsState::write_priv_reg32(u32 addr, u32 val) {
    const u32 base = addr & ~7u;
    const bool hi = (addr & 4u) != 0u;
    auto merge = [hi, val](u64& reg) {
        if (hi) {
            reg = (reg & 0x00000000FFFFFFFFULL) | (static_cast<u64>(val) << 32);
        } else {
            reg = (reg & 0xFFFFFFFF00000000ULL) | static_cast<u64>(val);
        }
    };

    switch (base) {
        case 0x12000000u: merge(m_pmode); break;
        case 0x12000020u: merge(m_smode2); break;
        case 0x12000070u: {
            const u64 prev = m_dispfb1;
            merge(m_dispfb1);
            // Avoid 2x double-present when games store DISPFB via two 32-bit writes (low + unchanged high=0)
            if (!hi || m_dispfb1 != prev) on_dispfb_written(1);
            break;
        }
        case 0x12000080u: merge(m_display1); break;
        case 0x12000090u: {
            const u64 prev = m_dispfb2;
            merge(m_dispfb2);
            if (!hi || m_dispfb2 != prev) on_dispfb_written(2);
            break;
        }
        case 0x120000A0u: merge(m_display2); break;
        case 0x120000E0u: merge(m_bgcolor); break;
        case 0x12001000u:
            if (!hi) {
                if (val & (1u << 9)) reset();
                else m_csr &= ~(static_cast<u64>(val) & 0x1Fu);
            }
            break;
        case 0x12001010u: merge(m_imr); break;
        default: break;
    }
}

void GsState::set_packed_q(u32 qbits) {
    if (qbits != 0) std::memcpy(&m_cur_q, &qbits, 4);
    else m_cur_q = 1.0f;
}

void GsState::write_internal_reg(u8 reg, u64 val) {
    switch (reg) {
        case 0x00: { // PRIM
            const u32 new_prim = static_cast<u32>(val & 0x7FFu);
            const u32 new_type = new_prim & 7u;
            if (!m_tri_batch.empty()) {
                const bool same_tri_attrs = (new_type >= 3u && new_type <= 5u) &&
                    (!m_prmodecont || ((new_prim & 0x7F8u) == (m_prim_reg & 0x7F8u)));
                if (!same_tri_attrs) {
                    flush_tri_batch();
                }
            }
            m_prim_reg = new_prim;
            m_prim_type = new_type;
            m_vtx_queue.clear();
            break;
        }
        case 0x01: { // RGBAQ
            m_cur_r = static_cast<u8>(val & 0xFFu);
            m_cur_g = static_cast<u8>((val >> 8) & 0xFFu);
            m_cur_b = static_cast<u8>((val >> 16) & 0xFFu);
            m_cur_a = static_cast<u8>((val >> 24) & 0xFFu);
            const u32 qbits = static_cast<u32>(val >> 32);
            if (qbits != 0) std::memcpy(&m_cur_q, &qbits, 4);
            else m_cur_q = 1.0f;
            break;
        }
        case 0x02: { // ST
            const u32 sbits = static_cast<u32>(val & 0xFFFFFFFFu);
            const u32 tbits = static_cast<u32>(val >> 32);
            std::memcpy(&m_cur_s, &sbits, 4);
            std::memcpy(&m_cur_t, &tbits, 4);
            break;
        }
        case 0x03: // UV (10.4 fixed-point texel coordinates)
            m_cur_u = static_cast<f32>(val & 0x3FFFu) * (1.0f / 16.0f);
            m_cur_v = static_cast<f32>((val >> 16) & 0x3FFFu) * (1.0f / 16.0f);
            break;
        case 0x04: // XYZF2 (drawing kick, 24-bit Z + 8-bit Fog)
            kick_vertex(val & 0x00FFFFFFFFFFFFFFULL, true);
            break;
        case 0x05: // XYZ2 (drawing kick, 32-bit Z)
            kick_vertex(val, true);
            break;
        case 0x0C: // XYZF3 (no kick, 24-bit Z + 8-bit Fog)
            kick_vertex(val & 0x00FFFFFFFFFFFFFFULL, false);
            break;
        case 0x0D: // XYZ3 (no kick, 32-bit Z)
            kick_vertex(val, false);
            break;
        case 0x06: // TEX0_1
        case 0x07: { // TEX0_2
            const u32 reg_ctx = (reg == 0x06) ? 0u : 1u;
            if (!m_tri_batch.empty()) {
                const u32 cld = static_cast<u32>((val >> 61) & 7u);
                const u32 psm = static_cast<u32>((val >> 20) & 0x3Fu);
                bool will_load_clut = false;
                if (is_paletted_psm(psm)) {
                    const u32 cbp = static_cast<u32>((val >> 37) & 0x3FFFu);
                    if (cld == 1u || cld == 2u || cld == 3u) will_load_clut = true;
                    else if (cld == 4u) will_load_clut = (cbp != m_cbp0);
                    else if (cld == 5u) will_load_clut = (cbp != m_cbp1);
                    else if (cld == 0u && !m_clut_explicit_cld) {
                        const u32 cpsm = static_cast<u32>((val >> 51) & 0xFu);
                        const u32 csm  = static_cast<u32>((val >> 55) & 1u);
                        const u32 csa  = static_cast<u32>((val >> 56) & 0x1Fu);
                        const bool is8 = is_8bit_paletted_psm(psm);
                        will_load_clut = !m_clut_fallback_valid || m_clut_fb_cbp != cbp ||
                                         m_clut_fb_cpsm != cpsm || m_clut_fb_csm != csm ||
                                         m_clut_fb_csa != csa || m_clut_fb_is8 != is8;
                    }
                }
                const u32 active_c = (active_prim_attrs() >> 9) & 1u;
                constexpr u64 kTex0NonCldMask = ~(7ULL << 61);
                if (will_load_clut || (active_c == reg_ctx && ((val & kTex0NonCldMask) != (m_ctx[reg_ctx].tex0_reg & kTex0NonCldMask)))) {
                    flush_tri_batch();
                }
            }
            m_ctx[reg_ctx].tex0_reg = val;
            handle_tex0_clut_load(val);
            break;
        }
        case 0x08: // CLAMP_1
            if (!m_tri_batch.empty() && (((active_prim_attrs() >> 9) & 1u) == 0u) && val != m_ctx[0].clamp_reg) flush_tri_batch();
            m_ctx[0].clamp_reg = val;
            break;
        case 0x09: // CLAMP_2
            if (!m_tri_batch.empty() && (((active_prim_attrs() >> 9) & 1u) == 1u) && val != m_ctx[1].clamp_reg) flush_tri_batch();
            m_ctx[1].clamp_reg = val;
            break;
        case 0x14: // TEX1_1
            if (!m_tri_batch.empty() && (((active_prim_attrs() >> 9) & 1u) == 0u) && val != m_ctx[0].tex1_reg) flush_tri_batch();
            m_ctx[0].tex1_reg = val;
            break;
        case 0x15: // TEX1_2
            if (!m_tri_batch.empty() && (((active_prim_attrs() >> 9) & 1u) == 1u) && val != m_ctx[1].tex1_reg) flush_tri_batch();
            m_ctx[1].tex1_reg = val;
            break;
        case 0x18: // XYOFFSET_1
            m_ctx[0].xyoffset_x = static_cast<s32>(val & 0xFFFFu);
            m_ctx[0].xyoffset_y = static_cast<s32>((val >> 32) & 0xFFFFu);
            break;
        case 0x19: // XYOFFSET_2
            m_ctx[1].xyoffset_x = static_cast<s32>(val & 0xFFFFu);
            m_ctx[1].xyoffset_y = static_cast<s32>((val >> 32) & 0xFFFFu);
            break;
        case 0x1A: { // PRMODECONT
            const bool nv = (val & 1u) != 0;
            if (!m_tri_batch.empty() && nv != m_prmodecont) flush_tri_batch();
            m_prmodecont = nv;
            break;
        }
        case 0x1B: { // PRMODE
            const u32 nv = static_cast<u32>(val & 0x7FFu);
            if (!m_tri_batch.empty() && !m_prmodecont && nv != m_prmode_reg) flush_tri_batch();
            m_prmode_reg = nv;
            break;
        }
        case 0x1C: // TEXCLUT (CSM2 CBW/COU/COV)
            if (!m_tri_batch.empty()) flush_tri_batch();
            m_texclut_cbw = static_cast<u32>(val & 0x3Fu);
            m_texclut_cou = static_cast<u32>((val >> 6) & 0x3Fu);
            m_texclut_cov = static_cast<u32>((val >> 12) & 0x3FFu);
            break;
        case 0x22: { // SCANMSK
            const u32 nv = static_cast<u32>(val & 3u);
            if (!m_tri_batch.empty() && nv != m_scanmsk) flush_tri_batch();
            m_scanmsk = nv;
            break;
        }
        case 0x3B: // TEXA
            if (!m_tri_batch.empty()) flush_tri_batch();
            m_texa_ta0 = static_cast<u8>(val & 0xFFu);
            m_texa_aem = ((val >> 15) & 1u) != 0;
            m_texa_ta1 = static_cast<u8>((val >> 32) & 0xFFu);
            refresh_clut_rgba_from_raw();
            m_clut_fallback_valid = false;
            break;
        case 0x40: { // SCISSOR_1
            const s32 nx0 = static_cast<s32>(val & 0x7FFu);
            const s32 nx1 = static_cast<s32>((val >> 16) & 0x7FFu);
            const s32 ny0 = static_cast<s32>((val >> 32) & 0x7FFu);
            const s32 ny1 = static_cast<s32>((val >> 48) & 0x7FFu);
            if (!m_tri_batch.empty() && (((active_prim_attrs() >> 9) & 1u) == 0u) &&
                (nx0 != m_ctx[0].scissor_x0 || nx1 != m_ctx[0].scissor_x1 ||
                 ny0 != m_ctx[0].scissor_y0 || ny1 != m_ctx[0].scissor_y1)) {
                flush_tri_batch();
            }
            m_ctx[0].scissor_x0 = nx0; m_ctx[0].scissor_x1 = nx1;
            m_ctx[0].scissor_y0 = ny0; m_ctx[0].scissor_y1 = ny1;
            break;
        }
        case 0x41: { // SCISSOR_2
            const s32 nx0 = static_cast<s32>(val & 0x7FFu);
            const s32 nx1 = static_cast<s32>((val >> 16) & 0x7FFu);
            const s32 ny0 = static_cast<s32>((val >> 32) & 0x7FFu);
            const s32 ny1 = static_cast<s32>((val >> 48) & 0x7FFu);
            if (!m_tri_batch.empty() && (((active_prim_attrs() >> 9) & 1u) == 1u) &&
                (nx0 != m_ctx[1].scissor_x0 || nx1 != m_ctx[1].scissor_x1 ||
                 ny0 != m_ctx[1].scissor_y0 || ny1 != m_ctx[1].scissor_y1)) {
                flush_tri_batch();
            }
            m_ctx[1].scissor_x0 = nx0; m_ctx[1].scissor_x1 = nx1;
            m_ctx[1].scissor_y0 = ny0; m_ctx[1].scissor_y1 = ny1;
            break;
        }
        case 0x42: // ALPHA_1
            if (!m_tri_batch.empty() && (((active_prim_attrs() >> 9) & 1u) == 0u) && val != m_ctx[0].alpha_reg) flush_tri_batch();
            m_ctx[0].alpha_reg = val;
            break;
        case 0x43: // ALPHA_2
            if (!m_tri_batch.empty() && (((active_prim_attrs() >> 9) & 1u) == 1u) && val != m_ctx[1].alpha_reg) flush_tri_batch();
            m_ctx[1].alpha_reg = val;
            break;
        case 0x47: // TEST_1
            if (!m_tri_batch.empty() && (((active_prim_attrs() >> 9) & 1u) == 0u) && val != m_ctx[0].test_reg) flush_tri_batch();
            m_ctx[0].test_reg = val;
            break;
        case 0x48: // TEST_2
            if (!m_tri_batch.empty() && (((active_prim_attrs() >> 9) & 1u) == 1u) && val != m_ctx[1].test_reg) flush_tri_batch();
            m_ctx[1].test_reg = val;
            break;
        case 0x49: { // PABE
            const bool nv = (val & 1u) != 0;
            if (!m_tri_batch.empty() && nv != m_pabe) flush_tri_batch();
            m_pabe = nv;
            break;
        }
        case 0x4A: { // FBA_1
            const bool nv = (val & 1u) != 0;
            if (!m_tri_batch.empty() && (((active_prim_attrs() >> 9) & 1u) == 0u) && nv != m_ctx[0].fba) flush_tri_batch();
            m_ctx[0].fba = nv;
            break;
        }
        case 0x4B: { // FBA_2
            const bool nv = (val & 1u) != 0;
            if (!m_tri_batch.empty() && (((active_prim_attrs() >> 9) & 1u) == 1u) && nv != m_ctx[1].fba) flush_tri_batch();
            m_ctx[1].fba = nv;
            break;
        }
        case 0x4C: // FRAME_1
            if (!m_tri_batch.empty() && (((active_prim_attrs() >> 9) & 1u) == 0u) && val != m_ctx[0].frame_reg) flush_tri_batch();
            m_ctx[0].frame_reg = val;
            m_ctx[0].active_draw_fbp = static_cast<u32>(val & 0x1FFu);
            invalidate_draw_cache();
            break;
        case 0x4D: // FRAME_2
            if (!m_tri_batch.empty() && (((active_prim_attrs() >> 9) & 1u) == 1u) && val != m_ctx[1].frame_reg) flush_tri_batch();
            m_ctx[1].frame_reg = val;
            m_ctx[1].active_draw_fbp = static_cast<u32>(val & 0x1FFu);
            invalidate_draw_cache();
            break;
        case 0x4E: // ZBUF_1
            if (!m_tri_batch.empty() && (((active_prim_attrs() >> 9) & 1u) == 0u) && val != m_ctx[0].zbuf_reg) flush_tri_batch();
            m_ctx[0].zbuf_reg = val;
            invalidate_draw_cache();
            break;
        case 0x4F: // ZBUF_2
            if (!m_tri_batch.empty() && (((active_prim_attrs() >> 9) & 1u) == 1u) && val != m_ctx[1].zbuf_reg) flush_tri_batch();
            m_ctx[1].zbuf_reg = val;
            invalidate_draw_cache();
            break;
        case 0x50: // BITBLTBUF
            m_bitblt_sbp  = static_cast<u32>(val & 0x3FFFu);
            m_bitblt_sbw  = static_cast<u32>((val >> 16) & 0x3Fu);
            m_bitblt_spsm = static_cast<u32>((val >> 24) & 0x3Fu);
            m_bitblt_dbp  = static_cast<u32>((val >> 32) & 0x3FFFu);
            m_bitblt_dbw  = static_cast<u32>((val >> 48) & 0x3Fu);
            m_bitblt_dpsm = static_cast<u32>((val >> 56) & 0x3Fu);
            break;
        case 0x51: // TRXPOS
            m_trx_ssax = static_cast<u32>(val & 0x7FFu);
            m_trx_ssay = static_cast<u32>((val >> 16) & 0x7FFu);
            m_trx_dsax = static_cast<u32>((val >> 32) & 0x7FFu);
            m_trx_dsay = static_cast<u32>((val >> 48) & 0x7FFu);
            m_trx_dir  = static_cast<u32>((val >> 59) & 0x3u);
            break;
        case 0x52: // TRXREG
            m_trx_rrw = static_cast<u32>(val & 0xFFFu);
            m_trx_rrh = static_cast<u32>((val >> 32) & 0xFFFu);
            break;
        case 0x53: { // TRXDIR
            if (!m_tri_batch.empty()) flush_tri_batch();
            PerfScope perf_vram(PerfCat::GsVram);
            if ((val & 3u) == 0u && m_trx_rrw > 0u && m_trx_rrh > 0u) {
                m_trx_x = 0;
                m_trx_y = 0;
                m_trx_active = true;
                m_rgb24_cnt = 0;
                m_clut_fallback_valid = false;
                const u32 span_blocks = vram_block_span(m_bitblt_dbw, m_bitblt_dpsm,
                                                        m_trx_dsax + m_trx_rrw, m_trx_dsay + m_trx_rrh);
                invalidate_overlapping_fbs(m_bitblt_dbp, span_blocks);
                if (m_hw) m_hw->on_vram_write(m_bitblt_dbp, span_blocks);
                bump_vram_pages(m_bitblt_dbp, span_blocks);
                const bool is_color_fb_upload =
                    (m_bitblt_dpsm == 0x00u || m_bitblt_dpsm == 0x01u || m_bitblt_dpsm == 0x02u || m_bitblt_dpsm == 0x0Au) &&
                    m_trx_rrw >= 32u && m_trx_rrh >= 32u && m_bitblt_dbw > 0u;
                mark_vram_pages_color_upload(m_bitblt_dbp, span_blocks,
                                             is_color_fb_upload ? static_cast<u8>(m_bitblt_dbw) : u8{0});
                if (ranges_overlap_vram(m_bitblt_dbp, span_blocks, m_last_presented_fbp * 32u, 56u * 32u)) {
                    m_drew_to_disp_fbp = true;
                }
                hwreg_uploads++;
                if (debug_log.size() < 24) {
                    debug_log.push_back("UPLOAD dbp=" + std::to_string(m_bitblt_dbp) +
                                        " dbw=" + std::to_string(m_bitblt_dbw) +
                                        " psm=" + std::to_string(m_bitblt_dpsm) +
                                        " rect=" + std::to_string(m_trx_rrw) + "x" + std::to_string(m_trx_rrh));
                }
            } else if ((val & 3u) == 2u && m_trx_rrw > 0u && m_trx_rrh > 0u) {
                m_clut_fallback_valid = false;
                exec_local_to_local_trx();
                m_trx_active = false;
            } else {
                m_trx_active = false;
            }
            break;
        }
        case 0x54: // HWREG
            if (m_trx_active) {
                handle_hwreg_write(val);
            }
            break;
        default:
            break;
    }
}

void GsState::mark_fb_dirty(u32 fbp, u32 y0, u32 y1_excl) {
    if (y0 >= y1_excl) return;
    fbp &= 0x1FFu;
    GsFbTarget& rt = m_fb_targets[fbp];
    const bool was_dirty = (rt.dirty_min_y <= rt.dirty_max_y);
    if (y0 < rt.dirty_min_y) rt.dirty_min_y = y0;
    if ((y1_excl - 1u) > rt.dirty_max_y) rt.dirty_max_y = y1_excl - 1u;
    if (!was_dirty && rt.dirty_min_y <= rt.dirty_max_y) {
        m_dirty_fbps.push_back(fbp);
    }
    if (fbp == m_last_presented_fbp) {
        m_drew_to_disp_fbp = true;
    }
}

void GsState::flush_fb_target(u32 fbp) {
    fbp &= 0x1FFu;
    GsFbTarget& rt = m_fb_targets[fbp];
    if (!rt.valid || rt.dirty_min_y > rt.dirty_max_y || rt.fbw == 0u) {
        rt.dirty_min_y = 0xFFFFFFFFu;
        rt.dirty_max_y = 0;
        for (std::size_t i = 0; i < m_dirty_fbps.size(); ++i) {
            if (m_dirty_fbps[i] == fbp) {
                m_dirty_fbps[i] = m_dirty_fbps.back();
                m_dirty_fbps.pop_back();
                break;
            }
        }
        return;
    }

    const u32 stride = rt.fbw * 64u;
    PerfScope perf_vram(PerfCat::GsVram);
    if (g_perf.enabled) g_perf.fb_flushes.fetch_add(1, std::memory_order_relaxed);
    const u32 y_start = rt.dirty_min_y;
    const u32 y_end = std::min<u32>(rt.dirty_max_y + 1u, rt.height);
    const u32 bp = fbp * 32u;
    u8* vram = m_vram.data();

    // Block-hoisted flush into 4 MiB swizzled VRAM
    if (rt.psm == 0x00u || rt.psm == 0x30u) {
        const u32 first_py = y_start >> 5u;
        const u32 last_py = (y_end - 1u) >> 5u;
        for (u32 py = first_py; py <= last_py; ++py) {
            const u32 y_lo = std::max<u32>(y_start, py << 5u);
            const u32 y_hi = std::min<u32>(y_end, (py + 1u) << 5u);
            for (u32 px = 0; px < rt.fbw; ++px) {
                const u32 page_blk = bp + (py * rt.fbw + px) * 32u;
                for (u32 y = y_lo; y < y_hi; ++y) {
                    const u32 by = (y >> 3u) & 3u;
                    const u32 word_y = ((y >> 1u) & 3u) * 16u;
                    const u8* col_row = kCol32[y & 1u];
                    const u32* srow = rt.pixels.data() + static_cast<std::size_t>(y) * stride + (px << 6u);
                    for (u32 bx = 0; bx < 8u; ++bx) {
                        const u32 blk_byte = (((page_blk + kBlock32[by][bx]) & 0x3FFFu) * 64u + word_y) * 4u;
                        const u32* sp = srow + (bx << 3u);
                        for (u32 cx = 0; cx < 8u; ++cx) {
                            write_le32(vram + ((blk_byte + col_row[cx] * 4u) & (kGsVramSize - 4u)), sp[cx]);
                        }
                    }
                }
            }
        }
    } else if (rt.psm == 0x01u || rt.psm == 0x31u) {
        const u32 first_py = y_start >> 5u;
        const u32 last_py = (y_end - 1u) >> 5u;
        for (u32 py = first_py; py <= last_py; ++py) {
            const u32 y_lo = std::max<u32>(y_start, py << 5u);
            const u32 y_hi = std::min<u32>(y_end, (py + 1u) << 5u);
            for (u32 px = 0; px < rt.fbw; ++px) {
                const u32 page_blk = bp + (py * rt.fbw + px) * 32u;
                for (u32 y = y_lo; y < y_hi; ++y) {
                    const u32 by = (y >> 3u) & 3u;
                    const u32 word_y = ((y >> 1u) & 3u) * 16u;
                    const u8* col_row = kCol32[y & 1u];
                    const u32* srow = rt.pixels.data() + static_cast<std::size_t>(y) * stride + (px << 6u);
                    for (u32 bx = 0; bx < 8u; ++bx) {
                        const u32 blk_byte = (((page_blk + kBlock32[by][bx]) & 0x3FFFu) * 64u + word_y) * 4u;
                        const u32* sp = srow + (bx << 3u);
                        for (u32 cx = 0; cx < 8u; ++cx) {
                            const u32 addr = (blk_byte + col_row[cx] * 4u) & (kGsVramSize - 4u);
                            const u32 cur = read_le32(vram + addr);
                            write_le32(vram + addr, (cur & 0xFF000000u) | (sp[cx] & 0x00FFFFFFu));
                        }
                    }
                }
            }
        }
    } else if (rt.psm == 0x02u || rt.psm == 0x32u || rt.psm == 0x0Au || rt.psm == 0x3Au) {
        const bool is_16s = (rt.psm == 0x0Au || rt.psm == 0x3Au);
        const u32 first_py = y_start >> 6u;
        const u32 last_py = (y_end - 1u) >> 6u;
        for (u32 py = first_py; py <= last_py; ++py) {
            const u32 y_lo = std::max<u32>(y_start, py << 6u);
            const u32 y_hi = std::min<u32>(y_end, (py + 1u) << 6u);
            for (u32 px = 0; px < rt.fbw; ++px) {
                const u32 page_blk = bp + (py * rt.fbw + px) * 32u;
                for (u32 y = y_lo; y < y_hi; ++y) {
                    const u32 by = (y >> 3u) & 7u;
                    const u32 hw_y = ((y >> 1u) & 3u) * 32u;
                    const u8* col_row = kCol32[y & 1u];
                    const u32* srow = rt.pixels.data() + static_cast<std::size_t>(y) * stride + (px << 6u);
                    for (u32 bx = 0; bx < 4u; ++bx) {
                        const u32 b_idx = is_16s ? kBlock16S[by][bx] : kBlock16[by][bx];
                        const u32 blk_byte = (((page_blk + b_idx) & 0x3FFFu) * 128u + hw_y) * 2u;
                        const u32* sp = srow + (bx << 4u);
                        for (u32 cx = 0; cx < 8u; ++cx) {
                            write_le16(vram + ((blk_byte + col_row[cx] * 2u) & (kGsVramSize - 2u)), pack_rgba_to_psm16(sp[cx]));
                            write_le16(vram + ((blk_byte + (16u + col_row[cx]) * 2u) & (kGsVramSize - 2u)), pack_rgba_to_psm16(sp[cx + 8u]));
                        }
                    }
                }
            }
        }
    }

    const u32 ph = psm_page_height(rt.psm);
    const u32 first_py = y_start / ph;
    const u32 last_py = (y_end - 1u) / ph;
    const u32 start_blk = bp + first_py * rt.fbw * 32u;
    const u32 num_blks = (last_py - first_py + 1u) * rt.fbw * 32u;
    bump_vram_pages(start_blk, num_blks);

    rt.dirty_min_y = 0xFFFFFFFFu;
    rt.dirty_max_y = 0;
    for (std::size_t i = 0; i < m_dirty_fbps.size(); ++i) {
        if (m_dirty_fbps[i] == fbp) {
            m_dirty_fbps[i] = m_dirty_fbps.back();
            m_dirty_fbps.pop_back();
            break;
        }
    }
}

void GsState::flush_overlapping_fbs(u32 start_block, u32 num_blocks, u32 exclude_fbp) {
    if (m_dirty_fbps.empty()) return;
    std::size_t i = 0;
    while (i < m_dirty_fbps.size()) {
        const u32 fbp = m_dirty_fbps[i];
        if (fbp == exclude_fbp) {
            ++i;
            continue;
        }
        const GsFbTarget& rt = m_fb_targets[fbp];
        if (!rt.valid || rt.dirty_min_y > rt.dirty_max_y || rt.fbw == 0u) {
            m_dirty_fbps[i] = m_dirty_fbps.back();
            m_dirty_fbps.pop_back();
            continue;
        }
        const u32 ph = psm_page_height(rt.psm);
        const u32 first_py = rt.dirty_min_y / ph;
        const u32 last_py  = rt.dirty_max_y / ph;
        const u32 rt_start = fbp * 32u + first_py * rt.fbw * 32u;
        const u32 rt_len   = (last_py - first_py + 1u) * rt.fbw * 32u;
        if (ranges_overlap_vram(start_block, num_blocks, rt_start, rt_len)) {
            flush_fb_target(fbp); // removes fbp from m_dirty_fbps
        } else {
            ++i;
        }
    }
}

void GsState::invalidate_overlapping_fbs(u32 start_block, u32 num_blocks) {
    flush_overlapping_fbs(start_block, num_blocks);
    std::size_t i = 0;
    while (i < m_valid_fbps.size()) {
        const u32 fbp = m_valid_fbps[i];
        GsFbTarget& rt = m_fb_targets[fbp];
        if (!rt.valid) {
            m_valid_fbps[i] = m_valid_fbps.back();
            m_valid_fbps.pop_back();
            continue;
        }
        const u32 ph = psm_page_height(rt.psm);
        const u32 rt_pages_y = std::max<u32>(1u, (rt.height + ph - 1u) / ph);
        const u32 rt_start = fbp * 32u;
        const u32 rt_len   = rt_pages_y * std::max<u32>(1u, rt.fbw) * 32u;
        if (ranges_overlap_vram(start_block, num_blocks, rt_start, rt_len)) {
            rt.valid = false;
            rt.dirty_min_y = 0xFFFFFFFFu;
            rt.dirty_max_y = 0;
            m_valid_fbps[i] = m_valid_fbps.back();
            m_valid_fbps.pop_back();
            if (m_cached_rt_fbp == fbp) invalidate_draw_cache();
        } else {
            ++i;
        }
    }
}

GsState::GsFbTarget* GsState::ensure_fb_target(u32 fbp, u32 fbw, u32 psm, u32 min_height, bool will_overwrite_all) {
    fbp &= 0x1FFu;
    const u32 eff_fbw = (fbw == 0u) ? ((m_fb_width + 63u) / 64u) : fbw;
    const u32 eff_psm = (psm == 0x00u || psm == 0x01u || psm == 0x02u || psm == 0x0Au) ? psm : 0x00u;
    const u32 ph = psm_page_height(eff_psm);
    // Size height to actual min_height rounded up to page height (prevents false 448-line overlap on 224-line RTs!)
    const u32 req_h = std::clamp<u32>(((std::max<u32>(ph, min_height) + ph - 1u) / ph) * ph, ph, 1024u);

    // Ultra-fast consecutive-primitive cache hit
    if (m_cached_rt != nullptr && m_cached_rt_fbp == fbp && m_cached_rt_fbw == eff_fbw &&
        m_cached_rt_psm == eff_psm && m_cached_rt->valid && !m_cached_rt->pixels.empty() &&
        m_cached_rt->height >= req_h) {
        return m_cached_rt;
    }
    PerfScope perf_vram(PerfCat::GsVram);

    GsFbTarget& rt = m_fb_targets[fbp];
    const u32 eff_h = (rt.valid && rt.fbw == eff_fbw && rt.psm == eff_psm && rt.height > req_h) ? rt.height : req_h;
    const u32 rt_blocks = (eff_h / ph) * eff_fbw * 32u;

    if (rt.valid && !rt.pixels.empty() && rt.fbw == eff_fbw && rt.psm == eff_psm && rt.height >= req_h) {
        if (m_dirty_fbps.size() > 1u || (!m_dirty_fbps.empty() && m_dirty_fbps[0] != fbp)) {
            bool other_overlaps = false;
            for (u32 dfbp : m_dirty_fbps) {
                if (dfbp == fbp) continue;
                const GsFbTarget& drt = m_fb_targets[dfbp];
                if (!drt.valid || drt.dirty_min_y > drt.dirty_max_y || drt.fbw == 0u) continue;
                const u32 dph = psm_page_height(drt.psm);
                const u32 dstart = dfbp * 32u + (drt.dirty_min_y / dph) * drt.fbw * 32u;
                const u32 dlen = ((drt.dirty_max_y / dph) - (drt.dirty_min_y / dph) + 1u) * drt.fbw * 32u;
                if (ranges_overlap_vram(fbp * 32u, rt_blocks, dstart, dlen)) {
                    other_overlaps = true;
                    break;
                }
            }
            if (!other_overlaps) {
                m_cached_rt = &rt;
                m_cached_rt_fbp = fbp;
                m_cached_rt_fbw = eff_fbw;
                m_cached_rt_psm = eff_psm;
                m_cached_rt_min_h = req_h;
                return &rt;
            }
            if (rt.dirty_min_y <= rt.dirty_max_y) {
                flush_fb_target(fbp);
            }
            flush_overlapping_fbs(fbp * 32u, rt_blocks, fbp);
            rt.valid = false;
        } else {
            m_cached_rt = &rt;
            m_cached_rt_fbp = fbp;
            m_cached_rt_fbw = eff_fbw;
            m_cached_rt_psm = eff_psm;
            m_cached_rt_min_h = req_h;
            return &rt;
        }
    }

    if (rt.valid && rt.dirty_min_y <= rt.dirty_max_y) {
        flush_fb_target(fbp);
    }
    flush_overlapping_fbs(fbp * 32u, rt_blocks, fbp);

    const u32 stride = eff_fbw * 64u;
    rt.fbw = eff_fbw;
    rt.height = req_h;
    rt.psm = eff_psm;
    rt.pixels.resize(static_cast<std::size_t>(stride) * req_h);

    if (!will_overwrite_all) {
        if (g_perf.enabled) g_perf.fb_reloads.fetch_add(1, std::memory_order_relaxed);
        // Page-tiled reload from VRAM; untouched zero pages use fast std::fill_n!
        const u32 bp = fbp * 32u;
        const u8* vram = m_vram.data();
        const u32 zero_px = (eff_psm == 0x00u) ? 0u
                          : (eff_psm == 0x01u) ? (m_texa_aem ? 0u : (static_cast<u32>(m_texa_ta0) << 24))
                          : unpack_psm16_texa(0u);
        const u32 pages_y = req_h / ph;
        for (u32 py = 0; py < pages_y; ++py) {
            for (u32 px = 0; px < eff_fbw; ++px) {
                const u32 page_idx = (fbp + py * eff_fbw + px) & 511u;
                const u32 y0 = py * ph;
                const u32 x0 = px << 6u;
                if (m_vram_page_gen[page_idx] == 0u) {
                    for (u32 ly = 0; ly < ph; ++ly) {
                        std::fill_n(rt.pixels.data() + static_cast<std::size_t>(y0 + ly) * stride + x0, 64u, zero_px);
                    }
                    continue;
                }
                const u32 page_blk = bp + (py * eff_fbw + px) * 32u;
                if (eff_psm == 0x00u) {
                    for (u32 ly = 0; ly < 32u; ++ly) {
                        const u32 by = (ly >> 3u) & 3u;
                        const u32 word_y = ((ly >> 1u) & 3u) * 16u;
                        const u8* col_row = kCol32[ly & 1u];
                        u32* drow = rt.pixels.data() + static_cast<std::size_t>(y0 + ly) * stride + x0;
                        for (u32 bx = 0; bx < 8u; ++bx) {
                            const u32 blk_byte = (((page_blk + kBlock32[by][bx]) & 0x3FFFu) * 64u + word_y) * 4u;
                            u32* dp = drow + (bx << 3u);
                            for (u32 cx = 0; cx < 8u; ++cx) {
                                dp[cx] = read_le32(vram + ((blk_byte + col_row[cx] * 4u) & (kGsVramSize - 4u)));
                            }
                        }
                    }
                } else if (eff_psm == 0x01u) {
                    const u32 ta0_sh = static_cast<u32>(m_texa_ta0) << 24;
                    for (u32 ly = 0; ly < 32u; ++ly) {
                        const u32 by = (ly >> 3u) & 3u;
                        const u32 word_y = ((ly >> 1u) & 3u) * 16u;
                        const u8* col_row = kCol32[ly & 1u];
                        u32* drow = rt.pixels.data() + static_cast<std::size_t>(y0 + ly) * stride + x0;
                        for (u32 bx = 0; bx < 8u; ++bx) {
                            const u32 blk_byte = (((page_blk + kBlock32[by][bx]) & 0x3FFFu) * 64u + word_y) * 4u;
                            u32* dp = drow + (bx << 3u);
                            for (u32 cx = 0; cx < 8u; ++cx) {
                                const u32 rgb = read_le32(vram + ((blk_byte + col_row[cx] * 4u) & (kGsVramSize - 4u))) & 0x00FFFFFFu;
                                dp[cx] = rgb | ((m_texa_aem && rgb == 0u) ? 0u : ta0_sh);
                            }
                        }
                    }
                } else {
                    const bool is_16s = (eff_psm == 0x0Au);
                    for (u32 ly = 0; ly < 64u; ++ly) {
                        const u32 by = (ly >> 3u) & 7u;
                        const u32 hw_y = ((ly >> 1u) & 3u) * 32u;
                        const u8* col_row = kCol32[ly & 1u];
                        u32* drow = rt.pixels.data() + static_cast<std::size_t>(y0 + ly) * stride + x0;
                        for (u32 bx = 0; bx < 4u; ++bx) {
                            const u32 b_idx = is_16s ? kBlock16S[by][bx] : kBlock16[by][bx];
                            const u32 blk_byte = (((page_blk + b_idx) & 0x3FFFu) * 128u + hw_y) * 2u;
                            u32* dp = drow + (bx << 4u);
                            for (u32 cx = 0; cx < 8u; ++cx) {
                                dp[cx]      = unpack_psm16_texa(read_le16(vram + ((blk_byte + col_row[cx] * 2u) & (kGsVramSize - 2u))));
                                dp[cx + 8u] = unpack_psm16_texa(read_le16(vram + ((blk_byte + (16u + col_row[cx]) * 2u) & (kGsVramSize - 2u))));
                            }
                        }
                    }
                }
            }
        }
    }

    if (std::find(m_valid_fbps.begin(), m_valid_fbps.end(), fbp) == m_valid_fbps.end()) {
        m_valid_fbps.push_back(fbp);
    }
    rt.valid = true;
    rt.dirty_min_y = 0xFFFFFFFFu;
    rt.dirty_max_y = 0;

    m_cached_rt = &rt;
    m_cached_rt_fbp = fbp;
    m_cached_rt_fbw = eff_fbw;
    m_cached_rt_psm = eff_psm;
    m_cached_rt_min_h = req_h;
    return &rt;
}

GsState::GsZTarget* GsState::ensure_z_target(u32 zbp, u32 stride, u32 height) {
    zbp &= 0x1FFu;
    if (m_cached_zt != nullptr && m_cached_zt_zbp == zbp &&
        m_cached_zt->stride == stride && m_cached_zt->height >= height) {
        return m_cached_zt;
    }
    GsZTarget& zt = m_z_targets[zbp];
    const std::size_t need = static_cast<std::size_t>(stride) * height;
    if (zt.stride != stride || zt.height < height || zt.z.size() < need) {
        zt.stride = stride;
        zt.height = height;
        zt.z.assign(need, 0u);
    }
    m_cached_zt = &zt;
    m_cached_zt_zbp = zbp;
    return &zt;
}

const GsState::GsFbTarget* GsState::prepare_texture_source(u64 tex0) {
    const u32 tbp0 = static_cast<u32>(tex0 & 0x3FFFu);
    const u32 tbw  = static_cast<u32>((tex0 >> 14) & 0x3Fu);
    const u32 psm  = static_cast<u32>((tex0 >> 20) & 0x3Fu);
    const u32 tw   = 1u << std::min<u32>(10u, static_cast<u32>((tex0 >> 26) & 0xFu));
    const u32 th   = 1u << std::min<u32>(10u, static_cast<u32>((tex0 >> 30) & 0xFu));

    // Fast linear render-target path: exact page-aligned FBP match with compatible width & RGB/H-alpha format
    const bool is_rt_rgb = (psm == 0x00u || psm == 0x01u || psm == 0x02u || psm == 0x0Au);
    const bool is_rt_clut_h = (m_hw != nullptr) && (psm == 0x1Bu || psm == 0x24u || psm == 0x2Cu);
    if ((tbp0 & 31u) == 0u && (is_rt_rgb || is_rt_clut_h)) {
        const u32 fbp = (tbp0 >> 5u) & 0x1FFu;
        const GsFbTarget& rt = m_fb_targets[fbp];
        const u32 eff_tbw = (tbw == 0u) ? rt.fbw : tbw;
        const bool psm_compat = is_rt_clut_h ? (rt.psm <= 1u)
                                             : ((rt.psm == psm) || (rt.psm <= 1u && psm <= 1u));
        if (rt.valid && rt.fbw > 0u && rt.fbw == eff_tbw && psm_compat) {
            if (m_hw != nullptr && m_hw->is_rt_resident(fbp)) {
                return &rt;
            }
            if (!is_rt_clut_h && !rt.pixels.empty()) {
                if (m_dirty_fbps.size() > 1u || (!m_dirty_fbps.empty() && m_dirty_fbps[0] != fbp)) {
                    const u32 span_blocks = vram_block_span(eff_tbw, psm, tw, th);
                    const std::size_t before = m_dirty_fbps.size();
                    flush_overlapping_fbs(tbp0, span_blocks, fbp);
                    if (m_dirty_fbps.size() != before) {
                        flush_fb_target(fbp);
                        return nullptr;
                    }
                }
                return &rt;
            }
        }
    }

    // Flush any dirty RT overlapping sampled VRAM blocks before texture cache / VRAM read
    if (!m_dirty_fbps.empty()) {
        const u32 span_blocks = vram_block_span(tbw, psm, tw, th);
        flush_overlapping_fbs(tbp0, span_blocks);
    }
    return nullptr;
}

GsState::LinearTexView GsState::resolve_linear_texture(u64 tex0, const GsFbTarget* rt_src) {
    PerfScope perf_tex(PerfCat::GsTex);
    const u32 psm = static_cast<u32>((tex0 >> 20) & 0x3Fu);
    if (rt_src != nullptr) {
        if (g_perf.enabled) g_perf.tex_rt_hits.fetch_add(1, std::memory_order_relaxed);
        const u32 rt_fbp = static_cast<u32>(rt_src - m_fb_targets.data());
        const u32 clut_mode = (psm == 0x1Bu) ? 1u : (psm == 0x24u) ? 2u : (psm == 0x2Cu) ? 3u : 0u;
        return LinearTexView{
            rt_src->pixels.empty() ? nullptr : rt_src->pixels.data(),
            rt_src->fbw * 64u,
            rt_src->height,
            (psm == 0x01u),
            true,
            static_cast<u32>(kTexCacheSlots),
            0u,
            rt_fbp,
            rt_src->fbw,
            clut_mode,
            false,
            0u
        };
    }

    const u32 tbp0   = static_cast<u32>(tex0 & 0x3FFFu);
    // GPU-resident Z-buffer sampled directly as texture (PSMZ32/24/16/16S depth fog & post-process)
    if (m_hw != nullptr && (tbp0 & 31u) == 0u &&
        (psm == 0x30u || psm == 0x31u || psm == 0x32u || psm == 0x3Au)) {
        const u32 zbp = (tbp0 >> 5u) & 0x1FFu;
        if (m_hw->is_zb_resident(zbp)) {
            if (g_perf.enabled) g_perf.tex_rt_hits.fetch_add(1, std::memory_order_relaxed);
            return LinearTexView{
                nullptr, 0u, 0u, false, false,
                static_cast<u32>(kTexCacheSlots), 0u,
                0u, 0u, 0u, true, zbp
            };
        }
    }
    const u32 tbw    = static_cast<u32>((tex0 >> 14) & 0x3Fu);
    const u32 tw_log = std::min<u32>(10u, static_cast<u32>((tex0 >> 26) & 0xFu));
    const u32 th_log = std::min<u32>(10u, static_cast<u32>((tex0 >> 30) & 0xFu));
    const u32 tw     = 1u << tw_log;
    const u32 th     = 1u << th_log;
    if (tw * th > 1024u * 512u) {
        return LinearTexView{};
    }

    const bool is_pal = is_paletted_psm(psm);
    const u32 cpsm = is_pal ? static_cast<u32>((tex0 >> 51) & 0xFu) : 0u;
    const u32 csa  = is_pal ? static_cast<u32>((tex0 >> 56) & 0x1Fu) : 0u;
    const u64 cur_clut_hash = is_pal ? compute_active_clut_hash(psm, cpsm, csa) : 0u;

    const u64 key = static_cast<u64>(tbp0) |
                    (static_cast<u64>(tbw) << 14) |
                    (static_cast<u64>(psm) << 20) |
                    (static_cast<u64>(tw_log) << 26) |
                    (static_cast<u64>(th_log) << 30) |
                    (static_cast<u64>(cpsm) << 34) |
                    (static_cast<u64>(csa) << 38);

    const bool uses_texa = (psm == 0x01u || psm == 0x02u || psm == 0x0Au ||
                            (is_pal && (cpsm == 0x02u || cpsm == 0x0Au)));
    const u32 texa_sig = uses_texa
        ? (static_cast<u32>(m_texa_ta0) | (static_cast<u32>(m_texa_ta1) << 8) | (m_texa_aem ? 0x10000u : 0u))
        : 0u;

    // 4-way set-associative lookup across 512 cache slots (128 sets x 4 ways LRU)
    const std::size_t clut_mix = static_cast<std::size_t>(cur_clut_hash ^ (cur_clut_hash >> 32));
    const std::size_t set_base =
        (static_cast<std::size_t>(tbp0 ^ (psm << 7) ^ (csa << 2) ^ (tw_log << 11) ^ (th_log << 4) ^ (tbp0 >> 6) ^ clut_mix) &
         (kTexCacheSets - 1u)) * kTexCacheWays;

    auto matches = [&](const GsTexCacheEntry& e) -> bool {
        return e.valid && e.key == key && e.texa_sig == texa_sig && (!is_pal || e.clut_hash == cur_clut_hash);
    };

    GsTexCacheEntry* hit = nullptr;
    for (std::size_t w = 0; w < kTexCacheWays; ++w) {
        GsTexCacheEntry& cand = m_tex_cache[set_base + w];
        if (matches(cand)) {
            hit = &cand;
            break;
        }
    }

    if (hit != nullptr) {
        const u32 slot_idx = static_cast<u32>(hit - m_tex_cache.data());
        if (hit->last_vram_gen == m_vram_write_gen) {
            if (g_perf.enabled) g_perf.tex_cache_hits.fetch_add(1, std::memory_order_relaxed);
            hit->last_use = ++m_tex_cache_tick;
            return LinearTexView{ hit->rgba.data(), tw, th, false, false, slot_idx, hit->gpu_gen, 0u, 0u };
        }
        const u64 cur_hash = compute_page_gen_hash(hit->first_page, hit->num_pages);
        if (cur_hash == hit->page_gen_hash) {
            if (g_perf.enabled) g_perf.tex_cache_hits.fetch_add(1, std::memory_order_relaxed);
            hit->last_vram_gen = m_vram_write_gen;
            hit->last_use = ++m_tex_cache_tick;
            return LinearTexView{ hit->rgba.data(), tw, th, false, false, slot_idx, hit->gpu_gen, 0u, 0u };
        }
        if (g_perf.enabled) g_perf.tex_cache_redecodes.fetch_add(1, std::memory_order_relaxed);
        // VRAM pages for this texture changed -> re-decode in place
    }
    if (g_perf.enabled) g_perf.tex_cache_misses.fetch_add(1, std::memory_order_relaxed);

    GsTexCacheEntry* dst = hit;
    if (dst == nullptr) {
        dst = &m_tex_cache[set_base];
        for (std::size_t w = 0; w < kTexCacheWays; ++w) {
            GsTexCacheEntry& cand = m_tex_cache[set_base + w];
            if (!cand.valid) {
                dst = &cand;
                break;
            }
            if (cand.last_use < dst->last_use) {
                dst = &cand;
            }
        }
    }

    const u32 span_blocks = vram_block_span(tbw, psm, tw, th);
    const u32 first_page = (tbp0 >> 5u) & 511u;
    const u32 num_pages = std::min<u32>(512u, std::max<u32>(1u, (span_blocks + (tbp0 & 31u) + 31u) >> 5u));

    dst->rgba.resize(static_cast<std::size_t>(tw) * th);
    u32* out = dst->rgba.data();
    const u8* vram = m_vram.data();

    // Fast specialized unswizzle + CLUT decode into contiguous linear RGBA8888
    if (psm == 0x13u) { // PSMT8
        u32 pal[256];
        const u32 bank = (cpsm == 0u) ? 0u : ((csa & 16u) * 16u);
        const u32 off = (csa & 15u) * 16u;
        for (u32 i = 0; i < 256u; ++i) pal[i] = m_clut_rgba[bank + ((i + off) & 0xFFu)];
        const u32 bw_pages = std::max<u32>(1u, tbw >> 1u);
        for (u32 y = 0; y < th; ++y) {
            const u32 py_off = (y >> 6u) * bw_pages;
            const u32 by = (y >> 4u) & 3u;
            const u32 cy = y & 15u;
            const u32 swap = (((cy + 2u) >> 2u) & 1u) * 4u;
            const u32 py_c = (((cy & ~3u) >> 1u) + (cy & 1u)) & 7u;
            const u32 word_y = (py_c >> 1u) * 16u;
            const u8* col_row = kCol32[py_c & 1u];
            const u32 lane_y = (cy >> 1u) & 1u;
            u32* drow = out + static_cast<std::size_t>(y) * tw;
            for (u32 x = 0; x < tw; ++x) {
                const u32 blk = (tbp0 + (py_off + (x >> 7u)) * 32u + kBlock32[by][(x >> 4u) & 7u]) & 0x3FFFu;
                const u32 cx = x & 15u;
                const u32 word = word_y + col_row[(cx + swap) & 7u];
                const u32 lane = lane_y + ((cx >> 2u) & 2u);
                const u32 addr = ((blk * 64u + word) * 4u + lane) & (kGsVramSize - 1u);
                drow[x] = pal[vram[addr]];
            }
        }
    } else if (psm == 0x14u) { // PSMT4
        u32 pal[16];
        const u32 base = (cpsm == 0u) ? ((csa & 15u) * 16u) : (csa * 16u);
        for (u32 i = 0; i < 16u; ++i) pal[i] = m_clut_rgba[(base + i) & 0x1FFu];
        const u32 bw_pages = std::max<u32>(1u, tbw >> 1u);
        for (u32 y = 0; y < th; ++y) {
            const u32 py_off = (y >> 7u) * bw_pages;
            const u32 by = (y >> 4u) & 7u;
            const u32 cy = y & 15u;
            const u32 swap = (((cy + 2u) >> 2u) & 1u) * 4u;
            const u32 py_c = (((cy & ~3u) >> 1u) + (cy & 1u)) & 7u;
            const u32 word_y = (py_c >> 1u) * 16u;
            const u8* col_row = kCol32[py_c & 1u];
            const u32 shift = ((cy >> 1u) & 1u) * 4u;
            u32* drow = out + static_cast<std::size_t>(y) * tw;
            for (u32 x = 0; x < tw; ++x) {
                const u32 blk = (tbp0 + (py_off + (x >> 7u)) * 32u + kBlock16[by][(x >> 5u) & 3u]) & 0x3FFFu;
                const u32 cx = x & 31u;
                const u32 word = word_y + col_row[(cx ^ swap) & 7u];
                const u32 lane = (cx >> 3u) & 3u;
                const u32 addr = ((blk * 64u + word) * 4u + lane) & (kGsVramSize - 1u);
                drow[x] = pal[(vram[addr] >> shift) & 0xFu];
            }
        }
    } else if (psm == 0x00u) { // PSMCT32
        const u32 bw_pages = std::max<u32>(1u, tbw);
        for (u32 y = 0; y < th; ++y) {
            const u32 py_off = (y >> 5u) * bw_pages;
            const u32 by = (y >> 3u) & 3u;
            const u32 word_y = ((y >> 1u) & 3u) * 16u;
            const u8* col_row = kCol32[y & 1u];
            u32* drow = out + static_cast<std::size_t>(y) * tw;
            for (u32 x = 0; x < tw; ++x) {
                const u32 blk = (tbp0 + (py_off + (x >> 6u)) * 32u + kBlock32[by][(x >> 3u) & 7u]) & 0x3FFFu;
                const u32 addr = ((blk * 64u + word_y + col_row[x & 7u]) * 4u) & (kGsVramSize - 4u);
                drow[x] = read_le32(vram + addr);
            }
        }
    } else if (psm == 0x01u) { // PSMCT24
        const u32 bw_pages = std::max<u32>(1u, tbw);
        const u32 ta0_sh = static_cast<u32>(m_texa_ta0) << 24;
        for (u32 y = 0; y < th; ++y) {
            const u32 py_off = (y >> 5u) * bw_pages;
            const u32 by = (y >> 3u) & 3u;
            const u32 word_y = ((y >> 1u) & 3u) * 16u;
            const u8* col_row = kCol32[y & 1u];
            u32* drow = out + static_cast<std::size_t>(y) * tw;
            for (u32 x = 0; x < tw; ++x) {
                const u32 blk = (tbp0 + (py_off + (x >> 6u)) * 32u + kBlock32[by][(x >> 3u) & 7u]) & 0x3FFFu;
                const u32 addr = ((blk * 64u + word_y + col_row[x & 7u]) * 4u) & (kGsVramSize - 4u);
                const u32 rgb = read_le32(vram + addr) & 0x00FFFFFFu;
                drow[x] = rgb | ((m_texa_aem && rgb == 0u) ? 0u : ta0_sh);
            }
        }
    } else if (psm == 0x02u || psm == 0x0Au) { // PSMCT16 / PSMCT16S
        const bool is_16s = (psm == 0x0Au);
        const u32 bw_pages = std::max<u32>(1u, tbw);
        for (u32 y = 0; y < th; ++y) {
            const u32 py_off = (y >> 6u) * bw_pages;
            const u32 by = (y >> 3u) & 7u;
            const u32 hw_y = ((y >> 1u) & 3u) * 32u;
            const u8* col_row = kCol32[y & 1u];
            u32* drow = out + static_cast<std::size_t>(y) * tw;
            for (u32 x = 0; x < tw; ++x) {
                const u32 b_idx = is_16s ? kBlock16S[by][(x >> 4u) & 3u] : kBlock16[by][(x >> 4u) & 3u];
                const u32 blk = (tbp0 + (py_off + (x >> 6u)) * 32u + b_idx) & 0x3FFFu;
                const u32 hw = hw_y + ((x & 8u) << 1u) + col_row[x & 7u];
                const u32 addr = ((blk * 128u + hw) * 2u) & (kGsVramSize - 2u);
                drow[x] = unpack_psm16_texa(read_le16(vram + addr));
            }
        }
    } else if (psm == 0x1Bu) { // PSMT8H
        u32 pal[256];
        const u32 bank = (cpsm == 0u) ? 0u : ((csa & 16u) * 16u);
        const u32 off = (csa & 15u) * 16u;
        for (u32 i = 0; i < 256u; ++i) pal[i] = m_clut_rgba[bank + ((i + off) & 0xFFu)];
        for (u32 y = 0; y < th; ++y) {
            u32* drow = out + static_cast<std::size_t>(y) * tw;
            for (u32 x = 0; x < tw; ++x) {
                drow[x] = pal[vram[vram_addr32(tbp0, tbw, x, y) + 3u]];
            }
        }
    } else if (psm == 0x24u || psm == 0x2Cu) { // PSMT4HL / PSMT4HH
        u32 pal[16];
        const u32 base = (cpsm == 0u) ? ((csa & 15u) * 16u) : (csa * 16u);
        for (u32 i = 0; i < 16u; ++i) pal[i] = m_clut_rgba[(base + i) & 0x1FFu];
        const u32 sh = (psm == 0x2Cu) ? 4u : 0u;
        for (u32 y = 0; y < th; ++y) {
            u32* drow = out + static_cast<std::size_t>(y) * tw;
            for (u32 x = 0; x < tw; ++x) {
                drow[x] = pal[(vram[vram_addr32(tbp0, tbw, x, y) + 3u] >> sh) & 0xFu];
            }
        }
    } else {
        for (u32 y = 0; y < th; ++y) {
            u32* drow = out + static_cast<std::size_t>(y) * tw;
            for (u32 x = 0; x < tw; ++x) {
                drow[x] = read_vram_rgba(tbp0, tbw, psm, x, y);
            }
        }
    }

    dst->key = key;
    dst->texa_sig = texa_sig;
    dst->clut_hash = cur_clut_hash;
    dst->first_page = first_page;
    dst->num_pages = num_pages;
    dst->page_gen_hash = compute_page_gen_hash(first_page, num_pages);
    dst->last_vram_gen = m_vram_write_gen;
    dst->tw = tw;
    dst->th = th;
    dst->last_use = ++m_tex_cache_tick;
    dst->gpu_gen = ++m_hw_tex_gen;
    dst->valid = true;

    const u32 dst_slot = static_cast<u32>(dst - m_tex_cache.data());
    return LinearTexView{ dst->rgba.data(), tw, th, false, false, dst_slot, dst->gpu_gen, 0u, 0u };
}

void GsState::write_vram_pixel(u32 bp, u32 bw, u32 psm, u32 x, u32 y, u32 raw_val) {
    switch (psm) {
        case 0x00u:
        case 0x30u: {
            const u32 addr = vram_addr32(bp, bw, x, y);
            write_le32(m_vram.data() + addr, raw_val);
            break;
        }
        case 0x01u:
        case 0x31u: {
            const u32 addr = vram_addr32(bp, bw, x, y);
            const u32 cur = read_le32(m_vram.data() + addr);
            write_le32(m_vram.data() + addr, (cur & 0xFF000000u) | (raw_val & 0x00FFFFFFu));
            break;
        }
        case 0x02u:
        case 0x32u: {
            const u32 addr = vram_addr16(bp, bw, x, y, false);
            write_le16(m_vram.data() + addr, static_cast<u16>(raw_val & 0xFFFFu));
            break;
        }
        case 0x0Au:
        case 0x3Au: {
            const u32 addr = vram_addr16(bp, bw, x, y, true);
            write_le16(m_vram.data() + addr, static_cast<u16>(raw_val & 0xFFFFu));
            break;
        }
        case 0x13u: {
            const u32 addr = vram_addr8(bp, bw, x, y);
            m_vram[addr] = static_cast<u8>(raw_val & 0xFFu);
            break;
        }
        case 0x1Bu: {
            const u32 addr = vram_addr32(bp, bw, x, y);
            m_vram[addr + 3u] = static_cast<u8>(raw_val & 0xFFu);
            break;
        }
        case 0x14u: {
            u32 shift = 0;
            const u32 addr = vram_addr4(bp, bw, x, y, shift);
            m_vram[addr] = static_cast<u8>((m_vram[addr] & ~(0xFu << shift)) | ((raw_val & 0xFu) << shift));
            break;
        }
        case 0x24u: {
            const u32 addr = vram_addr32(bp, bw, x, y);
            m_vram[addr + 3u] = static_cast<u8>((m_vram[addr + 3u] & 0xF0u) | (raw_val & 0x0Fu));
            break;
        }
        case 0x2Cu: {
            const u32 addr = vram_addr32(bp, bw, x, y);
            m_vram[addr + 3u] = static_cast<u8>((m_vram[addr + 3u] & 0x0Fu) | ((raw_val & 0x0Fu) << 4u));
            break;
        }
        default:
            break;
    }

    if (m_dispfb1 == 0u && m_dispfb2 == 0u && (bp & 31u) == 0u &&
        (psm == 0x00u || psm == 0x01u || psm == 0x02u || psm == 0x0Au) &&
        x < m_fb_width && y < m_fb_height) {
        const u32 target_fbp = (bp >> 5u) & 0x1FFu;
        if (target_fbp == m_ctx[0].active_draw_fbp || target_fbp == m_ctx[1].active_draw_fbp) {
            const u32 rgba = read_vram_rgba(bp, bw, psm, x, y);
            m_present_fb[static_cast<std::size_t>(y) * m_fb_width + x] = (rgba & 0x00FFFFFFu) | 0xFF000000u;
            pixels_drawn++;
        }
    }
}

u32 GsState::read_vram_texel_raw(u32 bp, u32 bw, u32 psm, u32 x, u32 y) const {
    switch (psm) {
        case 0x00u:
        case 0x01u:
        case 0x30u:
        case 0x31u:
            return read_le32(m_vram.data() + vram_addr32(bp, bw, x, y));
        case 0x02u:
        case 0x32u:
            return read_le16(m_vram.data() + vram_addr16(bp, bw, x, y, false));
        case 0x0Au:
        case 0x3Au:
            return read_le16(m_vram.data() + vram_addr16(bp, bw, x, y, true));
        case 0x13u:
            return m_vram[vram_addr8(bp, bw, x, y)];
        case 0x1Bu:
            return m_vram[vram_addr32(bp, bw, x, y) + 3u];
        case 0x14u: {
            u32 shift = 0;
            const u32 addr = vram_addr4(bp, bw, x, y, shift);
            return (m_vram[addr] >> shift) & 0xFu;
        }
        case 0x24u:
            return m_vram[vram_addr32(bp, bw, x, y) + 3u] & 0xFu;
        case 0x2Cu:
            return (m_vram[vram_addr32(bp, bw, x, y) + 3u] >> 4u) & 0xFu;
        default:
            return 0;
    }
}

u32 GsState::read_vram_rgba(u32 bp, u32 bw, u32 psm, u32 x, u32 y) const {
    const u32 raw = read_vram_texel_raw(bp, bw, psm, x, y);
    switch (psm) {
        case 0x00u:
        case 0x30u:
            return raw;
        case 0x01u:
        case 0x31u: {
            const u32 rgb = raw & 0x00FFFFFFu;
            const u32 a = (m_texa_aem && rgb == 0u) ? 0u : static_cast<u32>(m_texa_ta0);
            return rgb | (a << 24);
        }
        case 0x02u:
        case 0x0Au:
        case 0x32u:
        case 0x3Au:
            return unpack_psm16_texa(static_cast<u16>(raw));
        default:
            return raw;
    }
}

void GsState::exec_local_to_local_trx() {
    PerfScope perf_vram(PerfCat::GsVram);
    const u32 src_blocks = vram_block_span(m_bitblt_sbw, m_bitblt_spsm, m_trx_ssax + m_trx_rrw, m_trx_ssay + m_trx_rrh);
    const u32 dst_blocks = vram_block_span(m_bitblt_dbw, m_bitblt_dpsm, m_trx_dsax + m_trx_rrw, m_trx_dsay + m_trx_rrh);
    flush_overlapping_fbs(m_bitblt_sbp, src_blocks);
    invalidate_overlapping_fbs(m_bitblt_dbp, dst_blocks);
    if (m_hw) m_hw->on_vram_write(m_bitblt_dbp, dst_blocks);
    bump_vram_pages(m_bitblt_dbp, dst_blocks);
    const bool is_color_fb_trx =
        (m_bitblt_dpsm == 0x00u || m_bitblt_dpsm == 0x01u || m_bitblt_dpsm == 0x02u || m_bitblt_dpsm == 0x0Au) &&
        m_trx_rrw >= 32u && m_trx_rrh >= 32u && m_bitblt_dbw > 0u;
    mark_vram_pages_color_upload(m_bitblt_dbp, dst_blocks,
                                 is_color_fb_trx ? static_cast<u8>(m_bitblt_dbw) : u8{0});

    const bool rev_y = (m_trx_dir == 1u || m_trx_dir == 3u);
    const bool rev_x = (m_trx_dir == 2u || m_trx_dir == 3u);
    for (u32 yi = 0; yi < m_trx_rrh; ++yi) {
        const u32 y = rev_y ? (m_trx_rrh - 1u - yi) : yi;
        for (u32 xi = 0; xi < m_trx_rrw; ++xi) {
            const u32 x = rev_x ? (m_trx_rrw - 1u - xi) : xi;
            const u32 v = read_vram_texel_raw(m_bitblt_sbp, m_bitblt_sbw, m_bitblt_spsm, m_trx_ssax + x, m_trx_ssay + y);
            write_vram_pixel(m_bitblt_dbp, m_bitblt_dbw, m_bitblt_dpsm, m_trx_dsax + x, m_trx_dsay + y, v);
        }
    }
}

void GsState::write_hwreg_qwords(const u8* qwords, std::size_t qwc) {
    if (!m_trx_active || !qwords || qwc == 0u) return;

    // Preserve direct-present mirror behavior if DISPFB1/2 are both unset
    if (m_dispfb1 == 0u && m_dispfb2 == 0u) {
        for (std::size_t q = 0; q < qwc && m_trx_active; ++q) {
            handle_hwreg_write(read_le64(qwords + q * 16));
            if (m_trx_active) handle_hwreg_write(read_le64(qwords + q * 16 + 8));
        }
        return;
    }

    u8* vram = m_vram.data();
    const u32 bp = m_bitblt_dbp;
    const u32 bw = m_bitblt_dbw;

    if (m_bitblt_dpsm == 0x00u || m_bitblt_dpsm == 0x30u) {
        // PSMCT32: 4 pixels per qword, block-hoisted 8-pixel runs
        const u32 bw_pages = std::max<u32>(1u, bw);
        const u8* p = qwords;
        std::size_t px_rem = qwc * 4u;
        while (m_trx_active && px_rem > 0u) {
            if (m_trx_x >= m_trx_rrw || m_trx_y >= m_trx_rrh) { m_trx_active = false; break; }
            const u32 row_rem = m_trx_rrw - m_trx_x;
            const u32 chunk = static_cast<u32>(std::min<std::size_t>(px_rem, row_rem));
            const u32 y = m_trx_dsay + m_trx_y;
            const u32 py_off = (y >> 5u) * bw_pages;
            const u32 by = (y >> 3u) & 3u;
            const u32 word_y = ((y >> 1u) & 3u) * 16u;
            const u8* col_row = kCol32[y & 1u];
            u32 x = m_trx_dsax + m_trx_x;
            u32 i = 0;

            while (i < chunk && (x & 7u) != 0u) {
                const u32 blk = (bp + (py_off + (x >> 6u)) * 32u + kBlock32[by][(x >> 3u) & 7u]) & 0x3FFFu;
                write_le32(vram + ((blk * 64u + word_y + col_row[x & 7u]) * 4u), read_le32(p + i * 4u));
                ++x; ++i;
            }
            for (; i + 8u <= chunk; i += 8u, x += 8u) {
                const u32 blk = (bp + (py_off + (x >> 6u)) * 32u + kBlock32[by][(x >> 3u) & 7u]) & 0x3FFFu;
                u8* dst_blk = vram + ((blk * 64u + word_y) * 4u);
                const u8* sp = p + i * 4u;
                for (u32 c = 0; c < 8u; ++c) {
                    write_le32(dst_blk + col_row[c] * 4u, read_le32(sp + c * 4u));
                }
            }
            while (i < chunk) {
                const u32 blk = (bp + (py_off + (x >> 6u)) * 32u + kBlock32[by][(x >> 3u) & 7u]) & 0x3FFFu;
                write_le32(vram + ((blk * 64u + word_y + col_row[x & 7u]) * 4u), read_le32(p + i * 4u));
                ++x; ++i;
            }

            p += static_cast<std::size_t>(chunk) * 4u;
            px_rem -= chunk;
            m_trx_x += chunk;
            if (m_trx_x >= m_trx_rrw) {
                m_trx_x = 0;
                m_trx_y++;
                if (m_trx_y >= m_trx_rrh) m_trx_active = false;
            }
        }
        return;
    }

    if (m_bitblt_dpsm == 0x13u) {
        // PSMT8: 16 bytes per qword, row-hoisted offset LUT + 16-byte block writes
        const u32 bw_pages = std::max<u32>(1u, bw >> 1u);
        const u8* p = qwords;
        std::size_t px_rem = qwc * 16u;
        while (m_trx_active && px_rem > 0u) {
            if (m_trx_x >= m_trx_rrw || m_trx_y >= m_trx_rrh) { m_trx_active = false; break; }
            const u32 row_rem = m_trx_rrw - m_trx_x;
            const u32 chunk = static_cast<u32>(std::min<std::size_t>(px_rem, row_rem));
            const u32 y = m_trx_dsay + m_trx_y;
            const u32 py_off = (y >> 6u) * bw_pages;
            const u32 by = (y >> 4u) & 3u;
            const u32 cy = y & 15u;
            const u32 swap = (((cy + 2u) >> 2u) & 1u) * 4u;
            const u32 py = (((cy & ~3u) >> 1u) + (cy & 1u)) & 7u;
            const u32 word_y = (py >> 1u) * 16u;
            const u8* col_row = kCol32[py & 1u];
            const u32 lane_y = (cy >> 1u) & 1u;
            u8 off16[16];
            for (u32 cx = 0; cx < 16u; ++cx) {
                off16[cx] = static_cast<u8>(col_row[(cx + swap) & 7u] * 4u + lane_y + ((cx >> 2u) & 2u));
            }
            u32 x = m_trx_dsax + m_trx_x;
            u32 i = 0;

            while (i < chunk && (x & 15u) != 0u) {
                const u32 blk = (bp + (py_off + (x >> 7u)) * 32u + kBlock32[by][(x >> 4u) & 7u]) & 0x3FFFu;
                vram[(blk * 64u + word_y) * 4u + off16[x & 15u]] = p[i];
                ++x; ++i;
            }
            for (; i + 16u <= chunk; i += 16u, x += 16u) {
                const u32 blk = (bp + (py_off + (x >> 7u)) * 32u + kBlock32[by][(x >> 4u) & 7u]) & 0x3FFFu;
                u8* dst_blk = vram + ((blk * 64u + word_y) * 4u);
                const u8* sp = p + i;
                for (u32 cx = 0; cx < 16u; ++cx) {
                    dst_blk[off16[cx]] = sp[cx];
                }
            }
            while (i < chunk) {
                const u32 blk = (bp + (py_off + (x >> 7u)) * 32u + kBlock32[by][(x >> 4u) & 7u]) & 0x3FFFu;
                vram[(blk * 64u + word_y) * 4u + off16[x & 15u]] = p[i];
                ++x; ++i;
            }

            p += chunk;
            px_rem -= chunk;
            m_trx_x += chunk;
            if (m_trx_x >= m_trx_rrw) {
                m_trx_x = 0;
                m_trx_y++;
                if (m_trx_y >= m_trx_rrh) m_trx_active = false;
            }
        }
        return;
    }

    if (m_bitblt_dpsm == 0x14u && (m_trx_rrw & 1u) == 0u && (m_trx_x & 1u) == 0u) {
        // PSMT4: 32 nibbles (16 bytes) per qword, row-hoisted offset LUT + 32-nibble block writes
        const u32 bw_pages = std::max<u32>(1u, bw >> 1u);
        const u8* p = qwords;
        std::size_t px_rem = qwc * 32u;
        while (m_trx_active && px_rem >= 2u) {
            if (m_trx_x >= m_trx_rrw || m_trx_y >= m_trx_rrh) { m_trx_active = false; break; }
            const u32 row_rem = m_trx_rrw - m_trx_x;
            const u32 chunk = static_cast<u32>(std::min<std::size_t>(px_rem, row_rem));
            const u32 y = m_trx_dsay + m_trx_y;
            const u32 py_off = (y >> 7u) * bw_pages;
            const u32 by = (y >> 4u) & 7u;
            const u32 cy = y & 15u;
            const u32 swap = (((cy + 2u) >> 2u) & 1u) * 4u;
            const u32 py = (((cy & ~3u) >> 1u) + (cy & 1u)) & 7u;
            const u32 word_y = (py >> 1u) * 16u;
            const u8* col_row = kCol32[py & 1u];
            const u32 shift = ((cy >> 1u) & 1u) * 4u;
            const u8 keep_mask = static_cast<u8>(~(0xFu << shift));
            u8 off32[32];
            for (u32 cx = 0; cx < 32u; ++cx) {
                off32[cx] = static_cast<u8>(col_row[(cx ^ swap) & 7u] * 4u + ((cx >> 3u) & 3u));
            }
            u32 x = m_trx_dsax + m_trx_x;
            u32 i = 0;

            while (i + 2u <= chunk && (x & 31u) != 0u) {
                const u8 packed = p[i >> 1u];
                const u32 blk0 = (bp + (py_off + (x >> 7u)) * 32u + kBlock16[by][(x >> 5u) & 3u]) & 0x3FFFu;
                u8& d0 = vram[(blk0 * 64u + word_y) * 4u + off32[x & 31u]];
                d0 = static_cast<u8>((d0 & keep_mask) | ((packed & 0xFu) << shift));
                ++x;
                const u32 blk1 = (bp + (py_off + (x >> 7u)) * 32u + kBlock16[by][(x >> 5u) & 3u]) & 0x3FFFu;
                u8& d1 = vram[(blk1 * 64u + word_y) * 4u + off32[x & 31u]];
                d1 = static_cast<u8>((d1 & keep_mask) | (((packed >> 4u) & 0xFu) << shift));
                ++x;
                i += 2u;
            }
            for (; i + 32u <= chunk; i += 32u, x += 32u) {
                const u32 blk = (bp + (py_off + (x >> 7u)) * 32u + kBlock16[by][(x >> 5u) & 3u]) & 0x3FFFu;
                u8* dst_blk = vram + ((blk * 64u + word_y) * 4u);
                const u8* sp = p + (i >> 1u);
                for (u32 b = 0; b < 16u; ++b) {
                    const u8 packed = sp[b];
                    u8& d0 = dst_blk[off32[b * 2u]];
                    d0 = static_cast<u8>((d0 & keep_mask) | ((packed & 0xFu) << shift));
                    u8& d1 = dst_blk[off32[b * 2u + 1u]];
                    d1 = static_cast<u8>((d1 & keep_mask) | (((packed >> 4u) & 0xFu) << shift));
                }
            }
            while (i + 2u <= chunk) {
                const u8 packed = p[i >> 1u];
                const u32 blk0 = (bp + (py_off + (x >> 7u)) * 32u + kBlock16[by][(x >> 5u) & 3u]) & 0x3FFFu;
                u8& d0 = vram[(blk0 * 64u + word_y) * 4u + off32[x & 31u]];
                d0 = static_cast<u8>((d0 & keep_mask) | ((packed & 0xFu) << shift));
                ++x;
                const u32 blk1 = (bp + (py_off + (x >> 7u)) * 32u + kBlock16[by][(x >> 5u) & 3u]) & 0x3FFFu;
                u8& d1 = vram[(blk1 * 64u + word_y) * 4u + off32[x & 31u]];
                d1 = static_cast<u8>((d1 & keep_mask) | (((packed >> 4u) & 0xFu) << shift));
                ++x;
                i += 2u;
            }

            p += (chunk >> 1u);
            px_rem -= chunk;
            m_trx_x += chunk;
            if (m_trx_x >= m_trx_rrw) {
                m_trx_x = 0;
                m_trx_y++;
                if (m_trx_y >= m_trx_rrh) m_trx_active = false;
            }
        }
        return;
    }

    if (m_bitblt_dpsm == 0x02u || m_bitblt_dpsm == 0x0Au || m_bitblt_dpsm == 0x32u || m_bitblt_dpsm == 0x3Au) {
        // PSMCT16 / PSMCT16S: 8 halfwords per qword, block-hoisted 16-pixel runs
        const bool is_16s = (m_bitblt_dpsm == 0x0Au || m_bitblt_dpsm == 0x3Au);
        const u32 bw_pages = std::max<u32>(1u, bw);
        const u8* p = qwords;
        std::size_t px_rem = qwc * 8u;
        while (m_trx_active && px_rem > 0u) {
            if (m_trx_x >= m_trx_rrw || m_trx_y >= m_trx_rrh) { m_trx_active = false; break; }
            const u32 row_rem = m_trx_rrw - m_trx_x;
            const u32 chunk = static_cast<u32>(std::min<std::size_t>(px_rem, row_rem));
            const u32 y = m_trx_dsay + m_trx_y;
            const u32 py_off = (y >> 6u) * bw_pages;
            const u32 by = (y >> 3u) & 7u;
            const u32 hw_y = ((y >> 1u) & 3u) * 32u;
            const u8* col_row = kCol32[y & 1u];
            u32 x = m_trx_dsax + m_trx_x;
            u32 i = 0;

            while (i < chunk && (x & 15u) != 0u) {
                const u32 b_idx = is_16s ? kBlock16S[by][(x >> 4u) & 3u] : kBlock16[by][(x >> 4u) & 3u];
                const u32 blk = (bp + (py_off + (x >> 6u)) * 32u + b_idx) & 0x3FFFu;
                const u32 hw = hw_y + ((x & 8u) << 1u) + col_row[x & 7u];
                write_le16(vram + ((blk * 128u + hw) * 2u), read_le16(p + i * 2u));
                ++x; ++i;
            }
            for (; i + 16u <= chunk; i += 16u, x += 16u) {
                const u32 b_idx = is_16s ? kBlock16S[by][(x >> 4u) & 3u] : kBlock16[by][(x >> 4u) & 3u];
                const u32 blk = (bp + (py_off + (x >> 6u)) * 32u + b_idx) & 0x3FFFu;
                u8* dst_blk = vram + ((blk * 128u + hw_y) * 2u);
                const u8* sp = p + i * 2u;
                for (u32 c = 0; c < 8u; ++c) {
                    write_le16(dst_blk + col_row[c] * 2u, read_le16(sp + c * 2u));
                    write_le16(dst_blk + (16u + col_row[c]) * 2u, read_le16(sp + (c + 8u) * 2u));
                }
            }
            while (i < chunk) {
                const u32 b_idx = is_16s ? kBlock16S[by][(x >> 4u) & 3u] : kBlock16[by][(x >> 4u) & 3u];
                const u32 blk = (bp + (py_off + (x >> 6u)) * 32u + b_idx) & 0x3FFFu;
                const u32 hw = hw_y + ((x & 8u) << 1u) + col_row[x & 7u];
                write_le16(vram + ((blk * 128u + hw) * 2u), read_le16(p + i * 2u));
                ++x; ++i;
            }

            p += static_cast<std::size_t>(chunk) * 2u;
            px_rem -= chunk;
            m_trx_x += chunk;
            if (m_trx_x >= m_trx_rrw) {
                m_trx_x = 0;
                m_trx_y++;
                if (m_trx_y >= m_trx_rrh) m_trx_active = false;
            }
        }
        return;
    }

    // Fallback for PSMCT24 / PSMT8H / PSMT4HL / PSMT4HH / odd-width PSMT4
    for (std::size_t q = 0; q < qwc && m_trx_active; ++q) {
        handle_hwreg_write(read_le64(qwords + q * 16));
        if (m_trx_active) handle_hwreg_write(read_le64(qwords + q * 16 + 8));
    }
}

void GsState::handle_hwreg_write(u64 val) {
    auto advance = [&]() {
        m_trx_x++;
        if (m_trx_x >= m_trx_rrw) {
            m_trx_x = 0;
            m_trx_y++;
            if (m_trx_y >= m_trx_rrh) {
                m_trx_active = false;
            }
        }
    };

    switch (m_bitblt_dpsm) {
        case 0x00u:
        case 0x30u: {
            for (int i = 0; i < 2 && m_trx_active; ++i) {
                const u32 px = static_cast<u32>((val >> (i * 32)) & 0xFFFFFFFFu);
                write_vram_pixel(m_bitblt_dbp, m_bitblt_dbw, m_bitblt_dpsm, m_trx_dsax + m_trx_x, m_trx_dsay + m_trx_y, px);
                advance();
            }
            break;
        }
        case 0x01u:
        case 0x31u: {
            for (int b = 0; b < 8 && m_trx_active; ++b) {
                m_rgb24_buf[m_rgb24_cnt++] = static_cast<u8>((val >> (b * 8)) & 0xFFu);
                if (m_rgb24_cnt == 3u) {
                    const u32 rgb = static_cast<u32>(m_rgb24_buf[0]) |
                                    (static_cast<u32>(m_rgb24_buf[1]) << 8) |
                                    (static_cast<u32>(m_rgb24_buf[2]) << 16);
                    write_vram_pixel(m_bitblt_dbp, m_bitblt_dbw, m_bitblt_dpsm, m_trx_dsax + m_trx_x, m_trx_dsay + m_trx_y, rgb);
                    m_rgb24_cnt = 0;
                    advance();
                }
            }
            break;
        }
        case 0x02u:
        case 0x0Au:
        case 0x32u:
        case 0x3Au: {
            for (int i = 0; i < 4 && m_trx_active; ++i) {
                const u16 c = static_cast<u16>((val >> (i * 16)) & 0xFFFFu);
                write_vram_pixel(m_bitblt_dbp, m_bitblt_dbw, m_bitblt_dpsm, m_trx_dsax + m_trx_x, m_trx_dsay + m_trx_y, c);
                advance();
            }
            break;
        }
        case 0x13u: {
            if (m_trx_x + 8u <= m_trx_rrw) {
                const u32 x0 = m_trx_dsax + m_trx_x;
                const u32 y0 = m_trx_dsay + m_trx_y;
                u8* vram = m_vram.data();
                for (u32 i = 0; i < 8u; ++i) {
                    vram[vram_addr8(m_bitblt_dbp, m_bitblt_dbw, x0 + i, y0)] = static_cast<u8>((val >> (i * 8u)) & 0xFFu);
                }
                m_trx_x += 8u;
                if (m_trx_x >= m_trx_rrw) {
                    m_trx_x = 0;
                    m_trx_y++;
                    if (m_trx_y >= m_trx_rrh) m_trx_active = false;
                }
            } else {
                for (int i = 0; i < 8 && m_trx_active; ++i) {
                    const u32 idx = static_cast<u32>((val >> (i * 8)) & 0xFFu);
                    m_vram[vram_addr8(m_bitblt_dbp, m_bitblt_dbw, m_trx_dsax + m_trx_x, m_trx_dsay + m_trx_y)] = static_cast<u8>(idx);
                    advance();
                }
            }
            break;
        }
        case 0x1Bu: {
            for (int i = 0; i < 8 && m_trx_active; ++i) {
                const u32 idx = static_cast<u32>((val >> (i * 8)) & 0xFFu);
                write_vram_pixel(m_bitblt_dbp, m_bitblt_dbw, m_bitblt_dpsm, m_trx_dsax + m_trx_x, m_trx_dsay + m_trx_y, idx);
                advance();
            }
            break;
        }
        case 0x14u:
        case 0x24u:
        case 0x2Cu: {
            for (int i = 0; i < 16 && m_trx_active; ++i) {
                const u32 idx = static_cast<u32>((val >> (i * 4)) & 0xFu);
                write_vram_pixel(m_bitblt_dbp, m_bitblt_dbw, m_bitblt_dpsm, m_trx_dsax + m_trx_x, m_trx_dsay + m_trx_y, idx);
                advance();
            }
            break;
        }
        default:
            for (int i = 0; i < 2 && m_trx_active; ++i) advance();
            break;
    }
}

void GsState::update_clut_block_hash(u32 block_idx) {
    block_idx &= 31u;
    const u32 base = block_idx * 16u;
    u64 h = 14695981039346656037ULL;
    for (u32 i = 0; i < 16u; ++i) {
        h ^= static_cast<u64>(m_clut_rgba[base + i]);
        h *= 1099511628211ULL;
    }
    m_clut_block_hash[block_idx] = h;
}

u64 GsState::compute_active_clut_hash(u32 psm, u32 cpsm, u32 csa) const {
    if (is_8bit_paletted_psm(psm)) {
        const u32 bank_blk = (cpsm == 0u) ? 0u : ((csa & 16u) != 0u ? 16u : 0u);
        const u32 off_blk = csa & 15u;
        u64 h = 14695981039346656037ULL;
        for (u32 b = 0; b < 16u; ++b) {
            h ^= m_clut_block_hash[bank_blk + ((off_blk + b) & 15u)];
            h *= 1099511628211ULL;
        }
        return h;
    }
    const u32 blk = (cpsm == 0u) ? (csa & 15u) : (csa & 31u);
    return m_clut_block_hash[blk];
}

void GsState::refresh_clut_rgba_from_raw() {
    for (std::size_t i = 0; i < m_clut_raw.size(); ++i) {
        m_clut_rgba[i] = unpack_clut_raw(m_clut_raw[i], m_clut_slot_cpsm[i]);
    }
    for (u32 b = 0; b < 32u; ++b) {
        update_clut_block_hash(b);
    }
    m_clut_gen++;
    if (m_clut_gen == 0u) m_clut_gen = 1u;
}

void GsState::load_clut_from_vram(u32 cbp, u32 cpsm, u32 csm, u32 csa, bool is_8bit) {
    PerfScope perf_vram(PerfCat::GsVram);
    if (!m_dirty_fbps.empty()) {
        const u32 blocks = (csm == 0u) ? 4u : vram_block_span(std::max<u32>(1u, m_texclut_cbw), cpsm,
                                                               m_texclut_cou * 16u + (is_8bit ? 256u : 16u),
                                                               m_texclut_cov + 1u);
        flush_overlapping_fbs(cbp, blocks);
    }

    const u32 cbw = std::max<u32>(1u, m_texclut_cbw);
    const u32 cou_px = m_texclut_cou * 16u;
    const u32 cov_px = m_texclut_cov;
    u32 dirty_blk_mask = 0u;

    if (is_8bit) {
        for (u32 i = 0; i < 256u; ++i) {
            u32 raw = 0;
            if (csm == 0u) {
                u32 idx = i;
                if ((idx & 0x18u) == 0x08u || (idx & 0x18u) == 0x10u) {
                    idx ^= 0x18u;
                }
                raw = read_vram_texel_raw(cbp, 1u, cpsm, idx & 15u, idx >> 4u);
            } else {
                raw = read_vram_texel_raw(cbp, cbw, cpsm, cou_px + i, cov_px);
            }
            const u32 slot = (cpsm == 0u)
                ? ((i + (csa & 15u) * 16u) & 0xFFu)
                : (((csa & 16u) * 16u) + ((i + (csa & 15u) * 16u) & 0xFFu));
            const u32 rgba = unpack_clut_raw(raw, cpsm);
            if (m_clut_rgba[slot] != rgba || m_clut_raw[slot] != raw || m_clut_slot_cpsm[slot] != static_cast<u8>(cpsm)) {
                m_clut_raw[slot] = raw;
                m_clut_slot_cpsm[slot] = static_cast<u8>(cpsm);
                m_clut_rgba[slot] = rgba;
                dirty_blk_mask |= (1u << (slot >> 4u));
            }
        }
    } else {
        const u32 base = (cpsm == 0u) ? ((csa & 15u) * 16u) : (csa * 16u);
        for (u32 i = 0; i < 16u; ++i) {
            u32 raw = 0;
            if (csm == 0u) {
                raw = read_vram_texel_raw(cbp, 1u, cpsm, i & 7u, i >> 3u);
            } else {
                raw = read_vram_texel_raw(cbp, cbw, cpsm, cou_px + i, cov_px);
            }
            const u32 slot = (base + i) & 0x1FFu;
            const u32 rgba = unpack_clut_raw(raw, cpsm);
            if (m_clut_rgba[slot] != rgba || m_clut_raw[slot] != raw || m_clut_slot_cpsm[slot] != static_cast<u8>(cpsm)) {
                m_clut_raw[slot] = raw;
                m_clut_slot_cpsm[slot] = static_cast<u8>(cpsm);
                m_clut_rgba[slot] = rgba;
                dirty_blk_mask |= (1u << (slot >> 4u));
            }
        }
    }

    if (dirty_blk_mask != 0u) {
        for (u32 b = 0; b < 32u; ++b) {
            if (dirty_blk_mask & (1u << b)) {
                update_clut_block_hash(b);
            }
        }
        m_clut_gen++;
        if (m_clut_gen == 0u) m_clut_gen = 1u;
    }
}

void GsState::handle_tex0_clut_load(u64 tex0) {
    const u32 psm  = static_cast<u32>((tex0 >> 20) & 0x3Fu);
    if (!is_paletted_psm(psm)) return;

    const u32 cbp  = static_cast<u32>((tex0 >> 37) & 0x3FFFu);
    const u32 cpsm = static_cast<u32>((tex0 >> 51) & 0xFu);
    const u32 csm  = static_cast<u32>((tex0 >> 55) & 1u);
    const u32 csa  = static_cast<u32>((tex0 >> 56) & 0x1Fu);
    const u32 cld  = static_cast<u32>((tex0 >> 61) & 7u);
    const bool is8 = is_8bit_paletted_psm(psm);

    if (cld == 0u) {
        if (!m_clut_explicit_cld) {
            if (!m_clut_fallback_valid || m_clut_fb_cbp != cbp || m_clut_fb_cpsm != cpsm ||
                m_clut_fb_csm != csm || m_clut_fb_csa != csa || m_clut_fb_is8 != is8) {
                load_clut_from_vram(cbp, cpsm, csm, csa, is8);
                m_clut_fb_cbp = cbp;
                m_clut_fb_cpsm = cpsm;
                m_clut_fb_csm = csm;
                m_clut_fb_csa = csa;
                m_clut_fb_is8 = is8;
                m_clut_fallback_valid = true;
            }
        }
        return;
    }

    m_clut_explicit_cld = true;
    bool do_load = false;
    switch (cld) {
        case 1u:
            do_load = true;
            break;
        case 2u:
            do_load = true;
            m_cbp0 = cbp;
            break;
        case 3u:
            do_load = true;
            m_cbp1 = cbp;
            break;
        case 4u:
            if (cbp != m_cbp0) {
                do_load = true;
                m_cbp0 = cbp;
            }
            break;
        case 5u:
            if (cbp != m_cbp1) {
                do_load = true;
                m_cbp1 = cbp;
            }
            break;
        default:
            break;
    }
    if (do_load) {
        load_clut_from_vram(cbp, cpsm, csm, csa, is8);
    }
}

u32 GsState::sample_texel_at(const GsContext& c, const GsFbTarget* rt_src, u32 ix, u32 iy) const {
    const u64 tex0 = c.tex0_reg;
    const u32 psm  = static_cast<u32>((tex0 >> 20) & 0x3Fu);

    if (rt_src) {
        const u32 rt_w = rt_src->fbw * 64u;
        const u32 cx = std::min<u32>(ix, rt_w - 1u);
        const u32 cy = std::min<u32>(iy, rt_src->height - 1u);
        u32 px = rt_src->pixels[static_cast<std::size_t>(cy) * rt_w + cx];
        if (psm == 0x01u) {
            const u32 rgb = px & 0x00FFFFFFu;
            const u32 a = (m_texa_aem && rgb == 0u) ? 0u : static_cast<u32>(m_texa_ta0);
            px = rgb | (a << 24);
        }
        return px;
    }

    const u32 tbp0 = static_cast<u32>(tex0 & 0x3FFFu);
    const u32 tbw  = static_cast<u32>((tex0 >> 14) & 0x3Fu);
    const u32 cpsm = static_cast<u32>((tex0 >> 51) & 0xFu);
    const u32 csa  = static_cast<u32>((tex0 >> 56) & 0x1Fu);

    if (psm == 0x13u || psm == 0x1Bu) {
        const u32 idx = read_vram_texel_raw(tbp0, tbw, psm, ix, iy) & 0xFFu;
        const u32 slot = (cpsm == 0u)
            ? ((idx + (csa & 15u) * 16u) & 0xFFu)
            : (((csa & 16u) * 16u) + ((idx + (csa & 15u) * 16u) & 0xFFu));
        return m_clut_rgba[slot];
    }
    if (psm == 0x14u || psm == 0x24u || psm == 0x2Cu) {
        const u32 idx = read_vram_texel_raw(tbp0, tbw, psm, ix, iy) & 0xFu;
        const u32 base = (cpsm == 0u) ? ((csa & 15u) * 16u) : (csa * 16u);
        return m_clut_rgba[(base + idx) & 0x1FFu];
    }
    return read_vram_rgba(tbp0, tbw, psm, ix, iy);
}

u32 GsState::sample_texture(const GsContext& c, const GsFbTarget* rt_src, f32 u_tex, f32 v_tex, bool bilinear) const {
    const u64 tex0 = c.tex0_reg;
    const u32 tw   = 1u << std::min<u32>(10u, static_cast<u32>((tex0 >> 26) & 0xFu));
    const u32 th   = 1u << std::min<u32>(10u, static_cast<u32>((tex0 >> 30) & 0xFu));

    const u32 wms  = static_cast<u32>(c.clamp_reg & 3u);
    const u32 wmt  = static_cast<u32>((c.clamp_reg >> 2) & 3u);
    const s32 minu = static_cast<s32>((c.clamp_reg >> 4) & 0x3FFu);
    const s32 maxu = static_cast<s32>((c.clamp_reg >> 14) & 0x3FFu);
    const s32 minv = static_cast<s32>((c.clamp_reg >> 24) & 0x3FFu);
    const s32 maxv = static_cast<s32>((c.clamp_reg >> 34) & 0x3FFu);

    if (!bilinear) {
        const u32 ix = apply_clamp_axis(fast_floor(u_tex), tw, wms, minu, maxu);
        const u32 iy = apply_clamp_axis(fast_floor(v_tex), th, wmt, minv, maxv);
        return sample_texel_at(c, rt_src, ix, iy);
    }

    const f32 bu = u_tex - 0.5f;
    const f32 bv = v_tex - 0.5f;
    const s32 iu0 = fast_floor(bu);
    const s32 iv0 = fast_floor(bv);
    const u32 fx = static_cast<u32>(std::clamp<s32>(static_cast<s32>((bu - static_cast<f32>(iu0)) * 256.0f + 0.5f), 0, 256));
    const u32 fy = static_cast<u32>(std::clamp<s32>(static_cast<s32>((bv - static_cast<f32>(iv0)) * 256.0f + 0.5f), 0, 256));

    const u32 ix0 = apply_clamp_axis(iu0, tw, wms, minu, maxu);
    const u32 iy0 = apply_clamp_axis(iv0, th, wmt, minv, maxv);
    if (fx == 0u && fy == 0u) {
        return sample_texel_at(c, rt_src, ix0, iy0);
    }
    const u32 ix1 = apply_clamp_axis(iu0 + 1, tw, wms, minu, maxu);
    const u32 iy1 = apply_clamp_axis(iv0 + 1, th, wmt, minv, maxv);

    const u32 c00 = sample_texel_at(c, rt_src, ix0, iy0);
    const u32 c10 = (fx != 0u) ? sample_texel_at(c, rt_src, ix1, iy0) : c00;
    const u32 c01 = (fy != 0u) ? sample_texel_at(c, rt_src, ix0, iy1) : c00;
    const u32 c11 = (fx != 0u && fy != 0u) ? sample_texel_at(c, rt_src, ix1, iy1) : ((fx != 0u) ? c10 : c01);
    return bilerp_rgba(c00, c10, c01, c11, fx, fy);
}

u32 GsState::apply_tex_func(const GsContext& c, u32 tex, u32 vr, u32 vg, u32 vb, u32 va) const {
    const u32 tr = tex & 0xFFu;
    const u32 tg = (tex >> 8) & 0xFFu;
    const u32 tb = (tex >> 16) & 0xFFu;
    const u32 ta = (tex >> 24) & 0xFFu;
    const bool tcc = ((c.tex0_reg >> 34) & 1u) != 0;
    const u32 tfx  = static_cast<u32>((c.tex0_reg >> 35) & 3u);

    u32 out_r = tr, out_g = tg, out_b = tb, out_a = va;
    switch (tfx) {
        case 0u:
            out_r = std::min<u32>(255u, (tr * vr) >> 7);
            out_g = std::min<u32>(255u, (tg * vg) >> 7);
            out_b = std::min<u32>(255u, (tb * vb) >> 7);
            out_a = tcc ? std::min<u32>(255u, (ta * va) >> 7) : va;
            break;
        case 1u:
            out_r = tr; out_g = tg; out_b = tb;
            out_a = tcc ? ta : va;
            break;
        case 2u:
            out_r = std::min<u32>(255u, ((tr * vr) >> 7) + va);
            out_g = std::min<u32>(255u, ((tg * vg) >> 7) + va);
            out_b = std::min<u32>(255u, ((tb * vb) >> 7) + va);
            out_a = tcc ? std::min<u32>(255u, ta + va) : va;
            break;
        case 3u:
            out_r = std::min<u32>(255u, ((tr * vr) >> 7) + va);
            out_g = std::min<u32>(255u, ((tg * vg) >> 7) + va);
            out_b = std::min<u32>(255u, ((tb * vb) >> 7) + va);
            out_a = tcc ? ta : va;
            break;
    }
    return out_r | (out_g << 8) | (out_b << 16) | (out_a << 24);
}

bool GsState::copy_presented_framebuffer(std::vector<u32>& out_rgba, u32& out_w, u32& out_h) const {
    std::lock_guard<std::mutex> reader_lk(m_reader_mutex);
    int slot = 0;
    u32 w = 0;
    u32 h = 0;
    {
        std::lock_guard<std::mutex> lk(m_fb_mutex);
        if (!m_has_presented) {
            out_w = 0;
            out_h = 0;
            return false;
        }
        if (m_has_new_ready) {
            std::swap(m_front_idx, m_ready_idx);
            m_front_w = m_ready_w;
            m_front_h = m_ready_h;
            m_front_seq = m_ready_seq;
            m_has_new_ready = false;
        }
        slot = m_front_idx;
        w = m_front_w;
        h = m_front_h;
    }
    if (w == 0u || h == 0u) {
        out_w = 0;
        out_h = 0;
        return false;
    }
    const std::vector<u32>& src = m_triple_fb[slot];
    const std::size_t n = static_cast<std::size_t>(w) * h;
    if (out_rgba.size() != n) out_rgba.resize(n);
    std::memcpy(out_rgba.data(), src.data(), n * sizeof(u32));
    out_w = w;
    out_h = h;
    return true;
}

bool GsState::copy_presented_framebuffer_if_newer(u64& in_out_seq, std::vector<u32>& out_rgba, u32& out_w, u32& out_h) const {
    const u64 cur_seq = m_presented_seq.load(std::memory_order_acquire);
    if (cur_seq == 0u || cur_seq == in_out_seq) {
        return false;
    }

    std::lock_guard<std::mutex> reader_lk(m_reader_mutex);
    int slot = 0;
    u32 w = 0;
    u32 h = 0;
    u64 seq = 0;
    {
        std::lock_guard<std::mutex> lk(m_fb_mutex);
        if (!m_has_presented) {
            return false;
        }
        if (m_has_new_ready) {
            std::swap(m_front_idx, m_ready_idx);
            m_front_w = m_ready_w;
            m_front_h = m_ready_h;
            m_front_seq = m_ready_seq;
            m_has_new_ready = false;
        }
        slot = m_front_idx;
        w = m_front_w;
        h = m_front_h;
        seq = m_front_seq;
    }
    if (w == 0u || h == 0u || seq == 0u || seq == in_out_seq) {
        return false;
    }
    const std::vector<u32>& src = m_triple_fb[slot];
    const std::size_t n = static_cast<std::size_t>(w) * h;
    if (out_rgba.size() != n) out_rgba.resize(n);
    std::memcpy(out_rgba.data(), src.data(), n * sizeof(u32));
    out_w = w;
    out_h = h;
    in_out_seq = seq;
    return true;
}

void GsState::set_present_callback(std::function<void()> cb) {
    std::lock_guard<std::mutex> lk(m_cb_mutex);
    m_present_cb = std::move(cb);
    m_has_callbacks.store(static_cast<bool>(m_present_cb) || static_cast<bool>(m_pacing_cb), std::memory_order_release);
}

void GsState::set_frame_pacing_callback(std::function<void()> cb) {
    std::lock_guard<std::mutex> lk(m_cb_mutex);
    m_pacing_cb = std::move(cb);
    m_has_callbacks.store(static_cast<bool>(m_present_cb) || static_cast<bool>(m_pacing_cb), std::memory_order_release);
}

void GsState::notify_present_callbacks() {
    // Invoke instant presenter wakeup FIRST (<0.05ms SetEvent), coalescing <6ms bursts during
    // 550-975 FPS uncapped loading (0ms impact on 60 Hz gameplay where frames are 16.67ms apart),
    // then post-present frame pacer strictly outside m_fb_mutex and m_cb_mutex!
    if (m_has_callbacks.load(std::memory_order_acquire)) {
        std::function<void()> present_cb;
        std::function<void()> pacing_cb;
        {
            std::lock_guard<std::mutex> lk(m_cb_mutex);
            present_cb = m_present_cb;
            pacing_cb = m_pacing_cb;
        }
        if (present_cb) {
            const u64 now_us = static_cast<u64>(
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()).count());
            if (m_last_present_cb_us == 0u || (now_us - m_last_present_cb_us) >= 6000u) {
                m_last_present_cb_us = now_us;
                present_cb();
            }
        }
        if (pacing_cb) pacing_cb();
    }
}

void GsState::mark_vram_pages_color_upload(u32 start_block, u32 num_blocks, u8 dbw_val) {
    const u32 first_page = (start_block >> 5u) & 511u;
    const u32 total_blocks = num_blocks + (start_block & 31u);
    const u32 num_pages = std::min<u32>(512u, std::max<u32>(1u, (total_blocks + 31u) >> 5u));
    for (u32 p = 0; p < num_pages; ++p) {
        m_vram_page_color_dbw[(first_page + p) & 511u] = dbw_val;
    }
}

bool GsState::has_color_upload_pages(u32 fbp, u32 fbw, u32 psm, u32 height) const {
    const u32 ph = psm_page_height(psm);
    const u32 pages_y = std::max<u32>(1u, (height + ph - 1u) / ph);
    const u32 eff_fbw = std::max<u32>(1u, fbw);
    const u8 target_dbw = static_cast<u8>(eff_fbw);
    for (u32 py = 0; py < pages_y; ++py) {
        for (u32 px = 0; px < eff_fbw; ++px) {
            if (m_vram_page_color_dbw[(fbp + py * eff_fbw + px) & 511u] == target_dbw) {
                return true;
            }
        }
    }
    return false;
}

void GsState::populate_hw_draw_desc(const GsContext& c, u32 attrs, u32 eff_fbw, u32 eff_psm, u32 req_h,
                                    bool use_bilinear, const LinearTexView& tex_view, const GsFbTarget* rt_src,
                                    bool will_overwrite_all, GsHwDrawDesc& out_desc) {
    const u32 fbp = c.active_draw_fbp & 0x1FFu;
    GsFbTarget& rt = m_fb_targets[fbp];
    const u32 eff_h = (rt.valid && rt.fbw == eff_fbw && rt.psm == eff_psm && rt.height > req_h) ? rt.height : req_h;

    out_desc = GsHwDrawDesc{};
    out_desc.fbp = fbp;
    out_desc.fbw = eff_fbw;
    out_desc.fb_psm = eff_psm;
    out_desc.fb_height = eff_h;
    out_desc.fbmsk = static_cast<u32>((c.frame_reg >> 32) & 0xFFFFFFFFu);
    out_desc.fba = c.fba;

    out_desc.zbp = static_cast<u32>(c.zbuf_reg & 0x1FFu);
    out_desc.z_psm = static_cast<u32>((c.zbuf_reg >> 24) & 0xFu);
    out_desc.z_write_en = (((c.zbuf_reg >> 32) & 1u) == 0u);

    out_desc.scissor_x0 = c.scissor_x0;
    out_desc.scissor_y0 = c.scissor_y0;
    out_desc.scissor_x1 = c.scissor_x1;
    out_desc.scissor_y1 = c.scissor_y1;

    out_desc.prim_attrs = attrs;
    out_desc.test_reg = c.test_reg;
    out_desc.alpha_reg = c.alpha_reg;
    out_desc.tex0_reg = c.tex0_reg;
    out_desc.tex1_reg = c.tex1_reg;
    out_desc.clamp_reg = c.clamp_reg;
    out_desc.pabe = m_pabe;
    out_desc.scanmsk = m_scanmsk;
    out_desc.use_bilinear = use_bilinear;
    out_desc.will_overwrite_all = will_overwrite_all;

    // Seed GPU RT from CPU VRAM ONCE if CPU TRXDIR=0 uploaded a matching-stride color image to this FBP
    if (m_hw && !m_hw->is_rt_resident(fbp)) {
        if (!will_overwrite_all && has_color_upload_pages(fbp, eff_fbw, eff_psm, eff_h)) {
            GsFbTarget* cpu_rt = ensure_fb_target(fbp, eff_fbw, eff_psm, eff_h, false);
            if (cpu_rt && !cpu_rt->pixels.empty()) {
                out_desc.seed_rt_rgba = cpu_rt->pixels.data();
                out_desc.seed_rt_stride = cpu_rt->fbw * 64u;
                out_desc.seed_rt_height = cpu_rt->height;
            }
        }
        const u32 ph = psm_page_height(eff_psm);
        const u32 rt_blocks = std::max<u32>(1u, (eff_h + ph - 1u) / ph) * eff_fbw * 32u;
        mark_vram_pages_color_upload(fbp * 32u, rt_blocks, 0u);
    }

    const bool tme = (attrs & (1u << 4)) != 0u;
    const u32 tpsm = static_cast<u32>((c.tex0_reg >> 20) & 0x3Fu);
    if (tme) {
        out_desc.texa_ta0 = m_texa_ta0;
        out_desc.texa_aem = m_texa_aem;
        out_desc.tex_psm24_alpha = tex_view.apply_psm24_alpha;
        if (tex_view.is_zb) {
            out_desc.tex_is_zb = true;
            out_desc.tex_zb_zbp = tex_view.zb_zbp;
        } else if (tex_view.is_rt && rt_src != nullptr) {
            out_desc.tex_is_rt = true;
            out_desc.tex_rt_fbp = tex_view.rt_fbp;
            out_desc.tex_rt_fbw = tex_view.rt_fbw;
            out_desc.tex_rt_height = tex_view.height;
            out_desc.tex_rt_clut_mode = tex_view.rt_clut_mode;
            if (tex_view.rt_clut_mode != 0u) {
                const u32 cpsm = static_cast<u32>((c.tex0_reg >> 51) & 0xFu);
                const u32 csa  = static_cast<u32>((c.tex0_reg >> 56) & 0x1Fu);
                if (tex_view.rt_clut_mode == 1u) { // PSMT8H (256 entries)
                    const u32 bank = (cpsm == 0u) ? 0u : ((csa & 16u) * 16u);
                    const u32 off  = (csa & 15u) * 16u;
                    for (u32 i = 0; i < 256u; ++i) {
                        m_hw_clut_lut[i] = m_clut_rgba[bank + ((i + off) & 0xFFu)];
                    }
                } else { // PSMT4HL / PSMT4HH (16 entries)
                    const u32 base = (cpsm == 0u) ? ((csa & 15u) * 16u) : (csa * 16u);
                    for (u32 i = 0; i < 16u; ++i) {
                        m_hw_clut_lut[i] = m_clut_rgba[(base + i) & 0x1FFu];
                    }
                }
                out_desc.tex_clut_rgba = m_hw_clut_lut.data();
                out_desc.tex_clut_hash = compute_active_clut_hash(tpsm, cpsm, csa);
            }
            out_desc.tex_cache_slot = static_cast<u32>(kTexCacheSlots);
            out_desc.tex_cache_gen = ++m_hw_tex_gen;
            out_desc.tex_rgba = tex_view.pixels; // non-null fallback if CPU-only RT
        } else {
            out_desc.tex_is_rt = false;
            out_desc.tex_cache_slot = tex_view.cache_slot;
            out_desc.tex_cache_gen = tex_view.cache_gen;
            out_desc.tex_w = tex_view.stride;
            out_desc.tex_h = tex_view.height;
            out_desc.tex_rgba = tex_view.pixels;
        }
    }

    // Track lightweight RT metadata on CPU without allocating rt.pixels!
    // When PS2 shuffles a 32-bit Z-buffer via PSMZ16 -> PSMCT16 into a double-height target (for subsequent
    // PSMT8H depth-fog readback), D3D11GsRenderer stores the target as 32-bit (height / 2) in m_rts[fbp].
    const bool is_z16_shuffle = out_desc.tex_is_zb && (tpsm == 0x32u || tpsm == 0x3Au) &&
                                (eff_psm == 0x02u || eff_psm == 0x0Au) && eff_h >= 512u;
    rt.fbw = eff_fbw;
    rt.height = is_z16_shuffle ? (eff_h / 2u) : eff_h;
    rt.psm = is_z16_shuffle ? 0u : eff_psm;
    rt.valid = true;
    if (fbp == m_last_presented_fbp) {
        m_drew_to_disp_fbp = true;
    }
}

void GsState::commit_back_buffer_to_ready(u32 w, u32 h) {
    const std::vector<u32>& back = m_triple_fb[m_back_idx];
    const u64 next_frame = frames_presented + 1u;
    if (m_last_nonblack_fb.empty() || (next_frame & 31u) == 0u) {
        const u32* p = back.data();
        const std::size_t n = static_cast<std::size_t>(w) * h;
        for (std::size_t i = 0; i < n; i += 29) {
            if ((p[i] & 0x00FFFFFFu) != 0) {
                if (m_last_nonblack_fb.size() != n) m_last_nonblack_fb.resize(n);
                std::memcpy(m_last_nonblack_fb.data(), p, n * sizeof(u32));
                break;
            }
        }
    }

    m_last_presented_slot = m_back_idx;
    m_has_presented_single_thread = true;
    {
        std::lock_guard<std::mutex> lk(m_fb_mutex);
        std::swap(m_back_idx, m_ready_idx);
        m_ready_w = w;
        m_ready_h = h;
        m_has_new_ready = true;
        m_has_presented = true;
        const u64 seq = m_presented_seq.fetch_add(1u, std::memory_order_acq_rel) + 1u;
        m_ready_seq = seq;
    }
    frames_presented = next_frame;
    notify_present_callbacks();
}

void GsState::publish_present_fb_to_shared() {
    std::vector<u32>& dst = m_triple_fb[m_back_idx];
    const std::size_t n = m_present_fb.size();
    std::memcpy(dst.data(), m_present_fb.data(), n * sizeof(u32));
    commit_back_buffer_to_ready(m_fb_width, m_fb_height);
}

void GsState::publish_direct_framebuffer_if_active() {
    if (!m_tri_batch.empty()) flush_tri_batch();
    if (m_dispfb1 == 0u && m_dispfb2 == 0u) {
        if (pixels_drawn == 0u) return;
        publish_present_fb_to_shared();
        return;
    }
    // Single-buffered fallback: only re-present if DISPFB has not flipped for 2+ VBlanks
    // while drawing occurred directly into the displayed FBP (prevents double-presenting!).
    if (m_drew_to_disp_fbp && ++m_vblanks_since_present >= 2u) {
        const bool en1 = (m_pmode & 1u) != 0u;
        const bool en2 = (m_pmode & 2u) != 0u;
        if (en2 && m_dispfb2 != 0u) {
            present_dispfb(m_dispfb2, m_display2 != 0u ? m_display2 : m_display1);
        } else if (en1 && m_dispfb1 != 0u) {
            present_dispfb(m_dispfb1, m_display1 != 0u ? m_display1 : m_display2);
        } else if (m_dispfb1 != 0u) {
            present_dispfb(m_dispfb1, m_display1);
        } else if (m_dispfb2 != 0u) {
            present_dispfb(m_dispfb2, m_display2);
        }
    }
}

void GsState::present_fbp(u32 fbp) {
    if (!m_tri_batch.empty()) flush_tri_batch();
    const GsFbTarget& rt = m_fb_targets[fbp & 0x1FFu];
    const u32 fbw = (rt.valid && rt.fbw > 0u) ? rt.fbw : ((m_fb_width + 63u) / 64u);
    const u32 psm = rt.valid ? rt.psm : 0u;
    const u64 synth_dispfb = static_cast<u64>(fbp & 0x1FFu) |
                             (static_cast<u64>(fbw & 0x3Fu) << 9) |
                             (static_cast<u64>(psm & 0x1Fu) << 15);
    present_dispfb(synth_dispfb, m_display1 != 0u ? m_display1 : m_display2);
}

void GsState::present_dispfb(u64 dispfb, u64 display) {
    if (!m_tri_batch.empty()) flush_tri_batch();
    PerfScope perf_pres(PerfCat::GsPresent);
    const u32 fbp = static_cast<u32>(dispfb & 0x1FFu);
    u32 fbw = static_cast<u32>((dispfb >> 9) & 0x3Fu);
    const u32 psm = static_cast<u32>((dispfb >> 15) & 0x1Fu);
    const u32 dbx = static_cast<u32>((dispfb >> 32) & 0x7FFu);
    const u32 dby = static_cast<u32>((dispfb >> 43) & 0x7FFu);

    const GsFbTarget& rt = m_fb_targets[fbp];
    if (fbw == 0u) {
        fbw = (rt.valid && rt.fbw > 0u) ? rt.fbw : ((m_fb_width + 63u) / 64u);
    }

    u32 disp_w = 0;
    u32 disp_h = 0;
    if (display != 0u) {
        const u32 magh = static_cast<u32>((display >> 23) & 0xFu) + 1u;
        const u32 dw   = static_cast<u32>((display >> 32) & 0xFFFu) + 1u;
        disp_w = dw / magh;
        const u32 magv = static_cast<u32>((display >> 27) & 3u) + 1u;
        const u32 dh   = static_cast<u32>((display >> 44) & 0x7FFu) + 1u;
        disp_h = dh / magv;
    }

    const u32 buf_w = fbw * 64u;
    const u32 src_w = (disp_w >= 256u && disp_w <= buf_w) ? disp_w : std::min<u32>(buf_w, m_fb_width);
    u32 src_h = (disp_h >= 192u && disp_h < m_fb_height) ? disp_h : m_fb_height;

    // Interlaced field mode (SMODE2 INT=1, FFMD=0) or half-height RT: scale rt.height cleanly to 448
    if (rt.valid && rt.fbw == fbw && (rt.psm == psm || (rt.psm <= 1u && psm <= 1u)) &&
        dby == 0u && rt.height >= 192u && rt.height < src_h) {
        src_h = rt.height;
    }

    // Hardware GS zero-copy GPU-to-Swapchain presentation fast path!
    if (m_hw != nullptr) {
        const u64 next_seq = m_presented_seq.load(std::memory_order_relaxed) + 1u;
        if (m_hw->present_dispfb(fbp, fbw, psm, dbx, dby, src_w, src_h, next_seq)) {
            m_presented_seq.fetch_add(1u, std::memory_order_acq_rel);
            frames_presented++;
            m_last_presented_fbp = fbp;
            m_drew_to_disp_fbp = false;
            m_vblanks_since_present = 0;
            notify_present_callbacks();
            return;
        }
    }

    const bool rt_compat = rt.valid && !rt.pixels.empty() && rt.fbw == fbw && (rt.psm == psm || (rt.psm <= 1u && psm <= 1u)) &&
                           (dby + src_h <= rt.height);

    if (!rt_compat && !m_dirty_fbps.empty()) {
        const u32 span_blocks = vram_block_span(fbw, psm, dbx + src_w, dby + src_h);
        flush_overlapping_fbs(fbp * 32u, span_blocks);
    }

    const bool scale_x = (src_w >= 256u && src_w != m_fb_width);
    if (scale_x && (m_scale_disp_w != src_w || m_scale_dbx != dbx || m_scale_buf_w != buf_w || m_scale_sx0.size() != m_fb_width)) {
        m_scale_disp_w = src_w;
        m_scale_dbx = dbx;
        m_scale_buf_w = buf_w;
        m_scale_sx0.resize(m_fb_width);
        m_scale_sx1.resize(m_fb_width);
        m_scale_frac.resize(m_fb_width);
        for (u32 x = 0; x < m_fb_width; ++x) {
            const u32 sx_fp = (x * src_w * 256u) / m_fb_width;
            const u32 sx0 = std::min<u32>(sx_fp >> 8u, src_w - 1u);
            const u32 sx1 = std::min<u32>(sx0 + 1u, src_w - 1u);
            m_scale_sx0[x] = static_cast<u16>(std::min<u32>(dbx + sx0, buf_w - 1u));
            m_scale_sx1[x] = static_cast<u16>(std::min<u32>(dbx + sx1, buf_w - 1u));
            m_scale_frac[x] = static_cast<u16>(sx_fp & 0xFFu);
        }
    }

    // Write directly into triple-buffer back slot (eliminates redundant 1.15 MiB copy!)
    u32* out_fb = m_triple_fb[m_back_idx].data();
    const u32 bp = fbp * 32u;
    const u16* sx0_lut = scale_x ? m_scale_sx0.data() : nullptr;
    const u16* sx1_lut = scale_x ? m_scale_sx1.data() : nullptr;
    const u16* frac_lut = scale_x ? m_scale_frac.data() : nullptr;

    if (rt_compat) {
        const u32* sdata = rt.pixels.data();
        const bool fast_unscaled = (!scale_x && dbx + m_fb_width <= buf_w);
        if (scale_x && !m_workers.empty()) {
            struct PresentScaleCtx {
                const u32* sdata;
                u32* out_fb;
                const u16* sx0_lut;
                const u16* sx1_lut;
                const u16* frac_lut;
                u32 fb_w;
                u32 fb_h;
                u32 buf_w;
                u32 src_h;
                u32 dby;
            } pctx{ sdata, out_fb, sx0_lut, sx1_lut, frac_lut, m_fb_width, m_fb_height, buf_w, src_h, dby };

            dispatch_bands([](void* p, int band, int num_bands) {
                auto* ctx = static_cast<PresentScaleCtx*>(p);
                const u32 y0 = (ctx->fb_h * static_cast<u32>(band)) / static_cast<u32>(num_bands);
                const u32 y1 = (ctx->fb_h * static_cast<u32>(band + 1)) / static_cast<u32>(num_bands);
                for (u32 y = y0; y < y1; ++y) {
                    const u32 sy = ctx->dby + ((ctx->src_h == ctx->fb_h) ? y : ((y * ctx->src_h) / ctx->fb_h));
                    const u32* srow = ctx->sdata + static_cast<std::size_t>(sy) * ctx->buf_w;
                    u32* drow = ctx->out_fb + static_cast<std::size_t>(y) * ctx->fb_w;
                    scale_row_opaque_simd(drow, srow, ctx->sx0_lut, ctx->sx1_lut, ctx->frac_lut, ctx->fb_w);
                }
            }, &pctx);
        } else {
            for (u32 y = 0; y < m_fb_height; ++y) {
                const u32 sy = dby + ((src_h == m_fb_height) ? y : ((y * src_h) / m_fb_height));
                const u32* srow = sdata + static_cast<std::size_t>(sy) * buf_w;
                u32* drow = out_fb + static_cast<std::size_t>(y) * m_fb_width;
                if (fast_unscaled) {
                    copy_row_opaque_simd(drow, srow + dbx, m_fb_width);
                } else if (scale_x) {
                    scale_row_opaque_simd(drow, srow, sx0_lut, sx1_lut, frac_lut, m_fb_width);
                } else {
                    for (u32 x = 0; x < m_fb_width; ++x) {
                        const u32 sx = std::min<u32>(dbx + x, buf_w - 1u);
                        drow[x] = srow[sx] | 0xFF000000u;
                    }
                }
            }
        }
    } else {
        for (u32 y = 0; y < m_fb_height; ++y) {
            const u32 sy = dby + ((src_h == m_fb_height) ? y : ((y * src_h) / m_fb_height));
            u32* drow = out_fb + static_cast<std::size_t>(y) * m_fb_width;
            if (scale_x) {
                for (u32 x = 0; x < m_fb_width; ++x) {
                    const u32 p0 = read_vram_rgba(bp, fbw, psm, sx0_lut[x], sy);
                    const u32 p1 = read_vram_rgba(bp, fbw, psm, sx1_lut[x], sy);
                    drow[x] = lerp_rgb_opaque_px(p0, p1, frac_lut[x]);
                }
            } else {
                for (u32 x = 0; x < m_fb_width; ++x) {
                    drow[x] = read_vram_rgba(bp, fbw, psm, dbx + x, sy) | 0xFF000000u;
                }
            }
        }
    }

    m_last_presented_fbp = fbp;
    m_drew_to_disp_fbp = false;
    m_vblanks_since_present = 0;
    commit_back_buffer_to_ready(m_fb_width, m_fb_height);
}

void GsState::plot_pixel(const GsContext& c, GsFbTarget& rt, GsZTarget* zt, u32 z_mask,
                         s32 x, s32 y, u32 z, u8 r, u8 g, u8 b, u8 a) {
    if (x < c.scissor_x0 || x > c.scissor_x1 || y < c.scissor_y0 || y > c.scissor_y1) return;
    const u32 stride = rt.fbw * 64u;
    if (x < 0 || y < 0 || static_cast<u32>(x) >= stride || static_cast<u32>(y) >= rt.height) return;
    if (m_scanmsk == 2u && (y & 1) == 0) return;
    if (m_scanmsk == 3u && (y & 1) != 0) return;

    const std::size_t idx = static_cast<std::size_t>(y) * stride + static_cast<std::size_t>(x);
    u32* fb = rt.pixels.data();
    const u32 dst = fb[idx];

    if (((c.test_reg >> 14) & 1u) != 0u) {
        const u32 datm = static_cast<u32>((c.test_reg >> 15) & 1u);
        if (((dst >> 31) & 1u) != datm) return;
    }

    bool write_fb = true;
    bool write_alpha = true;
    bool write_zb = (((c.zbuf_reg >> 32) & 1u) == 0u);

    if ((c.test_reg & 1u) != 0u) {
        const u32 atst  = static_cast<u32>((c.test_reg >> 1) & 7u);
        const u32 aref  = static_cast<u32>((c.test_reg >> 4) & 0xFFu);
        const u32 afail = static_cast<u32>((c.test_reg >> 12) & 3u);
        bool apass = true;
        switch (atst) {
            case 0u: apass = false; break;
            case 1u: apass = true; break;
            case 2u: apass = (a <  aref); break;
            case 3u: apass = (a <= aref); break;
            case 4u: apass = (a == aref); break;
            case 5u: apass = (a >= aref); break;
            case 6u: apass = (a >  aref); break;
            case 7u: apass = (a != aref); break;
        }
        if (!apass) {
            switch (afail) {
                case 0u: return;
                case 1u: write_zb = false; break;
                case 2u: write_fb = false; break;
                case 3u: write_zb = false; write_alpha = false; break;
            }
        }
    }

    const u32 pz = z & z_mask;
    if (((c.test_reg >> 16) & 1u) != 0u && zt != nullptr) {
        const u32 ztst = static_cast<u32>((c.test_reg >> 17) & 3u);
        const u32 cur_z = zt->z[idx];
        switch (ztst) {
            case 0u: return;
            case 1u: break;
            case 2u: if (pz < cur_z) return; break;
            case 3u: if (pz <= cur_z) return; break;
        }
        if (write_zb) {
            zt->z[idx] = pz;
        }
    }

    if (!write_fb) return;

    const bool abe = ((active_prim_attrs() & (1u << 6)) != 0u) && (!m_pabe || (a & 0x80u) != 0u);
    u32 out_r = r, out_g = g, out_b = b;
    if (abe) {
        const u32 spec_a = static_cast<u32>(c.alpha_reg & 3u);
        const u32 spec_b = static_cast<u32>((c.alpha_reg >> 2) & 3u);
        const u32 spec_c = static_cast<u32>((c.alpha_reg >> 4) & 3u);
        const u32 spec_d = static_cast<u32>((c.alpha_reg >> 6) & 3u);
        const s32 fix    = static_cast<s32>((c.alpha_reg >> 32) & 0xFFu);

        const s32 sr = r, sg = g, sb = b, sa = a;
        const s32 dr = static_cast<s32>(dst & 0xFFu);
        const s32 dg = static_cast<s32>((dst >> 8) & 0xFFu);
        const s32 db = static_cast<s32>((dst >> 16) & 0xFFu);
        const s32 da = static_cast<s32>((dst >> 24) & 0xFFu);

        const s32 C = (spec_c == 0u) ? sa : (spec_c == 1u) ? da : (spec_c == 2u) ? fix : 0;
        auto blend_ch = [&](s32 Cs, s32 Cd) -> u32 {
            const s32 A = (spec_a == 0u) ? Cs : (spec_a == 1u) ? Cd : 0;
            const s32 B = (spec_b == 0u) ? Cs : (spec_b == 1u) ? Cd : 0;
            const s32 D = (spec_d == 0u) ? Cs : (spec_d == 1u) ? Cd : 0;
            return static_cast<u32>(std::clamp((((A - B) * C) >> 7) + D, 0, 255));
        };
        out_r = blend_ch(sr, dr);
        out_g = blend_ch(sg, dg);
        out_b = blend_ch(sb, db);
    }

    u32 out_a = write_alpha ? static_cast<u32>(a) : ((dst >> 24) & 0xFFu);
    if (c.fba && write_alpha) out_a |= 0x80u;
    const u32 new_px = out_r | (out_g << 8) | (out_b << 16) | (out_a << 24);
    const u32 fbmsk = static_cast<u32>((c.frame_reg >> 32) & 0xFFFFFFFFu);
    const u32 out_px = (new_px & ~fbmsk) | (dst & fbmsk);

    fb[idx] = out_px;

    const u32 zbp0 = static_cast<u32>(m_ctx[0].zbuf_reg & 0x1FFu);
    const u32 zbp1 = static_cast<u32>(m_ctx[1].zbuf_reg & 0x1FFu);
    if (c.active_draw_fbp != 0u && (c.active_draw_fbp == zbp0 || c.active_draw_fbp == zbp1)) {
        const u64 zreg = (c.active_draw_fbp == zbp0) ? m_ctx[0].zbuf_reg : m_ctx[1].zbuf_reg;
        GsZTarget* clear_zt = ensure_z_target(c.active_draw_fbp, stride, rt.height);
        clear_zt->z[idx] = out_px & zbuf_psm_mask(zreg);
    }

    if (m_dispfb1 == 0u && m_dispfb2 == 0u &&
        static_cast<u32>(x) < m_fb_width && static_cast<u32>(y) < m_fb_height) {
        m_present_fb[static_cast<std::size_t>(y) * m_fb_width + static_cast<std::size_t>(x)] =
            (out_px & 0x00FFFFFFu) | 0xFF000000u;
    }
    pixels_drawn++;
}

void GsState::rasterize_line(const GsVertex& v0, const GsVertex& v1) {
    if (!m_tri_batch.empty()) flush_tri_batch();
    const GsContext& c = active_ctx();
    const u32 fbw = static_cast<u32>((c.frame_reg >> 16) & 0x3Fu);
    const u32 fbpsm = static_cast<u32>((c.frame_reg >> 24) & 0x3Fu);
    const u32 min_h = static_cast<u32>(std::max<s32>(0, c.scissor_y1 + 1));

    if (m_hw != nullptr && !(m_dispfb1 == 0u && m_dispfb2 == 0u)) {
        const u32 attrs = active_prim_attrs();
        const bool iip = (attrs & (1u << 3)) != 0u;
        const u32 eff_fbw = (fbw == 0u) ? ((m_fb_width + 63u) / 64u) : fbw;
        const u32 eff_psm = (fbpsm == 0x00u || fbpsm == 0x01u || fbpsm == 0x02u || fbpsm == 0x0Au) ? fbpsm : 0x00u;
        const u32 ph = psm_page_height(eff_psm);
        const u32 req_h = std::clamp<u32>(((std::max<u32>(ph, min_h) + ph - 1u) / ph) * ph, ph, 1024u);

        GsHwDrawDesc desc{};
        LinearTexView empty_tex{};
        populate_hw_draw_desc(c, attrs & ~(1u << 4), eff_fbw, eff_psm, req_h, false, empty_tex, nullptr, false, desc);

        const f32 x0 = static_cast<f32>(v0.x4) * (1.0f / 16.0f) + 0.5f;
        const f32 y0 = static_cast<f32>(v0.y4) * (1.0f / 16.0f) + 0.5f;
        const f32 x1 = static_cast<f32>(v1.x4) * (1.0f / 16.0f) + 0.5f;
        const f32 y1 = static_cast<f32>(v1.y4) * (1.0f / 16.0f) + 0.5f;
        const f32 dx = x1 - x0;
        const f32 dy = y1 - y0;
        const f32 len = std::sqrt(dx * dx + dy * dy);
        const f32 nx = (len > 1e-4f) ? (-dy / len * 0.5f) : 0.5f;
        const f32 ny = (len > 1e-4f) ? ( dx / len * 0.5f) : 0.0f;

        const u32 z_mask = zbuf_psm_mask(c.zbuf_reg);
        const double inv_z = 1.0 / static_cast<double>(z_mask);
        const f32 z0 = static_cast<f32>(static_cast<double>(v0.z & z_mask) * inv_z);
        const f32 z1 = static_cast<f32>(static_cast<double>(v1.z & z_mask) * inv_z);
        const u32 c1 = static_cast<u32>(v1.r) | (static_cast<u32>(v1.g) << 8) |
                       (static_cast<u32>(v1.b) << 16) | (static_cast<u32>(v1.a) << 24);
        const u32 c0 = iip ? (static_cast<u32>(v0.r) | (static_cast<u32>(v0.g) << 8) |
                              (static_cast<u32>(v0.b) << 16) | (static_cast<u32>(v0.a) << 24))
                           : c1;

        const GsHwVertex verts[6] = {
            { x0 - nx, y0 - ny, z0, c0, 0.0f, 0.0f, 1.0f },
            { x0 + nx, y0 + ny, z0, c0, 0.0f, 0.0f, 1.0f },
            { x1 - nx, y1 - ny, z1, c1, 0.0f, 0.0f, 1.0f },
            { x0 + nx, y0 + ny, z0, c0, 0.0f, 0.0f, 1.0f },
            { x1 + nx, y1 + ny, z1, c1, 0.0f, 0.0f, 1.0f },
            { x1 - nx, y1 - ny, z1, c1, 0.0f, 0.0f, 1.0f }
        };
        m_hw->draw_triangles(desc, verts, 6u);
        primitives_drawn++;
        return;
    }

    GsFbTarget& rt = *ensure_fb_target(c.active_draw_fbp, fbw, fbpsm, min_h);
    const u32 zbp = static_cast<u32>(c.zbuf_reg & 0x1FFu);
    const u32 z_mask = zbuf_psm_mask(c.zbuf_reg);
    GsZTarget* zt = ensure_z_target(zbp, rt.fbw * 64u, rt.height);

    const bool iip = (active_prim_attrs() & (1u << 3)) != 0u;
    s32 x0 = v0.x, y0 = v0.y;
    const s32 x1 = v1.x, y1 = v1.y;
    const s32 dx = std::abs(x1 - x0);
    const s32 dy = std::abs(y1 - y0);
    const s32 sx = (x0 < x1) ? 1 : -1;
    const s32 sy = (y0 < y1) ? 1 : -1;
    const s32 steps = std::max<s32>(1, std::max(dx, dy));
    s32 err = dx - dy;

    u32 dirty_y0 = rt.height, dirty_y1 = 0;
    for (s32 s = 0; s <= steps && s < 4096; ++s) {
        const f32 t = static_cast<f32>(s) / static_cast<f32>(steps);
        const u8 r = iip ? static_cast<u8>(std::clamp(v0.r + (v1.r - v0.r) * t, 0.0f, 255.0f)) : v1.r;
        const u8 g = iip ? static_cast<u8>(std::clamp(v0.g + (v1.g - v0.g) * t, 0.0f, 255.0f)) : v1.g;
        const u8 b = iip ? static_cast<u8>(std::clamp(v0.b + (v1.b - v0.b) * t, 0.0f, 255.0f)) : v1.b;
        const u8 a = iip ? static_cast<u8>(std::clamp(v0.a + (v1.a - v0.a) * t, 0.0f, 255.0f)) : v1.a;
        const u32 z = static_cast<u32>(std::clamp(static_cast<double>(v0.z) + (static_cast<double>(v1.z) - static_cast<double>(v0.z)) * t, 0.0, 4294967295.0));
        plot_pixel(c, rt, zt, z_mask, x0, y0, z, r, g, b, a);
        if (y0 >= 0 && static_cast<u32>(y0) < rt.height) {
            dirty_y0 = std::min<u32>(dirty_y0, static_cast<u32>(y0));
            dirty_y1 = std::max<u32>(dirty_y1, static_cast<u32>(y0) + 1u);
        }
        if (x0 == x1 && y0 == y1) break;
        const s32 e2 = err * 2;
        if (e2 > -dy) { err -= dy; x0 += sx; }
        if (e2 < dx)  { err += dx; y0 += sy; }
    }
    if (dirty_y0 < dirty_y1) {
        mark_fb_dirty(c.active_draw_fbp, dirty_y0, dirty_y1);
    }
    primitives_drawn++;
}

void GsState::rasterize_sprite(const GsVertex& v0, const GsVertex& v1) {
    if (!m_tri_batch.empty()) flush_tri_batch();
    PerfScope perf_spr(PerfCat::GsSprite);
    if (g_perf.enabled) g_perf.sprite_draws.fetch_add(1, std::memory_order_relaxed);
    const GsContext& c = active_ctx();
    const u32 fbw = static_cast<u32>((c.frame_reg >> 16) & 0x3Fu);
    const u32 fbpsm = static_cast<u32>((c.frame_reg >> 24) & 0x3Fu);
    const u32 min_h = static_cast<u32>(std::max<s32>(0, c.scissor_y1 + 1));

    const u32 attrs = active_prim_attrs();
    const bool tme = (attrs & (1u << 4)) != 0u;
    const bool abe = (attrs & (1u << 6)) != 0u;
    const bool fst = (attrs & (1u << 8)) != 0u;

    const bool ate   = (c.test_reg & 1u) != 0u;
    const u32 atst   = static_cast<u32>((c.test_reg >> 1) & 7u);
    const u32 aref   = static_cast<u32>((c.test_reg >> 4) & 0xFFu);
    const u32 afail  = static_cast<u32>((c.test_reg >> 12) & 3u);
    const bool eff_ate = ate && (atst != 1u);
    const bool date  = ((c.test_reg >> 14) & 1u) != 0u;
    const bool zte   = ((c.test_reg >> 16) & 1u) != 0u;
    const u32 ztst   = static_cast<u32>((c.test_reg >> 17) & 3u);
    const u32 fbmsk  = static_cast<u32>((c.frame_reg >> 32) & 0xFFFFFFFFu);

    const s32 min_x4 = std::min(v0.x4, v1.x4);
    const s32 max_x4 = std::max(v0.x4, v1.x4);
    const s32 min_y4 = std::min(v0.y4, v1.y4);
    const s32 max_y4 = std::max(v0.y4, v1.y4);
    const s32 pre_x0 = std::max<s32>(0, std::max(c.scissor_x0, (min_x4 + 15) >> 4));
    const s32 pre_y0 = std::max<s32>(0, std::max(c.scissor_y0, (min_y4 + 15) >> 4));
    const s32 pre_x1 = std::min<s32>(c.scissor_x1 + 1, (max_x4 + 15) >> 4);
    const s32 pre_y1 = std::min<s32>(c.scissor_y1 + 1, (max_y4 + 15) >> 4);
    if (pre_x0 >= pre_x1 || pre_y0 >= pre_y1) return;

    if ((zte && ztst == 0u) || (ate && atst == 0u && afail == 0u)) {
        primitives_drawn++;
        return;
    }

    const u32 eff_fbw = (fbw == 0u) ? ((m_fb_width + 63u) / 64u) : fbw;
    const u32 eff_stride = eff_fbw * 64u;
    const u32 ph = psm_page_height((fbpsm <= 2u || fbpsm == 0x0Au) ? fbpsm : 0u);
    const u32 req_h = std::clamp<u32>(((std::max<u32>(ph, min_h) + ph - 1u) / ph) * ph, ph, 1024u);

    // Skip VRAM reload if a flat opaque clear covers the entire render target!
    const bool is_full_clear = !tme && !eff_ate && !date && !abe && fbmsk == 0u && m_scanmsk == 0u &&
                               (!zte || ztst == 1u) &&
                               pre_x0 == 0 && pre_y0 == 0 &&
                               pre_x1 >= static_cast<s32>(eff_stride) &&
                               pre_y1 >= static_cast<s32>(req_h);

    if (m_hw != nullptr && !(m_dispfb1 == 0u && m_dispfb2 == 0u)) {
        const u32 zbp0 = static_cast<u32>(m_ctx[0].zbuf_reg & 0x1FFu);
        const u32 zbp1 = static_cast<u32>(m_ctx[1].zbuf_reg & 0x1FFu);
        const bool target_is_zbuf = (c.active_draw_fbp != 0u) &&
                                    (c.active_draw_fbp == zbp0 || c.active_draw_fbp == zbp1);
        const u64 est_px = static_cast<u64>(pre_x1 - pre_x0) * static_cast<u64>(pre_y1 - pre_y0);

        // PS2 Z-buffer clear sprite: FRAME.FBP points at ZBUF.ZBP
        if (target_is_zbuf && !tme && !eff_ate && !date && !abe) {
            if (g_perf.enabled) g_perf.sprite_fast_clear.fetch_add(1, std::memory_order_relaxed);
            const u32 out_a = c.fba ? (static_cast<u32>(v1.a) | 0x80u) : static_cast<u32>(v1.a);
            const u32 fill_px = static_cast<u32>(v1.r) | (static_cast<u32>(v1.g) << 8) |
                                (static_cast<u32>(v1.b) << 16) | (out_a << 24);
            const u64 zreg = (c.active_draw_fbp == zbp0) ? m_ctx[0].zbuf_reg : m_ctx[1].zbuf_reg;
            const u32 clear_z_mask = zbuf_psm_mask(zreg);
            const f32 z_norm = static_cast<f32>(static_cast<double>(fill_px & clear_z_mask) /
                                                static_cast<double>(clear_z_mask));
            m_hw->clear_depth(c.active_draw_fbp, eff_stride, req_h, z_norm,
                              pre_x0, pre_y0, pre_x1 - 1, pre_y1 - 1);
            pixels_drawn += est_px;
            primitives_drawn++;
            return;
        }

        const u64 tex0 = c.tex0_reg;
        const u32 tw = 1u << std::min<u32>(10u, static_cast<u32>((tex0 >> 26) & 0xFu));
        const u32 th = 1u << std::min<u32>(10u, static_cast<u32>((tex0 >> 30) & 0xFu));
        const f32 dx4 = static_cast<f32>(v1.x4 - v0.x4);
        const f32 dy4 = static_cast<f32>(v1.y4 - v0.y4);
        const f32 q0_safe = (std::isfinite(v0.q) && std::fabs(v0.q) > 1e-6f) ? v0.q : 1.0f;
        const f32 q1_safe = (std::isfinite(v1.q) && std::fabs(v1.q) > 1e-6f) ? v1.q : 1.0f;
        const f32 u0  = fst ? v0.u : (sanitize_f32(v0.s) / q0_safe) * static_cast<f32>(tw);
        const f32 v0t = fst ? v0.v : (sanitize_f32(v0.t) / q0_safe) * static_cast<f32>(th);
        const f32 u1  = fst ? v1.u : (sanitize_f32(v1.s) / q1_safe) * static_cast<f32>(tw);
        const f32 v1t = fst ? v1.v : (sanitize_f32(v1.t) / q1_safe) * static_cast<f32>(th);
        const f32 du  = (dx4 != 0.0f) ? ((u1 - u0) * (16.0f / dx4)) : 0.0f;
        const f32 dv  = (dy4 != 0.0f) ? ((v1t - v0t) * (16.0f / dy4)) : 0.0f;

        const bool mmag_lin = ((c.tex1_reg >> 5) & 1u) != 0u;
        const u32 mmin = static_cast<u32>((c.tex1_reg >> 6) & 7u);
        const bool mmin_lin = (mmin == 1u || mmin == 4u || mmin == 5u);
        const bool is_mag = (std::fabs(du) < 1.0f || std::fabs(dv) < 1.0f);
        const bool use_bilinear = tme && (is_mag ? mmag_lin : mmin_lin);

        const GsFbTarget* rt_src = tme ? prepare_texture_source(tex0) : nullptr;
        LinearTexView tex_view{};
        if (tme) {
            textured_primitives++;
            tex_view = resolve_linear_texture(tex0, rt_src);
        }

        const u32 eff_psm = (fbpsm == 0x00u || fbpsm == 0x01u || fbpsm == 0x02u || fbpsm == 0x0Au) ? fbpsm : 0x00u;
        GsHwDrawDesc desc{};
        populate_hw_draw_desc(c, attrs, eff_fbw, eff_psm, req_h, use_bilinear, tex_view, rt_src, is_full_clear, desc);

        const f32 x0f = static_cast<f32>(v0.x4) * (1.0f / 16.0f);
        const f32 y0f = static_cast<f32>(v0.y4) * (1.0f / 16.0f);
        const f32 x1f = static_cast<f32>(v1.x4) * (1.0f / 16.0f);
        const f32 y1f = static_cast<f32>(v1.y4) * (1.0f / 16.0f);
        const u32 z_mask = zbuf_psm_mask(c.zbuf_reg);
        const f32 z_norm = static_cast<f32>(static_cast<double>(v1.z & z_mask) / static_cast<double>(z_mask));
        const u32 rgba = static_cast<u32>(v1.r) | (static_cast<u32>(v1.g) << 8) |
                         (static_cast<u32>(v1.b) << 16) | (static_cast<u32>(v1.a) << 24);
        const f32 inv_tw = 1.0f / static_cast<f32>(tw);
        const f32 inv_th = 1.0f / static_cast<f32>(th);
        // Compensate D3D11 pixel-center (+0.5, +0.5) evaluation vs PS2 bilinear texel-center sampling
        // so iterative RT-to-RT bloom passes accumulate zero subpixel drift at any 1x..4x resolution scale.
        // Point-sampled sprites (e.g. 1:1 FBP=56 -> FBP=0 present blit) keep 0 bias to avoid negative wrap on row 0.
        const f32 uv_bias_x = use_bilinear ? (-0.5f * du) : 0.0f;
        const f32 uv_bias_y = use_bilinear ? (-0.5f * dv) : 0.0f;
        const f32 s0 = (u0  + uv_bias_x) * inv_tw;
        const f32 t0 = (v0t + uv_bias_y) * inv_th;
        const f32 s1 = (u1  + uv_bias_x) * inv_tw;
        const f32 t1 = (v1t + uv_bias_y) * inv_th;

        const GsHwVertex quad[6] = {
            { x0f, y0f, z_norm, rgba, s0, t0, 1.0f },
            { x1f, y0f, z_norm, rgba, s1, t0, 1.0f },
            { x0f, y1f, z_norm, rgba, s0, t1, 1.0f },
            { x1f, y0f, z_norm, rgba, s1, t0, 1.0f },
            { x1f, y1f, z_norm, rgba, s1, t1, 1.0f },
            { x0f, y1f, z_norm, rgba, s0, t1, 1.0f }
        };
        m_hw->draw_triangles(desc, quad, 6u);
        pixels_drawn += est_px;
        primitives_drawn++;
        return;
    }

    const GsFbTarget* rt_src = tme ? prepare_texture_source(c.tex0_reg) : nullptr;
    GsFbTarget& rt = *ensure_fb_target(c.active_draw_fbp, fbw, fbpsm, min_h, is_full_clear);
    const u32 stride = rt.fbw * 64u;
    if (tme && rt_src != nullptr && !rt_src->valid) {
        rt_src = prepare_texture_source(c.tex0_reg);
    }

    const s32 x0 = pre_x0;
    const s32 x1 = std::min<s32>(static_cast<s32>(stride), pre_x1);
    const s32 y0 = pre_y0;
    const s32 y1 = std::min<s32>(static_cast<s32>(rt.height), pre_y1);
    if (x0 >= x1 || y0 >= y1) return;

    const u64 tex0 = c.tex0_reg;
    const u32 tw = 1u << std::min<u32>(10u, static_cast<u32>((tex0 >> 26) & 0xFu));
    const u32 th = 1u << std::min<u32>(10u, static_cast<u32>((tex0 >> 30) & 0xFu));

    // Signed 12.4 deltas properly support flipped sprites (v1.x4 < v0.x4 or v1.y4 < v0.y4)
    const f32 dx4 = static_cast<f32>(v1.x4 - v0.x4);
    const f32 dy4 = static_cast<f32>(v1.y4 - v0.y4);
    const f32 q0_safe = (std::isfinite(v0.q) && std::fabs(v0.q) > 1e-6f) ? v0.q : 1.0f;
    const f32 q1_safe = (std::isfinite(v1.q) && std::fabs(v1.q) > 1e-6f) ? v1.q : 1.0f;
    const f32 u0  = fst ? v0.u : (sanitize_f32(v0.s) / q0_safe) * static_cast<f32>(tw);
    const f32 v0t = fst ? v0.v : (sanitize_f32(v0.t) / q0_safe) * static_cast<f32>(th);
    const f32 u1  = fst ? v1.u : (sanitize_f32(v1.s) / q1_safe) * static_cast<f32>(tw);
    const f32 v1t = fst ? v1.v : (sanitize_f32(v1.t) / q1_safe) * static_cast<f32>(th);
    const f32 du  = (dx4 != 0.0f) ? ((u1 - u0) * (16.0f / dx4)) : 0.0f;
    const f32 dv  = (dy4 != 0.0f) ? ((v1t - v0t) * (16.0f / dy4)) : 0.0f;

    const bool mmag_lin = ((c.tex1_reg >> 5) & 1u) != 0u;
    const u32 mmin = static_cast<u32>((c.tex1_reg >> 6) & 7u);
    const bool mmin_lin = (mmin == 1u || mmin == 4u || mmin == 5u);
    const bool is_mag = (std::fabs(du) < 1.0f || std::fabs(dv) < 1.0f);
    const bool use_bilinear = tme && (is_mag ? mmag_lin : mmin_lin);

    if (tme) {
        textured_primitives++;
        if (debug_log.size() < 40) {
            const u32 tbp0 = static_cast<u32>(tex0 & 0x3FFFu);
            const u32 psm  = static_cast<u32>((tex0 >> 20) & 0x3Fu);
            const u32 cbp  = static_cast<u32>((tex0 >> 37) & 0x3FFFu);
            debug_log.push_back("SPRITE_TEX tbp0=" + std::to_string(tbp0) +
                                " psm=" + std::to_string(psm) +
                                " cbp=" + std::to_string(cbp) +
                                " size=" + std::to_string(tw) + "x" + std::to_string(th) +
                                " dst=(" + std::to_string(x0) + "," + std::to_string(y0) + ")-(" + std::to_string(x1) + "," + std::to_string(y1) + ")" +
                                " col=(" + std::to_string(v1.r) + "," + std::to_string(v1.g) + "," + std::to_string(v1.b) + "," + std::to_string(v1.a) + ")" +
                                " fbp=" + std::to_string(c.active_draw_fbp));
        }
    } else if (debug_log.size() < 40) {
        debug_log.push_back("SPRITE_FLAT dst=(" + std::to_string(x0) + "," + std::to_string(y0) + ")-(" + std::to_string(x1) + "," + std::to_string(y1) + ")" +
                            " col=(" + std::to_string(v1.r) + "," + std::to_string(v1.g) + "," + std::to_string(v1.b) + "," + std::to_string(v1.a) + ")" +
                            " fbp=" + std::to_string(c.active_draw_fbp));
    }

    const u32 zbp = static_cast<u32>(c.zbuf_reg & 0x1FFu);
    const u32 z_mask = zbuf_psm_mask(c.zbuf_reg);
    GsZTarget* zt = ensure_z_target(zbp, stride, rt.height);

    const bool write_zb = zte && (((c.zbuf_reg >> 32) & 1u) == 0u);
    const u32 zbp0 = static_cast<u32>(m_ctx[0].zbuf_reg & 0x1FFu);
    const u32 zbp1 = static_cast<u32>(m_ctx[1].zbuf_reg & 0x1FFu);
    const bool target_is_zbuf = (c.active_draw_fbp != 0u) &&
                                (c.active_draw_fbp == zbp0 || c.active_draw_fbp == zbp1);

    u32* fb = rt.pixels.data();
    const bool mirror_pres = (m_dispfb1 == 0u && m_dispfb2 == 0u);
    const u32 pz = v1.z & z_mask;

    // Fast path 1: Flat opaque sprite / screen clear / Z-buffer clear
    if (!tme && !eff_ate && !date && !abe && m_scanmsk == 0u && (!zte || ztst == 1u)) {
        if (g_perf.enabled) g_perf.sprite_fast_clear.fetch_add(1, std::memory_order_relaxed);
        const u32 out_a = c.fba ? (static_cast<u32>(v1.a) | 0x80u) : static_cast<u32>(v1.a);
        const u32 fill_px = static_cast<u32>(v1.r) | (static_cast<u32>(v1.g) << 8) |
                            (static_cast<u32>(v1.b) << 16) | (out_a << 24);
        const u32 pres_px = (fill_px & 0x00FFFFFFu) | 0xFF000000u;
        const s32 span = x1 - x0;

        GsZTarget* clear_zt = nullptr;
        u32 clear_z_mask = 0xFFFFFFFFu;
        if (target_is_zbuf) {
            const u64 zreg = (c.active_draw_fbp == zbp0) ? m_ctx[0].zbuf_reg : m_ctx[1].zbuf_reg;
            clear_z_mask = zbuf_psm_mask(zreg);
            clear_zt = ensure_z_target(c.active_draw_fbp, stride, rt.height);
        }

        for (s32 y = y0; y < y1; ++y) {
            const std::size_t row_off = static_cast<std::size_t>(y) * stride + static_cast<std::size_t>(x0);
            if (fbmsk == 0u) {
                std::fill_n(fb + row_off, span, fill_px);
                if (clear_zt) {
                    std::fill_n(clear_zt->z.data() + row_off, span, fill_px & clear_z_mask);
                }
            } else {
                const u32 inv_msk = ~fbmsk;
                const u32 masked_new = fill_px & inv_msk;
                for (s32 i = 0; i < span; ++i) {
                    const u32 out_px = masked_new | (fb[row_off + i] & fbmsk);
                    fb[row_off + i] = out_px;
                    if (clear_zt) clear_zt->z[row_off + i] = out_px & clear_z_mask;
                }
            }
            if (write_zb) {
                std::fill_n(zt->z.data() + row_off, span, pz);
            }
            if (mirror_pres && static_cast<u32>(y) < m_fb_height && x0 < static_cast<s32>(m_fb_width)) {
                const s32 pspan = std::min<s32>(x1, static_cast<s32>(m_fb_width)) - x0;
                const std::size_t poff = static_cast<std::size_t>(y) * m_fb_width + static_cast<std::size_t>(x0);
                if (fbmsk == 0u) {
                    std::fill_n(m_present_fb.data() + poff, pspan, pres_px);
                } else {
                    for (s32 i = 0; i < pspan; ++i) {
                        m_present_fb[poff + i] = (fb[row_off + i] & 0x00FFFFFFu) | 0xFF000000u;
                    }
                }
            }
        }
        mark_fb_dirty(c.active_draw_fbp, static_cast<u32>(y0), static_cast<u32>(y1));
        pixels_drawn += static_cast<u64>(span) * static_cast<u64>(y1 - y0);
        primitives_drawn++;
        return;
    }

    // Resolve linear texture view (RT linear buffer OR 256-slot decoded texture cache)
    LinearTexView tex_view{};
    if (tme) {
        tex_view = resolve_linear_texture(tex0, rt_src);
        // Safe self-blit snapshot reusing persistent scratch buffer (only overlapping row slice!)
        if (rt_src == &rt && tex_view.pixels != nullptr) {
            const f32 tv_start = v0t + (static_cast<f32>(y0 * 16 - v0.y4) / 16.0f) * dv;
            const f32 tv_end   = v0t + (static_cast<f32>((y1 - 1) * 16 - v0.y4) / 16.0f) * dv;
            const u32 wmt_chk  = static_cast<u32>((c.clamp_reg >> 2) & 3u);
            s32 sy_lo = 0, sy_hi = static_cast<s32>(rt.height);
            if (wmt_chk <= 1u) {
                sy_lo = std::clamp<s32>(fast_floor(std::min(tv_start, tv_end)) - 1, 0, static_cast<s32>(rt.height));
                sy_hi = std::clamp<s32>(fast_floor(std::max(tv_start, tv_end)) + 2, 0, static_cast<s32>(rt.height));
            }
            const bool rows_overlap = !(sy_hi <= y0 || sy_lo >= y1);
            if (rows_overlap && sy_lo < sy_hi) {
                if (m_self_blit_scratch.size() < rt.pixels.size()) {
                    m_self_blit_scratch.resize(rt.pixels.size());
                }
                const std::size_t off = static_cast<std::size_t>(sy_lo) * stride;
                const std::size_t cnt = static_cast<std::size_t>(sy_hi - sy_lo) * stride;
                std::memcpy(m_self_blit_scratch.data() + off, rt.pixels.data() + off, cnt * sizeof(u32));
                tex_view.pixels = m_self_blit_scratch.data();
            }
        }
    }

    const u32 wms  = static_cast<u32>(c.clamp_reg & 3u);
    const u32 wmt  = static_cast<u32>((c.clamp_reg >> 2) & 3u);
    const s32 minu = static_cast<s32>((c.clamp_reg >> 4) & 0x3FFu);
    const s32 maxu = static_cast<s32>((c.clamp_reg >> 14) & 0x3FFu);
    const s32 minv = static_cast<s32>((c.clamp_reg >> 24) & 0x3FFu);
    const s32 maxv = static_cast<s32>((c.clamp_reg >> 34) & 0x3FFu);
    const bool tcc = ((tex0 >> 34) & 1u) != 0u;
    const u32 tfx  = static_cast<u32>((tex0 >> 35) & 3u);

    // Fast path 2: 1:1 non-filtered opaque copy from linear texture/RT
    if (tme && tex_view.pixels != nullptr && !tex_view.apply_psm24_alpha && !use_bilinear &&
        !eff_ate && !date && !abe && fbmsk == 0u && !c.fba && m_scanmsk == 0u && (!zte || ztst == 1u) &&
        !mirror_pres && !target_is_zbuf && wms <= 1u && wmt <= 1u) {
        const bool identity_mod = (tfx == 1u && tcc) ||
                                  (tfx == 0u && tcc && v1.r == 128 && v1.g == 128 && v1.b == 128 && v1.a == 128);
        const s32 span = x1 - x0;
        const f32 x_base_tu = u0 + (static_cast<f32>(x0 * 16 - v0.x4) / 16.0f) * du;
        if (identity_mod && std::fabs(du - 1.0f) < 1e-4f && std::fabs(dv - 1.0f) < 1e-4f) {
            const s32 sx_start = fast_floor(x_base_tu);
            const u32 lim_w = std::min<u32>(tex_view.stride, tw);
            const u32 lim_h = std::min<u32>(tex_view.height, th);
            if (sx_start >= 0 && sx_start + span <= static_cast<s32>(lim_w)) {
                if (g_perf.enabled) g_perf.sprite_fast_copy.fetch_add(1, std::memory_order_relaxed);
                const u32* sdata = tex_view.pixels;
                for (s32 y = y0; y < y1; ++y) {
                    const f32 tv = v0t + (static_cast<f32>(y * 16 - v0.y4) / 16.0f) * dv;
                    const s32 sy = std::clamp<s32>(fast_floor(tv), 0, static_cast<s32>(lim_h) - 1);
                    const u32* srow = sdata + static_cast<std::size_t>(sy) * tex_view.stride + static_cast<std::size_t>(sx_start);
                    u32* drow = fb + static_cast<std::size_t>(y) * stride + static_cast<std::size_t>(x0);
                    std::memmove(drow, srow, static_cast<std::size_t>(span) * sizeof(u32));
                    if (write_zb) {
                        std::fill_n(zt->z.data() + static_cast<std::size_t>(y) * stride + static_cast<std::size_t>(x0), span, pz);
                    }
                }
                mark_fb_dirty(c.active_draw_fbp, static_cast<u32>(y0), static_cast<u32>(y1));
                pixels_drawn += static_cast<u64>(span) * static_cast<u64>(y1 - y0);
                primitives_drawn++;
                return;
            }
        }
    }

    const u32 datm   = static_cast<u32>((c.test_reg >> 15) & 1u);
    const bool zb_en = (((c.zbuf_reg >> 32) & 1u) == 0u);
    const u32 spec_a = static_cast<u32>(c.alpha_reg & 3u);
    const u32 spec_b = static_cast<u32>((c.alpha_reg >> 2) & 3u);
    const u32 spec_c = static_cast<u32>((c.alpha_reg >> 4) & 3u);
    const u32 spec_d = static_cast<u32>((c.alpha_reg >> 6) & 3u);
    const s32 fix    = static_cast<s32>((c.alpha_reg >> 32) & 0xFFu);

    u32* zb = zt->z.data();
    const u32* tpx = tex_view.pixels;
    const u32 tstride = tex_view.stride;
    const u32 th_lim = tex_view.height;
    const bool psm24_a = tex_view.apply_psm24_alpha;
    const u32 ta0_sh = static_cast<u32>(m_texa_ta0) << 24;

    const f32 u_start_f = u0 + (static_cast<f32>(x0 * 16 - v0.x4) / 16.0f) * du;
    const s64 u_start_fp = fast_f32_to_fp8(u_start_f);
    const s64 du_fp      = fast_f32_to_fp8(std::clamp(du, -4096.0f, 4096.0f));
    const u32 vr = v1.r, vg = v1.g, vb = v1.b, va = v1.a;
    const u32 vcol_packed = vr | (vg << 8) | (vb << 16) | (va << 24);
    const bool std_alpha_blend = (spec_a == 0u && spec_b == 1u && spec_d == 1u);
    const bool add_alpha_blend = (spec_a == 0u && spec_b == 2u && spec_d == 1u);

    // Fast path 3: Specialized SIMD + multi-band linear textured sprite (opaque, alpha cutout, bloom/glow/UI blend)
    if (tme && tpx != nullptr && !psm24_a && (!eff_ate || afail == 0u) && !date &&
        fbmsk == 0u && !c.fba && !mirror_pres && !target_is_zbuf && (!abe || !m_pabe)) {
        const u32 max_tx = tstride - 1u;
        const u32 max_ty = th_lim - 1u;
        const bool pot_repeat_x = (tstride == tw) && (wms == 0u);
        const u32 tw_mask = tw - 1u;
        const u32 pz4[4] = { pz, pz, pz, pz };
        const u32 vcol4[4] = { vcol_packed, vcol_packed, vcol_packed, vcol_packed };

        auto render_sprite_band = [&](s32 by0, s32 by1) -> u64 {
            u64 band_px = 0;
            for (s32 y = by0; y < by1; ++y) {
                if (m_scanmsk == 2u && (y & 1) == 0) continue;
                if (m_scanmsk == 3u && (y & 1) != 0) continue;

                const std::size_t row_base = static_cast<std::size_t>(y) * stride;
                const f32 tv = v0t + (static_cast<f32>(y * 16 - v0.y4) / 16.0f) * dv;
                u32 fy = 0;
                const u32* trow0 = nullptr;
                const u32* trow1 = nullptr;
                if (!use_bilinear) {
                    const u32 iy0 = clamp_axis_fast(fast_floor(tv), th, wmt, minv, maxv);
                    trow0 = tpx + static_cast<std::size_t>(std::min<u32>(iy0, max_ty)) * tstride;
                } else {
                    const s64 bv_fp = fast_f32_to_fp8(tv) - 128;
                    const s32 iv0 = static_cast<s32>(bv_fp >> 8);
                    fy = static_cast<u32>(bv_fp & 255);
                    const u32 iy0 = clamp_axis_fast(iv0, th, wmt, minv, maxv);
                    const u32 iy1 = (fy != 0u) ? clamp_axis_fast(iv0 + 1, th, wmt, minv, maxv) : iy0;
                    trow0 = tpx + static_cast<std::size_t>(std::min<u32>(iy0, max_ty)) * tstride;
                    trow1 = tpx + static_cast<std::size_t>(std::min<u32>(iy1, max_ty)) * tstride;
                }

                auto sample_row = [&](s64 ufp) -> u32 {
                    if (pot_repeat_x) {
                        if (!use_bilinear) {
                            return trow0[static_cast<u32>(ufp >> 8) & tw_mask];
                        }
                        const s64 bu_fp = ufp - 128;
                        const u32 iu0 = static_cast<u32>(bu_fp >> 8);
                        const u32 fx = static_cast<u32>(bu_fp & 255);
                        const u32 x0c = iu0 & tw_mask;
                        if ((fx | fy) == 0u) return trow0[x0c];
                        const u32 x1c = (iu0 + 1u) & tw_mask;
                        return bilerp_rgba(trow0[x0c], trow0[x1c], trow1[x0c], trow1[x1c], fx, fy);
                    }
                    if (!use_bilinear) {
                        const u32 ix = clamp_axis_fast(static_cast<s32>(ufp >> 8), tw, wms, minu, maxu);
                        return trow0[std::min<u32>(ix, max_tx)];
                    }
                    const s64 bu_fp = ufp - 128;
                    const s32 iu0 = static_cast<s32>(bu_fp >> 8);
                    const u32 fx = static_cast<u32>(bu_fp & 255);
                    const u32 ix0 = clamp_axis_fast(iu0, tw, wms, minu, maxu);
                    const u32 x0c = std::min<u32>(ix0, max_tx);
                    if ((fx | fy) == 0u) return trow0[x0c];
                    const u32 ix1 = (fx != 0u) ? clamp_axis_fast(iu0 + 1, tw, wms, minu, maxu) : ix0;
                    const u32 x1c = std::min<u32>(ix1, max_tx);
                    return bilerp_rgba(trow0[x0c], trow0[x1c], trow1[x0c], trow1[x1c], fx, fy);
                };

                auto apply_tfx_scalar = [&](u32 tex) -> u32 {
                    if (tfx == 0u) return modulate_px_tfx0(tex, vcol_packed, tcc);
                    if (tfx == 1u) return tcc ? tex : ((tex & 0x00FFFFFFu) | (vcol_packed & 0xFF000000u));
                    const u32 r = std::min<u32>(255u, (((tex & 0xFFu) * vr) >> 7) + va);
                    const u32 g = std::min<u32>(255u, ((((tex >> 8) & 0xFFu) * vg) >> 7) + va);
                    const u32 b = std::min<u32>(255u, ((((tex >> 16) & 0xFFu) * vb) >> 7) + va);
                    const u32 ta = (tex >> 24) & 0xFFu;
                    const u32 a = tcc ? ((tfx == 2u) ? std::min<u32>(255u, ta + va) : ta) : va;
                    return r | (g << 8) | (b << 16) | (a << 24);
                };

                auto blend_one = [&](u32 src_px, u32 dst_px) -> u32 {
                    const s32 C = (spec_c == 0u) ? static_cast<s32>((src_px >> 24) & 0xFFu)
                                : (spec_c == 1u) ? static_cast<s32>((dst_px >> 24) & 0xFFu)
                                : (spec_c == 2u) ? fix : 0;
                    if (std_alpha_blend) return blend_std_rgba_px(src_px, dst_px, C);
                    if (add_alpha_blend) return blend_add_rgba_px(src_px, dst_px, C);
                    const s32 sr = static_cast<s32>(src_px & 0xFFu), dr = static_cast<s32>(dst_px & 0xFFu);
                    const s32 sg = static_cast<s32>((src_px >> 8) & 0xFFu), dg = static_cast<s32>((dst_px >> 8) & 0xFFu);
                    const s32 sb = static_cast<s32>((src_px >> 16) & 0xFFu), db = static_cast<s32>((dst_px >> 16) & 0xFFu);
                    auto bch = [&](s32 Cs, s32 Cd) -> u32 {
                        const s32 A = (spec_a == 0u) ? Cs : (spec_a == 1u) ? Cd : 0;
                        const s32 B = (spec_b == 0u) ? Cs : (spec_b == 1u) ? Cd : 0;
                        const s32 D = (spec_d == 0u) ? Cs : (spec_d == 1u) ? Cd : 0;
                        return static_cast<u32>(std::clamp((((A - B) * C) >> 7) + D, 0, 255));
                    };
                    return bch(sr, dr) | (bch(sg, dg) << 8) | (bch(sb, db) << 16) | (src_px & 0xFF000000u);
                };

                s64 u_fp = u_start_fp;
                s32 x = x0;
                // 4-pixel vectorized span loop
                for (; x + 4 <= x1; x += 4, u_fp += du_fp * 4) {
                    const std::size_t idx = row_base + static_cast<std::size_t>(x);
                    const u32 zmask = zte ? z_pass_mask_4px(pz4, zb + idx, ztst) : 0xFu;
                    if (zmask == 0u) continue;

                    if (zmask == 0xFu && !eff_ate && tfx <= 1u) {
                        u32 col4[4];
                        const u32 tex4[4] = {
                            sample_row(u_fp),
                            sample_row(u_fp + du_fp),
                            sample_row(u_fp + du_fp * 2),
                            sample_row(u_fp + du_fp * 3)
                        };
                        if (tfx == 0u) {
                            modulate_4px_tfx0(tex4, vcol4, col4, tcc);
                        } else if (tcc) {
                            col4[0] = tex4[0]; col4[1] = tex4[1]; col4[2] = tex4[2]; col4[3] = tex4[3];
                        } else {
                            const u32 am = vcol_packed & 0xFF000000u;
                            col4[0] = (tex4[0] & 0x00FFFFFFu) | am;
                            col4[1] = (tex4[1] & 0x00FFFFFFu) | am;
                            col4[2] = (tex4[2] & 0x00FFFFFFu) | am;
                            col4[3] = (tex4[3] & 0x00FFFFFFu) | am;
                        }
                        if (write_zb) {
                            zb[idx] = pz; zb[idx + 1] = pz; zb[idx + 2] = pz; zb[idx + 3] = pz;
                        }
                        if (!abe) {
                            fb[idx] = col4[0]; fb[idx + 1] = col4[1]; fb[idx + 2] = col4[2]; fb[idx + 3] = col4[3];
                        } else if (std_alpha_blend || add_alpha_blend) {
                            const u32 d0 = fb[idx], d1 = fb[idx + 1], d2 = fb[idx + 2], d3 = fb[idx + 3];
                            const s32 c0 = (spec_c == 0u) ? static_cast<s32>((col4[0] >> 24) & 0xFFu) : (spec_c == 1u) ? static_cast<s32>((d0 >> 24) & 0xFFu) : (spec_c == 2u) ? fix : 0;
                            const s32 c1 = (spec_c == 0u) ? static_cast<s32>((col4[1] >> 24) & 0xFFu) : (spec_c == 1u) ? static_cast<s32>((d1 >> 24) & 0xFFu) : (spec_c == 2u) ? fix : 0;
                            const s32 c2 = (spec_c == 0u) ? static_cast<s32>((col4[2] >> 24) & 0xFFu) : (spec_c == 1u) ? static_cast<s32>((d2 >> 24) & 0xFFu) : (spec_c == 2u) ? fix : 0;
                            const s32 c3 = (spec_c == 0u) ? static_cast<s32>((col4[3] >> 24) & 0xFFu) : (spec_c == 1u) ? static_cast<s32>((d3 >> 24) & 0xFFu) : (spec_c == 2u) ? fix : 0;
                            if (std_alpha_blend) {
                                blend_std_2px(col4[0], col4[1], d0, d1, c0, c1, fb[idx], fb[idx + 1]);
                                blend_std_2px(col4[2], col4[3], d2, d3, c2, c3, fb[idx + 2], fb[idx + 3]);
                            } else {
                                blend_add_2px(col4[0], col4[1], d0, d1, c0, c1, fb[idx], fb[idx + 1]);
                                blend_add_2px(col4[2], col4[3], d2, d3, c2, c3, fb[idx + 2], fb[idx + 3]);
                            }
                        } else {
                            for (int k = 0; k < 4; ++k) fb[idx + k] = blend_one(col4[k], fb[idx + k]);
                        }
                        band_px += 4u;
                    } else {
                        for (int k = 0; k < 4; ++k) {
                            if (!(zmask & (1u << k))) continue;
                            const u32 col = apply_tfx_scalar(sample_row(u_fp + du_fp * k));
                            if (eff_ate && !alpha_test_pass((col >> 24) & 0xFFu, atst, aref)) continue;
                            if (write_zb) zb[idx + k] = pz;
                            fb[idx + k] = abe ? blend_one(col, fb[idx + k]) : col;
                            band_px++;
                        }
                    }
                }
                // Tail 0..3 pixels
                for (; x < x1; ++x, u_fp += du_fp) {
                    const std::size_t idx = row_base + static_cast<std::size_t>(x);
                    if (zte) {
                        const u32 cur_z = zb[idx];
                        if ((ztst == 2u && pz < cur_z) || (ztst == 3u && pz <= cur_z)) continue;
                    }
                    const u32 col = apply_tfx_scalar(sample_row(u_fp));
                    if (eff_ate && !alpha_test_pass((col >> 24) & 0xFFu, atst, aref)) continue;
                    if (write_zb) zb[idx] = pz;
                    fb[idx] = abe ? blend_one(col, fb[idx]) : col;
                    band_px++;
                }
            }
            return band_px;
        };

        u64 local_px = 0;
        const s64 area_px = static_cast<s64>(x1 - x0) * static_cast<s64>(y1 - y0);
        if ((y1 - y0) >= 128 && area_px >= 100000 && (use_bilinear || abe || eff_ate || area_px >= 200000)) {
            if (g_perf.enabled) {
                g_perf.sprite_fast_simd.fetch_add(1, std::memory_order_relaxed);
                g_perf.sprite_parallel_dispatches.fetch_add(1, std::memory_order_relaxed);
            }
            struct SpriteBandCtx {
                decltype(&render_sprite_band) fn;
                s32 y0;
                s32 y1;
                u64 band_px[kGsMaxBands]{};
            } sctx{ &render_sprite_band, y0, y1, {} };

            dispatch_bands([](void* p, int band, int num_bands) {
                auto* ctx = static_cast<SpriteBandCtx*>(p);
                const s32 total_h = ctx->y1 - ctx->y0;
                const s32 by0 = ctx->y0 + (total_h * band) / num_bands;
                const s32 by1 = ctx->y0 + (total_h * (band + 1)) / num_bands;
                if (by0 < by1) {
                    ctx->band_px[band] = (*ctx->fn)(by0, by1);
                }
            }, &sctx);
            for (int b = 0; b < m_num_bands; ++b) local_px += sctx.band_px[b];
        } else {
            if (g_perf.enabled) g_perf.sprite_fast_simd.fetch_add(1, std::memory_order_relaxed);
            local_px = render_sprite_band(y0, y1);
        }

        if (local_px > 0u) {
            mark_fb_dirty(c.active_draw_fbp, static_cast<u32>(y0), static_cast<u32>(y1));
        }
        pixels_drawn += local_px;
        primitives_drawn++;
        return;
    }

    // General Hoisted Sprite Rasterizer (DATE, PABE, FBMSK, FBA, AFAIL!=0, mirror_pres, untextured, PSMCT24 RT)
    if (g_perf.enabled) g_perf.sprite_general.fetch_add(1, std::memory_order_relaxed);
    GsZTarget* clear_zt = nullptr;
    u32 clear_z_mask = 0xFFFFFFFFu;
    if (target_is_zbuf) {
        const u64 zreg = (c.active_draw_fbp == zbp0) ? m_ctx[0].zbuf_reg : m_ctx[1].zbuf_reg;
        clear_z_mask = zbuf_psm_mask(zreg);
        clear_zt = ensure_z_target(c.active_draw_fbp, stride, rt.height);
    }

    u32* pres = m_present_fb.data();
    const bool early_z = zte && (!eff_ate || afail == 0u);

    auto fetch_lin = [&](u32 ix, u32 iy) -> u32 {
        if (tpx != nullptr) {
            u32 px = tpx[static_cast<std::size_t>(std::min<u32>(iy, th_lim - 1u)) * tstride +
                         std::min<u32>(ix, tstride - 1u)];
            if (psm24_a) {
                const u32 rgb = px & 0x00FFFFFFu;
                px = rgb | ((m_texa_aem && rgb == 0u) ? 0u : ta0_sh);
            }
            return px;
        }
        return sample_texel_at(c, rt_src, ix, iy);
    };

    u64 local_px = 0;

    for (s32 y = y0; y < y1; ++y) {
        if (m_scanmsk == 2u && (y & 1) == 0) continue;
        if (m_scanmsk == 3u && (y & 1) != 0) continue;

        const std::size_t row_base = static_cast<std::size_t>(y) * stride;
        const bool row_in_pres = mirror_pres && (static_cast<u32>(y) < m_fb_height);
        const std::size_t pres_row = row_in_pres ? (static_cast<std::size_t>(y) * m_fb_width) : 0u;

        // Precompute vertical texture row indices ONCE per scanline!
        u32 iy0 = 0, iy1 = 0, fy = 0;
        const u32* trow0 = nullptr;
        const u32* trow1 = nullptr;
        if (tme) {
            const f32 tv = v0t + (static_cast<f32>(y * 16 - v0.y4) / 16.0f) * dv;
            if (!use_bilinear) {
                iy0 = clamp_axis_fast(fast_floor(tv), th, wmt, minv, maxv);
                if (tpx != nullptr) {
                    trow0 = tpx + static_cast<std::size_t>(std::min<u32>(iy0, th_lim - 1u)) * tstride;
                }
            } else {
                const s64 bv_fp = fast_f32_to_fp8(tv) - 128;
                const s32 iv0 = static_cast<s32>(bv_fp >> 8);
                fy = static_cast<u32>(bv_fp & 255);
                iy0 = clamp_axis_fast(iv0, th, wmt, minv, maxv);
                iy1 = (fy != 0u) ? clamp_axis_fast(iv0 + 1, th, wmt, minv, maxv) : iy0;
                if (tpx != nullptr) {
                    trow0 = tpx + static_cast<std::size_t>(std::min<u32>(iy0, th_lim - 1u)) * tstride;
                    trow1 = tpx + static_cast<std::size_t>(std::min<u32>(iy1, th_lim - 1u)) * tstride;
                }
            }
        }

        s64 u_fp = u_start_fp;
        for (s32 x = x0; x < x1; ++x, u_fp += du_fp) {
            const std::size_t idx = row_base + static_cast<std::size_t>(x);

            if (early_z) {
                const u32 cur_z = zb[idx];
                if ((ztst == 2u && pz < cur_z) || (ztst == 3u && pz <= cur_z)) continue;
            }

            u32 dst = 0u;
            bool dst_loaded = false;
            if (date) {
                dst = fb[idx];
                dst_loaded = true;
                if (((dst >> 31) & 1u) != datm) continue;
            }

            u32 r = vr, g = vg, b = vb, a = va;
            if (tme) {
                u32 tex;
                if (!use_bilinear) {
                    const u32 ix = clamp_axis_fast(static_cast<s32>(u_fp >> 8), tw, wms, minu, maxu);
                    if (trow0 != nullptr) {
                        tex = trow0[std::min<u32>(ix, tstride - 1u)];
                        if (psm24_a) {
                            const u32 rgb = tex & 0x00FFFFFFu;
                            tex = rgb | ((m_texa_aem && rgb == 0u) ? 0u : ta0_sh);
                        }
                    } else {
                        tex = fetch_lin(ix, iy0);
                    }
                } else {
                    const s64 bu_fp = u_fp - 128;
                    const s32 iu0 = static_cast<s32>(bu_fp >> 8);
                    const u32 fx = static_cast<u32>(bu_fp & 255);
                    const u32 ix0 = clamp_axis_fast(iu0, tw, wms, minu, maxu);
                    if (fx == 0u && fy == 0u) {
                        if (trow0 != nullptr) {
                            tex = trow0[std::min<u32>(ix0, tstride - 1u)];
                            if (psm24_a) {
                                const u32 rgb = tex & 0x00FFFFFFu;
                                tex = rgb | ((m_texa_aem && rgb == 0u) ? 0u : ta0_sh);
                            }
                        } else {
                            tex = fetch_lin(ix0, iy0);
                        }
                    } else {
                        const u32 ix1 = (fx != 0u) ? clamp_axis_fast(iu0 + 1, tw, wms, minu, maxu) : ix0;
                        u32 c00, c10, c01, c11;
                        if (trow0 != nullptr) {
                            const u32 x0c = std::min<u32>(ix0, tstride - 1u);
                            const u32 x1c = std::min<u32>(ix1, tstride - 1u);
                            c00 = trow0[x0c];
                            c10 = trow0[x1c];
                            c01 = trow1[x0c];
                            c11 = trow1[x1c];
                            if (psm24_a) {
                                auto fix24 = [&](u32 p) {
                                    const u32 rgb = p & 0x00FFFFFFu;
                                    return rgb | ((m_texa_aem && rgb == 0u) ? 0u : ta0_sh);
                                };
                                c00 = fix24(c00); c10 = fix24(c10); c01 = fix24(c01); c11 = fix24(c11);
                            }
                        } else {
                            c00 = fetch_lin(ix0, iy0);
                            c10 = (fx != 0u) ? fetch_lin(ix1, iy0) : c00;
                            c01 = (fy != 0u) ? fetch_lin(ix0, iy1) : c00;
                            c11 = (fx != 0u && fy != 0u) ? fetch_lin(ix1, iy1) : ((fx != 0u) ? c10 : c01);
                        }
                        tex = bilerp_rgba(c00, c10, c01, c11, fx, fy);
                    }
                }

                const u32 tr = tex & 0xFFu;
                const u32 tg = (tex >> 8) & 0xFFu;
                const u32 tb = (tex >> 16) & 0xFFu;
                const u32 ta = (tex >> 24) & 0xFFu;
                if (tfx == 0u) {
                    r = std::min<u32>(255u, (tr * r) >> 7);
                    g = std::min<u32>(255u, (tg * g) >> 7);
                    b = std::min<u32>(255u, (tb * b) >> 7);
                    if (tcc) a = std::min<u32>(255u, (ta * a) >> 7);
                } else if (tfx == 1u) {
                    r = tr; g = tg; b = tb;
                    if (tcc) a = ta;
                } else {
                    r = std::min<u32>(255u, ((tr * r) >> 7) + a);
                    g = std::min<u32>(255u, ((tg * g) >> 7) + a);
                    b = std::min<u32>(255u, ((tb * b) >> 7) + a);
                    if (tcc) a = (tfx == 2u) ? std::min<u32>(255u, ta + a) : ta;
                }
            }

            bool write_fb = true;
            bool write_alpha = true;
            bool w_zb = zb_en;

            if (eff_ate) {
                if (!alpha_test_pass(a, atst, aref)) {
                    if (afail == 0u) continue;
                    if (afail == 1u) w_zb = false;
                    else if (afail == 2u) write_fb = false;
                    else { w_zb = false; write_alpha = false; }
                }
            }

            if (zte) {
                if (!early_z) {
                    const u32 cur_z = zb[idx];
                    if ((ztst == 2u && pz < cur_z) || (ztst == 3u && pz <= cur_z)) continue;
                }
                if (w_zb) zb[idx] = pz;
            }
            if (!write_fb) continue;

            const bool do_blend = abe && (!m_pabe || (a & 0x80u) != 0u);
            if (!dst_loaded && (do_blend || fbmsk != 0u || !write_alpha)) {
                dst = fb[idx];
            }

            u32 out_r = r, out_g = g, out_b = b;
            if (do_blend) {
                const s32 da = static_cast<s32>((dst >> 24) & 0xFFu);
                const s32 C = (spec_c == 0u) ? static_cast<s32>(a) : (spec_c == 1u) ? da : (spec_c == 2u) ? fix : 0;
                if (std_alpha_blend) {
                    const u32 blended = blend_std_rgba_px(r | (g << 8) | (b << 16) | (a << 24), dst, C);
                    out_r = blended & 0xFFu; out_g = (blended >> 8) & 0xFFu; out_b = (blended >> 16) & 0xFFu;
                } else if (add_alpha_blend) {
                    const u32 blended = blend_add_rgba_px(r | (g << 8) | (b << 16) | (a << 24), dst, C);
                    out_r = blended & 0xFFu; out_g = (blended >> 8) & 0xFFu; out_b = (blended >> 16) & 0xFFu;
                } else {
                    const s32 dr = static_cast<s32>(dst & 0xFFu);
                    const s32 dg = static_cast<s32>((dst >> 8) & 0xFFu);
                    const s32 db = static_cast<s32>((dst >> 16) & 0xFFu);
                    auto bch = [&](s32 Cs, s32 Cd) -> u32 {
                        const s32 A = (spec_a == 0u) ? Cs : (spec_a == 1u) ? Cd : 0;
                        const s32 B = (spec_b == 0u) ? Cs : (spec_b == 1u) ? Cd : 0;
                        const s32 D = (spec_d == 0u) ? Cs : (spec_d == 1u) ? Cd : 0;
                        return static_cast<u32>(std::clamp((((A - B) * C) >> 7) + D, 0, 255));
                    };
                    out_r = bch(static_cast<s32>(r), dr);
                    out_g = bch(static_cast<s32>(g), dg);
                    out_b = bch(static_cast<s32>(b), db);
                }
            }

            u32 out_a = write_alpha ? a : ((dst >> 24) & 0xFFu);
            if (c.fba && write_alpha) out_a |= 0x80u;
            const u32 new_px = out_r | (out_g << 8) | (out_b << 16) | (out_a << 24);
            const u32 out_px = (fbmsk == 0u) ? new_px : ((new_px & ~fbmsk) | (dst & fbmsk));
            fb[idx] = out_px;
            if (clear_zt) clear_zt->z[idx] = out_px & clear_z_mask;
            if (row_in_pres && static_cast<u32>(x) < m_fb_width) {
                pres[pres_row + static_cast<std::size_t>(x)] = (out_px & 0x00FFFFFFu) | 0xFF000000u;
            }
            local_px++;
        }
    }

    if (local_px > 0u) {
        mark_fb_dirty(c.active_draw_fbp, static_cast<u32>(y0), static_cast<u32>(y1));
    }
    pixels_drawn += local_px;
    primitives_drawn++;
}

void GsState::rasterize_triangle_range(const GsContext& c, u32 attrs, GsFbTarget& rt, GsZTarget* zt,
                                       u32 z_mask, const LinearTexView& tex_view, const GsFbTarget* rt_src,
                                       GsZTarget* clear_zt, u32 clear_z_mask, bool mirror_pres, u32* pres_fb,
                                       const GsVertex& v0, const GsVertex& v1, const GsVertex& v2,
                                       s32 band_y0, s32 band_y1, u64& out_px,
                                       GsTriPerfStats* perf_stats) const {
    const u32 stride = rt.fbw * 64u;
    const s32 min_x4 = std::min({v0.x4, v1.x4, v2.x4});
    const s32 max_x4 = std::max({v0.x4, v1.x4, v2.x4});
    const s32 min_y4 = std::min({v0.y4, v1.y4, v2.y4});
    const s32 max_y4 = std::max({v0.y4, v1.y4, v2.y4});

    const s32 min_x = std::max<s32>(0, std::max(c.scissor_x0, (min_x4 + 15) >> 4));
    const s32 max_x = std::min<s32>(static_cast<s32>(stride) - 1, std::min(c.scissor_x1, max_x4 >> 4));
    const s32 min_y = std::max<s32>(band_y0, std::max<s32>(0, std::max(c.scissor_y0, (min_y4 + 15) >> 4)));
    const s32 max_y = std::min<s32>(band_y1 - 1, std::min<s32>(static_cast<s32>(rt.height) - 1, std::min(c.scissor_y1, max_y4 >> 4)));
    if (min_x > max_x || min_y > max_y) return;

    auto edge12_4 = [](const GsVertex& a, const GsVertex& b, s32 px4, s32 py4) -> s64 {
        return static_cast<s64>(px4 - a.x4) * static_cast<s64>(b.y4 - a.y4) -
               static_cast<s64>(py4 - a.y4) * static_cast<s64>(b.x4 - a.x4);
    };

    const s64 area = edge12_4(v0, v1, v2.x4, v2.y4);
    if (area == 0) return;

    const bool tme = (attrs & (1u << 4)) != 0u;
    const bool abe = (attrs & (1u << 6)) != 0u;
    const bool fst = (attrs & (1u << 8)) != 0u;
    const bool iip = (attrs & (1u << 3)) != 0u;

    const s64 sign = (area > 0) ? 1 : -1;
    const double inv_area_d = 1.0 / static_cast<double>(area * sign);
    const f32 inv_area = static_cast<f32>(inv_area_d);

    const s64 g_x0 = static_cast<s64>(v2.y4 - v1.y4) * sign;
    const s64 g_y0 = -static_cast<s64>(v2.x4 - v1.x4) * sign;
    const s64 g_x1 = static_cast<s64>(v0.y4 - v2.y4) * sign;
    const s64 g_y1 = -static_cast<s64>(v0.x4 - v2.x4) * sign;
    const s64 g_x2 = static_cast<s64>(v1.y4 - v0.y4) * sign;
    const s64 g_y2 = -static_cast<s64>(v1.x4 - v0.x4) * sign;

    // Top-left fill rule (eliminates double-blended 1px H/V shared edges)
    const s64 bias0 = ((g_x0 > 0) || (g_x0 == 0 && g_y0 > 0)) ? 0 : -1;
    const s64 bias1 = ((g_x1 > 0) || (g_x1 == 0 && g_y1 > 0)) ? 0 : -1;
    const s64 bias2 = ((g_x2 > 0) || (g_x2 == 0 && g_y2 > 0)) ? 0 : -1;

    const s64 step_x0 = g_x0 * 16;
    const s64 step_y0 = g_y0 * 16;
    const s64 step_x1 = g_x1 * 16;
    const s64 step_y1 = g_y1 * 16;
    const s64 step_x2 = g_x2 * 16;
    const s64 step_y2 = g_y2 * 16;

    const s32 start_x4 = min_x * 16;
    const s32 start_y4 = min_y * 16;
    s64 row_e0 = edge12_4(v1, v2, start_x4, start_y4) * sign + bias0;
    s64 row_e1 = edge12_4(v2, v0, start_x4, start_y4) * sign + bias1;
    s64 row_e2 = edge12_4(v0, v1, start_x4, start_y4) * sign + bias2;

    const u64 tex0 = c.tex0_reg;
    const u32 tw_log = std::min<u32>(10u, static_cast<u32>((tex0 >> 26) & 0xFu));
    const u32 th_log = std::min<u32>(10u, static_cast<u32>((tex0 >> 30) & 0xFu));
    const u32 tw   = 1u << tw_log;
    const u32 th   = 1u << th_log;
    const u32 tw_mask = tw - 1u;
    const u32 th_mask = th - 1u;
    const bool tcc = ((tex0 >> 34) & 1u) != 0u;
    const u32 tfx  = static_cast<u32>((tex0 >> 35) & 3u);
    const u32 wms  = static_cast<u32>(c.clamp_reg & 3u);
    const u32 wmt  = static_cast<u32>((c.clamp_reg >> 2) & 3u);
    const s32 minu = static_cast<s32>((c.clamp_reg >> 4) & 0x3FFu);
    const s32 maxu = static_cast<s32>((c.clamp_reg >> 14) & 0x3FFu);
    const s32 minv = static_cast<s32>((c.clamp_reg >> 24) & 0x3FFu);
    const s32 maxv = static_cast<s32>((c.clamp_reg >> 34) & 0x3FFu);

    const bool mmag_lin = ((c.tex1_reg >> 5) & 1u) != 0u;
    const u32 mmin = static_cast<u32>((c.tex1_reg >> 6) & 7u);
    const bool mmin_lin = (mmin == 1u || mmin == 4u || mmin == 5u);
    const bool use_bilinear = tme && (mmag_lin || mmin_lin);

    const bool ate     = (c.test_reg & 1u) != 0u;
    const u32 atst     = static_cast<u32>((c.test_reg >> 1) & 7u);
    const u32 aref     = static_cast<u32>((c.test_reg >> 4) & 0xFFu);
    const u32 afail    = static_cast<u32>((c.test_reg >> 12) & 3u);
    const bool eff_ate = ate && (atst != 1u);
    const bool date    = ((c.test_reg >> 14) & 1u) != 0u;
    const u32 datm     = static_cast<u32>((c.test_reg >> 15) & 1u);
    const bool zte     = ((c.test_reg >> 16) & 1u) != 0u;
    const u32 ztst     = static_cast<u32>((c.test_reg >> 17) & 3u);
    const bool zb_en   = (((c.zbuf_reg >> 32) & 1u) == 0u);
    const u32 spec_a   = static_cast<u32>(c.alpha_reg & 3u);
    const u32 spec_b   = static_cast<u32>((c.alpha_reg >> 2) & 3u);
    const u32 spec_c   = static_cast<u32>((c.alpha_reg >> 4) & 3u);
    const u32 spec_d   = static_cast<u32>((c.alpha_reg >> 6) & 3u);
    const s32 fix      = static_cast<s32>((c.alpha_reg >> 32) & 0xFFu);
    const u32 fbmsk    = static_cast<u32>((c.frame_reg >> 32) & 0xFFFFFFFFu);

    u32* fb = rt.pixels.data();
    u32* zb = zt->z.data();

    const f32 tw_f = static_cast<f32>(tw);
    const f32 th_f = static_cast<f32>(th);
    const f32 s0 = sanitize_f32(v0.s), s1 = sanitize_f32(v1.s), s2 = sanitize_f32(v2.s);
    const f32 t0 = sanitize_f32(v0.t), t1 = sanitize_f32(v1.t), t2 = sanitize_f32(v2.t);
    const f32 q0 = sanitize_f32(v0.q, 1.0f), q1 = sanitize_f32(v1.q, 1.0f), q2 = sanitize_f32(v2.q, 1.0f);

    const bool affine_st = fst || (std::fabs(q0 - q1) < 1e-6f && std::fabs(q0 - q2) < 1e-6f);
    f32 au0 = 0.0f, au1 = 0.0f, au2 = 0.0f;
    f32 av0 = 0.0f, av1 = 0.0f, av2 = 0.0f;
    if (tme && affine_st) {
        if (fst) {
            au0 = v0.u; au1 = v1.u; au2 = v2.u;
            av0 = v0.v; av1 = v1.v; av2 = v2.v;
        } else {
            const f32 inv_q_c = (std::fabs(q0) > 1e-6f) ? (1.0f / q0) : 1.0f;
            au0 = s0 * inv_q_c * tw_f; au1 = s1 * inv_q_c * tw_f; au2 = s2 * inv_q_c * tw_f;
            av0 = t0 * inv_q_c * th_f; av1 = t1 * inv_q_c * th_f; av2 = t2 * inv_q_c * th_f;
        }
    }

    const double dw0_dx_d = static_cast<double>(step_x0) * inv_area_d;
    const double dw1_dx_d = static_cast<double>(step_x1) * inv_area_d;
    const double dw2_dx_d = static_cast<double>(step_x2) * inv_area_d;
    const f32 dw0_dx = static_cast<f32>(dw0_dx_d);
    const f32 dw1_dx = static_cast<f32>(dw1_dx_d);
    const f32 dw2_dx = static_cast<f32>(dw2_dx_d);

    const double dz_dx = dw0_dx_d * v0.z + dw1_dx_d * v1.z + dw2_dx_d * v2.z;
    const s64 dz_fp = fast_f64_to_fp16(dz_dx);

    const s32 dr_fp = iip ? fast_f32_to_fp16(dw0_dx * v0.r + dw1_dx * v1.r + dw2_dx * v2.r) : 0;
    const s32 dg_fp = iip ? fast_f32_to_fp16(dw0_dx * v0.g + dw1_dx * v1.g + dw2_dx * v2.g) : 0;
    const s32 db_fp = iip ? fast_f32_to_fp16(dw0_dx * v0.b + dw1_dx * v1.b + dw2_dx * v2.b) : 0;
    const s32 da_fp = iip ? fast_f32_to_fp16(dw0_dx * v0.a + dw1_dx * v1.a + dw2_dx * v2.a) : 0;

    const f32 du_dx = (tme && affine_st) ? (dw0_dx * au0 + dw1_dx * au1 + dw2_dx * au2) : 0.0f;
    const f32 dv_dx = (tme && affine_st) ? (dw0_dx * av0 + dw1_dx * av1 + dw2_dx * av2) : 0.0f;
    const s64 du_fp_aff = (tme && affine_st) ? fast_f32_to_fp8(std::clamp(du_dx, -4096.0f, 4096.0f)) : 0;
    const s64 dv_fp_aff = (tme && affine_st) ? fast_f32_to_fp8(std::clamp(dv_dx, -4096.0f, 4096.0f)) : 0;

    const f32 ds_dx = (tme && !affine_st) ? (dw0_dx * s0 + dw1_dx * s1 + dw2_dx * s2) : 0.0f;
    const f32 dt_dx = (tme && !affine_st) ? (dw0_dx * t0 + dw1_dx * t1 + dw2_dx * t2) : 0.0f;
    const f32 dq_dx = (tme && !affine_st) ? (dw0_dx * q0 + dw1_dx * q1 + dw2_dx * q2) : 0.0f;

    const u32* tpx = tex_view.pixels;
    const u32 tstride = tex_view.stride;
    const u32 th_lim = tex_view.height;
    const bool psm24_a = tex_view.apply_psm24_alpha;
    const u32 ta0_sh = static_cast<u32>(m_texa_ta0) << 24;
    const bool fast_lin_tex = (tpx != nullptr && !psm24_a);
    const bool pot_repeat_lin = fast_lin_tex && (tstride == tw) && (th_lim == th) && (wms == 0u) && (wmt == 0u);
    const bool pot_clamp_lin  = fast_lin_tex && (tstride == tw) && (th_lim == th) && (wms <= 1u) && (wmt <= 1u);
    const u32 max_tx = (tstride > 0u) ? (tstride - 1u) : 0u;
    const u32 max_ty = (th_lim > 0u) ? (th_lim - 1u) : 0u;

    auto fetch_tex = [&](u32 ix, u32 iy) -> u32 {
        if (tpx != nullptr) {
            u32 px = tpx[static_cast<std::size_t>(std::min<u32>(iy, max_ty)) * tstride +
                         std::min<u32>(ix, max_tx)];
            if (psm24_a) {
                const u32 rgb = px & 0x00FFFFFFu;
                px = rgb | ((m_texa_aem && rgb == 0u) ? 0u : ta0_sh);
            }
            return px;
        }
        return sample_texel_at(c, rt_src, ix, iy);
    };

    auto sample_uv_fp = [&](s64 ufp, s64 vfp) -> u32 {
        if (pot_repeat_lin) {
            if (!use_bilinear) {
                return tpx[(static_cast<std::size_t>(static_cast<u32>(vfp >> 8) & th_mask) << tw_log) |
                           (static_cast<u32>(ufp >> 8) & tw_mask)];
            }
            const s64 bu_fp = ufp - 128;
            const s64 bv_fp = vfp - 128;
            const u32 iu0 = static_cast<u32>(bu_fp >> 8);
            const u32 iv0 = static_cast<u32>(bv_fp >> 8);
            const u32 fx = static_cast<u32>(bu_fp & 255);
            const u32 fy = static_cast<u32>(bv_fp & 255);
            const u32 x0 = iu0 & tw_mask;
            const u32* r0 = tpx + (static_cast<std::size_t>(iv0 & th_mask) << tw_log);
            if ((fx | fy) == 0u) return r0[x0];
            const u32 x1 = (iu0 + 1u) & tw_mask;
            const u32* r1 = tpx + (static_cast<std::size_t>((iv0 + 1u) & th_mask) << tw_log);
            return bilerp_rgba(r0[x0], r0[x1], r1[x0], r1[x1], fx, fy);
        }
        if (pot_clamp_lin) {
            if (!use_bilinear) {
                const u32 ix = clamp_axis_fast(static_cast<s32>(ufp >> 8), tw, wms, minu, maxu);
                const u32 iy = clamp_axis_fast(static_cast<s32>(vfp >> 8), th, wmt, minv, maxv);
                return tpx[(static_cast<std::size_t>(iy) << tw_log) | ix];
            }
            const s64 bu_fp = ufp - 128;
            const s64 bv_fp = vfp - 128;
            const s32 iu0 = static_cast<s32>(bu_fp >> 8);
            const s32 iv0 = static_cast<s32>(bv_fp >> 8);
            const u32 fx = static_cast<u32>(bu_fp & 255);
            const u32 fy = static_cast<u32>(bv_fp & 255);
            const u32 ix0 = clamp_axis_fast(iu0, tw, wms, minu, maxu);
            const u32 iy0 = clamp_axis_fast(iv0, th, wmt, minv, maxv);
            const u32* r0 = tpx + (static_cast<std::size_t>(iy0) << tw_log);
            if ((fx | fy) == 0u) return r0[ix0];
            const u32 ix1 = (fx != 0u) ? clamp_axis_fast(iu0 + 1, tw, wms, minu, maxu) : ix0;
            const u32 iy1 = (fy != 0u) ? clamp_axis_fast(iv0 + 1, th, wmt, minv, maxv) : iy0;
            const u32* r1 = tpx + (static_cast<std::size_t>(iy1) << tw_log);
            return bilerp_rgba(r0[ix0], r0[ix1], r1[ix0], r1[ix1], fx, fy);
        }
        if (!use_bilinear) {
            const u32 ix = clamp_axis_fast(static_cast<s32>(ufp >> 8), tw, wms, minu, maxu);
            const u32 iy = clamp_axis_fast(static_cast<s32>(vfp >> 8), th, wmt, minv, maxv);
            if (fast_lin_tex) {
                return tpx[static_cast<std::size_t>(std::min<u32>(iy, max_ty)) * tstride + std::min<u32>(ix, max_tx)];
            }
            return fetch_tex(ix, iy);
        }
        const s64 bu_fp = ufp - 128;
        const s64 bv_fp = vfp - 128;
        const s32 iu0 = static_cast<s32>(bu_fp >> 8);
        const s32 iv0 = static_cast<s32>(bv_fp >> 8);
        const u32 fx = static_cast<u32>(bu_fp & 255);
        const u32 fy = static_cast<u32>(bv_fp & 255);
        const u32 ix0 = clamp_axis_fast(iu0, tw, wms, minu, maxu);
        const u32 iy0 = clamp_axis_fast(iv0, th, wmt, minv, maxv);
        if (fast_lin_tex) {
            const u32 x0c = std::min<u32>(ix0, max_tx);
            const u32* r0 = tpx + static_cast<std::size_t>(std::min<u32>(iy0, max_ty)) * tstride;
            if ((fx | fy) == 0u) return r0[x0c];
            const u32 ix1 = (fx != 0u) ? clamp_axis_fast(iu0 + 1, tw, wms, minu, maxu) : ix0;
            const u32 iy1 = (fy != 0u) ? clamp_axis_fast(iv0 + 1, th, wmt, minv, maxv) : iy0;
            const u32 x1c = std::min<u32>(ix1, max_tx);
            const u32* r1 = tpx + static_cast<std::size_t>(std::min<u32>(iy1, max_ty)) * tstride;
            return bilerp_rgba(r0[x0c], r0[x1c], r1[x0c], r1[x1c], fx, fy);
        }
        if ((fx | fy) == 0u) return fetch_tex(ix0, iy0);
        const u32 ix1 = (fx != 0u) ? clamp_axis_fast(iu0 + 1, tw, wms, minu, maxu) : ix0;
        const u32 iy1 = (fy != 0u) ? clamp_axis_fast(iv0 + 1, th, wmt, minv, maxv) : iy0;
        const u32 c00 = fetch_tex(ix0, iy0);
        const u32 c10 = (fx != 0u) ? fetch_tex(ix1, iy0) : c00;
        const u32 c01 = (fy != 0u) ? fetch_tex(ix0, iy1) : c00;
        const u32 c11 = (fx != 0u && fy != 0u) ? fetch_tex(ix1, iy1) : ((fx != 0u) ? c10 : c01);
        return bilerp_rgba(c00, c10, c01, c11, fx, fy);
    };

    auto apply_tfx_px = [&](u32 tex, u32 vcol) -> u32 {
        if (tfx == 0u) return modulate_px_tfx0(tex, vcol, tcc);
        if (tfx == 1u) return tcc ? tex : ((tex & 0x00FFFFFFu) | (vcol & 0xFF000000u));
        const u32 vr = vcol & 0xFFu, vg = (vcol >> 8) & 0xFFu, vb = (vcol >> 16) & 0xFFu, va = (vcol >> 24) & 0xFFu;
        const u32 r = std::min<u32>(255u, (((tex & 0xFFu) * vr) >> 7) + va);
        const u32 g = std::min<u32>(255u, ((((tex >> 8) & 0xFFu) * vg) >> 7) + va);
        const u32 b = std::min<u32>(255u, ((((tex >> 16) & 0xFFu) * vb) >> 7) + va);
        const u32 ta = (tex >> 24) & 0xFFu;
        const u32 a = tcc ? ((tfx == 2u) ? std::min<u32>(255u, ta + va) : ta) : va;
        return r | (g << 8) | (b << 16) | (a << 24);
    };

    const bool std_alpha_blend = (spec_a == 0u && spec_b == 1u && spec_d == 1u);
    const bool add_alpha_blend = (spec_a == 0u && spec_b == 2u && spec_d == 1u);

    auto blend_one = [&](u32 src_px, u32 dst_px) -> u32 {
        const s32 C = (spec_c == 0u) ? static_cast<s32>((src_px >> 24) & 0xFFu)
                    : (spec_c == 1u) ? static_cast<s32>((dst_px >> 24) & 0xFFu)
                    : (spec_c == 2u) ? fix : 0;
        if (std_alpha_blend) return blend_std_rgba_px(src_px, dst_px, C);
        if (add_alpha_blend) return blend_add_rgba_px(src_px, dst_px, C);
        const s32 sr = static_cast<s32>(src_px & 0xFFu), dr = static_cast<s32>(dst_px & 0xFFu);
        const s32 sg = static_cast<s32>((src_px >> 8) & 0xFFu), dg = static_cast<s32>((dst_px >> 8) & 0xFFu);
        const s32 sb = static_cast<s32>((src_px >> 16) & 0xFFu), db = static_cast<s32>((dst_px >> 16) & 0xFFu);
        auto bch = [&](s32 Cs, s32 Cd) -> u32 {
            const s32 A = (spec_a == 0u) ? Cs : (spec_a == 1u) ? Cd : 0;
            const s32 B = (spec_b == 0u) ? Cs : (spec_b == 1u) ? Cd : 0;
            const s32 D = (spec_d == 0u) ? Cs : (spec_d == 1u) ? Cd : 0;
            return static_cast<u32>(std::clamp((((A - B) * C) >> 7) + D, 0, 255));
        };
        return bch(sr, dr) | (bch(sg, dg) << 8) | (bch(sb, db) << 16) | (src_px & 0xFF000000u);
    };

    const u32 fba_or = c.fba ? 0x80000000u : 0u;
    const u32 inv_fbmsk = ~fbmsk;
    auto store_fb_px = [&](std::size_t idx, u32 px) {
        px |= fba_or;
        fb[idx] = (fbmsk == 0u) ? px : ((px & inv_fbmsk) | (fb[idx] & fbmsk));
    };

    const s32 bbox_w = max_x - min_x;
    const bool target_is_zbuf = (clear_zt != nullptr);
    const bool fast_tex_simd  = tme && (!eff_ate || afail == 0u) && !date && (!abe || !m_pabe) && !mirror_pres && !target_is_zbuf;
    const bool fast_flat_simd = !tme && (!eff_ate || afail == 0u) && !date && (!abe || !m_pabe) && !mirror_pres && !target_is_zbuf;
    const bool early_z_gen    = zte && (!eff_ate || afail == 0u);
    const u32 flat_vcol = static_cast<u32>(v2.r) | (static_cast<u32>(v2.g) << 8) |
                          (static_cast<u32>(v2.b) << 16) | (static_cast<u32>(v2.a) << 24);
    u64 local_px = 0;

    for (s32 y = min_y; y <= max_y; ++y, row_e0 += step_y0, row_e1 += step_y1, row_e2 += step_y2) {
        if (m_scanmsk == 2u && (y & 1) == 0) continue;
        if (m_scanmsk == 3u && (y & 1) != 0) continue;

        // Algebraic scanline span clipping [span_lo .. span_hi] relative to min_x
        s64 span_lo = 0;
        s64 span_hi = bbox_w;
        auto clip_edge = [&](s64 e, s64 sx) -> bool {
            if (sx > 0) {
                if (e < 0) {
                    const s64 d = (-e + sx - 1) / sx;
                    if (d > span_lo) span_lo = d;
                }
            } else if (sx < 0) {
                if (e < 0) return false;
                const s64 d = e / (-sx);
                if (d < span_hi) span_hi = d;
            } else {
                if (e < 0) return false;
            }
            return span_lo <= span_hi;
        };
        if (!clip_edge(row_e0, step_x0) || !clip_edge(row_e1, step_x1) || !clip_edge(row_e2, step_x2)) {
            continue;
        }

        // Compute starting weights & fixed-point attributes ONCE at span_lo (zero libm llround/lrint!)
        const f32 w0 = static_cast<f32>(row_e0 + span_lo * step_x0 - bias0) * inv_area;
        const f32 w1 = static_cast<f32>(row_e1 + span_lo * step_x1 - bias1) * inv_area;
        const f32 w2 = static_cast<f32>(row_e2 + span_lo * step_x2 - bias2) * inv_area;

        const double z_start_d = static_cast<double>(w0) * v0.z + static_cast<double>(w1) * v1.z + static_cast<double>(w2) * v2.z;
        s64 z_fp = fast_f64_to_fp16(z_start_d);
        s32 r_fp = iip ? fast_f32_to_fp16(w0 * v0.r + w1 * v1.r + w2 * v2.r) : 0;
        s32 g_fp = iip ? fast_f32_to_fp16(w0 * v0.g + w1 * v1.g + w2 * v2.g) : 0;
        s32 b_fp = iip ? fast_f32_to_fp16(w0 * v0.b + w1 * v1.b + w2 * v2.b) : 0;
        s32 a_fp = iip ? fast_f32_to_fp16(w0 * v0.a + w1 * v1.a + w2 * v2.a) : 0;

        const s32 x_start = min_x + static_cast<s32>(span_lo);
        const s32 x_end   = min_x + static_cast<s32>(span_hi);
        const std::size_t row_base = static_cast<std::size_t>(y) * stride;

        if (fast_tex_simd) {
            const bool do_blend = abe;
            s64 u_fp_cur = 0, v_fp_cur = 0;
            f32 s_cur = 0.0f, t_cur = 0.0f, q_cur = 1.0f;
            f32 tu_cur = 0.0f, tv_cur = 0.0f;
            if (affine_st) {
                u_fp_cur = fast_f32_to_fp8(w0 * au0 + w1 * au1 + w2 * au2);
                v_fp_cur = fast_f32_to_fp8(w0 * av0 + w1 * av1 + w2 * av2);
            } else {
                s_cur = w0 * s0 + w1 * s1 + w2 * s2;
                t_cur = w0 * t0 + w1 * t1 + w2 * t2;
                q_cur = w0 * q0 + w1 * q1 + w2 * q2;
                const f32 inv_q = (std::fabs(q_cur) > 1e-6f) ? (1.0f / q_cur) : 1.0f;
                tu_cur = std::clamp(s_cur * inv_q * tw_f, -65536.0f, 65536.0f);
                tv_cur = std::clamp(t_cur * inv_q * th_f, -65536.0f, 65536.0f);
            }

            // 16-pixel affine subdivision + 4-pixel SIMD Early-Z/Modulate/Blend spans (now includes FBMSK & FBA!)
            for (s32 x = x_start; x <= x_end; ) {
                const s32 chunk = std::min<s32>(16, x_end - x + 1);
                bool linear_chunk = affine_st;
                s64 cu_fp = u_fp_cur;
                s64 cv_fp = v_fp_cur;
                s64 cdu_fp = du_fp_aff;
                s64 cdv_fp = dv_fp_aff;

                if (affine_st) {
                    u_fp_cur += du_fp_aff * chunk;
                    v_fp_cur += dv_fp_aff * chunk;
                } else {
                    const f32 cf = static_cast<f32>(chunk);
                    const f32 s_next = s_cur + ds_dx * cf;
                    const f32 t_next = t_cur + dt_dx * cf;
                    const f32 q_next = q_cur + dq_dx * cf;
                    const f32 q_max = std::max(std::fabs(q_cur), std::fabs(q_next));
                    if (chunk > 1 && (q_cur * q_next > 0.0f) && (q_max > 1e-6f) && (std::fabs(q_next - q_cur) <= 0.125f * q_max)) {
                        const f32 inv_qn = 1.0f / q_next;
                        const f32 tu_next = std::clamp(s_next * inv_qn * tw_f, -65536.0f, 65536.0f);
                        const f32 tv_next = std::clamp(t_next * inv_qn * th_f, -65536.0f, 65536.0f);
                        cu_fp  = fast_f32_to_fp8(tu_cur);
                        cv_fp  = fast_f32_to_fp8(tv_cur);
                        const f32 inv_chunk = 256.0f / cf;
                        cdu_fp = static_cast<s64>((tu_next - tu_cur) * inv_chunk);
                        cdv_fp = static_cast<s64>((tv_next - tv_cur) * inv_chunk);
                        s_cur = s_next; t_cur = t_next; q_cur = q_next;
                        tu_cur = tu_next; tv_cur = tv_next;
                        linear_chunk = true;
                    }
                }

                if (linear_chunk) {
                    s32 rem = chunk;
                    std::size_t idx = row_base + static_cast<std::size_t>(x);
                    while (rem >= 4) {
                        const u32 pz4[4] = {
                            clamp_z_fp(z_fp, z_mask),
                            clamp_z_fp(z_fp + dz_fp, z_mask),
                            clamp_z_fp(z_fp + dz_fp * 2, z_mask),
                            clamp_z_fp(z_fp + dz_fp * 3, z_mask)
                        };
                        z_fp += dz_fp * 4;
                        const u32 zmask = zte ? z_pass_mask_4px(pz4, zb + idx, ztst) : 0xFu;
                        if (zmask == 0u) {
                            cu_fp += cdu_fp * 4;
                            cv_fp += cdv_fp * 4;
                            if (iip) { r_fp += dr_fp * 4; g_fp += dg_fp * 4; b_fp += db_fp * 4; a_fp += da_fp * 4; }
                            idx += 4;
                            rem -= 4;
                            continue;
                        }

                        if (zmask == 0xFu && !eff_ate && tfx <= 1u) {
                            const u32 tex4[4] = {
                                sample_uv_fp(cu_fp, cv_fp),
                                sample_uv_fp(cu_fp + cdu_fp, cv_fp + cdv_fp),
                                sample_uv_fp(cu_fp + cdu_fp * 2, cv_fp + cdv_fp * 2),
                                sample_uv_fp(cu_fp + cdu_fp * 3, cv_fp + cdv_fp * 3)
                            };
                            u32 vcol4[4];
                            if (iip) {
                                for (int k = 0; k < 4; ++k) {
                                    vcol4[k] = clamp_fp16_u8(r_fp) | (clamp_fp16_u8(g_fp) << 8) |
                                               (clamp_fp16_u8(b_fp) << 16) | (clamp_fp16_u8(a_fp) << 24);
                                    r_fp += dr_fp; g_fp += dg_fp; b_fp += db_fp; a_fp += da_fp;
                                }
                            } else {
                                vcol4[0] = flat_vcol; vcol4[1] = flat_vcol; vcol4[2] = flat_vcol; vcol4[3] = flat_vcol;
                            }

                            u32 col4[4];
                            if (tfx == 0u) {
                                modulate_4px_tfx0(tex4, vcol4, col4, tcc);
                            } else if (tcc) {
                                col4[0] = tex4[0]; col4[1] = tex4[1]; col4[2] = tex4[2]; col4[3] = tex4[3];
                            } else {
                                for (int k = 0; k < 4; ++k) {
                                    col4[k] = (tex4[k] & 0x00FFFFFFu) | (vcol4[k] & 0xFF000000u);
                                }
                            }

                            if (zte && zb_en) {
                                zb[idx] = pz4[0]; zb[idx + 1] = pz4[1]; zb[idx + 2] = pz4[2]; zb[idx + 3] = pz4[3];
                            }
                            if (!do_blend) {
                                if (fbmsk == 0u && !c.fba) {
                                    fb[idx] = col4[0]; fb[idx + 1] = col4[1]; fb[idx + 2] = col4[2]; fb[idx + 3] = col4[3];
                                } else {
                                    for (int k = 0; k < 4; ++k) store_fb_px(idx + k, col4[k]);
                                }
                            } else if (std_alpha_blend || add_alpha_blend) {
                                const u32 d0 = fb[idx], d1 = fb[idx + 1], d2 = fb[idx + 2], d3 = fb[idx + 3];
                                const s32 c0 = (spec_c == 0u) ? static_cast<s32>((col4[0] >> 24) & 0xFFu) : (spec_c == 1u) ? static_cast<s32>((d0 >> 24) & 0xFFu) : (spec_c == 2u) ? fix : 0;
                                const s32 c1 = (spec_c == 0u) ? static_cast<s32>((col4[1] >> 24) & 0xFFu) : (spec_c == 1u) ? static_cast<s32>((d1 >> 24) & 0xFFu) : (spec_c == 2u) ? fix : 0;
                                const s32 c2 = (spec_c == 0u) ? static_cast<s32>((col4[2] >> 24) & 0xFFu) : (spec_c == 1u) ? static_cast<s32>((d2 >> 24) & 0xFFu) : (spec_c == 2u) ? fix : 0;
                                const s32 c3 = (spec_c == 0u) ? static_cast<s32>((col4[3] >> 24) & 0xFFu) : (spec_c == 1u) ? static_cast<s32>((d3 >> 24) & 0xFFu) : (spec_c == 2u) ? fix : 0;
                                u32 o0, o1, o2, o3;
                                if (std_alpha_blend) {
                                    blend_std_2px(col4[0], col4[1], d0, d1, c0, c1, o0, o1);
                                    blend_std_2px(col4[2], col4[3], d2, d3, c2, c3, o2, o3);
                                } else {
                                    blend_add_2px(col4[0], col4[1], d0, d1, c0, c1, o0, o1);
                                    blend_add_2px(col4[2], col4[3], d2, d3, c2, c3, o2, o3);
                                }
                                if (fbmsk == 0u && !c.fba) {
                                    fb[idx] = o0; fb[idx + 1] = o1; fb[idx + 2] = o2; fb[idx + 3] = o3;
                                } else {
                                    fb[idx]     = (((o0 | fba_or) & inv_fbmsk) | (d0 & fbmsk));
                                    fb[idx + 1] = (((o1 | fba_or) & inv_fbmsk) | (d1 & fbmsk));
                                    fb[idx + 2] = (((o2 | fba_or) & inv_fbmsk) | (d2 & fbmsk));
                                    fb[idx + 3] = (((o3 | fba_or) & inv_fbmsk) | (d3 & fbmsk));
                                }
                            } else {
                                for (int k = 0; k < 4; ++k) store_fb_px(idx + k, blend_one(col4[k], fb[idx + k]));
                            }
                            local_px += 4u;
                        } else {
                            for (int k = 0; k < 4; ++k) {
                                if (zmask & (1u << k)) {
                                    const u32 vcol = iip
                                        ? (clamp_fp16_u8(r_fp) | (clamp_fp16_u8(g_fp) << 8) |
                                           (clamp_fp16_u8(b_fp) << 16) | (clamp_fp16_u8(a_fp) << 24))
                                        : flat_vcol;
                                    const u32 col = apply_tfx_px(sample_uv_fp(cu_fp + cdu_fp * k, cv_fp + cdv_fp * k), vcol);
                                    if (!eff_ate || alpha_test_pass((col >> 24) & 0xFFu, atst, aref)) {
                                        if (zte && zb_en) zb[idx + k] = pz4[k];
                                        store_fb_px(idx + k, do_blend ? blend_one(col, fb[idx + k]) : col);
                                        local_px++;
                                    }
                                }
                                if (iip) { r_fp += dr_fp; g_fp += dg_fp; b_fp += db_fp; a_fp += da_fp; }
                            }
                        }
                        cu_fp += cdu_fp * 4;
                        cv_fp += cdv_fp * 4;
                        idx += 4;
                        rem -= 4;
                    }
                    // Tail 0..3 pixels of affine chunk
                    while (rem > 0) {
                        const u32 pz = clamp_z_fp(z_fp, z_mask);
                        z_fp += dz_fp;
                        bool zpass = true;
                        if (zte) {
                            const u32 cur_z = zb[idx];
                            if ((ztst == 2u && pz < cur_z) || (ztst == 3u && pz <= cur_z)) zpass = false;
                        }
                        if (zpass) {
                            const u32 vcol = iip
                                ? (clamp_fp16_u8(r_fp) | (clamp_fp16_u8(g_fp) << 8) |
                                   (clamp_fp16_u8(b_fp) << 16) | (clamp_fp16_u8(a_fp) << 24))
                                : flat_vcol;
                            const u32 col = apply_tfx_px(sample_uv_fp(cu_fp, cv_fp), vcol);
                            if (!eff_ate || alpha_test_pass((col >> 24) & 0xFFu, atst, aref)) {
                                if (zte && zb_en) zb[idx] = pz;
                                store_fb_px(idx, do_blend ? blend_one(col, fb[idx]) : col);
                                local_px++;
                            }
                        }
                        cu_fp += cdu_fp;
                        cv_fp += cdv_fp;
                        if (iip) { r_fp += dr_fp; g_fp += dg_fp; b_fp += db_fp; a_fp += da_fp; }
                        idx++;
                        rem--;
                    }
                } else {
                    // Steep perspective / single-pixel exact 1/q fallback
                    for (s32 k = 0; k < chunk; ++k) {
                        const std::size_t idx = row_base + static_cast<std::size_t>(x + k);
                        const u32 pz = clamp_z_fp(z_fp, z_mask);
                        z_fp += dz_fp;
                        bool zpass = true;
                        if (zte) {
                            const u32 cur_z = zb[idx];
                            if ((ztst == 2u && pz < cur_z) || (ztst == 3u && pz <= cur_z)) zpass = false;
                        }
                        if (zpass) {
                            const f32 inv_q = (std::fabs(q_cur) > 1e-6f) ? (1.0f / q_cur) : 1.0f;
                            const s64 ufp = fast_f32_to_fp8(s_cur * inv_q * tw_f);
                            const s64 vfp = fast_f32_to_fp8(t_cur * inv_q * th_f);
                            const u32 vcol = iip
                                ? (clamp_fp16_u8(r_fp) | (clamp_fp16_u8(g_fp) << 8) |
                                   (clamp_fp16_u8(b_fp) << 16) | (clamp_fp16_u8(a_fp) << 24))
                                : flat_vcol;
                            const u32 col = apply_tfx_px(sample_uv_fp(ufp, vfp), vcol);
                            if (!eff_ate || alpha_test_pass((col >> 24) & 0xFFu, atst, aref)) {
                                if (zte && zb_en) zb[idx] = pz;
                                store_fb_px(idx, do_blend ? blend_one(col, fb[idx]) : col);
                                local_px++;
                            }
                        }
                        s_cur += ds_dx; t_cur += dt_dx; q_cur += dq_dx;
                        if (iip) { r_fp += dr_fp; g_fp += dg_fp; b_fp += db_fp; a_fp += da_fp; }
                    }
                    if (x + chunk <= x_end) {
                        const f32 inv_q = (std::fabs(q_cur) > 1e-6f) ? (1.0f / q_cur) : 1.0f;
                        tu_cur = std::clamp(s_cur * inv_q * tw_f, -65536.0f, 65536.0f);
                        tv_cur = std::clamp(t_cur * inv_q * th_f, -65536.0f, 65536.0f);
                    }
                }
                x += chunk;
            }
            continue;
        }

        if (fast_flat_simd) {
            const s32 span_len = x_end - x_start + 1;
            std::size_t idx = row_base + static_cast<std::size_t>(x_start);
            if (!iip && eff_ate && !alpha_test_pass(v2.a, atst, aref)) {
                continue;
            }
            if (!iip && !abe && (!zte || (ztst == 1u && !zb_en)) && fbmsk == 0u) {
                std::fill_n(fb + idx, span_len, flat_vcol | fba_or);
                local_px += static_cast<u64>(span_len);
                continue;
            }
            s32 rem = span_len;
            while (rem >= 4) {
                const u32 pz4[4] = {
                    clamp_z_fp(z_fp, z_mask),
                    clamp_z_fp(z_fp + dz_fp, z_mask),
                    clamp_z_fp(z_fp + dz_fp * 2, z_mask),
                    clamp_z_fp(z_fp + dz_fp * 3, z_mask)
                };
                z_fp += dz_fp * 4;
                const u32 zmask = zte ? z_pass_mask_4px(pz4, zb + idx, ztst) : 0xFu;
                if (zmask == 0u) {
                    if (iip) { r_fp += dr_fp * 4; g_fp += dg_fp * 4; b_fp += db_fp * 4; a_fp += da_fp * 4; }
                    idx += 4; rem -= 4;
                    continue;
                }
                if (zmask == 0xFu && !eff_ate) {
                    u32 col4[4];
                    if (iip) {
                        for (int k = 0; k < 4; ++k) {
                            col4[k] = clamp_fp16_u8(r_fp) | (clamp_fp16_u8(g_fp) << 8) |
                                      (clamp_fp16_u8(b_fp) << 16) | (clamp_fp16_u8(a_fp) << 24);
                            r_fp += dr_fp; g_fp += dg_fp; b_fp += db_fp; a_fp += da_fp;
                        }
                    } else {
                        col4[0] = flat_vcol; col4[1] = flat_vcol; col4[2] = flat_vcol; col4[3] = flat_vcol;
                    }
                    if (zte && zb_en) {
                        zb[idx] = pz4[0]; zb[idx + 1] = pz4[1]; zb[idx + 2] = pz4[2]; zb[idx + 3] = pz4[3];
                    }
                    if (!abe) {
                        if (fbmsk == 0u && !c.fba) {
                            fb[idx] = col4[0]; fb[idx + 1] = col4[1]; fb[idx + 2] = col4[2]; fb[idx + 3] = col4[3];
                        } else {
                            for (int k = 0; k < 4; ++k) store_fb_px(idx + k, col4[k]);
                        }
                    } else if (std_alpha_blend || add_alpha_blend) {
                        const u32 d0 = fb[idx], d1 = fb[idx + 1], d2 = fb[idx + 2], d3 = fb[idx + 3];
                        const s32 c0 = (spec_c == 0u) ? static_cast<s32>((col4[0] >> 24) & 0xFFu) : (spec_c == 1u) ? static_cast<s32>((d0 >> 24) & 0xFFu) : (spec_c == 2u) ? fix : 0;
                        const s32 c1 = (spec_c == 0u) ? static_cast<s32>((col4[1] >> 24) & 0xFFu) : (spec_c == 1u) ? static_cast<s32>((d1 >> 24) & 0xFFu) : (spec_c == 2u) ? fix : 0;
                        const s32 c2 = (spec_c == 0u) ? static_cast<s32>((col4[2] >> 24) & 0xFFu) : (spec_c == 1u) ? static_cast<s32>((d2 >> 24) & 0xFFu) : (spec_c == 2u) ? fix : 0;
                        const s32 c3 = (spec_c == 0u) ? static_cast<s32>((col4[3] >> 24) & 0xFFu) : (spec_c == 1u) ? static_cast<s32>((d3 >> 24) & 0xFFu) : (spec_c == 2u) ? fix : 0;
                        u32 o0, o1, o2, o3;
                        if (std_alpha_blend) {
                            blend_std_2px(col4[0], col4[1], d0, d1, c0, c1, o0, o1);
                            blend_std_2px(col4[2], col4[3], d2, d3, c2, c3, o2, o3);
                        } else {
                            blend_add_2px(col4[0], col4[1], d0, d1, c0, c1, o0, o1);
                            blend_add_2px(col4[2], col4[3], d2, d3, c2, c3, o2, o3);
                        }
                        if (fbmsk == 0u && !c.fba) {
                            fb[idx] = o0; fb[idx + 1] = o1; fb[idx + 2] = o2; fb[idx + 3] = o3;
                        } else {
                            fb[idx]     = (((o0 | fba_or) & inv_fbmsk) | (d0 & fbmsk));
                            fb[idx + 1] = (((o1 | fba_or) & inv_fbmsk) | (d1 & fbmsk));
                            fb[idx + 2] = (((o2 | fba_or) & inv_fbmsk) | (d2 & fbmsk));
                            fb[idx + 3] = (((o3 | fba_or) & inv_fbmsk) | (d3 & fbmsk));
                        }
                    } else {
                        for (int k = 0; k < 4; ++k) store_fb_px(idx + k, blend_one(col4[k], fb[idx + k]));
                    }
                    local_px += 4u;
                } else {
                    for (int k = 0; k < 4; ++k) {
                        if (zmask & (1u << k)) {
                            const u32 col = iip
                                ? (clamp_fp16_u8(r_fp) | (clamp_fp16_u8(g_fp) << 8) |
                                   (clamp_fp16_u8(b_fp) << 16) | (clamp_fp16_u8(a_fp) << 24))
                                : flat_vcol;
                            if (!eff_ate || alpha_test_pass((col >> 24) & 0xFFu, atst, aref)) {
                                if (zte && zb_en) zb[idx + k] = pz4[k];
                                store_fb_px(idx + k, abe ? blend_one(col, fb[idx + k]) : col);
                                local_px++;
                            }
                        }
                        if (iip) { r_fp += dr_fp; g_fp += dg_fp; b_fp += db_fp; a_fp += da_fp; }
                    }
                }
                idx += 4; rem -= 4;
            }
            while (rem > 0) {
                const u32 pz = clamp_z_fp(z_fp, z_mask);
                z_fp += dz_fp;
                bool zpass = true;
                if (zte) {
                    const u32 cur_z = zb[idx];
                    if ((ztst == 2u && pz < cur_z) || (ztst == 3u && pz <= cur_z)) zpass = false;
                }
                if (zpass) {
                    const u32 col = iip
                        ? (clamp_fp16_u8(r_fp) | (clamp_fp16_u8(g_fp) << 8) |
                           (clamp_fp16_u8(b_fp) << 16) | (clamp_fp16_u8(a_fp) << 24))
                        : flat_vcol;
                    if (!eff_ate || alpha_test_pass((col >> 24) & 0xFFu, atst, aref)) {
                        if (zte && zb_en) zb[idx] = pz;
                        store_fb_px(idx, abe ? blend_one(col, fb[idx]) : col);
                        local_px++;
                    }
                }
                if (iip) { r_fp += dr_fp; g_fp += dg_fp; b_fp += db_fp; a_fp += da_fp; }
                idx++; rem--;
            }
            continue;
        }

        // General Path (DATE, PABE, AFAIL!=0, mirror_pres, Z-clear)
        s64 u_fp_cur = (tme && affine_st) ? fast_f32_to_fp8(w0 * au0 + w1 * au1 + w2 * au2) : 0;
        s64 v_fp_cur = (tme && affine_st) ? fast_f32_to_fp8(w0 * av0 + w1 * av1 + w2 * av2) : 0;
        f32 s_cur = (tme && !affine_st) ? (w0 * s0 + w1 * s1 + w2 * s2) : 0.0f;
        f32 t_cur = (tme && !affine_st) ? (w0 * t0 + w1 * t1 + w2 * t2) : 0.0f;
        f32 q_cur = (tme && !affine_st) ? (w0 * q0 + w1 * q1 + w2 * q2) : 1.0f;

        const bool row_in_pres = mirror_pres && pres_fb != nullptr && (static_cast<u32>(y) < m_fb_height);
        const std::size_t pres_row = row_in_pres ? (static_cast<std::size_t>(y) * m_fb_width) : 0u;

        for (s32 x = x_start; x <= x_end; ++x) {
            const std::size_t idx = row_base + static_cast<std::size_t>(x);
            const u32 pz = clamp_z_fp(z_fp, z_mask);

            auto step_all = [&]() {
                z_fp += dz_fp;
                if (iip) { r_fp += dr_fp; g_fp += dg_fp; b_fp += db_fp; a_fp += da_fp; }
                if (tme) {
                    if (affine_st) { u_fp_cur += du_fp_aff; v_fp_cur += dv_fp_aff; }
                    else           { s_cur += ds_dx; t_cur += dt_dx; q_cur += dq_dx; }
                }
            };

            if (early_z_gen) {
                const u32 cur_z = zb[idx];
                if ((ztst == 2u && pz < cur_z) || (ztst == 3u && pz <= cur_z)) { step_all(); continue; }
            }

            u32 dst = 0u;
            bool dst_loaded = false;
            if (date) {
                dst = fb[idx];
                dst_loaded = true;
                if (((dst >> 31) & 1u) != datm) { step_all(); continue; }
            }

            u32 col = iip
                ? (clamp_fp16_u8(r_fp) | (clamp_fp16_u8(g_fp) << 8) |
                   (clamp_fp16_u8(b_fp) << 16) | (clamp_fp16_u8(a_fp) << 24))
                : flat_vcol;

            if (tme) {
                s64 ufp, vfp;
                if (affine_st) {
                    ufp = u_fp_cur; vfp = v_fp_cur;
                } else {
                    const f32 inv_q = (std::fabs(q_cur) > 1e-6f) ? (1.0f / q_cur) : 1.0f;
                    ufp = fast_f32_to_fp8(s_cur * inv_q * tw_f);
                    vfp = fast_f32_to_fp8(t_cur * inv_q * th_f);
                }
                col = apply_tfx_px(sample_uv_fp(ufp, vfp), col);
            }

            const u32 a = (col >> 24) & 0xFFu;
            bool write_fb = true;
            bool write_alpha = true;
            bool write_zb = zb_en;

            if (eff_ate) {
                if (!alpha_test_pass(a, atst, aref)) {
                    if (afail == 0u) { step_all(); continue; }
                    if (afail == 1u) write_zb = false;
                    else if (afail == 2u) write_fb = false;
                    else { write_zb = false; write_alpha = false; }
                }
            }

            if (zte) {
                if (!early_z_gen) {
                    const u32 cur_z = zb[idx];
                    if ((ztst == 2u && pz < cur_z) || (ztst == 3u && pz <= cur_z)) { step_all(); continue; }
                }
                if (write_zb) zb[idx] = pz;
            }
            if (!write_fb) { step_all(); continue; }

            const bool do_blend = abe && (!m_pabe || (a & 0x80u) != 0u);
            if (!dst_loaded && (do_blend || fbmsk != 0u || !write_alpha)) {
                dst = fb[idx];
            }

            u32 new_px = do_blend ? blend_one(col, dst) : col;
            u32 out_a = write_alpha ? a : ((dst >> 24) & 0xFFu);
            if (c.fba && write_alpha) out_a |= 0x80u;
            new_px = (new_px & 0x00FFFFFFu) | (out_a << 24);
            const u32 out_px = (fbmsk == 0u) ? new_px : ((new_px & ~fbmsk) | (dst & fbmsk));
            fb[idx] = out_px;
            if (clear_zt) clear_zt->z[idx] = out_px & clear_z_mask;
            if (row_in_pres && static_cast<u32>(x) < m_fb_width) {
                pres_fb[pres_row + static_cast<std::size_t>(x)] = (out_px & 0x00FFFFFFu) | 0xFF000000u;
            }
            local_px++;
            step_all();
        }
    }

    out_px += local_px;
    if (perf_stats) {
        if (fast_tex_simd) {
            perf_stats->fast_tex_px += local_px;
            perf_stats->fast_tex_cnt++;
        } else if (fast_flat_simd) {
            perf_stats->fast_flat_px += local_px;
            perf_stats->fast_flat_cnt++;
        } else {
            perf_stats->general_px += local_px;
            perf_stats->general_cnt++;
            if (date) perf_stats->gen_date_cnt++;
            if (abe && m_pabe) perf_stats->gen_pabe_cnt++;
            if (eff_ate && afail != 0u) perf_stats->gen_afail_cnt++;
        }
    }
}

void GsState::rasterize_triangle(const GsVertex& v0, const GsVertex& v1, const GsVertex& v2) {
    if (!m_tri_batch.empty()) flush_tri_batch();
    PerfScope perf_tri(PerfCat::GsTri);
    const GsContext& c = active_ctx();

    const s32 min_x4 = std::min({v0.x4, v1.x4, v2.x4});
    const s32 max_x4 = std::max({v0.x4, v1.x4, v2.x4});
    const s32 min_y4 = std::min({v0.y4, v1.y4, v2.y4});
    const s32 max_y4 = std::max({v0.y4, v1.y4, v2.y4});

    const s32 pre_min_x = std::max<s32>(0, std::max(c.scissor_x0, (min_x4 + 15) >> 4));
    const s32 pre_max_x = std::min<s32>(c.scissor_x1, max_x4 >> 4);
    const s32 pre_min_y = std::max<s32>(0, std::max(c.scissor_y0, (min_y4 + 15) >> 4));
    const s32 pre_max_y = std::min<s32>(c.scissor_y1, max_y4 >> 4);
    if (pre_min_x > pre_max_x || pre_min_y > pre_max_y) {
        primitives_drawn++;
        return;
    }

    const s64 area = static_cast<s64>(v2.x4 - v0.x4) * static_cast<s64>(v1.y4 - v0.y4) -
                     static_cast<s64>(v2.y4 - v0.y4) * static_cast<s64>(v1.x4 - v0.x4);
    if (area == 0) {
        primitives_drawn++;
        return;
    }

    const bool zte = ((c.test_reg >> 16) & 1u) != 0u;
    const u32 ztst = static_cast<u32>((c.test_reg >> 17) & 3u);
    const bool ate = (c.test_reg & 1u) != 0u;
    const u32 atst = static_cast<u32>((c.test_reg >> 1) & 7u);
    const u32 afail = static_cast<u32>((c.test_reg >> 12) & 3u);
    if ((zte && ztst == 0u) || (ate && atst == 0u && afail == 0u)) {
        primitives_drawn++;
        return;
    }

    const u32 fbw = static_cast<u32>((c.frame_reg >> 16) & 0x3Fu);
    const u32 fbpsm = static_cast<u32>((c.frame_reg >> 24) & 0x3Fu);
    const u32 min_h = static_cast<u32>(std::max<s32>(0, c.scissor_y1 + 1));
    const u32 attrs = active_prim_attrs();
    const bool tme = (attrs & (1u << 4)) != 0u;

    if (m_hw != nullptr && !(m_dispfb1 == 0u && m_dispfb2 == 0u)) {
        const u32 zbp0 = static_cast<u32>(m_ctx[0].zbuf_reg & 0x1FFu);
        const u32 zbp1 = static_cast<u32>(m_ctx[1].zbuf_reg & 0x1FFu);
        const bool target_is_zbuf = (c.active_draw_fbp != 0u) &&
                                    (c.active_draw_fbp == zbp0 || c.active_draw_fbp == zbp1);
        const u32 eff_fbw = (fbw == 0u) ? ((m_fb_width + 63u) / 64u) : fbw;
        const u32 eff_psm = (fbpsm == 0x00u || fbpsm == 0x01u || fbpsm == 0x02u || fbpsm == 0x0Au) ? fbpsm : 0x00u;
        const u32 ph = psm_page_height(eff_psm);
        const u32 req_h = std::clamp<u32>(((std::max<u32>(ph, min_h) + ph - 1u) / ph) * ph, ph, 1024u);

        if (target_is_zbuf) {
            const u32 out_a = c.fba ? (static_cast<u32>(v2.a) | 0x80u) : static_cast<u32>(v2.a);
            const u32 fill_px = static_cast<u32>(v2.r) | (static_cast<u32>(v2.g) << 8) |
                                (static_cast<u32>(v2.b) << 16) | (out_a << 24);
            const u64 zreg = (c.active_draw_fbp == zbp0) ? m_ctx[0].zbuf_reg : m_ctx[1].zbuf_reg;
            const u32 clear_z_mask = zbuf_psm_mask(zreg);
            const f32 z_norm = static_cast<f32>(static_cast<double>(fill_px & clear_z_mask) /
                                                static_cast<double>(clear_z_mask));
            m_hw->clear_depth(c.active_draw_fbp, eff_fbw * 64u, req_h, z_norm,
                              pre_min_x, pre_min_y, pre_max_x, pre_max_y);
            primitives_drawn++;
            return;
        }

        const GsFbTarget* rt_src = tme ? prepare_texture_source(c.tex0_reg) : nullptr;
        LinearTexView tex_view{};
        if (tme) {
            textured_primitives++;
            tex_view = resolve_linear_texture(c.tex0_reg, rt_src);
        }

        const bool mmag_lin = ((c.tex1_reg >> 5) & 1u) != 0u;
        const u32 mmin = static_cast<u32>((c.tex1_reg >> 6) & 7u);
        const bool mmin_lin = (mmin == 1u || mmin == 4u || mmin == 5u);
        const bool use_bilinear = tme && (mmag_lin || mmin_lin);

        GsHwDrawDesc desc{};
        populate_hw_draw_desc(c, attrs, eff_fbw, eff_psm, req_h, use_bilinear, tex_view, rt_src, false, desc);

        const u32 z_mask = zbuf_psm_mask(c.zbuf_reg);
        const double inv_z_mask = 1.0 / static_cast<double>(z_mask);
        const bool iip = (attrs & (1u << 3)) != 0u;
        const bool fst = (attrs & (1u << 8)) != 0u;
        const u32 tw = 1u << std::min<u32>(10u, static_cast<u32>((c.tex0_reg >> 26) & 0xFu));
        const u32 th = 1u << std::min<u32>(10u, static_cast<u32>((c.tex0_reg >> 30) & 0xFu));
        const f32 inv_tw = 1.0f / static_cast<f32>(tw);
        const f32 inv_th = 1.0f / static_cast<f32>(th);
        const u32 flat_rgba = static_cast<u32>(v2.r) | (static_cast<u32>(v2.g) << 8) |
                              (static_cast<u32>(v2.b) << 16) | (static_cast<u32>(v2.a) << 24);
        auto make_v = [&](const GsVertex& v) -> GsHwVertex {
            GsHwVertex hv{};
            hv.x = static_cast<f32>(v.x4) * (1.0f / 16.0f);
            hv.y = static_cast<f32>(v.y4) * (1.0f / 16.0f);
            hv.z = static_cast<f32>(static_cast<double>(v.z & z_mask) * inv_z_mask);
            hv.rgba = iip ? (static_cast<u32>(v.r) | (static_cast<u32>(v.g) << 8) |
                             (static_cast<u32>(v.b) << 16) | (static_cast<u32>(v.a) << 24))
                          : flat_rgba;
            if (fst) {
                hv.s = v.u * inv_tw;
                hv.t = v.v * inv_th;
                hv.q = 1.0f;
            } else {
                hv.s = sanitize_f32(v.s);
                hv.t = sanitize_f32(v.t);
                hv.q = (std::isfinite(v.q) && std::fabs(v.q) > 1e-6f) ? v.q : 1.0f;
            }
            return hv;
        };
        const GsHwVertex verts[3] = { make_v(v0), make_v(v1), make_v(v2) };
        m_hw->draw_triangles(desc, verts, 3u);
        if (g_perf.enabled) g_perf.tri_single_draws.fetch_add(1, std::memory_order_relaxed);
        primitives_drawn++;
        return;
    }

    const GsFbTarget* rt_src = tme ? prepare_texture_source(c.tex0_reg) : nullptr;
    GsFbTarget& rt = *ensure_fb_target(c.active_draw_fbp, fbw, fbpsm, min_h);
    const u32 stride = rt.fbw * 64u;
    if (tme && rt_src != nullptr && !rt_src->valid) {
        rt_src = prepare_texture_source(c.tex0_reg);
    }

    const s32 min_y = pre_min_y;
    const s32 max_y = std::min<s32>(static_cast<s32>(rt.height) - 1, pre_max_y);
    if (pre_min_x > static_cast<s32>(stride) - 1 || min_y > max_y) {
        primitives_drawn++;
        return;
    }

    LinearTexView tex_view{};
    if (tme) {
        textured_primitives++;
        tex_view = resolve_linear_texture(c.tex0_reg, rt_src);
        if (rt_src == &rt && tex_view.pixels != nullptr) {
            if (m_self_blit_scratch.size() < rt.pixels.size()) {
                m_self_blit_scratch.resize(rt.pixels.size());
            }
            std::memcpy(m_self_blit_scratch.data(), rt.pixels.data(), rt.pixels.size() * sizeof(u32));
            tex_view.pixels = m_self_blit_scratch.data();
        }
    }

    const u32 zbp = static_cast<u32>(c.zbuf_reg & 0x1FFu);
    const u32 z_mask = zbuf_psm_mask(c.zbuf_reg);
    GsZTarget* zt = ensure_z_target(zbp, stride, rt.height);

    const u32 zbp0 = static_cast<u32>(m_ctx[0].zbuf_reg & 0x1FFu);
    const u32 zbp1 = static_cast<u32>(m_ctx[1].zbuf_reg & 0x1FFu);
    const bool target_is_zbuf = (c.active_draw_fbp != 0u) &&
                                (c.active_draw_fbp == zbp0 || c.active_draw_fbp == zbp1);
    GsZTarget* clear_zt = nullptr;
    u32 clear_z_mask = 0xFFFFFFFFu;
    if (target_is_zbuf) {
        const u64 zreg = (c.active_draw_fbp == zbp0) ? m_ctx[0].zbuf_reg : m_ctx[1].zbuf_reg;
        clear_z_mask = zbuf_psm_mask(zreg);
        clear_zt = ensure_z_target(c.active_draw_fbp, stride, rt.height);
    }

    const bool mirror_pres = (m_dispfb1 == 0u && m_dispfb2 == 0u);
    u64 local_px = 0;
    GsTriPerfStats tstats{};
    rasterize_triangle_range(c, attrs, rt, zt, z_mask, tex_view, rt_src,
                             clear_zt, clear_z_mask, mirror_pres, m_present_fb.data(),
                             v0, v1, v2, min_y, max_y + 1, local_px,
                             g_perf.enabled ? &tstats : nullptr);
    if (g_perf.enabled) {
        g_perf.tri_single_draws.fetch_add(1, std::memory_order_relaxed);
        g_perf.tri_fast_tex_px.fetch_add(tstats.fast_tex_px, std::memory_order_relaxed);
        g_perf.tri_fast_flat_px.fetch_add(tstats.fast_flat_px, std::memory_order_relaxed);
        g_perf.tri_general_px.fetch_add(tstats.general_px, std::memory_order_relaxed);
        g_perf.tri_fast_tex_cnt.fetch_add(tstats.fast_tex_cnt, std::memory_order_relaxed);
        g_perf.tri_fast_flat_cnt.fetch_add(tstats.fast_flat_cnt, std::memory_order_relaxed);
        g_perf.tri_general_cnt.fetch_add(tstats.general_cnt, std::memory_order_relaxed);
        g_perf.tri_gen_date_cnt.fetch_add(tstats.gen_date_cnt, std::memory_order_relaxed);
        g_perf.tri_gen_pabe_cnt.fetch_add(tstats.gen_pabe_cnt, std::memory_order_relaxed);
        g_perf.tri_gen_afail_cnt.fetch_add(tstats.gen_afail_cnt, std::memory_order_relaxed);
    }

    if (local_px > 0u) {
        mark_fb_dirty(c.active_draw_fbp, static_cast<u32>(min_y), static_cast<u32>(max_y + 1));
    }
    pixels_drawn += local_px;
    primitives_drawn++;
}

void GsState::enqueue_triangle(const GsVertex& v0, const GsVertex& v1, const GsVertex& v2) {
    const GsContext& c = active_ctx();
    const bool mirror_pres = (m_dispfb1 == 0u && m_dispfb2 == 0u);
    const u32 zbp0 = static_cast<u32>(m_ctx[0].zbuf_reg & 0x1FFu);
    const u32 zbp1 = static_cast<u32>(m_ctx[1].zbuf_reg & 0x1FFu);
    const bool target_is_zbuf = (c.active_draw_fbp != 0u) &&
                                (c.active_draw_fbp == zbp0 || c.active_draw_fbp == zbp1);

    if (mirror_pres || target_is_zbuf) {
        rasterize_triangle(v0, v1, v2);
        return;
    }

    // Early degenerate & off-scissor cull BEFORE queuing!
    const s32 min_x4 = std::min({v0.x4, v1.x4, v2.x4});
    const s32 max_x4 = std::max({v0.x4, v1.x4, v2.x4});
    const s32 min_y4 = std::min({v0.y4, v1.y4, v2.y4});
    const s32 max_y4 = std::max({v0.y4, v1.y4, v2.y4});

    const s32 pre_min_x = std::max<s32>(0, std::max(c.scissor_x0, (min_x4 + 15) >> 4));
    const s32 pre_max_x = std::min<s32>(c.scissor_x1, max_x4 >> 4);
    const s32 pre_min_y = std::max<s32>(0, std::max(c.scissor_y0, (min_y4 + 15) >> 4));
    const s32 pre_max_y = std::min<s32>(c.scissor_y1, max_y4 >> 4);
    if (pre_min_x > pre_max_x || pre_min_y > pre_max_y) {
        primitives_drawn++;
        return;
    }

    const s64 area = static_cast<s64>(v2.x4 - v0.x4) * static_cast<s64>(v1.y4 - v0.y4) -
                     static_cast<s64>(v2.y4 - v0.y4) * static_cast<s64>(v1.x4 - v0.x4);
    if (area == 0) {
        primitives_drawn++;
        return;
    }

    const bool zte = ((c.test_reg >> 16) & 1u) != 0u;
    const u32 ztst = static_cast<u32>((c.test_reg >> 17) & 3u);
    const bool ate = (c.test_reg & 1u) != 0u;
    const u32 atst = static_cast<u32>((c.test_reg >> 1) & 7u);
    const u32 afail = static_cast<u32>((c.test_reg >> 12) & 3u);
    if ((zte && ztst == 0u) || (ate && atst == 0u && afail == 0u)) {
        primitives_drawn++;
        return;
    }

    m_tri_batch.push_back(TriBatchItem{ v0, v1, v2, pre_min_x, pre_max_x, pre_min_y, pre_max_y });
    if (pre_min_y < m_tri_batch_min_y) m_tri_batch_min_y = pre_min_y;
    if (pre_max_y > m_tri_batch_max_y) m_tri_batch_max_y = pre_max_y;

    const std::size_t max_batch = (m_hw != nullptr) ? 8192u : 1024u;
    if (m_tri_batch.size() >= max_batch) {
        flush_tri_batch();
    }
}

void GsState::flush_tri_batch() {
    if (m_tri_batch.empty()) return;
    PerfScope perf_tri(PerfCat::GsTri);

    const GsContext& c = active_ctx();
    const u32 attrs = active_prim_attrs();
    const bool tme = (attrs & (1u << 4)) != 0u;
    const u32 fbw = static_cast<u32>((c.frame_reg >> 16) & 0x3Fu);
    const u32 fbpsm = static_cast<u32>((c.frame_reg >> 24) & 0x3Fu);
    const u32 min_h = static_cast<u32>(std::max<s32>(0, c.scissor_y1 + 1));

    if (m_hw != nullptr) {
        const std::size_t count = m_tri_batch.size();
        const u32 eff_fbw = (fbw == 0u) ? ((m_fb_width + 63u) / 64u) : fbw;
        const u32 eff_psm = (fbpsm == 0x00u || fbpsm == 0x01u || fbpsm == 0x02u || fbpsm == 0x0Au) ? fbpsm : 0x00u;
        const u32 ph = psm_page_height(eff_psm);
        const u32 req_h = std::clamp<u32>(((std::max<u32>(ph, min_h) + ph - 1u) / ph) * ph, ph, 1024u);

        const GsFbTarget* rt_src = tme ? prepare_texture_source(c.tex0_reg) : nullptr;
        LinearTexView tex_view{};
        if (tme) {
            textured_primitives += count;
            tex_view = resolve_linear_texture(c.tex0_reg, rt_src);
        }

        const bool mmag_lin = ((c.tex1_reg >> 5) & 1u) != 0u;
        const u32 mmin = static_cast<u32>((c.tex1_reg >> 6) & 7u);
        const bool mmin_lin = (mmin == 1u || mmin == 4u || mmin == 5u);
        const bool use_bilinear = tme && (mmag_lin || mmin_lin);

        GsHwDrawDesc desc{};
        populate_hw_draw_desc(c, attrs, eff_fbw, eff_psm, req_h, use_bilinear, tex_view, rt_src, false, desc);

        const u32 z_mask = zbuf_psm_mask(c.zbuf_reg);
        const double inv_z_mask = 1.0 / static_cast<double>(z_mask);
        const bool iip = (attrs & (1u << 3)) != 0u;
        const bool fst = (attrs & (1u << 8)) != 0u;
        const u32 tw = 1u << std::min<u32>(10u, static_cast<u32>((c.tex0_reg >> 26) & 0xFu));
        const u32 th = 1u << std::min<u32>(10u, static_cast<u32>((c.tex0_reg >> 30) & 0xFu));
        const f32 inv_tw = 1.0f / static_cast<f32>(tw);
        const f32 inv_th = 1.0f / static_cast<f32>(th);

        m_hw_verts.resize(count * 3u);
        GsHwVertex* dst = m_hw_verts.data();
        for (const auto& item : m_tri_batch) {
            const u32 flat_rgba = static_cast<u32>(item.v2.r) | (static_cast<u32>(item.v2.g) << 8) |
                                  (static_cast<u32>(item.v2.b) << 16) | (static_cast<u32>(item.v2.a) << 24);
            const GsVertex* src_v[3] = { &item.v0, &item.v1, &item.v2 };
            for (int k = 0; k < 3; ++k) {
                const GsVertex& v = *src_v[k];
                GsHwVertex& hv = *dst++;
                hv.x = static_cast<f32>(v.x4) * (1.0f / 16.0f);
                hv.y = static_cast<f32>(v.y4) * (1.0f / 16.0f);
                hv.z = static_cast<f32>(static_cast<double>(v.z & z_mask) * inv_z_mask);
                hv.rgba = iip ? (static_cast<u32>(v.r) | (static_cast<u32>(v.g) << 8) |
                                 (static_cast<u32>(v.b) << 16) | (static_cast<u32>(v.a) << 24))
                              : flat_rgba;
                if (fst) {
                    hv.s = v.u * inv_tw;
                    hv.t = v.v * inv_th;
                    hv.q = 1.0f;
                } else {
                    hv.s = sanitize_f32(v.s);
                    hv.t = sanitize_f32(v.t);
                    hv.q = (std::isfinite(v.q) && std::fabs(v.q) > 1e-6f) ? v.q : 1.0f;
                }
            }
        }

        m_hw->draw_triangles(desc, m_hw_verts.data(), static_cast<u32>(m_hw_verts.size()));
        if (g_perf.enabled) {
            g_perf.tri_batches.fetch_add(1, std::memory_order_relaxed);
            g_perf.tri_batch_tris.fetch_add(count, std::memory_order_relaxed);
        }
        primitives_drawn += count;
        m_tri_batch.clear();
        m_tri_batch_min_y = 0x7FFFFFFF;
        m_tri_batch_max_y = -1;
        return;
    }

    const GsFbTarget* rt_src = tme ? prepare_texture_source(c.tex0_reg) : nullptr;
    GsFbTarget& rt = *ensure_fb_target(c.active_draw_fbp, fbw, fbpsm, min_h);
    const u32 stride = rt.fbw * 64u;
    if (tme && rt_src != nullptr && !rt_src->valid) {
        rt_src = prepare_texture_source(c.tex0_reg);
    }

    const s32 batch_y0 = std::max<s32>(0, m_tri_batch_min_y);
    const s32 batch_y1 = std::min<s32>(static_cast<s32>(rt.height), m_tri_batch_max_y + 1);
    const std::size_t count = m_tri_batch.size();

    if (batch_y0 >= batch_y1) {
        primitives_drawn += count;
        m_tri_batch.clear();
        m_tri_batch_min_y = 0x7FFFFFFF;
        m_tri_batch_max_y = -1;
        return;
    }

    LinearTexView tex_view{};
    if (tme) {
        textured_primitives += count;
        tex_view = resolve_linear_texture(c.tex0_reg, rt_src);
    }

    const u32 zbp = static_cast<u32>(c.zbuf_reg & 0x1FFu);
    const u32 z_mask = zbuf_psm_mask(c.zbuf_reg);
    GsZTarget* zt = ensure_z_target(zbp, stride, rt.height);

    u64 total_px = 0;
    GsTriPerfStats batch_stats{};
    const bool perf_on = g_perf.enabled;
    if (perf_on) {
        g_perf.tri_batches.fetch_add(1, std::memory_order_relaxed);
        g_perf.tri_batch_tris.fetch_add(count, std::memory_order_relaxed);
    }

    if (tme && rt_src == &rt && tex_view.pixels != nullptr) {
        if (perf_on) g_perf.tri_single_draws.fetch_add(1, std::memory_order_relaxed);
        // Self-sampling render target: snapshot sequentially per triangle
        if (m_self_blit_scratch.size() < rt.pixels.size()) {
            m_self_blit_scratch.resize(rt.pixels.size());
        }
        tex_view.pixels = m_self_blit_scratch.data();
        for (const auto& item : m_tri_batch) {
            std::memcpy(m_self_blit_scratch.data(), rt.pixels.data(), rt.pixels.size() * sizeof(u32));
            rasterize_triangle_range(c, attrs, rt, zt, z_mask, tex_view, rt_src,
                                     nullptr, 0xFFFFFFFFu, false, nullptr,
                                     item.v0, item.v1, item.v2, batch_y0, batch_y1, total_px,
                                     perf_on ? &batch_stats : nullptr);
        }
    } else {
        bool run_parallel = false;
        const s32 batch_h = batch_y1 - batch_y0;
        if (count >= 24u && batch_h >= 96) {
            u64 est_area = 0;
            u32 band_mask = 0u;
            const s32 b_cut1 = batch_y0 + batch_h / 4;
            const s32 b_cut2 = batch_y0 + (batch_h * 2) / 4;
            const s32 b_cut3 = batch_y0 + (batch_h * 3) / 4;
            for (const auto& item : m_tri_batch) {
                const s32 w = std::max<s32>(0, item.max_x - item.min_x + 1);
                const s32 h = std::max<s32>(0, item.max_y - item.min_y + 1);
                est_area += static_cast<u64>(w) * static_cast<u64>(h);
                if (item.min_y < b_cut1) band_mask |= 1u;
                if (item.max_y >= b_cut1 && item.min_y < b_cut2) band_mask |= 2u;
                if (item.max_y >= b_cut2 && item.min_y < b_cut3) band_mask |= 4u;
                if (item.max_y >= b_cut3) band_mask |= 8u;
            }
            const u32 active_bands = ((band_mask >> 0) & 1u) + ((band_mask >> 1) & 1u) +
                                     ((band_mask >> 2) & 1u) + ((band_mask >> 3) & 1u);
            run_parallel = (active_bands >= 2u) && (count >= 40u || est_area >= 24576u);
        }
        if (run_parallel) {
            if (perf_on) g_perf.tri_parallel_dispatches.fetch_add(1, std::memory_order_relaxed);
            // Lock-free dynamic multi-band (up to 8 bands) horizontal parallel rasterization (exact submission order per row!)
            struct TriBandCtx {
                const GsState* self;
                const GsContext* ctx;
                u32 attrs;
                GsFbTarget* rt;
                GsZTarget* zt;
                u32 z_mask;
                LinearTexView tex_view;
                const GsFbTarget* rt_src;
                const TriBatchItem* items;
                std::size_t count;
                s32 y0;
                s32 y1;
                bool perf_on;
                u64 band_px[kGsMaxBands]{};
                GsTriPerfStats band_stats[kGsMaxBands]{};
            } tctx{ this, &c, attrs, &rt, zt, z_mask, tex_view, rt_src, m_tri_batch.data(), count, batch_y0, batch_y1, perf_on, {}, {} };

            dispatch_bands([](void* p, int band, int num_bands) {
                auto* bctx = static_cast<TriBandCtx*>(p);
                const s32 total_h = bctx->y1 - bctx->y0;
                const s32 by0 = bctx->y0 + (total_h * band) / num_bands;
                const s32 by1 = bctx->y0 + (total_h * (band + 1)) / num_bands;
                if (by0 >= by1) return;
                u64 px = 0;
                GsTriPerfStats* ps = bctx->perf_on ? &bctx->band_stats[band] : nullptr;
                for (std::size_t i = 0; i < bctx->count; ++i) {
                    const TriBatchItem& item = bctx->items[i];
                    if (item.max_y < by0 || item.min_y >= by1) continue;
                    bctx->self->rasterize_triangle_range(
                        *bctx->ctx, bctx->attrs, *bctx->rt, bctx->zt, bctx->z_mask,
                        bctx->tex_view, bctx->rt_src, nullptr, 0xFFFFFFFFu, false, nullptr,
                        item.v0, item.v1, item.v2, by0, by1, px, ps);
                }
                bctx->band_px[band] = px;
            }, &tctx);

            for (int b = 0; b < m_num_bands; ++b) {
                total_px += tctx.band_px[b];
                if (perf_on) {
                    batch_stats.fast_tex_px += tctx.band_stats[b].fast_tex_px;
                    batch_stats.fast_flat_px += tctx.band_stats[b].fast_flat_px;
                    batch_stats.general_px += tctx.band_stats[b].general_px;
                    batch_stats.fast_tex_cnt += tctx.band_stats[b].fast_tex_cnt;
                    batch_stats.fast_flat_cnt += tctx.band_stats[b].fast_flat_cnt;
                    batch_stats.general_cnt += tctx.band_stats[b].general_cnt;
                    batch_stats.gen_date_cnt += tctx.band_stats[b].gen_date_cnt;
                    batch_stats.gen_pabe_cnt += tctx.band_stats[b].gen_pabe_cnt;
                    batch_stats.gen_afail_cnt += tctx.band_stats[b].gen_afail_cnt;
                }
            }
        } else {
            if (perf_on) g_perf.tri_single_draws.fetch_add(1, std::memory_order_relaxed);
            for (const auto& item : m_tri_batch) {
                rasterize_triangle_range(c, attrs, rt, zt, z_mask, tex_view, rt_src,
                                         nullptr, 0xFFFFFFFFu, false, nullptr,
                                         item.v0, item.v1, item.v2, batch_y0, batch_y1, total_px,
                                         perf_on ? &batch_stats : nullptr);
            }
        }
    }
    if (perf_on) {
        g_perf.tri_fast_tex_px.fetch_add(batch_stats.fast_tex_px, std::memory_order_relaxed);
        g_perf.tri_fast_flat_px.fetch_add(batch_stats.fast_flat_px, std::memory_order_relaxed);
        g_perf.tri_general_px.fetch_add(batch_stats.general_px, std::memory_order_relaxed);
        g_perf.tri_fast_tex_cnt.fetch_add(batch_stats.fast_tex_cnt, std::memory_order_relaxed);
        g_perf.tri_fast_flat_cnt.fetch_add(batch_stats.fast_flat_cnt, std::memory_order_relaxed);
        g_perf.tri_general_cnt.fetch_add(batch_stats.general_cnt, std::memory_order_relaxed);
        g_perf.tri_gen_date_cnt.fetch_add(batch_stats.gen_date_cnt, std::memory_order_relaxed);
        g_perf.tri_gen_pabe_cnt.fetch_add(batch_stats.gen_pabe_cnt, std::memory_order_relaxed);
        g_perf.tri_gen_afail_cnt.fetch_add(batch_stats.gen_afail_cnt, std::memory_order_relaxed);
    }

    if (total_px > 0u) {
        mark_fb_dirty(c.active_draw_fbp, static_cast<u32>(batch_y0), static_cast<u32>(batch_y1));
    }
    pixels_drawn += total_px;
    primitives_drawn += count;
    m_tri_batch.clear();
    m_tri_batch_min_y = 0x7FFFFFFF;
    m_tri_batch_max_y = -1;
}

void GsState::kick_vertex(u64 xyz_val, bool drawing_kick) {
    const GsContext& c = active_ctx();
    GsVertex v;
    const s32 raw_x = static_cast<s32>(xyz_val & 0xFFFFu);
    const s32 raw_y = static_cast<s32>((xyz_val >> 16) & 0xFFFFu);
    v.x4 = raw_x - c.xyoffset_x;
    v.y4 = raw_y - c.xyoffset_y;
    v.x = v.x4 >> 4;
    v.y = v.y4 >> 4;
    v.z = static_cast<u32>((xyz_val >> 32) & 0xFFFFFFFFu);
    v.r = m_cur_r; v.g = m_cur_g; v.b = m_cur_b; v.a = m_cur_a;
    v.s = m_cur_s; v.t = m_cur_t; v.q = m_cur_q;
    v.u = m_cur_u; v.v = m_cur_v;

    switch (m_prim_type) {
        case 0u: // Point
            if (drawing_kick) {
                if (!m_tri_batch.empty()) flush_tri_batch();
                const u32 fbw = static_cast<u32>((c.frame_reg >> 16) & 0x3Fu);
                const u32 fbpsm = static_cast<u32>((c.frame_reg >> 24) & 0x3Fu);
                const u32 min_h = static_cast<u32>(std::max<s32>(0, c.scissor_y1 + 1));
                if (m_hw != nullptr && !(m_dispfb1 == 0u && m_dispfb2 == 0u)) {
                    const u32 eff_fbw = (fbw == 0u) ? ((m_fb_width + 63u) / 64u) : fbw;
                    const u32 eff_psm = (fbpsm == 0x00u || fbpsm == 0x01u || fbpsm == 0x02u || fbpsm == 0x0Au) ? fbpsm : 0x00u;
                    const u32 ph = psm_page_height(eff_psm);
                    const u32 req_h = std::clamp<u32>(((std::max<u32>(ph, min_h) + ph - 1u) / ph) * ph, ph, 1024u);
                    GsHwDrawDesc desc{};
                    LinearTexView empty_tex{};
                    populate_hw_draw_desc(c, active_prim_attrs() & ~(1u << 4), eff_fbw, eff_psm, req_h,
                                          false, empty_tex, nullptr, false, desc);
                    const u32 z_mask = zbuf_psm_mask(c.zbuf_reg);
                    const f32 z_norm = static_cast<f32>(static_cast<double>(v.z & z_mask) / static_cast<double>(z_mask));
                    const u32 rgba = static_cast<u32>(v.r) | (static_cast<u32>(v.g) << 8) |
                                     (static_cast<u32>(v.b) << 16) | (static_cast<u32>(v.a) << 24);
                    const f32 px = static_cast<f32>(v.x);
                    const f32 py = static_cast<f32>(v.y);
                    const GsHwVertex quad[6] = {
                        { px,        py,        z_norm, rgba, 0.0f, 0.0f, 1.0f },
                        { px + 1.0f, py,        z_norm, rgba, 0.0f, 0.0f, 1.0f },
                        { px,        py + 1.0f, z_norm, rgba, 0.0f, 0.0f, 1.0f },
                        { px + 1.0f, py,        z_norm, rgba, 0.0f, 0.0f, 1.0f },
                        { px + 1.0f, py + 1.0f, z_norm, rgba, 0.0f, 0.0f, 1.0f },
                        { px,        py + 1.0f, z_norm, rgba, 0.0f, 0.0f, 1.0f }
                    };
                    m_hw->draw_triangles(desc, quad, 6u);
                    pixels_drawn++;
                    primitives_drawn++;
                } else {
                    GsFbTarget& rt = *ensure_fb_target(c.active_draw_fbp, fbw, fbpsm, min_h);
                    const u32 zbp = static_cast<u32>(c.zbuf_reg & 0x1FFu);
                    GsZTarget* zt = ensure_z_target(zbp, rt.fbw * 64u, rt.height);
                    plot_pixel(c, rt, zt, zbuf_psm_mask(c.zbuf_reg), v.x, v.y, v.z, v.r, v.g, v.b, v.a);
                    if (v.y >= 0 && static_cast<u32>(v.y) < rt.height) {
                        mark_fb_dirty(c.active_draw_fbp, static_cast<u32>(v.y), static_cast<u32>(v.y) + 1u);
                    }
                    primitives_drawn++;
                }
            }
            m_vtx_queue.clear();
            break;
        case 1u: // Line
            m_vtx_queue.push_back(v);
            if (m_vtx_queue.size() >= 2u) {
                if (drawing_kick) {
                    rasterize_line(m_vtx_queue[0], m_vtx_queue[1]);
                }
                m_vtx_queue.clear();
            }
            break;
        case 2u: // Line Strip
            m_vtx_queue.push_back(v);
            if (m_vtx_queue.size() >= 2u) {
                if (drawing_kick) {
                    rasterize_line(m_vtx_queue[0], m_vtx_queue[1]);
                }
                m_vtx_queue[0] = m_vtx_queue[1];
                m_vtx_queue.resize(1u);
            }
            break;
        case 3u: // Triangle (always consumes 3 vertices even when ADC=1)
            m_vtx_queue.push_back(v);
            if (m_vtx_queue.size() >= 3u) {
                if (drawing_kick) {
                    enqueue_triangle(m_vtx_queue[0], m_vtx_queue[1], m_vtx_queue[2]);
                }
                m_vtx_queue.clear();
            }
            break;
        case 4u: // Triangle Strip (3-vertex shift register; ADC=1 shifts without kicking)
            if (m_vtx_queue.size() >= 3u) {
                m_vtx_queue[0] = m_vtx_queue[1];
                m_vtx_queue[1] = m_vtx_queue[2];
                m_vtx_queue[2] = v;
            } else {
                m_vtx_queue.push_back(v);
            }
            if (m_vtx_queue.size() == 3u && drawing_kick) {
                enqueue_triangle(m_vtx_queue[0], m_vtx_queue[1], m_vtx_queue[2]);
            }
            break;
        case 5u: // Triangle Fan (keeps anchor v[0] + last 2 vertices)
            if (m_vtx_queue.size() >= 3u) {
                m_vtx_queue[1] = m_vtx_queue[2];
                m_vtx_queue[2] = v;
            } else {
                m_vtx_queue.push_back(v);
            }
            if (m_vtx_queue.size() == 3u && drawing_kick) {
                enqueue_triangle(m_vtx_queue[0], m_vtx_queue[1], m_vtx_queue[2]);
            }
            break;
        case 6u: // Sprite (always consumes 2 vertices even when ADC=1)
            m_vtx_queue.push_back(v);
            if (m_vtx_queue.size() >= 2u) {
                if (drawing_kick) {
                    rasterize_sprite(m_vtx_queue[0], m_vtx_queue[1]);
                }
                m_vtx_queue.clear();
            }
            break;
        default:
            m_vtx_queue.clear();
            break;
    }
}

} // namespace tf::ps2
