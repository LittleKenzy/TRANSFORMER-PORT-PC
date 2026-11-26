#include "TestFramework.h"
#include "../Runtime/RuntimeHost.h"
#include "../Runtime/CPU/R5900Helpers.h"
#include "../Runtime/CPU/R5900Interpreter.h"
#include "../Builder/Cache/BuildCache.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <thread>

using namespace tf::ps2;

TF_TEST(RuntimeTests, GuestMemoryMirrorsAndScratchpad) {
    GuestMemory mem;
    mem.write32(0x00123450u, 0xDEADBEEFu);
    // KSEG0, KSEG1, Uncached (0x20000000), and Uncached Accelerated (0x30000000) mirrors
    TF_EXPECT_EQ(mem.read32(0x80123450u), 0xDEADBEEFu);
    TF_EXPECT_EQ(mem.read32(0xA0123450u), 0xDEADBEEFu);
    TF_EXPECT_EQ(mem.read32(0x20123450u), 0xDEADBEEFu);
    TF_EXPECT_EQ(mem.read32(0x30123450u), 0xDEADBEEFu);

    // Scratchpad 0x70000000 is distinct from RAM 0x00000000
    mem.write32(0x70000010u, 0xCAFEBABEu);
    TF_EXPECT_EQ(mem.read32(0x70000010u), 0xCAFEBABEu);
    TF_EXPECT_EQ(mem.read32(0x00000010u), 0u);
}

TF_TEST(RuntimeTests, SyscallDispatcherThreadsSemasAndMemory) {
    RuntimeHost rt;

    // GetMemorySize (0x7F)
    rt.cpu.set_gpr_s32(3, 0x7F);
    rt.cpu.do_syscall(0);
    TF_EXPECT_EQ(rt.cpu.gpr_u32(2), kEeRamSize);

    // CreateSema (0x40) with init=1, max=2
    rt.memory.write32(0x00100000u, 0);
    rt.memory.write32(0x00100004u, 2);
    rt.memory.write32(0x00100008u, 1);
    rt.cpu.set_gpr_s32(3, 0x40);
    rt.cpu.set_gpr_u32_sext(4, 0x00100000u);
    rt.cpu.do_syscall(0);
    s32 sid = rt.cpu.gpr_s32(2);
    TF_EXPECT(sid > 0);

    // PollSema succeeds once, fails second time
    rt.cpu.set_gpr_s32(3, 0x45);
    rt.cpu.set_gpr_s32(4, sid);
    rt.cpu.do_syscall(0);
    TF_EXPECT_EQ(rt.cpu.gpr_s32(2), sid);

    rt.cpu.set_gpr_s32(3, 0x45);
    rt.cpu.set_gpr_s32(4, sid);
    rt.cpu.do_syscall(0);
    TF_EXPECT_EQ(rt.cpu.gpr_s32(2), -1);
}

TF_TEST(RuntimeTests, VU0VectorMathAndCrossProduct) {
    RuntimeHost rt;
    // vf1 = (1, 0, 0, 1), vf2 = (0, 1, 0, 1) -> cross product = (0, 0, 1)
    rt.cpu.vf[1] = Vec4f{1.0f, 0.0f, 0.0f, 1.0f};
    rt.cpu.vf[2] = Vec4f{0.0f, 1.0f, 0.0f, 1.0f};

    // VOPMULA.xyz ACC, vf1, vf2 -> special2 fn2=0x2E => fn=0x3E, upper bits=(0x2E>>2)=0x0B at bits 10..6
    u32 vopmula = (0x12u << 26) | (0x1Eu << 21) | (2u << 16) | (1u << 11) | (0x0Bu << 6) | 0x3Eu;
    // VOPMSUB.xyz vf3, vf2, vf1 -> special1 fn=0x2E
    u32 vopmsub = (0x12u << 26) | (0x1Eu << 21) | (1u << 16) | (2u << 11) | (3u << 6) | 0x2Eu;

    rt.cpu.do_vu0_macro(vopmula);
    rt.cpu.do_vu0_macro(vopmsub);

    TF_EXPECT(std::fabs(rt.cpu.vf[3].x - 0.0f) < 1e-5f);
    TF_EXPECT(std::fabs(rt.cpu.vf[3].y - 0.0f) < 1e-5f);
    TF_EXPECT(std::fabs(rt.cpu.vf[3].z - 1.0f) < 1e-5f);
}

TF_TEST(RuntimeTests, DmacToGifToGsSpriteRasterization) {
    RuntimeHost rt;

    // Build a GIF PACKED A+D packet in EE RAM at 0x00200000:
    // Draws a 4x4 red sprite at (10,10)-(14,14)
    u32 pkt = 0x00200000u;
    // GIFtag: NLOOP=4, EOP=1, FLG=0(PACKED), NREG=1, REGS=0xE(A+D)
    u64 tag_lo = 4ULL | (1ULL << 15) | (1ULL << 60);
    u64 tag_hi = 0x0EULL;
    rt.memory.write128(pkt + 0, u128(tag_lo, tag_hi));
    // Reg 0: PRIM = 6 (Sprite)
    rt.memory.write128(pkt + 16, u128(6ULL, 0x00ULL));
    // Reg 1: RGBAQ = R=255, G=0, B=128, A=255
    rt.memory.write128(pkt + 32, u128(0xFF8000FFULL, 0x01ULL));
    // Reg 2: XYZ2 = (10<<4, 10<<4)
    rt.memory.write128(pkt + 48, u128((10ULL << 4) | ((10ULL << 4) << 16), 0x05ULL));
    // Reg 3: XYZ2 = (14<<4, 14<<4)
    rt.memory.write128(pkt + 64, u128((14ULL << 4) | ((14ULL << 4) << 16), 0x05ULL));

    // Trigger GIF DMA channel (normal mode, 5 quadwords) via MMIO!
    rt.memory.write32(0x1000A010u, pkt);        // GIF MADR
    rt.memory.write32(0x1000A020u, 5u);         // GIF QWC
    rt.memory.write32(0x1000A000u, 0x101u);     // GIF CHCR: DIR=1, STR=1

    TF_EXPECT_EQ(rt.gs.primitives_drawn, u64{1});
    TF_EXPECT_EQ(rt.gs.pixels_drawn, u64{16});
    u32 px = rt.gs.framebuffer_rgba()[11 * rt.gs.fb_width() + 11];
    TF_EXPECT_EQ(px, 0xFF8000FFu);
}

TF_TEST(RuntimeTests, Spu2AdpcmInputAndVirtualMemoryCard) {
    RuntimeHost rt;

    // SPU2 ADPCM block decode & stereo mix
    u8 block[16] = {};
    block[0] = 0x00; // filter 0, shift 0
    block[1] = 0x01; // end flag
    block[2] = 0x21; // two non-zero nibbles
    rt.spu2.write_ram(0, block, 16);
    rt.spu2.key_on(0, 0);
    s16 pcm[56] = {};
    rt.spu2.mix_stereo_samples(pcm, 28);
    TF_EXPECT(pcm[0] != 0 || pcm[2] != 0);

    // DualShock 2 pad packing (active-low Cross button)
    rt.pad0.set_button(Ps2Button::Cross, true);
    u8 pad_buf[32] = {};
    rt.pad0.pack_ps2_pad_buffer(pad_buf);
    u16 buttons_raw = read_le16(pad_buf + 2);
    TF_EXPECT_EQ(static_cast<u16>(~buttons_raw), static_cast<u16>(Ps2Button::Cross));

    // VirtualMemoryCard save/load + traversal protection
    namespace fs = std::filesystem;
    fs::path mc_dir = fs::temp_directory_path() / "tf_vmc_test";
    fs::remove_all(mc_dir);
    std::string err;
    TF_EXPECT(rt.memcard0.mount_directory(mc_dir.string(), err));
    std::vector<u8> payload{1, 2, 3, 4, 5, 6};
    TF_EXPECT(rt.memcard0.write_save_file("BASLUS-21561/save0.dat", payload, err));
    std::vector<u8> loaded;
    TF_EXPECT(rt.memcard0.read_save_file("BASLUS-21561/save0.dat", loaded, err));
    TF_EXPECT(loaded == payload);
    TF_EXPECT(!rt.memcard0.write_save_file("../../escape.txt", payload, err));

    // BuildCache store, lookup, invalidate
    fs::path cache_dir = fs::temp_directory_path() / "tf_cache_test";
    fs::remove_all(cache_dir);
    BuildCache cache(cache_dir.string());
    CacheIdentity id;
    id.disc_serial = "SLUS_215.61";
    id.elf_sha256 = "abcd1234";
    id.iso_fingerprint_sha256 = "ef567890";
    id.target_arch = "win64-x86_64";
    id.build_config = "full";

    CacheManifest m;
    m.identity = id;
    m.generated_dir = "gen";
    m.functions_emitted = 42;
    m.instructions_translated = 1200;
    TF_EXPECT(cache.store(m, err));

    CacheManifest out_m;
    TF_EXPECT(cache.lookup(id, out_m));
    TF_EXPECT_EQ(out_m.functions_emitted, u64{42});
    TF_EXPECT(cache.invalidate(id));
    TF_EXPECT(!cache.lookup(id, out_m));
}

TF_TEST(RuntimeTests, GsVramPsmct32ToPsmt8UnswizzleClutAndBlend) {
    RuntimeHost rt;
    auto& gs = rt.gs;

    // 1. Upload a 16x2 PSMCT32 CLUT at cbp=352 (page 11) via BITBLTBUF/TRXPOS/TRXREG/TRXDIR/HWREG.
    // CSM1 swizzle swaps indices 0x08..0x0F with 0x10..0x17:
    //   idx 0x00 -> CLUT (0,0) = 0x80102030
    //   idx 0x08 -> swizzled to 0x10 -> CLUT (0,1) = 0x80708090
    //   idx 0x10 -> swizzled to 0x08 -> CLUT (8,0) = 0x80405060
    //   idx 0x18 -> CLUT (8,1) = 0x80A0B0C0
    const u32 cbp = 352u;
    gs.write_internal_reg(0x50, (static_cast<u64>(cbp) << 32) | (1ULL << 48) | (0ULL << 56)); // BITBLTBUF: dbp=352, dbw=1, dpsm=PSMCT32
    gs.write_internal_reg(0x51, 0ULL);                                                        // TRXPOS: dsax=0, dsay=0
    gs.write_internal_reg(0x52, 16ULL | (2ULL << 32));                                        // TRXREG: 16x2
    gs.write_internal_reg(0x53, 0ULL);                                                        // TRXDIR: host->local
    for (u32 y = 0; y < 2; ++y) {
        for (u32 x = 0; x < 16; x += 2) {
            auto clut_color = [](u32 cx, u32 cy) -> u32 {
                if (cx == 0 && cy == 0) return 0x80102030u;
                if (cx == 8 && cy == 0) return 0x80405060u;
                if (cx == 0 && cy == 1) return 0x80708090u;
                if (cx == 8 && cy == 1) return 0x80A0B0C0u;
                return 0x80000000u;
            };
            u64 pair = static_cast<u64>(clut_color(x, y)) | (static_cast<u64>(clut_color(x + 1, y)) << 32);
            gs.write_internal_reg(0x54, pair); // HWREG
        }
    }

    // 2. Upload a single PSMCT32 word 0x18100800 at tbp0=320 (page 10), (x=0, y=0).
    // In PS2 4 MiB GS VRAM layout, bytes [0,1,2,3] of PSMCT32 (0,0) unswizzle in PSMT8 to:
    //   (0,0)->0x00, (4,2)->0x08, (8,0)->0x10, (12,2)->0x18.
    const u32 tbp = 320u;
    gs.write_internal_reg(0x50, (static_cast<u64>(tbp) << 32) | (1ULL << 48) | (0ULL << 56)); // PSMCT32
    gs.write_internal_reg(0x51, 0ULL);
    gs.write_internal_reg(0x52, 2ULL | (1ULL << 32));
    gs.write_internal_reg(0x53, 0ULL);
    gs.write_internal_reg(0x54, 0x0000000018100800ULL);

    // Draw a 16x16 textured sprite (PRIM=Sprite|TME|FST) sampling PSMT8 (0x13) with CSM1 CLUT at cbp=352, tfx=Decal(1), tcc=1
    u64 tex0 = static_cast<u64>(tbp)
             | (2ULL << 14)       // TBW = 2
             | (0x13ULL << 20)    // PSM = PSMT8
             | (4ULL << 26)       // TW = 16
             | (4ULL << 30)       // TH = 16
             | (1ULL << 34)       // TCC = RGBA
             | (1ULL << 35)       // TFX = Decal
             | (static_cast<u64>(cbp) << 37)
             | (0ULL << 51);      // CPSM = PSMCT32
    gs.write_internal_reg(0x06, tex0); // TEX0_1
    gs.write_internal_reg(0x00, 6ULL | (1ULL << 4) | (1ULL << 8)); // PRIM: Sprite + TME + FST
    gs.write_internal_reg(0x01, 0x80808080ULL); // RGBAQ
    gs.write_internal_reg(0x03, 0ULL);          // UV = (0, 0)
    gs.write_internal_reg(0x05, 0ULL);          // XYZ2 = (0, 0)
    gs.write_internal_reg(0x03, (16ULL << 4) | ((16ULL << 4) << 16)); // UV = (16, 16)
    gs.write_internal_reg(0x05, (16ULL << 4) | ((16ULL << 4) << 16)); // XYZ2 = (16, 16)

    const auto& fb = gs.framebuffer_rgba();
    const u32 w = gs.fb_width();
    TF_EXPECT_EQ(fb[0 * w + 0],  0xFF102030u); // idx 0x00 -> CLUT(0,0)
    TF_EXPECT_EQ(fb[2 * w + 4],  0xFF708090u); // idx 0x08 -> CSM1(0x10) -> CLUT(0,1)
    TF_EXPECT_EQ(fb[0 * w + 8],  0xFF405060u); // idx 0x10 -> CSM1(0x08) -> CLUT(8,0)
    TF_EXPECT_EQ(fb[2 * w + 12], 0xFFA0B0C0u); // idx 0x18 -> CLUT(8,1)

    // 3. TEST_1 (alpha test + reversed-Z depth test) and ALPHA_1 blending
    auto draw_flat_rect = [&](u32 x0, u32 y0, u32 x1, u32 y1, u32 z, u32 rgba, bool abe) {
        gs.write_internal_reg(0x00, 6ULL | (abe ? (1ULL << 6) : 0ULL));
        gs.write_internal_reg(0x01, rgba);
        gs.write_internal_reg(0x05, (static_cast<u64>(x0 << 4)) | (static_cast<u64>(y0 << 4) << 16) | (static_cast<u64>(z) << 32));
        gs.write_internal_reg(0x05, (static_cast<u64>(x1 << 4)) | (static_cast<u64>(y1 << 4) << 16) | (static_cast<u64>(z) << 32));
    };

    // Enable Z-test ALWAYS initially to seed Z=100 and background RGB=(100,40,20) at (20,20)-(24,24)
    gs.write_internal_reg(0x47, (1ULL << 16) | (1ULL << 17)); // ZTE=1, ZTST=ALWAYS
    draw_flat_rect(20, 20, 24, 24, 100u, 0x80142864u, false);
    TF_EXPECT_EQ(fb[21 * w + 21], 0xFF142864u);

    // Enable Alpha Test (GEQUAL 0x40, AFAIL=KEEP) and Z Test (GEQUAL)
    u64 test_reg = 1ULL | (5ULL << 1) | (0x40ULL << 4) | (0ULL << 12) | (1ULL << 16) | (2ULL << 17);
    gs.write_internal_reg(0x47, test_reg);

    // Fails alpha test (A=0x20 < 0x40) -> pixel untouched
    draw_flat_rect(20, 20, 24, 24, 200u, 0x20FFFFFFu, false);
    TF_EXPECT_EQ(fb[21 * w + 21], 0xFF142864u);

    // Passes alpha test (A=0x80) but fails reversed-Z test (Z=50 < 100) -> pixel untouched
    draw_flat_rect(20, 20, 24, 24, 50u, 0x80FFFFFFu, false);
    TF_EXPECT_EQ(fb[21 * w + 21], 0xFF142864u);

    // ALPHA_1: ((Cs - Cd) * As >> 7) + Cd => A=0, B=1, C=0, D=1 => 0x44
    gs.write_internal_reg(0x42, 0x44ULL);
    // Source RGB=(200,140,120), A=0x40 (50% PS2 alpha), Z=200 (>=100) -> blended RGB=(150,90,70)
    draw_flat_rect(20, 20, 24, 24, 200u, (200u) | (140u << 8) | (120u << 16) | (0x40u << 24), true);
    u32 expected_blend = 150u | (90u << 8) | (70u << 16) | 0xFF000000u;
    TF_EXPECT_EQ(fb[21 * w + 21], expected_blend);
}

TF_TEST(RuntimeTests, McManSdk300RpcsBackedByVirtualMemoryCard) {
    RuntimeHost rt;
    namespace fs = std::filesystem;
    fs::path mc_dir = fs::temp_directory_path() / "tf_mcman_rpc_test";
    fs::remove_all(mc_dir);
    std::string err;
    TF_EXPECT(rt.memcard0.mount_directory(mc_dir.string(), err));

    u8 send[128]{};
    u8 recv[32]{};

    // fn=254: MC_RPCCMD_INIT -> returns 0, mcman >= 0x20A, mcserv >= 0x20E
    TF_EXPECT_EQ(rt.iop.call_rpc(kSifSidMcMan, 254u, send, 16, recv, 16), 0);
    TF_EXPECT_EQ(read_le32(recv + 0), 0u);
    TF_EXPECT(read_le32(recv + 4) >= 0x20Au);
    TF_EXPECT(read_le32(recv + 8) >= 0x20Eu);

    // fn=21: sceMcGetSlotMax -> 1 slot
    std::memset(recv, 0, sizeof(recv));
    TF_EXPECT_EQ(rt.iop.call_rpc(kSifSidMcMan, 21u, send, 16, recv, 16), 0);
    TF_EXPECT_EQ(read_le32(recv + 0), 1u);

    // fn=1: sceMcGetInfo -> first call returns -1 (ChangedCard), second returns 0; writes enddata
    const u32 enddata_addr = 0x00210000u;
    std::memset(send, 0, sizeof(send));
    write_le32(send + 4, 0u);           // port 0
    write_le32(send + 8, 0u);           // slot 0
    write_le32(send + 28, enddata_addr);
    TF_EXPECT_EQ(rt.iop.call_rpc(kSifSidMcMan, 1u, send, 32, recv, 16), 0);
    TF_EXPECT_EQ(static_cast<s32>(read_le32(recv + 0)), -1);
    TF_EXPECT_EQ(rt.memory.read32(enddata_addr + 0), 2u);    // sceMcTypePS2
    TF_EXPECT(rt.memory.read32(enddata_addr + 4) > 0u);      // free KB
    TF_EXPECT_EQ(rt.memory.read32(enddata_addr + 144), 1u);  // formatted
    TF_EXPECT_EQ(rt.iop.call_rpc(kSifSidMcMan, 1u, send, 32, recv, 16), 0);
    TF_EXPECT_EQ(static_cast<s32>(read_le32(recv + 0)), 0);

    // fn=2 (flags=0x40): sceMcMkdir("/BASLUS-21602")
    std::memset(send, 0, sizeof(send));
    write_le32(send + 0, 0u);     // port
    write_le32(send + 4, 0u);     // slot
    write_le32(send + 8, 0x40u);  // mkdir flag
    std::memcpy(send + 20, "/BASLUS-21602", 14);
    TF_EXPECT_EQ(rt.iop.call_rpc(kSifSidMcMan, 2u, send, 64, recv, 16), 0);
    TF_EXPECT_EQ(static_cast<s32>(read_le32(recv + 0)), 0);

    // fn=12: sceMcChdir("/BASLUS-21602") and back to "/"
    const u32 pwd_addr = 0x00211000u;
    std::memset(send, 0, sizeof(send));
    write_le32(send + 0, 0u);
    write_le32(send + 4, 0u);
    write_le32(send + 16, pwd_addr);
    std::memcpy(send + 20, "/BASLUS-21602", 14);
    TF_EXPECT_EQ(rt.iop.call_rpc(kSifSidMcMan, 12u, send, 64, recv, 16), 0);
    TF_EXPECT_EQ(static_cast<s32>(read_le32(recv + 0)), 0);
    TF_EXPECT_EQ(rt.memory.read8(pwd_addr + 0), static_cast<u8>('/'));

    // Chdir to nonexistent directory returns -4 (sceMcResNoEntry)
    std::memset(send + 20, 0, 32);
    std::memcpy(send + 20, "/NO_SUCH_DIR", 13);
    TF_EXPECT_EQ(rt.iop.call_rpc(kSifSidMcMan, 12u, send, 64, recv, 16), 0);
    TF_EXPECT_EQ(static_cast<s32>(read_le32(recv + 0)), -4);

    // fn=2: sceMcOpen relative "save.bin" (inside /BASLUS-21602) with create|write (0x202)
    std::memset(send, 0, sizeof(send));
    write_le32(send + 0, 0u);
    write_le32(send + 4, 0u);
    write_le32(send + 8, 0x202u);
    std::memcpy(send + 20, "save.bin", 9);
    TF_EXPECT_EQ(rt.iop.call_rpc(kSifSidMcMan, 2u, send, 64, recv, 16), 0);
    s32 fd = static_cast<s32>(read_le32(recv + 0));
    TF_EXPECT(fd >= 1);

    // fn=6: sceMcWrite with 4-byte inline header + 8-byte EE RAM body
    const u32 write_buf_addr = 0x00212000u;
    const u8 body_bytes[8] = {0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88};
    rt.memory.write_block(write_buf_addr, body_bytes, sizeof(body_bytes));
    std::memset(send, 0, sizeof(send));
    write_le32(send + 0, static_cast<u32>(fd));
    write_le32(send + 12, 8u); // body_sz
    write_le32(send + 20, 4u); // head_sz
    write_le32(send + 24, write_buf_addr);
    send[32] = 'T'; send[33] = 'F'; send[34] = '0'; send[35] = '1';
    TF_EXPECT_EQ(rt.iop.call_rpc(kSifSidMcMan, 6u, send, 40, recv, 16), 0);
    TF_EXPECT_EQ(read_le32(recv + 0), 12u);

    // fn=3: sceMcClose
    std::memset(send, 0, sizeof(send));
    write_le32(send + 0, static_cast<u32>(fd));
    TF_EXPECT_EQ(rt.iop.call_rpc(kSifSidMcMan, 3u, send, 16, recv, 16), 0);
    TF_EXPECT_EQ(static_cast<s32>(read_le32(recv + 0)), 0);

    // fn=13: sceMcGetDir("/BASLUS-21602/*") -> finds save.bin with size 12
    const u32 tbl_addr = 0x00213000u;
    std::memset(send, 0, sizeof(send));
    write_le32(send + 0, 0u);
    write_le32(send + 4, 0u);
    write_le32(send + 12, 8u); // maxent
    write_le32(send + 16, tbl_addr);
    std::memcpy(send + 20, "/BASLUS-21602/*", 16);
    TF_EXPECT_EQ(rt.iop.call_rpc(kSifSidMcMan, 13u, send, 64, recv, 16), 0);
    TF_EXPECT_EQ(read_le32(recv + 0), 1u);
    TF_EXPECT_EQ(rt.memory.read32(tbl_addr + 24), 12u);     // file size
    TF_EXPECT_EQ(rt.memory.read16(tbl_addr + 28), 0x8417u); // regular file attr
    char entry_name[16]{};
    rt.memory.read_block(tbl_addr + 32, reinterpret_cast<u8*>(entry_name), 9);
    TF_EXPECT_EQ(std::string(entry_name), std::string("save.bin"));

    // fn=2 (open read) + fn=5 (sceMcRead) + fn=3 (close)
    std::memset(send, 0, sizeof(send));
    write_le32(send + 8, 0x01u); // read-only
    std::memcpy(send + 20, "/BASLUS-21602/save.bin", 23);
    TF_EXPECT_EQ(rt.iop.call_rpc(kSifSidMcMan, 2u, send, 64, recv, 16), 0);
    s32 rfd = static_cast<s32>(read_le32(recv + 0));
    TF_EXPECT(rfd >= 1);

    const u32 read_buf_addr = 0x00214000u;
    std::memset(send, 0, sizeof(send));
    write_le32(send + 0, static_cast<u32>(rfd));
    write_le32(send + 12, 12u);
    write_le32(send + 24, read_buf_addr);
    TF_EXPECT_EQ(rt.iop.call_rpc(kSifSidMcMan, 5u, send, 32, recv, 16), 0);
    TF_EXPECT_EQ(read_le32(recv + 0), 12u);

    u8 read_back[12]{};
    rt.memory.read_block(read_buf_addr, read_back, 12);
    const u8 expected[12] = {'T', 'F', '0', '1', 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88};
    for (int i = 0; i < 12; ++i) {
        TF_EXPECT_EQ(read_back[i], expected[i]);
    }

    std::memset(send, 0, sizeof(send));
    write_le32(send + 0, static_cast<u32>(rfd));
    TF_EXPECT_EQ(rt.iop.call_rpc(kSifSidMcMan, 3u, send, 16, recv, 16), 0);
    fs::remove_all(mc_dir);
}

TF_TEST(RuntimeTests, DualShock2PackAndPadManLayout) {
    RuntimeHost rt;

    // Digital buttons + analog sticks + DS2 pressure bytes
    rt.pad0.set_button(Ps2Button::Start, true);
    rt.pad0.set_button(Ps2Button::Cross, true);
    rt.pad0.set_button(Ps2Button::DpadUp, true);
    rt.pad0.set_button(Ps2Button::R1, true);
    rt.pad0.set_left_stick(1.0f, -1.0f, rt.input_config);

    u8 buf[32]{};
    rt.pad0.pack_ps2_pad_buffer(buf);
    TF_EXPECT_EQ(buf[0], u8{0x00});
    TF_EXPECT_EQ(buf[1], u8{0x79}); // DualShock 2 Analog + Pressure
    u16 expected_mask = static_cast<u16>(Ps2Button::Start) |
                        static_cast<u16>(Ps2Button::Cross) |
                        static_cast<u16>(Ps2Button::DpadUp) |
                        static_cast<u16>(Ps2Button::R1);
    TF_EXPECT_EQ(static_cast<u16>(~read_le16(buf + 2)), expected_mask);
    TF_EXPECT_EQ(buf[6], u8{0xFF}); // LX full right
    TF_EXPECT_EQ(buf[7], u8{0x00}); // LY full up
    // DS2 pressure byte order: [0]=Right,[1]=Left,[2]=Up,[3]=Down,[4]=Tri,[5]=Cir,[6]=Cross,[7]=Sq,[8]=L1,[9]=R1,[10]=L2,[11]=R2
    TF_EXPECT_EQ(buf[8 + 2], u8{0xFF}); // DpadUp pressure
    TF_EXPECT_EQ(buf[8 + 6], u8{0xFF}); // Cross pressure
    TF_EXPECT_EQ(buf[8 + 9], u8{0xFF}); // R1 pressure
    TF_EXPECT_EQ(buf[8 + 4], u8{0x00}); // Triangle released

    // Releasing button clears both digital bit and pressure byte
    rt.pad0.set_button(Ps2Button::Cross, false);
    rt.pad0.pack_ps2_pad_buffer(buf);
    TF_EXPECT_EQ(buf[8 + 6], u8{0x00});

    // PADMAN RPC: GET_MODVER (cmd=0x12) -> Sony SDK 3.00 requires (ver >> 8) == 4
    u8 send[32]{};
    u8 recv[32]{};
    write_le32(send + 0, 0x12u);
    TF_EXPECT_EQ(rt.iop.call_rpc(kSifSidPadMan, 1u, send, 24, recv, 24), 0);
    TF_EXPECT_EQ(read_le32(recv + 12) >> 8, 4u);

    // PADMAN RPC: PAD_RPCCMD_OPEN (cmd=1, port=0, slot=0, ee_buf=0x00220000)
    const u32 pad_dma_addr = 0x00220000u;
    std::memset(send, 0, sizeof(send));
    std::memset(recv, 0, sizeof(recv));
    write_le32(send + 0, 1u);
    write_le32(send + 4, 0u);
    write_le32(send + 8, 0u);
    write_le32(send + 16, pad_dma_addr);
    TF_EXPECT_EQ(rt.iop.call_rpc(kSifSidPadMan, 1u, send, 24, recv, 24), 0);
    TF_EXPECT_EQ(read_le32(recv + 12), 1u);

    // Verify 128-byte Sony SDK 3.00 pad_data_t layout across both double-buffer slots
    for (u32 slot = 0; slot < 2; ++slot) {
        u32 base = pad_dma_addr + slot * 128u;
        TF_EXPECT_EQ(rt.memory.read8(base + 0), u8{0x00});
        TF_EXPECT_EQ(rt.memory.read8(base + 1), u8{0x79});
        TF_EXPECT_EQ(rt.memory.read8(base + 48), u8{0x01});     // actuator 0 func
        TF_EXPECT_EQ(rt.memory.read8(base + 49), u8{0x01});     // actuator 0 sub (small motor)
        TF_EXPECT_EQ(rt.memory.read8(base + 52), u8{0x01});     // actuator 1 func
        TF_EXPECT_EQ(rt.memory.read8(base + 53), u8{0x02});     // actuator 1 sub (large motor)
        TF_EXPECT_EQ(rt.memory.read16(base + 80), u16{4});      // modeTable[0] = Digital
        TF_EXPECT_EQ(rt.memory.read16(base + 82), u16{7});      // modeTable[1] = DualShock 2
        TF_EXPECT_EQ(rt.memory.read32(base + 96), u32{32});     // len = 32
        TF_EXPECT_EQ(rt.memory.read8(base + 100), u8{2});       // modeConfig = 2
        TF_EXPECT_EQ(rt.memory.read8(base + 101), u8{0x79});    // modeCurId = DualShock 2 Analog + Pressure
        TF_EXPECT((rt.memory.read8(base + 102) & 0x02u) != 0);  // buttonInfo press-mode bit
        TF_EXPECT_EQ(rt.memory.read8(base + 103), u8{1});       // ok
        TF_EXPECT_EQ(rt.memory.read8(base + 104), u8{2});       // modeTableNum
        TF_EXPECT_EQ(rt.memory.read8(base + 105), u8{1});       // modeCurOffs
        TF_EXPECT_EQ(rt.memory.read8(base + 106), u8{2});       // numActuators = 2
        TF_EXPECT_EQ(rt.memory.read8(base + 112), u8{6});       // scePadStateStable
        TF_EXPECT_EQ(rt.memory.read8(base + 113), u8{0});       // scePadReqStateComplete
        TF_EXPECT_EQ(rt.memory.read8(base + 114), u8{1});       // active/connected
    }

    // PADMAN RPC: EnterPressMode (cmd=10) sets recv[16] == 1
    std::memset(send, 0, sizeof(send));
    std::memset(recv, 0, sizeof(recv));
    write_le32(send + 0, 10u);
    TF_EXPECT_EQ(rt.iop.call_rpc(kSifSidPadMan, 1u, send, 24, recv, 24), 0);
    TF_EXPECT_EQ(read_le32(recv + 16), 1u);

    // tick_vblank refreshes pad DMA buffer with new button state & incremented frame counter
    u32 prev_frame = rt.memory.read32(pad_dma_addr + 88);
    rt.pad0.set_button(Ps2Button::Triangle, true);
    rt.iop.tick_vblank();
    TF_EXPECT(rt.memory.read32(pad_dma_addr + 88) > prev_frame);
    TF_EXPECT_EQ(rt.memory.read8(pad_dma_addr + 8 + 4), u8{0xFF}); // Triangle pressure updated in DMA buffer
}

TF_TEST(RuntimeTests, CooperativeEeWorkerThreadRoundRobinScheduling) {
    RuntimeHost rt;

    static std::vector<s32> s_order;
    static std::vector<u32> s_args;
    static s32 s_tid_a = 0;
    static s32 s_tid_b = 0;
    s_order.clear();
    s_args.clear();
    s_tid_a = 0;
    s_tid_b = 0;

    constexpr u32 kMainAddr    = 0x00100100u;
    constexpr u32 kWorkerAAddr = 0x00100200u;
    constexpr u32 kWorkerBAddr = 0x00100300u;

    rt.register_function(kWorkerAAddr, [](EEContext& ctx) {
        s_args.push_back(ctx.gpr_u32(4));
        for (int i = 0; i < 3; ++i) {
            ctx.set_gpr_s32(3, 0x2F); // GetThreadId
            ctx.do_syscall(0);
            s_order.push_back(ctx.gpr_s32(2));
            ctx.set_gpr_s32(3, 0x2B); // RotateThreadReadyQueue -> yield back to Main
            ctx.set_gpr_s32(4, 0);
            ctx.do_syscall(0);
        }
        ctx.set_gpr_s32(3, 0x23); // ExitThread
        ctx.do_syscall(0);
    });

    rt.register_function(kWorkerBAddr, [](EEContext& ctx) {
        s_args.push_back(ctx.gpr_u32(4));
        for (int i = 0; i < 2; ++i) {
            ctx.set_gpr_s32(3, 0x2F); // GetThreadId
            ctx.do_syscall(0);
            s_order.push_back(ctx.gpr_s32(2));
            ctx.set_gpr_s32(3, 0x2B); // RotateThreadReadyQueue -> yield back to Main
            ctx.set_gpr_s32(4, 0);
            ctx.do_syscall(0);
        }
        ctx.set_gpr_s32(3, 0x23); // ExitThread
        ctx.do_syscall(0);
    });

    rt.register_function(kMainAddr, [](EEContext& ctx) {
        auto create_and_start = [&](u32 param_addr, u32 entry, u32 stack, u32 arg) -> s32 {
            ctx.mem->write32(param_addr + 4, entry);
            ctx.mem->write32(param_addr + 8, stack);
            ctx.mem->write32(param_addr + 12, 0x4000u);
            ctx.mem->write32(param_addr + 16, 0u);
            ctx.mem->write32(param_addr + 20, 16u);
            ctx.set_gpr_s32(3, 0x20); // CreateThread
            ctx.set_gpr_u32_sext(4, param_addr);
            ctx.do_syscall(0);
            s32 tid = ctx.gpr_s32(2);
            ctx.set_gpr_s32(3, 0x22); // StartThread
            ctx.set_gpr_s32(4, tid);
            ctx.set_gpr_u32_sext(5, arg);
            ctx.do_syscall(0);
            return tid;
        };

        s_tid_a = create_and_start(0x00230000u, kWorkerAAddr, 0x00240000u, 0xAAu);
        s_tid_b = create_and_start(0x00230100u, kWorkerBAddr, 0x00248000u, 0xBBu);

        auto yield_once = [&]() {
            ctx.set_gpr_s32(3, 0x2B); // RotateThreadReadyQueue
            ctx.set_gpr_s32(4, 0);
            ctx.do_syscall(0);
        };

        // Round 1: A then B, then A again (next turn belongs to B)
        yield_once(); // A (slice 1)
        yield_once(); // B (slice 1)
        yield_once(); // A (slice 2); rr now points to B

        // Suspend B when it is B's turn: next yield must skip B and run A (slice 3)
        ctx.set_gpr_s32(3, 0x37); // SuspendThread(B)
        ctx.set_gpr_s32(4, s_tid_b);
        ctx.do_syscall(0);
        yield_once(); // skips B, runs A (slice 3)

        // Resume B and finish remaining slices + clean exits
        ctx.set_gpr_s32(3, 0x39); // ResumeThread(B)
        ctx.set_gpr_s32(4, s_tid_b);
        ctx.do_syscall(0);
        for (int i = 0; i < 6; ++i) {
            yield_once();
        }

        // Main thread ID must still be 1
        ctx.set_gpr_s32(3, 0x2F);
        ctx.do_syscall(0);
        TF_EXPECT_EQ(ctx.gpr_s32(2), 1);
    });

    rt.call_guest(kMainAddr);

    TF_EXPECT(s_tid_a >= 2);
    TF_EXPECT(s_tid_b >= 2);
    TF_EXPECT(s_tid_a != s_tid_b);
    TF_EXPECT_EQ(s_args.size(), std::size_t{2});
    TF_EXPECT_EQ(s_args[0], 0xAAu);
    TF_EXPECT_EQ(s_args[1], 0xBBu);
    // Exact cooperative schedule: A, B, A, A (B skipped while suspended), B
    TF_EXPECT_EQ(s_order.size(), std::size_t{5});
    TF_EXPECT_EQ(s_order[0], s_tid_a);
    TF_EXPECT_EQ(s_order[1], s_tid_b);
    TF_EXPECT_EQ(s_order[2], s_tid_a);
    TF_EXPECT_EQ(s_order[3], s_tid_a);
    TF_EXPECT_EQ(s_order[4], s_tid_b);
}

TF_TEST(RuntimeTests, GsPresentedFramebufferThreadSafetyAndResetRace) {
    RuntimeHost rt;

    // 1. Concurrent writer (GS draw + present_fbp + reset) vs UI reader (copy_presented_framebuffer)
    std::atomic<bool> stop_reader{false};
    std::atomic<u64> valid_copies{0};
    std::atomic<u64> nonblack_copies{0};
    std::atomic<bool> invariant_failed{false};

    std::thread reader([&]() {
        std::vector<u32> snap;
        u32 w = 0, h = 0;
        while (!stop_reader.load(std::memory_order_acquire)) {
            if (rt.gs.copy_presented_framebuffer(snap, w, h)) {
                if (w == 0 || h == 0 || snap.size() != static_cast<std::size_t>(w) * h) {
                    invariant_failed.store(true, std::memory_order_release);
                    return;
                }
                valid_copies.fetch_add(1, std::memory_order_relaxed);
                if (snap.size() > 11u * w + 11u && (snap[11u * w + 11u] & 0x00FFFFFFu) != 0u) {
                    nonblack_copies.fetch_add(1, std::memory_order_relaxed);
                }
            }
            std::this_thread::yield();
        }
    });

    for (int iter = 0; iter < 40; ++iter) {
        u32 color = (iter & 1) ? 0x80FF4020u : 0x8020A0FFu;
        rt.gs.write_internal_reg(0x4C, 0ULL); // FRAME_1 fbp=0
        rt.gs.write_internal_reg(0x00, 6ULL); // PRIM = Sprite
        rt.gs.write_internal_reg(0x01, color);
        rt.gs.write_internal_reg(0x05, (8ULL << 4) | ((8ULL << 4) << 16));
        rt.gs.write_internal_reg(0x05, (24ULL << 4) | ((24ULL << 4) << 16));
        rt.gs.present_fbp(0u);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    stop_reader.store(true, std::memory_order_release);
    reader.join();

    TF_EXPECT(!invariant_failed.load(std::memory_order_acquire));
    TF_EXPECT(valid_copies.load(std::memory_order_relaxed) > 0u);
    TF_EXPECT(nonblack_copies.load(std::memory_order_relaxed) > 0u);
    TF_EXPECT(rt.gs.frames_presented > 0u);

    // Final deterministic copy check
    std::vector<u32> final_copy;
    u32 fw = 0, fh = 0;
    TF_EXPECT(rt.gs.copy_presented_framebuffer(final_copy, fw, fh));
    TF_EXPECT_EQ(fw, rt.gs.fb_width());
    TF_EXPECT_EQ(fh, rt.gs.fb_height());
    TF_EXPECT_EQ(final_copy.size(), static_cast<std::size_t>(fw) * fh);
    TF_EXPECT((final_copy[12u * fw + 12u] & 0x00FFFFFFu) != 0u);

    // 2. RuntimeHost::reset() restart safety while m_iop_ticker is actively ticking heartbeat + pad DMA
    auto arm_heartbeat = [&]() {
        // NUSND3M cmd 73: register EE heartbeat addr 0x00250000 (big-endian 16-bit halfwords)
        u8 send[16]{};
        u8 recv[16]{};
        write_le16(send + 0, 1354u); // magic
        write_le16(send + 2, 1u);    // num_cmds
        write_le16(send + 4, 73u);   // cmd_id = 73 (register heartbeat)
        write_le16(send + 6, 2u);    // argc = 2
        write_le16(send + 8, 0x0025u);
        write_le16(send + 10, 0x0000u);
        rt.iop.call_rpc(0x00012345u, 0u, send, sizeof(send), recv, sizeof(recv));
    };

    for (int cycle = 0; cycle < 8; ++cycle) {
        arm_heartbeat();
        TF_EXPECT(rt.iop.has_heartbeat());
        // Let background m_iop_ticker actively write heartbeat into EE RAM
        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
        while (rt.memory.read32(0x00250000u) < 2u && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        TF_EXPECT(rt.memory.read32(0x00250000u) >= 1u);
        // Reset while m_iop_ticker is actively hammering tick_vblank()
        rt.reset();
        TF_EXPECT(!rt.iop.has_heartbeat());
        TF_EXPECT_EQ(rt.memory.read32(0x00250000u), 0u);
    }
}

TF_TEST(RuntimeTests, GsPsmt4Psmct16AndNonPageAlignedBpSwizzle) {
    RuntimeHost rt;
    auto& gs = rt.gs;

    // 1. Upload 16-entry PSMCT32 CLUT (8x2) at non-page-aligned cbp = 354 (page 11 + block 2)
    const u32 cbp = 354u;
    gs.write_internal_reg(0x50, (static_cast<u64>(cbp) << 32) | (1ULL << 48) | (0ULL << 56));
    gs.write_internal_reg(0x51, 0ULL);
    gs.write_internal_reg(0x52, 8ULL | (2ULL << 32));
    gs.write_internal_reg(0x53, 0ULL);
    for (u32 i = 0; i < 16; i += 2) {
        u32 c0 = 0x80000000u | ((i * 16u + 1u) << 0) | ((i * 8u + 2u) << 8) | ((i * 4u + 3u) << 16);
        u32 c1 = 0x80000000u | (((i + 1u) * 16u + 1u) << 0) | (((i + 1u) * 8u + 2u) << 8) | (((i + 1u) * 4u + 3u) << 16);
        gs.write_internal_reg(0x54, static_cast<u64>(c0) | (static_cast<u64>(c1) << 32));
    }

    // 2. Upload single PSMCT32 word 0x76543210 at non-page-aligned tbp = 322 (page 10 + block 2), (0,0).
    // Under exact PS2 GS PSMT4 column layout:
    //   byte lanes 0..3 -> x = 0, 8, 16, 24; low nibble -> y = 0, high nibble -> y = 2.
    const u32 tbp4 = 322u;
    gs.write_internal_reg(0x50, (static_cast<u64>(tbp4) << 32) | (1ULL << 48) | (0ULL << 56));
    gs.write_internal_reg(0x51, 0ULL);
    gs.write_internal_reg(0x52, 2ULL | (1ULL << 32));
    gs.write_internal_reg(0x53, 0ULL);
    gs.write_internal_reg(0x54, 0x0000000076543210ULL);

    gs.write_internal_reg(0x4C, (10ULL << 16)); // FRAME_1: fbp=0, fbw=10 (640px), psm=PSMCT32
    u64 tex0_4 = static_cast<u64>(tbp4)
               | (2ULL << 14)       // TBW = 2
               | (0x14ULL << 20)    // PSM = PSMT4
               | (5ULL << 26)       // TW = 32
               | (4ULL << 30)       // TH = 16
               | (1ULL << 34)       // TCC = RGBA
               | (1ULL << 35)       // TFX = Decal
               | (static_cast<u64>(cbp) << 37)
               | (0ULL << 51)       // CPSM = PSMCT32
               | (1ULL << 61);      // CLD = 1 (load)
    gs.write_internal_reg(0x06, tex0_4);
    gs.write_internal_reg(0x00, 6ULL | (1ULL << 4) | (1ULL << 8)); // Sprite + TME + FST
    gs.write_internal_reg(0x01, 0x80808080ULL);
    gs.write_internal_reg(0x03, 0ULL);
    gs.write_internal_reg(0x05, 0ULL);
    gs.write_internal_reg(0x03, (32ULL << 4) | ((16ULL << 4) << 16));
    gs.write_internal_reg(0x05, (32ULL << 4) | ((16ULL << 4) << 16));
    gs.present_fbp(0u);

    auto expect_idx = [&](u32 x, u32 y, u32 idx) {
        u32 expected = 0xFF000000u | ((idx * 16u + 1u) << 0) | ((idx * 8u + 2u) << 8) | ((idx * 4u + 3u) << 16);
        TF_EXPECT_EQ(gs.framebuffer_rgba()[y * gs.fb_width() + x], expected);
    };
    expect_idx(0,  0, 0);
    expect_idx(4,  2, 1);
    expect_idx(8,  0, 2);
    expect_idx(12, 2, 3);
    expect_idx(16, 0, 4);
    expect_idx(20, 2, 5);
    expect_idx(24, 0, 6);
    expect_idx(28, 2, 7);

    // 3. PSMCT16 column layout: upload 16x2 PSMCT16 pixels at tbp = 384 (page 12), distinct colors at x=0 and x=8
    const u32 tbp16 = 384u;
    gs.write_internal_reg(0x50, (static_cast<u64>(tbp16) << 32) | (1ULL << 48) | (0x02ULL << 56)); // PSMCT16
    gs.write_internal_reg(0x51, 0ULL);
    gs.write_internal_reg(0x52, 16ULL | (2ULL << 32));
    gs.write_internal_reg(0x53, 0ULL);
    for (u32 y = 0; y < 2; ++y) {
        for (u32 x = 0; x < 16; x += 4) {
            auto p16 = [](u32 px, u32 py) -> u16 {
                if (px == 0 && py == 0) return static_cast<u16>(0x801Fu); // R=31, G=0, B=0
                if (px == 8 && py == 0) return static_cast<u16>(0x83E0u); // R=0, G=31, B=0
                if (px == 8 && py == 1) return static_cast<u16>(0xFC00u); // R=0, G=0, B=31
                return 0x8000u;
            };
            u64 q = static_cast<u64>(p16(x, y))
                  | (static_cast<u64>(p16(x + 1, y)) << 16)
                  | (static_cast<u64>(p16(x + 2, y)) << 32)
                  | (static_cast<u64>(p16(x + 3, y)) << 48);
            gs.write_internal_reg(0x54, q);
        }
    }
    u64 tex0_16 = static_cast<u64>(tbp16)
                | (1ULL << 14)       // TBW = 1
                | (0x02ULL << 20)    // PSM = PSMCT16
                | (4ULL << 26)       // TW = 16
                | (2ULL << 30)       // TH = 4
                | (1ULL << 34)       // TCC = RGBA
                | (1ULL << 35);      // TFX = Decal
    gs.write_internal_reg(0x06, tex0_16);
    gs.write_internal_reg(0x03, 0ULL);
    gs.write_internal_reg(0x05, (40ULL << 4) | ((0ULL << 4) << 16));
    gs.write_internal_reg(0x03, (16ULL << 4) | ((2ULL << 4) << 16));
    gs.write_internal_reg(0x05, (56ULL << 4) | ((2ULL << 4) << 16));
    gs.present_fbp(0u);
    const u32 w = gs.fb_width();
    TF_EXPECT_EQ(gs.framebuffer_rgba()[0 * w + 40] & 0x00FFFFFFu, 0x000000F8u); // (0,0) red
    TF_EXPECT_EQ(gs.framebuffer_rgba()[0 * w + 48] & 0x00FFFFFFu, 0x0000F800u); // (8,0) green
    TF_EXPECT_EQ(gs.framebuffer_rgba()[1 * w + 48] & 0x00FFFFFFu, 0x00F80000u); // (8,1) blue
}

TF_TEST(RuntimeTests, GsTopLeftTriangleFillRuleAndBilinearTex1) {
    RuntimeHost rt;
    auto& gs = rt.gs;

    // 1. Two adjacent additive-blended triangles sharing vertical edge x=16: Cs + Cd (A=0,B=2,C=2,D=1,FIX=0x80)
    gs.write_internal_reg(0x4C, (10ULL << 16)); // FRAME_1 fbp=0, fbw=10
    gs.write_internal_reg(0x42, (0ULL << 0) | (2ULL << 2) | (2ULL << 4) | (1ULL << 6) | (0x80ULL << 32)); // ALPHA_1
    gs.write_internal_reg(0x00, 3ULL | (1ULL << 6)); // PRIM = Triangle + ABE
    gs.write_internal_reg(0x01, 0x80202020ULL);      // RGB = (32, 32, 32)
    auto vtx = [&](u32 x, u32 y) {
        gs.write_internal_reg(0x05, (static_cast<u64>(x) << 4) | (static_cast<u64>(y) << 20));
    };
    // Left quad (8..16, 8..24) split into 2 triangles
    vtx(8, 8);  vtx(16, 8);  vtx(8, 24);
    vtx(16, 8); vtx(16, 24); vtx(8, 24);
    // Right quad (16..24, 8..24) sharing vertical edge x=16
    vtx(16, 8); vtx(24, 8);  vtx(16, 24);
    vtx(24, 8); vtx(24, 24); vtx(16, 24);
    gs.present_fbp(0u);

    const u32 w = gs.fb_width();
    // Shared seam x=16 AND diagonal interior must be blended ONCE (32,32,32), never doubled (64,64,64)
    for (u32 y = 9; y < 23; ++y) {
        TF_EXPECT_EQ(gs.framebuffer_rgba()[y * w + 16] & 0x00FFFFFFu, 0x00202020u);
        TF_EXPECT_EQ(gs.framebuffer_rgba()[y * w + 12] & 0x00FFFFFFu, 0x00202020u);
        TF_EXPECT_EQ(gs.framebuffer_rgba()[y * w + 20] & 0x00FFFFFFu, 0x00202020u);
    }

    // 2. TEX1 bilinear filtering on a 2x2 black/white texture stretched to 8x8
    const u32 tbp = 416u; // page 13
    gs.write_internal_reg(0x50, (static_cast<u64>(tbp) << 32) | (1ULL << 48) | (0ULL << 56));
    gs.write_internal_reg(0x51, 0ULL);
    gs.write_internal_reg(0x52, 2ULL | (2ULL << 32));
    gs.write_internal_reg(0x53, 0ULL);
    gs.write_internal_reg(0x54, 0x80FFFFFF80000000ULL); // row 0: (0,0)=black, (1,0)=white
    gs.write_internal_reg(0x54, 0x8000000080FFFFFFULL); // row 1: (0,1)=white, (1,1)=black

    u64 tex0 = static_cast<u64>(tbp) | (1ULL << 14) | (0ULL << 20) | (1ULL << 26) | (1ULL << 30) | (1ULL << 34) | (1ULL << 35);
    gs.write_internal_reg(0x06, tex0);
    gs.write_internal_reg(0x08, (1ULL << 0) | (1ULL << 2));       // CLAMP_1: CLAMP U & V
    gs.write_internal_reg(0x14, (1ULL << 5) | (1ULL << 6));       // TEX1_1: MMAG=LINEAR(1), MMIN=LINEAR(1)
    gs.write_internal_reg(0x00, 6ULL | (1ULL << 4) | (1ULL << 8)); // Sprite + TME + FST
    gs.write_internal_reg(0x01, 0x80808080ULL);
    gs.write_internal_reg(0x03, 0ULL);
    gs.write_internal_reg(0x05, (40ULL << 4) | ((40ULL << 4) << 16));
    gs.write_internal_reg(0x03, (2ULL << 4) | ((2ULL << 4) << 16));
    gs.write_internal_reg(0x05, (48ULL << 4) | ((48ULL << 4) << 16));
    gs.present_fbp(0u);

    // Center pixels of the stretched 2x2 checkerboard must be interpolated gray (strictly between 0x10 and 0xF0)
    u32 mid_r = gs.framebuffer_rgba()[44 * w + 44] & 0xFFu;
    TF_EXPECT(mid_r > 0x10u && mid_r < 0xF0u);
}

TF_TEST(RuntimeTests, GifMultiPathIsolationAndPackedTex0Xyz3) {
    RuntimeHost rt;
    auto& gs = rt.gs;
    auto& gif = rt.gif;

    gs.write_internal_reg(0x4C, (10ULL << 16)); // FRAME_1 fbp=0, fbw=10
    // Start a 2-qword IMAGE transfer to page 14 (dbp=448) via PATH3, feeding header + 1st qword only
    const u32 tbp = 448u;
    gs.write_internal_reg(0x50, (static_cast<u64>(tbp) << 32) | (1ULL << 48) | (0ULL << 56));
    gs.write_internal_reg(0x51, 0ULL);
    gs.write_internal_reg(0x52, 4ULL | (1ULL << 32));
    gs.write_internal_reg(0x53, 0ULL);

    u8 path3_part1[32]{};
    // GIFtag: NLOOP=2, EOP=1, FLG=2 (IMAGE)
    write_le64(path3_part1 + 0, 2ULL | (1ULL << 15) | (2ULL << 58));
    write_le64(path3_part1 + 8, 0ULL);
    write_le64(path3_part1 + 16, 0x8000FF00800000FFULL); // px0=red, px1=green
    write_le64(path3_part1 + 24, 0x80FFFF0080FF0000ULL); // px2=blue, px3=cyan
    gif.feed_path(GifUnit::kPath3, path3_part1, 2, gs);

    // Interleave a PATH1 PACKED packet with TEX0_1 (0x6), RGBAQ (0x1), XYZ3 (0xD, no kick pair), XYZ2 (0x5, kick pair)
    // PRIM=Sprite(6), TME=0 (flat): suppressed sprite (0,0)-(10,10) via XYZ3, followed by drawn sprite (60,60)-(64,64)
    u8 path1_pkt[16 * 7]{};
    // GIFtag: NLOOP=1, EOP=1, PRE=1, PRIM=6(Sprite), FLG=0(PACKED), NREG=6:
    // reg0=TEX0_1(0x6), reg1=RGBAQ(0x1), reg2=XYZ3(0xD), reg3=XYZ3(0xD), reg4=XYZ2(0x5), reg5=XYZ2(0x5)
    u64 tag1_lo = 1ULL | (1ULL << 15) | (1ULL << 46) | (6ULL << 47) | (6ULL << 60);
    u64 tag1_hi = (0x6ULL << 0) | (0x1ULL << 4) | (0xDULL << 8) | (0xDULL << 12) | (0x5ULL << 16) | (0x5ULL << 20);
    write_le64(path1_pkt + 0, tag1_lo);
    write_le64(path1_pkt + 8, tag1_hi);
    // reg0: TEX0_1 raw 64-bit value
    u64 tex0_val = static_cast<u64>(tbp) | (1ULL << 14) | (2ULL << 26) | (1ULL << 30);
    write_le64(path1_pkt + 16, tex0_val);
    write_le64(path1_pkt + 24, 0ULL);
    // reg1: PACKED RGBAQ (R, G, B, A in 32-bit words 0..3)
    write_le32(path1_pkt + 32, 0x44u);
    write_le32(path1_pkt + 36, 0x88u);
    write_le32(path1_pkt + 40, 0xCCu);
    write_le32(path1_pkt + 44, 0x80u);
    // reg2 & reg3: PACKED XYZ3 pair (0,0)-(10,10) — no drawing kick, clears sprite queue!
    write_le32(path1_pkt + 48, 0u);
    write_le32(path1_pkt + 52, 0u);
    write_le32(path1_pkt + 64, 10u << 4);
    write_le32(path1_pkt + 68, 10u << 4);
    // reg4 & reg5: PACKED XYZ2 pair (60,60)-(64,64) — draws cleanly
    write_le32(path1_pkt + 80, 60u << 4);
    write_le32(path1_pkt + 84, 60u << 4);
    write_le32(path1_pkt + 96, 64u << 4);
    write_le32(path1_pkt + 100, 64u << 4);
    gif.feed_path(GifUnit::kPath1, path1_pkt, 7, gs);

    // Finish PATH3 IMAGE transfer (2nd qword) — PATH3 state must remain intact
    u8 path3_part2[16]{};
    write_le64(path3_part2 + 0, 0ULL);
    write_le64(path3_part2 + 8, 0ULL);
    gif.feed_path(GifUnit::kPath3, path3_part2, 1, gs);

    gs.present_fbp(0u);
    const u32 w = gs.fb_width();
    // Sprite must have drawn cleanly at (60..64, 60..64) and NOT at (0..60, 0..60)
    TF_EXPECT_EQ(gs.primitives_drawn, u64{1});
    TF_EXPECT_EQ(gs.framebuffer_rgba()[61 * w + 61] & 0x00FFFFFFu, 0x00CC8844u);
    TF_EXPECT_EQ(gs.framebuffer_rgba()[10 * w + 10] & 0x00FFFFFFu, 0u);
}

TF_TEST(RuntimeTests, DmacScratchpadBit31AndMmiPmfhlLhSh) {
    RuntimeHost rt;

    // 1. Write GIF packet to Scratchpad (0x70000100) while Main RAM (0x00000100) stays zero.
    const u32 spr_pkt = 0x70000100u;
    rt.memory.write128(spr_pkt + 0,  u128(4ULL | (1ULL << 15) | (1ULL << 60), 0x0EULL)); // A+D, 4 regs
    rt.memory.write128(spr_pkt + 16, u128(6ULL, 0x00ULL));                                // PRIM = Sprite
    rt.memory.write128(spr_pkt + 32, u128(0x8033AA55ULL, 0x01ULL));                       // RGBAQ
    rt.memory.write128(spr_pkt + 48, u128((20ULL << 4) | ((20ULL << 4) << 16), 0x05ULL)); // XYZ2
    rt.memory.write128(spr_pkt + 64, u128((24ULL << 4) | ((24ULL << 4) << 16), 0x05ULL)); // XYZ2

    // Trigger GIF DMA with SPR bit 31 set (0x80000100)
    rt.memory.write32(0x1000A010u, 0x80000100u); // GIF MADR (bit 31 = SPR!)
    rt.memory.write32(0x1000A020u, 5u);          // GIF QWC
    rt.memory.write32(0x1000A000u, 0x101u);      // GIF CHCR: DIR=1, STR=1

    TF_EXPECT_EQ(rt.gs.primitives_drawn, u64{1});
    rt.gs.present_fbp(0u);
    TF_EXPECT_EQ(rt.gs.framebuffer_rgba()[21 * rt.gs.fb_width() + 21] & 0x00FFFFFFu, 0x0033AA55u);

    // 2. MMI PMFHL.LH (fmt=3) and PMFHL.SH (fmt=4)
    EEContext ctx{};
    ctx.lo  = (static_cast<u64>(0x00010002u) << 32) | 0x00001234u; // lane0=+4660, lane1=+65538 (>32767)
    ctx.hi  = (static_cast<u64>(0xFFFF0000u) << 32) | 0xFFFF8001u; // lane2=-32767, lane3=-65536 (<-32768)
    ctx.lo1 = (static_cast<u64>(0x00007FFFu) << 32) | 0x12345678u; // lane4=+0x12345678, lane5=+32767
    ctx.hi1 = (static_cast<u64>(0x80000000u) << 32) | 0x00000000u; // lane6=0, lane7=INT32_MIN

    u128 lh = helpers::mmi_pmfhl(ctx, 3u);
    TF_EXPECT_EQ(lh.uh(0), u16{0x1234});
    TF_EXPECT_EQ(lh.uh(1), u16{0x0002});
    TF_EXPECT_EQ(lh.uh(2), u16{0x8001});
    TF_EXPECT_EQ(lh.uh(3), u16{0x0000});
    TF_EXPECT_EQ(lh.uh(4), u16{0x5678});
    TF_EXPECT_EQ(lh.uh(5), u16{0x7FFF});
    TF_EXPECT_EQ(lh.uh(6), u16{0x0000});
    TF_EXPECT_EQ(lh.uh(7), u16{0x0000});

    u128 sh = helpers::mmi_pmfhl(ctx, 4u);
    TF_EXPECT_EQ(sh.sh(0), s16{0x1234});
    TF_EXPECT_EQ(sh.sh(1), s16{32767});
    TF_EXPECT_EQ(sh.sh(2), s16{-32767});
    TF_EXPECT_EQ(sh.sh(3), s16{-32768});
    TF_EXPECT_EQ(sh.sh(4), s16{32767});
    TF_EXPECT_EQ(sh.sh(5), s16{32767});
    TF_EXPECT_EQ(sh.sh(6), s16{0});
    TF_EXPECT_EQ(sh.sh(7), s16{-32768});
}

TF_TEST(RuntimeTests, Vu1MaxMiniPreserveMacAndXgkickWrap) {
    VU1Unit vu1;
    std::vector<u8> code_mem(16384, 0);
    std::vector<u8> data_mem(16384, 0);
    vu1.attach_memory(code_mem.data(), data_mem.data());

    // Place a 2-qword GIF packet wrapping across 16 KiB boundary: qword 1023 (0x3FF0) + qword 0 (0x0000)
    // GIFtag at 0x3FF0: NLOOP=1, EOP=1, FLG=2(IMAGE) -> 1 payload qword at 0x0000
    write_le64(data_mem.data() + 1023 * 16 + 0, 1ULL | (1ULL << 15) | (2ULL << 58));
    write_le64(data_mem.data() + 1023 * 16 + 8, 0xCAFEBABE11223344ULL);
    write_le64(data_mem.data() + 0 * 16 + 0,    0xDEADBEEF55667788ULL);
    write_le64(data_mem.data() + 0 * 16 + 8,    0x0102030405060708ULL);

    std::vector<u8> kicked;
    vu1.set_xgkick_callback([&](const u8* data, std::size_t max_bytes) {
        if (max_bytes >= 32) {
            kicked.assign(data, data + 32);
        }
    });

    vu1.vf[3] = Vec4f{5.0f, 6.0f, 7.0f, 8.0f};
    vu1.vf[4] = Vec4f{2.0f, 9.0f, 4.0f, 1.0f};
    vu1.vi[1] = 1023u;

    const u32 nop_lower = 0x8000033Cu; // NOP lower
    const u32 nop_upper = 0x000002FFu; // NOP upper
    // Inst 0: Upper SUBx.xyzw vf1, vf0, vf0 (sets Z flags on x,y,z -> mac_flags = 0x000E) | Lower NOP
    const u32 subx_upper = (0xFu << 21) | (0u << 16) | (0u << 11) | (1u << 6) | 0x04u;
    // Inst 1: Upper MAXx.xyzw vf2, vf3, vf4 (must NOT clobber mac_flags!) | Lower XGKICK vi1 (sub=0x6C)
    const u32 maxx_upper = (0xFu << 21) | (4u << 16) | (3u << 11) | (2u << 6) | 0x10u;
    const u32 xgkick_vi1 = (0x40u << 25) | (1u << 11) | (0x1Bu << 6) | 0x3Cu;
    // Inst 2: E-bit NOP | Lower NOP
    // Inst 3: delay slot NOP | Lower NOP
    auto emit_pair = [&](u32 idx, u32 lo, u32 hi) {
        write_le32(code_mem.data() + idx * 8 + 0, lo);
        write_le32(code_mem.data() + idx * 8 + 4, hi);
    };
    emit_pair(0, nop_lower, subx_upper);
    emit_pair(1, xgkick_vi1, maxx_upper);
    emit_pair(2, nop_lower, nop_upper | 0x40000000u); // E-bit
    emit_pair(3, nop_lower, nop_upper);

    vu1.run_microprogram(0u, 64u);

    // MAXx computed vf2 = max(vf3, vf4.x=2.0) = (5, 6, 7, 8) without clobbering MAC Z flags (0x0E)
    TF_EXPECT_EQ(vu1.vf[2].x, 5.0f);
    TF_EXPECT_EQ(vu1.vf[2].y, 6.0f);
    TF_EXPECT_EQ(vu1.mac_flags & 0x0Eu, 0x0Eu);
    // XGKICK seamlessly wrapped across 16 KiB boundary
    TF_EXPECT_EQ(vu1.xgkick_count, u64{1});
    TF_EXPECT_EQ(kicked.size(), std::size_t{32});
    TF_EXPECT_EQ(read_le64(kicked.data() + 8),  0xCAFEBABE11223344ULL);
    TF_EXPECT_EQ(read_le64(kicked.data() + 16), 0xDEADBEEF55667788ULL);
    TF_EXPECT_EQ(read_le64(kicked.data() + 24), 0x0102030405060708ULL);
}

TF_TEST(RuntimeTests, Vu1FlagOpcodesQPipelineFmacLatencyAndViBranchHazard) {
    VU1Unit vu1;
    std::vector<u8> code_mem(16384, 0);
    std::vector<u8> data_mem(16384, 0);
    vu1.attach_memory(code_mem.data(), data_mem.data());

    const u32 nop_lo = 0x8000033Cu;
    const u32 nop_hi = 0x000002FFu;
    auto emit = [&](u32 idx, u32 lo, u32 hi) {
        write_le32(code_mem.data() + idx * 8 + 0, lo);
        write_le32(code_mem.data() + idx * 8 + 4, hi);
    };
    auto op_imm24 = [](u32 op, u32 imm24) { return (op << 25) | (imm24 & 0xFFFFFFu); };
    auto op_it_is_imm15 = [](u32 op, u32 it, u32 is, u32 imm15) {
        return (op << 25) | (((imm15 >> 11) & 0xFu) << 21) | ((it & 0x1Fu) << 16) | ((is & 0x1Fu) << 11) | (imm15 & 0x7FFu);
    };

    // Initial state: Q = 2.0f, vf2 = (4,0,0,1), vf4 = (10,20,30,40), vf6 = (5,0,0,1) (+x out of clip)
    vu1.q = 2.0f;
    vu1.vf[2] = Vec4f{4.0f, 0.0f, 0.0f, 1.0f};
    vu1.vf[4] = Vec4f{10.0f, 20.0f, 30.0f, 40.0f};
    vu1.vf[6] = Vec4f{5.0f, 0.0f, 0.0f, 1.0f};
    vu1.vi[6] = 0x000Eu; // mask for FMAND

    // 0: FCSET 0x000005  (0x11)
    emit(0, op_imm24(0x11u, 0x000005u), nop_hi);
    // 1: FCGET vi3 (0x1C) -> expects 0x0005
    emit(1, (0x1Cu << 25) | (3u << 16), nop_hi);
    // 2: FCAND vi1, 0x000004 (0x12) -> expects 1 (saved into vi4 next via IOR)
    emit(2, op_imm24(0x12u, 0x000004u), nop_hi);
    // 3: IOR vi4, vi1, vi0 (lower op=0x40, fn=0x35)
    emit(3, (0x40u << 25) | (0u << 16) | (1u << 11) | (4u << 6) | 0x35u, nop_hi);
    // 4: FSSET 0x0C0 (0x15, sticky bits 11:6) | Upper SUBx.xyzw vf1, vf0, vf0 (sets Z=0x0E in 4-cycle FMAC pipe)
    const u32 subx_hi = (0xFu << 21) | (0u << 16) | (0u << 11) | (1u << 6) | 0x04u;
    emit(4, op_it_is_imm15(0x15u, 0u, 0u, 0x0C0u), subx_hi);
    // 5: FSAND vi5, 0x0C0 (0x16) -> expects 0x0C0 | Upper CLIPw vf6, vf6 (sets +x clip bit 0 in 4-cycle pipe)
    const u32 clipw_hi = (0xEu << 21) | (6u << 16) | (6u << 11) | (0x7u << 6) | 0x3Fu; // Special2 0x1F = CLIPw
    emit(5, op_it_is_imm15(0x16u, 5u, 0u, 0x0C0u), clipw_hi);
    // 6 (d=1 after CLIPw): FCAND vi1, 0x000001 (0x12) -> still reads pre-CLIPw clip_flags (0x05 -> 1)
    //   Also DIV Q, vf0w(3), vf2x(0) (sub=0x38, fsf=3<<21, ftf=0<<23) -> new Q=0.25f after 7 cycles!
    const u32 div_q_lo = (0x40u << 25) | (0u << 23) | (3u << 21) | (2u << 16) | (0u << 11) | (0x0Eu << 6) | 0x3Cu;
    emit(6, div_q_lo, nop_hi);
    // 7 (d=1 after DIV): Upper MULq.xyzw vf3, vf4, Q (fn=0x1C) -> MUST read OLD Q=2.0f => vf3.x = 20.0f!
    const u32 mulq_vf3 = (0xFu << 21) | (0u << 16) | (4u << 11) | (3u << 6) | 0x1Cu;
    emit(7, nop_lo, mulq_vf3);
    // 8 (d=4 after SUBx, d=3 after CLIPw): FMAND vi7, vi6 (0x1A) -> sees delayed SUBx MAC Z flags (0x0E)!
    emit(8, (0x1Au << 25) | (7u << 16) | (6u << 11), nop_hi);
    // 9 (d=4 after CLIPw): FCGET vi8 (0x1C) -> sees shifted clip_flags = (0x05 << 6) | 0x01 = 0x0141!
    //   Also WAITQ (sub=0x3B) at 10
    emit(9, (0x1Cu << 25) | (8u << 16), nop_hi);
    const u32 waitq_lo = (0x40u << 25) | (0x0Eu << 6) | 0x3Fu; // sub=0x3B
    emit(10, waitq_lo, nop_hi);
    // 11: Upper MULq.xyzw vf5, vf4, Q -> after WAITQ, MUST read NEW Q=0.25f => vf5.x = 2.5f!
    //   Lower: IADDI vi2, vi0, 1 (fn=0x32, id=1) — 1-cycle VI ALU hazard before IBNE at 12!
    const u32 mulq_vf5 = (0xFu << 21) | (0u << 16) | (4u << 11) | (5u << 6) | 0x1Cu;
    const u32 iaddi_vi2 = (0x40u << 25) | (2u << 16) | (0u << 11) | (1u << 6) | 0x32u;
    emit(11, iaddi_vi2, mulq_vf5);
    // 12: IBNE vi0, vi2, +2 (op=0x29) -> reads PRE-ALU vi2==0, so branch MUST NOT be taken!
    const u32 ibne_lo = (0x29u << 25) | (2u << 16) | (0u << 11) | 2u;
    emit(12, ibne_lo, nop_hi);
    // 13: IADDIU vi9, vi0, 77 (executed because IBNE was NOT taken!)
    emit(13, op_it_is_imm15(0x08u, 9u, 0u, 77u), nop_hi);
    // 14: E-bit NOP
    emit(14, nop_lo, nop_hi | 0x40000000u);
    emit(15, nop_lo, nop_hi);

    vu1.run_microprogram(0u, 64u);

    TF_EXPECT_EQ(vu1.vi[3], u16{0x0005});   // FCSET + FCGET
    TF_EXPECT_EQ(vu1.vi[4], u16{1});        // FCAND
    TF_EXPECT_EQ(vu1.vi[5], u16{0x00C0});   // FSSET + FSAND
    TF_EXPECT_EQ(vu1.vi[7], u16{0x000E});   // 4-cycle delayed MAC -> FMAND
    TF_EXPECT_EQ(vu1.vi[8], u16{0x0141});   // 4-cycle delayed CLIPw -> FCGET
    TF_EXPECT(std::fabs(vu1.vf[3].x - 20.0f) < 1e-4f); // Q pipeline: old Q=2.0f before WAITQ
    TF_EXPECT(std::fabs(vu1.vf[5].x - 2.5f) < 1e-4f);  // Q pipeline: new Q=0.25f after WAITQ
    TF_EXPECT_EQ(vu1.vi[2], u16{1});        // IADDI still committed
    TF_EXPECT_EQ(vu1.vi[9], u16{77});       // 1-cycle VI branch hazard prevented premature branch
}

TF_TEST(RuntimeTests, GifReglistPreAndGsTexCacheVramInvalidation) {
    RuntimeHost rt;
    auto& gs = rt.gs;
    auto& gif = rt.gif;

    gs.write_internal_reg(0x4C, (10ULL << 16)); // FRAME_1 fbp=0, fbw=10

    // 1. REGLIST mode (flg=1) with PRE=1, PRIM=6 (Sprite): NLOOP=2, NREG=2 (RGBAQ=0x1, XYZ2=0x5)
    u8 rl_pkt[16 * 3]{};
    u64 rl_tag_lo = 2ULL | (1ULL << 15) | (1ULL << 46) | (6ULL << 47) | (1ULL << 58) | (2ULL << 60);
    u64 rl_tag_hi = 0x1ULL | (0x5ULL << 4);
    write_le64(rl_pkt + 0,  rl_tag_lo);
    write_le64(rl_pkt + 8,  rl_tag_hi);
    write_le64(rl_pkt + 16, 0x80112233ULL);                                    // RGBAQ
    write_le64(rl_pkt + 24, (10ULL << 4) | ((10ULL << 4) << 16));              // XYZ2 (10,10)
    write_le64(rl_pkt + 32, 0x80112233ULL);                                    // RGBAQ
    write_le64(rl_pkt + 40, (16ULL << 4) | ((16ULL << 4) << 16));              // XYZ2 (16,16)
    gif.feed_path(GifUnit::kPath3, rl_pkt, 3, gs);
    gs.present_fbp(0u);
    const u32 w = gs.fb_width();
    TF_EXPECT_EQ(gs.primitives_drawn, u64{1});
    TF_EXPECT_EQ(gs.framebuffer_rgba()[12 * w + 12] & 0x00FFFFFFu, 0x00112233u);

    // 2. GsTexCache hit + VRAM write invalidation coherence
    const u32 tbp = 512u; // page 16
    auto upload_solid_4x4 = [&](u32 rgba) {
        gs.write_internal_reg(0x50, (static_cast<u64>(tbp) << 32) | (1ULL << 48) | (0ULL << 56));
        gs.write_internal_reg(0x51, 0ULL);
        gs.write_internal_reg(0x52, 4ULL | (4ULL << 32));
        gs.write_internal_reg(0x53, 0ULL);
        u64 pair = static_cast<u64>(rgba) | (static_cast<u64>(rgba) << 32);
        for (int i = 0; i < 8; ++i) gs.write_internal_reg(0x54, pair);
    };
    auto draw_tex_sprite = [&](u32 dx) {
        u64 tex0 = static_cast<u64>(tbp) | (1ULL << 14) | (0ULL << 20) | (2ULL << 26) | (2ULL << 30) | (1ULL << 34) | (1ULL << 35);
        gs.write_internal_reg(0x06, tex0);
        gs.write_internal_reg(0x00, 6ULL | (1ULL << 4) | (1ULL << 8)); // Sprite + TME + FST
        gs.write_internal_reg(0x01, 0x80808080ULL);
        gs.write_internal_reg(0x03, 0ULL);
        gs.write_internal_reg(0x05, (static_cast<u64>(dx) << 4) | ((30ULL << 4) << 16));
        gs.write_internal_reg(0x03, (4ULL << 4) | ((4ULL << 4) << 16));
        gs.write_internal_reg(0x05, (static_cast<u64>(dx + 4u) << 4) | ((34ULL << 4) << 16));
    };

    upload_solid_4x4(0x8000AA11u);
    draw_tex_sprite(30u); // populates GsTexCache
    draw_tex_sprite(40u); // hits GsTexCache
    upload_solid_4x4(0x80BB2200u); // overwrites VRAM page -> must invalidate GsTexCache
    draw_tex_sprite(50u); // must sample updated color!
    gs.present_fbp(0u);

    TF_EXPECT_EQ(gs.framebuffer_rgba()[32 * w + 32] & 0x00FFFFFFu, 0x0000AA11u);
    TF_EXPECT_EQ(gs.framebuffer_rgba()[32 * w + 42] & 0x00FFFFFFu, 0x0000AA11u);
    TF_EXPECT_EQ(gs.framebuffer_rgba()[32 * w + 52] & 0x00FFFFFFu, 0x00BB2200u);
}

TF_TEST(RuntimeTests, GsTripleBufferedPresentedSeqAndCopyIfNewer) {
    RuntimeHost rt;
    auto& gs = rt.gs;

    const u64 seq0 = gs.presented_seq();
    gs.write_internal_reg(0x4C, (10ULL << 16)); // FRAME_1 fbp=0, fbw=10
    gs.write_internal_reg(0x00, 6ULL);         // Sprite
    gs.write_internal_reg(0x01, 0x80224466ULL);
    gs.write_internal_reg(0x05, (10ULL << 4) | ((10ULL << 4) << 16));
    gs.write_internal_reg(0x05, (20ULL << 4) | ((20ULL << 4) << 16));
    gs.present_fbp(0u);

    const u64 seq1 = gs.presented_seq();
    TF_EXPECT(seq1 > seq0);

    std::vector<u32> snap;
    u32 w = 0, h = 0;
    u64 seen_seq = seq0;
    TF_EXPECT(gs.copy_presented_framebuffer_if_newer(seen_seq, snap, w, h));
    TF_EXPECT_EQ(seen_seq, seq1);
    TF_EXPECT_EQ(w, gs.fb_width());
    TF_EXPECT_EQ(h, gs.fb_height());
    TF_EXPECT_EQ(snap[15u * w + 15u] & 0x00FFFFFFu, 0x00224466u);

    // Second call with unchanged seen_seq must return false immediately without re-copying
    TF_EXPECT(!gs.copy_presented_framebuffer_if_newer(seen_seq, snap, w, h));
    TF_EXPECT_EQ(seen_seq, seq1);

    // New frame increments seq and returns true with updated pixel color
    gs.write_internal_reg(0x01, 0x80998877ULL);
    gs.write_internal_reg(0x05, (10ULL << 4) | ((10ULL << 4) << 16));
    gs.write_internal_reg(0x05, (20ULL << 4) | ((20ULL << 4) << 16));
    gs.present_fbp(0u);
    TF_EXPECT(gs.copy_presented_framebuffer_if_newer(seen_seq, snap, w, h));
    TF_EXPECT(seen_seq > seq1);
    TF_EXPECT_EQ(snap[15u * w + 15u] & 0x00FFFFFFu, 0x00998877u);
}

TF_TEST(RuntimeTests, GsMultiBandBatchedTrianglesAndAteKeepEarlyZ) {
    RuntimeHost rt;
    auto& gs = rt.gs;

    gs.write_internal_reg(0x4C, (10ULL << 16)); // FRAME_1 fbp=0, fbw=10
    gs.write_internal_reg(0x4E, (200ULL << 0)); // ZBUF_1 zbp=200, PSMZ32
    // Seed Z=100, dark-blue background across 640x448 with ZTE=1, ZTST=ALWAYS
    gs.write_internal_reg(0x47, (1ULL << 16) | (1ULL << 17));
    gs.write_internal_reg(0x00, 6ULL);
    gs.write_internal_reg(0x01, 0x80102030ULL);
    gs.write_internal_reg(0x05, (0ULL << 4) | ((0ULL << 4) << 16) | (100ULL << 32));
    gs.write_internal_reg(0x05, (640ULL << 4) | ((448ULL << 4) << 16) | (100ULL << 32));

    // Enable ATE=1 (ATST=GEQUAL=5, AREF=0x40, AFAIL=KEEP=0) + ZTE=1 (ZTST=GEQUAL=2)
    const u64 test_ate_z = 1ULL | (5ULL << 1) | (0x40ULL << 4) | (0ULL << 12) | (1ULL << 16) | (2ULL << 17);
    gs.write_internal_reg(0x47, test_ate_z);
    gs.write_internal_reg(0x00, 3ULL); // PRIM = Triangle (batches across 4 horizontal bands when >= 20 tris)

    auto emit_tri = [&](u32 x0, u32 y0, u32 x1, u32 y1, u32 x2, u32 y2, u32 z, u32 rgba) {
        gs.write_internal_reg(0x01, rgba);
        gs.write_internal_reg(0x05, (static_cast<u64>(x0) << 4) | (static_cast<u64>(y0) << 20) | (static_cast<u64>(z) << 32));
        gs.write_internal_reg(0x05, (static_cast<u64>(x1) << 4) | (static_cast<u64>(y1) << 20) | (static_cast<u64>(z) << 32));
        gs.write_internal_reg(0x05, (static_cast<u64>(x2) << 4) | (static_cast<u64>(y2) << 20) | (static_cast<u64>(z) << 32));
    };

    // Emit 28 triangles spanning all 4 horizontal bands (y = 0..448):
    // - 14 passing triangles (Z=200 >= 100, A=0x80 >= 0x40) in green
    // - 7 alpha-rejected triangles (Z=300, A=0x20 < 0x40 -> must NOT overwrite color OR Z!)
    // - 7 Z-rejected triangles (Z=50 < 200, A=0x80 -> Early-Z rejects)
    for (u32 i = 0; i < 14; ++i) {
        u32 y0 = i * 30u + 4u;
        u32 y1 = y0 + 24u;
        emit_tri(20, y0, 120, y0, 20, y1, 200u, 0x8000CC00u);
    }
    for (u32 i = 0; i < 7; ++i) {
        u32 y0 = i * 60u + 4u;
        u32 y1 = y0 + 24u;
        emit_tri(20, y0, 120, y0, 20, y1, 300u, 0x200000FFu); // fails ATE -> KEEP
        emit_tri(20, y0, 120, y0, 20, y1, 50u,  0x80FF0000u); // fails Z -> Early-Z reject
    }
    gs.present_fbp(0u);

    const u32 w = gs.fb_width();
    // Verify pixels across all 4 vertical bands retained green (not overwritten by ATE-fail red or Z-fail blue)
    TF_EXPECT_EQ(gs.framebuffer_rgba()[10u  * w + 30u] & 0x00FFFFFFu, 0x0000CC00u);
    TF_EXPECT_EQ(gs.framebuffer_rgba()[130u * w + 30u] & 0x00FFFFFFu, 0x0000CC00u);
    TF_EXPECT_EQ(gs.framebuffer_rgba()[250u * w + 30u] & 0x00FFFFFFu, 0x0000CC00u);
    TF_EXPECT_EQ(gs.framebuffer_rgba()[370u * w + 30u] & 0x00FFFFFFu, 0x0000CC00u);
}

TF_TEST(RuntimeTests, Vu1PreDecodedSlotCacheCoherenceAcrossCodeOverwrite) {
    VU1Unit vu1;
    std::vector<u8> code_mem(16384, 0);
    std::vector<u8> data_mem(16384, 0);
    vu1.attach_memory(code_mem.data(), data_mem.data());

    const u32 nop_lo = 0x8000033Cu;
    const u32 nop_hi = 0x000002FFu;
    auto emit = [&](u32 idx, u32 lo, u32 hi) {
        write_le32(code_mem.data() + idx * 8 + 0, lo);
        write_le32(code_mem.data() + idx * 8 + 4, hi);
    };
    auto iaddiu = [](u32 it, u32 is, u32 imm15) {
        return (0x08u << 25) | (((imm15 >> 11) & 0xFu) << 21) | ((it & 0x1Fu) << 16) | ((is & 0x1Fu) << 11) | (imm15 & 0x7FFu);
    };

    // Program A: vi2 = vi0 + 11
    emit(0, iaddiu(2u, 0u, 11u), nop_hi | 0x40000000u);
    emit(1, nop_lo, nop_hi);
    vu1.run_microprogram(0u, 16u);
    TF_EXPECT_EQ(vu1.vi[2], u16{11});

    // Second run hits pre-decoded cache
    vu1.vi[2] = 0u;
    vu1.run_microprogram(0u, 16u);
    TF_EXPECT_EQ(vu1.vi[2], u16{11});

    // Overwrite slot 0 in-place with Program B: vi2 = vi0 + 42 -> raw64 tag MUST detect overwrite!
    emit(0, iaddiu(2u, 0u, 42u), nop_hi | 0x40000000u);
    vu1.run_microprogram(0u, 16u);
    TF_EXPECT_EQ(vu1.vi[2], u16{42});
}

TF_TEST(RuntimeTests, R5900FallbackDecodeCacheAndGuestMemoryInlineFastPath) {
    RuntimeHost rt;

    // 1. Inline GuestMemory fast-path round-trip (8/16/32/64/128-bit in EE RAM)
    const u32 base = 0x00260000u;
    rt.memory.write8(base + 0, 0x5Au);
    rt.memory.write16(base + 2, 0xBEEFu);
    rt.memory.write32(base + 4, 0x12345678u);
    rt.memory.write64(base + 8, 0x0123456789ABCDEFULL);
    rt.memory.write128(base + 16, u128(0xCAFEBABE11223344ULL, 0xDEADBEEF55667788ULL));
    TF_EXPECT_EQ(rt.memory.read8(base + 0), u8{0x5A});
    TF_EXPECT_EQ(rt.memory.read16(base + 2), u16{0xBEEF});
    TF_EXPECT_EQ(rt.memory.read32(base + 4), 0x12345678u);
    TF_EXPECT_EQ(rt.memory.read64(base + 8), 0x0123456789ABCDEFULL);
    u128 q = rt.memory.read128(base + 16);
    TF_EXPECT_EQ(q.lo, 0xCAFEBABE11223344ULL);
    TF_EXPECT_EQ(q.hi, 0xDEADBEEF55667788ULL);

    // 2. Fallback interpreter loop (sum 1..10 -> v0 = 55) exercising 64K decode cache
    // 0x00270000: addiu $v0, $zero, 0     (0x24020000)
    // 0x00270004: addiu $t0, $zero, 10    (0x2408000A)
    // 0x00270008: addu  $v0, $v0, $t0     (0x00481021)
    // 0x0027000C: addiu $t0, $t0, -1      (0x2508FFFF)
    // 0x00270010: bne   $t0, $zero, -3    (0x1500FFFD) -> branch back to 0x00270008
    // 0x00270014: nop                     (0x00000000)
    // 0x00270018: jr    $ra               (0x03E00008)
    // 0x0027001C: nop                     (0x00000000)
    const u32 fn = 0x00270000u;
    rt.memory.write32(fn + 0x00, 0x24020000u);
    rt.memory.write32(fn + 0x04, 0x2408000Au);
    rt.memory.write32(fn + 0x08, 0x00481021u);
    rt.memory.write32(fn + 0x0C, 0x2508FFFFu);
    rt.memory.write32(fn + 0x10, 0x1500FFFDu);
    rt.memory.write32(fn + 0x14, 0x00000000u);
    rt.memory.write32(fn + 0x18, 0x03E00008u);
    rt.memory.write32(fn + 0x1C, 0x00000000u);

    rt.call_guest(fn);
    TF_EXPECT_EQ(rt.cpu.gpr_u32(2), 55u);
    TF_EXPECT(rt.fallback_steps_executed >= 30u);
}

TF_TEST(RuntimeTests, GsPresentAndFramePacingCallbackOrdering) {
    RuntimeHost rt;
    auto& gs = rt.gs;

    std::vector<int> events;
    gs.set_present_callback([&]() { events.push_back(1); });
    gs.set_frame_pacing_callback([&]() { events.push_back(2); });

    gs.write_internal_reg(0x4C, (10ULL << 16)); // FRAME_1 fbp=0, fbw=10
    gs.write_internal_reg(0x00, 6ULL);
    gs.write_internal_reg(0x01, 0x80336699ULL);
    gs.write_internal_reg(0x05, (4ULL << 4) | ((4ULL << 4) << 16));
    gs.write_internal_reg(0x05, (16ULL << 4) | ((16ULL << 4) << 16));
    gs.present_fbp(0u);

    // present_cb (1) MUST fire strictly BEFORE frame_pacing_cb (2)
    TF_EXPECT_EQ(events.size(), std::size_t{2});
    TF_EXPECT_EQ(events[0], 1);
    TF_EXPECT_EQ(events[1], 2);

    // reset() MUST NOT invoke either callback
    events.clear();
    gs.reset();
    TF_EXPECT_EQ(events.size(), std::size_t{0});

    gs.set_present_callback(nullptr);
    gs.set_frame_pacing_callback(nullptr);
}

TF_TEST(RuntimeTests, GsEightBandBatchFbmskFbaAndUntexturedFastPath) {
    RuntimeHost rt;
    auto& gs = rt.gs;

    // 1. Seed background with alpha = 0x55 across 640x448 (FBMSK = 0)
    gs.write_internal_reg(0x4C, (10ULL << 16)); // FRAME_1: fbp=0, fbw=10, fbmsk=0
    gs.write_internal_reg(0x00, 6ULL);
    gs.write_internal_reg(0x01, 0x55102030ULL);
    gs.write_internal_reg(0x05, 0ULL);
    gs.write_internal_reg(0x05, (640ULL << 4) | ((448ULL << 4) << 16));

    // 2. Enable FBMSK = 0xFF000000 (mask alpha writes, allow RGB writes) and FBA_1 = 1
    gs.write_internal_reg(0x4C, (10ULL << 16) | (0xFF000000ULL << 32));
    gs.write_internal_reg(0x4A, 1ULL); // FBA_1 = 1
    gs.write_internal_reg(0x00, 3ULL); // PRIM = Untextured Triangle (48 tris -> exercises 8-band pool + fast_flat_simd)

    auto emit_tri = [&](u32 x0, u32 y0, u32 x1, u32 y1, u32 x2, u32 y2, u32 rgba) {
        gs.write_internal_reg(0x01, rgba);
        gs.write_internal_reg(0x05, (static_cast<u64>(x0) << 4) | (static_cast<u64>(y0) << 20));
        gs.write_internal_reg(0x05, (static_cast<u64>(x1) << 4) | (static_cast<u64>(y1) << 20));
        gs.write_internal_reg(0x05, (static_cast<u64>(x2) << 4) | (static_cast<u64>(y2) << 20));
    };

    // Emit 48 untextured triangles covering y = 0..448 across all 8 bands (56 rows/band)
    for (u32 i = 0; i < 48; ++i) {
        u32 y0 = (i * 9u) + 2u;
        u32 y1 = std::min<u32>(y0 + 14u, 446u);
        u32 col = 0x20000000u | ((30u + i * 3u) << 0) | (0xA0u << 8) | (0x40u << 16);
        emit_tri(40, y0, 200, y0, 40, y1, col);
    }
    gs.present_fbp(0u);

    const u32 w = gs.fb_width();
    // Verify RGB written cleanly in all 8 bands (e.g. y = 8, 62, 118, 174, 230, 286, 342, 398)
    for (u32 band = 0; band < 8; ++band) {
        u32 py = band * 56u + 8u;
        u32 px = gs.framebuffer_rgba()[py * w + 50u];
        TF_EXPECT_EQ((px >> 8) & 0xFFu, 0xA0u);
        TF_EXPECT_EQ((px >> 16) & 0xFFu, 0x40u);
    }
}

TF_TEST(RuntimeTests, RuntimeHostOptimalFramePacingAndJitInputSync) {
    RuntimeHost rt;

    // Open PADMAN Port 0 DMA buffer at 0x00280000
    const u32 pad_dma_addr = 0x00280000u;
    u8 send[32]{};
    u8 recv[32]{};
    write_le32(send + 0, 1u); // PAD_RPCCMD_OPEN
    write_le32(send + 16, pad_dma_addr);
    TF_EXPECT_EQ(rt.iop.call_rpc(kSifSidPadMan, 1u, send, 24, recv, 24), 0);

    // Initially Square is NOT pressed in DMA buffer
    TF_EXPECT_EQ(rt.memory.read8(pad_dma_addr + 8 + 7), u8{0x00});

    // Enable 60 Hz pacing and register JIT input callback that presses Square right after pacing sleep
    rt.enable_frame_pacing = true;
    int jit_calls = 0;
    std::chrono::steady_clock::time_point present_tp{};
    std::chrono::steady_clock::time_point jit_tp{};
    rt.gs.set_present_callback([&]() { present_tp = std::chrono::steady_clock::now(); });
    rt.set_jit_input_callback([&]() {
        jit_tp = std::chrono::steady_clock::now();
        jit_calls++;
        rt.pad0.set_button(Ps2Button::Square, true);
    });

    // Trigger two presents: 1st anchors pacer, 2nd paces ~16.7ms then fires JIT input + sync_pad0_now()
    rt.gs.write_internal_reg(0x4C, (10ULL << 16));
    rt.gs.write_internal_reg(0x00, 6ULL);
    rt.gs.write_internal_reg(0x01, 0x80112233ULL);
    rt.gs.write_internal_reg(0x05, 0ULL);
    rt.gs.write_internal_reg(0x05, (16ULL << 4) | ((16ULL << 4) << 16));
    rt.gs.present_fbp(0u);
    rt.gs.present_fbp(0u);

    TF_EXPECT(jit_calls >= 2);
    TF_EXPECT(jit_tp >= present_tp);
    // Square digital bit (active-low) and pressure byte [7] must be synced immediately into EE RAM!
    u16 buttons_raw = rt.memory.read16(pad_dma_addr + 2);
    TF_EXPECT((static_cast<u16>(~buttons_raw) & static_cast<u16>(Ps2Button::Square)) != 0);
    TF_EXPECT_EQ(rt.memory.read8(pad_dma_addr + 8 + 7), u8{0xFF});

    rt.enable_frame_pacing = false;
    rt.set_jit_input_callback(nullptr);
    rt.gs.set_present_callback(nullptr);
}

TF_TEST(RuntimeTests, Vu1JitX64BitExactEquivalenceWithInterpreter) {
    auto run_with_mode = [](bool use_jit, bool with_mac_reader, std::vector<u8>& out_data, VU1Unit& out_vu, u64& out_xg_bytes) {
        std::vector<u8> code_mem(16384, 0);
        out_data.assign(16384, 0);
        out_vu.reset();
        out_vu.enable_jit = use_jit;
        out_vu.attach_memory(code_mem.data(), out_data.data());

        // Seed 4 input vertices at qwords 0..3
        auto put_vec = [&](u32 qw, f32 x, f32 y, f32 z, f32 w) {
            Vec4f v{x, y, z, w};
            std::memcpy(out_data.data() + qw * 16u, &v, 16);
        };
        put_vec(0,  1.5f, -2.0f, 3.0f, 1.0f);
        put_vec(1, -0.5f,  4.0f, 1.25f, 1.0f);
        put_vec(2,  8.0f,  0.5f, -1.5f, 1.0f);
        put_vec(3,  0.25f, 1.0f,  2.5f, 1.0f);

        // Seed a minimal EOP=1 GIFtag at qword 32 for XGKICK
        write_le64(out_data.data() + 32u * 16u + 0, 0ULL | (1ULL << 15)); // NLOOP=0, EOP=1
        write_le64(out_data.data() + 32u * 16u + 8, 0ULL);

        // 4x4 transform matrix in vf1..vf4
        out_vu.vf[1] = Vec4f{ 1.25f,  0.5f,  0.0f,  0.1f};
        out_vu.vf[2] = Vec4f{-0.25f,  1.5f,  0.2f,  0.05f};
        out_vu.vf[3] = Vec4f{ 0.0f,  -0.5f,  1.2f,  0.2f};
        out_vu.vf[4] = Vec4f{ 10.0f, 20.0f,  5.0f,  2.0f};
        out_vu.vi[5] = 0u;  // src qword ptr
        out_vu.vi[6] = 16u; // dst qword ptr
        out_vu.vi[3] = 4u;  // vertex count
        out_vu.vi[7] = 32u; // xgkick addr
        out_vu.vi[8] = 0u;  // accumulated clip hit count

        out_xg_bytes = 0;
        out_vu.set_xgkick_callback([&](const u8*, std::size_t n) { out_xg_bytes += n; });

        const u32 nop_lo = 0x8000033Cu;
        const u32 nop_hi = 0x000002FFu;
        auto emit = [&](u32 idx, u32 lo, u32 hi) {
            write_le32(code_mem.data() + idx * 8 + 0, lo);
            write_le32(code_mem.data() + idx * 8 + 4, hi);
        };
        auto upper_sp2 = [](u32 dest, u32 ft, u32 fs, u32 sub) {
            return ((dest & 0xFu) << 21) | ((ft & 0x1Fu) << 16) | ((fs & 0x1Fu) << 11) |
                   (((sub >> 2) & 0x1Fu) << 6) | (0x3Cu | (sub & 3u));
        };
        auto lower_sp2 = [](u32 dest, u32 it, u32 is, u32 sub) {
            return (0x40u << 25) | ((dest & 0xFu) << 21) | ((it & 0x1Fu) << 16) | ((is & 0x1Fu) << 11) |
                   (((sub >> 2) & 0x1Fu) << 6) | (0x3Cu | (sub & 3u));
        };
        auto op_it_is_imm15 = [](u32 op, u32 it, u32 is, u32 imm15) {
            return (op << 25) | (((imm15 >> 11) & 0xFu) << 21) | ((it & 0x1Fu) << 16) | ((is & 0x1Fu) << 11) | (imm15 & 0x7FFu);
        };

        // Vertex transform loop (slots 0..12):
        emit(0, lower_sp2(0xFu, 10u, 5u, 0x34u), nop_hi); // LQI vf10, (vi5++)
        emit(1, nop_lo, upper_sp2(0xFu, 10u, 1u, 0x18u)); // MULAx ACC, vf1, vf10x
        emit(2, nop_lo, upper_sp2(0xFu, 10u, 2u, 0x09u)); // MADDAy ACC, vf2, vf10y
        emit(3, nop_lo, upper_sp2(0xFu, 10u, 3u, 0x0Au)); // MADDAz ACC, vf3, vf10z
        emit(4, nop_lo, (0xFu << 21) | (10u << 16) | (4u << 11) | (11u << 6) | 0x0Bu); // MADDw vf11, vf4, vf10w
        const u32 div_q_lo = (0x40u << 25) | (3u << 23) | (3u << 21) | (11u << 16) | (0u << 11) | (0x0Eu << 6) | 0x3Cu;
        emit(5, div_q_lo, upper_sp2(0xEu, 11u, 11u, 0x1Fu)); // CLIPw vf11, vf11 | DIV Q, vf0.w, vf11.w
        emit(6, (0x40u << 25) | (0x0Eu << 6) | 0x3Fu, nop_hi); // WAITQ
        emit(7, (0x12u << 25) | 0x3Fu, (0xFu << 21) | (0u << 16) | (11u << 11) | (12u << 6) | 0x1Cu); // MULq | FCAND
        emit(8, (0x40u << 25) | (1u << 16) | (8u << 11) | (8u << 6) | 0x30u, upper_sp2(0xFu, 13u, 12u, 0x15u)); // FTOI4 | IADD
        emit(9, (0x40u << 25) | (3u << 16) | (3u << 11) | (0x1Fu << 6) | 0x32u, nop_hi); // IADDI vi3, vi3, -1
        emit(10, nop_lo, nop_hi);
        emit(11, (0x29u << 25) | (0u << 16) | (3u << 11) | (static_cast<u32>(-12) & 0x7FFu), nop_hi); // IBNE vi3, vi0, -12
        emit(12, lower_sp2(0xFu, 6u, 13u, 0x35u), nop_hi); // delay slot: SQI vf13, (vi6++)
        emit(13, lower_sp2(0u, 0u, 7u, 0x6Cu), nop_hi); // XGKICK vi7
        // 1-cycle VI ALU branch hazard test: IADDI vi4, vi0, 1 immediately followed by IBNE vi4, vi0, +2
        emit(14, (0x40u << 25) | (4u << 16) | (0u << 11) | (1u << 6) | 0x32u, nop_hi);
        emit(15, (0x29u << 25) | (0u << 16) | (4u << 11) | 2u, nop_hi);
        emit(16, op_it_is_imm15(0x08u, 9u, 0u, 99u), nop_hi); // IADDIU vi9, vi0, 99
        // Optional FSAND reader to exercise both dead-MAC-skip (!with_mac_reader) & live-MAC (with_mac_reader) modes
        emit(17, with_mac_reader ? op_it_is_imm15(0x16u, 10u, 0u, 0x0FFu) : nop_lo, nop_hi | 0x40000000u);
        emit(18, nop_lo, nop_hi);

        out_vu.mark_code_dirty();
        out_vu.run_microprogram(0u, 512u);
    };

    for (bool with_mac_reader : {false, true}) {
        VU1Unit vu_interp, vu_jit;
        std::vector<u8> data_interp, data_jit;
        u64 xg_interp = 0, xg_jit = 0;

        run_with_mode(false, with_mac_reader, data_interp, vu_interp, xg_interp);
        run_with_mode(true,  with_mac_reader, data_jit,    vu_jit,    xg_jit);

#if defined(_WIN32) && (defined(__x86_64__) || defined(_M_X64))
        TF_EXPECT(vu_jit.jit_blocks_compiled() > 0u);
#endif
        TF_EXPECT_EQ(vu_interp.vi[9], u16{99});
        TF_EXPECT_EQ(vu_jit.vi[9], u16{99});
        TF_EXPECT_EQ(vu_jit.xgkick_count, vu_interp.xgkick_count);
        TF_EXPECT_EQ(xg_jit, xg_interp);
        TF_EXPECT_EQ(vu_jit.clip_flags, vu_interp.clip_flags);
        TF_EXPECT_EQ(vu_jit.pc, vu_interp.pc);
        if (with_mac_reader) {
            TF_EXPECT_EQ(vu_jit.mac_flags, vu_interp.mac_flags);
            TF_EXPECT_EQ(vu_jit.status_flags, vu_interp.status_flags);
        }
        for (int i = 0; i < 16; ++i) {
            TF_EXPECT_EQ(vu_jit.vi[i], vu_interp.vi[i]);
        }
        for (int i = 0; i < 32; ++i) {
            u32 a[4], b[4];
            std::memcpy(a, &vu_interp.vf[i], 16);
            std::memcpy(b, &vu_jit.vf[i], 16);
            for (int c = 0; c < 4; ++c) {
                TF_EXPECT_EQ(b[c], a[c]);
            }
        }
        u32 acc_a[4], acc_b[4];
        std::memcpy(acc_a, &vu_interp.acc, 16);
        std::memcpy(acc_b, &vu_jit.acc, 16);
        for (int c = 0; c < 4; ++c) TF_EXPECT_EQ(acc_b[c], acc_a[c]);
        u32 qa = 0, qb = 0;
        std::memcpy(&qa, &vu_interp.q, 4);
        std::memcpy(&qb, &vu_jit.q, 4);
        TF_EXPECT_EQ(qb, qa);
        TF_EXPECT(data_jit == data_interp);
    }
}

TF_TEST(RuntimeTests, R5900InterpreterWorkerSpinLoopCooperativeYield) {
    RuntimeHost rt;
    const u32 fn_pc = 0x00102000u;
    const u32 flag_base = 0x00103000u; // +0: ready, +4: ack, +8: done
    rt.memory.write32(flag_base + 0u, 0u);
    rt.memory.write32(flag_base + 4u, 0u);
    rt.memory.write32(flag_base + 8u, 0u);

    // Pure-RAM spin-wait mimicking 0x00287018 / L_00287348:
    // pushes 160-byte stack frame, sets ready=1, spins until ack!=0, sets done=1, restores $sp, returns.
    const u32 code[] = {
        0x27BDFF60u, // addiu $sp, $sp, -160
        0x3C030010u, // lui   $v1, 0x0010
        0x34633000u, // ori   $v1, $v1, 0x3000
        0x24080001u, // addiu $t0, $zero, 1
        0xAC680000u, // sw    $t0, 0($v1)          ; ready = 1
        0x8C620004u, // loop: lw $v0, 4($v1)       ; poll ack
        0x1040FFFEu, // beq   $v0, $zero, loop     ; backward branch (-2)
        0x00000000u, // nop
        0xAC680008u, // sw    $t0, 8($v1)          ; done = 1
        0x27BD00A0u, // addiu $sp, $sp, 160
        0x03E00008u, // jr    $ra
        0x00000000u, // nop
    };
    for (std::size_t i = 0; i < sizeof(code) / sizeof(code[0]); ++i) {
        rt.memory.write32(fn_pc + static_cast<u32>(i * 4u), code[i]);
    }

    EEContext ctx{};
    ctx.reset();
    ctx.mem = &rt.memory;
    const u32 init_sp = 0x01FF0000u;
    const u32 ret_pc  = 0x00109990u;
    ctx.set_gpr_u32_sext(29, init_sp);
    ctx.set_gpr_u32_sext(31, ret_pc);

    struct YieldState {
        GuestMemory* mem = nullptr;
        u32 flag_base = 0;
        u32 yields = 0;
    } ys{&rt.memory, flag_base, 0u};

    auto yield_cb = [](void* raw) {
        auto* s = static_cast<YieldState*>(raw);
        s->yields++;
        if (s->yields == 12u) {
            // Simulate main thread observing ready==1, fading out, and flipping ack=1
            if (s->mem->read32(s->flag_base + 0u) == 1u) {
                s->mem->write32(s->flag_base + 4u, 1u);
            }
        }
    };

    // Intentionally pass a small max_steps (500) to verify cooperative-yield workers do NOT abort mid-function.
    u64 steps = R5900FallbackInterpreter::run_until_return_or_handoff(
        ctx, fn_pc, rt.function_table(), 500ULL, nullptr, 0u, 0u, yield_cb, &ys);

    TF_EXPECT(steps > 500ULL);
    TF_EXPECT(ys.yields >= 12u);
    TF_EXPECT_EQ(rt.memory.read32(flag_base + 0u), 1u);
    TF_EXPECT_EQ(rt.memory.read32(flag_base + 4u), 1u);
    TF_EXPECT_EQ(rt.memory.read32(flag_base + 8u), 1u);
    TF_EXPECT_EQ(ctx.gpr_u32(29), init_sp);
    TF_EXPECT_EQ(ctx.pc, ret_pc);
    TF_EXPECT(!ctx.halted);
}

TF_TEST(RuntimeTests, Vu1IswIswrWaitpAndXgkickDelay) {
    auto lower_sp2 = [](u32 dest, u32 it, u32 is, u32 sub) -> u32 {
        return (0x40u << 25) | ((dest & 0xFu) << 21) | ((it & 0x1Fu) << 16) | ((is & 0x1Fu) << 11) |
               (((sub >> 2) & 0x1Fu) << 6) | (0x3Cu | (sub & 3u));
    };
    auto lower_isw = [](u32 dest, u32 it, u32 is, s32 imm11) -> u32 {
        return (0x05u << 25) | ((dest & 0xFu) << 21) | ((it & 0x1Fu) << 16) | ((is & 0x1Fu) << 11) |
               (static_cast<u32>(imm11) & 0x7FFu);
    };
    const u32 nop_lo = 0x8000033Cu;
    const u32 nop_hi = 0x000002FFu;

    for (bool jit : {false, true}) {
        VU1Unit vu;
        std::vector<u8> code(16384u, 0u);
        std::vector<u8> data(16384u, 0u);
        vu.reset();
        vu.enable_jit = jit;
        vu.attach_memory(code.data(), data.data());

        u32 kicked_word0 = 0u;
        u32 kick_addr_seen = 0u;
        vu.set_xgkick_callback([&](const u8* ptr, std::size_t) {
            kick_addr_seen = static_cast<u32>(ptr - data.data());
            kicked_word0 = static_cast<u32>(ptr[0]) |
                           (static_cast<u32>(ptr[1]) << 8) |
                           (static_cast<u32>(ptr[2]) << 16) |
                           (static_cast<u32>(ptr[3]) << 24);
        });

        auto emit = [&](u32 slot, u32 lo, u32 hi) {
            std::memcpy(code.data() + slot * 8u, &lo, 4);
            std::memcpy(code.data() + slot * 8u + 4u, &hi, 4);
        };

        vu.vi[1] = 10u;     // base addr for ISW (+2 -> qword 12)
        vu.vi[5] = 0x1234u; // value for ISW
        vu.vi[2] = 20u;     // base addr for ISWR (qword 20)
        vu.vi[6] = 0x5678u; // value for ISWR
        vu.vi[10] = 30u;    // XGKICK addr (qword 30)
        vu.vi[11] = 0x8002u;// EOP|NLOOP=2 written in 1-cycle XGKICK delay slot!
        vu.vf[1] = Vec4f{4.0f, 0.0f, 0.0f, 1.0f};

        // 0: ISW.x vi5, 2(vi1)  -> data[12].x = 0x1234 (NOT data[vi5+2]!)
        emit(0, lower_isw(0x8u, 5u, 1u, 2), nop_hi);
        // 1: ISWR.y vi6, (vi2)  -> data[20].y = 0x5678 (NOT data[vi6]!)
        emit(1, lower_sp2(0x4u, 6u, 2u, 0x3Fu), nop_hi);
        // 2: ERCPR P, vf1.x     -> pending P = 1/4.0 = 0.25
        emit(2, lower_sp2(0x0u, 0u, 1u, 0x78u), nop_hi);
        // 3: WAITP (sub=0x7B)   -> flush P immediately (must NOT decode as ESIN!)
        emit(3, lower_sp2(0x0u, 0u, 0u, 0x7Bu), nop_hi);
        // 4: MFP.x vf2, P       -> vf2.x = 0.25
        emit(4, lower_sp2(0x8u, 2u, 0u, 0x64u), nop_hi);
        // 5: XGKICK vi10        -> delayed by 1 cycle!
        emit(5, lower_sp2(0x0u, 0u, 10u, 0x6Cu), nop_hi);
        // 6: ISW.x vi11, 0(vi10)-> executes BEFORE delayed XGKICK reads qword 30!
        emit(6, lower_isw(0x8u, 11u, 10u, 0), nop_hi | 0x40000000u); // E-bit
        emit(7, nop_lo, nop_hi);

        vu.mark_code_dirty();
        vu.run_microprogram(0u, 128u);

        auto rd32 = [&](u32 qw, u32 field) -> u32 {
            u32 v = 0;
            std::memcpy(&v, data.data() + qw * 16u + field * 4u, 4);
            return v;
        };
        TF_EXPECT_EQ(rd32(12u, 0u), 0x1234u);
        TF_EXPECT_EQ(rd32(20u, 1u), 0x5678u);
        TF_EXPECT(std::fabs(vu.vf[2].x - 0.25f) < 1e-5f);
        TF_EXPECT_EQ(vu.xgkick_count, u64{1});
        TF_EXPECT_EQ(kick_addr_seen, 30u * 16u);
        TF_EXPECT_EQ(kicked_word0, 0x8002u);
    }
}

TF_TEST(RuntimeTests, Vu1LqiSqiLqdSqdBranchHazardAndDivQLatency) {
    auto lower_sp2 = [](u32 dest, u32 it, u32 is, u32 sub) -> u32 {
        return (0x40u << 25) | ((dest & 0xFu) << 21) | ((it & 0x1Fu) << 16) | ((is & 0x1Fu) << 11) |
               (((sub >> 2) & 0x1Fu) << 6) | (0x3Cu | (sub & 3u));
    };
    auto op_it_is_imm11 = [](u32 op, u32 it, u32 is, s32 imm11) -> u32 {
        return (op << 25) | ((it & 0x1Fu) << 16) | ((is & 0x1Fu) << 11) | (static_cast<u32>(imm11) & 0x7FFu);
    };
    const u32 nop_lo = 0x8000033Cu;
    const u32 nop_hi = 0x000002FFu;

    for (bool jit : {false, true}) {
        VU1Unit vu;
        std::vector<u8> code(16384u, 0u);
        std::vector<u8> data(16384u, 0u);
        vu.reset();
        vu.enable_jit = jit;
        vu.attach_memory(code.data(), data.data());

        auto emit = [&](u32 slot, u32 lo, u32 hi) {
            write_le32(code.data() + slot * 8u + 0u, lo);
            write_le32(code.data() + slot * 8u + 4u, hi);
        };
        auto wr_qw_x = [&](u32 qw, f32 x) {
            Vec4f v{x, 0.0f, 0.0f, 0.0f};
            std::memcpy(data.data() + qw * 16u, &v, 16);
        };
        auto rd_qw_x = [&](u32 qw) -> f32 {
            f32 x = 0.0f;
            std::memcpy(&x, data.data() + qw * 16u, 4);
            return x;
        };

        // Seed data for LQI loop (qwords 10, 11) and LQD (qword 29)
        wr_qw_x(10u, 1.0f);
        wr_qw_x(11u, 2.0f);
        wr_qw_x(29u, 9.0f);

        vu.q = 2.0f;
        vu.vi[2] = 11u; // loop bound (exits when pre-LQI vi3 == 11)
        vu.vi[3] = 10u; // LQI loop pointer
        vu.vi[4] = 20u; // chained SQI pointer
        vu.vi[5] = 20u; // compare target == pre-chain vi4
        vu.vi[7] = 31u; // LQD/SQD pointer
        vu.vi[8] = 31u; // compare target == pre-SQD vi7
        vu.vf[2] = Vec4f{0.0f, 0.0f, 0.0f, 0.0f};
        vu.vf[3] = Vec4f{3.0f, 0.0f, 0.0f, 0.0f};
        vu.vf[4] = Vec4f{4.0f, 0.0f, 0.0f, 0.0f};
        vu.vf[5] = Vec4f{5.0f, 0.0f, 0.0f, 0.0f};
        vu.vf[10] = Vec4f{4.0f, 0.0f, 0.0f, 1.0f};
        vu.vf[11] = Vec4f{10.0f, 0.0f, 0.0f, 1.0f};

        // (a) LQI + IBNE self-loop (must read pre-LQI vi3 and exit after 2 iterations, never infinite-loop!)
        // 0: LQI.xyzw vf1, (vi3++)
        emit(0u, lower_sp2(0xFu, 1u, 3u, 0x34u), nop_hi);
        // 1: Upper ADDx.x vf2, vf2, vf1 | Lower IBNE vi3, vi2, -2 (target slot 0)
        emit(1u, op_it_is_imm11(0x29u, 2u, 3u, -2), (0x8u << 21) | (1u << 16) | (2u << 11) | (2u << 6) | 0x00u);
        // 2: delay slot NOP
        emit(2u, nop_lo, nop_hi);

        // (b) Chained SQI (vi4++); SQI (vi4++); IBNE vi4, vi5 -> IBNE must read pre-chain vi4 (20 == 20 -> NOT taken!)
        // 3: SQI.xyzw vf3, (vi4++)
        emit(3u, lower_sp2(0xFu, 4u, 3u, 0x35u), nop_hi);
        // 4: SQI.xyzw vf4, (vi4++)
        emit(4u, lower_sp2(0xFu, 4u, 4u, 0x35u), nop_hi);
        // 5: IBNE vi4, vi5, +2 (skip slot 7 if wrongly taken)
        emit(5u, op_it_is_imm11(0x29u, 5u, 4u, 2), nop_hi);
        // 6: delay slot NOP
        emit(6u, nop_lo, nop_hi);
        // 7: IADDIU vi6, vi0, 1 (proves chained IBNE was NOT taken)
        emit(7u, op_it_is_imm11(0x08u, 6u, 0u, 1), nop_hi);

        // (c) SQD + IBEQ (taken on pre-SQD vi7 == 31) and LQD + IBEQ (not taken on pre-LQD vi7 == 30)
        // 8: SQD.xyzw vf5, (--vi7) -> vi7: 31->30, old=31
        emit(8u, lower_sp2(0xFu, 7u, 5u, 0x37u), nop_hi);
        // 9: IBEQ vi7, vi8, +2 (target slot 12, skips slot 11)
        emit(9u, op_it_is_imm11(0x28u, 8u, 7u, 2), nop_hi);
        // 10: delay slot NOP
        emit(10u, nop_lo, nop_hi);
        // 11: IADDIU vi9, vi0, 99 (MUST be skipped!)
        emit(11u, op_it_is_imm11(0x08u, 9u, 0u, 99), nop_hi);
        // 12: LQD.xyzw vf6, (--vi7) -> vi7: 30->29, old=30
        emit(12u, lower_sp2(0xFu, 6u, 7u, 0x36u), nop_hi);
        // 13: IBEQ vi7, vi8, +2 (reads old=30 != 31 -> NOT taken)
        emit(13u, op_it_is_imm11(0x28u, 8u, 7u, 2), nop_hi);
        // 14: delay slot NOP
        emit(14u, nop_lo, nop_hi);
        // 15: IADDIU vi10, vi0, 42 (proves post-LQD IBEQ was NOT taken)
        emit(15u, op_it_is_imm11(0x08u, 10u, 0u, 42), nop_hi);

        // (d) DIV Q latency (7 cycles: m_q_delay=6 -> T+5 reads OLD Q=2.0f, T+6 reads NEW Q=0.25f)
        // 16 (T=0): DIV Q, vf0w(3), vf10x(0)
        emit(16u, (0x40u << 25) | (0u << 23) | (3u << 21) | (10u << 16) | (0u << 11) | (0x0Eu << 6) | 0x3Cu, nop_hi);
        // 17..20 (T+1..T+4): 4 NOPs
        for (u32 s = 17u; s <= 20u; ++s) emit(s, nop_lo, nop_hi);
        // 21 (T+5): Upper MULq.xyzw vf12, vf11, Q -> OLD Q=2.0f => vf12.x = 20.0f
        emit(21u, nop_lo, (0xFu << 21) | (0u << 16) | (11u << 11) | (12u << 6) | 0x1Cu);
        // 22 (T+6): Upper MULq.xyzw vf13, vf11, Q -> NEW Q=0.25f => vf13.x = 2.5f
        emit(22u, nop_lo, (0xFu << 21) | (0u << 16) | (11u << 11) | (13u << 6) | 0x1Cu);
        // 23: E-bit NOP + delay NOP
        emit(23u, nop_lo, nop_hi | 0x40000000u);
        emit(24u, nop_lo, nop_hi);

        vu.mark_code_dirty();
        const u32 steps = vu.run_microprogram(0u, 128u);
        TF_EXPECT(steps < 64u);

        TF_EXPECT_EQ(vu.vi[3], u16{12});
        TF_EXPECT(std::fabs(vu.vf[2].x - 3.0f) < 1e-5f);
        TF_EXPECT_EQ(vu.vi[4], u16{22});
        TF_EXPECT_EQ(vu.vi[6], u16{1});
        TF_EXPECT(std::fabs(rd_qw_x(20u) - 3.0f) < 1e-5f);
        TF_EXPECT(std::fabs(rd_qw_x(21u) - 4.0f) < 1e-5f);
        TF_EXPECT_EQ(vu.vi[7], u16{29});
        TF_EXPECT_EQ(vu.vi[9], u16{0});
        TF_EXPECT_EQ(vu.vi[10], u16{42});
        TF_EXPECT(std::fabs(rd_qw_x(30u) - 5.0f) < 1e-5f);
        TF_EXPECT(std::fabs(vu.vf[6].x - 9.0f) < 1e-5f);
        TF_EXPECT(std::fabs(vu.vf[12].x - 20.0f) < 1e-4f);
        TF_EXPECT(std::fabs(vu.vf[13].x - 2.5f) < 1e-4f);
    }
}

TF_TEST(RuntimeTests, GsScanmskEvenOddAndClearParity) {
    RuntimeHost rt;
    auto& gs = rt.gs;

    gs.write_internal_reg(0x4C, (10ULL << 16)); // FRAME_1 fbp=0, fbw=10
    // Clear 32x32 region to dark blue (0x10,0x20,0x30) with SCANMSK=0
    gs.write_internal_reg(0x22, 0ULL);
    gs.write_internal_reg(0x00, 6ULL); // Sprite
    gs.write_internal_reg(0x01, 0x80302010ULL);
    gs.write_internal_reg(0x05, 0ULL);
    gs.write_internal_reg(0x05, (32ULL << 4) | ((32ULL << 4) << 16));

    // SCANMSK=2: discard even Y, draw odd Y (sprite + triangle in green)
    gs.write_internal_reg(0x22, 2ULL);
    gs.write_internal_reg(0x00, 6ULL);
    gs.write_internal_reg(0x01, 0x8000CC00ULL);
    gs.write_internal_reg(0x05, (4ULL << 4) | ((4ULL << 4) << 16));
    gs.write_internal_reg(0x05, (12ULL << 4) | ((12ULL << 4) << 16));

    gs.write_internal_reg(0x00, 3ULL); // Triangle
    gs.write_internal_reg(0x01, 0x8000CC00ULL);
    gs.write_internal_reg(0x05, (16ULL << 4) | ((4ULL << 4) << 16));
    gs.write_internal_reg(0x05, (28ULL << 4) | ((4ULL << 4) << 16));
    gs.write_internal_reg(0x05, (16ULL << 4) | ((12ULL << 4) << 16));

    // SCANMSK=3: discard odd Y, draw even Y (sprite + triangle in red)
    gs.write_internal_reg(0x22, 3ULL);
    gs.write_internal_reg(0x00, 6ULL);
    gs.write_internal_reg(0x01, 0x80FF0000ULL);
    gs.write_internal_reg(0x05, (4ULL << 4) | ((16ULL << 4) << 16));
    gs.write_internal_reg(0x05, (12ULL << 4) | ((24ULL << 4) << 16));

    gs.write_internal_reg(0x00, 3ULL); // Triangle
    gs.write_internal_reg(0x01, 0x80FF0000ULL);
    gs.write_internal_reg(0x05, (16ULL << 4) | ((16ULL << 4) << 16));
    gs.write_internal_reg(0x05, (28ULL << 4) | ((16ULL << 4) << 16));
    gs.write_internal_reg(0x05, (16ULL << 4) | ((24ULL << 4) << 16));

    // Restore SCANMSK=0: draw all rows in white
    gs.write_internal_reg(0x22, 0ULL);
    gs.write_internal_reg(0x00, 6ULL);
    gs.write_internal_reg(0x01, 0x80FFFFFFULL);
    gs.write_internal_reg(0x05, (4ULL << 4) | ((24ULL << 4) << 16));
    gs.write_internal_reg(0x05, (12ULL << 4) | ((30ULL << 4) << 16));

    gs.present_fbp(0u);
    const u32 w = gs.fb_width();
    // SCANMSK=2 sprite & triangle: even rows keep background, odd rows draw green
    TF_EXPECT_EQ(gs.framebuffer_rgba()[6u  * w + 8u]  & 0x00FFFFFFu, 0x00302010u);
    TF_EXPECT_EQ(gs.framebuffer_rgba()[7u  * w + 8u]  & 0x00FFFFFFu, 0x0000CC00u);
    TF_EXPECT_EQ(gs.framebuffer_rgba()[6u  * w + 18u] & 0x00FFFFFFu, 0x00302010u);
    TF_EXPECT_EQ(gs.framebuffer_rgba()[7u  * w + 18u] & 0x00FFFFFFu, 0x0000CC00u);
    // SCANMSK=3 sprite & triangle: even rows draw red, odd rows keep background
    TF_EXPECT_EQ(gs.framebuffer_rgba()[18u * w + 8u]  & 0x00FFFFFFu, 0x00FF0000u);
    TF_EXPECT_EQ(gs.framebuffer_rgba()[19u * w + 8u]  & 0x00FFFFFFu, 0x00302010u);
    TF_EXPECT_EQ(gs.framebuffer_rgba()[18u * w + 18u] & 0x00FFFFFFu, 0x00FF0000u);
    TF_EXPECT_EQ(gs.framebuffer_rgba()[19u * w + 18u] & 0x00FFFFFFu, 0x00302010u);
    // SCANMSK=0: both even and odd rows draw white
    TF_EXPECT_EQ(gs.framebuffer_rgba()[26u * w + 8u]  & 0x00FFFFFFu, 0x00FFFFFFu);
    TF_EXPECT_EQ(gs.framebuffer_rgba()[27u * w + 8u]  & 0x00FFFFFFu, 0x00FFFFFFu);
}
