#include "TestFramework.h"
#include "../Recompiler/ISO/IsoReader.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

namespace {

using namespace tf::ps2;

void write_dir_record(std::vector<u8>& buf, std::size_t& pos, u32 lba, u32 size, u8 flags, const std::string& name) {
    u8 rec_len = static_cast<u8>(33 + name.size() + ((name.size() % 2 == 0) ? 1 : 0));
    if (pos + rec_len > buf.size()) return;
    u8* r = buf.data() + pos;
    r[0] = rec_len;
    write_le32(r + 2, lba);
    write_le32(r + 10, size);
    r[25] = flags;
    r[32] = static_cast<u8>(name.size());
    std::memcpy(r + 33, name.data(), name.size());
    pos += rec_len;
}

std::vector<u8> make_minimal_ps2_elf_bytes() {
    std::vector<u8> elf(128, 0);
    elf[0] = 0x7F; elf[1] = 'E'; elf[2] = 'L'; elf[3] = 'F';
    elf[4] = 1; // ELFCLASS32
    elf[5] = 1; // ELFDATA2LSB
    elf[6] = 1; // EV_CURRENT
    write_le16(elf.data() + 16, 2); // ET_EXEC
    write_le16(elf.data() + 18, 8); // EM_MIPS
    write_le32(elf.data() + 20, 1);
    write_le32(elf.data() + 24, 0x00100000u); // entry
    write_le32(elf.data() + 28, 52);          // phoff
    write_le16(elf.data() + 40, 52);
    write_le16(elf.data() + 42, 32);
    write_le16(elf.data() + 44, 1);           // phnum
    // PT_LOAD at offset 96, vaddr 0x00100000, size 16
    write_le32(elf.data() + 52 + 0, 1);
    write_le32(elf.data() + 52 + 4, 96);
    write_le32(elf.data() + 52 + 8, 0x00100000u);
    write_le32(elf.data() + 52 + 12, 0x00100000u);
    write_le32(elf.data() + 52 + 16, 16);
    write_le32(elf.data() + 52 + 20, 16);
    write_le32(elf.data() + 52 + 24, 5); // PF_R | PF_X
    // JR $ra; NOP
    write_le32(elf.data() + 96, 0x03E00008u);
    write_le32(elf.data() + 100, 0x00000000u);
    return elf;
}

std::string write_synthetic_iso(
    const std::string& filename,
    bool include_sys_cnf,
    bool include_elf,
    bool inject_malformed_record,
    const std::vector<u8>& elf_payload
) {
    namespace fs = std::filesystem;
    fs::path p = fs::temp_directory_path() / filename;

    // 25 sectors = 51,200 bytes
    std::vector<u8> img(25 * kIsoSectorSize, 0);

    // Sector 16: Primary Volume Descriptor
    u8* pvd = img.data() + 16 * kIsoSectorSize;
    pvd[0] = 1;
    std::memcpy(pvd + 1, "CD001", 5);
    pvd[6] = 1;
    std::memcpy(pvd + 8, "PLAYSTATION", 11);
    std::memcpy(pvd + 40, "TF_SYNTH_TEST", 13);
    write_le32(pvd + 80, 25);
    write_le16(pvd + 128, kIsoSectorSize);
    // Root directory record at PVD+156 -> LBA 18, size 2048
    pvd[156] = 34;
    write_le32(pvd + 156 + 2, 18);
    write_le32(pvd + 156 + 10, kIsoSectorSize);
    pvd[156 + 25] = 0x02;
    pvd[156 + 32] = 1;

    // Sector 17: Volume Descriptor Set Terminator
    u8* term = img.data() + 17 * kIsoSectorSize;
    term[0] = 255;
    std::memcpy(term + 1, "CD001", 5);

    // Sector 18: Root Directory
    std::vector<u8> root_dir(kIsoSectorSize, 0);
    std::size_t pos = 0;
    write_dir_record(root_dir, pos, 18, kIsoSectorSize, 0x02, std::string("\x00", 1));
    write_dir_record(root_dir, pos, 18, kIsoSectorSize, 0x02, std::string("\x01", 1));

    const std::string cnf = "BOOT2 = cdrom0:\\SLUS_999.01;1\r\nVER = 1.00\r\nVMODE = NTSC\r\n";
    if (include_sys_cnf) {
        write_dir_record(root_dir, pos, 19, static_cast<u32>(cnf.size()), 0x00, "SYSTEM.CNF;1");
        std::memcpy(img.data() + 19 * kIsoSectorSize, cnf.data(), cnf.size());
    }
    if (include_elf) {
        write_dir_record(root_dir, pos, 20, static_cast<u32>(elf_payload.size()), 0x00, "SLUS_999.01;1");
        std::memcpy(img.data() + 20 * kIsoSectorSize, elf_payload.data(), elf_payload.size());
    }
    if (inject_malformed_record && pos + 8 < root_dir.size()) {
        root_dir[pos] = 12; // Invalid rec_len < 33
    }
    std::memcpy(img.data() + 18 * kIsoSectorSize, root_dir.data(), kIsoSectorSize);

    std::ofstream ofs(p, std::ios::binary | std::ios::trunc);
    ofs.write(reinterpret_cast<const char*>(img.data()), static_cast<std::streamsize>(img.size()));
    return p.string();
}

} // namespace

TF_TEST(IsoTests, ValidSyntheticIsoAndStreaming) {
    auto elf = make_minimal_ps2_elf_bytes();
    std::string path = write_synthetic_iso("tf_valid_test.iso", true, true, false, elf);

    IsoReader iso;
    std::string err;
    TF_EXPECT(iso.open(path, err));
    auto rep = iso.inspect();
    TF_EXPECT(rep.valid);
    TF_EXPECT(rep.has_system_cnf);
    TF_EXPECT(rep.has_boot_executable);
    TF_EXPECT_EQ(rep.system_cnf.serial, std::string("SLUS_999.01"));
    TF_EXPECT_EQ(rep.system_cnf.region, std::string("USA (NTSC-U)"));
    TF_EXPECT(rep.executable_format == ExecutableFormat::Ps2Elf32Mips);

    // Streaming API test
    auto stream = iso.open_stream("/SLUS_999.01");
    TF_EXPECT(stream != nullptr);
    TF_EXPECT_EQ(stream->size(), elf.size());
    u8 hdr[4] = {};
    TF_EXPECT_EQ(stream->read(hdr, 4), std::size_t{4});
    TF_EXPECT_EQ(hdr[0], u8{0x7F});
    TF_EXPECT_EQ(hdr[1], u8{'E'});
}

TF_TEST(IsoTests, MissingSystemCnfFailsGracefully) {
    auto elf = make_minimal_ps2_elf_bytes();
    std::string path = write_synthetic_iso("tf_nocnf_test.iso", false, true, false, elf);

    IsoReader iso;
    std::string err;
    TF_EXPECT(iso.open(path, err));
    auto rep = iso.inspect();
    TF_EXPECT(!rep.valid);
    TF_EXPECT(!rep.has_system_cnf);
}

TF_TEST(IsoTests, MissingBootExecutableFailsGracefully) {
    auto elf = make_minimal_ps2_elf_bytes();
    std::string path = write_synthetic_iso("tf_noelf_test.iso", true, false, false, elf);

    IsoReader iso;
    std::string err;
    TF_EXPECT(iso.open(path, err));
    auto rep = iso.inspect();
    TF_EXPECT(!rep.valid);
    TF_EXPECT(rep.has_system_cnf);
    TF_EXPECT(!rep.has_boot_executable);
}

TF_TEST(IsoTests, MalformedDirectoryRecordHandledSafely) {
    auto elf = make_minimal_ps2_elf_bytes();
    std::string path = write_synthetic_iso("tf_malformed_dir_test.iso", true, true, true, elf);

    IsoReader iso;
    std::string err;
    TF_EXPECT(iso.open(path, err));
    auto rep = iso.inspect();
    TF_EXPECT(rep.valid);
    TF_EXPECT(!rep.warnings.empty());
}

TF_TEST(IsoTests, CompressedAndPackedExecutableDetection) {
    std::string details;
    std::vector<u8> gz(32, 0); gz[0] = 0x1F; gz[1] = 0x8B;
    TF_EXPECT(IsoReader::detect_executable_format(gz, details) == ExecutableFormat::CompressedGzip);

    std::vector<u8> zl(32, 0); zl[0] = 0x78; zl[1] = 0x9C;
    TF_EXPECT(IsoReader::detect_executable_format(zl, details) == ExecutableFormat::CompressedZlib);

    auto packed = make_minimal_ps2_elf_bytes();
    std::memcpy(packed.data() + 104, "PS2-PACKER", 10);
    TF_EXPECT(IsoReader::detect_executable_format(packed, details) == ExecutableFormat::PackedPs2Packer);
}

TF_TEST(IsoTests, RealTransformersPs2IsoIfPresent) {
    namespace fs = std::filesystem;
    fs::path real_iso = fs::path(TF_WORKSPACE_ROOT) / "Transformers - The Game (USA).iso";
    if (!fs::exists(real_iso)) {
        std::cout << "  [INFO] Skipping live ISO test (ISO not present at " << real_iso.string() << ")\n";
        return;
    }

    IsoReader iso;
    std::string err;
    TF_EXPECT(iso.open(real_iso.string(), err));
    auto rep = iso.inspect();
    TF_EXPECT(rep.valid);
    TF_EXPECT(rep.has_system_cnf);
    TF_EXPECT(rep.has_boot_executable);
    TF_EXPECT(rep.iso_file_size > 2000000000ULL);
    std::cout << "  [LIVE ISO] Serial=" << rep.system_cnf.serial
              << " Region=" << rep.system_cnf.region
              << " BootELF=" << rep.boot_executable_entry.path
              << " (" << rep.boot_executable_entry.size << " bytes, "
              << IsoReader::executable_format_name(rep.executable_format) << ")\n";
}
