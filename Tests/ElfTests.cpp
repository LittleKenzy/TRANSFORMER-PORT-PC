#include "TestFramework.h"
#include "../Recompiler/ELF/ElfParser.h"
#include "../Recompiler/ISO/IsoReader.h"
#include "../Recompiler/Analyzer/FunctionAnalyzer.h"

#include <filesystem>

using namespace tf::ps2;

TF_TEST(ElfTests, RejectInvalidElfHeaders) {
    Ps2ElfImage elf;
    std::string err;

    std::vector<u8> bad_magic(64, 0);
    TF_EXPECT(!ElfParser::parse(bad_magic, elf, err));

    std::vector<u8> wrong_machine(64, 0);
    wrong_machine[0] = 0x7F; wrong_machine[1] = 'E'; wrong_machine[2] = 'L'; wrong_machine[3] = 'F';
    wrong_machine[4] = 1; wrong_machine[5] = 1;
    write_le16(wrong_machine.data() + 18, 62); // x86_64 instead of EM_MIPS
    TF_EXPECT(!ElfParser::parse(wrong_machine, elf, err));
}

TF_TEST(ElfTests, ParseRealTransformersElfIfPresent) {
    namespace fs = std::filesystem;
    fs::path real_iso = fs::path(TF_WORKSPACE_ROOT) / "Transformers - The Game (USA).iso";
    if (!fs::exists(real_iso)) return;

    IsoReader iso;
    std::string err;
    TF_EXPECT(iso.open(real_iso.string(), err));
    auto rep = iso.inspect();
    TF_EXPECT(rep.valid);
    TF_EXPECT(rep.executable_format == ExecutableFormat::Ps2Elf32Mips);

    std::vector<u8> elf_bytes;
    TF_EXPECT(iso.read_file(rep.boot_executable_entry, elf_bytes));

    Ps2ElfImage elf;
    TF_EXPECT(ElfParser::parse(elf_bytes, elf, err));
    TF_EXPECT(elf.valid);
    TF_EXPECT(elf.e_entry != 0);
    TF_EXPECT(!elf.executable_regions.empty());

    auto analysis = FunctionAnalyzer::analyze(elf);
    TF_EXPECT(!analysis.functions.empty());
    std::cout << "  [LIVE ELF] Entry=" << hex32(elf.e_entry)
              << " Functions=" << analysis.functions.size()
              << " InsnDecoded=" << analysis.coverage.decoded_instructions << "/" << analysis.coverage.total_instructions
              << " (" << analysis.coverage.coverage_percent() << "%)\n";
}
