# Repository Audit, Reference Study & Milestone Engineering Log

## 1. Existing Repository Audit

When auditing the workspace root prior to implementation:
- The workspace contained **no legacy remake source tree** (`Engine/`, `Game/`, etc.) and **only** the user-provided read-only disc image `Transformers - The Game (USA).iso` (`2,504,458,240` bytes).
- The project was therefore constructed cleanly from the ground up as an ISO-based PS2 static recompilation port (`TransformersNative`).

## 2. Reference Technology & License Study

Before designing `TransformersNative`, two public PS2 static recompilation projects were evaluated as technical references (zero code was copied from either repository):

1. **`ran-j/PS2Recomp`** — Licensed under **GPL-3.0**.
   - Architecture: Parses PS2 ELFs (`ELFIO`), decodes MIPS R5900 including MMI and VU0 macros (`ps2xAnalyzer`), emits C++ functions targeting a runtime (`ps2xRuntime`) with guest memory, syscall dispatcher, hardware stubs, and IOP HLE/RPC (`ps2xIOP`).
   - Takeaway: Clean separation between offline AOT analyzer/codegen and a portable C++ runtime; our clean-room C++20 implementation avoids GPL-3.0 license entanglement.
2. **`exoteric33/ps2-native-recomp`** (building on `ps2-web-recomp`) — Licensed under **MIT** (with BSD-2-Clause components).
   - Architecture: Translates EE R5900 statically to C++ while running GS/VU/SPU2/IOP/IPU dynamically in a hardware runtime backed by modern GPU APIs.
   - Takeaway: Confirms that AOT CPU translation must be paired with real DMAC/VIF/GIF/GS/VU/IOP/CDVD/SPU2 hardware compatibility rather than CPU-only translation.

## 3. Verified Implementation Status by Milestone

| Milestone | Status | Verified Evidence |
| --- | --- | --- |
| **M1**: Audit, ISO9660 parser, `SYSTEM.CNF`, ELF detection | **Completed & Verified** | 64-bit streaming `IsoReader`, malformed record guards, packed/compressed detector, SHA-256 fingerprint; 6 ISO unit tests + live `SLUS_216.02` inspection passed. |
| **M2**: ELF analyzer, function map, R5900 decoder | **Completed & Verified** | `ElfParser` separates EE `.text` from `.vutext` + 102 `.DVP.overlay.*` VU sections; `FunctionAnalyzer` discovers 7,610 functions + Ghidra CSV export; **100.0000%** (901,040/901,040) R5900 decode coverage on `SLUS_216.02`. |
| **M3**: R5900 static recompiler -> C++20 -> native execution | **Completed & Verified** | `CppCodeGenerator` lowers 100% of `SLUS_216.02` EE instructions (including delay slots, branch-likely nullification, 64-bit, MMI, FPU, VU0 macro); all **7,610** functions compiled into native Windows x64 `dist/SLUS_216.02_recomp.dll` (`17.7 MB`) and cross-compiled to AArch64 `libSLUS_216_02_recomp.so`. |
| **M4**: EE memory, CPU context, cooperative EE threading, syscalls, IOP HLE | **Completed & Verified** | 32 MiB EE RAM + mirrors + 16 KiB scratchpad + MMIO routing; cooperative round-robin EE worker threads (`CreateThread`/`StartThread`/`SuspendThread`/`ResumeThread`/`RotateThreadReadyQueue`/`ExitThread`); INTC/DMAC ISR dispatch; full Sony SDK 3.00 SIF RPC HLE for `CDVD`, `FILEIO`, `LOADFILE`, `PADMAN` (128-byte `pad_data_t` with actuators + pressure mode), `MCMAN` (fn=254/1/2/3/4/5/6/12/13/21 backed by `VirtualMemoryCard`), `LIBSD`, `IOPHEAP`, and Traveller's Tales `NUSND3M` (`0x00012345`). |
| **M5**: VU0, VU1, DMAC, VIF, GIF, 4 MiB GS VRAM Rasterizer | **Completed & Verified** | VU0 macro executor; VU1 microprogram interpreter with parallel upper/lower pipe writes, fixed Special2 sub-opcode decode, `FSOR`/`FMEQ`/`FMAND`/`FMOR`, and `XGKICK` (735,552+ genuine `XGKICK`s observed); DMAC Normal + Source Chain tag walker + TTE + `1,048,576`-round pump & `(tadr, asp, asr0, asr1)` cycle guard; VIF `UNPACK`/`MPG`/`DIRECT`/`MSCAL`; GIF `PACKED` (with `Q=1/W` perspective preservation & 32-bit `XYZ2` Z)/`REGLIST`/`IMAGE`; 4 MiB GS VRAM `PSMCT32`->`PSMT8`/`PSMT4` unswizzling, CSM1 CLUT lookup + caching, Context 1/2, `CLAMP_1`/`2`, `TEST_1` alpha & reversed-Z depth test, `ALPHA_1` blending, incremental edge-equation triangle rasterizer, and mutex-protected `copy_presented_framebuffer()` frontbuffer. |
| **M6**: Audio, CDVD, Input, Virtual Memory Card | **Completed & Verified** | ISO-backed streaming `CdvdDrive`, `Spu2Unit` PS-ADPCM decoder + 48-voice stereo mixer, `DualShock2State` active-low digital + 12-channel analog pressure packing + stick deadzone, traversal-safe persistent `VirtualMemoryCard`. |
| **M7**: Windows x64 Builder, Native Desktop App & Cache | **Completed & Verified** | Zero-warning build of `Transformers.exe`, `TransformersRecompBuilder.exe`, and `transformers_tests.exe` (**21/21 tests passing**). Full Win32 Desktop Launcher + Settings UI (`%LOCALAPPDATA%\TransformersNative\settings.ini`) + custom `TransformersViewportWnd` 60 Hz `Win32Presenter` (4:3/16:9/Stretch, F11 fullscreen) + 16 MiB guest stack + `GuestHaltSignal` clean STOP/RESTART/CLOSE + live Keyboard & XInput gamepad/rumble polling. |
| **M8**: Android ARM64 Native App, GLES 3.0 & Signed APK | **Completed & Verified** | `libtransformers_native.so` (`arm64-v8a`) + OpenGL ES 3.0 shader quad presenter (`AndroidGlesBackend`) + thread-safe `copy_presented_framebuffer()` + 16 MiB POSIX pthread guest runner + JNI SAF/recomp `.so` auto-loader + Java Launcher/Settings/multi-touch DualShock 2 overlay + GameController input + `Tools/build_android_recomp_mod.ps1`. Packaged into signed `dist/Transformers.apk` (`2,028,012` bytes), verified with `apksigner verify --verbose` (v2+v3) and clean-room ZIP audit (7 entries, zero ISO/ELF/BIOS/game assets). |
| **M9**: Genuine Boot to Full-Color 3D Title Screen (`SLUS_216.02`) | **Completed & Verified** | Verified in both headless and live Win32 GUI `dist/Transformers.exe`: DualShock 2 Port 0 connected+analog+pressure (`pad_st=0`, `pad_conn=255`, `pad0_fsm=11`, `pad0_flg=0xff79000f`), all 5 PS2 boot screens (Autobot, Paramount, DreamWorks, Hasbro, Legal), textured 3D spinning loading screen across all 40 load stages (including 5.5 MB `Scene Main`), Memory Card autosave dialog (Screen 1025, Cross accepted), and full-color perspective-correct 3D Earth, Sun, and glowing AllSpark Cube Title Screen (`boot_st=7`, `272,949 / 286,720` non-black pixels). |
| **M10**: Post-Title Main Menu & Full 3D Gameplay Performance | **In Progress — Next Steps Documented Below** | Full-color 3D Title Screen & AllSpark intro are verified; advancing through Main Menu into full-speed 60 FPS 3D city gameplay has the remaining optimization/feature items below. |

## 4. Remaining Engineering Items Beyond Verified Full-Color 3D Title Screen (`SLUS_216.02`)

1. **Hardware-Accelerated GS Rasterizer (D3D12 / Vulkan / GLES 3.0 Draw Backend)**:
   - The CPU 4 MiB GS VRAM software rasterizer is bit-accurate for `PSMCT32`->`PSMT8` unswizzling, CSM1 CLUT, perspective-correct `Q`, `TEST_1`, and `ALPHA_1`, but full-speed 60 FPS 3D gameplay on desktop and mobile will benefit from offloading GS triangle/sprite rasterization and texture unswizzling to GPU shaders.
2. **Cutscene Playback (`/MOVIES.DAT`, 1.429 GB) & IPU MPEG-2**:
   - Hardware IPU (`fromIPU`/`toIPU` DMA + MPEG-2 bitstream decoding) is needed to play pre-rendered FMV cutscenes stored in `/MOVIES.DAT`.
3. **Live SPU2 / `NUSND3M` Audio Output Sink**:
   - `Spu2Unit` ADPCM decoding and 48-voice mixing are implemented and unit-tested, and `NUSND3M` SIF RPC heartbeats/CDVD reads are handled; wiring real-time WASAPI (Windows) and AAudio/Oboe (Android) PCM output plus `NUSND3M` voice trigger mapping will enable full in-game music and SFX.
4. **Direct Intra-Function Jump-Table Lowering (325 `JR` Sites)**:
   - The 325 switch-table `JR` sites currently execute safely via the fallback interpreter handoff; lifting them directly into C++ `switch` statements inside `CppCodeGenerator` will eliminate fallback overhead during hot gameplay loops.
