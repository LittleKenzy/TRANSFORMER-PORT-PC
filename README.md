# TRANSFORMER-PORT-PC — Transformers: The Game (PS2) Native Windows x64 Port

A native Windows x64 static recompiler and PlayStation 2 hardware-compatibility runtime for **Transformers: The Game (USA)** (`SLUS_216.02`).

> **Legal Notice**: This repository contains **zero** copyrighted game assets, executables, BIOS/firmware, or disc images. You must supply your own legally dumped PlayStation 2 ISO of *Transformers: The Game* and generate the recompiled native module locally using `TransformersRecompBuilder.exe` or the `Transformers.exe` Desktop UI. Your ISO is accessed strictly **read-only**.

---

## Verified PC Features & Architecture

```
User PS2 ISO (read-only)
   │
   ├─► ISO9660 + ELF32 + R5900 Static Recompiler (Recompiler/ + Builder/)
   │     └─► Translates all 7,610 EE functions (901,040 / 901,040 R5900 insns) -> C++20 DLL
   │
   └─► Native Windows x64 Runtime (Runtime/ + Graphics/Direct3D12/ + Platform/Windows/)
         ├─► EE Memory, Cooperative EE Threads, Syscalls, DMAC, VIF, GIF, IOP HLE, SPU2, DS2, VMC
         ├─► Windows x64 SSE2 VU1 JIT (Vu1JitX64) + Cooperative R5900FallbackInterpreter
         └─► D3D11 Hardware GS Renderer (1x–4x HD) ──► DXGI Flip-Model Presenter
```

### Direct3D 11 Hardware GS Renderer (`D3D11 HW GS + Flip`)
- **Native 1×–4× HD Internal Resolution Scaling**: GPU-resident render targets and depth buffers with automatic CPU software GS fallback (`Runtime/GS/GsState.cpp`, `Graphics/Direct3D12/D3D11GsRenderer.cpp`).
- **Accurate PS2 Blending & Alpha Test**: GPU 2-pass `AFAIL` handling (`FB_ONLY`, `ZB_ONLY`, `RGB_ONLY`), dual-source PS2 alpha blending (`D3D11_BLEND_SRC1_ALPHA` / `D3D11_BLEND_INV_SRC1_ALPHA`), and logical `SCANMSK` parity masking.
- **GPU Depth Fog (`PSMZ16 -> PSMT8H`)**: Sampleable GPU Z-buffer (`DXGI_FORMAT_R32_TYPELESS` with `D32_FLOAT` DSV / `R32_FLOAT` SRV) and 256×1 GPU CLUT LUT for clean, hardware-accelerated depth-to-CLUT atmospheric fog.

### Ultra-Low-Latency Windows Presentation
- **DXGI Flip-Model**: `DXGI_SWAP_EFFECT_FLIP_DISCARD`, `SetMaximumFrameLatency(1)`, and `DXGI_PRESENT_ALLOW_TEARING` immediate presentation when VSync is disabled (`Win32Presenter.cpp`).
- **Event-Driven Wakeup & Frame Pacing**: Win32 event (`CreateEventA` / `MsgWaitForMultipleObjects`) presentation wakeup, Optimal 60 Hz Frame Pacing, and Just-In-Time (JIT) Input sampled right before PADMAN DMA (`DesktopUi.cpp`).
- **Alt-Tab / Minimize Safe**: Thread-safe D3D11 context protection via `ID3D10Multithread::SetMultithreadProtected(TRUE)`.

### CPU & Vector Unit Execution
- **Windows x64 SSE2 VU1 Basic-Block JIT (`Vu1JitX64`)**: Persistent L2 code cache with full PS2 VU1 hardware accuracy (`ISW`/`ISWR` operand order, `WAITP` `0x7B`, 1-cycle `XGKICK` delay, `LQI`/`SQI`/`LQD`/`SQD` VI branch-delay hazard with consecutive-write chaining, and cycle-accurate `DIV`/`SQRT`/`RSQRT` `Q`-pipeline latency).
- **Cooperative `R5900FallbackInterpreter`**: Handles indirect jump tables and uncompiled fallback paths with cooperative thread yielding alongside the statically recompiled `SLUS_216.02` x64 DLL.

### Verified Status (`SLUS_216.02`, 3× HD)
- **Title Screen**: **117+ FPS** @ 3× HD
- **AllSpark Menu**: **66–68 FPS** @ 3× HD
- **Autobot Mission 1 (*The Suburbs*)**: Live playable 3D Bumblebee gameplay with full HUD and minimap at **50–64 FPS** @ 3× HD
- **Automated Verification**: **40 / 40** unit tests passing (`transformers_tests.exe`) and **6 / 6** bit-exact D3D11 HW GS parity checks (`diff=0`).

---

## Quick Start (Windows x64)

### Prerequisites
- Windows 10 / 11 (x64) with Direct3D 11 capable GPU
- CMake 3.20+, Ninja, and a C++20 toolchain (MinGW-W64 GCC / Clang)
- A legally obtained PS2 ISO of *Transformers: The Game (USA)* (`SLUS_216.02`)

### 1. Configure & Build
Run from the root of the cloned `TRANSFORMER-PORT-PC` repository:

```powershell
cmake -S . -B build-win64 -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-win64
```

This produces `Transformers.exe`, `TransformersRecompBuilder.exe`, and `transformers_tests.exe` in `build-win64/`.

### 2. Run the Test Suite

```powershell
ctest --test-dir build-win64 --output-on-failure
```

### 3. Build the Recompiled Module & Play

**Option A — Desktop UI (Recommended):**
1. Launch `.\build-win64\Transformers.exe` (no CLI arguments required).
2. Click **Select PS2 ISO...** to browse to your legally obtained ISO.
3. Click **Recompile** to generate C++20 shards and compile the native x64 module DLL (cached via SHA-256).
4. Click **Play**! Settings (resolution scale `1x`–`4x`, aspect ratio `4:3` / `16:9` / `Stretch`, VSync, bilinear filtering, input deadzone/sensitivity, audio, and Virtual Memory Card path) are saved to `%LOCALAPPDATA%\TransformersNative\settings.ini`.

**Option B — CLI Builder & Headless Runner:**

```powershell
# 1. Inspect ISO / ELF / R5900 coverage only:
.\build-win64\TransformersRecompBuilder.exe --iso "path\to\Transformers.iso" --inspect-only

# 2. Generate C++20 shards and compile the Windows x64 module DLL:
.\build-win64\TransformersRecompBuilder.exe --iso "path\to\Transformers.iso" --out .\build-win64\recomp_out --compile-dll --runtime-root .

# 3. Launch GUI with preselected ISO & module, or run headless with D3D11 HW GS at 3x HD:
.\build-win64\Transformers.exe --iso "path\to\Transformers.iso" --module .\build-win64\recomp_out\SLUS_216.02_recomp.dll
.\build-win64\Transformers.exe --iso "path\to\Transformers.iso" --module .\build-win64\recomp_out\SLUS_216.02_recomp.dll --headless --hw-gs --res-scale 3 --ms 15000 --auto-press --profile
```

<details>
<summary>Verified CLI Flags (<code>Transformers.exe</code> &amp; <code>TransformersRecompBuilder.exe</code>)</summary>

**`Transformers.exe`** (`Platform/Windows/main_win.cpp`):
- `--iso <path.iso>`: Path to read-only PS2 ISO
- `--module <recomp.dll>` / `--dll <recomp.dll>`: Path to compiled x64 module DLL
- `--memcard <dir>`: Virtual Memory Card directory (default: `MemoryCard0`)
- `--headless`: Run without creating the Desktop UI window
- `--hw-gs`: Enable D3D11 Hardware GS Renderer in headless mode
- `--res-scale <1..4>`: Internal resolution scale factor (`1` to `4`) for headless `--hw-gs`
- `--ms <N>`: Headless run duration in milliseconds (default: `4000`)
- `--max-calls <N>`: Static call limit in headless mode (`0` = unlimited, default: `200000`)
- `--auto-press`: Pulse Start/Cross automatically in headless mode
- `--profile`: Print per-window subsystem frame timing and counter telemetry
- `--dump-frame <path.bmp>` / `--dump-dir <dir>`: Dump presented BMP frame(s) in headless mode
- `--help` / `-h`: Print usage information

**`TransformersRecompBuilder.exe`** (`Builder/Application/main.cpp`):
- `--select`: Open native Windows file picker for a PS2 ISO
- `--iso <path.iso>`: Input PS2 ISO path
- `--inspect-only`: Inspect ISO, `SYSTEM.CNF`, ELF, and R5900 coverage without codegen
- `--out <dir>`: Output directory for reports and generated C++20 shards
- `--cache <dir>`: Override persistent build cache directory
- `--ghidra-csv <file>`: Import additional function addresses from a Ghidra CSV
- `--max-funcs <N>`: Limit number of emitted functions (`0` = all)
- `--compile-dll`: Compile the generated C++20 shards into a Windows x64 DLL
- `--runtime-root <dir>`: Path to the `TransformersNative` source root (for `--compile-dll`)
- `--target <arch>`: Target architecture (`win64-x86_64`)
- `--force`: Bypass the persistent build cache

</details>

---

## Controls

`Transformers.exe` supports simultaneous **Keyboard** and **XInput Controller (Port 0)** input with DualShock 2 actuator rumble (`DesktopUi.cpp`):

| DualShock 2 Input | Keyboard Binding | XInput Gamepad |
| --- | --- | --- |
| **Left Analog Stick** | `W` / `A` / `S` / `D` | Left Thumbstick |
| **D-Pad** | `Up` / `Left` / `Down` / `Right` Arrows | D-Pad |
| **Right Analog Stick** | `I` / `J` / `K` / `L` or Numpad `8` / `4` / `5` / `6` | Right Thumbstick |
| **Start** | `Enter` | Start |
| **Select** | `Backspace` or `Right Shift` | Back |
| **Cross (✕)** | `Space`, `Z`, or `N` | `A` |
| **Circle (○)** | `X` or `M` | `B` |
| **Square (□)** | `C` or `B` | `X` |
| **Triangle (△)** | `V` or `F` | `Y` |
| **L1 / R1** | `Q` / `E` | LB / RB |
| **L2 / R2** | `1` / `3` | LT / RT |
| **L3 / R3** | — | Left / Right Thumbstick Click |
| **Fullscreen Toggle** | `F11` | — |

---

## Project Structure

```text
.
├── CMakeLists.txt                   # Windows x64 CMake build configuration
├── Builder/                         # SHA-256 build cache, recomp pipeline, CLI builder
├── Graphics/
│   ├── Common/                      # Shared IGraphicsBackend & GraphicsSettings interfaces
│   └── Direct3D12/                  # D3D11 Hardware GS renderer & Win32 DXGI Flip-Model presenter
├── Platform/
│   └── Windows/                     # Transformers.exe entrypoint, Win32 Desktop UI & input polling
├── Recompiler/                      # ISO9660 reader, ELF32 parser, R5900 decoder, analyzer, C++20 codegen
├── Runtime/                         # EE memory/CPU/syscalls, VU0/VU1 + x64 JIT, DMAC, VIF, GIF, GS, IOP, SPU2, DS2, VMC
├── Tests/                           # Unit & integration test suite (transformers_tests)
└── docs/
    └── AUDIT_AND_ARCHITECTURE.md    # Milestone engineering log & reference architecture study
```

See [docs/AUDIT_AND_ARCHITECTURE.md](docs/AUDIT_AND_ARCHITECTURE.md) for the milestone log and clean-room reference architecture notes.
