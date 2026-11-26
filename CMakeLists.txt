cmake_minimum_required(VERSION 3.20)
project(TransformersNative LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wno-unused-parameter)
    if(NOT ANDROID AND CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|x86_64|x64")
        add_compile_options($<$<CONFIG:Release>:-O3> $<$<CONFIG:Release>:-march=x86-64-v3>)
    endif()
endif()

# ---------------------------------------------------------------------------
# 1. Core PS2 Recompiler Library (ISO, ELF, Analyzer, R5900 Decoder, CodeGen)
# ---------------------------------------------------------------------------
add_library(ps2_recompiler STATIC
    Recompiler/Common/Sha256.cpp
    Recompiler/ISO/IsoReader.cpp
    Recompiler/ELF/ElfParser.cpp
    Recompiler/Decoder/R5900Decoder.cpp
    Recompiler/Analyzer/FunctionAnalyzer.cpp
    Recompiler/CodeGenerator/CppCodeGenerator.cpp
)
target_include_directories(ps2_recompiler PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# ---------------------------------------------------------------------------
# 2. Core PS2 Hardware Compatibility Runtime (portable Windows x64 + Android ARM64)
# ---------------------------------------------------------------------------
add_library(ps2_runtime STATIC
    Runtime/Config/AppSettings.cpp
    Runtime/Memory/GuestMemory.cpp
    Runtime/CPU/SyscallDispatcher.cpp
    Runtime/CPU/R5900Interpreter.cpp
    Runtime/VectorUnits/VU0.cpp
    Runtime/VectorUnits/VU1.cpp
    Runtime/VectorUnits/Vu1JitX64.cpp
    Runtime/DMA/DmaController.cpp
    Runtime/VIF/VifUnit.cpp
    Runtime/GIF/GifUnit.cpp
    Runtime/GS/GsState.cpp
    Runtime/CDVD/CdvdDrive.cpp
    Runtime/IOP/IopHle.cpp
    Runtime/Audio/Spu2.cpp
    Runtime/Input/DualShock2.cpp
    Runtime/MemoryCard/VirtualMemoryCard.cpp
    Runtime/RuntimeHost.cpp
)
target_include_directories(ps2_runtime PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ps2_runtime PUBLIC ps2_recompiler)

# ---------------------------------------------------------------------------
# 3. Android ARM64 Shared Native Library (when cross-compiled via Android NDK)
# ---------------------------------------------------------------------------
if(ANDROID)
    add_library(transformers_native SHARED
        Platform/Android/app/src/main/cpp/android_jni.cpp
        Graphics/OpenGLES/AndroidGlesBackend.cpp
    )
    target_link_libraries(transformers_native PRIVATE ps2_runtime GLESv3 EGL android log)
endif()

# ---------------------------------------------------------------------------
# 4. Windows x64 Deliverables: Builder, Native Runtime Host, and Test Suite
# ---------------------------------------------------------------------------
if(NOT ANDROID)
    add_library(ps2_builder_core STATIC
        Builder/Cache/BuildCache.cpp
        Builder/BuildPipeline/RecompPipeline.cpp
    )
    target_include_directories(ps2_builder_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(ps2_builder_core PUBLIC ps2_recompiler)

    # A. Recompilation Builder: TransformersRecompBuilder.exe
    add_executable(TransformersRecompBuilder
        Builder/Application/main.cpp
    )
    target_link_libraries(TransformersRecompBuilder PRIVATE ps2_builder_core)
    if(WIN32)
        target_link_libraries(TransformersRecompBuilder PRIVATE comdlg32)
        if(MINGW)
            target_link_options(TransformersRecompBuilder PRIVATE -static -static-libgcc -static-libstdc++)
        endif()
    endif()

    # B. Native Windows Game Host & Transformers Desktop Settings UI: Transformers.exe
    add_executable(Transformers
        Platform/Windows/main_win.cpp
        Platform/Windows/DesktopUi.cpp
        Graphics/Direct3D12/D3D11GsRenderer.cpp
        Graphics/Direct3D12/Win32Presenter.cpp
    )
    target_link_libraries(Transformers PRIVATE ps2_runtime ps2_builder_core)
    if(WIN32)
        target_link_libraries(Transformers PRIVATE gdi32 user32 comdlg32 opengl32 winmm d3d11 dxgi d3dcompiler dwmapi)
        if(MINGW)
            target_link_options(Transformers PRIVATE -static -static-libgcc -static-libstdc++)
        endif()
    endif()

    # C. Unit & End-to-End Recompilation Verification Suite
    enable_testing()
    add_executable(transformers_tests
        Tests/TestMain.cpp
        Tests/IsoTests.cpp
        Tests/ElfTests.cpp
        Tests/DecoderTests.cpp
        Tests/RecompilerExecTests.cpp
        Tests/RuntimeTests.cpp
    )
    target_link_libraries(transformers_tests PRIVATE ps2_runtime ps2_builder_core)
    target_compile_definitions(transformers_tests PRIVATE
        TF_SOURCE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
        TF_WORKSPACE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/.."
    )
    add_test(NAME TransformersNativeTests COMMAND transformers_tests)
endif()
