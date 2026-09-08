# The kernel compiler. Most of it is LLVM-free (the regex kernel parser, the PTX
# translator, SASS decoder, fusion, Triton frontend). The Clang front-end + the
# LLVM ORC JIT live in a handful of files that are compiled ONLY when
# VGRE_ENABLE_JIT is ON; when OFF, jit_stubs.cpp provides no-op replacements so
# VGRE builds and runs with no LLVM/Clang (Track Z — Zero-Burden Engine).

set(_vgre_compiler_srcs
    kernel_parser.cpp
    ptx/ptx_translator.cpp
    ptx/ptx_translator_map.cpp
    ptx/ptx_texture_ops.cpp
    ptx/ptx_shared_atomics.cpp
    ptx/ptx_conversion.cpp
    texture_builtins.cpp
    kernel_fusion_engine.cpp
    sass/sass_decoder.cpp
    triton/triton_frontend.cpp
)

if(VGRE_ENABLE_JIT)
    list(APPEND _vgre_compiler_srcs
        clang_kernel_parser.cpp
        clang_kernel_analysis.cpp
        kernel_cache.cpp
        llvm_translation_engine.cpp
        llvm_translation_codegen.cpp
    )
else()
    list(APPEND _vgre_compiler_srcs jit_stubs.cpp)
endif()

add_library(vgre_compiler STATIC ${_vgre_compiler_srcs})

target_include_directories(vgre_compiler PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

if(VGRE_ENABLE_JIT)
    target_link_libraries(vgre_compiler PRIVATE ${llvm_libs})
endif()
