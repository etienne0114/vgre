add_library(vgre_compiler STATIC
    kernel_parser.cpp
    clang_kernel_parser.cpp
    clang_kernel_analysis.cpp
    ptx/ptx_translator.cpp
    ptx/ptx_translator_map.cpp
    ptx/ptx_texture_ops.cpp
    ptx/ptx_shared_atomics.cpp
    ptx/ptx_conversion.cpp
    kernel_cache.cpp
    llvm_translation_engine.cpp
    llvm_translation_codegen.cpp
    texture_builtins.cpp
    kernel_fusion_engine.cpp
)

target_include_directories(vgre_compiler PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(vgre_compiler PUBLIC
    OpenMP::OpenMP_CXX
    ${llvm_libs}
)
