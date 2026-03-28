add_library(vgre_compiler STATIC
    kernel_parser.cpp
    clang_kernel_parser.cpp
    kernel_cache.cpp
    llvm_translation_engine.cpp
    texture_builtins.cpp
)

target_include_directories(vgre_compiler PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(vgre_compiler PUBLIC
    OpenMP::OpenMP_CXX
    ${llvm_libs}
)
