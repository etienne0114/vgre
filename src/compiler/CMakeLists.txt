add_library(vgre_compiler OBJECT
    kernel_parser.cpp
    llvm_translation_engine.cpp
)

target_include_directories(vgre_compiler PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(vgre_compiler PUBLIC
    OpenMP::OpenMP_CXX
    ${llvm_libs}
)
