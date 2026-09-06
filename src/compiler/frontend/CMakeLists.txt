# Track Z (Zero-Burden Engine): the from-scratch CUDA-C front-end. Lexer now;
# parser (-> AST) and PTX code generation land here next, each in its own file
# so nothing mixes. LLVM-free — this is what replaces the Clang front-end.
add_library(vgre_frontend OBJECT
    lexer.cpp
)

target_include_directories(vgre_frontend PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(vgre_frontend PRIVATE
    Threads::Threads
)
