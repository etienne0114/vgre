# Track Z (Zero-Burden Engine): the pluggable execution-backend module. It is
# the seam that lets VGRE run kernels through interchangeable engines. Tier 0
# (interpreter) lives here now; Tier 1 (copy_patch/) and Tier 2 (ssa/) land as
# their own subdirectories so nothing mixes. This module is LLVM-free.
add_library(vgre_backend OBJECT
    backend_registry.cpp
    interpreter/interpreter_backend.cpp
)

target_include_directories(vgre_backend PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(vgre_backend PRIVATE
    Threads::Threads
)
