# ──────────────────────────────────────────────────────────────────────────────
# §  vgre_debug — CUDA-GDB-compatible debug stepping (missingFeatures §6.1):
#    PTX single-step interpreter + GDB remote-serial-protocol server.
add_library(vgre_debug OBJECT
    ptx_interpreter.cpp
    gdb_rsp_server.cpp
)

target_include_directories(vgre_debug PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)

set_target_properties(vgre_debug PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(WIN32)
    target_link_libraries(vgre_debug PUBLIC ws2_32)
endif()
