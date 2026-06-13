# §  vgre_compliance — enterprise audit / compliance (tamper-evident audit log)
add_library(vgre_compliance OBJECT
    audit_log.cpp
    policy_engine.cpp
)

target_include_directories(vgre_compliance PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)

# Position-independent — folded into the vgre SHARED library.
set_target_properties(vgre_compliance PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Crypto primitives live in vgre_advanced's headers; symbols resolve when both
# object libraries are linked into the single `vgre` shared library.
if(TARGET Threads::Threads)
    target_link_libraries(vgre_compliance PUBLIC Threads::Threads)
endif()
