# §  vgre_mig — Multi-Instance GPU partitioning (multi-tenant isolation)
add_library(vgre_mig OBJECT
    mig_manager.cpp
    mig_c_api.cpp
)

target_include_directories(vgre_mig PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)

set_target_properties(vgre_mig PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(TARGET Threads::Threads)
    target_link_libraries(vgre_mig PUBLIC Threads::Threads)
endif()
