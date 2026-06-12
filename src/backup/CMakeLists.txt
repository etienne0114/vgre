# §  vgre_backup — content-addressed snapshot archive (backup / DR)
add_library(vgre_backup OBJECT
    backup_archive.cpp
    backup_c_api.cpp
)

target_include_directories(vgre_backup PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)

set_target_properties(vgre_backup PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(TARGET Threads::Threads)
    target_link_libraries(vgre_backup PUBLIC Threads::Threads)
endif()
