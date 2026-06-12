# §  vgre_secrets — sealed, versioned secret store (rotation + access policy)
add_library(vgre_secrets OBJECT
    secret_store.cpp
)

target_include_directories(vgre_secrets PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)

set_target_properties(vgre_secrets PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(TARGET Threads::Threads)
    target_link_libraries(vgre_secrets PUBLIC Threads::Threads)
endif()
