# §  vgre_xla — XLA/HLO execution engine (the compute core of a PJRT backend)
add_library(vgre_xla OBJECT
    hlo_evaluator.cpp
)

target_include_directories(vgre_xla PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)

set_target_properties(vgre_xla PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(TARGET Threads::Threads)
    target_link_libraries(vgre_xla PUBLIC Threads::Threads)
endif()
