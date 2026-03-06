add_library(vgre_runtime OBJECT
    cpu_parallel_executor.cpp
    vector_engine.cpp
)

target_include_directories(vgre_runtime PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(vgre_runtime PUBLIC
    OpenMP::OpenMP_CXX
)
