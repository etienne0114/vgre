add_library(vgre_advanced OBJECT
    hybrid_compute_manager.cpp
    adaptive_execution_engine.cpp
    memory_compression.cpp
    vendor/lz4/lz4.c
    runtime_profiler.cpp
    vgre_workload_engine.cpp
    ipc_manager.cpp
    tcp_cluster.cpp
    secure_channel.cpp
    workload_partitioner.cpp
    resource_ledger.cpp
    hardware_token_manager.cpp
)

target_include_directories(vgre_advanced PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src/advanced/vendor/lz4
)

target_link_libraries(vgre_advanced PUBLIC
    OpenMP::OpenMP_CXX
    Threads::Threads
)

if(APPLE)
    target_link_libraries(vgre_advanced PUBLIC "-framework IOKit")
endif()
