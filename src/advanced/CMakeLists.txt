add_library(vgre_advanced OBJECT
    hybrid_compute_manager.cpp
    adaptive_execution_engine.cpp
    memory_compression.cpp
    runtime_profiler.cpp
    vgre_workload_engine.cpp
    ipc_manager.cpp
    tcp_cluster.cpp
)

target_include_directories(vgre_advanced PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(vgre_advanced PUBLIC
    OpenMP::OpenMP_CXX
    Threads::Threads
)
