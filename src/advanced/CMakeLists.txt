add_library(vgre_advanced OBJECT
    hybrid_compute_manager.cpp
    adaptive_execution_engine.cpp
    memory_compression.cpp
    vendor/lz4/lz4.c
    runtime_profiler.cpp
    vgre_workload_engine.cpp
    ipc_manager.cpp
    tcp_cluster.cpp
    tcp_cluster/tcp_cluster_manager.cpp
    tcp_cluster/connection_manager.cpp
    tcp_cluster/discovery_manager.cpp
    tcp_cluster/discovery_loops.cpp
    tcp_cluster/packet_handler.cpp
    tcp_cluster/security_manager.cpp
    tcp_cluster/memory_sync_manager.cpp
    tcp_cluster/collective_ops_manager.cpp
    tcp_cluster/dispatch_manager.cpp
    tcp_cluster/dispatch_impl.cpp
    tcp_cluster/server_loop.cpp
    tcp_cluster/client_loop.cpp
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

# ── Standalone Worker CLI ────────────────────────────────────────────────────
add_executable(vgre-worker vgre_worker_cli.cpp)
target_sources(vgre-worker PRIVATE
    $<TARGET_OBJECTS:vgre_common>
    $<TARGET_OBJECTS:vgre_core>
    $<TARGET_OBJECTS:vgre_compiler>
    $<TARGET_OBJECTS:vgre_api>
    $<TARGET_OBJECTS:vgre_runtime>
    $<TARGET_OBJECTS:vgre_advanced>
)
target_link_libraries(vgre-worker PRIVATE
    OpenMP::OpenMP_CXX
    ${OpenMP_CXX_LIBRARIES}
    Threads::Threads
    ${llvm_libs}
)
if(WIN32)
    target_link_libraries(vgre-worker PRIVATE Advapi32 ws2_32)
elseif(APPLE)
    target_link_libraries(vgre-worker PRIVATE "-framework Security")
elseif(UNIX)
    find_library(KEYUTILS_LIB keyutils)
    if(KEYUTILS_LIB)
        target_link_libraries(vgre-worker PRIVATE ${KEYUTILS_LIB})
    endif()
endif()
if(VGRE_USE_OPENCL_BACKEND)
    target_link_libraries(vgre-worker PRIVATE OpenCL::OpenCL)
endif()
target_include_directories(vgre-worker PRIVATE ${PROJECT_SOURCE_DIR}/include)
