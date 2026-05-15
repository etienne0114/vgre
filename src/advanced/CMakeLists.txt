add_library(vgre_advanced OBJECT
    hybrid_compute_manager.cpp
    hybrid_compute_manager_remote.cpp
    hybrid_compute_manager_workload.cpp
    adaptive_execution_engine.cpp
    adaptive_execution_engine_record.cpp
    adaptive_execution_engine_tune.cpp
    memory_compression.cpp
    vendor/lz4/lz4.c
    runtime_profiler.cpp
    vgre_workload_engine.cpp
    ipc_manager.cpp
    tcp_cluster/tcp_cluster_manager.cpp
    tcp_cluster/tcp_cluster_init.cpp
    tcp_cluster/tcp_cluster_shutdown.cpp
    tcp_cluster/tcp_cluster_state.cpp
    tcp_cluster/tcp_cluster_telemetry.cpp
    tcp_cluster/tcp_cluster_restart.cpp
    tcp_cluster/tcp_cluster_metrics.cpp
    tcp_cluster/tcp_cluster_transport.cpp
    tcp_cluster/configuration_manager_core.cpp
    tcp_cluster/configuration_manager_file_io.cpp
    tcp_cluster/configuration_manager_backup.cpp
    tcp_cluster/configuration_manager_monitoring.cpp
    tcp_cluster/configuration_manager_validation.cpp
    tcp_cluster/configuration_manager_documentation.cpp
    tcp_cluster/connection_manager.cpp
    tcp_cluster/discovery_manager.cpp
    tcp_cluster/discovery_loops_udp.cpp
    tcp_cluster/discovery_loops_proactive.cpp
    tcp_cluster/packet_handler.cpp
    tcp_cluster/security_manager.cpp
    tcp_cluster/memory_sync_manager.cpp
    tcp_cluster/collective_ops_manager.cpp
    tcp_cluster/dispatch_manager_core.cpp
    tcp_cluster/dispatch_manager_remote.cpp
    tcp_cluster/dispatch_manager_partitioned.cpp
    tcp_cluster/dispatch_impl.cpp
    tcp_cluster/server_loop_core.cpp
    tcp_cluster/server_loop_auth_mgmt.cpp
    tcp_cluster/server_loop_connection_handling.cpp
    tcp_cluster/server_loop_data_handling.cpp
    tcp_cluster/server_packet_dispatch.cpp
    tcp_cluster/client_loop.cpp
    tcp_cluster/client_packet_dispatch.cpp
    tcp_cluster/windows_socket_manager_lifecycle.cpp
    tcp_cluster/windows_socket_manager_errors.cpp
    tcp_cluster/windows_socket_manager_recovery.cpp
    tcp_cluster/interfaces.cpp
    tcp_cluster/shared_utilities_base.cpp
    tcp_cluster/shared_utilities_network_crypto.cpp
    tcp_cluster/shared_utilities_packets.cpp
    tcp_cluster/diagnostic_logger.cpp
    tcp_cluster/mesh_topology_impl.cpp
    tcp_cluster/tcp_cluster_validation.cpp
    secure_channel_crypto.cpp
    secure_channel.cpp
    gpu_passthrough.cpp
    workload_partitioner.cpp
    resource_ledger.cpp
    token/hardware_token_manager.cpp
    token/token_manager_linux.cpp
    token/token_manager_macos.cpp
    token/token_manager_win32.cpp
    token/token_manager_tpm2.cpp
    token/token_manager_fallback.cpp
    rdma_transport.cpp
    mps_control.cpp
    grpc_transport.cpp
    websocket_transport.cpp
    ${VGRE_GRPC_GENERATED}
)

target_include_directories(vgre_advanced PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src/advanced/vendor/lz4
)

target_link_libraries(vgre_advanced PUBLIC
    OpenMP::OpenMP_CXX
    Threads::Threads
)

# Link libibverbs when RDMA is enabled
if(VGRE_ENABLE_RDMA AND VGRE_RDMA_LIBS)
    target_link_libraries(vgre_advanced PUBLIC ${VGRE_RDMA_LIBS})
endif()

# Link gRPC/Protobuf when gRPC transport is enabled
if(VGRE_ENABLE_GRPC AND VGRE_GRPC_LIBS)
    target_link_libraries(vgre_advanced PUBLIC ${VGRE_GRPC_LIBS})
    target_include_directories(vgre_advanced PRIVATE ${VGRE_PROTO_GEN})
endif()

if(APPLE)
    target_link_libraries(vgre_advanced PUBLIC
        "-framework IOKit"
        "-framework Security"
        "-framework CoreFoundation"
    )
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
    target_link_libraries(vgre-worker PRIVATE Advapi32 ws2_32 bcrypt)
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
