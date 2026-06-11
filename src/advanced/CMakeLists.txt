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
    runtime_profiler_instruction.cpp
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
    tcp_cluster/dispatch_partition_impl.cpp
    tcp_cluster/server_loop_core.cpp
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

target_link_libraries(vgre_advanced PRIVATE
    Threads::Threads
    vgre_llvm_iface
)

# macOS: adaptive_execution_engine.cpp probes IOKit (AppleSMC) for thermal state.
# PUBLIC so consumers of these objects (shared libs + tests) resolve the symbols.
if(APPLE)
    find_library(_vgre_adv_iokit IOKit)
    find_library(_vgre_adv_cf CoreFoundation)
    if(_vgre_adv_iokit AND _vgre_adv_cf)
        target_link_libraries(vgre_advanced PUBLIC ${_vgre_adv_iokit} ${_vgre_adv_cf})
    endif()
endif()

# Propagate SIMD compile options to vgre_advanced so that AES-NI/PCLMUL/AVX2
# intrinsics in secure_channel_crypto.cpp etc. compile with the right ISA flags.
if(VGRE_SIMD_COMPILE_OPTIONS)
    target_compile_options(vgre_advanced PRIVATE ${VGRE_SIMD_COMPILE_OPTIONS})
endif()
if(VGRE_SIMD_COMPILE_DEFS)
    target_compile_definitions(vgre_advanced PRIVATE ${VGRE_SIMD_COMPILE_DEFS})
endif()

# RDMA: set compile definitions + include paths so rdma_transport.cpp compiles.
# The libibverbs link itself is handled by _vgre_configure_library() on the
# final shared libraries (vgre, vgre_cudart) and the vgre-worker executable.
if(_VGRE_RDMA_FOUND)
    target_compile_definitions(vgre_advanced PRIVATE VGRE_HAS_RDMA)
    target_include_directories(vgre_advanced PRIVATE "${IBVERBS_INCLUDE}")
endif()

# OpenSSL: compile secure_channel.cpp and websocket_transport.cpp mTLS blocks.
# VGRE_ENABLE_SSL must be set on the OBJECT library (not just on the final
# shared targets) so that #ifdef VGRE_ENABLE_SSL guards in these source files
# are active during compilation of vgre_advanced objects.
# The include directories are needed here; actual linking happens on the final
# shared libraries via _vgre_configure_library().
if(_VGRE_HAS_SSL)
    target_compile_definitions(vgre_advanced PRIVATE VGRE_ENABLE_SSL)
    target_include_directories(vgre_advanced PRIVATE
        ${OPENSSL_INCLUDE_DIR})
endif()

# gRPC: set compile definitions + generated proto include path so
# grpc_transport.cpp sees the generated .pb.h headers.
# Library linking is handled by _vgre_configure_library() on final targets.
if(_VGRE_GRPC_FOUND)
    target_compile_definitions(vgre_advanced PRIVATE VGRE_HAS_GRPC)
    target_include_directories(vgre_advanced PRIVATE "${_gen_dir}")
endif()

if(APPLE)
    target_link_libraries(vgre_advanced PUBLIC
        "-framework IOKit"
        "-framework Security"
        "-framework CoreFoundation"
    )
endif()

# ── Standalone Worker CLI ────────────────────────────────────────────────────
# Link against the vgre shared library instead of embedding all object files.
# This eliminates duplicate singletons, static initialization order issues,
# and OpenMP runtime conflicts (MSVC vcomp.dll vs LLVM libomp.dll) that cause
# STATUS_ACCESS_VIOLATION (0xC0000005) on Windows.
#
# All test executables already use this pattern (target_link_libraries PRIVATE vgre)
# and pass 100% of CTests.  The worker must behave identically.
#
# On Windows, vgre-worker.exe finds vgre.dll because both are deployed to the
# same INSTALL_DIR by vgre_sync.bat, and Windows searches the application
# directory first.  On Linux/macOS, the launch wrapper sets LD_LIBRARY_PATH /
# DYLD_LIBRARY_PATH.
add_executable(vgre-worker vgre_worker_cli.cpp)
target_link_libraries(vgre-worker PRIVATE vgre)
target_include_directories(vgre-worker PRIVATE ${PROJECT_SOURCE_DIR}/include)

# ── Cross-platform dependencies ──────────────────────────────────────────────
# The vgre shared library already links these PRIVATE, so they are resolved
# internally at load time.  We add them here defensively so that:
#   (a) any direct platform calls from the CLI itself (Win32 diagnostics,
#       signal handling, etc.) resolve cleanly at link time,
#   (b) linkers with strict --no-undefined / LINK.EXE /FORCE:UNRESOLVED never
#       report missing symbols,
#   (c) cross-platform intent is documented explicitly in the build system.
if(WIN32)
    target_link_libraries(vgre-worker PRIVATE Advapi32 ws2_32 bcrypt)
    set_target_properties(vgre-worker PROPERTIES
        MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
elseif(APPLE)
    target_link_libraries(vgre-worker PRIVATE "-framework Security")
elseif(UNIX)
    find_library(KEYUTILS_LIB keyutils)
    if(KEYUTILS_LIB)
        target_link_libraries(vgre-worker PRIVATE ${KEYUTILS_LIB})
    endif()
endif()

# OpenCL backend — worker probes iGPU capabilities on startup.
if(_VGRE_OPENCL_FOUND)
    target_compile_definitions(vgre-worker PRIVATE VGRE_HAS_OPENCL_BACKEND)
    target_link_libraries(vgre-worker PRIVATE OpenCL::OpenCL)
endif()

