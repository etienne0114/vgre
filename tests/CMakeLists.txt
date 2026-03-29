# Test CMakeLists.txt
# Unit tests and integration tests for VGRE

# ── Unit tests ──────────────────────────────────────────────────────────────
add_executable(test_virtual_gpu_device unit/test_virtual_gpu_device.cpp)
target_link_libraries(test_virtual_gpu_device PRIVATE vgre)
add_test(NAME VirtualGPUDevice COMMAND test_virtual_gpu_device)

add_executable(test_memory_manager unit/test_memory_manager.cpp)
target_link_libraries(test_memory_manager PRIVATE vgre)
add_test(NAME MemoryManager COMMAND test_memory_manager)

add_executable(test_scheduler unit/test_scheduler.cpp)
target_link_libraries(test_scheduler PRIVATE vgre)
add_test(NAME Scheduler COMMAND test_scheduler)

add_executable(test_vector_engine unit/test_vector_engine.cpp)
target_link_libraries(test_vector_engine PRIVATE vgre)
add_test(NAME VectorEngine COMMAND test_vector_engine)

# ── Integration tests ──────────────────────────────────────────────────────
add_executable(test_vector_addition integration/test_vector_addition.cpp)
target_link_libraries(test_vector_addition PRIVATE vgre)
add_test(NAME VectorAdditionIntegration COMMAND test_vector_addition)

add_executable(test_uvm_managed integration/test_uvm_managed.cpp)
target_link_libraries(test_uvm_managed PRIVATE vgre)
add_test(NAME UVMManagedIntegration COMMAND test_uvm_managed)

add_executable(test_cuda_graphs integration/test_cuda_graphs.cpp)
target_link_libraries(test_cuda_graphs PRIVATE vgre)
add_test(NAME CUDAGraphsIntegration COMMAND test_cuda_graphs)

add_executable(test_graph_c_api integration/test_graph_c_api.cpp)
target_link_libraries(test_graph_c_api PRIVATE vgre)
add_test(NAME GraphCAPIIntegration COMMAND test_graph_c_api)

add_executable(test_multi_device integration/test_multi_device.cpp)
target_link_libraries(test_multi_device PRIVATE vgre)
add_test(NAME MultiDeviceIntegration COMMAND test_multi_device)

add_executable(test_stream_concurrency integration/test_stream_concurrency.cpp)
target_link_libraries(test_stream_concurrency PRIVATE vgre)
add_test(NAME test_stream_concurrency COMMAND test_stream_concurrency)

add_executable(test_cudart_shim_regression integration/test_cudart_shim_regression.cpp)
target_link_libraries(test_cudart_shim_regression PRIVATE vgre_cudart)
add_test(NAME CUDARTShimRegression COMMAND test_cudart_shim_regression)

add_executable(test_tcp_cluster integration/test_tcp_cluster.cpp)
target_link_libraries(test_tcp_cluster PRIVATE vgre)
add_test(NAME TCPClusterIntegration COMMAND test_tcp_cluster)

add_executable(test_p2p_performance integration/test_p2p_performance.cpp)
target_link_libraries(test_p2p_performance PRIVATE vgre)
add_test(NAME P2PPerformanceIntegration COMMAND test_p2p_performance)

add_executable(test_igpu_backend integration/test_igpu_backend.cpp)
target_link_libraries(test_igpu_backend PRIVATE vgre)
add_test(NAME IGPUBackendIntegration COMMAND test_igpu_backend)

add_executable(test_complex_jit integration/test_complex_jit.cpp)
target_link_libraries(test_complex_jit PRIVATE vgre)
add_test(NAME ComplexJITIntegration COMMAND test_complex_jit)

add_executable(test_syncthreads integration/test_syncthreads.cpp)
target_link_libraries(test_syncthreads PRIVATE vgre)
add_test(NAME SyncthreadsIntegration COMMAND test_syncthreads)

add_executable(test_extern_shared integration/test_extern_shared.cpp)
target_link_libraries(test_extern_shared PRIVATE vgre)
add_test(NAME ExternSharedIntegration COMMAND test_extern_shared)

add_executable(test_extern_c_kernel integration/test_extern_c_kernel.cpp)
target_link_libraries(test_extern_c_kernel PRIVATE vgre)
add_test(NAME ExternCKernelIntegration COMMAND test_extern_c_kernel)

add_executable(test_static_shared integration/test_static_shared.cpp)
target_link_libraries(test_static_shared PRIVATE vgre)
add_test(NAME StaticSharedIntegration COMMAND test_static_shared)

add_executable(test_cuda_api_attrs integration/test_cuda_api_attrs.cpp)
target_link_libraries(test_cuda_api_attrs PRIVATE vgre)
add_test(NAME CUDAAPIAttributesIntegration COMMAND test_cuda_api_attrs)

add_executable(test_driver_api integration/test_driver_api.cpp)
target_link_libraries(test_driver_api PRIVATE vgre_cudart)
add_test(NAME DriverAPIIntegration COMMAND test_driver_api)
add_executable(test_cubin_load integration/test_cubin_load.cpp)
target_link_libraries(test_cubin_load PRIVATE vgre_cudart)
add_executable(test_texture_driver_api integration/test_texture_driver_api.cpp)
target_link_libraries(test_texture_driver_api vgre)

add_executable(test_async_sync integration/test_async_sync.cpp)
target_link_libraries(test_async_sync vgre)

add_executable(test_graph_cycle integration/test_graph_cycle.cpp)
target_link_libraries(test_graph_cycle vgre)

add_executable(test_struct_args integration/test_struct_args.cpp)
target_link_libraries(test_struct_args vgre)
add_test(NAME StructArgsIntegration COMMAND test_struct_args)

add_test(NAME test_cubin_load COMMAND test_cubin_load)
add_test(NAME test_texture_driver_api COMMAND test_texture_driver_api)
add_test(NAME test_async_sync COMMAND test_async_sync)
add_test(NAME test_graph_cycle COMMAND test_graph_cycle)

add_executable(test_uvm_hints integration/test_uvm_hints.cpp)
target_link_libraries(test_uvm_hints PRIVATE vgre)
add_test(NAME test_uvm_hints COMMAND test_uvm_hints)

add_executable(test_graph_update integration/test_graph_update.cpp)
target_link_libraries(test_graph_update PRIVATE vgre)
add_test(NAME test_graph_update COMMAND test_graph_update)

add_executable(test_phase3_cluster integration/test_phase3_cluster.cpp)
target_link_libraries(test_phase3_cluster PRIVATE vgre)
add_test(NAME test_phase3_cluster COMMAND test_phase3_cluster)
set_tests_properties(test_phase3_cluster PROPERTIES TIMEOUT 60)

# ── Phase 5 tests ───────────────────────────────────────────────────────────
add_executable(test_phase5_secure_channel unit/test_phase5_secure_channel.cpp)
target_link_libraries(test_phase5_secure_channel PRIVATE vgre)
add_test(NAME test_phase5_secure_channel COMMAND test_phase5_secure_channel)

add_executable(test_phase5_workload_partition unit/test_phase5_workload_partition.cpp)
target_link_libraries(test_phase5_workload_partition PRIVATE vgre)
add_test(NAME test_phase5_workload_partition COMMAND test_phase5_workload_partition)

add_executable(test_phase5_resource_ledger unit/test_phase5_resource_ledger.cpp)
target_link_libraries(test_phase5_resource_ledger PRIVATE vgre)
add_test(NAME test_phase5_resource_ledger COMMAND test_phase5_resource_ledger)
add_executable(test_texture_depth unit/test_texture_depth.cpp)
target_link_libraries(test_texture_depth PRIVATE vgre)
add_test(NAME TextureDepth COMMAND test_texture_depth)

add_executable(test_clang_parser unit/test_clang_parser.cpp)
target_link_libraries(test_clang_parser PRIVATE vgre)
add_test(NAME ClangParser COMMAND test_clang_parser)

add_executable(test_clang_enhanced compiler/test_clang_enhanced.cpp)
target_link_libraries(test_clang_enhanced PRIVATE vgre)
add_test(NAME ClangEnhanced COMMAND test_clang_enhanced)
set_tests_properties(ClangEnhanced PROPERTIES TIMEOUT 300)

add_executable(test_kernel_parser_enhanced compiler/test_kernel_parser_enhanced.cpp)
target_link_libraries(test_kernel_parser_enhanced PRIVATE vgre)
add_test(NAME KernelParserEnhanced COMMAND test_kernel_parser_enhanced)

add_executable(test_flop_counting compiler/test_flop_counting.cpp)
target_link_libraries(test_flop_counting PRIVATE vgre)
add_test(NAME FLOPCounting COMMAND test_flop_counting)

add_executable(test_texture_view unit/test_texture_view.cpp)
target_link_libraries(test_texture_view PRIVATE vgre)
add_test(NAME TextureView COMMAND test_texture_view)

add_executable(test_texture_view_advanced unit/test_texture_view_advanced.cpp)
target_link_libraries(test_texture_view_advanced PRIVATE vgre)
add_test(NAME TextureViewAdvanced COMMAND test_texture_view_advanced)

add_executable(test_dirty_page_tracking core/test_dirty_page_tracking.cpp)
target_link_libraries(test_dirty_page_tracking PRIVATE vgre)
add_test(NAME DirtyPageTracking COMMAND test_dirty_page_tracking)
add_executable(test_memory_concurrency unit/test_memory_concurrency.cpp)
target_link_libraries(test_memory_concurrency PRIVATE vgre)
add_test(NAME MemoryConcurrency COMMAND test_memory_concurrency)

add_executable(test_tpm_auth unit/test_tpm_auth.cpp)
target_link_libraries(test_tpm_auth PRIVATE vgre)
add_test(NAME TPMAuth COMMAND test_tpm_auth)

add_executable(test_tcp_poll advanced/test_tcp_poll.cpp)
target_link_libraries(test_tcp_poll PRIVATE vgre)
add_test(NAME TCPPoll COMMAND test_tcp_poll)

add_executable(test_cost_benefit_model advanced/test_cost_benefit_model.cpp)
target_link_libraries(test_cost_benefit_model PRIVATE vgre)
add_test(NAME CostBenefitModel COMMAND test_cost_benefit_model)

add_executable(test_hardware_token_manager advanced/test_hardware_token_manager.cpp)
target_link_libraries(test_hardware_token_manager PRIVATE vgre)
add_test(NAME HardwareTokenManager COMMAND test_hardware_token_manager)

add_executable(test_input_validation unit/test_input_validation.cpp)
target_link_libraries(test_input_validation PRIVATE vgre)
add_test(NAME InputValidation COMMAND test_input_validation)

add_executable(test_texture_manager core/test_texture_manager.cpp)
target_link_libraries(test_texture_manager PRIVATE vgre)
add_test(NAME TextureManager COMMAND test_texture_manager)

add_executable(test_memory_pool unit/test_memory_pool.cpp)
target_link_libraries(test_memory_pool PRIVATE vgre)
add_test(NAME MemoryPool COMMAND test_memory_pool)
