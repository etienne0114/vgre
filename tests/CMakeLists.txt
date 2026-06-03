# Test CMakeLists.txt
# Unit tests and integration tests for VGRE

# ── Pre-test cleanup fixture ─────────────────────────────────────────────────
# Cleans stale OpenMP (libomp) KMP registration files left by killed processes.
# These files accumulate in /dev/shm when processes are SIGKILL'd, causing the
# next test that initializes OpenMP to deadlock during library registration.
add_test(NAME KMPCleanup
    COMMAND bash -c "rm -f /dev/shm/__KMP_REGISTERED_LIB_* 2>/dev/null; rm -f \$HOME/.vgre_cache/*.ll.tmp.* 2>/dev/null; exit 0")
set_tests_properties(KMPCleanup PROPERTIES FIXTURES_SETUP KMPRegistrationClean)

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

# cuDNN activation backward numerical gradient tests
add_executable(test_cudnn_activation_backward unit/test_cudnn_activation_backward.cpp)
target_link_libraries(test_cudnn_activation_backward PRIVATE vgre)
add_test(NAME CUDNNActivationBackward COMMAND test_cudnn_activation_backward)

# Chase-Lev work-stealing deque stress test (QUEUE-02)
add_executable(test_chase_lev_stress unit/test_chase_lev_stress.cpp)
target_link_libraries(test_chase_lev_stress PRIVATE vgre)
add_test(NAME ChaseLevStress COMMAND test_chase_lev_stress)
set_tests_properties(ChaseLevStress PROPERTIES TIMEOUT 120)

# cuDNN backward conv: group + dilated + depthwise gradient check (QUEUE-03)
add_executable(test_cudnn_conv_group_dilated api/test_cudnn_conv_group_dilated.cpp)
target_link_libraries(test_cudnn_conv_group_dilated PRIVATE vgre)
add_test(NAME CUDNNConvGroupDilated COMMAND test_cudnn_conv_group_dilated)
set_tests_properties(CUDNNConvGroupDilated PROPERTIES TIMEOUT 60)

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

add_executable(test_multi_device_cooperative_comprehensive integration/test_multi_device_cooperative_comprehensive.cpp)
target_link_libraries(test_multi_device_cooperative_comprehensive PRIVATE vgre)
add_test(NAME MultiDeviceCooperativeComprehensive COMMAND test_multi_device_cooperative_comprehensive)
set_tests_properties(MultiDeviceCooperativeComprehensive PROPERTIES
    TIMEOUT 300
    RESOURCE_LOCK vgre_llvm_jit)

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
set_tests_properties(ComplexJITIntegration PROPERTIES RESOURCE_LOCK vgre_llvm_jit)

add_executable(test_syncthreads integration/test_syncthreads.cpp)
target_link_libraries(test_syncthreads PRIVATE vgre)
add_test(NAME SyncthreadsIntegration COMMAND test_syncthreads)
set_tests_properties(SyncthreadsIntegration PROPERTIES FIXTURES_REQUIRED KMPRegistrationClean)

add_executable(test_warp_shuffle_jit integration/test_warp_shuffle.cpp)
target_link_libraries(test_warp_shuffle_jit PRIVATE vgre)
add_test(NAME WarpShuffleJITIntegration COMMAND test_warp_shuffle_jit)
set_tests_properties(WarpShuffleJITIntegration PROPERTIES
    TIMEOUT 120
    FIXTURES_REQUIRED KMPRegistrationClean
    RESOURCE_LOCK vgre_llvm_jit)

add_executable(test_extern_shared integration/test_extern_shared.cpp)
target_link_libraries(test_extern_shared PRIVATE vgre)
add_test(NAME ExternSharedIntegration COMMAND test_extern_shared)
set_tests_properties(ExternSharedIntegration PROPERTIES FIXTURES_REQUIRED KMPRegistrationClean)

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
set_tests_properties(test_phase3_cluster PROPERTIES
    TIMEOUT 600
    RESOURCE_LOCK vgre_tcp_cluster_port
    RUN_SERIAL TRUE)

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
set_tests_properties(ClangParser PROPERTIES TIMEOUT 300)


add_executable(test_clang_enhanced compiler/test_clang_enhanced.cpp)
target_link_libraries(test_clang_enhanced PRIVATE vgre)
add_test(NAME ClangEnhanced COMMAND test_clang_enhanced)
set_tests_properties(ClangEnhanced PROPERTIES TIMEOUT 600 RESOURCE_LOCK vgre_llvm_jit)

add_executable(test_kernel_parser_enhanced compiler/test_kernel_parser_enhanced.cpp)
target_link_libraries(test_kernel_parser_enhanced PRIVATE vgre)
add_test(NAME KernelParserEnhanced COMMAND test_kernel_parser_enhanced)
set_tests_properties(KernelParserEnhanced PROPERTIES TIMEOUT 600 RESOURCE_LOCK vgre_llvm_jit)


add_executable(test_flop_counting compiler/test_flop_counting.cpp)
target_link_libraries(test_flop_counting PRIVATE vgre)
add_test(NAME FLOPCounting COMMAND test_flop_counting)
set_tests_properties(FLOPCounting PROPERTIES TIMEOUT 600 RESOURCE_LOCK vgre_llvm_jit)

add_executable(test_texture_view unit/test_texture_view.cpp)
target_link_libraries(test_texture_view PRIVATE vgre)
add_test(NAME TextureView COMMAND test_texture_view)

add_executable(test_texture_view_advanced unit/test_texture_view_advanced.cpp)
target_link_libraries(test_texture_view_advanced PRIVATE vgre)
add_test(NAME TextureViewAdvanced COMMAND test_texture_view_advanced)

add_executable(test_dirty_page_tracking core/test_dirty_page_tracking.cpp)
target_link_libraries(test_dirty_page_tracking PRIVATE vgre)
add_test(NAME DirtyPageTracking COMMAND test_dirty_page_tracking)

add_executable(test_roofline_autotuner core/test_roofline_autotuner.cpp)
target_link_libraries(test_roofline_autotuner PRIVATE vgre)
add_test(NAME RooflineAutoTuner COMMAND test_roofline_autotuner)
set_tests_properties(RooflineAutoTuner PROPERTIES
    TIMEOUT 120
    RESOURCE_LOCK vgre_llvm_jit)

add_executable(test_memory_concurrency unit/test_memory_concurrency.cpp)
target_link_libraries(test_memory_concurrency PRIVATE vgre)
add_test(NAME MemoryConcurrency COMMAND test_memory_concurrency)

# QUEUE-22: Michael-Scott two-lock queue concurrency tests
add_executable(test_pool_two_lock unit/test_pool_two_lock.cpp)
target_link_libraries(test_pool_two_lock PRIVATE vgre)
add_test(NAME PoolTwoLock COMMAND test_pool_two_lock)
set_tests_properties(PoolTwoLock PROPERTIES TIMEOUT 60)

# QUEUE-23: Halley's method fast inverse square root
add_executable(test_vector_engine_rsqrt unit/test_vector_engine_rsqrt.cpp)
target_link_libraries(test_vector_engine_rsqrt PRIVATE vgre)
add_test(NAME VectorEngineRsqrt COMMAND test_vector_engine_rsqrt)
set_tests_properties(VectorEngineRsqrt PROPERTIES TIMEOUT 30)

# QUEUE-24: cuBLAS GemmBatched all precisions (FP16, BF16, INT8, FP64)
add_executable(test_cublas_gemm_ex_batched api/test_cublas_gemm_ex_batched.cpp)
target_link_libraries(test_cublas_gemm_ex_batched vgre)
add_test(NAME CublasGemmExBatched COMMAND test_cublas_gemm_ex_batched)
set_tests_properties(CublasGemmExBatched PROPERTIES TIMEOUT 30 ENVIRONMENT "VGRE_BENCHMARK=OFF;VGRE_IPC_MODE=OFF")

# QUEUE-25: mTLS — certificate pinning + ECDSA-P256 verify
# All 5 tests are self-contained (in-memory keys/certs via OpenSSL EVP APIs).
# The OpenSSL-dependent tests are guarded by #ifdef VGRE_ENABLE_SSL inside the
# source file; the fallback test runs on every build.
add_executable(test_mtls unit/test_mtls.cpp)
target_link_libraries(test_mtls PRIVATE vgre)
# test_mtls directly calls OpenSSL EVP APIs (keygen, sign, etc.) to generate
# in-memory test keys — it must link OpenSSL::Crypto directly.
find_package(OpenSSL QUIET)
if(OpenSSL_FOUND)
    target_link_libraries(test_mtls PRIVATE OpenSSL::SSL OpenSSL::Crypto)
endif()
add_test(NAME mTLSCertPinningECDSA COMMAND test_mtls)
set_tests_properties(mTLSCertPinningECDSA PROPERTIES TIMEOUT 30)

add_executable(test_events api/test_events.cpp)
target_link_libraries(test_events PRIVATE vgre_cudart)
add_test(NAME Events COMMAND test_events)

add_executable(test_stream_wait_event api/test_stream_wait_event.cpp)
target_link_libraries(test_stream_wait_event PRIVATE vgre_cudart)
add_test(NAME StreamWaitEvent COMMAND test_stream_wait_event)

add_executable(test_event_query api/test_event_query.cpp)
target_link_libraries(test_event_query PRIVATE vgre_cudart)
add_test(NAME EventQuery COMMAND test_event_query)

add_executable(test_stream_add_callback api/test_stream_add_callback.cpp)
target_link_libraries(test_stream_add_callback PRIVATE vgre_cudart)
add_test(NAME StreamAddCallback COMMAND test_stream_add_callback)

add_executable(test_launch_host_func api/test_launch_host_func.cpp)
target_link_libraries(test_launch_host_func PRIVATE vgre_cudart)
add_test(NAME LaunchHostFunc COMMAND test_launch_host_func)

add_executable(test_error_name_string api/test_error_name_string.cpp)
target_link_libraries(test_error_name_string PRIVATE vgre_cudart)
add_test(NAME ErrorNameString COMMAND test_error_name_string)

add_executable(test_memcpy_symbol api/test_memcpy_symbol.cpp)
target_link_libraries(test_memcpy_symbol PRIVATE vgre_cudart)
add_test(NAME MemcpySymbol COMMAND test_memcpy_symbol)

add_executable(test_malloc_array api/test_malloc_array.cpp)
target_link_libraries(test_malloc_array PRIVATE vgre_cudart)
add_test(NAME MallocArray COMMAND test_malloc_array)

add_executable(test_pointer_attributes api/test_pointer_attributes.cpp)
target_link_libraries(test_pointer_attributes PRIVATE vgre_cudart)
add_test(NAME PointerAttributes COMMAND test_pointer_attributes)

add_executable(test_cublas_level1 api/test_cublas_level1.cpp)
target_link_libraries(test_cublas_level1 PRIVATE vgre)
add_test(NAME CublasLevel1 COMMAND test_cublas_level1)

add_executable(test_cublas_pointer_atomics_mode api/test_cublas_pointer_atomics_mode.cpp)
target_link_libraries(test_cublas_pointer_atomics_mode PRIVATE vgre)
add_test(NAME CublasPointerAtomicsMode COMMAND test_cublas_pointer_atomics_mode)

add_executable(test_cublas_level2 api/test_cublas_level2.cpp)
target_link_libraries(test_cublas_level2 PRIVATE vgre)
add_test(NAME CublasLevel2 COMMAND test_cublas_level2)

add_executable(test_cublas_level3 api/test_cublas_level3.cpp)
target_link_libraries(test_cublas_level3 PRIVATE vgre)
add_test(NAME CublasLevel3 COMMAND test_cublas_level3)

add_executable(test_cublas_complex api/test_cublas_complex.cpp)
target_link_libraries(test_cublas_complex PRIVATE vgre)
add_test(NAME CublasComplex COMMAND test_cublas_complex)

add_executable(test_cufft api/test_cufft.cpp)
target_link_libraries(test_cufft PRIVATE vgre_cudart)
add_test(NAME CufftFFT COMMAND test_cufft)

add_executable(test_cufft_xt api/test_cufft_xt.cpp)
target_link_libraries(test_cufft_xt PRIVATE vgre_cudart)
add_test(NAME CufftXt COMMAND test_cufft_xt)

# Radix-4 FFT correctness (QUEUE-04)
add_executable(test_cufft_radix4 api/test_cufft_radix4.cpp)
target_link_libraries(test_cufft_radix4 PRIVATE vgre_cudart)
add_test(NAME CufftRadix4 COMMAND test_cufft_radix4)
set_tests_properties(CufftRadix4 PROPERTIES TIMEOUT 60)

# Winograd F(4x4,3x3) forward-pass correctness (QUEUE-05)
add_executable(test_cudnn_winograd api/test_cudnn_winograd.cpp)
target_link_libraries(test_cudnn_winograd PRIVATE vgre)
add_test(NAME CUDNNWinograd COMMAND test_cudnn_winograd)
set_tests_properties(CUDNNWinograd PROPERTIES TIMEOUT 120)

# AES-256-GCM authenticated encryption (QUEUE-06)
add_executable(test_aes_gcm api/test_aes_gcm.cpp)
target_link_libraries(test_aes_gcm PRIVATE vgre)
add_test(NAME AES256GCM COMMAND test_aes_gcm)
set_tests_properties(AES256GCM PROPERTIES TIMEOUT 60)

# Min-segment tree stream dependency tracker (QUEUE-07)
add_executable(test_stream_dep_tracker unit/test_stream_dep_tracker.cpp)
target_link_libraries(test_stream_dep_tracker PRIVATE vgre)
add_test(NAME StreamDepTracker COMMAND test_stream_dep_tracker)
set_tests_properties(StreamDepTracker PROPERTIES TIMEOUT 30)

# NCCL Ring AllReduce — 2(N-1) step bandwidth-optimal ring (QUEUE-09)
add_executable(test_nccl_ring_allreduce unit/test_nccl_ring_allreduce.cpp)
target_link_libraries(test_nccl_ring_allreduce PRIVATE vgre)
add_test(NAME NCCLRingAllReduce COMMAND test_nccl_ring_allreduce)
set_tests_properties(NCCLRingAllReduce PROPERTIES TIMEOUT 60)

# PTX warp shuffle emulation — group-aware permutation network (QUEUE-10)
# Self-contained: provides its own JIT runtime stubs; does not link vgre.
add_executable(test_warp_shuffle unit/test_warp_shuffle.cpp)
target_include_directories(test_warp_shuffle PRIVATE
    ${CMAKE_SOURCE_DIR}/include)
add_test(NAME WarpShuffle COMMAND test_warp_shuffle)
set_tests_properties(WarpShuffle PROPERTIES TIMEOUT 30)

# cuBLAS DOT/NRM2/ASUM Kahan summation + cublasDnrm2 (QUEUE-08)
add_executable(test_kahan_level1 api/test_kahan_level1.cpp)
target_link_libraries(test_kahan_level1 PRIVATE vgre)
add_test(NAME KahanLevel1 COMMAND test_kahan_level1)
set_tests_properties(KahanLevel1 PROPERTIES TIMEOUT 60)

add_executable(test_curand api/test_curand.cpp)
target_link_libraries(test_curand PRIVATE vgre_cudart)
add_test(NAME CuRANDFunctional COMMAND test_curand)

add_executable(test_cusolver api/test_cusolver.cpp)
target_link_libraries(test_cusolver PRIVATE vgre_cudart)
add_test(NAME CusolverLAPACK COMMAND test_cusolver)

add_executable(test_cusolver_sp api/test_cusolver_sp.cpp)
target_link_libraries(test_cusolver_sp PRIVATE vgre_cudart)
add_test(NAME CusolverSparse COMMAND test_cusolver_sp)

add_executable(test_cusolver_complex api/test_cusolver_complex.cpp)
target_link_libraries(test_cusolver_complex PRIVATE vgre_cudart)
add_test(NAME CusolverComplexCZ COMMAND test_cusolver_complex)

add_executable(test_cusparse api/test_cusparse.cpp)
target_link_libraries(test_cusparse PRIVATE vgre_cudart)
add_test(NAME CuSPARSEFunctional COMMAND test_cusparse)

add_executable(test_cublaslt api/test_cublaslt.cpp)
target_link_libraries(test_cublaslt PRIVATE vgre_cudart)
add_test(NAME CuBLASLtFunctional COMMAND test_cublaslt)

add_executable(test_cudnn_backward_data api/test_cudnn_backward_data.cpp)
target_link_libraries(test_cudnn_backward_data PRIVATE vgre)
add_test(NAME CudnnBackwardData COMMAND test_cudnn_backward_data)

add_executable(test_cudnn_backward_filter api/test_cudnn_backward_filter.cpp)
target_link_libraries(test_cudnn_backward_filter PRIVATE vgre)
add_test(NAME CudnnBackwardFilter COMMAND test_cudnn_backward_filter)

add_executable(test_cudnn_bn_forward_training api/test_cudnn_bn_forward_training.cpp)
target_link_libraries(test_cudnn_bn_forward_training PRIVATE vgre)
add_test(NAME CudnnBnForwardTraining COMMAND test_cudnn_bn_forward_training)

add_executable(test_cudnn_bn_backward api/test_cudnn_bn_backward.cpp)
target_link_libraries(test_cudnn_bn_backward PRIVATE vgre)
add_test(NAME CudnnBnBackward COMMAND test_cudnn_bn_backward)

add_executable(test_cudnn_softmax_backward api/test_cudnn_softmax_backward.cpp)
target_link_libraries(test_cudnn_softmax_backward PRIVATE vgre)
add_test(NAME CudnnSoftmaxBackward COMMAND test_cudnn_softmax_backward)

add_executable(test_cudnn_pooling_backward api/test_cudnn_pooling_backward.cpp)
target_link_libraries(test_cudnn_pooling_backward PRIVATE vgre)
add_test(NAME CudnnPoolingBackward COMMAND test_cudnn_pooling_backward)

add_executable(test_cudnn_dropout api/test_cudnn_dropout.cpp)
target_link_libraries(test_cudnn_dropout PRIVATE vgre)
add_test(NAME CudnnDropout COMMAND test_cudnn_dropout)

add_executable(test_cudnn_tensor_ops api/test_cudnn_tensor_ops.cpp)
target_link_libraries(test_cudnn_tensor_ops PRIVATE vgre)
add_test(NAME CudnnTensorOps COMMAND test_cudnn_tensor_ops)

add_executable(test_cudnn_rnn api/test_cudnn_rnn.cpp)
target_link_libraries(test_cudnn_rnn PRIVATE vgre)
add_test(NAME CudnnRNN COMMAND test_cudnn_rnn)

add_executable(test_cudnn_attention api/test_cudnn_attention.cpp)
target_link_libraries(test_cudnn_attention PRIVATE vgre)
add_test(NAME CudnnAttention COMMAND test_cudnn_attention)

add_executable(test_cudnn_backend_v8 api/test_cudnn_backend_v8.cpp)
target_link_libraries(test_cudnn_backend_v8 PRIVATE vgre)
add_test(NAME CudnnBackendV8 COMMAND test_cudnn_backend_v8)

add_executable(test_cudnn_resample_bwd api/test_cudnn_resample_bwd.cpp)
target_link_libraries(test_cudnn_resample_bwd PRIVATE vgre)
add_test(NAME CudnnResampleBwd COMMAND test_cudnn_resample_bwd)

add_executable(test_cudnn_graph_v9 api/test_cudnn_graph_v9.cpp)
target_link_libraries(test_cudnn_graph_v9 PRIVATE vgre)
add_test(NAME CudnnGraphV9 COMMAND test_cudnn_graph_v9)

add_executable(test_ptx_cross_module_link api/test_ptx_cross_module_link.cpp)
target_link_libraries(test_ptx_cross_module_link PRIVATE vgre_cudart)
add_test(NAME PTXCrossModuleLink COMMAND test_ptx_cross_module_link)

add_executable(test_ptx_fp8 api/test_ptx_fp8.cpp)
target_link_libraries(test_ptx_fp8 PRIVATE vgre_cudart)
add_test(NAME PTXFp8Translation COMMAND test_ptx_fp8)

add_executable(test_sass_detection api/test_sass_detection.cpp)
target_link_libraries(test_sass_detection PRIVATE vgre_cudart)
add_test(NAME SASSDetection COMMAND test_sass_detection)

add_executable(test_cupti_hw_counters api/test_cupti_hw_counters.cpp)
target_link_libraries(test_cupti_hw_counters PRIVATE vgre_api vgre_cudart)
add_test(NAME CuptiHwCounters COMMAND test_cupti_hw_counters)

add_executable(test_hgemm_batched api/test_hgemm_batched.cpp)
target_link_libraries(test_hgemm_batched PRIVATE vgre_cudart)
add_test(NAME HgemmBatched COMMAND test_hgemm_batched)

add_executable(test_cusparse_spsm_sddmm api/test_cusparse_spsm_sddmm.cpp)
target_link_libraries(test_cusparse_spsm_sddmm PRIVATE vgre_cudart)
add_test(NAME CuSPARSESpSMSDDMM COMMAND test_cusparse_spsm_sddmm)

add_executable(test_cusparse_spvec api/test_cusparse_spvec.cpp)
target_link_libraries(test_cusparse_spvec PRIVATE vgre_cudart)
add_test(NAME CuSPARSESpVec COMMAND test_cusparse_spvec)

add_executable(test_cudnn_spatial_tf api/test_cudnn_spatial_tf.cpp)
target_link_libraries(test_cudnn_spatial_tf PRIVATE vgre)
add_test(NAME CuDNNSpatialTf COMMAND test_cudnn_spatial_tf)

add_executable(test_cublas_complex_batched api/test_cublas_complex_batched.cpp)
target_link_libraries(test_cublas_complex_batched PRIVATE vgre_cudart)
add_test(NAME CuBLASComplexBatched COMMAND test_cublas_complex_batched)

add_executable(test_cudnn_normalization api/test_cudnn_normalization.cpp)
target_link_libraries(test_cudnn_normalization PRIVATE vgre)
add_test(NAME CuDNNNormalization COMMAND test_cudnn_normalization)

add_executable(test_curand_mtgp32 api/test_curand_mtgp32.cpp)
target_link_libraries(test_curand_mtgp32 PRIVATE vgre_cudart)
add_test(NAME CuRANDMTGP32Uniformity COMMAND test_curand_mtgp32)

add_executable(test_cusolver_batched api/test_cusolver_batched.cpp)
target_link_libraries(test_cusolver_batched PRIVATE vgre_cudart)
add_test(NAME CusolverBatchedAPIs COMMAND test_cusolver_batched)

add_executable(test_memset2d_graph_kernelnode api/test_memset2d_graph_kernelnode.cpp)
target_link_libraries(test_memset2d_graph_kernelnode PRIVATE vgre_cudart)
add_test(NAME Memset2DGraphKernelNode COMMAND test_memset2d_graph_kernelnode)

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

# CUDA Virtual Memory unit tests (QUEUE-13)
add_executable(test_virtual_memory unit/test_virtual_memory.cpp)
target_link_libraries(test_virtual_memory PRIVATE vgre)
add_test(NAME VirtualMemory COMMAND test_virtual_memory)
set_tests_properties(VirtualMemory PROPERTIES TIMEOUT 30)

# TCP reconnect decorrelated jitter backoff (QUEUE-16)
add_executable(test_reconnect_backoff unit/test_reconnect_backoff.cpp)
add_test(NAME ReconnectBackoff COMMAND test_reconnect_backoff)
set_tests_properties(ReconnectBackoff PROPERTIES TIMEOUT 30)

# OpenCL NDRange arithmetic: 3D grid, local_size, offset (QUEUE-18)
add_executable(test_opencl_ndrange unit/test_opencl_ndrange.cpp)
add_test(NAME OpenCLNDRange COMMAND test_opencl_ndrange)
set_tests_properties(OpenCLNDRange PROPERTIES TIMEOUT 30)

# AVX2 blocked matrix transpose with cache-oblivious tiling (QUEUE-19)
add_executable(test_cache_oblivious_transpose unit/test_cache_oblivious_transpose.cpp)
target_link_libraries(test_cache_oblivious_transpose PRIVATE vgre)
add_test(NAME CacheObliviousTranspose COMMAND test_cache_oblivious_transpose)
set_tests_properties(CacheObliviousTranspose PROPERTIES TIMEOUT 30)

# cudaMemcpy3D / cudaMemcpy3DAsync all 4 transfer directions (QUEUE-21)
add_executable(test_memcpy3d unit/test_memcpy3d.cpp)
target_link_libraries(test_memcpy3d PRIVATE vgre)
add_test(NAME Memcpy3D COMMAND test_memcpy3d)
set_tests_properties(Memcpy3D PROPERTIES TIMEOUT 30)


add_executable(test_tcp_cluster_preservation advanced/test_tcp_cluster_preservation.cpp)
target_link_libraries(test_tcp_cluster_preservation PRIVATE vgre)
add_test(NAME TCPClusterPreservation COMMAND test_tcp_cluster_preservation)

add_executable(test_tcp_cluster_security_policy advanced/test_tcp_cluster_security_policy.cpp)
target_link_libraries(test_tcp_cluster_security_policy PRIVATE vgre)
add_test(NAME TCPClusterSecurityPolicy COMMAND test_tcp_cluster_security_policy)

add_executable(test_tcp_cluster_architecture advanced/test_tcp_cluster_architecture.cpp)
target_link_libraries(test_tcp_cluster_architecture PRIVATE vgre)
add_test(NAME TCPClusterArchitecture COMMAND test_tcp_cluster_architecture
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(test_tcp_cluster_comprehensive_preservation advanced/test_tcp_cluster_comprehensive_preservation.cpp)
target_link_libraries(test_tcp_cluster_comprehensive_preservation PRIVATE vgre)
add_test(NAME TCPClusterComprehensivePreservation COMMAND test_tcp_cluster_comprehensive_preservation)

add_executable(test_tcp_cluster_hybrid_auth advanced/test_tcp_cluster_hybrid_auth.cpp)
target_link_libraries(test_tcp_cluster_hybrid_auth PRIVATE vgre)
add_test(NAME TCPClusterHybridAuth COMMAND test_tcp_cluster_hybrid_auth)

# D2: Wire up the security hybrid integration test (was on disk but not in CTest)
add_executable(test_tcp_cluster_security_hybrid integration/test_tcp_cluster_security_hybrid.cpp)
target_link_libraries(test_tcp_cluster_security_hybrid PRIVATE vgre)
add_test(NAME TCPClusterSecurityHybrid COMMAND test_tcp_cluster_security_hybrid)
set_tests_properties(TCPClusterSecurityHybrid PROPERTIES TIMEOUT 60)

add_executable(test_texture_manager core/test_texture_manager.cpp)
target_link_libraries(test_texture_manager PRIVATE vgre)
add_test(NAME TextureManager COMMAND test_texture_manager)

add_executable(test_memory_pool unit/test_memory_pool.cpp)
target_link_libraries(test_memory_pool PRIVATE vgre)
add_test(NAME MemoryPool COMMAND test_memory_pool)

add_executable(test_memory_scheduler_benchmarks unit/test_memory_scheduler_benchmarks.cpp)
target_link_libraries(test_memory_scheduler_benchmarks PRIVATE vgre)
add_test(NAME MemorySchedulerBenchmarks COMMAND test_memory_scheduler_benchmarks)
set_tests_properties(MemorySchedulerBenchmarks PROPERTIES TIMEOUT 180)

add_executable(test_scheduler_benchmark unit/test_scheduler_benchmark.cpp)
target_link_libraries(test_scheduler_benchmark PRIVATE vgre)
add_test(NAME SchedulerBenchmark COMMAND test_scheduler_benchmark)
set_tests_properties(SchedulerBenchmark PROPERTIES TIMEOUT 120)

add_executable(test_vector_engine_benchmark unit/test_vector_engine_benchmark.cpp)
target_link_libraries(test_vector_engine_benchmark PRIVATE vgre)
add_test(NAME VectorEngineBenchmark COMMAND test_vector_engine_benchmark)
set_tests_properties(VectorEngineBenchmark PROPERTIES TIMEOUT 120)

add_executable(test_matmul_amx unit/test_matmul_amx.cpp)
target_link_libraries(test_matmul_amx PRIVATE vgre)
add_test(NAME MatMulAMX COMMAND test_matmul_amx)
set_tests_properties(MatMulAMX PROPERTIES TIMEOUT 60)

add_executable(test_memory_manager_benchmark unit/test_memory_manager_benchmark.cpp)
target_link_libraries(test_memory_manager_benchmark PRIVATE vgre)
add_test(NAME MemoryManagerBenchmark COMMAND test_memory_manager_benchmark)
set_tests_properties(MemoryManagerBenchmark PROPERTIES TIMEOUT 180)

# ── TCP Cluster Module Unit Tests (Phase 5, Tasks 33.4–39.4) ──────────────
add_executable(test_packet_handler
    advanced/tcp_cluster/test_packet_handler.cpp)
target_link_libraries(test_packet_handler PRIVATE vgre)
add_test(NAME PacketHandler COMMAND test_packet_handler)

add_executable(test_collective_ops_manager
    advanced/tcp_cluster/test_collective_ops_manager.cpp)
target_link_libraries(test_collective_ops_manager PRIVATE vgre)
add_test(NAME CollectiveOpsManager COMMAND test_collective_ops_manager)

add_executable(test_connection_manager
    advanced/tcp_cluster/test_connection_manager.cpp)
target_link_libraries(test_connection_manager PRIVATE vgre)
add_test(NAME ConnectionManager COMMAND test_connection_manager)

add_executable(test_security_manager
    advanced/tcp_cluster/test_security_manager.cpp)
target_link_libraries(test_security_manager PRIVATE vgre)
add_test(NAME SecurityManager COMMAND test_security_manager)

add_executable(test_discovery_manager
    advanced/tcp_cluster/test_discovery_manager.cpp)
target_link_libraries(test_discovery_manager PRIVATE vgre)
add_test(NAME DiscoveryManager COMMAND test_discovery_manager)

add_executable(test_dispatch_manager
    advanced/tcp_cluster/test_dispatch_manager.cpp)
target_link_libraries(test_dispatch_manager PRIVATE vgre)
add_test(NAME DispatchManager COMMAND test_dispatch_manager)

add_executable(test_memory_sync_manager
    advanced/tcp_cluster/test_memory_sync_manager.cpp)
target_link_libraries(test_memory_sync_manager PRIVATE vgre)
add_test(NAME MemorySyncManager COMMAND test_memory_sync_manager)


# ── TCP Cluster Modular Unit Tests (Hardening Phase) ──────────────────────
add_executable(test_tcp_cluster_testability
    advanced/test_tcp_cluster_testability.cpp
    advanced/tcp_cluster/mocks.cpp)
target_link_libraries(test_tcp_cluster_testability PRIVATE vgre)
target_include_directories(test_tcp_cluster_testability PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME TCPClusterTestability COMMAND test_tcp_cluster_testability)
set_tests_properties(TCPClusterTestability PROPERTIES TIMEOUT 60)

add_executable(test_tcp_cluster_preservation_testability
    advanced/test_tcp_cluster_preservation_testability.cpp
    advanced/tcp_cluster/mocks.cpp)
target_link_libraries(test_tcp_cluster_preservation_testability PRIVATE vgre)
target_include_directories(test_tcp_cluster_preservation_testability PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME TCPClusterPreservationTestability COMMAND test_tcp_cluster_preservation_testability)
set_tests_properties(TCPClusterPreservationTestability PROPERTIES TIMEOUT 60)

add_executable(test_tcp_cluster_all_operations_preservation
    advanced/test_tcp_cluster_all_operations_preservation.cpp
    advanced/tcp_cluster/mocks.cpp)
target_link_libraries(test_tcp_cluster_all_operations_preservation PRIVATE vgre)
target_include_directories(test_tcp_cluster_all_operations_preservation PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME TCPClusterAllOpsPreservation COMMAND test_tcp_cluster_all_operations_preservation)
set_tests_properties(TCPClusterAllOpsPreservation PROPERTIES TIMEOUT 600 RESOURCE_LOCK vgre_tcp_cluster_port)

add_executable(test_diagnostic_logger
    advanced/test_diagnostic_logger.cpp)
target_link_libraries(test_diagnostic_logger PRIVATE vgre)
add_test(NAME DiagnosticLogger COMMAND test_diagnostic_logger)

# ── Cross-Platform Integration Tests ────────────────────────────────────────
add_executable(test_cross_platform_tcp_cluster
    integration/test_cross_platform_tcp_cluster.cpp)
target_link_libraries(test_cross_platform_tcp_cluster PRIVATE vgre)
add_test(NAME CrossPlatformTCPCluster COMMAND test_cross_platform_tcp_cluster)
set_tests_properties(CrossPlatformTCPCluster PROPERTIES TIMEOUT 120)

add_executable(test_cross_platform_networking
    integration/test_cross_platform_networking.cpp)
target_link_libraries(test_cross_platform_networking PRIVATE vgre)
add_test(NAME CrossPlatformNetworking COMMAND test_cross_platform_networking)
set_tests_properties(CrossPlatformNetworking PROPERTIES TIMEOUT 60)

add_executable(test_cross_platform_socket_compatibility
    integration/test_cross_platform_socket_compatibility.cpp)
target_link_libraries(test_cross_platform_socket_compatibility PRIVATE vgre)
add_test(NAME CrossPlatformSocketCompat COMMAND test_cross_platform_socket_compatibility)
set_tests_properties(CrossPlatformSocketCompat PROPERTIES TIMEOUT 60)

add_executable(test_cross_platform_error_handling
    integration/test_cross_platform_error_handling.cpp)
target_link_libraries(test_cross_platform_error_handling PRIVATE vgre)
add_test(NAME CrossPlatformErrorHandling COMMAND test_cross_platform_error_handling)

add_executable(test_cross_platform_memory_management
    integration/test_cross_platform_memory_management.cpp)
target_link_libraries(test_cross_platform_memory_management PRIVATE vgre)
add_test(NAME CrossPlatformMemoryManagement COMMAND test_cross_platform_memory_management)

add_executable(test_cross_platform_windows_access_violation_prevention
    integration/test_cross_platform_windows_access_violation_prevention.cpp)
target_link_libraries(test_cross_platform_windows_access_violation_prevention PRIVATE vgre)
add_test(NAME CrossPlatformWinAVPrevention
    COMMAND test_cross_platform_windows_access_violation_prevention)

add_executable(test_tcp_cluster_preservation_integration
    integration/test_tcp_cluster_preservation.cpp)
target_link_libraries(test_tcp_cluster_preservation_integration PRIVATE vgre)
add_test(NAME TCPClusterPreservationIntegration
    COMMAND test_tcp_cluster_preservation_integration)
set_tests_properties(TCPClusterPreservationIntegration PROPERTIES TIMEOUT 120)

add_executable(test_tcp_cluster_windows_bug_exploration
    integration/test_tcp_cluster_windows_bug_exploration.cpp)
target_link_libraries(test_tcp_cluster_windows_bug_exploration PRIVATE vgre)
add_test(NAME TCPClusterWindowsBugExploration
    COMMAND test_tcp_cluster_windows_bug_exploration)
set_tests_properties(TCPClusterWindowsBugExploration PROPERTIES TIMEOUT 60)

# ── Unit Safety Tests ────────────────────────────────────────────────────────
add_executable(test_tcp_cluster_struct_packing
    unit/test_tcp_cluster_struct_packing.cpp)
target_link_libraries(test_tcp_cluster_struct_packing PRIVATE vgre)
add_test(NAME TCPClusterStructPacking COMMAND test_tcp_cluster_struct_packing)

add_executable(test_security_handshake_buffer_safety
    unit/test_security_handshake_buffer_safety.cpp)
target_link_libraries(test_security_handshake_buffer_safety PRIVATE vgre)
add_test(NAME SecurityHandshakeBufferSafety
    COMMAND test_security_handshake_buffer_safety)

add_executable(test_udp_discovery_buffer_safety
    unit/test_udp_discovery_buffer_safety.cpp)
target_link_libraries(test_udp_discovery_buffer_safety PRIVATE vgre)
add_test(NAME UDPDiscoveryBufferSafety COMMAND test_udp_discovery_buffer_safety)

# ── Kernel Fusion Integration Test ─────────────────────────────────────────
add_executable(test_kernel_fusion integration/test_kernel_fusion.cpp)
target_link_libraries(test_kernel_fusion PRIVATE vgre)
add_test(NAME KernelFusionIntegration COMMAND test_kernel_fusion)
set_tests_properties(KernelFusionIntegration PROPERTIES TIMEOUT 30)

# ── WebSocket Transport Integration Test ───────────────────────────────────
add_executable(test_websocket_transport integration/test_websocket_transport.cpp)
target_link_libraries(test_websocket_transport PRIVATE vgre)
add_test(NAME WebSocketTransportIntegration COMMAND test_websocket_transport)
set_tests_properties(WebSocketTransportIntegration PROPERTIES TIMEOUT 10)

# ── Python end-to-end test (CUDA-on-CPU via C API) ─────────────────────────
# Use the project .venv interpreter so numpy and other dependencies are available.
set(VGRE_VENV_PYTHON "${CMAKE_SOURCE_DIR}/.venv/bin/python3")
if(EXISTS "${VGRE_VENV_PYTHON}")
    add_test(NAME PythonCAPIVectorAdd
        COMMAND "${VGRE_VENV_PYTHON}" ${CMAKE_SOURCE_DIR}/test_cuda_on_cpu.py)
    set_tests_properties(PythonCAPIVectorAdd PROPERTIES
        ENVIRONMENT "VGRE_BENCHMARK=OFF;VGRE_IPC_MODE=OFF;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}"
        LABELS "python;e2e"
    )
else()
    # Fallback: use the system Python3 found by CMake.
    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_FOUND)
        add_test(NAME PythonCAPIVectorAdd
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/test_cuda_on_cpu.py)
        set_tests_properties(PythonCAPIVectorAdd PROPERTIES
            ENVIRONMENT "VGRE_BENCHMARK=OFF;VGRE_IPC_MODE=OFF;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}"
            LABELS "python;e2e"
        )
    endif()
endif()

# ── Global Test Configuration ──────────────────────────────────────────────
# Disable background hardware benchmarks and optional IPC services during
# tests to ensure deterministic performance and prevent CPU resource
# contention/leakage.
#
# LD_LIBRARY_PATH: Force tests to use the freshly-built libvgre.so from the
# build tree. On Linux, RUNPATH is searched after the ldconfig cache, so a
# stale system-installed libvgre (e.g. /usr/local/lib/libvgre.so.0) would
# otherwise be loaded instead of the current build — causing ABI mismatches
# and hard-to-diagnose SEGFAULTs.
get_directory_property(vgre_full_test_list TESTS)
set_tests_properties(${vgre_full_test_list} PROPERTIES
    ENVIRONMENT "VGRE_BENCHMARK=OFF;VGRE_IPC_MODE=OFF;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}"
)

# ── Network/TCP serialization ──────────────────────────────────────────────
# All tests that bind to TCP/UDP ports share vgre_tcp_cluster_port so CTest
# never runs two of them at the same time, preventing port conflicts and
# SEGFAULT/timeout under parallel -j execution.
# Network-binding tests — serialized via RESOURCE_LOCK to prevent port conflicts.
# Only tests that actually bind TCP/UDP sockets to localhost ports are listed.
# Pure in-memory unit tests (PacketHandler, ConnectionManager, etc.) are excluded.
set_tests_properties(
    TCPClusterIntegration
    P2PPerformanceIntegration
    test_phase3_cluster
    TCPClusterPreservation
    TCPClusterSecurityPolicy
    TCPClusterArchitecture
    TCPClusterComprehensivePreservation
    TCPClusterHybridAuth
    TCPClusterSecurityHybrid
    TCPClusterPreservationIntegration
    TCPClusterWindowsBugExploration
    TCPClusterTestability
    TCPClusterPreservationTestability
    TCPClusterAllOpsPreservation
    CrossPlatformNetworking
    CrossPlatformSocketCompat
    WebSocketTransportIntegration
    PROPERTIES RESOURCE_LOCK vgre_tcp_cluster_port
)

# ── Python integration tests ─────────────────────────────────────────────────
find_program(PYTHON3 python3)
if(PYTHON3)
    add_test(NAME PyTorchIntegration
             COMMAND ${PYTHON3} ${CMAKE_SOURCE_DIR}/tests/python/test_pytorch.py
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    set_tests_properties(PyTorchIntegration PROPERTIES
                         ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}"
                         TIMEOUT 120
                         LABELS "python"
                         SKIP_REGULAR_EXPRESSION "ModuleNotFoundError|ImportError|No module named")
    add_test(NAME TensorFlowIntegration
             COMMAND ${PYTHON3} ${CMAKE_SOURCE_DIR}/tests/python/test_tensorflow.py
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    set_tests_properties(TensorFlowIntegration PROPERTIES
                         ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}"
                         TIMEOUT 120
                         LABELS "python"
                         SKIP_REGULAR_EXPRESSION "ModuleNotFoundError|ImportError|No module named")
endif()
