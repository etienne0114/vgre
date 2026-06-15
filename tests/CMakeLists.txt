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

# Phase 3, P3-14: CUDA-graph capture-from-stream fidelity (memset node capture).
add_executable(test_graph_capture_fidelity integration/test_graph_capture_fidelity.cpp)
target_link_libraries(test_graph_capture_fidelity PRIVATE vgre)
add_test(NAME GraphCaptureFidelity COMMAND test_graph_capture_fidelity)
set_tests_properties(GraphCaptureFidelity PROPERTIES TIMEOUT 30)

# Phase 3, P3-13: Triton IR (TTIR) frontend — parse + run from IR.
add_executable(test_triton_frontend integration/test_triton_frontend.cpp)
target_link_libraries(test_triton_frontend PRIVATE vgre)
add_test(NAME TritonFrontend COMMAND test_triton_frontend)
set_tests_properties(TritonFrontend PROPERTIES TIMEOUT 30)

# Phase-3 extension API (vgre::ext) — capability modules through the shipped lib.
add_executable(test_vgre_ext integration/test_vgre_ext.cpp)
target_link_libraries(test_vgre_ext PRIVATE vgre)
add_test(NAME Phase3ExtAPI COMMAND test_vgre_ext)
set_tests_properties(Phase3ExtAPI PROPERTIES TIMEOUT 30)

add_executable(test_graph_c_api integration/test_graph_c_api.cpp)
target_link_libraries(test_graph_c_api PRIVATE vgre)
add_test(NAME GraphCAPIIntegration COMMAND test_graph_c_api)

add_executable(test_multi_device integration/test_multi_device.cpp)
target_link_libraries(test_multi_device PRIVATE vgre)
add_test(NAME MultiDeviceIntegration COMMAND test_multi_device)
# Force ≥2 virtual devices so the multi-device path is actually exercised even on
# single-NUMA-node CI hosts (otherwise the test SKIPs).
set_tests_properties(MultiDeviceIntegration PROPERTIES
    ENVIRONMENT "VGRE_VIRTUAL_DEVICE_COUNT=2")

add_executable(test_multi_device_cooperative_comprehensive integration/test_multi_device_cooperative_comprehensive.cpp)
target_link_libraries(test_multi_device_cooperative_comprehensive PRIVATE vgre)
add_test(NAME MultiDeviceCooperativeComprehensive COMMAND test_multi_device_cooperative_comprehensive)
set_tests_properties(MultiDeviceCooperativeComprehensive PROPERTIES
    TIMEOUT 300
    RESOURCE_LOCK vgre_llvm_jit)

add_executable(test_cooperative_groups_partition unit/test_cooperative_groups_partition.cpp)
target_link_libraries(test_cooperative_groups_partition PRIVATE vgre)
add_test(NAME CooperativeGroupsPartition COMMAND test_cooperative_groups_partition)
set_tests_properties(CooperativeGroupsPartition PROPERTIES
    TIMEOUT 120
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

add_executable(test_device_curand integration/test_device_curand.cpp)
target_link_libraries(test_device_curand PRIVATE vgre)
add_test(NAME DeviceCurandIntegration COMMAND test_device_curand)
set_tests_properties(DeviceCurandIntegration PROPERTIES
    TIMEOUT 120
    FIXTURES_REQUIRED KMPRegistrationClean
    RESOURCE_LOCK vgre_llvm_jit)

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
target_link_libraries(test_texture_driver_api vgre_cudart)

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

# QUEUE-27: cudaGraph full replay — topological sort + dependency-ordered
# kernel execution.  Five tests: diamond, multi-root, relaunch, memcpy chain,
# and disconnected components.
add_executable(test_graph_topo_advanced integration/test_graph_topo_advanced.cpp)
target_link_libraries(test_graph_topo_advanced PRIVATE vgre)
add_test(NAME GraphTopoAdvanced COMMAND test_graph_topo_advanced)
set_tests_properties(GraphTopoAdvanced PROPERTIES TIMEOUT 60)

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

add_executable(test_vectorization_hints compiler/test_vectorization_hints.cpp)
target_link_libraries(test_vectorization_hints PRIVATE vgre)
add_test(NAME VectorizationHints COMMAND test_vectorization_hints)
set_tests_properties(VectorizationHints PROPERTIES TIMEOUT 30)

add_executable(test_texture_view unit/test_texture_view.cpp)
target_link_libraries(test_texture_view PRIVATE vgre)
add_test(NAME TextureView COMMAND test_texture_view)

add_executable(test_texture_view_advanced unit/test_texture_view_advanced.cpp)
target_link_libraries(test_texture_view_advanced PRIVATE vgre)
add_test(NAME TextureViewAdvanced COMMAND test_texture_view_advanced)

add_executable(test_dirty_page_tracking core/test_dirty_page_tracking.cpp)
target_link_libraries(test_dirty_page_tracking PRIVATE vgre)
add_test(NAME DirtyPageTracking COMMAND test_dirty_page_tracking)

# FP8 (E4M3/E5M2) minifloat codec — round-trip + subnormal correctness.
add_executable(test_fp8_minifloat core/test_fp8_minifloat.cpp)
target_link_libraries(test_fp8_minifloat PRIVATE vgre)
target_include_directories(test_fp8_minifloat PRIVATE ${PROJECT_SOURCE_DIR}/src/core/math)
add_test(NAME FP8Minifloat COMMAND test_fp8_minifloat)
set_tests_properties(FP8Minifloat PROPERTIES TIMEOUT 30)

# QUEUE-60: Blelloch (1990) parallel prefix scan
add_executable(test_prefix_scan core/test_prefix_scan.cpp)
target_link_libraries(test_prefix_scan PRIVATE vgre)
add_test(NAME BlellochPrefixScan COMMAND test_prefix_scan)

# QUEUE-62: LSD radix sort O(n) for uint32_t keys
add_executable(test_radix_sort core/test_radix_sort.cpp)
target_link_libraries(test_radix_sort PRIVATE vgre)
add_test(NAME RadixSort COMMAND test_radix_sort)

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

# Identity: OIDC/JWT verification (RS256/ES256) + RBAC. Generates in-memory keys
# via OpenSSL EVP, so it links OpenSSL::Crypto directly (like test_mtls).
add_executable(test_jwt_rbac identity/test_jwt_rbac.cpp)
target_link_libraries(test_jwt_rbac PRIVATE vgre)
if(OpenSSL_FOUND)
    target_link_libraries(test_jwt_rbac PRIVATE OpenSSL::SSL OpenSSL::Crypto)
endif()
add_test(NAME IdentityJwtRbac COMMAND test_jwt_rbac)
set_tests_properties(IdentityJwtRbac PROPERTIES TIMEOUT 30)

# Identity: end-to-end /metrics Bearer-JWT protection (real HTTP server + token).
add_executable(test_metrics_jwt identity/test_metrics_jwt.cpp)
target_link_libraries(test_metrics_jwt PRIVATE vgre)
if(OpenSSL_FOUND)
    target_link_libraries(test_metrics_jwt PRIVATE OpenSSL::SSL OpenSSL::Crypto)
endif()
add_test(NAME IdentityMetricsJwt COMMAND test_metrics_jwt)
set_tests_properties(IdentityMetricsJwt PROPERTIES TIMEOUT 30)

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

add_executable(test_cublas_workspace api/test_cublas_workspace.cpp)
target_link_libraries(test_cublas_workspace PRIVATE vgre)
add_test(NAME CublasWorkspace COMMAND test_cublas_workspace)

add_executable(test_cublas_strassen api/test_cublas_strassen.cpp)
target_link_libraries(test_cublas_strassen PRIVATE vgre)
add_test(NAME CublasStrassen COMMAND test_cublas_strassen)

# cuDNN RNN peephole LSTM (QUEUE-38)
add_executable(test_cudnn_rnn_peephole api/test_cudnn_rnn_peephole.cpp)
target_link_libraries(test_cudnn_rnn_peephole PRIVATE vgre)
add_test(NAME CudnnRNNPeephole COMMAND test_cudnn_rnn_peephole)

# Secure memory zeroing (QUEUE-39)
add_executable(test_secure_zero unit/test_secure_zero.cpp)
target_link_libraries(test_secure_zero PRIVATE vgre)
add_test(NAME SecureZero COMMAND test_secure_zero)

# cuSolver batch QR factorization (QUEUE-40)
add_executable(test_cusolver_batchqr api/test_cusolver_batchqr.cpp)
target_link_libraries(test_cusolver_batchqr PRIVATE vgre_cudart)
add_test(NAME CusolverBatchQR COMMAND test_cusolver_batchqr)

# cuSolver batch LU factorization (QUEUE-41)
add_executable(test_cusolver_batchlu api/test_cusolver_batchlu.cpp)
target_link_libraries(test_cusolver_batchlu PRIVATE vgre_cudart)
add_test(NAME CusolverBatchLU COMMAND test_cusolver_batchlu)

# cuBLAS DGMM diagonal matrix scaling (QUEUE-42)
add_executable(test_cublas_dgmm api/test_cublas_dgmm.cpp)
target_link_libraries(test_cublas_dgmm PRIVATE vgre)
add_test(NAME CublasDGMM COMMAND test_cublas_dgmm)

# cuBLAS GEAM general matrix add/transpose (QUEUE-43)
add_executable(test_cublas_geam api/test_cublas_geam.cpp)
target_link_libraries(test_cublas_geam PRIVATE vgre)
add_test(NAME CublasGEAM COMMAND test_cublas_geam)

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

# cuFFT 2D/3D test coverage — C2C roundtrip, R2C/C2R, 3D C2C, batched, single-frequency (QUEUE-26)
add_executable(test_cufft_2d_3d unit/test_cufft_2d_3d.cpp)
target_link_libraries(test_cufft_2d_3d PRIVATE vgre_cudart)
add_test(NAME CufftFFT2D3D COMMAND test_cufft_2d_3d)
set_tests_properties(CufftFFT2D3D PROPERTIES TIMEOUT 60)

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

# QUEUE-64: ChaCha20-Poly1305 AEAD
add_executable(test_chacha20_poly1305 api/test_chacha20_poly1305.cpp)
target_link_libraries(test_chacha20_poly1305 PRIVATE vgre)
add_test(NAME ChaCha20Poly1305 COMMAND test_chacha20_poly1305)

# QUEUE-65: cuFFT DCT-II/III via FFT embedding
add_executable(test_cufft_dct api/test_cufft_dct.cpp)
target_link_libraries(test_cufft_dct PRIVATE vgre_cudart)
add_test(NAME CuFFTDCT COMMAND test_cufft_dct)

# QUEUE-66: SkipList ordered event tracker
add_executable(test_skip_list core/test_skip_list.cpp)
target_include_directories(test_skip_list PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_skip_list PRIVATE vgre)
add_test(NAME SkipList COMMAND test_skip_list)

# QUEUE-67: cudnnNormalizationBackward full BN backward
add_executable(test_cudnn_bn_backward_full api/test_cudnn_bn_backward_full.cpp)
target_link_libraries(test_cudnn_bn_backward_full PRIVATE vgre)
add_test(NAME CuDNNBNBackwardFull COMMAND test_cudnn_bn_backward_full)

# QUEUE-69: cudaStreamCreateWithPriority / cudaStreamGetPriority
add_executable(test_stream_priority api/test_stream_priority.cpp)
target_link_libraries(test_stream_priority PRIVATE vgre_cudart)
add_test(NAME StreamPriority COMMAND test_stream_priority)

# QUEUE-70: Auth rate-limiting — exponential backoff for cluster auth failures
add_executable(test_auth_rate_limit unit/test_auth_rate_limit.cpp)
target_link_libraries(test_auth_rate_limit PRIVATE vgre)
add_test(NAME AuthRateLimit COMMAND test_auth_rate_limit)

# QUEUE-73: Fast GELU via degree-7 Horner polynomial (Chebyshev minimax)
add_executable(test_gelu_horner unit/test_gelu_horner.cpp)
target_link_libraries(test_gelu_horner PRIVATE vgre)
add_test(NAME FastGELUHorner7 COMMAND test_gelu_horner)

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

# NCCL ReduceScatter — flat-barrier + ring path correctness
add_executable(test_nccl_reduce_scatter unit/test_nccl_reduce_scatter.cpp)
target_link_libraries(test_nccl_reduce_scatter PRIVATE vgre)
add_test(NAME NCCLReduceScatter COMMAND test_nccl_reduce_scatter)
set_tests_properties(NCCLReduceScatter PROPERTIES TIMEOUT 60)

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

add_executable(test_sparse_view api/test_sparse_view.cpp)
target_link_libraries(test_sparse_view PRIVATE vgre)
target_include_directories(test_sparse_view PRIVATE ${CMAKE_SOURCE_DIR}/src/api/cusparse)
add_test(NAME SparseViewConversions COMMAND test_sparse_view)
set_tests_properties(SparseViewConversions PROPERTIES TIMEOUT 30)

add_executable(test_ptx_translate api/test_ptx_translate.cpp)
target_link_libraries(test_ptx_translate PRIVATE vgre)
add_test(NAME PTXTranslate COMMAND test_ptx_translate)
set_tests_properties(PTXTranslate PROPERTIES TIMEOUT 30)

add_executable(test_metrics_server api/test_metrics_server.cpp)
target_link_libraries(test_metrics_server PRIVATE vgre_cudart)
add_test(NAME MetricsServer COMMAND test_metrics_server)
set_tests_properties(MetricsServer PROPERTIES TIMEOUT 120)

add_executable(test_config_buildinfo api/test_config_buildinfo.cpp)
target_link_libraries(test_config_buildinfo PRIVATE vgre_cudart)
add_test(NAME ConfigBuildInfo COMMAND test_config_buildinfo)
set_tests_properties(ConfigBuildInfo PROPERTIES TIMEOUT 30)

add_executable(test_flash_attention integration/test_flash_attention.cpp)
target_link_libraries(test_flash_attention PRIVATE vgre)
add_test(NAME FlashAttention COMMAND test_flash_attention)
# JIT-compiles its kernels, so serialise it against the other LLVM-JIT tests:
# under -j2 CI load a concurrent clang fork can transiently fail the first
# (cold) compile, which surfaced as a non-deterministic non-causal failure.
set_tests_properties(FlashAttention PROPERTIES TIMEOUT 120 RESOURCE_LOCK vgre_llvm_jit)

add_executable(test_tensorcore_mma integration/test_tensorcore_mma.cpp)
target_link_libraries(test_tensorcore_mma PRIVATE vgre)
add_test(NAME TensorCoreMMA COMMAND test_tensorcore_mma)
# JIT-compiles warp-collective mma kernels — serialise with the other JIT tests.
set_tests_properties(TensorCoreMMA PROPERTIES TIMEOUT 120 RESOURCE_LOCK vgre_llvm_jit)

# Phase 3, P3-4: Hopper warp-group MMA (wgmma).
add_executable(test_wgmma integration/test_wgmma.cpp)
target_link_libraries(test_wgmma PRIVATE vgre)
add_test(NAME WgmmaWarpGroup COMMAND test_wgmma)
set_tests_properties(WgmmaWarpGroup PROPERTIES TIMEOUT 60)

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

# CTC loss forward + backward terminal-state constraint
add_executable(test_cudnn_ctc_loss unit/test_cudnn_ctc_loss.cpp)
target_link_libraries(test_cudnn_ctc_loss PRIVATE vgre)
add_test(NAME CudnnCTCLoss COMMAND test_cudnn_ctc_loss)
set_tests_properties(CudnnCTCLoss PROPERTIES TIMEOUT 30)

add_executable(test_cudnn_resample_bwd api/test_cudnn_resample_bwd.cpp)
target_link_libraries(test_cudnn_resample_bwd PRIVATE vgre)
add_test(NAME CudnnResampleBwd COMMAND test_cudnn_resample_bwd)

add_executable(test_cudnn_graph_v9 api/test_cudnn_graph_v9.cpp)
target_link_libraries(test_cudnn_graph_v9 PRIVATE vgre)
add_test(NAME CudnnGraphV9 COMMAND test_cudnn_graph_v9)

# Track Q: Nsight .nsys-rep export (links SQLite for read-back verification).
add_executable(test_profiler_timeline unit/test_profiler_timeline.cpp)
target_link_libraries(test_profiler_timeline PRIVATE vgre)
add_test(NAME ProfilerTimeline COMMAND test_profiler_timeline)
set_tests_properties(ProfilerTimeline PROPERTIES TIMEOUT 30)

# Security: device-memory bounds checker (red-zone canaries).
add_executable(test_bounds_checker core/test_bounds_checker.cpp)
target_link_libraries(test_bounds_checker PRIVATE vgre)
add_test(NAME BoundsChecker COMMAND test_bounds_checker)
set_tests_properties(BoundsChecker PROPERTIES TIMEOUT 30)

# HIP/ROCm compatibility layer (delegates to the cuda* runtime → links vgre_cudart).
add_executable(test_hip_compat integration/test_hip_compat.cpp)
target_link_libraries(test_hip_compat PRIVATE vgre_cudart)
add_test(NAME HipCompat COMMAND test_hip_compat)
set_tests_properties(HipCompat PROPERTIES TIMEOUT 30)

# QA: chaos fault injector.
add_executable(test_fault_injector common/test_fault_injector.cpp)
target_link_libraries(test_fault_injector PRIVATE vgre)
add_test(NAME ChaosFaultInjector COMMAND test_fault_injector)
set_tests_properties(ChaosFaultInjector PROPERTIES TIMEOUT 30)

# QA: mutation-testing runner — must kill 100% of demo mutants (strong test).
find_program(Python3_EXECUTABLE NAMES python3 python)
# Prefer the project .venv interpreter (has the project's deps); fall back to a
# system python. validate_iac/validate_chart use the in-tree minyaml loader when
# PyYAML is absent, so these pass with any reasonably-complete interpreter.
if(EXISTS "${PROJECT_SOURCE_DIR}/.venv/bin/python")
    set(_vgre_py "${PROJECT_SOURCE_DIR}/.venv/bin/python")
else()
    set(_vgre_py "${Python3_EXECUTABLE}")
endif()
if(Python3_EXECUTABLE)
    add_test(NAME MutationTesting COMMAND ${Python3_EXECUTABLE}
        ${PROJECT_SOURCE_DIR}/tools/testing/mutate.py
        --target ${PROJECT_SOURCE_DIR}/tools/testing/mutation_demo/target.cpp
        --test   ${PROJECT_SOURCE_DIR}/tools/testing/mutation_demo/test.cpp
        --cxx    ${CMAKE_CXX_COMPILER}
        --min-score 1.0)
    set_tests_properties(MutationTesting PROPERTIES TIMEOUT 60)

    # Deploy: validate the VGRE Helm chart (structural + helm lint when present).
    add_test(NAME HelmChart COMMAND ${_vgre_py}
        ${PROJECT_SOURCE_DIR}/deploy/helm/validate_chart.py)
    set_tests_properties(HelmChart PROPERTIES TIMEOUT 30)

    # Deploy: validate Terraform module + GitOps (ArgoCD/Kustomize/Flux) manifests.
    add_test(NAME IaCManifests COMMAND ${_vgre_py}
        ${PROJECT_SOURCE_DIR}/deploy/validate_iac.py)
    set_tests_properties(IaCManifests PROPERTIES TIMEOUT 30)
endif()

# PyTorch backend (§4.2): runs a real PyTorch program on VGRE via a PrivateUse1
# device (alloc/copy → emulated CUDA, matmul → emulated cuBLAS). Uses the project
# .venv python (which has torch); skipped if that interpreter isn't present. The
# first run JIT-compiles the extension against libtorch (hence the long timeout).
if(EXISTS ${PROJECT_SOURCE_DIR}/.venv/bin/python)
    add_test(NAME PyTorchBackend COMMAND ${PROJECT_SOURCE_DIR}/.venv/bin/python
        ${PROJECT_SOURCE_DIR}/frameworks/pytorch/test_vgre_backend.py)
    set_tests_properties(PyTorchBackend PROPERTIES TIMEOUT 600
        ENVIRONMENT "VGRE_BUILD_DIR=${CMAKE_BINARY_DIR}")

    # JAX end-to-end (§4.2 step 3): lower a jax.jit fn to StableHLO, translate it to
    # VGRE HLO via the builder C ABI, execute on the engine, and assert the result
    # matches jax's own CPU backend. Self-skips at runtime if jax isn't installed.
    add_test(NAME JaxStableHloBackend COMMAND ${PROJECT_SOURCE_DIR}/.venv/bin/python
        ${PROJECT_SOURCE_DIR}/frameworks/jax/test_jax_e2e.py)
    set_tests_properties(JaxStableHloBackend PROPERTIES TIMEOUT 300
        ENVIRONMENT "VGRE_BUILD_DIR=${CMAKE_BINARY_DIR}")

    # TensorFlow end-to-end (§4.2): XLA-compile a tf.function, dump its StableHLO in
    # an isolated subprocess (TF + jaxlib MLIR cannot co-load), translate to VGRE HLO
    # and execute, asserting against TF's own result. Self-skips if tf isn't present.
    add_test(NAME TensorFlowStableHloBackend COMMAND ${PROJECT_SOURCE_DIR}/.venv/bin/python
        ${PROJECT_SOURCE_DIR}/frameworks/tf/test_tf_e2e.py)
    set_tests_properties(TensorFlowStableHloBackend PROPERTIES TIMEOUT 400
        ENVIRONMENT "VGRE_BUILD_DIR=${CMAKE_BINARY_DIR}")

    # Keras 3 high-level models end-to-end (§4.2): full CNN / BatchNorm & LayerNorm
    # MLPs / a Transformer (MultiHeadAttention) block lowered via the Keras JAX
    # backend and run on VGRE vs Keras. Self-skips if keras isn't installed.
    add_test(NAME KerasModelsBackend COMMAND ${PROJECT_SOURCE_DIR}/.venv/bin/python
        ${PROJECT_SOURCE_DIR}/frameworks/keras/test_keras_e2e.py)
    set_tests_properties(KerasModelsBackend PROPERTIES TIMEOUT 500
        ENVIRONMENT "VGRE_BUILD_DIR=${CMAKE_BINARY_DIR}")
endif()

# WASM target (§3.3): compile VGRE compute kernels to wasm32 (clang) and run them
# in a WebAssembly sandbox (node). The test self-skips if clang can't target wasm.
find_program(NODE_EXE NAMES node nodejs)
if(NODE_EXE)
    add_test(NAME WasmKernels COMMAND ${NODE_EXE} ${PROJECT_SOURCE_DIR}/frameworks/wasm/test_wasm.mjs)
    set_tests_properties(WasmKernels PROPERTIES TIMEOUT 60)
endif()

# RISC-V target (§3.3): cross-compile the same kernels to riscv64 and run them
# under QEMU. Self-skips if the riscv64 toolchain / qemu-riscv64 isn't installed.
if(Python3_EXECUTABLE)
    add_test(NAME RiscvKernels COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/frameworks/riscv/test_riscv.py)
    set_tests_properties(RiscvKernels PROPERTIES TIMEOUT 60)
endif()

# Compliance: tamper-evident audit log + GDPR crypto-erasure.
add_executable(test_audit_log compliance/test_audit_log.cpp)
target_link_libraries(test_audit_log PRIVATE vgre)
add_test(NAME ComplianceAuditLog COMMAND test_audit_log)
set_tests_properties(ComplianceAuditLog PROPERTIES TIMEOUT 30)

# Compliance: data classification + policy-as-code engine.
add_executable(test_policy_engine compliance/test_policy_engine.cpp)
target_link_libraries(test_policy_engine PRIVATE vgre)
add_test(NAME CompliancePolicyEngine COMMAND test_policy_engine)
set_tests_properties(CompliancePolicyEngine PROPERTIES TIMEOUT 30)

# Secrets: sealed, versioned secret store (rotation + access policy).
add_executable(test_secret_store secrets/test_secret_store.cpp)
target_link_libraries(test_secret_store PRIVATE vgre)
add_test(NAME SecretsStore COMMAND test_secret_store)
set_tests_properties(SecretsStore PROPERTIES TIMEOUT 30)

# Backup / DR: content-addressed snapshot archive.
add_executable(test_backup_archive backup/test_backup_archive.cpp)
target_link_libraries(test_backup_archive PRIVATE vgre)
add_test(NAME BackupArchive COMMAND test_backup_archive)
set_tests_properties(BackupArchive PROPERTIES TIMEOUT 30)

# MIG: multi-instance GPU partitioning (multi-tenant isolation).
add_executable(test_mig mig/test_mig.cpp)
target_link_libraries(test_mig PRIVATE vgre)
add_test(NAME MigPartitioning COMMAND test_mig)
set_tests_properties(MigPartitioning PROPERTIES TIMEOUT 30)

# Fuzzing: deterministic smoke test over the JSON/JWT/PTX fuzz-target cores.
add_executable(test_fuzz_smoke fuzzing/test_fuzz_smoke.cpp)
target_link_libraries(test_fuzz_smoke PRIVATE vgre_fuzz_targets)
add_test(NAME FuzzSmoke COMMAND test_fuzz_smoke)
set_tests_properties(FuzzSmoke PROPERTIES TIMEOUT 60)

# PQC: FIPS-202 Keccak KATs + ML-KEM-768 (FIPS 203) + X25519 hybrid KEM.
add_executable(test_pqc pqc/test_pqc.cpp)
target_link_libraries(test_pqc PRIVATE vgre)
add_test(NAME PostQuantumKEM COMMAND test_pqc)
set_tests_properties(PostQuantumKEM PROPERTIES TIMEOUT 60)

# XLA HLO interpreter (§4.2): the compute engine JAX/TF/PyTorch-XLA lower into.
add_executable(test_hlo xla/test_hlo.cpp)
target_link_libraries(test_hlo PRIVATE vgre)
add_test(NAME XlaHloInterpreter COMMAND test_hlo)
set_tests_properties(XlaHloInterpreter PROPERTIES TIMEOUT 30)

# XLA executable C ABI: serialize → compile → execute (PJRT lifecycle substrate).
add_executable(test_xla_c_api xla/test_xla_c_api.cpp)
target_link_libraries(test_xla_c_api PRIVATE vgre)
add_test(NAME XlaExecutableCApi COMMAND test_xla_c_api)
set_tests_properties(XlaExecutableCApi PROPERTIES TIMEOUT 30)

# Work-stealing thread pool: parallel coverage, reentrancy, concurrent jobs.
add_executable(test_thread_pool xla/test_thread_pool.cpp)
target_link_libraries(test_thread_pool PRIVATE vgre Threads::Threads)
add_test(NAME XlaThreadPool COMMAND test_thread_pool)
set_tests_properties(XlaThreadPool PROPERTIES TIMEOUT 60)

# Real BLAS/tiled GEMM: correctness (all transpose combos + batched) + >=10x
# throughput vs the naive triple loop (milestone L2).
add_executable(test_blas_gemm xla/test_blas_gemm.cpp)
target_link_libraries(test_blas_gemm PRIVATE vgre Threads::Threads)
add_test(NAME XlaBlasGemm COMMAND test_blas_gemm)
set_tests_properties(XlaBlasGemm PROPERTIES TIMEOUT 120)

# Memory-mapped safetensors checkpoint loader (F32/F16/BF16/int dtypes) — L1.
add_executable(test_safetensors xla/test_safetensors.cpp)
target_link_libraries(test_safetensors PRIVATE vgre)
add_test(NAME XlaSafetensors COMMAND test_safetensors)
set_tests_properties(XlaSafetensors PROPERTIES TIMEOUT 30)

# bf16/f16 native Literal storage (the rest of L1) — half-width weights + engine
# decompression; matmul with bf16 params matches fp32.
add_executable(test_bf16_literal xla/test_bf16_literal.cpp)
target_link_libraries(test_bf16_literal PRIVATE vgre)
add_test(NAME XlaBf16Literal COMMAND test_bf16_literal)
set_tests_properties(XlaBf16Literal PROPERTIES TIMEOUT 30)

# Perf analytics: roofline model + flamegraph collapsed-stack.
add_executable(test_perf_analytics advanced/test_perf_analytics.cpp)
target_link_libraries(test_perf_analytics PRIVATE vgre)
add_test(NAME PerfAnalytics COMMAND test_perf_analytics)
set_tests_properties(PerfAnalytics PROPERTIES TIMEOUT 30)

# Capacity planner: Holt-Winters forecasting + FFD bin-packing.
add_executable(test_capacity_planner advanced/test_capacity_planner.cpp)
target_link_libraries(test_capacity_planner PRIVATE vgre)
add_test(NAME CapacityPlanner COMMAND test_capacity_planner)
set_tests_properties(CapacityPlanner PROPERTIES TIMEOUT 30)

# Fault tolerance: Byzantine-robust aggregation + elastic membership.
add_executable(test_fault_tolerance advanced/test_fault_tolerance.cpp)
target_link_libraries(test_fault_tolerance PRIVATE vgre)
add_test(NAME FaultTolerance COMMAND test_fault_tolerance)
set_tests_properties(FaultTolerance PROPERTIES TIMEOUT 30)

# API management: HMAC-signed webhook delivery + OpenAPI spec.
add_executable(test_api_management advanced/test_api_management.cpp)
target_link_libraries(test_api_management PRIVATE vgre)
add_test(NAME ApiManagement COMMAND test_api_management)
set_tests_properties(ApiManagement PROPERTIES TIMEOUT 30)

# Traffic management: load balancing + circuit breaker + rate limiting.
add_executable(test_traffic_manager advanced/test_traffic_manager.cpp)
target_link_libraries(test_traffic_manager PRIVATE vgre)
add_test(NAME TrafficManager COMMAND test_traffic_manager)
set_tests_properties(TrafficManager PROPERTIES TIMEOUT 30)

# Distributed: consistent hashing + distributed lock + leader election.
add_executable(test_distributed advanced/test_distributed.cpp)
target_link_libraries(test_distributed PRIVATE vgre)
add_test(NAME DistributedPrimitives COMMAND test_distributed)
set_tests_properties(DistributedPrimitives PROPERTIES TIMEOUT 30)

# AIOps + security analytics: anomaly detection + audit-log threat analysis.
add_executable(test_aiops advanced/test_aiops.cpp)
target_link_libraries(test_aiops PRIVATE vgre)
add_test(NAME AIOpsAnalytics COMMAND test_aiops)
set_tests_properties(AIOpsAnalytics PROPERTIES TIMEOUT 30)

# ETL pipeline: extract/validate/transform/load + data quality + lineage.
add_executable(test_etl advanced/test_etl.cpp)
target_link_libraries(test_etl PRIVATE vgre)
add_test(NAME ETLPipeline COMMAND test_etl)
set_tests_properties(ETLPipeline PROPERTIES TIMEOUT 30)

add_executable(test_nsight_export api/test_nsight_export.cpp)
target_link_libraries(test_nsight_export PRIVATE vgre)
if(SQLITE3_LIBRARY)
    target_link_libraries(test_nsight_export PRIVATE SQLite::SQLite3)
    add_test(NAME NsightExport COMMAND test_nsight_export)
    set_tests_properties(NsightExport PROPERTIES
        TIMEOUT 120
        FIXTURES_REQUIRED KMPRegistrationClean
        RESOURCE_LOCK vgre_llvm_jit)
endif()

add_executable(test_ptx_cross_module_link api/test_ptx_cross_module_link.cpp)
target_link_libraries(test_ptx_cross_module_link PRIVATE vgre_cudart)
add_test(NAME PTXCrossModuleLink COMMAND test_ptx_cross_module_link)

add_executable(test_llvm_bitcode_module api/test_llvm_bitcode_module.cpp)
target_link_libraries(test_llvm_bitcode_module PRIVATE vgre_cudart)
add_test(NAME LLVMBitcodeModuleDirect COMMAND test_llvm_bitcode_module)
set_tests_properties(LLVMBitcodeModuleDirect PROPERTIES TIMEOUT 120 RESOURCE_LOCK vgre_llvm_jit)

add_executable(test_ptx_fp8 api/test_ptx_fp8.cpp)
target_link_libraries(test_ptx_fp8 PRIVATE vgre_cudart)
add_test(NAME PTXFp8Translation COMMAND test_ptx_fp8)

add_executable(test_sass_detection api/test_sass_detection.cpp)
target_link_libraries(test_sass_detection PRIVATE vgre_cudart)
add_test(NAME SASSDetection COMMAND test_sass_detection)
set_tests_properties(SASSDetection PROPERTIES TIMEOUT 120 RESOURCE_LOCK vgre_llvm_jit)

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

# QUEUE-59: cuBLAS HEMM/HERK/HER2K batched variants
add_executable(test_cublas_hermitian_batched api/test_cublas_hermitian_batched.cpp)
target_link_libraries(test_cublas_hermitian_batched PRIVATE vgre_cudart)
add_test(NAME CuBLASHermitianBatched COMMAND test_cublas_hermitian_batched)

# QUEUE-61: cudaMemcpyPeerAsync cross-device async copy
add_executable(test_memcpy_peer_async api/test_memcpy_peer_async.cpp)
target_link_libraries(test_memcpy_peer_async PRIVATE vgre_cudart)
add_test(NAME MemcpyPeerAsync COMMAND test_memcpy_peer_async)

# QUEUE-63: cublasStrsmStridedBatched / cublasDtrsmStridedBatched
add_executable(test_cublas_trsm_strided api/test_cublas_trsm_strided_batched.cpp)
target_link_libraries(test_cublas_trsm_strided PRIVATE vgre_cudart)
add_test(NAME CuBLASTrsmStridedBatched COMMAND test_cublas_trsm_strided)

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
set_tests_properties(HardwareTokenManager PROPERTIES TIMEOUT 60)

add_executable(test_mps_quota advanced/test_mps_quota.cpp)
target_link_libraries(test_mps_quota PRIVATE vgre)
add_test(NAME MPSQuota COMMAND test_mps_quota)
set_tests_properties(MPSQuota PROPERTIES TIMEOUT 60)

# Portable software RDMA — one-sided WRITE/READ over TCP on every platform.
add_executable(test_software_rdma advanced/test_software_rdma.cpp)
target_link_libraries(test_software_rdma PRIVATE vgre)
add_test(NAME SoftwareRDMA COMMAND test_software_rdma)
set_tests_properties(SoftwareRDMA PROPERTIES TIMEOUT 60)

add_executable(test_adaptive_partition advanced/test_adaptive_partition.cpp)
target_link_libraries(test_adaptive_partition PRIVATE vgre)
add_test(NAME AdaptivePartition COMMAND test_adaptive_partition)
set_tests_properties(AdaptivePartition PROPERTIES TIMEOUT 60)

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

# QUEUE-28: JIT disk cache — SHA-256 key, atomic write, integrity footer, corruption detection.
# Standalone: no LLVM linkage, no vgre linkage — header-only jit_cache_utils.h only.
add_executable(test_jit_disk_cache unit/test_jit_disk_cache.cpp)
target_include_directories(test_jit_disk_cache PRIVATE
    ${CMAKE_SOURCE_DIR}/include)
add_test(NAME JITDiskCache COMMAND test_jit_disk_cache)
set_tests_properties(JITDiskCache PROPERTIES TIMEOUT 30)

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

add_executable(test_graph_dce core/test_graph_dce.cpp)
target_link_libraries(test_graph_dce PRIVATE vgre)
add_test(NAME GraphDCE COMMAND test_graph_dce)
set_tests_properties(GraphDCE PROPERTIES TIMEOUT 120)

add_executable(test_kv_cache core/test_kv_cache.cpp)
target_link_libraries(test_kv_cache PRIVATE vgre)
add_test(NAME KVCachePaged COMMAND test_kv_cache)
set_tests_properties(KVCachePaged PROPERTIES TIMEOUT 30)

# Header-only utility (gradient_checkpointing.h) — wires it into the build/tests.
add_executable(test_gradient_checkpointing core/test_gradient_checkpointing.cpp)
target_link_libraries(test_gradient_checkpointing PRIVATE vgre)
add_test(NAME GradientCheckpointing COMMAND test_gradient_checkpointing)
set_tests_properties(GradientCheckpointing PROPERTIES TIMEOUT 30)

# Phase 3, P3-2: weight-only INT4 (AWQ/GPTQ, W4A16).
add_executable(test_int4_weight_quant core/test_int4_weight_quant.cpp)
target_link_libraries(test_int4_weight_quant PRIVATE vgre)
add_test(NAME Int4WeightQuant COMMAND test_int4_weight_quant)
set_tests_properties(Int4WeightQuant PROPERTIES TIMEOUT 30)

# Phase 3, P3-9: MoE top-k router + grouped expert GEMM.
add_executable(test_moe core/test_moe.cpp)
target_link_libraries(test_moe PRIVATE vgre)
add_test(NAME MoEGroupedGemm COMMAND test_moe)
set_tests_properties(MoEGroupedGemm PROPERTIES TIMEOUT 30)

# Phase 3, P3-12: Mamba/SSM selective scan.
add_executable(test_ssm_scan core/test_ssm_scan.cpp)
target_link_libraries(test_ssm_scan PRIVATE vgre)
add_test(NAME SSMSelectiveScan COMMAND test_ssm_scan)
set_tests_properties(SSMSelectiveScan PROPERTIES TIMEOUT 30)

# Phase 3, P3-11: structured attention masks (sliding-window/sink/ALiBi).
add_executable(test_attention_masks core/test_attention_masks.cpp)
target_link_libraries(test_attention_masks PRIVATE vgre)
add_test(NAME StructuredAttentionMasks COMMAND test_attention_masks)
set_tests_properties(StructuredAttentionMasks PROPERTIES TIMEOUT 30)

# Phase 3, P3-3: OCP Microscaling (MX) format family.
add_executable(test_mx_quant core/test_mx_quant.cpp)
target_link_libraries(test_mx_quant PRIVATE vgre)
add_test(NAME MXFormats COMMAND test_mx_quant)
set_tests_properties(MXFormats PROPERTIES TIMEOUT 30)

# Phase 3, P3-17: tensor/pipeline parallel primitives (Megatron-LM).
add_executable(test_tensor_parallel core/test_tensor_parallel.cpp)
target_link_libraries(test_tensor_parallel PRIVATE vgre)
add_test(NAME TensorParallel COMMAND test_tensor_parallel)
set_tests_properties(TensorParallel PROPERTIES TIMEOUT 30)

# Phase 3, P3-8: speculative decoding (draft + verify).
add_executable(test_speculative core/test_speculative.cpp)
target_link_libraries(test_speculative PRIVATE vgre)
add_test(NAME SpeculativeDecoding COMMAND test_speculative)
set_tests_properties(SpeculativeDecoding PROPERTIES TIMEOUT 60)

# Phase 3, P3-15: interconnect topology + collective cost model.
add_executable(test_interconnect core/test_interconnect.cpp)
target_link_libraries(test_interconnect PRIVATE vgre)
add_test(NAME InterconnectCostModel COMMAND test_interconnect)
set_tests_properties(InterconnectCostModel PROPERTIES TIMEOUT 30)

# Phase 3, P3-21: SM occupancy calculator + roofline.
add_executable(test_occupancy core/test_occupancy.cpp)
target_link_libraries(test_occupancy PRIVATE vgre)
add_test(NAME OccupancyRoofline COMMAND test_occupancy)
set_tests_properties(OccupancyRoofline PROPERTIES TIMEOUT 30)

# Phase 3, P3-16: NVSHMEM symmetric memory + one-sided collectives.
add_executable(test_nvshmem core/test_nvshmem.cpp)
target_link_libraries(test_nvshmem PRIVATE vgre)
add_test(NAME NvshmemSymmetricMem COMMAND test_nvshmem)
set_tests_properties(NvshmemSymmetricMem PROPERTIES TIMEOUT 30)

# Phase 3, P3-19: differential / property-based testing harness.
add_executable(test_differential api/test_differential.cpp)
target_link_libraries(test_differential PRIVATE vgre)
add_test(NAME DifferentialFuzz COMMAND test_differential)
set_tests_properties(DifferentialFuzz PROPERTIES TIMEOUT 60)

# Phase 3, P3-18: bit-deterministic mode.
add_executable(test_deterministic core/test_deterministic.cpp)
target_link_libraries(test_deterministic PRIVATE vgre)
add_test(NAME DeterministicMode COMMAND test_deterministic)
set_tests_properties(DeterministicMode PROPERTIES TIMEOUT 30)

# Phase 3, P3-10: prefix caching (content-addressed KV-block sharing).
add_executable(test_prefix_cache core/test_prefix_cache.cpp)
target_link_libraries(test_prefix_cache PRIVATE vgre)
add_test(NAME PrefixCaching COMMAND test_prefix_cache)
set_tests_properties(PrefixCaching PROPERTIES TIMEOUT 30)

# Phase 3, P3-5: Tensor Memory Accelerator (TMA).
add_executable(test_tma core/test_tma.cpp)
target_link_libraries(test_tma PRIVATE vgre)
add_test(NAME TensorMemoryAccelerator COMMAND test_tma)
set_tests_properties(TensorMemoryAccelerator PROPERTIES TIMEOUT 30)

# Phase 3, P3-6: Thread-block clusters + Distributed Shared Memory (DSMEM).
add_executable(test_cluster core/test_cluster.cpp)
target_link_libraries(test_cluster PRIVATE vgre Threads::Threads)
add_test(NAME ThreadBlockCluster COMMAND test_cluster)
set_tests_properties(ThreadBlockCluster PROPERTIES TIMEOUT 30)

# Phase 3, P3-6: end-to-end cluster execution via the executor (executeClustered).
add_executable(test_cluster_exec integration/test_cluster_exec.cpp)
target_link_libraries(test_cluster_exec PRIVATE vgre Threads::Threads)
add_test(NAME ClusterExecution COMMAND test_cluster_exec)
set_tests_properties(ClusterExecution PROPERTIES TIMEOUT 30)

# Phase 3, P3-7: Blackwell tcgen05 Tensor Memory (TMEM) data path.
add_executable(test_tmem core/test_tmem.cpp)
target_link_libraries(test_tmem PRIVATE vgre)
add_test(NAME TensorMemoryTcgen05 COMMAND test_tmem)
set_tests_properties(TensorMemoryTcgen05 PROPERTIES TIMEOUT 30)

# Phase 3, P3-20: UVM oversubscription via disk-backed LRU eviction.
add_executable(test_page_evictor core/test_page_evictor.cpp)
target_link_libraries(test_page_evictor PRIVATE vgre)
add_test(NAME UVMPageEvictor COMMAND test_page_evictor)
set_tests_properties(UVMPageEvictor PROPERTIES TIMEOUT 30)

# Phase 3, P3-20: end-to-end oversubscription through the live MemoryManager.
add_executable(test_uvm_oversubscription integration/test_uvm_oversubscription.cpp)
target_link_libraries(test_uvm_oversubscription PRIVATE vgre)
add_test(NAME UVMOversubscription COMMAND test_uvm_oversubscription)
set_tests_properties(UVMOversubscription PROPERTIES TIMEOUT 60)

# Header-only test of the device cuRAND skip-ahead (no JIT/link needed).
add_executable(test_curand_skipahead unit/test_curand_skipahead.cpp)
target_include_directories(test_curand_skipahead PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME CurandSkipAhead COMMAND test_curand_skipahead)
set_tests_properties(CurandSkipAhead PROPERTIES TIMEOUT 30)

# Header-only FP8/FP4 quantized GEMM test.
add_executable(test_fp_quant unit/test_fp_quant.cpp)
target_include_directories(test_fp_quant PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME FPQuantGemm COMMAND test_fp_quant)
set_tests_properties(FPQuantGemm PROPERTIES TIMEOUT 30)

add_executable(test_golden_vectors api/test_golden_vectors.cpp)
target_link_libraries(test_golden_vectors PRIVATE vgre_cudart)
add_test(NAME GoldenVectors COMMAND test_golden_vectors)
set_tests_properties(GoldenVectors PROPERTIES TIMEOUT 60)

# The iGPU transpiler compile-check needs the OpenCL backend: the executor
# header includes <CL/cl.h> and transpileToOpenCL lives in the OpenCL-gated .cpp.
# Only build it when the backend is enabled (otherwise the OpenCL headers are
# absent, e.g. on CI runners without an OpenCL SDK).
if(VGRE_USE_OPENCL_BACKEND)
    add_executable(test_igpu_transpile integration/test_igpu_transpile.cpp)
    target_link_libraries(test_igpu_transpile PRIVATE vgre)
    add_test(NAME IGPUTranspileCompile COMMAND test_igpu_transpile)
    set_tests_properties(IGPUTranspileCompile PROPERTIES TIMEOUT 60)
endif()

# PyTorch end-to-end validation harness (Track 24): runs VGRE's cuBLAS via
# ctypes and compares to torch. Prefers the project .venv (which has torch); the
# script self-SKIPs if torch/numpy are unavailable.
if(EXISTS "${CMAKE_SOURCE_DIR}/.venv/bin/python")
    set(_vgre_py "${CMAKE_SOURCE_DIR}/.venv/bin/python")
else()
    find_program(_vgre_py NAMES python3 python)
endif()
if(_vgre_py)
    add_test(NAME PyTorchValidation
             COMMAND ${_vgre_py} ${CMAKE_CURRENT_SOURCE_DIR}/integration/pytorch_validation.py)
    set_tests_properties(PyTorchValidation PROPERTIES TIMEOUT 120)
endif()

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

# QUEUE-29: B-tree region lookup — O(log n) std::map<uintptr_t, ManagedRegion>
# Standalone: no vgre linkage — pure std::map instrumented comparison test.
add_executable(test_btree_region_lookup unit/test_btree_region_lookup.cpp)
add_test(NAME BTreeRegionLookup COMMAND test_btree_region_lookup)
set_tests_properties(BTreeRegionLookup PROPERTIES TIMEOUT 30)

# QUEUE-34: Robin Hood open-addressing hash table — 10k-entry stress test.
# Standalone: header-only, no vgre linkage needed.
add_executable(test_robin_hood_hash_table unit/test_robin_hood_hash_table.cpp)
target_include_directories(test_robin_hood_hash_table PRIVATE
    ${CMAKE_SOURCE_DIR}/include)
add_test(NAME RobinHoodHashTable COMMAND test_robin_hood_hash_table)
set_tests_properties(RobinHoodHashTable PROPERTIES TIMEOUT 30)

# QUEUE-50: SipHash-2-4 — known test vector + collision resistance in RobinHoodHashTable.
add_executable(test_siphash unit/test_siphash.cpp)
target_link_libraries(test_siphash PRIVATE vgre_common)
target_include_directories(test_siphash PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME SipHash24 COMMAND test_siphash)
set_tests_properties(SipHash24 PROPERTIES TIMEOUT 30)

# QUEUE-51: Fibonacci Max-Heap — O(1) insert, O(log n) pop, O(1) increase_key.
add_executable(test_fib_heap unit/test_fib_heap.cpp)
target_link_libraries(test_fib_heap PRIVATE vgre)
target_include_directories(test_fib_heap PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME FibHeap COMMAND test_fib_heap)
set_tests_properties(FibHeap PROPERTIES TIMEOUT 30)

# QUEUE-58: Treiber lock-free stack — LIFO, empty-pop, 8-thread concurrency.
add_executable(test_treiber_stack unit/test_treiber_stack.cpp)
target_include_directories(test_treiber_stack PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME TreiberStack COMMAND test_treiber_stack)
set_tests_properties(TreiberStack PROPERTIES TIMEOUT 60)

# ── QUEUE-47/52: TRMM strided-batched + TRSV batched ─────────────────────────
add_executable(test_trmm_trsv_batched unit/test_trmm_trsv_batched.cpp)
target_link_libraries(test_trmm_trsv_batched PRIVATE vgre)
add_test(NAME TrmmTrsvBatched COMMAND test_trmm_trsv_batched)
set_tests_properties(TrmmTrsvBatched PROPERTIES
    ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}" TIMEOUT 30)

# ── QUEUE-49: Event creation flags + elapsed time ────────────────────────────
add_executable(test_event_flags unit/test_event_flags.cpp)
target_link_libraries(test_event_flags PRIVATE vgre)
add_test(NAME EventFlags COMMAND test_event_flags)
set_tests_properties(EventFlags PROPERTIES
    ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}" TIMEOUT 30)

# ── QUEUE-57: cuSOLVER sygvd generalized eigenvalue ──────────────────────────
add_executable(test_cusolver_sygvd api/test_cusolver_sygvd.cpp)
target_link_libraries(test_cusolver_sygvd PRIVATE vgre_cudart)
add_test(NAME CusolverSygvd COMMAND test_cusolver_sygvd)
set_tests_properties(CusolverSygvd PROPERTIES
    ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}" TIMEOUT 30)

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
        SKIP_RETURN_CODE 77
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
            SKIP_RETURN_CODE 77
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

# HardwareTokenManager validates token encrypt/decrypt/store *correctness*, not
# KDF strength, so run it with the minimum PBKDF2 iteration count instead of the
# deliberately-expensive 600k production default (the test was the suite's single
# biggest time sink). Re-set ENVIRONMENT here, AFTER the global setter above
# (which replaces — not appends — the property), preserving the global vars.
# Production code is unaffected; only this test process gets the override.
if("HardwareTokenManager" IN_LIST vgre_full_test_list)
    set_tests_properties(HardwareTokenManager PROPERTIES
        ENVIRONMENT "VGRE_BENCHMARK=OFF;VGRE_IPC_MODE=OFF;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR};VGRE_PBKDF2_ITERATIONS=10000")
endif()

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

# ── Heaviest JIT + multi-device tests (Track 6) ────────────────────────────
# A *small* curated set that each fork clang AND spin multiple device worker
# pools (multi-device tests double the thread count). Even under `ctest -j2
# --test-load`, two of these running together can oversubscribe a small runner
# and time one out. Serialize ONLY these few with a shared lock (measured to
# remove the residual flake) — not a blanket lock, which previously backfired.
foreach(_jh
        MultiDeviceCooperativeComprehensive MultiDeviceIntegration
        ClangEnhanced SASSDetection DeviceCurandIntegration)
    if("${_jh}" IN_LIST vgre_full_test_list)
        set_tests_properties(${_jh} PROPERTIES RESOURCE_LOCK vgre_jit_heavy)
    endif()
endforeach()

# ── Python integration tests ─────────────────────────────────────────────────
# Prefer the project .venv (has numpy/torch/tf); fall back to a system python.
if(EXISTS "${CMAKE_SOURCE_DIR}/.venv/bin/python")
    set(_vgre_intpy "${CMAKE_SOURCE_DIR}/.venv/bin/python")
else()
    find_program(_vgre_intpy NAMES python3 python)
endif()
if(_vgre_intpy)
    add_test(NAME PyTorchIntegration
             COMMAND ${_vgre_intpy} ${CMAKE_SOURCE_DIR}/tests/python/test_pytorch.py
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    set_tests_properties(PyTorchIntegration PROPERTIES
                         ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}"
                         TIMEOUT 120
                         LABELS "python"
                         SKIP_REGULAR_EXPRESSION "ModuleNotFoundError|ImportError|No module named|SKIP:")
    add_test(NAME TensorFlowIntegration
             COMMAND ${_vgre_intpy} ${CMAKE_SOURCE_DIR}/tests/python/test_tensorflow.py
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    set_tests_properties(TensorFlowIntegration PROPERTIES
                         ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}"
                         TIMEOUT 120
                         LABELS "python"
                         SKIP_REGULAR_EXPRESSION "ModuleNotFoundError|ImportError|No module named|SKIP:")
endif()
