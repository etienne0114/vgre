add_library(vgre_api OBJECT
    cuda_interceptor.cpp
    cuda_interceptor_device.cpp
    cuda_interceptor_memory.cpp
    cuda_interceptor_graphs.cpp
    cuda_ipc_memory.cpp
    cuda_virtual_memory.cpp
    cuda_external_semaphore.cpp
    cublas/cublas_core.cpp
    cublas/cublas_level1.cpp
    cublas/cublas_level2.cpp
    cublas/cublas_level2_packed.cpp
    cublas/cublas_level3.cpp
    cublas/cublas_level3_blas3.cpp
    cublas/cublas_gemm_ex.cpp
    cublas/cublas_complex.cpp
    cublas/cublas_complex_level2.cpp
    cublas/cublas_complex_level3.cpp
    cublas/cublas_hermitian.cpp
    cudnn/cudnn_core.cpp
    cudnn/cudnn_convolution.cpp
    cudnn/cudnn_activation.cpp
    cudnn/cudnn_softmax.cpp
    cudnn/cudnn_pooling.cpp
    cudnn/cudnn_batchnorm.cpp
    cudnn/cudnn_normalization.cpp
    cudnn/cudnn_dropout.cpp
    cudnn/cudnn_tensor_ops.cpp
    cudnn/cudnn_rnn.cpp
    cudnn/cudnn_attention.cpp
    cudnn/cudnn_ctc_loss.cpp
    cudnn/cudnn_backend_api.cpp
    cudnn/cudnn_lrn.cpp
    cudnn/cudnn_divisive_norm.cpp
    nccl/nccl_core.cpp
    nccl/nccl_communicator.cpp
    nccl/nccl_collectives.cpp
    nccl/nccl_p2p.cpp
    nvtx_shim.cpp
    opencl_adapter.cpp
    vgre_c_api.cpp
    vgre_c_api_graphs.cpp
    vgre_c_api_telemetry.cpp
    cupti/cupti_shim.cpp
)

# New CUDART shim files — compiled as a separate OBJECT library so they can
# be linked into vgre_cudart.so only (they reference CUDAModuleRegistry which
# lives in cudart_shim.cpp, part of vgre_cudart, not libvgre.so).
add_library(vgre_cudart_shims OBJECT
    cudart/cudart_shim_array_memcpy.cpp
    cudart/cudart_shim_memset_nd.cpp
    cudart/cudart_shim_graph_nodes.cpp
    cudart/cudart_shim_graph_introspection.cpp
    cudart/cudart_shim_graph_exec.cpp
    cudart/cudart_shim_graph_user_objects.cpp
    cudart/cudart_shim_capture.cpp
    cudart/cudart_shim_texture_objects.cpp
    cudart/cudart_shim_device_attrs.cpp
    cudart/cudart_shim_external_memory.cpp
    cudart/cudart_memory_pool.cpp
    cudart/cudart_cooperative.cpp
    cudart/cudart_mipmap.cpp
    cudart/cudart_shim_graph_basic.cpp
)
target_include_directories(vgre_cudart_shims PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_include_directories(vgre_api PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)
