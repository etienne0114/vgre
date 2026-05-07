add_library(vgre_api OBJECT
    cuda_interceptor.cpp
    cuda_interceptor_memory.cpp
    cuda_interceptor_graphs.cpp
    cuda_ipc_memory.cpp
    cuda_virtual_memory.cpp
    cuda_external_semaphore.cpp
    cublas_shim.cpp
    cudnn_shim.cpp
    nccl_shim.cpp
    nvtx_shim.cpp
    opencl_adapter.cpp
    vgre_c_api.cpp
    vgre_c_api_graphs.cpp
    vgre_c_api_telemetry.cpp
)

target_include_directories(vgre_api PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)
