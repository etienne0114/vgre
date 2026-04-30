add_library(vgre_api OBJECT
    cuda_interceptor.cpp
    cuda_interceptor_memory.cpp
    cuda_interceptor_graphs.cpp
    opencl_adapter.cpp
    vgre_c_api.cpp
    vgre_c_api_graphs.cpp
    vgre_c_api_telemetry.cpp
)

target_include_directories(vgre_api PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)
