add_library(vgre_api OBJECT
    cuda_interceptor.cpp
    opencl_adapter.cpp
    vgre_c_api.cpp
)

target_include_directories(vgre_api PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)
