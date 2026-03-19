add_library(vgre_runtime OBJECT
    cpu_parallel_executor.cpp
    vector_engine.cpp
    gpu_thread_context.cpp
    block_worker_pool.cpp
)

if(VGRE_USE_OPENCL_BACKEND)
    target_sources(vgre_runtime PRIVATE igpu_opencl_executor.cpp)
endif()

target_include_directories(vgre_runtime PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(vgre_runtime PUBLIC
    OpenMP::OpenMP_CXX
)
