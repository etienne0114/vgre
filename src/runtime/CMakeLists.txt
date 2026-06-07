add_library(vgre_runtime OBJECT
    cpu_parallel_executor.cpp
    cdp_executor.cpp
    vector_engine.cpp
    vector_engine_float.cpp
    vector_engine_double.cpp
    vector_engine_amx.cpp
    gpu_thread_context.cpp
    block_worker_pool.cpp
)

# AVX-VNNI + AMX compile flags for the AMX/VNNI translation unit.
# Only applied on x86/x86_64 hosts where the compiler supports these flags.
# On ARM/Apple Silicon the flags don't exist and ENABLE_VGRE_AMX is not set,
# so vector_engine_amx.cpp compiles to an empty translation unit.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i686|i386")
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mavxvnni" _cc_avxvnni)
    check_cxx_compiler_flag("-mamx-tile" _cc_amxtile_rt)
    check_cxx_compiler_flag("-mamx-bf16" _cc_amxbf16_rt)
    if(_cc_avxvnni AND _cc_amxtile_rt AND _cc_amxbf16_rt)
        set_source_files_properties(vector_engine_amx.cpp PROPERTIES
            COMPILE_FLAGS "-mavxvnni -mamx-tile -mamx-bf16"
        )
    endif()
endif()

if(VGRE_USE_OPENCL_BACKEND)
    target_sources(vgre_runtime PRIVATE igpu_opencl_executor.cpp)
endif()

target_include_directories(vgre_runtime PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(vgre_runtime PRIVATE
)
