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
# -mavxvnni    enables _mm256_dpbusd_avx_epi32 (Alder Lake AVX-VNNI)
# -mamx-tile   enables _tile_loadconfig, _tile_loadd, _tile_stored
# -mamx-bf16   enables _tile_dpbf16ps (BF16 tile multiply-accumulate)
# These flags are localised to vector_engine_amx.cpp so other TUs are unaffected.
set_source_files_properties(vector_engine_amx.cpp PROPERTIES
    COMPILE_FLAGS "-mavxvnni -mamx-tile -mamx-bf16"
)

if(VGRE_USE_OPENCL_BACKEND)
    target_sources(vgre_runtime PRIVATE igpu_opencl_executor.cpp)
endif()

target_include_directories(vgre_runtime PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(vgre_runtime PRIVATE
)
