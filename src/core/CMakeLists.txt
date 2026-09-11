add_library(vgre_core OBJECT
    virtual_gpu_device.cpp
    memory/memory_manager.cpp
    memory/bounds_checker.cpp
    memory/memory_manager_managed.cpp
    memory/memory_manager_copy.cpp
    memory/pool_allocator.cpp
    memory/uvm_migration.cpp
    memory/bandwidth_model.cpp
    scheduler.cpp
    scheduler_numa.cpp
    scheduler_worker.cpp
    scheduler_tasks.cpp
    scheduler_control.cpp
    event.cpp
    runtime_engine.cpp
    runtime_engine_launch.cpp
    runtime_engine_backend.cpp  # no-LLVM backend routing (#ifndef VGRE_ENABLE_JIT)
    kv_cache.cpp
    runtime_engine_graph.cpp
    runtime_engine_graph_extended.cpp
    graph/graph_manager.cpp
    graph/graph_manager_nodes.cpp
    graph/graph_manager_serde.cpp
    graph_optimizer.cpp
    graph/graph_manager_extended_nodes.cpp
    graph/graph_manager_introspection.cpp
    graph/graph_manager_exec_update.cpp
    graph/graph_manager_dependencies.cpp
    texture/texture_manager_lifecycle.cpp
    texture/texture_manager_sampling.cpp
    texture/texture_manager_surface.cpp
    texture/texture_manager_array.cpp
    texture/texture_manager_mipmap.cpp
    shm_manager.cpp
    math/cache_oblivious.cpp
    math/mixed_precision.cpp
    math/block_sparse.cpp
    math/tensor_core_emulation.cpp
    math/scan.cpp
    math/radix_sort.cpp
    graph_dependency_tracker.cpp
)

target_include_directories(vgre_core PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(vgre_core PRIVATE
    Threads::Threads
    vgre_common
)

# clang-cl only: its <immintrin.h> gates <avxintrin.h> (which declares __m256i)
# behind __AVX__ when _MSC_VER is defined, so without /arch:AVX(2) the intrinsic
# TYPES are never declared and __attribute__((target("avx"))) in memory/*.cpp and
# math/*.cpp cannot recover them. GCC/Clang on Linux/macOS include the header
# unconditionally, so they must NOT get these flags: -mfma would let the compiler
# contract a*b+c into an FMA in the scalar reference paths, changing float
# rounding and breaking exact-comparison tests. So this is MSVC-scoped only.
if(MSVC)
    if(VGRE_SIMD_COMPILE_OPTIONS)
        target_compile_options(vgre_core PRIVATE ${VGRE_SIMD_COMPILE_OPTIONS})
    endif()
    if(VGRE_SIMD_COMPILE_DEFS)
        target_compile_definitions(vgre_core PRIVATE ${VGRE_SIMD_COMPILE_DEFS})
    endif()
endif()
