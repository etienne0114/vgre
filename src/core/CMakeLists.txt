add_library(vgre_core OBJECT
    virtual_gpu_device.cpp
    memory/memory_manager.cpp
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
)

target_include_directories(vgre_core PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(vgre_core PRIVATE
    Threads::Threads
    vgre_common
)
