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
    graph_optimizer.cpp
    graph/graph_manager_extended_nodes.cpp
    graph/graph_manager_introspection.cpp
    graph/graph_manager_exec_update.cpp
    graph/graph_manager_dependencies.cpp
    texture_manager.cpp
    shm_manager.cpp
)

target_include_directories(vgre_core PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(vgre_core PUBLIC
    OpenMP::OpenMP_CXX
    Threads::Threads
)
