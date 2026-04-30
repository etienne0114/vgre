add_library(vgre_core OBJECT
    virtual_gpu_device.cpp
    memory/memory_manager.cpp
    memory/memory_manager_managed.cpp
    memory/memory_manager_copy.cpp
    memory/pool_allocator.cpp
    memory/uvm_migration.cpp
    memory/bandwidth_model.cpp
    scheduler.cpp
    event.cpp
    runtime_engine.cpp
    runtime_engine_launch.cpp
    runtime_engine_graph.cpp
    graph_manager.cpp
    graph_optimizer.cpp
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
