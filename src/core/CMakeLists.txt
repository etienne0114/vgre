add_library(vgre_core OBJECT
    virtual_gpu_device.cpp
    memory_manager.cpp
    scheduler.cpp
    event.cpp
    runtime_engine.cpp
    graph_manager.cpp
    graph_optimizer.cpp
    texture_manager.cpp
)

target_include_directories(vgre_core PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(vgre_core PUBLIC
    OpenMP::OpenMP_CXX
    Threads::Threads
)
