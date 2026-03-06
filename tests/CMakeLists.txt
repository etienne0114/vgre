# Test CMakeLists.txt
# Unit tests and integration tests for VGRE

# ── Unit tests ──────────────────────────────────────────────────────────────
add_executable(test_virtual_gpu_device unit/test_virtual_gpu_device.cpp)
target_link_libraries(test_virtual_gpu_device PRIVATE vgre)
add_test(NAME VirtualGPUDevice COMMAND test_virtual_gpu_device)

add_executable(test_memory_manager unit/test_memory_manager.cpp)
target_link_libraries(test_memory_manager PRIVATE vgre)
add_test(NAME MemoryManager COMMAND test_memory_manager)

add_executable(test_scheduler unit/test_scheduler.cpp)
target_link_libraries(test_scheduler PRIVATE vgre)
add_test(NAME Scheduler COMMAND test_scheduler)

add_executable(test_vector_engine unit/test_vector_engine.cpp)
target_link_libraries(test_vector_engine PRIVATE vgre)
add_test(NAME VectorEngine COMMAND test_vector_engine)

# ── Integration tests ──────────────────────────────────────────────────────
add_executable(test_vector_addition integration/test_vector_addition.cpp)
target_link_libraries(test_vector_addition PRIVATE vgre)
add_test(NAME VectorAdditionIntegration COMMAND test_vector_addition)

add_executable(test_uvm_managed integration/test_uvm_managed.cpp)
target_link_libraries(test_uvm_managed PRIVATE vgre)
add_test(NAME UVMManagedIntegration COMMAND test_uvm_managed)

add_executable(test_cuda_graphs integration/test_cuda_graphs.cpp)
target_link_libraries(test_cuda_graphs PRIVATE vgre)
add_test(NAME CUDAGraphsIntegration COMMAND test_cuda_graphs)

add_executable(test_multi_device integration/test_multi_device.cpp)
target_link_libraries(test_multi_device PRIVATE vgre)
add_test(NAME MultiDeviceIntegration COMMAND test_multi_device)

add_executable(test_stream_concurrency integration/test_stream_concurrency.cpp)
target_link_libraries(test_stream_concurrency PRIVATE vgre)
add_test(NAME test_stream_concurrency COMMAND test_stream_concurrency)
