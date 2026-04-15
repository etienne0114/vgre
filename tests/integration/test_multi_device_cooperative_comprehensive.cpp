#include "vgre/core/runtime_engine.h"
#include "vgre/common/logger.h"
#include <thread>
#include <chrono>
#include <atomic>
#include <vector>
#include <memory>
#include <iostream>
#include <cassert>
#include <cstring>

using namespace vgre;
using namespace vgre::core;

// Simple test framework
#define ASSERT_EQ(a, b) do { \
    if ((a) != (b)) { \
        std::cerr << "ASSERTION FAILED: " << #a << " != " << #b << " at line " << __LINE__ << std::endl; \
        std::cerr << "  Expected: " << static_cast<int>(b) << ", Got: " << static_cast<int>(a) << std::endl; \
        exit(1); \
    } \
} while(0)

#define EXPECT_EQ(a, b) do { \
    if ((a) != (b)) { \
        std::cerr << "EXPECTATION FAILED: " << #a << " != " << #b << " at line " << __LINE__ << std::endl; \
        std::cerr << "  Expected: " << static_cast<int>(b) << ", Got: " << static_cast<int>(a) << std::endl; \
        test_failed = true; \
    } \
} while(0)

#define EXPECT_NE(a, b) do { \
    if ((a) == (b)) { \
        std::cerr << "EXPECTATION FAILED: " << #a << " == " << #b << " at line " << __LINE__ << std::endl; \
        test_failed = true; \
    } \
} while(0)

#define EXPECT_LT(a, b) do { \
    if ((a) >= (b)) { \
        std::cerr << "EXPECTATION FAILED: " << #a << " >= " << #b << " at line " << __LINE__ << std::endl; \
        test_failed = true; \
    } \
} while(0)

bool test_failed = false;

void setUp() {
    // Initialize runtime engine
    auto result = RuntimeEngine::instance().initialize();
    ASSERT_EQ(result, VGREResult::SUCCESS);
}

void tearDown() {
    RuntimeEngine::instance().shutdown();
}

// Test kernel that uses grid-wide barriers
const char* COOPERATIVE_KERNEL_SOURCE = R"(
extern "C" __global__ void cooperative_test_kernel(int* data, int* barrier_count, int num_elements) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Phase 1: Each thread writes its ID
    if (tid < num_elements) {
        data[tid] = tid;
    }
    
    // Grid-wide barrier - all blocks must reach this point
    cooperative_groups::this_grid().sync();
    
    // Phase 2: Increment barrier count atomically (only after all threads have written)
    if (tid == 0) {
        atomicAdd(barrier_count, 1);
    }
    
    // Another grid-wide barrier
    cooperative_groups::this_grid().sync();
    
    // Phase 3: Each thread adds 1000 to its value
    if (tid < num_elements) {
        data[tid] += 1000;
    }
}
)";

// Test kernel for multi-device coordination
const char* MULTI_DEVICE_KERNEL_SOURCE = R"(
extern "C" __global__ void multi_device_kernel(int* device_data, int device_id, int* sync_counter) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each device writes its device ID
    if (tid == 0) {
        device_data[0] = device_id;
        atomicAdd(sync_counter, 1);
    }
    
    // Grid-wide barrier within this device
    cooperative_groups::this_grid().sync();
    
    // All threads on this device increment the counter
    if (tid < 32) {  // First warp only
        atomicAdd(sync_counter, 1);
    }
}
)";

void testSingleDeviceCooperativeBarrier() {
    std::cout << "Running testSingleDeviceCooperativeBarrier..." << std::endl;
    
    int NUM_ELEMENTS = 1024;
    const int NUM_BLOCKS = 8;
    const int BLOCK_SIZE = 128;
    
    // Allocate memory
    MemoryHandle data_handle;
    MemoryHandle barrier_count_handle;
    
    auto& engine = RuntimeEngine::instance();
    auto result = engine.malloc(NUM_ELEMENTS * sizeof(int), data_handle);
    ASSERT_EQ(result, VGREResult::SUCCESS);
    
    result = engine.malloc(sizeof(int), barrier_count_handle);
    ASSERT_EQ(result, VGREResult::SUCCESS);
    
    // Get pointers from handles
    int* data = reinterpret_cast<int*>(data_handle);
    int* barrier_count = reinterpret_cast<int*>(barrier_count_handle);
    
    // Initialize data
    memset(data, 0, NUM_ELEMENTS * sizeof(int));
    *barrier_count = 0;
    
    // Register kernel
    KernelId kernelId;
    result = engine.registerKernel("cooperative_test_kernel", COOPERATIVE_KERNEL_SOURCE, kernelId);
    ASSERT_EQ(result, VGREResult::SUCCESS);
    
    // Prepare arguments
    void* args[] = {&data, &barrier_count, &NUM_ELEMENTS};
    
    // Launch cooperative kernel
    dim3 gridDim(NUM_BLOCKS, 1, 1);
    dim3 blockDim(BLOCK_SIZE, 1, 1);
    
    result = engine.launchCooperativeKernel(
        kernelId, gridDim, blockDim, args, 0, 0
    );
    ASSERT_EQ(result, VGREResult::SUCCESS);
    
    // Verify results
    // All threads should have written their IDs + 1000
    for (int i = 0; i < NUM_ELEMENTS; i++) {
        EXPECT_EQ(data[i], i + 1000);
    }
    
    // Barrier count should be exactly NUM_BLOCKS (one per block)
    EXPECT_EQ(*barrier_count, NUM_BLOCKS);
    
    std::cout << "testSingleDeviceCooperativeBarrier: " << (test_failed ? "FAILED" : "PASSED") << std::endl;
}

void testMultiDeviceCooperativeLaunch() {
    std::cout << "Running testMultiDeviceCooperativeLaunch..." << std::endl;
    
    const int NUM_DEVICES = 4;
    const int NUM_BLOCKS_PER_DEVICE = 4;
    const int BLOCK_SIZE = 32;
    
    // Allocate shared memory for all devices
    std::vector<MemoryHandle> device_data_handles(NUM_DEVICES);
    MemoryHandle sync_counter_handle;
    
    auto& engine = RuntimeEngine::instance();
    
    // Allocate memory for each device
    for (int i = 0; i < NUM_DEVICES; i++) {
        auto result = engine.malloc(sizeof(int), device_data_handles[i]);
        ASSERT_EQ(result, VGREResult::SUCCESS);
    }
    
    // Allocate sync counter
    auto result = engine.malloc(sizeof(int), sync_counter_handle);
    ASSERT_EQ(result, VGREResult::SUCCESS);
    
    // Get pointers from handles
    std::vector<int*> device_data(NUM_DEVICES);
    for (int i = 0; i < NUM_DEVICES; i++) {
        device_data[i] = reinterpret_cast<int*>(device_data_handles[i]);
        *device_data[i] = -1;  // Initialize to invalid value
    }
    int* sync_counter = reinterpret_cast<int*>(sync_counter_handle);
    *sync_counter = 0;
    
    // Register kernel
    KernelId kernelId;
    result = engine.registerKernel("multi_device_kernel", MULTI_DEVICE_KERNEL_SOURCE, kernelId);
    ASSERT_EQ(result, VGREResult::SUCCESS);
    
    // Prepare launch parameters for each device
    std::vector<RuntimeEngine::CoopMultiLaunchParams> launchParams;
    std::vector<std::vector<void*>> argsList(NUM_DEVICES);
    
    for (int i = 0; i < NUM_DEVICES; i++) {
        argsList[i] = {&device_data[i], &i, &sync_counter};
        
        RuntimeEngine::CoopMultiLaunchParams params;
        params.name = "multi_device_kernel";
        params.source = MULTI_DEVICE_KERNEL_SOURCE;
        params.gridDim = dim3(NUM_BLOCKS_PER_DEVICE, 1, 1);
        params.blockDim = dim3(BLOCK_SIZE, 1, 1);
        params.args = argsList[i].data();
        params.sharedMem = 0;
        params.stream = 0;
        
        launchParams.push_back(params);
    }
    
    // Record start time
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Launch cooperative kernel on multiple devices
    result = engine.launchCooperativeKernelMultiDevice(launchParams);
    ASSERT_EQ(result, VGREResult::SUCCESS);
    
    // Record end time
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "Multi-device cooperative launch completed in " << duration.count() << " ms" << std::endl;
    
    // Verify results
    // Each device should have written its device ID
    for (int i = 0; i < NUM_DEVICES; i++) {
        EXPECT_EQ(*device_data[i], i);
    }
    
    // Sync counter should reflect all device activity
    // Each device: 1 (device ID write) + 32 (first warp increments) = 33 per device
    int expected_count = NUM_DEVICES * (1 + 32);  // 4 * 33 = 132
    EXPECT_EQ(*sync_counter, expected_count);
    
    // Cleanup
    for (int i = 0; i < NUM_DEVICES; i++) {
        // Note: MemoryHandle cleanup would be done through engine.free() if needed
    }
    
    std::cout << "testMultiDeviceCooperativeLaunch: " << (test_failed ? "FAILED" : "PASSED") << std::endl;
}

void testEmptyLaunchList() {
    std::cout << "Running testEmptyLaunchList..." << std::endl;
    
    // Test edge case: empty launch list should succeed
    std::vector<RuntimeEngine::CoopMultiLaunchParams> emptyParams;
    
    auto result = RuntimeEngine::instance().launchCooperativeKernelMultiDevice(emptyParams);
    EXPECT_EQ(result, VGREResult::SUCCESS);
    
    std::cout << "testEmptyLaunchList: " << (test_failed ? "FAILED" : "PASSED") << std::endl;
}

void testInvalidParameters() {
    std::cout << "Running testInvalidParameters..." << std::endl;
    
    // Test error handling for invalid parameters
    std::vector<RuntimeEngine::CoopMultiLaunchParams> invalidParams(1);
    
    // Invalid grid dimension (0)
    invalidParams[0].name = "test_kernel";
    invalidParams[0].source = "extern \"C\" __global__ void test_kernel() {}";
    invalidParams[0].gridDim = dim3(0, 1, 1);  // Invalid
    invalidParams[0].blockDim = dim3(32, 1, 1);
    invalidParams[0].args = nullptr;
    invalidParams[0].sharedMem = 0;
    invalidParams[0].stream = 0;
    
    auto result = RuntimeEngine::instance().launchCooperativeKernelMultiDevice(invalidParams);
    EXPECT_EQ(result, VGREResult::ERR_INVALID_VALUE);
    
    // Invalid block dimension (0)
    invalidParams[0].gridDim = dim3(1, 1, 1);
    invalidParams[0].blockDim = dim3(0, 1, 1);  // Invalid
    
    result = RuntimeEngine::instance().launchCooperativeKernelMultiDevice(invalidParams);
    EXPECT_EQ(result, VGREResult::ERR_INVALID_VALUE);
    
    std::cout << "testInvalidParameters: " << (test_failed ? "FAILED" : "PASSED") << std::endl;
}

void testLargeMultiDeviceLaunch() {
    std::cout << "Running testLargeMultiDeviceLaunch..." << std::endl;
    
    // Test with larger number of "devices" to verify scalability
    const int NUM_DEVICES = 8;
    const int NUM_BLOCKS_PER_DEVICE = 16;
    const int BLOCK_SIZE = 128;
    
    std::vector<MemoryHandle> result_handles(NUM_DEVICES);
    auto& engine = RuntimeEngine::instance();
    
    // Allocate result arrays
    for (int i = 0; i < NUM_DEVICES; i++) {
        auto result = engine.malloc(sizeof(int), result_handles[i]);
        ASSERT_EQ(result, VGREResult::SUCCESS);
    }
    
    // Get pointers from handles
    std::vector<int*> results(NUM_DEVICES);
    for (int i = 0; i < NUM_DEVICES; i++) {
        results[i] = reinterpret_cast<int*>(result_handles[i]);
        *results[i] = 0;
    }
    
    // Simple kernel that computes device_id * 100 + block_count
    const char* COMPUTE_KERNEL = R"(
    extern "C" __global__ void compute_kernel(int* result, int device_id) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            *result = device_id * 100 + gridDim.x;
        }
        
        // Ensure all blocks participate
        cooperative_groups::this_grid().sync();
    }
    )";
    
    KernelId kernelId;
    auto result = engine.registerKernel("compute_kernel", COMPUTE_KERNEL, kernelId);
    ASSERT_EQ(result, VGREResult::SUCCESS);
    
    // Prepare launch parameters
    std::vector<RuntimeEngine::CoopMultiLaunchParams> launchParams;
    std::vector<std::vector<void*>> argsList(NUM_DEVICES);
    
    for (int i = 0; i < NUM_DEVICES; i++) {
        argsList[i] = {&results[i], &i};
        
        RuntimeEngine::CoopMultiLaunchParams params;
        params.name = "compute_kernel";
        params.source = COMPUTE_KERNEL;
        params.gridDim = dim3(NUM_BLOCKS_PER_DEVICE, 1, 1);
        params.blockDim = dim3(BLOCK_SIZE, 1, 1);
        params.args = argsList[i].data();
        params.sharedMem = 0;
        params.stream = 0;
        
        launchParams.push_back(params);
    }
    
    // Launch
    result = engine.launchCooperativeKernelMultiDevice(launchParams);
    ASSERT_EQ(result, VGREResult::SUCCESS);
    
    // Verify results
    for (int i = 0; i < NUM_DEVICES; i++) {
        int expected = i * 100 + NUM_BLOCKS_PER_DEVICE;
        EXPECT_EQ(*results[i], expected);
    }
    
    // Cleanup
    // Note: MemoryHandle cleanup would be done through engine.free() if needed
    
    std::cout << "testLargeMultiDeviceLaunch: " << (test_failed ? "FAILED" : "PASSED") << std::endl;
}

int main() {
    std::cout << "=== Multi-Device Cooperative Launch Comprehensive Test ===" << std::endl;
    
    setUp();
    
    // Run all tests
    testSingleDeviceCooperativeBarrier();
    testMultiDeviceCooperativeLaunch();
    testEmptyLaunchList();
    testInvalidParameters();
    testLargeMultiDeviceLaunch();
    
    tearDown();
    
    if (test_failed) {
        std::cout << "\n=== SOME TESTS FAILED ===" << std::endl;
        return 1;
    } else {
        std::cout << "\n=== ALL TESTS PASSED ===" << std::endl;
        return 0;
    }
}