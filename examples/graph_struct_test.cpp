#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include "vgre/api/vgre_c_api.h"

// Define a custom struct to test authoritative capture
struct Config {
    float gain;
    int offset;
    float bias;
};

const char* kernel_source = R"(
struct Config {
    float gain;
    int offset;
    float bias;
};

__global__ void struct_kernel(float* data, Config config, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        data[tid] = data[tid] * config.gain + config.offset + config.bias;
    }
}
)";

int main() {
    std::cout << "VGRE Graph STRUCT Support Verification" << std::endl;

    // 1. Initialize
    vgre_init();

    // 2. Prepare data
    const int n = 1024;
    size_t size = n * sizeof(float);
    float* h_data = (float*)malloc(size);
    for (int i = 0; i < n; ++i) h_data[i] = (float)i;

    void* d_data;
    vgre_malloc(&d_data, size);
    vgre_memcpy(d_data, h_data, size, VGRE_MEMCPY_HOST_TO_DEVICE);

    // 3. Register Kernel
    uint64_t kernel_id;
    vgre_register_kernel("struct_kernel", kernel_source, &kernel_id);

    // 4. Create Graph
    uint64_t graph;
    vgre_graphCreate(&graph);

    // 5. Setup Struct Argument
    Config cfg = {2.5f, 10, 1.5f};
    
    // C API uses uint8_t for arg types
    std::vector<uint8_t> argTypes = {
        VGRE_ARG_POINTER, // data
        VGRE_ARG_STRUCT,  // config
        VGRE_ARG_INT32    // n
    };

    void* kernel_args[] = { &d_data, &cfg, (void*)&n };
    
    uint32_t grid[] = {4, 1, 1};
    uint32_t block[] = {256, 1, 1};
    
    uint64_t node_id;
    vgre_graphAddKernelNodeEx(graph, kernel_id, "struct_kernel", grid, block, 
                                kernel_args, argTypes.data(), (int)argTypes.size(), 
                                nullptr, 0, &node_id);

    // 6. Instantiate and Launch
    uint64_t exec;
    vgre_graphInstantiate(graph, &exec);
    vgre_graphLaunch(exec, 0);
    vgre_synchronize();

    // 7. Verify Results
    vgre_memcpy(h_data, d_data, size, VGRE_MEMCPY_DEVICE_TO_HOST);

    bool passed = true;
    for (int i = 0; i < n; ++i) {
        float expected = (float)i * cfg.gain + cfg.offset + cfg.bias;
        if (std::abs(h_data[i] - expected) > 1e-4) {
            std::cerr << "Mismatch at index " << i << ": got " << h_data[i] 
                      << ", expected " << expected << std::endl;
            passed = false;
            break;
        }
    }

    if (passed) {
        std::cout << "✓ Graph STRUCT Verification PASSED" << std::endl;
    } else {
        std::cout << "✗ Graph STRUCT Verification FAILED" << std::endl;
    }

    // Cleanup
    vgre_graphExecDestroy(exec);
    vgre_graphDestroy(graph);
    vgre_free(d_data);
    free(h_data);
    vgre_shutdown();

    return passed ? 0 : 1;
}
