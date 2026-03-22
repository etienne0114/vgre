#include "vgre/api/vgre_c_api.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cstring>

typedef uint64_t vgre_kernel_id_t;
typedef uint64_t vgre_graph_id_t;
typedef uint64_t vgre_graph_exec_id_t;

const char* vectorAddSource = R"(
extern "C" __global__ void vectorAdd(float* a, float* b, float* c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = a[i] + b[i];
}
)";

const char* vectorMulSource = R"(
extern "C" __global__ void vectorMul(float* c, float* d, float* e) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    e[i] = c[i] * d[i];
}
)";

int main() {
    vgre_init();
    
    int deviceCount = 0;
    vgre_get_device_count(&deviceCount);
    assert(deviceCount > 0);
    vgre_set_device(0);

    // Register kernels
    vgre_kernel_id_t addId, mulId;
    assert(vgre_register_kernel("vectorAdd", vectorAddSource, &addId) == 0);
    assert(vgre_register_kernel("vectorMul", vectorMulSource, &mulId) == 0);

    const int N = 256;
    size_t size = N * sizeof(float);

    // Allocate buffers
    void *d_a, *d_b, *d_c1, *d_c2, *d_d, *d_e;
    vgre_malloc(&d_a, size);
    vgre_malloc(&d_b, size);
    vgre_malloc(&d_c1, size);
    vgre_malloc(&d_c2, size);
    vgre_malloc(&d_d, size);
    vgre_malloc(&d_e, size);

    // Initialize data
    std::vector<float> h_a(N, 1.0f);
    std::vector<float> h_b(N, 2.0f); // 1 + 2 = 3
    std::vector<float> h_d(N, 10.0f);
    
    vgre_memcpy(d_a, h_a.data(), size, VGRE_MEMCPY_HOST_TO_DEVICE);
    vgre_memcpy(d_b, h_b.data(), size, VGRE_MEMCPY_HOST_TO_DEVICE);
    vgre_memcpy(d_d, h_d.data(), size, VGRE_MEMCPY_HOST_TO_DEVICE);

    // Create Graph
    vgre_graph_id_t graph;
    assert(vgre_graphCreate(&graph) == 0);

    uint32_t grid[] = {1, 1, 1};
    uint32_t block[] = {N, 1, 1};

    // Add nodes: Add1 -> Add2 -> Mul
    uint64_t addNode1, addNode2, mulNode;
    
    // Add1: c1 = a + b (Expect 3.0)
    void* addArgs1[] = {&d_a, &d_b, &d_c1};
    uint8_t types[] = {VGRE_ARG_POINTER, VGRE_ARG_POINTER, VGRE_ARG_POINTER};
    assert(vgre_graphAddKernelNodeEx(graph, addId, "vectorAdd_1", grid, block, 
                                     addArgs1, types, 3, nullptr, 0, &addNode1) == 0);

    // Add2: c2 = c1 + b (Expect 3 + 2 = 5.0)
    // This uses the SAME kernel ID (addId) as Add1. 
    // Hardening Test: Fusing Add1 and Add2 will fail if name collisions are not handled.
    void* addArgs2[] = {&d_c1, &d_b, &d_c2};
    uint64_t dep1[] = {addNode1};
    assert(vgre_graphAddKernelNodeEx(graph, addId, "vectorAdd_2", grid, block, 
                                     addArgs2, types, 3, dep1, 1, &addNode2) == 0);

    // Mul: e = c2 * d (Expect 5 * 10 = 50.0)
    void* mulArgs[] = {&d_c2, &d_d, &d_e};
    uint64_t dep2[] = {addNode2};
    assert(vgre_graphAddKernelNodeEx(graph, mulId, "vectorMul", grid, block, 
                                     mulArgs, types, 3, dep2, 1, &mulNode) == 0);

    // Instantiate - This should trigger JIT Fusion
    std::cout << "[Test] Instantiating graph with identical kernel fusion case..." << std::endl;
    vgre_graph_exec_id_t exec;
    assert(vgre_graphInstantiate(graph, &exec) == 0);

    // Launch
    std::cout << "[Test] Launching exec..." << std::endl;
    assert(vgre_graphLaunch(exec, 0) == 0);
    vgre_synchronize();

    // Verify results
    std::vector<float> h_e(N);
    vgre_memcpy(h_e.data(), d_e, size, VGRE_MEMCPY_DEVICE_TO_HOST);

    for(int i = 0; i < N; ++i) {
        if (h_e[i] != 50.0f) {
            std::cerr << "Verification failed at index " << i << ": " << h_e[i] << " != 50.0" << std::endl;
            return 1;
        }
    }

    std::cout << "Zero-Simulation Hardened JIT Fusion SUCCESS!" << std::endl;

    vgre_graphExecDestroy(exec);
    vgre_graphDestroy(graph);
    vgre_free(d_a);
    vgre_free(d_b);
    vgre_free(d_c1);
    vgre_free(d_c2);
    vgre_free(d_d);
    vgre_free(d_e);

    vgre_shutdown();
    return 0;
}
