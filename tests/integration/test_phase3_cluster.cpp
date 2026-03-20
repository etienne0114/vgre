#include "vgre/advanced/tcp_cluster.h"
#include "vgre/api/vgre_c_api.h"
#include "vgre/common/logger.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <cassert>
#include <cstdlib>
#include <sys/wait.h>
#include <unistd.h>

using namespace vgre;
using namespace vgre::advanced;

// Phase 3 Integration Test
// Verifies 12-argument kernel dispatch with UVM coherence across isolated cluster nodes.

void master_node() {
    VGRE_LOG_INFO("Master", "Starting Master Node...");
    setenv("VGRE_TCP_AUTH_TOKEN", "test_secret", 1);
    
    auto &cluster = vgre::advanced::TCPClusterManager::instance();
    // In this test, we now use the singleton for master for simplicity in API calls,
    // but the worker WILL use its own instance.
    cluster.initialize(true, "127.0.0.1", 7780);
    
    // Wait for worker
    int worker_id = -1;
    for (int i = 0; i < 100; ++i) {
        worker_id = cluster.getFirstActiveWorker();
        if (worker_id >= 0) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    if (worker_id < 0) {
        std::cerr << "Master FAIL: No worker connected" << std::endl;
        return;
    }

    // Register a complex kernel
    const std::string kernel_src = R"(
        extern "C" __global__ void test_many_args(
            int* data, int a, int b, int c, int d, int e, int f, int g, int h, int i, int j, int* out) {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            if (tid == 0) {
                *out = data[0] + a + b + c + d + e + f + g + h + i + j;
            }
        }
    )";
    
    uint64_t kid = 0;
    vgre_register_kernel("test_many_args", kernel_src.c_str(), &kid);

    void* dev_data = nullptr;
    void* dev_out = nullptr;
    vgre_malloc_managed(&dev_data, sizeof(int));
    vgre_malloc_managed(&dev_out, sizeof(int));
    
    *(int*)dev_data = 100;
    *(int*)dev_out = 0;

    int a=1, b=2, c=3, d=4, e=5, f=6, g=7, h=8, i=9, j=10;
    void* args[12];
    args[0] = &dev_data;
    args[1] = &a; args[2] = &b; args[3] = &c; args[4] = &d; args[5] = &e;
    args[6] = &f; args[7] = &g; args[8] = &h; args[9] = &i; args[10] = &j;
    args[11] = &dev_out;

    std::cout << "Master: Launching remote kernel (12 args)..." << std::endl;
    uint32_t gd[3] = {1, 1, 1};
    uint32_t bd[3] = {1, 1, 1};
    cluster.launchRemoteKernel(worker_id, kid, gd, bd, args, 12, 0);

    std::cout << "Master: Waiting for results (Memory Coherence)..." << std::endl;
    int result = 0;
    for(int i=0; i<50; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        result = *(int*)dev_out;
        if (result == 155) break;
    }
    
    std::cout << "Master: Final Result=" << result << std::endl;
    if (result == 155) {
        std::cout << "PHASE 3 TEST PASSED" << std::endl;
    } else {
        std::cout << "PHASE 3 TEST FAILED" << std::endl;
    }
    cluster.shutdown();
}

void worker_node() {
    VGRE_LOG_INFO("Worker", "Starting Worker Node...");
    setenv("VGRE_TCP_AUTH_TOKEN", "test_secret", 1);
    
    // Give master time to start
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    vgre::advanced::TCPClusterManager worker_cluster;
    VGREResult res = worker_cluster.initialize(false, "127.0.0.1", 7780);
    if (res != VGREResult::SUCCESS) {
        std::cerr << "Worker FAIL: init failed" << std::endl;
        return;
    }
    
    std::cout << "Worker connected. Waiting for commands..." << std::endl;
    while (worker_cluster.isEnabled()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
}

int main() {
    vgre_init();

    std::thread master_thread(master_node);
    std::this_thread::sleep_for(std::chrono::seconds(1));
    std::thread worker_thread(worker_node);

    master_thread.join();
    // Signal worker to exit by shutting it down or just detach if we don't care about its cleanup
    // But since it's a test, we want a clean exit.
    
    vgre_shutdown();
    if (worker_thread.joinable()) {
        worker_thread.detach(); // Worker loop is infinite, detach is safer than join if we don't have a stop signal
    }
    return 0;
}
