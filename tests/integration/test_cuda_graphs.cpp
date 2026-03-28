#include "vgre/api/cuda_interceptor.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include "vgre/core/runtime_engine.h"

using namespace vgre;
using namespace vgre::api;

int main() {
  std::cout << "=============================================" << std::endl;
  std::cout << "  VGRE Integration Test — CUDA Graphs        " << std::endl;
  std::cout << "=============================================" << std::endl;

  CUDAInterceptor &cuda = CUDAInterceptor::instance();
  cuda.init();

  const int N = 1024;
  size_t size = N * sizeof(float);
  float *h_a, *h_b, *h_c;
  float *d_a, *d_b, *d_c;

  h_a = (float *)malloc(size);
  h_b = (float *)malloc(size);
  h_c = (float *)malloc(size);

  for (int i = 0; i < N; i++) {
    h_a[i] = 1.0f;
    h_b[i] = 2.0f;
  }

  cuda.malloc((void **)&d_a, size);
  cuda.malloc((void **)&d_b, size);
  cuda.malloc((void **)&d_c, size);

  cuda.memcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
  cuda.memcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

  std::string source = R"(
        extern "C" __global__ void vadd(float* a, float* b, float* c, int n) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) {
                c[i] = a[i] + b[i];
            }
        }
    )";

  dim3 grid(4, 1, 1);
  dim3 block(256, 1, 1);
  void *args[] = {&d_a, &d_b, &d_c, (void *)&N};

  // --- Graph Capture ---
  std::cout << "--- Step 1: Capturing graph ---" << std::endl;
  cudaStream_t stream;
  cuda.streamCreate(&stream);

  cudaGraph_t graph;
  cuda.streamBeginCapture(stream);

  // Record 3 kernel launches
  for (int i = 0; i < 3; i++) {
    cuda.launchKernel("vadd", source, grid, block, args, 0, stream);
  }

  cuda.streamEndCapture(stream, &graph);
  std::cout << "  Captured graph ID: " << graph << std::endl;

  // --- Instantiate and Launch ---
  std::cout << "--- Step 2: Instantiating and launching graph ---" << std::endl;
  cudaGraphExec_t exec;
  cuda.graphInstantiate(&exec, graph);

  // Reset output
  cuda.memset(d_c, 0, size);

  // Launch graph
  cuda.graphLaunch(exec, stream);
  cuda.streamSynchronize(stream);

  // --- Verify ---
  std::cout << "--- Step 3: Verifying results ---" << std::endl;
  cuda.memcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

  int errors = 0;
  for (int i = 0; i < N; i++) {
    if (std::abs(h_c[i] - 3.0f) > 1e-5) {
      errors++;
    }
  }

  if (errors == 0) {
    std::cout << "  ✓ Graph execution verified correctly" << std::endl;
    std::cout << "[PASS] CUDA Graphs test" << std::endl;
  } else {
    std::cout << "  ✗ Verification failed with " << errors << " errors"
              << std::endl;
    std::cout << "  Sample values (first 10):" << std::endl;
    for (int i = 0; i < 10; i++) {
      std::cout << "    [" << i << "] = " << h_c[i] << " (expected 3.0)" << std::endl;
    }
    return 1;
  }

  cuda.free(d_a);
  cuda.free(d_b);
  cuda.free(d_c);
  free(h_a);
  free(h_b);
  free(h_c);

  // --- Native Graph with Dependencies ---
  std::cout << "--- Step 4: Native graph with dependencies ---" << std::endl;
  auto &engine = vgre::core::RuntimeEngine::instance();
  engine.initialize();

  KernelId kid = 0;
  engine.registerKernel("vadd_native", source, kid);

  GraphId graph2 = 0;
  engine.graphCreate(graph2);

  uint64_t node1 = 0;
  uint64_t node2 = 0;
  uint64_t node3 = 0;
  void *args2[] = {&d_a, &d_b, &d_c, (void *)&N};

  std::vector<ArgType> argTypes = {ArgType::POINTER, ArgType::POINTER,
                                   ArgType::POINTER, ArgType::INT32};
  engine.graphAddKernelNode(graph2, kid, "vadd_native", grid, block, args2,
                            argTypes, {}, node1);
  engine.graphAddKernelNode(graph2, kid, "vadd_native", grid, block, args2,
                            argTypes, {node1}, node2);
  engine.graphAddKernelNode(graph2, kid, "vadd_native", grid, block, args2,
                            argTypes, {node2}, node3);

  GraphExecId exec2 = 0;
  engine.graphInstantiate(graph2, exec2);
  engine.graphLaunch(exec2, 0);
  engine.streamSynchronize(0);

  std::cout << "  ✓ Native dependency graph executed" << std::endl;

  // --- Update node args ---
  std::cout << "--- Step 5: Updating graph node args ---" << std::endl;
  int N2 = 512;
  void *args3[] = {&d_a, &d_b, &d_c, (void *)&N2};
  engine.graphUpdateKernelNode(graph2, node1, args3, argTypes);
  engine.graphUpdateKernelNode(graph2, node2, args3, argTypes);
  engine.graphUpdateKernelNode(graph2, node3, args3, argTypes);

  engine.graphLaunch(exec2, 0);
  engine.streamSynchronize(0);
  std::cout << "  ✓ Graph node args updated and executed" << std::endl;

  return 0;
}
