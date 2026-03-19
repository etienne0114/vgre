#include "vgre/api/cuda_interceptor.h"
#include <iostream>
#include <vector>

using namespace vgre::api;

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " code=" \
                << err << std::endl;                                           \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

int main() {
  std::cout << "Starting test_graph_update..." << std::endl;
  CUDAInterceptor &cuda = CUDAInterceptor::instance();
  cuda.init();

  const size_t N = 256;
  const size_t bytes = N * sizeof(int);

  int *d_src, *d_dst;
  CHECK_CUDA(cuda.malloc((void **)&d_src, bytes));
  CHECK_CUDA(cuda.malloc((void **)&d_dst, bytes));

  std::vector<int> h_src(N, 7);
  std::vector<int> h_dst(N, 0);

  CHECK_CUDA(cuda.memcpy(d_src, h_src.data(), bytes, cudaMemcpyHostToDevice));

  // 1. Create a graph
  cudaGraph_t graph1;
  CHECK_CUDA(cuda.graphCreate(&graph1, 0));

  // 2. Add Memcpy Node
  CUDAInterceptor::cudaGraphNode_t memcpyNode1;
  // In our interceptor, cudaMemcpy3DParms is used for graphAddMemcpyNode, 
  // but we added graphAddMemcpyNode1D as a helper.
  CHECK_CUDA(cuda.graphAddMemcpyNode1D(&memcpyNode1, graph1, nullptr, 0, d_dst, d_src, bytes, cudaMemcpyDeviceToDevice));

  // 3. Instantiate Executable Graph
  cudaGraphExec_t graphExec;
  CHECK_CUDA(cuda.graphInstantiate(&graphExec, graph1));

  // 4. Create a second identical graph (same topology) to test update
  cudaGraph_t graph2;
  CHECK_CUDA(cuda.graphCreate(&graph2, 0));
  
  CUDAInterceptor::cudaGraphNode_t memcpyNode2;
  CHECK_CUDA(cuda.graphAddMemcpyNode1D(&memcpyNode2, graph2, nullptr, 0, d_dst, d_src, bytes, cudaMemcpyDeviceToDevice));
  
  // 5. Update Executable Graph
  // This should trigger the new rigorous topological verification in GraphManager::updateExec
  std::cout << "Triggering Graph Exec Update topological verification..." << std::endl;
  CUDAInterceptor::cudaGraphExecUpdateResult updateResult;
  CHECK_CUDA(cuda.graphExecUpdate(graphExec, graph2, nullptr, &updateResult));
  
  if (updateResult != CUDAInterceptor::cudaGraphExecUpdateSuccess) {
      std::cerr << "Graph execution update rejected topology mismatch incorrectly!" << std::endl;
      return 1;
  }

  // 6. Launch the updated graph
  CHECK_CUDA(cuda.graphLaunch(graphExec, 0));
  CHECK_CUDA(cuda.deviceSynchronize());

  // Verify
  CHECK_CUDA(cuda.memcpy(h_dst.data(), d_dst, bytes, cudaMemcpyDeviceToHost));
  
  bool success = true;
  for (size_t i = 0; i < N; ++i) {
    if (h_dst[i] != 7) {
      success = false;
      break;
    }
  }

  CHECK_CUDA(cuda.graphExecDestroy(graphExec));
  CHECK_CUDA(cuda.graphDestroy(graph1));
  CHECK_CUDA(cuda.graphDestroy(graph2));
  CHECK_CUDA(cuda.free(d_src));
  CHECK_CUDA(cuda.free(d_dst));

  if (success) {
    std::cout << "test_graph_update PASSED" << std::endl;
    return 0;
  } else {
    std::cout << "test_graph_update FAILED" << std::endl;
    return 1;
  }
}
