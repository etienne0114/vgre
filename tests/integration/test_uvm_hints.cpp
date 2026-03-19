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
  std::cout << "Starting test_uvm_hints..." << std::endl;
  CUDAInterceptor &cuda = CUDAInterceptor::instance();
  cuda.init();

  const size_t N = 1024 * 1024;
  const size_t bytes = N * sizeof(float);
  float *data;

  // 1. Allocate Managed Memory
  CHECK_CUDA(cuda.mallocManaged(reinterpret_cast<void **>(&data), bytes));

  // 2. Apply memAdvise (SetReadMostly)
  // This physically invokes the POSIX madvise(MADV_RANDOM) on Linux!
  std::cout << "Applying cudaMemAdvise (SetReadMostly)..." << std::endl;
  // Using 1 for cudaMemAdviseSetReadMostly (as per our implementation)
  CHECK_CUDA(cuda.memAdvise(data, bytes, 1, 0));

  // 3. Pre-fault memory asynchronously
  // This triggers proactive read/write faults to warm the page cache exactly as requested!
  std::cout << "Applying cudaMemPrefetchAsync..." << std::endl;
  cudaStream_t stream;
  CHECK_CUDA(cuda.streamCreate(&stream));
  CHECK_CUDA(cuda.memPrefetchAsync(data, bytes, 0, stream));
  CHECK_CUDA(cuda.streamSynchronize(stream));

  // 4. Verify we can still write and read correctly
  for (size_t i = 0; i < N; ++i) {
    data[i] = static_cast<float>(i) * 2.0f;
  }

  // 5. Apply memAdvise (SetPreferredLocation)
  // Using 3 for cudaMemAdviseSetPreferredLocation
  CHECK_CUDA(cuda.memAdvise(data, bytes, 3, 0));

  bool success = true;
  for (size_t i = 0; i < N; ++i) {
    if (data[i] != static_cast<float>(i) * 2.0f) {
      success = false;
      break;
    }
  }

  CHECK_CUDA(cuda.free(data));
  CHECK_CUDA(cuda.streamDestroy(stream));

  if (success) {
    std::cout << "test_uvm_hints PASSED" << std::endl;
    return 0;
  } else {
    std::cout << "test_uvm_hints FAILED" << std::endl;
    return 1;
  }
}
