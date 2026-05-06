#include "vgre/api/cuda_interceptor.h"
#include "vgre/core/runtime_engine.h"

#include <cassert>
#include <iostream>

using namespace vgre::api;
using namespace vgre::core;

void test_stream_priority() {
  std::cout << "\n--- Test: Stream Priority ---\n";

  auto &cuda = CUDAInterceptor::instance();
  auto err = cuda.init();
  (void)err;
  assert(err == cudaSuccess);

  int least = 0, greatest = 0;
  err = cuda.deviceGetStreamPriorityRange(&least, &greatest);
  (void)err;
  assert(err == cudaSuccess);

  // Create a stream using the greatest priority supported
  int testPriority = greatest;
  cudaStream_t stream = 0;
  err = cuda.streamCreateWithPriority(&stream, 0, testPriority);
  (void)err;
  assert(err == cudaSuccess);

  // Verify the runtime stored the priority on the stream
  int got = 0;
  VGREResult r = RuntimeEngine::instance().getDevice().getStreamPriority(stream, got);
  (void)r;
  assert(r == VGREResult::SUCCESS);
  assert(got == testPriority);

  err = cuda.streamDestroy(stream);
  (void)err;
  assert(err == cudaSuccess);

  std::cout << "  ✓ Stream priority stored and retrieved\n";
}

int main() {
  test_stream_priority();
  return 0;
}
