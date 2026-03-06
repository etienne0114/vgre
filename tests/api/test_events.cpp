#include <iostream>
#include <unistd.h>
// Standard CUDART API that unmodified frameworks use natively
extern "C" {
typedef int cudaError_t;
typedef struct CUstream_st *cudaStream_t;
typedef struct CUevent_st *cudaEvent_t;

cudaError_t cudaStreamCreate(cudaStream_t *pStream);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaStreamDestroy(cudaStream_t stream);

cudaError_t cudaEventCreate(cudaEvent_t *event);
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream);
cudaError_t cudaEventSynchronize(cudaEvent_t event);
cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end);
cudaError_t cudaEventDestroy(cudaEvent_t event);
}

int main() {
  std::cout << "--- Testing VGRE Extended CUDA API (Events) ---\n";

  cudaStream_t stream;
  if (cudaStreamCreate(&stream) != 0) {
    std::cerr << "Stream creation failed." << std::endl;
    return 1;
  }

  cudaEvent_t start, end;
  if (cudaEventCreate(&start) != 0 || cudaEventCreate(&end) != 0) {
    std::cerr << "Event creation failed." << std::endl;
    return 1;
  }

  std::cout << "Events created successfully.\n";

  // Record start event on stream
  if (cudaEventRecord(start, stream) != 0) {
    std::cerr << "Failed to record start event." << std::endl;
    return 1;
  }

  std::cout << "Start event recorded. Sleeping for 1500us (simulating async "
               "stream work)...\n";
  usleep(150000); // Sleep for 150ms

  // Record end event on stream
  if (cudaEventRecord(end, stream) != 0) {
    std::cerr << "Failed to record end event." << std::endl;
    return 1;
  }

  // Wait for the stream to complete the enqueued events
  if (cudaEventSynchronize(end) != 0) {
    std::cerr << "Failed to sync end event." << std::endl;
    return 1;
  }
  std::cout << "End event synchronized.\n";

  float ms = 0.0f;
  if (cudaEventElapsedTime(&ms, start, end) != 0) {
    std::cerr << "Failed to calculate elapsed time." << std::endl;
    return 1;
  }

  std::cout << "Elapsed time: " << ms << " ms (Expected ~150ms)\n";

  if (ms < 140.0f || ms > 250.0f) {
    std::cerr << "Elapsed time looks suspiciously off. Test failed."
              << std::endl;
    return 1;
  }

  cudaEventDestroy(start);
  cudaEventDestroy(end);
  cudaStreamDestroy(stream);

  std::cout << "[PASS] CUDA Event interception and timing worked flawlessly!\n";
  return 0;
}
