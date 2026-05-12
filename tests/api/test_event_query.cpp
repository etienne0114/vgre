#include <iostream>
#include <unistd.h>

// Standard CUDART API — resolved via LD_PRELOAD when libvgre_cudart.so is loaded
extern "C" {
typedef int cudaError_t;
#define cudaSuccess 0
#define cudaErrorInvalidValue 1
#define cudaErrorNotReady 34

typedef struct CUstream_st *cudaStream_t;
typedef struct CUevent_st *cudaEvent_t;

cudaError_t cudaStreamCreate(cudaStream_t *pStream);
cudaError_t cudaStreamDestroy(cudaStream_t stream);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);

cudaError_t cudaEventCreate(cudaEvent_t *event);
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream);
cudaError_t cudaEventQuery(cudaEvent_t event);
cudaError_t cudaEventSynchronize(cudaEvent_t event);
cudaError_t cudaEventDestroy(cudaEvent_t event);
}

int main() {
  std::cout << "--- Testing cudaEventQuery ---\n";

  cudaStream_t stream;
  if (cudaStreamCreate(&stream) != 0) {
    std::cerr << "Stream creation failed.\n";
    return 1;
  }

  cudaEvent_t event;
  if (cudaEventCreate(&event) != 0) {
    std::cerr << "Event creation failed.\n";
    return 1;
  }

  // 1. Query before recording → should be cudaErrorNotReady
  std::cout << "Querying unrecorded event...\n";
  cudaError_t err = cudaEventQuery(event);
  if (err != cudaErrorNotReady) {
    std::cerr << "FAIL: Expected cudaErrorNotReady (" << cudaErrorNotReady
              << "), got " << err << "\n";
    return 1;
  }
  std::cout << "  -> cudaErrorNotReady (correct)\n";

  // 2. Record event on stream and synchronize
  std::cout << "Recording event on stream...\n";
  if (cudaEventRecord(event, stream) != 0) {
    std::cerr << "Event record failed.\n";
    return 1;
  }

  // 3. Query after recording → should eventually be cudaSuccess
  std::cout << "Querying recorded event (polling)...\n";
  int attempts = 0;
  const int maxAttempts = 1000;
  do {
    err = cudaEventQuery(event);
    if (err == cudaSuccess) break;
    if (err != cudaErrorNotReady) {
      std::cerr << "FAIL: Unexpected error code " << err << "\n";
      return 1;
    }
    usleep(1000); // 1ms
    ++attempts;
  } while (attempts < maxAttempts);

  if (err != cudaSuccess) {
    std::cerr << "FAIL: Event never became ready after " << maxAttempts
              << " attempts.\n";
    return 1;
  }
  std::cout << "  -> cudaSuccess after " << attempts << " attempt(s)\n";

  // 4. Null event → should return cudaErrorInvalidValue
  std::cout << "Querying null event...\n";
  err = cudaEventQuery(nullptr);
  if (err != cudaErrorInvalidValue) {
    std::cerr << "FAIL: Expected cudaErrorInvalidValue (" << cudaErrorInvalidValue
              << "), got " << err << "\n";
    return 1;
  }
  std::cout << "  -> cudaErrorInvalidValue (correct)\n";

  // Cleanup
  cudaEventDestroy(event);
  cudaStreamDestroy(stream);

  std::cout << "[PASS] cudaEventQuery works correctly!\n";
  return 0;
}
