#include <iostream>
#include <atomic>
#include <unistd.h>

// Standard CUDART API — resolved via LD_PRELOAD when libvgre_cudart.so is loaded
extern "C" {
typedef int cudaError_t;
#define cudaSuccess 0
#define cudaErrorInvalidValue 1

typedef struct CUstream_st *cudaStream_t;

cudaError_t cudaStreamCreate(cudaStream_t *pStream);
cudaError_t cudaStreamDestroy(cudaStream_t stream);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaStreamAddCallback(cudaStream_t stream,
                                   void (*callback)(cudaStream_t, cudaError_t,
                                                    void *),
                                   void *userData, unsigned int flags);
}

static std::atomic<int> g_callbackCount{0};
static std::atomic<int> g_callbackStatus{-1};

static void myCallback(cudaStream_t stream, cudaError_t status,
                                  void *userData) {
  (void)stream;
  int *value = static_cast<int *>(userData);
  *value = 42;
  g_callbackStatus.store(status, std::memory_order_relaxed);
  g_callbackCount.fetch_add(1, std::memory_order_relaxed);
}

int main() {
  std::cout << "--- Testing cudaStreamAddCallback ---\n";

  cudaStream_t stream;
  if (cudaStreamCreate(&stream) != 0) {
    std::cerr << "Stream creation failed.\n";
    return 1;
  }

  int userData = 0;

  // 1. Add callback to stream
  std::cout << "Adding callback to stream...\n";
  if (cudaStreamAddCallback(stream, myCallback, &userData, 0) != 0) {
    std::cerr << "cudaStreamAddCallback failed.\n";
    return 1;
  }

  // 2. Synchronize to ensure callback has run
  std::cout << "Synchronizing stream...\n";
  if (cudaStreamSynchronize(stream) != 0) {
    std::cerr << "Stream synchronize failed.\n";
    return 1;
  }

  // 3. Verify callback was invoked with correct status and userData modified
  if (g_callbackCount.load() != 1) {
    std::cerr << "FAIL: Callback was not invoked (count="
              << g_callbackCount.load() << ")\n";
    return 1;
  }
  if (g_callbackStatus.load() != cudaSuccess) {
    std::cerr << "FAIL: Callback status was not cudaSuccess (status="
              << g_callbackStatus.load() << ")\n";
    return 1;
  }
  if (userData != 42) {
    std::cerr << "FAIL: userData was not modified (value=" << userData
              << ")\n";
    return 1;
  }
  std::cout << "  -> callback invoked, status=cudaSuccess, userData=42\n";

  // 4. Test null callback rejection
  std::cout << "Testing null callback rejection...\n";
  cudaError_t err = cudaStreamAddCallback(stream, nullptr, &userData, 0);
  if (err != cudaErrorInvalidValue) {
    std::cerr << "FAIL: Expected cudaErrorInvalidValue (" << cudaErrorInvalidValue
              << "), got " << err << "\n";
    return 1;
  }
  std::cout << "  -> cudaErrorInvalidValue (correct)\n";

  cudaStreamDestroy(stream);

  std::cout << "[PASS] cudaStreamAddCallback works correctly!\n";
  return 0;
}
