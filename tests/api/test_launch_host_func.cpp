#include <iostream>
#include <atomic>

// Standard CUDART API — resolved via LD_PRELOAD when libvgre_cudart.so is loaded
extern "C" {
typedef int cudaError_t;
#define cudaSuccess 0
#define cudaErrorInvalidValue 1

typedef struct CUstream_st *cudaStream_t;

cudaError_t cudaStreamCreate(cudaStream_t *pStream);
cudaError_t cudaStreamDestroy(cudaStream_t stream);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaLaunchHostFunc(cudaStream_t stream, void (*fn)(void *),
                                void *userData);
}

static std::atomic<int> g_fnCount{0};
static std::atomic<int> g_fnValue{-1};

static void myHostFn(void *userData) {
  int *value = static_cast<int *>(userData);
  *value = 99;
  g_fnValue.store(*value, std::memory_order_relaxed);
  g_fnCount.fetch_add(1, std::memory_order_relaxed);
}

int main() {
  std::cout << "--- Testing cudaLaunchHostFunc ---\n";

  cudaStream_t stream;
  if (cudaStreamCreate(&stream) != 0) {
    std::cerr << "Stream creation failed.\n";
    return 1;
  }

  int userData = 0;

  // 1. Launch host function on stream
  std::cout << "Launching host function on stream...\n";
  if (cudaLaunchHostFunc(stream, myHostFn, &userData) != 0) {
    std::cerr << "cudaLaunchHostFunc failed.\n";
    return 1;
  }

  // 2. Synchronize to ensure host function has run
  std::cout << "Synchronizing stream...\n";
  if (cudaStreamSynchronize(stream) != 0) {
    std::cerr << "Stream synchronize failed.\n";
    return 1;
  }

  // 3. Verify host function was invoked and userData modified
  if (g_fnCount.load() != 1) {
    std::cerr << "FAIL: Host function was not invoked (count="
              << g_fnCount.load() << ")\n";
    return 1;
  }
  if (userData != 99) {
    std::cerr << "FAIL: userData was not modified (value=" << userData
              << ")\n";
    return 1;
  }
  std::cout << "  -> host fn invoked, userData=99\n";

  // 4. Test null function rejection
  std::cout << "Testing null function rejection...\n";
  cudaError_t err = cudaLaunchHostFunc(stream, nullptr, &userData);
  if (err != cudaErrorInvalidValue) {
    std::cerr << "FAIL: Expected cudaErrorInvalidValue (" << cudaErrorInvalidValue
              << "), got " << err << "\n";
    return 1;
  }
  std::cout << "  -> cudaErrorInvalidValue (correct)\n";

  cudaStreamDestroy(stream);

  std::cout << "[PASS] cudaLaunchHostFunc works correctly!\n";
  return 0;
}
