#include <iostream>
#include <cstring>

// Standard CUDART API — resolved via LD_PRELOAD when libvgre_cudart.so is loaded
extern "C" {
typedef int cudaError_t;
#define cudaSuccess 0
#define cudaErrorInvalidValue 1
#define cudaErrorMemoryAllocation 2
#define cudaErrorNotReady 34
#define cudaErrorUnknown 999

const char *cudaGetErrorString(cudaError_t error);
const char *cudaGetErrorName(cudaError_t error);
}

int main() {
  std::cout << "--- Testing cudaGetErrorName / cudaGetErrorString ---\n";

  // 1. cudaSuccess
  const char *name = cudaGetErrorName(cudaSuccess);
  const char *str = cudaGetErrorString(cudaSuccess);
  if (std::strcmp(name, "cudaSuccess") != 0) {
    std::cerr << "FAIL: cudaSuccess name='" << name << "'\n";
    return 1;
  }
  if (std::strcmp(str, "no error") != 0) {
    std::cerr << "FAIL: cudaSuccess string='" << str << "'\n";
    return 1;
  }
  std::cout << "  cudaSuccess: name='" << name << "', string='" << str << "'\n";

  // 2. cudaErrorInvalidValue
  name = cudaGetErrorName(cudaErrorInvalidValue);
  str = cudaGetErrorString(cudaErrorInvalidValue);
  if (std::strcmp(name, "cudaErrorInvalidValue") != 0) {
    std::cerr << "FAIL: cudaErrorInvalidValue name='" << name << "'\n";
    return 1;
  }
  if (std::strcmp(str, "invalid argument") != 0) {
    std::cerr << "FAIL: cudaErrorInvalidValue string='" << str << "'\n";
    return 1;
  }
  std::cout << "  cudaErrorInvalidValue: name='" << name << "', string='" << str << "'\n";

  // 3. cudaErrorMemoryAllocation
  name = cudaGetErrorName(cudaErrorMemoryAllocation);
  str = cudaGetErrorString(cudaErrorMemoryAllocation);
  if (std::strcmp(name, "cudaErrorMemoryAllocation") != 0) {
    std::cerr << "FAIL: cudaErrorMemoryAllocation name='" << name << "'\n";
    return 1;
  }
  if (std::strcmp(str, "out of memory") != 0) {
    std::cerr << "FAIL: cudaErrorMemoryAllocation string='" << str << "'\n";
    return 1;
  }
  std::cout << "  cudaErrorMemoryAllocation: name='" << name << "', string='" << str << "'\n";

  // 4. cudaErrorNotReady
  name = cudaGetErrorName(cudaErrorNotReady);
  str = cudaGetErrorString(cudaErrorNotReady);
  if (std::strcmp(name, "cudaErrorNotReady") != 0) {
    std::cerr << "FAIL: cudaErrorNotReady name='" << name << "'\n";
    return 1;
  }
  if (std::strcmp(str, "device not ready") != 0) {
    std::cerr << "FAIL: cudaErrorNotReady string='" << str << "'\n";
    return 1;
  }
  std::cout << "  cudaErrorNotReady: name='" << name << "', string='" << str << "'\n";

  // 5. Unknown / unmapped error code
  name = cudaGetErrorName(cudaErrorUnknown);
  str = cudaGetErrorString(cudaErrorUnknown);
  if (std::strcmp(name, "cudaErrorUnknown") != 0) {
    std::cerr << "FAIL: cudaErrorUnknown name='" << name << "'\n";
    return 1;
  }
  if (std::strcmp(str, "unknown error") != 0) {
    std::cerr << "FAIL: cudaErrorUnknown string='" << str << "'\n";
    return 1;
  }
  std::cout << "  cudaErrorUnknown: name='" << name << "', string='" << str << "'\n";

  std::cout << "[PASS] cudaGetErrorName and cudaGetErrorString work correctly!\n";
  return 0;
}
