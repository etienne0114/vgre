#include <iostream>

// Standard CUDART API — resolved via LD_PRELOAD when libvgre_cudart.so is loaded
extern "C" {
typedef int cudaError_t;
#define cudaSuccess 0
#define cudaErrorInvalidValue 1
#define cudaMemoryTypeHost 1
#define cudaMemoryTypeDevice 2
#define cudaMemoryTypeManaged 3

struct cudaPointerAttributes {
  int memoryType;
  int device;
  void *devicePointer;
  void *hostPointer;
  int isManaged;
};

cudaError_t cudaPointerGetAttributes(struct cudaPointerAttributes *attributes,
                                      const void *ptr);
cudaError_t cudaMalloc(void **devPtr, size_t size);
cudaError_t cudaFree(void *devPtr);
cudaError_t cudaMallocManaged(void **devPtr, size_t size, unsigned int flags);
}

int main() {
  std::cout << "--- Testing cudaPointerGetAttributes ---\n";

  // 1. Null attributes rejected
  std::cout << "Testing null attributes rejection...\n";
  int dummy = 0;
  cudaError_t err = cudaPointerGetAttributes(nullptr, &dummy);
  if (err != cudaErrorInvalidValue) {
    std::cerr << "FAIL: Expected cudaErrorInvalidValue for null attributes\n";
    return 1;
  }
  std::cout << "  -> null attributes correctly rejected\n";

  // 2. Device allocation (cudaMalloc)
  std::cout << "Testing device pointer (cudaMalloc)...\n";
  void *dptr = nullptr;
  err = cudaMalloc(&dptr, 64);
  if (err != cudaSuccess || !dptr) {
    std::cerr << "FAIL: cudaMalloc failed\n";
    return 1;
  }
  cudaPointerAttributes attr;
  err = cudaPointerGetAttributes(&attr, dptr);
  if (err != cudaSuccess) {
    std::cerr << "FAIL: cudaPointerGetAttributes failed (err=" << err << ")\n";
    return 1;
  }
  if (attr.memoryType != cudaMemoryTypeDevice) {
    std::cerr << "FAIL: Expected cudaMemoryTypeDevice (" << cudaMemoryTypeDevice
              << "), got " << attr.memoryType << "\n";
    return 1;
  }
  if (attr.isManaged != 0) {
    std::cerr << "FAIL: Expected isManaged=0, got " << attr.isManaged << "\n";
    return 1;
  }
  if (attr.devicePointer != dptr) {
    std::cerr << "FAIL: devicePointer mismatch\n";
    return 1;
  }
  if (attr.hostPointer != dptr) {
    std::cerr << "FAIL: hostPointer mismatch\n";
    return 1;
  }
  std::cout << "  -> device type, isManaged=0, device=" << attr.device << "\n";
  cudaFree(dptr);

  // 3. Managed allocation (cudaMallocManaged)
  std::cout << "Testing managed pointer (cudaMallocManaged)...\n";
  void *mptr = nullptr;
  err = cudaMallocManaged(&mptr, 64, 0);
  if (err != cudaSuccess || !mptr) {
    std::cerr << "FAIL: cudaMallocManaged failed\n";
    return 1;
  }
  err = cudaPointerGetAttributes(&attr, mptr);
  if (err != cudaSuccess) {
    std::cerr << "FAIL: cudaPointerGetAttributes failed (err=" << err << ")\n";
    return 1;
  }
  if (attr.memoryType != cudaMemoryTypeManaged) {
    std::cerr << "FAIL: Expected cudaMemoryTypeManaged (" << cudaMemoryTypeManaged
              << "), got " << attr.memoryType << "\n";
    return 1;
  }
  if (attr.isManaged != 1) {
    std::cerr << "FAIL: Expected isManaged=1, got " << attr.isManaged << "\n";
    return 1;
  }
  std::cout << "  -> managed type, isManaged=1, device=" << attr.device << "\n";
  cudaFree(mptr);

  // 4. Unregistered host pointer
  std::cout << "Testing unregistered host pointer...\n";
  int hostVal = 42;
  err = cudaPointerGetAttributes(&attr, &hostVal);
  if (err != cudaSuccess) {
    std::cerr << "FAIL: cudaPointerGetAttributes failed for host ptr (err=" << err << ")\n";
    return 1;
  }
  if (attr.memoryType != cudaMemoryTypeHost) {
    std::cerr << "FAIL: Expected cudaMemoryTypeHost (" << cudaMemoryTypeHost
              << "), got " << attr.memoryType << "\n";
    return 1;
  }
  if (attr.devicePointer != nullptr) {
    std::cerr << "FAIL: Expected devicePointer=nullptr for host ptr\n";
    return 1;
  }
  if (attr.hostPointer != &hostVal) {
    std::cerr << "FAIL: hostPointer mismatch\n";
    return 1;
  }
  std::cout << "  -> host type, devicePointer=nullptr\n";

  std::cout << "[PASS] cudaPointerGetAttributes works correctly!\n";
  return 0;
}
