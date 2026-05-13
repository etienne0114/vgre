#include <iostream>
#include <cstring>

// Standard CUDART API — resolved via LD_PRELOAD when libvgre_cudart.so is loaded
extern "C" {
typedef int cudaError_t;
#define cudaSuccess 0
#define cudaErrorInvalidValue 1
#define cudaErrorInvalidSymbol 13
#define cudaMemcpyHostToDevice 1
#define cudaMemcpyDeviceToHost 2

void __cudaRegisterVar(void **fatCubinHandle, char *hostVar,
                       char *deviceAddress, const char *deviceName,
                       int ext, size_t size, int constant, int global);

cudaError_t cudaMemcpyToSymbol(const void *symbol, const void *src, size_t count,
                               size_t offset, int kind);
cudaError_t cudaMemcpyToSymbolAsync(const void *symbol, const void *src,
                                    size_t count, size_t offset, int kind,
                                    void *stream);
cudaError_t cudaMemcpyFromSymbol(void *dst, const void *symbol, size_t count,
                                  size_t offset, int kind);
cudaError_t cudaMemcpyFromSymbolAsync(void *dst, const void *symbol,
                                      size_t count, size_t offset, int kind,
                                      void *stream);
cudaError_t cudaGetSymbolAddress(void **devPtr, const void *symbol);
cudaError_t cudaStreamCreate(void **pStream);
cudaError_t cudaStreamDestroy(void *stream);
cudaError_t cudaStreamSynchronize(void *stream);
}

int main() {
  std::cout << "--- Testing cudaMemcpyToSymbol / cudaMemcpyFromSymbol ---\n";

  // 1. Register a dummy host variable as a CUDA device symbol.
  //    In real CUDA this is done by compiler-generated init code.
  int hostSymbol = 0;
  __cudaRegisterVar(nullptr, reinterpret_cast<char *>(&hostSymbol),
                      nullptr, "mySymbol", 0, sizeof(int), 0, 1);

  // 2. Verify symbol address lookup works
  void *symbolAddr = nullptr;
  cudaError_t err = cudaGetSymbolAddress(&symbolAddr, &hostSymbol);
  if (err != cudaSuccess || !symbolAddr) {
    std::cerr << "FAIL: cudaGetSymbolAddress failed (err=" << err << ")\n";
    return 1;
  }
  std::cout << "  Symbol address resolved: " << symbolAddr << "\n";

  // 3. Test cudaMemcpyToSymbol
  int srcData = 42;
  std::cout << "Testing cudaMemcpyToSymbol...\n";
  err = cudaMemcpyToSymbol(&hostSymbol, &srcData, sizeof(int), 0,
                           cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    std::cerr << "FAIL: cudaMemcpyToSymbol failed (err=" << err << ")\n";
    return 1;
  }
  std::cout << "  -> wrote 42 to symbol\n";

  // 4. Test cudaMemcpyFromSymbol
  int dstData = 0;
  std::cout << "Testing cudaMemcpyFromSymbol...\n";
  err = cudaMemcpyFromSymbol(&dstData, &hostSymbol, sizeof(int), 0,
                             cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    std::cerr << "FAIL: cudaMemcpyFromSymbol failed (err=" << err << ")\n";
    return 1;
  }
  if (dstData != 42) {
    std::cerr << "FAIL: Expected 42, got " << dstData << "\n";
    return 1;
  }
  std::cout << "  -> read back 42 (correct)\n";

  // 5. Test with offset
  int srcArr[4] = {10, 20, 30, 40};
  std::cout << "Testing offset copy...\n";
  err = cudaMemcpyToSymbol(&hostSymbol, srcArr, sizeof(int) * 4, 0,
                           cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    std::cerr << "FAIL: cudaMemcpyToSymbol (array) failed (err=" << err << ")\n";
    return 1;
  }
  int readBack = 0;
  err = cudaMemcpyFromSymbol(&readBack, &hostSymbol, sizeof(int), sizeof(int) * 2,
                             cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    std::cerr << "FAIL: cudaMemcpyFromSymbol (offset) failed (err=" << err << ")\n";
    return 1;
  }
  if (readBack != 30) {
    std::cerr << "FAIL: Expected 30 at offset 8, got " << readBack << "\n";
    return 1;
  }
  std::cout << "  -> offset read back 30 (correct)\n";

  // 6. Test unregistered symbol returns cudaErrorInvalidSymbol
  std::cout << "Testing unregistered symbol rejection...\n";
  int unregistered = 0;
  err = cudaMemcpyToSymbol(&unregistered, &srcData, sizeof(int), 0,
                           cudaMemcpyHostToDevice);
  if (err != cudaErrorInvalidSymbol) {
    std::cerr << "FAIL: Expected cudaErrorInvalidSymbol (" << cudaErrorInvalidSymbol
              << "), got " << err << "\n";
    return 1;
  }
  std::cout << "  -> cudaErrorInvalidSymbol (correct)\n";

  // 7. Test null parameter rejection
  std::cout << "Testing null parameter rejection...\n";
  err = cudaMemcpyToSymbol(nullptr, &srcData, sizeof(int), 0,
                           cudaMemcpyHostToDevice);
  if (err != cudaErrorInvalidValue) {
    std::cerr << "FAIL: Expected cudaErrorInvalidValue for null symbol\n";
    return 1;
  }
  err = cudaMemcpyFromSymbol(&dstData, nullptr, sizeof(int), 0,
                             cudaMemcpyDeviceToHost);
  if (err != cudaErrorInvalidValue) {
    std::cerr << "FAIL: Expected cudaErrorInvalidValue for null symbol\n";
    return 1;
  }
  std::cout << "  -> null parameters correctly rejected\n";

  // 8. Test async variants (basic smoke test with null stream)
  int asyncData = 77;
  std::cout << "Testing async variants...\n";
  err = cudaMemcpyToSymbolAsync(&hostSymbol, &asyncData, sizeof(int), 0,
                                cudaMemcpyHostToDevice, nullptr);
  if (err != cudaSuccess) {
    std::cerr << "FAIL: cudaMemcpyToSymbolAsync failed (err=" << err << ")\n";
    return 1;
  }
  int asyncRead = 0;
  err = cudaMemcpyFromSymbolAsync(&asyncRead, &hostSymbol, sizeof(int), 0,
                                  cudaMemcpyDeviceToHost, nullptr);
  if (err != cudaSuccess) {
    std::cerr << "FAIL: cudaMemcpyFromSymbolAsync failed (err=" << err << ")\n";
    return 1;
  }
  if (asyncRead != 77) {
    std::cerr << "FAIL: Async read expected 77, got " << asyncRead << "\n";
    return 1;
  }
  std::cout << "  -> async round-trip 77 (correct)\n";

  std::cout << "[PASS] cudaMemcpyToSymbol / cudaMemcpyFromSymbol work correctly!\n";
  return 0;
}
