/**
 * VGRE Integration Test — CUDA API attributes & mem info
 */
#include "vgre/api/cuda_interceptor.h"
#include "vgre/core/runtime_engine.h"

#include <cassert>
#include <iostream>
#include <vector>

using namespace vgre::api;

void test_cuda_api_attributes() {
  std::cout << "\n--- Test: CUDA API attributes ---\n";

  auto &cuda = CUDAInterceptor::instance();
  auto err = cuda.init();
  (void)err;
  assert(err == cudaSuccess);

  int count = 0;
  err = cuda.getDeviceCount(&count);
  (void)err;
  assert(err == cudaSuccess);
  assert(count >= 1);

  int value = 0;
  err = cuda.deviceGetAttribute(&value, 1, 0); // MaxThreadsPerBlock
  (void)err;
  assert(err == cudaSuccess);
  assert(value > 0);

  err = cuda.deviceGetAttribute(&value, 10, 0); // WarpSize
  (void)err;
  assert(err == cudaSuccess);
  assert(value > 0);

  err = cuda.deviceGetAttribute(&value, 16, 0); // MultiProcessorCount
  (void)err;
  assert(err == cudaSuccess);
  assert(value >= 1);

  err = cuda.deviceGetAttribute(&value, 75, 0); // ComputeCapabilityMajor
  (void)err;
  assert(err == cudaSuccess);
  assert(value >= 1);

  err = cuda.deviceGetAttribute(&value, 76, 0); // ComputeCapabilityMinor
  (void)err;
  assert(err == cudaSuccess);
  assert(value >= 0);

  err = cuda.deviceGetAttribute(&value, 11, 0); // MaxPitch
  (void)err;
  assert(err == cudaSuccess);
  assert(value > 0);

  value = 1234;
  err = cuda.deviceGetAttribute(&value, 38, 0); // L2CacheSize is not modeled.
  assert(err == cudaErrorNotSupported);
  assert(value == 1234);

  err = cuda.deviceGetAttribute(&value, 41, 0); // UnifiedAddressing
  (void)err;
  assert(err == cudaSuccess);

  err = cuda.deviceGetAttribute(&value, 50, 0); // PciDomainId
  (void)err;
  assert(err == cudaSuccess);

  // Regression coverage for CUDA 11 cudaDeviceAttr numeric values. These
  // attributes were previously shifted, causing valid queries to report an
  // unrelated property and unknown IDs to succeed with a fabricated zero.
  vgre::DeviceProperties properties;
  auto propertyResult = vgre::core::RuntimeEngine::instance().getDeviceProperties(0, properties);
  assert(propertyResult == vgre::VGREResult::SUCCESS);
  const auto expectAttribute = [&cuda, &value](int attr, int expected) {
    const auto result = cuda.deviceGetAttribute(&value, attr, 0);
    assert(result == cudaSuccess);
    assert(value == expected);
  };
  expectAttribute(12, properties.maxRegsPerSM); // MaxRegistersPerBlock
  const int textureAttributes[][2] = {
      {45, 1 << 15}, {46, 1 << 15}, {47, 1 << 12}, {48, 1 << 12},
      {49, 1 << 12}, {50, properties.pciDomainId}, {51, 256},
      {52, 1 << 14}, {53, 1 << 14}, {54, 2046}, {55, 1 << 20},
      {56, 1 << 15}, {57, 1 << 15}, {58, 1 << 12}, {59, 1 << 12},
      {60, 1 << 12}, {61, 1 << 20}, {62, 2048}, {63, 1 << 15},
      {64, 1 << 15}, {65, 2048}, {66, 1 << 14}, {67, 1 << 14},
      {68, 2046}, {69, 1 << 27}, {70, 1 << 15}, {71, 1 << 15},
      {72, 1 << 27}, {73, 1 << 15}, {74, 1 << 15}};
  for (const auto &expected : textureAttributes)
    expectAttribute(expected[0], expected[1]);

  const int emulatedAttributes[][2] = {
      {77, 1 << 20}, {78, 1}, {79, 1}, {80, 1},
      {81, properties.maxSharedMemPerSM}, {82, properties.maxRegsPerSM},
      {83, 1}, {84, 0}, {85, 0}, {86, 1}, {88, 0}, {89, 1}, {90, 1},
      {91, 1}, {95, 1}, {96, 0},
      {97, static_cast<int>(properties.sharedMemPerBlock)},
      {98, 0}, {99, 1}, {100, 0}, {101, 0},
      {106, properties.maxBlocksPerSM}, {111, 0}};
  for (const auto &expected : emulatedAttributes)
    expectAttribute(expected[0], expected[1]);

  for (int unavailableAttr : {36, 37, 87}) {
    value = 1234;
    err = cuda.deviceGetAttribute(&value, unavailableAttr, 0);
    assert(err == cudaErrorNotSupported);
    assert(value == 1234);
  }

  for (int invalidAttr : {-1, 0, 44, 92, 102, 105, 107, 112}) {
    value = 1234;
    err = cuda.deviceGetAttribute(&value, invalidAttr, 0);
    assert(err == cudaErrorInvalidValue);
    assert(value == 1234);
  }

  size_t freeBytes = 0;
  size_t totalBytes = 0;
  err = cuda.memGetInfo(&freeBytes, &totalBytes);
  (void)err;
  assert(err == cudaSuccess);
  assert(totalBytes > 0);
  assert(freeBytes <= totalBytes);

  int drv = 0, rt = 0;
  err = cuda.driverGetVersion(&drv);
  (void)err;
  assert(err == cudaSuccess);
  err = cuda.runtimeGetVersion(&rt);
  (void)err;
  assert(err == cudaSuccess);
  assert(drv > 0 && rt > 0);

  int least = 0, greatest = 0;
  err = cuda.deviceGetStreamPriorityRange(&least, &greatest);
  (void)err;
  assert(err == cudaSuccess);

  cudaStream_t stream = 0;
  err = cuda.streamCreateWithPriority(&stream, 0, 0);
  (void)err;
  assert(err == cudaSuccess);
  err = cuda.streamDestroy(stream);
  (void)err;
  assert(err == cudaSuccess);

  unsigned int flags = 0;
  err = cuda.setDeviceFlags(0x1);
  (void)err;
  assert(err == cudaSuccess);
  err = cuda.getDeviceFlags(&flags);
  (void)err;
  assert(err == cudaSuccess);
  assert(flags == 0x1);

  char pciBuf[32] = {0};
  err = cuda.deviceGetPCIBusId(pciBuf, sizeof(pciBuf), 0);
  (void)err;
  assert(err == cudaSuccess);

  int devFromPci = -1;
  err = cuda.deviceGetByPCIBusId(&devFromPci, pciBuf);
  (void)err;
  assert(err == cudaSuccess);
  assert(devFromPci == 0);

  void *hostPtr = nullptr;
  err = cuda.hostAlloc(&hostPtr, 1024, 0);
  (void)err;
  assert(err == cudaSuccess && hostPtr != nullptr);
  err = cuda.hostRegister(hostPtr, 1024, 0);
  (void)err;
  assert(err == cudaSuccess);
  err = cuda.hostUnregister(hostPtr);
  (void)err;
  assert(err == cudaSuccess);
  err = cuda.freeHost(hostPtr);
  (void)err;
  assert(err == cudaSuccess);

  size_t pitch = 0;
  void *devPitch = nullptr;
  const size_t width = 64;
  const size_t height = 8;
  err = cuda.mallocPitch(&devPitch, &pitch, width, height);
  (void)err;
  assert(err == cudaSuccess && devPitch != nullptr);
  assert(pitch >= width);

  std::vector<unsigned char> hSrc(pitch * height, 0);
  std::vector<unsigned char> hDst(pitch * height, 0);
  for (size_t r = 0; r < height; ++r) {
    for (size_t c = 0; c < width; ++c) {
      hSrc[r * pitch + c] = static_cast<unsigned char>((r + c) & 0xFF);
    }
  }

  err = cuda.memcpy2D(devPitch, pitch, hSrc.data(), pitch, width, height,
                      cudaMemcpyHostToDevice);
  (void)err;
  assert(err == cudaSuccess);
  err = cuda.memcpy2D(hDst.data(), pitch, devPitch, pitch, width, height,
                      cudaMemcpyDeviceToHost);
  (void)err;
  assert(err == cudaSuccess);

  for (size_t r = 0; r < height; ++r) {
    for (size_t c = 0; c < width; ++c) {
      assert(hDst[r * pitch + c] == hSrc[r * pitch + c]);
    }
  }

  err = cuda.free(devPitch);
  (void)err;
  assert(err == cudaSuccess);

  err = cuda.streamQuery(0);
  (void)err;
  assert(err == cudaSuccess);

  if (count > 1) {
    int canAccess = 0;
    err = cuda.deviceCanAccessPeer(&canAccess, 0, 1);
    (void)err;
    assert(err == cudaSuccess);
    err = cuda.deviceEnablePeerAccess(1, 0);
    (void)err;
    assert(err == cudaSuccess);
    err = cuda.deviceDisablePeerAccess(1);
    (void)err;
    assert(err == cudaSuccess);
  }

  std::cout << "  ✓ CUDA attribute APIs verified\n";
  std::cout << "[PASS] CUDA API attributes\n";
}

int main() {
  test_cuda_api_attributes();
  return 0;
}
