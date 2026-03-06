/**
 * VGRE Unit Tests — Memory Manager
 */
#include "vgre/core/memory_manager.h"

#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>

using namespace vgre;
using namespace vgre::core;

void test_allocate_free() {
  MemoryManager mm(1024 * 1024); // 1 MB pool

  MemoryHandle h1 = nullptr;
  VGREResult r = mm.allocate(1024, h1);
  (void)r;
  assert(r == VGREResult::SUCCESS);
  assert(h1 != nullptr);
  assert(mm.isValidHandle(h1));
  assert(mm.getUsedMemory() >= 1024);

  r = mm.free(h1);
  (void)r;
  assert(r == VGREResult::SUCCESS);
  assert(!mm.isValidHandle(h1));
  assert(mm.getUsedMemory() == 0);

  std::cout << "[PASS] Allocate / Free" << std::endl;
}

void test_host_device_copy() {
  MemoryManager mm(1024 * 1024);

  const int N = 256;
  std::vector<float> hostSrc(N);
  std::vector<float> hostDst(N, 0.0f);

  for (int i = 0; i < N; ++i) {
    hostSrc[i] = static_cast<float>(i);
  }

  MemoryHandle devMem = nullptr;
  VGREResult r = mm.allocate(N * sizeof(float), devMem);
  (void)r;
  assert(r == VGREResult::SUCCESS);
  assert(devMem != nullptr);

  // Host → Device
  r = mm.copyHostToDevice(devMem, hostSrc.data(), N * sizeof(float));
  (void)r;
  assert(r == VGREResult::SUCCESS);

  // Device → Host
  r = mm.copyDeviceToHost(hostDst.data(), devMem, N * sizeof(float));
  (void)r;
  assert(r == VGREResult::SUCCESS);

  for (int i = 0; i < N; ++i) {
    assert(hostDst[i] == static_cast<float>(i));
  }

  mm.free(devMem);
  std::cout << "[PASS] Host ↔ Device copy" << std::endl;
}

void test_device_device_copy() {
  MemoryManager mm(1024 * 1024);

  const size_t SIZE = 512;
  MemoryHandle h1 = nullptr;
  MemoryHandle h2 = nullptr;
  VGREResult r = mm.allocate(SIZE, h1);
  (void)r;
  assert(r == VGREResult::SUCCESS);
  assert(h1 != nullptr);

  r = mm.allocate(SIZE, h2);
  (void)r;
  assert(r == VGREResult::SUCCESS);
  assert(h2 != nullptr);

  // Fill h1 via host copy
  std::vector<uint8_t> pattern(SIZE, 0xAB);
  r = mm.copyHostToDevice(h1, pattern.data(), SIZE);
  (void)r;
  assert(r == VGREResult::SUCCESS);

  // D2D copy
  r = mm.copyDeviceToDevice(h2, h1, SIZE);
  (void)r;
  assert(r == VGREResult::SUCCESS);

  // Read h2 back and verify
  std::vector<uint8_t> readback(SIZE, 0);
  r = mm.copyDeviceToHost(readback.data(), h2, SIZE);
  (void)r;
  assert(r == VGREResult::SUCCESS);

  for (size_t i = 0; i < SIZE; ++i) {
    assert(readback[i] == 0xAB);
  }

  mm.free(h1);
  mm.free(h2);
  std::cout << "[PASS] Device ↔ Device copy" << std::endl;
}

void test_oom() {
  MemoryManager mm(1024); // Tiny pool: 1 KB

  MemoryHandle h = nullptr;
  // Attempt to allocate more than pool
  VGREResult r = mm.allocate(2048, h);
  (void)r;
  assert(r == VGREResult::ERROR_OUT_OF_MEMORY);
  assert(h == nullptr);

  std::cout << "[PASS] OOM detection" << std::endl;
}

void test_memory_stats() {
  MemoryManager mm(1024 * 1024); // 1 MB

  assert(mm.getTotalMemory() == 1024 * 1024);
  assert(mm.getUsedMemory() == 0);
  assert(mm.getFreeMemory() == 1024 * 1024);

  MemoryHandle h = nullptr;
  mm.allocate(4096, h);
  assert(h != nullptr);
  assert(mm.getUsedMemory() >= 4096);
  assert(mm.getFreeMemory() < 1024 * 1024);

  mm.free(h);
  std::cout << "[PASS] Memory statistics" << std::endl;
}

int main() {
  std::cout << "=== VGRE Memory Manager Unit Tests ===" << std::endl;

  test_allocate_free();
  test_host_device_copy();
  test_device_device_copy();
  test_oom();
  test_memory_stats();

  std::cout << "\nAll memory manager tests passed!" << std::endl;
  return 0;
}
