/**
 * VGRE Unit Tests — Virtual GPU Device
 * Tests device instance creation, properties, context, and stream management.
 */
#include "vgre/core/virtual_gpu_device.h"

#include <cassert>
#include <cstring>
#include <iostream>

using namespace vgre;
using namespace vgre::core;

void test_device_creation() {
  VirtualGPUDevice dev(0);
  dev.detectHardware();

  assert(dev.getId() == 0);

  const auto &props = dev.getProperties();
  assert(strlen(props.name) > 0);
  assert(props.totalGlobalMem > 0);
  assert(props.maxThreadsPerBlock > 0);
  assert(props.multiProcessorCount > 0);
  assert(props.warpSize == 32);
  assert(props.major >= 1);

  std::cout << "[PASS] Device creation: " << props.name << std::endl;
  std::cout << "[PASS] VRAM: " << props.totalGlobalMem / (1024 * 1024) << " MB"
            << std::endl;
  std::cout << "[PASS] SM Count: " << props.multiProcessorCount << std::endl;
}

void test_context() {
  VirtualGPUDevice dev(0);
  dev.detectHardware();

  assert(!dev.hasContext());
  assert(dev.createContext() == VGREResult::SUCCESS);
  assert(dev.hasContext());

  // Double create should return ALREADY_EXISTS
  assert(dev.createContext() == VGREResult::ERR_ALREADY_EXISTS);

  assert(dev.destroyContext() == VGREResult::SUCCESS);
  assert(!dev.hasContext());

  std::cout << "[PASS] Context management" << std::endl;
}

void test_streams() {
  VirtualGPUDevice dev(0);
  dev.detectHardware();
  dev.createContext();

  StreamId s1, s2;
  assert(dev.createStream(s1) == VGREResult::SUCCESS);
  (void)s1;
  assert(dev.createStream(s2) == VGREResult::SUCCESS);
  (void)s2;
  assert(s1 != s2);

  assert(dev.synchronizeStream(s1) == VGREResult::SUCCESS);
  assert(dev.destroyStream(s1) == VGREResult::SUCCESS);
  assert(dev.destroyStream(s2) == VGREResult::SUCCESS);

  // Destroy non-existent stream
  assert(dev.destroyStream(999) == VGREResult::ERR_INVALID_VALUE);

  dev.destroyContext();
  std::cout << "[PASS] Stream management" << std::endl;
}

void test_device_id() {
  VirtualGPUDevice dev0(0);
  VirtualGPUDevice dev1(1);

  assert(dev0.getId() == 0);
  assert(dev1.getId() == 1);

  dev0.setId(42);
  assert(dev0.getId() == 42);

  std::cout << "[PASS] Device ID management" << std::endl;
}

int main() {
  std::cout << "=== VGRE Virtual GPU Device Unit Tests ===" << std::endl;

  test_device_creation();
  test_context();
  test_streams();
  test_device_id();

  std::cout << "\nAll device tests passed!" << std::endl;
  return 0;
}
