#include "vgre/api/vgre_c_api.h"
#include <cassert>
#include <iostream>
#include <vector>

int main() {
  std::cout << "=============================================" << std::endl;
  std::cout << "  VGRE P2P Performance & Topology Test" << std::endl;
  std::cout << "=============================================" << std::endl;

  vgre_init();

  int deviceCount = 0;
  vgre_get_device_count(&deviceCount);
  std::cout << "Devices available: " << deviceCount << std::endl;

  if (deviceCount < 4) {
    std::cerr << "Test requires at least 4 devices for topology testing."
              << std::endl;
    return 1;
  }

  size_t size = 1024 * 1024 * 10; // 10 MB

  // Enable P2P: Dev 1 to Dev 0 (Implicitly Dev 0 is current)
  vgre_set_device(0);
  vgre_device_enable_peer_access(1);

  // Enable P2P: Dev 2 to Dev 0 (Implicitly Dev 0 is current)
  vgre_device_enable_peer_access(2);

  void *d0 = nullptr;
  void *d1 = nullptr;
  void *d2 = nullptr;

  vgre_malloc(&d0, size);

  vgre_set_device(1);
  vgre_malloc(&d1, size);

  vgre_set_device(2);
  vgre_malloc(&d2, size);

  std::cout
      << "Performing P2P Copy: Dev 0 -> Dev 1 (Same Bus - NVLink Modeled)..."
      << std::endl;
  vgre_memcpy(d1, d0, size, VGRE_MEMCPY_DEVICE_TO_DEVICE);

  std::cout
      << "Performing P2P Copy: Dev 0 -> Dev 2 (Diff Bus - PCIe Modeled)..."
      << std::endl;
  vgre_memcpy(d2, d0, size, VGRE_MEMCPY_DEVICE_TO_DEVICE);

  std::cout
      << "Telemetry recorded. Check console logs for modeled bandwidth outputs."
      << std::endl;

  vgre_free(d0);
  vgre_set_device(1);
  vgre_free(d1);
  vgre_set_device(2);
  vgre_free(d2);

  vgre_shutdown();
  std::cout << "[PASS] P2P Performance & Topology test" << std::endl;
  return 0;
}
