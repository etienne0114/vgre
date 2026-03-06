#include "vgre/advanced/tcp_cluster.h"
#include <cassert>
#include <iostream>

using namespace vgre::advanced;

void test_initialization() {
  std::cout << "\n--- Test: TCP Cluster Initialization & Shutdown ---\n";
  auto &cluster = TCPClusterManager::instance();

  // Shut down just in case
  cluster.shutdown();

  // Test Server Initialization
  vgre::VGREResult r = cluster.initialize(true, "127.0.0.1", 7780);
  (void)r;
  assert(r == vgre::VGREResult::SUCCESS);
  assert(cluster.isEnabled());
  assert(cluster.isMaster());

  cluster.shutdown();
  assert(!cluster.isEnabled());

  std::cout << "[PASS] Initialization and Shutdown\n";
}

void test_telemetry_aggregation() {
  std::cout << "\n--- Test: Remote Telemetry Aggregation (Empty state) ---\n";
  auto &cluster = TCPClusterManager::instance();

  // Shut down just in case
  cluster.shutdown();

  // Master port
  vgre::VGREResult r = cluster.initialize(true, "127.0.0.1", 7781);
  (void)r;
  assert(r == vgre::VGREResult::SUCCESS);

  // Feed local state
  vgre_telemetry_t t1{};
  t1.gflops = 55.5;
  t1.memory_used_bytes = 1024;
  t1.active_kernels = 2;
  cluster.broadcastLocalTelemetry(t1); // Local broadcast does nothing on master

  vgre_telemetry_t agg{};
  cluster.aggregateRemoteTelemetry(agg);

  // 0.0 because there are no remote clients feeding the master via sockets
  assert(agg.gflops == 0.0f);

  cluster.shutdown();

  std::cout << "[PASS] Telemetry Aggregation\n";
}

int main() {
  std::cout << "=============================================" << std::endl;
  std::cout << "  VGRE Integration Test — TCP Cluster        " << std::endl;
  std::cout << "=============================================" << std::endl;

  test_initialization();
  test_telemetry_aggregation();

  std::cout << "\n✓ All TCP Cluster integration tests passed!" << std::endl;
  return 0;
}
