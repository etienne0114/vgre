#ifndef VGRE_WORKLOAD_ENGINE_H
#define VGRE_WORKLOAD_ENGINE_H

#include <atomic>
#include <thread>

namespace vgre {
namespace advanced {

/**
 * @brief WorkloadEngine provides real background compute tasks to demonstrate
 * the high-performance capabilities of the VGRE runtime.
 */
class WorkloadEngine {
public:
  static WorkloadEngine &instance();

  void setEnabled(bool enabled);
  bool isEnabled() const { return running_.load(); }

private:
  WorkloadEngine();
  ~WorkloadEngine();

  void workloadLoop();

  std::atomic<bool> running_{false};
  std::thread workerThread_;

  // Workload resources
  void *d_A = nullptr;
  void *d_B = nullptr;
  void *d_C = nullptr;
  const size_t N = 1024 * 1024; // 1M elements
};

} // namespace advanced
} // namespace vgre

#endif // VGRE_WORKLOAD_ENGINE_H
