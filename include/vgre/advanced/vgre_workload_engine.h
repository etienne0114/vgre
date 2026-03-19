#ifndef VGRE_WORKLOAD_ENGINE_H
#define VGRE_WORKLOAD_ENGINE_H

#include "vgre/common/types.h"

#include <atomic>
#include <mutex>
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
  
  void setLoadFactor(double factor) { loadFactor_.store(factor); }
  double getLoadFactor() const { return loadFactor_.load(); }

private:
  WorkloadEngine();
  ~WorkloadEngine();

  void workloadLoop();

  std::atomic<bool> running_{false};
  std::atomic<double> loadFactor_{0.2}; // Default 20% load
  std::thread workerThread_;
  mutable std::mutex mutex_;

  // Workload resources
  void *d_A = nullptr;
  void *d_B = nullptr;
  void *d_C = nullptr;
  StreamId workloadStream_ = 0;
  const size_t N = 4 * 1024 * 1024; // Increase to 4M for better measurement
};

} // namespace advanced
} // namespace vgre

#endif // VGRE_WORKLOAD_ENGINE_H
