#ifndef VGRE_SIMULATION_ENGINE_H
#define VGRE_SIMULATION_ENGINE_H

#include <atomic>
#include <thread>

namespace vgre {
namespace advanced {

class SimulationEngine {
public:
  static SimulationEngine &instance();

  void setEnabled(bool enabled);
  bool isEnabled() const { return running_.load(); }

private:
  SimulationEngine();
  ~SimulationEngine();

  void simulationLoop();

  std::atomic<bool> running_{false};
  std::thread workerThread_;

  // Simulation workload resources
  void *d_A = nullptr;
  void *d_B = nullptr;
  void *d_C = nullptr;
  const size_t N = 1024 * 1024; // 1M elements
};

} // namespace advanced
} // namespace vgre

#endif // VGRE_SIMULATION_ENGINE_H
