#include "vgre/advanced/vgre_simulation_engine.h"
#include "vgre/common/logger.h"
#include "vgre/common/types.h"
#include "vgre/core/memory_manager.h"
#include "vgre/core/runtime_engine.h"
#include <chrono>
#include <string>

namespace vgre {
namespace advanced {

SimulationEngine &SimulationEngine::instance() {
  static SimulationEngine engine;
  return engine;
}

SimulationEngine::SimulationEngine() = default;

SimulationEngine::~SimulationEngine() { setEnabled(false); }

void SimulationEngine::setEnabled(bool enabled) {
  bool expected = !enabled;
  if (running_.compare_exchange_strong(expected, enabled)) {
    if (enabled) {
      workerThread_ = std::thread([this]() { this->simulationLoop(); });
      VGRE_LOG_INFO("SimulationEngine", "VGRE Auto-Pilot Simulation STARTED");
    } else {
      if (workerThread_.joinable()) {
        workerThread_.join();
      }
      VGRE_LOG_INFO("SimulationEngine", "VGRE Auto-Pilot Simulation STOPPED");
    }
  }
}

void SimulationEngine::simulationLoop() {
  auto &runtime = core::RuntimeEngine::instance();
  if (!runtime.isInitialized()) {
    runtime.initialize();
  }

  // Register a simple simulation kernel
  std::string kernelName = "sim_workload";
  std::string source =
      "extern \"C\" __global__ void sim_workload(float* A, float* B, float* C, "
      "int N) {\n"
      "  int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
      "  if (i < N) {\n"
      "    for (int j=0; j<10; ++j) C[i] = A[i] * B[i] + (float)j;\n"
      "  }\n"
      "}";

  KernelId kid;
  runtime.registerKernel(kernelName, source, kid);

  // Allocate VRAM
  runtime.getMemoryManager().allocate(N * sizeof(float), d_A);
  runtime.getMemoryManager().allocate(N * sizeof(float), d_B);
  runtime.getMemoryManager().allocate(N * sizeof(float), d_C);

  dim3 grid(static_cast<uint32_t>(N / 256), 1, 1);
  dim3 block(256, 1, 1);
  int n_val = static_cast<int>(N);
  void *args[] = {&d_A, &d_B, &d_C, &n_val};

  while (running_.load()) {
    // Launch workload on default stream 0
    runtime.launchKernel(kid, grid, block, args, 0, 0);

    // Controlled frequency to simulate load
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  // Cleanup
  runtime.getMemoryManager().free(d_A);
  runtime.getMemoryManager().free(d_B);
  runtime.getMemoryManager().free(d_C);
}

} // namespace advanced
} // namespace vgre
