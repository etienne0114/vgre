#include "vgre/advanced/vgre_workload_engine.h"
#include "vgre/common/logger.h"
#include "vgre/common/types.h"
#include "vgre/core/memory_manager.h"
#include "vgre/core/runtime_engine.h"
#include <chrono>
#include <string>

namespace vgre {
namespace advanced {

WorkloadEngine &WorkloadEngine::instance() {
  static WorkloadEngine engine;
  return engine;
}

WorkloadEngine::WorkloadEngine() = default;

WorkloadEngine::~WorkloadEngine() { setEnabled(false); }

void WorkloadEngine::setEnabled(bool enabled) {
  bool expected = !enabled;
  if (running_.compare_exchange_strong(expected, enabled)) {
    if (enabled) {
      workerThread_ = std::thread([this]() { this->workloadLoop(); });
      VGRE_LOG_INFO("WorkloadEngine",
                    "VGRE Background Workload Engine STARTED");
    } else {
      running_ = false;
      if (workerThread_.joinable()) {
        workerThread_.join();
      }
      VGRE_LOG_INFO("WorkloadEngine",
                    "VGRE Background Workload Engine STOPPED");
    }
  }
}

void WorkloadEngine::workloadLoop() {
  auto &runtime = core::RuntimeEngine::instance();
  if (!runtime.isInitialized()) {
    runtime.initialize();
  }

  // Register a real background workload kernel (Iterative Matrix Transform)
  std::string kernelName = "background_compute";
  std::string source = "extern \"C\" __global__ void background_compute(float* "
                       "A, float* B, float* C, "
                       "int N) {\n"
                       "  int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
                       "  if (i < N) {\n"
                       "    float val = A[i];\n"
                       "    for (int j=0; j<50; ++j) {\n"
                       "       val = val * B[i] + 0.1f;\n"
                       "       if (val > 100.0f) val = 1.0f;\n"
                       "    }\n"
                       "    C[i] = val;\n"
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

    // Controlled frequency to maintain high but stable performance recording
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }

  // Cleanup
  runtime.getMemoryManager().free(d_A);
  runtime.getMemoryManager().free(d_B);
  runtime.getMemoryManager().free(d_C);
}

} // namespace advanced
} // namespace vgre
