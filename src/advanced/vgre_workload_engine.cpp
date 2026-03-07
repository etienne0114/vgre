#include "vgre/advanced/vgre_workload_engine.h"
#include "vgre/common/logger.h"
#include "vgre/common/types.h"
#include "vgre/core/memory_manager.h"
#include "vgre/core/runtime_engine.h"
#include <chrono>
#include <exception>
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
  try {
  auto &runtime = core::RuntimeEngine::instance();
  if (!runtime.isInitialized() &&
      runtime.initialize() != VGREResult::SUCCESS) {
    VGRE_LOG_ERROR("WorkloadEngine",
                   "Failed to initialize runtime for background workload");
    running_ = false;
    return;
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

  KernelId kid = 0;
  if (runtime.registerKernel(kernelName, source, kid) != VGREResult::SUCCESS) {
    VGRE_LOG_ERROR("WorkloadEngine",
                   "Failed to register background workload kernel");
    running_ = false;
    return;
  }

  // Allocate VRAM
  auto &mm = runtime.getMemoryManager();
  if (mm.allocate(N * sizeof(float), d_A) != VGREResult::SUCCESS ||
      mm.allocate(N * sizeof(float), d_B) != VGREResult::SUCCESS ||
      mm.allocate(N * sizeof(float), d_C) != VGREResult::SUCCESS) {
    VGRE_LOG_ERROR("WorkloadEngine",
                   "Failed to allocate memory for background workload");
    if (d_A)
      mm.free(d_A);
    if (d_B)
      mm.free(d_B);
    if (d_C)
      mm.free(d_C);
    d_A = d_B = d_C = nullptr;
    running_ = false;
    return;
  }

  dim3 grid(static_cast<uint32_t>((N + 255) / 256), 1, 1);
  dim3 block(256, 1, 1);
  int n_val = static_cast<int>(N);
  void *args[] = {&d_A, &d_B, &d_C, &n_val};

  while (running_.load()) {
    // Launch workload on default stream 0
    auto launchRes = runtime.launchKernel(kid, grid, block, args, 0, 0);
    if (launchRes != VGREResult::SUCCESS) {
      VGRE_LOG_ERROR("WorkloadEngine",
                     "Background workload kernel launch failed");
      running_ = false;
      break;
    }

    // Controlled frequency to maintain high but stable performance recording
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }

  // Cleanup
  if (d_A)
    mm.free(d_A);
  if (d_B)
    mm.free(d_B);
  if (d_C)
    mm.free(d_C);
  d_A = d_B = d_C = nullptr;
  } catch (const std::exception &e) {
    VGRE_LOG_ERROR("WorkloadEngine",
                   std::string("Background workload loop failed: ") + e.what());
    running_ = false;
  } catch (...) {
    VGRE_LOG_ERROR("WorkloadEngine",
                   "Background workload loop failed with unknown exception");
    running_ = false;
  }
}

} // namespace advanced
} // namespace vgre
