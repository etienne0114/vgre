#include "vgre/advanced/vgre_workload_engine.h"
#include "vgre/common/logger.h"
#include "vgre/common/types.h"
#include "vgre/core/memory_manager.h"
#include "vgre/core/runtime_engine.h"
#include "vgre/core/virtual_gpu_device.h"
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
  std::thread toJoin;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (enabled == running_.load())
      return;

    if (enabled) {
      VGRE_LOG_INFO("WorkloadEngine", "Starting background compute...");
      running_.store(true);
      if (workerThread_.joinable()) {
        toJoin = std::move(workerThread_);
      }
      workerThread_ = std::thread(&WorkloadEngine::workloadLoop, this);
    } else {
      VGRE_LOG_INFO("WorkloadEngine", "Stopping background compute...");
      running_.store(false);
      if (workerThread_.joinable()) {
        toJoin = std::move(workerThread_);
      }
    }
  }

  // Join outside the lock to avoid blocking other threads (like telemetry) 
  // that might want to check status or re-enable.
  if (toJoin.joinable()) {
    toJoin.join();
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

  // Create a dedicated stream for background compute to avoid contention.
  if (workloadStream_ == 0) {
    auto &dev = runtime.getDevice();
    if (dev.createStream(workloadStream_, 0) != VGREResult::SUCCESS) {
      VGRE_LOG_ERROR("WorkloadEngine",
                     "Failed to create stream for background workload");
      running_ = false;
      return;
    }
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

  if (workloadKernel_ == 0) {
    if (runtime.registerKernel(kernelName, source, workloadKernel_) != VGREResult::SUCCESS) {
      VGRE_LOG_ERROR("WorkloadEngine",
                     "Failed to register background workload kernel");
      running_ = false;
      return;
    }
  }

  // Allocate VRAM and Initialize Data
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

  // Authoritative Data Initialization (Ensures valid shared state)
  std::vector<float> h_init(N);
  for (size_t i = 0; i < N; ++i) h_init[i] = static_cast<float>(i % 100) / 100.0f;
  mm.copyHostToDevice(d_A, h_init.data(), N * sizeof(float));
  for (size_t i = 0; i < N; ++i) h_init[i] = 1.01f; // Slow growth factor
  mm.copyHostToDevice(d_B, h_init.data(), N * sizeof(float));

  dim3 grid(static_cast<uint32_t>((N + 255) / 256), 1, 1);
  dim3 block(256, 1, 1);
  int n_val = static_cast<int>(N);
  void *args[] = {&d_A, &d_B, &d_C, &n_val};

  while (running_.load()) {
    auto t1 = std::chrono::steady_clock::now();

    // Launch workload on dedicated stream
    auto launchRes = runtime.launchKernel(workloadKernel_, grid, block, args, 0,
                                          workloadStream_);
    if (launchRes != VGREResult::SUCCESS) {
      VGRE_LOG_ERROR("WorkloadEngine",
                     "Background workload kernel launch failed");
      running_ = false;
      break;
    }

    static int iteration = 0;
    if (++iteration % 50 == 0) {
      VGRE_LOG_INFO("WorkloadEngine", "Iteration " + std::to_string(iteration) + ": Background compute heart-beat [OK]");
    }

    // Synchronize to measure actual Host-side busy time
    runtime.streamSynchronize(workloadStream_);
    
    auto t2 = std::chrono::steady_clock::now();
    double busyMs = std::chrono::duration<double, std::milli>(t2 - t1).count();

    double factor = loadFactor_.load();
    if (factor < 0.01) factor = 0.01; // Avoid divide by zero
    if (factor > 1.0) factor = 1.0;

    // Authoritative Pacing: Instead of artificial sleeping, we align to the next
    // real-time scheduling slot based on the target load factor.
    auto nextTarget = t1 + std::chrono::milliseconds(static_cast<int>(busyMs / factor));
    auto now = std::chrono::steady_clock::now();
    
    if (nextTarget > now) {
      // Use high-precision wait for the next real interval
      std::this_thread::sleep_until(nextTarget);
    }
  }

  // CRITICAL FIX: Synchronize stream 0 to ensure all kernels are DONE 
  // before we free the memory buffers (d_A, d_B, d_C).
  // This prevents the SIGSEGV/Invalid Access during dashboard toggle.
  if (workloadStream_ != 0) {
    runtime.streamSynchronize(workloadStream_);
  }

  // Cleanup
  if (d_A)
    mm.free(d_A);
  if (d_B)
    mm.free(d_B);
  if (d_C)
    mm.free(d_C);
  d_A = d_B = d_C = nullptr;

  if (workloadStream_ != 0) {
    runtime.getDevice().destroyStream(workloadStream_);
    workloadStream_ = 0;
  }
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
