/**
 * VGRE Dispatch Manager - Remote Execution Implementation
 */

#include "vgre/advanced/gpu_passthrough.h"
#include "vgre/advanced/tcp_cluster.h"
#include "vgre/advanced/tcp_cluster/internal/interfaces.h"
#include "vgre/advanced/tcp_cluster/internal/dispatch_manager.h"
#include "vgre/common/logger.h"
#include "vgre/core/runtime_engine.h"
#include "vgre/core/memory_manager.h"
#ifdef VGRE_HAS_OPENCL_BACKEND
#include "vgre/runtime/igpu_opencl_executor.h"
#endif
#include <cstring>
#include <thread>

namespace vgre {
namespace advanced {

VGREResult DispatchManager::launchRemoteKernel(
    int worker_idx, uint64_t kernel_id, const uint32_t grid_dim[3],
    const uint32_t block_dim[3], void **args, int num_args, size_t shared_mem) {
  if (!parent_->is_master_)
    return VGREResult::ERR_INVALID_VALUE;

  // Pre-compile JIT before acquiring clients_mutex_ or sending the packet.
  // Master and worker share the same RuntimeEngine singleton, so once the
  // kernel is in kernelCache_ the worker's launchKernel() finds it instantly
  // (no cold-JIT blocking on the cluster wait path).
  {
    auto preRes = core::RuntimeEngine::instance().ensureKernelCompiled(kernel_id);
    if (preRes != VGREResult::SUCCESS &&
        preRes != VGREResult::ERR_INVALID_KERNEL) {
      VGRE_LOG_ERROR("TCPCluster",
                     "JIT pre-compilation failed for kernel " +
                         std::to_string(kernel_id) +
                         " — aborting remote launch");
      return preRes;
    }
  }

  std::lock_guard<std::recursive_mutex> lock(parent_->clients_mutex_);
  if (worker_idx < 0 ||
      worker_idx >= static_cast<int>(parent_->clients_.size()) ||
      !parent_->clients_[worker_idx] ||
      !parent_->clients_[worker_idx]->active) {
    return VGREResult::ERR_INVALID_VALUE;
  }

  // Stream arguments
  VGREResult argResult = parent_->streamArgumentsToWorker(
      args, num_args, kernel_id, parent_->clients_[worker_idx]);
  if (argResult != VGREResult::SUCCESS)
    return argResult;

  // Send launch packet
  RemoteCommandPacket pkt{};
  pkt.auth_token = parent_->auth_token_;
  pkt.kernel_id = kernel_id;
  std::memcpy(pkt.grid_dim, grid_dim, sizeof(pkt.grid_dim));
  std::memcpy(pkt.block_dim, block_dim, sizeof(pkt.block_dim));
  pkt.shared_mem = shared_mem;
  pkt.num_args = num_args;

  VGREResult sendResult = parent_->send_packet(
      parent_->clients_[worker_idx]->socket_fd, PacketType::LAUNCH_KERNEL, &pkt,
      sizeof(RemoteCommandPacket),
      parent_->clients_[worker_idx]->secure_channel.get());

  if (sendResult == VGREResult::SUCCESS) {
    parent_->clients_[worker_idx]->in_flight_kernels++;
  }
  return sendResult;
}

void DispatchManager::handleRemoteCommand(const RemoteCommandPacket &pkt) {
  if (parent_->auth_token_ != 0 && pkt.auth_token != parent_->auth_token_) {
    VGRE_LOG_ERROR("TCPCluster", "Auth failure in remote command");
    return;
  }

  std::vector<void *> local_args(pkt.num_args, nullptr);
  for (int i = 0; i < pkt.num_args; ++i) {
    auto it = parent_->pending_args_.find(i);
    if (it != parent_->pending_args_.end()) {
      if (it->second.type == (uint8_t)ArgType::STRUCT)
        local_args[i] = it->second.data.data();
      else
        local_args[i] = &it->second.value;
    }
  }

  dim3 gd(pkt.grid_dim[0], pkt.grid_dim[1], pkt.grid_dim[2]);
  dim3 bd(pkt.block_dim[0], pkt.block_dim[1], pkt.block_dim[2]);

  // ── GPU dispatch cascade: dGPU → iGPU (OpenCL) → CPU JIT ────────────────
  // Try the highest-throughput available backend.  Each path falls through to
  // the next on failure so workers are always productive even without a GPU.
  VGREResult r = VGREResult::ERR_LAUNCH_FAILURE;

  // 1. Discrete NVIDIA GPU via libcuda passthrough
  GPUPassthrough& gp = GPUPassthrough::instance();
  if (gp.isAvailable()) {
    auto* ir = core::RuntimeEngine::instance().getKernelIR(pkt.kernel_id);
    if (ir) {
      std::vector<size_t> argSizes(static_cast<size_t>(pkt.num_args), 0);
      auto& memMgr = core::RuntimeEngine::instance().getMemoryManager();
      for (int i = 0; i < pkt.num_args; ++i) {
        if (i < static_cast<int>(ir->argTypes.size()) &&
            ir->argTypes[i] == ArgType::POINTER && local_args[i]) {
          void* p = *reinterpret_cast<void**>(local_args[i]);
          if (p) {
            try { argSizes[static_cast<size_t>(i)] = memMgr.getAllocationSize(p); }
            catch (...) {}
          }
        }
      }
      r = gp.launchOnGPU(ir->source, ir->name, gd, bd,
                         local_args.data(), pkt.num_args,
                         pkt.shared_mem, ir->argTypes, argSizes);
      if (r == VGREResult::SUCCESS)
        VGRE_LOG_DEBUG("TCPCluster", "Kernel dispatched via GPU passthrough (dGPU)");
      else
        VGRE_LOG_DEBUG("TCPCluster", "GPU passthrough failed — trying iGPU");
    }
  }

  // 2. Integrated GPU via OpenCL (CUDA→OpenCL transpilation)
#ifdef VGRE_HAS_OPENCL_BACKEND
  if (r != VGREResult::SUCCESS) {
    auto& igpu = runtime::IGPUOpenCLExecutor::instance();
    if (igpu.isInitialized()) {
      auto* ir = core::RuntimeEngine::instance().getKernelIR(pkt.kernel_id);
      if (ir) {
        std::vector<size_t> argSizes(static_cast<size_t>(pkt.num_args), 0);
        auto& memMgr = core::RuntimeEngine::instance().getMemoryManager();
        for (int i = 0; i < pkt.num_args; ++i) {
          if (i < static_cast<int>(ir->argTypes.size()) &&
              ir->argTypes[i] == ArgType::POINTER && local_args[i]) {
            void* p = *reinterpret_cast<void**>(local_args[i]);
            if (p) {
              try { argSizes[static_cast<size_t>(i)] = memMgr.getAllocationSize(p); }
              catch (...) {}
            }
          }
        }
        r = igpu.execute(ir->name, ir->source, ir->argTypes,
                         gd, bd, local_args.data(), argSizes);
        if (r == VGREResult::SUCCESS)
          VGRE_LOG_DEBUG("TCPCluster", "Kernel dispatched via OpenCL iGPU");
        else
          VGRE_LOG_DEBUG("TCPCluster", "OpenCL iGPU failed — falling back to CPU JIT");
      }
    }
  }
#endif

  // 3. CPU JIT fallback (always available)
  if (r != VGREResult::SUCCESS) {
    r = vgre::core::RuntimeEngine::instance().launchKernel(
        pkt.kernel_id, gd, bd, local_args.data(), pkt.shared_mem, 0);
  }

  ResponsePacket resp{pkt.kernel_id, r};
  parent_->send_packet(parent_->client_fd_, PacketType::RESPONSE, &resp,
                       sizeof(ResponsePacket),
                       parent_->client_secure_channel_.get());
  parent_->pending_args_.clear();
}

VGREResult DispatchManager::waitForRemoteResult(uint64_t kernel_id,
                                                int timeout_ms) {
  auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  std::unique_lock<std::mutex> lock(parent_->remote_results_mutex_);
  while (remote_kernel_results_.find(kernel_id) ==
         remote_kernel_results_.end()) {
    if (parent_->remote_results_cv_.wait_until(lock, deadline) ==
        std::cv_status::timeout)
      return VGREResult::ERR_TIMEOUT;
  }
  VGREResult r = remote_kernel_results_[kernel_id].first;
  remote_kernel_results_.erase(kernel_id);
  return r;
}

} // namespace advanced
} // namespace vgre
