/**
 * VGRE Dispatch Manager Implementation
 * 
 * Handles kernel dispatch operations for TCP cluster including remote launches,
 * partitioned dispatch, result collection, and kernel registration broadcasting.
 */

#include "vgre/advanced/tcp_cluster/internal/dispatch_manager.h"
#include "vgre/advanced/tcp_cluster.h"
#include "vgre/advanced/secure_channel.h"
#include "vgre/advanced/workload_partitioner.h"
#include "vgre/advanced/hybrid_compute_manager.h"
#include "vgre/core/runtime_engine.h"
#include "vgre/core/memory_manager.h"
#include "vgre/common/logger.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

// Configurable via VGRE_CLUSTER_MAX_IN_FLIGHT (default 16).
const uint32_t kMaxInFlight = []() -> uint32_t {
    const char* env = std::getenv("VGRE_CLUSTER_MAX_IN_FLIGHT");
    if (env) {
        try {
            long v = std::stol(env);
            if (v > 0 && v <= 1024) return static_cast<uint32_t>(v);
        } catch (...) {}
    }
    return 16;
}();

// Configurable via VGRE_CLUSTER_SHM_RESULT_OFFSET (bytes, default 128 MB).
// Defines the start of the pull-back result region inside the SHM segment.
// Must be >= the size of argument/data written into the first part of SHM.
const size_t kResultRegionStart = []() -> size_t {
    const char* env = std::getenv("VGRE_CLUSTER_SHM_RESULT_OFFSET");
    if (env) {
        try {
            long long v = std::stoll(env);
            if (v >= 1024 * 1024) return static_cast<size_t>(v);
        } catch (...) {}
    }
    return 128ULL * 1024 * 1024;
}();

// One-shot memcpy throughput benchmark used to estimate local SHM bandwidth.
// Runs 10 × 1 MB copies and converts to Gbps. Result is cached in a static
// so it runs only once per process rather than before every dispatch call.
double measureLocalShmBandwidthGbps() {
    constexpr size_t kBenchBytes = 1024 * 1024; // 1 MB
    std::vector<uint8_t> src(kBenchBytes, 0xAB);
    std::vector<uint8_t> dst(kBenchBytes, 0);
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 10; ++i) std::memcpy(dst.data(), src.data(), kBenchBytes);
    double elapsed_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    if (elapsed_s <= 0.0) return 100.0; // unreachable, but safe fallback
    double bw_gbps = (10.0 * kBenchBytes * 8.0) / (elapsed_s * 1e9);
    // Clamp to [1, 1000] Gbps — covers LPDDR5 to HBM3 range.
    return std::max(1.0, std::min(1000.0, bw_gbps));
}

} // anonymous namespace

namespace vgre {
namespace advanced {

DispatchManager::DispatchManager(TCPClusterManager* parent)
    : parent_(parent),
      result_shm_offset_(kResultRegionStart) {
  if (!parent_) {
    throw std::invalid_argument("DispatchManager: parent cannot be null");
  }
}

VGREResult DispatchManager::launchRemoteKernel(
    int worker_idx, uint64_t kernel_id,
    const uint32_t grid_dim[3], const uint32_t block_dim[3],
    void** args, int num_args, size_t shared_mem) {
  
  if (!parent_->is_master_)
    return VGREResult::ERR_INVALID_VALUE;
  if (!grid_dim || !block_dim || num_args < 0)
    return VGREResult::ERR_INVALID_VALUE;
  if (num_args > 0 && !args)
    return VGREResult::ERR_INVALID_VALUE;
  if (grid_dim[0] == 0 || block_dim[0] == 0)
    return VGREResult::ERR_INVALID_VALUE;

  std::lock_guard<std::recursive_mutex> lock(parent_->clients_mutex_);
  if (worker_idx < 0 || worker_idx >= static_cast<int>(parent_->clients_.size()) ||
      !parent_->clients_[worker_idx] || !parent_->clients_[worker_idx]->active) {
    return VGREResult::ERR_INVALID_VALUE;
  }

  if (parent_->auth_token_ == 0) {
    VGRE_LOG_ERROR("TCPCluster",
                   "Remote kernel dispatch blocked: missing "
                   "VGRE_TCP_AUTH_TOKEN");
    return VGREResult::ERR_INVALID_VALUE;
  }

  if (parent_->clients_[worker_idx]->in_flight_kernels >= kMaxInFlight) {
      VGRE_LOG_WARN("TCPCluster", "Congestion: Max in-flight kernels reached for " + 
                    parent_->clients_[worker_idx]->ip_address);
      return VGREResult::ERR_BUSY;
  }

  // 1. Stream all arguments FIRST
  VGREResult argResult = parent_->streamArgumentsToWorker(
      args, num_args, kernel_id, parent_->clients_[worker_idx]);
  if (argResult != VGREResult::SUCCESS) {
      return argResult;
  }

  // 2. Send LAUNCH_KERNEL header LAST (this triggers execution on worker side)
  RemoteCommandPacket pkt{};
  pkt.auth_token = parent_->auth_token_;
  pkt.kernel_id = kernel_id;
  std::memcpy(pkt.grid_dim, grid_dim, sizeof(pkt.grid_dim));
  std::memcpy(pkt.block_dim, block_dim, sizeof(pkt.block_dim));
  pkt.shared_mem = shared_mem;
  pkt.num_args = num_args; 

  if (parent_->send_packet(
          parent_->clients_[worker_idx]->socket_fd,
          PacketType::LAUNCH_KERNEL,
          &pkt, sizeof(RemoteCommandPacket),
          parent_->clients_[worker_idx]->secureChannel.get()) != VGREResult::SUCCESS) {
    return VGREResult::ERR_IO;
  }
  parent_->clients_[worker_idx]->in_flight_kernels++;

  VGRE_LOG_INFO("TCPCluster",
                "Dispatched remote kernel launch (" + std::to_string(num_args) +
                    " args) to worker " + std::to_string(worker_idx));
  return VGREResult::SUCCESS;
}

void DispatchManager::broadcastKernelRegistration(
    uint64_t kernel_id,
    const std::string& name,
    const std::string& source) {
  
  if (!parent_->enabled_ || !parent_->is_master_)
    return;

  std::lock_guard<std::recursive_mutex> lock(parent_->clients_mutex_);
  KernelRegisterPacket kpkt{};
  kpkt.auth_token = parent_->auth_token_;
  kpkt.kernel_id = kernel_id;
  std::strncpy(kpkt.name, name.c_str(), sizeof(kpkt.name) - 1);
  kpkt.source_len = static_cast<uint32_t>(source.length());

  for (auto& client : parent_->clients_) {
    if (client && client->active) {
      VGREResult r1 = parent_->send_packet(
          client->socket_fd,
          PacketType::REGISTER_KERNEL,
          &kpkt, sizeof(KernelRegisterPacket),
          client->secureChannel.get());
      if (r1 != VGREResult::SUCCESS) {
        VGRE_LOG_ERROR("TCPCluster",
            "broadcastKernelRegistration: REGISTER_KERNEL send failed for " +
            client->ip_address + " — worker will silently miss kernel " +
            std::to_string(kernel_id) + " and fail on dispatch");
        continue; // skip RAW_DATA; no point sending source without the header
      }
      VGREResult r2 = parent_->send_packet(
          client->socket_fd,
          PacketType::RAW_DATA,
          source.c_str(), source.length(),
          client->secureChannel.get());
      if (r2 != VGREResult::SUCCESS) {
        VGRE_LOG_ERROR("TCPCluster",
            "broadcastKernelRegistration: RAW_DATA (kernel source) send failed for " +
            client->ip_address + " — worker has header but no source for kernel " +
            std::to_string(kernel_id));
      }
    }
  }
}

void DispatchManager::handleRemoteCommand(const RemoteCommandPacket& pkt) {
  // Early authentication validation — B5: use constant-time compare to prevent
  // timing side-channel enumeration of the auth token hash.
  bool tokenOk = (parent_->auth_token_ != 0) &&
      vgre::advanced::crypto::secure_compare(
          reinterpret_cast<const uint8_t*>(&pkt.auth_token),
          reinterpret_cast<const uint8_t*>(&parent_->auth_token_),
          sizeof(parent_->auth_token_));
  if (!tokenOk) {
    VGRE_LOG_ERROR("TCPCluster",
                   "Rejected remote command due to missing/invalid auth token");
    parent_->pending_args_.clear(); // Clear any pending arguments on auth failure
    return;
  }
  
  // Packet structure validation after auth check
  if (pkt.grid_dim[0] == 0 || pkt.block_dim[0] == 0) {
    VGRE_LOG_ERROR("TCPCluster", "Invalid Remote Command Packet Received");
    parent_->pending_args_.clear(); // Clear pending arguments on validation failure
    return;
  }

  VGRE_LOG_INFO("TCPCluster",
                "Executing remote kernel launch request (Kernel ID: " +
                    std::to_string(pkt.kernel_id) + " | Args: " + 
                    std::to_string(pkt.num_args) + ")");

  int numArgs = pkt.num_args;
  std::vector<void*> local_args(numArgs, nullptr);
  
  // Build argument array from pending_args_
  for (int i = 0; i < numArgs; ++i) {
    auto it = parent_->pending_args_.find(i);
    if (it == parent_->pending_args_.end()) {
        VGRE_LOG_ERROR("TCPCluster", "Remote execution failed: missing argument data for index " + 
                       std::to_string(i));
        parent_->pending_args_.clear();
        return;
    }
    
    TCPClusterManager::PendingArg& arg = it->second;
    ArgType type = static_cast<ArgType>(arg.type);
    
    if (type == ArgType::STRUCT) {
       local_args[i] = arg.data.data();
    } else if (type == ArgType::POINTER) {
       local_args[i] = &arg.value;
    } else {
       // For scalars, value is already stored in PendingArg::value (64-bit).
       local_args[i] = &arg.value;
    }
  }

  dim3 gd(pkt.grid_dim[0], pkt.grid_dim[1], pkt.grid_dim[2]);
  dim3 bd(pkt.block_dim[0], pkt.block_dim[1], pkt.block_dim[2]);

  auto r = vgre::core::RuntimeEngine::instance().launchKernel(
      pkt.kernel_id, gd, bd, local_args.data(), pkt.shared_mem, 0);
  
  if (r != VGREResult::SUCCESS) {
    VGRE_LOG_ERROR("TCPCluster",
                   "Remote kernel execution failed with code " +
                       std::to_string(static_cast<int>(r)));
  }

  // Wait for local execution to complete before syncing back memory
  vgre::core::RuntimeEngine::instance().streamSynchronize(0);

  // Memory Coherence (Pull Back) - sync modified pointers back to master.
  // Reset the SHM result cursor to kResultRegionStart at the beginning of each
  // kernel's pull-back phase. This prevents wrap-around from overwriting earlier
  // results from the SAME kernel that the master has not yet read:
  //   - handleRemoteCommand is called serially (one at a time from the staging
  //     buffer loop), so the master has already consumed all results from the
  //     PREVIOUS kernel before this kernel is dispatched.
  //   - Within a single kernel, results accumulate sequentially and are never
  //     overwritten — if the region is full, remaining results fall back to TCP.
  {
    std::lock_guard<std::mutex> shm_lock(shm_result_mutex_);
    result_shm_offset_ = kResultRegionStart;
  }

  for (int i = 0; i < numArgs; ++i) {
      auto it = parent_->pending_args_.find(i);
      if (it == parent_->pending_args_.end()) continue;

      TCPClusterManager::PendingArg& arg = it->second;
      if (arg.type == (uint8_t)ArgType::POINTER) {
          void* ptr = reinterpret_cast<void*>(arg.value);
          size_t size = vgre::core::RuntimeEngine::instance()
                            .getMemoryManager()
                            .getAllocationSize(ptr);
          if (size > 0) {
              if (parent_->client_shm_enabled_ && parent_->client_shm_manager_) {
                  // SHM pull-back path. Advance cursor; if it would exceed the SHM
                  // region, fall back to TCP for this result.
                  const size_t shmSize = parent_->client_shm_manager_->getSize();
                  bool usedShm = false;
                  if (shmSize > kResultRegionStart && size <= shmSize - kResultRegionStart) {
                      // Try to allocate from the result region. allocated=true means
                      // we got a valid slot; offset is set only in that case.
                      bool allocated = false;
                      uint64_t offset = 0;
                      {
                          std::lock_guard<std::mutex> shm_lock(shm_result_mutex_);
                          if (result_shm_offset_ + size <= shmSize) {
                              offset = result_shm_offset_;
                              result_shm_offset_ += size;
                              allocated = true;
                          }
                          // else: region exhausted — fall back to TCP (allocated stays false).
                          // Do NOT wrap: would overwrite earlier results for this kernel.
                      }
                      if (allocated) {
                          std::memcpy(
                              static_cast<uint8_t*>(parent_->client_shm_manager_->getBasePtr()) + offset,
                              ptr, size);
                          DataShmPacket dspkt{};
                          dspkt.target_ptr = arg.value;
                          dspkt.shm_offset = offset;
                          dspkt.size = size;
                          parent_->send_packet(
                              parent_->client_fd_,
                              PacketType::DATA_SHM,
                              &dspkt, sizeof(DataShmPacket),
                              parent_->client_secure_channel_.get());
                          usedShm = true;
                      }
                  }
                  if (!usedShm) {
                      // Fallback to TCP (result too large, SHM exhausted, or size check failed)
                      DataHeaderPacket dptr_pkt{};
                      dptr_pkt.target_ptr = arg.value;
                      dptr_pkt.size = size;
                      parent_->send_packet(
                          parent_->client_fd_,
                          PacketType::DATA_HEADER,
                          &dptr_pkt, sizeof(DataHeaderPacket),
                          parent_->client_secure_channel_.get());
                      parent_->send_packet(
                          parent_->client_fd_,
                          PacketType::DATA_BODY,
                          ptr, size,
                          parent_->client_secure_channel_.get());
                  }
              } else {
                  // TCP path
                  DataHeaderPacket dptr_pkt{};
                  dptr_pkt.target_ptr = arg.value;
                  dptr_pkt.size = size;
                  parent_->send_packet(
                      parent_->client_fd_,
                      PacketType::DATA_HEADER,
                      &dptr_pkt, sizeof(DataHeaderPacket),
                      parent_->client_secure_channel_.get());
                  parent_->send_packet(
                      parent_->client_fd_,
                      PacketType::DATA_BODY,
                      ptr, size,
                      parent_->client_secure_channel_.get());
              }
          }
      }
  }

  // Send RESPONSE packet
  ResponsePacket resp{};
  resp.kernel_id = pkt.kernel_id;
  resp.result = r;
  parent_->send_packet(
      parent_->client_fd_,
      PacketType::RESPONSE,
      &resp, sizeof(ResponsePacket),
      parent_->client_secure_channel_.get());
  
  // Clear pending args after launch
  parent_->pending_args_.clear();
}

VGREResult DispatchManager::launchPartitionedKernel(
    uint64_t kernel_id,
    const uint32_t grid_dim[3], const uint32_t block_dim[3],
    void** args, int num_args, size_t shared_mem) {
  
  if (!parent_->is_master_ || !parent_->enabled_) {
    return VGREResult::ERR_NOT_INITIALIZED;
  }
  if (!grid_dim || !block_dim || grid_dim[0] == 0 || block_dim[0] == 0) {
    return VGREResult::ERR_INVALID_VALUE;
  }

  // Build node capability list
  std::vector<NodeCapability> nodes;
  auto resources = HybridComputeManager::instance().getResources();

  // Local node
  NodeCapability local;
  local.address = "local";
  local.worker_idx = -1;
  local.cpu_cores = resources.cpuCores;
  local.measured_gflops = resources.gflops;
  local.avg_latency_ms = resources.avgLatency;
  local.is_local = true;
  // Measure actual memory copy bandwidth for the local node (cached after first call).
  static const double kLocalBwGbps = measureLocalShmBandwidthGbps();
  local.network_bandwidth_gbps = kLocalBwGbps;
  nodes.push_back(local);

  // Remote workers
  {
    std::lock_guard<std::recursive_mutex> lock(parent_->clients_mutex_);
    for (size_t i = 0; i < parent_->clients_.size(); ++i) {
      if (!parent_->clients_[i] || !parent_->clients_[i]->active || 
          parent_->clients_[i]->cpu_cores <= 0)
        continue;
      NodeCapability cap;
      cap.address = parent_->clients_[i]->ip_address;
      cap.worker_idx = static_cast<int>(i);
      cap.cpu_cores = parent_->clients_[i]->cpu_cores;
      cap.measured_gflops = parent_->clients_[i]->last_telemetry.gflops;
      cap.avg_latency_ms = parent_->clients_[i]->network_latency_ms;
      cap.is_local = false;
      cap.network_bandwidth_gbps = parent_->clients_[i]->network_bandwidth_gbps;
      nodes.push_back(cap);
    }
  }

  // Create partition plan
  PartitionPlan plan;
  auto r = WorkloadPartitioner::instance().createPartitionPlan(
      grid_dim, block_dim, nodes, plan);
  if (r != VGREResult::SUCCESS) {
    return r;
  }

  if (!WorkloadPartitioner::instance().validatePlan(plan)) {
    VGRE_LOG_ERROR("TCPCluster", "Partition plan validation failed");
    return VGREResult::ERR_INVALID_VALUE;
  }

  // Clear previous results
  {
    std::lock_guard<std::mutex> lock(parent_->partition_mutex_);
    partition_results_.clear();
  }

  // Dispatch partitions
  for (const auto& slice : plan.slices) {
    if (slice.worker_idx == -1) {
      // Local partition: execute directly
      dim3 gd(slice.partition_grid_x, grid_dim[1], grid_dim[2]);
      dim3 bd(block_dim[0], block_dim[1], block_dim[2]);

      dim3 gridOffset(slice.grid_start[0], slice.grid_start[1], slice.grid_start[2]);
      auto start = std::chrono::steady_clock::now();
      auto kr = core::RuntimeEngine::instance().launchKernel(
          kernel_id, gd, bd, args, shared_mem, 0, gridOffset);
      auto end = std::chrono::steady_clock::now();
      double execMs = std::chrono::duration<double, std::milli>(end - start).count();

      std::lock_guard<std::mutex> lock(parent_->partition_mutex_);
      PartitionResult pr;
      pr.partition_id = slice.partition_id;
      pr.kernel_id = kernel_id;
      pr.result = kr;
      pr.execution_time_ms = execMs;
      partition_results_.push_back(pr);
      parent_->partition_cv_.notify_all();
    } else {
      // Remote partition: dispatch via TCP
      std::lock_guard<std::recursive_mutex> lock(parent_->clients_mutex_);
      if (slice.worker_idx >= static_cast<int>(parent_->clients_.size()) ||
          !parent_->clients_[slice.worker_idx] || 
          !parent_->clients_[slice.worker_idx]->active) {
        continue;
      }
      
      if (parent_->clients_[slice.worker_idx]->in_flight_kernels >= kMaxInFlight) {
          VGRE_LOG_WARN("TCPCluster", "Congestion: Partition dispatch skipped for worker " +
                        std::to_string(slice.worker_idx) +
                        " — inserting ERR_BUSY so collectPartitionResults() does not deadlock");
          std::lock_guard<std::mutex> lock(parent_->partition_mutex_);
          PartitionResult pr;
          pr.partition_id      = slice.partition_id;
          pr.kernel_id         = kernel_id;
          pr.result            = VGREResult::ERR_BUSY;
          pr.execution_time_ms = 0.0;
          partition_results_.push_back(pr);
          parent_->partition_cv_.notify_all();
          continue;
      }

      // Stream arguments first
      VGREResult argResult = parent_->streamArgumentsToWorker(
          args, num_args, kernel_id, parent_->clients_[slice.worker_idx]);
      if (argResult != VGREResult::SUCCESS) {
          VGRE_LOG_ERROR("TCPCluster", "Failed to stream arguments for partition " +
                         std::to_string(slice.partition_id) +
                         " (error=" + std::to_string(static_cast<int>(argResult)) +
                         ") — recording real error so collectPartitionResults() reports it");
          std::lock_guard<std::mutex> lock(parent_->partition_mutex_);
          PartitionResult pr;
          pr.partition_id      = slice.partition_id;
          pr.kernel_id         = kernel_id;
          pr.result            = argResult;  // preserve real error, not generic ERR_INVALID_VALUE
          pr.execution_time_ms = 0.0;
          partition_results_.push_back(pr);
          parent_->partition_cv_.notify_all();
          continue;
      }

      // Send partition dispatch packet
      PartitionDispatchPacket pdpkt{};
      pdpkt.auth_token = parent_->auth_token_;
      pdpkt.kernel_id = kernel_id;
      std::memcpy(pdpkt.full_grid_dim, grid_dim, sizeof(pdpkt.full_grid_dim));
      std::memcpy(pdpkt.partition_grid_dim, slice.partition_dim, sizeof(pdpkt.partition_grid_dim));
      std::memcpy(pdpkt.block_dim, block_dim, sizeof(pdpkt.block_dim));
      std::memcpy(pdpkt.grid_start, slice.grid_start, sizeof(pdpkt.grid_start));
      pdpkt.shared_mem = shared_mem;
      pdpkt.num_args = num_args;
      pdpkt.partition_id = slice.partition_id;
      pdpkt.total_partitions = plan.total_partitions;

      VGREResult dispatch_res = parent_->send_packet(
          parent_->clients_[slice.worker_idx]->socket_fd,
          PacketType::PARTITION_DISPATCH,
          &pdpkt, sizeof(PartitionDispatchPacket),
          parent_->clients_[slice.worker_idx]->secureChannel.get());
      if (dispatch_res != VGREResult::SUCCESS) {
          VGRE_LOG_ERROR("TCPCluster",
              "Partition " + std::to_string(slice.partition_id) +
              " send failed for worker " + std::to_string(slice.worker_idx) +
              " — inserting synthetic failure so collectPartitionResults() does not deadlock");
          std::lock_guard<std::mutex> lock(parent_->partition_mutex_);
          PartitionResult pr;
          pr.partition_id = slice.partition_id;
          pr.kernel_id    = kernel_id;
          pr.result       = VGREResult::ERR_IO;
          pr.execution_time_ms = 0.0;
          partition_results_.push_back(pr);
          parent_->partition_cv_.notify_all();
          continue;
      }
      parent_->clients_[slice.worker_idx]->in_flight_kernels++;
    }
  }

  VGRE_LOG_INFO("TCPCluster",
                "Dispatched " + std::to_string(plan.total_partitions) +
                    " partitions for kernel " + std::to_string(kernel_id));
  return VGREResult::SUCCESS;
}

VGREResult DispatchManager::collectPartitionResults(
    uint64_t kernel_id, uint32_t total_partitions, int timeout_ms) {
  (void)kernel_id; // Used for logging
  auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);

  // Results are cleared in dispatchPartitionedKernel() immediately before the
  // partition packets are sent — that is the only safe clear point.  A clear
  // here would race with results that arrived between dispatch and this call.
  std::unique_lock<std::mutex> lock(parent_->partition_mutex_);
  while (partition_results_.size() < total_partitions) {
    if (parent_->partition_cv_.wait_until(lock, deadline) == std::cv_status::timeout) {
      VGRE_LOG_ERROR("TCPCluster",
                     "Partition result collection timed out: received " +
                         std::to_string(partition_results_.size()) + "/" +
                         std::to_string(total_partitions));
      return VGREResult::ERR_TIMEOUT;
    }
  }

  // Check all results
  for (const auto& pr : partition_results_) {
    if (pr.result != VGREResult::SUCCESS) {
      VGRE_LOG_ERROR("TCPCluster",
                     "Partition " + std::to_string(pr.partition_id) +
                         " failed with code " +
                         std::to_string(static_cast<int>(pr.result)));
      return pr.result;
    }
  }

  VGRE_LOG_INFO("TCPCluster",
                "All " + std::to_string(total_partitions) +
                    " partitions completed successfully");
  return VGREResult::SUCCESS;
}

VGREResult DispatchManager::waitForRemoteResult(uint64_t kernel_id, int timeout_ms) {
    auto deadline = std::chrono::steady_clock::now() + 
                    std::chrono::milliseconds(timeout_ms);
    std::unique_lock<std::mutex> lock(parent_->remote_results_mutex_);
    
    while (remote_kernel_results_.find(kernel_id) == remote_kernel_results_.end()) {
        if (parent_->remote_results_cv_.wait_until(lock, deadline) == std::cv_status::timeout) {
            VGRE_LOG_ERROR("TCPCluster", "Wait for remote result (kernel=" +
                           std::to_string(kernel_id) + ") timed out");
            return VGREResult::ERR_TIMEOUT;
        }
    }

    VGREResult r = remote_kernel_results_[kernel_id].first;
    // C1: Erase after reading so re-launched kernels don't get the stale result.
    remote_kernel_results_.erase(kernel_id);
    return r;
}

// handlePartitionDispatch is defined in dispatch_impl.cpp.

void DispatchManager::storeRemoteResult(uint64_t kernel_id, VGREResult result) {
  std::lock_guard<std::mutex> lock(parent_->remote_results_mutex_);

  // Purge results that were stored more than 60 s ago and never claimed.
  // This prevents unbounded map growth when callers abandon (e.g. timed out
  // waitForRemoteResult but the worker replied late).
  auto now = std::chrono::steady_clock::now();
  for (auto it = remote_kernel_results_.begin(); it != remote_kernel_results_.end(); ) {
    if (now - it->second.second > std::chrono::seconds(60)) {
      VGRE_LOG_WARN("TCPCluster",
          "Purging stale remote result for kernel=" + std::to_string(it->first) +
          " (unclaimed for >60s)");
      it = remote_kernel_results_.erase(it);
    } else {
      ++it;
    }
  }

  remote_kernel_results_[kernel_id] = {result, now};
  parent_->remote_results_cv_.notify_all();
}

void DispatchManager::storePartitionResult(
    uint32_t partition_id, uint64_t kernel_id,
    VGREResult result, double execution_time_ms) {
  std::lock_guard<std::mutex> lock(parent_->partition_mutex_);
  PartitionResult pr;
  pr.partition_id = partition_id;
  pr.kernel_id = kernel_id;
  pr.result = result;
  pr.execution_time_ms = execution_time_ms;
  partition_results_.push_back(pr);
  parent_->partition_cv_.notify_all();
}

} // namespace advanced
} // namespace vgre
