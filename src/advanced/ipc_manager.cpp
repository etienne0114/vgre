#include "vgre/advanced/ipc_manager.h"
#include "vgre/advanced/tcp_cluster.h"
#include "vgre/common/logger.h"
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace vgre {
namespace advanced {

IPCManager &IPCManager::instance() {
  static IPCManager manager;
  return manager;
}

IPCManager::IPCManager() = default;

IPCManager::~IPCManager() { shutdown(); }

bool IPCManager::initialize(bool isMaster) {
  std::lock_guard<std::mutex> lock(mutex_);

  // If we're already enabled as master, or enabled as client and only client is
  // requested, skip.
  if (enabled_ && (isMaster_ || !isMaster))
    return true;

  // If we are upgrading from client to master, we must shutdown first.
  if (enabled_ && isMaster && !isMaster_) {
    VGRE_LOG_DEBUG("IPCManager", "Upgrading from Client to Master mode");
    // Note: shutdown() also takes the mutex, so we shouldn't call it here
    // directly. We inline the necessary cleanup.
    if (state_ && local_slot_ != -1) {
      state_->slots[local_slot_].active = false;
    }
    if (state_)
      munmap(state_, sizeof(GlobalState));
    if (shm_fd_ != -1)
      close(shm_fd_);
    enabled_ = false;
    state_ = nullptr;
    shm_fd_ = -1;
  }

  isMaster_ = isMaster;
  shm_fd_ = shm_open(VGRE_SHM_NAME, O_CREAT | O_RDWR, 0666);
  if (shm_fd_ == -1) {
    VGRE_LOG_ERROR("IPCManager", "Failed to open shared memory: " +
                                     std::string(VGRE_SHM_NAME));
    return false;
  }

  struct stat st;
  if (fstat(shm_fd_, &st) == 0) {
    if (st.st_size < static_cast<off_t>(sizeof(GlobalState))) {
      if (ftruncate(shm_fd_, sizeof(GlobalState)) == -1) {
        VGRE_LOG_ERROR("IPCManager", "Failed to ftruncate shared memory");
        return false;
      }
    }
  }

  state_ = (GlobalState *)mmap(NULL, sizeof(GlobalState),
                               PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_, 0);
  if (state_ == MAP_FAILED) {
    VGRE_LOG_ERROR("IPCManager", "Failed to mmap shared memory");
    return false;
  }

  if (isMaster_) {
    // Master initializes the state
    std::memset(state_, 0, sizeof(GlobalState));
    state_->master_pid = (int32_t)getpid();
    VGRE_LOG_INFO("IPCManager",
                  "VGRE Global Service Initialized (Master Mode)");
  }

  // Register local process
  int32_t myPid = (int32_t)getpid();
  for (int i = 0; i < VGRE_MAX_PROCESSES; ++i) {
    // Atomic-like claim: if slot is inactive or matching our PID (re-init)
    if (!state_->slots[i].active || state_->slots[i].pid == myPid) {
      state_->slots[i].pid = myPid;
      state_->slots[i].active = true;
      local_slot_ = i;
      break;
    }
  }

  if (local_slot_ == -1) {
    VGRE_LOG_ERROR("IPCManager",
                   "No available slots in shared memory registry");
    return false;
  }

  enabled_ = true;
  VGRE_LOG_INFO("IPCManager",
                "Process registered in slot " + std::to_string(local_slot_));
  TCPClusterManager::instance().initialize(isMaster);
  return true;
}

void IPCManager::shutdown() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!enabled_)
    return;

  if (state_ && local_slot_ != -1) {
    state_->slots[local_slot_].active = false;
    state_->slots[local_slot_].pid = 0;
  }

  if (state_) {
    munmap(state_, sizeof(GlobalState));
    state_ = nullptr;
  }

  if (shm_fd_ != -1) {
    close(shm_fd_);
    shm_fd_ = -1;
  }

  if (isMaster_) {
    shm_unlink(VGRE_SHM_NAME);
  }

  TCPClusterManager::instance().shutdown();

  enabled_ = false;
  VGRE_LOG_INFO("IPCManager", "IPC Shutdown complete");
}

void IPCManager::updateLocalTelemetry(const vgre_telemetry_t &telemetry) {
  if (!enabled_ || !state_ || local_slot_ == -1)
    return;
  state_->slots[local_slot_].telemetry = telemetry;

  TCPClusterManager::instance().broadcastLocalTelemetry(telemetry);
}

void IPCManager::getGlobalTelemetry(vgre_telemetry_t &outCombined) {
  if (!enabled_ || !state_)
    return;

  // Aggregation logic
  std::memset(&outCombined, 0, sizeof(vgre_telemetry_t));

  // Use master metadata or first slot
  if (local_slot_ != -1) {
    outCombined = state_->slots[local_slot_].telemetry;
  }

  // sum performance metrics from ALL active processes
  float total_gflops = 0;
  float total_bw = 0;
  size_t total_used_mem = 0;
  int total_kernels = 0;
  int total_threads = 0;

  for (int i = 0; i < VGRE_MAX_PROCESSES; ++i) {
    if (state_->slots[i].active) {
      total_gflops += state_->slots[i].telemetry.gflops;
      total_bw += state_->slots[i].telemetry.memory_bandwidth_gbps;
      total_used_mem += state_->slots[i].telemetry.memory_used_bytes;
      total_kernels += state_->slots[i].telemetry.active_kernels;
      total_threads += state_->slots[i].telemetry.active_threads;
    }
  }

  outCombined.gflops = total_gflops;
  outCombined.memory_bandwidth_gbps = total_bw;
  outCombined.memory_used_bytes = total_used_mem;
  outCombined.active_kernels = total_kernels;
  outCombined.active_threads = total_threads;

  // Preserve essential state from the local slot
  if (local_slot_ != -1) {
    outCombined.simulation_enabled =
        state_->slots[local_slot_].telemetry.simulation_enabled;
    outCombined.ecc_enabled = state_->slots[local_slot_].telemetry.ecc_enabled;
  }
}

} // namespace advanced
} // namespace vgre
