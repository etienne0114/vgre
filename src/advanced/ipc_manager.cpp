#include "vgre/advanced/ipc_manager.h"
#include "vgre/advanced/tcp_cluster.h"
#include "vgre/common/logger.h"
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

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
#if defined(_WIN32)
    if (state_)
      UnmapViewOfFile(state_);
    if (shm_fd_)
      CloseHandle(shm_fd_);
    shm_fd_ = nullptr;
#else
    if (state_)
      munmap(state_, sizeof(GlobalState));
    if (shm_fd_ != -1)
      close(shm_fd_);
    shm_fd_ = -1;
#endif
    enabled_ = false;
    state_ = nullptr;
  }

  isMaster_ = isMaster;

#if defined(_WIN32)
  std::string shmName = "Local\\VGRE_GLOBAL_STATE";
  shm_fd_ = CreateFileMappingA(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0,
                               sizeof(GlobalState), shmName.c_str());
  if (!shm_fd_) {
    VGRE_LOG_ERROR("IPCManager", "Failed to open shared memory: " + shmName);
    return false;
  }

  state_ = (GlobalState *)MapViewOfFile(shm_fd_, FILE_MAP_ALL_ACCESS, 0, 0,
                                        sizeof(GlobalState));
  if (!state_) {
    VGRE_LOG_ERROR("IPCManager", "Failed to map shared memory");
    CloseHandle(shm_fd_);
    shm_fd_ = nullptr;
    return false;
  }
#else
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
#endif

  if (isMaster_) {
    // Master initializes the state
    std::memset(state_, 0, sizeof(GlobalState));
    state_->master_pid = (int32_t)getpid();
    VGRE_LOG_INFO("IPCManager",
                  "VGRE Global Service Initialized (Master Mode)");
  }

  // Register local process
#if defined(_WIN32)
  int32_t myPid = (int32_t)GetCurrentProcessId();
#else
  int32_t myPid = (int32_t)getpid();
#endif

  for (int i = 0; i < VGRE_MAX_PROCESSES; ++i) {
    // Atomic slot claim
    if (state_->slots[i].pid == myPid) {
      // Re-claiming our own slot
      state_->slots[i].active = true;
      local_slot_ = i;
      break;
    }

    // Attempt to claim an empty slot (pid == 0)
    int32_t expected = 0;
#if defined(_WIN32)
    if (_InterlockedCompareExchange((volatile long *)&state_->slots[i].pid,
                                    myPid, expected) == expected) {
      state_->slots[i].active = true;
      local_slot_ = i;
      break;
    }
#else
    if (__atomic_compare_exchange_n(&state_->slots[i].pid, &expected, myPid,
                                    false, __ATOMIC_SEQ_CST,
                                    __ATOMIC_SEQ_CST)) {
      state_->slots[i].active = true;
      local_slot_ = i;
      break;
    }
#endif
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
#if defined(_WIN32)
    UnmapViewOfFile(state_);
#else
    munmap(state_, sizeof(GlobalState));
#endif
    state_ = nullptr;
  }

#if defined(_WIN32)
  if (shm_fd_) {
    CloseHandle(shm_fd_);
    shm_fd_ = nullptr;
  }
#else
  if (shm_fd_ != -1) {
    close(shm_fd_);
    shm_fd_ = -1;
  }

  if (isMaster_) {
    shm_unlink(VGRE_SHM_NAME);
  }
#endif

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
    outCombined.background_compute_active =
        state_->slots[local_slot_].telemetry.background_compute_active;
    outCombined.ecc_enabled = state_->slots[local_slot_].telemetry.ecc_enabled;
  }
}

} // namespace advanced
} // namespace vgre
