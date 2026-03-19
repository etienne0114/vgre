#include "vgre/advanced/ipc_manager.h"
#include "vgre/advanced/tcp_cluster.h"
#include "vgre/common/logger.h"
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <signal.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace vgre {
namespace advanced {

namespace {
static bool processIsAlive(int32_t pid) {
  if (pid <= 0) {
    return false;
  }
#if defined(_WIN32)
  HANDLE process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE,
                               static_cast<DWORD>(pid));
  if (!process) {
    return false;
  }
  DWORD exitCode = 0;
  bool alive = (GetExitCodeProcess(process, &exitCode) != 0) &&
               (exitCode == STILL_ACTIVE);
  CloseHandle(process);
  return alive;
#else
  return kill(static_cast<pid_t>(pid), 0) == 0 || errno == EPERM;
#endif
}
} // namespace

IPCManager &IPCManager::instance() {
  static IPCManager manager;
  return manager;
}

IPCManager::IPCManager() = default;

IPCManager::~IPCManager() { shutdown(); }

bool IPCManager::initialize(bool isMaster) {
  std::unique_lock<std::mutex> lock(mutex_);
  auto logInitFailure = [isMaster](const std::string &msg) {
    if (isMaster) {
      VGRE_LOG_ERROR("IPCManager", msg);
    } else {
      VGRE_LOG_WARN("IPCManager", msg + " (IPC client mode is optional)");
    }
  };

  // If we're already enabled as master, or enabled as client and only client is
  // requested, skip.
  if (enabled_ && (isMaster_ || !isMaster))
    return true;

  // If we are upgrading from client to master, we must shutdown first.
  if (enabled_ && isMaster && !isMaster_) {
    VGRE_LOG_DEBUG("IPCManager", "Upgrading from Client to Master mode");
    
    // Reset TCP cluster if it was active as a client
    TCPClusterManager::instance().shutdown();

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
    logInitFailure("Failed to open shared memory: " + shmName);
    return false;
  }

  state_ = (GlobalState *)MapViewOfFile(shm_fd_, FILE_MAP_ALL_ACCESS, 0, 0,
                                        sizeof(GlobalState));
  if (!state_) {
    logInitFailure("Failed to map shared memory");
    CloseHandle(shm_fd_);
    shm_fd_ = nullptr;
    return false;
  }
#else
  auto cleanupPosix = [this]() {
    if (state_ && state_ != MAP_FAILED) {
      munmap(state_, sizeof(GlobalState));
      state_ = nullptr;
    }
    if (shm_fd_ != -1) {
      close(shm_fd_);
      shm_fd_ = -1;
    }
  };

  // Restrict shared memory to owner-only: prevents unauthorized processes
  // from reading or corrupting VGRE global state.
  shm_fd_ = shm_open(VGRE_SHM_NAME, O_CREAT | O_RDWR, 0600);
  if (shm_fd_ == -1) {
    logInitFailure("Failed to open shared memory: " +
                   std::string(VGRE_SHM_NAME));
    return false;
  }

  struct stat st;
  if (fstat(shm_fd_, &st) != 0) {
    logInitFailure("Failed to stat shared memory");
    cleanupPosix();
    return false;
  }
  if (st.st_size < static_cast<off_t>(sizeof(GlobalState))) {
    if (ftruncate(shm_fd_, sizeof(GlobalState)) == -1) {
      logInitFailure("Failed to ftruncate shared memory");
      cleanupPosix();
      return false;
    }
  }

  state_ = (GlobalState *)mmap(NULL, sizeof(GlobalState),
                               PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_, 0);
  if (state_ == MAP_FAILED) {
    logInitFailure("Failed to mmap shared memory");
    state_ = nullptr;
    cleanupPosix();
    return false;
  }
#endif

  if (isMaster_) {
    // Master initializes the state
    std::memset(state_, 0, sizeof(GlobalState));
#if defined(_WIN32)
    state_->master_pid = static_cast<int32_t>(GetCurrentProcessId());
#else
    state_->master_pid = (int32_t)getpid();
#endif
    VGRE_LOG_INFO("IPCManager",
                  "VGRE Global Service Initialized (Master Mode)");
  }

  // Register local process
#if defined(_WIN32)
  int32_t myPid = (int32_t)GetCurrentProcessId();
#else
  int32_t myPid = (int32_t)getpid();
#endif
  local_slot_ = -1;

  // Reclaim stale slots left by dead processes.
  for (int i = 0; i < VGRE_MAX_PROCESSES; ++i) {
    int32_t pid = state_->slots[i].pid;
    if (pid != 0 && !processIsAlive(pid)) {
      state_->slots[i].active = false;
      state_->slots[i].pid = 0;
      std::memset(&state_->slots[i].telemetry, 0, sizeof(vgre_telemetry_t));
    }
  }

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
#if defined(_WIN32)
    if (state_) {
      UnmapViewOfFile(state_);
      state_ = nullptr;
    }
    if (shm_fd_) {
      CloseHandle(shm_fd_);
      shm_fd_ = nullptr;
    }
#else
    if (state_) {
      munmap(state_, sizeof(GlobalState));
      state_ = nullptr;
    }
    if (shm_fd_ != -1) {
      close(shm_fd_);
      shm_fd_ = -1;
    }
#endif
    return false;
  }

  int activeCount = 0;
  for (int i = 0; i < VGRE_MAX_PROCESSES; ++i) {
    if (state_->slots[i].active) {
      ++activeCount;
    }
  }
  state_->active_count = activeCount;

  enabled_ = true;
  VGRE_LOG_INFO("IPCManager",
                "Process registered in slot " + std::to_string(local_slot_));
  const bool startAsMaster = isMaster_;
  lock.unlock();

  auto clusterRes = TCPClusterManager::instance().initialize(startAsMaster);
  if (clusterRes != VGREResult::SUCCESS) {
    if (startAsMaster) {
      VGRE_LOG_ERROR("IPCManager", "TCP cluster initialization failed");
      shutdown();
      return false;
    }
    VGRE_LOG_DEBUG("IPCManager",
                   "TCP cluster client initialization failed (optional)");
  }
  return true;
}

void IPCManager::shutdown() {
  std::unique_lock<std::mutex> lock(mutex_);
  if (!enabled_)
    return;

  if (state_ && local_slot_ != -1) {
    state_->slots[local_slot_].active = false;
    state_->slots[local_slot_].pid = 0;
    if (state_->active_count > 0) {
      --state_->active_count;
    }
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

  enabled_ = false;
  local_slot_ = -1;
  lock.unlock();

  TCPClusterManager::instance().shutdown();
  VGRE_LOG_INFO("IPCManager", "IPC Shutdown complete");
}

void IPCManager::updateLocalTelemetry(const vgre_telemetry_t &telemetry) {
  bool shouldBroadcast = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!enabled_ || !state_ || local_slot_ == -1)
      return;
    state_->slots[local_slot_].telemetry = telemetry;
    shouldBroadcast = true;
  }
  if (shouldBroadcast) {
    TCPClusterManager::instance().broadcastLocalTelemetry(telemetry);
  }
}

void IPCManager::getGlobalTelemetry(vgre_telemetry_t &outCombined) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!enabled_ || !state_)
      return;

    // Use master metadata or first slot as base for non-additive fields
    // (We do NOT memset to 0 because outCombined might already contain 
    // local hardware stats from vgre_get_telemetry)
    
    // Sum performance metrics from ALL active processes
    double total_gflops = 0;
    double total_bw = 0;
    uint64_t total_used_mem = 0;
    int64_t total_kernels = 0;
    int64_t total_threads = 0;

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

    // Aggregate compute utilization
    if (outCombined.max_gflops > 0) {
      outCombined.compute_utilization = (outCombined.gflops / outCombined.max_gflops) * 100.0;
    }
  }

  // Merge remote node telemetry when this process acts as cluster master.
  TCPClusterManager::instance().aggregateRemoteTelemetry(outCombined);
}

} // namespace advanced
} // namespace vgre
