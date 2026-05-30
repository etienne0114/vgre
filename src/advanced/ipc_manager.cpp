#include "vgre/advanced/ipc_manager.h"
#include "vgre/advanced/secure_channel.h"
#include "vgre/advanced/tcp_cluster.h"
#include "vgre/api/vgre_c_api.h"
#include "vgre/common/logger.h"
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <random>
#include <chrono>
#if !defined(_WIN32)
#include <sys/stat.h>
#include <sys/random.h>   // getrandom()
#endif
#include <thread>

#include "vgre/common/os_backend.h"

// Platform-specific crypto includes
#if defined(_WIN32)
#include <windows.h>
#include <wincrypt.h>
#elif defined(__APPLE__)
#include <CommonCrypto/CommonHMAC.h>
#include <CommonCrypto/CommonDigest.h>
#else
#include <openssl/hmac.h>
#include <openssl/sha.h>
#endif

// Platform-specific shared memory includes
#if defined(__APPLE__)
#include <sys/stat.h>
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

// Generate HMAC-SHA256 for authentication using VGRE's internal crypto.
// This avoids a dependency on OpenSSL while using the same HMAC-SHA256
// implementation already present in secure_channel_crypto.cpp.
static void generateHMAC(const uint8_t* data, size_t len, const std::string& secret, uint8_t* hmac) {
    vgre::advanced::crypto::hmac_sha256(
        reinterpret_cast<const uint8_t*>(secret.data()), secret.size(),
        data, len, hmac);
}

// Generate random token (platform-specific)
static void generateRandomToken(uint8_t* token, size_t size) {
#if defined(_WIN32)
    HCRYPTPROV hProv = 0;
    if (CryptAcquireContext(&hProv, NULL, NULL, PROV_RSA_FULL, CRYPT_VERIFYCONTEXT)) {
        CryptGenRandom(hProv, size, token);
        CryptReleaseContext(hProv, 0);
    } else {
        // Fallback to std::random_device
        std::random_device rd;
        std::mt19937_64 gen(rd());
        std::uniform_int_distribution<uint8_t> dist(0, 255);
        for (size_t i = 0; i < size; ++i) {
            token[i] = dist(gen);
        }
    }
#elif defined(__APPLE__)
    // macOS: Use arc4random_buf
    arc4random_buf(token, size);
#else
    // Linux: Use getrandom if available, otherwise std::random_device
    #if defined(__linux__)
    ssize_t ret = getrandom(token, size, 0);
    if (ret == static_cast<ssize_t>(size)) {
        return;
    }
    #endif
    // Fallback
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint8_t> dist(0, 255);
    for (size_t i = 0; i < size; ++i) {
        token[i] = dist(gen);
    }
#endif
}
} // namespace

IPCManager &IPCManager::instance() {
  // D5: Meyers singleton — destructor is called at static-storage teardown,
  // so IPCManager::shutdown() runs on process exit and SHM/handles are released.
  // The previous `new IPCManager()` leaked the object and prevented ASAN/valgrind
  // from reporting clean exits.
  static IPCManager inst;
  return inst;
}

IPCManager::IPCManager() {
    // Initialize authentication secret from environment or generate random
    const char* secret = vgre_get_config("VGRE_IPC_SECRET");
    if (secret) {
        auth_secret_ = secret;
    } else {
        // Generate random secret if not provided
        uint8_t random_bytes[32];
        generateRandomToken(random_bytes, 32);
        auth_secret_.assign(reinterpret_cast<char*>(random_bytes), 32);
    }
    
    // Generate local authentication token
    generateAuthToken(local_auth_, 0);
}

IPCManager::~IPCManager() {
  // Skip shutdown during static destruction to prevent deadlock
  // In test environments, the destruction order of singletons can cause hangs
  // The OS will reclaim resources when the process exits
  if (!enabled_) return;
  
  // Try shutdown with timeout
  auto future = std::async(std::launch::async, [this]() { shutdown(); });
  if (future.wait_for(std::chrono::seconds(5)) == std::future_status::timeout) {
    // Timeout - skip shutdown to prevent hang
    VGRE_LOG_WARN("IPCManager", "Shutdown timeout during destruction - skipping");
  }
}

bool IPCManager::initialize(bool isMaster) {
  std::unique_lock<std::recursive_mutex> lock(mutex_);
  auto logInitFailure = [isMaster](const std::string &msg) {
    if (isMaster) {
      VGRE_LOG_ERROR("IPCManager", msg);
    } else {
      VGRE_LOG_WARN("IPCManager", msg + " (IPC client mode is optional)");
    }
  };

  // Spin-wait if another thread is mid-upgrade (client→master teardown/reinit).
  // Releases the lock while waiting so the upgrading thread can re-acquire it.
  while (upgrading_.load(std::memory_order_acquire)) {
    lock.unlock();
    std::this_thread::yield();
    lock.lock();
  }

  // If we're already enabled as master, or enabled as client and only client is
  // requested, skip.
  if (enabled_ && (isMaster_ || !isMaster))
    return true;

  // If we are upgrading from client to master, we must shutdown first.
  if (enabled_ && isMaster && !isMaster_) {
    VGRE_LOG_DEBUG("IPCManager", "Upgrading from Client to Master mode");

    // Mark upgrade in progress so concurrent callers spin-wait rather than
    // entering initialize() with a partially-torn-down state_.
    upgrading_.store(true, std::memory_order_release);

    // Release the mutex before calling TCPCluster::shutdown().
    // TCPCluster::shutdown() joins worker threads that may call back into
    // IPCManager (e.g. updateTelemetry). Holding mutex_ while waiting for
    // those threads to exit would deadlock on this recursive_mutex.
    lock.unlock();
    TCPClusterManager::instance().shutdown();
    lock.lock();

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
    // upgrading_ cleared after the new SHM is fully initialized below.
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

  // Phase 10: Dynamic SHM name for test isolation
  // macOS requires shared memory names to start with '/' for shm_open
  std::string fullShmName = VGRE_SHM_NAME;
#if defined(__APPLE__)
  if (fullShmName[0] != '/') {
      fullShmName = "/" + fullShmName;
  }
#endif
  const char* shmSuffix = vgre_get_config("VGRE_SHM_SUFFIX");
  if (shmSuffix) {
      fullShmName += "_" + std::string(shmSuffix);
  }

  // Restrict shared memory to owner-only: prevents unauthorized processes
  // from reading or corrupting VGRE global state.
  // On macOS, ensure the shared memory is created with proper permissions
  int shm_flags = O_CREAT | O_RDWR;
#if defined(__APPLE__)
  // macOS: Try to open existing first, then create if it doesn't exist
  shm_fd_ = shm_open(fullShmName.c_str(), O_RDWR, 0600);
  if (shm_fd_ == -1 && errno == ENOENT) {
      shm_fd_ = shm_open(fullShmName.c_str(), shm_flags, 0600);
  }
#else
  shm_fd_ = shm_open(fullShmName.c_str(), shm_flags, 0600);
#endif
  if (shm_fd_ == -1) {
    logInitFailure("Failed to open shared memory: " + fullShmName);
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
    memset(state_, 0, sizeof(GlobalState));
#if defined(_WIN32)
    state_->master_pid = static_cast<int32_t>(GetCurrentProcessId());
#else
    state_->master_pid = (int32_t)getpid();
#endif
    
    // Generate master authentication token
    generateAuthToken(state_->master_auth, state_->master_pid);
    state_->state_version = 1;
    updateStateHMAC();
    
    VGRE_LOG_INFO("IPCManager",
                  "VGRE Global Service Initialized (Master Mode with IPC hardening)");
  }

  // Register local process
#if defined(_WIN32)
  int32_t myPid = (int32_t)GetCurrentProcessId();
#else
  int32_t myPid = (int32_t)getpid();
#endif
  local_slot_ = -1;
  
  // Update local auth token with actual PID
  generateAuthToken(local_auth_, myPid);

  // Verify state integrity before accessing
  if (!verifyStateIntegrity()) {
    VGRE_LOG_ERROR("IPCManager", "State integrity check failed - possible hijacking attempt");
    cleanupPosix();
    return false;
  }
  
  // Reclaim stale slots left by dead processes.
  for (int i = 0; i < VGRE_MAX_PROCESSES; ++i) {
    int32_t pid = state_->slots[i].pid;
    if (pid != 0 && !processIsAlive(pid)) {
      state_->slots[i].active = false;
      state_->slots[i].pid = 0;
      memset(&state_->slots[i].telemetry, 0, sizeof(vgre_telemetry_t));
      memset(&state_->slots[i].auth, 0, sizeof(AuthToken));
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
      state_->slots[i].auth = local_auth_;
      state_->slots[i].last_heartbeat = std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now().time_since_epoch()).count();
      local_slot_ = i;
      break;
    }
#else
    if (__atomic_compare_exchange_n(&state_->slots[i].pid, &expected, myPid,
                                    false, __ATOMIC_SEQ_CST,
                                    __ATOMIC_SEQ_CST)) {
      state_->slots[i].active = true;
      state_->slots[i].auth = local_auth_;
      state_->slots[i].last_heartbeat = std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now().time_since_epoch()).count();
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
  
  // Update state HMAC after modifications
  if (isMaster_) {
      updateStateHMAC();
  }

  enabled_ = true;
  upgrading_.store(false, std::memory_order_release); // upgrade complete; clear the barrier
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
  std::unique_lock<std::recursive_mutex> lock(mutex_);
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
    std::string fullShmName = VGRE_SHM_NAME;
    const char* shmSuffix = vgre_get_config("VGRE_SHM_SUFFIX");
    if (shmSuffix) {
        fullShmName += "_" + std::string(shmSuffix);
    }
    shm_unlink(fullShmName.c_str());
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
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!enabled_ || !state_ || local_slot_ == -1)
      return;
    
    // Verify authentication before updating
    if (!validateAuthToken(state_->slots[local_slot_].auth)) {
        VGRE_LOG_ERROR("IPCManager", "Authentication failed - telemetry update rejected");
        return;
    }
    
    state_->slots[local_slot_].telemetry = telemetry;
    state_->slots[local_slot_].last_heartbeat = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    shouldBroadcast = true;
    
    if (isMaster_) {
        updateStateHMAC();
    }
  }
  if (shouldBroadcast) {
    TCPClusterManager::instance().broadcastLocalTelemetry(telemetry);
  }
}

void IPCManager::getGlobalTelemetry(vgre_telemetry_t &outCombined) {
  {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!enabled_ || !state_)
      return;
    
    // Verify state integrity
    if (!verifyStateIntegrity()) {
        VGRE_LOG_ERROR("IPCManager", "State integrity check failed during telemetry read");
        return;
    }

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

void IPCManager::updateClusterNodes(const std::vector<vgre_cluster_node_t>& nodes) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (!enabled_ || !state_) return;
  
  if (!verifyStateIntegrity()) {
      VGRE_LOG_ERROR("IPCManager", "State integrity check failed during cluster update");
      return;
  }
  
  uint32_t count = std::min(static_cast<uint32_t>(nodes.size()), static_cast<uint32_t>(VGRE_MAX_NODES));
  state_->cluster_node_count = count;
  for (uint32_t i = 0; i < count; ++i) {
    state_->cluster_nodes[i] = nodes[i];
  }
  
  if (isMaster_) {
      updateStateHMAC();
  }
}

void IPCManager::getClusterNodes(std::vector<vgre_cluster_node_t>& outNodes) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (!enabled_ || !state_) return;
  
  if (!verifyStateIntegrity()) {
      VGRE_LOG_ERROR("IPCManager", "State integrity check failed during cluster read");
      return;
  }
  
  outNodes.clear();
  uint32_t count = state_->cluster_node_count;
  for (uint32_t i = 0; i < count; ++i) {
    outNodes.push_back(state_->cluster_nodes[i]);
  }
}

// Generate authentication token for a process
void IPCManager::generateAuthToken(AuthToken& token, int32_t pid) {
    generateRandomToken(token.token, VGRE_AUTH_TOKEN_SIZE);
    token.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    token.pid = pid;
    
    // Generate HMAC for integrity
    generateHMAC(reinterpret_cast<uint8_t*>(&token), 
                 sizeof(AuthToken) - sizeof(token.hmac),
                 auth_secret_, token.hmac);
}

// Validate authentication token
bool IPCManager::validateAuthToken(const AuthToken& token) const {
    if (token.pid <= 0) return false;
  
    // Check if token is too old (5 minute expiry)
  auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
  if (now - token.timestamp > 300000) return false;
  
  // Verify HMAC
  uint8_t computed_hmac[32];
  generateHMAC(reinterpret_cast<const uint8_t*>(&token),
               sizeof(AuthToken) - sizeof(token.hmac),
               auth_secret_, computed_hmac);
  
  return memcmp(computed_hmac, token.hmac, 32) == 0;
}

// Verify shared memory state integrity
bool IPCManager::verifyStateIntegrity() const {
    if (!state_) return false;
  
    // Compute HMAC of state (excluding the HMAC field itself)
  size_t hmac_offset = offsetof(GlobalState, state_hmac);
  uint8_t computed_hmac[32];
  generateHMAC(reinterpret_cast<uint8_t*>(state_), hmac_offset,
               auth_secret_, computed_hmac);
  
  return memcmp(computed_hmac, state_->state_hmac, 32) == 0;
}

// Update state HMAC after modifications
void IPCManager::updateStateHMAC() {
    if (!state_) return;
  
  state_->state_version++;
  size_t hmac_offset = offsetof(GlobalState, state_hmac);
  generateHMAC(reinterpret_cast<uint8_t*>(state_), hmac_offset,
               auth_secret_, state_->state_hmac);
}

} // namespace advanced
} // namespace vgre
