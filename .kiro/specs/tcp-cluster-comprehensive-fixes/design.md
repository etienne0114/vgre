# TCP Cluster Comprehensive Fixes Design

## Overview

This design addresses all critical issues in tcp_cluster.cpp (3,267 lines) to transform it from a monolithic, bug-prone implementation into a production-ready, maintainable, and testable distributed computing system. The fixes span missing implementations, code duplication elimination, stub replacement, business logic improvements, architectural refactoring, and testability enhancements.

## Glossary

- **Bug_Condition (C)**: The set of conditions that trigger each identified bug (missing methods, duplicated code, stubs, poor heuristics, etc.)
- **Property (P)**: The desired correct behavior after fixes are applied
- **Preservation**: Existing functionality that must remain unchanged (protocol compatibility, performance characteristics, error handling)
- **SIMD**: Single Instruction Multiple Data - vectorized CPU operations for performance
- **RAII**: Resource Acquisition Is Initialization - C++ idiom for automatic resource management
- **TSS2**: Traffic-Shaping-Sync 2.0 - the priority-based packet queuing system
- **VSBP**: VGRE Structured Binary Protocol v0.1.2 - the wire protocol format
- **Delta-Sync**: Incremental memory synchronization using dirty page tracking
- **SHM**: Shared Memory - inter-process memory optimization for local connections

## Bug Details

### 1. Missing Implementations

**Bug Condition:**

```
FUNCTION isMissingImplementation(symbol)
  INPUT: symbol of type string (function name)
  OUTPUT: boolean
  
  RETURN symbol IN ['reportComputeFromWorker', 'allReduce']
         AND symbol is declared in header
         AND symbol has no definition in implementation file
END FUNCTION
```

**Examples:**
- `reportComputeFromWorker(double seconds, int cores, uint64_t kernel_id)` - declared in header, no implementation → linker error
- `allReduce(void* ptr, size_t count, int datatype)` - declared in header, no implementation → linker error

### 2. Code Duplication

**Bug Condition:**

```
FUNCTION isCodeDuplication(codeBlock)
  INPUT: codeBlock of type CodeFragment
  OUTPUT: boolean
  
  RETURN (codeBlock appears in multiple locations
         AND codeBlock performs identical logic
         AND codeBlock is not abstracted into shared function)
END FUNCTION
```

**Examples:**
- Packet construction logic duplicated in `send_packet()` and `send_packet_direct()`
- Delta-sync detection duplicated in `launchRemoteKernel()` and `launchPartitionedKernel()`
- Argument serialization duplicated across dispatch methods
- SHM fallback `goto` patterns duplicated 4+ times

### 3. Stub/Mock Implementations

**Bug Condition:**

```
FUNCTION isStubImplementation(function)
  INPUT: function of type FunctionDefinition
  OUTPUT: boolean
  
  RETURN (function.body is trivial OR function.body is placeholder)
         AND function is marked for production use
         AND function lacks optimization or proper implementation
END FUNCTION
```

**Examples:**
- `sum_reduce<T>()` uses simple loop instead of SIMD-optimized reduction
- Security handshake failure silently accepts plaintext without policy enforcement


### 4. Poor Business Logic/Heuristics

**Bug Condition:**

```
FUNCTION hasPoorBusinessLogic(code)
  INPUT: code of type CodeFragment
  OUTPUT: boolean
  
  RETURN code contains magic numbers
         OR code has race conditions
         OR code leaks resources
         OR code has inconsistent error handling
         OR code uses inefficient algorithms
END FUNCTION
```

**Examples:**
- Magic number 5 seconds for timeout without named constant
- Magic numbers 0/1 for priority without enum
- Magic number 16 for MAX_IN_FLIGHT without configuration
- Magic number 128MB for SHM offset without explanation
- TOCTOU race in duplicate connection detection
- UDP socket leaks on error paths
- Thread leaks when shutdown called before join
- Busy-wait consuming CPU instead of blocking I/O

## Expected Behavior

### 1. Complete Implementations

**reportComputeFromWorker Implementation:**
```cpp
void TCPClusterManager::reportComputeFromWorker(double seconds, int cores, uint64_t kernel_id) {
  if (!enabled_ || is_master_) return;
  
  CreditReportPacket crpkt{};
  crpkt.compute_seconds = seconds;
  crpkt.cpu_cores = cores;
  crpkt.kernel_id = kernel_id;
  crpkt.timestamp = getCurrentTimestampMs();
  
  send_packet(client_fd_, PacketType::CREDIT_REPORT, &crpkt, 
              sizeof(CreditReportPacket), client_secure_channel_.get());
}
```

**allReduce Implementation:**
```cpp
VGREResult TCPClusterManager::allReduce(void* ptr, size_t count, int datatype) {
  if (!enabled_) return VGREResult::ERR_NOT_INITIALIZED;
  
  // Worker path: send local data to master
  if (!is_master_) {
    CollectiveOpPacket cop{};
    cop.op_type = 0; // all_reduce
    cop.datatype = datatype;
    cop.count = count;
    cop.sequence = reduction_sequence_++;
    
    send_packet(client_fd_, PacketType::COLLECTIVE_OP, &cop, sizeof(cop), 
                client_secure_channel_.get());
    send_packet(client_fd_, PacketType::RAW_DATA, ptr, 
                count * getTypeSize(datatype), client_secure_channel_.get());
    
    // Wait for master to broadcast result
    std::unique_lock<std::mutex> lock(reduction_mutex_);
    reduction_cv_.wait(lock, [this]() { return !is_reducing_; });
    return VGREResult::SUCCESS;
  }
  
  // Master path: collect from all workers, reduce, broadcast
  // Implementation details in design section below
}
```


### 2. Eliminate Code Duplication

**Unified Packet Construction:**
```cpp
// Extract common packet construction logic
std::vector<uint8_t> TCPClusterManager::constructPacket(
    PacketType type, const void* payload, size_t payloadLen) {
  static std::atomic<uint32_t> s_seq{1};
  
  size_t totalLen = sizeof(VSBPHeader) + payloadLen;
  std::vector<uint8_t> staging(totalLen);
  
  VSBPHeader* header = reinterpret_cast<VSBPHeader*>(staging.data());
  header->magic = VSBP_MAGIC;
  header->version = VSBP_VERSION;
  header->type = static_cast<uint16_t>(type);
  header->sequence = s_seq.fetch_add(1, std::memory_order_relaxed);
  header->payloadSize = payloadLen;
  
  if (payload && payloadLen > 0) {
    std::memcpy(staging.data() + sizeof(VSBPHeader), payload, payloadLen);
  }
  
  return staging;
}
```

**Unified Delta-Sync Logic:**
```cpp
// Extract delta-sync detection and transmission
VGREResult TCPClusterManager::syncPointerToWorker(
    void* ptr, uint64_t handle, std::shared_ptr<ClientConnection> client) {
  
  auto& mm = core::RuntimeEngine::instance().getMemoryManager();
  size_t size = mm.getAllocationSize(ptr);
  if (size == 0 || !ptr) return VGREResult::SUCCESS;
  
  std::vector<std::pair<size_t, size_t>> dirtyRanges;
  bool useDelta = client->synced_handles.count(ptr) > 0;
  
  if (useDelta) {
    mm.getDirtyPages(ptr, dirtyRanges);
  }
  
  if (useDelta && !dirtyRanges.empty()) {
    return sendDeltaSync(ptr, handle, dirtyRanges, client);
  } else {
    client->synced_handles.insert(ptr);
    return sendFullSync(ptr, handle, size, client);
  }
}
```

**Unified Argument Serialization:**
```cpp
// Extract argument streaming logic
VGREResult TCPClusterManager::streamArgumentsToWorker(
    void** args, int num_args, uint64_t kernel_id,
    std::shared_ptr<ClientConnection> client) {
  
  std::vector<ArgType> argTypes;
  if (core::RuntimeEngine::instance().getKernelArgTypes(kernel_id, argTypes) 
      != VGREResult::SUCCESS) {
    return VGREResult::ERR_INVALID_KERNEL;
  }
  
  for (int i = 0; i < num_args; ++i) {
    ArgType type = (i < argTypes.size()) ? argTypes[i] : ArgType::UINT64;
    
    if (type == ArgType::STRUCT) {
      VGREResult r = sendStructArg(i, args[i], kernel_id, client);
      if (r != VGREResult::SUCCESS) return r;
    } else if (type == ArgType::POINTER) {
      VGREResult r = sendPointerArg(i, args[i], client);
      if (r != VGREResult::SUCCESS) return r;
    } else {
      VGREResult r = sendScalarArg(i, type, args[i], client);
      if (r != VGREResult::SUCCESS) return r;
    }
  }
  
  return VGREResult::SUCCESS;
}
```


**RAII-Based SHM Fallback:**
```cpp
// Replace goto-based fallback with RAII
class ShmFallbackGuard {
public:
  ShmFallbackGuard(std::shared_ptr<ClientConnection> client, 
                   uint64_t size_needed)
    : client_(client), succeeded_(false) {
    if (client_->is_local && client_->shmManager) {
      offset_ = client_->shm_offset;
      if (offset_ + size_needed <= client_->shmManager->getSize()) {
        client_->shm_offset += size_needed;
        succeeded_ = true;
      }
    }
  }
  
  ~ShmFallbackGuard() {
    if (!succeeded_ && client_->is_local && client_->shmManager) {
      // Rollback on failure
      client_->shm_offset = offset_;
    }
  }
  
  bool succeeded() const { return succeeded_; }
  uint64_t offset() const { return offset_; }
  
private:
  std::shared_ptr<ClientConnection> client_;
  uint64_t offset_;
  bool succeeded_;
};
```

### 3. Production-Ready Implementations

**SIMD-Optimized Reduction:**
```cpp
template <typename T>
void sum_reduce_simd(T* dst, const T* src, size_t count) {
  size_t i = 0;
  
#if defined(__AVX2__)
  // AVX2 path for 256-bit SIMD
  if constexpr (std::is_same_v<T, float>) {
    for (; i + 8 <= count; i += 8) {
      __m256 d = _mm256_loadu_ps(dst + i);
      __m256 s = _mm256_loadu_ps(src + i);
      __m256 r = _mm256_add_ps(d, s);
      _mm256_storeu_ps(dst + i, r);
    }
  } else if constexpr (std::is_same_v<T, double>) {
    for (; i + 4 <= count; i += 4) {
      __m256d d = _mm256_loadu_pd(dst + i);
      __m256d s = _mm256_loadu_pd(src + i);
      __m256d r = _mm256_add_pd(d, s);
      _mm256_storeu_pd(dst + i, r);
    }
  }
#elif defined(__SSE2__)
  // SSE2 path for 128-bit SIMD
  if constexpr (std::is_same_v<T, float>) {
    for (; i + 4 <= count; i += 4) {
      __m128 d = _mm_loadu_ps(dst + i);
      __m128 s = _mm_loadu_ps(src + i);
      __m128 r = _mm_add_ps(d, s);
      _mm_storeu_ps(dst + i, r);
    }
  }
#endif
  
  // Scalar fallback for remaining elements
  for (; i < count; ++i) {
    dst[i] += src[i];
  }
}
```

**Enforced Security Policy:**
```cpp
VGREResult TCPClusterManager::performSecureHandshake(
    std::shared_ptr<ClientConnection> clientPtr) {
  
  if (!security_enabled_) {
    return VGREResult::SUCCESS; // Security not required
  }
  
  if (auth_token_str_.empty()) {
    VGRE_LOG_ERROR("TCPCluster", 
      "Security enabled but VGRE_TCP_AUTH_TOKEN not set");
    return VGREResult::ERR_AUTH_FAILED;
  }
  
  // Perform handshake...
  VGREResult r = doHandshake(clientPtr);
  
  if (r != VGREResult::SUCCESS) {
    // ENFORCE: No plaintext fallback when security is enabled
    VGRE_LOG_ERROR("TCPCluster", 
      "Security handshake failed - closing connection (no plaintext fallback)");
    clientPtr->active = false;
    vgre_close_socket(clientPtr->socket_fd);
    return r;
  }
  
  return VGREResult::SUCCESS;
}
```


### 4. Proper Business Logic/Heuristics

**Named Constants:**
```cpp
// Replace magic numbers with named constants
namespace TCPClusterConstants {
  // Timeouts
  constexpr int SEND_TIMEOUT_SECONDS = 5;
  constexpr int HANDSHAKE_TIMEOUT_MS = 5000;
  constexpr int PEEK_TIMEOUT_MS = 200;
  constexpr int POLL_TIMEOUT_MS = 50;
  
  // Flow Control
  constexpr uint32_t MAX_IN_FLIGHT_KERNELS = 16;
  constexpr uint32_t HIGH_PRIORITY = 0;
  constexpr uint32_t LOW_PRIORITY = 1;
  
  // Memory
  constexpr uint64_t SHM_RESULT_OFFSET_BASE = 128 * 1024 * 1024; // 128MB
  constexpr size_t DEFAULT_SHM_SIZE = 256 * 1024 * 1024; // 256MB
  
  // Backoff
  constexpr int INITIAL_BACKOFF_SECONDS = 5;
  constexpr int MAX_BACKOFF_SECONDS = 300;
  constexpr int BACKOFF_MULTIPLIER = 4;
}

enum class PacketPriority : uint32_t {
  HIGH = 0,  // Control/Sync packets
  LOW = 1    // Data/Bulk packets
};
```

**Atomic Duplicate Connection Check:**
```cpp
// Fix TOCTOU race in duplicate connection detection
bool TCPClusterManager::addClientIfNotDuplicate(
    const std::string& ip, vgre_socket_t new_socket) {
  
  std::lock_guard<std::recursive_mutex> lock(clients_mutex_);
  
  // Atomic check-and-insert under single lock
  for (const auto& ec : clients_) {
    if (ec && (ec->active || ec->is_authenticating) &&
        ec->socket_fd != VGRE_INVALID_SOCKET &&
        ec->ip_address == ip) {
      // Duplicate found
      VGRE_LOG_WARN("TCPCluster",
        "Dropping duplicate connection from " + ip);
      vgre_close_socket(new_socket);
      return false;
    }
  }
  
  // No duplicate - safe to add
  auto conn = std::make_shared<ClientConnection>();
  conn->socket_fd = new_socket;
  conn->ip_address = ip;
  conn->active = true;
  clients_.push_back(std::move(conn));
  
  return true;
}
```

**RAII Socket Wrapper:**
```cpp
// Prevent socket leaks with RAII
class SocketGuard {
public:
  explicit SocketGuard(vgre_socket_t fd = VGRE_INVALID_SOCKET) 
    : fd_(fd) {}
  
  ~SocketGuard() {
    if (fd_ != VGRE_INVALID_SOCKET) {
      vgre_close_socket(fd_);
    }
  }
  
  SocketGuard(const SocketGuard&) = delete;
  SocketGuard& operator=(const SocketGuard&) = delete;
  
  SocketGuard(SocketGuard&& other) noexcept : fd_(other.fd_) {
    other.fd_ = VGRE_INVALID_SOCKET;
  }
  
  vgre_socket_t get() const { return fd_; }
  vgre_socket_t release() {
    vgre_socket_t tmp = fd_;
    fd_ = VGRE_INVALID_SOCKET;
    return tmp;
  }
  
private:
  vgre_socket_t fd_;
};
```


**RAII Thread Wrapper:**
```cpp
// Prevent thread leaks with RAII
class JoinableThread {
public:
  JoinableThread() = default;
  
  template<typename Func, typename... Args>
  explicit JoinableThread(Func&& f, Args&&... args)
    : thread_(std::forward<Func>(f), std::forward<Args>(args)...) {}
  
  ~JoinableThread() {
    if (thread_.joinable()) {
      thread_.join();
    }
  }
  
  JoinableThread(const JoinableThread&) = delete;
  JoinableThread& operator=(const JoinableThread&) = delete;
  
  JoinableThread(JoinableThread&&) = default;
  JoinableThread& operator=(JoinableThread&&) = default;
  
  bool joinable() const { return thread_.joinable(); }
  void join() { if (thread_.joinable()) thread_.join(); }
  
private:
  std::thread thread_;
};
```

**RAII SHM Wrapper:**
```cpp
// Prevent SHM leaks with RAII
class ShmSegment {
public:
  ShmSegment() = default;
  
  ~ShmSegment() {
    if (manager_) {
      manager_->close();
    }
  }
  
  VGREResult open(const std::string& name, size_t size, bool create) {
    manager_ = std::make_unique<vgre::core::ShmManager>();
    return manager_->open(name, size, create);
  }
  
  void* getBasePtr() const {
    return manager_ ? manager_->getBasePtr() : nullptr;
  }
  
  size_t getSize() const {
    return manager_ ? manager_->getSize() : 0;
  }
  
private:
  std::unique_ptr<vgre::core::ShmManager> manager_;
};
```

**Exponential Backoff:**
```cpp
// Replace hardcoded backoff with exponential strategy
class ExponentialBackoff {
public:
  ExponentialBackoff(int initial_ms, int max_ms, double multiplier = 2.0)
    : initial_ms_(initial_ms), max_ms_(max_ms), 
      multiplier_(multiplier), current_ms_(initial_ms) {}
  
  int next() {
    int result = current_ms_;
    current_ms_ = std::min(
      static_cast<int>(current_ms_ * multiplier_), 
      max_ms_
    );
    return result;
  }
  
  void reset() {
    current_ms_ = initial_ms_;
  }
  
private:
  int initial_ms_;
  int max_ms_;
  double multiplier_;
  int current_ms_;
};
```

**Consistent Error Handling:**
```cpp
// Standardize all error-prone operations to return VGREResult
VGREResult TCPClusterManager::sendPacketSafe(
    vgre_socket_t fd, PacketType type, 
    const void* payload, size_t payloadLen,
    SecureChannel* sc) {
  
  if (fd == VGRE_INVALID_SOCKET) {
    return VGREResult::ERR_INVALID_VALUE;
  }
  
  if (common::InputValidator::validatePacketSize(payloadLen) 
      != VGREResult::SUCCESS) {
    return VGREResult::ERR_INVALID_VALUE;
  }
  
  auto packet = constructPacket(type, payload, payloadLen);
  
  bool success = false;
  if (sc && sc->isInitialized()) {
    success = (sc->sendSecure(fd, packet.data(), packet.size()) 
               == VGREResult::SUCCESS);
  } else {
    success = send_all(fd, packet.data(), packet.size(), &enabled_);
  }
  
  return success ? VGREResult::SUCCESS : VGREResult::ERR_IO;
}
```


**Blocking I/O with Timeout:**
```cpp
// Replace busy-wait with proper blocking I/O
VGREResult TCPClusterManager::waitForData(
    vgre_socket_t fd, int timeout_ms) {
  
  vgre_pollfd pfd;
  pfd.fd = fd;
  pfd.events = POLLIN;
  pfd.revents = 0;
  
  int result = vgre_poll(&pfd, 1, timeout_ms);
  
  if (result < 0) {
    if (!vgre_is_would_block(vgre_get_last_socket_error())) {
      return VGREResult::ERR_IO;
    }
    return VGREResult::ERR_TIMEOUT;
  }
  
  if (result == 0) {
    return VGREResult::ERR_TIMEOUT;
  }
  
  if (pfd.revents & (POLLERR | POLLHUP)) {
    return VGREResult::ERR_IO;
  }
  
  return VGREResult::SUCCESS;
}
```

**Delta-Sync with Retry:**
```cpp
// Add retry logic before falling back to full sync
VGREResult TCPClusterManager::sendDeltaSyncWithRetry(
    void* ptr, uint64_t handle,
    const std::vector<std::pair<size_t, size_t>>& dirtyRanges,
    std::shared_ptr<ClientConnection> client) {
  
  ExponentialBackoff backoff(100, 5000);
  int max_retries = 3;
  
  for (int attempt = 0; attempt < max_retries; ++attempt) {
    VGREResult r = sendDeltaSync(ptr, handle, dirtyRanges, client);
    
    if (r == VGREResult::SUCCESS) {
      return VGREResult::SUCCESS;
    }
    
    if (r == VGREResult::ERR_IO) {
      // Connection lost - don't retry
      return r;
    }
    
    // Transient failure - retry with backoff
    VGRE_LOG_WARN("TCPCluster", 
      "Delta-sync attempt " + std::to_string(attempt + 1) + 
      " failed, retrying...");
    std::this_thread::sleep_for(
      std::chrono::milliseconds(backoff.next())
    );
  }
  
  // All retries exhausted - fall back to full sync
  VGRE_LOG_INFO("TCPCluster", 
    "Delta-sync failed after retries, falling back to full sync");
  return sendFullSync(ptr, handle, 
    core::RuntimeEngine::instance().getMemoryManager()
      .getAllocationSize(ptr), client);
}
```

**Early Authentication Validation:**
```cpp
// Validate auth token before processing packet data
void TCPClusterManager::handleRemoteCommand(
    const RemoteCommandPacket &pkt) {
  
  // CRITICAL: Validate auth FIRST before any processing
  if (auth_token_ == 0 || pkt.auth_token != auth_token_) {
    VGRE_LOG_ERROR("TCPCluster",
      "Rejected remote command: invalid auth token");
    // Clear any pending args to prevent data leakage
    pending_args_.clear();
    return;
  }
  
  // Validate packet structure
  if (pkt.grid_dim[0] == 0 || pkt.block_dim[0] == 0) {
    VGRE_LOG_ERROR("TCPCluster", 
      "Invalid remote command packet");
    pending_args_.clear();
    return;
  }
  
  // Now safe to process...
}
```


### 5. Modular Architecture

**File Structure:**
```
src/advanced/tcp_cluster/
├── tcp_cluster_manager.cpp       // Main coordinator (500 lines)
├── connection_manager.cpp        // Client connection lifecycle (400 lines)
├── packet_handler.cpp            // Packet construction/parsing (300 lines)
├── security_manager.cpp          // Handshake and encryption (300 lines)
├── discovery_manager.cpp         // UDP discovery (300 lines)
├── dispatch_manager.cpp          // Kernel dispatch logic (400 lines)
├── memory_sync_manager.cpp       // Delta-sync and SHM (400 lines)
└── collective_ops_manager.cpp    // allReduce and barriers (300 lines)

include/vgre/advanced/tcp_cluster/
├── tcp_cluster_manager.h         // Public API
├── connection_manager.h          // Internal
├── packet_handler.h              // Internal
├── security_manager.h            // Internal
├── discovery_manager.h           // Internal
├── dispatch_manager.h            // Internal
├── memory_sync_manager.h         // Internal
└── collective_ops_manager.h      // Internal
```

**ConnectionManager Class:**
```cpp
class ConnectionManager {
public:
  ConnectionManager(TCPClusterManager* parent);
  
  // Connection lifecycle
  VGREResult acceptConnection(vgre_socket_t server_fd);
  VGREResult connectToMaster(const std::string& host, int port);
  void closeConnection(std::shared_ptr<ClientConnection> client);
  
  // Client management
  std::shared_ptr<ClientConnection> getClient(int worker_idx);
  std::vector<std::shared_ptr<ClientConnection>> getActiveClients();
  bool addClientIfNotDuplicate(const std::string& ip, vgre_socket_t fd);
  void purgeDeadClients();
  
private:
  TCPClusterManager* parent_;
  std::vector<std::shared_ptr<ClientConnection>> clients_;
  mutable std::recursive_mutex clients_mutex_;
};
```

**PacketHandler Class:**
```cpp
class PacketHandler {
public:
  // Packet construction
  std::vector<uint8_t> constructPacket(
    PacketType type, const void* payload, size_t payloadLen);
  
  // Packet sending
  VGREResult sendPacket(
    vgre_socket_t fd, PacketType type,
    const void* payload, size_t payloadLen,
    SecureChannel* sc = nullptr);
  
  VGREResult sendPacketDirect(
    vgre_socket_t fd, PacketType type,
    const void* payload, size_t payloadLen,
    SecureChannel* sc = nullptr);
  
  // Packet receiving
  VGREResult recvPacket(
    vgre_socket_t fd, std::vector<uint8_t>& outBuffer,
    SecureChannel* sc = nullptr);
  
  // Packet parsing
  VGREResult parseVSBPHeader(
    const std::vector<uint8_t>& buffer,
    VSBPHeader& header);
  
private:
  std::atomic<uint32_t> sequence_counter_{1};
};
```

**SecurityManager Class:**
```cpp
class SecurityManager {
public:
  SecurityManager(TCPClusterManager* parent);
  
  // Security configuration
  VGREResult enableSecurity(bool enabled);
  bool isSecurityEnabled() const;
  SessionInfo getSecurityInfo() const;
  
  // Handshake operations
  VGREResult performServerHandshake(
    std::shared_ptr<ClientConnection> client);
  VGREResult performClientHandshake(vgre_socket_t fd);
  
  // Key management
  VGREResult rotateSessionKey(std::shared_ptr<ClientConnection> client);
  
private:
  TCPClusterManager* parent_;
  std::atomic<bool> security_enabled_{false};
  std::string auth_token_str_;
  uint64_t auth_token_{0};
};
```


**DiscoveryManager Class:**
```cpp
class DiscoveryManager {
public:
  DiscoveryManager(TCPClusterManager* parent);
  
  // UDP discovery
  void startMasterAnnouncer();
  void startWorkerDiscovery();
  void startProactiveConnections();
  
  void stopAll();
  
  // Proactive connection management
  void addProactiveAddress(const std::string& address);
  void removeProactiveAddress(const std::string& address);
  
private:
  void udpAnnouncerLoop();
  void udpMasterDiscoveryLoop();
  void udpDiscoveryLoop();
  void udpWorkerAnnouncerLoop();
  void proactiveConnectionLoop();
  
  TCPClusterManager* parent_;
  JoinableThread announcer_thread_;
  JoinableThread discovery_thread_;
  JoinableThread proactive_thread_;
  std::atomic<bool> stop_flag_{false};
  std::vector<std::string> proactive_addresses_;
  std::mutex addresses_mutex_;
};
```

**DispatchManager Class:**
```cpp
class DispatchManager {
public:
  DispatchManager(TCPClusterManager* parent);
  
  // Remote kernel dispatch
  VGREResult launchRemoteKernel(
    int worker_idx, uint64_t kernel_id,
    const uint32_t grid_dim[3], const uint32_t block_dim[3],
    void** args, int num_args, size_t shared_mem);
  
  // Partitioned dispatch
  VGREResult launchPartitionedKernel(
    uint64_t kernel_id,
    const uint32_t grid_dim[3], const uint32_t block_dim[3],
    void** args, int num_args, size_t shared_mem);
  
  VGREResult collectPartitionResults(
    uint64_t kernel_id, uint32_t total_partitions,
    int timeout_ms = 30000);
  
  // Kernel registration
  void broadcastKernelRegistration(
    uint64_t kernel_id, const std::string& name,
    const std::string& source);
  
  // Result tracking
  VGREResult waitForRemoteResult(
    uint64_t kernel_id, int timeout_ms = 30000);
  
private:
  TCPClusterManager* parent_;
  std::map<uint64_t, VGREResult> remote_results_;
  std::mutex results_mutex_;
  std::condition_variable results_cv_;
  
  std::vector<PartitionResult> partition_results_;
  std::mutex partition_mutex_;
  std::condition_variable partition_cv_;
};
```

**MemorySyncManager Class:**
```cpp
class MemorySyncManager {
public:
  MemorySyncManager(TCPClusterManager* parent);
  
  // Argument streaming
  VGREResult streamArgumentsToWorker(
    void** args, int num_args, uint64_t kernel_id,
    std::shared_ptr<ClientConnection> client);
  
  // Memory synchronization
  VGREResult syncPointerToWorker(
    void* ptr, uint64_t handle,
    std::shared_ptr<ClientConnection> client);
  
  VGREResult syncPointerFromWorker(
    void* ptr, uint64_t handle, size_t size);
  
  // SHM management
  VGREResult initializeShmForClient(
    std::shared_ptr<ClientConnection> client);
  
private:
  VGREResult sendDeltaSync(
    void* ptr, uint64_t handle,
    const std::vector<std::pair<size_t, size_t>>& dirtyRanges,
    std::shared_ptr<ClientConnection> client);
  
  VGREResult sendFullSync(
    void* ptr, uint64_t handle, size_t size,
    std::shared_ptr<ClientConnection> client);
  
  VGREResult sendScalarArg(
    int index, ArgType type, void* value,
    std::shared_ptr<ClientConnection> client);
  
  VGREResult sendPointerArg(
    int index, void* arg,
    std::shared_ptr<ClientConnection> client);
  
  VGREResult sendStructArg(
    int index, void* arg, uint64_t kernel_id,
    std::shared_ptr<ClientConnection> client);
  
  TCPClusterManager* parent_;
};
```


**CollectiveOpsManager Class:**
```cpp
class CollectiveOpsManager {
public:
  CollectiveOpsManager(TCPClusterManager* parent);
  
  // Collective operations
  VGREResult allReduce(void* ptr, size_t count, int datatype);
  VGREResult barrier();
  
  // Reduction operations
  template<typename T>
  void sumReduce(T* dst, const T* src, size_t count);
  
private:
  void masterAllReduce(void* ptr, size_t count, int datatype);
  void workerAllReduce(void* ptr, size_t count, int datatype);
  
  TCPClusterManager* parent_;
  std::mutex reduction_mutex_;
  std::condition_variable reduction_cv_;
  std::atomic<int> reduction_count_{0};
  std::vector<uint8_t> reduction_buffer_;
  bool is_reducing_{false};
  uint32_t reduction_datatype_{0};
  size_t reduction_element_count_{0};
  uint64_t reduction_sequence_{0};
  
  std::mutex barrier_mutex_;
  std::condition_variable barrier_cv_;
  uint32_t barrier_count_{0};
};
```

### 6. Testable Architecture

**Dependency Injection:**
```cpp
// Replace singleton with dependency injection
class TCPClusterManager {
public:
  // Constructor injection for testability
  TCPClusterManager(
    std::unique_ptr<ISocketFactory> socket_factory,
    std::unique_ptr<ISecureChannelFactory> security_factory,
    std::unique_ptr<IMemoryManager> memory_manager);
  
  // For production use
  static TCPClusterManager& instance() {
    static TCPClusterManager inst(
      std::make_unique<RealSocketFactory>(),
      std::make_unique<RealSecureChannelFactory>(),
      std::make_unique<RealMemoryManager>()
    );
    return inst;
  }
  
private:
  std::unique_ptr<ISocketFactory> socket_factory_;
  std::unique_ptr<ISecureChannelFactory> security_factory_;
  std::unique_ptr<IMemoryManager> memory_manager_;
};
```

**Interface Abstractions:**
```cpp
// Socket abstraction for testing
class ISocketFactory {
public:
  virtual ~ISocketFactory() = default;
  virtual vgre_socket_t createSocket(int domain, int type, int protocol) = 0;
  virtual int bind(vgre_socket_t fd, const sockaddr* addr, socklen_t len) = 0;
  virtual int listen(vgre_socket_t fd, int backlog) = 0;
  virtual vgre_socket_t accept(vgre_socket_t fd, sockaddr* addr, socklen_t* len) = 0;
  virtual int connect(vgre_socket_t fd, const sockaddr* addr, socklen_t len) = 0;
  virtual int send(vgre_socket_t fd, const void* buf, size_t len, int flags) = 0;
  virtual int recv(vgre_socket_t fd, void* buf, size_t len, int flags) = 0;
  virtual void close(vgre_socket_t fd) = 0;
};

// Mock implementation for testing
class MockSocketFactory : public ISocketFactory {
public:
  MOCK_METHOD(vgre_socket_t, createSocket, (int, int, int), (override));
  MOCK_METHOD(int, bind, (vgre_socket_t, const sockaddr*, socklen_t), (override));
  MOCK_METHOD(int, listen, (vgre_socket_t, int), (override));
  MOCK_METHOD(vgre_socket_t, accept, (vgre_socket_t, sockaddr*, socklen_t*), (override));
  MOCK_METHOD(int, connect, (vgre_socket_t, const sockaddr*, socklen_t), (override));
  MOCK_METHOD(int, send, (vgre_socket_t, const void*, size_t, int), (override));
  MOCK_METHOD(int, recv, (vgre_socket_t, void*, size_t, int), (override));
  MOCK_METHOD(void, close, (vgre_socket_t), (override));
};
```


**Eliminate Global State:**
```cpp
// Replace global state with instance state
class TCPClusterManager {
public:
  // No static instance() - use explicit instances
  TCPClusterManager();
  
  // Configuration via constructor or setters
  void setAuthToken(const std::string& token);
  void setPort(int port);
  void setHost(const std::string& host);
  
private:
  // All state is instance-specific
  std::atomic<bool> enabled_{false};
  std::atomic<bool> security_enabled_{false};
  uint64_t auth_token_{0};
  std::string auth_token_str_;
  bool is_master_{false};
  int port_{7777};
  std::string host_;
  
  // No static variables
};

// For tests: create isolated instances
TEST(TCPClusterTest, ParallelExecution) {
  TCPClusterManager manager1;
  TCPClusterManager manager2;
  
  // Both can run in parallel without interference
  std::thread t1([&]() { manager1.initialize(true, "127.0.0.1", 7777); });
  std::thread t2([&]() { manager2.initialize(true, "127.0.0.1", 7778); });
  
  t1.join();
  t2.join();
}
```

## Hypothesized Root Cause

Based on the bug analysis, the root causes are:

1. **Missing Implementations**: Methods were declared for future use but never implemented, causing linker errors when code attempts to use them

2. **Code Duplication**: Rapid feature development led to copy-paste programming without refactoring, resulting in 4+ instances of identical logic

3. **Stub Implementations**: Placeholder code was left in place instead of being replaced with production-ready implementations (SIMD optimization, security enforcement)

4. **Poor Business Logic**: 
   - Magic numbers used for quick prototyping without documentation
   - Race conditions from insufficient locking granularity
   - Resource leaks from missing RAII patterns
   - Inconsistent error handling from incremental API evolution

5. **Monolithic Architecture**: Single 3,267-line file grew organically without modularization, making navigation and maintenance difficult

6. **Untestable Design**: Singleton pattern and global state prevent dependency injection and parallel test execution

## Correctness Properties

Property 1: Complete Implementations

_For any_ call to `reportComputeFromWorker()` or `allReduce()`, the system SHALL successfully link and execute the method with correct behavior.

**Validates: Requirements 2.1, 2.2**

Property 2: No Code Duplication

_For any_ code fragment that performs packet construction, delta-sync, argument serialization, or SHM fallback, the system SHALL use a single shared implementation instead of duplicated code.

**Validates: Requirements 2.3, 2.4, 2.5, 2.6**

Property 3: Production-Ready Implementations

_For any_ reduction operation or security handshake failure, the system SHALL use SIMD-optimized code and enforce security policy without silent fallback.

**Validates: Requirements 2.7, 2.8**

Property 4: Proper Business Logic

_For any_ timeout, priority, flow control limit, or memory offset, the system SHALL use named constants with documentation instead of magic numbers.

**Validates: Requirements 2.9, 2.10, 2.11, 2.12, 2.13**

Property 5: No Race Conditions

_For any_ duplicate connection check, the system SHALL perform atomic check-and-insert under a single lock to prevent TOCTOU races.

**Validates: Requirements 2.16**

Property 6: No Resource Leaks

_For any_ socket, thread, or SHM allocation, the system SHALL use RAII wrappers to ensure cleanup on all code paths.

**Validates: Requirements 2.17, 2.18, 2.19**

Property 7: Consistent Error Handling

_For any_ error-prone operation, the system SHALL return VGREResult consistently instead of mixing bool, int, and VGREResult.

**Validates: Requirements 2.15**

Property 8: Efficient I/O

_For any_ wait operation, the system SHALL use blocking I/O with timeout or proper poll/epoll instead of busy-waiting.

**Validates: Requirements 2.21**

Property 9: Modular Architecture

_For any_ subsystem (connection management, packet handling, security, discovery, dispatch, memory sync, collective ops), the system SHALL implement it in a separate focused file under 500 lines.

**Validates: Requirements 2.25**

Property 10: Testable Design

_For any_ external dependency (sockets, security, memory), the system SHALL provide interface abstractions allowing mock injection for unit testing.

**Validates: Requirements 2.30, 2.31, 2.32**


## Fix Implementation

### Phase 1: Implement Missing Methods (Priority: CRITICAL)

**File**: `src/advanced/tcp_cluster.cpp`

**Changes Required**:

1. Implement `reportComputeFromWorker()`:
```cpp
void TCPClusterManager::reportComputeFromWorker(
    double seconds, int cores, uint64_t kernel_id) {
  if (!enabled_ || is_master_) return;
  
  CreditReportPacket crpkt{};
  crpkt.compute_seconds = seconds;
  crpkt.cpu_cores = cores;
  crpkt.kernel_id = kernel_id;
  crpkt.timestamp = static_cast<uint64_t>(
    std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::system_clock::now().time_since_epoch()).count());
  
  send_packet(client_fd_, PacketType::CREDIT_REPORT, &crpkt, 
              sizeof(CreditReportPacket), client_secure_channel_.get());
  
  VGRE_LOG_DEBUG("TCPCluster", 
    "Reported compute: " + std::to_string(seconds) + "s on " + 
    std::to_string(cores) + " cores");
}
```

2. Implement `allReduce()`:
```cpp
VGREResult TCPClusterManager::allReduce(
    void* ptr, size_t count, int datatype) {
  if (!enabled_) return VGREResult::ERR_NOT_INITIALIZED;
  
  if (!is_master_) {
    // Worker path: send local data to master
    CollectiveOpPacket cop{};
    cop.op_type = 0; // all_reduce
    cop.datatype = datatype;
    cop.count = count;
    cop.sequence = reduction_sequence_++;
    
    size_t type_size = getTypeSizeFromDatatype(datatype);
    
    send_packet(client_fd_, PacketType::COLLECTIVE_OP, &cop, 
                sizeof(cop), client_secure_channel_.get());
    send_packet(client_fd_, PacketType::RAW_DATA, ptr, 
                count * type_size, client_secure_channel_.get());
    
    // Wait for master to broadcast result
    std::unique_lock<std::mutex> lock(reduction_mutex_);
    auto deadline = std::chrono::steady_clock::now() + 
                    std::chrono::seconds(30);
    
    if (!reduction_cv_.wait_until(lock, deadline, 
        [this]() { return !is_reducing_; })) {
      return VGREResult::ERR_TIMEOUT;
    }
    
    // Copy result back to ptr
    std::memcpy(ptr, active_reduction_buffer_.data(), 
                count * type_size);
    return VGREResult::SUCCESS;
  }
  
  // Master path: collect from all workers, reduce, broadcast
  {
    std::lock_guard<std::mutex> lock(reduction_mutex_);
    is_reducing_ = true;
    reduction_count_ = 0;
    reduction_datatype_ = datatype;
    reduction_element_count_ = count;
    
    size_t type_size = getTypeSizeFromDatatype(datatype);
    active_reduction_buffer_.resize(count * type_size);
    std::memcpy(active_reduction_buffer_.data(), ptr, 
                count * type_size);
  }
  
  // Wait for all workers to send their data
  int expected_workers = 0;
  {
    std::lock_guard<std::recursive_mutex> lock(clients_mutex_);
    for (const auto& c : clients_) {
      if (c && c->active) expected_workers++;
    }
  }
  
  std::unique_lock<std::mutex> lock(reduction_mutex_);
  auto deadline = std::chrono::steady_clock::now() + 
                  std::chrono::seconds(30);
  
  if (!reduction_cv_.wait_until(lock, deadline,
      [this, expected_workers]() { 
        return reduction_count_ >= expected_workers; 
      })) {
    is_reducing_ = false;
    return VGREResult::ERR_TIMEOUT;
  }
  
  // Broadcast result to all workers
  size_t type_size = getTypeSizeFromDatatype(datatype);
  broadcastPacket(PacketType::RAW_DATA, 
                  active_reduction_buffer_.data(),
                  count * type_size);
  
  // Copy result back to master's ptr
  std::memcpy(ptr, active_reduction_buffer_.data(), 
              count * type_size);
  
  is_reducing_ = false;
  reduction_cv_.notify_all();
  
  return VGREResult::SUCCESS;
}

size_t TCPClusterManager::getTypeSizeFromDatatype(int datatype) {
  switch (static_cast<ArgType>(datatype)) {
    case ArgType::INT32:
    case ArgType::UINT32:
    case ArgType::FLOAT32:
      return 4;
    case ArgType::INT64:
    case ArgType::UINT64:
    case ArgType::FLOAT64:
      return 8;
    default:
      return 8;
  }
}
```

3. Add handler in `processClientStagingBuffer()` for COLLECTIVE_OP:
```cpp
} else if (type == PacketType::COLLECTIVE_OP) {
  CollectiveOpPacket cop;
  std::memcpy(&cop, payload, sizeof(CollectiveOpPacket));
  client_rx_buffer_.erase(client_rx_buffer_.begin(), 
                          client_rx_buffer_.begin() + totalLen);
  
  // Expect RAW_DATA next with the actual values
  // (handled in next iteration when RAW_DATA arrives)
  
} else if (type == PacketType::RAW_DATA && is_reducing_) {
  // This is reduction data from a worker
  std::lock_guard<std::mutex> lock(reduction_mutex_);
  
  size_t type_size = getTypeSizeFromDatatype(reduction_datatype_);
  const uint8_t* worker_data = payload;
  
  // Perform reduction based on datatype
  if (reduction_datatype_ == static_cast<uint32_t>(ArgType::FLOAT32)) {
    sum_reduce_simd(
      reinterpret_cast<float*>(active_reduction_buffer_.data()),
      reinterpret_cast<const float*>(worker_data),
      reduction_element_count_);
  } else if (reduction_datatype_ == static_cast<uint32_t>(ArgType::FLOAT64)) {
    sum_reduce_simd(
      reinterpret_cast<double*>(active_reduction_buffer_.data()),
      reinterpret_cast<const double*>(worker_data),
      reduction_element_count_);
  } else if (reduction_datatype_ == static_cast<uint32_t>(ArgType::INT32)) {
    sum_reduce_simd(
      reinterpret_cast<int32_t*>(active_reduction_buffer_.data()),
      reinterpret_cast<const int32_t*>(worker_data),
      reduction_element_count_);
  } else if (reduction_datatype_ == static_cast<uint32_t>(ArgType::INT64)) {
    sum_reduce_simd(
      reinterpret_cast<int64_t*>(active_reduction_buffer_.data()),
      reinterpret_cast<const int64_t*>(worker_data),
      reduction_element_count_);
  }
  
  reduction_count_++;
  reduction_cv_.notify_all();
  
  client_rx_buffer_.erase(client_rx_buffer_.begin(), 
                          client_rx_buffer_.begin() + totalLen);
}
```


### Phase 2: Eliminate Code Duplication (Priority: HIGH)

**File**: `src/advanced/tcp_cluster.cpp`

**Changes Required**:

1. Extract common packet construction:
```cpp
// Add new private method
std::vector<uint8_t> TCPClusterManager::constructPacket(
    PacketType type, const void* payload, size_t payloadLen) {
  static std::atomic<uint32_t> s_seq{1};
  
  size_t totalLen = sizeof(VSBPHeader) + payloadLen;
  std::vector<uint8_t> staging(totalLen);
  
  VSBPHeader* header = reinterpret_cast<VSBPHeader*>(staging.data());
  header->magic = VSBP_MAGIC;
  header->version = VSBP_VERSION;
  header->type = static_cast<uint16_t>(type);
  header->sequence = s_seq.fetch_add(1, std::memory_order_relaxed);
  header->payloadSize = payloadLen;
  
  if (payload && payloadLen > 0) {
    std::memcpy(staging.data() + sizeof(VSBPHeader), payload, payloadLen);
  }
  
  return staging;
}

// Refactor send_packet to use constructPacket
bool TCPClusterManager::send_packet(
    vgre_socket_t fd, PacketType type, 
    const void *payload, size_t payloadLen, 
    SecureChannel *sc) {
  
  if (common::InputValidator::validatePacketSize(payloadLen) 
      != VGREResult::SUCCESS) {
    VGRE_LOG_ERROR("TCPCluster", 
      "Packet size validation failed: " + std::to_string(payloadLen));
    return false;
  }
  
  auto staging = constructPacket(type, payload, payloadLen);
  
  uint32_t priority = (type == PacketType::RESPONSE || 
                       type == PacketType::PARTITION_RESULT ||
                       type == PacketType::TELEMETRY || 
                       type == PacketType::SECURE_HANDSHAKE ||
                       type == PacketType::SECURE_HANDSHAKE_ACK || 
                       type == PacketType::ROTATE_KEY ||
                       type == PacketType::CREDIT_REPORT || 
                       type == PacketType::COOP_BARRIER_SYNC ||
                       type == PacketType::COOP_BARRIER_RESUME) 
                      ? 0 : 1;
  
  // Enqueue logic (unchanged)...
}

// Refactor send_packet_direct to use constructPacket
bool TCPClusterManager::send_packet_direct(
    vgre_socket_t fd, PacketType type,
    const void *payload, size_t payloadLen,
    SecureChannel *sc) {
  
  auto staging = constructPacket(type, payload, payloadLen);
  
  global_packets_sent_++;
  global_bytes_sent_ += staging.size();
  
  if (sc && sc->isInitialized()) {
    return sc->sendSecure(fd, staging.data(), staging.size()) 
           == VGREResult::SUCCESS;
  } else {
    std::atomic<bool> enabled{true};
    return send_all(fd, staging.data(), staging.size(), &enabled);
  }
}
```

2. Extract delta-sync logic:
```cpp
// Add new private methods
VGREResult TCPClusterManager::syncPointerToWorker(
    void* ptr, uint64_t handle, 
    std::shared_ptr<ClientConnection> client) {
  
  auto& mm = core::RuntimeEngine::instance().getMemoryManager();
  size_t size = mm.getAllocationSize(ptr);
  if (size == 0 || !ptr) return VGREResult::SUCCESS;
  
  std::vector<std::pair<size_t, size_t>> dirtyRanges;
  bool useDelta = client->synced_handles.count(ptr) > 0;
  
  if (useDelta) {
    mm.getDirtyPages(ptr, dirtyRanges);
  }
  
  if (useDelta && !dirtyRanges.empty()) {
    VGREResult r = sendDeltaSync(ptr, handle, dirtyRanges, client);
    if (r == VGREResult::SUCCESS) {
      mm.clearDirtyPages(ptr);
      return VGREResult::SUCCESS;
    }
    // Fall through to full sync on failure
  }
  
  client->synced_handles.insert(ptr);
  VGREResult r = sendFullSync(ptr, handle, size, client);
  if (r == VGREResult::SUCCESS) {
    mm.clearDirtyPages(ptr);
  }
  return r;
}

VGREResult TCPClusterManager::sendDeltaSync(
    void* ptr, uint64_t handle,
    const std::vector<std::pair<size_t, size_t>>& dirtyRanges,
    std::shared_ptr<ClientConnection> client) {
  
  if (client->is_local && client->shmManager) {
    return sendDeltaSyncSHM(ptr, handle, dirtyRanges, client);
  } else {
    return sendDeltaSyncTCP(ptr, handle, dirtyRanges, client);
  }
}

VGREResult TCPClusterManager::sendDeltaSyncSHM(
    void* ptr, uint64_t handle,
    const std::vector<std::pair<size_t, size_t>>& dirtyRanges,
    std::shared_ptr<ClientConnection> client) {
  
  uint64_t totalSize = 0;
  for (const auto& range : dirtyRanges) {
    totalSize += range.second;
  }
  
  uint64_t baseOffset = client->shm_offset;
  if (baseOffset + totalSize > client->shmManager->getSize()) {
    return VGREResult::ERR_OUT_OF_MEMORY;
  }
  
  client->shm_offset += totalSize;
  
  DataShmDirtyPacket dspkt{};
  dspkt.target_ptr = handle;
  dspkt.num_ranges = static_cast<uint32_t>(dirtyRanges.size());
  dspkt.shm_offset = baseOffset;
  
  if (!send_packet(client->socket_fd, PacketType::DATA_SHM_DIRTY, 
                   &dspkt, sizeof(dspkt), client->secureChannel.get())) {
    return VGREResult::ERR_IO;
  }
  
  uint64_t currentOffset = baseOffset;
  for (const auto& range : dirtyRanges) {
    DirtyRangePacket rpkt{range.first, range.second};
    if (!send_packet(client->socket_fd, PacketType::DIRTY_RANGE, 
                     &rpkt, sizeof(rpkt), client->secureChannel.get())) {
      return VGREResult::ERR_IO;
    }
    
    std::memcpy(
      static_cast<uint8_t*>(client->shmManager->getBasePtr()) + currentOffset,
      static_cast<uint8_t*>(ptr) + range.first,
      range.second);
    currentOffset += range.second;
  }
  
  return VGREResult::SUCCESS;
}

VGREResult TCPClusterManager::sendDeltaSyncTCP(
    void* ptr, uint64_t handle,
    const std::vector<std::pair<size_t, size_t>>& dirtyRanges,
    std::shared_ptr<ClientConnection> client) {
  
  DataHeaderDirtyPacket dhpkt{};
  dhpkt.target_ptr = handle;
  dhpkt.num_ranges = static_cast<uint32_t>(dirtyRanges.size());
  
  if (!send_packet(client->socket_fd, PacketType::DATA_HEADER_DIRTY,
                   &dhpkt, sizeof(dhpkt), client->secureChannel.get())) {
    return VGREResult::ERR_IO;
  }
  
  for (const auto& range : dirtyRanges) {
    DirtyRangePacket rpkt{range.first, range.second};
    if (!send_packet(client->socket_fd, PacketType::DIRTY_RANGE,
                     &rpkt, sizeof(rpkt), client->secureChannel.get())) {
      return VGREResult::ERR_IO;
    }
    
    if (!send_packet(client->socket_fd, PacketType::DATA_BODY,
                     static_cast<uint8_t*>(ptr) + range.first,
                     range.second, client->secureChannel.get())) {
      return VGREResult::ERR_IO;
    }
  }
  
  return VGREResult::SUCCESS;
}

VGREResult TCPClusterManager::sendFullSync(
    void* ptr, uint64_t handle, size_t size,
    std::shared_ptr<ClientConnection> client) {
  
  if (client->is_local && client->shmManager) {
    return sendFullSyncSHM(ptr, handle, size, client);
  } else {
    return sendFullSyncTCP(ptr, handle, size, client);
  }
}

VGREResult TCPClusterManager::sendFullSyncSHM(
    void* ptr, uint64_t handle, size_t size,
    std::shared_ptr<ClientConnection> client) {
  
  uint64_t offset = client->shm_offset;
  if (offset + size > client->shmManager->getSize()) {
    return VGREResult::ERR_OUT_OF_MEMORY;
  }
  
  client->shm_offset += size;
  
  std::memcpy(
    static_cast<uint8_t*>(client->shmManager->getBasePtr()) + offset,
    ptr, size);
  
  DataShmPacket dspkt{};
  dspkt.target_ptr = handle;
  dspkt.shm_offset = offset;
  dspkt.size = size;
  
  if (!send_packet(client->socket_fd, PacketType::DATA_SHM,
                   &dspkt, sizeof(dspkt), client->secureChannel.get())) {
    return VGREResult::ERR_IO;
  }
  
  return VGREResult::SUCCESS;
}

VGREResult TCPClusterManager::sendFullSyncTCP(
    void* ptr, uint64_t handle, size_t size,
    std::shared_ptr<ClientConnection> client) {
  
  DataHeaderPacket dpkt{};
  dpkt.target_ptr = handle;
  dpkt.size = size;
  
  if (!send_packet(client->socket_fd, PacketType::DATA_HEADER,
                   &dpkt, sizeof(dpkt), client->secureChannel.get())) {
    return VGREResult::ERR_IO;
  }
  
  if (!send_packet(client->socket_fd, PacketType::DATA_BODY,
                   ptr, size, client->secureChannel.get())) {
    return VGREResult::ERR_IO;
  }
  
  return VGREResult::SUCCESS;
}

// Now refactor launchRemoteKernel and launchPartitionedKernel
// to use syncPointerToWorker instead of duplicated logic
```


3. Extract argument serialization:
```cpp
// Add new private methods
VGREResult TCPClusterManager::streamArgumentsToWorker(
    void** args, int num_args, uint64_t kernel_id,
    std::shared_ptr<ClientConnection> client) {
  
  std::vector<ArgType> argTypes;
  if (core::RuntimeEngine::instance().getKernelArgTypes(kernel_id, argTypes)
      != VGREResult::SUCCESS) {
    return VGREResult::ERR_INVALID_KERNEL;
  }
  
  for (int i = 0; i < num_args; ++i) {
    ArgType type = (i < static_cast<int>(argTypes.size())) 
                   ? argTypes[i] : ArgType::UINT64;
    
    VGREResult r = VGREResult::SUCCESS;
    if (type == ArgType::STRUCT) {
      r = sendStructArg(i, args[i], kernel_id, client);
    } else if (type == ArgType::POINTER) {
      r = sendPointerArg(i, args[i], client);
    } else {
      r = sendScalarArg(i, type, args[i], client);
    }
    
    if (r != VGREResult::SUCCESS) {
      return r;
    }
  }
  
  return VGREResult::SUCCESS;
}

VGREResult TCPClusterManager::sendStructArg(
    int index, void* arg, uint64_t kernel_id,
    std::shared_ptr<ClientConnection> client) {
  
  const auto* ir = core::RuntimeEngine::instance().getKernelIR(kernel_id);
  if (!ir || index >= static_cast<int>(ir->argSizes.size())) {
    return VGREResult::ERR_INVALID_VALUE;
  }
  
  size_t size = ir->argSizes[index];
  
  StructDataPacket spkt{};
  spkt.arg_index = index;
  spkt.size = static_cast<uint32_t>(size);
  
  if (!send_packet(client->socket_fd, PacketType::STRUCT_DATA,
                   &spkt, sizeof(spkt), client->secureChannel.get())) {
    return VGREResult::ERR_IO;
  }
  
  if (!send_packet(client->socket_fd, PacketType::DATA_BODY,
                   arg, size, client->secureChannel.get())) {
    return VGREResult::ERR_IO;
  }
  
  return VGREResult::SUCCESS;
}

VGREResult TCPClusterManager::sendPointerArg(
    int index, void* arg,
    std::shared_ptr<ClientConnection> client) {
  
  void* ptr = *static_cast<void**>(arg);
  uint64_t handle = reinterpret_cast<uint64_t>(ptr);
  
  // Sync memory first
  VGREResult r = syncPointerToWorker(ptr, handle, client);
  if (r != VGREResult::SUCCESS) {
    return r;
  }
  
  // Send pointer argument
  ArgScalarPacket apkt{};
  apkt.arg_index = index;
  apkt.arg_type = static_cast<uint8_t>(ArgType::POINTER);
  apkt.value = handle;
  
  if (!send_packet(client->socket_fd, PacketType::ARG_POINTER,
                   &apkt, sizeof(apkt), client->secureChannel.get())) {
    return VGREResult::ERR_IO;
  }
  
  return VGREResult::SUCCESS;
}

VGREResult TCPClusterManager::sendScalarArg(
    int index, ArgType type, void* arg,
    std::shared_ptr<ClientConnection> client) {
  
  ArgScalarPacket apkt{};
  apkt.arg_index = index;
  apkt.arg_type = static_cast<uint8_t>(type);
  
  size_t arg_size = 8;
  switch (type) {
    case ArgType::INT32:
    case ArgType::UINT32:
    case ArgType::FLOAT32:
      arg_size = 4;
      break;
    default:
      arg_size = 8;
      break;
  }
  
  std::memset(&apkt.value, 0, 8);
  std::memcpy(&apkt.value, arg, arg_size);
  
  if (!send_packet(client->socket_fd, PacketType::ARG_SCALAR,
                   &apkt, sizeof(apkt), client->secureChannel.get())) {
    return VGREResult::ERR_IO;
  }
  
  return VGREResult::SUCCESS;
}

// Refactor launchRemoteKernel to use streamArgumentsToWorker
VGREResult TCPClusterManager::launchRemoteKernel(
    int worker_idx, uint64_t kernel_id, const uint32_t grid_dim[3],
    const uint32_t block_dim[3], void **args, int num_args, 
    size_t shared_mem) {
  
  // Validation...
  
  std::lock_guard<std::recursive_mutex> lock(clients_mutex_);
  if (worker_idx < 0 || worker_idx >= static_cast<int>(clients_.size()) ||
      !clients_[worker_idx] || !clients_[worker_idx]->active) {
    return VGREResult::ERR_INVALID_VALUE;
  }
  
  // Flow control check...
  
  // Stream arguments using unified method
  VGREResult r = streamArgumentsToWorker(
    args, num_args, kernel_id, clients_[worker_idx]);
  if (r != VGREResult::SUCCESS) {
    return r;
  }
  
  // Send launch command...
}
```


### Phase 3: Replace Stubs with Production Code (Priority: HIGH)

**File**: `src/advanced/tcp_cluster.cpp`

**Changes Required**:

1. Replace `sum_reduce` with SIMD-optimized version:
```cpp
// Add SIMD intrinsics header
#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__SSE2__)
#include <emmintrin.h>
#endif

// Replace simple loop with SIMD-optimized version
template <typename T>
void sum_reduce(T* dst, const T* src, size_t count) {
  size_t i = 0;
  
#if defined(__AVX2__)
  if constexpr (std::is_same_v<T, float>) {
    for (; i + 8 <= count; i += 8) {
      __m256 d = _mm256_loadu_ps(dst + i);
      __m256 s = _mm256_loadu_ps(src + i);
      __m256 r = _mm256_add_ps(d, s);
      _mm256_storeu_ps(dst + i, r);
    }
  } else if constexpr (std::is_same_v<T, double>) {
    for (; i + 4 <= count; i += 4) {
      __m256d d = _mm256_loadu_pd(dst + i);
      __m256d s = _mm256_loadu_pd(src + i);
      __m256d r = _mm256_add_pd(d, s);
      _mm256_storeu_pd(dst + i, r);
    }
  } else if constexpr (std::is_same_v<T, int32_t>) {
    for (; i + 8 <= count; i += 8) {
      __m256i d = _mm256_loadu_si256((__m256i*)(dst + i));
      __m256i s = _mm256_loadu_si256((__m256i*)(src + i));
      __m256i r = _mm256_add_epi32(d, s);
      _mm256_storeu_si256((__m256i*)(dst + i), r);
    }
  } else if constexpr (std::is_same_v<T, int64_t>) {
    for (; i + 4 <= count; i += 4) {
      __m256i d = _mm256_loadu_si256((__m256i*)(dst + i));
      __m256i s = _mm256_loadu_si256((__m256i*)(src + i));
      __m256i r = _mm256_add_epi64(d, s);
      _mm256_storeu_si256((__m256i*)(dst + i), r);
    }
  }
#elif defined(__SSE2__)
  if constexpr (std::is_same_v<T, float>) {
    for (; i + 4 <= count; i += 4) {
      __m128 d = _mm_loadu_ps(dst + i);
      __m128 s = _mm_loadu_ps(src + i);
      __m128 r = _mm_add_ps(d, s);
      _mm_storeu_ps(dst + i, r);
    }
  } else if constexpr (std::is_same_v<T, double>) {
    for (; i + 2 <= count; i += 2) {
      __m128d d = _mm_loadu_pd(dst + i);
      __m128d s = _mm_loadu_pd(src + i);
      __m128d r = _mm_add_pd(d, s);
      _mm_storeu_pd(dst + i, r);
    }
  } else if constexpr (std::is_same_v<T, int32_t>) {
    for (; i + 4 <= count; i += 4) {
      __m128i d = _mm_loadu_si128((__m128i*)(dst + i));
      __m128i s = _mm_loadu_si128((__m128i*)(src + i));
      __m128i r = _mm_add_epi32(d, s);
      _mm_storeu_si128((__m128i*)(dst + i), r);
    }
  } else if constexpr (std::is_same_v<T, int64_t>) {
    for (; i + 2 <= count; i += 2) {
      __m128i d = _mm_loadu_si128((__m128i*)(dst + i));
      __m128i s = _mm_loadu_si128((__m128i*)(src + i));
      __m128i r = _mm_add_epi64(d, s);
      _mm_storeu_si128((__m128i*)(dst + i), r);
    }
  }
#endif
  
  // Scalar fallback for remaining elements
  for (; i < count; ++i) {
    dst[i] += src[i];
  }
}
```

2. Enforce security policy (no silent plaintext fallback):
```cpp
// In performSecureHandshake, remove plaintext fallback logic
VGREResult TCPClusterManager::performSecureHandshake(
    std::shared_ptr<ClientConnection> clientPtr) {
  
  if (!clientPtr) return VGREResult::ERR_INVALID_VALUE;
  auto &client = *clientPtr;
  
  if (!security_enabled_) {
    // Security not required - skip handshake
    return VGREResult::SUCCESS;
  }
  
  if (auth_token_str_.empty()) {
    VGRE_LOG_ERROR("TCPCluster", 
      "Security enabled but VGRE_TCP_AUTH_TOKEN not set");
    return VGREResult::ERR_AUTH_FAILED;
  }
  
  // Perform handshake...
  SecureHandshakePacket shpkt{};
  SecureChannel::generateNonce(shpkt.nonce);
  
  if (!send_packet_direct(client.socket_fd, 
                          PacketType::SECURE_HANDSHAKE,
                          &shpkt, sizeof(shpkt), nullptr)) {
    VGRE_LOG_ERROR("TCPCluster", 
      "Failed to send handshake nonce");
    return VGREResult::ERR_IO;
  }
  
  // Receive response...
  const size_t expectedLen = sizeof(VSBPHeader) + 
                             sizeof(SecureHandshakePacket);
  std::vector<uint8_t> rx;
  rx.reserve(expectedLen);
  
  auto deadline = std::chrono::steady_clock::now() + 
                  std::chrono::seconds(5);
  
  while (rx.size() < expectedLen) {
    if (std::chrono::steady_clock::now() > deadline) {
      VGRE_LOG_ERROR("TCPCluster", 
        "Handshake timeout for " + client.ip_address);
      return VGREResult::ERR_TIMEOUT;
    }
    
    uint8_t buf[256];
    size_t toRead = std::min(sizeof(buf), expectedLen - rx.size());
    int n = recv(client.socket_fd, buf, static_cast<int>(toRead), 0);
    
    if (n > 0) {
      rx.insert(rx.end(), buf, buf + n);
    } else if (n == 0) {
      VGRE_LOG_ERROR("TCPCluster",
        "Peer closed connection during handshake: " + 
        client.ip_address);
      return VGREResult::ERR_IO;
    } else {
      if (vgre_is_would_block(vgre_get_last_socket_error())) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        continue;
      }
      VGRE_LOG_ERROR("TCPCluster",
        "Handshake recv failed for " + client.ip_address);
      return VGREResult::ERR_IO;
    }
  }
  
  VSBPHeader* header = reinterpret_cast<VSBPHeader*>(rx.data());
  if (header->magic != VSBP_MAGIC) {
    VGRE_LOG_ERROR("TCPCluster", 
      "Invalid magic in handshake response");
    return VGREResult::ERR_AUTH_FAILED;
  }
  
  if (header->type != static_cast<uint16_t>(
        PacketType::SECURE_HANDSHAKE_ACK)) {
    // CRITICAL: No plaintext fallback when security is enabled
    VGRE_LOG_ERROR("TCPCluster",
      "Expected SECURE_HANDSHAKE_ACK, got type " + 
      std::to_string(header->type) + 
      " from " + client.ip_address + 
      " - closing connection (security policy enforced)");
    return VGREResult::ERR_AUTH_FAILED;
  }
  
  SecureHandshakePacket clientHs{};
  std::memcpy(&clientHs, rx.data() + sizeof(VSBPHeader), 
              sizeof(SecureHandshakePacket));
  
  // Derive session key
  client.secureChannel = std::make_unique<SecureChannel>();
  VGREResult r = client.secureChannel->initializeFromSecret(
    auth_token_str_, shpkt.nonce, clientHs.nonce);
  
  if (r != VGREResult::SUCCESS) {
    VGRE_LOG_ERROR("TCPCluster", 
      "Key derivation failed for " + client.ip_address);
    return r;
  }
  
  client.security_established = true;
  VGRE_LOG_INFO("TCPCluster",
    "Security handshake completed with " + client.ip_address);
  
  return VGREResult::SUCCESS;
}
```


### Phase 4: Fix Business Logic Issues (Priority: CRITICAL)

**File**: `src/advanced/tcp_cluster.cpp`

**Changes Required**:

1. Add named constants at top of file:
```cpp
namespace {

// Using vgre::common types and helpers
using vgre::common::vgre_socket_t;
using vgre::common::VGRE_INVALID_SOCKET;
// ... existing using declarations ...

// Named Constants
namespace Constants {
  // Timeouts
  constexpr int SEND_TIMEOUT_SECONDS = 5;
  constexpr int HANDSHAKE_TIMEOUT_SECONDS = 5;
  constexpr int PEEK_TIMEOUT_MS = 200;
  constexpr int POLL_TIMEOUT_MS = 50;
  constexpr int RECV_TIMEOUT_MS = 1000;
  constexpr int BACKOFF_AFTER_DISCONNECT_SECONDS = 8;
  constexpr int HANDSHAKE_STUCK_TIMEOUT_MS = 15000;
  
  // Flow Control
  constexpr uint32_t MAX_IN_FLIGHT_KERNELS = 16;
  
  // Memory
  constexpr uint64_t SHM_RESULT_OFFSET_BASE = 128 * 1024 * 1024; // 128MB
  constexpr size_t DEFAULT_SHM_SIZE = 256 * 1024 * 1024; // 256MB
  
  // Backoff
  constexpr int INITIAL_BACKOFF_SECONDS = 5;
  constexpr int MAX_BACKOFF_SECONDS = 300;
  constexpr int BACKOFF_MULTIPLIER = 4;
  constexpr int PROACTIVE_BACKOFF_SECONDS = 4;
  
  // Discovery
  constexpr int UDP_ANNOUNCE_INTERVAL_SECONDS = 2;
  constexpr int UDP_DISCOVERY_PORT = 7778;
  constexpr int UDP_WORKER_PORT = 7779;
  
  // Retry
  constexpr int MAX_DELTA_SYNC_RETRIES = 3;
  constexpr int INITIAL_RETRY_BACKOFF_MS = 100;
  constexpr int MAX_RETRY_BACKOFF_MS = 5000;
}

enum class PacketPriority : uint32_t {
  HIGH = 0,  // Control/Sync packets
  LOW = 1    // Data/Bulk packets
};

// ... rest of anonymous namespace ...
}
```

2. Replace all magic numbers with named constants:
```cpp
// Example replacements throughout the file:

// Before:
if (std::chrono::duration_cast<std::chrono::seconds>(
      std::chrono::steady_clock::now() - start).count() > 5) {
  return false;
}

// After:
if (std::chrono::duration_cast<std::chrono::seconds>(
      std::chrono::steady_clock::now() - start).count() 
    > Constants::SEND_TIMEOUT_SECONDS) {
  return false;
}

// Before:
uint32_t priority = 1; // Default: LOW
if (type == PacketType::RESPONSE || ...) {
  priority = 0; // HIGH
}

// After:
PacketPriority priority = PacketPriority::LOW;
if (type == PacketType::RESPONSE || ...) {
  priority = PacketPriority::HIGH;
}

// Before:
const uint32_t MAX_IN_FLIGHT = 16;

// After:
if (clients_[worker_idx]->in_flight_kernels >= 
    Constants::MAX_IN_FLIGHT_KERNELS) {
  // ...
}

// Before:
static std::atomic<uint64_t> resultOffset{128 * 1024 * 1024};

// After:
static std::atomic<uint64_t> resultOffset{
  Constants::SHM_RESULT_OFFSET_BASE};

// Before:
size_t shmSize = 256 * 1024 * 1024; // 256MB default

// After:
size_t shmSize = Constants::DEFAULT_SHM_SIZE;
```

3. Fix TOCTOU race in duplicate connection detection:
```cpp
// In serverLoop, replace the duplicate check with atomic version:

// Before (TOCTOU race):
bool is_duplicate = false;
for (const auto& ec : clients_) {
  if (ec && (ec->active || ec->is_authenticating) &&
      ec->socket_fd != VGRE_INVALID_SOCKET &&
      ec->ip_address == inbound_ip) {
    is_duplicate = true;
    break;
  }
}
if (is_duplicate) {
  vgre_close_socket(new_socket);
  goto server_loop_next_iter;
}

auto conn = std::make_shared<ClientConnection>();
// ... initialize conn ...
clients_.push_back(std::move(conn));

// After (atomic check-and-insert):
if (!addClientIfNotDuplicate(inbound_ip, new_socket, address)) {
  goto server_loop_next_iter;
}

// Add new method:
bool TCPClusterManager::addClientIfNotDuplicate(
    const std::string& ip, vgre_socket_t new_socket,
    const sockaddr_in& address) {
  
  // Lock is already held by caller (serverLoop)
  // Check for duplicate
  for (const auto& ec : clients_) {
    if (ec && (ec->active || ec->is_authenticating) &&
        ec->socket_fd != VGRE_INVALID_SOCKET &&
        ec->ip_address == ip) {
      VGRE_LOG_WARN("TCPCluster",
        "Dropping duplicate inbound connection from " + ip);
      vgre_close_socket(new_socket);
      return false;
    }
  }
  
  // No duplicate - safe to add
  auto conn = std::make_shared<ClientConnection>();
  conn->socket_fd = new_socket;
  conn->ip_address = ip;
  conn->port = ntohs(address.sin_port);
  conn->active = true;
  conn->expecting_type = true;
  conn->rx_buffer.clear();
  clients_.push_back(std::move(conn));
  
  VGRE_LOG_INFO("TCPCluster",
    "Master: Accepted connection from " + ip + ":" + 
    std::to_string(conn->port));
  
  return true;
}
```


4. Add RAII wrappers for resource management:
```cpp
// Add to tcp_cluster.cpp or separate utility header

// Socket RAII wrapper
class SocketGuard {
public:
  explicit SocketGuard(vgre_socket_t fd = VGRE_INVALID_SOCKET) 
    : fd_(fd) {}
  
  ~SocketGuard() {
    if (fd_ != VGRE_INVALID_SOCKET) {
      vgre_close_socket(fd_);
    }
  }
  
  SocketGuard(const SocketGuard&) = delete;
  SocketGuard& operator=(const SocketGuard&) = delete;
  
  SocketGuard(SocketGuard&& other) noexcept : fd_(other.fd_) {
    other.fd_ = VGRE_INVALID_SOCKET;
  }
  
  SocketGuard& operator=(SocketGuard&& other) noexcept {
    if (this != &other) {
      if (fd_ != VGRE_INVALID_SOCKET) {
        vgre_close_socket(fd_);
      }
      fd_ = other.fd_;
      other.fd_ = VGRE_INVALID_SOCKET;
    }
    return *this;
  }
  
  vgre_socket_t get() const { return fd_; }
  
  vgre_socket_t release() {
    vgre_socket_t tmp = fd_;
    fd_ = VGRE_INVALID_SOCKET;
    return tmp;
  }
  
private:
  vgre_socket_t fd_;
};

// Use in udpAnnouncerLoop:
void TCPClusterManager::udpAnnouncerLoop() {
  SocketGuard udp_fd(socket(AF_INET, SOCK_DGRAM, 0));
  if (udp_fd.get() == VGRE_INVALID_SOCKET) return;
  
  int opt = 1;
  vgre_setsockopt(udp_fd.get(), SOL_SOCKET, SO_BROADCAST, &opt, sizeof(opt));
  
  // ... rest of function ...
  // Socket automatically closed when udp_fd goes out of scope
}

// Use in udpMasterDiscoveryLoop:
void TCPClusterManager::udpMasterDiscoveryLoop() {
  SocketGuard udp_fd(socket(AF_INET, SOCK_DGRAM, 0));
  if (udp_fd.get() == VGRE_INVALID_SOCKET) return;
  
  // ... rest of function ...
  // Socket automatically closed on all exit paths
}
```

5. Consistent error handling (return VGREResult everywhere):
```cpp
// Convert all error-prone operations to return VGREResult

// Before:
bool send_packet(...) {
  // ...
  return success;
}

// After:
VGREResult send_packet(...) {
  if (fd == VGRE_INVALID_SOCKET) {
    return VGREResult::ERR_INVALID_VALUE;
  }
  
  if (common::InputValidator::validatePacketSize(payloadLen) 
      != VGREResult::SUCCESS) {
    return VGREResult::ERR_INVALID_VALUE;
  }
  
  // ... packet construction ...
  
  bool success = /* send logic */;
  return success ? VGREResult::SUCCESS : VGREResult::ERR_IO;
}

// Update all call sites:
// Before:
if (!send_packet(...)) {
  return VGREResult::ERR_IO;
}

// After:
VGREResult r = send_packet(...);
if (r != VGREResult::SUCCESS) {
  return r;
}
```

6. Replace busy-wait with blocking I/O:
```cpp
// Before (busy-wait):
while (rx.size() < expectedLen) {
  if (std::chrono::steady_clock::now() > deadline) {
    return VGREResult::ERR_TIMEOUT;
  }
  int n = recv(fd, buf, toRead, 0);
  if (n > 0) {
    rx.insert(rx.end(), buf, buf + n);
  } else if (n == 0) {
    return VGREResult::ERR_IO;
  } else {
    if (vgre_is_would_block(...)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }
    return VGREResult::ERR_IO;
  }
}

// After (blocking I/O with poll):
while (rx.size() < expectedLen) {
  auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
    deadline - std::chrono::steady_clock::now()).count();
  
  if (remaining <= 0) {
    return VGREResult::ERR_TIMEOUT;
  }
  
  vgre_pollfd pfd;
  pfd.fd = fd;
  pfd.events = POLLIN;
  pfd.revents = 0;
  
  int poll_res = vgre_poll(&pfd, 1, static_cast<int>(remaining));
  
  if (poll_res < 0) {
    if (!vgre_is_would_block(vgre_get_last_socket_error())) {
      return VGREResult::ERR_IO;
    }
    continue;
  }
  
  if (poll_res == 0) {
    return VGREResult::ERR_TIMEOUT;
  }
  
  if (pfd.revents & (POLLERR | POLLHUP)) {
    return VGREResult::ERR_IO;
  }
  
  int n = recv(fd, buf, toRead, 0);
  if (n > 0) {
    rx.insert(rx.end(), buf, buf + n);
  } else if (n == 0) {
    return VGREResult::ERR_IO;
  } else {
    if (!vgre_is_would_block(vgre_get_last_socket_error())) {
      return VGREResult::ERR_IO;
    }
  }
}
```


7. Add exponential backoff for delta-sync retry:
```cpp
// Add ExponentialBackoff class
class ExponentialBackoff {
public:
  ExponentialBackoff(int initial_ms, int max_ms, double multiplier = 2.0)
    : initial_ms_(initial_ms), max_ms_(max_ms), 
      multiplier_(multiplier), current_ms_(initial_ms) {}
  
  int next() {
    int result = current_ms_;
    current_ms_ = std::min(
      static_cast<int>(current_ms_ * multiplier_), 
      max_ms_
    );
    return result;
  }
  
  void reset() {
    current_ms_ = initial_ms_;
  }
  
private:
  int initial_ms_;
  int max_ms_;
  double multiplier_;
  int current_ms_;
};

// Use in sendDeltaSyncWithRetry:
VGREResult TCPClusterManager::sendDeltaSyncWithRetry(
    void* ptr, uint64_t handle,
    const std::vector<std::pair<size_t, size_t>>& dirtyRanges,
    std::shared_ptr<ClientConnection> client) {
  
  ExponentialBackoff backoff(
    Constants::INITIAL_RETRY_BACKOFF_MS,
    Constants::MAX_RETRY_BACKOFF_MS);
  
  for (int attempt = 0; 
       attempt < Constants::MAX_DELTA_SYNC_RETRIES; 
       ++attempt) {
    
    VGREResult r = sendDeltaSync(ptr, handle, dirtyRanges, client);
    
    if (r == VGREResult::SUCCESS) {
      return VGREResult::SUCCESS;
    }
    
    if (r == VGREResult::ERR_IO) {
      // Connection lost - don't retry
      return r;
    }
    
    // Transient failure - retry with backoff
    VGRE_LOG_WARN("TCPCluster", 
      "Delta-sync attempt " + std::to_string(attempt + 1) + 
      " failed, retrying...");
    
    std::this_thread::sleep_for(
      std::chrono::milliseconds(backoff.next())
    );
  }
  
  // All retries exhausted - fall back to full sync
  VGRE_LOG_INFO("TCPCluster", 
    "Delta-sync failed after " + 
    std::to_string(Constants::MAX_DELTA_SYNC_RETRIES) +
    " retries, falling back to full sync");
  
  auto& mm = core::RuntimeEngine::instance().getMemoryManager();
  size_t size = mm.getAllocationSize(ptr);
  return sendFullSync(ptr, handle, size, client);
}
```

8. Early authentication validation:
```cpp
// In handleRemoteCommand, validate auth FIRST:
void TCPClusterManager::handleRemoteCommand(
    const RemoteCommandPacket &pkt) {
  
  // CRITICAL: Validate auth token BEFORE any processing
  if (auth_token_ == 0 || pkt.auth_token != auth_token_) {
    VGRE_LOG_ERROR("TCPCluster",
      "Rejected remote command: invalid auth token");
    // Clear any pending args to prevent data leakage
    pending_args_.clear();
    return;
  }
  
  // Validate packet structure
  if (pkt.grid_dim[0] == 0 || pkt.block_dim[0] == 0) {
    VGRE_LOG_ERROR("TCPCluster", 
      "Invalid remote command packet");
    pending_args_.clear();
    return;
  }
  
  // Now safe to process kernel launch...
  VGRE_LOG_INFO("TCPCluster",
    "Executing remote kernel launch request (Kernel ID: " +
    std::to_string(pkt.kernel_id) + ")");
  
  // ... rest of function ...
}

// Similarly in handlePartitionDispatch:
void TCPClusterManager::handlePartitionDispatch(
    const PartitionDispatchPacket &pkt) {
  
  // CRITICAL: Validate auth token FIRST
  if (auth_token_ == 0 || pkt.auth_token != auth_token_) {
    VGRE_LOG_ERROR("TCPCluster",
      "Rejected partition dispatch: invalid auth token");
    pending_args_.clear();
    return;
  }
  
  // ... rest of function ...
}
```

9. Preserve buffer contents for diagnostics:
```cpp
// Before (loses diagnostic info):
if (header->magic != VSBP_MAGIC || header->version != VSBP_VERSION) {
  VGRE_LOG_ERROR("TCPCluster", "VSBP protocol violation");
  client->rx_buffer.clear();
  break;
}

// After (preserve for diagnostics):
if (header->magic != VSBP_MAGIC || header->version != VSBP_VERSION) {
  // Log first 64 bytes for diagnostics
  std::string hex_dump;
  size_t dump_size = std::min(client->rx_buffer.size(), size_t(64));
  for (size_t i = 0; i < dump_size; ++i) {
    char buf[4];
    std::snprintf(buf, sizeof(buf), "%02x ", client->rx_buffer[i]);
    hex_dump += buf;
  }
  
  VGRE_LOG_ERROR("TCPCluster",
    "VSBP protocol violation from " + client->ip_address +
    " - magic=" + std::to_string(header->magic) +
    " version=" + std::to_string(header->version) +
    " buffer_size=" + std::to_string(client->rx_buffer.size()) +
    " hex_dump=" + hex_dump);
  
  // Now safe to clear
  client->rx_buffer.clear();
  break;
}
```


### Phase 5: Refactor into Modular Architecture (Priority: MEDIUM)

This phase splits the monolithic 3,267-line file into focused modules. This is a large refactoring that should be done incrementally.

**Step 1: Create Module Structure**

Create new directory structure:
```
src/advanced/tcp_cluster/
├── tcp_cluster_manager.cpp
├── connection_manager.cpp
├── packet_handler.cpp
├── security_manager.cpp
├── discovery_manager.cpp
├── dispatch_manager.cpp
├── memory_sync_manager.cpp
└── collective_ops_manager.cpp

include/vgre/advanced/tcp_cluster/
├── tcp_cluster_manager.h (public API)
└── internal/
    ├── connection_manager.h
    ├── packet_handler.h
    ├── security_manager.h
    ├── discovery_manager.h
    ├── dispatch_manager.h
    ├── memory_sync_manager.h
    └── collective_ops_manager.h
```

**Step 2: Extract ConnectionManager**

Move all client connection lifecycle code to `connection_manager.cpp`:
- `acceptConnection()` - extracted from `serverLoop()`
- `connectToMaster()` - extracted from `initialize()`
- `closeConnection()` - extracted from `shutdown()`
- `getClient()` - accessor for worker index
- `getActiveClients()` - list of active connections
- `addClientIfNotDuplicate()` - atomic duplicate check
- `purgeDeadClients()` - cleanup dead connections

**Step 3: Extract PacketHandler**

Move all packet I/O code to `packet_handler.cpp`:
- `constructPacket()` - unified packet construction
- `sendPacket()` - enqueue to TSS2
- `sendPacketDirect()` - bypass queue
- `recvPacket()` - receive with optional encryption
- `parseVSBPHeader()` - header validation
- `flush_tx_queues()` - TSS2 priority flush

**Step 4: Extract SecurityManager**

Move all security code to `security_manager.cpp`:
- `enableSecurity()` - toggle security mode
- `isSecurityEnabled()` - query security state
- `getSecurityInfo()` - session info for dashboard
- `performServerHandshake()` - master-side handshake
- `performClientHandshake()` - worker-side handshake
- `rotateSessionKey()` - key rotation

**Step 5: Extract DiscoveryManager**

Move all UDP discovery code to `discovery_manager.cpp`:
- `startMasterAnnouncer()` - UDP broadcast loop
- `startWorkerDiscovery()` - UDP listen loop
- `startProactiveConnections()` - proactive dial-out
- `stopAll()` - shutdown all discovery threads
- `addProactiveAddress()` - add to connection list
- `removeProactiveAddress()` - remove from list

**Step 6: Extract DispatchManager**

Move all kernel dispatch code to `dispatch_manager.cpp`:
- `launchRemoteKernel()` - single-worker dispatch
- `launchPartitionedKernel()` - multi-worker dispatch
- `collectPartitionResults()` - wait for partitions
- `broadcastKernelRegistration()` - register kernel
- `waitForRemoteResult()` - wait for completion
- `handleRemoteCommand()` - worker-side execution
- `handlePartitionDispatch()` - worker-side partition

**Step 7: Extract MemorySyncManager**

Move all memory synchronization code to `memory_sync_manager.cpp`:
- `streamArgumentsToWorker()` - unified arg streaming
- `syncPointerToWorker()` - memory coherence
- `syncPointerFromWorker()` - pull back results
- `initializeShmForClient()` - SHM setup
- `sendDeltaSync()` - incremental sync
- `sendFullSync()` - full memory transfer
- `sendScalarArg()` - scalar argument
- `sendPointerArg()` - pointer argument
- `sendStructArg()` - struct argument

**Step 8: Extract CollectiveOpsManager**

Move all collective operations code to `collective_ops_manager.cpp`:
- `allReduce()` - distributed reduction
- `barrier()` - synchronization barrier
- `sumReduce()` - SIMD-optimized reduction
- `masterAllReduce()` - master-side coordination
- `workerAllReduce()` - worker-side participation

**Step 9: Update TCPClusterManager**

The main `TCPClusterManager` class becomes a thin coordinator:
```cpp
class TCPClusterManager {
public:
  static TCPClusterManager& instance();
  
  // Lifecycle
  VGREResult initialize(bool is_master, const std::string& host, int port);
  void shutdown();
  
  // Delegate to modules
  VGREResult launchRemoteKernel(...) {
    return dispatch_manager_->launchRemoteKernel(...);
  }
  
  VGREResult allReduce(...) {
    return collective_ops_manager_->allReduce(...);
  }
  
  // ... other delegating methods ...
  
private:
  std::unique_ptr<ConnectionManager> connection_manager_;
  std::unique_ptr<PacketHandler> packet_handler_;
  std::unique_ptr<SecurityManager> security_manager_;
  std::unique_ptr<DiscoveryManager> discovery_manager_;
  std::unique_ptr<DispatchManager> dispatch_manager_;
  std::unique_ptr<MemorySyncManager> memory_sync_manager_;
  std::unique_ptr<CollectiveOpsManager> collective_ops_manager_;
};
```


### Phase 6: Make Code Testable (Priority: MEDIUM)

**Step 1: Define Interface Abstractions**

Create `include/vgre/advanced/tcp_cluster/internal/interfaces.h`:
```cpp
#pragma once

#include "vgre/common/sockets.h"
#include "vgre/common/error_codes.h"
#include <memory>

namespace vgre {
namespace advanced {
namespace tcp_cluster {

// Socket abstraction for testing
class ISocketFactory {
public:
  virtual ~ISocketFactory() = default;
  
  virtual vgre_socket_t createSocket(
    int domain, int type, int protocol) = 0;
  virtual int bind(
    vgre_socket_t fd, const sockaddr* addr, socklen_t len) = 0;
  virtual int listen(vgre_socket_t fd, int backlog) = 0;
  virtual vgre_socket_t accept(
    vgre_socket_t fd, sockaddr* addr, socklen_t* len) = 0;
  virtual int connect(
    vgre_socket_t fd, const sockaddr* addr, socklen_t len) = 0;
  virtual int send(
    vgre_socket_t fd, const void* buf, size_t len, int flags) = 0;
  virtual int recv(
    vgre_socket_t fd, void* buf, size_t len, int flags) = 0;
  virtual int poll(
    vgre_pollfd* fds, size_t nfds, int timeout) = 0;
  virtual void close(vgre_socket_t fd) = 0;
};

// Real implementation
class RealSocketFactory : public ISocketFactory {
public:
  vgre_socket_t createSocket(int domain, int type, int protocol) override {
    return socket(domain, type, protocol);
  }
  
  int bind(vgre_socket_t fd, const sockaddr* addr, socklen_t len) override {
    return ::bind(fd, addr, len);
  }
  
  // ... implement all methods using real system calls ...
};

// Memory manager abstraction
class IMemoryManager {
public:
  virtual ~IMemoryManager() = default;
  
  virtual size_t getAllocationSize(void* ptr) = 0;
  virtual void* getPointer(void* handle) = 0;
  virtual bool isValidHandle(void* handle) = 0;
  virtual VGREResult allocateManagedAt(
    void* handle, size_t size, void*& actual_ptr) = 0;
  virtual void getDirtyPages(
    void* ptr, std::vector<std::pair<size_t, size_t>>& ranges) = 0;
  virtual void clearDirtyPages(void* ptr) = 0;
};

// Real implementation
class RealMemoryManager : public IMemoryManager {
public:
  size_t getAllocationSize(void* ptr) override {
    return core::RuntimeEngine::instance()
      .getMemoryManager().getAllocationSize(ptr);
  }
  
  // ... implement all methods delegating to RuntimeEngine ...
};

// Secure channel factory abstraction
class ISecureChannelFactory {
public:
  virtual ~ISecureChannelFactory() = default;
  
  virtual std::unique_ptr<SecureChannel> create() = 0;
};

// Real implementation
class RealSecureChannelFactory : public ISecureChannelFactory {
public:
  std::unique_ptr<SecureChannel> create() override {
    return std::make_unique<SecureChannel>();
  }
};

} // namespace tcp_cluster
} // namespace advanced
} // namespace vgre
```

**Step 2: Add Dependency Injection to TCPClusterManager**

Update `tcp_cluster_manager.h`:
```cpp
class TCPClusterManager {
public:
  // Constructor with dependency injection for testing
  TCPClusterManager(
    std::unique_ptr<tcp_cluster::ISocketFactory> socket_factory,
    std::unique_ptr<tcp_cluster::IMemoryManager> memory_manager,
    std::unique_ptr<tcp_cluster::ISecureChannelFactory> security_factory);
  
  // Singleton for production use
  static TCPClusterManager& instance() {
    static TCPClusterManager inst(
      std::make_unique<tcp_cluster::RealSocketFactory>(),
      std::make_unique<tcp_cluster::RealMemoryManager>(),
      std::make_unique<tcp_cluster::RealSecureChannelFactory>()
    );
    return inst;
  }
  
  // ... rest of public API ...
  
private:
  std::unique_ptr<tcp_cluster::ISocketFactory> socket_factory_;
  std::unique_ptr<tcp_cluster::IMemoryManager> memory_manager_;
  std::unique_ptr<tcp_cluster::ISecureChannelFactory> security_factory_;
  
  // Modules also use injected dependencies
  std::unique_ptr<ConnectionManager> connection_manager_;
  std::unique_ptr<PacketHandler> packet_handler_;
  // ... other modules ...
};
```

**Step 3: Create Mock Implementations**

Create `tests/advanced/tcp_cluster/mocks.h`:
```cpp
#pragma once

#include "vgre/advanced/tcp_cluster/internal/interfaces.h"
#include <gmock/gmock.h>
#include <queue>

namespace vgre {
namespace advanced {
namespace tcp_cluster {
namespace test {

class MockSocketFactory : public ISocketFactory {
public:
  MOCK_METHOD(vgre_socket_t, createSocket, (int, int, int), (override));
  MOCK_METHOD(int, bind, (vgre_socket_t, const sockaddr*, socklen_t), (override));
  MOCK_METHOD(int, listen, (vgre_socket_t, int), (override));
  MOCK_METHOD(vgre_socket_t, accept, (vgre_socket_t, sockaddr*, socklen_t*), (override));
  MOCK_METHOD(int, connect, (vgre_socket_t, const sockaddr*, socklen_t), (override));
  MOCK_METHOD(int, send, (vgre_socket_t, const void*, size_t, int), (override));
  MOCK_METHOD(int, recv, (vgre_socket_t, void*, size_t, int), (override));
  MOCK_METHOD(int, poll, (vgre_pollfd*, size_t, int), (override));
  MOCK_METHOD(void, close, (vgre_socket_t), (override));
};

class MockMemoryManager : public IMemoryManager {
public:
  MOCK_METHOD(size_t, getAllocationSize, (void*), (override));
  MOCK_METHOD(void*, getPointer, (void*), (override));
  MOCK_METHOD(bool, isValidHandle, (void*), (override));
  MOCK_METHOD(VGREResult, allocateManagedAt, 
    (void*, size_t, void*&), (override));
  MOCK_METHOD(void, getDirtyPages, 
    (void*, std::vector<std::pair<size_t, size_t>>&), (override));
  MOCK_METHOD(void, clearDirtyPages, (void*), (override));
};

class MockSecureChannelFactory : public ISecureChannelFactory {
public:
  MOCK_METHOD(std::unique_ptr<SecureChannel>, create, (), (override));
};

// Fake socket for integration tests
class FakeSocketFactory : public ISocketFactory {
public:
  vgre_socket_t createSocket(int, int, int) override {
    return next_fd_++;
  }
  
  int bind(vgre_socket_t, const sockaddr*, socklen_t) override {
    return 0; // Success
  }
  
  int connect(vgre_socket_t fd, const sockaddr*, socklen_t) override {
    connected_sockets_.insert(fd);
    return 0;
  }
  
  int send(vgre_socket_t fd, const void* buf, size_t len, int) override {
    if (connected_sockets_.count(fd) == 0) return -1;
    
    std::vector<uint8_t> data(
      static_cast<const uint8_t*>(buf),
      static_cast<const uint8_t*>(buf) + len);
    send_buffers_[fd].push(std::move(data));
    return static_cast<int>(len);
  }
  
  int recv(vgre_socket_t fd, void* buf, size_t len, int) override {
    if (recv_buffers_[fd].empty()) {
      errno = EWOULDBLOCK;
      return -1;
    }
    
    auto& data = recv_buffers_[fd].front();
    size_t to_copy = std::min(len, data.size());
    std::memcpy(buf, data.data(), to_copy);
    
    if (to_copy == data.size()) {
      recv_buffers_[fd].pop();
    } else {
      data.erase(data.begin(), data.begin() + to_copy);
    }
    
    return static_cast<int>(to_copy);
  }
  
  void close(vgre_socket_t fd) override {
    connected_sockets_.erase(fd);
    send_buffers_.erase(fd);
    recv_buffers_.erase(fd);
  }
  
  // Test helpers
  void injectRecvData(vgre_socket_t fd, const std::vector<uint8_t>& data) {
    recv_buffers_[fd].push(data);
  }
  
  std::vector<uint8_t> getSentData(vgre_socket_t fd) {
    if (send_buffers_[fd].empty()) return {};
    auto data = send_buffers_[fd].front();
    send_buffers_[fd].pop();
    return data;
  }
  
private:
  vgre_socket_t next_fd_{100};
  std::set<vgre_socket_t> connected_sockets_;
  std::map<vgre_socket_t, std::queue<std::vector<uint8_t>>> send_buffers_;
  std::map<vgre_socket_t, std::queue<std::vector<uint8_t>>> recv_buffers_;
};

} // namespace test
} // namespace tcp_cluster
} // namespace advanced
} // namespace vgre
```


**Step 4: Write Unit Tests**

Create `tests/advanced/tcp_cluster/test_packet_handler.cpp`:
```cpp
#include "vgre/advanced/tcp_cluster/internal/packet_handler.h"
#include "mocks.h"
#include <gtest/gtest.h>

using namespace vgre::advanced::tcp_cluster;
using namespace vgre::advanced::tcp_cluster::test;
using ::testing::_;
using ::testing::Return;

class PacketHandlerTest : public ::testing::Test {
protected:
  void SetUp() override {
    socket_factory_ = std::make_unique<MockSocketFactory>();
    handler_ = std::make_unique<PacketHandler>(socket_factory_.get());
  }
  
  std::unique_ptr<MockSocketFactory> socket_factory_;
  std::unique_ptr<PacketHandler> handler_;
};

TEST_F(PacketHandlerTest, ConstructPacket_ValidInput_CreatesCorrectHeader) {
  const char* payload = "test";
  size_t payload_len = 4;
  
  auto packet = handler_->constructPacket(
    PacketType::TELEMETRY, payload, payload_len);
  
  ASSERT_GE(packet.size(), sizeof(VSBPHeader) + payload_len);
  
  VSBPHeader* header = reinterpret_cast<VSBPHeader*>(packet.data());
  EXPECT_EQ(header->magic, VSBP_MAGIC);
  EXPECT_EQ(header->version, VSBP_VERSION);
  EXPECT_EQ(header->type, static_cast<uint16_t>(PacketType::TELEMETRY));
  EXPECT_EQ(header->payloadSize, payload_len);
  
  // Verify payload
  const char* packet_payload = 
    reinterpret_cast<const char*>(packet.data() + sizeof(VSBPHeader));
  EXPECT_EQ(std::memcmp(packet_payload, payload, payload_len), 0);
}

TEST_F(PacketHandlerTest, SendPacket_SocketError_ReturnsError) {
  EXPECT_CALL(*socket_factory_, send(_, _, _, _))
    .WillOnce(Return(-1));
  
  const char* payload = "test";
  VGREResult result = handler_->sendPacketDirect(
    100, PacketType::TELEMETRY, payload, 4, nullptr);
  
  EXPECT_EQ(result, VGREResult::ERR_IO);
}

TEST_F(PacketHandlerTest, SendPacket_Success_ReturnsSuccess) {
  EXPECT_CALL(*socket_factory_, send(_, _, _, _))
    .WillRepeatedly(Return(100)); // Return bytes sent
  
  const char* payload = "test";
  VGREResult result = handler_->sendPacketDirect(
    100, PacketType::TELEMETRY, payload, 4, nullptr);
  
  EXPECT_EQ(result, VGREResult::SUCCESS);
}
```

Create `tests/advanced/tcp_cluster/test_connection_manager.cpp`:
```cpp
#include "vgre/advanced/tcp_cluster/internal/connection_manager.h"
#include "mocks.h"
#include <gtest/gtest.h>

using namespace vgre::advanced::tcp_cluster;
using namespace vgre::advanced::tcp_cluster::test;

class ConnectionManagerTest : public ::testing::Test {
protected:
  void SetUp() override {
    socket_factory_ = std::make_unique<FakeSocketFactory>();
    manager_ = std::make_unique<ConnectionManager>(socket_factory_.get());
  }
  
  std::unique_ptr<FakeSocketFactory> socket_factory_;
  std::unique_ptr<ConnectionManager> manager_;
};

TEST_F(ConnectionManagerTest, AddClient_NoDuplicate_Success) {
  vgre_socket_t fd1 = socket_factory_->createSocket(AF_INET, SOCK_STREAM, 0);
  vgre_socket_t fd2 = socket_factory_->createSocket(AF_INET, SOCK_STREAM, 0);
  
  bool added1 = manager_->addClientIfNotDuplicate("192.168.1.100", fd1);
  bool added2 = manager_->addClientIfNotDuplicate("192.168.1.101", fd2);
  
  EXPECT_TRUE(added1);
  EXPECT_TRUE(added2);
  EXPECT_EQ(manager_->getActiveClients().size(), 2);
}

TEST_F(ConnectionManagerTest, AddClient_Duplicate_Rejected) {
  vgre_socket_t fd1 = socket_factory_->createSocket(AF_INET, SOCK_STREAM, 0);
  vgre_socket_t fd2 = socket_factory_->createSocket(AF_INET, SOCK_STREAM, 0);
  
  bool added1 = manager_->addClientIfNotDuplicate("192.168.1.100", fd1);
  bool added2 = manager_->addClientIfNotDuplicate("192.168.1.100", fd2);
  
  EXPECT_TRUE(added1);
  EXPECT_FALSE(added2); // Duplicate rejected
  EXPECT_EQ(manager_->getActiveClients().size(), 1);
}

TEST_F(ConnectionManagerTest, PurgeDeadClients_RemovesInactive) {
  vgre_socket_t fd = socket_factory_->createSocket(AF_INET, SOCK_STREAM, 0);
  manager_->addClientIfNotDuplicate("192.168.1.100", fd);
  
  auto client = manager_->getClient(0);
  ASSERT_NE(client, nullptr);
  
  client->active = false;
  manager_->purgeDeadClients();
  
  EXPECT_EQ(manager_->getActiveClients().size(), 0);
}
```

Create `tests/advanced/tcp_cluster/test_integration.cpp`:
```cpp
#include "vgre/advanced/tcp_cluster_manager.h"
#include "mocks.h"
#include <gtest/gtest.h>
#include <thread>

using namespace vgre::advanced;
using namespace vgre::advanced::tcp_cluster::test;

class TCPClusterIntegrationTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Create isolated instances for parallel testing
    master_socket_factory_ = std::make_unique<FakeSocketFactory>();
    worker_socket_factory_ = std::make_unique<FakeSocketFactory>();
    
    master_ = std::make_unique<TCPClusterManager>(
      std::move(master_socket_factory_),
      std::make_unique<RealMemoryManager>(),
      std::make_unique<RealSecureChannelFactory>()
    );
    
    worker_ = std::make_unique<TCPClusterManager>(
      std::move(worker_socket_factory_),
      std::make_unique<RealMemoryManager>(),
      std::make_unique<RealSecureChannelFactory>()
    );
  }
  
  std::unique_ptr<FakeSocketFactory> master_socket_factory_;
  std::unique_ptr<FakeSocketFactory> worker_socket_factory_;
  std::unique_ptr<TCPClusterManager> master_;
  std::unique_ptr<TCPClusterManager> worker_;
};

TEST_F(TCPClusterIntegrationTest, MasterWorkerConnection_Success) {
  VGREResult r1 = master_->initialize(true, "127.0.0.1", 7777);
  ASSERT_EQ(r1, VGREResult::SUCCESS);
  
  VGREResult r2 = worker_->initialize(false, "127.0.0.1", 7777);
  ASSERT_EQ(r2, VGREResult::SUCCESS);
  
  // Wait for connection
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  
  std::vector<TCPClusterManager::ClusterNodeInfo> nodes;
  master_->getConnectedNodes(nodes);
  
  EXPECT_EQ(nodes.size(), 1);
  EXPECT_TRUE(nodes[0].active);
}

TEST_F(TCPClusterIntegrationTest, ParallelInstances_NoInterference) {
  // Create two separate master-worker pairs
  auto master1 = std::make_unique<TCPClusterManager>(
    std::make_unique<FakeSocketFactory>(),
    std::make_unique<RealMemoryManager>(),
    std::make_unique<RealSecureChannelFactory>()
  );
  
  auto master2 = std::make_unique<TCPClusterManager>(
    std::make_unique<FakeSocketFactory>(),
    std::make_unique<RealMemoryManager>(),
    std::make_unique<RealSecureChannelFactory>()
  );
  
  // Both should initialize without interference
  std::thread t1([&]() {
    VGREResult r = master1->initialize(true, "127.0.0.1", 7777);
    EXPECT_EQ(r, VGREResult::SUCCESS);
  });
  
  std::thread t2([&]() {
    VGREResult r = master2->initialize(true, "127.0.0.1", 7778);
    EXPECT_EQ(r, VGREResult::SUCCESS);
  });
  
  t1.join();
  t2.join();
}
```


## Testing Strategy

### Validation Approach

The testing strategy follows a multi-layered approach: unit tests for individual modules, integration tests for module interactions, and system tests for end-to-end functionality.

### Unit Tests

**Goal**: Verify each module in isolation with mocked dependencies.

**Test Coverage**:
1. **PacketHandler Tests**:
   - Packet construction with valid/invalid inputs
   - Send/receive with socket errors
   - VSBP header validation
   - Priority queue ordering

2. **ConnectionManager Tests**:
   - Duplicate connection detection (TOCTOU fix)
   - Client lifecycle (add, remove, purge)
   - Active client enumeration
   - Thread-safe access

3. **SecurityManager Tests**:
   - Handshake success/failure paths
   - Plaintext fallback prevention
   - Key derivation
   - Session info reporting

4. **MemorySyncManager Tests**:
   - Delta-sync vs full-sync decision
   - SHM vs TCP path selection
   - Retry logic with exponential backoff
   - Argument serialization

5. **CollectiveOpsManager Tests**:
   - allReduce with different datatypes
   - SIMD reduction correctness
   - Timeout handling
   - Master/worker coordination

### Integration Tests

**Goal**: Verify module interactions with fake implementations.

**Test Coverage**:
1. **Master-Worker Connection**:
   - UDP discovery
   - Proactive connection
   - Security handshake
   - Capability exchange

2. **Kernel Dispatch**:
   - Argument streaming
   - Memory synchronization
   - Result collection
   - Error propagation

3. **Partitioned Execution**:
   - Workload partitioning
   - Multi-worker dispatch
   - Result aggregation
   - Partial failure handling

4. **Collective Operations**:
   - allReduce across multiple workers
   - Barrier synchronization
   - Timeout scenarios

### System Tests

**Goal**: Verify end-to-end functionality with real sockets (localhost).

**Test Coverage**:
1. **Full Cluster Lifecycle**:
   - Master initialization
   - Worker discovery and connection
   - Kernel registration and execution
   - Graceful shutdown

2. **Security End-to-End**:
   - Encrypted communication
   - Key rotation
   - Auth token validation
   - Plaintext rejection

3. **Performance Tests**:
   - Throughput measurement
   - Latency measurement
   - Memory overhead
   - CPU utilization

4. **Stress Tests**:
   - Many concurrent kernels
   - Large data transfers
   - Connection churn
   - Resource exhaustion

### Property-Based Tests

**Goal**: Generate random inputs to find edge cases.

**Test Coverage**:
1. **Packet Construction**:
   - Random payload sizes
   - Random packet types
   - Verify VSBP compliance

2. **Memory Synchronization**:
   - Random dirty ranges
   - Random allocation sizes
   - Verify data integrity

3. **Reduction Operations**:
   - Random array sizes
   - Random datatypes
   - Verify associativity and commutativity

### Regression Tests

**Goal**: Ensure fixes don't break existing functionality.

**Test Coverage**:
1. **Protocol Compatibility**:
   - VSBP v0.1.2 format unchanged
   - PacketType enum values unchanged
   - Header structure layout unchanged

2. **Performance Characteristics**:
   - TSS2 priority ordering maintained
   - Flow control limits respected
   - Rate limiting functional

3. **Error Handling**:
   - Socket errors handled correctly
   - Timeout behavior unchanged
   - Memory allocation failures handled

## Implementation Plan

### Phase 1: Missing Implementations (Week 1)
- Implement `reportComputeFromWorker()`
- Implement `allReduce()`
- Add COLLECTIVE_OP packet handling
- Write unit tests for new methods
- **Deliverable**: Linker errors resolved, collective ops functional

### Phase 2: Code Duplication Elimination (Week 2)
- Extract `constructPacket()`
- Extract delta-sync logic
- Extract argument serialization
- Refactor `launchRemoteKernel()` and `launchPartitionedKernel()`
- Write unit tests for extracted functions
- **Deliverable**: Code duplication reduced by 60%

### Phase 3: Stub Replacement (Week 3)
- Implement SIMD-optimized `sum_reduce()`
- Enforce security policy (no plaintext fallback)
- Add AVX2/SSE2 detection
- Write performance benchmarks
- **Deliverable**: Production-ready implementations

### Phase 4: Business Logic Fixes (Week 4-5)
- Add named constants
- Replace all magic numbers
- Fix TOCTOU race with atomic check-and-insert
- Add RAII wrappers (SocketGuard, etc.)
- Standardize error handling to VGREResult
- Replace busy-wait with blocking I/O
- Add exponential backoff for retries
- Early authentication validation
- Preserve buffer contents for diagnostics
- Write unit tests for each fix
- **Deliverable**: All business logic issues resolved

### Phase 5: Modular Architecture (Week 6-8)
- Create module directory structure
- Extract ConnectionManager
- Extract PacketHandler
- Extract SecurityManager
- Extract DiscoveryManager
- Extract DispatchManager
- Extract MemorySyncManager
- Extract CollectiveOpsManager
- Update TCPClusterManager to delegate
- Write integration tests
- **Deliverable**: Monolithic file split into 8 focused modules

### Phase 6: Testability (Week 9-10)
- Define interface abstractions
- Add dependency injection
- Create mock implementations
- Create fake implementations
- Write comprehensive unit tests
- Write integration tests
- Write system tests
- **Deliverable**: 80%+ test coverage, parallel test execution

### Phase 7: Documentation and Cleanup (Week 11)
- Update API documentation
- Add inline comments
- Create architecture diagram
- Write migration guide
- Remove deprecated code
- Final code review
- **Deliverable**: Production-ready, documented codebase

## Success Criteria

1. **Functionality**: All missing methods implemented and tested
2. **Maintainability**: No code duplication, modular architecture
3. **Quality**: All stubs replaced with production code
4. **Reliability**: No race conditions, no resource leaks
5. **Testability**: 80%+ test coverage, parallel test execution
6. **Performance**: SIMD optimizations, efficient I/O
7. **Security**: Enforced security policy, no silent fallbacks
8. **Documentation**: Complete API docs, architecture guide

## Risks and Mitigation

**Risk 1**: Refactoring breaks existing functionality
- **Mitigation**: Comprehensive regression test suite, incremental changes

**Risk 2**: Performance degradation from abstraction layers
- **Mitigation**: Benchmark before/after, inline hot paths

**Risk 3**: Module boundaries unclear
- **Mitigation**: Clear responsibility matrix, code review

**Risk 4**: Test infrastructure complexity
- **Mitigation**: Start with simple mocks, add complexity as needed

**Risk 5**: Timeline slippage
- **Mitigation**: Prioritize critical fixes (Phase 1-4), defer nice-to-haves (Phase 5-6)

