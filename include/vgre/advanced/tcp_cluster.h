#pragma once

#include "vgre/api/vgre_c_api.h"
#include "vgre/common/error_codes.h"
#include "vgre/common/types.h"
#include "vgre/advanced/secure_channel.h"
#include "vgre/core/shm_manager.h"
#include <atomic>
#include <cstdint>
#include <mutex>
#include <condition_variable>
#include "vgre/common/sockets.h"
#include <string>
#include <thread>
#include <vector>
#include <deque>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <chrono>
#include <memory>

namespace vgre {
namespace advanced {

// Forward declarations for modular architecture
class ConnectionManager;
class DiscoveryManager;
class PacketHandler;
class SecurityManager;
class MemorySyncManager;
class CollectiveOpsManager;
class DispatchManager;

// ── VGRE Structured Binary Protocol (VSBP) v0.1.2 ─────────────────────────
constexpr uint32_t VSBP_MAGIC = 0x56475245; // 'VGRE'
constexpr uint16_t VSBP_VERSION = 0x0102;   // v0.1.2

enum class PacketType : uint32_t {
  TELEMETRY = 1,
  LAUNCH_KERNEL = 2,
  RESPONSE = 3,
  DATA_HEADER = 4,   // size and target pointer
  DATA_BODY = 5,      // raw bytes
  STRUCT_DATA = 6,    // size and arg index
  ARG_SCALAR = 7,     // scalar arg (8 bytes)
  ARG_POINTER = 8,    // pointer arg (handle)
  CAPABILITY = 9,     // per-node hardware info
  REGISTER_KERNEL = 10,
  // Phase 5: Global Compute Network
  SECURE_HANDSHAKE = 11,      // nonce exchange for key derivation
  SECURE_HANDSHAKE_ACK = 12,  // handshake acknowledgment
  PARTITION_DISPATCH = 13,    // sub-grid dispatch for partitioned kernel
  PARTITION_RESULT = 14,      // result from a partition execution
  CREDIT_REPORT = 15,         // compute-unit-seconds billing report
  ROTATE_KEY = 16,            // Phase 10: dynamic session key rotation
  SHM_INIT = 17,              // Phase 11: negotiate SHM segment
  DATA_SHM = 18,              // Phase 11: data in SHM
  DATA_HEADER_DIRTY = 19,     // Phase 11: dirty ranges header (TCP)
  DATA_SHM_DIRTY = 20,        // Phase 11: dirty ranges header (SHM)
  DIRTY_RANGE = 21,           // Phase 11: offset/size range packet
  COOP_BARRIER_SYNC = 22,     // Phase 13: Decentralized Grid Barrier
  COOP_BARRIER_RESUME = 23,   // Phase 13: Master resume signal
  RAW_DATA = 24,              // Generic unstructured payload
  COLLECTIVE_OP = 25,         // Phase 14: Collective Ops (all_reduce, etc.)
  COLLECTIVE_COMPLETE = 26,   // Phase 14: Master signal that reduction is ready
};

struct VSBPHeader {
  uint32_t magic;      // Explicit 32-bit magic
  uint16_t version;    // Platform-independent versioning
  uint16_t type;       // PacketType as uint16
  uint32_t sequence;   // Message sequencing
  uint64_t payloadSize; // 64-bit size for massive transfers
};

enum class ReceiveState : uint8_t {
  IDLE = 0,
  EXPECTING_RANGES_TCP = 1,
  EXPECTING_RANGES_SHM = 2,
  EXPECTING_BODY = 3,
  EXPECTING_KERNEL_SOURCE = 4
};

struct ShmInitPacket {
  char shm_name[64];
  uint64_t shm_size;
};

struct DataShmPacket {
  uint64_t target_ptr;
  uint64_t shm_offset;
  uint64_t size;
};

struct DataHeaderDirtyPacket {
  uint64_t target_ptr;
  uint32_t num_ranges;
};

struct DataShmDirtyPacket {
  uint64_t target_ptr;
  uint32_t num_ranges;
  uint64_t shm_offset;
};

struct DirtyRangePacket {
  uint64_t offset;
  uint64_t size;
};

struct KernelRegisterPacket {
  uint64_t auth_token;
  uint64_t kernel_id;
  char name[64];
  uint32_t source_len;
};

struct TelemetryPacket {
  vgre_telemetry_t telemetry;
};

struct RemoteCommandPacket {
  uint64_t auth_token;
  uint64_t kernel_id;
  uint32_t grid_dim[3];
  uint32_t block_dim[3];
  size_t shared_mem;
  int num_args;
};

struct ArgScalarPacket {
  uint32_t arg_index;
  uint8_t arg_type;
  uint64_t value;
};

struct StructDataPacket {
  uint32_t arg_index;
  uint32_t size;
};

struct DataHeaderPacket {
  uint64_t target_ptr;
  uint64_t size;
};

struct ResponsePacket {
  uint64_t kernel_id;
  VGREResult result;
};

struct CapabilityPacket {
  int cpu_cores;
  uint64_t cpu_memory;
  bool has_igpu;
  char igpu_name[64];
};

struct SecureHandshakePacket {
  uint8_t nonce[crypto::kNonceLen];
  uint8_t key_verification[crypto::kSHA256DigestLen];
};

struct PartitionDispatchPacket {
  uint64_t auth_token;
  uint64_t kernel_id;
  uint32_t full_grid_dim[3];
  uint32_t partition_grid_dim[3];
  uint32_t block_dim[3];
  uint32_t grid_start[3];
  uint64_t shared_mem;
  int num_args;
  uint32_t partition_id;
  uint32_t total_partitions;
};

struct PartitionResultPacket {
  uint64_t kernel_id;
  uint32_t partition_id;
  VGREResult result;
  double execution_time_ms;
};

struct CreditReportPacket {
  double compute_seconds;
  int cpu_cores;
  uint64_t kernel_id;
  uint64_t timestamp;
};

struct CollectiveOpPacket {
  uint32_t op_type;    // 0 = all_reduce
  uint32_t datatype;   // VGRE_ARG_...
  uint64_t count;
  uint64_t sequence;   // sync ID
};

class TCPClusterManager {
public:
  static TCPClusterManager &instance() {
    static TCPClusterManager* inst = new TCPClusterManager();
    return *inst;
  }

  // Initialize as Master (Server) or Client (Worker) Node
  VGREResult initialize(bool is_master, const std::string &host = "127.0.0.1",
                        int port = 7777);
  void shutdown();

  // Telemetry Aggregation
  void broadcastLocalTelemetry(const vgre_telemetry_t &telemetry);
  void aggregateRemoteTelemetry(vgre_telemetry_t &outCombined);

  // Remote Execution
  VGREResult launchRemoteKernel(int worker_idx, uint64_t kernel_id,
                                const uint32_t grid_dim[3],
                                const uint32_t block_dim[3], void **args,
                                int num_args, size_t shared_mem);

  /**
   * @brief Broadcasts a kernel registration to all workers.
   */
  void broadcastKernelRegistration(uint64_t kernel_id, const std::string &name,
                                 const std::string &source);

  int getFirstActiveWorker() const;

  // Phase 5: Secure Packet I/O
  // VSBP v0.1.2: Raw packet dispatch with automatic header construction
  VGREResult send_packet(vgre_socket_t fd, PacketType type, const void* payload, size_t payloadLen, SecureChannel* sc = nullptr);
  VGREResult send_packet_direct(vgre_socket_t fd, PacketType type, const void* payload, size_t payloadLen, SecureChannel* sc = nullptr);
  int recv_packet(vgre_socket_t fd, std::vector<uint8_t> &outBuffer, SecureChannel *sc = nullptr);
  void reportComputeFromWorker(double seconds, int cores, uint64_t kernel_id);
  VGREResult broadcastPacket(PacketType type, const void* payload, size_t payloadLen);


  // ── Phase 5: Security ────────────────────────────────────────────────
  VGREResult enableSecurity(bool enabled);
  bool isSecurityEnabled() const { return security_enabled_.load(); }
  SessionInfo getSecurityInfo() const;

  // ── Phase 5: Partitioned Dispatch ────────────────────────────────────
  VGREResult launchPartitionedKernel(uint64_t kernel_id,
                                     const uint32_t grid_dim[3],
                                     const uint32_t block_dim[3],
                                     void **args, int num_args,
                                     size_t shared_mem);
  VGREResult collectPartitionResults(uint64_t kernel_id,
                                     uint32_t total_partitions,
                                     int timeout_ms = 30000);
  
  // Distributed Collective Operations
  VGREResult allReduce(void* ptr, size_t count, int datatype);
  
  struct ClientConnection {
    vgre_socket_t socket_fd;
    vgre_telemetry_t last_telemetry;
    std::atomic<bool> active{false};
    std::atomic<bool> is_authenticating{false}; // Phase 13: Background handshake guard
    uint64_t handshake_start_ms{0};
    std::vector<uint8_t> rx_buffer;
    bool expecting_type = true;
    PacketType pending_type = PacketType::TELEMETRY;
    uint64_t pending_target_ptr = 0;
    uint64_t pending_data_size = 0;
    uint32_t pending_num_ranges = 0;
    uint64_t pending_shm_offset = 0;
    uint64_t pending_range_offset = 0;
    ReceiveState receive_state = ReceiveState::IDLE;
    int cpu_cores = 0;
    uint64_t cpu_memory = 0;
    bool has_igpu = false;
    char igpu_name[64] = {};
    std::string ip_address;
    int port = 0;
    std::unique_ptr<SecureChannel> secureChannel;
    std::atomic<bool> security_established{false};
    uint32_t packets_sent = 0; // Phase 10: for rotation trigger
    
    // Phase 12: TSS2 (Traffic-Shaping-Sync 2.0)
    struct OutgoingPacket {
      std::vector<uint8_t> data;
      uint32_t priority; // 0 = high, 1 = low
    };
    mutable std::mutex tx_mutex;
    std::deque<OutgoingPacket> high_priority_tx;
    std::deque<OutgoingPacket> low_priority_tx;
    std::atomic<uint32_t> in_flight_kernels{0};
    
    // Phase 11: SHM Support
    bool is_local = false;
    std::unique_ptr<vgre::core::ShmManager> shmManager;
    uint64_t shm_offset = 0;
    std::unordered_set<void*> synced_handles;

    // Measured inter-node bandwidth in Gb/s.
    // Updated by the proactive connection loop after a micro-benchmark probe.
    // Default 10.0 (typical 10GbE); local connections use a large sentinel.
    double network_bandwidth_gbps = 10.0;
  };

  struct ClusterNodeInfo {
    std::string ip_address;
    int port;
    vgre_telemetry_t last_telemetry;
    bool active;
    int cpu_cores;
    uint64_t cpu_memory;
    bool has_igpu;
    char igpu_name[64];
    bool security_established;
    bool is_authenticating;
  };

  void getConnectedNodes(std::vector<ClusterNodeInfo> &outNodes) const;

  // Phase 10: Wait for a specific kernel result from the cluster
  VGREResult waitForRemoteResult(uint64_t kernel_id, int timeout_ms = 30000);

  bool isEnabled() const { return enabled_.load(); }
  bool isMaster() const { return is_master_; }
  bool isWorker() const { return !is_master_; }

  TCPClusterManager();
  ~TCPClusterManager();

private:
  // Friend classes for modular architecture
  friend class ConnectionManager;
  friend class DiscoveryManager;
  friend class SecurityManager;
  friend class CollectiveOpsManager;
  friend class DispatchManager;
  
  // Modular architecture components
  std::unique_ptr<ConnectionManager> connection_manager_;
  std::unique_ptr<DiscoveryManager> discovery_manager_;
  std::unique_ptr<PacketHandler> packet_handler_;
  std::unique_ptr<SecurityManager> security_manager_;
  std::unique_ptr<MemorySyncManager> memory_sync_manager_;
  std::unique_ptr<CollectiveOpsManager> collective_ops_manager_;
  std::unique_ptr<DispatchManager> dispatch_manager_;
  
  // Using vgre::common types
  using vgre_pollfd = vgre::common::vgre_pollfd;
  // ── Connection rate limiter ─────────────────────────────────────────────
  // Prevents handshake-flood DoS: limits new connections per source IP to
  // kMaxPerWindow within a rolling kWindowSeconds window.
  struct ConnectionRateLimiter {
    static constexpr int kMaxPerWindow = 10;
    static constexpr int kWindowSeconds = 60;

    std::mutex mtx;
    std::unordered_map<std::string,
        std::deque<std::chrono::steady_clock::time_point>> attempts;

    bool isAllowed(const std::string &ip) {
      std::lock_guard<std::mutex> lk(mtx);
      auto now = std::chrono::steady_clock::now();
      auto &q = attempts[ip];
      // Prune entries outside the window
      while (!q.empty() &&
             std::chrono::duration_cast<std::chrono::seconds>(now - q.front()).count()
                 >= kWindowSeconds) {
        q.pop_front();
      }
      return static_cast<int>(q.size()) < kMaxPerWindow;
    }

    void record(const std::string &ip) {
      std::lock_guard<std::mutex> lk(mtx);
      attempts[ip].push_back(std::chrono::steady_clock::now());
    }
  };
  ConnectionRateLimiter rateLimiter_;

  // Socket logic
  void serverLoop();
  void clientLoop();
  void handleRemoteCommand(const RemoteCommandPacket &pkt);
  void handlePartitionDispatch(const PartitionDispatchPacket &pkt);
  
  // Packet construction helper
  std::vector<uint8_t> constructPacket(PacketType type, const void* payload, size_t payloadLen);
  
  // Delta-sync helpers
  VGREResult syncPointerToWorker(void* ptr, uint64_t handle, std::shared_ptr<ClientConnection> client);
  VGREResult sendDeltaSync(void* ptr, uint64_t handle, const std::vector<std::pair<size_t, size_t>>& dirtyRanges, std::shared_ptr<ClientConnection> client);
  VGREResult sendDeltaSyncWithRetry(void* ptr, uint64_t handle, const std::vector<std::pair<size_t, size_t>>& dirtyRanges, std::shared_ptr<ClientConnection> client);
  VGREResult sendDeltaSyncSHM(void* ptr, uint64_t handle, const std::vector<std::pair<size_t, size_t>>& dirtyRanges, std::shared_ptr<ClientConnection> client);
  VGREResult sendDeltaSyncTCP(void* ptr, uint64_t handle, const std::vector<std::pair<size_t, size_t>>& dirtyRanges, std::shared_ptr<ClientConnection> client);
  VGREResult sendFullSync(void* ptr, uint64_t handle, size_t size, std::shared_ptr<ClientConnection> client);
  VGREResult sendFullSyncSHM(void* ptr, uint64_t handle, size_t size, std::shared_ptr<ClientConnection> client);
  VGREResult sendFullSyncTCP(void* ptr, uint64_t handle, size_t size, std::shared_ptr<ClientConnection> client);
  
  // Argument serialization helpers
  VGREResult streamArgumentsToWorker(void** args, int num_args, uint64_t kernel_id, std::shared_ptr<ClientConnection> client);
  VGREResult sendStructArg(void* arg, int arg_index, uint64_t kernel_id, std::shared_ptr<ClientConnection> client);
  VGREResult sendPointerArg(void* arg, int arg_index, std::shared_ptr<ClientConnection> client);
  VGREResult sendScalarArg(void* arg, int arg_index, ArgType type, std::shared_ptr<ClientConnection> client);
  
  // Blocking I/O helper
  VGREResult waitForData(vgre_socket_t fd, int timeout_ms);
  
  // Diagnostic helper
  std::string hexDump(const uint8_t* data, size_t max_bytes);
  
  // UDP Auto-Discovery
  void udpAnnouncerLoop();       // Master (announces master to workers)
  void udpMasterDiscoveryLoop();  // Master (discovers active workers)
  void udpDiscoveryLoop();       // Worker (discovers master)
  void udpWorkerAnnouncerLoop();  // Worker (announces worker to master)
  void proactiveConnectionLoop(); // Master (proactive worker connections)
  void processClientStagingBuffer(); // Client data processor
  void flush_tx_queues(std::shared_ptr<ClientConnection> client);
  void parseProactiveNodes();

  // Phase 5: Security handshake
  // ── Phase 13: IPC Visibility Support ──────────────────────────────────
  void syncToIPC();

  VGREResult performSecureHandshake(std::shared_ptr<ClientConnection> client);
  VGREResult performClientSecureHandshake();

  std::atomic<bool> enabled_{false};
  std::atomic<bool> security_enabled_{false};
  uint64_t auth_token_ = 0;
  std::string auth_token_str_; // raw token string for PBKDF2
  
  // Worker-side state for incoming data
  uint64_t pending_target_ptr_ = 0;
  uint32_t pending_data_size_ = 0;
  uint32_t pending_num_ranges_ = 0;
  uint64_t pending_shm_offset_ = 0;
  uint64_t pending_range_offset_ = 0;
  uint64_t pending_kernel_id_ = 0;
  std::string pending_kernel_name_;
  uint32_t pending_kernel_source_len_ = 0;
  ReceiveState receive_state_ = ReceiveState::IDLE;
  
  // Collective operation state (master-side)
  uint32_t pending_collective_op_type_ = 0;
  uint32_t pending_collective_datatype_ = 0;
  uint64_t pending_collective_count_ = 0;

  bool is_master_ = false;
  int port_ = 7777;
  std::string host_;

  // Threading
  std::thread cluster_thread_;
  std::thread client_loop_thread_;   // standby worker: clientLoop when master dials in
  std::thread data_processor_thread_;
  std::vector<std::string> proactive_worker_addresses_;

  // Proactive-connection handshake backoff (per remote IP).
  // After consecutive handshake failures the proactive loop backs off
  // exponentially (5 s → 20 s → 60 s → … up to 300 s) to avoid log spam
  // and unnecessary network churn against workers that don't yet support
  // Phase 5 security.  Cleared whenever enableSecurity() is called so the
  // new security state takes effect with a fresh start.
  mutable std::mutex proactive_backoff_mutex_;
  std::map<std::string, int> proactive_fail_counts_;
  std::map<std::string, std::chrono::steady_clock::time_point> proactive_backoff_until_;

  // Master State
  vgre_socket_t server_fd_ = (vgre_socket_t)-1;
  std::vector<std::shared_ptr<ClientConnection>> clients_;
  mutable std::recursive_mutex clients_mutex_;

  // Client State
  vgre_socket_t client_fd_ = (vgre_socket_t)-1;
  vgre_telemetry_t client_telemetry_buffer_{};
  std::unique_ptr<SecureChannel> client_secure_channel_;
  bool client_security_established_ = false;
  std::atomic<bool> is_authenticating_{false};
  uint64_t last_handshake_start_ms_{0};
  std::vector<uint8_t> client_rx_buffer_;
  std::mutex client_mutex_;
  std::mutex client_tx_mutex_;
  std::deque<ClientConnection::OutgoingPacket> client_high_priority_tx_;
  std::deque<ClientConnection::OutgoingPacket> client_low_priority_tx_;
  
  // Phase 11: Client-side SHM
  std::unique_ptr<vgre::core::ShmManager> client_shm_manager_;
  bool client_shm_enabled_ = false;

  // Double-buffered async data receiving
  std::vector<uint8_t> client_rx_staging_A_;
  std::vector<uint8_t> client_rx_staging_B_;
  std::vector<uint8_t>* active_staging_ = &client_rx_staging_A_;
  std::vector<uint8_t>* processing_staging_ = &client_rx_staging_B_;
  std::mutex staging_mutex_;
  std::condition_variable staging_cv_;
  std::atomic<bool> staging_ready_{false};

  // Worker side: storage for incoming arguments
  struct PendingArg {
    uint8_t type;
    uint64_t value;
    std::vector<uint8_t> data;
  };
  std::map<uint32_t, PendingArg> pending_args_;

  // Dispatch result tracking (managed by DispatchManager but kept here for shutdown)
  std::mutex remote_results_mutex_;
  std::condition_variable remote_results_cv_;
  std::mutex partition_mutex_;
  std::condition_variable partition_cv_;

  // Distributed Collective Primitives (moved to CollectiveOpsManager, kept here for backward compat)
  // Note: These are now managed by collective_ops_manager_ but kept as references for friend access
  std::mutex reduction_mutex_;
  std::condition_variable reduction_cv_;
  std::atomic<int> reduction_count_{0};
  std::vector<uint8_t> active_reduction_buffer_;
  bool is_reducing_{false};
  uint32_t reduction_datatype_{0};
  size_t reduction_element_count_{0};
  uint64_t reduction_sequence_{0};
  
  // Phase 12: Global Traffic Telemetry
  mutable std::atomic<uint64_t> global_packets_sent_{0};
  mutable std::atomic<uint64_t> global_bytes_sent_{0};
  mutable std::atomic<uint64_t> global_packets_received_{0};
  mutable std::atomic<uint64_t> global_bytes_received_{0};

  // Cooperative Barrier (Zero-Simulation)
  std::mutex barrier_mutex_;
  uint32_t barrier_count_ = 0;
  std::condition_variable barrier_cv_;
};

} // namespace advanced
} // namespace vgre

// End of file extension for proactive connections
