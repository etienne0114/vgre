#pragma once
// sockets.h must come first on Windows: it includes <winsock2.h> which
// must precede <windows.h>, and also defines ERROR_* macros via winerror.h.
// Keeping it first prevents those macros from stomping on our error_codes.h
// enum values when other headers include both in the wrong order.
#include "vgre/advanced/rdma_transport.h"
#include "vgre/advanced/secure_channel.h"
#include "vgre/advanced/tcp_cluster/internal/diagnostic_logger.h"
#include "vgre/advanced/tcp_cluster_protocol.h"
#include "vgre/advanced/tcp_cluster_validation.h"
#include "vgre/api/vgre_c_api.h"
#include "vgre/common/error_codes.h"
#include "vgre/common/logger.h"
#include "vgre/common/sockets.h"
#include "vgre/common/types.h"
#include "vgre/core/shm_manager.h"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Forward declarations for dependency injection interfaces
namespace vgre {
namespace advanced {

class ISocketFactory;
class IMemoryManager;
class IKernelManager;
class ISecureChannelFactory;
struct ClientConnection;
class ConnectionManager;
class DiscoveryManager;
class PacketHandler;
class SecurityManager;
class MemorySyncManager;
class CollectiveOpsManager;
class DispatchManager;
class WindowsSocketManager;

// VSBP is a binary protocol with no byte-swapping — it assumes little-endian
// hosts on both ends.  This holds for every supported platform (x86-64, ARM64).
// If you port to a big-endian system, add htons/htonl/htobe wrappers to every
// VSBPHeader and packet-struct field before removing this assertion.
#if defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__)
static_assert(__BYTE_ORDER__ != __ORDER_BIG_ENDIAN__,
              "VSBP protocol assumes little-endian host; "
              "add byte-swap wrappers before enabling big-endian support.");
#endif

class TCPClusterManager {
public:
  static TCPClusterManager &instance();
  VGREResult initialize(bool is_master, const std::string &host = "127.0.0.1",
                        int port = 7777);
  void shutdown();
  static VGREResult validateMemoryAlignment();
  VGREResult initializeMeshTopology() {
    return MeshTopologyManager::initializeMeshTopology(this);
  }
  VGREResult addMeshPeer(const std::string &ip_address, int port = 7777);
  struct MeshTopologyStatus {
    detail::PlatformType local_platform;
    std::string local_platform_name;
    size_t total_peers;
    size_t active_peers;
    size_t cross_platform_peers;
    bool mesh_enabled;
  };
  MeshTopologyStatus getMeshTopologyStatus() const;
  void broadcastLocalTelemetry(const vgre_telemetry_t &telemetry);
  void aggregateRemoteTelemetry(vgre_telemetry_t &outCombined);
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
  // Returns the index of the active worker with GPU capability and fewest
  // in-flight kernels. Falls back to the first active worker when no GPU
  // worker is available. Use this instead of getFirstActiveWorker() when
  // dispatching GPU-intensive kernels.
  int getGpuCapableWorker() const;
  VGREResult send_packet(vgre_socket_t fd, PacketType type, const void *payload,
                         size_t payloadLen, SecureChannel *sc = nullptr);
  VGREResult send_packet_direct(vgre_socket_t fd, PacketType type,
                                const void *payload, size_t payloadLen,
                                SecureChannel *sc = nullptr);
  int recv_packet(vgre_socket_t fd, std::vector<uint8_t> &outBuffer,
                  SecureChannel *sc = nullptr);
  void reportComputeFromWorker(double seconds, int cores, uint64_t kernel_id);
  VGREResult broadcastPacket(PacketType type, const void *payload,
                             size_t payloadLen);
  /**
   * Generate comprehensive operational report with metrics and health status
   */
  void generateOperationalReport();
  /**
   * Export metrics for external monitoring systems
   */
  void exportMetricsForMonitoring();
  /**
   * Perform graceful restart with coordination
   * @param preserve_connections Whether to preserve existing connections during
   * restart
   * @return VGREResult indicating restart success
   */
  VGREResult performGracefulRestart(bool preserve_connections = false);
  /**
   * Check if the cluster is ready for restart
   * @return True if restart can be performed safely
   */
  bool isReadyForRestart() const;
  /**
   * Coordinate restart with peer nodes in mesh topology
   * @param restart_delay_ms Delay before restart to allow coordination
   * @return VGREResult indicating coordination success
   */
  VGREResult coordinateRestartWithPeers(uint32_t restart_delay_ms = 5000);
  VGREResult enableSecurity(bool enabled);
  bool isSecurityEnabled() const { return security_enabled_.load(); }
  SessionInfo getSecurityInfo() const;
  VGREResult launchPartitionedKernel(uint64_t kernel_id,
                                     const uint32_t grid_dim[3],
                                     const uint32_t block_dim[3], void **args,
                                     int num_args, size_t shared_mem);
  VGREResult collectPartitionResults(uint64_t kernel_id,
                                     uint32_t total_partitions,
                                     int timeout_ms = 30000);
  VGREResult allReduce(void *ptr, size_t count, int datatype,
                       ReductionOp op = ReductionOp::Sum);
  VGREResult barrier();
  struct ClientConnection {
    vgre_socket_t socket_fd;
    vgre_telemetry_t last_telemetry;
    std::atomic<bool> active{false};
    std::atomic<bool> is_authenticating{false};
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
    char platform_name[32] = {};   // reported OS: "Linux"/"macOS"/"Windows"
    char arch_name[16] = {};       // reported arch: "x86_64"/"arm64"
    char node_hostname[64] = {};   // reported hostname
    bool capability_received = false;
    std::string ip_address;
    int port = 0;
    std::unique_ptr<SecureChannel> secure_channel;
    std::atomic<bool> security_established{false};
    bool security_enabled = true;
    uint32_t packets_sent = 0;
    detail::PlatformType remote_platform = detail::PlatformType::UNKNOWN;
    bool is_mesh_peer = false;
    bool can_be_master = true;
    bool can_be_worker = true;
    std::string platform_version;
    uint32_t protocol_capabilities = 0;
    bool supports_rdma = false;
    bool supports_shm = false;
    bool supports_security = true;
    bool supports_collective_ops = false;
    std::set<std::string> reachable_peers;
    double connection_quality = 1.0;
    uint64_t last_heartbeat_ms = 0;
    std::chrono::steady_clock::time_point last_activity_time{std::chrono::steady_clock::now()};
    // MT.6: Estimated clock offset of the remote peer relative to local clock.
    // Positive = remote is ahead by this many microseconds.
    // Populated by performClockSync() after the security handshake.
    int64_t clock_offset_us{0};
    // Pending T1 for clock sync computation (set when CLOCK_SYNC is sent).
    int64_t clock_sync_t1_us{0};
    struct OutgoingPacket {
      std::vector<uint8_t> data;
      uint32_t priority;
    };
    mutable std::mutex tx_mutex;
    std::deque<OutgoingPacket> high_priority_tx;
    std::deque<OutgoingPacket> low_priority_tx;
    std::atomic<uint32_t> in_flight_kernels{0};
    // Cumulative count of kernels this worker has actually executed and returned
    // a result for — the proof a node is *used*, not merely connected. Bumped on
    // each RESPONSE / PARTITION_RESULT received from the worker.
    std::atomic<uint64_t> kernels_completed{0};
    bool is_local = false;
    std::unique_ptr<vgre::core::ShmManager> shm_manager;
    uint64_t shm_offset = 0;
    std::unordered_set<void *> synced_handles;
    double network_bandwidth_gbps = 10.0;
    std::chrono::steady_clock::time_point bandwidth_probe_start{};
    bool bandwidth_probe_in_flight = false;
    std::chrono::steady_clock::time_point last_bandwidth_probe_time{};
    double network_latency_ms = 1.0;
    std::unique_ptr<RDMAContext> rdma_ctx;
    std::unique_ptr<RDMAConnection> rdma_conn;
    bool rdma_connected = false;
    int hmac_failure_count{0};
    std::string effective_auth_token;
    uint64_t rate_window_start_ms{0};
    uint32_t rate_window_count{0};
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
    char platform_name[32];   // reported OS ("" if a legacy worker)
    char arch_name[16];       // reported CPU arch
    char node_hostname[64];   // reported hostname
    uint32_t in_flight_kernels;   // kernels dispatched to this node, not yet returned
    uint64_t kernels_completed;   // cumulative kernels this node has executed
    bool security_established;
    bool is_authenticating;
    int worker_idx;
  };
  void getConnectedNodes(std::vector<ClusterNodeInfo> &outNodes) const;
  void getAggregatedTelemetry(vgre_telemetry_t &outCombined) const;
  VGREResult waitForRemoteResult(uint64_t kernel_id, int timeout_ms = 30000);
  bool isEnabled() const { return enabled_.load(); }
  bool isMaster() const { return is_master_; }
  bool isWorker() const { return !is_master_; }
  TCPClusterManager();
  ~TCPClusterManager();
  TCPClusterManager(std::unique_ptr<ISocketFactory> socket_factory,
                    std::unique_ptr<IMemoryManager> memory_manager,
                    std::unique_ptr<IKernelManager> kernel_manager,
                    std::unique_ptr<ISecureChannelFactory> security_factory);

private:
  friend class ConnectionManager;
  friend class DiscoveryManager;
  friend class SecurityManager;
  friend class CollectiveOpsManager;
  friend class DispatchManager;
  friend class MemorySyncManager;
  std::unique_ptr<ISocketFactory> socket_factory_;
  std::unique_ptr<IMemoryManager> memory_manager_;
  std::unique_ptr<IKernelManager> kernel_manager_;
  std::unique_ptr<ISecureChannelFactory> security_factory_;
  std::unique_ptr<ConnectionManager> connection_manager_;
  std::unique_ptr<DiscoveryManager> discovery_manager_;
  std::unique_ptr<PacketHandler> packet_handler_;
  std::unique_ptr<SecurityManager> security_manager_;
  std::unique_ptr<MemorySyncManager> memory_sync_manager_;
  std::unique_ptr<CollectiveOpsManager> collective_ops_manager_;
  std::unique_ptr<DispatchManager> dispatch_manager_;
  std::unique_ptr<WindowsSocketManager> windows_socket_manager_;
  using vgre_pollfd = vgre::common::vgre_pollfd;
  struct ConnectionRateLimiter {
    static int getMaxPerWindow() {
      static const int v = []() -> int {
        const char *e = std::getenv("VGRE_CLUSTER_MAX_CONN_PER_WINDOW");
        if (e) {
          int v = std::atoi(e);
          if (v > 0 && v <= 10000)
            return v;
        }
        return 10;
      }();
      return v;
    }
    static int getWindowSeconds() {
      static const int v = []() -> int {
        const char *e = std::getenv("VGRE_CLUSTER_CONN_WINDOW_SEC");
        if (e) {
          int v = std::atoi(e);
          if (v > 0 && v <= 3600)
            return v;
        }
        return 60;
      }();
      return v;
    }
    std::mutex mtx;
    std::unordered_map<std::string,
                       std::deque<std::chrono::steady_clock::time_point>>
        attempts;
    bool isAllowed(const std::string &ip) {
      std::lock_guard<std::mutex> lk(mtx);
      auto now = std::chrono::steady_clock::now();
      auto &q = attempts[ip];
      const int window = getWindowSeconds();
      while (!q.empty() &&
             std::chrono::duration_cast<std::chrono::seconds>(now - q.front())
                     .count() >= window) {
        q.pop_front();
      }
      return static_cast<int>(q.size()) < getMaxPerWindow();
    }
    void record(const std::string &ip) {
      std::lock_guard<std::mutex> lk(mtx);
      attempts[ip].push_back(std::chrono::steady_clock::now());
    }
  };
  ConnectionRateLimiter rateLimiter_;
  void serverLoop();
  void cleanupServerAuthThreads();
  void performServerMaintenance();
  void handleNewInboundConnection();
  void handleClientDataEvents(std::vector<vgre_pollfd> &fds);
  void processServerPackets(std::shared_ptr<ClientConnection> client);
  void clientLoop();
  void handleRemoteCommand(const RemoteCommandPacket &pkt);
  void handlePartitionDispatch(const PartitionDispatchPacket &pkt);
  std::vector<uint8_t> constructPacket(PacketType type, const void *payload,
                                       size_t payloadLen);
  std::vector<uint8_t>
  constructMeshPacket(PacketType type, const void *payload, size_t payloadLen,
                      detail::PlatformType source_platform);
  VGREResult syncPointerToWorker(void *ptr, uint64_t handle,
                                 std::shared_ptr<ClientConnection> client);
  VGREResult
  sendDeltaSync(void *ptr, uint64_t handle,
                const std::vector<std::pair<size_t, size_t>> &dirtyRanges,
                std::shared_ptr<ClientConnection> client);
  VGREResult sendDeltaSyncWithRetry(
      void *ptr, uint64_t handle,
      const std::vector<std::pair<size_t, size_t>> &dirtyRanges,
      std::shared_ptr<ClientConnection> client);
  VGREResult
  sendDeltaSyncSHM(void *ptr, uint64_t handle,
                   const std::vector<std::pair<size_t, size_t>> &dirtyRanges,
                   std::shared_ptr<ClientConnection> client);
  VGREResult
  sendDeltaSyncTCP(void *ptr, uint64_t handle,
                   const std::vector<std::pair<size_t, size_t>> &dirtyRanges,
                   std::shared_ptr<ClientConnection> client);
  VGREResult sendFullSync(void *ptr, uint64_t handle, size_t size,
                          std::shared_ptr<ClientConnection> client);
  VGREResult sendFullSyncSHM(void *ptr, uint64_t handle, size_t size,
                             std::shared_ptr<ClientConnection> client);
  VGREResult sendFullSyncTCP(void *ptr, uint64_t handle, size_t size,
                             std::shared_ptr<ClientConnection> client);
  VGREResult streamArgumentsToWorker(void **args, int num_args,
                                     uint64_t kernel_id,
                                     std::shared_ptr<ClientConnection> client);
  VGREResult sendStructArg(void *arg, int arg_index, uint64_t kernel_id,
                           std::shared_ptr<ClientConnection> client);
  VGREResult sendPointerArg(void *arg, int arg_index,
                            std::shared_ptr<ClientConnection> client);
  VGREResult sendScalarArg(void *arg, int arg_index, ArgType type,
                           std::shared_ptr<ClientConnection> client);
  VGREResult waitForData(vgre_socket_t fd, int timeout_ms);
  std::string hexDump(const uint8_t *data, size_t max_bytes);
  void processClientStagingBuffer();
  void flush_tx_queues(std::shared_ptr<ClientConnection> client);
  void parseProactiveNodes();

public:
  void parseMeshPeers();

private:
  void syncToIPC();
  VGREResult performSecureHandshake(std::shared_ptr<ClientConnection> client);
  VGREResult performClientSecureHandshake();
  VGREResult performPeerClientHandshake(std::shared_ptr<ClientConnection> peer);
  std::atomic<bool> enabled_{false};
  std::atomic<bool> security_enabled_{false};
  uint64_t auth_token_ = 0;
  mutable std::recursive_mutex auth_token_mutex_;
  std::string auth_token_str_;
  uint64_t pending_target_ptr_ = 0;
  uint32_t pending_data_size_ = 0;
  uint32_t pending_num_ranges_ = 0;
  uint64_t pending_shm_offset_ = 0;
  uint64_t pending_range_offset_ = 0;
  uint64_t pending_kernel_id_ = 0;
  std::string pending_kernel_name_;
  uint32_t pending_kernel_source_len_ = 0;
  uint32_t pending_struct_arg_index_ = 0;
  uint32_t pending_struct_arg_size_ = 0;
  ReceiveState receive_state_ = ReceiveState::IDLE;
  uint32_t pending_collective_op_type_ = 0;
  uint32_t pending_collective_datatype_ = 0;
  uint64_t pending_collective_count_ = 0;
  bool is_master_ = false;
  int port_ = 7777;
  std::string host_;
  std::thread cluster_thread_;
  std::thread client_loop_thread_;
  std::thread data_processor_thread_;
  std::thread monitoring_thread_;
  std::vector<std::string> proactive_worker_addresses_;
  std::set<std::string> mesh_peer_ips_;
  bool mesh_topology_enabled_ = false;
  detail::PlatformType local_platform_;
  std::string local_platform_name_;
  mutable std::mutex mesh_topology_mutex_;
  struct MeshPeerInfo {
    std::string ip_address;
    int port;
    detail::PlatformType platform;
    bool is_active;
    bool can_be_master;
    bool can_be_worker;
    std::chrono::steady_clock::time_point last_seen;
  };
  std::map<std::string, MeshPeerInfo> mesh_peers_;
  mutable std::mutex proactive_backoff_mutex_;
  std::map<std::string, int> proactive_fail_counts_;
  std::map<std::string, std::chrono::steady_clock::time_point>
      proactive_backoff_until_;
  struct AuthEntry {
    std::thread t;
    std::shared_ptr<std::atomic<bool>> done;
  };
  std::vector<AuthEntry> server_auth_threads_;
  std::mutex server_auth_mutex_;
  vgre_socket_t server_fd_ = (vgre_socket_t)-1;
  std::vector<std::shared_ptr<ClientConnection>> clients_;
  mutable std::recursive_mutex clients_mutex_;
  vgre_socket_t client_fd_ = (vgre_socket_t)-1;
  std::atomic<bool> has_master_fd_{false};
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
  std::unique_ptr<vgre::core::ShmManager> client_shm_manager_;
  bool client_shm_enabled_ = false;
  std::unique_ptr<RDMAContext> client_rdma_ctx_;
  std::unique_ptr<RDMAConnection> client_rdma_conn_;
  bool client_rdma_connected_ = false;
  std::vector<uint8_t> client_rx_staging_A_;
  std::vector<uint8_t> client_rx_staging_B_;
  std::vector<uint8_t> *active_staging_ = &client_rx_staging_A_;
  std::vector<uint8_t> *processing_staging_ = &client_rx_staging_B_;
  std::mutex staging_mutex_;
  
  std::mutex shutdown_mutex_;
  std::condition_variable shutdown_cv_;
  std::condition_variable staging_cv_;
  std::atomic<bool> staging_ready_{false};
  struct PendingArg {
    uint8_t type;
    uint64_t value;
    std::vector<uint8_t> data;
  };
  std::map<uint32_t, PendingArg> pending_args_;
  std::mutex remote_results_mutex_;
  std::condition_variable remote_results_cv_;
  std::mutex partition_mutex_;
  std::condition_variable partition_cv_;
  std::mutex reduction_mutex_;
  std::condition_variable reduction_cv_;
  std::atomic<int> reduction_count_{0};
  std::vector<uint8_t> active_reduction_buffer_;
  bool is_reducing_{false};
  uint32_t reduction_datatype_{0};
  size_t reduction_element_count_{0};
  uint64_t reduction_sequence_{0};
  mutable std::atomic<uint64_t> global_packets_sent_{0};
  mutable std::atomic<uint64_t> global_bytes_sent_{0};
  mutable std::atomic<uint64_t> global_packets_received_{0};
  mutable std::atomic<uint64_t> global_bytes_received_{0};
  std::mutex barrier_mutex_;
  uint32_t barrier_count_ = 0;
  std::condition_variable barrier_cv_;
};
} // namespace advanced
} // namespace vgre
// End of file extension for proactive connections
