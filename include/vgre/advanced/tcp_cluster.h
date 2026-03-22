#pragma once

#include "vgre/api/vgre_c_api.h"
#include "vgre/common/error_codes.h"
#include "vgre/advanced/secure_channel.h"
#include <atomic>
#include <cstdint>
#include <mutex>
#include <condition_variable>
#if defined(_WIN32)
typedef unsigned long long
    vgre_socket_t; // Abstracting Windows SOCKET (UINT_PTR)
#else
#include <netinet/in.h>
typedef int vgre_socket_t;
#endif
#include <string>
#include <thread>
#include <vector>
#include <map>

namespace vgre {
namespace advanced {

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
  ROTATE_KEY = 16             // Phase 10: dynamic session key rotation
};

struct KernelRegisterPacket {
  PacketType type;
  uint64_t auth_token;
  uint64_t kernel_id;
  char name[64];
  uint32_t source_len;
};

struct TelemetryPacket {
  PacketType type;
  vgre_telemetry_t telemetry;
};

struct RemoteCommandPacket {
  PacketType type;
  uint64_t auth_token;
  uint64_t kernel_id;
  uint32_t grid_dim[3];
  uint32_t block_dim[3];
  size_t shared_mem;
  int num_args;
};

struct ArgScalarPacket {
  PacketType type;
  uint32_t arg_index;
  uint8_t arg_type;
  uint64_t value;
};

struct StructDataPacket {
  PacketType type;
  uint32_t arg_index;
  uint32_t size;
};

struct DataHeaderPacket {
  PacketType type;
  uint64_t target_ptr;
  uint64_t size;
};

struct ResponsePacket {
  PacketType type;
  uint64_t kernel_id;
  VGREResult result;
};

struct CapabilityPacket {
  PacketType type;
  int cpu_cores;
  uint64_t cpu_memory;
  bool has_igpu;
  char igpu_name[64];
};

// Phase 5: Security handshake — exchanges nonces for PBKDF2 key derivation
struct SecureHandshakePacket {
  PacketType type;
  uint8_t nonce[crypto::kNonceLen];
  uint8_t key_verification[crypto::kSHA256DigestLen]; // SHA256(derived_key) for verification
};

// Phase 5: Partition dispatch — sub-grid for distributed kernel execution
struct PartitionDispatchPacket {
  PacketType type;
  uint64_t auth_token;
  uint64_t kernel_id;
  uint32_t full_grid_dim[3];   // original full grid
  uint32_t partition_grid_dim[3]; // this partition's grid
  uint32_t block_dim[3];
  uint32_t block_offset_x;     // X offset for blockIdx remapping
  size_t shared_mem;
  int num_args;
  uint32_t partition_id;
  uint32_t total_partitions;
};

// Phase 5: Partition result — response from a completed partition
struct PartitionResultPacket {
  PacketType type;
  uint64_t kernel_id;
  uint32_t partition_id;
  VGREResult result;
  double execution_time_ms;
};

// Phase 5: Credit report — compute usage billing
struct CreditReportPacket {
  PacketType type;
  double compute_seconds;  // wall-clock execution time
  int cpu_cores;           // cores used during execution
  uint64_t kernel_id;
  uint64_t timestamp;
};

class TCPClusterManager {
public:
  static TCPClusterManager &instance() {
    static TCPClusterManager inst;
    return inst;
  }

  // Initialize as Master (Server) or Client (Worker) Node
  VGREResult initialize(bool is_master, const std::string &host = "127.0.0.1",
                        int port = 7780);
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
  bool send_packet(vgre_socket_t fd, const void *data, size_t len, SecureChannel *sc = nullptr);
  int recv_packet(vgre_socket_t fd, std::vector<uint8_t> &outBuffer, SecureChannel *sc = nullptr);
  void reportComputeFromWorker(double seconds, int cores, uint64_t kernel_id);


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
  
  struct ClientConnection {
    vgre_socket_t socket_fd;
    vgre_telemetry_t last_telemetry;
    bool active;
    std::vector<uint8_t> rx_buffer;
    bool expecting_type = true;
    PacketType pending_type = PacketType::TELEMETRY;
    uint64_t pending_target_ptr = 0;
    uint64_t pending_data_size = 0;
    int cpu_cores = 0;
    uint64_t cpu_memory = 0;
    bool has_igpu = false;
    char igpu_name[64] = {};
    std::string ip_address;
    std::unique_ptr<SecureChannel> secureChannel;
    bool security_established = false;
    uint32_t packets_sent = 0; // Phase 10: for rotation trigger
  };

  struct ClusterNodeInfo {
    std::string ip_address;
    vgre_telemetry_t last_telemetry;
    bool active;
    int cpu_cores;
    uint64_t cpu_memory;
    bool has_igpu;
    char igpu_name[64];
    bool security_established;
  };

  void getConnectedNodes(std::vector<ClusterNodeInfo> &outNodes) const;

  bool isEnabled() const { return enabled_.load(); }
  bool isMaster() const { return is_master_; }
  bool isWorker() const { return !is_master_; }

  TCPClusterManager();
  ~TCPClusterManager();

private:
  // Socket logic
  void serverLoop();
  void clientLoop();
  void handleRemoteCommand(const RemoteCommandPacket &pkt);
  void handlePartitionDispatch(const PartitionDispatchPacket &pkt);
  
  // UDP Auto-Discovery
  void udpAnnouncerLoop();  // Master
  void udpDiscoveryLoop();  // Client
  void processClientStagingBuffer(); // Client data processor

  // Phase 5: Security handshake
  VGREResult performSecureHandshake(ClientConnection &client);
  VGREResult performClientSecureHandshake();

  std::atomic<bool> enabled_{false};
  std::atomic<bool> security_enabled_{false};
  uint64_t auth_token_ = 0;
  std::string auth_token_str_; // raw token string for PBKDF2
  
  // Worker-side state for incoming data
  uint64_t pending_target_ptr_ = 0;
  uint32_t pending_data_size_ = 0;

  bool is_master_ = false;
  int port_ = 7777;
  std::string host_;

  // Threading
  std::thread cluster_thread_;
  std::thread udp_thread_;
  std::thread data_processor_thread_;

  // Master State
  vgre_socket_t server_fd_ = (vgre_socket_t)-1;
  std::vector<ClientConnection> clients_;
  mutable std::mutex clients_mutex_;

  // Client State
  vgre_socket_t client_fd_ = (vgre_socket_t)-1;
  vgre_telemetry_t client_telemetry_buffer_{};
  std::unique_ptr<SecureChannel> client_secure_channel_;
  bool client_security_established_ = false;
  std::vector<uint8_t> client_rx_buffer_;
  std::mutex client_mutex_;

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

  // Phase 5: Partition results collected on master
  struct PartitionResult {
    uint32_t partition_id;
    VGREResult result;
    double execution_time_ms;
  };
  std::vector<PartitionResult> partition_results_;
  std::mutex partition_mutex_;
  std::condition_variable partition_cv_;
};

} // namespace advanced
} // namespace vgre
