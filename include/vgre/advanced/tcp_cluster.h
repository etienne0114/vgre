#pragma once

#include "vgre/api/vgre_c_api.h"
#include "vgre/common/error_codes.h"
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
  REGISTER_KERNEL = 10
};

struct KernelRegisterPacket {
  PacketType type;
  uint64_t kernel_id;
  char name[64];
  uint32_t source_len;
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
  };
  
  void getConnectedNodes(std::vector<ClientConnection> &outNodes) const;

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
  
  // UDP Auto-Discovery
  void udpAnnouncerLoop();  // Master
  void udpDiscoveryLoop();  // Client
  void processClientStagingBuffer(); // Client data processor

  std::atomic<bool> enabled_{false};
  uint64_t auth_token_ = 0;
  
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
};

} // namespace advanced
} // namespace vgre
