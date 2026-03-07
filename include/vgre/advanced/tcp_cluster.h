#pragma once

#include "vgre/api/vgre_c_api.h"
#include "vgre/common/error_codes.h"
#include <atomic>
#include <mutex>
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

namespace vgre {
namespace advanced {

enum class PacketType : uint32_t {
  TELEMETRY = 1,
  LAUNCH_KERNEL = 2,
  RESPONSE = 3
};

struct RemoteCommandPacket {
  PacketType type;
  uint64_t kernel_id;
  uint32_t grid_dim[3];
  uint32_t block_dim[3];
  size_t shared_mem;
  // Simplified: only supporting numeric/static args for now in proto
  double args[8];
  int num_args;
};

class TCPClusterManager {
public:
  static TCPClusterManager &instance() {
    static TCPClusterManager inst;
    return inst;
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

  bool isEnabled() const { return enabled_.load(); }
  bool isMaster() const { return is_master_; }

private:
  TCPClusterManager();
  ~TCPClusterManager();

  // Socket logic
  void serverLoop();
  void clientLoop();
  void handleRemoteCommand(const RemoteCommandPacket &pkt);

  std::atomic<bool> enabled_{false};
  bool is_master_ = false;
  int port_ = 7777;
  std::string host_;

  // Threading
  std::thread cluster_thread_;

  // Master State
  vgre_socket_t server_fd_ = (vgre_socket_t)-1;
  struct ClientConnection {
    vgre_socket_t socket_fd;
    vgre_telemetry_t last_telemetry;
    bool active;
  };
  std::vector<ClientConnection> clients_;
  std::mutex clients_mutex_;

  // Client State
  vgre_socket_t client_fd_ = (vgre_socket_t)-1;
  vgre_telemetry_t client_telemetry_buffer_{};
  std::mutex client_mutex_;
};

} // namespace advanced
} // namespace vgre
