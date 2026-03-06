#pragma once

#include "vgre/api/vgre_c_api.h"
#include "vgre/common/error_codes.h"
#include "vgre/common/types.h"
#include <atomic>
#include <mutex>
#include <netinet/in.h>
#include <string>
#include <thread>
#include <vector>

namespace vgre {
namespace advanced {

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

  bool isEnabled() const { return enabled_.load(); }
  bool isMaster() const { return is_master_; }

private:
  TCPClusterManager();
  ~TCPClusterManager();

  // Socket logic
  void serverLoop();
  void clientLoop();

  std::atomic<bool> enabled_{false};
  bool is_master_ = false;
  int port_ = 7777;
  std::string host_;

  // Threading
  std::thread cluster_thread_;

  // Master State
  int server_fd_ = -1;
  struct ClientConnection {
    int socket_fd;
    vgre_telemetry_t last_telemetry;
    bool active;
  };
  std::vector<ClientConnection> clients_;
  std::mutex clients_mutex_;

  // Client State
  int client_fd_ = -1;
  vgre_telemetry_t client_telemetry_buffer_{};
  std::mutex client_mutex_;
};

} // namespace advanced
} // namespace vgre
