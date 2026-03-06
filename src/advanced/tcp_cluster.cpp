#include "vgre/advanced/tcp_cluster.h"
#include "vgre/common/logger.h"
#include <arpa/inet.h>
#include <cstring>
#include <fcntl.h>
#include <sys/socket.h>
#include <unistd.h>

namespace vgre {
namespace advanced {

TCPClusterManager::TCPClusterManager() {}

TCPClusterManager::~TCPClusterManager() { shutdown(); }

VGREResult TCPClusterManager::initialize(bool is_master,
                                         const std::string &host, int port) {
  if (enabled_.exchange(true))
    return VGREResult::SUCCESS;

  is_master_ = is_master;
  host_ = host;
  port_ = port;

  if (is_master_) {
    // Master Node (Server)
    server_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd_ < 0) {
      VGRE_LOG_ERROR("TCPCluster", "Failed to create socket");
      return VGREResult::ERROR_IO;
    }

    int opt = 1;
    setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt,
               sizeof(opt));

    struct sockaddr_in address;
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port_);

    if (bind(server_fd_, (struct sockaddr *)&address, sizeof(address)) < 0) {
      VGRE_LOG_ERROR("TCPCluster",
                     "Bind failed on port " + std::to_string(port_));
      return VGREResult::ERROR_IO;
    }

    if (listen(server_fd_, 10) < 0) {
      VGRE_LOG_ERROR("TCPCluster", "Listen failed");
      return VGREResult::ERROR_IO;
    }

    // Set non-blocking
    fcntl(server_fd_, F_SETFL, fcntl(server_fd_, F_GETFL, 0) | O_NONBLOCK);

    VGRE_LOG_INFO("TCPCluster",
                  "Master Server Listening on port " + std::to_string(port_));
    cluster_thread_ = std::thread(&TCPClusterManager::serverLoop, this);
  } else {
    // Client Node (Worker)
    client_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (client_fd_ < 0) {
      VGRE_LOG_ERROR("TCPCluster", "Failed to create client socket");
      return VGREResult::ERROR_IO;
    }

    struct sockaddr_in serv_addr;
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(port_);

    if (inet_pton(AF_INET, host_.c_str(), &serv_addr.sin_addr) <= 0) {
      VGRE_LOG_ERROR("TCPCluster",
                     "Invalid address or not supported: " + host_);
      return VGREResult::ERROR_IO;
    }

    if (connect(client_fd_, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) <
        0) {
      VGRE_LOG_ERROR("TCPCluster", "Connection failed to " + host_ +
                                       " (Will run locally only)");
      // Don't fully fail, just don't enable the background loop.
      enabled_ = false;
      close(client_fd_);
      client_fd_ = -1;
      return VGREResult::SUCCESS;
    }

    // Set non-blocking
    fcntl(client_fd_, F_SETFL, fcntl(client_fd_, F_GETFL, 0) | O_NONBLOCK);

    VGRE_LOG_INFO("TCPCluster", "Connected to Remote Master Node at " + host_);
    cluster_thread_ = std::thread(&TCPClusterManager::clientLoop, this);
  }

  return VGREResult::SUCCESS;
}

void TCPClusterManager::shutdown() {
  if (!enabled_.exchange(false))
    return;

  if (cluster_thread_.joinable()) {
    cluster_thread_.join();
  }

  if (is_master_) {
    std::lock_guard<std::mutex> lock(clients_mutex_);
    for (auto &client : clients_) {
      if (client.socket_fd >= 0)
        close(client.socket_fd);
    }
    clients_.clear();
    if (server_fd_ >= 0)
      close(server_fd_);
  } else {
    if (client_fd_ >= 0)
      close(client_fd_);
  }
}

void TCPClusterManager::serverLoop() {
  while (enabled_) {
    // Accept new connections
    struct sockaddr_in address;
    int addrlen = sizeof(address);
    int new_socket =
        accept(server_fd_, (struct sockaddr *)&address, (socklen_t *)&addrlen);

    if (new_socket >= 0) {
      fcntl(new_socket, F_SETFL, fcntl(new_socket, F_GETFL, 0) | O_NONBLOCK);
      std::lock_guard<std::mutex> lock(clients_mutex_);
      clients_.push_back(
          {new_socket, {}, true}); // Initialize with zero'd struct
      std::memset(&clients_.back().last_telemetry, 0, sizeof(vgre_telemetry_t));
      VGRE_LOG_INFO("TCPCluster", "New remote node connected via TCP.");
    }

    // Read telemetry from all clients
    {
      std::lock_guard<std::mutex> lock(clients_mutex_);
      for (auto &client : clients_) {
        if (!client.active)
          continue;

        // Robust read for potentially partial TCP packets
        uint8_t *ptr = reinterpret_cast<uint8_t *>(&client.last_telemetry);
        int total_read = 0;
        const int expected = sizeof(vgre_telemetry_t);

        while (total_read < expected) {
          int valread =
              read(client.socket_fd, ptr + total_read, expected - total_read);
          if (valread > 0) {
            total_read += valread;
          } else if (valread == 0) {
            // Disconnected
            client.active = false;
            close(client.socket_fd);
            client.socket_fd = -1;
            VGRE_LOG_INFO("TCPCluster", "Remote node disconnected.");
            break;
          } else if (errno == EAGAIN || errno == EWOULDBLOCK) {
            // No more data for now
            break;
          } else {
            // Error
            client.active = false;
            close(client.socket_fd);
            client.socket_fd = -1;
            VGRE_LOG_ERROR("TCPCluster", "Read error on remote node.");
            break;
          }
        }
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
}

void TCPClusterManager::clientLoop() {
  while (enabled_) {
    vgre_telemetry_t payload;
    {
      std::lock_guard<std::mutex> lock(client_mutex_);
      payload = client_telemetry_buffer_;
    }

    // Ignore empty/zero payloads if not populated yet
    if (payload.timestamp > 0) {
      send(client_fd_, &payload, sizeof(vgre_telemetry_t), MSG_NOSIGNAL);
    }

    std::this_thread::sleep_for(
        std::chrono::milliseconds(100)); // 10Hz streaming TCP
  }
}

void TCPClusterManager::broadcastLocalTelemetry(
    const vgre_telemetry_t &telemetry) {
  if (enabled_ && !is_master_) {
    std::lock_guard<std::mutex> lock(client_mutex_);
    client_telemetry_buffer_ = telemetry;
  }
}

void TCPClusterManager::aggregateRemoteTelemetry(
    vgre_telemetry_t &outCombined) {
  if (enabled_ && is_master_) {
    std::lock_guard<std::mutex> lock(clients_mutex_);
    for (const auto &client : clients_) {
      if (!client.active)
        continue;
      outCombined.gflops += client.last_telemetry.gflops;
      outCombined.memory_bandwidth_gbps +=
          client.last_telemetry.memory_bandwidth_gbps;
      outCombined.memory_used_bytes += client.last_telemetry.memory_used_bytes;
      outCombined.active_kernels += client.last_telemetry.active_kernels;
      outCombined.active_threads += client.last_telemetry.active_threads;
    }
  }
}

} // namespace advanced
} // namespace vgre
