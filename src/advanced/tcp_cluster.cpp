#include "vgre/advanced/tcp_cluster.h"
#include "vgre/common/logger.h"
#include "vgre/core/runtime_engine.h"
#include <cstring>

#if defined(_WIN32)
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")

#define CLOSE_SOCKET(s) closesocket(s)
#define IOCTL_NONBLOCK(s)                                                      \
  do {                                                                         \
    u_long mode = 1;                                                           \
    ioctlsocket(s, FIONBIO, &mode);                                            \
  } while (0)
#ifndef MSG_DONTWAIT
#define MSG_DONTWAIT 0
#endif
#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif
typedef int socklen_t;

#else
#include <arpa/inet.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <unistd.h>
#define CLOSE_SOCKET(s) close(s)
#define IOCTL_NONBLOCK(s)                                                      \
  fcntl((s), F_SETFL, fcntl((s), F_GETFL, 0) | O_NONBLOCK)
#endif

namespace vgre {
namespace advanced {

TCPClusterManager::TCPClusterManager() {}

TCPClusterManager::~TCPClusterManager() { shutdown(); }

VGREResult TCPClusterManager::initialize(bool is_master,
                                         const std::string &host, int port) {
  if (enabled_.exchange(true))
    return VGREResult::SUCCESS;

#if defined(_WIN32)
  WSADATA wsaData;
  if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
    VGRE_LOG_ERROR("TCPCluster", "WSAStartup failed");
    return VGREResult::ERROR_IO;
  }
#endif

  is_master_ = is_master;
  host_ = host;
  port_ = port;

  if (is_master_) {
    // Master Node (Server)
    server_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd_ == (vgre_socket_t)-1) {
      VGRE_LOG_ERROR("TCPCluster", "Failed to create socket");
      return VGREResult::ERROR_IO;
    }

    int opt = 1;
#if defined(_WIN32) && !defined(SO_REUSEPORT)
    setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, (const char *)&opt,
               sizeof(opt));
#else
    setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt,
               sizeof(opt));
#endif

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
    IOCTL_NONBLOCK(server_fd_);

    VGRE_LOG_INFO("TCPCluster",
                  "Master Server Listening on port " + std::to_string(port_));
    cluster_thread_ = std::thread(&TCPClusterManager::serverLoop, this);
  } else {
    // Client Node (Worker)
    client_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (client_fd_ == (vgre_socket_t)-1) {
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
      CLOSE_SOCKET(client_fd_);
      client_fd_ = (vgre_socket_t)-1;
      return VGREResult::SUCCESS;
    }

    // Set non-blocking
    IOCTL_NONBLOCK(client_fd_);

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
      if (client.socket_fd != (vgre_socket_t)-1)
        CLOSE_SOCKET(client.socket_fd);
    }
    clients_.clear();
    if (server_fd_ != (vgre_socket_t)-1)
      CLOSE_SOCKET(server_fd_);
  } else {
    if (client_fd_ != (vgre_socket_t)-1)
      CLOSE_SOCKET(client_fd_);
  }

#if defined(_WIN32)
  WSACleanup();
#endif
}

void TCPClusterManager::serverLoop() {
  while (enabled_) {
    // Accept new connections
    struct sockaddr_in address;
    int addrlen = sizeof(address);
    vgre_socket_t new_socket =
        accept(server_fd_, (struct sockaddr *)&address, (socklen_t *)&addrlen);

    if (new_socket != (vgre_socket_t)-1) {
      IOCTL_NONBLOCK(new_socket);
      std::lock_guard<std::mutex> lock(clients_mutex_);
      clients_.push_back(
          {new_socket, {}, true}); // Initialize with zero'd struct
      std::memset(&clients_.back().last_telemetry, 0, sizeof(vgre_telemetry_t));
      VGRE_LOG_INFO("TCPCluster", "New remote node connected via TCP.");
    }

    // Read data from all clients
    {
      std::lock_guard<std::mutex> lock(clients_mutex_);
      for (auto &client : clients_) {
        if (!client.active)
          continue;

        // Read Packet Type
        PacketType type;
        int n = recv(client.socket_fd, (char *)&type, sizeof(PacketType),
                     MSG_DONTWAIT);
        if (n == sizeof(PacketType)) {
          if (type == PacketType::TELEMETRY) {
            ssize_t bytes_read =
                recv(client.socket_fd, (char *)&client.last_telemetry,
                     sizeof(vgre_telemetry_t), 0);
            if (bytes_read < 0 && (errno != EAGAIN && errno != EWOULDBLOCK)) {
              client.active = false;
            }
          }
        } else if (n == 0 ||
                   (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK)) {
          client.active = false;
          CLOSE_SOCKET(client.socket_fd);
          client.socket_fd = (vgre_socket_t)-1;
          VGRE_LOG_INFO("TCPCluster", "Remote node disconnected.");
        }
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
}

void TCPClusterManager::clientLoop() {
  while (enabled_) {
    // 1. Send Telemetry to Master
    vgre_telemetry_t telemetry;
    {
      std::lock_guard<std::mutex> lock(client_mutex_);
      telemetry = client_telemetry_buffer_;
    }

    if (telemetry.timestamp > 0) {
      PacketType type = PacketType::TELEMETRY;
      send(client_fd_, (const char *)&type, sizeof(PacketType), MSG_NOSIGNAL);
      send(client_fd_, (const char *)&telemetry, sizeof(vgre_telemetry_t),
           MSG_NOSIGNAL);
    }

    // 2. Check for incoming commands from Master
    RemoteCommandPacket pkt;
    int n = recv(client_fd_, (char *)&pkt, sizeof(RemoteCommandPacket),
                 MSG_DONTWAIT);
    if (n == sizeof(RemoteCommandPacket)) {
      if (pkt.type == PacketType::LAUNCH_KERNEL) {
        handleRemoteCommand(pkt);
      }
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

VGREResult TCPClusterManager::launchRemoteKernel(
    int worker_idx, uint64_t kernel_id, const uint32_t grid_dim[3],
    const uint32_t block_dim[3], void **args, int num_args, size_t shared_mem) {
  if (!is_master_)
    return VGREResult::ERROR_INVALID_VALUE;

  std::lock_guard<std::mutex> lock(clients_mutex_);
  if (worker_idx < 0 || worker_idx >= static_cast<int>(clients_.size()) ||
      !clients_[worker_idx].active) {
    return VGREResult::ERROR_INVALID_VALUE;
  }

  std::vector<ArgType> argTypes;
  if (core::RuntimeEngine::instance().getKernelArgTypes(kernel_id, argTypes) !=
      VGREResult::SUCCESS) {
    return VGREResult::ERROR_INVALID_KERNEL;
  }

  RemoteCommandPacket pkt;
  pkt.type = PacketType::LAUNCH_KERNEL;
  pkt.kernel_id = kernel_id;
  std::memcpy(pkt.grid_dim, grid_dim, sizeof(pkt.grid_dim));
  std::memcpy(pkt.block_dim, block_dim, sizeof(pkt.block_dim));
  pkt.shared_mem = shared_mem;
  pkt.num_args = std::min(num_args, 8);

  // Simplified: only supporting numeric args for now
  for (int i = 0; i < pkt.num_args; ++i) {
    if (i < argTypes.size() && argTypes[i] == ArgType::POINTER) {
      VGRE_LOG_ERROR("TCPCluster", "Remote execution with pointer arguments is "
                                   "currently unsupported over TCP");
      return VGREResult::ERROR_INVALID_VALUE;
    }

    if (args[i]) {
      // Best-effort value cast for the protocol
      pkt.args[i] = *reinterpret_cast<double *>(args[i]);
    } else {
      pkt.args[i] = 0;
    }
  }

  send(clients_[worker_idx].socket_fd, (const char *)&pkt,
       sizeof(RemoteCommandPacket), MSG_NOSIGNAL);
  VGRE_LOG_INFO("TCPCluster", "Dispatched remote kernel launch to worker " +
                                  std::to_string(worker_idx));

  return VGREResult::SUCCESS;
}

void TCPClusterManager::handleRemoteCommand(const RemoteCommandPacket &pkt) {
  VGRE_LOG_INFO("TCPCluster",
                "Executing remote kernel launch request (Kernel ID: " +
                    std::to_string(pkt.kernel_id) + ")");

  void *local_args[8];
  for (int i = 0; i < pkt.num_args; ++i) {
    local_args[i] = (void *)&pkt.args[i];
  }

  dim3 gd(pkt.grid_dim[0], pkt.grid_dim[1], pkt.grid_dim[2]);
  dim3 bd(pkt.block_dim[0], pkt.block_dim[1], pkt.block_dim[2]);

  vgre::core::RuntimeEngine::instance().launchKernel(
      pkt.kernel_id, gd, bd, local_args, pkt.shared_mem, 0);
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
