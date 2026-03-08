#include "vgre/advanced/tcp_cluster.h"
#include "vgre/common/logger.h"
#include "vgre/core/memory_manager.h"
#include "vgre/core/runtime_engine.h"

// System Headers
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <functional>

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

namespace {
static bool socket_would_block() {
#if defined(_WIN32)
  int err = WSAGetLastError();
  return err == WSAEWOULDBLOCK || err == WSAEINPROGRESS;
#else
  return errno == EAGAIN || errno == EWOULDBLOCK;
#endif
}

static bool send_all(vgre_socket_t fd, const void *buf, size_t len) {
  const char *p = static_cast<const char *>(buf);
  size_t sent = 0;
  while (sent < len) {
    int n = send(fd, p + sent, static_cast<int>(len - sent), MSG_NOSIGNAL);
    if (n == 0) {
      return false;
    }
    if (n < 0) {
      if (socket_would_block()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        continue;
      }
      return false;
    }
    sent += static_cast<size_t>(n);
  }
  return true;
}
} // namespace

TCPClusterManager::TCPClusterManager() {}

TCPClusterManager::~TCPClusterManager() { shutdown(); }

VGREResult TCPClusterManager::initialize(bool is_master,
                                         const std::string &host, int port) {
  if (enabled_)
    return VGREResult::SUCCESS;

#if defined(_WIN32)
  WSADATA wsaData;
  if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
    VGRE_LOG_ERROR("TCPCluster", "WSAStartup failed");
    enabled_ = false;
    return VGREResult::ERROR_IO;
  }
#endif

  is_master_ = is_master;
  host_ = host;
  port_ = port;
  auth_token_ = 0;
  if (const char *token = std::getenv("VGRE_TCP_AUTH_TOKEN")) {
    if (token[0] != '\0') {
      auth_token_ = static_cast<uint64_t>(std::hash<std::string>{}(token));
    }
  }

  if (is_master_) {
    // Master Node (Server)
    server_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd_ == (vgre_socket_t)-1) {
      VGRE_LOG_ERROR("TCPCluster", "Failed to create socket");
      enabled_ = false;
      return VGREResult::ERROR_IO;
    }

    int opt = 1;
#if defined(_WIN32)
    setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, (const char *)&opt,
               sizeof(opt));
#else
    setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#ifdef SO_REUSEPORT
    setsockopt(server_fd_, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(opt));
#endif
#endif

    struct sockaddr_in address;
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port_);

    if (bind(server_fd_, (struct sockaddr *)&address, sizeof(address)) < 0) {
      VGRE_LOG_ERROR("TCPCluster",
                     "Bind failed on port " + std::to_string(port_));
      CLOSE_SOCKET(server_fd_);
      server_fd_ = (vgre_socket_t)-1;
      enabled_ = false;
      return VGREResult::ERROR_IO;
    }

    if (listen(server_fd_, 10) < 0) {
      VGRE_LOG_ERROR("TCPCluster", "Listen failed");
      CLOSE_SOCKET(server_fd_);
      server_fd_ = (vgre_socket_t)-1;
      enabled_ = false;
      return VGREResult::ERROR_IO;
    }

    IOCTL_NONBLOCK(server_fd_);

    VGRE_LOG_INFO("TCPCluster",
                  "Master Server Listening on port " + std::to_string(port_));
    cluster_thread_ = std::thread(&TCPClusterManager::serverLoop, this);
    udp_thread_ = std::thread(&TCPClusterManager::udpAnnouncerLoop, this);
  } else {
    // Client Node (Worker)
    if (host_ == "auto" || host_.empty()) {
      cluster_thread_ = std::thread(&TCPClusterManager::udpDiscoveryLoop, this);
    } else {
    client_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (client_fd_ == (vgre_socket_t)-1) {
      VGRE_LOG_ERROR("TCPCluster", "Failed to create client socket");
      enabled_ = false;
      return VGREResult::ERROR_IO;
    }

    struct sockaddr_in serv_addr;
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(port_);

    if (inet_pton(AF_INET, host_.c_str(), &serv_addr.sin_addr) <= 0) {
      VGRE_LOG_ERROR("TCPCluster",
                     "Invalid address or not supported: " + host_);
      CLOSE_SOCKET(client_fd_);
      client_fd_ = (vgre_socket_t)-1;
      enabled_ = false;
      return VGREResult::ERROR_IO;
    }

    if (connect(client_fd_, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) <
        0) {
      VGRE_LOG_ERROR("TCPCluster", "Connection failed to " + host_ + ":" +
                                       std::to_string(port_));
      enabled_ = false;
      CLOSE_SOCKET(client_fd_);
      client_fd_ = (vgre_socket_t)-1;
#if defined(_WIN32)
      WSACleanup();
#endif
      return VGREResult::ERROR_IO;
    }

    // Set non-blocking
    IOCTL_NONBLOCK(client_fd_);

    VGRE_LOG_INFO("TCPCluster", "Connected to Remote Master Node at " + host_);
    cluster_thread_ = std::thread(&TCPClusterManager::clientLoop, this);
    data_processor_thread_ = std::thread(&TCPClusterManager::processClientStagingBuffer, this);
    }
  }

  enabled_ = true;
  return VGREResult::SUCCESS;
}

void TCPClusterManager::shutdown() {
  if (!enabled_.exchange(false))
    return;

  // Unblock wait conditions
  {
    std::lock_guard<std::mutex> lock(staging_mutex_);
    staging_ready_ = true;
  }
  staging_cv_.notify_all();

  if (udp_thread_.joinable()) {
    udp_thread_.join();
  }
  if (data_processor_thread_.joinable()) {
    data_processor_thread_.join();
  }
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
      ClientConnection conn{};
      conn.socket_fd = new_socket;
      conn.active = true;
      conn.expecting_type = true;
      conn.pending_type = PacketType::TELEMETRY;
      conn.rx_buffer.clear();
      clients_.push_back(std::move(conn));
      std::memset(&clients_.back().last_telemetry, 0, sizeof(vgre_telemetry_t));
      VGRE_LOG_INFO("TCPCluster", "New remote node connected via TCP.");
    }

    // Read data from all clients
    {
      std::lock_guard<std::mutex> lock(clients_mutex_);
      for (auto &client : clients_) {
        if (!client.active)
          continue;

        char temp[2048];
        int n = recv(client.socket_fd, temp, sizeof(temp), MSG_DONTWAIT);
        if (n > 0) {
          client.rx_buffer.insert(client.rx_buffer.end(), temp, temp + n);
          while (client.active) {
            if (client.expecting_type) {
              if (client.rx_buffer.size() < sizeof(PacketType))
                break;
              std::memcpy(&client.pending_type, client.rx_buffer.data(),
                          sizeof(PacketType));
              client.rx_buffer.erase(client.rx_buffer.begin(),
                                     client.rx_buffer.begin() +
                                         static_cast<long>(sizeof(PacketType)));
              client.expecting_type = false;
            }

            if (client.pending_type == PacketType::TELEMETRY) {
              if (client.rx_buffer.size() < sizeof(vgre_telemetry_t))
                break;
              std::memcpy(&client.last_telemetry, client.rx_buffer.data(),
                          sizeof(vgre_telemetry_t));
              client.rx_buffer.erase(client.rx_buffer.begin(),
                                     client.rx_buffer.begin() +
                                         static_cast<long>(sizeof(vgre_telemetry_t)));
              client.expecting_type = true;
            } else {
              VGRE_LOG_ERROR("TCPCluster", "Unknown packet type from client");
              client.active = false;
              CLOSE_SOCKET(client.socket_fd);
              client.socket_fd = (vgre_socket_t)-1;
              break;
            }
          }
        } else if (n == 0 || (n < 0 && !socket_would_block())) {
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
      if (!send_all(client_fd_, &type, sizeof(PacketType)) ||
          !send_all(client_fd_, &telemetry, sizeof(vgre_telemetry_t))) {
        VGRE_LOG_ERROR("TCPCluster",
                       "Failed to send telemetry to master; disconnecting");
        enabled_ = false;
        break;
      }
    }

    // 2. Buffer incoming commands from Master asynchronously
    char temp[4096];
    int n = recv(client_fd_, temp, sizeof(temp), MSG_DONTWAIT);
    if (n > 0) {
      std::lock_guard<std::mutex> lock(staging_mutex_);
      active_staging_->insert(active_staging_->end(), temp, temp + n);
      staging_ready_ = true;
      staging_cv_.notify_one();
    } else if (n == 0 || (n < 0 && !socket_would_block())) {
      VGRE_LOG_ERROR("TCPCluster", "Client command channel disconnected");
      enabled_ = false;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

void TCPClusterManager::processClientStagingBuffer() {
  while (enabled_) {
    std::unique_lock<std::mutex> lock(staging_mutex_);
    staging_cv_.wait(lock, [this]() { return staging_ready_.load() || !enabled_; });
    if (!enabled_) break;
    
    // Swap buffers
    std::swap(active_staging_, processing_staging_);
    staging_ready_ = false;
    lock.unlock();

    if (processing_staging_->empty()) continue;

    // Append to internal rx buffer and process
    client_rx_buffer_.insert(client_rx_buffer_.end(), 
                             processing_staging_->begin(), 
                             processing_staging_->end());
    processing_staging_->clear();

    while (true) {
        if (client_rx_buffer_.size() < sizeof(PacketType))
          break;
        
        PacketType type;
        std::memcpy(&type, client_rx_buffer_.data(), sizeof(PacketType));

        if (type == PacketType::LAUNCH_KERNEL) {
            if (client_rx_buffer_.size() < sizeof(RemoteCommandPacket) + sizeof(PacketType))
              break;
            RemoteCommandPacket pkt{};
            std::memcpy(&pkt, client_rx_buffer_.data() + sizeof(PacketType), sizeof(RemoteCommandPacket));
            client_rx_buffer_.erase(client_rx_buffer_.begin(),
                                    client_rx_buffer_.begin() + sizeof(PacketType) + sizeof(RemoteCommandPacket));
            handleRemoteCommand(pkt);
        } else if (type == PacketType::DATA_HEADER) {
            if (client_rx_buffer_.size() < sizeof(DataHeaderPacket) + sizeof(PacketType))
              break;
            DataHeaderPacket dpkt{};
            std::memcpy(&dpkt, client_rx_buffer_.data() + sizeof(PacketType), sizeof(DataHeaderPacket));
            
            size_t required = sizeof(DataHeaderPacket) + sizeof(PacketType) * 2 + dpkt.size;
            if (client_rx_buffer_.size() < required) break; 

            // We have the full data!
            client_rx_buffer_.erase(client_rx_buffer_.begin(), client_rx_buffer_.begin() + sizeof(DataHeaderPacket) + sizeof(PacketType));
            
            PacketType btype;
            std::memcpy(&btype, client_rx_buffer_.data(), sizeof(PacketType));
            if (btype == PacketType::DATA_BODY) {
                void* target = reinterpret_cast<void*>(dpkt.target_ptr);
                auto& mm = core::RuntimeEngine::instance().getMemoryManager();
                if (mm.isValidHandle(target)) {
                    std::memcpy(mm.getPointer(target), 
                                client_rx_buffer_.data() + sizeof(PacketType), 
                                dpkt.size);
                }
            }
            client_rx_buffer_.erase(client_rx_buffer_.begin(), client_rx_buffer_.begin() + sizeof(PacketType) + dpkt.size);
        } else {
            client_rx_buffer_.clear(); // Corrupt state
            break;
        }
    }
  }
}

void TCPClusterManager::udpAnnouncerLoop() {
  vgre_socket_t udp_fd = socket(AF_INET, SOCK_DGRAM, 0);
  if (udp_fd == (vgre_socket_t)-1) return;
  
  int opt = 1;
#if defined(_WIN32)
  setsockopt(udp_fd, SOL_SOCKET, SO_BROADCAST, (const char*)&opt, sizeof(opt));
#else
  setsockopt(udp_fd, SOL_SOCKET, SO_BROADCAST, &opt, sizeof(opt));
#endif

  struct sockaddr_in broadcast_addr{};
  broadcast_addr.sin_family = AF_INET;
  broadcast_addr.sin_addr.s_addr = INADDR_BROADCAST;
  broadcast_addr.sin_port = htons(7778);

  const char* ping_msg = "VGRE_DISCOVERY_PING";

  while (enabled_ && is_master_) {
    sendto(udp_fd, ping_msg, strlen(ping_msg), 0, (struct sockaddr*)&broadcast_addr, sizeof(broadcast_addr));
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
  CLOSE_SOCKET(udp_fd);
}

void TCPClusterManager::udpDiscoveryLoop() {
  vgre_socket_t udp_fd = socket(AF_INET, SOCK_DGRAM, 0);
  if (udp_fd == (vgre_socket_t)-1) return;

  int opt = 1;
#if defined(_WIN32)
  setsockopt(udp_fd, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt, sizeof(opt));
#else
  setsockopt(udp_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#ifdef SO_REUSEPORT
  setsockopt(udp_fd, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(opt));
#endif
#endif

  struct sockaddr_in listen_addr{};
  listen_addr.sin_family = AF_INET;
  listen_addr.sin_addr.s_addr = INADDR_ANY;
  listen_addr.sin_port = htons(7778);

  if (bind(udp_fd, (struct sockaddr*)&listen_addr, sizeof(listen_addr)) < 0) {
    VGRE_LOG_WARN("TCPCluster", "UDP discovery bind failed");
    CLOSE_SOCKET(udp_fd);
    return;
  }

  struct timeval tv;
  tv.tv_sec = 1;
  tv.tv_usec = 0;
#if defined(_WIN32)
  DWORD timeout = 1000;
  setsockopt(udp_fd, SOL_SOCKET, SO_RCVTIMEO, (const char*)&timeout, sizeof(timeout));
#else
  setsockopt(udp_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
#endif

  char buffer[64];
  struct sockaddr_in sender_addr{};
  socklen_t sender_len = sizeof(sender_addr);

  VGRE_LOG_INFO("TCPCluster", "Scanning local subnet for Master node broadcasts...");

  while (enabled_ && client_fd_ == (vgre_socket_t)-1) {
    int n = recvfrom(udp_fd, buffer, sizeof(buffer) - 1, 0, (struct sockaddr*)&sender_addr, &sender_len);
    if (n > 0) {
      buffer[n] = '\0';
      if (std::string(buffer) == "VGRE_DISCOVERY_PING") {
        char ip[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &(sender_addr.sin_addr), ip, INET_ADDRSTRLEN);
        host_ = ip;
        VGRE_LOG_INFO("TCPCluster", "Discovered Master node at " + host_);
        
        client_fd_ = socket(AF_INET, SOCK_STREAM, 0);
        struct sockaddr_in serv_addr;
        serv_addr.sin_family = AF_INET;
        serv_addr.sin_port = htons(port_);
        inet_pton(AF_INET, host_.c_str(), &serv_addr.sin_addr);
        
        if (connect(client_fd_, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) >= 0) {
            IOCTL_NONBLOCK(client_fd_);
            VGRE_LOG_INFO("TCPCluster", "Connected to Remote Master Node at " + host_);
            break;
        } else {
            CLOSE_SOCKET(client_fd_);
            client_fd_ = (vgre_socket_t)-1;
        }
      }
    }
  }
  CLOSE_SOCKET(udp_fd);

  if (client_fd_ != (vgre_socket_t)-1 && enabled_) {
      data_processor_thread_ = std::thread(&TCPClusterManager::processClientStagingBuffer, this);
      clientLoop();
  }
}


VGREResult TCPClusterManager::launchRemoteKernel(
    int worker_idx, uint64_t kernel_id, const uint32_t grid_dim[3],
    const uint32_t block_dim[3], void **args, int num_args, size_t shared_mem) {
  if (!is_master_)
    return VGREResult::ERROR_INVALID_VALUE;
  if (!grid_dim || !block_dim || num_args < 0)
    return VGREResult::ERROR_INVALID_VALUE;
  if (num_args > 0 && !args)
    return VGREResult::ERROR_INVALID_VALUE;
  if (grid_dim[0] == 0 || block_dim[0] == 0)
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

  RemoteCommandPacket pkt{};
  pkt.type = PacketType::LAUNCH_KERNEL;
  if (auth_token_ == 0) {
    VGRE_LOG_ERROR("TCPCluster",
                   "Remote kernel dispatch blocked: missing "
                   "VGRE_TCP_AUTH_TOKEN");
    return VGREResult::ERROR_INVALID_VALUE;
  }
  pkt.auth_token = auth_token_;
  pkt.kernel_id = kernel_id;
  std::memcpy(pkt.grid_dim, grid_dim, sizeof(pkt.grid_dim));
  std::memcpy(pkt.block_dim, block_dim, sizeof(pkt.block_dim));
  pkt.shared_mem = shared_mem;
  pkt.num_args = std::min(num_args, 8);
  std::memset(pkt.arg_types, 0, sizeof(pkt.arg_types));
  std::memset(pkt.args, 0, sizeof(pkt.args));

  for (int i = 0; i < pkt.num_args; ++i) {
    if (i >= static_cast<int>(argTypes.size())) {
      VGRE_LOG_ERROR("TCPCluster",
                     "Remote dispatch failed: kernel argument metadata missing");
      return VGREResult::ERROR_INVALID_KERNEL;
    }
    if (argTypes[i] == ArgType::POINTER) {
      void* host_ptr = *static_cast<void **>(args[i]);
      if (host_ptr) {
        auto &mm = core::RuntimeEngine::instance().getMemoryManager();
        if (mm.isValidHandle(host_ptr)) {
          size_t size = mm.getAllocationSize(host_ptr);
          
          // 1. Send DATA_HEADER
          DataHeaderPacket dptr_pkt{};
          dptr_pkt.type = PacketType::DATA_HEADER;
          dptr_pkt.target_ptr = reinterpret_cast<uintptr_t>(host_ptr);
          dptr_pkt.size = size;
          send_all(clients_[worker_idx].socket_fd, &dptr_pkt, sizeof(DataHeaderPacket));

          // 2. Send DATA_BODY type and content
          PacketType body_type = PacketType::DATA_BODY;
          send_all(clients_[worker_idx].socket_fd, &body_type, sizeof(PacketType));
          send_all(clients_[worker_idx].socket_fd, mm.getPointer(host_ptr), size);

          // The actual argument in the LAUNCH_KERNEL packet is the pointer value
          pkt.args[i] = reinterpret_cast<uint64_t>(host_ptr);
          pkt.arg_types[i] = static_cast<uint8_t>(ArgType::POINTER);
          continue;
        }
      }
      VGRE_LOG_ERROR("TCPCluster", "Remote execution failed: pointer argument is not a managed VGRE handle");
      return VGREResult::ERROR_INVALID_VALUE;
    }

    if (!args || !args[i]) {
      VGRE_LOG_ERROR("TCPCluster",
                     "Remote dispatch failed: null scalar argument at index " +
                         std::to_string(i));
      return VGREResult::ERROR_INVALID_VALUE;
    }

    ArgType type = argTypes[i];
    pkt.arg_types[i] = static_cast<uint8_t>(type);
    switch (type) {
    case ArgType::INT32:
    case ArgType::UINT32:
    case ArgType::FLOAT32:
      std::memcpy(&pkt.args[i], args[i], sizeof(uint32_t));
      break;
    case ArgType::INT64:
    case ArgType::UINT64:
    case ArgType::FLOAT64:
      std::memcpy(&pkt.args[i], args[i], sizeof(uint64_t));
      break;
    case ArgType::POINTER:
      // Already rejected above
      return VGREResult::ERROR_INVALID_VALUE;
    }
  }

  if (!send_all(clients_[worker_idx].socket_fd, &pkt,
                sizeof(RemoteCommandPacket))) {
    clients_[worker_idx].active = false;
    CLOSE_SOCKET(clients_[worker_idx].socket_fd);
    clients_[worker_idx].socket_fd = (vgre_socket_t)-1;
    VGRE_LOG_ERROR("TCPCluster", "Failed to dispatch remote kernel packet");
    return VGREResult::ERROR_IO;
  }
  VGRE_LOG_INFO("TCPCluster", "Dispatched remote kernel launch to worker " +
                                  std::to_string(worker_idx));

  return VGREResult::SUCCESS;
}

int TCPClusterManager::getFirstActiveWorker() const {
  if (!is_master_) {
    return -1;
  }
  std::lock_guard<std::mutex> lock(clients_mutex_);
  for (size_t i = 0; i < clients_.size(); ++i) {
    if (clients_[i].active) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

void TCPClusterManager::handleRemoteCommand(const RemoteCommandPacket &pkt) {
  if (auth_token_ == 0 || pkt.auth_token != auth_token_) {
    VGRE_LOG_ERROR("TCPCluster",
                   "Rejected remote command due to missing/invalid auth token");
    return;
  }
  if (pkt.type != PacketType::LAUNCH_KERNEL || pkt.grid_dim[0] == 0 ||
      pkt.block_dim[0] == 0) {
    VGRE_LOG_ERROR("TCPCluster", "Rejected malformed remote command packet");
    return;
  }

  VGRE_LOG_INFO("TCPCluster",
                "Executing remote kernel launch request (Kernel ID: " +
                    std::to_string(pkt.kernel_id) + ")");

  int32_t arg_i32[8] = {};
  uint32_t arg_u32[8] = {};
  int64_t arg_i64[8] = {};
  uint64_t arg_u64[8] = {};
  float arg_f32[8] = {};
  double arg_f64[8] = {};
  void *local_args[8] = {};
  int numArgs = std::clamp(pkt.num_args, 0, 8);
  for (int i = 0; i < numArgs; ++i) {
    ArgType type = static_cast<ArgType>(pkt.arg_types[i]);
    switch (type) {
    case ArgType::INT32: {
      std::memcpy(&arg_i32[i], &pkt.args[i], sizeof(int32_t));
      local_args[i] = static_cast<void *>(&arg_i32[i]);
      break;
    }
    case ArgType::UINT32: {
      std::memcpy(&arg_u32[i], &pkt.args[i], sizeof(uint32_t));
      local_args[i] = static_cast<void *>(&arg_u32[i]);
      break;
    }
    case ArgType::FLOAT32: {
      uint32_t bits = 0;
      std::memcpy(&bits, &pkt.args[i], sizeof(uint32_t));
      std::memcpy(&arg_f32[i], &bits, sizeof(float));
      local_args[i] = static_cast<void *>(&arg_f32[i]);
      break;
    }
    case ArgType::INT64: {
      std::memcpy(&arg_i64[i], &pkt.args[i], sizeof(int64_t));
      local_args[i] = static_cast<void *>(&arg_i64[i]);
      break;
    }
    case ArgType::UINT64: {
      std::memcpy(&arg_u64[i], &pkt.args[i], sizeof(uint64_t));
      local_args[i] = static_cast<void *>(&arg_u64[i]);
      break;
    }
    case ArgType::FLOAT64: {
      std::memcpy(&arg_f64[i], &pkt.args[i], sizeof(double));
      local_args[i] = static_cast<void *>(&arg_f64[i]);
      break;
    }
    case ArgType::POINTER: {
      arg_u64[i] = pkt.args[i];
      local_args[i] = static_cast<void*>(&arg_u64[i]);
      break;
    }
    default:
      VGRE_LOG_ERROR("TCPCluster",
                     "Rejected remote command: unknown argument type " +
                         std::to_string(static_cast<int>(pkt.arg_types[i])));
      return;
    }
  }

  dim3 gd(pkt.grid_dim[0], pkt.grid_dim[1], pkt.grid_dim[2]);
  dim3 bd(pkt.block_dim[0], pkt.block_dim[1], pkt.block_dim[2]);

  auto r = vgre::core::RuntimeEngine::instance().launchKernel(
      pkt.kernel_id, gd, bd, local_args, pkt.shared_mem, 0);
  if (r != VGREResult::SUCCESS) {
    VGRE_LOG_ERROR("TCPCluster",
                   "Remote kernel execution failed with code " +
                       std::to_string(static_cast<int>(r)));
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
