#include "vgre/advanced/tcp_cluster.h"
#include "vgre/advanced/hybrid_compute_manager.h"
#include "vgre/common/logger.h"
#include "vgre/core/memory_manager.h"
#include "vgre/core/runtime_engine.h"

// System Headers
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <fstream>
#include <sstream>

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
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <poll.h>

#define CLOSE_SOCKET(s) close(s)
#define IOCTL_NONBLOCK(s)                                                      \
  do {                                                                         \
    int mode = 1;                                                              \
    ioctl(s, FIONBIO, &mode);                                                  \
  } while (0)
#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif
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
  if (enabled_ && is_master_ == is_master)
    return VGREResult::SUCCESS;

  if (enabled_) {
    shutdown();
  }

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
      // Use stable FNV-1a hash instead of std::hash (which is not stable across processes)
      uint64_t hash = 0xcbf29ce484222325ULL;
      const char* p = token;
      while (*p) {
          hash ^= static_cast<uint64_t>(*p++);
          hash *= 0x100000001b3ULL;
      }
      auth_token_ = hash;
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
      VGRE_LOG_DEBUG("TCPCluster", "Connection failed to " + host_ + ":" +
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
  VGRE_LOG_DEBUG("TCPCluster", "Master Server Loop starting...");

  while (enabled_) {
    // 1. Prepare poll fds
    std::vector<struct pollfd> fds;
    
    // Listening socket
    struct pollfd pfd_server;
    pfd_server.fd = server_fd_;
    pfd_server.events = POLLIN;
    fds.push_back(pfd_server);
    
    // Client sockets
    {
        std::lock_guard<std::mutex> lock(clients_mutex_);
        for (const auto &client : clients_) {
            if (client.active && client.socket_fd != (vgre_socket_t)-1) {
                struct pollfd pfd_client;
                pfd_client.fd = client.socket_fd;
                pfd_client.events = POLLIN;
                fds.push_back(pfd_client);
            }
        }
    }
    
    int poll_res = 0;
#if defined(_WIN32)
    poll_res = WSAPoll(fds.data(), fds.size(), 50);
#else
    poll_res = poll(fds.data(), fds.size(), 50);
#endif

    if (poll_res < 0) {
        if (!socket_would_block()) {
           VGRE_LOG_ERROR("TCPCluster", "Master: poll() failed");
           break;
        }
        continue;
    }
    
    if (poll_res == 0) continue; // Timeout

    // 2. Handle new connections
    if (fds[0].revents & POLLIN) {
        struct sockaddr_in address;
        socklen_t addrlen = sizeof(address);
        vgre_socket_t new_socket = accept(server_fd_, (struct sockaddr *)&address, &addrlen);
        
        if (new_socket != (vgre_socket_t)-1) {
            char ipstr[INET_ADDRSTRLEN];
            inet_ntop(AF_INET, &(address.sin_addr), ipstr, sizeof(ipstr));

            IOCTL_NONBLOCK(new_socket);
            std::lock_guard<std::mutex> lock(clients_mutex_);
            ClientConnection conn{};
            conn.socket_fd = new_socket;
            conn.ip_address = std::string(ipstr);
            conn.active = true;
            conn.expecting_type = true;
            conn.rx_buffer.clear();
            clients_.push_back(std::move(conn));
            VGRE_LOG_INFO("TCPCluster", "Master: Accepted new remote node from " + std::string(ipstr));
        }
    }

    // 3. Handle data from existing clients
    {
        std::lock_guard<std::mutex> lock(clients_mutex_);
        for (size_t i = 1; i < fds.size(); ++i) {
            if (fds[i].revents & (POLLIN | POLLERR | POLLHUP)) {
                // Find matching client
                for (auto &client : clients_) {
                    if (client.socket_fd == fds[i].fd) {
                        if (fds[i].revents & POLLIN) {
                            char temp[4096];
                            int n = recv(client.socket_fd, temp, sizeof(temp), 0);
                            if (n > 0) {
                                client.rx_buffer.insert(client.rx_buffer.end(), temp, temp + n);
                            } else if (n == 0) {
                                VGRE_LOG_INFO("TCPCluster", "Master: Worker disconnected.");
                                client.active = false;
                                CLOSE_SOCKET(client.socket_fd);
                            }
                        } else {
                            VGRE_LOG_WARN("TCPCluster", "Master: Client socket error or hup.");
                            client.active = false;
                            CLOSE_SOCKET(client.socket_fd);
                        }
                        break;
                    }
                }
            }
        }
        
        // Process buffers
        for (auto &client : clients_) {
          if (!client.active || client.rx_buffer.empty()) continue;
          
          while (client.active && !client.rx_buffer.empty()) {
            PacketType type = static_cast<PacketType>(client.rx_buffer[0]);
            
            if (type == PacketType::TELEMETRY) {
              if (client.rx_buffer.size() < sizeof(vgre_telemetry_t) + sizeof(PacketType)) break;
              std::memcpy(&client.last_telemetry, client.rx_buffer.data() + sizeof(PacketType), sizeof(vgre_telemetry_t));
              client.rx_buffer.erase(client.rx_buffer.begin(), client.rx_buffer.begin() + sizeof(PacketType) + sizeof(vgre_telemetry_t));
            } else if (type == PacketType::RESPONSE) {
              if (client.rx_buffer.size() < sizeof(ResponsePacket)) break;
              ResponsePacket resp;
              std::memcpy(&resp, client.rx_buffer.data(), sizeof(ResponsePacket));
              client.rx_buffer.erase(client.rx_buffer.begin(), client.rx_buffer.begin() + sizeof(ResponsePacket));
              VGRE_LOG_DEBUG("TCPCluster", "Master: Received RESPONSE from worker (Kernel: " + std::to_string(resp.kernel_id) + ")");
            } else if (type == PacketType::DATA_HEADER) {
              if (client.rx_buffer.size() < sizeof(DataHeaderPacket)) break;
              DataHeaderPacket dpkt;
              std::memcpy(&dpkt, client.rx_buffer.data(), sizeof(DataHeaderPacket));
              client.rx_buffer.erase(client.rx_buffer.begin(), client.rx_buffer.begin() + sizeof(DataHeaderPacket));
              client.pending_target_ptr = dpkt.target_ptr;
              client.pending_data_size = dpkt.size;
              client.expecting_type = false;
            } else if (type == PacketType::DATA_BODY) {
              if (client.rx_buffer.size() < sizeof(PacketType) + client.pending_data_size) break;
              void* host_ptr = reinterpret_cast<void*>(client.pending_target_ptr);
              if (host_ptr) {
                  auto &mm = core::RuntimeEngine::instance().getMemoryManager();
                  if (mm.isValidHandle(host_ptr)) {
                      std::memcpy(mm.getPointer(host_ptr), client.rx_buffer.data() + sizeof(PacketType), client.pending_data_size);
                  }
              }
              client.rx_buffer.erase(client.rx_buffer.begin(), client.rx_buffer.begin() + sizeof(PacketType) + client.pending_data_size);
              client.expecting_type = true;
            } else if (type == PacketType::CAPABILITY) {
              if (client.rx_buffer.size() < sizeof(CapabilityPacket)) break;
              CapabilityPacket cpkt;
              std::memcpy(&cpkt, client.rx_buffer.data(), sizeof(CapabilityPacket));
              client.rx_buffer.erase(client.rx_buffer.begin(), client.rx_buffer.begin() + sizeof(CapabilityPacket));
              
              client.cpu_cores = cpkt.cpu_cores;
              client.cpu_memory = cpkt.cpu_memory;
              client.has_igpu = cpkt.has_igpu;
              std::snprintf(client.igpu_name, sizeof(client.igpu_name), "%s", cpkt.igpu_name);
              
              HybridComputeManager::instance().updateRemoteNodeCapability(
                  client.ip_address, cpkt.cpu_cores, cpkt.cpu_memory, cpkt.has_igpu, cpkt.igpu_name);
              
              VGRE_LOG_INFO("TCPCluster", "Master: Received CAPABILITY from worker " + client.ip_address + " (Cores: " + std::to_string(cpkt.cpu_cores) + ")");
            } else {
              VGRE_LOG_ERROR("TCPCluster", "Master: Protocol sync error, discarding buffer for client " + client.ip_address);
              client.rx_buffer.clear();
              break;
            }
          }
        }
    }
  }
  VGRE_LOG_DEBUG("TCPCluster", "Master Server Loop exiting.");
}

void TCPClusterManager::clientLoop() {
  // 1. Send Capability Handshake
  {
    CapabilityPacket cpkt{};
    cpkt.type = PacketType::CAPABILITY;
    
    // Use HybridComputeManager to get local resources if possible, or detect directly
    cpkt.cpu_cores = std::thread::hardware_concurrency();
    
#if defined(_WIN32)
    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(MEMORYSTATUSEX);
    if (GlobalMemoryStatusEx(&memInfo)) cpkt.cpu_memory = memInfo.ullTotalPhys;
#else
    std::ifstream meminfo("/proc/meminfo");
    std::string line;
    while (std::getline(meminfo, line)) {
      if (line.find("MemTotal") != std::string::npos) {
        std::istringstream iss(line);
        std::string label; size_t kb; iss >> label >> kb;
        cpkt.cpu_memory = kb * 1024;
        break;
      }
    }
#endif
    
    // Simplified iGPU check for handshake
#ifdef VGRE_HAS_OPENCL_BACKEND
    cpkt.has_igpu = true;
    std::strncpy(cpkt.igpu_name, "VGRE-Enabled iGPU", 63);
#else
    cpkt.has_igpu = false;
#endif

    send_all(client_fd_, &cpkt, sizeof(CapabilityPacket));
  }

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

    while (enabled_ && !client_rx_buffer_.empty()) {
        PacketType type = static_cast<PacketType>(client_rx_buffer_[0]);

        if (type == PacketType::ARG_SCALAR || type == PacketType::ARG_POINTER) {
            if (client_rx_buffer_.size() < sizeof(ArgScalarPacket)) break;
            ArgScalarPacket apkt;
            std::memcpy(&apkt, client_rx_buffer_.data(), sizeof(ArgScalarPacket));
            client_rx_buffer_.erase(client_rx_buffer_.begin(), client_rx_buffer_.begin() + sizeof(ArgScalarPacket));
            
            PendingArg arg;
            arg.type = apkt.arg_type;
            arg.value = apkt.value;
            pending_args_[apkt.arg_index] = std::move(arg);
        } else if (type == PacketType::DATA_HEADER) {
            if (client_rx_buffer_.size() < sizeof(DataHeaderPacket)) break;
            DataHeaderPacket dpkt;
            std::memcpy(&dpkt, client_rx_buffer_.data(), sizeof(DataHeaderPacket));
            client_rx_buffer_.erase(client_rx_buffer_.begin(), client_rx_buffer_.begin() + sizeof(DataHeaderPacket));
            
            pending_target_ptr_ = dpkt.target_ptr;
            pending_data_size_ = dpkt.size;
        } else if (type == PacketType::DATA_BODY) {
            if (client_rx_buffer_.size() < sizeof(PacketType) + pending_data_size_) break;
            
            void* handle = reinterpret_cast<void*>(pending_target_ptr_);
            auto& mm = vgre::core::RuntimeEngine::instance().getMemoryManager();
            
            if (!mm.isValidHandle(handle)) {
               void* actual_ptr = nullptr;
               mm.allocateManagedAt(handle, pending_data_size_, actual_ptr);
            }
            
            void* local_ptr = mm.getPointer(handle);
            if (local_ptr) {
                std::memcpy(local_ptr, client_rx_buffer_.data() + sizeof(PacketType), pending_data_size_);
            }
            client_rx_buffer_.erase(client_rx_buffer_.begin(), client_rx_buffer_.begin() + sizeof(PacketType) + pending_data_size_);
        } else if (type == PacketType::STRUCT_DATA) {
            if (client_rx_buffer_.size() < sizeof(StructDataPacket)) break;
            StructDataPacket spkt;
            std::memcpy(&spkt, client_rx_buffer_.data(), sizeof(StructDataPacket));
            
            size_t required = sizeof(StructDataPacket) + sizeof(PacketType) + spkt.size;
            if (client_rx_buffer_.size() < required) break;

            client_rx_buffer_.erase(client_rx_buffer_.begin(), client_rx_buffer_.begin() + sizeof(StructDataPacket));
            
            PacketType btype = static_cast<PacketType>(client_rx_buffer_[0]);
            if (btype == PacketType::DATA_BODY) {
                PendingArg arg;
                arg.type = static_cast<uint8_t>(ArgType::STRUCT);
                arg.data.assign(client_rx_buffer_.data() + sizeof(PacketType), client_rx_buffer_.data() + sizeof(PacketType) + spkt.size);
                pending_args_[spkt.arg_index] = std::move(arg);
            }
            client_rx_buffer_.erase(client_rx_buffer_.begin(), client_rx_buffer_.begin() + sizeof(PacketType) + spkt.size);
        } else if (type == PacketType::REGISTER_KERNEL) {
            if (client_rx_buffer_.size() < sizeof(KernelRegisterPacket)) break;
            KernelRegisterPacket kpkt;
            std::memcpy(&kpkt, client_rx_buffer_.data(), sizeof(KernelRegisterPacket));
            
            size_t total_size = sizeof(KernelRegisterPacket) + kpkt.source_len;
            if (client_rx_buffer_.size() < total_size) break;

            VGRE_LOG_INFO("TCPCluster", "Worker: Registering remote-provided kernel: " + std::string(kpkt.name));
            std::string source(reinterpret_cast<const char*>(client_rx_buffer_.data() + sizeof(KernelRegisterPacket)), kpkt.source_len);
            vgre::KernelId actual_id;
            core::RuntimeEngine::instance().registerKernel(kpkt.name, source, actual_id);
            
            client_rx_buffer_.erase(client_rx_buffer_.begin(), client_rx_buffer_.begin() + total_size);
        } else if (type == PacketType::LAUNCH_KERNEL) {
            if (client_rx_buffer_.size() < sizeof(RemoteCommandPacket)) break;
            RemoteCommandPacket pkt;
            std::memcpy(&pkt, client_rx_buffer_.data(), sizeof(RemoteCommandPacket));
            client_rx_buffer_.erase(client_rx_buffer_.begin(), client_rx_buffer_.begin() + sizeof(RemoteCommandPacket));
            
            handleRemoteCommand(pkt);
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

  if (auth_token_ == 0) {
    VGRE_LOG_ERROR("TCPCluster",
                   "Remote kernel dispatch blocked: missing "
                   "VGRE_TCP_AUTH_TOKEN");
    return VGREResult::ERROR_INVALID_VALUE;
  }

  // 1. Stream all arguments FIRST
  for (int i = 0; i < num_args; ++i) {
    ArgType type = (i < static_cast<int>(argTypes.size())) ? argTypes[i] : ArgType::UINT64;
    
    if (type == ArgType::STRUCT) {
      // For structs, we need to know the size. JIT metadata is the most authoritative source.
      const auto *ir = core::RuntimeEngine::instance().getKernelIR(kernel_id);
      if (ir && i < static_cast<int>(ir->argSizes.size())) {
          size_t size = ir->argSizes[i];
          StructDataPacket spkt{};
          spkt.type = PacketType::STRUCT_DATA;
          spkt.arg_index = i;
          spkt.size = static_cast<uint32_t>(size);
          send_all(clients_[worker_idx].socket_fd, &spkt, sizeof(StructDataPacket));

          PacketType body_type = PacketType::DATA_BODY;
          send_all(clients_[worker_idx].socket_fd, &body_type, sizeof(PacketType));
          send_all(clients_[worker_idx].socket_fd, args[i], size);
      }
    } else if (type == ArgType::POINTER) {
      void* ptr = *static_cast<void**>(args[i]);
      uint64_t handle = reinterpret_cast<uint64_t>(ptr);
      
      // Memory Coherence: If it's a valid managed pointer, sync its data to the worker
      size_t size = core::RuntimeEngine::instance().getMemoryManager().getAllocationSize(ptr);
      if (size > 0 && ptr) {
          DataHeaderPacket dpkt{};
          dpkt.type = PacketType::DATA_HEADER;
          dpkt.target_ptr = handle;
          dpkt.size = size;
          send_all(clients_[worker_idx].socket_fd, &dpkt, sizeof(DataHeaderPacket));

          PacketType body_type = PacketType::DATA_BODY;
          send_all(clients_[worker_idx].socket_fd, &body_type, sizeof(PacketType));
          send_all(clients_[worker_idx].socket_fd, ptr, size);
      }

      ArgScalarPacket apkt{};
      apkt.type = PacketType::ARG_POINTER;
      apkt.arg_index = i;
      apkt.arg_type = static_cast<uint8_t>(type);
      apkt.value = handle;
      send_all(clients_[worker_idx].socket_fd, &apkt, sizeof(ArgScalarPacket));
    } else {
      // Scalar arguments (INT32, FLOAT32, etc.)
      ArgScalarPacket apkt{};
      apkt.type = PacketType::ARG_SCALAR;
      apkt.arg_index = i;
      apkt.arg_type = static_cast<uint8_t>(type);
      
      // All scalars are currently treated as 8-byte values in the RPC layer for simplicity
      std::memcpy(&apkt.value, args[i], 8);
      send_all(clients_[worker_idx].socket_fd, &apkt, sizeof(ArgScalarPacket));
    }
  }

  // 2. Send LAUNCH_KERNEL header LAST (this triggers execution on worker side)
  RemoteCommandPacket pkt{};
  pkt.type = PacketType::LAUNCH_KERNEL;
  pkt.auth_token = auth_token_;
  pkt.kernel_id = kernel_id;
  std::memcpy(pkt.grid_dim, grid_dim, sizeof(pkt.grid_dim));
  std::memcpy(pkt.block_dim, block_dim, sizeof(pkt.block_dim));
  pkt.shared_mem = shared_mem;
  pkt.num_args = num_args; 

  if (!send_all(clients_[worker_idx].socket_fd, &pkt, sizeof(RemoteCommandPacket))) {
    return VGREResult::ERROR_IO;
  }

  VGRE_LOG_INFO("TCPCluster",
                "Dispatched remote kernel launch (" + std::to_string(num_args) +
                    " args) to worker " + std::to_string(worker_idx));
  return VGREResult::SUCCESS;
}

void TCPClusterManager::broadcastKernelRegistration(uint64_t kernel_id,
                                                   const std::string &name,
                                                   const std::string &source) {
  if (!enabled_ || !is_master_)
    return;

  std::lock_guard<std::mutex> lock(clients_mutex_);
  KernelRegisterPacket kpkt{};
  kpkt.type = PacketType::REGISTER_KERNEL;
  kpkt.kernel_id = kernel_id;
  std::strncpy(kpkt.name, name.c_str(), sizeof(kpkt.name) - 1);
  kpkt.source_len = static_cast<uint32_t>(source.length());

  for (auto &client : clients_) {
    if (client.active) {
      send_all(client.socket_fd, &kpkt, sizeof(KernelRegisterPacket));
      send_all(client.socket_fd, source.c_str(), source.length());
    }
  }
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
                    std::to_string(pkt.kernel_id) + " | Args: " + std::to_string(pkt.num_args) + ")");

  int numArgs = pkt.num_args;
  std::vector<void*> local_args(numArgs, nullptr);
  
  // We need to keep the actual values alive during launchKernel.
  // We can't just point into the map because it might reallocate (though not during this loop).
  // But more importantly, scalar values of different sizes need correct alignment and size.
  // However, VGRE's standard internal dispatch uses 8-byte slots for all scalars for simplicity in many paths,
  // or expects the pointer to the actual type.
  
  for (int i = 0; i < numArgs; ++i) {
    auto it = pending_args_.find(i);
    if (it == pending_args_.end()) {
        VGRE_LOG_ERROR("TCPCluster", "Remote execution failed: missing argument data for index " + std::to_string(i));
        pending_args_.clear();
        return;
    }
    
    PendingArg& arg = it->second;
    ArgType type = static_cast<ArgType>(arg.type);
    
    if (type == ArgType::STRUCT) {
       local_args[i] = arg.data.data();
    } else if (type == ArgType::POINTER) {
       local_args[i] = &arg.value;
    } else {
       // For scalars, value is already stored in PendingArg::value (64-bit).
       // We can point directly to it.
       local_args[i] = &arg.value;
    }
  }

  dim3 gd(pkt.grid_dim[0], pkt.grid_dim[1], pkt.grid_dim[2]);
  dim3 bd(pkt.block_dim[0], pkt.block_dim[1], pkt.block_dim[2]);

  auto r = vgre::core::RuntimeEngine::instance().launchKernel(
      pkt.kernel_id, gd, bd, local_args.data(), pkt.shared_mem, 0);
  
  if (r != VGREResult::SUCCESS) {
    VGRE_LOG_ERROR("TCPCluster",
                   "Remote kernel execution failed with code " +
                       std::to_string(static_cast<int>(r)));
  }

  // --- Memory Coherence (Pull Back) ---
  // If the kernel was successful, sync modified managed pointers back to master.
  // In a real GPU driver, we'd track dirty pages, but for VGRE Phase 3 coherence,
  // we assume all pointer arguments are potentially outputs.
  for (int i = 0; i < numArgs; ++i) {
      auto it = pending_args_.find(i);
      if (it == pending_args_.end()) continue;
      
      PendingArg& arg = it->second;
      if (arg.type == static_cast<uint8_t>(ArgType::POINTER)) {
          void* ptr = reinterpret_cast<void*>(arg.value);
          size_t size = vgre::core::RuntimeEngine::instance().getMemoryManager().getAllocationSize(ptr);
          if (size > 0) {
              // 1. Send DATA_HEADER
              DataHeaderPacket dptr_pkt{};
              dptr_pkt.type = PacketType::DATA_HEADER;
              dptr_pkt.target_ptr = arg.value;
              dptr_pkt.size = size;
              send_all(client_fd_, &dptr_pkt, sizeof(DataHeaderPacket));

              // 2. Send DATA_BODY
              PacketType body_type = PacketType::DATA_BODY;
              send_all(client_fd_, &body_type, sizeof(PacketType));
              send_all(client_fd_, ptr, size);
          }
      }
  }

  // Send RESPONSE packet
  ResponsePacket resp{};
  resp.type = PacketType::RESPONSE;
  resp.kernel_id = pkt.kernel_id;
  resp.result = r;
  send_all(client_fd_, &resp, sizeof(ResponsePacket));
  
  // Clear pending args after launch
  pending_args_.clear();
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

void TCPClusterManager::getConnectedNodes(std::vector<ClientConnection> &outNodes) const {
  std::lock_guard<std::mutex> lock(clients_mutex_);
  outNodes = clients_;
}

} // namespace advanced
} // namespace vgre
