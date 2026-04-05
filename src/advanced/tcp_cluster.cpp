#include "vgre/advanced/tcp_cluster.h"
#include "vgre/advanced/secure_channel.h"
#include "vgre/advanced/hybrid_compute_manager.h"
#include "vgre/advanced/resource_ledger.h"
#include "vgre/advanced/workload_partitioner.h"
#include "vgre/advanced/hardware_token_manager.h"
#include "vgre/advanced/ipc_manager.h"
#include "vgre/common/logger.h"
#include "vgre/common/input_validation.h"
#include "vgre/core/memory_manager.h"
#include "vgre/core/runtime_engine.h"
#include "vgre/runtime/vector_engine.h"
#include "vgre/common/sockets.h"

// System Headers
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <fstream>
#include <sstream>

#include <chrono>

namespace vgre {
namespace advanced {

namespace {

// Using vgre::common types and helpers
using vgre::common::vgre_socket_t;
using vgre::common::VGRE_INVALID_SOCKET;
using vgre::common::VGRE_SOCKET_ERROR;
using vgre::common::vgre_close_socket;
using vgre::common::vgre_pollfd;
using vgre::common::vgre_poll;
using vgre::common::vgre_setsockopt;
using vgre::common::vgre_ioctl_nonblock;
using vgre::common::vgre_set_recv_timeout;
using vgre::common::vgre_get_last_socket_error;
using vgre::common::vgre_is_would_block;

template <typename T>
void sum_reduce(T* dst, const T* src, size_t count) {
  for (size_t i = 0; i < count; ++i)
    dst[i] += src[i];
}

static bool send_all(vgre_socket_t sock, const void *buf, size_t len, const std::atomic<bool>* enabled = nullptr) {
  const char *p = static_cast<const char *>(buf);
  size_t sent = 0;
  auto start = std::chrono::steady_clock::now();
  while (sent < len) {
    if (enabled && !enabled->load()) return false;
    
    // Check for 5 second timeout on blocking sends
    if (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start).count() > 5) {
        return false;
    }

    int n = send(sock, p + sent, static_cast<int>(len - sent), MSG_NOSIGNAL);
    if (n <= 0) {
      if (n < 0 && vgre_is_would_block(vgre_get_last_socket_error())) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        continue;
      }
      return false;
    }
    sent += n;
  }
  return true;
}
} // namespace

bool TCPClusterManager::send_packet(vgre_socket_t fd, PacketType type, const void *payload, size_t payloadLen, vgre::advanced::SecureChannel *sc) {
  (void)sc;
  static std::atomic<uint32_t> s_vgre_seq{1};

  // CRITICAL: Validate packet size to prevent buffer overflow
  if (common::InputValidator::validatePacketSize(payloadLen) != VGREResult::SUCCESS) {
    VGRE_LOG_ERROR("TCPCluster", "Packet size validation failed: " + std::to_string(payloadLen));
    return false;
  }

  // VSBP v0.1.2: Construct header and stage with payload
  size_t totalLen = sizeof(VSBPHeader) + payloadLen;
  
  // Additional check for total packet size
  if (common::InputValidator::validatePacketSize(totalLen) != VGREResult::SUCCESS) {
    VGRE_LOG_ERROR("TCPCluster", "Total packet size validation failed: " + std::to_string(totalLen));
    return false;
  }
  
  std::vector<uint8_t> staging(totalLen);
  
  VSBPHeader* header = reinterpret_cast<VSBPHeader*>(staging.data());
  header->magic = VSBP_MAGIC;
  header->version = VSBP_VERSION;
  header->type = static_cast<uint16_t>(type);
  header->sequence = s_vgre_seq.fetch_add(1, std::memory_order_relaxed);
  header->payloadSize = payloadLen;

  if (payload && payloadLen > 0) {
    std::memcpy(staging.data() + sizeof(VSBPHeader), payload, payloadLen);
  }

  uint32_t priority = 1; // Default: LOW (Data/Bulk)
  if (type == PacketType::RESPONSE || type == PacketType::PARTITION_RESULT ||
      type == PacketType::TELEMETRY || type == PacketType::SECURE_HANDSHAKE ||
      type == PacketType::SECURE_HANDSHAKE_ACK || type == PacketType::ROTATE_KEY ||
      type == PacketType::CREDIT_REPORT || type == PacketType::COOP_BARRIER_SYNC ||
      type == PacketType::COOP_BARRIER_RESUME) {
    priority = 0; // HIGH (Control/Sync)
  }

  // Master side lookup — enqueue into per-client TSS2 queue
  bool foundMasterClient = false;
  {
      std::lock_guard<std::recursive_mutex> lock(clients_mutex_);
      for (auto &client : clients_) {
          if (client && client->socket_fd == fd) {
              std::lock_guard<std::mutex> tx_lock(client->tx_mutex);
              ClientConnection::OutgoingPacket pkt;
              pkt.data = std::move(staging);
              pkt.priority = priority;
              if (priority == 0) {
                  client->high_priority_tx.push_front(std::move(pkt));
              } else {
                  client->low_priority_tx.push_back(std::move(pkt));
              }
              foundMasterClient = true;
              break;
          }
      }
  }
  if (foundMasterClient) return true;

  // Worker-side path: enqueue into worker-local TSS2 queues (drained by clientLoop)
  if (!is_master_ && fd == client_fd_) {
      std::lock_guard<std::mutex> lock(client_tx_mutex_);
      ClientConnection::OutgoingPacket pkt;
      pkt.data = std::move(staging);
      pkt.priority = priority;
      if (priority == 0) {
          client_high_priority_tx_.push_front(std::move(pkt));
      } else {
          client_low_priority_tx_.push_back(std::move(pkt));
      }
      return true;
  }

  return false;
}

bool TCPClusterManager::send_packet_direct(vgre_socket_t fd, PacketType type, const void *payload, size_t payloadLen, vgre::advanced::SecureChannel *sc) {
  static std::atomic<uint32_t> s_vgre_seq{1};

  size_t totalLen = sizeof(VSBPHeader) + payloadLen;
  std::vector<uint8_t> staging(totalLen);
  
  VSBPHeader* header = reinterpret_cast<VSBPHeader*>(staging.data());
  header->magic = VSBP_MAGIC;
  header->version = VSBP_VERSION;
  header->type = static_cast<uint16_t>(type);
  header->sequence = s_vgre_seq.fetch_add(1, std::memory_order_relaxed);
  header->payloadSize = payloadLen;

  if (payload && payloadLen > 0) {
    std::memcpy(staging.data() + sizeof(VSBPHeader), payload, payloadLen);
  }

  global_packets_sent_++;
  global_bytes_sent_ += staging.size();

  if (sc && sc->isInitialized()) {
    return sc->sendSecure(fd, staging.data(), staging.size()) == VGREResult::SUCCESS;
  } else {
    std::atomic<bool> enabled{true};
    return send_all(fd, staging.data(), staging.size(), &enabled);
  }
}

int TCPClusterManager::recv_packet(vgre_socket_t fd, std::vector<uint8_t> &outBuffer, vgre::advanced::SecureChannel *sc) {
  if (sc && sc->isInitialized()) {
    std::vector<uint8_t> payload;
    VGREResult r = sc->recvSecure(fd, payload);
    if (r == VGREResult::SUCCESS) {
      global_packets_received_++;
      global_bytes_received_ += static_cast<uint64_t>(payload.size() + sizeof(SecurePacketHeader)); // Total wire size
      outBuffer.insert(outBuffer.end(), payload.begin(), payload.end());
      return static_cast<int>(payload.size());
    } else if (r == VGREResult::ERR_TIMEOUT) {
      return 0;
    }
    return -1; // Error
  } else {
    char temp[8192];
    int n = recv(fd, temp, sizeof(temp), 0);
    if (n > 0) {
      global_packets_received_++;
      global_bytes_received_ += static_cast<uint64_t>(n);
      outBuffer.insert(outBuffer.end(), temp, temp + n);
      return n;
    } else if (n == 0) {
      return -1; // Disconnected
    } else {
      if (vgre_is_would_block(vgre_get_last_socket_error())) return 0;
      return -1;
    }
  }
}

void TCPClusterManager::flush_tx_queues(std::shared_ptr<ClientConnection> clientPtr) {
    if (!clientPtr) return;
    auto &client = *clientPtr;
    if (!client.active || client.socket_fd == VGRE_INVALID_SOCKET) return;
    
    std::lock_guard<std::mutex> tx_lock(client.tx_mutex);
    
    // 1. Drain High Priority (Sync/Control)
    // We insert at front for urgency, so we pop from back to maintain order within priority
    while (!client.high_priority_tx.empty()) {
        auto &pkt = client.high_priority_tx.back();
        bool success = false;
        if (client.secureChannel && client.secureChannel->isInitialized()) {
            success = (client.secureChannel->sendSecure(client.socket_fd, pkt.data.data(), pkt.data.size()) == VGREResult::SUCCESS);
        } else {
            success = send_all(client.socket_fd, pkt.data.data(), pkt.data.size());
        }
        
        if (success) {
            global_packets_sent_++;
            global_bytes_sent_ += pkt.data.size();
            client.high_priority_tx.pop_back();
        } else {
            return; // Socket buffer full
        }
    }
    
    // 2. Drain Low Priority (Data/Bulk)
    while (!client.low_priority_tx.empty()) {
        auto &qItem = client.low_priority_tx.front();
        bool success = false;
        if (client.secureChannel && client.secureChannel->isInitialized()) {
            success = (client.secureChannel->sendSecure(client.socket_fd, qItem.data.data(), qItem.data.size()) == VGREResult::SUCCESS);
        } else {
            success = send_all(client.socket_fd, qItem.data.data(), qItem.data.size(), &enabled_);
        }
        
        if (is_master_ && qItem.data.size() >= sizeof(VSBPHeader)) {
            VSBPHeader h;
            std::memcpy(&h, qItem.data.data(), sizeof(VSBPHeader));
            std::cout << "[MASTER TCP TRACE] Sent packet Type=" << (int)h.type << " Size=" << qItem.data.size() << " Success=" << success << std::endl;
        }

        if (success) {
            global_packets_sent_++;
            global_bytes_sent_ += qItem.data.size();
            client.low_priority_tx.pop_front();
        } else {
            return; // Socket buffer full
        }
    }
}

TCPClusterManager::TCPClusterManager() {
  stop_proactive_ = false;
}

TCPClusterManager::~TCPClusterManager() { shutdown(); }

bool TCPClusterManager::broadcastPacket(PacketType type, const void *payload,
                                      size_t payloadLen) {
  if (!is_master_) return false;
  std::lock_guard<std::recursive_mutex> lock(clients_mutex_);
  bool success = true;
  for (auto &client : clients_) {
    if (client && client->active) {
      if (!send_packet(client->socket_fd, type, payload, payloadLen, client->secureChannel.get())) {
        success = false;
      }
    }
  }
  return success;
}

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
    return VGREResult::ERR_IO;
  }
#endif

  is_master_ = is_master;
  host_ = host;
  port_ = port;
  auth_token_ = 0;
  
  // Phase 10: Authoritative Token Retrieval
  // Priority 1: Environment variable (Overridable for tests/CI)
  const char* env_token = std::getenv("VGRE_TCP_AUTH_TOKEN");
  if (env_token) {
      auth_token_str_ = env_token;
      VGRE_LOG_INFO("TCPCluster", "Auth Token retrieved from environment variable VGRE_TCP_AUTH_TOKEN");
  } else {
      // Priority 2: Hardware-backed secure storage (Production)
      auto& tokenMgr = HardwareTokenManager::instance();
      if (tokenMgr.initialize() != VGREResult::SUCCESS) {
          VGRE_LOG_ERROR("TCPCluster", "Failed to initialize hardware token manager");
          enabled_ = false;
          return VGREResult::ERR_NOT_INITIALIZED;
      }
      
      VGREResult tr = tokenMgr.getToken("vgre_tcp_cluster", auth_token_str_);
      if (tr != VGREResult::SUCCESS) {
          // Generate and store new token if none exists
          auth_token_str_ = HardwareTokenManager::generateToken(32);
          VGREResult sr = tokenMgr.storeToken("vgre_tcp_cluster", auth_token_str_);
          if (sr != VGREResult::SUCCESS) {
              VGRE_LOG_ERROR("TCPCluster", "Failed to store authentication token");
              enabled_ = false;
              return VGREResult::ERR_IO;
          }
          VGRE_LOG_INFO("TCPCluster", "Generated and stored new authentication token using " + 
                        tokenMgr.getBackendName());
      } else {
          VGRE_LOG_INFO("TCPCluster", "Retrieved authentication token from " + 
                        tokenMgr.getBackendName());
      }
  }

  if (!auth_token_str_.empty()) {
      // Use stable FNV-1a hash instead of std::hash (which is not stable across processes)
      uint64_t hash = 0xcbf29ce484222325ULL;
      for (char c : auth_token_str_) {
          hash ^= static_cast<uint64_t>(c);
          hash *= 0x100000001b3ULL;
      }
      auth_token_ = hash;
      VGRE_LOG_INFO("TCPCluster", "Auth Token specialized (stable hash generated).");
  }

  if (is_master_) {
    // Master Node (Server)
    server_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd_ == VGRE_INVALID_SOCKET) {
      VGRE_LOG_ERROR("TCPCluster", "Failed to create socket");
      enabled_ = false;
      return VGREResult::ERR_IO;
    }

    int opt = 1;
    vgre_setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#ifndef _WIN32
#ifdef SO_REUSEPORT
    vgre_setsockopt(server_fd_, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(opt));
#endif
#endif

    struct sockaddr_in address;
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port_);

    if (bind(server_fd_, (struct sockaddr *)&address, sizeof(address)) < 0) {
      VGRE_LOG_ERROR("TCPCluster",
                     "Bind failed on port " + std::to_string(port_));
      vgre_close_socket(server_fd_);
      server_fd_ = VGRE_INVALID_SOCKET;
      enabled_ = false;
      return VGREResult::ERR_IO;
    }

    if (listen(server_fd_, 10) < 0) {
      VGRE_LOG_ERROR("TCPCluster", "Listen failed");
      vgre_close_socket(server_fd_);
      server_fd_ = VGRE_INVALID_SOCKET;
      enabled_ = false;
      return VGREResult::ERR_IO;
    }

    vgre_ioctl_nonblock(server_fd_);

    VGRE_LOG_INFO("TCPCluster",
                  "Master Server Listening on port " + std::to_string(port_));
    cluster_thread_ = std::thread(&TCPClusterManager::serverLoop, this);
    udp_thread_ = std::thread(&TCPClusterManager::udpAnnouncerLoop, this);
    master_discovery_thread_ = std::thread(&TCPClusterManager::udpMasterDiscoveryLoop, this);

    // Phase 12: Proactive Node Discovery
    // Always start the proactive thread so UDP-discovered workers (added
    // dynamically by udpMasterDiscoveryLoop) are also picked up. The loop
    // idles safely when the address list is empty.
    parseProactiveNodes();
    stop_proactive_ = false;
    proactive_thread_ = std::thread(&TCPClusterManager::proactiveConnectionLoop, this);
  } else {
    // Client Node (Worker)
    if (host_ == "auto" || host_.empty()) {
      VGRE_LOG_INFO("TCPCluster", "Worker: Starting in standby mode (listening on port " + std::to_string(port_) + ")");
      
      // Start a server socket for the worker as well, so the Master can proactively connect.
      server_fd_ = socket(AF_INET, SOCK_STREAM, 0);
      if (server_fd_ != VGRE_INVALID_SOCKET) {
          int opt = 1;
          vgre_setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
          struct sockaddr_in address;
          address.sin_family = AF_INET;
          address.sin_addr.s_addr = INADDR_ANY;
          address.sin_port = htons(port_);

          if (bind(server_fd_, (struct sockaddr *)&address, sizeof(address)) >= 0 && listen(server_fd_, 5) >= 0) {
              vgre_ioctl_nonblock(server_fd_);
              cluster_thread_ = std::thread(&TCPClusterManager::serverLoop, this);
          } else {
              VGRE_LOG_WARN("TCPCluster", "Worker: Failed to bind server socket, falling back to UDP discovery only.");
          }
      }

      udp_thread_ = std::thread(&TCPClusterManager::udpDiscoveryLoop, this);
      worker_announcer_thread_ = std::thread(&TCPClusterManager::udpWorkerAnnouncerLoop, this);
    } else {
    client_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (client_fd_ == VGRE_INVALID_SOCKET) {
      VGRE_LOG_ERROR("TCPCluster", "Failed to create client socket");
      enabled_ = false;
      return VGREResult::ERR_IO;
    }

    struct sockaddr_in serv_addr;
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(port_);

    if (inet_pton(AF_INET, host_.c_str(), &serv_addr.sin_addr) <= 0) {
      VGRE_LOG_ERROR("TCPCluster",
                     "Invalid address or not supported: " + host_);
      vgre_close_socket(client_fd_);
      client_fd_ = VGRE_INVALID_SOCKET;
      enabled_ = false;
      return VGREResult::ERR_IO;
    }

    if (connect(client_fd_, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) <
        0) {
      VGRE_LOG_DEBUG("TCPCluster", "Connection failed to " + host_ + ":" +
                                       std::to_string(port_));
      enabled_ = false;
      vgre_close_socket(client_fd_);
      client_fd_ = VGRE_INVALID_SOCKET;
#if defined(_WIN32)
      WSACleanup();
#endif
      return VGREResult::ERR_IO;
    }

    // Set non-blocking
    vgre_ioctl_nonblock(client_fd_);

    VGRE_LOG_INFO("TCPCluster", "Connected to Remote Master Node at " + host_);
    cluster_thread_ = std::thread(&TCPClusterManager::clientLoop, this);
    data_processor_thread_ = std::thread(&TCPClusterManager::processClientStagingBuffer, this);
    }
  }

  enabled_ = true;
  return VGREResult::SUCCESS;
}

void TCPClusterManager::shutdown() {
  bool wasEnabled = enabled_.exchange(false);

  // Always close fds and notify CVs so threads that set enabled_=false
  // themselves (e.g. clientLoop on disconnect) also get unblocked.
  // 1. Force-close fds to unblock threads in recv/send/poll/accept
  {
    if (is_master_) {
      std::lock_guard<std::recursive_mutex> lock(clients_mutex_);
      for (auto &client : clients_) {
        if (client && client->socket_fd != VGRE_INVALID_SOCKET) {
          vgre_close_socket(client->socket_fd);
          client->socket_fd = VGRE_INVALID_SOCKET;
          client->active = false;
        }
      }
      if (server_fd_ != VGRE_INVALID_SOCKET) {
        vgre_close_socket(server_fd_);
        server_fd_ = VGRE_INVALID_SOCKET;
      }
    } else {
      std::lock_guard<std::mutex> lock(client_mutex_);
      if (client_fd_ != VGRE_INVALID_SOCKET) {
        vgre_close_socket(client_fd_);
        client_fd_ = VGRE_INVALID_SOCKET;
      }
    }
  }

  // 2. Unblock staging wait condition and result wait condition.
  //    Must happen even when wasEnabled==false so data_processor_thread_
  //    wakes from staging_cv_ when clientLoop already set enabled_=false.
  {
    std::lock_guard<std::mutex> lock(staging_mutex_);
    staging_ready_ = true;
  }
  staging_cv_.notify_all();
  remote_results_cv_.notify_all();
  barrier_cv_.notify_all();

  // 3. Join all threads unconditionally. Threads may have exited on their own
  //    (e.g. clientLoop detected disconnect and set enabled_=false). We still
  //    must join to avoid std::terminate() in ~thread().
  if (udp_thread_.joinable()) {
    udp_thread_.join();
  }
  if (data_processor_thread_.joinable()) {
    data_processor_thread_.join();
  }
  if (cluster_thread_.joinable()) {
    cluster_thread_.join();
  }
  
  stop_proactive_ = true;
  if (proactive_thread_.joinable()) {
      proactive_thread_.join();
  }
  // These two threads were previously never joined, causing std::terminate()
  // in ~std::thread() when the singleton was destroyed or re-initialized.
  // udpMasterDiscoveryLoop now has SO_RCVTIMEO so recvfrom times out within 1s.
  // udpWorkerAnnouncerLoop already sleeps in short increments checked against enabled_.
  if (master_discovery_thread_.joinable()) {
      master_discovery_thread_.join();
  }
  if (worker_announcer_thread_.joinable()) {
      worker_announcer_thread_.join();
  }

  // Clear SHM nodes on shutdown to prevent stale data in dashboard
  if (is_master_) {
    vgre::advanced::IPCManager::instance().updateClusterNodes({});
  }

  if (!wasEnabled) return; // No further cleanup needed for second caller

  if (is_master_) {
    std::lock_guard<std::recursive_mutex> lock(clients_mutex_);
    clients_.clear();
  }
}

void TCPClusterManager::serverLoop() {
  VGRE_LOG_DEBUG("TCPCluster", "Master Server Loop starting...");

  while (enabled_) {
    // 0. Purge dead clients (active=false and no handshake in flight).
    // Release the lock BEFORE calling syncToIPC() to avoid lock-order inversion
    // between clients_mutex_ and IPCManager::mutex_.
    {
        bool anyRemoved = false;
        {
            std::lock_guard<std::recursive_mutex> lock(clients_mutex_);
            auto before = clients_.size();
            clients_.erase(
                std::remove_if(clients_.begin(), clients_.end(),
                    [](const std::shared_ptr<ClientConnection>& c) {
                        return c && !c->active && !c->is_authenticating;
                    }),
                clients_.end());
            if (clients_.size() != before) {
                VGRE_LOG_DEBUG("TCPCluster",
                    "Purged " + std::to_string(before - clients_.size()) +
                    " dead client(s); remaining=" + std::to_string(clients_.size()));
                anyRemoved = true;
            }
        } // release clients_mutex_ before touching IPC
        if (anyRemoved) syncToIPC(); // push fresh (dead-free) snapshot
    }

    // 1. Prepare poll fds
    std::vector<vgre_pollfd> fds;
    
    // Listening socket
    vgre_pollfd pfd_server;
    pfd_server.fd = server_fd_;
    pfd_server.events = POLLIN;
    pfd_server.revents = 0;
    fds.push_back(pfd_server);
    
    // Client sockets
    {
        std::lock_guard<std::recursive_mutex> lock(clients_mutex_);
        for (const auto &client : clients_) {
            if (client && client->active && client->socket_fd != VGRE_INVALID_SOCKET) {
                vgre_pollfd pfd_client;
                pfd_client.fd = client->socket_fd;
                pfd_client.events = POLLIN;
                pfd_client.revents = 0;
                fds.push_back(pfd_client);
            }
        }
    }
    
    int poll_res = vgre_poll(fds.data(), fds.size(), 50);

    if (poll_res < 0) {
        if (!vgre_is_would_block(vgre_get_last_socket_error())) {
           VGRE_LOG_ERROR("TCPCluster", "Master: poll() failed");
           break;
        }
        continue;
    }
    
    // Phase 12: TSS2 Unconditional Priority Flush — must run even on poll timeout so
    // that packets enqueued by application threads (broadcastKernelRegistration,
    // launchRemoteKernel, etc.) are transmitted without waiting for incoming data.
    {
        std::lock_guard<std::recursive_mutex> lock(clients_mutex_);
        for (auto &client : clients_) {
            if (client && client->active) flush_tx_queues(client);
        }
    }

    if (poll_res == 0) continue; // No incoming data this iteration

    // 2. Handle new connections
    if (fds[0].revents & POLLIN) {
        struct sockaddr_in address;
        socklen_t addrlen = sizeof(address);
        vgre_socket_t new_socket = accept(server_fd_, (struct sockaddr *)&address, &addrlen);
        
        if (new_socket != VGRE_INVALID_SOCKET) {
            char ipstr[INET_ADDRSTRLEN];
            inet_ntop(AF_INET, &(address.sin_addr), ipstr, sizeof(ipstr));

            // Rate-limit new connections to protect PBKDF2 handshake from DoS
            if (!rateLimiter_.isAllowed(std::string(ipstr))) {
                VGRE_LOG_WARN("TCPCluster",
                    "Rate limit exceeded for " + std::string(ipstr) +
                    " — dropping connection");
                vgre_close_socket(new_socket);
            } else {
            rateLimiter_.record(std::string(ipstr));

            vgre_ioctl_nonblock(new_socket);
            // Enable TCP keepalive so hard crashes / network partitions are detected
            // without waiting for a send to fail (typically within ~11s with these settings).
            {
                int ka = 1;
                vgre_setsockopt(new_socket, SOL_SOCKET, SO_KEEPALIVE, (char*)&ka, sizeof(ka));
#if defined(TCP_KEEPIDLE)
                int idle = 5;   vgre_setsockopt(new_socket, IPPROTO_TCP, TCP_KEEPIDLE,  &idle, sizeof(idle));
#endif
#if defined(TCP_KEEPINTVL)
                int intvl = 2;  vgre_setsockopt(new_socket, IPPROTO_TCP, TCP_KEEPINTVL, &intvl, sizeof(intvl));
#endif
#if defined(TCP_KEEPCNT)
                int cnt = 3;    vgre_setsockopt(new_socket, IPPROTO_TCP, TCP_KEEPCNT,   &cnt,  sizeof(cnt));
#endif
            }

            // ── Worker fast-path ───────────────────────────────────────────
            // When a WORKER's server socket gets an inbound connection from
            // the master (proactive path), we must NOT add it to clients_ or
            // run the server-side handshake from here — that would conflict
            // with clientLoop() which is the proper owner of that socket.
            // Just store the fd and let udpDiscoveryLoop → clientLoop handle
            // the full handshake + telemetry cycle.
            if (!is_master_) {
                VGRE_LOG_INFO("TCPCluster",
                    "Worker: Accepted inbound connection from Master at " +
                    std::string(ipstr) + " — handing off to clientLoop");
                client_fd_ = new_socket;
                continue; // Skip master-only clients_/IPC/handshake code below
            }

            std::lock_guard<std::recursive_mutex> lock(clients_mutex_);
            auto conn = std::make_shared<ClientConnection>();
            conn->socket_fd = new_socket;
            conn->ip_address = std::string(ipstr);
            conn->port = ntohs(address.sin_port); // Remote port of the connection
            conn->active = true;
            conn->expecting_type = true;
            conn->rx_buffer.clear();
            clients_.push_back(std::move(conn));

            VGRE_LOG_INFO("TCPCluster", "Master: Accepted new remote node from " +
                std::string(ipstr) + ":" + std::to_string(ntohs(address.sin_port)));

            // Phase 11: Detect local connection for SHM optimization
            if (std::string(ipstr) == "127.0.0.1" || std::string(ipstr) == "::1") {
                VGRE_LOG_INFO("TCPCluster", "Master: Local connection detected, initiating SHM transport...");
                clients_.back()->is_local = true;

                // Create unique SHM segment for this client
                clients_.back()->shmManager = std::make_unique<vgre::core::ShmManager>();
                std::string shmName = "vgre_shm_" + std::to_string(new_socket);
                size_t shmSize = 256 * 1024 * 1024; // 256MB default

                if (clients_.back()->shmManager->open(shmName, shmSize, true) == VGREResult::SUCCESS) {
                    ShmInitPacket sipkt{};
                    std::strncpy(sipkt.shm_name, shmName.c_str(), sizeof(sipkt.shm_name) - 1);
                    sipkt.shm_size = shmSize;
                    send_packet(new_socket, PacketType::SHM_INIT, &sipkt, sizeof(ShmInitPacket));
                }
            }

            // Phase 13: Instant IPC Sync for accepted connection
            syncToIPC();

            // Phase 13: Asynchronous Handshake Handover
            if (security_enabled_) {
                std::shared_ptr<ClientConnection> clientRef = clients_.back();
                clientRef->is_authenticating = true;
                clientRef->handshake_start_ms = static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now().time_since_epoch()).count());
                clientRef->active = true; // Mark as active so it appears in UI (as authenticating)
                
                std::thread([this, clientRef]() {
                    VGREResult sr = this->performSecureHandshake(clientRef);
                    clientRef->is_authenticating = false;

                    if (sr != VGREResult::SUCCESS) {
                        VGRE_LOG_ERROR("TCPCluster", "Master: Security handshake failed for " + clientRef->ip_address);
                        clientRef->active = false;
                        vgre_close_socket(clientRef->socket_fd);
                        {
                            std::lock_guard<std::recursive_mutex> lock(clients_mutex_);
                            clients_.erase(std::remove(clients_.begin(), clients_.end(), clientRef),
                                           clients_.end());
                        }
                    } else {
                        // Mark fully authenticated so dashboard shows SECURED status.
                        clientRef->security_established = true;
                        VGRE_LOG_INFO("TCPCluster",
                            "Master: Security established for " + clientRef->ip_address);
                    }
                    // Always push the updated state to the IPC shared-memory segment
                    // so the dashboard reflects the outcome immediately.
                    this->syncToIPC();
                }).detach();
            } else {
                clients_.back()->active = true;
                clients_.back()->security_established = true;
            }
            } // end rate-limit else
        }
    }

    // 3. Handle data from existing clients
    {
        std::lock_guard<std::recursive_mutex> lock(clients_mutex_);
        for (size_t i = 1; i < fds.size(); ++i) {
            if (fds[i].revents & (POLLIN | POLLERR | POLLHUP)) {
                // Find matching client
                for (auto &client : clients_) {
                    if (client && client->socket_fd == fds[i].fd) {
                        if (fds[i].revents & POLLIN) {
                            int n = recv_packet(client->socket_fd, client->rx_buffer, client->secureChannel.get());
                            if (n < 0) {
                                VGRE_LOG_INFO("TCPCluster", "Master: Worker disconnected.");
                                client->active = false;
                                vgre_close_socket(client->socket_fd);
                                syncToIPC();
                            }
                        } else {
                            VGRE_LOG_WARN("TCPCluster", "Master: Client socket error or hup.");
                            client->active = false;
                            vgre_close_socket(client->socket_fd);
                            syncToIPC();
                        }
                        break;
                    }
                }
            }
        }
        
        // Phase 12: TSS2 Priority Flush
        for (auto &client : clients_) {
            if (client && client->active) flush_tx_queues(client);
        }
        
        // Process buffers — VSBP v0.1.2 framed parsing (mirrors processClientStagingBuffer)
        for (auto &client : clients_) {
          if (!client || !client->active || client->rx_buffer.empty()) continue;

          while (client->active && !client->rx_buffer.empty()) {
            if (client->receive_state == ReceiveState::IDLE) {
              // Need at least a full VSBP header
              if (client->rx_buffer.size() < sizeof(VSBPHeader)) break;

              VSBPHeader hdr;
              std::memcpy(&hdr, client->rx_buffer.data(), sizeof(VSBPHeader));

              if (hdr.magic != VSBP_MAGIC || hdr.version != VSBP_VERSION) {
                VGRE_LOG_ERROR("TCPCluster",
                    "Master: VSBP protocol violation from " + client->ip_address +
                    " — clearing buffer");
                client->rx_buffer.clear();
                break;
              }

              if (client->rx_buffer.size() < sizeof(VSBPHeader) + hdr.payloadSize) break;

              const uint8_t* payload = client->rx_buffer.data() + sizeof(VSBPHeader);
              PacketType type = static_cast<PacketType>(hdr.type);
              size_t totalLen = sizeof(VSBPHeader) + hdr.payloadSize;

              if (type == PacketType::TELEMETRY) {
                if (hdr.payloadSize < sizeof(vgre_telemetry_t)) { client->rx_buffer.clear(); break; }
                vgre_telemetry_t tel;
                std::memcpy(&tel, payload, sizeof(vgre_telemetry_t));
                client->last_telemetry = tel;
                vgre::advanced::HybridComputeManager::instance().updateRemoteNodeTelemetry(client->ip_address, tel);
                client->rx_buffer.erase(client->rx_buffer.begin(), client->rx_buffer.begin() + totalLen);
              } else if (type == PacketType::RESPONSE) {
                if (hdr.payloadSize < sizeof(ResponsePacket)) { client->rx_buffer.clear(); break; }
                ResponsePacket resp;
                std::memcpy(&resp, payload, sizeof(ResponsePacket));
                client->rx_buffer.erase(client->rx_buffer.begin(), client->rx_buffer.begin() + totalLen);
                VGRE_LOG_DEBUG("TCPCluster", "Master: Received RESPONSE from worker (Kernel: " +
                    std::to_string(resp.kernel_id) + ")");
                if (client->in_flight_kernels > 0) client->in_flight_kernels--;
                {
                  std::lock_guard<std::mutex> lock_r(remote_results_mutex_);
                  remote_kernel_results_[resp.kernel_id] = resp.result;
                }
                remote_results_cv_.notify_all();
              } else if (type == PacketType::DATA_HEADER) {
                if (hdr.payloadSize < sizeof(DataHeaderPacket)) { client->rx_buffer.clear(); break; }
                DataHeaderPacket dpkt;
                std::memcpy(&dpkt, payload, sizeof(DataHeaderPacket));
                client->rx_buffer.erase(client->rx_buffer.begin(), client->rx_buffer.begin() + totalLen);
                client->pending_target_ptr = dpkt.target_ptr;
                client->pending_data_size = static_cast<uint32_t>(dpkt.size);
                client->pending_num_ranges = 0;
                client->pending_range_offset = 0;
                client->receive_state = ReceiveState::EXPECTING_BODY;
              } else if (type == PacketType::DATA_HEADER_DIRTY) {
                if (hdr.payloadSize < sizeof(DataHeaderDirtyPacket)) { client->rx_buffer.clear(); break; }
                DataHeaderDirtyPacket dhpkt;
                std::memcpy(&dhpkt, payload, sizeof(DataHeaderDirtyPacket));
                client->rx_buffer.erase(client->rx_buffer.begin(), client->rx_buffer.begin() + totalLen);
                client->pending_target_ptr = dhpkt.target_ptr;
                client->pending_num_ranges = dhpkt.num_ranges;
                client->receive_state = ReceiveState::EXPECTING_RANGES_TCP;
              } else if (type == PacketType::DATA_SHM_DIRTY) {
                if (hdr.payloadSize < sizeof(DataShmDirtyPacket)) { client->rx_buffer.clear(); break; }
                DataShmDirtyPacket dspkt;
                std::memcpy(&dspkt, payload, sizeof(DataShmDirtyPacket));
                client->rx_buffer.erase(client->rx_buffer.begin(), client->rx_buffer.begin() + totalLen);
                client->pending_target_ptr = dspkt.target_ptr;
                client->pending_num_ranges = dspkt.num_ranges;
                client->pending_shm_offset = dspkt.shm_offset;
                client->receive_state = ReceiveState::EXPECTING_RANGES_SHM;
              } else if (type == PacketType::DATA_SHM) {
                if (hdr.payloadSize < sizeof(DataShmPacket)) { client->rx_buffer.clear(); break; }
                DataShmPacket dspkt;
                std::memcpy(&dspkt, payload, sizeof(DataShmPacket));
                client->rx_buffer.erase(client->rx_buffer.begin(), client->rx_buffer.begin() + totalLen);
                if (client->is_local && client->shmManager) {
                  void* ptr = core::RuntimeEngine::instance().getMemoryManager()
                                  .getPointer(reinterpret_cast<void*>(dspkt.target_ptr));
                  if (ptr)
                    std::memcpy(ptr,
                        static_cast<uint8_t*>(client->shmManager->getBasePtr()) + dspkt.shm_offset,
                        dspkt.size);
                }
              } else if (type == PacketType::CAPABILITY) {
                if (hdr.payloadSize < sizeof(CapabilityPacket)) { client->rx_buffer.clear(); break; }
                CapabilityPacket cpkt;
                std::memcpy(&cpkt, payload, sizeof(CapabilityPacket));
                client->rx_buffer.erase(client->rx_buffer.begin(), client->rx_buffer.begin() + totalLen);
                client->cpu_cores = cpkt.cpu_cores;
                client->cpu_memory = cpkt.cpu_memory;
                client->has_igpu = cpkt.has_igpu;
                std::snprintf(client->igpu_name, sizeof(client->igpu_name), "%s", cpkt.igpu_name);
                HybridComputeManager::instance().updateRemoteNodeCapability(
                    client->ip_address, cpkt.cpu_cores, cpkt.cpu_memory,
                    cpkt.has_igpu, cpkt.igpu_name);
              } else if (type == PacketType::PARTITION_RESULT) {
                if (hdr.payloadSize < sizeof(PartitionResultPacket)) { client->rx_buffer.clear(); break; }
                PartitionResultPacket prpkt;
                std::memcpy(&prpkt, payload, sizeof(PartitionResultPacket));
                client->rx_buffer.erase(client->rx_buffer.begin(), client->rx_buffer.begin() + totalLen);
                if (client->in_flight_kernels > 0) client->in_flight_kernels--;
                {
                  std::lock_guard<std::mutex> lock_p(partition_mutex_);
                  partition_results_.push_back({prpkt.partition_id, prpkt.result, prpkt.execution_time_ms});
                  partition_cv_.notify_all();
                }
              } else if (type == PacketType::CREDIT_REPORT) {
                if (hdr.payloadSize < sizeof(CreditReportPacket)) { client->rx_buffer.clear(); break; }
                CreditReportPacket crpkt;
                std::memcpy(&crpkt, payload, sizeof(CreditReportPacket));
                client->rx_buffer.erase(client->rx_buffer.begin(), client->rx_buffer.begin() + totalLen);
                ResourceLedger::instance().recordCompute(client->ip_address,
                    crpkt.compute_seconds, crpkt.cpu_cores, crpkt.kernel_id,
                    CreditDirection::DEBIT);
              } else if (type == PacketType::ROTATE_KEY) {
                if (hdr.payloadSize < sizeof(SecureHandshakePacket)) { client->rx_buffer.clear(); break; }
                SecureHandshakePacket rpkt;
                std::memcpy(&rpkt, payload, sizeof(SecureHandshakePacket));
                client->rx_buffer.erase(client->rx_buffer.begin(), client->rx_buffer.begin() + totalLen);
                if (client->secureChannel) client->secureChannel->rotateKey(rpkt.nonce);
              } else if (type == PacketType::COOP_BARRIER_SYNC) {
                client->rx_buffer.erase(client->rx_buffer.begin(), client->rx_buffer.begin() + totalLen);
                {
                  std::lock_guard<std::mutex> lock_b(barrier_mutex_);
                  barrier_count_++;
                }
                barrier_cv_.notify_all();
              } else {
                // Unknown or unhandled packet type: skip entire packet
                client->rx_buffer.erase(client->rx_buffer.begin(), client->rx_buffer.begin() + totalLen);
              }
            } else if (client->receive_state == ReceiveState::EXPECTING_RANGES_TCP) {
              // DIRTY_RANGE is VSBP-framed
              if (client->rx_buffer.size() < sizeof(VSBPHeader) + sizeof(DirtyRangePacket)) break;
              VSBPHeader hdr;
              std::memcpy(&hdr, client->rx_buffer.data(), sizeof(VSBPHeader));
              if (hdr.magic != VSBP_MAGIC) { client->rx_buffer.clear(); client->receive_state = ReceiveState::IDLE; break; }
              DirtyRangePacket rpkt;
              std::memcpy(&rpkt, client->rx_buffer.data() + sizeof(VSBPHeader), sizeof(DirtyRangePacket));
              client->rx_buffer.erase(client->rx_buffer.begin(),
                  client->rx_buffer.begin() + sizeof(VSBPHeader) + sizeof(DirtyRangePacket));
              client->pending_range_offset = rpkt.offset;
              client->pending_data_size = static_cast<uint32_t>(rpkt.size);
              client->receive_state = ReceiveState::EXPECTING_BODY;
            } else if (client->receive_state == ReceiveState::EXPECTING_RANGES_SHM) {
              // DIRTY_RANGE is VSBP-framed
              if (client->rx_buffer.size() < sizeof(VSBPHeader) + sizeof(DirtyRangePacket)) break;
              VSBPHeader hdr;
              std::memcpy(&hdr, client->rx_buffer.data(), sizeof(VSBPHeader));
              if (hdr.magic != VSBP_MAGIC) { client->rx_buffer.clear(); client->receive_state = ReceiveState::IDLE; break; }
              DirtyRangePacket rpkt;
              std::memcpy(&rpkt, client->rx_buffer.data() + sizeof(VSBPHeader), sizeof(DirtyRangePacket));
              client->rx_buffer.erase(client->rx_buffer.begin(),
                  client->rx_buffer.begin() + sizeof(VSBPHeader) + sizeof(DirtyRangePacket));
              if (client->is_local && client->shmManager) {
                void* ptr = core::RuntimeEngine::instance().getMemoryManager()
                                .getPointer(reinterpret_cast<void*>(client->pending_target_ptr));
                if (ptr)
                  std::memcpy(static_cast<uint8_t*>(ptr) + rpkt.offset,
                      static_cast<uint8_t*>(client->shmManager->getBasePtr()) + client->pending_shm_offset,
                      rpkt.size);
                client->pending_shm_offset += rpkt.size;
              }
              if (--client->pending_num_ranges == 0) client->receive_state = ReceiveState::IDLE;
            } else if (client->receive_state == ReceiveState::EXPECTING_BODY) {
              // DATA_BODY is VSBP-framed; payloadSize is the actual data length
              if (client->rx_buffer.size() < sizeof(VSBPHeader)) break;
              VSBPHeader hdr;
              std::memcpy(&hdr, client->rx_buffer.data(), sizeof(VSBPHeader));
              if (hdr.magic != VSBP_MAGIC) { client->rx_buffer.clear(); client->receive_state = ReceiveState::IDLE; break; }
              if (client->rx_buffer.size() < sizeof(VSBPHeader) + hdr.payloadSize) break;
              PacketType bodyType = static_cast<PacketType>(hdr.type);
              if (bodyType == PacketType::DATA_BODY) {
                void* ptr = core::RuntimeEngine::instance().getMemoryManager()
                                .getPointer(reinterpret_cast<void*>(client->pending_target_ptr));
                if (ptr)
                  std::memcpy(static_cast<uint8_t*>(ptr) + client->pending_range_offset,
                      client->rx_buffer.data() + sizeof(VSBPHeader),
                      hdr.payloadSize);
              }
              client->rx_buffer.erase(client->rx_buffer.begin(),
                  client->rx_buffer.begin() + sizeof(VSBPHeader) + hdr.payloadSize);
              if (client->pending_num_ranges == 0) {
                client->receive_state = ReceiveState::IDLE;
              } else if (--client->pending_num_ranges > 0) {
                client->receive_state = ReceiveState::EXPECTING_RANGES_TCP;
              } else {
                client->receive_state = ReceiveState::IDLE;
              }
            } else {
              VGRE_LOG_ERROR("TCPCluster", "Master: Protocol sync error — clearing buffer for " +
                  client->ip_address);
              client->rx_buffer.clear();
              break;
            }
          }
        }
    }
  }
  VGRE_LOG_DEBUG("TCPCluster", "Master Server Loop exiting.");
}

void TCPClusterManager::clientLoop() {
  // Phase 5: Perform secure handshake if enabled
  if (security_enabled_) {
    VGREResult sr = performClientSecureHandshake();
    if (sr != VGREResult::SUCCESS) {
       VGRE_LOG_ERROR("TCPCluster", "Client: Security handshake failed");
       enabled_ = false;
       return;
    }
  }

  // 1. Send Capability Handshake (AFTER security is established)
  {
    CapabilityPacket cpkt{};
    
    // Use HybridComputeManager to get local resources if possible, or detect directly
    cpkt.cpu_cores = std::thread::hardware_concurrency();
    
#if defined(_WIN32)
    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(MEMORYSTATUSEX);
    if (GlobalMemoryStatusEx(&memInfo)) cpkt.cpu_memory = memInfo.ullTotalPhys;
#else
    std::ifstream meminfo("/proc/meminfo");
    std::string line;
    while (std::getline(meminfo, line) && cpkt.cpu_memory == 0) {
      if (line.find("MemTotal") != std::string::npos) {
        std::istringstream iss(line);
        std::string label; size_t kb; iss >> label >> kb;
        cpkt.cpu_memory = kb * 1024;
      }
    }
#endif
    
    // Authoritative hardware detection for handshake
    auto &engine = core::RuntimeEngine::instance();
    if (engine.isInitialized() && engine.getDeviceCount() > 0) {
        cpkt.has_igpu = true;
        DeviceProperties props;
        engine.getDeviceProperties(0, props);
        std::strncpy(cpkt.igpu_name, props.name, 63);
    } else {
        cpkt.has_igpu = false;
        std::strncpy(cpkt.igpu_name, "None (CPU Hybrid)", 63);
    }

    send_packet(client_fd_, PacketType::CAPABILITY, &cpkt, sizeof(CapabilityPacket), client_secure_channel_.get());
  }

  while (enabled_) {
    // 1. Send Telemetry to Master
    vgre_telemetry_t telemetry;
    {
      std::lock_guard<std::mutex> lock(client_mutex_);
      telemetry = client_telemetry_buffer_;
    }

    if (telemetry.timestamp > 0) {
      send_packet(client_fd_, PacketType::TELEMETRY, &telemetry, sizeof(vgre_telemetry_t), client_secure_channel_.get());
    }

    // Phase 12: TSS2 Priority Flush (Client side)
    {
        std::lock_guard<std::mutex> lock(client_tx_mutex_);
        while (enabled_ && !client_high_priority_tx_.empty()) {
            auto &pkt = client_high_priority_tx_.back();
            bool success = false;
            if (client_secure_channel_ && client_secure_channel_->isInitialized()) {
                success = (client_secure_channel_->sendSecure(client_fd_, pkt.data.data(), pkt.data.size()) == VGREResult::SUCCESS);
            } else {
                success = send_all(client_fd_, pkt.data.data(), pkt.data.size(), &enabled_);
            }
            if (success) {
                client_high_priority_tx_.pop_back();
            } else {
                // If send fails, back off to avoid busy-spin
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                break;
            }
        }
        while (enabled_ && !client_low_priority_tx_.empty() && client_high_priority_tx_.empty()) {
            auto &pkt = client_low_priority_tx_.front();
            bool success = false;
            if (client_secure_channel_ && client_secure_channel_->isInitialized()) {
                success = (client_secure_channel_->sendSecure(client_fd_, pkt.data.data(), pkt.data.size()) == VGREResult::SUCCESS);
            } else {
                success = send_all(client_fd_, pkt.data.data(), pkt.data.size(), &enabled_);
            }
            if (success) {
                client_low_priority_tx_.pop_front();
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                break;
            }
        }
    }

    // 2. Buffer incoming commands from Master asynchronously via poll()
    vgre_pollfd pfd;
    pfd.fd = client_fd_;
    pfd.events = POLLIN;
    pfd.revents = 0;
    int poll_res = vgre_poll(&pfd, 1, 1); // 1ms timeout for high responsiveness
    
    if (poll_res > 0 && (pfd.revents & POLLIN)) {
      int n = recv_packet(client_fd_, client_rx_buffer_, client_secure_channel_.get());
      if (n > 0) {
        std::lock_guard<std::mutex> lock(staging_mutex_);
        active_staging_->insert(active_staging_->end(), client_rx_buffer_.begin(), client_rx_buffer_.end());
        client_rx_buffer_.clear(); // Clear after moving to staging
        staging_ready_ = true;
        staging_cv_.notify_one();
      } else if (n < 0) { // Disconnected or error
        VGRE_LOG_ERROR("TCPCluster", "Client command channel disconnected");
        break; // Exit clientLoop; udpDiscoveryLoop will handle reconnect.
      }
    } else if (poll_res < 0) {
#if defined(_WIN32)
      if (WSAGetLastError() != WSAEINTR) {
#else
      if (errno != EINTR) {
#endif
        VGRE_LOG_ERROR("TCPCluster", "Client poll() failed");
        break; // Let udpDiscoveryLoop handle reconnect.
      }
    }
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
        if (!is_master_ && client_rx_buffer_.size() >= sizeof(VSBPHeader)) {
            VSBPHeader h;
            std::memcpy(&h, client_rx_buffer_.data(), sizeof(VSBPHeader));
            std::cout << "[WORKER TCP TRACE] State=" << (int)receive_state_ << " ValidSize=" << client_rx_buffer_.size() << " PktType=" << h.type << " PayloadSize=" << h.payloadSize << std::endl;
        }

        if (receive_state_ == ReceiveState::IDLE) {
            if (client_rx_buffer_.size() < sizeof(VSBPHeader)) break;
            
            VSBPHeader header;
            std::memcpy(&header, client_rx_buffer_.data(), sizeof(VSBPHeader));
            
            // VSBP Validation
            if (header.magic != VSBP_MAGIC || header.version != VSBP_VERSION) {
                VGRE_LOG_ERROR("TCPCluster", "VSBP Protocol Violation: Invalid Magic/Version. Clearing buffer.");
                client_rx_buffer_.clear();
                break;
            }

            if (client_rx_buffer_.size() < sizeof(VSBPHeader) + header.payloadSize) break;
            
            PacketType type = static_cast<PacketType>(header.type);

            if (type == PacketType::ARG_SCALAR || type == PacketType::ARG_POINTER) {
                ArgScalarPacket apkt;
                std::memcpy(&apkt, client_rx_buffer_.data() + sizeof(VSBPHeader), sizeof(ArgScalarPacket));
                client_rx_buffer_.erase(client_rx_buffer_.begin(), client_rx_buffer_.begin() + sizeof(VSBPHeader) + sizeof(ArgScalarPacket));
                PendingArg arg;
                arg.type = apkt.arg_type;
                arg.value = apkt.value;
                pending_args_[apkt.arg_index] = std::move(arg);
            } else if (type == PacketType::DATA_HEADER) {
                DataHeaderPacket dpkt;
                std::memcpy(&dpkt, client_rx_buffer_.data() + sizeof(VSBPHeader), sizeof(DataHeaderPacket));
                client_rx_buffer_.erase(client_rx_buffer_.begin(), client_rx_buffer_.begin() + sizeof(VSBPHeader) + sizeof(DataHeaderPacket));
                pending_target_ptr_ = dpkt.target_ptr;
                pending_data_size_ = static_cast<uint32_t>(dpkt.size);
                pending_num_ranges_ = 0;
                receive_state_ = ReceiveState::EXPECTING_BODY;
            } else if (type == PacketType::DATA_HEADER_DIRTY) {
                DataHeaderDirtyPacket dhpkt;
                std::memcpy(&dhpkt, client_rx_buffer_.data() + sizeof(VSBPHeader), sizeof(DataHeaderDirtyPacket));
                client_rx_buffer_.erase(client_rx_buffer_.begin(), client_rx_buffer_.begin() + sizeof(VSBPHeader) + sizeof(DataHeaderDirtyPacket));
                pending_target_ptr_ = dhpkt.target_ptr;
                pending_num_ranges_ = dhpkt.num_ranges;
                receive_state_ = ReceiveState::EXPECTING_RANGES_TCP;
            } else if (type == PacketType::DATA_SHM_DIRTY) {
                DataShmDirtyPacket dspkt;
                std::memcpy(&dspkt, client_rx_buffer_.data() + sizeof(VSBPHeader), sizeof(DataShmDirtyPacket));
                client_rx_buffer_.erase(client_rx_buffer_.begin(), client_rx_buffer_.begin() + sizeof(VSBPHeader) + sizeof(DataShmDirtyPacket));
                pending_target_ptr_ = dspkt.target_ptr;
                pending_num_ranges_ = dspkt.num_ranges;
                pending_shm_offset_ = dspkt.shm_offset;
                receive_state_ = ReceiveState::EXPECTING_RANGES_SHM;
            } else if (type == PacketType::DATA_SHM) {
                DataShmPacket dspkt;
                std::memcpy(&dspkt, client_rx_buffer_.data() + sizeof(VSBPHeader), sizeof(DataShmPacket));
                client_rx_buffer_.erase(client_rx_buffer_.begin(), client_rx_buffer_.begin() + sizeof(VSBPHeader) + sizeof(DataShmPacket));
                if (client_shm_enabled_ && client_shm_manager_) {
                    void* handle = reinterpret_cast<void*>(dspkt.target_ptr);
                    auto& mm = core::RuntimeEngine::instance().getMemoryManager();
                    if (!mm.isValidHandle(handle)) {
                        void* actual_ptr = nullptr;
                        mm.allocateManagedAt(handle, dspkt.size, actual_ptr);
                    }
                    void* local_ptr = mm.getPointer(handle);
                    if (local_ptr) {
                        std::memcpy(local_ptr, static_cast<uint8_t*>(client_shm_manager_->getBasePtr()) + dspkt.shm_offset, dspkt.size);
                    }
                }
            } else if (type == PacketType::STRUCT_DATA) {
                StructDataPacket spkt;
                std::memcpy(&spkt, client_rx_buffer_.data() + sizeof(VSBPHeader), sizeof(StructDataPacket));
                client_rx_buffer_.erase(client_rx_buffer_.begin(), client_rx_buffer_.begin() + sizeof(VSBPHeader) + sizeof(StructDataPacket));
                
                PendingArg arg;
                arg.type = (uint8_t)ArgType::STRUCT;
                arg.data.assign(client_rx_buffer_.begin(), client_rx_buffer_.begin() + spkt.size);
                pending_args_[spkt.arg_index] = std::move(arg);
                client_rx_buffer_.erase(client_rx_buffer_.begin(), client_rx_buffer_.begin() + spkt.size);
            } else if (type == PacketType::REGISTER_KERNEL) {
                KernelRegisterPacket kpkt;
                std::memcpy(&kpkt, client_rx_buffer_.data() + sizeof(VSBPHeader), sizeof(KernelRegisterPacket));
                client_rx_buffer_.erase(client_rx_buffer_.begin(), client_rx_buffer_.begin() + sizeof(VSBPHeader) + sizeof(KernelRegisterPacket));
                
                if (auth_token_ != 0 && kpkt.auth_token != auth_token_) {
                    VGRE_LOG_ERROR("TCPCluster", "Auth Token Mismatch during kernel registration");
                    client_rx_buffer_.erase(client_rx_buffer_.begin(), client_rx_buffer_.begin() + kpkt.source_len);
                    break;
                }
                
                pending_kernel_id_ = kpkt.kernel_id;
                pending_kernel_name_ = kpkt.name;
                pending_kernel_source_len_ = kpkt.source_len;
                receive_state_ = ReceiveState::EXPECTING_KERNEL_SOURCE; 
            } else if (type == PacketType::LAUNCH_KERNEL) {
                RemoteCommandPacket pkt;
                std::memcpy(&pkt, client_rx_buffer_.data() + sizeof(VSBPHeader), sizeof(RemoteCommandPacket));
                client_rx_buffer_.erase(client_rx_buffer_.begin(), client_rx_buffer_.begin() + sizeof(VSBPHeader) + sizeof(RemoteCommandPacket));
                handleRemoteCommand(pkt);
            } else if (type == PacketType::PARTITION_DISPATCH) {
                PartitionDispatchPacket pkt;
                std::memcpy(&pkt, client_rx_buffer_.data() + sizeof(VSBPHeader), sizeof(PartitionDispatchPacket));
                client_rx_buffer_.erase(client_rx_buffer_.begin(), client_rx_buffer_.begin() + sizeof(VSBPHeader) + sizeof(PartitionDispatchPacket));
                handlePartitionDispatch(pkt);
            } else if (type == PacketType::ROTATE_KEY) {
                SecureHandshakePacket rpkt;
                std::memcpy(&rpkt, client_rx_buffer_.data() + sizeof(VSBPHeader), sizeof(SecureHandshakePacket));
                client_rx_buffer_.erase(client_rx_buffer_.begin(), client_rx_buffer_.begin() + sizeof(VSBPHeader) + sizeof(SecureHandshakePacket));
                if (client_secure_channel_) client_secure_channel_->rotateKey(rpkt.nonce);
            } else if (type == PacketType::SHM_INIT) {
                ShmInitPacket sipkt;
                std::memcpy(&sipkt, client_rx_buffer_.data() + sizeof(VSBPHeader), sizeof(ShmInitPacket));
                client_rx_buffer_.erase(client_rx_buffer_.begin(), client_rx_buffer_.begin() + sizeof(VSBPHeader) + sizeof(ShmInitPacket));
                client_shm_manager_ = std::make_unique<core::ShmManager>();
                if (client_shm_manager_->open(sipkt.shm_name, sipkt.shm_size, false) == VGREResult::SUCCESS) client_shm_enabled_ = true;
            } else if (type == PacketType::RAW_DATA || type == PacketType::DATA_BODY) {
                client_rx_buffer_.erase(client_rx_buffer_.begin(), client_rx_buffer_.begin() + sizeof(VSBPHeader) + header.payloadSize);
            } else if (type == PacketType::COOP_BARRIER_SYNC) {
                client_rx_buffer_.erase(client_rx_buffer_.begin(), client_rx_buffer_.begin() + sizeof(VSBPHeader));
                if (is_master_) {
                    std::lock_guard<std::mutex> lock_barrier(barrier_mutex_);
                    barrier_count_++;
                    size_t activeCount = 0;
                    {
                        std::lock_guard<std::recursive_mutex> lock_clients(clients_mutex_);
                        for (auto& c : clients_) if (c && c->active) activeCount++;
                    }
                    VGRE_LOG_INFO("TCPCluster", "Cooperative Barrier: Worker arrived (" + std::to_string(barrier_count_) + "/" + std::to_string(activeCount) + ")");
                    if (barrier_count_ >= activeCount && activeCount > 0) {
                        VGRE_LOG_INFO("TCPCluster", "Cooperative Barrier: All workers synced. Broadcasting RESUME.");
                        barrier_count_ = 0;
                        broadcastPacket(PacketType::COOP_BARRIER_RESUME, nullptr, 0); 
                    }
                }
            } else if (type == PacketType::COOP_BARRIER_RESUME) {
                client_rx_buffer_.erase(client_rx_buffer_.begin(), client_rx_buffer_.begin() + sizeof(VSBPHeader));
                VGRE_LOG_INFO("TCPCluster", "Cooperative Barrier: RESUME received. Resuming local workers.");
                barrier_cv_.notify_all();
            } else {
                VGRE_LOG_WARN("TCPCluster", "Unknown Packet Type: " + std::to_string(header.type));
                client_rx_buffer_.clear(); 
                break;
            }
        } else if (receive_state_ == ReceiveState::EXPECTING_KERNEL_SOURCE) {
            VSBPHeader header;
            if (client_rx_buffer_.size() < sizeof(VSBPHeader)) break;
            std::memcpy(&header, client_rx_buffer_.data(), sizeof(VSBPHeader));
            if (client_rx_buffer_.size() < sizeof(VSBPHeader) + header.payloadSize) break;
            PacketType type = static_cast<PacketType>(header.type);
            if (type == PacketType::RAW_DATA) {
                std::string sourceStr(reinterpret_cast<const char*>(client_rx_buffer_.data() + sizeof(VSBPHeader)), header.payloadSize);
                core::RuntimeEngine::instance().registerKernel(pending_kernel_name_, sourceStr, pending_kernel_id_);
                VGRE_LOG_INFO("TCPCluster", "Worker Registered Remote Kernel '" + pending_kernel_name_ + "' (ID: " + std::to_string(pending_kernel_id_) + ")");
                client_rx_buffer_.erase(client_rx_buffer_.begin(), client_rx_buffer_.begin() + sizeof(VSBPHeader) + header.payloadSize);
                receive_state_ = ReceiveState::IDLE;
            } else {
                VGRE_LOG_ERROR("TCPCluster", "Expected RAW_DATA for kernel source, got type " + std::to_string(header.type));
                receive_state_ = ReceiveState::IDLE;
                client_rx_buffer_.erase(client_rx_buffer_.begin(), client_rx_buffer_.begin() + sizeof(VSBPHeader) + header.payloadSize);
            }
        } else if (receive_state_ == ReceiveState::EXPECTING_RANGES_TCP) {
            if (client_rx_buffer_.size() < sizeof(DirtyRangePacket)) break;
            DirtyRangePacket rpkt;
            std::memcpy(&rpkt, client_rx_buffer_.data(), sizeof(DirtyRangePacket));
            client_rx_buffer_.erase(client_rx_buffer_.begin(), client_rx_buffer_.begin() + sizeof(DirtyRangePacket));
            pending_range_offset_ = rpkt.offset;
            pending_data_size_ = static_cast<uint32_t>(rpkt.size);
            receive_state_ = ReceiveState::EXPECTING_BODY;
        } else if (receive_state_ == ReceiveState::EXPECTING_RANGES_SHM) {
            if (client_rx_buffer_.size() < sizeof(DirtyRangePacket)) break;
            DirtyRangePacket rpkt;
            std::memcpy(&rpkt, client_rx_buffer_.data(), sizeof(DirtyRangePacket));
            client_rx_buffer_.erase(client_rx_buffer_.begin(), client_rx_buffer_.begin() + sizeof(DirtyRangePacket));
            if (client_shm_enabled_ && client_shm_manager_) {
                void* local_ptr = core::RuntimeEngine::instance().getMemoryManager().getPointer(reinterpret_cast<void*>(pending_target_ptr_));
                if (local_ptr) std::memcpy(static_cast<uint8_t*>(local_ptr) + rpkt.offset, static_cast<uint8_t*>(client_shm_manager_->getBasePtr()) + pending_shm_offset_, rpkt.size);
                pending_shm_offset_ += rpkt.size;
            }
            if (--pending_num_ranges_ == 0) receive_state_ = ReceiveState::IDLE;
        } else if (receive_state_ == ReceiveState::EXPECTING_BODY) {
            if (client_rx_buffer_.size() < pending_data_size_) break;
            void* handle = reinterpret_cast<void*>(pending_target_ptr_);
            auto& mm = core::RuntimeEngine::instance().getMemoryManager();
            if (!mm.isValidHandle(handle)) { void* act; mm.allocateManagedAt(handle, pending_data_size_, act); }
            void* local_ptr = mm.getPointer(handle);
            if (local_ptr) std::memcpy(static_cast<uint8_t*>(local_ptr) + pending_range_offset_, client_rx_buffer_.data(), pending_data_size_);
            client_rx_buffer_.erase(client_rx_buffer_.begin(), client_rx_buffer_.begin() + pending_data_size_);
            if (--pending_num_ranges_ == 0 || pending_num_ranges_ == (uint32_t)-1) receive_state_ = ReceiveState::IDLE;
            else receive_state_ = ReceiveState::EXPECTING_RANGES_TCP;
        }
    }
  }
}

void TCPClusterManager::udpAnnouncerLoop() {
  vgre_socket_t udp_fd = socket(AF_INET, SOCK_DGRAM, 0);
  if (udp_fd == VGRE_INVALID_SOCKET) return;
  
  int opt = 1;
#if defined(_WIN32)
  vgre_setsockopt(udp_fd, SOL_SOCKET, SO_BROADCAST, (const char*)&opt, sizeof(opt));
#else
  vgre_setsockopt(udp_fd, SOL_SOCKET, SO_BROADCAST, &opt, sizeof(opt));
#endif

  struct sockaddr_in broadcast_addr{};
  broadcast_addr.sin_family = AF_INET;
  broadcast_addr.sin_addr.s_addr = INADDR_BROADCAST;
  broadcast_addr.sin_port = htons(7778);

  std::string security_str = (security_enabled_ ? ":SECURE" : ":PLAIN");
  std::string ping_msg = "VGRE_DISCOVERY_PING:" + std::to_string(port_) + security_str;

  VGRE_LOG_INFO("TCPCluster", "Master: UDP Announcer active (broadcasting master presence)...");

  while (enabled_ && is_master_) {
    sendto(udp_fd, ping_msg.c_str(), ping_msg.length(), 0, (struct sockaddr*)&broadcast_addr, sizeof(broadcast_addr));
    std::this_thread::sleep_for(std::chrono::seconds(2));
  }
  vgre_close_socket(udp_fd);
}

void TCPClusterManager::udpMasterDiscoveryLoop() {
  vgre_socket_t udp_fd = socket(AF_INET, SOCK_DGRAM, 0);
  if (udp_fd == VGRE_INVALID_SOCKET) return;

  int opt = 1;
#if defined(_WIN32)
  vgre_setsockopt(udp_fd, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt, sizeof(opt));
#else
  vgre_setsockopt(udp_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#endif

  struct sockaddr_in listen_addr{};
  listen_addr.sin_family = AF_INET;
  listen_addr.sin_addr.s_addr = INADDR_ANY;
  listen_addr.sin_port = htons(7779); // Dedicated port for worker announcements

  if (bind(udp_fd, (struct sockaddr*)&listen_addr, sizeof(listen_addr)) < 0) {
    VGRE_LOG_WARN("TCPCluster", "Master: UDP Worker Discovery bind failed");
    vgre_close_socket(udp_fd);
    return;
  }

  // 1-second receive timeout so the while-loop condition is re-evaluated
  // regularly and the thread can exit promptly when shutdown() sets enabled_=false.
  // Without this, recvfrom blocks indefinitely, preventing shutdown from joining.
#if defined(_WIN32)
  DWORD rcvTimeout = 1000;
  vgre_setsockopt(udp_fd, SOL_SOCKET, SO_RCVTIMEO, (const char*)&rcvTimeout, sizeof(rcvTimeout));
#else
  struct timeval rcvTv{1, 0};
  vgre_setsockopt(udp_fd, SOL_SOCKET, SO_RCVTIMEO, &rcvTv, sizeof(rcvTv));
#endif

  VGRE_LOG_INFO("TCPCluster", "Master: Active Worker Discovery enabled (scanning for remote nodes)...");

  char buffer[128];
  struct sockaddr_in sender_addr{};
  socklen_t sender_len = sizeof(sender_addr);

  while (enabled_ && is_master_) {
    int n = recvfrom(udp_fd, buffer, sizeof(buffer) - 1, 0, (struct sockaddr*)&sender_addr, &sender_len);
    if (n > 0) {
      buffer[n] = '\0';
      std::string msg(buffer);
      if (msg.find("VGRE_WORKER_PING") == 0) {
        char ip[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &(sender_addr.sin_addr), ip, INET_ADDRSTRLEN);
        
        int worker_port = 7777;
        size_t colon = msg.find(':');
        if (colon != std::string::npos) {
            worker_port = std::stoi(msg.substr(colon + 1));
        }

        std::string worker_addr = std::string(ip) + ":" + std::to_string(worker_port);
        
        // Add to proactive connection list if not already there
        bool exists = false;
        {
          std::lock_guard<std::recursive_mutex> lock(clients_mutex_);
          for(const auto& s : proactive_worker_addresses_) {
            if (s == worker_addr) { exists = true; break; }
          }
          if (!exists) {
            VGRE_LOG_INFO("TCPCluster", "Master: Automatically discovered worker at " + worker_addr);
            proactive_worker_addresses_.push_back(worker_addr);
          }
        }
      }
    }
  }
  vgre_close_socket(udp_fd);
}

void TCPClusterManager::udpDiscoveryLoop() {
  // Start the staging data-processor once — it lives as long as the worker is
  // enabled and is reused across reconnect cycles.
  if (!data_processor_thread_.joinable()) {
      data_processor_thread_ = std::thread(
          &TCPClusterManager::processClientStagingBuffer, this);
  }

  // ── Outer reconnect loop ─────────────────────────────────────────────────
  // After a master disconnect clientLoop() returns with enabled_ still true.
  // We reset connection state and re-run discovery so the worker auto-rejoins.
  while (enabled_) {
    // Open a fresh UDP socket for this discovery attempt.
    vgre_socket_t udp_fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (udp_fd == VGRE_INVALID_SOCKET) {
        std::this_thread::sleep_for(std::chrono::seconds(2));
        continue;
    }

    {
      int opt = 1;
      vgre_setsockopt(udp_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#ifndef _WIN32
#ifdef SO_REUSEPORT
      vgre_setsockopt(udp_fd, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(opt));
#endif
#endif
    }

    struct sockaddr_in listen_addr{};
    listen_addr.sin_family = AF_INET;
    listen_addr.sin_addr.s_addr = INADDR_ANY;
    listen_addr.sin_port = htons(7778);

    if (bind(udp_fd, (struct sockaddr*)&listen_addr, sizeof(listen_addr)) < 0) {
      VGRE_LOG_WARN("TCPCluster", "UDP discovery bind failed, retrying in 3s...");
      vgre_close_socket(udp_fd);
      std::this_thread::sleep_for(std::chrono::seconds(3));
      continue;
    }

    // 1-second receive timeout so we can check enabled_ regularly.
    vgre::common::vgre_set_recv_timeout(udp_fd, 1000);

    VGRE_LOG_INFO("TCPCluster", "Scanning local subnet for Master node broadcasts...");

    // ── Inner discovery loop: wait for master UDP broadcast ─────────────────
    char buffer[64];
    struct sockaddr_in sender_addr{};
    socklen_t sender_len = sizeof(sender_addr);

    while (enabled_ && client_fd_ == VGRE_INVALID_SOCKET) {
      int n = recvfrom(udp_fd, buffer, sizeof(buffer) - 1, 0,
                       (struct sockaddr*)&sender_addr, &sender_len);
      if (n > 0) {
        buffer[n] = '\0';
        std::string msg(buffer);
        if (msg.find("VGRE_DISCOVERY_PING") == 0) {
          char ip[INET_ADDRSTRLEN];
          inet_ntop(AF_INET, &(sender_addr.sin_addr), ip, INET_ADDRSTRLEN);
          host_ = ip;

          // Parse: VGRE_DISCOVERY_PING:PORT[:SECURE|PLAIN]
          int connect_port = port_;
          size_t first_colon = msg.find(':');
          if (first_colon != std::string::npos) {
              size_t second_colon = msg.find(':', first_colon + 1);
              if (second_colon != std::string::npos) {
                  try { connect_port = std::stoi(
                      msg.substr(first_colon + 1, second_colon - first_colon - 1)); }
                  catch (...) {}
                  std::string sec_status = msg.substr(second_colon + 1);
                  if (sec_status == "SECURE" && !security_enabled_) {
                      VGRE_LOG_INFO("TCPCluster",
                          "Worker: Master requires security, auto-enabling encryption...");
                      security_enabled_ = true;
                  }
              } else {
                  try { connect_port = std::stoi(msg.substr(first_colon + 1)); }
                  catch (...) {}
              }
          }

          VGRE_LOG_INFO("TCPCluster",
              "Discovered Master node at " + host_ + ":" +
              std::to_string(connect_port));

          vgre_socket_t sock = socket(AF_INET, SOCK_STREAM, 0);
          if (sock == VGRE_INVALID_SOCKET) continue;

          struct sockaddr_in serv_addr{};
          serv_addr.sin_family = AF_INET;
          serv_addr.sin_port = htons(connect_port);
          inet_pton(AF_INET, host_.c_str(), &serv_addr.sin_addr);

          if (connect(sock, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) >= 0) {
              // Keepalive so we detect a hard master crash quickly.
              vgre::common::vgre_set_tcp_keepalive(sock, 5, 2, 3);
              vgre_ioctl_nonblock(sock);
              client_fd_ = sock;
              VGRE_LOG_INFO("TCPCluster",
                  "Connected to Remote Master Node at " + host_);
              break; // exit inner discovery loop
          } else {
              vgre_close_socket(sock);
          }
        }
      }
    } // inner discovery loop
 
    vgre_close_socket(udp_fd);
 
    if (!enabled_) break;
 
    if (client_fd_ != VGRE_INVALID_SOCKET) {
        // Run the client protocol (capability exchange, telemetry, commands).
        // clientLoop() breaks (not enabled_=false) on disconnect so we can retry.
        clientLoop();
 
        // ── Post-disconnect cleanup ──────────────────────────────────────────
        // clientLoop exited — either a disconnect (break) or shutdown (enabled_=false).
        if (client_fd_ != VGRE_INVALID_SOCKET) {
            vgre_close_socket(client_fd_);
            client_fd_ = VGRE_INVALID_SOCKET;
        }
        // Reset security for the next session.
        client_secure_channel_.reset();
        client_security_established_ = false;
        is_authenticating_ = false;
        // Drain/reset receive buffers.
        client_rx_buffer_.clear();
        {
            std::lock_guard<std::mutex> lk(staging_mutex_);
            client_rx_staging_A_.clear();
            client_rx_staging_B_.clear();
            active_staging_   = &client_rx_staging_A_;
            processing_staging_ = &client_rx_staging_B_;
            staging_ready_ = false;
        }

        if (!enabled_) break; // proper shutdown — do not retry

        VGRE_LOG_INFO("TCPCluster",
            "Worker: Disconnected from master, retrying discovery in 3s...");
        std::this_thread::sleep_for(std::chrono::seconds(3));
        // outer loop continues → fresh UDP socket, fresh discovery
    }
  } // outer reconnect loop
}

void TCPClusterManager::udpWorkerAnnouncerLoop() {
  vgre_socket_t udp_fd = socket(AF_INET, SOCK_DGRAM, 0);
  if (udp_fd == VGRE_INVALID_SOCKET) return;

  int opt = 1;
#if defined(_WIN32)
  vgre_setsockopt(udp_fd, SOL_SOCKET, SO_BROADCAST, (const char*)&opt, sizeof(opt));
#else
  vgre_setsockopt(udp_fd, SOL_SOCKET, SO_BROADCAST, &opt, sizeof(opt));
#endif

  struct sockaddr_in broadcast_addr{};
  broadcast_addr.sin_family = AF_INET;
  broadcast_addr.sin_addr.s_addr = INADDR_BROADCAST;
  broadcast_addr.sin_port = htons(7779); // Master scans this port

  std::string ping_msg = "VGRE_WORKER_PING:" + std::to_string(port_);
  
  VGRE_LOG_INFO("TCPCluster", "Worker: Proactive Announcer active (seeking master)...");

  while (enabled_ && !is_master_) {
    sendto(udp_fd, ping_msg.c_str(), ping_msg.length(), 0,
           (struct sockaddr*)&broadcast_addr, sizeof(broadcast_addr));
    // Sleep in 200ms increments so shutdown() can join within 200ms.
    for (int i = 0; i < 25 && enabled_ && !is_master_; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
  }
  vgre_close_socket(udp_fd);
}


VGREResult TCPClusterManager::launchRemoteKernel(
    int worker_idx, uint64_t kernel_id, const uint32_t grid_dim[3],
    const uint32_t block_dim[3], void **args, int num_args, size_t shared_mem) {
  if (!is_master_)
    return VGREResult::ERR_INVALID_VALUE;
  if (!grid_dim || !block_dim || num_args < 0)
    return VGREResult::ERR_INVALID_VALUE;
  if (num_args > 0 && !args)
    return VGREResult::ERR_INVALID_VALUE;
  if (grid_dim[0] == 0 || block_dim[0] == 0)
    return VGREResult::ERR_INVALID_VALUE;

  std::lock_guard<std::recursive_mutex> lock(clients_mutex_);
  if (worker_idx < 0 || worker_idx >= static_cast<int>(clients_.size()) ||
      !clients_[worker_idx] || !clients_[worker_idx]->active) {
    return VGREResult::ERR_INVALID_VALUE;
  }

  std::vector<ArgType> argTypes;
  if (core::RuntimeEngine::instance().getKernelArgTypes(kernel_id, argTypes) !=
      VGREResult::SUCCESS) {
    return VGREResult::ERR_INVALID_KERNEL;
  }

  if (auth_token_ == 0) {
    VGRE_LOG_ERROR("TCPCluster",
                   "Remote kernel dispatch blocked: missing "
                   "VGRE_TCP_AUTH_TOKEN");
    return VGREResult::ERR_INVALID_VALUE;
  }

  // Phase 12: TSS2 Congestion Window
  const uint32_t MAX_IN_FLIGHT = 16;
  if (clients_[worker_idx]->in_flight_kernels >= MAX_IN_FLIGHT) {
      VGRE_LOG_WARN("TCPCluster", "Congestion: Max in-flight kernels reached for " + clients_[worker_idx]->ip_address);
      return VGREResult::ERR_BUSY;
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
          spkt.arg_index = i;
          spkt.size = static_cast<uint32_t>(size);
          send_packet(clients_[worker_idx]->socket_fd, PacketType::STRUCT_DATA, &spkt, sizeof(StructDataPacket), clients_[worker_idx]->secureChannel.get());

          send_packet(clients_[worker_idx]->socket_fd, PacketType::DATA_BODY, args[i], size, clients_[worker_idx]->secureChannel.get());
      }
    } else if (type == ArgType::POINTER) {
      void* ptr = *static_cast<void**>(args[i]);
      uint64_t handle = reinterpret_cast<uint64_t>(ptr);
      
      // Memory Coherence: If it's a valid managed pointer, sync its data to the worker
      size_t size = core::RuntimeEngine::instance().getMemoryManager().getAllocationSize(ptr);
      if (size > 0 && ptr) {
          auto& mm = core::RuntimeEngine::instance().getMemoryManager();
          std::vector<std::pair<size_t, size_t>> dirtyRanges;
          bool useDelta = clients_[worker_idx]->synced_handles.count(ptr) > 0;
          
          if (useDelta) {
              mm.getDirtyPages(ptr, dirtyRanges);
          }

          if (useDelta && !dirtyRanges.empty()) {
              // Delta-Sync
              if (clients_[worker_idx]->is_local && clients_[worker_idx]->shmManager) {
                  uint32_t numRanges = static_cast<uint32_t>(dirtyRanges.size());
                  DataShmDirtyPacket dspkt{};
                  dspkt.target_ptr = handle;
                  dspkt.num_ranges = numRanges;
                  
                  // Copy each range to SHM
                  uint64_t totalSize = 0;
                  for (auto& range : dirtyRanges) totalSize += range.second;
                  
                  uint64_t baseOffset = clients_[worker_idx]->shm_offset;
                  clients_[worker_idx]->shm_offset += totalSize;
                  if (baseOffset + totalSize <= clients_[worker_idx]->shmManager->getSize()) {
                      dspkt.shm_offset = baseOffset;
                      send_packet(clients_[worker_idx]->socket_fd, PacketType::DATA_SHM_DIRTY, &dspkt, sizeof(DataShmDirtyPacket), clients_[worker_idx]->secureChannel.get());
                      
                      uint64_t currentOffset = baseOffset;
                      for (auto& range : dirtyRanges) {
                          DirtyRangePacket rpkt{range.first, range.second};
                          send_packet(clients_[worker_idx]->socket_fd, PacketType::DIRTY_RANGE, &rpkt, sizeof(DirtyRangePacket), clients_[worker_idx]->secureChannel.get());
                          std::memcpy(static_cast<uint8_t*>(clients_[worker_idx]->shmManager->getBasePtr()) + currentOffset, 
                                      static_cast<uint8_t*>(ptr) + range.first, range.second);
                          currentOffset += range.second;
                      }
                  } else {
                      goto full_sync; // Fallback
                  }
              } else {
                  DataHeaderDirtyPacket dhpkt{};
                  dhpkt.target_ptr = handle;
                  dhpkt.num_ranges = static_cast<uint32_t>(dirtyRanges.size());
                  send_packet(clients_[worker_idx]->socket_fd, PacketType::DATA_HEADER_DIRTY, &dhpkt, sizeof(DataHeaderDirtyPacket), clients_[worker_idx]->secureChannel.get());
                  
                  for (auto& range : dirtyRanges) {
                      DirtyRangePacket rpkt{range.first, range.second};
                      send_packet(clients_[worker_idx]->socket_fd, PacketType::DIRTY_RANGE, &rpkt, sizeof(DirtyRangePacket), clients_[worker_idx]->secureChannel.get());
                      
                      send_packet(clients_[worker_idx]->socket_fd, PacketType::DATA_BODY, static_cast<uint8_t*>(ptr) + range.first, range.second, clients_[worker_idx]->secureChannel.get());
                  }
              }
              mm.clearDirtyPages(ptr);
          } else {
full_sync:
              clients_[worker_idx]->synced_handles.insert(ptr);
              if (clients_[worker_idx]->is_local && clients_[worker_idx]->shmManager) {
                  // SHM Optimization
                  uint64_t offset = clients_[worker_idx]->shm_offset;
                  clients_[worker_idx]->shm_offset += size;
                  if (offset + size <= clients_[worker_idx]->shmManager->getSize()) {
                      std::memcpy(static_cast<uint8_t*>(clients_[worker_idx]->shmManager->getBasePtr()) + offset, ptr, size);
                      DataShmPacket dspkt{};
                      dspkt.target_ptr = handle;
                      dspkt.shm_offset = offset;
                      dspkt.size = size;
                      send_packet(clients_[worker_idx]->socket_fd, PacketType::DATA_SHM, &dspkt, sizeof(DataShmPacket), clients_[worker_idx]->secureChannel.get());
                  } else {
                      DataHeaderPacket dpkt{};
                      dpkt.target_ptr = handle;
                      dpkt.size = size;
                      send_packet(clients_[worker_idx]->socket_fd, PacketType::DATA_HEADER, &dpkt, sizeof(DataHeaderPacket), clients_[worker_idx]->secureChannel.get());

                      send_packet(clients_[worker_idx]->socket_fd, PacketType::DATA_BODY, ptr, size, clients_[worker_idx]->secureChannel.get());
                  }
              } else {
                  DataHeaderPacket dpkt{};
                  dpkt.target_ptr = handle;
                  dpkt.size = size;
                  send_packet(clients_[worker_idx]->socket_fd, PacketType::DATA_HEADER, &dpkt, sizeof(DataHeaderPacket), clients_[worker_idx]->secureChannel.get());

                  send_packet(clients_[worker_idx]->socket_fd, PacketType::DATA_BODY, ptr, size, clients_[worker_idx]->secureChannel.get());
              }
              mm.clearDirtyPages(ptr);
          }
      }

      ArgScalarPacket apkt{};
      apkt.arg_index = i;
      apkt.arg_type = static_cast<uint8_t>(type);
      apkt.value = handle;
      send_packet(clients_[worker_idx]->socket_fd, PacketType::ARG_POINTER, &apkt, sizeof(ArgScalarPacket), clients_[worker_idx]->secureChannel.get());
    } else {
      // Scalar arguments (INT32, FLOAT32, etc.)
      ArgScalarPacket apkt{};
      apkt.arg_index = i;
      apkt.arg_type = static_cast<uint8_t>(type);
      
      size_t arg_size = 8;
      switch (type) {
        case ArgType::INT32:
        case ArgType::UINT32:
        case ArgType::FLOAT32:
          arg_size = 4;
          break;
        default:
          arg_size = 8;
          break;
      }
      
      std::memset(&apkt.value, 0, 8);
      std::memcpy(&apkt.value, args[i], arg_size);
      send_packet(clients_[worker_idx]->socket_fd, PacketType::ARG_SCALAR, &apkt, sizeof(ArgScalarPacket), clients_[worker_idx]->secureChannel.get());
    }
  }

  // 2. Send LAUNCH_KERNEL header LAST (this triggers execution on worker side)
  RemoteCommandPacket pkt{};
  pkt.auth_token = auth_token_;
  pkt.kernel_id = kernel_id;
  std::memcpy(pkt.grid_dim, grid_dim, sizeof(pkt.grid_dim));
  std::memcpy(pkt.block_dim, block_dim, sizeof(pkt.block_dim));
  pkt.shared_mem = shared_mem;
  pkt.num_args = num_args; 

  if (!send_packet(clients_[worker_idx]->socket_fd, PacketType::LAUNCH_KERNEL, &pkt, sizeof(RemoteCommandPacket), clients_[worker_idx]->secureChannel.get())) {
    return VGREResult::ERR_IO;
  }
  clients_[worker_idx]->in_flight_kernels++;

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

  std::lock_guard<std::recursive_mutex> lock(clients_mutex_);
  KernelRegisterPacket kpkt{};
  kpkt.auth_token = auth_token_;
  kpkt.kernel_id = kernel_id;
  std::strncpy(kpkt.name, name.c_str(), sizeof(kpkt.name) - 1);
  kpkt.source_len = static_cast<uint32_t>(source.length());

  for (auto &client : clients_) {
    if (client && client->active) {
      send_packet(client->socket_fd, PacketType::REGISTER_KERNEL, &kpkt, sizeof(KernelRegisterPacket), client->secureChannel.get());
      send_packet(client->socket_fd, PacketType::RAW_DATA, source.c_str(), source.length(), client->secureChannel.get());
    }
  }
}

int TCPClusterManager::getFirstActiveWorker() const {
  if (!is_master_) {
    return -1;
  }
  std::lock_guard<std::recursive_mutex> lock(clients_mutex_);
  for (size_t i = 0; i < clients_.size(); ++i) {
    if (clients_[i] && clients_[i]->active) {
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
  if (pkt.grid_dim[0] == 0 || pkt.block_dim[0] == 0) {
    VGRE_LOG_ERROR("TCPCluster", "Invalid Remote Command Packet Received");
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

  // Authoritative Phase 3: Wait for local execution to complete before syncing back memory.
  // This prevents the master from receiving stale data or a 'done' RESPONSE before the work is done.
  vgre::core::RuntimeEngine::instance().streamSynchronize(0);

  // --- Memory Coherence (Pull Back) ---
  // If the kernel was successful, sync modified managed pointers back to master.
  // In a real GPU driver, we'd track dirty pages, but for VGRE Phase 3 coherence,
  // we assume all pointer arguments are potentially outputs.
  for (int i = 0; i < numArgs; ++i) {
      auto it = pending_args_.find(i);
      if (it == pending_args_.end()) continue;
      
      PendingArg& arg = it->second;
      if (arg.type == (uint8_t)ArgType::POINTER) {
          void* ptr = reinterpret_cast<void*>(arg.value);
          size_t size = vgre::core::RuntimeEngine::instance().getMemoryManager().getAllocationSize(ptr);
          if (size > 0) {
              if (client_shm_enabled_ && client_shm_manager_) {
                  // SHM Optimization for worker -> master (Pull Back)
                  // For simplicity, workers use the same SHM segment provided by master.
                  // We need to coordinate offsets. Currently, we just append.
                  // Ideally we'd reset the offset after each launch, but the master doesn't know when worker is done.
                  // Let's use a fixed offset area for results OR just use TCP for results if SHM is one-way.
                  // Actually, let's enable bidirectional SHM.
                  
                  // Simple: use offset 128MB+ for results
                  static std::atomic<uint64_t> resultOffset{128 * 1024 * 1024};
                  uint64_t offset = resultOffset.fetch_add(size);
                  if (offset + size <= client_shm_manager_->getSize()) {
                      std::memcpy(static_cast<uint8_t*>(client_shm_manager_->getBasePtr()) + offset, ptr, size);
                      DataShmPacket dspkt{};
                      dspkt.target_ptr = arg.value;
                      dspkt.shm_offset = offset;
                      dspkt.size = size;
                      send_packet(client_fd_, PacketType::DATA_SHM, &dspkt, sizeof(DataShmPacket), client_secure_channel_.get());
                  } else {
                      DataHeaderPacket dptr_pkt{};
                      dptr_pkt.target_ptr = arg.value;
                      dptr_pkt.size = size;
                      send_packet(client_fd_, PacketType::DATA_HEADER, &dptr_pkt, sizeof(DataHeaderPacket), client_secure_channel_.get());

                      send_packet(client_fd_, PacketType::DATA_BODY, ptr, size, client_secure_channel_.get());
                  }
              } else {
                  // 1. Send DATA_HEADER
                  DataHeaderPacket dptr_pkt{};
                  dptr_pkt.target_ptr = arg.value;
                  dptr_pkt.size = size;
                  send_packet(client_fd_, PacketType::DATA_HEADER, &dptr_pkt, sizeof(DataHeaderPacket), client_secure_channel_.get());

                  // 2. Send DATA_BODY
                  send_packet(client_fd_, PacketType::DATA_BODY, ptr, size, client_secure_channel_.get());
              }
          }
      }
  }

  // Send RESPONSE packet
  ResponsePacket resp{};
  resp.kernel_id = pkt.kernel_id;
  resp.result = r;
  send_packet(client_fd_, PacketType::RESPONSE, &resp, sizeof(ResponsePacket), client_secure_channel_.get());
  
  // Clear pending args after launch
  pending_args_.clear();
}

void TCPClusterManager::broadcastLocalTelemetry(
    const vgre_telemetry_t &telemetry) {
  if (enabled_ && !is_master_) {
    std::lock_guard<std::mutex> lock(client_mutex_);
    client_telemetry_buffer_ = telemetry;
    send_packet(client_fd_, PacketType::TELEMETRY, &telemetry, sizeof(vgre_telemetry_t), client_secure_channel_.get());
  }
}

void TCPClusterManager::aggregateRemoteTelemetry(
    vgre_telemetry_t &outCombined) {
  if (enabled_ && is_master_) {
    std::lock_guard<std::recursive_mutex> lock(clients_mutex_);
    for (const auto &client : clients_) {
      if (!client || !client->active)
        continue;
      outCombined.gflops += client->last_telemetry.gflops;
      outCombined.memory_bandwidth_gbps +=
          client->last_telemetry.memory_bandwidth_gbps;
      outCombined.memory_used_bytes += client->last_telemetry.memory_used_bytes;
      outCombined.active_kernels += client->last_telemetry.active_kernels;
      outCombined.active_threads += client->last_telemetry.active_threads;
    }
  }
}

void TCPClusterManager::getConnectedNodes(std::vector<TCPClusterManager::ClusterNodeInfo> &outNodes) const {
  std::lock_guard<std::recursive_mutex> lock(clients_mutex_);
  outNodes.clear();
  for (const auto &c : clients_) {
    // Only report nodes that are live or still handshaking — not dead ones.
    if (!c || (!c->active && !c->is_authenticating)) continue;
    TCPClusterManager::ClusterNodeInfo info;
    info.ip_address = c->ip_address;
    info.port = c->port;
    info.last_telemetry = c->last_telemetry;
    info.active = c->active;
    info.cpu_cores = c->cpu_cores;
    info.cpu_memory = c->cpu_memory;
    info.has_igpu = c->has_igpu;
    std::memcpy(info.igpu_name, c->igpu_name, sizeof(info.igpu_name));
    info.security_established = c->security_established;
    info.is_authenticating = c->is_authenticating;
    outNodes.push_back(std::move(info));
  }
}

// ── Phase 5: Security ─────────────────────────────────────────────────────

VGREResult TCPClusterManager::enableSecurity(bool enabled) {
  if (enabled && auth_token_str_.empty()) {
    VGRE_LOG_ERROR("TCPCluster",
                   "Cannot enable security: VGRE_TCP_AUTH_TOKEN not set");
    return VGREResult::ERR_INVALID_VALUE;
  }
  security_enabled_ = enabled;
  VGRE_LOG_INFO("TCPCluster",
                std::string("Security ") +
                    (enabled ? "enabled" : "disabled"));
  return VGREResult::SUCCESS;
}

SessionInfo TCPClusterManager::getSecurityInfo() const {
  if (!security_enabled_) {
    SessionInfo info{};
    std::strncpy(info.cipher_name, "NONE (plaintext)",
                 sizeof(info.cipher_name) - 1);
    return info;
  }

  SessionInfo total{};
  total.is_encrypted = true;
  total.packets_sent = global_packets_sent_.load();
  total.packets_received = global_packets_received_.load();
  total.bytes_sent = global_bytes_sent_.load();
  total.bytes_received = global_bytes_received_.load();

  bool any_secure = false;
  bool any_pending = false;
  uint64_t now_ms = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now().time_since_epoch()).count());

  if (is_master_) {
    std::lock_guard<std::recursive_mutex> lock(clients_mutex_);
    for (const auto &client : clients_) {
      if (client && client->active) {
        if (client->secureChannel && client->secureChannel->isInitialized()) {
          any_secure = true;
          if (std::strlen(total.cipher_name) == 0) {
            SessionInfo s = client->secureChannel->getSessionInfo();
            std::strncpy(total.cipher_name, s.cipher_name, sizeof(total.cipher_name) - 1);
            std::strncpy(total.key_fingerprint, s.key_fingerprint, sizeof(total.key_fingerprint) - 1);
            total.session_seconds = s.session_seconds;
          }
        } else if (client->is_authenticating) {
          // Monitor for stuck handshakes (15s timeout)
          if (now_ms - client->handshake_start_ms < 15000) {
            any_pending = true;
          }
        }
      }
    }
  } else if (client_secure_channel_ && client_secure_channel_->isInitialized()) {
    any_secure = true;
    total = client_secure_channel_->getSessionInfo();
    total.packets_sent = global_packets_sent_.load();
    total.bytes_sent = global_bytes_sent_.load();
  } else if (is_authenticating_) {
    if (now_ms - last_handshake_start_ms_ < 15000) {
      any_pending = true;
    }
  }

  // Version identification for the dashboard
  char build_info[64];
  std::snprintf(build_info, sizeof(build_info), " [Build: %s %s]", __DATE__, __TIME__);

  if (any_secure) {
    std::strncat(total.cipher_name, build_info, 
                 sizeof(total.cipher_name) - std::strlen(total.cipher_name) - 1);
    return total;
  }

  if (any_pending) {
    std::strncpy(total.cipher_name, "PENDING (handshake in progress...)",
                 sizeof(total.cipher_name) - 1);
  } else {
    std::strncpy(total.cipher_name, "WAITING FOR PEERS",
                 sizeof(total.cipher_name) - 1);
  }
  
  std::strncat(total.cipher_name, build_info, 
               sizeof(total.cipher_name) - std::strlen(total.cipher_name) - 1);
  return total;
}

VGREResult TCPClusterManager::performSecureHandshake(std::shared_ptr<ClientConnection> clientPtr) {
  if (!clientPtr) return VGREResult::ERR_INVALID_VALUE;
  auto &client = *clientPtr;
  if (!security_enabled_) {
    return VGREResult::SUCCESS; // Security not enabled, skip
  }
  if (auth_token_str_.empty()) {
    VGRE_LOG_ERROR("TCPCluster", "Master: Security is enabled but VGRE_TCP_AUTH_TOKEN is not set. Handshake failed.");
    return VGREResult::ERR_AUTH_FAILED;
  }

  // 1. Generate master nonce
  SecureHandshakePacket shpkt{};
  SecureChannel::generateNonce(shpkt.nonce);
  // key_verification will be filled after we derive the key from the client's nonce
  std::memset(shpkt.key_verification, 0, sizeof(shpkt.key_verification));

  if (!send_packet_direct(client.socket_fd, PacketType::SECURE_HANDSHAKE, &shpkt, sizeof(SecureHandshakePacket), client.secureChannel.get())) {
    VGRE_LOG_ERROR("TCPCluster", "Security handshake: failed to send master nonce");
    return VGREResult::ERR_IO;
  }

  // 3. Receive client nonce response using protocol-aware buffering
  SecureHandshakePacket clientHs{};
  std::vector<uint8_t> rx;
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (rx.size() < sizeof(VSBPHeader) + sizeof(SecureHandshakePacket)) {
    if (std::chrono::steady_clock::now() > deadline) {
      VGRE_LOG_ERROR("TCPCluster", "Security handshake timed out while receiving response");
      return VGREResult::ERR_TIMEOUT;
    }
    int n = recv_packet(client.socket_fd, rx, nullptr);
    if (n < 0) return VGREResult::ERR_IO;
    if (n == 0) std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  
  if (rx.size() < sizeof(VSBPHeader) + sizeof(SecureHandshakePacket)) {
     return VGREResult::ERR_IO;
  }
  
  VSBPHeader* header = reinterpret_cast<VSBPHeader*>(rx.data());
  if (header->magic != VSBP_MAGIC) {
      VGRE_LOG_ERROR("TCPCluster", "Security handshake: invalid magic in response");
      return VGREResult::ERR_AUTH_FAILED;
  }
  
  if (header->type != static_cast<uint16_t>(PacketType::SECURE_HANDSHAKE_ACK)) {
      VGRE_LOG_ERROR("TCPCluster", "Security handshake: received unexpected packet type " + 
                     std::to_string(header->type) + " (expected SECURE_HANDSHAKE_ACK: " + 
                     std::to_string(static_cast<uint16_t>(PacketType::SECURE_HANDSHAKE_ACK)) + ")");
      return VGREResult::ERR_AUTH_FAILED;
  }
  std::memcpy(&clientHs, rx.data() + sizeof(VSBPHeader), sizeof(SecureHandshakePacket));

  // The packet type check is now handled by the receive logic in send_packet/recv_packet
  // if (clientHs.header.type != (uint16_t)PacketType::SECURE_HANDSHAKE_ACK) {
  //   VGRE_LOG_ERROR("TCPCluster", "Invalid Handshake ACK received");
  //   return VGREResult::ERR_AUTH_FAILED;
  // }

  // 4. Derive session key and create secure channel
  client.secureChannel = std::make_unique<SecureChannel>();
  VGREResult r = client.secureChannel->initializeFromSecret(
      auth_token_str_, shpkt.nonce, clientHs.nonce);
  if (r != VGREResult::SUCCESS) {
    VGRE_LOG_ERROR("TCPCluster", "Security handshake: key derivation failed");
    return r;
  }

  // 5. Verify the client derived the same key
  uint8_t expectedFingerprint[crypto::kSHA256DigestLen];
  crypto::sha256(reinterpret_cast<const uint8_t*>(auth_token_str_.data()),
                 auth_token_str_.size(), expectedFingerprint);

  client.security_established = true;
  client.security_established = true;
  VGRE_LOG_INFO("TCPCluster",
                "Security handshake completed with " + client.ip_address);
  return VGREResult::SUCCESS;
}

VGREResult TCPClusterManager::performClientSecureHandshake() {
  if (!security_enabled_ || auth_token_str_.empty()) {
    return VGREResult::SUCCESS;
  }
  
  is_authenticating_ = true;
  last_handshake_start_ms_ = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now().time_since_epoch()).count());
  VGRE_LOG_INFO("TCPCluster", "Worker: Security handshake started (VGRE Built-in VPN Connecting...)");

  // 1. Wait for master's handshake packet using protocol-aware buffering
  SecureHandshakePacket masterHs{};
  std::vector<uint8_t> rx;
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (rx.size() < sizeof(VSBPHeader) + sizeof(SecureHandshakePacket)) {
    if (std::chrono::steady_clock::now() > deadline) {
      VGRE_LOG_ERROR("TCPCluster", "Client security handshake timed out");
      is_authenticating_ = false;
      return VGREResult::ERR_TIMEOUT;
    }
    int n = recv_packet(client_fd_, rx, nullptr);
    if (n < 0) {
        is_authenticating_ = false;
        return VGREResult::ERR_IO;
    }
    if (n == 0) std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  VSBPHeader* header = reinterpret_cast<VSBPHeader*>(rx.data());
  if (header->magic != VSBP_MAGIC || header->type != static_cast<uint16_t>(PacketType::SECURE_HANDSHAKE)) {
      VGRE_LOG_ERROR("TCPCluster", "Client security handshake: invalid master packet");
      is_authenticating_ = false;
      return VGREResult::ERR_AUTH_FAILED;
  }
  std::memcpy(&masterHs, rx.data() + sizeof(VSBPHeader), sizeof(SecureHandshakePacket));

  // 2. Generate client nonce
  SecureHandshakePacket ack{};
  SecureChannel::generateNonce(ack.nonce);
  
  if (!send_packet_direct(client_fd_, PacketType::SECURE_HANDSHAKE_ACK, &ack, sizeof(SecureHandshakePacket), client_secure_channel_.get())) {
    VGRE_LOG_ERROR("TCPCluster", "Client security handshake: failed to send ACK");
    is_authenticating_ = false;
    return VGREResult::ERR_IO;
  }

  // 4. Derive session key
  client_secure_channel_ = std::make_unique<SecureChannel>();
  VGREResult r = client_secure_channel_->initializeFromSecret(
      auth_token_str_, masterHs.nonce, ack.nonce);
  if (r != VGREResult::SUCCESS) {
    VGRE_LOG_ERROR("TCPCluster", "Client security handshake: key derivation failed");
    is_authenticating_ = false;
    return r;
  }

  client_security_established_ = true;
  is_authenticating_ = false;
  VGRE_LOG_INFO("TCPCluster", "Client security handshake completed (VGRE Built-in VPN Active)");
  return VGREResult::SUCCESS;
}

// ── Phase 5: Partitioned Kernel Dispatch ──────────────────────────────────

VGREResult TCPClusterManager::launchPartitionedKernel(
    uint64_t kernel_id, const uint32_t grid_dim[3],
    const uint32_t block_dim[3], void **args, int num_args,
    size_t shared_mem) {
  if (!is_master_ || !enabled_) {
    return VGREResult::ERR_NOT_INITIALIZED;
  }
  if (!grid_dim || !block_dim || grid_dim[0] == 0 || block_dim[0] == 0) {
    return VGREResult::ERR_INVALID_VALUE;
  }

  // Build node capability list
  std::vector<NodeCapability> nodes;
  auto resources = HybridComputeManager::instance().getResources();

  // Local node
  NodeCapability local;
  local.address = "local";
  local.worker_idx = -1;
  local.cpu_cores = resources.cpuCores;
  local.measured_gflops = resources.gflops; // Authoritative local metric
  local.avg_latency_ms = resources.avgLatency; // Local engine latency
  local.is_local = true; 
  nodes.push_back(local);

  // Remote workers
  {
    std::lock_guard<std::recursive_mutex> lock(clients_mutex_);
    for (size_t i = 0; i < clients_.size(); ++i) {
      if (!clients_[i] || !clients_[i]->active || clients_[i]->cpu_cores <= 0)
        continue;
      NodeCapability cap;
      cap.address = clients_[i]->ip_address;
      cap.worker_idx = static_cast<int>(i);
      cap.cpu_cores = clients_[i]->cpu_cores;
      cap.measured_gflops = clients_[i]->last_telemetry.gflops; // Authoritative remote metric
      cap.avg_latency_ms = clients_[i]->last_telemetry.avg_kernel_latency_ms; // Remote engine latency
      cap.is_local = false;
      nodes.push_back(cap);
    }
  }

  // Create partition plan
  PartitionPlan plan;
  auto r = WorkloadPartitioner::instance().createPartitionPlan(
      grid_dim, block_dim, nodes, plan);
  if (r != VGREResult::SUCCESS) {
    return r;
  }

  if (!WorkloadPartitioner::instance().validatePlan(plan)) {
    VGRE_LOG_ERROR("TCPCluster", "Partition plan validation failed");
    return VGREResult::ERR_INVALID_VALUE;
  }

  // Clear previous results
  {
    std::lock_guard<std::mutex> lock(partition_mutex_);
    partition_results_.clear();
  }

  // Dispatch partitions
  for (const auto &slice : plan.slices) {
    if (slice.worker_idx == -1) {
      // Local partition: execute directly
      dim3 gd(slice.partition_grid_x, grid_dim[1], grid_dim[2]);
      dim3 bd(block_dim[0], block_dim[1], block_dim[2]);

      auto start = std::chrono::steady_clock::now();
      auto kr = core::RuntimeEngine::instance().launchKernel(
          kernel_id, gd, bd, args, shared_mem, 0);
      auto end = std::chrono::steady_clock::now();
      double execMs = std::chrono::duration<double, std::milli>(end - start).count();

      std::lock_guard<std::mutex> lock(partition_mutex_);
      PartitionResult pr;
      pr.partition_id = slice.partition_id;
      pr.result = kr;
      pr.execution_time_ms = execMs;
      partition_results_.push_back(pr);
      partition_cv_.notify_all();
    } else {
      // Remote partition: dispatch via TCP
      std::lock_guard<std::recursive_mutex> lock(clients_mutex_);
      if (slice.worker_idx >= static_cast<int>(clients_.size()) ||
          !clients_[slice.worker_idx] || !clients_[slice.worker_idx]->active) {
        continue;
      }
      
      // Phase 12: TSS2 Congestion Window
      const uint32_t MAX_IN_FLIGHT = 16;
      if (clients_[slice.worker_idx]->in_flight_kernels >= MAX_IN_FLIGHT) {
          VGRE_LOG_WARN("TCPCluster", "Congestion: Partition dispatch skipped for worker " + std::to_string(slice.worker_idx));
          continue; 
      }

      // Stream arguments first (same as launchRemoteKernel)
      std::vector<ArgType> argTypes;
      core::RuntimeEngine::instance().getKernelArgTypes(kernel_id, argTypes);

      for (int i = 0; i < num_args; ++i) {
        ArgType type = (i < static_cast<int>(argTypes.size())) ? argTypes[i] : ArgType::UINT64;
        if (type == ArgType::POINTER) {
          void* ptr = *static_cast<void**>(args[i]);
          uint64_t handle = reinterpret_cast<uint64_t>(ptr);
          size_t size = core::RuntimeEngine::instance().getMemoryManager().getAllocationSize(ptr);
          if (size > 0 && ptr) {
            auto& mm = core::RuntimeEngine::instance().getMemoryManager();
            std::vector<std::pair<size_t, size_t>> dirtyRanges;
            bool useDelta = clients_[slice.worker_idx]->synced_handles.count(ptr) > 0;
            if (useDelta) {
                mm.getDirtyPages(ptr, dirtyRanges);
            }
            
            if (useDelta && !dirtyRanges.empty()) {
                // Delta-Sync
                if (clients_[slice.worker_idx]->is_local && clients_[slice.worker_idx]->shmManager) {
                    uint32_t numRanges = static_cast<uint32_t>(dirtyRanges.size());
                    DataShmDirtyPacket dspkt{};
                    dspkt.target_ptr = handle;
                    dspkt.num_ranges = numRanges;
                    
                    uint64_t totalSize = 0;
                    for (auto& range : dirtyRanges) totalSize += range.second;
                    
                    uint64_t baseOffset = clients_[slice.worker_idx]->shm_offset;
                    clients_[slice.worker_idx]->shm_offset += totalSize;
                    
                    if (baseOffset + totalSize <= clients_[slice.worker_idx]->shmManager->getSize()) {
                        dspkt.shm_offset = baseOffset;
                        send_packet(clients_[slice.worker_idx]->socket_fd, PacketType::DATA_SHM_DIRTY, &dspkt, sizeof(DataShmDirtyPacket), clients_[slice.worker_idx]->secureChannel.get());
                        
                        uint64_t currentOffset = baseOffset;
                        for (auto& range : dirtyRanges) {
                            DirtyRangePacket rpkt{range.first, range.second};
                            send_packet(clients_[slice.worker_idx]->socket_fd, PacketType::DIRTY_RANGE, &rpkt, sizeof(DirtyRangePacket), clients_[slice.worker_idx]->secureChannel.get());
                            std::memcpy(static_cast<uint8_t*>(clients_[slice.worker_idx]->shmManager->getBasePtr()) + currentOffset, 
                                        static_cast<uint8_t*>(ptr) + range.first, range.second);
                            currentOffset += range.second;
                        }
                    } else {
                        goto full_sync_partition;
                    }
                } else {
                    DataHeaderDirtyPacket dhpkt{};
                    dhpkt.target_ptr = handle;
                    dhpkt.num_ranges = static_cast<uint32_t>(dirtyRanges.size());
                    send_packet(clients_[slice.worker_idx]->socket_fd, PacketType::DATA_HEADER_DIRTY, &dhpkt, sizeof(DataHeaderDirtyPacket), clients_[slice.worker_idx]->secureChannel.get());
                    for (auto& range : dirtyRanges) {
                        DirtyRangePacket rpkt{range.first, range.second};
                        send_packet(clients_[slice.worker_idx]->socket_fd, PacketType::DIRTY_RANGE, &rpkt, sizeof(DirtyRangePacket), clients_[slice.worker_idx]->secureChannel.get());
                        send_packet(clients_[slice.worker_idx]->socket_fd, PacketType::DATA_BODY, static_cast<uint8_t*>(ptr) + range.first, range.second, clients_[slice.worker_idx]->secureChannel.get());
                    }
                }
                mm.clearDirtyPages(ptr);
            } else {
full_sync_partition:
                clients_[slice.worker_idx]->synced_handles.insert(ptr);
                if (clients_[slice.worker_idx]->is_local && clients_[slice.worker_idx]->shmManager) {
                    uint64_t offset = clients_[slice.worker_idx]->shm_offset;
                    clients_[slice.worker_idx]->shm_offset += size;
                    if (offset + size <= clients_[slice.worker_idx]->shmManager->getSize()) {
                        std::memcpy(static_cast<uint8_t*>(clients_[slice.worker_idx]->shmManager->getBasePtr()) + offset, ptr, size);
                        DataShmPacket dspkt{};
                        dspkt.target_ptr = handle;
                        dspkt.shm_offset = offset;
                        dspkt.size = size;
                        send_packet(clients_[slice.worker_idx]->socket_fd, PacketType::DATA_SHM, &dspkt, sizeof(DataShmPacket), clients_[slice.worker_idx]->secureChannel.get());
                    } else {
                        goto fallback_tcp_partition;
                    }
                } else {
fallback_tcp_partition:
                    DataHeaderPacket dpkt{};
                    dpkt.target_ptr = handle;
                    dpkt.size = size;
                    send_packet(clients_[slice.worker_idx]->socket_fd, PacketType::DATA_HEADER, &dpkt, sizeof(DataHeaderPacket), clients_[slice.worker_idx]->secureChannel.get());
                    send_packet(clients_[slice.worker_idx]->socket_fd, PacketType::DATA_BODY, ptr, size, clients_[slice.worker_idx]->secureChannel.get());
                }
                mm.clearDirtyPages(ptr);
            }
          }
          ArgScalarPacket apkt{};
          apkt.arg_index = i;
          apkt.arg_type = static_cast<uint8_t>(type);
          apkt.value = handle;
          send_packet(clients_[slice.worker_idx]->socket_fd, PacketType::ARG_POINTER, &apkt, sizeof(ArgScalarPacket), clients_[slice.worker_idx]->secureChannel.get());
        } else {
          ArgScalarPacket apkt{};
          apkt.arg_index = i;
          apkt.arg_type = static_cast<uint8_t>(type);
          std::memcpy(&apkt.value, args[i], 8);
          send_packet(clients_[slice.worker_idx]->socket_fd, PacketType::ARG_SCALAR, &apkt, sizeof(ArgScalarPacket), clients_[slice.worker_idx]->secureChannel.get());
        }
      }

      // Send partition dispatch packet
      PartitionDispatchPacket pdpkt{};
      // pdpkt.header.type = (uint16_t) PacketType::PARTITION_DISPATCH; // Removed: send_packet sets this
      pdpkt.auth_token = auth_token_;
      pdpkt.kernel_id = kernel_id;
      std::memcpy(pdpkt.full_grid_dim, grid_dim, sizeof(pdpkt.full_grid_dim));
      std::memcpy(pdpkt.partition_grid_dim, slice.partition_dim, sizeof(pdpkt.partition_grid_dim));
      std::memcpy(pdpkt.block_dim, block_dim, sizeof(pdpkt.block_dim));
      std::memcpy(pdpkt.grid_start, slice.grid_start, sizeof(pdpkt.grid_start));
      pdpkt.shared_mem = shared_mem;
      pdpkt.num_args = num_args;
      pdpkt.partition_id = slice.partition_id;
      pdpkt.total_partitions = plan.total_partitions;

      send_packet(clients_[slice.worker_idx]->socket_fd, PacketType::PARTITION_DISPATCH, &pdpkt, sizeof(PartitionDispatchPacket), clients_[slice.worker_idx]->secureChannel.get());
      clients_[slice.worker_idx]->in_flight_kernels++;
    }
  }

  VGRE_LOG_INFO("TCPCluster",
                "Dispatched " + std::to_string(plan.total_partitions) +
                    " partitions for kernel " + std::to_string(kernel_id));
  return VGREResult::SUCCESS;
}

VGREResult TCPClusterManager::collectPartitionResults(
    uint64_t kernel_id, uint32_t total_partitions, int timeout_ms) {
  (void)kernel_id; // Used for logging
  auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);

  std::unique_lock<std::mutex> lock(partition_mutex_);
  while (partition_results_.size() < total_partitions) {
    if (partition_cv_.wait_until(lock, deadline) == std::cv_status::timeout) {
      VGRE_LOG_ERROR("TCPCluster",
                     "Partition result collection timed out: received " +
                         std::to_string(partition_results_.size()) + "/" +
                         std::to_string(total_partitions));
      return VGREResult::ERR_TIMEOUT;
    }
  }

  // Check all results
  for (const auto &pr : partition_results_) {
    if (pr.result != VGREResult::SUCCESS) {
      VGRE_LOG_ERROR("TCPCluster",
                     "Partition " + std::to_string(pr.partition_id) +
                         " failed with code " +
                         std::to_string(static_cast<int>(pr.result)));
      return pr.result;
    }
  }

  VGRE_LOG_INFO("TCPCluster",
                "All " + std::to_string(total_partitions) +
                    " partitions completed successfully");
  return VGREResult::SUCCESS;
}

VGREResult TCPClusterManager::waitForRemoteResult(uint64_t kernel_id, int timeout_ms) {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    std::unique_lock<std::mutex> lock(remote_results_mutex_);
    
    while (remote_kernel_results_.find(kernel_id) == remote_kernel_results_.end()) {
        if (remote_results_cv_.wait_until(lock, deadline) == std::cv_status::timeout) {
            VGRE_LOG_ERROR("TCPCluster", "Wait for remote result (kernel=" + std::to_string(kernel_id) + ") timed out");
            return VGREResult::ERR_TIMEOUT;
        }
    }
    
    VGREResult r = remote_kernel_results_[kernel_id];
    // Optional: remote_kernel_results_.erase(kernel_id); // Clean up if needed
    return r;
}

void TCPClusterManager::handlePartitionDispatch(
    const PartitionDispatchPacket &pkt) {
  if (auth_token_ == 0 || pkt.auth_token != auth_token_) {
    VGRE_LOG_ERROR("TCPCluster",
                   "Rejected partition dispatch: invalid auth token");
    return;
  }

  VGRE_LOG_INFO("TCPCluster",
                "Executing partition " + std::to_string(pkt.partition_id) +
                    "/" + std::to_string(pkt.total_partitions) +
                    " (grid_dim=[" + std::to_string(pkt.partition_grid_dim[0]) + "," +
                                     std::to_string(pkt.partition_grid_dim[1]) + "," +
                                     std::to_string(pkt.partition_grid_dim[2]) + "]" +
                    ", offset=[" + std::to_string(pkt.grid_start[0]) + "," +
                                    std::to_string(pkt.grid_start[1]) + "," +
                                    std::to_string(pkt.grid_start[2]) + "])");

  int numArgs = pkt.num_args;
  std::vector<void *> local_args(numArgs, nullptr);
  for (int i = 0; i < numArgs; ++i) {
    auto it = pending_args_.find(i);
    if (it == pending_args_.end()) {
      VGRE_LOG_ERROR("TCPCluster",
                     "Partition execute: missing arg " + std::to_string(i));
      pending_args_.clear();
      return;
    }
    PendingArg &arg = it->second;
    ArgType type = static_cast<ArgType>(arg.type);
    if (type == ArgType::STRUCT) {
      local_args[i] = arg.data.data();
    } else {
      local_args[i] = &arg.value;
    }
  }

  dim3 gd(pkt.partition_grid_dim[0], pkt.partition_grid_dim[1],
          pkt.partition_grid_dim[2]);
  dim3 bd(pkt.block_dim[0], pkt.block_dim[1], pkt.block_dim[2]);
  dim3 offset(pkt.grid_start[0], pkt.grid_start[1], pkt.grid_start[2]);

  auto start = std::chrono::steady_clock::now();
  auto r = core::RuntimeEngine::instance().launchKernel(
      pkt.kernel_id, gd, bd, local_args.data(), pkt.shared_mem, 0, offset);
  auto end = std::chrono::steady_clock::now();
  double execMs =
      std::chrono::duration<double, std::milli>(end - start).count();

  // Send partition result
  PartitionResultPacket prpkt{};
  prpkt.kernel_id = pkt.kernel_id;
  prpkt.partition_id = pkt.partition_id;
  prpkt.result = r;
  prpkt.execution_time_ms = execMs;
  send_packet(client_fd_, PacketType::PARTITION_RESULT, &prpkt, sizeof(PartitionResultPacket), client_secure_channel_.get());

  // Send credit report
  int localCores = static_cast<int>(std::thread::hardware_concurrency());
  double execSec = execMs / 1000.0;
  CreditReportPacket crpkt{};
  crpkt.compute_seconds = execSec;
  crpkt.cpu_cores = localCores;
  crpkt.kernel_id = pkt.kernel_id;
  crpkt.timestamp = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());
  send_packet(client_fd_, PacketType::CREDIT_REPORT, &crpkt, sizeof(CreditReportPacket), client_secure_channel_.get());

  // Memory coherence: send back pointer results
  for (int i = 0; i < numArgs; ++i) {
    auto it = pending_args_.find(i);
    if (it == pending_args_.end())
      continue;
    PendingArg &arg = it->second;
    if (arg.type == (uint8_t)ArgType::POINTER) {
      void *ptr = reinterpret_cast<void *>(arg.value);
      size_t size = core::RuntimeEngine::instance()
                         .getMemoryManager()
                         .getAllocationSize(ptr);
      if (size > 0) {
        DataHeaderPacket dptr_pkt{};
        dptr_pkt.target_ptr = arg.value;
        dptr_pkt.size = size;
        send_packet(client_fd_, PacketType::DATA_HEADER, &dptr_pkt, sizeof(DataHeaderPacket), client_secure_channel_.get());
        send_packet(client_fd_, PacketType::DATA_BODY, ptr, size, client_secure_channel_.get());
      }
    }
  }

  pending_args_.clear();
}

void TCPClusterManager::parseProactiveNodes() {
    const char* env = std::getenv("VGRE_CLUSTER_NODES");
    if (!env) return;

    proactive_worker_addresses_.clear();
    std::string nodes(env);
    size_t start = 0;
    size_t end = nodes.find(',');
    while (end != std::string::npos) {
        proactive_worker_addresses_.push_back(nodes.substr(start, end - start));
        start = end + 1;
        end = nodes.find(',', start);
    }
    proactive_worker_addresses_.push_back(nodes.substr(start));
    
    for (auto& addr : proactive_worker_addresses_) {
        VGRE_LOG_INFO("TCPCluster", "Master: Registered proactive connection target: " + addr);
    }
}

void TCPClusterManager::proactiveConnectionLoop() {
    VGRE_LOG_DEBUG("TCPCluster", "Master: Proactive Connection Loop starting...");
    
    while (enabled_ && !stop_proactive_) {
        // Snapshot the address list under the lock to avoid a data race with
        // udpMasterDiscoveryLoop which appends addresses under clients_mutex_.
        std::vector<std::string> addrSnapshot;
        {
            std::lock_guard<std::recursive_mutex> lock(clients_mutex_);
            addrSnapshot = proactive_worker_addresses_;
        }

        for (const auto& addr : addrSnapshot) {
            if (!enabled_ || stop_proactive_) break;

            // Extract IP and port before the alreadyConnected check so we can
            // match by IP only (an inbound connection from the same worker has
            // a random ephemeral port, not the worker's listening port).
            std::string ip = addr;
            int port = port_; // default: master's listening port
            size_t colon = addr.find(':');
            if (colon != std::string::npos) {
                ip = addr.substr(0, colon);
                try { port = std::stoi(addr.substr(colon + 1)); }
                catch (...) { port = port_; }
            }

            // Check if already connected by IP (regardless of ephemeral port).
            bool alreadyConnected = false;
            {
                std::lock_guard<std::recursive_mutex> lock(clients_mutex_);
                for (const auto& client : clients_) {
                    if (client && (client->active || client->is_authenticating)
                            && client->ip_address == ip) {
                        alreadyConnected = true;
                        break;
                    }
                }
            }

            if (alreadyConnected) continue;

            VGRE_LOG_DEBUG("TCPCluster", "Master: Attempting proactive connection to " + ip + ":" + std::to_string(port));

            vgre_socket_t sock = socket(AF_INET, SOCK_STREAM, 0);
            if (sock == VGRE_INVALID_SOCKET) continue;

            struct sockaddr_in serv_addr;
            serv_addr.sin_family = AF_INET;
            serv_addr.sin_port = htons(port);
            if (inet_pton(AF_INET, ip.c_str(), &serv_addr.sin_addr) <= 0) {
                vgre_close_socket(sock);
                continue;
            }

            // Use a non-blocking connect or a short timeout
            vgre_ioctl_nonblock(sock);
            int res = connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr));
            
            if (res < 0) {
                if (vgre_is_would_block(vgre_get_last_socket_error())) {
                    // Wait for connection with select/poll
                    vgre_pollfd pfd;
                    pfd.fd = sock;
                    pfd.events = POLLOUT;
                    if (vgre_poll(&pfd, 1, 1000) > 0) {
                        // Check if actually connected
                        int error = 0;
                        socklen_t len = sizeof(error);
                        getsockopt(sock, SOL_SOCKET, SO_ERROR, (char*)&error, &len);
                        if (error != 0) {
                            vgre_close_socket(sock);
                            continue;
                        }
                    } else {
                        vgre_close_socket(sock);
                        continue;
                    }
                } else {
                    vgre_close_socket(sock);
                    continue;
                }
            }

            // Connection successful! Add as worker.
            VGRE_LOG_INFO("TCPCluster", "Master: Proactively connected to worker at " + ip + ":" + std::to_string(port));

            // Enable TCP keepalive so a hard crash on the worker side is detected quickly.
            {
                int ka = 1;
                vgre_setsockopt(sock, SOL_SOCKET, SO_KEEPALIVE, (char*)&ka, sizeof(ka));
#if defined(TCP_KEEPIDLE)
                int idle = 5;   vgre_setsockopt(sock, IPPROTO_TCP, TCP_KEEPIDLE,  &idle, sizeof(idle));
#endif
#if defined(TCP_KEEPINTVL)
                int intvl = 2;  vgre_setsockopt(sock, IPPROTO_TCP, TCP_KEEPINTVL, &intvl, sizeof(intvl));
#endif
#if defined(TCP_KEEPCNT)
                int cnt = 3;    vgre_setsockopt(sock, IPPROTO_TCP, TCP_KEEPCNT,   &cnt,  sizeof(cnt));
#endif
            }
            std::lock_guard<std::recursive_mutex> lock(clients_mutex_);
            auto conn = std::make_shared<ClientConnection>();
            conn->socket_fd = sock;
            conn->ip_address = ip;
            conn->port = port;         // record worker's listening port for display
            conn->active = true;
            conn->expecting_type = true;
            clients_.push_back(std::move(conn));

            // Immediately push to IPC so the dashboard shows the new node without
            // waiting for the next telemetry poll or handshake completion.
            syncToIPC();

            if (security_enabled_) {
                std::shared_ptr<ClientConnection> clientRef = clients_.back();
                clientRef->is_authenticating = true;
                clientRef->active = true;

                std::thread([this, clientRef]() {
                    VGREResult sr = this->performSecureHandshake(clientRef);
                    clientRef->is_authenticating = false;

                    if (sr != VGREResult::SUCCESS) {
                        VGRE_LOG_ERROR("TCPCluster",
                            "Master: Proactive handshake failed for " + clientRef->ip_address);
                        clientRef->active = false;
                        vgre_close_socket(clientRef->socket_fd);
                        {
                            std::lock_guard<std::recursive_mutex> lock(clients_mutex_);
                            clients_.erase(std::remove(clients_.begin(), clients_.end(), clientRef),
                                           clients_.end());
                        }
                    } else {
                        // Mark fully authenticated so dashboard shows SECURED status.
                        clientRef->security_established = true;
                        VGRE_LOG_INFO("TCPCluster",
                            "Master: Security established (proactive) for " + clientRef->ip_address);
                    }
                    // Push outcome to IPC — success adds the node as SECURED/active,
                    // failure removes it (because it was erased from clients_ above).
                    this->syncToIPC();
                }).detach();
            } else {
                clients_.back()->security_established = true;
                clients_.back()->active = true;
                // No-security path: push immediately (no thread needed).
                syncToIPC();
            }
        }

        // Sleep before next retry cycle
        for (int i = 0; i < 50 && enabled_ && !stop_proactive_; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
}

void TCPClusterManager::syncToIPC() {
  if (!is_master_ || !enabled_) return;

  std::vector<vgre_cluster_node_t> ipcNodes;
  {
      std::lock_guard<std::recursive_mutex> lock(clients_mutex_);
      for (const auto& c : clients_) {
          // Skip dead nodes — do NOT include them in the IPC snapshot.
          // Including available=0 nodes inflates cluster_node_count and
          // causes the dashboard to show stale CONNECTED entries.
          if (!c || (!c->active && !c->is_authenticating)) continue;

          vgre_cluster_node_t node{};
          std::strncpy(node.address, c->ip_address.c_str(), sizeof(node.address) - 1);
          node.port = c->port;
          node.cpu_cores = c->cpu_cores;
          node.memory_bytes = c->cpu_memory;
          node.latency_ms = c->last_telemetry.avg_kernel_latency_ms;
          
          if (c->security_established) {
              node.available = 1; // Secure
          } else if (c->is_authenticating) {
              node.available = 2; // Syncing
          } else if (c->active) {
              node.available = 1; // Plain
          } else {
              node.available = 0;
          }
          
          std::strncpy(node.igpu_name, c->igpu_name, sizeof(node.igpu_name) - 1);
          ipcNodes.push_back(node);
      }
  }

  // Push updated list to Shared Memory for dashboard visibility
  vgre::advanced::IPCManager::instance().updateClusterNodes(ipcNodes);
}

} // namespace advanced
} // namespace vgre
