/**
 * VGRE TCP Cluster Manager — Master Server Loop
 *
 * Contains serverLoop(): the master's accept/poll/dispatch state machine.
 * Extracted from tcp_cluster.cpp to keep that file under 1000 lines.
 */

#include "vgre/advanced/tcp_cluster.h"
#include "vgre/advanced/tcp_cluster/internal/connection_manager.h"
#include "vgre/advanced/tcp_cluster/internal/dispatch_manager.h"
#include "vgre/advanced/tcp_cluster/internal/security_manager.h"
#include "vgre/advanced/hybrid_compute_manager.h"
#include "vgre/advanced/resource_ledger.h"
#include "vgre/advanced/ipc_manager.h"
#include "vgre/core/runtime_engine.h"
#include "vgre/core/memory_manager.h"
#include "vgre/common/logger.h"
#include "vgre/common/sockets.h"

#include <chrono>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

// All platform socket headers are provided by vgre/common/sockets.h above.

namespace vgre {
namespace advanced {

using vgre::common::vgre_socket_t;
using vgre::common::VGRE_INVALID_SOCKET;
using vgre::common::vgre_pollfd;
using vgre::common::vgre_poll;
using vgre::common::vgre_close_socket;
using vgre::common::vgre_ioctl_nonblock;
using vgre::common::vgre_is_would_block;
using vgre::common::vgre_get_last_socket_error;

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
            connection_manager_->purgeDeadClients();
            if (clients_.size() != before) {
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

    // Client sockets — only poll clients whose handshake is complete.
    // Authenticating clients are exclusively owned by a detached handshake
    // thread that reads directly from the socket; including them here would
    // create a data race (two threads consuming bytes from the same FD).
    {
        std::lock_guard<std::recursive_mutex> lock(clients_mutex_);
        for (const auto &client : clients_) {
            if (client && client->active && !client->is_authenticating
                    && client->socket_fd != VGRE_INVALID_SOCKET) {
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

    // Phase 10: Periodic session-key rotation.
    // Every KEY_ROTATION_THRESHOLD packets, the master sends a ROTATE_KEY
    // packet to the worker and advances its own nonce.  packets_sent is
    // reset to 0 after rotation so the threshold fires once per interval.
    if (security_enabled_) {
        constexpr uint32_t KEY_ROTATION_THRESHOLD = 10000;
        std::lock_guard<std::recursive_mutex> lock(clients_mutex_);
        for (auto &client : clients_) {
            if (client && client->active && client->security_established &&
                    client->packets_sent >= KEY_ROTATION_THRESHOLD) {
                VGREResult rr = security_manager_->rotateSessionKey(client);
                if (rr == VGREResult::SUCCESS) {
                    client->packets_sent = 0; // reset counter for next interval
                } else {
                    VGRE_LOG_WARN("TCPCluster",
                        "Key rotation failed for " + client->ip_address +
                        " — will retry next interval");
                }
            }
        }
    }

    if (poll_res == 0) continue; // No incoming data this iteration

    // 2. Handle new connections
    if (fds[0].revents & POLLIN) {
        ::sockaddr_in address;
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
            // Enable TCP keepalive via the cross-platform helper (works on both
            // Linux and Windows via SIO_KEEPALIVE_VALS). Dead connections are
            // detected within ~11 s (idle=5s, intvl=2s, cnt=3).
            vgre::common::vgre_set_tcp_keepalive(new_socket, 5, 2, 3);

            // ── Worker fast-path ───────────────────────────────────────────
            // When a WORKER's server socket gets an inbound connection from
            // the master (proactive path), we must NOT add it to clients_ or
            // run the server-side handshake from here — that would conflict
            // with clientLoop() which is the proper owner of that socket.
            // Just store the fd and let udpDiscoveryLoop → clientLoop handle
            // the full handshake + telemetry cycle.
            if (!is_master_) {
                // Only accept the connection if we're not already connected to a master.
                // If both serverLoop (inbound) and udpDiscoveryLoop (outbound) race to
                // set client_fd_ at the same time, the second one must drop its socket.
                vgre_socket_t expected = VGRE_INVALID_SOCKET;
                vgre_socket_t desired  = new_socket;
                // client_fd_ is not atomic, but it is only written here (serverLoop, single
                // thread) and in udpDiscoveryLoop (also single-thread). Use client_mutex_
                // to make the check-then-set atomic across both sites.
                {
                    std::lock_guard<std::mutex> lock(client_mutex_);
                    if (client_fd_ != VGRE_INVALID_SOCKET) {
                        // Already have a master connection — drop this one.
                        VGRE_LOG_WARN("TCPCluster",
                            "Worker: Duplicate master connection from " +
                            std::string(ipstr) + " — dropping (already connected)");
                        vgre_close_socket(new_socket);
                        (void)expected; (void)desired;
                        continue;
                    }
                    client_fd_ = new_socket;
                }
                VGRE_LOG_INFO("TCPCluster",
                    "Worker: Accepted inbound connection from Master at " +
                    std::string(ipstr) + " — starting clientLoop");

                // clientLoop and processClientStagingBuffer are already running
                // (started in initialize()).  client_fd_ being set above is
                // sufficient — clientLoop Phase 0 will wake up and proceed.
                continue; // Skip master-only clients_/IPC/handshake code below
            }

            std::lock_guard<std::recursive_mutex> lock(clients_mutex_);

            // ── Duplicate-connection guard ─────────────────────────────────
            // Atomic check-and-insert prevents double-connection race.
            // If proactiveConnectionLoop already has an active/authenticating
            // entry for this worker IP, drop the inbound duplicate to ensure
            // exactly ONE session key exists per peer.
            const std::string inbound_ip(ipstr);
            if (!connection_manager_->addClientIfNotDuplicate(inbound_ip, new_socket, address)) {
                goto server_loop_next_iter; // skip past rate-limit else block
            }

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
                            // n < 0 → error/EOF from recv()
                            // n == 0 with POLLHUP → peer closed gracefully but recvAll timed out
                            bool hangup = (fds[i].revents & (POLLHUP | POLLERR)) != 0;
                            if (n < 0 || (n == 0 && hangup)) {
                                VGRE_LOG_INFO("TCPCluster", "Master: Worker " +
                                    client->ip_address + " disconnected.");
                                client->active = false;
                                vgre_close_socket(client->socket_fd);
                                client->socket_fd = VGRE_INVALID_SOCKET;
                                syncToIPC();
                                // Give the dashboard a visible disconnect window before
                                // the proactive loop reconnects.
                                {
                                    std::lock_guard<std::mutex> bk(proactive_backoff_mutex_);
                                    proactive_backoff_until_[client->ip_address] =
                                        std::chrono::steady_clock::now() + std::chrono::seconds(8);
                                }
                            }
                        } else {
                            VGRE_LOG_WARN("TCPCluster", "Master: Client socket error or hup from " +
                                client->ip_address);
                            client->active = false;
                            vgre_close_socket(client->socket_fd);
                            client->socket_fd = VGRE_INVALID_SOCKET;
                            syncToIPC();
                            // Give the dashboard a visible disconnect window before
                            // the proactive loop reconnects.
                            {
                                std::lock_guard<std::mutex> bk(proactive_backoff_mutex_);
                                proactive_backoff_until_[client->ip_address] =
                                    std::chrono::steady_clock::now() + std::chrono::seconds(8);
                            }
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
        // Skip clients still in the handshake thread (is_authenticating) to avoid
        // a data race: the handshake thread may concurrently write leftover bytes
        // to rx_buffer before setting is_authenticating=false.
        for (auto &client : clients_) {
          if (!client || !client->active || client->is_authenticating || client->rx_buffer.empty()) continue;

          while (client->active && !client->rx_buffer.empty()) {
            if (client->receive_state == ReceiveState::IDLE) {
              // Need at least a full VSBP header
              if (client->rx_buffer.size() < sizeof(VSBPHeader)) break;

              VSBPHeader hdr;
              std::memcpy(&hdr, client->rx_buffer.data(), sizeof(VSBPHeader));

              if (hdr.magic != VSBP_MAGIC || hdr.version != VSBP_VERSION) {
                // Preserve buffer contents for diagnostics before clearing
                size_t dump_size = std::min(client->rx_buffer.size(), size_t(64));
                std::string hex = hexDump(client->rx_buffer.data(), dump_size);
                VGRE_LOG_ERROR("TCPCluster",
                    "Master: VSBP protocol violation from " + client->ip_address +
                    " (magic=0x" + std::to_string(hdr.magic) +
                    ", version=" + std::to_string(hdr.version) +
                    ", buffer_size=" + std::to_string(client->rx_buffer.size()) + ")" +
                    "\nFirst " + std::to_string(dump_size) + " bytes (hex):\n" + hex +
                    "\n— clearing buffer");
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
                dispatch_manager_->storeRemoteResult(resp.kernel_id, resp.result);
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
                // Push real hardware info to IPC immediately so the dashboard
                // shows actual CPU cores and RAM instead of the initial 0 values.
                syncToIPC();
              } else if (type == PacketType::PARTITION_RESULT) {
                if (hdr.payloadSize < sizeof(PartitionResultPacket)) { client->rx_buffer.clear(); break; }
                PartitionResultPacket prpkt;
                std::memcpy(&prpkt, payload, sizeof(PartitionResultPacket));
                client->rx_buffer.erase(client->rx_buffer.begin(), client->rx_buffer.begin() + totalLen);
                if (client->in_flight_kernels > 0) client->in_flight_kernels--;
                dispatch_manager_->storePartitionResult(
                    prpkt.partition_id, prpkt.kernel_id, prpkt.result, prpkt.execution_time_ms);
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
    server_loop_next_iter: ; // jumped to by the duplicate-connection guard above
  }
  VGRE_LOG_DEBUG("TCPCluster", "Master Server Loop exiting.");
}

} // namespace advanced
} // namespace vgre
