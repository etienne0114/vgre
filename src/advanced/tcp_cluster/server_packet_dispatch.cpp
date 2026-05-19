#include "vgre/advanced/hybrid_compute_manager.h"
#include "vgre/advanced/rdma_transport.h"
#include "vgre/advanced/resource_ledger.h"
#include "vgre/advanced/tcp_cluster.h"
#include "vgre/advanced/tcp_cluster/internal/interfaces.h"
#include "vgre/advanced/tcp_cluster/internal/dispatch_manager.h"
#include "vgre/advanced/tcp_cluster/internal/shared_utilities.h"
#include "vgre/api/vgre_c_api.h"
#include "vgre/common/logger.h"
#include "vgre/core/memory_manager.h"
#include "vgre/core/runtime_engine.h"

namespace vgre {
namespace advanced {

namespace {
const size_t kProbePayloadBytes = []() -> size_t {
  const char *env = vgre_get_config("VGRE_CLUSTER_PROBE_BYTES");
  if (env) {
    try {
      long long v = std::stoll(env);
      if (v >= 4096 && v <= 64 * 1024 * 1024)
        return static_cast<size_t>(v);
    } catch (...) {
    }
  }
  return 1024ULL * 1024;
}();
const size_t kMaxPacketsPerSec = []() -> size_t {
  const char *env = vgre_get_config("VGRE_CLUSTER_MAX_PACKETS_PER_SEC");
  if (env) {
    try {
      long long v = std::stoll(env);
      if (v > 0)
        return static_cast<size_t>(v);
    } catch (...) {
    }
  }
  return 10000;
}();
} // namespace

void TCPClusterManager::processServerPackets(
    std::shared_ptr<ClientConnection> client) {
  while (client->active && !client->rx_buffer.empty()) {
    if (client->receive_state == ReceiveState::IDLE) {
      // Need at least a full VSBP header
      if (client->rx_buffer.size() < sizeof(VSBPHeader))
        break;

      VSBPHeader hdr;
      memcpy(&hdr, client->rx_buffer.data(), sizeof(VSBPHeader));

      // Per-connection packet rate limiter (token bucket, 1-second window).
      // Rejects packets from peers that exceed kMaxPacketsPerSec to prevent
      // packet-flood DoS while allowing legitimate bursting within the window.
      {
        uint64_t now_ms = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch())
                .count());
        if (now_ms - client->rate_window_start_ms >= 1000) {
          client->rate_window_start_ms = now_ms;
          client->rate_window_count = 0;
        }
        client->rate_window_count++;
        if (client->rate_window_count > kMaxPacketsPerSec) {
          VGRE_LOG_WARN("TCPCluster",
                        "Rate limit exceeded for " + client->ip_address + " (" +
                            std::to_string(client->rate_window_count) +
                            " pkts/s, limit " +
                            std::to_string(kMaxPacketsPerSec) +
                            ") — dropping packet");
          // Consume the packet from the buffer. Guard against untrusted
          // payloadSize causing an integer overflow in totalLen.
          if (hdr.payloadSize <=
              client->rx_buffer.size() - sizeof(VSBPHeader)) {
            size_t totalLenRl =
                sizeof(VSBPHeader) + static_cast<size_t>(hdr.payloadSize);
            client->rx_buffer.erase(client->rx_buffer.begin(),
                                    client->rx_buffer.begin() + totalLenRl);
          } else {
            client->rx_buffer.clear(); // incomplete packet — flush and re-sync
          }
          break;
        }
      }

      if (!PacketUtils::validateVSBPHeader(hdr)) {
        // Protocol violation — close connection to prevent desync cascade.
        size_t dump_size = std::min(client->rx_buffer.size(), size_t(64));
        std::string hex = hexDump(client->rx_buffer.data(), dump_size);
        VGRE_LOG_ERROR(
            "TCPCluster",
            "Master: VSBP protocol violation from " + client->ip_address +
                " (magic=0x" + std::to_string(hdr.magic) +
                ", version=" + std::to_string(hdr.version) +
                ", buffer_size=" + std::to_string(client->rx_buffer.size()) +
                ")" + "\nFirst " + std::to_string(dump_size) +
                " bytes (hex):\n" + hex + "\n— disconnecting client");
        client->active =
            false; // close instead of just clearing — prevents frame-desync
        break;
      }

      // Guard against integer overflow: payloadSize is uint32_t so adding
      // sizeof(VSBPHeader) can overflow size_t on 32-bit platforms; cap it.
      static constexpr uint32_t kMaxPayload = 256u * 1024u * 1024u; // 256 MB
      if (hdr.payloadSize > kMaxPayload) {
        VGRE_LOG_ERROR("TCPCluster", "Master: oversized payload (" +
                                         std::to_string(hdr.payloadSize) +
                                         " bytes) from " + client->ip_address +
                                         " — disconnecting");
        client->active = false;
        break;
      }

      if (client->rx_buffer.size() < sizeof(VSBPHeader) + hdr.payloadSize)
        break;

      const uint8_t *payload = client->rx_buffer.data() + sizeof(VSBPHeader);
      PacketType type = static_cast<PacketType>(hdr.type);
      size_t totalLen = sizeof(VSBPHeader) + hdr.payloadSize;

      if (type == PacketType::TELEMETRY) {
        if (hdr.payloadSize < sizeof(vgre_telemetry_t)) {
          client->rx_buffer.clear();
          break;
        }
        vgre_telemetry_t tel;
        memcpy(&tel, payload, sizeof(vgre_telemetry_t));
        client->last_telemetry = tel;
        vgre::advanced::HybridComputeManager::instance()
            .updateRemoteNodeTelemetry(client->ip_address, tel);
        client->rx_buffer.erase(client->rx_buffer.begin(),
                                client->rx_buffer.begin() + totalLen);
      } else if (type == PacketType::RESPONSE) {
        if (hdr.payloadSize < sizeof(ResponsePacket)) {
          client->rx_buffer.clear();
          break;
        }
        ResponsePacket resp;
        memcpy(&resp, payload, sizeof(ResponsePacket));
        client->rx_buffer.erase(client->rx_buffer.begin(),
                                client->rx_buffer.begin() + totalLen);
        VGRE_LOG_DEBUG("TCPCluster",
                       "Master: Received RESPONSE from worker (Kernel: " +
                           std::to_string(resp.kernel_id) + ")");
        if (client->in_flight_kernels > 0)
          client->in_flight_kernels--;
        dispatch_manager_->storeRemoteResult(resp.kernel_id, resp.result);
      } else if (type == PacketType::DATA_HEADER) {
        if (hdr.payloadSize < sizeof(DataHeaderPacket)) {
          client->rx_buffer.clear();
          break;
        }
        DataHeaderPacket dpkt;
        memcpy(&dpkt, payload, sizeof(DataHeaderPacket));
        client->rx_buffer.erase(client->rx_buffer.begin(),
                                client->rx_buffer.begin() + totalLen);
        client->pending_target_ptr = dpkt.target_ptr;
        client->pending_data_size = static_cast<uint32_t>(dpkt.size);
        client->pending_num_ranges = 0;
        client->pending_range_offset = 0;
        client->receive_state = ReceiveState::EXPECTING_BODY;
      } else if (type == PacketType::DATA_HEADER_DIRTY) {
        if (hdr.payloadSize < sizeof(DataHeaderDirtyPacket)) {
          client->rx_buffer.clear();
          break;
        }
        DataHeaderDirtyPacket dhpkt;
        memcpy(&dhpkt, payload, sizeof(DataHeaderDirtyPacket));
        client->rx_buffer.erase(client->rx_buffer.begin(),
                                client->rx_buffer.begin() + totalLen);
        client->pending_target_ptr = dhpkt.target_ptr;
        client->pending_num_ranges = dhpkt.num_ranges;
        client->receive_state = ReceiveState::EXPECTING_RANGES_TCP;
      } else if (type == PacketType::DATA_SHM_DIRTY) {
        if (hdr.payloadSize < sizeof(DataShmDirtyPacket)) {
          client->rx_buffer.clear();
          break;
        }
        DataShmDirtyPacket dspkt;
        memcpy(&dspkt, payload, sizeof(DataShmDirtyPacket));
        client->rx_buffer.erase(client->rx_buffer.begin(),
                                client->rx_buffer.begin() + totalLen);
        client->pending_target_ptr = dspkt.target_ptr;
        client->pending_num_ranges = dspkt.num_ranges;
        client->pending_shm_offset = dspkt.shm_offset;
        client->receive_state = ReceiveState::EXPECTING_RANGES_SHM;
      } else if (type == PacketType::DATA_SHM) {
        if (hdr.payloadSize < sizeof(DataShmPacket)) {
          client->rx_buffer.clear();
          break;
        }
        DataShmPacket dspkt;
        memcpy(&dspkt, payload, sizeof(DataShmPacket));
        client->rx_buffer.erase(client->rx_buffer.begin(),
                                client->rx_buffer.begin() + totalLen);
        if (client->is_local && client->shm_manager) {
          void *shm_base = client->shm_manager->getBasePtr();
          size_t shm_sz = client->shm_manager->getSize();
          if (!shm_base) {
            VGRE_LOG_ERROR("TCPCluster",
                           "DATA_SHM: SHM base pointer is null for " +
                               client->ip_address);
          } else if (dspkt.shm_offset > shm_sz ||
                     dspkt.size > shm_sz - dspkt.shm_offset) {
            VGRE_LOG_ERROR(
                "TCPCluster",
                "DATA_SHM: out-of-bounds SHM read for " + client->ip_address +
                    " (offset=" + std::to_string(dspkt.shm_offset) +
                    " size=" + std::to_string(dspkt.size) +
                    " shm_sz=" + std::to_string(shm_sz) + ") — skipping");
          } else {
            void *ptr =
                core::RuntimeEngine::instance().getMemoryManager().getPointer(
                    reinterpret_cast<void *>(dspkt.target_ptr));
            if (ptr)
              memcpy(ptr,
                          static_cast<uint8_t *>(shm_base) + dspkt.shm_offset,
                          dspkt.size);
          }
        }
      } else if (type == PacketType::DATA_HEADER_RDMA) {
        // RDMA zero-copy receive (worker → master result pull-back):
        // The worker already RDMA-wrote the result into this master's
        // pre-registered bounce buffer before sending this notification.
        // Copy from the bounce buffer into the actual host allocation.
        if (hdr.payloadSize < sizeof(DataHeaderRDMAPacket)) {
          client->rx_buffer.clear();
          break;
        }
        DataHeaderRDMAPacket rdmaPkt;
        memcpy(&rdmaPkt, payload, sizeof(DataHeaderRDMAPacket));
        client->rx_buffer.erase(client->rx_buffer.begin(),
                                client->rx_buffer.begin() + totalLen);

        if (client->rdma_connected && client->rdma_conn &&
            client->rdma_conn->bounceBuf() &&
            rdmaPkt.chunk_size <= client->rdma_conn->bounceCapacity()) {
          auto &mm = core::RuntimeEngine::instance().getMemoryManager();
          void *handle = reinterpret_cast<void *>(rdmaPkt.target_ptr);
          // Allocate only for the first chunk (dst_offset == 0)
          if (rdmaPkt.dst_offset == 0 && !mm.isValidHandle(handle)) {
            void *act = nullptr;
            mm.allocateManagedAt(handle, rdmaPkt.chunk_size, act);
          }
          void *local_ptr = mm.getPointer(handle);
          if (local_ptr) {
            // Copy this chunk to the correct byte offset inside the allocation.
            memcpy(static_cast<uint8_t*>(local_ptr) + rdmaPkt.dst_offset,
                        client->rdma_conn->bounceBuf(),
                        rdmaPkt.chunk_size);
            // Acquire fence: ensures CPU reads NIC-written data, not stale cache
            // (needed on non-x86 ARM where RDMA DMA bypasses the CPU cache hierarchy).
            std::atomic_thread_fence(std::memory_order_acquire);
            VGRE_LOG_DEBUG("TCPCluster",
                           "Master: RDMA receive chunk " +
                               std::to_string(rdmaPkt.chunk_size) + " bytes at offset " +
                               std::to_string(rdmaPkt.dst_offset) + " from " +
                               client->ip_address + " → handle " +
                               std::to_string(rdmaPkt.target_ptr));
          } else {
            VGRE_LOG_ERROR(
                "TCPCluster",
                "Master: DATA_HEADER_RDMA: no local pointer for handle " +
                    std::to_string(rdmaPkt.target_ptr) + " from " +
                    client->ip_address);
          }
        } else {
          VGRE_LOG_WARN("TCPCluster",
                        "Master: DATA_HEADER_RDMA from " + client->ip_address +
                            " but RDMA not active or chunk_size " +
                            std::to_string(rdmaPkt.chunk_size) +
                            " exceeds bounce capacity — data lost");
        }
      } else if (type == PacketType::CAPABILITY) {
        if (hdr.payloadSize < sizeof(CapabilityPacket)) {
          client->rx_buffer.clear();
          break;
        }
        CapabilityPacket cpkt;
        memcpy(&cpkt, payload, sizeof(CapabilityPacket));
        // B3: Force null-termination on all string fields.
        cpkt.igpu_name[sizeof(cpkt.igpu_name) - 1] = '\0';
        cpkt.gpu_name[sizeof(cpkt.gpu_name) - 1] = '\0';
        client->rx_buffer.erase(client->rx_buffer.begin(),
                                client->rx_buffer.begin() + totalLen);
        client->cpu_cores = cpkt.cpu_cores;
        client->cpu_memory = cpkt.cpu_memory;
        client->has_igpu = cpkt.has_igpu;
        std::snprintf(client->igpu_name, sizeof(client->igpu_name), "%s",
                      cpkt.igpu_name);
        // Gate: node is now visible in the dashboard with real hardware info.
        client->capability_received = true;

        // Log GPU capability if the worker reported a discrete GPU.
        if (cpkt.gpu_count > 0) {
          VGRE_LOG_INFO(
              "TCPCluster",
              "Worker " + client->ip_address + " has " +
                  std::to_string(cpkt.gpu_count) + " GPU(s): " + cpkt.gpu_name +
                  " (" + std::to_string(cpkt.gpu_memory_bytes / (1024 * 1024)) +
                  " MB, " + "SM " + std::to_string(cpkt.gpu_compute_major) +
                  "." + std::to_string(cpkt.gpu_compute_minor) + ")");
        }

        HybridComputeManager::instance().updateRemoteNodeCapability(
            client->ip_address, cpkt.cpu_cores, cpkt.cpu_memory, cpkt.has_igpu,
            cpkt.igpu_name);

        // Auto-enable full-mesh when a second worker joins — without requiring
        // VGRE_ENABLE_MESH_TOPOLOGY to be set explicitly.  Once enabled, mesh
        // is never auto-disabled (workers can leave and re-join freely).
        if (!mesh_topology_enabled_) {
            int ready = 0;
            for (const auto& c : clients_) {
                if (c && c->active && c->capability_received) ++ready;
            }
            if (ready >= 2) {
                mesh_topology_enabled_ = true;
                VGRE_LOG_INFO("TCPCluster",
                    "Full-mesh topology auto-enabled (" + std::to_string(ready) +
                    " workers connected).");
            }
        }

        syncToIPC();

        // Local-only SHM transport: negotiate after CAPABILITY so the worker
        // has fully transitioned to encrypted traffic. SHM_INIT is sent
        // encrypted to prevent plaintext/secure framing desync.
        if (client->is_local && client->security_established &&
            client->secure_channel && client->secure_channel->isInitialized() &&
            !client->shm_manager) {
          client->shm_manager = std::make_unique<vgre::core::ShmManager>();
          std::string shmName = "vgre_shm_" + std::to_string(client->socket_fd);
          static const size_t shmSize = []() -> size_t {
            const char *env = vgre_get_config("VGRE_CLUSTER_SHM_SIZE");
            if (env) {
              try {
                long long v = std::stoll(env);
                if (v > 0) return static_cast<size_t>(v);
              } catch (...) {
              }
            }
            return 256ULL * 1024 * 1024;
          }();
          if (client->shm_manager->open(shmName, shmSize, true) ==
              VGREResult::SUCCESS) {
            ShmInitPacket sipkt{};
            strncpy(sipkt.shm_name, shmName.c_str(),
                         sizeof(sipkt.shm_name) - 1);
            sipkt.shm_size = shmSize;
            (void)send_packet_direct(client->socket_fd, PacketType::SHM_INIT,
                                     &sipkt, sizeof(ShmInitPacket),
                                     client->secure_channel.get());
          } else {
            client->shm_manager.reset();
          }
        }

        // ── Optional RDMA upgrade after CAPABILITY ─────────────────
        // Try to establish an RDMA QP with this worker so that bulk
        // DATA_BODY packets (> 64 KB) can use zero-copy RDMA WRITE
        // instead of TCP copy.  Falls back silently if RDMA unavailable.
        if (!client->rdma_connected) {
          if (!client->rdma_ctx) {
            client->rdma_ctx.reset(RDMAContext::tryCreate());
          }
          if (client->rdma_ctx && client->secure_channel &&
              client->secure_channel->isInitialized()) {
            client->rdma_conn = std::make_unique<RDMAConnection>();
            if (client->rdma_conn->connect(*client->secure_channel,
                                           *client->rdma_ctx)) {
              client->rdma_connected = true;
              VGRE_LOG_INFO("TCPCluster", "Master: RDMA transport active for " +
                                              client->ip_address);
            } else {
              client->rdma_conn.reset();
              client->rdma_ctx.reset();
              VGRE_LOG_INFO("TCPCluster", "Master: RDMA not available for " +
                                              client->ip_address +
                                              " — using TCP");
            }
          }
        }

        // Launch bandwidth probe: send a BANDWIDTH_PROBE immediately
        // after CAPABILITY so the worker can reply with BANDWIDTH_ACK.
        // The round-trip time is used to estimate effective network bandwidth
        // which feeds the workload partitioner (analysis §1.1).
        //
        // IMPORTANT: Only send the probe after security is established.
        // Sending it before the handshake completes causes the probe's
        // 1MB zero-filled payload to arrive at the worker while
        // performClientHandshake() is still in its bounded-recv peek
        // window. The peek reads 88 bytes of the probe payload (not the
        // SECURE_HANDSHAKE), stashes them in staging, and the remaining
        // ~1MB of zeros floods processClientStagingBuffer as invalid
        // VSBP frames (magic=0x0). Gate on security_established to
        // ensure the handshake has fully completed before sending.
        if (!client->bandwidth_probe_in_flight && !client->is_local &&
            client->security_established) {
          uint64_t now_ms = static_cast<uint64_t>(
              std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::system_clock::now().time_since_epoch())
                  .count());
          // Build probe buffer: [8-byte timestamp][kProbePayloadBytes zeros]
          std::vector<uint8_t> probe_buf(sizeof(uint64_t) + kProbePayloadBytes,
                                         0);
          memcpy(probe_buf.data(), &now_ms, sizeof(uint64_t));
          client->bandwidth_probe_start = std::chrono::steady_clock::now();
          client->bandwidth_probe_in_flight = true;
          send_packet(client->socket_fd, PacketType::BANDWIDTH_PROBE,
                      probe_buf.data(), probe_buf.size(),
                      client->secure_channel.get());
        }
      } else if (type == PacketType::BANDWIDTH_ACK) {
        if (hdr.payloadSize < sizeof(BandwidthAckPacket)) {
          client->rx_buffer.clear();
          break;
        }
        BandwidthAckPacket ack;
        memcpy(&ack, payload, sizeof(BandwidthAckPacket));
        client->rx_buffer.erase(client->rx_buffer.begin(),
                                client->rx_buffer.begin() + totalLen);
        if (client->bandwidth_probe_in_flight) {
          auto elapsed =
              std::chrono::steady_clock::now() - client->bandwidth_probe_start;
          double rtt_s = std::chrono::duration<double>(elapsed).count();
          // Bandwidth = (probe_size_bytes × 8 bits) / (rtt_s × 1e9) Gbps.
          // RTT is used as a conservative lower-bound; latency = RTT/2.
          if (rtt_s > 0.0) {
            const double kProbeBytes = static_cast<double>(kProbePayloadBytes);
            double bw_gbps = (kProbeBytes * 8.0) / (rtt_s * 1e9);
            bw_gbps = std::max(0.001, std::min(100.0, bw_gbps));
            client->network_bandwidth_gbps = bw_gbps;
            // One-way latency estimate = RTT / 2
            client->network_latency_ms = (rtt_s * 1000.0) / 2.0;
            VGRE_LOG_INFO(
                "TCPCluster",
                "Bandwidth probe to " + client->ip_address +
                    ": RTT=" + std::to_string(static_cast<int>(rtt_s * 1000)) +
                    "ms, latency=" +
                    std::to_string(client->network_latency_ms) +
                    "ms, estimated bandwidth=" + std::to_string(bw_gbps) +
                    " Gbps");
          }
          client->last_bandwidth_probe_time = std::chrono::steady_clock::now();
          client->bandwidth_probe_in_flight = false;
        }
      } else if (type == PacketType::PARTITION_RESULT) {
        if (hdr.payloadSize < sizeof(PartitionResultPacket)) {
          client->rx_buffer.clear();
          break;
        }
        PartitionResultPacket prpkt;
        memcpy(&prpkt, payload, sizeof(PartitionResultPacket));
        client->rx_buffer.erase(client->rx_buffer.begin(),
                                client->rx_buffer.begin() + totalLen);
        if (client->in_flight_kernels > 0)
          client->in_flight_kernels--;
        dispatch_manager_->storePartitionResult(prpkt.partition_id,
                                                prpkt.kernel_id, prpkt.result,
                                                prpkt.execution_time_ms);
      } else if (type == PacketType::CREDIT_REPORT) {
        if (hdr.payloadSize < sizeof(CreditReportPacket)) {
          client->rx_buffer.clear();
          break;
        }
        CreditReportPacket crpkt;
        memcpy(&crpkt, payload, sizeof(CreditReportPacket));
        client->rx_buffer.erase(client->rx_buffer.begin(),
                                client->rx_buffer.begin() + totalLen);
        ResourceLedger::instance().recordCompute(
            client->ip_address, crpkt.compute_seconds, crpkt.cpu_cores,
            crpkt.kernel_id, CreditDirection::DEBIT);
      } else if (type == PacketType::ROTATE_KEY) {
        if (hdr.payloadSize < sizeof(SecureHandshakePacket)) {
          client->rx_buffer.clear();
          break;
        }
        SecureHandshakePacket rpkt;
        memcpy(&rpkt, payload, sizeof(SecureHandshakePacket));
        client->rx_buffer.erase(client->rx_buffer.begin(),
                                client->rx_buffer.begin() + totalLen);
        if (client->secure_channel)
          client->secure_channel->rotateKey(rpkt.nonce);
      } else if (type == PacketType::COLLECTIVE_COMPLETE) {
        // Worker → master ACK for collective result receipt. No master-side
        // state change needed; just consume the packet.
        client->rx_buffer.erase(client->rx_buffer.begin(),
                                client->rx_buffer.begin() + totalLen);
        VGRE_LOG_DEBUG("TCPCluster", "Master: COLLECTIVE_COMPLETE from " +
                                         client->ip_address);
      } else if (type == PacketType::COOP_BARRIER_SYNC) {
        client->rx_buffer.erase(client->rx_buffer.begin(),
                                client->rx_buffer.begin() + totalLen);
        {
          std::lock_guard<std::mutex> lock_b(barrier_mutex_);
          barrier_count_++;
        }
        barrier_cv_.notify_all();
      } else if (type == PacketType::CLOCK_SYNC_REPLY) {
        // MT.6: compute clock offset from NTP-style 3-timestamp exchange.
        if (hdr.payloadSize >= sizeof(ClockSyncReplyPayload)) {
          ClockSyncReplyPayload rpl;
          memcpy(&rpl, payload, sizeof(rpl));
          int64_t t4 = static_cast<int64_t>(
              std::chrono::duration_cast<std::chrono::microseconds>(
                  std::chrono::system_clock::now().time_since_epoch()).count());
          // NTP offset formula: ((T2-T1) + (T3-T4)) / 2
          // Positive = remote clock is ahead of local clock.
          client->clock_offset_us =
              ((rpl.t2_us - rpl.t1_us) + (rpl.t3_us - t4)) / 2;
          VGRE_LOG_INFO("TCPCluster",
              "Clock offset for " + client->ip_address + ": " +
              std::to_string(client->clock_offset_us) + " µs (RTT=" +
              std::to_string(t4 - rpl.t1_us) + " µs)");
        }
        client->rx_buffer.erase(client->rx_buffer.begin(),
                                client->rx_buffer.begin() + totalLen);
      } else {
        // Unknown or unhandled packet type: skip entire packet
        client->rx_buffer.erase(client->rx_buffer.begin(),
                                client->rx_buffer.begin() + totalLen);
      }
    } else if (client->receive_state == ReceiveState::EXPECTING_RANGES_TCP) {
      // DIRTY_RANGE is VSBP-framed
      if (client->rx_buffer.size() <
          sizeof(VSBPHeader) + sizeof(DirtyRangePacket))
        break;
      VSBPHeader hdr;
      memcpy(&hdr, client->rx_buffer.data(), sizeof(VSBPHeader));
      if (hdr.magic != VSBP_MAGIC) {
        client->rx_buffer.clear();
        client->receive_state = ReceiveState::IDLE;
        break;
      }
      DirtyRangePacket rpkt;
      memcpy(&rpkt, client->rx_buffer.data() + sizeof(VSBPHeader),
                  sizeof(DirtyRangePacket));
      client->rx_buffer.erase(client->rx_buffer.begin(),
                              client->rx_buffer.begin() + sizeof(VSBPHeader) +
                                  sizeof(DirtyRangePacket));
      client->pending_range_offset = rpkt.offset;
      client->pending_data_size = static_cast<uint32_t>(rpkt.size);
      client->receive_state = ReceiveState::EXPECTING_BODY;
    } else if (client->receive_state == ReceiveState::EXPECTING_RANGES_SHM) {
      // DIRTY_RANGE is VSBP-framed
      if (client->rx_buffer.size() <
          sizeof(VSBPHeader) + sizeof(DirtyRangePacket))
        break;
      VSBPHeader hdr;
      memcpy(&hdr, client->rx_buffer.data(), sizeof(VSBPHeader));
      if (hdr.magic != VSBP_MAGIC) {
        client->rx_buffer.clear();
        client->receive_state = ReceiveState::IDLE;
        break;
      }
      DirtyRangePacket rpkt;
      memcpy(&rpkt, client->rx_buffer.data() + sizeof(VSBPHeader),
                  sizeof(DirtyRangePacket));
      client->rx_buffer.erase(client->rx_buffer.begin(),
                              client->rx_buffer.begin() + sizeof(VSBPHeader) +
                                  sizeof(DirtyRangePacket));
      if (client->is_local && client->shm_manager) {
        void *shm_base = client->shm_manager->getBasePtr();
        size_t shm_sz = client->shm_manager->getSize();
        if (!shm_base) {
          VGRE_LOG_ERROR("TCPCluster",
                         "RANGES_SHM: SHM base pointer is null for " +
                             client->ip_address);
        } else if (client->pending_shm_offset > shm_sz ||
                   rpkt.size > shm_sz - client->pending_shm_offset) {
          VGRE_LOG_ERROR(
              "TCPCluster",
              "RANGES_SHM: out-of-bounds SHM read for " + client->ip_address +
                  " (offset=" + std::to_string(client->pending_shm_offset) +
                  " size=" + std::to_string(rpkt.size) +
                  " shm_sz=" + std::to_string(shm_sz) + ") — skipping");
        } else {
          void *ptr =
              core::RuntimeEngine::instance().getMemoryManager().getPointer(
                  reinterpret_cast<void *>(client->pending_target_ptr));
          if (ptr)
            memcpy(static_cast<uint8_t *>(ptr) + rpkt.offset,
                        static_cast<uint8_t *>(shm_base) +
                            client->pending_shm_offset,
                        rpkt.size);
        }
        client->pending_shm_offset += rpkt.size;
      }
      if (--client->pending_num_ranges == 0)
        client->receive_state = ReceiveState::IDLE;
    } else if (client->receive_state == ReceiveState::EXPECTING_BODY) {
      // DATA_BODY is VSBP-framed; payloadSize is the actual data length
      if (client->rx_buffer.size() < sizeof(VSBPHeader))
        break;
      VSBPHeader hdr;
      memcpy(&hdr, client->rx_buffer.data(), sizeof(VSBPHeader));
      if (hdr.magic != VSBP_MAGIC) {
        client->rx_buffer.clear();
        client->receive_state = ReceiveState::IDLE;
        break;
      }
      if (client->rx_buffer.size() < sizeof(VSBPHeader) + hdr.payloadSize)
        break;
      PacketType bodyType = static_cast<PacketType>(hdr.type);
      if (bodyType == PacketType::DATA_BODY) {
        void *ptr =
            core::RuntimeEngine::instance().getMemoryManager().getPointer(
                reinterpret_cast<void *>(client->pending_target_ptr));
        if (ptr)
          memcpy(
              static_cast<uint8_t *>(ptr) + client->pending_range_offset,
              client->rx_buffer.data() + sizeof(VSBPHeader), hdr.payloadSize);
      }
      client->rx_buffer.erase(client->rx_buffer.begin(),
                              client->rx_buffer.begin() + sizeof(VSBPHeader) +
                                  hdr.payloadSize);
      if (client->pending_num_ranges == 0) {
        client->receive_state = ReceiveState::IDLE;
      } else if (--client->pending_num_ranges > 0) {
        client->receive_state = ReceiveState::EXPECTING_RANGES_TCP;
      } else {
        client->receive_state = ReceiveState::IDLE;
      }
    } else {
      VGRE_LOG_ERROR("TCPCluster",
                     "Master: Protocol sync error — clearing buffer for " +
                         client->ip_address);
      client->rx_buffer.clear();
      break;
    }
  }
}

} // namespace advanced
} // namespace vgre
