#include "vgre/advanced/rdma_transport.h"
#include "vgre/advanced/tcp_cluster.h"
#include "vgre/advanced/tcp_cluster/internal/interfaces.h"
#include "vgre/advanced/tcp_cluster/internal/collective_ops_manager.h"
#include "vgre/advanced/tcp_cluster/internal/shared_utilities.h"
#include "vgre/common/logger.h"
#include "vgre/core/memory_manager.h"
#include "vgre/core/runtime_engine.h"
#include <cstring>

namespace vgre {
namespace advanced {

void TCPClusterManager::processClientStagingBuffer() {
  while (enabled_) {
    std::unique_lock<std::mutex> lock(staging_mutex_);
    staging_cv_.wait(lock,
                     [this]() { return staging_ready_.load() || !enabled_; });
    if (!enabled_)
      break;

    // Swap buffers
    std::swap(active_staging_, processing_staging_);
    staging_ready_ = false;

    if (processing_staging_->empty()) {
      lock.unlock();
      continue;
    }

    // Append to internal rx buffer and process (all under lock)
    client_rx_buffer_.insert(client_rx_buffer_.end(),
                             processing_staging_->begin(),
                             processing_staging_->end());
    processing_staging_->clear();

    while (enabled_ && !client_rx_buffer_.empty()) {
      if (receive_state_ == ReceiveState::IDLE) {
        if (client_rx_buffer_.size() < sizeof(VSBPHeader))
          break;

        VSBPHeader header;
        std::memcpy(&header, client_rx_buffer_.data(), sizeof(VSBPHeader));

        // VSBP Validation
        if (!PacketUtils::validateVSBPHeader(header)) {
          size_t dump_size = std::min(client_rx_buffer_.size(), size_t(64));
          std::string hex = hexDump(client_rx_buffer_.data(), dump_size);
          VGRE_LOG_ERROR(
              "TCPCluster",
              std::string("VSBP Protocol Violation: Invalid Magic/Version") +
                  " (magic=0x" + std::to_string(header.magic) +
                  ", version=" + std::to_string(header.version) +
                  ", buffer_size=" + std::to_string(client_rx_buffer_.size()) +
                  ")" + "\nFirst " + std::to_string(dump_size) +
                  " bytes (hex):\n" + hex + "\nClearing buffer.");
          client_rx_buffer_.clear();
          break;
        }

        if (client_rx_buffer_.size() < sizeof(VSBPHeader) + header.payloadSize)
          break;

        PacketType type = static_cast<PacketType>(header.type);

        if (type == PacketType::ARG_SCALAR || type == PacketType::ARG_POINTER) {
          ArgScalarPacket apkt;
          std::memcpy(&apkt, client_rx_buffer_.data() + sizeof(VSBPHeader),
                      sizeof(ArgScalarPacket));
          client_rx_buffer_.erase(client_rx_buffer_.begin(),
                                  client_rx_buffer_.begin() +
                                      sizeof(VSBPHeader) +
                                      sizeof(ArgScalarPacket));
          PendingArg arg;
          arg.type = apkt.arg_type;
          arg.value = apkt.value;
          pending_args_[apkt.arg_index] = std::move(arg);
        } else if (type == PacketType::DATA_HEADER) {
          DataHeaderPacket dpkt;
          std::memcpy(&dpkt, client_rx_buffer_.data() + sizeof(VSBPHeader),
                      sizeof(DataHeaderPacket));
          client_rx_buffer_.erase(client_rx_buffer_.begin(),
                                  client_rx_buffer_.begin() +
                                      sizeof(VSBPHeader) +
                                      sizeof(DataHeaderPacket));
          pending_target_ptr_ = dpkt.target_ptr;
          pending_data_size_ = static_cast<uint32_t>(dpkt.size);
          pending_num_ranges_ = 0;
          receive_state_ = ReceiveState::EXPECTING_BODY;
        } else if (type == PacketType::DATA_HEADER_DIRTY) {
          DataHeaderDirtyPacket dhpkt;
          std::memcpy(&dhpkt, client_rx_buffer_.data() + sizeof(VSBPHeader),
                      sizeof(DataHeaderDirtyPacket));
          client_rx_buffer_.erase(client_rx_buffer_.begin(),
                                  client_rx_buffer_.begin() +
                                      sizeof(VSBPHeader) +
                                      sizeof(DataHeaderDirtyPacket));
          pending_target_ptr_ = dhpkt.target_ptr;
          pending_num_ranges_ = dhpkt.num_ranges;
          receive_state_ = ReceiveState::EXPECTING_RANGES_TCP;
        } else if (type == PacketType::DATA_SHM_DIRTY) {
          DataShmDirtyPacket dspkt;
          std::memcpy(&dspkt, client_rx_buffer_.data() + sizeof(VSBPHeader),
                      sizeof(DataShmDirtyPacket));
          client_rx_buffer_.erase(client_rx_buffer_.begin(),
                                  client_rx_buffer_.begin() +
                                      sizeof(VSBPHeader) +
                                      sizeof(DataShmDirtyPacket));
          pending_target_ptr_ = dspkt.target_ptr;
          pending_num_ranges_ = dspkt.num_ranges;
          pending_shm_offset_ = dspkt.shm_offset;
          receive_state_ = ReceiveState::EXPECTING_RANGES_SHM;
        } else if (type == PacketType::DATA_SHM) {
          DataShmPacket dspkt;
          std::memcpy(&dspkt, client_rx_buffer_.data() + sizeof(VSBPHeader),
                      sizeof(DataShmPacket));
          client_rx_buffer_.erase(client_rx_buffer_.begin(),
                                  client_rx_buffer_.begin() +
                                      sizeof(VSBPHeader) +
                                      sizeof(DataShmPacket));
          if (client_shm_enabled_ && client_shm_manager_) {
            void *shm_base = client_shm_manager_->getBasePtr();
            size_t shm_sz = client_shm_manager_->getSize();
            if (!shm_base) {
              VGRE_LOG_ERROR("TCPCluster", "DATA_SHM: SHM base pointer is null "
                                           "— disabling SHM transport");
              client_shm_enabled_ = false;
            } else if (dspkt.shm_offset > shm_sz ||
                       dspkt.size > shm_sz - dspkt.shm_offset) {
              VGRE_LOG_ERROR("TCPCluster",
                             "DATA_SHM: out-of-bounds SHM read (offset=" +
                                 std::to_string(dspkt.shm_offset) +
                                 " size=" + std::to_string(dspkt.size) +
                                 " shm_sz=" + std::to_string(shm_sz) +
                                 ") — skipping");
            } else {
              void *handle = reinterpret_cast<void *>(dspkt.target_ptr);
              auto &mm = core::RuntimeEngine::instance().getMemoryManager();
              if (!mm.isValidHandle(handle)) {
                void *actual_ptr = nullptr;
                mm.allocateManagedAt(handle, dspkt.size, actual_ptr);
              }
              void *local_ptr = mm.getPointer(handle);
              if (local_ptr) {
                std::memcpy(local_ptr,
                            static_cast<uint8_t *>(shm_base) + dspkt.shm_offset,
                            dspkt.size);
              }
            }
          }
        } else if (type == PacketType::DATA_HEADER_RDMA) {
          // RDMA zero-copy receive: the master already RDMA-wrote the data
          // into this worker's pre-registered bounce buffer before sending
          // this notification. Copy from the bounce buffer to the allocation
          // identified by target_ptr. No DATA_BODY packet follows.
          if (client_rx_buffer_.size() <
              sizeof(VSBPHeader) + sizeof(DataHeaderRDMAPacket))
            break;
          DataHeaderRDMAPacket rdmaPkt;
          std::memcpy(&rdmaPkt, client_rx_buffer_.data() + sizeof(VSBPHeader),
                      sizeof(DataHeaderRDMAPacket));
          client_rx_buffer_.erase(client_rx_buffer_.begin(),
                                  client_rx_buffer_.begin() +
                                      sizeof(VSBPHeader) +
                                      sizeof(DataHeaderRDMAPacket));

          if (client_rdma_connected_ && client_rdma_conn_ &&
              client_rdma_conn_->bounceBuf() &&
              rdmaPkt.chunk_size <= client_rdma_conn_->bounceCapacity()) {
            void *handle = reinterpret_cast<void *>(rdmaPkt.target_ptr);
            auto &mm = core::RuntimeEngine::instance().getMemoryManager();
            // For the first chunk (dst_offset==0), allocate if not yet present.
            // Subsequent chunks reuse the existing allocation.
            if (rdmaPkt.dst_offset == 0 && !mm.isValidHandle(handle)) {
              void *act = nullptr;
              mm.allocateManagedAt(handle, rdmaPkt.chunk_size, act);
            }
            void *local_ptr = mm.getPointer(handle);
            if (local_ptr) {
              // Copy this chunk to the correct offset inside the allocation.
              std::memcpy(static_cast<uint8_t*>(local_ptr) + rdmaPkt.dst_offset,
                          client_rdma_conn_->bounceBuf(),
                          rdmaPkt.chunk_size);
              // On non-x86 (ARM) the RDMA DMA may have bypassed the CPU cache;
              // the acquire fence ensures we read the NIC-written data, not
              // stale cached values.
              std::atomic_thread_fence(std::memory_order_acquire);
              VGRE_LOG_DEBUG("TCPCluster",
                             "RDMA receive: chunk " + std::to_string(rdmaPkt.chunk_size) +
                                 " bytes at offset " + std::to_string(rdmaPkt.dst_offset) +
                                 " → handle " + std::to_string(rdmaPkt.target_ptr));
            } else {
              VGRE_LOG_ERROR("TCPCluster",
                             "DATA_HEADER_RDMA: no local pointer for handle " +
                                 std::to_string(rdmaPkt.target_ptr));
            }
          } else {
            VGRE_LOG_WARN("TCPCluster",
                          "DATA_HEADER_RDMA: RDMA not active or chunk_size " +
                              std::to_string(rdmaPkt.chunk_size) +
                              " exceeds bounce capacity — data lost");
          }
        } else if (type == PacketType::STRUCT_DATA) {
          StructDataPacket spkt;
          std::memcpy(&spkt, client_rx_buffer_.data() + sizeof(VSBPHeader),
                      sizeof(StructDataPacket));
          client_rx_buffer_.erase(client_rx_buffer_.begin(),
                                  client_rx_buffer_.begin() +
                                      sizeof(VSBPHeader) +
                                      sizeof(StructDataPacket));
          pending_struct_arg_index_ = spkt.arg_index;
          pending_struct_arg_size_ = spkt.size;
          receive_state_ = ReceiveState::EXPECTING_STRUCT_BODY;
        } else if (type == PacketType::REGISTER_KERNEL) {
          KernelRegisterPacket kpkt;
          std::memcpy(&kpkt, client_rx_buffer_.data() + sizeof(VSBPHeader),
                      sizeof(KernelRegisterPacket));
          client_rx_buffer_.erase(client_rx_buffer_.begin(),
                                  client_rx_buffer_.begin() +
                                      sizeof(VSBPHeader) +
                                      sizeof(KernelRegisterPacket));
          // B5: constant-time compare prevents timing side-channel token
          // enumeration
          if (auth_token_ != 0 &&
              !crypto::secure_compare(
                  reinterpret_cast<const uint8_t *>(&kpkt.auth_token),
                  reinterpret_cast<const uint8_t *>(&auth_token_),
                  sizeof(auth_token_))) {
            VGRE_LOG_ERROR("TCPCluster",
                           "Auth Token Mismatch during kernel registration");
            client_rx_buffer_.erase(client_rx_buffer_.begin(),
                                    client_rx_buffer_.begin() +
                                        kpkt.source_len);
            break;
          }
          pending_kernel_id_ = kpkt.kernel_id;
          pending_kernel_name_ = kpkt.name;
          pending_kernel_source_len_ = kpkt.source_len;
          receive_state_ = ReceiveState::EXPECTING_KERNEL_SOURCE;
        } else if (type == PacketType::LAUNCH_KERNEL) {
          RemoteCommandPacket pkt;
          std::memcpy(&pkt, client_rx_buffer_.data() + sizeof(VSBPHeader),
                      sizeof(RemoteCommandPacket));
          client_rx_buffer_.erase(client_rx_buffer_.begin(),
                                  client_rx_buffer_.begin() +
                                      sizeof(VSBPHeader) +
                                      sizeof(RemoteCommandPacket));
          handleRemoteCommand(pkt);
        } else if (type == PacketType::PARTITION_DISPATCH) {
          PartitionDispatchPacket pkt;
          std::memcpy(&pkt, client_rx_buffer_.data() + sizeof(VSBPHeader),
                      sizeof(PartitionDispatchPacket));
          client_rx_buffer_.erase(client_rx_buffer_.begin(),
                                  client_rx_buffer_.begin() +
                                      sizeof(VSBPHeader) +
                                      sizeof(PartitionDispatchPacket));
          handlePartitionDispatch(pkt);
        } else if (type == PacketType::ROTATE_KEY) {
          SecureHandshakePacket rpkt;
          std::memcpy(&rpkt, client_rx_buffer_.data() + sizeof(VSBPHeader),
                      sizeof(SecureHandshakePacket));
          client_rx_buffer_.erase(client_rx_buffer_.begin(),
                                  client_rx_buffer_.begin() +
                                      sizeof(VSBPHeader) +
                                      sizeof(SecureHandshakePacket));
          if (client_secure_channel_)
            client_secure_channel_->rotateKey(rpkt.nonce);
        } else if (type == PacketType::SECURE_READY) {
          // Post-handshake barrier (normally consumed in clientLoop before
          // CAPABILITY is sent). If it arrives here due to retries or timing,
          // treat it as a no-op control packet.
          client_rx_buffer_.erase(client_rx_buffer_.begin(),
                                  client_rx_buffer_.begin() +
                                      sizeof(VSBPHeader) + header.payloadSize);
        } else if (type == PacketType::SHM_INIT) {
          ShmInitPacket sipkt;
          std::memcpy(&sipkt, client_rx_buffer_.data() + sizeof(VSBPHeader),
                      sizeof(ShmInitPacket));
          client_rx_buffer_.erase(client_rx_buffer_.begin(),
                                  client_rx_buffer_.begin() +
                                      sizeof(VSBPHeader) +
                                      sizeof(ShmInitPacket));
          client_shm_manager_ = std::make_unique<core::ShmManager>();
          if (client_shm_manager_->open(sipkt.shm_name, sipkt.shm_size,
                                        false) == VGREResult::SUCCESS)
            client_shm_enabled_ = true;
        } else if (type == PacketType::BANDWIDTH_PROBE) {
          // Echo the 8-byte probe timestamp back so master can compute RTT.
          // Discard the rest of the payload (zero-filled bandwidth padding).
          uint64_t probe_sent_ms = 0;
          if (header.payloadSize >= sizeof(uint64_t))
            std::memcpy(&probe_sent_ms,
                        client_rx_buffer_.data() + sizeof(VSBPHeader),
                        sizeof(uint64_t));
          client_rx_buffer_.erase(client_rx_buffer_.begin(),
                                  client_rx_buffer_.begin() +
                                      sizeof(VSBPHeader) + header.payloadSize);
          BandwidthAckPacket ack{};
          ack.probe_sent_ms = probe_sent_ms;
          ack.ack_sent_ms = static_cast<uint64_t>(
              std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::system_clock::now().time_since_epoch())
                  .count());
          {
            std::lock_guard<std::mutex> lk(client_mutex_);
            if (client_fd_ != vgre::common::VGRE_INVALID_SOCKET)
              send_packet(client_fd_, PacketType::BANDWIDTH_ACK, &ack,
                          sizeof(ack), client_secure_channel_.get());
          }
        } else if (type == PacketType::RAW_DATA ||
                   type == PacketType::DATA_BODY) {
          // Handle RAW_DATA for collective operations
          if (is_master_ && is_reducing_ && pending_collective_count_ > 0) {
            // Master receiving worker data for reduction
            size_t element_size = ::vgre::vgre_get_type_size(
                static_cast<int>(pending_collective_datatype_));
            size_t expected_bytes = pending_collective_count_ * element_size;

            if (header.payloadSize == expected_bytes) {
              std::lock_guard<std::mutex> lock(reduction_mutex_);
              ReductionOp redOp = decodeReductionOp(pending_collective_op_type_);
              if (pending_collective_datatype_ ==
                  static_cast<uint32_t>(ArgType::FLOAT32)) {
                collective_ops_manager_->applyReduce(
                    reinterpret_cast<float *>(active_reduction_buffer_.data()),
                    reinterpret_cast<const float *>(client_rx_buffer_.data() +
                                                    sizeof(VSBPHeader)),
                    pending_collective_count_, redOp);
              } else if (pending_collective_datatype_ ==
                         static_cast<uint32_t>(ArgType::FLOAT64)) {
                collective_ops_manager_->applyReduce(
                    reinterpret_cast<double *>(active_reduction_buffer_.data()),
                    reinterpret_cast<const double *>(client_rx_buffer_.data() +
                                                     sizeof(VSBPHeader)),
                    pending_collective_count_, redOp);
              } else if (pending_collective_datatype_ ==
                         static_cast<uint32_t>(ArgType::INT32)) {
                collective_ops_manager_->applyReduce(
                    reinterpret_cast<int32_t *>(
                        active_reduction_buffer_.data()),
                    reinterpret_cast<const int32_t *>(client_rx_buffer_.data() +
                                                      sizeof(VSBPHeader)),
                    pending_collective_count_, redOp);
              } else if (pending_collective_datatype_ ==
                         static_cast<uint32_t>(ArgType::INT64)) {
                collective_ops_manager_->applyReduce(
                    reinterpret_cast<int64_t *>(
                        active_reduction_buffer_.data()),
                    reinterpret_cast<const int64_t *>(client_rx_buffer_.data() +
                                                      sizeof(VSBPHeader)),
                    pending_collective_count_, redOp);
              } else if (pending_collective_datatype_ ==
                         static_cast<uint32_t>(ArgType::FLOAT16) ||
                         pending_collective_datatype_ ==
                         static_cast<uint32_t>(ArgType::BFLOAT16)) {
                collective_ops_manager_->applyReduceFp16(
                    active_reduction_buffer_.data(),
                    client_rx_buffer_.data() + sizeof(VSBPHeader),
                    pending_collective_count_, redOp,
                    static_cast<int>(pending_collective_datatype_));
              }
              reduction_count_++;
              reduction_cv_.notify_all();
              pending_collective_count_ = 0;
            }
          } else if (!is_master_ && header.payloadSize > 0) {
            // Worker receiving result from master
            std::lock_guard<std::mutex> lock(reduction_mutex_);
            active_reduction_buffer_.assign(
                client_rx_buffer_.begin() + sizeof(VSBPHeader),
                client_rx_buffer_.begin() + sizeof(VSBPHeader) +
                    header.payloadSize);
            reduction_cv_.notify_all();
          }
          client_rx_buffer_.erase(client_rx_buffer_.begin(),
                                  client_rx_buffer_.begin() +
                                      sizeof(VSBPHeader) + header.payloadSize);
        } else if (type == PacketType::COLLECTIVE_OP) {
          // Master receives collective operation request from worker
          if (is_master_) {
            CollectiveOpPacket op_packet;
            std::memcpy(&op_packet,
                        client_rx_buffer_.data() + sizeof(VSBPHeader),
                        sizeof(CollectiveOpPacket));
            client_rx_buffer_.erase(client_rx_buffer_.begin(),
                                    client_rx_buffer_.begin() +
                                        sizeof(VSBPHeader) +
                                        sizeof(CollectiveOpPacket));
            pending_collective_op_type_ = op_packet.op_type;
            pending_collective_datatype_ = op_packet.datatype;
            pending_collective_count_ = op_packet.count;
          }
        } else if (type == PacketType::COLLECTIVE_COMPLETE) {
          // Master → worker: the all-reduce result RAW_DATA packet that
          // immediately preceded this packet is now ready. Wake any
          // workerAllReduce() call that is waiting on reduction_cv_.
          client_rx_buffer_.erase(client_rx_buffer_.begin(),
                                  client_rx_buffer_.begin() +
                                      sizeof(VSBPHeader) + header.payloadSize);
          {
            std::lock_guard<std::mutex> lock(reduction_mutex_);
            reduction_cv_.notify_all();
          }
          VGRE_LOG_DEBUG("TCPCluster", "Worker: COLLECTIVE_COMPLETE received — "
                                       "reduction result acknowledged");
        } else if (type == PacketType::COOP_BARRIER_SYNC) {
          client_rx_buffer_.erase(client_rx_buffer_.begin(),
                                  client_rx_buffer_.begin() +
                                      sizeof(VSBPHeader));
          if (is_master_) {
            std::lock_guard<std::mutex> lock_barrier(barrier_mutex_);
            barrier_count_++;
            size_t activeCount = 0;
            {
              std::lock_guard<std::recursive_mutex> lock_clients(
                  clients_mutex_);
              for (auto &c : clients_)
                if (c && c->active)
                  activeCount++;
            }
            VGRE_LOG_INFO("TCPCluster",
                          "Cooperative Barrier: Worker arrived (" +
                              std::to_string(barrier_count_) + "/" +
                              std::to_string(activeCount) + ")");
            if (barrier_count_ >= activeCount && activeCount > 0) {
              VGRE_LOG_INFO("TCPCluster", "Cooperative Barrier: All workers "
                                          "synced. Broadcasting RESUME.");
              barrier_count_ = 0;
              broadcastPacket(PacketType::COOP_BARRIER_RESUME, nullptr, 0);
            }
          }
        } else if (type == PacketType::COOP_BARRIER_RESUME) {
          client_rx_buffer_.erase(client_rx_buffer_.begin(),
                                  client_rx_buffer_.begin() +
                                      sizeof(VSBPHeader));
          VGRE_LOG_INFO(
              "TCPCluster",
              "Cooperative Barrier: RESUME received. Resuming local workers.");
          barrier_cv_.notify_all();
        } else if (type == PacketType::SECURE_HANDSHAKE ||
                   type == PacketType::SECURE_HANDSHAKE_ACK) {
          // A security handshake packet arrived in the staging buffer.
          // This is a stale packet from a duplicate inbound connection that
          // the serverLoop's duplicate-guard rejected — its bytes may have
          // been partially received before the socket was closed.
          // Discard the packet; the real handshake already completed on the
          // correct socket via performClientSecureHandshake().
          VGRE_LOG_WARN(
              "TCPCluster",
              "Worker: Discarding unexpected security handshake packet "
              "(type " +
                  std::to_string(header.type) +
                  ") in staging buffer "
                  "— stale bytes from a rejected duplicate connection");
          client_rx_buffer_.erase(
              client_rx_buffer_.begin(),
              client_rx_buffer_.begin() +
                  std::min(sizeof(VSBPHeader) + (size_t)header.payloadSize,
                           client_rx_buffer_.size()));
        } else {
          VGRE_LOG_WARN("TCPCluster",
                        "Unknown Packet Type: " + std::to_string(header.type));
          client_rx_buffer_.clear();
          break;
        }
      } else if (receive_state_ == ReceiveState::EXPECTING_KERNEL_SOURCE) {
        VSBPHeader header;
        if (client_rx_buffer_.size() < sizeof(VSBPHeader))
          break;
        std::memcpy(&header, client_rx_buffer_.data(), sizeof(VSBPHeader));
        if (client_rx_buffer_.size() < sizeof(VSBPHeader) + header.payloadSize)
          break;
        PacketType type = static_cast<PacketType>(header.type);
        if (type == PacketType::RAW_DATA) {
          std::string sourceStr(
              reinterpret_cast<const char *>(client_rx_buffer_.data() +
                                             sizeof(VSBPHeader)),
              header.payloadSize);
          core::RuntimeEngine::instance().registerKernel(
              pending_kernel_name_, sourceStr, pending_kernel_id_);
          VGRE_LOG_INFO(
              "TCPCluster",
              "Worker Registered Remote Kernel '" + pending_kernel_name_ +
                  "' (ID: " + std::to_string(pending_kernel_id_) + ")");
          client_rx_buffer_.erase(client_rx_buffer_.begin(),
                                  client_rx_buffer_.begin() +
                                      sizeof(VSBPHeader) + header.payloadSize);
          receive_state_ = ReceiveState::IDLE;
        } else {
          VGRE_LOG_ERROR("TCPCluster",
                         "Expected RAW_DATA for kernel source, got type " +
                             std::to_string(header.type));
          receive_state_ = ReceiveState::IDLE;
          client_rx_buffer_.erase(client_rx_buffer_.begin(),
                                  client_rx_buffer_.begin() +
                                      sizeof(VSBPHeader) + header.payloadSize);
        }
      } else if (receive_state_ == ReceiveState::EXPECTING_RANGES_TCP) {
        if (client_rx_buffer_.size() <
            sizeof(VSBPHeader) + sizeof(DirtyRangePacket))
          break;
        VSBPHeader hdr;
        std::memcpy(&hdr, client_rx_buffer_.data(), sizeof(VSBPHeader));
        if (hdr.magic != VSBP_MAGIC || hdr.version != VSBP_VERSION ||
            static_cast<PacketType>(hdr.type) != PacketType::DIRTY_RANGE ||
            hdr.payloadSize < sizeof(DirtyRangePacket)) {
          VGRE_LOG_ERROR(
              "TCPCluster",
              "Worker: invalid DIRTY_RANGE packet while expecting ranges");
          client_rx_buffer_.clear();
          receive_state_ = ReceiveState::IDLE;
          break;
        }
        DirtyRangePacket rpkt;
        std::memcpy(&rpkt, client_rx_buffer_.data() + sizeof(VSBPHeader),
                    sizeof(DirtyRangePacket));
        client_rx_buffer_.erase(client_rx_buffer_.begin(),
                                client_rx_buffer_.begin() + sizeof(VSBPHeader) +
                                    sizeof(DirtyRangePacket));
        pending_range_offset_ = rpkt.offset;
        pending_data_size_ = static_cast<uint32_t>(rpkt.size);
        receive_state_ = ReceiveState::EXPECTING_BODY;
      } else if (receive_state_ == ReceiveState::EXPECTING_RANGES_SHM) {
        if (client_rx_buffer_.size() <
            sizeof(VSBPHeader) + sizeof(DirtyRangePacket))
          break;
        VSBPHeader hdr;
        std::memcpy(&hdr, client_rx_buffer_.data(), sizeof(VSBPHeader));
        if (hdr.magic != VSBP_MAGIC || hdr.version != VSBP_VERSION ||
            static_cast<PacketType>(hdr.type) != PacketType::DIRTY_RANGE ||
            hdr.payloadSize < sizeof(DirtyRangePacket)) {
          VGRE_LOG_ERROR(
              "TCPCluster",
              "Worker: invalid DIRTY_RANGE packet while expecting SHM ranges");
          client_rx_buffer_.clear();
          receive_state_ = ReceiveState::IDLE;
          break;
        }
        DirtyRangePacket rpkt;
        std::memcpy(&rpkt, client_rx_buffer_.data() + sizeof(VSBPHeader),
                    sizeof(DirtyRangePacket));
        client_rx_buffer_.erase(client_rx_buffer_.begin(),
                                client_rx_buffer_.begin() + sizeof(VSBPHeader) +
                                    sizeof(DirtyRangePacket));
        if (client_shm_enabled_ && client_shm_manager_) {
          void *local_ptr =
              core::RuntimeEngine::instance().getMemoryManager().getPointer(
                  reinterpret_cast<void *>(pending_target_ptr_));
          if (local_ptr)
            std::memcpy(
                static_cast<uint8_t *>(local_ptr) + rpkt.offset,
                static_cast<uint8_t *>(client_shm_manager_->getBasePtr()) +
                    pending_shm_offset_,
                rpkt.size);
          pending_shm_offset_ += rpkt.size;
        }
        if (--pending_num_ranges_ == 0)
          receive_state_ = ReceiveState::IDLE;
      } else if (receive_state_ == ReceiveState::EXPECTING_BODY) {
        if (client_rx_buffer_.size() < sizeof(VSBPHeader))
          break;
        VSBPHeader hdr;
        std::memcpy(&hdr, client_rx_buffer_.data(), sizeof(VSBPHeader));
        if (hdr.magic != VSBP_MAGIC || hdr.version != VSBP_VERSION ||
            static_cast<PacketType>(hdr.type) != PacketType::DATA_BODY) {
          VGRE_LOG_ERROR(
              "TCPCluster",
              "Worker: invalid DATA_BODY packet while expecting body");
          client_rx_buffer_.clear();
          receive_state_ = ReceiveState::IDLE;
          break;
        }
        if (hdr.payloadSize < pending_data_size_)
          break;
        if (client_rx_buffer_.size() < sizeof(VSBPHeader) + hdr.payloadSize)
          break;
        void *handle = reinterpret_cast<void *>(pending_target_ptr_);
        auto &mm = core::RuntimeEngine::instance().getMemoryManager();
        if (!mm.isValidHandle(handle)) {
          void *act;
          mm.allocateManagedAt(handle, pending_data_size_, act);
        }
        void *local_ptr = mm.getPointer(handle);
        if (local_ptr)
          std::memcpy(static_cast<uint8_t *>(local_ptr) + pending_range_offset_,
                      client_rx_buffer_.data() + sizeof(VSBPHeader),
                      pending_data_size_);
        client_rx_buffer_.erase(client_rx_buffer_.begin(),
                                client_rx_buffer_.begin() + sizeof(VSBPHeader) +
                                    hdr.payloadSize);
        if (--pending_num_ranges_ == 0 || pending_num_ranges_ == (uint32_t)-1)
          receive_state_ = ReceiveState::IDLE;
        else
          receive_state_ = ReceiveState::EXPECTING_RANGES_TCP;
      } else if (receive_state_ == ReceiveState::EXPECTING_STRUCT_BODY) {
        if (client_rx_buffer_.size() < sizeof(VSBPHeader))
          break;
        VSBPHeader hdr;
        std::memcpy(&hdr, client_rx_buffer_.data(), sizeof(VSBPHeader));
        if (hdr.magic != VSBP_MAGIC || hdr.version != VSBP_VERSION ||
            static_cast<PacketType>(hdr.type) != PacketType::DATA_BODY ||
            hdr.payloadSize < pending_struct_arg_size_ ||
            client_rx_buffer_.size() < sizeof(VSBPHeader) + hdr.payloadSize) {
          VGRE_LOG_ERROR("TCPCluster",
                         "Worker: invalid struct DATA_BODY payload");
          client_rx_buffer_.clear();
          receive_state_ = ReceiveState::IDLE;
          break;
        }
        PendingArg arg;
        arg.type = static_cast<uint8_t>(ArgType::STRUCT);
        arg.data.assign(client_rx_buffer_.begin() + sizeof(VSBPHeader),
                        client_rx_buffer_.begin() + sizeof(VSBPHeader) +
                            pending_struct_arg_size_);
        pending_args_[pending_struct_arg_index_] = std::move(arg);
        client_rx_buffer_.erase(client_rx_buffer_.begin(),
                                client_rx_buffer_.begin() + sizeof(VSBPHeader) +
                                    hdr.payloadSize);
        pending_struct_arg_index_ = 0;
        pending_struct_arg_size_ = 0;
        receive_state_ = ReceiveState::IDLE;
      }
    }
  }
}

} // namespace advanced
} // namespace vgre
