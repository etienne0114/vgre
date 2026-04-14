#include "vgre/advanced/tcp_cluster/internal/memory_sync_manager.h"
#include "vgre/advanced/tcp_cluster.h"
#include "vgre/core/runtime_engine.h"
#include "vgre/core/memory_manager.h"
#include "vgre/common/logger.h"
#include <cstring>
#include <thread>
#include <chrono>
#include <algorithm>

namespace vgre {
namespace advanced {

namespace {
  constexpr int MAX_DELTA_SYNC_RETRIES = 3;
  constexpr int INITIAL_RETRY_BACKOFF_MS = 100;
  constexpr int MAX_RETRY_BACKOFF_MS = 5000;
  
  class ExponentialBackoff {
  public:
    ExponentialBackoff(int initial_ms, int max_ms, double multiplier = 2.0)
      : initial_ms_(initial_ms), max_ms_(max_ms), multiplier_(multiplier), current_ms_(initial_ms) {}
    
    int next() {
      int delay = current_ms_;
      current_ms_ = std::min(static_cast<int>(current_ms_ * multiplier_), max_ms_);
      return delay;
    }
    
    void reset() {
      current_ms_ = initial_ms_;
    }
    
  private:
    int initial_ms_;
    int max_ms_;
    double multiplier_;
    int current_ms_;
  };
} // anonymous namespace

MemorySyncManager::MemorySyncManager(TCPClusterManager* parent)
  : parent_(parent) {}

VGREResult MemorySyncManager::streamArgumentsToWorker(void** args, int num_args, uint64_t kernel_id, 
                                                      std::shared_ptr<TCPClusterManager::ClientConnection> client) {
  if (!args || num_args <= 0 || !client) {
    return VGREResult::SUCCESS; // No arguments to stream
  }
  
  // Get argument types from RuntimeEngine
  std::vector<ArgType> argTypes;
  if (core::RuntimeEngine::instance().getKernelArgTypes(kernel_id, argTypes) != VGREResult::SUCCESS) {
    return VGREResult::ERR_INVALID_KERNEL;
  }
  
  // Stream each argument
  for (int i = 0; i < num_args; ++i) {
    ArgType type = (i < static_cast<int>(argTypes.size())) ? argTypes[i] : ArgType::UINT64;
    
    VGREResult result = VGREResult::SUCCESS;
    if (type == ArgType::STRUCT) {
      result = sendStructArg(args[i], i, kernel_id, client);
    } else if (type == ArgType::POINTER) {
      result = sendPointerArg(args[i], i, client);
    } else {
      result = sendScalarArg(args[i], i, type, client);
    }
    
    if (result != VGREResult::SUCCESS) {
      return result;
    }
  }
  
  return VGREResult::SUCCESS;
}

VGREResult MemorySyncManager::syncPointerToWorker(void* ptr, uint64_t handle, 
                                                  std::shared_ptr<TCPClusterManager::ClientConnection> client) {
  if (!ptr || !client) {
    return VGREResult::ERR_INVALID_VALUE;
  }
  
  auto& mm = core::RuntimeEngine::instance().getMemoryManager();
  size_t size = mm.getAllocationSize(ptr);
  if (size == 0) {
    return VGREResult::ERR_INVALID_VALUE;
  }
  
  // Check if pointer was previously synced
  bool useDelta = client->synced_handles.count(ptr) > 0;
  
  if (useDelta) {
    // Get dirty ranges from MemoryManager
    std::vector<std::pair<size_t, size_t>> dirtyRanges;
    mm.getDirtyPages(ptr, dirtyRanges);
    
    if (!dirtyRanges.empty()) {
      // Use delta-sync with retry and automatic fallback to full sync
      VGREResult result = sendDeltaSyncWithRetry(ptr, handle, dirtyRanges, client);
      if (result == VGREResult::SUCCESS) {
        mm.clearDirtyPages(ptr);
        return VGREResult::SUCCESS;
      }
      // sendDeltaSyncWithRetry already handles fallback to full sync
      return result;
    }
  }
  
  // Full sync (first time or no dirty ranges)
  client->synced_handles.insert(ptr);
  VGREResult result = sendFullSync(ptr, handle, size, client);
  if (result == VGREResult::SUCCESS) {
    mm.clearDirtyPages(ptr);
  }
  return result;
}

VGREResult MemorySyncManager::syncPointerFromWorker(void* ptr, uint64_t handle, size_t size,
                                                    std::shared_ptr<TCPClusterManager::ClientConnection> client) {
  // Worker-to-master memory sync (pull-back after kernel execution)
  // This is used when a worker needs to send modified memory back to the master
  
  if (!ptr || size == 0) {
    return VGREResult::ERR_INVALID_VALUE;
  }
  
  if (!client || client->socket_fd == vgre::common::VGRE_INVALID_SOCKET) {
    return VGREResult::ERR_INVALID_VALUE;
  }
  
  // Check if we can use SHM for local connections
  if (client->is_local && client->shmManager && client->shmManager->getBasePtr()) {
    // SHM path: copy data to shared memory
    uint64_t offset = client->shm_offset;
    
    // Check if there's enough space
    if (offset + size > client->shmManager->getSize()) {
      VGRE_LOG_WARN("TCPCluster", "SHM full for pull-back, falling back to TCP");
      // Fall through to TCP path
    } else {
      // Copy data to SHM
      std::memcpy(
          static_cast<uint8_t*>(client->shmManager->getBasePtr()) + offset,
          ptr, size);
      
      client->shm_offset += size;
      
      // Send SHM packet
      DataShmPacket dspkt{};
      dspkt.target_ptr = handle;
      dspkt.shm_offset = offset;
      dspkt.size = size;
      
      return parent_->send_packet(client->socket_fd, PacketType::DATA_SHM, 
                                  &dspkt, sizeof(DataShmPacket), 
                                  client->secureChannel.get());
    }
  }
  
  // TCP path: send data header + body
  DataHeaderPacket dhpkt{};
  dhpkt.target_ptr = handle;
  dhpkt.size = size;
  
  VGREResult result = parent_->send_packet(client->socket_fd, PacketType::DATA_HEADER,
                                           &dhpkt, sizeof(DataHeaderPacket),
                                           client->secureChannel.get());
  if (result != VGREResult::SUCCESS) {
    return result;
  }
  
  // Send data body
  return parent_->send_packet(client->socket_fd, PacketType::DATA_BODY,
                              ptr, size,
                              client->secureChannel.get());
}

VGREResult MemorySyncManager::initializeShmForClient(std::shared_ptr<TCPClusterManager::ClientConnection> client) {
  // Initialize shared memory for a local client connection
  
  if (!client) {
    return VGREResult::ERR_INVALID_VALUE;
  }
  
  // Only initialize SHM for local connections
  if (!client->is_local) {
    VGRE_LOG_DEBUG("TCPCluster", "Skipping SHM initialization for remote client " + client->ip_address);
    return VGREResult::SUCCESS;
  }
  
  // Check if SHM is already initialized
  if (client->shmManager && client->shmManager->getBasePtr()) {
    VGRE_LOG_DEBUG("TCPCluster", "SHM already initialized for client " + client->ip_address);
    return VGREResult::SUCCESS;
  }
  
  // Create SHM manager
  client->shmManager = std::make_unique<vgre::core::ShmManager>();
  
  // Generate unique SHM name based on socket FD
  std::string shmName = "vgre_shm_" + std::to_string(client->socket_fd);
  size_t shmSize = 256 * 1024 * 1024; // 256MB default
  
  // Open/create shared memory segment
  VGREResult result = client->shmManager->open(shmName, shmSize, true);
  if (result != VGREResult::SUCCESS) {
    VGRE_LOG_ERROR("TCPCluster", "Failed to create SHM segment for client " + client->ip_address);
    client->shmManager.reset();
    return result;
  }
  
  // Initialize SHM offset
  client->shm_offset = 0;
  
  // Send SHM initialization packet to client
  ShmInitPacket sipkt{};
  std::strncpy(sipkt.shm_name, shmName.c_str(), sizeof(sipkt.shm_name) - 1);
  sipkt.shm_name[sizeof(sipkt.shm_name) - 1] = '\0';
  sipkt.shm_size = shmSize;
  
  result = parent_->send_packet(client->socket_fd, PacketType::SHM_INIT,
                                &sipkt, sizeof(ShmInitPacket),
                                client->secureChannel.get());
  
  if (result != VGREResult::SUCCESS) {
    VGRE_LOG_ERROR("TCPCluster", "Failed to send SHM_INIT packet to client " + client->ip_address);
    client->shmManager.reset();
    return result;
  }
  
  VGRE_LOG_INFO("TCPCluster", "SHM initialized for local client " + client->ip_address + 
                " (name: " + shmName + ", size: " + std::to_string(shmSize) + " bytes)");
  
  return VGREResult::SUCCESS;
}

VGREResult MemorySyncManager::sendDeltaSync(void* ptr, uint64_t handle, 
                                            const std::vector<std::pair<size_t, size_t>>& dirtyRanges,
                                            std::shared_ptr<TCPClusterManager::ClientConnection> client) {
  if (client->is_local && client->shmManager) {
    return sendDeltaSyncSHM(ptr, handle, dirtyRanges, client);
  } else {
    return sendDeltaSyncTCP(ptr, handle, dirtyRanges, client);
  }
}

VGREResult MemorySyncManager::sendDeltaSyncWithRetry(void* ptr, uint64_t handle,
                                                     const std::vector<std::pair<size_t, size_t>>& dirtyRanges,
                                                     std::shared_ptr<TCPClusterManager::ClientConnection> client) {
  ExponentialBackoff backoff(INITIAL_RETRY_BACKOFF_MS, MAX_RETRY_BACKOFF_MS);
  
  for (int attempt = 0; attempt < MAX_DELTA_SYNC_RETRIES; ++attempt) {
    VGREResult result = sendDeltaSync(ptr, handle, dirtyRanges, client);
    
    if (result == VGREResult::SUCCESS) {
      return VGREResult::SUCCESS;
    }
    
    // If connection lost, don't retry
    if (result == VGREResult::ERR_IO) {
      VGRE_LOG_ERROR("TCPCluster", "Delta-sync failed due to I/O error (connection lost), not retrying");
      return result;
    }
    
    // For transient failures, sleep with exponential backoff and retry
    if (attempt < MAX_DELTA_SYNC_RETRIES - 1) {
      int delay_ms = backoff.next();
      VGRE_LOG_DEBUG("TCPCluster", "Delta-sync attempt " + std::to_string(attempt + 1) + 
                     " failed, retrying after " + std::to_string(delay_ms) + "ms");
      std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
    }
  }
  
  // All retries exhausted, fall back to full sync
  VGRE_LOG_WARN("TCPCluster", "Delta-sync failed after " + std::to_string(MAX_DELTA_SYNC_RETRIES) + 
                " attempts, falling back to full sync");
  size_t size = core::RuntimeEngine::instance().getMemoryManager().getAllocationSize(ptr);
  return sendFullSync(ptr, handle, size, client);
}

VGREResult MemorySyncManager::sendDeltaSyncSHM(void* ptr, uint64_t handle,
                                               const std::vector<std::pair<size_t, size_t>>& dirtyRanges,
                                               std::shared_ptr<TCPClusterManager::ClientConnection> client) {
  // Calculate total size of dirty ranges
  uint64_t totalSize = 0;
  for (const auto& range : dirtyRanges) {
    totalSize += range.second;
  }
  
  // Check SHM space availability
  uint64_t baseOffset = client->shm_offset;
  if (baseOffset + totalSize > client->shmManager->getSize()) {
    // Not enough SHM space, fall back to full sync
    size_t size = core::RuntimeEngine::instance().getMemoryManager().getAllocationSize(ptr);
    return sendFullSync(ptr, handle, size, client);
  }
  
  // Update SHM offset
  client->shm_offset += totalSize;
  
  // Send DataShmDirtyPacket
  DataShmDirtyPacket dspkt{};
  dspkt.target_ptr = handle;
  dspkt.num_ranges = static_cast<uint32_t>(dirtyRanges.size());
  dspkt.shm_offset = baseOffset;
  
  if (parent_->send_packet(client->socket_fd, PacketType::DATA_SHM_DIRTY, &dspkt, sizeof(DataShmDirtyPacket), client->secureChannel.get()) != VGREResult::SUCCESS) {
    return VGREResult::ERR_IO;
  }
  
  // Send each dirty range
  uint64_t currentOffset = baseOffset;
  for (const auto& range : dirtyRanges) {
    DirtyRangePacket rpkt{range.first, range.second};
    if (parent_->send_packet(client->socket_fd, PacketType::DIRTY_RANGE, &rpkt, sizeof(DirtyRangePacket), client->secureChannel.get()) != VGREResult::SUCCESS) {
      return VGREResult::ERR_IO;
    }
    
    // Copy data to SHM
    std::memcpy(static_cast<uint8_t*>(client->shmManager->getBasePtr()) + currentOffset,
                static_cast<uint8_t*>(ptr) + range.first, range.second);
    currentOffset += range.second;
  }
  
  return VGREResult::SUCCESS;
}

VGREResult MemorySyncManager::sendDeltaSyncTCP(void* ptr, uint64_t handle,
                                               const std::vector<std::pair<size_t, size_t>>& dirtyRanges,
                                               std::shared_ptr<TCPClusterManager::ClientConnection> client) {
  // Send DataHeaderDirtyPacket
  DataHeaderDirtyPacket dhpkt{};
  dhpkt.target_ptr = handle;
  dhpkt.num_ranges = static_cast<uint32_t>(dirtyRanges.size());
  
  if (parent_->send_packet(client->socket_fd, PacketType::DATA_HEADER_DIRTY, &dhpkt, sizeof(DataHeaderDirtyPacket), client->secureChannel.get()) != VGREResult::SUCCESS) {
    return VGREResult::ERR_IO;
  }
  
  // Send each dirty range
  for (const auto& range : dirtyRanges) {
    DirtyRangePacket rpkt{range.first, range.second};
    if (parent_->send_packet(client->socket_fd, PacketType::DIRTY_RANGE, &rpkt, sizeof(DirtyRangePacket), client->secureChannel.get()) != VGREResult::SUCCESS) {
      return VGREResult::ERR_IO;
    }
    
    // Send range data
    if (parent_->send_packet(client->socket_fd, PacketType::DATA_BODY, static_cast<uint8_t*>(ptr) + range.first, range.second, client->secureChannel.get()) != VGREResult::SUCCESS) {
      return VGREResult::ERR_IO;
    }
  }
  
  return VGREResult::SUCCESS;
}

VGREResult MemorySyncManager::sendFullSync(void* ptr, uint64_t handle, size_t size,
                                           std::shared_ptr<TCPClusterManager::ClientConnection> client) {
  if (client->is_local && client->shmManager) {
    return sendFullSyncSHM(ptr, handle, size, client);
  } else {
    return sendFullSyncTCP(ptr, handle, size, client);
  }
}

VGREResult MemorySyncManager::sendFullSyncSHM(void* ptr, uint64_t handle, size_t size,
                                              std::shared_ptr<TCPClusterManager::ClientConnection> client) {
  // Check SHM space availability
  uint64_t offset = client->shm_offset;
  if (offset + size > client->shmManager->getSize()) {
    // Not enough SHM space, fall back to TCP
    return sendFullSyncTCP(ptr, handle, size, client);
  }
  
  // Update SHM offset
  client->shm_offset += size;
  
  // Copy data to SHM
  std::memcpy(static_cast<uint8_t*>(client->shmManager->getBasePtr()) + offset, ptr, size);
  
  // Send DataShmPacket
  DataShmPacket dspkt{};
  dspkt.target_ptr = handle;
  dspkt.shm_offset = offset;
  dspkt.size = size;
  
  if (parent_->send_packet(client->socket_fd, PacketType::DATA_SHM, &dspkt, sizeof(DataShmPacket), client->secureChannel.get()) != VGREResult::SUCCESS) {
    return VGREResult::ERR_IO;
  }
  
  return VGREResult::SUCCESS;
}

VGREResult MemorySyncManager::sendFullSyncTCP(void* ptr, uint64_t handle, size_t size,
                                              std::shared_ptr<TCPClusterManager::ClientConnection> client) {
  // Send DataHeaderPacket
  DataHeaderPacket dpkt{};
  dpkt.target_ptr = handle;
  dpkt.size = size;
  
  if (parent_->send_packet(client->socket_fd, PacketType::DATA_HEADER, &dpkt, sizeof(DataHeaderPacket), client->secureChannel.get()) != VGREResult::SUCCESS) {
    return VGREResult::ERR_IO;
  }
  
  // Send data body
  if (parent_->send_packet(client->socket_fd, PacketType::DATA_BODY, ptr, size, client->secureChannel.get()) != VGREResult::SUCCESS) {
    return VGREResult::ERR_IO;
  }
  
  return VGREResult::SUCCESS;
}

VGREResult MemorySyncManager::sendScalarArg(void* arg, int arg_index, ArgType type,
                                            std::shared_ptr<TCPClusterManager::ClientConnection> client) {
  // Determine argument size based on type
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
  
  // Copy value to ArgScalarPacket
  ArgScalarPacket apkt{};
  apkt.arg_index = static_cast<uint32_t>(arg_index);
  apkt.arg_type = static_cast<uint8_t>(type);
  std::memset(&apkt.value, 0, 8);
  std::memcpy(&apkt.value, arg, arg_size);
  
  if (parent_->send_packet(client->socket_fd, PacketType::ARG_SCALAR, &apkt, sizeof(ArgScalarPacket), client->secureChannel.get()) != VGREResult::SUCCESS) {
    return VGREResult::ERR_IO;
  }
  
  return VGREResult::SUCCESS;
}

VGREResult MemorySyncManager::sendPointerArg(void* arg, int arg_index,
                                             std::shared_ptr<TCPClusterManager::ClientConnection> client) {
  // Dereference pointer to get actual address
  void* ptr = *static_cast<void**>(arg);
  uint64_t handle = reinterpret_cast<uint64_t>(ptr);
  
  // Sync memory if it's a valid managed pointer
  size_t size = core::RuntimeEngine::instance().getMemoryManager().getAllocationSize(ptr);
  if (size > 0 && ptr) {
    VGREResult syncResult = syncPointerToWorker(ptr, handle, client);
    if (syncResult != VGREResult::SUCCESS) {
      return syncResult;
    }
  }
  
  // Send ArgScalarPacket with pointer handle
  ArgScalarPacket apkt{};
  apkt.arg_index = static_cast<uint32_t>(arg_index);
  apkt.arg_type = static_cast<uint8_t>(ArgType::POINTER);
  apkt.value = handle;
  
  if (parent_->send_packet(client->socket_fd, PacketType::ARG_POINTER, &apkt, sizeof(ArgScalarPacket), client->secureChannel.get()) != VGREResult::SUCCESS) {
    return VGREResult::ERR_IO;
  }
  
  return VGREResult::SUCCESS;
}

VGREResult MemorySyncManager::sendStructArg(void* arg, int arg_index, uint64_t kernel_id,
                                            std::shared_ptr<TCPClusterManager::ClientConnection> client) {
  // Get struct size from kernel IR
  const auto* ir = core::RuntimeEngine::instance().getKernelIR(kernel_id);
  if (!ir || arg_index >= static_cast<int>(ir->argSizes.size())) {
    return VGREResult::ERR_INVALID_VALUE;
  }
  
  size_t size = ir->argSizes[arg_index];
  
  // Send StructDataPacket
  StructDataPacket spkt{};
  spkt.arg_index = static_cast<uint32_t>(arg_index);
  spkt.size = static_cast<uint32_t>(size);
  
  if (parent_->send_packet(client->socket_fd, PacketType::STRUCT_DATA, &spkt, sizeof(StructDataPacket), client->secureChannel.get()) != VGREResult::SUCCESS) {
    return VGREResult::ERR_IO;
  }
  
  // Send struct contents
  if (parent_->send_packet(client->socket_fd, PacketType::DATA_BODY, arg, size, client->secureChannel.get()) != VGREResult::SUCCESS) {
    return VGREResult::ERR_IO;
  }
  
  return VGREResult::SUCCESS;
}

} // namespace advanced
} // namespace vgre
