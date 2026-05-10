#include "vgre/advanced/tcp_cluster/internal/memory_sync_manager.h"
#include "vgre/advanced/tcp_cluster.h"
#include "vgre/advanced/rdma_transport.h"
#include "vgre/core/runtime_engine.h"
#include "vgre/core/memory_manager.h"
#include "vgre/common/logger.h"
#include <cstdlib>
#include <cstring>
#include <thread>
#include <chrono>
#include <algorithm>

namespace {
// Minimum transfer size for RDMA WRITE. Transfers below this threshold are sent over TCP (RDMA setup overhead exceeds the copy cost for small buffers).
// Configurable via VGRE_RDMA_THRESHOLD_BYTES (default 65536 = 64 KB).
const size_t kRdmaThreshold = []() -> size_t {
    const char* env = std::getenv("VGRE_RDMA_THRESHOLD_BYTES");
    if (env) { try { long long v = std::stoll(env); if (v >= 4096) return static_cast<size_t>(v); } catch (...) {} }
    return 65536;  // 64 KB
}();
} // anonymous namespace

namespace vgre {
namespace advanced {

namespace {
  // Configurable via VGRE_CLUSTER_MAX_DELTA_SYNC_RETRIES (default 3).
  const int MAX_DELTA_SYNC_RETRIES = []() -> int {
      const char* env = std::getenv("VGRE_CLUSTER_MAX_DELTA_SYNC_RETRIES");
      if (env) { try { int v = std::stoi(env); if (v > 0 && v <= 100) return v; } catch (...) {} }
      return 3;
  }();
  // Configurable via VGRE_CLUSTER_RETRY_BACKOFF_INITIAL_MS (default 100 ms).
  const int INITIAL_RETRY_BACKOFF_MS = []() -> int {
      const char* env = std::getenv("VGRE_CLUSTER_RETRY_BACKOFF_INITIAL_MS");
      if (env) { try { int v = std::stoi(env); if (v > 0) return v; } catch (...) {} }
      return 100;
  }();
  // Configurable via VGRE_CLUSTER_RETRY_BACKOFF_MAX_MS (default 5000 ms).
  const int MAX_RETRY_BACKOFF_MS = []() -> int {
      const char* env = std::getenv("VGRE_CLUSTER_RETRY_BACKOFF_MAX_MS");
      if (env) { try { int v = std::stoi(env); if (v > 0) return v; } catch (...) {} }
      return 5000;
  }();
  
  class ExponentialBackoff {
  public:
    ExponentialBackoff(int initial_ms, int max_ms, double multiplier = 2.0) : initial_ms_(initial_ms), max_ms_(max_ms), multiplier_(multiplier), current_ms_(initial_ms) {}
    int next() { int delay = current_ms_; current_ms_ = std::min(static_cast<int>(current_ms_ * multiplier_), max_ms_); return delay; }
    void reset() { current_ms_ = initial_ms_; }
  private:
    int initial_ms_; int max_ms_; double multiplier_; int current_ms_;
  };
} // anonymous namespace

MemorySyncManager::MemorySyncManager(TCPClusterManager* parent)
  : parent_(parent) {}

VGREResult MemorySyncManager::streamArgumentsToWorker(void** args, int num_args, uint64_t kernel_id, std::shared_ptr<TCPClusterManager::ClientConnection> client) {
  if (!client) return VGREResult::ERR_INVALID_VALUE;
  if (!args || num_args <= 0) return VGREResult::SUCCESS;
  std::vector<ArgType> argTypes;
  if (core::RuntimeEngine::instance().getKernelArgTypes(kernel_id, argTypes) != VGREResult::SUCCESS) { return VGREResult::ERR_INVALID_KERNEL; }
  
  // All arguments must have registered type info. A silent UINT64 fallback would produce the wrong marshal layout and silently corrupt data on the worker. Reject up-front if type info is incomplete.
  if (static_cast<int>(argTypes.size()) < num_args) {
    VGRE_LOG_ERROR("TCPCluster", "streamArgumentsToWorker: kernel " + std::to_string(kernel_id) + " has " + std::to_string(argTypes.size()) + " registered arg types but " + std::to_string(num_args) + " args were supplied — register all argument types before launching remotely.");
    return VGREResult::ERR_INVALID_KERNEL;
  }

  // Stream each argument
  for (int i = 0; i < num_args; ++i) {
    ArgType type = argTypes[i];
    VGREResult result = VGREResult::SUCCESS;
    if (type == ArgType::STRUCT) { result = sendStructArg(args[i], i, kernel_id, client); }
    else if (type == ArgType::POINTER) { result = sendPointerArg(args[i], i, client); }
    else { result = sendScalarArg(args[i], i, type, client); }
    if (result != VGREResult::SUCCESS) { return result; }
  }
  return VGREResult::SUCCESS;
}

VGREResult MemorySyncManager::syncPointerToWorker(void* ptr, uint64_t handle, std::shared_ptr<TCPClusterManager::ClientConnection> client) {
  if (!ptr || !client) { return VGREResult::ERR_INVALID_VALUE; }
  
  auto& mm = core::RuntimeEngine::instance().getMemoryManager();
  size_t size = mm.getAllocationSize(ptr);
  if (size == 0) { return VGREResult::ERR_INVALID_VALUE; }
  
  // Check if pointer was previously synced
  bool useDelta = client->synced_handles.count(ptr) > 0;
  
  if (useDelta) {
    // Get dirty ranges from MemoryManager
    std::vector<std::pair<size_t, size_t>> dirtyRanges;
    mm.getDirtyPages(ptr, dirtyRanges);
    
    if (!dirtyRanges.empty()) {
      // Use delta-sync with retry and automatic fallback to full sync
      VGREResult result = sendDeltaSyncWithRetry(ptr, handle, dirtyRanges, client);
      if (result == VGREResult::SUCCESS) { mm.clearDirtyPages(ptr); return VGREResult::SUCCESS; }
      // sendDeltaSyncWithRetry already handles fallback to full sync
      return result;
    }
  }
  
  // Full sync (first time or no dirty ranges)
  client->synced_handles.insert(ptr);
  VGREResult result = sendFullSync(ptr, handle, size, client);
  if (result == VGREResult::SUCCESS) { mm.clearDirtyPages(ptr); }
  return result;
}

VGREResult MemorySyncManager::syncPointerFromWorker(void* ptr, uint64_t handle, size_t size, std::shared_ptr<TCPClusterManager::ClientConnection> client) {
  // Worker-to-master memory sync (pull-back after kernel execution)
  if (!ptr || size == 0) { return VGREResult::ERR_INVALID_VALUE; }
  if (!client || client->socket_fd == vgre::common::VGRE_INVALID_SOCKET) { return VGREResult::ERR_INVALID_VALUE; }
  
  // Check if we can use SHM for local connections
  if (client->is_local && client->shm_manager && client->shm_manager->getBasePtr()) {
    // SHM path: copy data to shared memory
    uint64_t offset = client->shm_offset;
    
    // Check if there's enough space
    if (offset + size > client->shm_manager->getSize()) {
      VGRE_LOG_WARN("TCPCluster", "SHM full for pull-back, falling back to TCP");
      // Fall through to TCP path
    } else {
      // Copy data to SHM
      std::memcpy(static_cast<uint8_t*>(client->shm_manager->getBasePtr()) + offset, ptr, size);
      client->shm_offset += size;
      
      // Send SHM packet
      DataShmPacket dspkt{}; dspkt.target_ptr = handle; dspkt.shm_offset = offset; dspkt.size = size;
      return parent_->send_packet(client->socket_fd, PacketType::DATA_SHM, &dspkt, sizeof(DataShmPacket), client->secure_channel.get());
    }
  }
  
  // TCP path: send data header + body
  DataHeaderPacket dhpkt{}; dhpkt.target_ptr = handle; dhpkt.size = size;
  VGREResult result = parent_->send_packet(client->socket_fd, PacketType::DATA_HEADER, &dhpkt, sizeof(DataHeaderPacket), client->secure_channel.get());
  if (result != VGREResult::SUCCESS) { return result; }
  
  // Send data body
  return parent_->send_packet(client->socket_fd, PacketType::DATA_BODY, ptr, size, client->secure_channel.get());
}

VGREResult MemorySyncManager::initializeShmForClient(std::shared_ptr<TCPClusterManager::ClientConnection> client) {
  // Initialize shared memory for a local client connection
  if (!client) { return VGREResult::ERR_INVALID_VALUE; }
  
  // Only initialize SHM for local connections
  if (!client->is_local) {
    VGRE_LOG_DEBUG("TCPCluster", "Skipping SHM initialization for remote client " + client->ip_address);
    return VGREResult::SUCCESS;
  }
  
  // Check if SHM is already initialized
  if (client->shm_manager && client->shm_manager->getBasePtr()) {
    VGRE_LOG_DEBUG("TCPCluster", "SHM already initialized for client " + client->ip_address);
    return VGREResult::SUCCESS;
  }
  
  // Create SHM manager
  client->shm_manager = std::make_unique<vgre::core::ShmManager>();

  // Generate unique SHM name using a process-wide counter + socket FD to avoid collisions when the OS recycles socket descriptors across connections.
  static std::atomic<uint64_t> s_shmCounter{0};
  std::string shmName = "vgre_shm_" + std::to_string(++s_shmCounter) + "_" + std::to_string(client->socket_fd);
  // Configurable via VGRE_CLUSTER_SHM_SIZE (bytes, default 256 MB).
  // Mirrors the server_loop.cpp constant so both sides negotiate the same size.
  static const size_t kShmSize = []() -> size_t {
    const char* env = std::getenv("VGRE_CLUSTER_SHM_SIZE");
    if (env) {
      try { long long v = std::stoll(env); if (v > 0) return static_cast<size_t>(v); }
      catch (...) {}
    }
    return 256ULL * 1024 * 1024;
  }();
  size_t shmSize = kShmSize;
  
  // Open/create shared memory segment
  VGREResult result = client->shm_manager->open(shmName, shmSize, true);
  if (result != VGREResult::SUCCESS) {
    VGRE_LOG_ERROR("TCPCluster", "Failed to create SHM segment for client " + client->ip_address);
    client->shm_manager.reset(); return result;
  }
  
  // Initialize SHM offset
  client->shm_offset = 0;
  
  // Send SHM initialization packet to client
  ShmInitPacket sipkt{};
  std::strncpy(sipkt.shm_name, shmName.c_str(), sizeof(sipkt.shm_name) - 1);
  sipkt.shm_name[sizeof(sipkt.shm_name) - 1] = '\0'; sipkt.shm_size = shmSize;
  
  result = parent_->send_packet(client->socket_fd, PacketType::SHM_INIT, &sipkt, sizeof(ShmInitPacket), client->secure_channel.get());
  if (result != VGREResult::SUCCESS) {
    VGRE_LOG_ERROR("TCPCluster", "Failed to send SHM_INIT packet to client " + client->ip_address);
    client->shm_manager.reset(); return result;
  }
  
  VGRE_LOG_INFO("TCPCluster", "SHM initialized for local client " + client->ip_address + " (name: " + shmName + ", size: " + std::to_string(shmSize) + " bytes)");
  return VGREResult::SUCCESS;
}

VGREResult MemorySyncManager::sendDeltaSync(void* ptr, uint64_t handle,
                                            const std::vector<std::pair<size_t, size_t>>& dirtyRanges,
                                            std::shared_ptr<TCPClusterManager::ClientConnection> client) {
  if (client->is_local && client->shm_manager) {
    return sendDeltaSyncSHM(ptr, handle, dirtyRanges, client);
  } else {
    return sendDeltaSyncTCP(ptr, handle, dirtyRanges, client);
  }
}

VGREResult MemorySyncManager::sendDeltaSyncWithRetry(void* ptr, uint64_t handle, const std::vector<std::pair<size_t, size_t>>& dirtyRanges, std::shared_ptr<TCPClusterManager::ClientConnection> client) {
  ExponentialBackoff backoff(INITIAL_RETRY_BACKOFF_MS, MAX_RETRY_BACKOFF_MS);
  
  for (int attempt = 0; attempt < MAX_DELTA_SYNC_RETRIES; ++attempt) {
    VGREResult result = sendDeltaSync(ptr, handle, dirtyRanges, client);
    if (result == VGREResult::SUCCESS) { return VGREResult::SUCCESS; }
    
    // If connection lost, don't retry
    if (result == VGREResult::ERR_IO) {
      VGRE_LOG_ERROR("TCPCluster", "Delta-sync failed due to I/O error (connection lost), not retrying");
      return result;
    }
    
    // For transient failures, sleep with exponential backoff and retry
    if (attempt < MAX_DELTA_SYNC_RETRIES - 1) {
      int delay_ms = backoff.next();
      VGRE_LOG_DEBUG("TCPCluster", "Delta-sync attempt " + std::to_string(attempt + 1) + " failed, retrying after " + std::to_string(delay_ms) + "ms");
      std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
    }
  }
  
  // All retries exhausted, fall back to full sync
  VGRE_LOG_WARN("TCPCluster", "Delta-sync failed after " + std::to_string(MAX_DELTA_SYNC_RETRIES) + " attempts, falling back to full sync");
  size_t size = core::RuntimeEngine::instance().getMemoryManager().getAllocationSize(ptr);
  return sendFullSync(ptr, handle, size, client);
}

VGREResult MemorySyncManager::sendDeltaSyncSHM(void* ptr, uint64_t handle, const std::vector<std::pair<size_t, size_t>>& dirtyRanges, std::shared_ptr<TCPClusterManager::ClientConnection> client) {
  if (!client->shm_manager || !client->shm_manager->getBasePtr()) { return sendDeltaSyncTCP(ptr, handle, dirtyRanges, client); }
  size_t allocSize = core::RuntimeEngine::instance().getMemoryManager().getAllocationSize(ptr);
  for (const auto& range : dirtyRanges) {
    // Use >= so that range.first == allocSize (past-the-end start) is also rejected.
    if (range.first >= allocSize || range.second > allocSize - range.first)
      return VGREResult::ERR_INVALID_VALUE;
  }
  uint64_t totalSize = 0;
  for (const auto& range : dirtyRanges) { totalSize += range.second; }

  uint64_t baseOffset = client->shm_offset;
  if (baseOffset + totalSize > client->shm_manager->getSize()) {
    size_t size = core::RuntimeEngine::instance().getMemoryManager().getAllocationSize(ptr);
    return sendFullSync(ptr, handle, size, client);
  }
  client->shm_offset += totalSize;
  DataShmDirtyPacket dspkt{}; dspkt.target_ptr = handle; dspkt.num_ranges = static_cast<uint32_t>(dirtyRanges.size()); dspkt.shm_offset = baseOffset;
  
  if (parent_->send_packet(client->socket_fd, PacketType::DATA_SHM_DIRTY, &dspkt, sizeof(DataShmDirtyPacket), client->secure_channel.get()) != VGREResult::SUCCESS) {
    return VGREResult::ERR_IO;
  }
  uint64_t currentOffset = baseOffset;
  for (const auto& range : dirtyRanges) {
    DirtyRangePacket rpkt{range.first, range.second};
    if (parent_->send_packet(client->socket_fd, PacketType::DIRTY_RANGE, &rpkt, sizeof(DirtyRangePacket), client->secure_channel.get()) != VGREResult::SUCCESS) {
      return VGREResult::ERR_IO;
    }
    
    // Copy data to SHM
    std::memcpy(static_cast<uint8_t*>(client->shm_manager->getBasePtr()) + currentOffset, static_cast<uint8_t*>(ptr) + range.first, range.second);
    currentOffset += range.second;
  }
  return VGREResult::SUCCESS;
}

VGREResult MemorySyncManager::sendDeltaSyncTCP(void* ptr, uint64_t handle, const std::vector<std::pair<size_t, size_t>>& dirtyRanges, std::shared_ptr<TCPClusterManager::ClientConnection> client) {
  // Send DataHeaderDirtyPacket
  DataHeaderDirtyPacket dhpkt{}; dhpkt.target_ptr = handle; dhpkt.num_ranges = static_cast<uint32_t>(dirtyRanges.size());

  if (parent_->send_packet(client->socket_fd, PacketType::DATA_HEADER_DIRTY, &dhpkt, sizeof(DataHeaderDirtyPacket), client->secure_channel.get()) != VGREResult::SUCCESS) {
    return VGREResult::ERR_IO;
  }

  // Send each dirty range — use RDMA WRITE for ranges above the threshold
  for (const auto& range : dirtyRanges) {
    DirtyRangePacket rpkt{range.first, range.second};
    if (parent_->send_packet(client->socket_fd, PacketType::DIRTY_RANGE, &rpkt, sizeof(DirtyRangePacket), client->secure_channel.get()) != VGREResult::SUCCESS) {
      return VGREResult::ERR_IO;
    }

    uint8_t* rangeData = static_cast<uint8_t*>(ptr) + range.first;
    const size_t rangeSize = range.second;

    // ── RDMA fast path for large dirty ranges ──────────────────────────────
    if (client->rdma_connected && client->rdma_conn && client->rdma_ctx && rangeSize >= kRdmaThreshold) {
      if (client->rdma_conn->rdmaWriteToRemote(*client->rdma_ctx, rangeData, rangeSize)) {
        DataHeaderRDMAPacket rdmaPkt{}; rdmaPkt.target_ptr = handle; rdmaPkt.size = rangeSize;
        if (parent_->send_packet(client->socket_fd, PacketType::DATA_HEADER_RDMA, &rdmaPkt, sizeof(DataHeaderRDMAPacket), client->secure_channel.get()) == VGREResult::SUCCESS) {
          VGRE_LOG_DEBUG("TCPCluster", "RDMA delta range: " + std::to_string(rangeSize) + " bytes → " + client->ip_address);
          continue;  // skip the TCP DATA_BODY send for this range
        }
        VGRE_LOG_WARN("TCPCluster", "RDMA WRITE ok but DATA_HEADER_RDMA failed for delta range — TCP fallback");
      }
    }

    // TCP fallback for this range
    if (parent_->send_packet(client->socket_fd, PacketType::DATA_BODY, rangeData, rangeSize, client->secure_channel.get()) != VGREResult::SUCCESS) {
      return VGREResult::ERR_IO;
    }
  }
  return VGREResult::SUCCESS;
}

VGREResult MemorySyncManager::sendFullSync(void* ptr, uint64_t handle, size_t size,
                                           std::shared_ptr<TCPClusterManager::ClientConnection> client) {
  if (client->is_local && client->shm_manager) {
    return sendFullSyncSHM(ptr, handle, size, client);
  } else {
    return sendFullSyncTCP(ptr, handle, size, client);
  }
}

VGREResult MemorySyncManager::sendFullSyncSHM(void* ptr, uint64_t handle, size_t size, std::shared_ptr<TCPClusterManager::ClientConnection> client) {
  if (!client->shm_manager || !client->shm_manager->getBasePtr()) { return sendFullSyncTCP(ptr, handle, size, client); }
  // Check SHM space availability
  uint64_t offset = client->shm_offset;
  if (offset + size > client->shm_manager->getSize()) {
    // Not enough SHM space, fall back to TCP
    return sendFullSyncTCP(ptr, handle, size, client);
  }
  
  // Update SHM offset
  client->shm_offset += size;
  
  // Copy data to SHM
  std::memcpy(static_cast<uint8_t*>(client->shm_manager->getBasePtr()) + offset, ptr, size);
  
  // Send DataShmPacket
  DataShmPacket dspkt{}; dspkt.target_ptr = handle; dspkt.shm_offset = offset; dspkt.size = size;
  if (parent_->send_packet(client->socket_fd, PacketType::DATA_SHM, &dspkt, sizeof(DataShmPacket), client->secure_channel.get()) != VGREResult::SUCCESS) {
    return VGREResult::ERR_IO;
  }
  return VGREResult::SUCCESS;
}

VGREResult MemorySyncManager::sendFullSyncTCP(void* ptr, uint64_t handle, size_t size, std::shared_ptr<TCPClusterManager::ClientConnection> client) {
  // ── RDMA fast path: zero-copy WRITE for large transfers ─────────────────
  // When RDMA is active and the payload exceeds the threshold, write directly into the worker's pre-registered bounce buffer, then send a slim DATA_HEADER_RDMA notification over TCP so the worker copies from its bounce buffer to the actual allocation. No DATA_BODY packet is sent.
  if (client->rdma_connected && client->rdma_conn && client->rdma_ctx && size >= kRdmaThreshold) {
    if (client->rdma_conn->rdmaWriteToRemote(*client->rdma_ctx, ptr, size)) {
      DataHeaderRDMAPacket rdmaPkt{}; rdmaPkt.target_ptr = handle; rdmaPkt.size = size;
      if (parent_->send_packet(client->socket_fd, PacketType::DATA_HEADER_RDMA, &rdmaPkt, sizeof(DataHeaderRDMAPacket), client->secure_channel.get()) == VGREResult::SUCCESS) {
        VGRE_LOG_DEBUG("TCPCluster", "RDMA WRITE: " + std::to_string(size) + " bytes → " + client->ip_address);
        return VGREResult::SUCCESS;
      }
      // TCP notification failed — fall through to the TCP data path.
      VGRE_LOG_WARN("TCPCluster", "RDMA WRITE succeeded but DATA_HEADER_RDMA send failed for " + client->ip_address + " — retrying via TCP");
    } else {
      VGRE_LOG_WARN("TCPCluster", "RDMA WRITE failed for " + client->ip_address + " (" + std::to_string(size) + " bytes) — falling back to TCP");
    }
  }

  // ── TCP path ─────────────────────────────────────────────────────────────
  DataHeaderPacket dpkt{}; dpkt.target_ptr = handle; dpkt.size = size;

  if (parent_->send_packet(client->socket_fd, PacketType::DATA_HEADER, &dpkt, sizeof(DataHeaderPacket), client->secure_channel.get()) != VGREResult::SUCCESS) {
    return VGREResult::ERR_IO;
  }

  if (parent_->send_packet(client->socket_fd, PacketType::DATA_BODY, ptr, size, client->secure_channel.get()) != VGREResult::SUCCESS) {
    return VGREResult::ERR_IO;
  }
  return VGREResult::SUCCESS;
}

VGREResult MemorySyncManager::sendScalarArg(void* arg, int arg_index, ArgType type, std::shared_ptr<TCPClusterManager::ClientConnection> client) {
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
  ArgScalarPacket apkt{}; apkt.arg_index = static_cast<uint32_t>(arg_index); apkt.arg_type = static_cast<uint8_t>(type);
  std::memset(&apkt.value, 0, 8); std::memcpy(&apkt.value, arg, arg_size);
  
  if (parent_->send_packet(client->socket_fd, PacketType::ARG_SCALAR, &apkt, sizeof(ArgScalarPacket), client->secure_channel.get()) != VGREResult::SUCCESS) {
    return VGREResult::ERR_IO;
  }
  return VGREResult::SUCCESS;
}

VGREResult MemorySyncManager::sendPointerArg(void* arg, int arg_index, std::shared_ptr<TCPClusterManager::ClientConnection> client) {
  if (!arg) {
    // Null arg slot — send a null pointer handle so the worker allocates nothing.
    ArgScalarPacket apkt{}; apkt.arg_index = static_cast<uint32_t>(arg_index); apkt.arg_type = static_cast<uint8_t>(ArgType::POINTER); apkt.value = 0;
    parent_->send_packet(client->socket_fd, PacketType::ARG_POINTER, &apkt, sizeof(ArgScalarPacket), client->secure_channel.get());
    return VGREResult::SUCCESS;
  }

  // Dereference to get the actual host pointer stored at this argument slot.
  void* ptr = *static_cast<void**>(arg);
  uint64_t handle = reinterpret_cast<uint64_t>(ptr);

  if (!ptr) {
    // Explicitly-null pointer: send handle=0 so worker passes null to kernel.
    ArgScalarPacket apkt{}; apkt.arg_index = static_cast<uint32_t>(arg_index); apkt.arg_type = static_cast<uint8_t>(ArgType::POINTER); apkt.value = 0;
    parent_->send_packet(client->socket_fd, PacketType::ARG_POINTER, &apkt, sizeof(ArgScalarPacket), client->secure_channel.get());
    return VGREResult::SUCCESS;
  }

  // Get allocation size.  If the pointer is not managed by VGRE (e.g., a user-space static buffer), size will be 0 — log a warning so users can register the memory correctly, but still send the handle so the remote side can at least attempt the launch.
  auto& mm = core::RuntimeEngine::instance().getMemoryManager();
  size_t size = mm.getAllocationSize(ptr);
  if (size == 0) {
    VGRE_LOG_WARN("TCPCluster", "sendPointerArg[" + std::to_string(arg_index) + "]: pointer " + std::to_string(handle) + " has size=0 — not a VGRE-managed allocation. Use vgre_malloc/cudaMalloc to ensure the buffer is tracked.");
    // Still send the pointer handle without data — worker will try to use whatever local allocation it has at this address (unlikely to work, but we must not silently discard the arg as that corrupts the argument count protocol).
  } else {
    VGREResult syncResult = syncPointerToWorker(ptr, handle, client);
    if (syncResult != VGREResult::SUCCESS) return syncResult;
  }

  // Send ARG_POINTER packet with the handle so the worker maps the data.
  ArgScalarPacket apkt{}; apkt.arg_index = static_cast<uint32_t>(arg_index); apkt.arg_type = static_cast<uint8_t>(ArgType::POINTER); apkt.value = handle;

  if (parent_->send_packet(client->socket_fd, PacketType::ARG_POINTER, &apkt, sizeof(ArgScalarPacket), client->secure_channel.get()) != VGREResult::SUCCESS) {
    return VGREResult::ERR_IO;
  }
  return VGREResult::SUCCESS;
}

VGREResult MemorySyncManager::sendStructArg(void* arg, int arg_index, uint64_t kernel_id, std::shared_ptr<TCPClusterManager::ClientConnection> client) {
  // Get struct size from kernel IR
  const auto* ir = core::RuntimeEngine::instance().getKernelIR(kernel_id);
  if (!ir || arg_index >= static_cast<int>(ir->argSizes.size())) { return VGREResult::ERR_INVALID_VALUE; }
  
  size_t size = ir->argSizes[arg_index];
  
  // Send StructDataPacket
  StructDataPacket spkt{}; spkt.arg_index = static_cast<uint32_t>(arg_index); spkt.size = static_cast<uint32_t>(size);
  
  if (parent_->send_packet(client->socket_fd, PacketType::STRUCT_DATA, &spkt, sizeof(StructDataPacket), client->secure_channel.get()) != VGREResult::SUCCESS) {
    return VGREResult::ERR_IO;
  }
  
  // Send struct contents
  if (parent_->send_packet(client->socket_fd, PacketType::DATA_BODY, arg, size, client->secure_channel.get()) != VGREResult::SUCCESS) {
    return VGREResult::ERR_IO;
  }
  return VGREResult::SUCCESS;
}

} // namespace advanced
} // namespace vgre
