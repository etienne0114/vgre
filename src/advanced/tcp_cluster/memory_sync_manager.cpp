#include "vgre/advanced/tcp_cluster/internal/memory_sync_manager.h"
#include "vgre/advanced/rdma_transport.h"
#include "vgre/advanced/tcp_cluster.h"
#include "vgre/advanced/tcp_cluster/internal/interfaces.h"
#include "vgre/api/vgre_c_api.h"
#include "vgre/common/logger.h"
#include "vgre/core/memory_manager.h"
#include "vgre/core/runtime_engine.h"
#include <algorithm>
#include <chrono>
#include <random>
#include <thread>

namespace {

// ── Configuration Constants ──────────────────────────────────────────────────
// Use function-local statics (initialized on first call, NOT at DLL/exe load
// time) to avoid static-initialization-order issues on Windows.

static size_t kRdmaThreshold_get() {
  static const size_t v = []() -> size_t {
    const char* env = vgre_get_config("VGRE_RDMA_THRESHOLD_BYTES");
    if (env) { try { return static_cast<size_t>(std::stoll(env)); } catch (...) {} }
    return 65536; // 64 KB default
  }();
  return v;
}
#define kRdmaThreshold (kRdmaThreshold_get())

static int MAX_DELTA_SYNC_RETRIES_get() {
  static const int v = []() -> int {
    const char* env = vgre_get_config("VGRE_CLUSTER_MAX_DELTA_SYNC_RETRIES");
    if (env) { try { int x = std::stoi(env); if (x > 0 && x <= 100) return x; } catch (...) {} }
    return 3;
  }();
  return v;
}
#define MAX_DELTA_SYNC_RETRIES (MAX_DELTA_SYNC_RETRIES_get())

static int INITIAL_RETRY_BACKOFF_MS_get() {
  static const int v = []() -> int {
    const char* env = vgre_get_config("VGRE_CLUSTER_RETRY_BACKOFF_INITIAL_MS");
    if (env) { try { int x = std::stoi(env); if (x > 0) return x; } catch (...) {} }
    return 100;
  }();
  return v;
}
#define INITIAL_RETRY_BACKOFF_MS (INITIAL_RETRY_BACKOFF_MS_get())

static int MAX_RETRY_BACKOFF_MS_get() {
  static const int v = []() -> int {
    const char* env = vgre_get_config("VGRE_CLUSTER_RETRY_BACKOFF_MAX_MS");
    if (env) { try { int x = std::stoi(env); if (x > 0) return x; } catch (...) {} }
    return 30000; // 30 s (AWS decorrelated jitter cap)
  }();
  return v;
}
#define MAX_RETRY_BACKOFF_MS (MAX_RETRY_BACKOFF_MS_get())

// ── Decorrelated Jitter Backoff (AWS formula) ────────────────────────────────
//
// Formula: delay_n = min(cap, uniform(base, prev_{n-1} * 3))
// Invariant: E[delay_n] = min(cap, (base + min(cap, prev*3)) / 2)
//             ≈ min(cap, 1.5 * prev_{n-1})  when base << cap
//
// Per-connection instance — each sendDeltaSyncWithRetry call constructs one.

class DecorrelatedJitterBackoff {
public:
  explicit DecorrelatedJitterBackoff(int initial_ms, int max_ms)
      : kBaseMs_(static_cast<int64_t>(initial_ms)),
        kCapMs_(static_cast<int64_t>(max_ms)),
        prev_ms_(static_cast<int64_t>(initial_ms)),
        rng_(std::random_device{}()) {}

  int next() {
    // Overflow guard: prev*3 can overflow int32_t for large prev; use int64_t
    // arithmetic. If prev > cap/3 clamp upper to cap directly.
    int64_t upper = (prev_ms_ > kCapMs_ / 3) ? kCapMs_ : prev_ms_ * 3;
    if (upper < kBaseMs_) upper = kBaseMs_;

    // AWS decorrelated jitter: E[delay_n] = min(cap, 1.5*prev_{n-1})
    std::uniform_int_distribution<int64_t> dist(kBaseMs_, upper);
    int64_t delay = dist(rng_);

    prev_ms_ = delay; // update state for next call
    return static_cast<int>(delay);
  }

private:
  int64_t         kBaseMs_;
  int64_t         kCapMs_;
  int64_t         prev_ms_;
  std::mt19937_64 rng_;
};

} // anonymous namespace

namespace vgre {
namespace advanced {

MemorySyncManager::MemorySyncManager(TCPClusterManager *parent) 
    : parent_(parent) {}

// ── Pointer Synchronization ──────────────────────────────────────────────────

VGREResult MemorySyncManager::syncPointerToWorker(
    void *ptr, uint64_t handle, 
    std::shared_ptr<TCPClusterManager::ClientConnection> client) {
  if (!ptr || !client) {
    return VGREResult::ERR_INVALID_VALUE;
  }

  auto &mm = core::RuntimeEngine::instance().getMemoryManager();
  size_t size = mm.getAllocationSize(ptr);
  if (size == 0) {
    VGRE_LOG_WARN("TCPCluster", "Attempted to sync unmanaged pointer: " + std::to_string(handle));
    return VGREResult::ERR_INVALID_VALUE;
  }

  // Check if pointer was previously synced (delta sync available)
  if (client->synced_handles.count(ptr) > 0) {
    std::vector<std::pair<size_t, size_t>> dirtyRanges;
    mm.getDirtyPages(ptr, dirtyRanges);

    if (!dirtyRanges.empty()) {
      VGRE_LOG_DEBUG("TCPCluster", "Delta syncing " + std::to_string(dirtyRanges.size()) + " ranges for pointer " + std::to_string(handle));
      VGREResult r = sendDeltaSyncWithRetry(ptr, handle, dirtyRanges, client);
      if (r == VGREResult::SUCCESS) {
        mm.clearDirtyPages(ptr);
        return VGREResult::SUCCESS;
      }
      VGRE_LOG_WARN("TCPCluster", "Delta sync failed, falling back to full sync for pointer " + std::to_string(handle));
    }
  }

  // Full sync (first time or delta sync fallback)
  client->synced_handles.insert(ptr);
  VGREResult r = sendFullSync(ptr, handle, size, client);
  if (r == VGREResult::SUCCESS) {
    mm.clearDirtyPages(ptr);
  }
  return r;
}

VGREResult MemorySyncManager::syncPointerFromWorker(
    void *ptr, uint64_t handle, size_t size, 
    std::shared_ptr<TCPClusterManager::ClientConnection> client) {
  if (!ptr || size == 0 || !client || client->socket_fd == common::VGRE_INVALID_SOCKET) {
    return VGREResult::ERR_INVALID_VALUE;
  }

  // SHM Fast Path
  if (client->is_local && client->shm_manager && client->shm_manager->getBasePtr()) {
    uint64_t offset = client->shm_offset;
    if (offset + size <= client->shm_manager->getSize()) {
      memcpy(static_cast<uint8_t *>(client->shm_manager->getBasePtr()) + offset, ptr, size);
      client->shm_offset += size;
      
      DataShmPacket p{handle, offset, size};
      return parent_->send_packet(client->socket_fd, PacketType::DATA_SHM, &p, sizeof(p), client->secure_channel.get());
    }
    VGRE_LOG_WARN("TCPCluster", "SHM buffer full during pull-back, falling back to TCP");
  }

  // TCP Path
  DataHeaderPacket h{handle, size};
  if (parent_->send_packet(client->socket_fd, PacketType::DATA_HEADER, &h, sizeof(h), client->secure_channel.get()) != VGREResult::SUCCESS) {
    return VGREResult::ERR_IO;
  }
  return parent_->send_packet(client->socket_fd, PacketType::DATA_BODY, ptr, size, client->secure_channel.get());
}

// ── Delta Synchronization ────────────────────────────────────────────────────

VGREResult MemorySyncManager::sendDeltaSync(
    void *ptr, uint64_t handle, const std::vector<std::pair<size_t, size_t>> &dirty, 
    std::shared_ptr<TCPClusterManager::ClientConnection> client) {
  if (client->is_local && client->shm_manager) {
    return sendDeltaSyncSHM(ptr, handle, dirty, client);
  }
  return sendDeltaSyncTCP(ptr, handle, dirty, client);
}

VGREResult MemorySyncManager::sendDeltaSyncWithRetry(
    void *ptr, uint64_t handle, const std::vector<std::pair<size_t, size_t>> &dirty, 
    std::shared_ptr<TCPClusterManager::ClientConnection> client) {
  DecorrelatedJitterBackoff backoff(INITIAL_RETRY_BACKOFF_MS, MAX_RETRY_BACKOFF_MS);

  for (int attempt = 0; attempt < MAX_DELTA_SYNC_RETRIES; ++attempt) {
    VGREResult r = sendDeltaSync(ptr, handle, dirty, client);
    if (r == VGREResult::SUCCESS) {
      return VGREResult::SUCCESS;
    }
    
    if (r == VGREResult::ERR_IO) {
      VGRE_LOG_ERROR("TCPCluster", "Delta-sync failed due to I/O error (connection lost), not retrying");
      break;
    }
    
    if (attempt < MAX_DELTA_SYNC_RETRIES - 1) {
      int delay_ms = backoff.next();
      VGRE_LOG_DEBUG("TCPCluster", "Delta-sync attempt " + std::to_string(attempt + 1) + " failed, retrying after " + std::to_string(delay_ms) + "ms");
      std::unique_lock<std::mutex> lock(parent_->shutdown_mutex_);
      parent_->shutdown_cv_.wait_for(lock, std::chrono::milliseconds(delay_ms), [this]() { return !parent_->enabled_; });
    }
  }

  VGRE_LOG_WARN("TCPCluster", "Delta-sync failed after " + std::to_string(MAX_DELTA_SYNC_RETRIES) + " attempts, falling back to full sync");
  size_t allocSize = core::RuntimeEngine::instance().getMemoryManager().getAllocationSize(ptr);
  return sendFullSync(ptr, handle, allocSize, client);
}

VGREResult MemorySyncManager::sendDeltaSyncSHM(
    void *ptr, uint64_t handle, const std::vector<std::pair<size_t, size_t>> &dirty, 
    std::shared_ptr<TCPClusterManager::ClientConnection> client) {
  if (!client->shm_manager || !client->shm_manager->getBasePtr()) {
    return sendDeltaSyncTCP(ptr, handle, dirty, client);
  }

  size_t total = 0;
  size_t alloc = core::RuntimeEngine::instance().getMemoryManager().getAllocationSize(ptr);
  
  for (const auto &r : dirty) { 
    if (r.first >= alloc || r.second > alloc - r.first) return VGREResult::ERR_INVALID_VALUE; 
    total += r.second; 
  }

  uint64_t base = client->shm_offset;
  if (base + total > client->shm_manager->getSize()) {
    VGRE_LOG_WARN("TCPCluster", "SHM full during delta sync, falling back to full sync");
    return sendFullSync(ptr, handle, alloc, client);
  }
  
  client->shm_offset += total;
  DataShmDirtyPacket p{handle, static_cast<uint32_t>(dirty.size()), base};
  
  if (parent_->send_packet(client->socket_fd, PacketType::DATA_SHM_DIRTY, &p, sizeof(p), client->secure_channel.get()) != VGREResult::SUCCESS) {
    return VGREResult::ERR_IO;
  }

  uint64_t cur = base;
  for (const auto &r : dirty) {
    DirtyRangePacket rp{r.first, r.second};
    if (parent_->send_packet(client->socket_fd, PacketType::DIRTY_RANGE, &rp, sizeof(rp), client->secure_channel.get()) != VGREResult::SUCCESS) {
      return VGREResult::ERR_IO;
    }
    memcpy(static_cast<uint8_t *>(client->shm_manager->getBasePtr()) + cur, static_cast<uint8_t *>(ptr) + r.first, r.second);
    cur += r.second;
  }
  
  return VGREResult::SUCCESS;
}

VGREResult MemorySyncManager::sendDeltaSyncTCP(
    void *ptr, uint64_t handle, const std::vector<std::pair<size_t, size_t>> &dirty, 
    std::shared_ptr<TCPClusterManager::ClientConnection> client) {
  DataHeaderDirtyPacket h{handle, static_cast<uint32_t>(dirty.size())};
  if (parent_->send_packet(client->socket_fd, PacketType::DATA_HEADER_DIRTY, &h, sizeof(h), client->secure_channel.get()) != VGREResult::SUCCESS) {
    return VGREResult::ERR_IO;
  }

  for (const auto &r : dirty) {
    DirtyRangePacket rp{r.first, r.second};
    if (parent_->send_packet(client->socket_fd, PacketType::DIRTY_RANGE, &rp, sizeof(rp), client->secure_channel.get()) != VGREResult::SUCCESS) {
      return VGREResult::ERR_IO;
    }
    
    uint8_t *data = static_cast<uint8_t *>(ptr) + r.first;
    
    // RDMA Fast Path — chunked for transfers larger than the bounce buffer.
    // Each chunk is written into the remote bounce buffer at offset 0, then
    // a DATA_HEADER_RDMA notification is sent before the next chunk overwrites
    // the bounce buffer.  The receiver copies each chunk to the correct
    // dst_offset inside its local allocation.
    if (client->rdma_connected && client->rdma_conn && r.second >= kRdmaThreshold) {
      const size_t bsz  = client->rdma_conn->bounceCapacity();
      const uint8_t* src = static_cast<const uint8_t*>(data);
      size_t remaining   = r.second;
      size_t dstOff      = 0;
      bool rdmaOk        = true;

      while (remaining > 0 && rdmaOk) {
        size_t chunk = std::min(remaining, bsz);

        if (!client->rdma_conn->rdmaWriteToRemote(*client->rdma_ctx, src + dstOff, chunk)) {
          rdmaOk = false;
          break;
        }
        DataHeaderRDMAPacket rh{};
        rh.target_ptr = handle;
        rh.chunk_size = static_cast<uint32_t>(chunk);
        rh.dst_offset = static_cast<uint32_t>(dstOff);
        if (parent_->send_packet(client->socket_fd, PacketType::DATA_HEADER_RDMA, &rh, sizeof(rh), client->secure_channel.get()) != VGREResult::SUCCESS) {
          rdmaOk = false;
          break;
        }
        dstOff    += chunk;
        remaining -= chunk;
      }
      if (rdmaOk) {
        VGRE_LOG_DEBUG("TCPCluster", "RDMA WRITE successful for delta range (" + std::to_string(r.second) + " bytes)");
        continue; // Skip TCP payload
      }
      VGRE_LOG_WARN("TCPCluster", "RDMA failed — TCP fallback for this delta range");
    }
    
    // TCP Fallback
    if (parent_->send_packet(client->socket_fd, PacketType::DATA_BODY, data, r.second, client->secure_channel.get()) != VGREResult::SUCCESS) {
      return VGREResult::ERR_IO;
    }
  }
  
  return VGREResult::SUCCESS;
}

// ── Full Synchronization ─────────────────────────────────────────────────────

VGREResult MemorySyncManager::sendFullSync(
    void *ptr, uint64_t handle, size_t size, 
    std::shared_ptr<TCPClusterManager::ClientConnection> client) {
  if (client->is_local && client->shm_manager) {
    return sendFullSyncSHM(ptr, handle, size, client);
  }
  return sendFullSyncTCP(ptr, handle, size, client);
}

VGREResult MemorySyncManager::sendFullSyncSHM(
    void *ptr, uint64_t handle, size_t size, 
    std::shared_ptr<TCPClusterManager::ClientConnection> client) {
  if (!client->shm_manager || !client->shm_manager->getBasePtr() || 
      client->shm_offset + size > client->shm_manager->getSize()) {
    return sendFullSyncTCP(ptr, handle, size, client);
  }
  
  uint64_t off = client->shm_offset; 
  client->shm_offset += size;
  memcpy(static_cast<uint8_t *>(client->shm_manager->getBasePtr()) + off, ptr, size);
  
  DataShmPacket p{handle, off, size};
  return parent_->send_packet(client->socket_fd, PacketType::DATA_SHM, &p, sizeof(p), client->secure_channel.get());
}

VGREResult MemorySyncManager::sendFullSyncTCP(
    void *ptr, uint64_t handle, size_t size, 
    std::shared_ptr<TCPClusterManager::ClientConnection> client) {
  // RDMA Fast Path — chunked for transfers larger than the bounce buffer.
  if (client->rdma_connected && client->rdma_conn && size >= kRdmaThreshold) {
    const size_t bsz  = client->rdma_conn->bounceCapacity();
    const uint8_t* src = static_cast<const uint8_t*>(ptr);
    size_t remaining   = size;
    size_t dstOff      = 0;
    bool rdmaOk        = true;

    while (remaining > 0 && rdmaOk) {
      size_t chunk = std::min(remaining, bsz);

      if (!client->rdma_conn->rdmaWriteToRemote(*client->rdma_ctx, src + dstOff, chunk)) {
        rdmaOk = false;
        break;
      }
      DataHeaderRDMAPacket rh{};
      rh.target_ptr = handle;
      rh.chunk_size = static_cast<uint32_t>(chunk);
      rh.dst_offset = static_cast<uint32_t>(dstOff);
      if (parent_->send_packet(client->socket_fd, PacketType::DATA_HEADER_RDMA, &rh, sizeof(rh), client->secure_channel.get()) != VGREResult::SUCCESS) {
        rdmaOk = false;
        break;
      }
      dstOff    += chunk;
      remaining -= chunk;
    }
    if (rdmaOk) {
      VGRE_LOG_DEBUG("TCPCluster", "RDMA WRITE successful for full sync (" + std::to_string(size) + " bytes)");
      return VGREResult::SUCCESS;
    }
    VGRE_LOG_WARN("TCPCluster", "RDMA failed mid-stream — TCP fallback for full sync");
  }
  
  // TCP Path
  DataHeaderPacket h{handle, size};
  if (parent_->send_packet(client->socket_fd, PacketType::DATA_HEADER, &h, sizeof(h), client->secure_channel.get()) != VGREResult::SUCCESS) {
    return VGREResult::ERR_IO;
  }
  return parent_->send_packet(client->socket_fd, PacketType::DATA_BODY, ptr, size, client->secure_channel.get());
}

} // namespace advanced
} // namespace vgre
