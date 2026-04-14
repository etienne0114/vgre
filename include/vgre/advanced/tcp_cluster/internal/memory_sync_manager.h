#ifndef VGRE_ADVANCED_TCP_CLUSTER_MEMORY_SYNC_MANAGER_H
#define VGRE_ADVANCED_TCP_CLUSTER_MEMORY_SYNC_MANAGER_H

#include "vgre/common/error_codes.h"
#include "vgre/common/types.h"
#include "vgre/advanced/tcp_cluster.h"
#include <vector>
#include <utility>
#include <cstdint>
#include <memory>

namespace vgre {
namespace advanced {

/**
 * MemorySyncManager - Handles memory synchronization between master and workers
 * 
 * This module encapsulates all memory synchronization operations including:
 * - Delta-sync (dirty page tracking)
 * - Full-sync (complete memory transfer)
 * - SHM vs TCP path selection
 * - Argument serialization (scalar, pointer, struct)
 * - Retry logic with exponential backoff
 */
class MemorySyncManager {
public:
  explicit MemorySyncManager(TCPClusterManager* parent);
  ~MemorySyncManager() = default;

  // Friend class for access to private methods
  friend class TCPClusterManager;

  // High-level memory sync operations
  VGREResult streamArgumentsToWorker(void** args, int num_args, uint64_t kernel_id, 
                                     std::shared_ptr<TCPClusterManager::ClientConnection> client);
  
  VGREResult syncPointerToWorker(void* ptr, uint64_t handle, 
                                 std::shared_ptr<TCPClusterManager::ClientConnection> client);
  
  VGREResult syncPointerFromWorker(void* ptr, uint64_t handle, size_t size,
                                   std::shared_ptr<TCPClusterManager::ClientConnection> client);

  // SHM initialization
  VGREResult initializeShmForClient(std::shared_ptr<TCPClusterManager::ClientConnection> client);

private:
  // Delta-sync operations
  VGREResult sendDeltaSync(void* ptr, uint64_t handle, 
                           const std::vector<std::pair<size_t, size_t>>& dirtyRanges,
                           std::shared_ptr<TCPClusterManager::ClientConnection> client);
  
  VGREResult sendDeltaSyncWithRetry(void* ptr, uint64_t handle,
                                    const std::vector<std::pair<size_t, size_t>>& dirtyRanges,
                                    std::shared_ptr<TCPClusterManager::ClientConnection> client);
  
  VGREResult sendDeltaSyncSHM(void* ptr, uint64_t handle,
                              const std::vector<std::pair<size_t, size_t>>& dirtyRanges,
                              std::shared_ptr<TCPClusterManager::ClientConnection> client);
  
  VGREResult sendDeltaSyncTCP(void* ptr, uint64_t handle,
                              const std::vector<std::pair<size_t, size_t>>& dirtyRanges,
                              std::shared_ptr<TCPClusterManager::ClientConnection> client);

  // Full-sync operations
  VGREResult sendFullSync(void* ptr, uint64_t handle, size_t size,
                          std::shared_ptr<TCPClusterManager::ClientConnection> client);
  
  VGREResult sendFullSyncSHM(void* ptr, uint64_t handle, size_t size,
                             std::shared_ptr<TCPClusterManager::ClientConnection> client);
  
  VGREResult sendFullSyncTCP(void* ptr, uint64_t handle, size_t size,
                             std::shared_ptr<TCPClusterManager::ClientConnection> client);

  // Argument serialization
  VGREResult sendScalarArg(void* arg, int arg_index, ArgType type,
                           std::shared_ptr<TCPClusterManager::ClientConnection> client);
  
  VGREResult sendPointerArg(void* arg, int arg_index,
                            std::shared_ptr<TCPClusterManager::ClientConnection> client);
  
  VGREResult sendStructArg(void* arg, int arg_index, uint64_t kernel_id,
                           std::shared_ptr<TCPClusterManager::ClientConnection> client);

  TCPClusterManager* parent_;
};

} // namespace advanced
} // namespace vgre

#endif // VGRE_ADVANCED_TCP_CLUSTER_MEMORY_SYNC_MANAGER_H
