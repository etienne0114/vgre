/**
 * VGRE TCP Cluster Manager — Thin Coordinator
 *
 * Wires together the modular sub-systems that make up the TCP cluster:
 *   - ConnectionManager   : TCP socket lifecycle & rate limiting
 *   - DiscoveryManager    : UDP master/worker discovery & proactive connects
 *   - PacketHandler       : VSBP packet framing, send/receive
 *   - SecurityManager     : HMAC-SHA256 + AES-256-CTR handshake
 *   - MemorySyncManager   : Pointer / delta / full memory synchronisation
 *   - CollectiveOpsManager: AllReduce and telemetry aggregation
 *   - DispatchManager     : Remote kernel launch and partition dispatch
 *
 * Lifecycle (constructor, destructor, initialize, shutdown) lives here.
 * All per-subsystem logic is in the corresponding module file under
 * src/advanced/tcp_cluster/.
 */

#include "vgre/advanced/secure_channel.h"
#include "vgre/advanced/tcp_cluster.h"
#include "vgre/advanced/tcp_cluster/internal/collective_ops_manager.h"
#include "vgre/advanced/tcp_cluster/internal/connection_manager.h"
#include "vgre/advanced/tcp_cluster/internal/diagnostic_logger.h"
#include "vgre/advanced/tcp_cluster/internal/discovery_manager.h"
#include "vgre/advanced/tcp_cluster/internal/dispatch_manager.h"
#include "vgre/advanced/tcp_cluster/internal/interfaces.h"
#include "vgre/advanced/tcp_cluster/internal/memory_sync_manager.h"
#include "vgre/advanced/tcp_cluster/internal/packet_handler.h"
#include "vgre/advanced/tcp_cluster/internal/security_manager.h"
#include "vgre/advanced/tcp_cluster/internal/shared_utilities.h"
#include "vgre/advanced/tcp_cluster/internal/windows_socket_manager.h"
#include "vgre/common/logger.h"
#include "vgre/common/sockets.h"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <future>
#include <string>
#include <thread>

// All platform socket headers are provided by vgre/common/sockets.h above.

namespace vgre {
namespace advanced {

// Using vgre::common types (as needed)
using vgre::common::VGRE_INVALID_SOCKET;
using vgre::common::vgre_socket_t;

// ── Constructor: instantiate all sub-systems ──────────────────────────────
TCPClusterManager::TCPClusterManager()
    : socket_factory_(std::make_unique<RealSocketFactory>()),
      memory_manager_(std::make_unique<RealMemoryManager>()),
      kernel_manager_(std::make_unique<RealKernelManager>()),
      security_factory_(std::make_unique<RealSecureChannelFactory>()),
      connection_manager_(std::make_unique<ConnectionManager>(this)),
      discovery_manager_(std::make_unique<DiscoveryManager>(this)),
      packet_handler_(std::make_unique<PacketHandler>()),
      security_manager_(std::make_unique<SecurityManager>(this)),
      memory_sync_manager_(std::make_unique<MemorySyncManager>(this)),
      collective_ops_manager_(std::make_unique<CollectiveOpsManager>(this)),
      dispatch_manager_(std::make_unique<DispatchManager>(this)),
      windows_socket_manager_(std::make_unique<WindowsSocketManager>()) {
  local_platform_ = detail::detect_platform();
  local_platform_name_ = MeshTopologyManager::getPlatformName(local_platform_);
}

TCPClusterManager &TCPClusterManager::instance() {
  static TCPClusterManager instance;
  return instance;
}

VGREResult TCPClusterManager::validateMemoryAlignment() {
  return MemoryAlignmentValidator::validateAllPacketStructures();
}

// ── Constructor with dependency injection for testability ─────────────────
TCPClusterManager::TCPClusterManager(
    std::unique_ptr<ISocketFactory> socket_factory,
    std::unique_ptr<IMemoryManager> memory_manager,
    std::unique_ptr<IKernelManager> kernel_manager,
    std::unique_ptr<ISecureChannelFactory> security_factory)
    : socket_factory_(std::move(socket_factory)),
      memory_manager_(std::move(memory_manager)),
      kernel_manager_(std::move(kernel_manager)),
      security_factory_(std::move(security_factory)),
      connection_manager_(std::make_unique<ConnectionManager>(this)),
      discovery_manager_(std::make_unique<DiscoveryManager>(this)),
      packet_handler_(std::make_unique<PacketHandler>()),
      security_manager_(std::make_unique<SecurityManager>(this)),
      memory_sync_manager_(std::make_unique<MemorySyncManager>(this)),
      collective_ops_manager_(std::make_unique<CollectiveOpsManager>(this)),
      dispatch_manager_(std::make_unique<DispatchManager>(this)),
      windows_socket_manager_(std::make_unique<WindowsSocketManager>()) {
  local_platform_ = detail::detect_platform();
  local_platform_name_ = MeshTopologyManager::getPlatformName(local_platform_);
}

// ── Destructor: shutdown is a no-op if never initialised ─────────────────
TCPClusterManager::~TCPClusterManager() {
  fprintf(stderr, "DEBUG [TCPCluster] TCPClusterManager destructor starting...\n");
  try {
    shutdown();
  } catch (...) {
    fprintf(stderr, "ERROR [TCPCluster] Exception in TCPClusterManager destructor\n");
  }
  fprintf(stderr, "DEBUG [TCPCluster] TCPClusterManager destructor finished.\n");
}

// ── initialize ────────────────────────────────────────────────────────────

// ── shutdown ──────────────────────────────────────────────────────────────

// ── Restart Coordination and Management ─────────────────────────────────────

// ── Diagnostic Logging and Operational Monitoring Implementation ────────────

// ── Consolidated Methods from tcp_cluster.cpp ──────────────────────────────\n
// ── Unified Packet Construction ──────────────────────────────────────────

// ── Delta-Sync Logic Extraction ──────────────────────────────────────────
VGREResult TCPClusterManager::syncPointerToWorker(
    void *ptr, uint64_t handle, std::shared_ptr<ClientConnection> client) {
  return memory_sync_manager_->syncPointerToWorker(ptr, handle, client);
}

VGREResult TCPClusterManager::sendDeltaSync(
    void *ptr, uint64_t handle,
    const std::vector<std::pair<size_t, size_t>> &dirtyRanges,
    std::shared_ptr<ClientConnection> client) {
  return memory_sync_manager_->sendDeltaSync(ptr, handle, dirtyRanges, client);
}

VGREResult TCPClusterManager::sendDeltaSyncWithRetry(
    void *ptr, uint64_t handle,
    const std::vector<std::pair<size_t, size_t>> &dirtyRanges,
    std::shared_ptr<ClientConnection> client) {
  return memory_sync_manager_->sendDeltaSyncWithRetry(ptr, handle, dirtyRanges,
                                                      client);
}

VGREResult TCPClusterManager::sendDeltaSyncSHM(
    void *ptr, uint64_t handle,
    const std::vector<std::pair<size_t, size_t>> &dirtyRanges,
    std::shared_ptr<ClientConnection> client) {
  return memory_sync_manager_->sendDeltaSyncSHM(ptr, handle, dirtyRanges,
                                                client);
}

VGREResult TCPClusterManager::sendDeltaSyncTCP(
    void *ptr, uint64_t handle,
    const std::vector<std::pair<size_t, size_t>> &dirtyRanges,
    std::shared_ptr<ClientConnection> client) {
  return memory_sync_manager_->sendDeltaSyncTCP(ptr, handle, dirtyRanges,
                                                client);
}

VGREResult
TCPClusterManager::sendFullSync(void *ptr, uint64_t handle, size_t size,
                                std::shared_ptr<ClientConnection> client) {
  return memory_sync_manager_->sendFullSync(ptr, handle, size, client);
}

VGREResult
TCPClusterManager::sendFullSyncSHM(void *ptr, uint64_t handle, size_t size,
                                   std::shared_ptr<ClientConnection> client) {
  return memory_sync_manager_->sendFullSyncSHM(ptr, handle, size, client);
}

VGREResult
TCPClusterManager::sendFullSyncTCP(void *ptr, uint64_t handle, size_t size,
                                   std::shared_ptr<ClientConnection> client) {
  return memory_sync_manager_->sendFullSyncTCP(ptr, handle, size, client);
}

// ── Argument Serialization Logic Extraction ──────────────────────────────
VGREResult TCPClusterManager::streamArgumentsToWorker(
    void **args, int num_args, uint64_t kernel_id,
    std::shared_ptr<ClientConnection> client) {
  return memory_sync_manager_->streamArgumentsToWorker(args, num_args,
                                                       kernel_id, client);
}

VGREResult
TCPClusterManager::sendStructArg(void *arg, int arg_index, uint64_t kernel_id,
                                 std::shared_ptr<ClientConnection> client) {
  return memory_sync_manager_->sendStructArg(arg, arg_index, kernel_id, client);
}

VGREResult
TCPClusterManager::sendPointerArg(void *arg, int arg_index,
                                  std::shared_ptr<ClientConnection> client) {
  return memory_sync_manager_->sendPointerArg(arg, arg_index, client);
}

VGREResult
TCPClusterManager::sendScalarArg(void *arg, int arg_index, ArgType type,
                                 std::shared_ptr<ClientConnection> client) {
  return memory_sync_manager_->sendScalarArg(arg, arg_index, type, client);
}

// ── Blocking I/O Helper ───────────────────────────────────────────────────

// ── Diagnostic Helper ─────────────────────────────────────────────────────

// Constructor, destructor                     →
// tcp_cluster/tcp_cluster_manager.cpp initialize, shutdown →
// tcp_cluster/tcp_cluster_lifecycle.cpp serverLoop →
// tcp_cluster/server_loop.cpp clientLoop                                  →
// tcp_cluster/client_loop.cpp processClientStagingBuffer                  →
// tcp_cluster/client_packet_dispatch.cpp

VGREResult TCPClusterManager::launchRemoteKernel(
    int worker_idx, uint64_t kernel_id, const uint32_t grid_dim[3],
    const uint32_t block_dim[3], void **args, int num_args, size_t shared_mem) {
  return dispatch_manager_->launchRemoteKernel(
      worker_idx, kernel_id, grid_dim, block_dim, args, num_args, shared_mem);
}

void TCPClusterManager::broadcastKernelRegistration(uint64_t kernel_id,
                                                    const std::string &name,
                                                    const std::string &source) {
  dispatch_manager_->broadcastKernelRegistration(kernel_id, name, source);
}

int TCPClusterManager::getFirstActiveWorker() const {
  if (!is_master_) {
    return -1;
  }
  std::lock_guard<std::recursive_mutex> lock(clients_mutex_);
  if (!clients_.empty()) {
      VGRE_LOG_DEBUG("TCPCluster", "getFirstActiveWorker: scanning " + std::to_string(clients_.size()) + " clients");
  }
  for (size_t i = 0; i < clients_.size(); ++i) {
    if (clients_[i] && clients_[i]->active) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

void TCPClusterManager::handleRemoteCommand(const RemoteCommandPacket &pkt) {
  dispatch_manager_->handleRemoteCommand(pkt);
}

// ── Security ─────────────────────────────────────────────────────

VGREResult TCPClusterManager::enableSecurity(bool enabled) {
  return security_manager_->enableSecurity(enabled);
}

SessionInfo TCPClusterManager::getSecurityInfo() const {
  return security_manager_->getSecurityInfo();
}

VGREResult TCPClusterManager::performSecureHandshake(
    std::shared_ptr<ClientConnection> clientPtr) {
  return security_manager_->performServerHandshake(clientPtr);
}

VGREResult TCPClusterManager::performClientSecureHandshake() {
  return security_manager_->performClientHandshake();
}

VGREResult TCPClusterManager::performPeerClientHandshake(
    std::shared_ptr<ClientConnection> peer) {
  return security_manager_->performPeerClientHandshake(peer);
}

// ── Partitioned Kernel Dispatch ──────────────────────────────────

VGREResult TCPClusterManager::launchPartitionedKernel(
    uint64_t kernel_id, const uint32_t grid_dim[3], const uint32_t block_dim[3],
    void **args, int num_args, size_t shared_mem) {
  return dispatch_manager_->launchPartitionedKernel(
      kernel_id, grid_dim, block_dim, args, num_args, shared_mem);
}

VGREResult TCPClusterManager::collectPartitionResults(uint64_t kernel_id,
                                                      uint32_t total_partitions,
                                                      int timeout_ms) {
  return dispatch_manager_->collectPartitionResults(kernel_id, total_partitions,
                                                    timeout_ms);
}

VGREResult TCPClusterManager::waitForRemoteResult(uint64_t kernel_id,
                                                  int timeout_ms) {
  return dispatch_manager_->waitForRemoteResult(kernel_id, timeout_ms);
}

void TCPClusterManager::handlePartitionDispatch(
    const PartitionDispatchPacket &pkt) {
  dispatch_manager_->handlePartitionDispatch(pkt);
}

// ══════════════════════════════════════════════════════════════════════════════
// Missing Method Implementations
// ══════════════════════════════════════════════════════════════════════════════

VGREResult TCPClusterManager::allReduce(void *ptr, size_t count, int datatype,
                                         ReductionOp op) {
  return collective_ops_manager_->allReduce(ptr, count, datatype, op);
}

VGREResult TCPClusterManager::barrier() {
  return collective_ops_manager_->barrier();
}

} // namespace advanced
} // namespace vgre