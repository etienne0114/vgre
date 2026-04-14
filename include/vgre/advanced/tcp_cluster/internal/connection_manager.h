#pragma once

#include "vgre/advanced/tcp_cluster.h"  // Need ClientConnection definition
#include "vgre/api/vgre_c_api.h"
#include "vgre/common/error_codes.h"
#include "vgre/common/sockets.h"
#include <memory>
#include <mutex>
#include <string>
#include <vector>

// Forward declare sockaddr_in from global namespace
struct sockaddr_in;

namespace vgre {
namespace advanced {

/**
 * @brief ConnectionManager handles client connection lifecycle for TCPClusterManager.
 * 
 * This module extracts connection management logic from the monolithic tcp_cluster.cpp,
 * including:
 * - Connection acceptance and establishment
 * - Duplicate connection detection
 * - Client list management
 * - Connection cleanup and purging
 * 
 * Design: Facade pattern with parent pointer to TCPClusterManager for accessing
 * shared state and delegating operations.
 */
class ConnectionManager {
public:
  /**
   * @brief Construct a new ConnectionManager
   * @param parent Pointer to parent TCPClusterManager (non-owning)
   */
  explicit ConnectionManager(TCPClusterManager* parent);
  
  ~ConnectionManager() = default;
  
  // Delete copy/move to prevent accidental duplication
  ConnectionManager(const ConnectionManager&) = delete;
  ConnectionManager& operator=(const ConnectionManager&) = delete;
  ConnectionManager(ConnectionManager&&) = delete;
  ConnectionManager& operator=(ConnectionManager&&) = delete;
  
  /**
   * @brief Accept a new connection from server socket
   * @param server_fd Server socket file descriptor
   * @return VGREResult::SUCCESS on success, error code otherwise
   */
  VGREResult acceptConnection(vgre::common::vgre_socket_t server_fd);
  
  /**
   * @brief Connect to master node (worker-side)
   * @param host Master hostname or IP address
   * @param port Master port number
   * @return VGREResult::SUCCESS on success, error code otherwise
   */
  VGREResult connectToMaster(const std::string& host, int port);
  
  /**
   * @brief Close a client connection
   * @param client Shared pointer to ClientConnection to close
   */
  void closeConnection(std::shared_ptr<TCPClusterManager::ClientConnection> client);
  
  /**
   * @brief Get client by worker index
   * @param worker_idx Worker index (0-based)
   * @return Shared pointer to ClientConnection, or nullptr if not found
   */
  std::shared_ptr<TCPClusterManager::ClientConnection> getClient(int worker_idx);
  
  /**
   * @brief Get all active clients
   * @return Vector of shared pointers to active ClientConnections
   */
  std::vector<std::shared_ptr<TCPClusterManager::ClientConnection>> getActiveClients();
  
  /**
   * @brief Add client if not duplicate (atomic check-and-insert)
   * 
   * This method prevents duplicate connections from the same IP address.
   * It atomically checks for existing connections and adds the new one
   * only if no duplicate exists.
   * 
   * @param ip_address Client IP address
   * @param socket_fd Client socket file descriptor
   * @param address Socket address structure
   * @return true if client was added, false if duplicate was detected
   */
  bool addClientIfNotDuplicate(const std::string& ip_address, 
                                vgre::common::vgre_socket_t socket_fd, 
                                const ::sockaddr_in& address);
  
  /**
   * @brief Purge dead clients from the client list
   * 
   * Removes clients that are no longer active and not in the middle
   * of authentication handshake.
   */
  void purgeDeadClients();

private:
  TCPClusterManager* parent_; // Non-owning pointer to parent
};

} // namespace advanced
} // namespace vgre
