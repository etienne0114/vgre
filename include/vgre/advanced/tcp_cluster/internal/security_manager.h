#pragma once

#include "vgre/advanced/tcp_cluster.h"  // Need ClientConnection definition
#include "vgre/api/vgre_c_api.h"
#include "vgre/common/error_codes.h"
#include <memory>
#include <string>
#include <atomic>

namespace vgre {
namespace advanced {

// Forward declarations
class SecureChannel;

/**
 * @brief SecurityManager handles security operations for TCPClusterManager.
 * 
 * This module extracts security logic from the monolithic tcp_cluster.cpp,
 * including:
 * - Security configuration (enable/disable)
 * - Security handshake (master and worker sides)
 * - Session key management
 * - Security state tracking
 * 
 * Design: Facade pattern with parent pointer to TCPClusterManager for accessing
 * shared state and delegating operations.
 */
class SecurityManager {
public:
  /**
   * @brief Construct a new SecurityManager
   * @param parent Pointer to parent TCPClusterManager (non-owning)
   */
  explicit SecurityManager(TCPClusterManager* parent);
  
  ~SecurityManager() = default;
  
  // Delete copy/move to prevent accidental duplication
  SecurityManager(const SecurityManager&) = delete;
  SecurityManager& operator=(const SecurityManager&) = delete;
  SecurityManager(SecurityManager&&) = delete;
  SecurityManager& operator=(SecurityManager&&) = delete;
  
  /**
   * @brief Enable or disable security
   * @param enabled true to enable security, false to disable
   * @return VGREResult::SUCCESS on success, error code otherwise
   */
  VGREResult enableSecurity(bool enabled);
  
  /**
   * @brief Check if security is enabled
   * @return true if security is enabled, false otherwise
   */
  bool isSecurityEnabled() const;
  
  /**
   * @brief Get security session information
   * @return SessionInfo structure with cipher, encryption status, and traffic stats
   */
  SessionInfo getSecurityInfo() const;
  
  /**
   * @brief Perform security handshake (master side)
   * 
   * This method performs the server-side security handshake with a connecting worker.
   * It generates a nonce, sends it to the worker, receives the worker's nonce,
   * derives a session key, and establishes a secure channel.
   * 
   * @param client Shared pointer to ClientConnection
   * @return VGREResult::SUCCESS on success, error code otherwise
   */
  VGREResult performServerHandshake(std::shared_ptr<TCPClusterManager::ClientConnection> client);
  
  /**
   * @brief Perform security handshake (worker side)
   * 
   * This method performs the client-side security handshake with the master.
   * It uses auto-negotiation: peeks for a SECURE_HANDSHAKE packet with a short
   * timeout (200ms). If detected, completes the handshake. If not, proceeds
   * with plaintext connection.
   * 
   * @return VGREResult::SUCCESS on success, error code otherwise
   */
  VGREResult performClientHandshake();
  
  /**
   * @brief Rotate session key for a client connection
   * 
   * This method rotates the session key for an existing secure connection.
   * It performs an authenticated ROTATE_KEY packet exchange and applies
   * the nonce-derived key update on the active secure channel.
   * 
   * @param client Shared pointer to ClientConnection
   * @return VGREResult::SUCCESS on success, error code otherwise
   */
  VGREResult rotateSessionKey(std::shared_ptr<TCPClusterManager::ClientConnection> client);

  /**
   * @brief Compute SHA256 hash of token for logging/diagnostics.
   *        Same algorithm used in handshake failure messages — call this
   *        wherever a token fingerprint is needed so all log output is consistent.
   * @param token The authentication token string (no trailing newline)
   * @return Full 64-char hex SHA256 digest
   */
  std::string computeTokenHash(const std::string& token) const;

private:
  // Parent cluster manager (non-owning pointer)
  // Used to access: clients_, client_fd_, client_secure_channel_,
  // send_packet_direct, recv_packet, waitForData, and other shared state
  TCPClusterManager* parent_;

  // Hybrid authentication mode control
  bool strict_auth_mode_ = false;  // Read from VGRE_CLUSTER_STRICT_AUTH
};

} // namespace advanced
} // namespace vgre
