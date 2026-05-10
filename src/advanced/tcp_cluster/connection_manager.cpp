/**
 * @file connection_manager.cpp
 * @brief ConnectionManager implementation for TCPClusterManager
 * 
 * This module extracts connection management logic from the monolithic
 * tcp_cluster.cpp file, providing focused connection lifecycle management.
 */

#include "vgre/advanced/tcp_cluster/internal/connection_manager.h"
#include "vgre/advanced/tcp_cluster.h"
#include "vgre/common/logger.h"
#include "vgre/common/sockets.h"
#include "vgre/advanced/tcp_cluster/internal/shared_utilities.h"
#include <algorithm>
#include <cstring>

// All platform socket headers are provided by vgre/common/sockets.h above.

namespace vgre {
namespace advanced {

using vgre::common::vgre_socket_t;
using vgre::common::VGRE_INVALID_SOCKET;

ConnectionManager::ConnectionManager(TCPClusterManager* parent)
    : parent_(parent) {
  if (!parent_) {
    VGRE_LOG_ERROR("ConnectionManager", "Parent TCPClusterManager is null");
  }
}

void ConnectionManager::closeConnection(
    std::shared_ptr<TCPClusterManager::ClientConnection> client) {
  if (!client) {
    return;
  }
  
  if (client->socket_fd != VGRE_INVALID_SOCKET) {
    VGRE_LOG_INFO("ConnectionManager", 
        "Closing connection to " + client->ip_address);
    vgre::common::vgre_close_socket(client->socket_fd);
    client->socket_fd = VGRE_INVALID_SOCKET;
  }
  
  client->active = false;
  client->security_established = false;
}

std::shared_ptr<TCPClusterManager::ClientConnection> 
ConnectionManager::getClient(int worker_idx) {
  if (!parent_) {
    return nullptr;
  }
  
  std::lock_guard<std::recursive_mutex> lock(parent_->clients_mutex_);
  
  int active_count = 0;
  for (const auto& client : parent_->clients_) {
    if (client && client->active) {
      if (active_count == worker_idx) {
        return client;
      }
      active_count++;
    }
  }
  
  return nullptr;
}

std::vector<std::shared_ptr<TCPClusterManager::ClientConnection>> 
ConnectionManager::getActiveClients() {
  if (!parent_) {
    return {};
  }
  
  std::vector<std::shared_ptr<TCPClusterManager::ClientConnection>> active;
  std::lock_guard<std::recursive_mutex> lock(parent_->clients_mutex_);
  
  for (const auto& client : parent_->clients_) {
    if (client && client->active) {
      active.push_back(client);
    }
  }
  
  return active;
}

bool ConnectionManager::addClientIfNotDuplicate(
    const std::string& ip_address,
    vgre_socket_t socket_fd,
    const ::sockaddr_in& address) {
  if (!parent_) {
    vgre::common::vgre_close_socket(socket_fd);
    return false;
  }
  
  std::lock_guard<std::recursive_mutex> lock(parent_->clients_mutex_);
  
  // Check for duplicate: same IP and (active or is_authenticating)
  for (const auto& client : parent_->clients_) {
    if (client && (client->active || client->is_authenticating) &&
        client->socket_fd != VGRE_INVALID_SOCKET &&
        client->ip_address == ip_address) {
      // Duplicate found - close new socket and return false
      VGRE_LOG_WARN("ConnectionManager",
          "Dropping duplicate inbound connection from " + ip_address +
          " — active connection already exists "
          "(prevents double-handshake HMAC key mismatch)");
      vgre::common::vgre_close_socket(socket_fd);
      return false;
    }
  }
  
  // No duplicate - create and add new ClientConnection
  auto conn = std::make_shared<TCPClusterManager::ClientConnection>();
  conn->socket_fd = socket_fd;
  conn->ip_address = ip_address;
  conn->port = ntohs(address.sin_port);
  conn->active = true;
  conn->expecting_type = true;
  conn->rx_buffer.clear();
  parent_->clients_.push_back(std::move(conn));
  
  VGRE_LOG_INFO("ConnectionManager", 
      "Added new client from " + ip_address + ":" + 
      std::to_string(ntohs(address.sin_port)));
  
  return true;
}

void ConnectionManager::purgeDeadClients() {
  if (!parent_) {
    return;
  }
  
  std::lock_guard<std::recursive_mutex> lock(parent_->clients_mutex_);
  
  auto before = parent_->clients_.size();
  parent_->clients_.erase(
      std::remove_if(parent_->clients_.begin(), parent_->clients_.end(),
          [](const std::shared_ptr<TCPClusterManager::ClientConnection>& c) {
            return c && !c->active && !c->is_authenticating;
          }),
      parent_->clients_.end());
  
  if (parent_->clients_.size() != before) {
    VGRE_LOG_DEBUG("ConnectionManager",
        "Purged " + std::to_string(before - parent_->clients_.size()) +
        " dead client(s); remaining=" + std::to_string(parent_->clients_.size()));
  }
}

} // namespace advanced
} // namespace vgre
