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

VGREResult ConnectionManager::acceptConnection(vgre_socket_t server_fd) {
  if (server_fd == VGRE_INVALID_SOCKET) {
    return VGREResult::ERR_INVALID_VALUE;
  }
  
  struct sockaddr_in address;
  socklen_t addrlen = sizeof(address);
  vgre_socket_t new_socket = accept(server_fd, (struct sockaddr*)&address, &addrlen);
  
  if (new_socket == VGRE_INVALID_SOCKET) {
    return VGREResult::ERR_IO;
  }
  
  char ipstr[INET_ADDRSTRLEN];
  inet_ntop(AF_INET, &(address.sin_addr), ipstr, sizeof(ipstr));
  std::string ip_address(ipstr);
  
  // Configure socket
  vgre::common::vgre_ioctl_nonblock(new_socket);
  vgre::common::vgre_set_tcp_keepalive(new_socket, 5, 2, 3);
  
  // Add client if not duplicate
  if (!addClientIfNotDuplicate(ip_address, new_socket, address)) {
    // Duplicate detected, socket already closed by addClientIfNotDuplicate
    return VGREResult::ERR_ALREADY_EXISTS;
  }
  
  VGRE_LOG_INFO("ConnectionManager", 
      "Accepted new connection from " + ip_address + ":" + 
      std::to_string(ntohs(address.sin_port)));
  
  return VGREResult::SUCCESS;
}

VGREResult ConnectionManager::connectToMaster(const std::string& host, int port) {
  if (host.empty() || port <= 0) {
    return VGREResult::ERR_INVALID_VALUE;
  }
  
  vgre_socket_t client_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (client_fd == VGRE_INVALID_SOCKET) {
    VGRE_LOG_ERROR("ConnectionManager", "Failed to create client socket");
    return VGREResult::ERR_IO;
  }
  
  struct sockaddr_in serv_addr;
  serv_addr.sin_family = AF_INET;
  serv_addr.sin_port = htons(port);
  
  if (inet_pton(AF_INET, host.c_str(), &serv_addr.sin_addr) <= 0) {
    VGRE_LOG_ERROR("ConnectionManager", 
        "Invalid address or not supported: " + host);
    vgre::common::vgre_close_socket(client_fd);
    return VGREResult::ERR_INVALID_VALUE;
  }
  
  if (connect(client_fd, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0) {
    VGRE_LOG_DEBUG("ConnectionManager", 
        "Connection failed to " + host + ":" + std::to_string(port));
    vgre::common::vgre_close_socket(client_fd);
    return VGREResult::ERR_IO;
  }
  
  // Set non-blocking
  vgre::common::vgre_ioctl_nonblock(client_fd);

  // Store the connected fd in the parent so clientLoop() can use it.
  {
    std::lock_guard<std::mutex> lock(parent_->client_mutex_);
    parent_->client_fd_ = client_fd;
  }

  VGRE_LOG_INFO("ConnectionManager",
      "Connected to master node at " + host + ":" + std::to_string(port));
  return VGREResult::SUCCESS;
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
