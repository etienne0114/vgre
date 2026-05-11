/**
 * VGRE Discovery Manager - Proactive Connection Loop
 */

#include "vgre/advanced/tcp_cluster/internal/discovery_manager.h"
#include "vgre/advanced/tcp_cluster.h"
#include "vgre/common/logger.h"
#include "vgre/common/sockets.h"
#include <chrono>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <winsock2.h>
#else
#include <arpa/inet.h>
#endif

namespace vgre {
namespace advanced {

void DiscoveryManager::proactiveConnectionLoop() {
    while (parent_ && parent_->enabled_ && !stop_proactive_) {
        std::vector<std::string> addrSnapshot;
        { std::lock_guard<std::recursive_mutex> lock(parent_->clients_mutex_); addrSnapshot = parent_->proactive_worker_addresses_; }

        for (const auto& addr : addrSnapshot) {
            if (!parent_->enabled_ || stop_proactive_) break;
            std::string ip = addr; int port = parent_->port_;
            size_t colon = addr.find(':');
            if (colon != std::string::npos) { ip = addr.substr(0, colon); try { port = std::stoi(addr.substr(colon + 1)); } catch (...) {} }

            bool alreadyConnected = false;
            {
                std::lock_guard<std::recursive_mutex> lock(parent_->clients_mutex_);
                for (const auto& client : parent_->clients_) if (client && client->ip_address == ip) { alreadyConnected = true; break; }
            }
            if (alreadyConnected) continue;

            vgre::common::vgre_socket_t sock = socket(AF_INET, SOCK_STREAM, 0);
            struct sockaddr_in serv_addr{}; serv_addr.sin_family = AF_INET; serv_addr.sin_port = htons(static_cast<uint16_t>(port));
            if (inet_pton(AF_INET, ip.c_str(), &serv_addr.sin_addr) <= 0) { vgre::common::vgre_close_socket(sock); continue; }

            if (connect(sock, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) >= 0) {
                auto conn = std::make_shared<TCPClusterManager::ClientConnection>();
                conn->socket_fd = sock; conn->ip_address = ip; conn->port = port; conn->active = true;
                std::lock_guard<std::recursive_mutex> lock(parent_->clients_mutex_);
                parent_->clients_.push_back(std::move(conn));
                parent_->syncToIPC();
            } else vgre::common::vgre_close_socket(sock);
        }
        for (int i = 0; i < 50 && parent_->enabled_ && !stop_proactive_; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
}

} // namespace advanced
} // namespace vgre
