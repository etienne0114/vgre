/**
 * VGRE TCP Cluster Manager - Core State and Discovery Parsing
 */

#include "vgre/advanced/tcp_cluster.h"
#include "vgre/advanced/ipc_manager.h"
#include "vgre/api/vgre_c_api.h"
#include "vgre/common/logger.h"
#include <cstring>
#include <cstdlib>

namespace vgre {
namespace advanced {

void TCPClusterManager::syncToIPC() {
  if (!is_master_ || !enabled_) return;
  std::vector<vgre_cluster_node_t> ipcNodes;
  {
      std::lock_guard<std::recursive_mutex> lock(clients_mutex_);
      for (const auto& c : clients_) {
          if (!c || !c->active) continue;
          if (!c->capability_received && !c->is_local) continue;
          vgre_cluster_node_t node{};
          strncpy(node.address, c->ip_address.c_str(), sizeof(node.address) - 1);
          node.port = c->port; node.cpu_cores = c->cpu_cores; node.memory_bytes = c->cpu_memory;
          node.latency_ms = c->last_telemetry.avg_kernel_latency_ms; node.available = 1;
          strncpy(node.igpu_name, c->igpu_name, sizeof(node.igpu_name) - 1);
          ipcNodes.push_back(node);
      }
  }
  vgre::advanced::IPCManager::instance().updateClusterNodes(ipcNodes);
}

void TCPClusterManager::parseProactiveNodes() {
    const char* env = vgre_get_config("VGRE_CLUSTER_NODES");
    if (!env || env[0] == '\0') return;
    proactive_worker_addresses_.clear();
    std::string nodes(env);
    size_t start = 0;
    while (start <= nodes.size()) {
        size_t comma = nodes.find(',', start);
        std::string addr = nodes.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        start = (comma == std::string::npos ? nodes.size() + 1 : comma + 1);
        while (!addr.empty() && (addr.front() == ' ' || addr.front() == '\t')) addr.erase(addr.begin());
        while (!addr.empty() && (addr.back()  == ' ' || addr.back()  == '\t')) addr.pop_back();
        if (addr.empty()) continue;
        proactive_worker_addresses_.push_back(addr);
    }
}

void TCPClusterManager::parseMeshPeers() {
    const char* env = vgre_get_config("VGRE_MESH_PEERS");
    if (!env || env[0] == '\0') return;
    mesh_peer_ips_.clear();
    std::string peers_str(env);
    size_t start = 0;
    while (start <= peers_str.size()) {
        size_t comma = peers_str.find(',', start);
        std::string peer = peers_str.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        start = (comma == std::string::npos ? peers_str.size() + 1 : comma + 1);
        while (!peer.empty() && peer.front() == ' ') peer.erase(peer.begin());
        while (!peer.empty() && peer.back()  == ' ') peer.pop_back();
        if (peer.empty()) continue;
        size_t colon = peer.rfind(':');
        if (colon == std::string::npos) continue;
        std::string peer_ip = peer.substr(0, colon);
        int peer_port = port_;
        try { peer_port = std::stoi(peer.substr(colon + 1)); } catch (...) {}
        mesh_peer_ips_.insert(peer_ip);
        if ((port_ < peer_port) || (port_ == peer_port && host_ < peer_ip)) {
            bool found = false;
            for (const auto& existing : proactive_worker_addresses_) {
                if (existing == peer) { found = true; break; }
            }
            if (!found) proactive_worker_addresses_.push_back(peer);
        }
    }
}

} // namespace advanced
} // namespace vgre
