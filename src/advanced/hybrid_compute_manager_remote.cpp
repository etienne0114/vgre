#include "vgre/advanced/hybrid_compute_manager.h"
#include "vgre/advanced/adaptive_execution_engine.h"
#include "vgre/api/vgre_c_api.h"
#include "vgre/advanced/tcp_cluster.h"
#include "vgre/advanced/tcp_cluster/tcp_cluster_defaults.h"
#include "vgre/common/logger.h"
#include "vgre/runtime/cpu_parallel_executor.h"

#ifdef VGRE_HAS_OPENCL_BACKEND
#include "vgre/runtime/igpu_opencl_executor.h"
#endif

// System Headers
#include <chrono>
#include <fstream>
#include <mutex>
#include <sstream>
#include "vgre/core/memory_manager.h"
#include "vgre/core/runtime_engine.h"
#include <thread>
#include <vector>

#include "vgre/common/os_backend.h"
#if !defined(_WIN32)
#include <dirent.h>  // opendir/readdir — Linux/macOS iGPU device scan
#include <netdb.h>
#endif

namespace vgre {
namespace advanced {

// ── Remote node management ─────────────────────────────────────────────────
VGREResult HybridComputeManager::addRemoteNode(const std::string &address,
                                               int port) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (address.empty() || port <= 0 || port > 65535) {
    return VGREResult::ERR_INVALID_VALUE;
  }

  // Check for duplicates
  for (const auto &node : resources_.remoteNodes) {
    if (node.address == address && node.port == port) {
      return VGREResult::ERR_ALREADY_EXISTS;
    }
  }

  RemoteNode node;
  node.address = address;
  node.port = port;
  node.available = false; // not yet verified

  resources_.remoteNodes.push_back(node);

  VGRE_LOG_INFO("HybridComputeManager",
                "Added remote node: " + address + ":" + std::to_string(port));
  return VGREResult::SUCCESS;
}

VGREResult HybridComputeManager::removeRemoteNode(const std::string &address) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = std::remove_if(
      resources_.remoteNodes.begin(), resources_.remoteNodes.end(),
      [&](const RemoteNode &n) { return n.address == address; });
  if (it == resources_.remoteNodes.end()) {
    return VGREResult::ERR_INVALID_VALUE;
  }
  resources_.remoteNodes.erase(it, resources_.remoteNodes.end());
  return VGREResult::SUCCESS;
}

VGREResult HybridComputeManager::updateRemoteNodeCapability(const std::string &address, int cores,
                                                           size_t memory, bool hasIGPU,
                                                           const std::string &igpuName) {
  std::lock_guard<std::mutex> lock(mutex_);
  bool found = false;
  for (auto &node : resources_.remoteNodes) {
    if (node.address == address) {
      node.cpuCores = cores;
      node.cpuMemoryBytes = memory;
      node.hasIntegratedGPU = hasIGPU;
      node.igpuName = igpuName;
      node.available = true;
      found = true;
      break;
    }
  }
  
  if (!found) {
    RemoteNode node;
    node.address = address;
    node.port = vgre::advanced::kDefaultClusterPort;
    node.cpuCores = cores;
    node.cpuMemoryBytes = memory;
    node.hasIntegratedGPU = hasIGPU;
    node.igpuName = igpuName;
    node.available = true;
    resources_.remoteNodes.push_back(node);
  }

  // Recalculate authoritative total units
  resources_.totalComputeUnits = resources_.cpuCores;
  for (const auto &node : resources_.remoteNodes) {
    if (node.available) resources_.totalComputeUnits += node.cpuCores;
  }
  
  return VGREResult::SUCCESS;
}

VGREResult HybridComputeManager::updateRemoteNodeTelemetry(const std::string &address, const vgre_telemetry_t &telemetry) {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto &node : resources_.remoteNodes) {
    if (node.address == address) {
      node.lastTelemetry = telemetry;

      // Maintain EWMA of avg_kernel_latency_ms for smoothed cost estimation.
      constexpr double kLoadAlpha = 0.2;
      auto &hist = nodeLoadHistory_[address];
      // Use real telemetry latency; fall back to 2×RTT from the node's ping
      // result (node.latencyMs is the one-way RTT probe), not an arbitrary 50ms.
      double rawMs = telemetry.avg_kernel_latency_ms > 0.0
                         ? telemetry.avg_kernel_latency_ms
                         : (node.latencyMs > 0.0 ? node.latencyMs * 2.0 : 0.0);
      if (rawMs > 0.0) {
        if (hist.samples == 0) {
          hist.ewma_exec_ms = rawMs; // seed from first real observation
        } else {
          hist.ewma_exec_ms =
              hist.ewma_exec_ms * (1.0 - kLoadAlpha) + rawMs * kLoadAlpha;
        }
        hist.samples++;
      }

      return VGREResult::SUCCESS;
    }
  }
  return VGREResult::ERR_INVALID_VALUE;
}

VGREResult HybridComputeManager::pingRemoteNode(const std::string &address,
                                                double &latencyMs) {
  latencyMs = 0.0;
  int targetPort = 0;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto &node : resources_.remoteNodes) {
      if (node.address == address) {
        targetPort = node.port;
        break;
      }
    }
  }

  if (targetPort == 0)
    return VGREResult::ERR_INVALID_VALUE;

#if defined(_WIN32)
  // Windows: use WinSock for TCP connection test
  WSADATA wsaData;
  if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
    return VGREResult::ERR_IO;
  }

  auto start = std::chrono::steady_clock::now();

  struct addrinfo hints{}, *res = nullptr;
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;

  if (getaddrinfo(address.c_str(), std::to_string(targetPort).c_str(), &hints,
                  &res) != 0) {
    WSACleanup();
    return VGREResult::ERR_INVALID_VALUE;
  }

  bool connected = false;
  for (addrinfo *it = res; it != nullptr; it = it->ai_next) {
    SOCKET sock = socket(it->ai_family, it->ai_socktype, it->ai_protocol);
    if (sock == INVALID_SOCKET) {
      continue;
    }

    if (connect(sock, it->ai_addr, static_cast<int>(it->ai_addrlen)) !=
        SOCKET_ERROR) {
      connected = true;
      closesocket(sock);
      break;
    }
    closesocket(sock);
  }

  if (!connected) {
    freeaddrinfo(res);
    WSACleanup();
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto &node : resources_.remoteNodes) {
      if (node.address == address) {
        node.available = false;
        break;
      }
    }
    return VGREResult::ERR_IO;
  }

  freeaddrinfo(res);

  auto end = std::chrono::steady_clock::now();
  latencyMs = std::chrono::duration<double, std::milli>(end - start).count();
  WSACleanup();
#else
  auto start = std::chrono::steady_clock::now();

  struct addrinfo hints{}, *res = nullptr;
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;

  if (getaddrinfo(address.c_str(), std::to_string(targetPort).c_str(), &hints,
                  &res) != 0) {
    return VGREResult::ERR_INVALID_VALUE;
  }

  bool connected = false;
  for (addrinfo *it = res; it != nullptr; it = it->ai_next) {
    int sock = socket(it->ai_family, it->ai_socktype, it->ai_protocol);
    if (sock < 0) {
      continue;
    }

    if (connect(sock, it->ai_addr, it->ai_addrlen) == 0) {
      connected = true;
      close(sock);
      break;
    }
    close(sock);
  }

  if (!connected) {
    freeaddrinfo(res);
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto &node : resources_.remoteNodes) {
      if (node.address == address) {
        node.available = false;
        break;
      }
    }
    return VGREResult::ERR_IO;
  }

  freeaddrinfo(res);

  auto end = std::chrono::steady_clock::now();
  latencyMs = std::chrono::duration<double, std::milli>(end - start).count();
#endif

  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto &node : resources_.remoteNodes) {
      if (node.address == address) {
        node.latencyMs = latencyMs;
        node.available = true;
        // Capability will be handshaked via TCPClusterManager
        break;
      }
    }
  }

  VGRE_LOG_INFO("HybridComputeManager",
                "TCP Pinged " + address +
                    " — genuine latency: " + std::to_string(latencyMs) + " ms");

  return VGREResult::SUCCESS;
}

std::vector<RemoteNode> HybridComputeManager::getRemoteNodes() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return resources_.remoteNodes;
}


} // namespace advanced
} // namespace vgre
