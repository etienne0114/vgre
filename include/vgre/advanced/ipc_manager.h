#ifndef VGRE_IPC_MANAGER_H
#define VGRE_IPC_MANAGER_H
#include "vgre/api/vgre_c_api.h"
#include <atomic>
#include <mutex>
#include <string>
#include <vector>

namespace vgre {
namespace advanced {

#define VGRE_MAX_PROCESSES 32
#define VGRE_MAX_NODES 16
#define VGRE_SHM_NAME "/vgre_global_state"
#define VGRE_AUTH_TOKEN_SIZE 64

// Authentication token for IPC hardening
struct AuthToken {
    uint8_t token[VGRE_AUTH_TOKEN_SIZE];
    uint64_t timestamp;
    uint32_t pid;
    uint8_t hmac[32]; // HMAC-SHA256 for integrity
};

struct ProcessSlot {
  int32_t pid;
  bool active;
  vgre_telemetry_t telemetry;
  AuthToken auth; // Authentication token for this slot
  uint64_t last_heartbeat;
};

struct GlobalState {
  int32_t master_pid;
  uint32_t active_count;
  AuthToken master_auth; // Master authentication token
  uint64_t state_version; // Version for integrity checking
  uint8_t state_hmac[32]; // HMAC of entire state for integrity
  ProcessSlot slots[VGRE_MAX_PROCESSES];
  
  // Phase 12: Distributed Topology Synchronization
  vgre_cluster_node_t cluster_nodes[VGRE_MAX_NODES];
  uint32_t cluster_node_count;
  
  char padding[1024]; // Future expansion
};

class IPCManager {
public:
  static IPCManager &instance();

  bool initialize(bool isMaster = false);
  void shutdown();

  void updateLocalTelemetry(const vgre_telemetry_t &telemetry);
  void getGlobalTelemetry(vgre_telemetry_t &outCombined);
  
  // Phase 12: Cluster Sync
  void updateClusterNodes(const std::vector<vgre_cluster_node_t>& nodes);
  void getClusterNodes(std::vector<vgre_cluster_node_t>& outNodes);

  bool isEnabled() const { return enabled_; }
  
  // IPC Hardening: Authentication and integrity
  bool validateAuthToken(const AuthToken& token) const;
  void generateAuthToken(AuthToken& token, int32_t pid);
  bool verifyStateIntegrity() const;
  void updateStateHMAC();

private:
  IPCManager();
  ~IPCManager();

  bool enabled_ = false;
  bool isMaster_ = false;
  std::atomic<bool> upgrading_{false};
  AuthToken local_auth_; // Local process authentication token
  std::string auth_secret_; // Shared secret for HMAC generation
#if defined(_WIN32)
  void *shm_fd_ = nullptr; // HANDLE on Windows
#else
  int shm_fd_ = -1;
#endif
  GlobalState *state_ = nullptr;
  int local_slot_ = -1;
  std::recursive_mutex mutex_;
};

} // namespace advanced
} // namespace vgre

#endif // VGRE_IPC_MANAGER_H
