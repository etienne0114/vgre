#ifndef VGRE_IPC_MANAGER_H
#define VGRE_IPC_MANAGER_H
#include "vgre/api/vgre_c_api.h"
#include <mutex>
#include <vector>

namespace vgre {
namespace advanced {

#define VGRE_MAX_PROCESSES 32
#define VGRE_MAX_NODES 16
#define VGRE_SHM_NAME "/vgre_global_state"

struct ProcessSlot {
  int32_t pid;
  bool active;
  vgre_telemetry_t telemetry;
};

struct GlobalState {
  int32_t master_pid;
  uint32_t active_count;
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

private:
  IPCManager();
  ~IPCManager();

  bool enabled_ = false;
  bool isMaster_ = false;
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
