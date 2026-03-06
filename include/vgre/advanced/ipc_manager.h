#ifndef VGRE_IPC_MANAGER_H
#define VGRE_IPC_MANAGER_H
#include "vgre/api/vgre_c_api.h"
#include <mutex>

namespace vgre {
namespace advanced {

#define VGRE_MAX_PROCESSES 16
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
  char padding[256]; // Future expansion
};

class IPCManager {
public:
  static IPCManager &instance();

  bool initialize(bool isMaster = false);
  void shutdown();

  void updateLocalTelemetry(const vgre_telemetry_t &telemetry);
  void getGlobalTelemetry(vgre_telemetry_t &outCombined);

  bool isEnabled() const { return enabled_; }

private:
  IPCManager();
  ~IPCManager();

  bool enabled_ = false;
  bool isMaster_ = false;
  int shm_fd_ = -1;
  GlobalState *state_ = nullptr;
  int local_slot_ = -1;
  std::mutex mutex_;
};

} // namespace advanced
} // namespace vgre

#endif // VGRE_IPC_MANAGER_H
