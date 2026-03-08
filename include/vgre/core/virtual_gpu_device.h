#ifndef VGRE_CORE_VIRTUAL_GPU_DEVICE_H
#define VGRE_CORE_VIRTUAL_GPU_DEVICE_H

#include "vgre/common/error_codes.h"
#include "vgre/common/types.h"

#include <atomic>
#include <mutex>
#include <unordered_map>

namespace vgre {
namespace core {

// ── Stream descriptor ──────────────────────────────────────────────────────
struct Stream {
  StreamId id = 0;
  int priority = 0;
  StreamState state = StreamState::IDLE;
};

/**
 * @brief Representation of a Virtual GPU device instance.
 *
 * Each VirtualGPUDevice instance maps to a set of hardware resources
 * (physical cores, memory pools) and maintains its own stream/context state.
 */
class VirtualGPUDevice {
public:
  VirtualGPUDevice(DeviceId id = 0);
  ~VirtualGPUDevice();

  /**
   * @brief Returns the sensed or assigned properties for this virtual device.
   */
  DeviceProperties getProperties() const;
  void setProperties(const DeviceProperties &props);
  void setId(DeviceId id);
  DeviceId getId() const;

  // Context
  VGREResult createContext();
  VGREResult destroyContext();
  bool hasContext() const;

  // Streams
  VGREResult createStream(StreamId &outId, int priority = 0);
  VGREResult destroyStream(StreamId id);
  VGREResult synchronizeStream(StreamId id);
  VGREResult synchronizeDevice();

  void detectHardware();

private:
  DeviceId id_;
  DeviceProperties props_;
  bool contextActive_ = false;
  std::atomic<StreamId> nextStreamId_{1};
  std::unordered_map<StreamId, Stream> streams_;
  mutable std::mutex mutex_;
};

} // namespace core
} // namespace vgre

#endif // VGRE_CORE_VIRTUAL_GPU_DEVICE_H
