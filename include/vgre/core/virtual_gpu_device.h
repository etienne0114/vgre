#ifndef VGRE_CORE_VIRTUAL_GPU_DEVICE_H
#define VGRE_CORE_VIRTUAL_GPU_DEVICE_H

#include "vgre/common/error_codes.h"
#include "vgre/common/types.h"

#include <atomic>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace vgre {
namespace core {

// ── Stream descriptor ──────────────────────────────────────────────────────
struct Stream {
  StreamId id = 0;
  int priority = 0;
  StreamState state = StreamState::IDLE;
};

// ── Virtual GPU Device ─────────────────────────────────────────────────────
class VirtualGPUDevice {
public:
  VirtualGPUDevice(DeviceId id = 0);
  ~VirtualGPUDevice();

  // Properties
  const DeviceProperties &getProperties() const;
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
