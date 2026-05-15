#include "vgre/core/texture_manager.h"
#include "vgre/common/logger.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace vgre {
namespace core {

VGREResult TextureManager::surf2Dwrite(SurfaceId id, float value, int x,
                                       int y) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  auto it = surfaces_.find(id);
  if (it == surfaces_.end())
    return VGREResult::ERR_INVALID_VALUE;
  auto &surf = it->second;
  int w = static_cast<int>(surf.width);
  int h = static_cast<int>(surf.height);
  if (x < 0 || x >= w || y < 0 || y >= h)
    return VGREResult::ERR_INVALID_VALUE;
  uint8_t *base = static_cast<uint8_t *>(surf.data);
  uint8_t *elem = base + (static_cast<size_t>(y) * w + x) * surf.elementSize;
  switch (surf.elementType) {
  case TextureElementType::FLOAT32:
    *reinterpret_cast<float *>(elem) = value;
    break;
  case TextureElementType::FLOAT64:
    *reinterpret_cast<double *>(elem) = static_cast<double>(value);
    break;
  case TextureElementType::INT8:
    *reinterpret_cast<int8_t *>(elem) = static_cast<int8_t>(value);
    break;
  case TextureElementType::INT16:
    *reinterpret_cast<int16_t *>(elem) = static_cast<int16_t>(value);
    break;
  case TextureElementType::INT32:
    *reinterpret_cast<int32_t *>(elem) = static_cast<int32_t>(value);
    break;
  case TextureElementType::UINT8:
    *reinterpret_cast<uint8_t *>(elem) = static_cast<uint8_t>(value);
    break;
  case TextureElementType::UINT16:
    *reinterpret_cast<uint16_t *>(elem) = static_cast<uint16_t>(value);
    break;
  case TextureElementType::UINT32:
    *reinterpret_cast<uint32_t *>(elem) = static_cast<uint32_t>(value);
    break;
  }
  return VGREResult::SUCCESS;
}

VGREResult TextureManager::surf2Dread(SurfaceId id, float &value, int x,
                                      int y) const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  auto it = surfaces_.find(id);
  if (it == surfaces_.end())
    return VGREResult::ERR_INVALID_VALUE;
  const auto &surf = it->second;
  int w = static_cast<int>(surf.width);
  int h = static_cast<int>(surf.height);
  if (x < 0 || x >= w || y < 0 || y >= h)
    return VGREResult::ERR_INVALID_VALUE;
  const uint8_t *base = static_cast<const uint8_t *>(surf.data);
  const uint8_t *elem = base + (static_cast<size_t>(y) * w + x) * surf.elementSize;
  switch (surf.elementType) {
  case TextureElementType::FLOAT32:
    value = *reinterpret_cast<const float *>(elem);
    break;
  case TextureElementType::FLOAT64:
    value = static_cast<float>(*reinterpret_cast<const double *>(elem));
    break;
  case TextureElementType::INT8:
    value = static_cast<float>(*reinterpret_cast<const int8_t *>(elem));
    break;
  case TextureElementType::INT16:
    value = static_cast<float>(*reinterpret_cast<const int16_t *>(elem));
    break;
  case TextureElementType::INT32:
    value = static_cast<float>(*reinterpret_cast<const int32_t *>(elem));
    break;
  case TextureElementType::UINT8:
    value = static_cast<float>(*reinterpret_cast<const uint8_t *>(elem));
    break;
  case TextureElementType::UINT16:
    value = static_cast<float>(*reinterpret_cast<const uint16_t *>(elem));
    break;
  case TextureElementType::UINT32:
    value = static_cast<float>(*reinterpret_cast<const uint32_t *>(elem));
    break;
  }
  return VGREResult::SUCCESS;
}

} // namespace core
} // namespace vgre
