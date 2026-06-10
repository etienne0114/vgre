#include "vgre/core/texture_manager.h"
#include "vgre/common/logger.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace vgre {
namespace core {


TextureManager::TextureManager() {
  VGRE_LOG_DEBUG("TextureManager", "Initialized");
}

TextureManager::~TextureManager() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  textures_.clear();
  surfaces_.clear();
}

// ── Address mode helper ────────────────────────────────────────────────────
int TextureManager::applyAddressMode(int coord, int size,
                                     TextureAddressMode mode) const {
  if (size <= 0)
    return 0;

  switch (mode) {
  case TextureAddressMode::CLAMP:
    return std::max(0, std::min(coord, size - 1));

  case TextureAddressMode::WRAP:
    coord = coord % size;
    if (coord < 0)
      coord += size;
    return coord;

  case TextureAddressMode::MIRROR: {
    // Mirror: for negative coords, reflect into positive range
    if (coord < 0)
      coord = -(coord + 1);
    int period = 2 * size;
    coord = coord % period;
    if (coord >= size)
      coord = period - coord - 1;
    return std::max(0, std::min(coord, size - 1));
  }

  case TextureAddressMode::BORDER:
    // Return -1 to signal that borderColor should be used
    if (coord < 0 || coord >= size)
      return -1;
    return coord;
  }

  return std::max(0, std::min(coord, size - 1));
}

// ── Catmull-Rom cubic interpolation kernel ────────────────────────────────
float cubicWeight(float x) {
  float ax = std::abs(x);
  if (ax < 1.0f) {
    return (3.0f * ax * ax * ax - 5.0f * ax * ax + 2.0f) / 2.0f;
  } else if (ax < 2.0f) {
    return (-ax * ax * ax + 5.0f * ax * ax - 8.0f * ax + 4.0f) / 2.0f;
  }
  return 0.0f;
}

// ── Create texture ─────────────────────────────────────────────────────────
VGREResult TextureManager::createTexture(TextureId &outId, const void *data,
                                         size_t width, size_t height,
                                         size_t elementSize,
                                         const TextureDescriptor &desc,
                                         unsigned int layers) {
  if (!data || width == 0 || elementSize == 0)
    return VGREResult::ERR_INVALID_VALUE;

  std::lock_guard<std::recursive_mutex> lock(mutex_);

  TextureObject tex;
  tex.id = nextTextureId_++;
  tex.data = data;
  tex.width = width;
  tex.height = (height == 0) ? 1 : height;
  tex.depth = 1;  
  tex.elementSize = elementSize;
  tex.layers = layers;
  tex.desc = desc;

  textures_[tex.id] = tex;
  outId = tex.id;

  VGRE_LOG_INFO("TextureManager",
                "Created texture " + std::to_string(tex.id) + " (" +
                    std::to_string(width) + "x" + std::to_string(tex.height) +
                    ", " + std::to_string(elementSize) + " bytes/element, type=" +
                    std::to_string(static_cast<int>(desc.elementType)) + ")");

  return VGREResult::SUCCESS;
}

VGREResult TextureManager::createTextureView(TextureId &outId,
                                             TextureId baseTextureId,
                                             const ResourceViewDescriptor &viewDesc,
                                             const TextureDescriptor &texDesc) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  auto it = textures_.find(baseTextureId);
  if (it == textures_.end()) {
    VGRE_LOG_ERROR("TextureManager", "Base texture not found: " + std::to_string(baseTextureId));
    return VGREResult::ERR_INVALID_VALUE;
  }

  const auto &base = it->second;
  size_t elementSize = 0;
  switch (viewDesc.format) {
    case TextureElementType::FLOAT32:
    case TextureElementType::INT32:
    case TextureElementType::UINT32: elementSize = 4; break;
    case TextureElementType::FLOAT64: elementSize = 8; break;
    case TextureElementType::FP16:
    case TextureElementType::INT16:
    case TextureElementType::UINT16: elementSize = 2; break;
    case TextureElementType::INT8:
    case TextureElementType::UINT8: elementSize = 1; break;
  }

  // Calculate layer offset: each layer occupies width×height×depth×elementSize bytes.
  size_t layerStride = base.width * base.height * base.depth * base.elementSize;
  size_t layerOffset = viewDesc.firstLayer * layerStride;

  // Calculate mipmap level byte offset.
  // Mip level k has dimensions max(1, W>>k) × max(1, H>>k) × max(1, D>>k).
  // Sum the sizes of levels [0, firstMipmapLevel) to get the byte offset into
  // the layer's mip chain.
  size_t mipOffset = 0;
  {
    size_t mw = base.width, mh = base.height, md = base.depth;
    for (uint32_t m = 0; m < viewDesc.firstMipmapLevel; ++m) {
      mipOffset += mw * mh * md * base.elementSize;
      if (mw > 1) mw >>= 1;
      if (mh > 1) mh >>= 1;
      if (md > 1) md >>= 1;
    }
  }

  TextureObject view;
  view.id = nextTextureId_++;
  view.data = base.data;
  view.offsetInBytes = base.offsetInBytes + viewDesc.offsetInBytes + layerOffset + mipOffset;
  view.width = (viewDesc.width == 0) ? base.width : viewDesc.width;
  view.height = (viewDesc.height == 0) ? base.height : viewDesc.height;
  view.depth = (viewDesc.depth == 0) ? base.depth : viewDesc.depth;
  view.elementSize = elementSize;
  view.layers = (viewDesc.lastLayer >= viewDesc.firstLayer) ? (viewDesc.lastLayer - viewDesc.firstLayer + 1) : 1;
  view.mips = (viewDesc.lastMipmapLevel >= viewDesc.firstMipmapLevel) ? (viewDesc.lastMipmapLevel - viewDesc.firstMipmapLevel + 1) : 1;
  view.desc = texDesc;
  view.desc.elementType = viewDesc.format;

  // Final boundary check for "Zero Simulation" integrity
  size_t totalBaseBytes = base.width * base.height * base.depth * base.elementSize * base.layers;
  if (view.offsetInBytes + (view.width * view.height * view.depth * view.elementSize * view.layers) > base.offsetInBytes + totalBaseBytes) {
      VGRE_LOG_ERROR("TextureManager", "Texture view exceeds base resource boundaries.");
      return VGREResult::ERR_INVALID_VALUE;
  }

  textures_[view.id] = view;
  outId = view.id;

  VGRE_LOG_INFO("TextureManager",
                "Created texture view " + std::to_string(view.id) + " from base " +
                std::to_string(baseTextureId) + " (offset=" + std::to_string(view.offsetInBytes) +
                ", size=" + std::to_string(view.width) + "x" + std::to_string(view.height) +
                ", format=" + std::to_string(static_cast<int>(viewDesc.format)) + ")");

  return VGREResult::SUCCESS;
}

VGREResult TextureManager::destroyTexture(TextureId id) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  auto it = textures_.find(id);
  if (it == textures_.end())
    return VGREResult::ERR_INVALID_VALUE;

  textures_.erase(it);
  VGRE_LOG_DEBUG("TextureManager", "Destroyed texture " + std::to_string(id));
  return VGREResult::SUCCESS;
}

// ── Create surface ─────────────────────────────────────────────────────────
VGREResult TextureManager::createSurface(SurfaceId &outId, void *data,
                                         size_t width, size_t height,
                                         size_t elementSize,
                                         TextureElementType elementType) {
  if (!data || width == 0 || height == 0 || elementSize == 0)
    return VGREResult::ERR_INVALID_VALUE;

  std::lock_guard<std::recursive_mutex> lock(mutex_);

  SurfaceObject surf;
  surf.id = nextSurfaceId_++;
  surf.data = data;
  surf.width = width;
  surf.height = height;
  surf.elementSize = elementSize;
  surf.elementType = elementType;

  surfaces_[surf.id] = surf;
  outId = surf.id;

  VGRE_LOG_INFO("TextureManager", "Created surface " + std::to_string(surf.id) +
                                      " (" + std::to_string(width) + "x" +
                                      std::to_string(surf.height) + ", " +
                                      std::to_string(elementSize) + " bytes/element, type=" +
                                      std::to_string(static_cast<int>(elementType)) + ")");

  return VGREResult::SUCCESS;
}

VGREResult TextureManager::destroySurface(SurfaceId id) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  auto it = surfaces_.find(id);
  if (it == surfaces_.end())
    return VGREResult::ERR_INVALID_VALUE;

  surfaces_.erase(it);
  return VGREResult::SUCCESS;
}

} // namespace core
} // namespace vgre
