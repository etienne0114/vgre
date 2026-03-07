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
  std::lock_guard<std::mutex> lock(mutex_);
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

// ── Create texture ─────────────────────────────────────────────────────────
VGREResult TextureManager::createTexture(TextureId &outId, const void *data,
                                         size_t width, size_t height,
                                         size_t elementSize,
                                         const TextureDescriptor &desc) {
  if (!data || width == 0 || elementSize == 0)
    return VGREResult::ERROR_INVALID_VALUE;

  std::lock_guard<std::mutex> lock(mutex_);

  TextureObject tex;
  tex.id = nextTextureId_++;
  tex.data = data;
  tex.width = width;
  tex.height = (height == 0) ? 1 : height;
  tex.elementSize = elementSize;
  tex.desc = desc;

  textures_[tex.id] = tex;
  outId = tex.id;

  VGRE_LOG_INFO("TextureManager",
                "Created texture " + std::to_string(tex.id) + " (" +
                    std::to_string(width) + "x" + std::to_string(tex.height) +
                    ", " + std::to_string(elementSize) + " bytes/element)");

  return VGREResult::SUCCESS;
}

VGREResult TextureManager::destroyTexture(TextureId id) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = textures_.find(id);
  if (it == textures_.end())
    return VGREResult::ERROR_INVALID_VALUE;

  textures_.erase(it);
  VGRE_LOG_DEBUG("TextureManager", "Destroyed texture " + std::to_string(id));
  return VGREResult::SUCCESS;
}

// ── Create surface ─────────────────────────────────────────────────────────
VGREResult TextureManager::createSurface(SurfaceId &outId, void *data,
                                         size_t width, size_t height,
                                         size_t elementSize) {
  if (!data || width == 0 || elementSize == 0)
    return VGREResult::ERROR_INVALID_VALUE;

  std::lock_guard<std::mutex> lock(mutex_);

  SurfaceObject surf;
  surf.id = nextSurfaceId_++;
  surf.data = data;
  surf.width = width;
  surf.height = (height == 0) ? 1 : height;
  surf.elementSize = elementSize;

  surfaces_[surf.id] = surf;
  outId = surf.id;

  VGRE_LOG_INFO("TextureManager", "Created surface " + std::to_string(surf.id) +
                                      " (" + std::to_string(width) + "x" +
                                      std::to_string(surf.height) + ")");

  return VGREResult::SUCCESS;
}

VGREResult TextureManager::destroySurface(SurfaceId id) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = surfaces_.find(id);
  if (it == surfaces_.end())
    return VGREResult::ERROR_INVALID_VALUE;

  surfaces_.erase(it);
  return VGREResult::SUCCESS;
}

// ── 1D texture fetch (integer coordinate) ──────────────────────────────────
float TextureManager::tex1Dfetch(TextureId id, int x) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = textures_.find(id);
  if (it == textures_.end())
    return 0.0f;

  const auto &tex = it->second;
  int idx =
      applyAddressMode(x, static_cast<int>(tex.width), tex.desc.addressMode);

  if (idx < 0) // Border mode, out of bounds
    return tex.desc.borderColor;

  // Read float value from the texture data
  const auto *fdata = static_cast<const float *>(tex.data);
  return fdata[idx];
}

// ── 2D texture fetch (floating-point coordinates with interpolation) ──────
float TextureManager::tex2D(TextureId id, float x, float y) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = textures_.find(id);
  if (it == textures_.end())
    return 0.0f;

  const auto &tex = it->second;
  const auto *fdata = static_cast<const float *>(tex.data);
  int w = static_cast<int>(tex.width);
  int h = static_cast<int>(tex.height);

  if (tex.desc.filterMode == TextureFilterMode::POINT) {
    // Nearest-neighbor sampling
    int ix = static_cast<int>(std::floor(x));
    int iy = static_cast<int>(std::floor(y));

    ix = applyAddressMode(ix, w, tex.desc.addressMode);
    iy = applyAddressMode(iy, h, tex.desc.addressMode);

    if (ix < 0 || iy < 0)
      return tex.desc.borderColor;

    return fdata[iy * w + ix];
  }

  // Bilinear interpolation
  float fx = x - 0.5f; // Shift to texel center
  float fy = y - 0.5f;
  int x0 = static_cast<int>(std::floor(fx));
  int y0 = static_cast<int>(std::floor(fy));
  int x1 = x0 + 1;
  int y1 = y0 + 1;
  float fracX = fx - static_cast<float>(x0);
  float fracY = fy - static_cast<float>(y0);

  // Apply address modes
  auto sample = [&](int sx, int sy) -> float {
    sx = applyAddressMode(sx, w, tex.desc.addressMode);
    sy = applyAddressMode(sy, h, tex.desc.addressMode);
    if (sx < 0 || sy < 0)
      return tex.desc.borderColor;
    return fdata[sy * w + sx];
  };

  float v00 = sample(x0, y0);
  float v10 = sample(x1, y0);
  float v01 = sample(x0, y1);
  float v11 = sample(x1, y1);

  // Bilinear blend
  float top = v00 * (1.0f - fracX) + v10 * fracX;
  float bot = v01 * (1.0f - fracX) + v11 * fracX;
  return top * (1.0f - fracY) + bot * fracY;
}

// ── Surface write ──────────────────────────────────────────────────────────
VGREResult TextureManager::surf2Dwrite(SurfaceId id, float value, int x,
                                       int y) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = surfaces_.find(id);
  if (it == surfaces_.end())
    return VGREResult::ERROR_INVALID_VALUE;

  auto &surf = it->second;
  int w = static_cast<int>(surf.width);
  int h = static_cast<int>(surf.height);

  if (x < 0 || x >= w || y < 0 || y >= h)
    return VGREResult::ERROR_INVALID_VALUE;

  auto *fdata = static_cast<float *>(surf.data);
  fdata[y * w + x] = value;
  return VGREResult::SUCCESS;
}

VGREResult TextureManager::surf2Dread(SurfaceId id, float &value, int x,
                                      int y) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = surfaces_.find(id);
  if (it == surfaces_.end())
    return VGREResult::ERROR_INVALID_VALUE;

  const auto &surf = it->second;
  int w = static_cast<int>(surf.width);
  int h = static_cast<int>(surf.height);

  if (x < 0 || x >= w || y < 0 || y >= h)
    return VGREResult::ERROR_INVALID_VALUE;

  const auto *fdata = static_cast<const float *>(surf.data);
  value = fdata[y * w + x];
  return VGREResult::SUCCESS;
}

// ── Queries ────────────────────────────────────────────────────────────────
size_t TextureManager::getTextureCount() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return textures_.size();
}

size_t TextureManager::getSurfaceCount() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return surfaces_.size();
}

// ── Singleton ──────────────────────────────────────────────────────────────
TextureManager &TextureManager::instance() {
  static TextureManager mgr;
  return mgr;
}

} // namespace core
} // namespace vgre
