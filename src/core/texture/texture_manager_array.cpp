#include "vgre/core/texture_manager.h"
#include "vgre/common/logger.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace vgre {
namespace core {

// ── Queries ────────────────────────────────────────────────────────────────
size_t TextureManager::getTextureCount() const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  return textures_.size();
}

size_t TextureManager::getSurfaceCount() const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  return surfaces_.size();
}

// ── Singleton ────────────────────────────────────────────────────────────────────
TextureManager &TextureManager::instance() {
  static TextureManager mgr;
  return mgr;
}
// ── 3D texture fetch ─────────────────────────────────────────────────────────
float TextureManager::tex3D(TextureId id, float x, float y, float z) const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  auto it = textures_.find(id);
  if (it == textures_.end()) return 0.0f;

  const auto &tex = it->second;
  int w = static_cast<int>(tex.width);
  int h = static_cast<int>(tex.height);
  int d = static_cast<int>(tex.depth);

  float sampleX = x, sampleY = y, sampleZ = z;
  if (tex.desc.normalizedCoords) {
    sampleX *= static_cast<float>(w);
    sampleY *= static_cast<float>(h);
    sampleZ *= static_cast<float>(d);
  }

  if (tex.desc.filterMode == TextureFilterMode::POINT) {
    // Nearest-neighbor
    int ix = static_cast<int>(std::floor(sampleX));
    int iy = static_cast<int>(std::floor(sampleY));
    int iz = static_cast<int>(std::floor(sampleZ));
    double value = sampleTexel(tex, ix, iy, iz);
    return static_cast<float>(value);
  }

  // Trilinear interpolation
  float fx = sampleX - 0.5f;
  float fy = sampleY - 0.5f;
  float fz = sampleZ - 0.5f;
  int x0 = static_cast<int>(std::floor(fx));
  int y0 = static_cast<int>(std::floor(fy));
  int z0 = static_cast<int>(std::floor(fz));
  int x1 = x0 + 1, y1 = y0 + 1, z1 = z0 + 1;
  float fracX = fx - static_cast<float>(x0);
  float fracY = fy - static_cast<float>(y0);
  float fracZ = fz - static_cast<float>(z0);

  auto sample3D = [&](int sx, int sy, int sz) -> double {
    return sampleTexel(tex, sx, sy, sz);
  };

  // Interpolate along x for each (y, z) corner
  double c000 = sample3D(x0, y0, z0), c100 = sample3D(x1, y0, z0);
  double c010 = sample3D(x0, y1, z0), c110 = sample3D(x1, y1, z0);
  double c001 = sample3D(x0, y0, z1), c101 = sample3D(x1, y0, z1);
  double c011 = sample3D(x0, y1, z1), c111 = sample3D(x1, y1, z1);

  // Bilinear on z=0 plane
  double t0 = c000 * (1.0 - fracX) + c100 * fracX;
  double b0 = c010 * (1.0 - fracX) + c110 * fracX;
  double front = t0 * (1.0 - fracY) + b0 * fracY;

  // Bilinear on z=1 plane
  double t1 = c001 * (1.0 - fracX) + c101 * fracX;
  double b1 = c011 * (1.0 - fracX) + c111 * fracX;
  double back = t1 * (1.0 - fracY) + b1 * fracY;

  // Linear between front and back
  return static_cast<float>(front * (1.0 - fracZ) + back * fracZ);
}

// ── 3D Texture creation (explicit depth) ─────────────────────────────────────
VGREResult TextureManager::createTexture3D(TextureId &outId, const void *data,
                                           size_t width, size_t height,
                                           size_t depth, size_t elementSize,
                                           const TextureDescriptor &desc) {
  if (!data || width == 0 || height == 0 || depth == 0 || elementSize == 0)
    return VGREResult::ERR_INVALID_VALUE;

  std::lock_guard<std::recursive_mutex> lock(mutex_);

  TextureObject tex;
  tex.id = nextTextureId_++;
  tex.data = data;
  tex.offsetInBytes = 0;
  tex.width = width;
  tex.height = height;
  tex.depth = depth;
  tex.elementSize = elementSize;
  tex.desc = desc;

  textures_[tex.id] = tex;
  outId = tex.id;

  VGRE_LOG_INFO("TextureManager",
                "Created 3D texture " + std::to_string(tex.id) + " (" +
                    std::to_string(width) + "x" + std::to_string(height) +
                    "x" + std::to_string(depth) + ", " +
                    std::to_string(elementSize) + " bytes/element)");

  return VGREResult::SUCCESS;
}

// ── cudaArray lifecycle (owns backing memory) ─────────────────────────────────
VGREResult TextureManager::createCudaArray(TextureId &outId, size_t width,
                                           size_t height, size_t elementSize,
                                           const TextureDescriptor &desc) {
  if (width == 0 || elementSize == 0)
    return VGREResult::ERR_INVALID_VALUE;

  size_t h = (height == 0) ? 1 : height;
  size_t totalBytes = width * h * elementSize;

  // Allocate managed backing memory
  std::vector<uint8_t> backing(totalBytes, 0);

  std::lock_guard<std::recursive_mutex> lock(mutex_);

  TextureObject tex;
  tex.id = nextTextureId_++;
  tex.offsetInBytes = 0;
  tex.width = width;
  tex.height = h;
  tex.depth = 1;
  tex.elementSize = elementSize;
  tex.desc = desc;

  // Store the owned backing buffer first, then point the texture to it
  ownedArrays_[tex.id] = std::move(backing);
  tex.data = ownedArrays_[tex.id].data();

  textures_[tex.id] = tex;
  outId = tex.id;

  VGRE_LOG_INFO("TextureManager",
                "Created cudaArray " + std::to_string(tex.id) + " (" +
                    std::to_string(width) + "x" + std::to_string(h) +
                    ", " + std::to_string(totalBytes) + " bytes owned)");

  return VGREResult::SUCCESS;
}

VGREResult TextureManager::createCudaArray3D(TextureId &outId, size_t width,
                                             size_t height, size_t depth,
                                             size_t elementSize,
                                             const TextureDescriptor &desc) {
  if (width == 0 || height == 0 || depth == 0 || elementSize == 0)
    return VGREResult::ERR_INVALID_VALUE;

  size_t totalBytes = width * height * depth * elementSize;

  // Allocate managed backing memory
  std::vector<uint8_t> backing(totalBytes, 0);

  std::lock_guard<std::recursive_mutex> lock(mutex_);

  TextureObject tex;
  tex.id = nextTextureId_++;
  tex.offsetInBytes = 0;
  tex.width = width;
  tex.height = height;
  tex.depth = depth;
  tex.elementSize = elementSize;
  tex.desc = desc;

  // Store the owned backing buffer first, then point the texture to it
  ownedArrays_[tex.id] = std::move(backing);
  tex.data = ownedArrays_[tex.id].data();

  textures_[tex.id] = tex;
  outId = tex.id;

  VGRE_LOG_INFO("TextureManager",
                "Created 3D cudaArray " + std::to_string(tex.id) + " (" +
                    std::to_string(width) + "x" + std::to_string(height) +
                    "x" + std::to_string(depth) + ", " +
                    std::to_string(totalBytes) + " bytes owned)");

  return VGREResult::SUCCESS;
}

VGREResult TextureManager::destroyCudaArray(TextureId id) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  auto texIt = textures_.find(id);
  auto arrIt = ownedArrays_.find(id);

  if (texIt == textures_.end() && arrIt == ownedArrays_.end())
    return VGREResult::ERR_INVALID_VALUE;

  if (texIt != textures_.end())
    textures_.erase(texIt);
  if (arrIt != ownedArrays_.end())
    ownedArrays_.erase(arrIt);

  VGRE_LOG_DEBUG("TextureManager",
                 "Destroyed cudaArray " + std::to_string(id));
  return VGREResult::SUCCESS;
}

void *TextureManager::getCudaArrayData(TextureId id) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  auto it = ownedArrays_.find(id);
  if (it == ownedArrays_.end()) return nullptr;
  return it->second.data();
}

bool TextureManager::getCudaArrayInfo(TextureId id, ArrayInfo &out) const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  auto arrIt = ownedArrays_.find(id);
  if (arrIt == ownedArrays_.end()) return false;
  auto texIt = textures_.find(id);
  if (texIt == textures_.end()) return false;
  const TextureObject &tex = texIt->second;
  out.width       = tex.width;
  out.height      = (tex.height > 0) ? tex.height : 1;
  out.depth       = (tex.depth  > 0) ? tex.depth  : 1;
  out.elementSize = tex.elementSize;
  out.elementType = tex.desc.elementType;
  return true;
}

const void *TextureManager::getCudaArrayData(TextureId id) const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  auto it = ownedArrays_.find(id);
  if (it == ownedArrays_.end()) return nullptr;
  return it->second.data();
}

} // namespace core
} // namespace vgre
