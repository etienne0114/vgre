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

// ── 1D texture fetch (integer coordinate) ──────────────────────────────────
float TextureManager::tex1Dfetch(TextureId id, int x) const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  auto it = textures_.find(id);
  if (it == textures_.end())
    return 0.0f;

  const auto &tex = it->second;
  double value =
      sampleTexel(tex, x, 0, 0);
  return static_cast<float>(value);
}

// ── 2D texture fetch (floating-point coordinates with interpolation) ──────
float TextureManager::tex2D(TextureId id, float x, float y) const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  auto it = textures_.find(id);
  if (it == textures_.end())
    return 0.0f;

  const auto &tex = it->second;
  int w = static_cast<int>(tex.width);
  int h = static_cast<int>(tex.height);

  float sampleX = x;
  float sampleY = y;
  if (tex.desc.normalizedCoords) {
    sampleX *= static_cast<float>(w);
    sampleY *= static_cast<float>(h);
  }

  if (tex.desc.filterMode == TextureFilterMode::POINT) {
    // Nearest-neighbor sampling
    int ix = static_cast<int>(std::floor(sampleX));
    int iy = static_cast<int>(std::floor(sampleY));

    double value = sampleTexel(tex, ix, iy, 0);
    return static_cast<float>(value);
  }

  // Bilinear helper: bilinear sample at (bx, by) in texel space.
  auto bilinearSample = [&](float bx, float by) -> double {
    float fx = bx - 0.5f;
    float fy = by - 0.5f;
    int x0 = static_cast<int>(std::floor(fx));
    int y0 = static_cast<int>(std::floor(fy));
    float fracX = fx - static_cast<float>(x0);
    float fracY = fy - static_cast<float>(y0);
    double v00 = sampleTexel(tex, x0,     y0,     0);
    double v10 = sampleTexel(tex, x0 + 1, y0,     0);
    double v01 = sampleTexel(tex, x0,     y0 + 1, 0);
    double v11 = sampleTexel(tex, x0 + 1, y0 + 1, 0);
    double top = v00 * (1.0 - fracX) + v10 * fracX;
    double bot = v01 * (1.0 - fracX) + v11 * fracX;
    return top * (1.0 - fracY) + bot * fracY;
  };

  if (tex.desc.filterMode == TextureFilterMode::LINEAR) {
    return static_cast<float>(bilinearSample(sampleX, sampleY));
  }

  if (tex.desc.filterMode == TextureFilterMode::ANISOTROPIC ||
      (tex.desc.filterMode == TextureFilterMode::LINEAR &&
       tex.desc.maxAnisotropy > 1)) {
    // Anisotropic filtering via multi-sample line averaging.
    // Without per-pixel derivative information, we approximate by gathering
    // maxAnisotropy bilinear samples along both U and V axes and averaging them.
    // This reduces aliasing along surfaces viewed at a glancing angle.
    unsigned int N = std::max(1u, std::min(tex.desc.maxAnisotropy, 16u));
    if (N == 1) return static_cast<float>(bilinearSample(sampleX, sampleY));

    // Sample N points along X and N along Y, centred on (sampleX, sampleY).
    // Step is sub-texel: spacing = 1/(N) texels.
    double stepX = 1.0 / static_cast<double>(N);
    double stepY = 1.0 / static_cast<double>(N);

    double accumX = 0.0, accumY = 0.0;
    double startX = sampleX - 0.5 * stepX * static_cast<double>(N - 1);
    double startY = sampleY - 0.5 * stepY * static_cast<double>(N - 1);
    for (unsigned int i = 0; i < N; ++i) {
      accumX += bilinearSample(static_cast<float>(startX + stepX * i), sampleY);
      accumY += bilinearSample(sampleX, static_cast<float>(startY + stepY * i));
    }
    // Blend X-pass and Y-pass evenly.
    return static_cast<float>((accumX + accumY) / (2.0 * N));
  }

  // Bicubic (Catmull-Rom) interpolation - 16 samples (4x4 neighborhood)
  float fx = sampleX - 0.5f;
  float fy = sampleY - 0.5f;
  int x1 = static_cast<int>(std::floor(fx));
  int y1 = static_cast<int>(std::floor(fy));
  float fracX = fx - static_cast<float>(x1);
  float fracY = fy - static_cast<float>(y1);

  auto sample = [&](int sx, int sy) -> double {
    return sampleTexel(tex, sx, sy, 0);
  };

  // Sample 4x4 neighborhood and apply Catmull-Rom weights
  double result = 0.0;
  for (int dy = -1; dy <= 2; dy++) {
    double rowSum = 0.0;
    float wy = cubicWeight(fracY - static_cast<float>(dy));
    for (int dx = -1; dx <= 2; dx++) {
      float wx = cubicWeight(fracX - static_cast<float>(dx));
      rowSum += sample(x1 + dx, y1 + dy) * wx;
    }
    result += rowSum * wy;
  }

  return static_cast<float>(result);
}

// ── 2D texture fetch at explicit LOD (trilinear across mip levels) ──────────
float TextureManager::tex2DLod(TextureId id, float x, float y, float lod) const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);

  auto texIt = textures_.find(id);
  if (texIt == textures_.end()) return 0.0f;
  const auto &tex = texIt->second;

  auto arrIt = ownedArrays_.find(id);
  auto offIt = mipmapLevelOffsets_.find(id);

  // Not a mipmapped texture — bilinear sample from base level (no unlock needed,
  // mutex_ is recursive).
  if (arrIt == ownedArrays_.end() || offIt == mipmapLevelOffsets_.end() ||
      tex.mips <= 1) {
    // Delegate to the regular tex2D logic inline (avoids re-locking the recursive mutex).
    int w = static_cast<int>(tex.width);
    int h = static_cast<int>(tex.height);
    float sx = tex.desc.normalizedCoords ? x * static_cast<float>(w) : x;
    float sy = tex.desc.normalizedCoords ? y * static_cast<float>(h) : y;
    float fx = sx - 0.5f, fy = sy - 0.5f;
    int x0 = static_cast<int>(std::floor(fx));
    int y0 = static_cast<int>(std::floor(fy));
    float fracX = fx - static_cast<float>(x0);
    float fracY = fy - static_cast<float>(y0);
    double v00 = sampleTexel(tex, x0,     y0,     0);
    double v10 = sampleTexel(tex, x0 + 1, y0,     0);
    double v01 = sampleTexel(tex, x0,     y0 + 1, 0);
    double v11 = sampleTexel(tex, x0 + 1, y0 + 1, 0);
    double top = v00 * (1.0 - fracX) + v10 * fracX;
    double bot = v01 * (1.0 - fracX) + v11 * fracX;
    return static_cast<float>(top * (1.0 - fracY) + bot * fracY);
  }

  // Clamp LOD to valid mip range.
  float clampedLod = std::max(0.0f, std::min(lod, static_cast<float>(tex.mips - 1)));
  int   level0     = static_cast<int>(std::floor(clampedLod));
  int   level1     = std::min(level0 + 1, static_cast<int>(tex.mips - 1));
  float blend      = clampedLod - static_cast<float>(level0);

  const std::vector<uint8_t> &arr  = arrIt->second;
  const std::vector<size_t>  &offs = offIt->second;

  // Bilinear sample from a specific mip level.
  // Pixel coordinates (px, py) are in texel space of the BASE texture; they
  // are divided by 2^level to land in the correct level's texel space.
  auto sampleLevel = [&](int lvl, float bx, float by) -> double {
    size_t lw  = std::max<size_t>(1, tex.width  >> lvl);
    size_t lh  = std::max<size_t>(1, tex.height >> lvl);
    float  fx  = bx - 0.5f;
    float  fy  = by - 0.5f;
    int    x0l = static_cast<int>(std::floor(fx));
    int    y0l = static_cast<int>(std::floor(fy));
    float  frX = fx - static_cast<float>(x0l);
    float  frY = fy - static_cast<float>(y0l);

    const float *data =
        reinterpret_cast<const float *>(arr.data() + offs[static_cast<size_t>(lvl)]);

    auto readPx = [&](int px, int py) -> double {
      px = applyAddressMode(px, static_cast<int>(lw), tex.desc.addressMode);
      py = applyAddressMode(py, static_cast<int>(lh), tex.desc.addressMode);
      if (px < 0 || py < 0 ||
          static_cast<size_t>(px) >= lw || static_cast<size_t>(py) >= lh)
        return static_cast<double>(tex.desc.borderColor);
      return static_cast<double>(
          data[static_cast<size_t>(py) * lw + static_cast<size_t>(px)]);
    };

    double v00 = readPx(x0l,     y0l);
    double v10 = readPx(x0l + 1, y0l);
    double v01 = readPx(x0l,     y0l + 1);
    double v11 = readPx(x0l + 1, y0l + 1);
    double top = v00 * (1.0 - frX) + v10 * frX;
    double bot = v01 * (1.0 - frX) + v11 * frX;
    return top * (1.0 - frY) + bot * frY;
  };

  // Convert input (x,y) to texel coordinates for level0.
  // Normalized: multiply by level-0 dimensions.
  // Non-normalized: coords are in base-level texel space; scale by 1/2^level.
  float scale0 = static_cast<float>(1u << level0); // 2^level0
  float sx0 = tex.desc.normalizedCoords
                   ? x * static_cast<float>(std::max<size_t>(1, tex.width  >> level0))
                   : x / scale0;
  float sy0 = tex.desc.normalizedCoords
                   ? y * static_cast<float>(std::max<size_t>(1, tex.height >> level0))
                   : y / scale0;

  double s0 = sampleLevel(level0, sx0, sy0);

  if (blend < 1e-5f || level0 == level1)
    return static_cast<float>(s0);

  // Sample from the next finer mip level and blend (trilinear).
  float scale1 = static_cast<float>(1u << level1);
  float sx1 = tex.desc.normalizedCoords
                   ? x * static_cast<float>(std::max<size_t>(1, tex.width  >> level1))
                   : x / scale1;
  float sy1 = tex.desc.normalizedCoords
                   ? y * static_cast<float>(std::max<size_t>(1, tex.height >> level1))
                   : y / scale1;

  double s1 = sampleLevel(level1, sx1, sy1);
  return static_cast<float>(s0 * (1.0 - blend) + s1 * blend);
}

float TextureManager::tex1D(TextureId id, float x) const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  auto it = textures_.find(id);
  if (it == textures_.end())
    return 0.0f;

  const auto &tex = it->second;
  int w = static_cast<int>(tex.width);

  if (tex.desc.normalizedCoords) {
    x *= static_cast<float>(w);
  }

  if (tex.desc.filterMode == TextureFilterMode::POINT) {
    int ix = static_cast<int>(std::floor(x));
    double value = sampleTexel(tex, ix, 0, 0);
    return static_cast<float>(value);
  }

  if (tex.desc.filterMode == TextureFilterMode::LINEAR) {
    // Linear interpolation
    float fx = x - 0.5f;
    int x0 = static_cast<int>(std::floor(fx));
    int x1 = x0 + 1;
    float fracX = fx - static_cast<float>(x0);

    auto sample = [&](int sx) -> double {
      return sampleTexel(tex, sx, 0, 0);
    };

    double v0 = sample(x0);
    double v1 = sample(x1);
    return static_cast<float>(v0 * (1.0 - fracX) + v1 * fracX);
  }

  // Cubic (Catmull-Rom) interpolation - 4 samples
  float fx = x - 0.5f;
  int x1 = static_cast<int>(std::floor(fx));
  int x0 = x1 - 1;
  int x2 = x1 + 1;
  int x3 = x1 + 2;
  float fracX = fx - static_cast<float>(x1);

  auto sample = [&](int sx) -> double {
    return sampleTexel(tex, sx, 0, 0);
  };

  // Apply Catmull-Rom weights
  double v0 = sample(x0) * cubicWeight(fracX + 1.0f);
  double v1 = sample(x1) * cubicWeight(fracX);
  double v2 = sample(x2) * cubicWeight(fracX - 1.0f);
  double v3 = sample(x3) * cubicWeight(fracX - 2.0f);

  return static_cast<float>(v0 + v1 + v2 + v3);
}

// ── Surface write ──────────────────────────────────────────────────────────
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
  static TextureManager* mgr = new TextureManager();
  return *mgr;
}

// ── Type-aware element read ───────────────────────────────────────────────────
double TextureManager::readElementValue(const TextureObject &tex, size_t linearIndex) const {
  const uint8_t *base = static_cast<const uint8_t *>(tex.data);
  const uint8_t *elem = base + tex.offsetInBytes + linearIndex * tex.elementSize;

  switch (tex.desc.elementType) {
  case TextureElementType::FLOAT32:
    return static_cast<double>(*reinterpret_cast<const float *>(elem));
  case TextureElementType::FLOAT64:
    return *reinterpret_cast<const double *>(elem);
  case TextureElementType::INT8:
    return static_cast<double>(*reinterpret_cast<const int8_t *>(elem));
  case TextureElementType::INT16:
    return static_cast<double>(*reinterpret_cast<const int16_t *>(elem));
  case TextureElementType::INT32:
    return static_cast<double>(*reinterpret_cast<const int32_t *>(elem));
  case TextureElementType::UINT8:
    return static_cast<double>(*reinterpret_cast<const uint8_t *>(elem));
  case TextureElementType::UINT16:
    return static_cast<double>(*reinterpret_cast<const uint16_t *>(elem));
  case TextureElementType::UINT32:
    return static_cast<double>(*reinterpret_cast<const uint32_t *>(elem));
  }
  return 0.0;
}

double TextureManager::readElementAsDouble(TextureId id, int linearIndex) const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  auto it = textures_.find(id);
  if (it == textures_.end()) return 0.0;

  const auto &tex = it->second;
  size_t totalElements = tex.width * tex.height * tex.depth;
  if (linearIndex < 0 || static_cast<size_t>(linearIndex) >= totalElements)
    return 0.0;

  if (!tex.data) return 0.0;

  return readElementValue(tex, static_cast<size_t>(linearIndex));
}

// ── Read element as float (no internal locking) ──────────────────────────────
float TextureManager::readElementAsFloat(const TextureObject &tex, size_t linearIndex) const {
  size_t totalElements = tex.width * tex.height * tex.depth;
  if (linearIndex >= totalElements)
    return 0.0f;

  if (!tex.data) return 0.0f;

  const uint8_t *base = static_cast<const uint8_t *>(tex.data);
  const uint8_t *elem = base + tex.offsetInBytes + linearIndex * tex.elementSize;

  switch (tex.desc.elementType) {
  case TextureElementType::FLOAT32:
    return *reinterpret_cast<const float *>(elem);
  case TextureElementType::FLOAT64:
    return static_cast<float>(*reinterpret_cast<const double *>(elem));
  case TextureElementType::INT8:
    return static_cast<float>(*reinterpret_cast<const int8_t *>(elem));
  case TextureElementType::INT16:
    return static_cast<float>(*reinterpret_cast<const int16_t *>(elem));
  case TextureElementType::INT32:
    return static_cast<float>(*reinterpret_cast<const int32_t *>(elem));
  case TextureElementType::UINT8:
    return static_cast<float>(*reinterpret_cast<const uint8_t *>(elem));
  case TextureElementType::UINT16:
    return static_cast<float>(*reinterpret_cast<const uint16_t *>(elem));
  case TextureElementType::UINT32:
    return static_cast<float>(*reinterpret_cast<const uint32_t *>(elem));
  }
  return 0.0f;
}

double TextureManager::sampleTexel(const TextureObject &tex, int x, int y, int z) const {
  if (tex.width == 0 || tex.height == 0 || tex.depth == 0)
    return tex.desc.borderColor;

  int rx = applyAddressMode(x, static_cast<int>(tex.width), tex.desc.addressMode);
  int ry = applyAddressMode(y, static_cast<int>(tex.height), tex.desc.addressMode);
  int rz = applyAddressMode(z, static_cast<int>(tex.depth), tex.desc.addressMode);

  if (rx < 0 || ry < 0 || rz < 0)
    return tex.desc.borderColor;

  size_t linear = static_cast<size_t>(rz) * tex.width * tex.height +
                  static_cast<size_t>(ry) * tex.width +
                  static_cast<size_t>(rx);

  return readElementValue(tex, linear);
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

const void *TextureManager::getCudaArrayData(TextureId id) const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  auto it = ownedArrays_.find(id);
  if (it == ownedArrays_.end()) return nullptr;
  return it->second.data();
}

// ── Mipmapped array pipeline ──────────────────────────────────────────────

VGREResult TextureManager::createMipmappedArray(TextureId &outId,
                                                 size_t width, size_t height,
                                                 size_t elementSize,
                                                 unsigned int mipLevels,
                                                 const TextureDescriptor &desc) {
    if (width == 0 || elementSize == 0)
        return VGREResult::ERR_INVALID_VALUE;

    size_t h = (height == 0) ? 1 : height;

    // Auto-determine level count if not specified
    if (mipLevels == 0) {
        size_t maxDim = std::max(width, h);
        mipLevels = 1;
        while ((maxDim >> mipLevels) > 0) ++mipLevels;
    }

    // Compute total storage and per-level offsets
    std::vector<size_t> offsets;
    offsets.reserve(mipLevels);
    size_t totalBytes = 0;
    size_t lw = width, lh = h;
    for (unsigned int m = 0; m < mipLevels; ++m) {
        offsets.push_back(totalBytes);
        totalBytes += lw * lh * elementSize;
        lw = std::max<size_t>(1, lw >> 1);
        lh = std::max<size_t>(1, lh >> 1);
    }

    std::lock_guard<std::recursive_mutex> lock(mutex_);

    TextureObject tex;
    tex.id = nextTextureId_++;
    tex.offsetInBytes = 0;
    tex.width = width;
    tex.height = h;
    tex.depth = 1;
    tex.elementSize = elementSize;
    tex.mips = mipLevels;
    tex.desc = desc;

    ownedArrays_[tex.id] = std::vector<uint8_t>(totalBytes, 0);
    tex.data = ownedArrays_[tex.id].data();
    mipmapLevelOffsets_[tex.id] = std::move(offsets);

    textures_[tex.id] = tex;
    outId = tex.id;

    VGRE_LOG_INFO("TextureManager",
                  "Created mipmapped array " + std::to_string(tex.id) +
                  " (" + std::to_string(width) + "x" + std::to_string(h) +
                  ", " + std::to_string(mipLevels) + " levels, " +
                  std::to_string(totalBytes) + " bytes)");

    return VGREResult::SUCCESS;
}

// Box-filter 2×2 downscale of float32 data.
// src is (sw × sh) floats; dst is (dw × dh) floats where dw = sw/2, dh = sh/2.
static void boxFilter2D(const float *src, size_t sw, size_t sh,
                        float *dst, size_t dw, size_t dh) {
    for (size_t y = 0; y < dh; ++y) {
        for (size_t x = 0; x < dw; ++x) {
            size_t sx = x * 2, sy = y * 2;
            // Gather 2×2 neighbourhood (clamp at edges)
            float s00 = src[sy * sw + sx];
            float s10 = (sx + 1 < sw) ? src[sy * sw + sx + 1] : s00;
            float s01 = (sy + 1 < sh) ? src[(sy + 1) * sw + sx] : s00;
            float s11 = (sx + 1 < sw && sy + 1 < sh)
                            ? src[(sy + 1) * sw + sx + 1]
                            : s00;
            dst[y * dw + x] = (s00 + s10 + s01 + s11) * 0.25f;
        }
    }
}

VGREResult TextureManager::generateMipmaps(TextureId id) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);

    auto texIt = textures_.find(id);
    auto arrIt = ownedArrays_.find(id);
    auto offIt = mipmapLevelOffsets_.find(id);

    if (texIt == textures_.end() || arrIt == ownedArrays_.end() ||
        offIt == mipmapLevelOffsets_.end()) {
        VGRE_LOG_ERROR("TextureManager",
                       "generateMipmaps: id " + std::to_string(id) +
                       " is not a mipmapped array");
        return VGREResult::ERR_INVALID_VALUE;
    }

    const TextureObject &tex = texIt->second;
    if (tex.elementSize != sizeof(float)) {
        VGRE_LOG_WARN("TextureManager",
                      "generateMipmaps: only float32 textures supported; "
                      "element size = " + std::to_string(tex.elementSize));
        return VGREResult::ERR_NOT_SUPPORTED;
    }

    uint8_t *base = arrIt->second.data();
    const std::vector<size_t> &offs = offIt->second;
    unsigned int levels = tex.mips;

    size_t lw = tex.width, lh = tex.height;
    for (unsigned int m = 1; m < levels; ++m) {
        size_t dw = std::max<size_t>(1, lw >> 1);
        size_t dh = std::max<size_t>(1, lh >> 1);

        const float *src = reinterpret_cast<const float *>(base + offs[m - 1]);
        float *dst = reinterpret_cast<float *>(base + offs[m]);
        boxFilter2D(src, lw, lh, dst, dw, dh);

        lw = dw;
        lh = dh;
    }

    VGRE_LOG_INFO("TextureManager",
                  "Generated " + std::to_string(levels - 1) +
                  " mip levels for texture " + std::to_string(id));
    return VGREResult::SUCCESS;
}

void *TextureManager::getMipmapLevelData(TextureId id, unsigned int level) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    auto arrIt = ownedArrays_.find(id);
    auto offIt = mipmapLevelOffsets_.find(id);
    if (arrIt == ownedArrays_.end() || offIt == mipmapLevelOffsets_.end())
        return nullptr;
    if (level >= offIt->second.size()) return nullptr;
    return arrIt->second.data() + offIt->second[level];
}

const void *TextureManager::getMipmapLevelData(TextureId id, unsigned int level) const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    auto arrIt = ownedArrays_.find(id);
    auto offIt = mipmapLevelOffsets_.find(id);
    if (arrIt == ownedArrays_.end() || offIt == mipmapLevelOffsets_.end())
        return nullptr;
    if (level >= offIt->second.size()) return nullptr;
    return arrIt->second.data() + offIt->second[level];
}

} // namespace core
} // namespace vgre
