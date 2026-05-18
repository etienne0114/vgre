#include "vgre/core/texture_manager.h"
#include "vgre/common/logger.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace vgre {
namespace core {

// Forward declaration (defined in texture_manager_lifecycle.cpp)
float cubicWeight(float x);


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
// ── SRGB gamma decode helper ──────────────────────────────────────────────────
// Converts a normalised [0,1] sRGB value to linear light per IEC 61966-2-1.
static double srgbToLinear(double v) {
  if (v <= 0.04045) return v / 12.92;
  return std::pow((v + 0.055) / 1.055, 2.4);
}

// ── Type-aware element read ───────────────────────────────────────────────────
double TextureManager::readElementValue(const TextureObject &tex, size_t linearIndex) const {
  const uint8_t *base = static_cast<const uint8_t *>(tex.data);
  const uint8_t *elem = base + tex.offsetInBytes + linearIndex * tex.elementSize;

  double raw = 0.0;
  switch (tex.desc.elementType) {
  case TextureElementType::FLOAT32:
    raw = static_cast<double>(*reinterpret_cast<const float *>(elem));
    break;
  case TextureElementType::FLOAT64:
    raw = *reinterpret_cast<const double *>(elem);
    break;
  case TextureElementType::INT8:
    raw = static_cast<double>(*reinterpret_cast<const int8_t *>(elem));
    break;
  case TextureElementType::INT16:
    raw = static_cast<double>(*reinterpret_cast<const int16_t *>(elem));
    break;
  case TextureElementType::INT32:
    raw = static_cast<double>(*reinterpret_cast<const int32_t *>(elem));
    break;
  case TextureElementType::UINT8:
    raw = static_cast<double>(*reinterpret_cast<const uint8_t *>(elem));
    if (tex.desc.srgbDecode) raw = srgbToLinear(raw / 255.0) * 255.0;
    break;
  case TextureElementType::UINT16:
    raw = static_cast<double>(*reinterpret_cast<const uint16_t *>(elem));
    if (tex.desc.srgbDecode) raw = srgbToLinear(raw / 65535.0) * 65535.0;
    break;
  case TextureElementType::UINT32:
    raw = static_cast<double>(*reinterpret_cast<const uint32_t *>(elem));
    break;
  }
  return raw;
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

// ── Per-channel element read ──────────────────────────────────────────────────
// Reads the nth float-channel within a packed vector element (e.g. float4).
// For FLOAT32: element layout is [ch0 f32, ch1 f32, ...].
// For all other element types channel 0 is the whole element; ch>0 returns 0.
float TextureManager::readElementChannel(const TextureObject &tex,
                                          size_t linearIndex,
                                          unsigned channel) const {
  if (!tex.data) return 0.0f;
  const uint8_t *base = static_cast<const uint8_t *>(tex.data);
  const uint8_t *elem = base + tex.offsetInBytes + linearIndex * tex.elementSize;

  if (tex.desc.elementType == TextureElementType::FLOAT32) {
    unsigned numCh = static_cast<unsigned>(tex.elementSize / sizeof(float));
    if (numCh == 0) numCh = 1;
    if (channel >= numCh) return 0.0f;
    return reinterpret_cast<const float *>(elem)[channel];
  }
  // For non-float element types multi-channel packing is not supported;
  // channel 0 returns the scalar value, higher channels return 0.
  if (channel > 0) return 0.0f;
  return readElementAsFloat(tex, linearIndex);
}

// ── Per-channel texel sample ──────────────────────────────────────────────────
float TextureManager::sampleTexelChan(const TextureObject &tex,
                                       int x, int y, int z,
                                       unsigned channel) const {
  if (tex.width == 0) return 0.0f;

  int rx = applyAddressMode(x, static_cast<int>(tex.width),  tex.desc.addressMode);
  int ry = applyAddressMode(y, static_cast<int>(tex.height), tex.desc.addressMode);
  int rz = applyAddressMode(z, static_cast<int>(tex.depth),  tex.desc.addressMode);

  if (rx < 0 || ry < 0 || rz < 0)
    return tex.desc.borderColor;

  size_t linear = static_cast<size_t>(rz) * tex.width * tex.height +
                  static_cast<size_t>(ry) * tex.width +
                  static_cast<size_t>(rx);

  return readElementChannel(tex, linear, channel);
}

// ── getTextureInfo ────────────────────────────────────────────────────────────
bool TextureManager::getTextureInfo(TextureId id, TextureInfo &out) const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  auto it = textures_.find(id);
  if (it == textures_.end()) return false;
  const auto &tex = it->second;
  out.width  = tex.width;
  out.height = (tex.height == 0) ? 1 : tex.height;
  out.depth  = (tex.depth  == 0) ? 1 : tex.depth;
  if (tex.desc.elementType == TextureElementType::FLOAT32 && tex.elementSize > sizeof(float)) {
    out.channels = static_cast<unsigned>(tex.elementSize / sizeof(float));
  } else {
    out.channels = 1;
  }
  return true;
}

// ── Per-channel tex1D/2D/3D ──────────────────────────────────────────────────
float TextureManager::tex2DChan(TextureId id, float x, float y,
                                 unsigned channel) const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  auto it = textures_.find(id);
  if (it == textures_.end()) return 0.0f;
  const auto &tex = it->second;

  int w = static_cast<int>(tex.width);
  int h = static_cast<int>(tex.height);
  float sx = tex.desc.normalizedCoords ? x * static_cast<float>(w) : x;
  float sy = tex.desc.normalizedCoords ? y * static_cast<float>(h) : y;

  if (tex.desc.filterMode == TextureFilterMode::POINT) {
    int ix = static_cast<int>(std::floor(sx));
    int iy = static_cast<int>(std::floor(sy));
    return sampleTexelChan(tex, ix, iy, 0, channel);
  }

  // Bilinear interpolation per-channel
  float fx = sx - 0.5f, fy = sy - 0.5f;
  int x0 = static_cast<int>(std::floor(fx));
  int y0 = static_cast<int>(std::floor(fy));
  float frX = fx - static_cast<float>(x0);
  float frY = fy - static_cast<float>(y0);
  float v00 = sampleTexelChan(tex, x0,     y0,     0, channel);
  float v10 = sampleTexelChan(tex, x0 + 1, y0,     0, channel);
  float v01 = sampleTexelChan(tex, x0,     y0 + 1, 0, channel);
  float v11 = sampleTexelChan(tex, x0 + 1, y0 + 1, 0, channel);
  float top = v00 * (1.0f - frX) + v10 * frX;
  float bot = v01 * (1.0f - frX) + v11 * frX;
  return top * (1.0f - frY) + bot * frY;
}

float TextureManager::tex1DChan(TextureId id, float x, unsigned channel) const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  auto it = textures_.find(id);
  if (it == textures_.end()) return 0.0f;
  const auto &tex = it->second;
  int w  = static_cast<int>(tex.width);
  float sx = tex.desc.normalizedCoords ? x * static_cast<float>(w) : x;
  int ix = static_cast<int>(std::floor(sx));
  return sampleTexelChan(tex, ix, 0, 0, channel);
}

float TextureManager::tex3DChan(TextureId id, float x, float y, float z,
                                 unsigned channel) const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  auto it = textures_.find(id);
  if (it == textures_.end()) return 0.0f;
  const auto &tex = it->second;
  int w = static_cast<int>(tex.width);
  int h = static_cast<int>(tex.height);
  int d = static_cast<int>(tex.depth);
  float sx = tex.desc.normalizedCoords ? x * static_cast<float>(w) : x;
  float sy = tex.desc.normalizedCoords ? y * static_cast<float>(h) : y;
  float sz = tex.desc.normalizedCoords ? z * static_cast<float>(d) : z;
  int ix = static_cast<int>(std::floor(sx));
  int iy = static_cast<int>(std::floor(sy));
  int iz = static_cast<int>(std::floor(sz));
  return sampleTexelChan(tex, ix, iy, iz, channel);
}

} // namespace core
} // namespace vgre
