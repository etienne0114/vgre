#ifndef VGRE_CORE_TEXTURE_MANAGER_H
#define VGRE_CORE_TEXTURE_MANAGER_H

#include "vgre/common/error_codes.h"
#include "vgre/common/types.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace vgre {
namespace core {

// ── Texture addressing and filter modes ────────────────────────────────────
enum class TextureAddressMode : uint8_t { CLAMP, WRAP, MIRROR, BORDER };

enum class TextureFilterMode : uint8_t { POINT, LINEAR, CUBIC };

// ── Texture element types ──────────────────────────────────────────────────
enum class TextureElementType : uint8_t {
  FLOAT32,
  FLOAT64,
  INT8,
  INT16,
  INT32,
  UINT8,
  UINT16,
  UINT32
};

// ── Texture descriptor ─────────────────────────────────────────────────────
struct TextureDescriptor {
  TextureAddressMode addressMode = TextureAddressMode::CLAMP;
  TextureFilterMode filterMode = TextureFilterMode::POINT;
  TextureElementType elementType = TextureElementType::FLOAT32;
  bool normalizedCoords = false;
  float borderColor = 0.0f;
};

// ── Resource view descriptor ───────────────────────────────────────────────
struct ResourceViewDescriptor {
  TextureElementType format;
  size_t width;
  size_t height;
  size_t depth;
  size_t offsetInBytes = 0;
  unsigned int firstMipmapLevel = 0;
  unsigned int lastMipmapLevel = 0;
  unsigned int firstLayer = 0;
  unsigned int lastLayer = 0;
};

// ── Texture object ─────────────────────────────────────────────────────────
using TextureId = uint64_t;

struct TextureObject {
  TextureId id = 0;
  const void *data = nullptr; // Pointer to underlying memory
  size_t offsetInBytes = 0;   // Offset into data for views
  size_t width = 0;           // Width in elements
  size_t height = 0;          // Height in elements (1 for 1D textures)
  size_t depth = 0;           // Depth in elements (1 for 1D/2D textures)
  size_t elementSize = 0;     // Size per element in bytes
  unsigned int layers = 1;    // Number of layers for array textures
  unsigned int mips = 1;      // Number of mipmap levels
  TextureDescriptor desc;
};

// ── Surface object ─────────────────────────────────────────────────────────
using SurfaceId = uint64_t;

struct SurfaceObject {
  SurfaceId id = 0;
  void *data = nullptr; // Writable memory
  size_t width = 0;
  size_t height = 0;
  size_t elementSize = 0;
  TextureElementType elementType = TextureElementType::FLOAT32;
};

// ── Texture Manager ────────────────────────────────────────────────────────
// Provides CUDA-compatible texture/surface memory APIs with CPU-side
// bilinear interpolation and configurable addressing modes.
class TextureManager {
public:
  TextureManager();
  ~TextureManager();

  // ── Texture object lifecycle ─────────────────────────────────────────────
  VGREResult createTexture(TextureId &outId, const void *data, size_t width,
                           size_t height, size_t elementSize,
                           const TextureDescriptor &desc,
                           unsigned int layers = 1);

  // Create a view of an existing texture with a different format/size/offset
  VGREResult createTextureView(TextureId &outId, TextureId baseTextureId,
                               const ResourceViewDescriptor &viewDesc,
                               const TextureDescriptor &texDesc);

  VGREResult destroyTexture(TextureId id);

  // ── Surface object lifecycle ─────────────────────────────────────────────
  VGREResult createSurface(SurfaceId &outId, void *data, size_t width,
                           size_t height, size_t elementSize,
                           TextureElementType elementType = TextureElementType::FLOAT32);
  VGREResult destroySurface(SurfaceId id);

  // ── Texture fetch operations ─────────────────────────────────────────────
  // 1D integer-coordinate fetch (like tex1Dfetch)
  float tex1Dfetch(TextureId id, int x) const;

  // 1D texture fetch with optional interpolation (like tex1D)
  float tex1D(TextureId id, float x) const;

  // 2D floating-point fetch with optional interpolation (like tex2D)
  float tex2D(TextureId id, float x, float y) const;

  // 3D floating-point fetch with optional trilinear interpolation (like tex3D)
  float tex3D(TextureId id, float x, float y, float z) const;

  // ── Type-aware fetch operations ───────────────────────────────────────────
  // Read a single element as double (supports all element types)
  double readElementAsDouble(TextureId id, int linearIndex) const;
  
  // Read a single element as float from the raw data pointer based on its descriptor type
  float readElementAsFloat(const TextureObject &tex, size_t linearIndex) const;

  // ── Surface write operations ─────────────────────────────────────────────
  VGREResult surf2Dwrite(SurfaceId id, float value, int x, int y);
  VGREResult surf2Dread(SurfaceId id, float &value, int x, int y) const;

  // ── Queries ──────────────────────────────────────────────────────────────
  size_t getTextureCount() const;
  size_t getSurfaceCount() const;

  // ── 3D Texture creation (explicit depth) ─────────────────────────────────
  VGREResult createTexture3D(TextureId &outId, const void *data,
                             size_t width, size_t height, size_t depth,
                             size_t elementSize, const TextureDescriptor &desc);

  // ── cudaArray lifecycle (owns backing memory) ────────────────────────────
  // Allocates a managed memory block and wraps it as a texture for sampling.
  VGREResult createCudaArray(TextureId &outId, size_t width, size_t height,
                             size_t elementSize, const TextureDescriptor &desc);
  VGREResult destroyCudaArray(TextureId id);
  // Returns raw pointer to the cudaArray's owned backing memory for memcpy.
  void *getCudaArrayData(TextureId id);
  const void *getCudaArrayData(TextureId id) const;

  // ── Singleton ────────────────────────────────────────────────────────────
  static TextureManager &instance();

private:
  // Address mode helper: translates coordinates based on addressing mode
  int applyAddressMode(int coord, int size, TextureAddressMode mode) const;

  std::unordered_map<TextureId, TextureObject> textures_;
  std::unordered_map<SurfaceId, SurfaceObject> surfaces_;
  // Backing memory owned by cudaArray lifecycle (freed on destroyCudaArray)
  std::unordered_map<TextureId, std::vector<uint8_t>> ownedArrays_;
  mutable std::recursive_mutex mutex_;
  TextureId nextTextureId_ = 1;
  SurfaceId nextSurfaceId_ = 1;

  // ── Helpers ─────────────────────────────────────────────────────────────
  double readElementValue(const TextureObject &tex,
                          size_t linearIndex) const;
  double sampleTexel(const TextureObject &tex, int x, int y, int z) const;
};

} // namespace core
} // namespace vgre

#endif // VGRE_CORE_TEXTURE_MANAGER_H
