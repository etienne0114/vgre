#include "vgre/core/texture_manager.h"
#include "vgre/common/logger.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "vgre/common/openmp_helper.h"
#include "vgre/compiler/cpu_cuda_fp16.h"

namespace vgre {
namespace core {


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
    #ifdef _OPENMP
    #pragma omp parallel for collapse(2) if (dw * dh > 1024)
    #endif
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

static size_t baseTypeSize(TextureElementType t) {
    switch (t) {
        case TextureElementType::UINT8:
        case TextureElementType::INT8:     return 1;
        case TextureElementType::UINT16:
        case TextureElementType::INT16:
        case TextureElementType::FP16:      return 2;
        case TextureElementType::UINT32:
        case TextureElementType::INT32:
        case TextureElementType::FLOAT32:   return 4;
        default:                           return 0;
    }
}

static void channelToFloat(const uint8_t* src, size_t srcStride, size_t count,
                           TextureElementType type, float* dst) {
    switch (type) {
        case TextureElementType::FLOAT32:
            #ifdef _OPENMP
            #pragma omp parallel for if (count > 1024)
            #endif
            for (size_t i = 0; i < count; ++i)
                dst[i] = reinterpret_cast<const float*>(src + i * srcStride)[0];
            break;
        case TextureElementType::FP16:
            #ifdef _OPENMP
            #pragma omp parallel for if (count > 1024)
            #endif
            for (size_t i = 0; i < count; ++i)
                dst[i] = vgre_cuda::__half2float(reinterpret_cast<const vgre_cuda::__half*>(src + i * srcStride)[0]);
            break;
        case TextureElementType::INT8:
            #ifdef _OPENMP
            #pragma omp parallel for if (count > 1024)
            #endif
            for (size_t i = 0; i < count; ++i)
                dst[i] = static_cast<float>(reinterpret_cast<const int8_t*>(src + i * srcStride)[0]);
            break;
        case TextureElementType::INT16:
            #ifdef _OPENMP
            #pragma omp parallel for if (count > 1024)
            #endif
            for (size_t i = 0; i < count; ++i)
                dst[i] = static_cast<float>(reinterpret_cast<const int16_t*>(src + i * srcStride)[0]);
            break;
        case TextureElementType::INT32:
            #ifdef _OPENMP
            #pragma omp parallel for if (count > 1024)
            #endif
            for (size_t i = 0; i < count; ++i)
                dst[i] = static_cast<float>(reinterpret_cast<const int32_t*>(src + i * srcStride)[0]);
            break;
        case TextureElementType::UINT8:
            #ifdef _OPENMP
            #pragma omp parallel for if (count > 1024)
            #endif
            for (size_t i = 0; i < count; ++i)
                dst[i] = static_cast<float>(reinterpret_cast<const uint8_t*>(src + i * srcStride)[0]);
            break;
        case TextureElementType::UINT16:
            #ifdef _OPENMP
            #pragma omp parallel for if (count > 1024)
            #endif
            for (size_t i = 0; i < count; ++i)
                dst[i] = static_cast<float>(reinterpret_cast<const uint16_t*>(src + i * srcStride)[0]);
            break;
        case TextureElementType::UINT32:
            #ifdef _OPENMP
            #pragma omp parallel for if (count > 1024)
            #endif
            for (size_t i = 0; i < count; ++i)
                dst[i] = static_cast<float>(reinterpret_cast<const uint32_t*>(src + i * srcStride)[0]);
            break;
        default:
            break;
    }
}

static void floatToChannel(const float* src, uint8_t* dst, size_t dstStride,
                           size_t count, TextureElementType type) {
    switch (type) {
        case TextureElementType::FLOAT32:
            #ifdef _OPENMP
            #pragma omp parallel for if (count > 1024)
            #endif
            for (size_t i = 0; i < count; ++i)
                reinterpret_cast<float*>(dst + i * dstStride)[0] = src[i];
            break;
        case TextureElementType::FP16:
            #ifdef _OPENMP
            #pragma omp parallel for if (count > 1024)
            #endif
            for (size_t i = 0; i < count; ++i)
                reinterpret_cast<vgre_cuda::__half*>(dst + i * dstStride)[0] = vgre_cuda::__float2half(src[i]);
            break;
        case TextureElementType::INT8:
            #ifdef _OPENMP
            #pragma omp parallel for if (count > 1024)
            #endif
            for (size_t i = 0; i < count; ++i)
                reinterpret_cast<int8_t*>(dst + i * dstStride)[0] = static_cast<int8_t>(src[i]);
            break;
        case TextureElementType::INT16:
            #ifdef _OPENMP
            #pragma omp parallel for if (count > 1024)
            #endif
            for (size_t i = 0; i < count; ++i)
                reinterpret_cast<int16_t*>(dst + i * dstStride)[0] = static_cast<int16_t>(src[i]);
            break;
        case TextureElementType::INT32:
            #ifdef _OPENMP
            #pragma omp parallel for if (count > 1024)
            #endif
            for (size_t i = 0; i < count; ++i)
                reinterpret_cast<int32_t*>(dst + i * dstStride)[0] = static_cast<int32_t>(src[i]);
            break;
        case TextureElementType::UINT8:
            #ifdef _OPENMP
            #pragma omp parallel for if (count > 1024)
            #endif
            for (size_t i = 0; i < count; ++i)
                reinterpret_cast<uint8_t*>(dst + i * dstStride)[0] = static_cast<uint8_t>(src[i]);
            break;
        case TextureElementType::UINT16:
            #ifdef _OPENMP
            #pragma omp parallel for if (count > 1024)
            #endif
            for (size_t i = 0; i < count; ++i)
                reinterpret_cast<uint16_t*>(dst + i * dstStride)[0] = static_cast<uint16_t>(src[i]);
            break;
        case TextureElementType::UINT32:
            #ifdef _OPENMP
            #pragma omp parallel for if (count > 1024)
            #endif
            for (size_t i = 0; i < count; ++i)
                reinterpret_cast<uint32_t*>(dst + i * dstStride)[0] = static_cast<uint32_t>(src[i]);
            break;
        default:
            break;
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
    size_t bSize = baseTypeSize(tex.desc.elementType);
    if (bSize == 0 || tex.elementSize % bSize != 0) {
        VGRE_LOG_WARN("TextureManager",
                      "generateMipmaps: unsupported element size/type combo");
        return VGREResult::ERR_NOT_SUPPORTED;
    }

    uint8_t *base = arrIt->second.data();
    const std::vector<size_t> &offs = offIt->second;
    unsigned int levels = tex.mips;
    size_t channelCount = tex.elementSize / bSize;

    size_t lw = tex.width, lh = tex.height;
    for (unsigned int m = 1; m < levels; ++m) {
        size_t dw = std::max<size_t>(1, lw >> 1);
        size_t dh = std::max<size_t>(1, lh >> 1);

        uint8_t *srcBase = base + offs[m - 1];
        uint8_t *dstBase = base + offs[m];
        size_t srcPixels = lw * lh;
        size_t dstPixels = dw * dh;

        std::vector<float> srcF(srcPixels), dstF(dstPixels);
        for (size_t ch = 0; ch < channelCount; ++ch) {
            channelToFloat(srcBase + ch * bSize, tex.elementSize, srcPixels,
                           tex.desc.elementType, srcF.data());
            boxFilter2D(srcF.data(), lw, lh, dstF.data(), dw, dh);
            floatToChannel(dstF.data(), dstBase + ch * bSize, tex.elementSize,
                           dstPixels, tex.desc.elementType);
        }

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
