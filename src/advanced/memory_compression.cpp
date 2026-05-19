#include "vgre/advanced/memory_compression.h"
#include "vgre/common/logger.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <lz4.h>

namespace vgre {
namespace advanced {

namespace {
constexpr uint32_t kCompressionMagic = 0x56475245U; // "VGRE"
constexpr uint8_t kCompressionVersion = 2; // Updated for industrial LZ4
constexpr uint8_t kFlagCompressed = 0x1;

struct CompressionHeader {
  uint32_t magic = kCompressionMagic;
  uint8_t version = kCompressionVersion;
  uint8_t flags = 0;
  uint16_t reserved = 0;
  uint64_t originalSize = 0;
  uint64_t payloadSize = 0;
};
} // namespace

MemoryCompression::MemoryCompression() {
  VGRE_LOG_INFO("MemoryCompression", "Initialized with industrial LZ4 (min transfer size " +
                                         std::to_string(minTransferSize_) +
                                         " bytes)");
}

MemoryCompression::~MemoryCompression() = default;

VGREResult MemoryCompression::compress(const void *src, size_t srcSize,
                                       std::vector<uint8_t> &dst) {
  if (!src && srcSize > 0) {
    return VGREResult::ERR_INVALID_VALUE;
  }
  auto start = std::chrono::steady_clock::now();

  const int maxCompressedSize = LZ4_compressBound(static_cast<int>(srcSize));
  std::vector<uint8_t> compressed;
  const bool shouldTryCompress = shouldCompress(srcSize);
  
  int compressedSize = 0;
  if (shouldTryCompress) {
    compressed.resize(static_cast<size_t>(maxCompressedSize));
    compressedSize = LZ4_compress_default(
        static_cast<const char *>(src), 
        reinterpret_cast<char *>(compressed.data()), 
        static_cast<int>(srcSize), 
        maxCompressedSize);
  }

  const bool useCompressed = (compressedSize > 0 && compressedSize < static_cast<int>(srcSize));
  const uint64_t payloadSize =
      static_cast<uint64_t>(useCompressed ? compressedSize : srcSize);

  CompressionHeader hdr{};
  hdr.flags = useCompressed ? kFlagCompressed : 0;
  hdr.originalSize = static_cast<uint64_t>(srcSize);
  hdr.payloadSize = payloadSize;

  dst.resize(sizeof(CompressionHeader) + static_cast<size_t>(payloadSize));
  memcpy(dst.data(), &hdr, sizeof(CompressionHeader));
  if (payloadSize > 0) {
    if (useCompressed) {
      memcpy(dst.data() + sizeof(CompressionHeader), compressed.data(),
                  static_cast<size_t>(payloadSize));
    } else if (srcSize > 0) {
      memcpy(dst.data() + sizeof(CompressionHeader), src, srcSize);
    }
  }

  auto end = std::chrono::steady_clock::now();
  double ms = std::chrono::duration<double, std::milli>(end - start).count();

  std::lock_guard<std::mutex> lock(mutex_);
  stats_.totalBytesIn += srcSize;
  stats_.totalBytesOut += dst.size();
  stats_.compressCount++;
  stats_.avgCompressTimeMs =
      (stats_.avgCompressTimeMs * (stats_.compressCount - 1) + ms) /
      stats_.compressCount;
  stats_.avgCompressionRatio = static_cast<double>(stats_.totalBytesIn) /
                               std::max(stats_.totalBytesOut, uint64_t(1));

  VGRE_LOG_DEBUG("MemoryCompression",
                 "Industrial LZ4: " + std::to_string(srcSize) + " → " +
                     std::to_string(dst.size()) + " bytes (" +
                     std::to_string(ms) + " ms)");

  return VGREResult::SUCCESS;
}

VGREResult MemoryCompression::decompress(const void *src, size_t srcSize,
                                         void *dst, size_t dstCapacity,
                                         size_t &outActualSize) {
  if ((!src && srcSize > 0) || !dst) {
    return VGREResult::ERR_INVALID_VALUE;
  }
  auto start = std::chrono::steady_clock::now();

  outActualSize = 0;

  if (srcSize >= sizeof(CompressionHeader)) {
    CompressionHeader hdr{};
    memcpy(&hdr, src, sizeof(CompressionHeader));
    if (hdr.magic == kCompressionMagic && hdr.version == kCompressionVersion) {
      const char *payload =
          static_cast<const char *>(src) + sizeof(CompressionHeader);
      const size_t payloadSize = static_cast<size_t>(hdr.payloadSize);
      const size_t originalSize = static_cast<size_t>(hdr.originalSize);
      
      if (sizeof(CompressionHeader) + payloadSize > srcSize ||
          originalSize > dstCapacity) {
        return VGREResult::ERR_COMPRESSION;
      }

      if ((hdr.flags & kFlagCompressed) != 0) {
        int decompressedSize = LZ4_decompress_safe(
            payload, static_cast<char *>(dst), 
            static_cast<int>(payloadSize), 
            static_cast<int>(dstCapacity));
            
        if (decompressedSize < 0 || static_cast<size_t>(decompressedSize) != originalSize) {
          VGRE_LOG_ERROR("MemoryCompression", "LZ4 Decompression failed (err=" + 
                         std::to_string(decompressedSize) + ")");
          return VGREResult::ERR_COMPRESSION;
        }
      } else {
        if (payloadSize != originalSize) {
          return VGREResult::ERR_COMPRESSION;
        }
        if (originalSize > 0) {
          memcpy(dst, payload, originalSize);
        }
      }
      outActualSize = originalSize;

      auto end = std::chrono::steady_clock::now();
      double ms = std::chrono::duration<double, std::milli>(end - start).count();
      std::lock_guard<std::mutex> lock(mutex_);
      stats_.decompressCount++;
      stats_.avgDecompressTimeMs =
          (stats_.avgDecompressTimeMs * (stats_.decompressCount - 1) + ms) /
          stats_.decompressCount;
      return VGREResult::SUCCESS;
    }
  }

  // Legacy fallback if signature fails (direct LZ4 decompress attempt)
  int decompressedSize = LZ4_decompress_safe(
      static_cast<const char *>(src), static_cast<char *>(dst),
      static_cast<int>(srcSize), static_cast<int>(dstCapacity));
      
  if (decompressedSize <= 0) {
    // If not LZ4, assume raw copy as last resort
    if (srcSize > dstCapacity) {
      return VGREResult::ERR_COMPRESSION;
    }
    memcpy(dst, src, srcSize);
    decompressedSize = static_cast<int>(srcSize);
  }
  outActualSize = static_cast<size_t>(decompressedSize);

  auto end = std::chrono::steady_clock::now();
  double ms = std::chrono::duration<double, std::milli>(end - start).count();

  std::lock_guard<std::mutex> lock(mutex_);
  stats_.decompressCount++;
  stats_.avgDecompressTimeMs =
      (stats_.avgDecompressTimeMs * (stats_.decompressCount - 1) + ms) /
      stats_.decompressCount;

  return VGREResult::SUCCESS;
}

void MemoryCompression::setMinTransferSize(size_t bytes) {
  minTransferSize_ = bytes;
}

size_t MemoryCompression::getMinTransferSize() const {
  return minTransferSize_;
}

bool MemoryCompression::shouldCompress(size_t transferSize) const {
  return transferSize >= minTransferSize_;
}

CompressionStats MemoryCompression::getStats() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return stats_;
}

void MemoryCompression::resetStats() {
  std::lock_guard<std::mutex> lock(mutex_);
  stats_ = {};
}

MemoryCompression &MemoryCompression::instance() {
  static MemoryCompression* mc = new MemoryCompression();
  return *mc;
}

} // namespace advanced
} // namespace vgre
