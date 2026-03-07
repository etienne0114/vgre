#include "vgre/advanced/memory_compression.h"
#include "vgre/common/logger.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>

namespace vgre {
namespace advanced {

namespace {
constexpr uint32_t kCompressionMagic = 0x56475245U; // "VGRE"
constexpr uint8_t kCompressionVersion = 1;
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
  VGRE_LOG_INFO("MemoryCompression", "Initialized with min transfer size " +
                                         std::to_string(minTransferSize_) +
                                         " bytes");
}

MemoryCompression::~MemoryCompression() = default;

// ── LZ4-style fast compression ─────────────────────────────────────────────
// Simplified run-length encoding with literal blocks for fast throughput.
// Format: [literal_len(1 byte)][match_len(1 byte)][literal_data...]
size_t MemoryCompression::lz4Compress(const uint8_t *src, size_t srcSize,
                                      uint8_t *dst, size_t dstCapacity) {
  if (srcSize == 0 || dstCapacity < srcSize + (srcSize / 255) + 16) {
    // Not enough space — just copy
    return 0;
  }

  size_t srcPos = 0, dstPos = 0;
  size_t lastLiteral = 0;

  while (srcPos < srcSize) {
    // Search for a run of repeated bytes
    size_t runStart = srcPos;
    uint8_t runByte = src[srcPos];
    size_t runLen = 0;

    while (srcPos < srcSize && src[srcPos] == runByte && runLen < 255) {
      ++srcPos;
      ++runLen;
    }

    if (runLen >= 4) {
      // Write literal block before the run
      size_t litLen = runStart - lastLiteral;

      while (litLen > 255) {
        if (dstPos + 2 + 255 > dstCapacity)
          return 0;
        dst[dstPos++] = 255;
        dst[dstPos++] = 0; // no run
        std::memcpy(&dst[dstPos], &src[lastLiteral], 255);
        dstPos += 255;
        lastLiteral += 255;
        litLen -= 255;
      }

      if (dstPos + 2 + litLen + 1 > dstCapacity)
        return 0;
      dst[dstPos++] = static_cast<uint8_t>(litLen);
      dst[dstPos++] = static_cast<uint8_t>(runLen);

      // Copy literal data
      if (litLen > 0) {
        std::memcpy(&dst[dstPos], &src[lastLiteral], litLen);
        dstPos += litLen;
      }

      // Write run byte
      dst[dstPos++] = runByte;

      lastLiteral = srcPos;
    }
  }

  // Final literal block
  size_t remaining = srcSize - lastLiteral;
  while (remaining > 0) {
    size_t chunk = std::min(remaining, size_t(255));
    if (dstPos + 2 + chunk > dstCapacity)
      return 0;
    dst[dstPos++] = static_cast<uint8_t>(chunk);
    dst[dstPos++] = 0; // no run
    std::memcpy(&dst[dstPos], &src[lastLiteral], chunk);
    dstPos += chunk;
    lastLiteral += chunk;
    remaining -= chunk;
  }

  return dstPos;
}

// ── LZ4-style decompression ───────────────────────────────────────────────
size_t MemoryCompression::lz4Decompress(const uint8_t *src, size_t srcSize,
                                        uint8_t *dst, size_t dstCapacity) {
  size_t srcPos = 0, dstPos = 0;

  while (srcPos + 1 < srcSize) {
    uint8_t litLen = src[srcPos++];
    uint8_t runLen = src[srcPos++];

    // Copy literal data
    if (litLen > 0) {
      if (srcPos + litLen > srcSize || dstPos + litLen > dstCapacity)
        return 0;
      std::memcpy(&dst[dstPos], &src[srcPos], litLen);
      dstPos += litLen;
      srcPos += litLen;
    }

    // Expand run
    if (runLen > 0) {
      if (srcPos >= srcSize || dstPos + runLen > dstCapacity)
        return 0;
      uint8_t runByte = src[srcPos++];
      std::memset(&dst[dstPos], runByte, runLen);
      dstPos += runLen;
    }
  }

  return dstPos;
}

// ── Public compress ────────────────────────────────────────────────────────
VGREResult MemoryCompression::compress(const void *src, size_t srcSize,
                                       std::vector<uint8_t> &dst) {
  if (!src && srcSize > 0) {
    return VGREResult::ERROR_INVALID_VALUE;
  }
  auto start = std::chrono::steady_clock::now();

  std::vector<uint8_t> compressed;
  compressed.resize(srcSize + (srcSize / 255) + 64);
  size_t compressedSize = lz4Compress(static_cast<const uint8_t *>(src), srcSize,
                                      compressed.data(), compressed.size());

  const bool useCompressed = (compressedSize > 0 && compressedSize < srcSize);
  const uint64_t payloadSize =
      static_cast<uint64_t>(useCompressed ? compressedSize : srcSize);

  CompressionHeader hdr{};
  hdr.flags = useCompressed ? kFlagCompressed : 0;
  hdr.originalSize = static_cast<uint64_t>(srcSize);
  hdr.payloadSize = payloadSize;

  dst.resize(sizeof(CompressionHeader) + static_cast<size_t>(payloadSize));
  std::memcpy(dst.data(), &hdr, sizeof(CompressionHeader));
  if (payloadSize > 0) {
    if (useCompressed) {
      std::memcpy(dst.data() + sizeof(CompressionHeader), compressed.data(),
                  static_cast<size_t>(payloadSize));
    } else if (srcSize > 0) {
      std::memcpy(dst.data() + sizeof(CompressionHeader), src, srcSize);
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
                 "Compressed " + std::to_string(srcSize) + " → " +
                     std::to_string(dst.size()) + " bytes (" +
                     std::to_string(ms) + " ms)");

  return VGREResult::SUCCESS;
}

// ── Public decompress ──────────────────────────────────────────────────────
VGREResult MemoryCompression::decompress(const void *src, size_t srcSize,
                                         void *dst, size_t dstCapacity,
                                         size_t &outActualSize) {
  if ((!src && srcSize > 0) || !dst) {
    return VGREResult::ERROR_INVALID_VALUE;
  }
  auto start = std::chrono::steady_clock::now();

  outActualSize = 0;

  if (srcSize >= sizeof(CompressionHeader)) {
    CompressionHeader hdr{};
    std::memcpy(&hdr, src, sizeof(CompressionHeader));
    if (hdr.magic == kCompressionMagic && hdr.version == kCompressionVersion) {
      const uint8_t *payload =
          static_cast<const uint8_t *>(src) + sizeof(CompressionHeader);
      const size_t payloadSize = static_cast<size_t>(hdr.payloadSize);
      const size_t originalSize = static_cast<size_t>(hdr.originalSize);
      if (sizeof(CompressionHeader) + payloadSize > srcSize ||
          originalSize > dstCapacity) {
        return VGREResult::ERROR_COMPRESSION;
      }

      if ((hdr.flags & kFlagCompressed) != 0) {
        size_t decompressedSize =
            lz4Decompress(payload, payloadSize, static_cast<uint8_t *>(dst),
                          dstCapacity);
        if (decompressedSize != originalSize) {
          return VGREResult::ERROR_COMPRESSION;
        }
      } else {
        if (payloadSize != originalSize) {
          return VGREResult::ERROR_COMPRESSION;
        }
        if (originalSize > 0) {
          std::memcpy(dst, payload, originalSize);
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

  // Legacy fallback for pre-header payloads.
  size_t decompressedSize =
      lz4Decompress(static_cast<const uint8_t *>(src), srcSize,
                    static_cast<uint8_t *>(dst), dstCapacity);
  if (decompressedSize == 0) {
    if (srcSize > dstCapacity) {
      return VGREResult::ERROR_COMPRESSION;
    }
    std::memcpy(dst, src, srcSize);
    decompressedSize = srcSize;
  }
  outActualSize = decompressedSize;

  auto end = std::chrono::steady_clock::now();
  double ms = std::chrono::duration<double, std::milli>(end - start).count();

  std::lock_guard<std::mutex> lock(mutex_);
  stats_.decompressCount++;
  stats_.avgDecompressTimeMs =
      (stats_.avgDecompressTimeMs * (stats_.decompressCount - 1) + ms) /
      stats_.decompressCount;

  return VGREResult::SUCCESS;
}

// ── Configuration ──────────────────────────────────────────────────────────
void MemoryCompression::setMinTransferSize(size_t bytes) {
  minTransferSize_ = bytes;
}

size_t MemoryCompression::getMinTransferSize() const {
  return minTransferSize_;
}

bool MemoryCompression::shouldCompress(size_t transferSize) const {
  return transferSize >= minTransferSize_;
}

// ── Statistics ─────────────────────────────────────────────────────────────
CompressionStats MemoryCompression::getStats() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return stats_;
}

void MemoryCompression::resetStats() {
  std::lock_guard<std::mutex> lock(mutex_);
  stats_ = {};
}

// ── Singleton ──────────────────────────────────────────────────────────────
MemoryCompression &MemoryCompression::instance() {
  static MemoryCompression mc;
  return mc;
}

} // namespace advanced
} // namespace vgre
