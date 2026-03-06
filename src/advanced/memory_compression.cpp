#include "vgre/advanced/memory_compression.h"
#include "vgre/common/logger.h"

#include <chrono>
#include <cstring>
#include <algorithm>

namespace vgre {
namespace advanced {

MemoryCompression::MemoryCompression() {
    VGRE_LOG_INFO("MemoryCompression",
                  "Initialized with min transfer size " +
                  std::to_string(minTransferSize_) + " bytes");
}

MemoryCompression::~MemoryCompression() = default;

// ── LZ4-style fast compression ─────────────────────────────────────────────
// Simplified run-length encoding with literal blocks for fast throughput.
// Format: [literal_len(1 byte)][match_len(1 byte)][literal_data...]
size_t MemoryCompression::lz4Compress(const uint8_t* src, size_t srcSize,
                                       uint8_t* dst, size_t dstCapacity) {
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
            if (dstPos + 2 + litLen > dstCapacity) return 0;

            dst[dstPos++] = static_cast<uint8_t>(std::min(litLen, size_t(255)));
            dst[dstPos++] = static_cast<uint8_t>(runLen);

            // Copy literal data
            if (litLen > 0) {
                std::memcpy(&dst[dstPos], &src[lastLiteral], litLen);
                dstPos += litLen;
            }

            // Write run byte
            if (dstPos + 1 > dstCapacity) return 0;
            dst[dstPos++] = runByte;

            lastLiteral = srcPos;
        }
    }

    // Final literal block
    size_t remaining = srcSize - lastLiteral;
    if (remaining > 0) {
        if (dstPos + 2 + remaining > dstCapacity) return 0;
        dst[dstPos++] = static_cast<uint8_t>(std::min(remaining, size_t(255)));
        dst[dstPos++] = 0; // no run
        std::memcpy(&dst[dstPos], &src[lastLiteral], remaining);
        dstPos += remaining;
    }

    return dstPos;
}

// ── LZ4-style decompression ───────────────────────────────────────────────
size_t MemoryCompression::lz4Decompress(const uint8_t* src, size_t srcSize,
                                         uint8_t* dst, size_t dstCapacity) {
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
            if (srcPos >= srcSize || dstPos + runLen > dstCapacity) return 0;
            uint8_t runByte = src[srcPos++];
            std::memset(&dst[dstPos], runByte, runLen);
            dstPos += runLen;
        }
    }

    return dstPos;
}

// ── Public compress ────────────────────────────────────────────────────────
VGREResult MemoryCompression::compress(const void* src, size_t srcSize,
                                        std::vector<uint8_t>& dst) {
    auto start = std::chrono::steady_clock::now();

    // Allocate worst-case output buffer
    dst.resize(srcSize + (srcSize / 255) + 64);

    size_t compressedSize = lz4Compress(
        static_cast<const uint8_t*>(src), srcSize,
        dst.data(), dst.size());

    if (compressedSize == 0 || compressedSize >= srcSize) {
        // Compression didn't help — store uncompressed
        dst.resize(srcSize);
        std::memcpy(dst.data(), src, srcSize);
        compressedSize = srcSize;
    } else {
        dst.resize(compressedSize);
    }

    auto end = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::lock_guard<std::mutex> lock(mutex_);
    stats_.totalBytesIn += srcSize;
    stats_.totalBytesOut += compressedSize;
    stats_.compressCount++;
    stats_.avgCompressTimeMs = (stats_.avgCompressTimeMs *
                                (stats_.compressCount - 1) + ms) /
                               stats_.compressCount;
    stats_.avgCompressionRatio = static_cast<double>(stats_.totalBytesIn) /
                                 std::max(stats_.totalBytesOut, uint64_t(1));

    VGRE_LOG_DEBUG("MemoryCompression",
                   "Compressed " + std::to_string(srcSize) + " → " +
                   std::to_string(compressedSize) + " bytes (" +
                   std::to_string(ms) + " ms)");

    return VGREResult::SUCCESS;
}

// ── Public decompress ──────────────────────────────────────────────────────
VGREResult MemoryCompression::decompress(const void* src, size_t srcSize,
                                          void* dst, size_t dstCapacity,
                                          size_t& outActualSize) {
    auto start = std::chrono::steady_clock::now();

    size_t decompressedSize = lz4Decompress(
        static_cast<const uint8_t*>(src), srcSize,
        static_cast<uint8_t*>(dst), dstCapacity);

    if (decompressedSize == 0) {
        // Fallback: data was stored uncompressed
        if (srcSize <= dstCapacity) {
            std::memcpy(dst, src, srcSize);
            decompressedSize = srcSize;
        } else {
            return VGREResult::ERROR_COMPRESSION;
        }
    }

    outActualSize = decompressedSize;

    auto end = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::lock_guard<std::mutex> lock(mutex_);
    stats_.decompressCount++;
    stats_.avgDecompressTimeMs = (stats_.avgDecompressTimeMs *
                                  (stats_.decompressCount - 1) + ms) /
                                 stats_.decompressCount;

    return VGREResult::SUCCESS;
}

// ── Configuration ──────────────────────────────────────────────────────────
void   MemoryCompression::setMinTransferSize(size_t bytes) {
    minTransferSize_ = bytes;
}

size_t MemoryCompression::getMinTransferSize() const {
    return minTransferSize_;
}

bool MemoryCompression::shouldCompress(size_t transferSize) const {
    return transferSize >= minTransferSize_;
}

// ── Statistics ─────────────────────────────────────────────────────────────
const CompressionStats& MemoryCompression::getStats() const {
    return stats_;
}

void MemoryCompression::resetStats() {
    std::lock_guard<std::mutex> lock(mutex_);
    stats_ = {};
}

// ── Singleton ──────────────────────────────────────────────────────────────
MemoryCompression& MemoryCompression::instance() {
    static MemoryCompression mc;
    return mc;
}

} // namespace advanced
} // namespace vgre
