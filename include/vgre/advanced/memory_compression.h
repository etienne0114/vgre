#ifndef VGRE_ADVANCED_MEMORY_COMPRESSION_H
#define VGRE_ADVANCED_MEMORY_COMPRESSION_H

#include "vgre/common/types.h"
#include "vgre/common/error_codes.h"

#include <cstddef>
#include <vector>
#include <mutex>

namespace vgre {
namespace advanced {

// ── Compression statistics ─────────────────────────────────────────────────
struct CompressionStats {
    uint64_t totalBytesIn       = 0;
    uint64_t totalBytesOut      = 0;
    uint64_t compressCount      = 0;
    uint64_t decompressCount    = 0;
    double   avgCompressionRatio= 1.0;
    double   avgCompressTimeMs  = 0.0;
    double   avgDecompressTimeMs= 0.0;
};

// ── Memory Compression ────────────────────────────────────────────────────
// Provides fast LZ4-style compression for memory transfers to reduce
// bandwidth bottlenecks between virtual GPU memory and host memory.
class MemoryCompression {
public:
    MemoryCompression();
    ~MemoryCompression();

    // Compress data (LZ4-style fast compression)
    VGREResult compress(const void* src, size_t srcSize,
                         std::vector<uint8_t>& dst);

    // Decompress data
    VGREResult decompress(const void* src, size_t srcSize,
                           void* dst, size_t dstCapacity,
                           size_t& outActualSize);

    // Configuration
    void   setMinTransferSize(size_t bytes); // only compress above this
    size_t getMinTransferSize() const;
    bool   shouldCompress(size_t transferSize) const;

    // Statistics
    CompressionStats getStats() const;
    void  resetStats();

    // Singleton
    static MemoryCompression& instance();

private:
    // Simple LZ4-like fast compression (run-length + literal)
    size_t lz4Compress(const uint8_t* src, size_t srcSize,
                        uint8_t* dst, size_t dstCapacity);
    size_t lz4Decompress(const uint8_t* src, size_t srcSize,
                          uint8_t* dst, size_t dstCapacity);

    size_t            minTransferSize_ = 4096; // 4 KB default threshold
    CompressionStats  stats_;
    mutable std::mutex mutex_;
};

} // namespace advanced
} // namespace vgre

#endif // VGRE_ADVANCED_MEMORY_COMPRESSION_H
