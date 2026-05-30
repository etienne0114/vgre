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
    
    // Compression budgeting to control CPU overhead
    void setCompressionBudgetMs(double budgetMs); // Max time per compression
    double getCompressionBudgetMs() const;
    bool isWithinBudget(double elapsedMs) const;

    // Singleton
    static MemoryCompression& instance();

private:
    size_t            minTransferSize_ = 4096; // 4 KB default threshold
    std::atomic<size_t> adaptiveThreshold_{4096}; // Dynamically adjusted threshold
    std::chrono::steady_clock::time_point lastAdjustmentTime_;
    std::atomic<double> compressionBudgetMs_{10.0}; // Max 10ms per compression
    std::atomic<uint64_t> totalCompressionTimeUs_{0}; // Track total compression time
    std::atomic<uint64_t> compressionCount_{0}; // Track number of compressions
    CompressionStats  stats_;
    mutable std::mutex mutex_;
    
    // Adaptive threshold adjustment based on performance metrics
    void adjustAdaptiveThreshold(size_t inputSize, size_t outputSize, double compressTimeMs);
};

} // namespace advanced
} // namespace vgre

#endif // VGRE_ADVANCED_MEMORY_COMPRESSION_H
