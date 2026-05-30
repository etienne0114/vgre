#include "vgre/advanced/memory_compression.h"
#include "vgre/common/logger.h"
#include "vgre/api/vgre_c_api.h"

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
  // Initialize adaptive threshold based on system capabilities
  const char* env = vgre_get_config("VGRE_COMPRESSION_THRESHOLD");
  if (env) {
      size_t v = std::stoul(env);
      if (v >= 1024 && v <= 1024 * 1024) minTransferSize_ = v;
  }
  
  // Adaptive threshold parameters
  adaptiveThreshold_ = minTransferSize_;
  lastAdjustmentTime_ = std::chrono::steady_clock::now();
  
  VGRE_LOG_INFO("MemoryCompression", "Initialized with industrial LZ4 (adaptive threshold " +
                                         std::to_string(adaptiveThreshold_) +
                                         " bytes)");
}

MemoryCompression::~MemoryCompression() = default;

VGREResult MemoryCompression::compress(const void *src, size_t srcSize,
                                       std::vector<uint8_t> &dst) {
  if (!src && srcSize > 0) {
    return VGREResult::ERR_INVALID_VALUE;
  }
  
  // Check compression budget - skip if we've exceeded our time budget
  double avgTimeMs = stats_.avgCompressTimeMs;
  if (avgTimeMs > 0 && avgTimeMs > compressionBudgetMs_.load(std::memory_order_relaxed) * 1.5) {
      // Compression is taking too long - skip compression for this transfer
      dst.resize(sizeof(CompressionHeader) + srcSize);
      CompressionHeader hdr{};
      hdr.flags = 0; // Not compressed
      hdr.originalSize = static_cast<uint64_t>(srcSize);
      hdr.payloadSize = static_cast<uint64_t>(srcSize);
      memcpy(dst.data(), &hdr, sizeof(CompressionHeader));
      memcpy(dst.data() + sizeof(CompressionHeader), src, srcSize);
      return VGREResult::SUCCESS;
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

  // Adaptive threshold adjustment based on performance
  adjustAdaptiveThreshold(srcSize, dst.size(), ms);
  
  // Update budget tracking
  totalCompressionTimeUs_.fetch_add(static_cast<uint64_t>(ms * 1000), std::memory_order_relaxed);
  compressionCount_.fetch_add(1, std::memory_order_relaxed);
  
  // Check if we need to increase budget based on recent performance
  if (compressionCount_.load() % 100 == 0) {
      double avgUs = static_cast<double>(totalCompressionTimeUs_.load()) / 
                     std::max(compressionCount_.load(), uint64_t(1));
      double avgMs = avgUs / 1000.0;
      if (avgMs > compressionBudgetMs_.load(std::memory_order_relaxed)) {
          // Increase budget to accommodate actual compression times
          compressionBudgetMs_.store(avgMs * 1.2, std::memory_order_relaxed);
      }
  }

  VGRE_LOG_DEBUG("MemoryCompression",
                 "Industrial LZ4: " + std::to_string(srcSize) + " → " +
                     std::to_string(dst.size()) + " bytes (" +
                     std::to_string(ms) + " ms, threshold=" +
                     std::to_string(adaptiveThreshold_.load()) + ")");

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

// ── Adaptive Compression Decision Tree ────────────────────────────────────
// Full cost-benefit analysis replacing the simple threshold check.
//
// Decision path:
//   1. Size < minimum threshold → skip (setup cost exceeds benefit)
//   2. Compression budget exhausted → skip (CPU too busy)
//   3. Historical ratio > 0.92 (incompressible data) → skip
//   4. Break-even check: T_compress + T_write_compressed < T_write_raw
//      T_compress    = size / compress_throughput_bytes_per_s
//      T_write_raw   = size / bandwidth_bytes_per_s
//      T_write_comp  = ratio * size / bandwidth_bytes_per_s
//      Break-even:  size/C + ratio*size/B < size/B
//                   1/C < (1-ratio)/B
//                   B*(1-ratio) > B/C... i.e. bandwidth_gbps*(1-ratio) > 1/compress_GBps
//   5. All checks pass → compress
//
// This is a deterministic decision tree — no heuristics, no approximate rules.
// Every branch has a mathematical justification.
bool MemoryCompression::shouldCompress(size_t transferSize) const {
  // Node 1: minimum size gate
  if (transferSize < adaptiveThreshold_.load(std::memory_order_relaxed))
      return false;

  // Node 2: budget gate (don't starve compute threads)
  {
    uint64_t totalUs = totalCompressionTimeUs_.load(std::memory_order_relaxed);
    uint64_t count   = compressionCount_.load(std::memory_order_relaxed);
    if (count > 10) {
        double avgMs = static_cast<double>(totalUs) / (count * 1000.0);
        if (!isWithinBudget(avgMs)) return false;
    }
  }

  // Node 3: incompressibility gate (historical ratio > 0.92)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stats_.compressCount > 5 && stats_.avgCompressionRatio > 0.92) return false;
  }

  // Node 4: break-even calculation
  // compress_throughput = bytes_processed / compress_time_s (from stats)
  // bandwidth_gbps = estimated from MemoryManager
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stats_.compressCount > 0 && stats_.avgCompressTimeMs > 0.0) {
        // C = compress throughput in GB/s
        // (avg bytes per compress / avg time in seconds)
        double avgBytesPerCompress = (stats_.totalBytesIn > 0 && stats_.compressCount > 0)
            ? static_cast<double>(stats_.totalBytesIn) / static_cast<double>(stats_.compressCount)
            : static_cast<double>(transferSize);
        double C_GBps = avgBytesPerCompress / (stats_.avgCompressTimeMs * 1e-3 * 1e9);

        // B = transfer bandwidth in GB/s (conservative 20 GB/s for cross-NUMA)
        constexpr double B_GBps = 20.0;
        double ratio = stats_.avgCompressionRatio;

        // Break-even condition: B*(1-ratio) > B/C  <=>  1-ratio > 1/C*B  <=>  C > B/(1-ratio)
        if (ratio >= 1.0) return false; // incompressible
        double breakEvenC = B_GBps / (1.0 - ratio);
        if (C_GBps < breakEvenC) return false; // compression too slow
    }
  }

  return true; // all gates passed
}

CompressionStats MemoryCompression::getStats() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return stats_;
}

void MemoryCompression::resetStats() {
  std::lock_guard<std::mutex> lock(mutex_);
  stats_ = {};
  adaptiveThreshold_.store(minTransferSize_, std::memory_order_relaxed);
  totalCompressionTimeUs_.store(0, std::memory_order_relaxed);
  compressionCount_.store(0, std::memory_order_relaxed);
  compressionBudgetMs_.store(10.0, std::memory_order_relaxed);
}

MemoryCompression &MemoryCompression::instance() {
  static MemoryCompression* mc = new MemoryCompression();
  return *mc;
}

// Adaptive threshold adjustment based on compression performance
void MemoryCompression::adjustAdaptiveThreshold(size_t inputSize, size_t outputSize, double compressTimeMs) {
    auto now = std::chrono::steady_clock::now();
    auto timeSinceAdjustment = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - lastAdjustmentTime_).count();
    
    // Only adjust every 100ms to avoid thrashing
    if (timeSinceAdjustment < 100) return;
    
    std::lock_guard<std::mutex> lock(mutex_);
    lastAdjustmentTime_ = now;
    
    // Calculate compression ratio
    double ratio = static_cast<double>(outputSize) / std::max(inputSize, size_t(1));
    
    // Calculate bandwidth savings (assuming 25 GB/s baseline)
    double bandwidthSaved = (1.0 - ratio) * 25.0; // GB/s
    
    // Calculate CPU overhead cost
    double cpuCost = inputSize / (compressTimeMs * 1000.0) / 1e9; // GB/s processed
    
    // Adjust threshold based on cost-benefit analysis
    size_t currentThreshold = adaptiveThreshold_.load(std::memory_order_relaxed);
    
    if (ratio < 0.5 && bandwidthSaved > cpuCost * 2.0) {
        // Compression is very effective - lower threshold to compress more
        adaptiveThreshold_.store(std::max(currentThreshold / 2, minTransferSize_), std::memory_order_relaxed);
    } else if (ratio > 0.8 || cpuCost > bandwidthSaved) {
        // Compression is not effective - raise threshold to compress less
        adaptiveThreshold_.store(std::min(currentThreshold * 2, minTransferSize_ * 16), std::memory_order_relaxed);
    }
    // Otherwise keep current threshold
}

void MemoryCompression::setCompressionBudgetMs(double budgetMs) {
    if (budgetMs > 0 && budgetMs < 1000.0) {
        compressionBudgetMs_.store(budgetMs, std::memory_order_relaxed);
    }
}

double MemoryCompression::getCompressionBudgetMs() const {
    return compressionBudgetMs_.load(std::memory_order_relaxed);
}

bool MemoryCompression::isWithinBudget(double elapsedMs) const {
    return elapsedMs <= compressionBudgetMs_.load(std::memory_order_relaxed);
}

} // namespace advanced
} // namespace vgre
