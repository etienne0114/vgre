#include "vgre/advanced/tcp_cluster/internal/collective_ops_manager.h"
#include "vgre/advanced/tcp_cluster.h"
#include "vgre/api/vgre_c_api.h"
#include "vgre/common/logger.h"
#include "vgre/common/sockets.h"
#include <cstdlib>
#include <cstring>
#include <chrono>
#include "vgre/common/os_backend.h"
#if defined(__linux__)
#include <sys/sysinfo.h>  // struct sysinfo — RAM stats
#elif defined(__APPLE__)
#include <sys/sysctl.h>   // sysctl — memory query
#endif

// SIMD intrinsics headers
#if defined(__AVX2__)
#include <immintrin.h>  // AVX2 + SSE2
#elif defined(__SSE2__) || defined(_M_X64) || defined(_M_AMD64)
#include <emmintrin.h>  // SSE2
#elif defined(__ARM_NEON__) || defined(__aarch64__)
#include <arm_neon.h>   // ARM NEON (AArch32 / AArch64)
#endif

namespace vgre {
namespace advanced {

using vgre::common::VGRE_INVALID_SOCKET;


CollectiveOpsManager::CollectiveOpsManager(TCPClusterManager* parent)
    : parent_(parent) {}

VGREResult CollectiveOpsManager::allReduce(void* ptr, size_t count, int datatype,
                                              ReductionOp op) {
  if (parent_->isMaster()) {
    return masterAllReduce(ptr, count, datatype, op);
  } else {
    return workerAllReduce(ptr, count, datatype, op);
  }
}

VGREResult CollectiveOpsManager::masterAllReduce(void* ptr, size_t count, int datatype,
                                                  ReductionOp op) {
  size_t element_size = ::vgre::vgre_get_type_size(datatype);
  // C3: Overflow check — count * element_size can wrap if count is huge.
  if (element_size > 0 && count > (SIZE_MAX / element_size)) {
    VGRE_LOG_ERROR("TCPCluster", "masterAllReduce: count overflow (count=" + std::to_string(count) + ", element_size=" + std::to_string(element_size) + ")");
    return VGREResult::ERR_INVALID_VALUE;
  }
  size_t total_bytes = count * element_size;

  // Cap allocation to at most 75% of available system RAM to prevent OOM.
  {
    size_t availableBytes = SIZE_MAX;
#if defined(__linux__)
    struct sysinfo si{}; if (sysinfo(&si) == 0) availableBytes = static_cast<size_t>(si.freeram) * si.mem_unit;
#elif defined(__APPLE__)
    int64_t physMem = 0; size_t sz = sizeof(physMem);
    sysctlbyname("hw.memsize", &physMem, &sz, nullptr, 0); if (physMem > 0) availableBytes = static_cast<size_t>(physMem * 3 / 4);
#elif defined(_WIN32)
    MEMORYSTATUSEX ms{}; ms.dwLength = sizeof(ms); if (GlobalMemoryStatusEx(&ms)) availableBytes = static_cast<size_t>(ms.ullAvailPhys);
#endif
    size_t cap = availableBytes * 3 / 4; // 75% of free RAM
    if (total_bytes > cap) {
      VGRE_LOG_ERROR("TCPCluster", "masterAllReduce: buffer (" + std::to_string(total_bytes) + " bytes) exceeds 75% of available RAM (" + std::to_string(cap) + " bytes)");
      return VGREResult::ERR_OUT_OF_MEMORY;
    }
  }

  std::unique_lock<std::mutex> lock(parent_->reduction_mutex_);
  // Initialize reduction state
  parent_->is_reducing_ = true; parent_->reduction_count_ = 0; parent_->reduction_datatype_ = datatype;
  parent_->reduction_element_count_ = count; parent_->reduction_sequence_++;
  parent_->pending_collective_op_type_ = static_cast<uint32_t>(op);
  // Allocate buffer and copy master's local data
  parent_->active_reduction_buffer_.resize(total_bytes); std::memcpy(parent_->active_reduction_buffer_.data(), ptr, total_bytes);

  // Configurable reduction timeout: VGRE_REDUCTION_TIMEOUT_MS (default 30 s).
  int reductionTimeoutMs = 30000;
  {
    const char *envT = vgre_get_config("VGRE_REDUCTION_TIMEOUT_MS");
    if (envT) { int v = std::atoi(envT); if (v > 0) reductionTimeoutMs = v; }
    else {
      // Estimate: count live workers, use their telemetry to compute timeout.
      double maxWorkerLatencyMs = 0.0;
      { std::lock_guard<std::recursive_mutex> cl(parent_->clients_mutex_);
        for (const auto &c : parent_->clients_) {
          if (c && c->active && c->last_telemetry.avg_kernel_latency_ms > 0)
            maxWorkerLatencyMs = std::max(maxWorkerLatencyMs, c->last_telemetry.avg_kernel_latency_ms);
        }
      }
      if (maxWorkerLatencyMs > 0.0) {
        reductionTimeoutMs = static_cast<int>(maxWorkerLatencyMs * 2.0) + 5000; // 2× max observed latency + 5 s network buffer
        reductionTimeoutMs = std::max(reductionTimeoutMs, 10000); // at least 10 s
      }
    }
  }
  // C4: Count active workers live inside the predicate — if a worker disconnects mid-reduction, the stale pre-captured count would deadlock.
  bool success = parent_->reduction_cv_.wait_for(lock, std::chrono::milliseconds(reductionTimeoutMs), [this]() {
    size_t live_workers = 0;
    { std::lock_guard<std::recursive_mutex> cl(parent_->clients_mutex_); for (const auto& c : parent_->clients_) if (c && c->active) live_workers++; }
    return parent_->reduction_count_ >= static_cast<int>(live_workers) || !parent_->isEnabled();
  });
  
  if (!success || !parent_->isEnabled()) { parent_->is_reducing_ = false; return VGREResult::ERR_TIMEOUT; }
  
  // Broadcast final result + COLLECTIVE_COMPLETE to all workers.
  {
    std::lock_guard<std::recursive_mutex> clients_lock(parent_->clients_mutex_);
    for (const auto& c : parent_->clients_) {
      // Only send reduction results to fully authenticated peers — mirrors broadcastPacket() which also guards on security_established.
      if (c && c->active && (!parent_->security_enabled_ || c->security_established)) {
        parent_->send_packet(c->socket_fd, PacketType::RAW_DATA, parent_->active_reduction_buffer_.data(), total_bytes, c->secure_channel.get());
        parent_->send_packet(c->socket_fd, PacketType::COLLECTIVE_COMPLETE, nullptr, 0, c->secure_channel.get());
      }
    }
  }
  
  // Copy result back to master's ptr
  std::memcpy(ptr, parent_->active_reduction_buffer_.data(), total_bytes);
  // Reset state
  parent_->is_reducing_ = false; parent_->reduction_cv_.notify_all();
  return VGREResult::SUCCESS;
}

VGREResult CollectiveOpsManager::workerAllReduce(void* ptr, size_t count, int datatype,
                                                    ReductionOp op) {
  size_t element_size = ::vgre::vgre_get_type_size(datatype); size_t total_bytes = count * element_size;

  // Create collective operation packet with encoded reduction op
  CollectiveOpPacket op_packet;
  op_packet.op_type = encodeCollectiveOpType(0 /* allReduce */, op);
  op_packet.datatype = datatype; op_packet.count = count;
  op_packet.sequence = parent_->reduction_sequence_++;
  
  // Send collective op packet followed by raw data
  if (parent_->client_fd_ == VGRE_INVALID_SOCKET) { return VGREResult::ERR_NOT_INITIALIZED; }
  parent_->send_packet(parent_->client_fd_, PacketType::COLLECTIVE_OP, &op_packet, sizeof(op_packet), parent_->client_secure_channel_.get());
  parent_->send_packet(parent_->client_fd_, PacketType::RAW_DATA, ptr, total_bytes, parent_->client_secure_channel_.get());
  
  // Wait for result from master (with 30-second timeout).
  std::unique_lock<std::mutex> lock(parent_->reduction_mutex_);
  parent_->active_reduction_buffer_.clear(); // discard stale data from prior call
  bool success = parent_->reduction_cv_.wait_for(lock, std::chrono::seconds(30), [this]() {
    return !parent_->active_reduction_buffer_.empty() || !parent_->isEnabled();
  });
  
  if (!success || !parent_->isEnabled() || parent_->active_reduction_buffer_.empty()) { return VGREResult::ERR_TIMEOUT; }

  // Guard against a master that sends a result buffer smaller than expected
  if (parent_->active_reduction_buffer_.size() < total_bytes) {
    VGRE_LOG_ERROR("TCPCluster", "workerAllReduce: received buffer (" + std::to_string(parent_->active_reduction_buffer_.size()) + " bytes) is smaller than expected (" + std::to_string(total_bytes) + " bytes) — possible datatype mismatch, discarding result");
    parent_->active_reduction_buffer_.clear(); return VGREResult::ERR_INVALID_VALUE;
  }

  // Copy result back to ptr
  std::memcpy(ptr, parent_->active_reduction_buffer_.data(), total_bytes); parent_->active_reduction_buffer_.clear();
  return VGREResult::SUCCESS;
}

VGREResult CollectiveOpsManager::barrier() {
  if (!parent_->enabled_) { return VGREResult::ERR_NOT_INITIALIZED; }

  if (parent_->is_master_) {
    // Master-side barrier: wait for all workers to reach barrier
    std::unique_lock<std::mutex> lock(parent_->barrier_mutex_);
    
    // Get active worker count
    int active_workers = 0;
    { std::lock_guard<std::recursive_mutex> client_lock(parent_->clients_mutex_);
      for (const auto& client : parent_->clients_) { if (client && client->active && client->security_established) { active_workers++; } }
    }
    
    if (active_workers == 0) { return VGREResult::SUCCESS; } // No workers, barrier is trivially satisfied
    
    // Wait for all workers to send COOP_BARRIER_SYNC
    parent_->barrier_count_ = 0;
    auto result = parent_->barrier_cv_.wait_for(lock, std::chrono::seconds(30), [this, active_workers]() {
      return parent_->barrier_count_ >= static_cast<uint32_t>(active_workers);
    });
    
    if (!result) {
      VGRE_LOG_ERROR("TCPCluster", "Master: Barrier timeout - not all workers reached barrier");
      parent_->barrier_count_ = 0; return VGREResult::ERR_TIMEOUT;
    }
    
    // All workers reached barrier - broadcast resume signal
    parent_->barrier_count_ = 0;
    VGREResult broadcast_result = parent_->broadcastPacket(PacketType::COOP_BARRIER_RESUME, nullptr, 0);
    if (broadcast_result != VGREResult::SUCCESS) {
      VGRE_LOG_ERROR("TCPCluster", "Master: Failed to broadcast barrier resume"); return broadcast_result;
    }
    
    VGRE_LOG_DEBUG("TCPCluster", "Master: Barrier completed for " + std::to_string(active_workers) + " workers");
    return VGREResult::SUCCESS;
    
  } else {
    // Worker-side barrier: notify master and wait for resume
    std::unique_lock<std::mutex> lock(parent_->barrier_mutex_);
    
    // Send barrier sync to master
    VGREResult send_result = parent_->send_packet(parent_->client_fd_, PacketType::COOP_BARRIER_SYNC, nullptr, 0, parent_->client_secure_channel_.get());
    if (send_result != VGREResult::SUCCESS) {
      VGRE_LOG_ERROR("TCPCluster", "Worker: Failed to send barrier sync to master"); return send_result;
    }
    
    // Wait for master to broadcast COOP_BARRIER_RESUME
    parent_->barrier_count_ = 0;
    auto result = parent_->barrier_cv_.wait_for(lock, std::chrono::seconds(30), [this]() { return parent_->barrier_count_ > 0; });
    if (!result) {
      VGRE_LOG_ERROR("TCPCluster", "Worker: Barrier timeout - master did not send resume");
      parent_->barrier_count_ = 0; return VGREResult::ERR_TIMEOUT;
    }
    
    parent_->barrier_count_ = 0;
    VGRE_LOG_DEBUG("TCPCluster", "Worker: Barrier completed"); return VGREResult::SUCCESS;
  }
}

// applyReduce — supports Sum (SIMD-optimized), Prod, Max, Min.
template<typename T>
void CollectiveOpsManager::applyReduce(T* dst, const T* src, size_t count, ReductionOp op) {
  if (op == ReductionOp::Sum) {
    // SIMD-optimized sum reduction
#if defined(__AVX2__)
    if constexpr (std::is_same_v<T, float>) {
      size_t i = 0; const size_t simd_width = 8; const size_t simd_end = (count / simd_width) * simd_width;
      for (; i < simd_end; i += simd_width) {
        __m256 dst_vec = _mm256_loadu_ps(dst + i); __m256 src_vec = _mm256_loadu_ps(src + i);
        _mm256_storeu_ps(dst + i, _mm256_add_ps(dst_vec, src_vec));
      }
      for (; i < count; ++i) dst[i] += src[i];
    } else if constexpr (std::is_same_v<T, double>) {
      size_t i = 0; const size_t simd_width = 4; const size_t simd_end = (count / simd_width) * simd_width;
      for (; i < simd_end; i += simd_width) {
        __m256d dst_vec = _mm256_loadu_pd(dst + i); __m256d src_vec = _mm256_loadu_pd(src + i);
        _mm256_storeu_pd(dst + i, _mm256_add_pd(dst_vec, src_vec));
      }
      for (; i < count; ++i) dst[i] += src[i];
    } else if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t>) {
      size_t i = 0; const size_t simd_width = 8; const size_t simd_end = (count / simd_width) * simd_width;
      for (; i < simd_end; i += simd_width) {
        __m256i dst_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dst + i));
        __m256i src_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), _mm256_add_epi32(dst_vec, src_vec));
      }
      for (; i < count; ++i) dst[i] += src[i];
    } else if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>) {
      size_t i = 0; const size_t simd_width = 4; const size_t simd_end = (count / simd_width) * simd_width;
      for (; i < simd_end; i += simd_width) {
        __m256i dst_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dst + i));
        __m256i src_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), _mm256_add_epi64(dst_vec, src_vec));
      }
      for (; i < count; ++i) dst[i] += src[i];
    } else { for (size_t i = 0; i < count; ++i) dst[i] += src[i]; }
#elif defined(__SSE2__) || defined(_M_X64) || defined(_M_AMD64)
    if constexpr (std::is_same_v<T, float>) {
      size_t i = 0; const size_t simd_width = 4; const size_t simd_end = (count / simd_width) * simd_width;
      for (; i < simd_end; i += simd_width) {
        __m128 dst_vec = _mm_loadu_ps(dst + i); __m128 src_vec = _mm_loadu_ps(src + i);
        _mm_storeu_ps(dst + i, _mm_add_ps(dst_vec, src_vec));
      }
      for (; i < count; ++i) dst[i] += src[i];
    } else if constexpr (std::is_same_v<T, double>) {
      size_t i = 0; const size_t simd_width = 2; const size_t simd_end = (count / simd_width) * simd_width;
      for (; i < simd_end; i += simd_width) {
        __m128d dst_vec = _mm_loadu_pd(dst + i); __m128d src_vec = _mm_loadu_pd(src + i);
        _mm_storeu_pd(dst + i, _mm_add_pd(dst_vec, src_vec));
      }
      for (; i < count; ++i) dst[i] += src[i];
    } else if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t>) {
      size_t i = 0; const size_t simd_width = 4; const size_t simd_end = (count / simd_width) * simd_width;
      for (; i < simd_end; i += simd_width) {
        __m128i dst_vec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(dst + i));
        __m128i src_vec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), _mm_add_epi32(dst_vec, src_vec));
      }
      for (; i < count; ++i) dst[i] += src[i];
    } else if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>) {
      size_t i = 0; const size_t simd_width = 2; const size_t simd_end = (count / simd_width) * simd_width;
      for (; i < simd_end; i += simd_width) {
        __m128i dst_vec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(dst + i));
        __m128i src_vec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), _mm_add_epi64(dst_vec, src_vec));
      }
      for (; i < count; ++i) dst[i] += src[i];
    } else { for (size_t i = 0; i < count; ++i) dst[i] += src[i]; }
#elif defined(__ARM_NEON__) || defined(__aarch64__)
    if constexpr (std::is_same_v<T, float>) {
      size_t i = 0; const size_t simd_end = (count / 4) * 4;
      for (; i < simd_end; i += 4) { float32x4_t d = vld1q_f32(dst + i); float32x4_t s = vld1q_f32(src + i); vst1q_f32(dst + i, vaddq_f32(d, s)); }
      for (; i < count; ++i) dst[i] += src[i];
    } else if constexpr (std::is_same_v<T, double>) {
      size_t i = 0; const size_t simd_end = (count / 2) * 2;
      for (; i < simd_end; i += 2) { float64x2_t d = vld1q_f64(dst + i); float64x2_t s = vld1q_f64(src + i); vst1q_f64(dst + i, vaddq_f64(d, s)); }
      for (; i < count; ++i) dst[i] += src[i];
    } else if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t>) {
      size_t i = 0; const size_t simd_end = (count / 4) * 4;
      for (; i < simd_end; i += 4) {
        int32x4_t d = vld1q_s32(reinterpret_cast<const int32_t*>(dst + i)); int32x4_t s = vld1q_s32(reinterpret_cast<const int32_t*>(src + i));
        vst1q_s32(reinterpret_cast<int32_t*>(dst + i), vaddq_s32(d, s));
      }
      for (; i < count; ++i) dst[i] += src[i];
    } else if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>) {
      size_t i = 0; const size_t simd_end = (count / 2) * 2;
      for (; i < simd_end; i += 2) {
        int64x2_t d = vld1q_s64(reinterpret_cast<const int64_t*>(dst + i)); int64x2_t s = vld1q_s64(reinterpret_cast<const int64_t*>(src + i));
        vst1q_s64(reinterpret_cast<int64_t*>(dst + i), vaddq_s64(d, s));
      }
      for (; i < count; ++i) dst[i] += src[i];
    } else { for (size_t i = 0; i < count; ++i) dst[i] += src[i]; }
#else
    for (size_t i = 0; i < count; ++i) dst[i] += src[i];
#endif
  } else if (op == ReductionOp::Prod) {
    for (size_t i = 0; i < count; ++i) dst[i] *= src[i];
  } else if (op == ReductionOp::Max) {
    for (size_t i = 0; i < count; ++i) if (src[i] > dst[i]) dst[i] = src[i];
  } else if (op == ReductionOp::Min) {
    for (size_t i = 0; i < count; ++i) if (src[i] < dst[i]) dst[i] = src[i];
  }
}

// Explicit template instantiations
template void CollectiveOpsManager::applyReduce<float>(float*, const float*, size_t, ReductionOp);
template void CollectiveOpsManager::applyReduce<double>(double*, const double*, size_t, ReductionOp);
template void CollectiveOpsManager::applyReduce<int32_t>(int32_t*, const int32_t*, size_t, ReductionOp);
template void CollectiveOpsManager::applyReduce<int64_t>(int64_t*, const int64_t*, size_t, ReductionOp);
template void CollectiveOpsManager::applyReduce<uint32_t>(uint32_t*, const uint32_t*, size_t, ReductionOp);
template void CollectiveOpsManager::applyReduce<uint64_t>(uint64_t*, const uint64_t*, size_t, ReductionOp);

} // namespace advanced
} // namespace vgre
