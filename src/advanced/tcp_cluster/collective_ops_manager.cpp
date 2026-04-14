#include "vgre/advanced/tcp_cluster/internal/collective_ops_manager.h"
#include "vgre/advanced/tcp_cluster.h"
#include "vgre/common/logger.h"
#include "vgre/common/sockets.h"
#include <cstring>
#include <chrono>

// SIMD intrinsics headers
#if defined(__AVX2__)
#include <immintrin.h>  // AVX2
#elif defined(__SSE2__) || defined(_M_X64) || defined(_M_AMD64)
#include <emmintrin.h>  // SSE2
#endif

namespace vgre {
namespace advanced {

using vgre::common::VGRE_INVALID_SOCKET;

namespace {
// Helper function to get type size from datatype
size_t getTypeSizeFromDatatype(int datatype) {
  switch (datatype) {
    case static_cast<int>(ArgType::INT32):
    case static_cast<int>(ArgType::UINT32):
    case static_cast<int>(ArgType::FLOAT32):
      return 4;
    case static_cast<int>(ArgType::INT64):
    case static_cast<int>(ArgType::UINT64):
    case static_cast<int>(ArgType::FLOAT64):
      return 8;
    default:
      return 8; // Default to 8 bytes
  }
}
} // anonymous namespace

CollectiveOpsManager::CollectiveOpsManager(TCPClusterManager* parent)
    : parent_(parent) {}

VGREResult CollectiveOpsManager::allReduce(void* ptr, size_t count, int datatype) {
  if (parent_->isMaster()) {
    return masterAllReduce(ptr, count, datatype);
  } else {
    return workerAllReduce(ptr, count, datatype);
  }
}

VGREResult CollectiveOpsManager::masterAllReduce(void* ptr, size_t count, int datatype) {
  size_t element_size = getTypeSizeFromDatatype(datatype);
  size_t total_bytes = count * element_size;
  
  std::unique_lock<std::mutex> lock(parent_->reduction_mutex_);
  
  // Initialize reduction state
  parent_->is_reducing_ = true;
  parent_->reduction_count_ = 0;
  parent_->reduction_datatype_ = datatype;
  parent_->reduction_element_count_ = count;
  parent_->reduction_sequence_++;
  
  // Allocate buffer and copy master's local data
  parent_->active_reduction_buffer_.resize(total_bytes);
  std::memcpy(parent_->active_reduction_buffer_.data(), ptr, total_bytes);
  
  // Get number of active workers
  size_t num_workers = 0;
  {
    std::lock_guard<std::recursive_mutex> clients_lock(parent_->clients_mutex_);
    for (const auto& c : parent_->clients_) {
      if (c && c->active) {
        num_workers++;
      }
    }
  }
  
  // Wait for all workers to send their data (with 30-second timeout)
  bool success = parent_->reduction_cv_.wait_for(lock, std::chrono::seconds(30), [this, num_workers]() {
    return parent_->reduction_count_ >= static_cast<int>(num_workers) || !parent_->isEnabled();
  });
  
  if (!success || !parent_->isEnabled()) {
    parent_->is_reducing_ = false;
    return VGREResult::ERR_TIMEOUT;
  }
  
  // Broadcast final result to all workers
  {
    std::lock_guard<std::recursive_mutex> clients_lock(parent_->clients_mutex_);
    for (const auto& c : parent_->clients_) {
      if (c && c->active) {
        parent_->send_packet(c->socket_fd, PacketType::RAW_DATA, 
                   parent_->active_reduction_buffer_.data(), total_bytes, 
                   c->secureChannel.get());
      }
    }
  }
  
  // Copy result back to master's ptr
  std::memcpy(ptr, parent_->active_reduction_buffer_.data(), total_bytes);
  
  // Reset state
  parent_->is_reducing_ = false;
  parent_->reduction_cv_.notify_all();
  
  return VGREResult::SUCCESS;
}

VGREResult CollectiveOpsManager::workerAllReduce(void* ptr, size_t count, int datatype) {
  size_t element_size = getTypeSizeFromDatatype(datatype);
  size_t total_bytes = count * element_size;
  
  // Create collective operation packet
  CollectiveOpPacket op_packet;
  op_packet.op_type = 0; // all_reduce
  op_packet.datatype = datatype;
  op_packet.count = count;
  op_packet.sequence = parent_->reduction_sequence_++;
  
  // Send collective op packet followed by raw data
  if (parent_->client_fd_ == VGRE_INVALID_SOCKET) {
    return VGREResult::ERR_NOT_INITIALIZED;
  }
  
  parent_->send_packet(parent_->client_fd_, PacketType::COLLECTIVE_OP, &op_packet, sizeof(op_packet), 
             parent_->client_secure_channel_.get());
  parent_->send_packet(parent_->client_fd_, PacketType::RAW_DATA, ptr, total_bytes, 
             parent_->client_secure_channel_.get());
  
  // Wait for result from master (with 30-second timeout)
  std::unique_lock<std::mutex> lock(parent_->reduction_mutex_);
  bool success = parent_->reduction_cv_.wait_for(lock, std::chrono::seconds(30), [this]() {
    return !parent_->active_reduction_buffer_.empty() || !parent_->isEnabled();
  });
  
  if (!success || !parent_->isEnabled() || parent_->active_reduction_buffer_.empty()) {
    return VGREResult::ERR_TIMEOUT;
  }
  
  // Copy result back to ptr
  std::memcpy(ptr, parent_->active_reduction_buffer_.data(), total_bytes);
  parent_->active_reduction_buffer_.clear();
  
  return VGREResult::SUCCESS;
}

VGREResult CollectiveOpsManager::barrier() {
  if (!parent_->enabled_) {
    return VGREResult::ERR_NOT_INITIALIZED;
  }

  if (parent_->is_master_) {
    // Master-side barrier: wait for all workers to reach barrier
    std::unique_lock<std::mutex> lock(parent_->barrier_mutex_);
    
    // Get active worker count
    int active_workers = 0;
    {
      std::lock_guard<std::recursive_mutex> client_lock(parent_->clients_mutex_);
      for (const auto& client : parent_->clients_) {
        if (client && client->active && client->security_established) {
          active_workers++;
        }
      }
    }
    
    if (active_workers == 0) {
      // No workers, barrier is trivially satisfied
      return VGREResult::SUCCESS;
    }
    
    // Wait for all workers to send COOP_BARRIER_SYNC
    parent_->barrier_count_ = 0;
    auto result = parent_->barrier_cv_.wait_for(lock, std::chrono::seconds(30), [this, active_workers]() {
      return parent_->barrier_count_ >= static_cast<uint32_t>(active_workers);
    });
    
    if (!result) {
      VGRE_LOG_ERROR("TCPCluster", "Master: Barrier timeout - not all workers reached barrier");
      parent_->barrier_count_ = 0;
      return VGREResult::ERR_TIMEOUT;
    }
    
    // All workers reached barrier - broadcast resume signal
    parent_->barrier_count_ = 0;
    VGREResult broadcast_result = parent_->broadcastPacket(PacketType::COOP_BARRIER_RESUME, nullptr, 0);
    
    if (broadcast_result != VGREResult::SUCCESS) {
      VGRE_LOG_ERROR("TCPCluster", "Master: Failed to broadcast barrier resume");
      return broadcast_result;
    }
    
    VGRE_LOG_DEBUG("TCPCluster", "Master: Barrier completed for " + std::to_string(active_workers) + " workers");
    return VGREResult::SUCCESS;
    
  } else {
    // Worker-side barrier: notify master and wait for resume
    std::unique_lock<std::mutex> lock(parent_->barrier_mutex_);
    
    // Send barrier sync to master
    VGREResult send_result = parent_->send_packet(
        parent_->client_fd_,
        PacketType::COOP_BARRIER_SYNC,
        nullptr, 0,
        parent_->client_secure_channel_.get());
    
    if (send_result != VGREResult::SUCCESS) {
      VGRE_LOG_ERROR("TCPCluster", "Worker: Failed to send barrier sync to master");
      return send_result;
    }
    
    // Wait for master to broadcast COOP_BARRIER_RESUME
    // The resume signal is processed in processClientStagingBuffer() which sets barrier_count_ = 1
    parent_->barrier_count_ = 0;
    auto result = parent_->barrier_cv_.wait_for(lock, std::chrono::seconds(30), [this]() {
      return parent_->barrier_count_ > 0;
    });
    
    if (!result) {
      VGRE_LOG_ERROR("TCPCluster", "Worker: Barrier timeout - master did not send resume");
      parent_->barrier_count_ = 0;
      return VGREResult::ERR_TIMEOUT;
    }
    
    parent_->barrier_count_ = 0;
    VGRE_LOG_DEBUG("TCPCluster", "Worker: Barrier completed");
    return VGREResult::SUCCESS;
  }
}

// Template specializations for sumReduce
template<typename T>
void CollectiveOpsManager::sumReduce(T* dst, const T* src, size_t count) {
  // SIMD-optimized sum reduction with AVX2/SSE2 support
  
#if defined(__AVX2__)
  // AVX2 path: process 8 floats, 4 doubles, 8 int32_t, or 4 int64_t per iteration
  if constexpr (std::is_same_v<T, float>) {
    size_t i = 0;
    const size_t simd_width = 8;
    const size_t simd_end = (count / simd_width) * simd_width;
    
    for (; i < simd_end; i += simd_width) {
      __m256 dst_vec = _mm256_loadu_ps(dst + i);
      __m256 src_vec = _mm256_loadu_ps(src + i);
      __m256 result = _mm256_add_ps(dst_vec, src_vec);
      _mm256_storeu_ps(dst + i, result);
    }
    
    for (; i < count; ++i) {
      dst[i] += src[i];
    }
  } else if constexpr (std::is_same_v<T, double>) {
    size_t i = 0;
    const size_t simd_width = 4;
    const size_t simd_end = (count / simd_width) * simd_width;
    
    for (; i < simd_end; i += simd_width) {
      __m256d dst_vec = _mm256_loadu_pd(dst + i);
      __m256d src_vec = _mm256_loadu_pd(src + i);
      __m256d result = _mm256_add_pd(dst_vec, src_vec);
      _mm256_storeu_pd(dst + i, result);
    }
    
    for (; i < count; ++i) {
      dst[i] += src[i];
    }
  } else if constexpr (std::is_same_v<T, int32_t>) {
    size_t i = 0;
    const size_t simd_width = 8;
    const size_t simd_end = (count / simd_width) * simd_width;
    
    for (; i < simd_end; i += simd_width) {
      __m256i dst_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dst + i));
      __m256i src_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
      __m256i result = _mm256_add_epi32(dst_vec, src_vec);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), result);
    }
    
    for (; i < count; ++i) {
      dst[i] += src[i];
    }
  } else if constexpr (std::is_same_v<T, int64_t>) {
    size_t i = 0;
    const size_t simd_width = 4;
    const size_t simd_end = (count / simd_width) * simd_width;
    
    for (; i < simd_end; i += simd_width) {
      __m256i dst_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dst + i));
      __m256i src_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
      __m256i result = _mm256_add_epi64(dst_vec, src_vec);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), result);
    }
    
    for (; i < count; ++i) {
      dst[i] += src[i];
    }
  } else {
    for (size_t i = 0; i < count; ++i) {
      dst[i] += src[i];
    }
  }
  
#elif defined(__SSE2__) || defined(_M_X64) || defined(_M_AMD64)
  // SSE2 path: process 4 floats, 2 doubles, 4 int32_t, or 2 int64_t per iteration
  if constexpr (std::is_same_v<T, float>) {
    size_t i = 0;
    const size_t simd_width = 4;
    const size_t simd_end = (count / simd_width) * simd_width;
    
    for (; i < simd_end; i += simd_width) {
      __m128 dst_vec = _mm_loadu_ps(dst + i);
      __m128 src_vec = _mm_loadu_ps(src + i);
      __m128 result = _mm_add_ps(dst_vec, src_vec);
      _mm_storeu_ps(dst + i, result);
    }
    
    for (; i < count; ++i) {
      dst[i] += src[i];
    }
  } else if constexpr (std::is_same_v<T, double>) {
    size_t i = 0;
    const size_t simd_width = 2;
    const size_t simd_end = (count / simd_width) * simd_width;
    
    for (; i < simd_end; i += simd_width) {
      __m128d dst_vec = _mm_loadu_pd(dst + i);
      __m128d src_vec = _mm_loadu_pd(src + i);
      __m128d result = _mm_add_pd(dst_vec, src_vec);
      _mm_storeu_pd(dst + i, result);
    }
    
    for (; i < count; ++i) {
      dst[i] += src[i];
    }
  } else if constexpr (std::is_same_v<T, int32_t>) {
    size_t i = 0;
    const size_t simd_width = 4;
    const size_t simd_end = (count / simd_width) * simd_width;
    
    for (; i < simd_end; i += simd_width) {
      __m128i dst_vec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(dst + i));
      __m128i src_vec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
      __m128i result = _mm_add_epi32(dst_vec, src_vec);
      _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), result);
    }
    
    for (; i < count; ++i) {
      dst[i] += src[i];
    }
  } else if constexpr (std::is_same_v<T, int64_t>) {
    size_t i = 0;
    const size_t simd_width = 2;
    const size_t simd_end = (count / simd_width) * simd_width;
    
    for (; i < simd_end; i += simd_width) {
      __m128i dst_vec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(dst + i));
      __m128i src_vec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
      __m128i result = _mm_add_epi64(dst_vec, src_vec);
      _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), result);
    }
    
    for (; i < count; ++i) {
      dst[i] += src[i];
    }
  } else {
    for (size_t i = 0; i < count; ++i) {
      dst[i] += src[i];
    }
  }
  
#else
  // Scalar fallback for systems without SIMD support
  for (size_t i = 0; i < count; ++i) {
    dst[i] += src[i];
  }
#endif
}

// Explicit template instantiations
template void CollectiveOpsManager::sumReduce<float>(float*, const float*, size_t);
template void CollectiveOpsManager::sumReduce<double>(double*, const double*, size_t);
template void CollectiveOpsManager::sumReduce<int32_t>(int32_t*, const int32_t*, size_t);
template void CollectiveOpsManager::sumReduce<int64_t>(int64_t*, const int64_t*, size_t);

} // namespace advanced
} // namespace vgre
