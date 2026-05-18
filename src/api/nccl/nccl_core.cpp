#include "vgre/api/nccl_internal.h"
#include "vgre/common/logger.h"

#include "vgre/common/os_backend.h"
#if !defined(_WIN32) && defined(__linux__)
#  include <sys/random.h>  // getrandom()
#endif

// SIMD for reduction
#if defined(__AVX2__)
#  include <immintrin.h>
#elif defined(__SSE2__) || defined(_M_X64)
#  include <emmintrin.h>
#elif defined(__ARM_NEON__) || defined(__aarch64__)
#  include <arm_neon.h>
#endif

// ── TCPCluster multi-node delegation helpers ──────────────────────────────────
// Map NCCL datatype to VGRE ArgType for TCPCluster collective ops.
int nccl_datatype_to_argtype(ncclDataType_t dt) {
    switch (dt) {
    case ncclFloat32:  return static_cast<int>(vgre::ArgType::FLOAT32);
    case ncclFloat64:  return static_cast<int>(vgre::ArgType::FLOAT64);
    case ncclInt32:    return static_cast<int>(vgre::ArgType::INT32);
    case ncclInt64:    return static_cast<int>(vgre::ArgType::INT64);
    case ncclUint32:   return static_cast<int>(vgre::ArgType::UINT32);
    case ncclUint64:   return static_cast<int>(vgre::ArgType::UINT64);
    default:           return static_cast<int>(vgre::ArgType::FLOAT32);
    }
}

// Map NCCL reduction op to TCPCluster ReductionOp.
vgre::advanced::ReductionOp nccl_op_to_tcpcluster(ncclRedOp_t op) {
    switch (op) {
    case ncclSum:  return vgre::advanced::ReductionOp::Sum;
    case ncclProd: return vgre::advanced::ReductionOp::Prod;
    case ncclMax:  return vgre::advanced::ReductionOp::Max;
    case ncclMin:  return vgre::advanced::ReductionOp::Min;
    case ncclAvg:  return vgre::advanced::ReductionOp::Avg;
    default:       return vgre::advanced::ReductionOp::Sum;
    }
}

// Returns true if this process should route NCCL collectives through
// TCPClusterManager rather than shared-memory p2p_slots.
bool tcpcluster_should_delegate() {
    auto& tcm = vgre::advanced::TCPClusterManager::instance();
    if (!tcm.isEnabled()) return false;
    // Master: delegate only if there are active remote workers.
    if (tcm.isMaster()) {
        std::vector<vgre::advanced::TCPClusterManager::ClusterNodeInfo> nodes;
        tcm.getConnectedNodes(nodes);
        return !nodes.empty();
    }
    // Worker: delegate if connected to a master.
    return tcm.isWorker();
}

// ── Element sizes ─────────────────────────────────────────────────────────────
size_t nccl_elem_size(ncclDataType_t dt) {
    switch (dt) {
    case ncclInt8:    case ncclUint8:   return 1;
    case ncclFloat16: case ncclBfloat16:return 2;
    case ncclInt32:   case ncclUint32:  case ncclFloat32: return 4;
    case ncclInt64:   case ncclUint64:  case ncclFloat64: return 8;
    default: return 4;
    }
}

static void reduce_sum(void* dst, const void* src, size_t count, ncclDataType_t dt) {
    if (dt == ncclFloat32) {
        float* d = static_cast<float*>(dst);
        const float* s = static_cast<const float*>(src);
#if defined(__AVX2__)
        size_t i = 0, end8 = (count / 8) * 8;
        for (; i < end8; i += 8) {
            __m256 dv = _mm256_loadu_ps(d + i);
            __m256 sv = _mm256_loadu_ps(s + i);
            _mm256_storeu_ps(d + i, _mm256_add_ps(dv, sv));
        }
        for (; i < count; ++i) d[i] += s[i];
#elif defined(__SSE2__) || defined(_M_X64)
        size_t i = 0, end4 = (count / 4) * 4;
        for (; i < end4; i += 4) {
            __m128 dv = _mm_loadu_ps(d + i);
            __m128 sv = _mm_loadu_ps(s + i);
            _mm_storeu_ps(d + i, _mm_add_ps(dv, sv));
        }
        for (; i < count; ++i) d[i] += s[i];
#else
        for (size_t i = 0; i < count; ++i) d[i] += s[i];
#endif
    } else if (dt == ncclFloat64) {
        double* d = static_cast<double*>(dst);
        const double* s = static_cast<const double*>(src);
        for (size_t i = 0; i < count; ++i) d[i] += s[i];
    } else if (dt == ncclInt32 || dt == ncclUint32) {
        int32_t* d = static_cast<int32_t*>(dst);
        const int32_t* s = static_cast<const int32_t*>(src);
#if defined(__AVX2__)
        size_t i = 0, end8 = (count / 8) * 8;
        for (; i < end8; i += 8) {
            __m256i dv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(d + i));
            __m256i sv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + i));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(d + i), _mm256_add_epi32(dv, sv));
        }
        for (; i < count; ++i) d[i] += s[i];
#else
        for (size_t i = 0; i < count; ++i) d[i] += s[i];
#endif
    } else if (dt == ncclInt64 || dt == ncclUint64) {
        int64_t* d = static_cast<int64_t*>(dst);
        const int64_t* s = static_cast<const int64_t*>(src);
        for (size_t i = 0; i < count; ++i) d[i] += s[i];
    } else {
        // Byte-level fallback for int8/uint8 sum
        uint8_t* d = static_cast<uint8_t*>(dst);
        const uint8_t* s = static_cast<const uint8_t*>(src);
        for (size_t i = 0; i < count; ++i) d[i] = static_cast<uint8_t>(d[i] + s[i]);
    }
}

static void reduce_prod(void* dst, const void* src, size_t count, ncclDataType_t dt) {
    if (dt == ncclFloat32) {
        float* d = static_cast<float*>(dst); const float* s = static_cast<const float*>(src);
        for (size_t i = 0; i < count; ++i) d[i] *= s[i];
    } else if (dt == ncclFloat64) {
        double* d = static_cast<double*>(dst); const double* s = static_cast<const double*>(src);
        for (size_t i = 0; i < count; ++i) d[i] *= s[i];
    } else if (dt == ncclInt32 || dt == ncclUint32) {
        int32_t* d = static_cast<int32_t*>(dst); const int32_t* s = static_cast<const int32_t*>(src);
        for (size_t i = 0; i < count; ++i) d[i] *= s[i];
    } else {
        int64_t* d = static_cast<int64_t*>(dst); const int64_t* s = static_cast<const int64_t*>(src);
        for (size_t i = 0; i < count; ++i) d[i] *= s[i];
    }
}

static void reduce_max(void* dst, const void* src, size_t count, ncclDataType_t dt) {
    if (dt == ncclFloat32) {
        float* d = static_cast<float*>(dst); const float* s = static_cast<const float*>(src);
        for (size_t i = 0; i < count; ++i) if (s[i] > d[i]) d[i] = s[i];
    } else if (dt == ncclFloat64) {
        double* d = static_cast<double*>(dst); const double* s = static_cast<const double*>(src);
        for (size_t i = 0; i < count; ++i) if (s[i] > d[i]) d[i] = s[i];
    } else if (dt == ncclInt32) {
        int32_t* d = static_cast<int32_t*>(dst); const int32_t* s = static_cast<const int32_t*>(src);
        for (size_t i = 0; i < count; ++i) if (s[i] > d[i]) d[i] = s[i];
    } else {
        int64_t* d = static_cast<int64_t*>(dst); const int64_t* s = static_cast<const int64_t*>(src);
        for (size_t i = 0; i < count; ++i) if (s[i] > d[i]) d[i] = s[i];
    }
}

static void reduce_min(void* dst, const void* src, size_t count, ncclDataType_t dt) {
    if (dt == ncclFloat32) {
        float* d = static_cast<float*>(dst); const float* s = static_cast<const float*>(src);
        for (size_t i = 0; i < count; ++i) if (s[i] < d[i]) d[i] = s[i];
    } else if (dt == ncclFloat64) {
        double* d = static_cast<double*>(dst); const double* s = static_cast<const double*>(src);
        for (size_t i = 0; i < count; ++i) if (s[i] < d[i]) d[i] = s[i];
    } else if (dt == ncclInt32) {
        int32_t* d = static_cast<int32_t*>(dst); const int32_t* s = static_cast<const int32_t*>(src);
        for (size_t i = 0; i < count; ++i) if (s[i] < d[i]) d[i] = s[i];
    } else {
        int64_t* d = static_cast<int64_t*>(dst); const int64_t* s = static_cast<const int64_t*>(src);
        for (size_t i = 0; i < count; ++i) if (s[i] < d[i]) d[i] = s[i];
    }
}

void apply_reduce(void* dst, const void* src, size_t count,
                          ncclDataType_t dt, ncclRedOp_t op) {
    switch (op) {
    case ncclSum:  reduce_sum(dst, src, count, dt); break;
    case ncclProd: reduce_prod(dst, src, count, dt); break;
    case ncclMax:  reduce_max(dst, src, count, dt); break;
    case ncclMin:  reduce_min(dst, src, count, dt); break;
    case ncclAvg:  reduce_sum(dst, src, count, dt); break; // divided after all ranks
    default:       reduce_sum(dst, src, count, dt); break;
    }
}

void scale_avg(void* buf, size_t count, ncclDataType_t dt, int nranks) {
    if (nranks <= 1) return;
    if (dt == ncclFloat32) {
        float* d = static_cast<float*>(buf); float inv = 1.0f / nranks;
        for (size_t i = 0; i < count; ++i) d[i] *= inv;
    } else if (dt == ncclFloat64) {
        double* d = static_cast<double*>(buf); double inv = 1.0 / nranks;
        for (size_t i = 0; i < count; ++i) d[i] *= inv;
    } else if (dt == ncclInt32) {
        int32_t* d = static_cast<int32_t*>(buf);
        for (size_t i = 0; i < count; ++i) d[i] /= nranks;
    } else {
        int64_t* d = static_cast<int64_t*>(buf);
        for (size_t i = 0; i < count; ++i) d[i] /= nranks;
    }
}

// Use function-local static initialization to prevent blocking during library load
// This ensures the mutex and map are only initialized when first used
std::mutex& getNCCLRegistryMutex() {
    static std::mutex mu;
    return mu;
}

std::unordered_map<std::string, std::shared_ptr<NcclGroupState>>& getNCCLRegistry() {
    static std::unordered_map<std::string, std::shared_ptr<NcclGroupState>> registry;
    return registry;
}

std::string id_key(const ncclUniqueId& id) {
    return std::string(id.internal, NCCL_UNIQUE_ID_BYTES);
}

std::shared_ptr<NcclGroupState> find_or_create(const ncclUniqueId& id, int nranks) {
    std::lock_guard<std::mutex> lg(getNCCLRegistryMutex());
    auto key = id_key(id);
    auto it = getNCCLRegistry().find(key);
    if (it != getNCCLRegistry().end()) return it->second;
    auto state = std::make_shared<NcclGroupState>(nranks);
    getNCCLRegistry()[key] = state;
    return state;
}

// ── Group call batching ───────────────────────────────────────────────────────
thread_local int g_group_depth = 0;

NcclGroupState::NcclGroupState(int nr)
    : nranks(nr), sendbufs(nr, nullptr), gather_slots(nr), p2p_slots(nr) {}
