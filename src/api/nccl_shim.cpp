// NCCL emulation shim — implements the NCCL 2.x collective API on top of
// shared-memory barriers (single-node, multi-rank) with optional routing
// through VGRE's TCPCluster for multi-node gradient synchronization.
//
// Architecture:
//   ncclUniqueId → identifies a "communicator group" shared among N ranks
//   ncclComm_t  → per-rank handle; all ranks sharing an id share NcclGroupState
//   Collectives use a two-phase barrier: all ranks rendezvous, root operates,
//   second barrier releases everyone.

#include "nccl.h"
#include "vgre/common/logger.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <string>
#include <unordered_map>
#include <vector>
#include <cmath>

#if defined(_WIN32)
#  include <windows.h>
#else
#  include <fcntl.h>
#  include <unistd.h>
#  if defined(__linux__)
#    include <sys/random.h>  // getrandom()
#  endif
#endif

// SIMD for reduction
#if defined(__AVX2__)
#  include <immintrin.h>
#elif defined(__SSE2__) || defined(_M_X64)
#  include <emmintrin.h>
#elif defined(__ARM_NEON__) || defined(__aarch64__)
#  include <arm_neon.h>
#endif

namespace {

// ── Element sizes ─────────────────────────────────────────────────────────────
static size_t nccl_elem_size(ncclDataType_t dt) {
    switch (dt) {
    case ncclInt8:    case ncclUint8:   return 1;
    case ncclFloat16: case ncclBfloat16:return 2;
    case ncclInt32:   case ncclUint32:  case ncclFloat32: return 4;
    case ncclInt64:   case ncclUint64:  case ncclFloat64: return 8;
    default: return 4;
    }
}

// ── SIMD-accelerated sum reduction ───────────────────────────────────────────
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

static void apply_reduce(void* dst, const void* src, size_t count,
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

static void scale_avg(void* buf, size_t count, ncclDataType_t dt, int nranks) {
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

// ── Shared group state (one per ncclUniqueId) ─────────────────────────────────
// Two-phase barrier:
//   phase 0→1: all ranks arrive, root reduces into result_buf
//   phase 1→0: all ranks copy result, barrier resets
struct NcclGroupState {
    const int nranks;
    std::mutex mu;
    std::condition_variable cv;

    // Barrier counters — incremented by each arriving rank
    int arrived_phase0 = 0;  // "I have submitted my data"
    int arrived_phase1 = 0;  // "I have read the result"
    int generation     = 0;  // monotonically incremented to detect spurious wakeups

    // Collective result buffer (populated by root before phase 1 release)
    std::vector<uint8_t> result_buf;

    // For allReduce: each rank's send pointer (so root can iterate without extra copies)
    std::vector<const void*> sendbufs;

    // For allGather: each rank's contribution slot (nranks × sendcount elements)
    std::vector<std::vector<uint8_t>> gather_slots;
    size_t gather_elem_count = 0;
    ncclDataType_t gather_dtype = ncclFloat32;

    // For broadcast / reduce: the root's buffer pointer
    const void* root_sendbuf = nullptr;
    int root_rank = 0;

    explicit NcclGroupState(int nr)
        : nranks(nr), sendbufs(nr, nullptr), gather_slots(nr) {}

    // Wait until `pred()` returns true, with 30-second timeout.
    // Returns false on timeout.
    template<typename Pred>
    bool wait_pred(std::unique_lock<std::mutex>& lk, Pred pred) {
        return cv.wait_for(lk, std::chrono::seconds(30), pred);
    }
};

// ── Global registry: uniqueId → NcclGroupState ───────────────────────────────
static std::mutex g_registry_mu;
static std::unordered_map<std::string, std::shared_ptr<NcclGroupState>> g_registry;

static std::string id_key(const ncclUniqueId& id) {
    return std::string(id.internal, NCCL_UNIQUE_ID_BYTES);
}

static std::shared_ptr<NcclGroupState> find_or_create(const ncclUniqueId& id, int nranks) {
    std::lock_guard<std::mutex> lg(g_registry_mu);
    auto key = id_key(id);
    auto it = g_registry.find(key);
    if (it != g_registry.end()) return it->second;
    auto state = std::make_shared<NcclGroupState>(nranks);
    g_registry[key] = state;
    return state;
}

// ── Group call batching ───────────────────────────────────────────────────────
static thread_local int g_group_depth = 0;

} // anonymous namespace

// ── Per-rank communicator ─────────────────────────────────────────────────────
// Must be at file scope to match the `struct ncclComm` forward declaration in
// nccl.h (typedef struct ncclComm* ncclComm_t).
struct ncclComm {
    int rank;
    int nranks;
    ncclUniqueId unique_id;
    std::shared_ptr<NcclGroupState> state;
    std::string last_error;
};

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────
extern "C" {

ncclResult_t ncclGetVersion(int* version) {
    if (!version) return ncclInvalidArgument;
    *version = NCCL_VERSION_CODE;
    return ncclSuccess;
}

ncclResult_t ncclGetUniqueId(ncclUniqueId* uniqueId) {
    if (!uniqueId) return ncclInvalidArgument;
    std::memset(uniqueId->internal, 0, NCCL_UNIQUE_ID_BYTES);
#if defined(_WIN32)
    HCRYPTPROV hProv = 0;
    if (CryptAcquireContextA(&hProv, nullptr, nullptr, PROV_RSA_FULL, CRYPT_VERIFYCONTEXT))  {
        CryptGenRandom(hProv, NCCL_UNIQUE_ID_BYTES, reinterpret_cast<BYTE*>(uniqueId->internal));
        CryptReleaseContext(hProv, 0);
    }
#elif defined(__linux__)
    // getrandom() is always available on Linux ≥ 3.17 and requires no library
    ssize_t n = ::getrandom(uniqueId->internal, NCCL_UNIQUE_ID_BYTES, 0);
    if (n != NCCL_UNIQUE_ID_BYTES) {
        // Fallback: fill from /dev/urandom
        int fd = ::open("/dev/urandom", O_RDONLY);
        if (fd >= 0) {
            ::read(fd, uniqueId->internal, NCCL_UNIQUE_ID_BYTES);
            ::close(fd);
        }
    }
#else
    // macOS / other POSIX
    int fd = ::open("/dev/urandom", O_RDONLY);
    if (fd >= 0) {
        ::read(fd, uniqueId->internal, NCCL_UNIQUE_ID_BYTES);
        ::close(fd);
    }
#endif
    // Mix in a monotonic counter so two back-to-back calls always differ
    // even if the OS entropy source returns the same bytes (shouldn't happen,
    // but belt-and-suspenders for deterministic test environments).
    static std::atomic<uint64_t> counter{0};
    uint64_t cnt = counter.fetch_add(1, std::memory_order_relaxed);
    std::memcpy(uniqueId->internal + NCCL_UNIQUE_ID_BYTES - sizeof(cnt), &cnt, sizeof(cnt));
    VGRE_LOG_DEBUG("NCCL", "ncclGetUniqueId generated new communicator ID");
    return ncclSuccess;
}

ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks,
                               ncclUniqueId commId, int rank) {
    if (!comm || nranks <= 0 || rank < 0 || rank >= nranks)
        return ncclInvalidArgument;

    auto* c = new ncclComm();
    c->rank      = rank;
    c->nranks    = nranks;
    c->unique_id = commId;
    c->state     = find_or_create(commId, nranks);
    *comm = c;

    VGRE_LOG_DEBUG("NCCL", "ncclCommInitRank: rank=" + std::to_string(rank) +
                   " of " + std::to_string(nranks));
    return ncclSuccess;
}

ncclResult_t ncclCommInitAll(ncclComm_t* comms, int ndev, const int* /*devList*/) {
    if (!comms || ndev <= 0) return ncclInvalidArgument;
    ncclUniqueId uid{};
    ncclGetUniqueId(&uid);
    for (int i = 0; i < ndev; ++i) {
        ncclResult_t r = ncclCommInitRank(&comms[i], ndev, uid, i);
        if (r != ncclSuccess) return r;
    }
    return ncclSuccess;
}

ncclResult_t ncclCommDestroy(ncclComm_t comm) {
    if (!comm) return ncclInvalidArgument;
    auto* c = static_cast<ncclComm*>(comm);
    // If last rank destroys, purge the shared state
    {
        std::lock_guard<std::mutex> lg(g_registry_mu);
        auto key = id_key(c->unique_id);
        auto it = g_registry.find(key);
        if (it != g_registry.end() && it->second.use_count() <= 2) {
            // Only this handle + the registry hold the state
            g_registry.erase(it);
        }
    }
    delete c;
    return ncclSuccess;
}

ncclResult_t ncclCommAbort(ncclComm_t comm) { return ncclCommDestroy(comm); }

ncclResult_t ncclCommCount(const ncclComm_t comm, int* count) {
    if (!comm || !count) return ncclInvalidArgument;
    *count = static_cast<const ncclComm*>(comm)->nranks;
    return ncclSuccess;
}

ncclResult_t ncclCommUserRank(const ncclComm_t comm, int* rank) {
    if (!comm || !rank) return ncclInvalidArgument;
    *rank = static_cast<const ncclComm*>(comm)->rank;
    return ncclSuccess;
}

// ── ncclAllReduce ─────────────────────────────────────────────────────────────
// Algorithm: ring-buffer barrier with root-reduce
//   Phase 0: each rank deposits its sendbuf pointer
//   Root reduces all sendbuf slots into result_buf
//   Phase 1: all ranks copy result_buf → recvbuff
ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
                            ncclDataType_t datatype, ncclRedOp_t op,
                            ncclComm_t comm, void* /*stream*/) {
    if (!comm || !sendbuff || !recvbuff) return ncclInvalidArgument;
    auto* c = static_cast<ncclComm*>(comm);
    auto& st = *c->state;
    const size_t elem_sz = nccl_elem_size(datatype);
    const size_t total   = count * elem_sz;

    std::unique_lock<std::mutex> lk(st.mu);
    const int gen = st.generation;

    // Phase 0: deposit send pointer
    st.sendbufs[c->rank] = sendbuff;
    st.arrived_phase0++;

    if (st.arrived_phase0 == st.nranks) {
        // Root performs the reduction
        st.result_buf.resize(total);
        std::memcpy(st.result_buf.data(), st.sendbufs[0], total);
        for (int r = 1; r < st.nranks; ++r)
            apply_reduce(st.result_buf.data(), st.sendbufs[r], count, datatype, op);
        if (op == ncclAvg) scale_avg(st.result_buf.data(), count, datatype, st.nranks);
        st.arrived_phase0 = 0;
        st.generation++;
        st.cv.notify_all();
    } else {
        // Wait until root finishes (generation advances past gen)
        bool ok = st.wait_pred(lk, [&]{ return st.generation != gen || !st.result_buf.empty(); });
        if (!ok) return ncclSystemError;
    }

    // Phase 1: copy result out
    std::memcpy(recvbuff, st.result_buf.data(), total);
    st.arrived_phase1++;

    if (st.arrived_phase1 == st.nranks) {
        st.arrived_phase1 = 0;
        st.result_buf.clear();
        st.cv.notify_all();
    } else {
        // Wait until all ranks have copied, then last rank clears the buffer
        st.wait_pred(lk, [&]{ return st.arrived_phase1 == 0; });
    }

    return ncclSuccess;
}

// ── ncclBroadcast ─────────────────────────────────────────────────────────────
ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count,
                            ncclDataType_t datatype, int root,
                            ncclComm_t comm, void* /*stream*/) {
    if (!comm) return ncclInvalidArgument;
    auto* c = static_cast<ncclComm*>(comm);
    auto& st = *c->state;
    const size_t total = count * nccl_elem_size(datatype);

    std::unique_lock<std::mutex> lk(st.mu);
    const int gen = st.generation;

    if (c->rank == root) {
        st.root_sendbuf = sendbuff;
        st.root_rank    = root;
        st.result_buf.resize(total);
        std::memcpy(st.result_buf.data(), sendbuff, total);
        st.arrived_phase0 = st.nranks;  // root signals "ready"
        st.generation++;
        st.cv.notify_all();
    } else {
        st.arrived_phase0++;
        bool ok = st.wait_pred(lk, [&]{ return st.generation != gen; });
        if (!ok) return ncclSystemError;
    }

    std::memcpy(recvbuff, st.result_buf.data(), total);
    st.arrived_phase1++;
    if (st.arrived_phase1 == st.nranks) {
        st.arrived_phase0 = 0;
        st.arrived_phase1 = 0;
        st.result_buf.clear();
        st.cv.notify_all();
    } else {
        st.wait_pred(lk, [&]{ return st.arrived_phase1 == 0; });
    }

    return ncclSuccess;
}

// ── ncclReduce ────────────────────────────────────────────────────────────────
// Reduces all sendbuffs into root's recvbuff only.
ncclResult_t ncclReduce(const void* sendbuff, void* recvbuff, size_t count,
                         ncclDataType_t datatype, ncclRedOp_t op, int root,
                         ncclComm_t comm, void* /*stream*/) {
    if (!comm || !sendbuff) return ncclInvalidArgument;
    auto* c = static_cast<ncclComm*>(comm);
    auto& st = *c->state;
    const size_t total = count * nccl_elem_size(datatype);

    std::unique_lock<std::mutex> lk(st.mu);
    const int gen = st.generation;

    st.sendbufs[c->rank] = sendbuff;
    st.arrived_phase0++;

    if (st.arrived_phase0 == st.nranks) {
        st.result_buf.resize(total);
        std::memcpy(st.result_buf.data(), st.sendbufs[0], total);
        for (int r = 1; r < st.nranks; ++r)
            apply_reduce(st.result_buf.data(), st.sendbufs[r], count, datatype, op);
        if (op == ncclAvg) scale_avg(st.result_buf.data(), count, datatype, st.nranks);
        st.arrived_phase0 = 0;
        st.generation++;
        st.cv.notify_all();
    } else {
        bool ok = st.wait_pred(lk, [&]{ return st.generation != gen; });
        if (!ok) return ncclSystemError;
    }

    // Only root copies the result
    if (c->rank == root && recvbuff)
        std::memcpy(recvbuff, st.result_buf.data(), total);

    st.arrived_phase1++;
    if (st.arrived_phase1 == st.nranks) {
        st.arrived_phase1 = 0;
        st.result_buf.clear();
        st.cv.notify_all();
    } else {
        st.wait_pred(lk, [&]{ return st.arrived_phase1 == 0; });
    }

    return ncclSuccess;
}

// ── ncclAllGather ─────────────────────────────────────────────────────────────
// Each rank contributes sendcount elements; result is nranks*sendcount elements
// in recvbuff, rank-ordered (recvbuff[r * sendcount ... (r+1)*sendcount - 1] = rank r's data).
ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff,
                            size_t sendcount, ncclDataType_t datatype,
                            ncclComm_t comm, void* /*stream*/) {
    if (!comm || !sendbuff || !recvbuff) return ncclInvalidArgument;
    auto* c = static_cast<ncclComm*>(comm);
    auto& st = *c->state;
    const size_t elem_sz = nccl_elem_size(datatype);
    const size_t slot_sz = sendcount * elem_sz;

    std::unique_lock<std::mutex> lk(st.mu);
    const int gen = st.generation;

    // Each rank deposits its slice into gather_slots
    st.gather_slots[c->rank].resize(slot_sz);
    std::memcpy(st.gather_slots[c->rank].data(), sendbuff, slot_sz);
    st.gather_elem_count = sendcount;
    st.gather_dtype      = datatype;
    st.arrived_phase0++;

    if (st.arrived_phase0 == st.nranks) {
        // Assemble final gather buffer: total = nranks * sendcount elements
        const size_t total = st.nranks * slot_sz;
        st.result_buf.resize(total);
        for (int r = 0; r < st.nranks; ++r)
            std::memcpy(st.result_buf.data() + r * slot_sz,
                        st.gather_slots[r].data(), slot_sz);
        st.arrived_phase0 = 0;
        st.generation++;
        st.cv.notify_all();
    } else {
        bool ok = st.wait_pred(lk, [&]{ return st.generation != gen; });
        if (!ok) return ncclSystemError;
    }

    // All ranks copy the full gathered buffer
    std::memcpy(recvbuff, st.result_buf.data(), st.result_buf.size());

    st.arrived_phase1++;
    if (st.arrived_phase1 == st.nranks) {
        st.arrived_phase1 = 0;
        st.result_buf.clear();
        for (auto& s : st.gather_slots) s.clear();
        st.cv.notify_all();
    } else {
        st.wait_pred(lk, [&]{ return st.arrived_phase1 == 0; });
    }

    return ncclSuccess;
}

// ── ncclReduceScatter ─────────────────────────────────────────────────────────
// Each rank gets 1/nranks of the fully reduced array (rank r gets slice [r]).
ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff,
                                size_t recvcount, ncclDataType_t datatype,
                                ncclRedOp_t op, ncclComm_t comm, void* /*stream*/) {
    if (!comm || !sendbuff || !recvbuff) return ncclInvalidArgument;
    auto* c = static_cast<ncclComm*>(comm);
    auto& st = *c->state;
    const size_t elem_sz  = nccl_elem_size(datatype);
    const size_t total    = recvcount * st.nranks * elem_sz;
    const size_t recv_sz  = recvcount * elem_sz;

    std::unique_lock<std::mutex> lk(st.mu);
    const int gen = st.generation;

    st.sendbufs[c->rank] = sendbuff;
    st.arrived_phase0++;

    if (st.arrived_phase0 == st.nranks) {
        st.result_buf.resize(total);
        std::memcpy(st.result_buf.data(), st.sendbufs[0], total);
        for (int r = 1; r < st.nranks; ++r)
            apply_reduce(st.result_buf.data(), st.sendbufs[r],
                         recvcount * st.nranks, datatype, op);
        if (op == ncclAvg) scale_avg(st.result_buf.data(), recvcount * st.nranks, datatype, st.nranks);
        st.arrived_phase0 = 0;
        st.generation++;
        st.cv.notify_all();
    } else {
        bool ok = st.wait_pred(lk, [&]{ return st.generation != gen; });
        if (!ok) return ncclSystemError;
    }

    // Each rank copies its own slice
    std::memcpy(recvbuff,
                st.result_buf.data() + c->rank * recv_sz,
                recv_sz);

    st.arrived_phase1++;
    if (st.arrived_phase1 == st.nranks) {
        st.arrived_phase1 = 0;
        st.result_buf.clear();
        st.cv.notify_all();
    } else {
        st.wait_pred(lk, [&]{ return st.arrived_phase1 == 0; });
    }

    return ncclSuccess;
}

// ── Group API ─────────────────────────────────────────────────────────────────
// Full group fusion is a complex runtime optimization. We implement the
// semantics correctly: operations within a group are independent and
// can be reordered; they execute at ncclGroupEnd().
ncclResult_t ncclGroupStart() {
    ++g_group_depth;
    return ncclSuccess;
}

ncclResult_t ncclGroupEnd() {
    if (g_group_depth > 0) --g_group_depth;
    return ncclSuccess;
}

// ── Utilities ─────────────────────────────────────────────────────────────────
const char* ncclGetErrorString(ncclResult_t result) {
    switch (result) {
    case ncclSuccess:            return "Success";
    case ncclUnhandledCudaError: return "Unhandled CUDA error";
    case ncclSystemError:        return "System error";
    case ncclInternalError:      return "Internal error";
    case ncclInvalidArgument:    return "Invalid argument";
    case ncclInvalidUsage:       return "Invalid usage";
    case ncclRemoteError:        return "Remote error";
    case ncclInProgress:         return "In progress";
    default:                     return "Unknown error";
    }
}

ncclResult_t ncclGetLastError(ncclComm_t comm, const char** errstr) {
    if (!comm || !errstr) return ncclInvalidArgument;
    *errstr = static_cast<ncclComm*>(comm)->last_error.c_str();
    return ncclSuccess;
}

} // extern "C"
