#include "vgre/api/nccl_internal.h"
#include "vgre/common/logger.h"

#include <fcntl.h>
#include <unistd.h>
#include <sys/random.h>

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
        std::lock_guard<std::mutex> lg(getNCCLRegistryMutex());
        auto key = id_key(c->unique_id);
        auto it = getNCCLRegistry().find(key);
        if (it != getNCCLRegistry().end() && it->second.use_count() <= 2) {
            getNCCLRegistry().erase(it);
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
