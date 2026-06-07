#ifndef VGRE_API_NCCL_INTERNAL_H
#define VGRE_API_NCCL_INTERNAL_H

#include "nccl.h"
#include "vgre/common/types.h"
#include "vgre/advanced/tcp_cluster.h"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <string>
#include <unordered_map>
#include <vector>

// Forward declaration from nccl.h (opaque in user code, complete here).
// We complete the definition in VGRE's internal header so that all
// NCCL translation units share the same struct layout.
struct ncclComm {
    int rank;
    int nranks;
    ncclUniqueId unique_id;
    std::shared_ptr<struct NcclGroupState> state;
    std::string last_error;
};

// ── Shared group state (one per ncclUniqueId) ────────────────────────────
// Two-phase barrier:
//   phase 0→1: all ranks arrive, root reduces into result_buf
//   phase 1→0: all ranks copy result, barrier resets
struct NcclGroupState {
    const int nranks;
    std::mutex mu;
    std::condition_variable cv;

    int arrived_phase0 = 0;
    int arrived_phase1 = 0;
    int generation     = 0;

    std::vector<uint8_t> result_buf;
    std::vector<const void*> sendbufs;
    std::vector<std::vector<uint8_t>> gather_slots;
    size_t gather_elem_count = 0;
    ncclDataType_t gather_dtype = ncclFloat32;

    const void* root_sendbuf = nullptr;
    int root_rank = 0;

    std::vector<std::vector<uint8_t>> p2p_slots;

    // ── Ring AllReduce state ─────────────────────────────────────────────
    // ring_accum[r] holds rank r's accumulated chunk buffer across ring steps.
    // Invariant: after reduce-scatter, ring_accum[r] contains the fully-reduced
    // chunk owned by rank r. O(N * chunk_size) total memory during operation.
    std::vector<std::vector<uint8_t>> ring_accum;
    int ring_arrived = 0;    // arrivals at current step boundary
    int ring_step    = 0;    // current step counter (0..2*(N-1)-1)
    int ring_gen     = 0;    // barrier generation for ring steps

    // ── Tensor-Parallel lock-free AllReduce state ─────────────────────
    // Activated when VGRE_TP_DEGREE is set and all ranks share a process.
    // Uses atomic_thread_fence(acq_rel) barriers instead of condvar to avoid
    // kernel-mode wakeups on same-process tensor-parallel workloads.
    int creator_pid = 0;    // getpid() at construction; used to detect same-process topology
    int tp_degree   = 0;    // from VGRE_TP_DEGREE env var; 0 = disabled
    std::unique_ptr<std::atomic<uintptr_t>[]> tp_sendbufs;  // rank i → sendbuff address
    std::unique_ptr<std::atomic<uintptr_t>[]> tp_recvbufs;  // rank i → recvbuff address
    std::atomic<int> tp_arrived{0};
    std::atomic<int> tp_gen{0};

    explicit NcclGroupState(int nr);

    template<typename Pred>
    bool wait_pred(std::unique_lock<std::mutex>& lk, Pred pred) {
        return cv.wait_for(lk, std::chrono::seconds(30), pred);
    }
};

// ── Registry ─────────────────────────────────────────────────────────────
std::mutex& getNCCLRegistryMutex();
std::unordered_map<std::string, std::shared_ptr<NcclGroupState>>& getNCCLRegistry();
std::shared_ptr<NcclGroupState> find_or_create(const ncclUniqueId& id, int nranks);
std::string id_key(const ncclUniqueId& id);

// ── TCPCluster helpers ───────────────────────────────────────────────────
int nccl_datatype_to_argtype(ncclDataType_t dt);
vgre::advanced::ReductionOp nccl_op_to_tcpcluster(ncclRedOp_t op);
bool tcpcluster_should_delegate();

// ── Element sizes / reduction helpers ─────────────────────────────────────
size_t nccl_elem_size(ncclDataType_t dt);
void apply_reduce(void* dst, const void* src, size_t count,
                  ncclDataType_t dt, ncclRedOp_t op);
void scale_avg(void* buf, size_t count, ncclDataType_t dt, int nranks);

// ── Group call batching ────────────────────────────────────────────────────
extern thread_local int g_group_depth;

#endif // VGRE_API_NCCL_INTERNAL_H
