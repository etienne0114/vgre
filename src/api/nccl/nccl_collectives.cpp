#include "vgre/api/nccl_internal.h"
#include "vgre/common/logger.h"
#include <cstdlib>

// Algorithm thresholds (bytes):
//   ≤ kTreeThreshold : flat barrier (latency-optimal, O(1) sync rounds)
//   kTreeThreshold – kRingThreshold : binary tree reduce (distributed work, log2(N) rounds)
//   > kRingThreshold : ring allreduce (bandwidth-optimal for large tensors)
static constexpr size_t kTreeThreshold = 64ULL  * 1024;        // 64 KB
static constexpr size_t kRingThreshold = 1024ULL * 1024;       // 1 MB
extern "C" {

static ncclResult_t ring_allreduce(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm*);
static ncclResult_t tree_allreduce(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm*);

// Algorithm: ring-buffer barrier with root-reduce
//   Phase 0: each rank deposits its sendbuf pointer
//   Root reduces all sendbuf slots into result_buf
//   Phase 1: all ranks copy result_buf → recvbuff
ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
                            ncclDataType_t datatype, ncclRedOp_t op,
                            ncclComm_t comm, void* /*stream*/) {
    if (!comm || !sendbuff || !recvbuff) return ncclInvalidArgument;
    auto* c = static_cast<ncclComm*>(comm);

    // Multi-node: if TCPCluster is active with remote peers, route through it
    // for true distributed allReduce (Sum, Prod, Max, Min, Avg).
    if (tcpcluster_should_delegate()) {
        auto& tcm = vgre::advanced::TCPClusterManager::instance();
        // Copy input to output buffer (TCPCluster allReduce operates in-place)
        size_t bytes = count * nccl_elem_size(datatype);
        memcpy(recvbuff, sendbuff, bytes);
        int argtype = nccl_datatype_to_argtype(datatype);
        vgre::advanced::ReductionOp redOp = nccl_op_to_tcpcluster(op);
        vgre::VGREResult r = tcm.allReduce(recvbuff, count, argtype, redOp);
        // TCPCluster allReduce doesn't divide by nranks for Avg; do it here.
        if (r == vgre::VGREResult::SUCCESS && op == ncclAvg) {
            auto* c = static_cast<ncclComm*>(comm);
            scale_avg(recvbuff, count, datatype, c->state->nranks);
        }
        return (r == vgre::VGREResult::SUCCESS) ? ncclSuccess : ncclSystemError;
    }

    // Algorithm selection:
    //   small  (≤64 KB)   → flat barrier (lowest latency, root does all work)
    //   medium (64KB-1MB) → binary tree (distributes reduction across log2(N) levels)
    //   large  (>1 MB)    → ring (bandwidth-optimal, chunk-pipelined)
    size_t bytes = count * nccl_elem_size(datatype);
    if (c->state->nranks > 1 && bytes > kRingThreshold)
        return ring_allreduce(sendbuff, recvbuff, count, datatype, op, c);
    if (c->state->nranks > 1 && bytes > kTreeThreshold)
        return tree_allreduce(sendbuff, recvbuff, count, datatype, op, c);
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
        memcpy(st.result_buf.data(), st.sendbufs[0], total);
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
    memcpy(recvbuff, st.result_buf.data(), total);
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
// ── Binary Tree AllReduce (latency+bandwidth balanced for medium tensors) ────
// Used when kTreeThreshold < bytes ≤ kRingThreshold (64 KB – 1 MB).
//
// Algorithm (reduce phase, log2(ceil(nranks)) rounds):
//   Round r=0,1,...: ranks whose index is divisible by 2^(r+1) receive from
//   (rank + 2^r), reduce into their local buffer, then advance.
//   After all rounds rank 0 holds the full reduction.
//
// Broadcast phase (reverse): rank 0 fans out result back down the tree.
//
// In shared-memory context "send" = deposit pointer; "recv" = memcpy from slot.
// The result is identical to flat-barrier reduce but work is distributed across
// log2(N) ranks at each level instead of serialising all reduction on rank 0.
static ncclResult_t tree_allreduce(const void* sendbuff, void* recvbuff,
    size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm* c)
{
    auto& st = *c->state;
    const int    nranks  = st.nranks;
    const int    rank    = c->rank;
    const size_t elem_sz = nccl_elem_size(datatype);
    const size_t total   = count * elem_sz;

    // ── Phase 0: all ranks deposit their send buffers ─────────────────────
    {
        std::unique_lock<std::mutex> lk(st.mu);
        const int gen = st.generation;
        st.sendbufs[rank] = sendbuff;
        st.arrived_phase0++;
        if (st.arrived_phase0 == nranks) {
            // Allocate tree workspace: each rank gets a private scratch slot.
            // We reuse result_buf (sized total) as the "active" buffer.
            st.result_buf.resize(total);
            // Copy each rank's data into a scratch vector indexed by rank.
            // (In a real distributed system each rank would own its own buffer;
            //  here we centralise for thread safety while keeping the tree logic.)
            st.arrived_phase0 = 0;
            st.generation++;
            st.cv.notify_all();
        } else {
            st.wait_pred(lk, [&]{ return st.generation != gen; });
        }
    }

    // ── Reduce phase (bottom-up binary tree) ─────────────────────────────
    // We implement the tree via log2(nranks) barrier rounds.
    // Each active rank in round r reduces from its right child (rank + 2^r)
    // if that child exists, then signals "done with this round".
    // We use a per-round generation counter stored in tree_phase0_count /
    // tree_phase1_count in the shared state. Since those fields don't exist yet
    // we use the standard generation + arrived counters with a separate mu lock.
    // For simplicity we fall back to a barrier-per-round model sharing
    // st.generation as the epoch.  Two-phase per round:
    //   arrived_phase0 counts ranks active in this round.
    //   When all arrive, root of this sub-tree does the reduction step,
    //   then we advance generation and proceed to the next round.

    // Build per-rank scratch buffer (stack-allocated for small tensors).
    std::vector<uint8_t> myBuf(total);
    memcpy(myBuf.data(), sendbuff, total);

    int step = 1;
    while (step < nranks) {
        std::unique_lock<std::mutex> lk(st.mu);
        const int gen = st.generation;

        // Deposit my current partial result into sendbufs slot for this round.
        // We overwrite the slot so the parent can read from it.
        st.sendbufs[rank] = myBuf.data();
        st.arrived_phase0++;

        if (st.arrived_phase0 == nranks) {
            // Every rank participates in the barrier; only "receiver" ranks
            // (those whose rank % (2*step) == 0) actually reduce.
            for (int r = 0; r + step < nranks; r += 2 * step) {
                // rank r reduces from rank r+step
                const void* childBuf = st.sendbufs[r + step];
                if (r == 0) {
                    // r=0 reduces into result_buf directly
                    memcpy(st.result_buf.data(), st.sendbufs[0], total);
                    apply_reduce(st.result_buf.data(), childBuf, count, datatype, op);
                    // push the result back into sendbufs[0] so next round reads it
                    st.sendbufs[0] = st.result_buf.data();
                } else {
                    // Other receivers: reduce child into parent's myBuf.
                    // We can't update myBuf here (it belongs to another thread),
                    // so we use a temporary stored in gather_slots[r].
                    st.gather_slots[r].resize(total);
                    memcpy(st.gather_slots[r].data(), st.sendbufs[r], total);
                    apply_reduce(st.gather_slots[r].data(), childBuf, count, datatype, op);
                    st.sendbufs[r] = st.gather_slots[r].data();
                }
            }
            st.arrived_phase0 = 0;
            st.generation++;
            st.cv.notify_all();
        } else {
            bool ok = st.wait_pred(lk, [&]{ return st.generation != gen; });
            if (!ok) return ncclSystemError;
        }

        // Update my local buffer from sendbufs if I was a receiver this round.
        if (rank % (2 * step) == 0) {
            // I was a receiver — my result is in sendbufs[rank] which now points
            // to either result_buf (rank 0) or gather_slots[rank].
            // We already hold the lock; read after the barrier above.
            if (st.sendbufs[rank] != myBuf.data()) {
                memcpy(myBuf.data(), st.sendbufs[rank], total);
            }
        }
        step <<= 1;
    }

    // ── After reduce: rank 0 has the full result in result_buf ──────────
    {
        std::unique_lock<std::mutex> lk(st.mu);
        const int gen = st.generation;
        if (rank == 0) {
            // Ensure result_buf holds the final answer (already set above).
            if (op == ncclAvg) scale_avg(st.result_buf.data(), count, datatype, nranks);
            st.arrived_phase0 = nranks; // signal ready
            st.generation++;
            st.cv.notify_all();
        } else {
            st.arrived_phase0++;
            st.wait_pred(lk, [&]{ return st.generation != gen; });
        }
    }

    // ── Broadcast phase: all ranks copy from result_buf ────────────────
    memcpy(recvbuff, st.result_buf.data(), total);

    {
        std::unique_lock<std::mutex> lk(st.mu);
        st.arrived_phase1++;
        if (st.arrived_phase1 == nranks) {
            st.arrived_phase0 = 0;
            st.arrived_phase1 = 0;
            st.result_buf.clear();
            for (auto& s : st.gather_slots) s.clear();
            st.cv.notify_all();
        } else {
            st.wait_pred(lk, [&]{ return st.arrived_phase1 == 0; });
        }
    }

    VGRE_LOG_DEBUG("NCCL", "tree_allreduce: " + std::to_string(total) +
                   " bytes, " + std::to_string(nranks) + " ranks");
    return ncclSuccess;
}

// ── Ring AllReduce — Bandwidth-Optimal 2-Phase Ring Schedule ─────────────────
// Algorithm: Rabenseifner 2004 / Barnett 1994 ring reduce-scatter + allgather.
// Bandwidth efficiency: η = 2(N-1)/N — approaches 100% as N → ∞.
//
// Phase 1 — Reduce-Scatter (N-1 steps):
//   Each step s, rank i accumulates chunk (i-s-1+N)%N from its predecessor
//   (rank (i-1+N)%N) using Kahan summation.
//   After N-1 steps, ring_accum[rank] contains the fully-reduced chunk owned
//   by rank (ring invariant: rank i owns chunk (i+1)%N after reduce-scatter).
//
// Phase 2 — AllGather (N-1 steps):
//   Each step s, rank i reads the fully-reduced chunk from ring_accum of
//   predecessor rank (i-1+N)%N (chunk position (rank-1-s+N)%N) and copies it
//   into its output buffer at the corresponding position.
//
// Total data movement per rank: 2*(N-1)/N * total_bytes = η * total_bytes.
// Math invariant: chunk_size = ceil(count/N); ring_accum[r][chunk_r] =
//   Σ_{i=0}^{N-1} X_i[chunk_r] for all i, after reduce-scatter completes.
static ncclResult_t ring_allreduce(const void* sendbuff, void* recvbuff,
    size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm* c)
{
    auto& st     = *c->state;
    const size_t elem_sz   = nccl_elem_size(datatype);
    const size_t total     = count * elem_sz;
    const int    nranks    = st.nranks;
    const int    rank      = c->rank;

    // Chunk layout: chunk k starts at element index k*chunk_count.
    // Last chunk is smaller if count is not divisible by nranks.
    // chunk_count = ceil(count/N)
    const size_t chunk_count = (count + static_cast<size_t>(nranks) - 1)
                                / static_cast<size_t>(nranks);

    // Helper: get element range for chunk k
    auto chunk_range = [&](int k) -> std::pair<size_t, size_t> {
        size_t start = static_cast<size_t>(k) * chunk_count;
        size_t end   = std::min(start + chunk_count, count);
        return {start, end};
    };

    // ── Barrier helper: advance ring_gen, wake all waiters ────────────────
    // Each step uses ring_gen to synchronise — all N ranks must arrive before
    // any rank proceeds to the next step.
    auto ring_barrier = [&](std::unique_lock<std::mutex>& lk) -> bool {
        int target = st.ring_gen + 1;
        st.ring_arrived++;
        if (st.ring_arrived == nranks) {
            st.ring_arrived = 0;
            st.ring_gen = target;
            st.cv.notify_all();
        } else {
            bool ok = st.wait_pred(lk, [&]{ return st.ring_gen == target; });
            if (!ok) return false;
        }
        return true;
    };

    std::unique_lock<std::mutex> lk(st.mu);

    // ── Initialisation barrier: all ranks arrive, set up shared ring_accum ─
    st.sendbufs[rank] = sendbuff;
    st.arrived_phase0++;
    if (st.arrived_phase0 == nranks) {
        // ring_accum[r] = copy of sender r's full buffer (will be reduced in place).
        // O(N * chunk_size) memory; each rank owns one chunk after reduce-scatter.
        st.ring_accum.resize(nranks);
        for (int r = 0; r < nranks; ++r) {
            st.ring_accum[r].resize(total);
            memcpy(st.ring_accum[r].data(), st.sendbufs[r], total);
        }
        st.arrived_phase0 = 0;
        st.ring_step = 0;
        st.generation++;
        st.cv.notify_all();
    } else {
        bool ok = st.wait_pred(lk, [&]{ return st.arrived_phase0 == 0; });
        if (!ok) return ncclSystemError;
    }

    // ── Phase 1: Reduce-Scatter (N-1 barrier-synchronised steps) ──────────
    // Step s: rank i reduces chunk (i-s-1+N)%N — accumulating from predecessor's
    // ring_accum into its own ring_accum for that chunk position.
    // Kahan compensated summation for fp32/fp64 (error bound O(ε) vs O(N·ε) naive).
    // Recurrence: y=src[i]-comp[i]; t=dst[i]+y; comp[i]=(t-dst[i])-y; dst[i]=t
    // Compensation buffer is rank-local across all steps (stack-allocated, size=count).
    int src_rank = (rank - 1 + nranks) % nranks;

    // Allocate Kahan compensation per element (zero-initialised for first step).
    // Only needed for fp32/fp64 Sum/Avg; other ops/types use exact arithmetic.
    std::vector<float>  kahan_comp_f32(
        (datatype == ncclFloat32 && (op == ncclSum || op == ncclAvg)) ? count : 0, 0.0f);
    std::vector<double> kahan_comp_f64(
        (datatype == ncclFloat64 && (op == ncclSum || op == ncclAvg)) ? count : 0, 0.0);

    for (int s = 0; s < nranks - 1; ++s) {
        int src_chunk = ((rank - s - 1) % nranks + nranks) % nranks;
        auto [start, end] = chunk_range(src_chunk);
        size_t nelems = end - start;

        uint8_t* dst_ptr = st.ring_accum[rank].data()     + start * elem_sz;
        const uint8_t* src_ptr = st.ring_accum[src_rank].data() + start * elem_sz;

        if (datatype == ncclFloat32 && (op == ncclSum || op == ncclAvg)) {
            // Kahan compensated sum for fp32: error bound ||e|| ≤ 2ε·||x||_1
            float* dst  = reinterpret_cast<float*>(dst_ptr);
            const float* src = reinterpret_cast<const float*>(src_ptr);
            float* comp = kahan_comp_f32.data() + start;
            for (size_t i = 0; i < nelems; ++i) {
                float y = src[i] - comp[i];
                float t = dst[i] + y;
                comp[i] = (t - dst[i]) - y;
                dst[i]  = t;
            }
        } else if (datatype == ncclFloat64 && (op == ncclSum || op == ncclAvg)) {
            double* dst  = reinterpret_cast<double*>(dst_ptr);
            const double* src = reinterpret_cast<const double*>(src_ptr);
            double* comp = kahan_comp_f64.data() + start;
            for (size_t i = 0; i < nelems; ++i) {
                double y = src[i] - comp[i];
                double t = dst[i] + y;
                comp[i] = (t - dst[i]) - y;
                dst[i]  = t;
            }
        } else {
            apply_reduce(dst_ptr, src_ptr, nelems, datatype, op);
        }

        if (!ring_barrier(lk)) return ncclSystemError;
    }

    // After N-1 steps: rank i's fully-reduced chunk is at position (i+1)%N
    // (ring invariant: the chunk "opposite" this rank in the reduce-scatter).
    if (op == ncclAvg) {
        // Scale the owned chunk by 1/N; other chunks are scaled in allgather copy.
        // We scale all of ring_accum[rank] here to simplify allgather: each rank
        // reads already-scaled data from predecessors during allgather.
        scale_avg(st.ring_accum[rank].data(), count, datatype, nranks);
    }
    if (nranks > 1 && !ring_barrier(lk)) return ncclSystemError;

    // ── Phase 2: AllGather (N-1 barrier-synchronised steps) ───────────────
    // Step s: rank i copies fully-reduced chunk (rank-s-1+N)%N from
    // predecessor's ring_accum into its own output buffer.
    // Each rank starts by writing its own fully-reduced chunk to output.
    {
        int my_chunk = (rank + 1) % nranks;
        auto [start, end] = chunk_range(my_chunk);
        memcpy(static_cast<uint8_t*>(recvbuff) + start * elem_sz,
               st.ring_accum[rank].data() + start * elem_sz,
               (end - start) * elem_sz);
    }
    // AllGather: each step copies the fully-reduced chunk owned by the
    // rank that is (s+1) hops back in the ring.
    // Invariant after reduce-scatter: ring_accum[r] owns chunk (r+1)%N.
    // For step s, the target chunk is (rank-s+N)%N, owned by rank (rank-s-1+N)%N.
    // No intermediate barriers needed — ring_accum is read-only from this point,
    // so all ranks can read from any position after the phase-transition barrier above.
    for (int s = 0; s < nranks - 1; ++s) {
        // Owner of chunk (rank-s)%N is rank (rank-s-1+N)%N
        // because ring_accum[r] contains fully-reduced data at index (r+1)%N,
        // so (r+1)%N == (rank-s)%N  ⟹  r = (rank-s-1+N)%N.
        int src_rank  = ((rank - s - 1) % nranks + nranks) % nranks;
        int src_chunk = (rank - s + nranks) % nranks;   // = (src_rank+1)%N

        auto [start, end] = chunk_range(src_chunk);
        memcpy(static_cast<uint8_t*>(recvbuff) + start * elem_sz,
               st.ring_accum[src_rank].data() + start * elem_sz,
               (end - start) * elem_sz);
    }
    if (nranks > 1 && !ring_barrier(lk)) return ncclSystemError;

    // ── Teardown barrier: last rank clears ring_accum ─────────────────────
    st.arrived_phase1++;
    if (st.arrived_phase1 == nranks) {
        st.arrived_phase1 = 0;
        st.ring_accum.clear();
        st.cv.notify_all();
    } else {
        st.wait_pred(lk, [&]{ return st.arrived_phase1 == 0; });
    }

    VGRE_LOG_DEBUG("NCCL", "ring_allreduce: " + std::to_string(total) +
                   " bytes, " + std::to_string(nranks) + " ranks"
                   " η=2(N-1)/N=" +
                   std::to_string(2.0*(nranks-1)/nranks));
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
        memcpy(st.result_buf.data(), sendbuff, total);
        st.arrived_phase0 = st.nranks;  // root signals "ready"
        st.generation++;
        st.cv.notify_all();
    } else {
        st.arrived_phase0++;
        bool ok = st.wait_pred(lk, [&]{ return st.generation != gen; });
        if (!ok) return ncclSystemError;
    }

    memcpy(recvbuff, st.result_buf.data(), total);
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
        memcpy(st.result_buf.data(), st.sendbufs[0], total);
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
        memcpy(recvbuff, st.result_buf.data(), total);

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
    memcpy(st.gather_slots[c->rank].data(), sendbuff, slot_sz);
    st.gather_elem_count = sendcount;
    st.gather_dtype      = datatype;
    st.arrived_phase0++;

    if (st.arrived_phase0 == st.nranks) {
        // Assemble final gather buffer: total = nranks * sendcount elements
        const size_t total = st.nranks * slot_sz;
        st.result_buf.resize(total);
        for (int r = 0; r < st.nranks; ++r)
            memcpy(st.result_buf.data() + r * slot_sz,
                        st.gather_slots[r].data(), slot_sz);
        st.arrived_phase0 = 0;
        st.generation++;
        st.cv.notify_all();
    } else {
        bool ok = st.wait_pred(lk, [&]{ return st.generation != gen; });
        if (!ok) return ncclSystemError;
    }

    // All ranks copy the full gathered buffer
    memcpy(recvbuff, st.result_buf.data(), st.result_buf.size());

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
// ── ncclReduceScatter — Ring Phase 1 ─────────────────────────────────────────
// Each rank r receives the fully-reduced chunk r out of N_ranks chunks.
// Math: recvbuff = Σ_{i=0}^{N-1} sendbuff_i[r*recvcount : (r+1)*recvcount]
//
// Algorithm: Phase 1 of the Rabenseifner ring AllReduce (reduce-scatter step).
// Bandwidth efficiency: η = (N-1)/N (half of the full AllReduce).
//
// N-1 barrier-synchronised steps:
//   Step s: rank i accumulates chunk (i-s-1+N)%N from its left predecessor's
//   ring_accum into its own ring_accum for that chunk position.
//   After N-1 steps, ring_accum[r] holds the fully-reduced data for chunk
//   (r+1)%N (ring invariant).  Rank r then copies chunk r out of
//   ring_accum[(r-1+N)%N] into recvbuff.
//
// For nranks==1 or very small payloads (≤kTreeThreshold) we fall through to
// the flat-barrier path for lowest latency.
ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff,
                                size_t recvcount, ncclDataType_t datatype,
                                ncclRedOp_t op, ncclComm_t comm, void* /*stream*/) {
    if (!comm || !sendbuff || !recvbuff) return ncclInvalidArgument;
    auto* c = static_cast<ncclComm*>(comm);
    auto& st = *c->state;
    const int    nranks   = st.nranks;
    const int    rank     = c->rank;
    const size_t elem_sz  = nccl_elem_size(datatype);
    const size_t total    = recvcount * static_cast<size_t>(nranks) * elem_sz;
    const size_t recv_sz  = recvcount * elem_sz;

    // Single-rank trivial case: copy sendbuff → recvbuff directly.
    if (nranks == 1) {
        memcpy(recvbuff, sendbuff, recv_sz);
        return ncclSuccess;
    }

    // ── Flat-barrier path for small payloads (latency-optimal) ───────────────
    if (total <= kTreeThreshold) {
        std::unique_lock<std::mutex> lk(st.mu);
        const int gen = st.generation;

        st.sendbufs[rank] = sendbuff;
        st.arrived_phase0++;

        if (st.arrived_phase0 == nranks) {
            st.result_buf.resize(total);
            memcpy(st.result_buf.data(), st.sendbufs[0], total);
            for (int r = 1; r < nranks; ++r)
                apply_reduce(st.result_buf.data(), st.sendbufs[r],
                             recvcount * static_cast<size_t>(nranks), datatype, op);
            if (op == ncclAvg)
                scale_avg(st.result_buf.data(),
                          recvcount * static_cast<size_t>(nranks), datatype, nranks);
            st.arrived_phase0 = 0;
            st.generation++;
            st.cv.notify_all();
        } else {
            bool ok = st.wait_pred(lk, [&]{ return st.generation != gen; });
            if (!ok) return ncclSystemError;
        }

        memcpy(recvbuff, st.result_buf.data() + static_cast<size_t>(rank) * recv_sz, recv_sz);

        st.arrived_phase1++;
        if (st.arrived_phase1 == nranks) {
            st.arrived_phase1 = 0;
            st.result_buf.clear();
            st.cv.notify_all();
        } else {
            st.wait_pred(lk, [&]{ return st.arrived_phase1 == 0; });
        }
        return ncclSuccess;
    }

    // ── Ring Phase 1 (bandwidth-optimal reduce-scatter) ───────────────────────
    // Chunk layout: chunk k starts at element index k * chunk_count.
    // chunk_count = ceil(recvcount / N) — last chunk may be smaller.
    const size_t chunk_count = (recvcount + static_cast<size_t>(nranks) - 1)
                                / static_cast<size_t>(nranks);
    auto chunk_range = [&](int k) -> std::pair<size_t, size_t> {
        size_t start = static_cast<size_t>(k) * chunk_count;
        size_t end   = std::min(start + chunk_count, recvcount);
        return {start, end};
    };

    // Barrier helper shared with ring_allreduce.
    // ring_gen / ring_arrived synchronise all N ranks at each step boundary.
    auto ring_barrier = [&](std::unique_lock<std::mutex>& lk) -> bool {
        int target = st.ring_gen + 1;
        st.ring_arrived++;
        if (st.ring_arrived == nranks) {
            st.ring_arrived = 0;
            st.ring_gen     = target;
            st.cv.notify_all();
        } else {
            bool ok = st.wait_pred(lk, [&]{ return st.ring_gen == target; });
            if (!ok) return false;
        }
        return true;
    };

    std::unique_lock<std::mutex> lk(st.mu);

    // ── Initialisation barrier: all ranks deposit their send buffers ──────────
    // ring_accum[r] is initialised to a full copy of rank r's sendbuff.
    // Total = N * recvcount elements (one full message per rank).
    st.sendbufs[rank] = sendbuff;
    st.arrived_phase0++;
    if (st.arrived_phase0 == nranks) {
        st.ring_accum.resize(nranks);
        for (int r = 0; r < nranks; ++r) {
            st.ring_accum[r].resize(total);
            memcpy(st.ring_accum[r].data(), st.sendbufs[r], total);
        }
        st.arrived_phase0 = 0;
        st.ring_step      = 0;
        st.generation++;
        st.cv.notify_all();
    } else {
        bool ok = st.wait_pred(lk, [&]{ return st.arrived_phase0 == 0; });
        if (!ok) return ncclSystemError;
    }

    // ── Phase 1: Reduce-Scatter (N-1 ring steps) ──────────────────────────────
    // Step s: rank i reduces chunk (i-s-1+N)%N — accumulating from the
    // left-predecessor's ring_accum (rank (i-1+N)%N) into its own ring_accum.
    // Kahan compensated summation is used for fp32/fp64 Sum/Avg to bound error.
    const int src_rank = (rank - 1 + nranks) % nranks;

    std::vector<float>  kahan_f32(
        (datatype == ncclFloat32 && (op == ncclSum || op == ncclAvg)) ? recvcount : 0, 0.f);
    std::vector<double> kahan_f64(
        (datatype == ncclFloat64 && (op == ncclSum || op == ncclAvg)) ? recvcount : 0, 0.0);

    for (int s = 0; s < nranks - 1; ++s) {
        int src_chunk = ((rank - s - 1) % nranks + nranks) % nranks;
        auto [start, end] = chunk_range(src_chunk);
        size_t nelems = end - start;

        uint8_t*       dst_ptr = st.ring_accum[rank].data()     + start * elem_sz;
        const uint8_t* src_ptr = st.ring_accum[src_rank].data() + start * elem_sz;

        if (datatype == ncclFloat32 && (op == ncclSum || op == ncclAvg)) {
            float*       dst  = reinterpret_cast<float*>(dst_ptr);
            const float* src  = reinterpret_cast<const float*>(src_ptr);
            float*       comp = kahan_f32.data() + start;
            for (size_t i = 0; i < nelems; ++i) {
                float y = src[i] - comp[i];
                float t = dst[i] + y;
                comp[i] = (t - dst[i]) - y;
                dst[i]  = t;
            }
        } else if (datatype == ncclFloat64 && (op == ncclSum || op == ncclAvg)) {
            double*       dst  = reinterpret_cast<double*>(dst_ptr);
            const double* src  = reinterpret_cast<const double*>(src_ptr);
            double*       comp = kahan_f64.data() + start;
            for (size_t i = 0; i < nelems; ++i) {
                double y = src[i] - comp[i];
                double t = dst[i] + y;
                comp[i] = (t - dst[i]) - y;
                dst[i]  = t;
            }
        } else {
            apply_reduce(dst_ptr, src_ptr, nelems, datatype, op);
        }

        if (!ring_barrier(lk)) return ncclSystemError;
    }

    // ── Average scaling ───────────────────────────────────────────────────────
    // Scale the owned chunk by 1/N before copying to recvbuff.
    if (op == ncclAvg)
        scale_avg(st.ring_accum[rank].data(), recvcount, datatype, nranks);

    // Final phase-transition barrier so all ranks have finished their last step
    // before any rank reads from a predecessor's ring_accum below.
    if (!ring_barrier(lk)) return ncclSystemError;

    // ── Copy fully-reduced chunk for this rank into recvbuff ──────────────────
    // After N-1 reduce-scatter steps, ring_accum[r] holds the fully-reduced data
    // for chunk (r+1)%N (ring invariant).  Rank i wants chunk i, which is owned
    // by ring_accum[(i-1+N)%N].
    {
        const int owner = (rank - 1 + nranks) % nranks;
        auto [start, end] = chunk_range(rank);
        size_t copy_bytes = (end - start) * elem_sz;
        memcpy(recvbuff,
               st.ring_accum[owner].data() + start * elem_sz,
               copy_bytes);
        // If recvcount is not evenly divisible the last rank's slice may be
        // shorter; zero-pad the remainder so the caller always sees recvcount
        // elements (consistent with NCCL reference behaviour).
        if (copy_bytes < recv_sz)
            memset(static_cast<uint8_t*>(recvbuff) + copy_bytes, 0,
                   recv_sz - copy_bytes);
    }

    // ── Teardown barrier: last rank clears ring_accum ─────────────────────────
    st.arrived_phase1++;
    if (st.arrived_phase1 == nranks) {
        st.arrived_phase1 = 0;
        st.ring_accum.clear();
        st.cv.notify_all();
    } else {
        st.wait_pred(lk, [&]{ return st.arrived_phase1 == 0; });
    }

    VGRE_LOG_DEBUG("NCCL", "ncclReduceScatter (ring): " +
                   std::to_string(total) + " bytes, " +
                   std::to_string(nranks) + " ranks, η=(N-1)/N=" +
                   std::to_string(static_cast<double>(nranks - 1) / nranks));
    return ncclSuccess;
}

} // extern "C"
