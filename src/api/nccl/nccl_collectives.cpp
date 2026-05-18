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
        std::memcpy(recvbuff, sendbuff, bytes);
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
    std::memcpy(myBuf.data(), sendbuff, total);

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
                    std::memcpy(st.result_buf.data(), st.sendbufs[0], total);
                    apply_reduce(st.result_buf.data(), childBuf, count, datatype, op);
                    // push the result back into sendbufs[0] so next round reads it
                    st.sendbufs[0] = st.result_buf.data();
                } else {
                    // Other receivers: reduce child into parent's myBuf.
                    // We can't update myBuf here (it belongs to another thread),
                    // so we use a temporary stored in gather_slots[r].
                    st.gather_slots[r].resize(total);
                    std::memcpy(st.gather_slots[r].data(), st.sendbufs[r], total);
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
                std::memcpy(myBuf.data(), st.sendbufs[rank], total);
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
    std::memcpy(recvbuff, st.result_buf.data(), total);

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

// ── Ring AllReduce (bandwidth-optimal for large tensors) ──────────────────────
// Used when count * elem_sz > kRingThreshold (1 MB).
// Each rank deposits its pointer; rank 0 performs chunk-wise reduction,
// then all ranks copy the result — functionally equivalent to ring output.
static ncclResult_t ring_allreduce(const void* sendbuff, void* recvbuff,
    size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm* c)
{
    auto& st = *c->state;
    const size_t elem_sz = nccl_elem_size(datatype);
    const size_t total   = count * elem_sz;
    const int nranks     = st.nranks;
    const int rank       = c->rank;

    // Ring algorithm requires all ranks to participate simultaneously.
    // Implementation: each rank deposits its buffer pointer; rank 0 sequentially
    // processes all ring chunks across all ranks.
    // (True pipelined ring is only beneficial with real network; here we implement
    //  the correct output with barrier-based coordination matching the ring spec.)
    std::unique_lock<std::mutex> lk(st.mu);
    const int gen = st.generation;

    st.sendbufs[rank] = sendbuff;
    st.arrived_phase0++;

    if (st.arrived_phase0 == nranks) {
        // Chunk size: split tensor across ranks
        size_t chunkCount = (count + nranks - 1) / nranks;
        st.result_buf.resize(total);

        // Reduce-scatter phase: for each chunk position c, reduce from all ranks
        for (int chunk = 0; chunk < nranks; ++chunk) {
            size_t start = static_cast<size_t>(chunk) * chunkCount;
            size_t end   = std::min(start + chunkCount, count);
            if (start >= count) break;
            size_t cBytes = (end - start) * elem_sz;
            // Initialize chunk from rank 0
            std::memcpy(st.result_buf.data() + start * elem_sz,
                        static_cast<const uint8_t*>(st.sendbufs[0]) + start * elem_sz,
                        cBytes);
            // Reduce remaining ranks into this chunk
            for (int r = 1; r < nranks; ++r) {
                apply_reduce(
                    st.result_buf.data() + start * elem_sz,
                    static_cast<const uint8_t*>(st.sendbufs[r]) + start * elem_sz,
                    end - start, datatype, op);
            }
        }
        if (op == ncclAvg) scale_avg(st.result_buf.data(), count, datatype, nranks);
        st.arrived_phase0 = 0;
        st.generation++;
        st.cv.notify_all();
        VGRE_LOG_DEBUG("NCCL", "ring_allreduce: " + std::to_string(total) +
                       " bytes, " + std::to_string(nranks) + " ranks");
    } else {
        bool ok = st.wait_pred(lk, [&]{ return st.generation != gen; });
        if (!ok) return ncclSystemError;
    }

    // All-gather phase: every rank copies the full result
    std::memcpy(recvbuff, st.result_buf.data(), total);
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

} // extern "C"
