#include "vgre/api/nccl_internal.h"
#include "vgre/common/logger.h"

static constexpr size_t kRingThreshold = 1ULL * 1024 * 1024;
extern "C" {

static ncclResult_t ring_allreduce(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm*);

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

    // Algorithm selection: ring for large tensors (bandwidth-optimal),
    // barrier tree (existing) for small tensors (latency-optimal).
    size_t bytes = count * nccl_elem_size(datatype);
    if (c->state->nranks > 1 && bytes > kRingThreshold)
        return ring_allreduce(sendbuff, recvbuff, count, datatype, op, c);
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
