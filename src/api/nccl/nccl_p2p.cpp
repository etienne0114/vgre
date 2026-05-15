#include "vgre/api/nccl_internal.h"
#include "vgre/common/logger.h"

extern "C" {

// ── Point-to-Point: ncclSend / ncclRecv ──────────────────────────────────────
// Single-node: copy via shared-memory p2p_slots.  Multi-node: would route
// through TCPClusterManager (future work; see implementationPlan.md Phase 7).
ncclResult_t ncclSend(const void* sendbuff, size_t count,
                      ncclDataType_t datatype, int peer,
                      ncclComm_t comm, void* /*stream*/) {
    if (!comm || !sendbuff) return ncclInvalidArgument;
    auto* c = static_cast<ncclComm*>(comm);
    auto& st = *c->state;
    const size_t bytes = count * nccl_elem_size(datatype);

    std::unique_lock<std::mutex> lk(st.mu);
    const int gen = st.generation;

    // Write sender's data into the peer's slot
    st.p2p_slots[peer].resize(bytes);
    std::memcpy(st.p2p_slots[peer].data(), sendbuff, bytes);

    st.arrived_phase0++;
    if (st.arrived_phase0 == st.nranks) {
        st.arrived_phase0 = 0;
        st.generation++;
        st.cv.notify_all();
    } else {
        bool ok = st.wait_pred(lk, [&]{ return st.generation != gen; });
        if (!ok) return ncclSystemError;
    }
    return ncclSuccess;
}
ncclResult_t ncclRecv(void* recvbuff, size_t count,
                      ncclDataType_t datatype, int peer,
                      ncclComm_t comm, void* /*stream*/) {
    if (!comm || !recvbuff) return ncclInvalidArgument;
    auto* c = static_cast<ncclComm*>(comm);
    auto& st = *c->state;
    const size_t bytes = count * nccl_elem_size(datatype);

    std::unique_lock<std::mutex> lk(st.mu);
    const int gen = st.generation;

    // Read from the slot that the peer wrote into for this rank
    st.arrived_phase0++;
    if (st.arrived_phase0 == st.nranks) {
        st.arrived_phase0 = 0;
        st.generation++;
        st.cv.notify_all();
    } else {
        bool ok = st.wait_pred(lk, [&]{ return st.generation != gen; });
        if (!ok) return ncclSystemError;
    }

    // After barrier, copy the peer's data
    if (st.p2p_slots[c->rank].size() >= bytes)
        std::memcpy(recvbuff, st.p2p_slots[c->rank].data(), bytes);
    return ncclSuccess;
}
// ── ncclAllToAll ─────────────────────────────────────────────────────────────
// Each rank sends recvcount elements to every other rank.
// recvbuff is nranks × recvcount elements; rank r receives from rank i at
// offset i × recvcount.
ncclResult_t ncclAllToAll(const void* sendbuff, void* recvbuff,
                          size_t count, ncclDataType_t datatype,
                          ncclComm_t comm, void* /*stream*/) {
    if (!comm || !sendbuff || !recvbuff) return ncclInvalidArgument;
    auto* c = static_cast<ncclComm*>(comm);
    auto& st = *c->state;
    const size_t elem_sz = nccl_elem_size(datatype);
    const size_t chunk_bytes = count * elem_sz;
    const size_t total_bytes = chunk_bytes * st.nranks;

    std::unique_lock<std::mutex> lk(st.mu);
    const int gen = st.generation;

    // Deposit this rank's chunks into every peer's slot
    for (int peer = 0; peer < st.nranks; ++peer) {
        st.p2p_slots[peer].resize(total_bytes);
        std::memcpy(st.p2p_slots[peer].data() + c->rank * chunk_bytes,
                    static_cast<const uint8_t*>(sendbuff) + peer * chunk_bytes,
                    chunk_bytes);
    }

    st.arrived_phase0++;
    if (st.arrived_phase0 == st.nranks) {
        st.arrived_phase0 = 0;
        st.generation++;
        st.cv.notify_all();
    } else {
        bool ok = st.wait_pred(lk, [&]{ return st.generation != gen; });
        if (!ok) return ncclSystemError;
    }

    // Copy data from all peers into this rank's recvbuff
    std::memcpy(recvbuff, st.p2p_slots[c->rank].data(), total_bytes);

    st.arrived_phase1++;
    if (st.arrived_phase1 == st.nranks) {
        st.arrived_phase1 = 0;
        for (auto& slot : st.p2p_slots) slot.clear();
        st.cv.notify_all();
    } else {
        st.wait_pred(lk, [&]{ return st.arrived_phase1 == 0; });
    }
    return ncclSuccess;
}
// ── ncclGather ───────────────────────────────────────────────────────────────
// All ranks send `sendcount` elements; root concatenates them into `recvbuff`.
ncclResult_t ncclGather(const void* sendbuff, void* recvbuff,
                        size_t sendcount, ncclDataType_t datatype,
                        int root, ncclComm_t comm, void* /*stream*/) {
    if (!comm || !sendbuff) return ncclInvalidArgument;
    auto* c = static_cast<ncclComm*>(comm);
    auto& st = *c->state;
    const size_t elem_sz = nccl_elem_size(datatype);
    const size_t chunk_bytes = sendcount * elem_sz;

    std::unique_lock<std::mutex> lk(st.mu);
    const int gen = st.generation;

    st.sendbufs[c->rank] = sendbuff;
    st.root_rank = root;
    st.arrived_phase0++;

    if (st.arrived_phase0 == st.nranks) {
        if (recvbuff) {
            for (int r = 0; r < st.nranks; ++r)
                std::memcpy(static_cast<uint8_t*>(recvbuff) + r * chunk_bytes,
                            st.sendbufs[r], chunk_bytes);
        }
        st.sendbufs.assign(st.nranks, nullptr);
        st.arrived_phase0 = 0;
        st.generation++;
        st.cv.notify_all();
    } else {
        bool ok = st.wait_pred(lk, [&]{ return st.generation != gen; });
        if (!ok) return ncclSystemError;
    }
    return ncclSuccess;
}
// ── ncclScatter ──────────────────────────────────────────────────────────────
// Root sends `sendcount` elements to each rank (total `nranks × sendcount`).
ncclResult_t ncclScatter(const void* sendbuff, void* recvbuff,
                         size_t recvcount, ncclDataType_t datatype,
                         int root, ncclComm_t comm, void* /*stream*/) {
    if (!comm || !recvbuff) return ncclInvalidArgument;
    auto* c = static_cast<ncclComm*>(comm);
    auto& st = *c->state;
    const size_t elem_sz = nccl_elem_size(datatype);
    const size_t chunk_bytes = recvcount * elem_sz;

    std::unique_lock<std::mutex> lk(st.mu);
    const int gen = st.generation;

    if (c->rank == root) {
        if (!sendbuff) return ncclInvalidArgument;
        st.root_sendbuf = sendbuff;
    }
    st.root_rank = root;
    st.arrived_phase0++;

    if (st.arrived_phase0 == st.nranks) {
        std::memcpy(recvbuff,
                    static_cast<const uint8_t*>(st.root_sendbuf) + c->rank * chunk_bytes,
                    chunk_bytes);
        st.root_sendbuf = nullptr;
        st.arrived_phase0 = 0;
        st.generation++;
        st.cv.notify_all();
    } else {
        bool ok = st.wait_pred(lk, [&]{ return st.generation != gen; });
        if (!ok) return ncclSystemError;
    }
    return ncclSuccess;
}

} // extern "C"
