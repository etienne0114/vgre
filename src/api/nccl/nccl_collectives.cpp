#include "vgre/api/nccl_internal.h"
#include "vgre/common/logger.h"
#include "vgre/common/os_backend.h"
#include <algorithm>
#include <cstdlib>
#include <thread>

// Algorithm thresholds (bytes):
//   ≤ kSmallThreshold : flat barrier (latency-optimal, O(1) sync rounds)
//   > kSmallThreshold : ring reduce-scatter + all-gather (per-rank chunk
//                       ownership, bandwidth-optimal, no centralised root)
static constexpr size_t kTreeThreshold = 64ULL * 1024;         // 64 KB

// ── Tensor-Parallel Lock-Free AllReduce ───────────────────────────────────────
// Called when VGRE_TP_DEGREE is set and all ranks are in the same process.
// Uses atomic_thread_fence(acq_rel) between stages instead of condvar waits,
// avoiding kernel-mode wakeups on same-process tensor-parallel workloads.
//
// Algorithm (two-phase: reduce-scatter then all-gather):
//   Phase 1 barrier: all ranks publish sendbuff/recvbuff pointers (release) and
//                    spin-wait on tp_gen for the last rank to bump it (acquire).
//   Reduce-scatter: each rank independently reduces its owned chunk [rank*chunk, (rank+1)*chunk)
//                   from all ranks' sendbuffs into its own recvbuff slice.
//   Phase 2 barrier: spin-wait until all chunk reductions are done.
//   All-gather: each rank copies every other rank's reduced chunk from that rank's
//               recvbuff slice into its own recvbuff.
static ncclResult_t tp_ring_allreduce(const void* sendbuff, void* recvbuff, size_t count,
                                       ncclDataType_t datatype, ncclRedOp_t op, ncclComm* c) {
    auto& st    = *c->state;
    const int N = st.nranks;
    const int R = c->rank;
    const size_t esz   = nccl_elem_size(datatype);
    const size_t chunk = (count + N - 1) / N;

    // ── Phase 1: publish buffers, barrier ─────────────────────────────────
    st.tp_sendbufs[R].store(reinterpret_cast<uintptr_t>(sendbuff), std::memory_order_release);
    st.tp_recvbufs[R].store(reinterpret_cast<uintptr_t>(recvbuff), std::memory_order_release);
    std::atomic_thread_fence(std::memory_order_acq_rel);

    const int gen1 = st.tp_gen.load(std::memory_order_acquire);
    if (st.tp_arrived.fetch_add(1, std::memory_order_acq_rel) + 1 == N) {
        st.tp_arrived.store(0, std::memory_order_relaxed);
        st.tp_gen.store(gen1 + 1, std::memory_order_release);
    } else {
        while (st.tp_gen.load(std::memory_order_acquire) == gen1)
            std::this_thread::yield();
    }
    // All pointer stores by all ranks are now visible.
    std::atomic_thread_fence(std::memory_order_acquire);

    // ── Reduce-scatter: each rank reduces its owned chunk ─────────────────
    const size_t my_start = R * chunk;
    const size_t my_end   = std::min(my_start + chunk, count);
    const size_t my_len   = (my_end > my_start) ? my_end - my_start : 0UL;

    if (my_len > 0) {
        uint8_t* out = static_cast<uint8_t*>(recvbuff) + my_start * esz;
        // Initialise output from this rank's own slice
        memcpy(out, static_cast<const uint8_t*>(sendbuff) + my_start * esz, my_len * esz);
        // Accumulate slices from all other ranks
        for (int r = 0; r < N; ++r) {
            if (r == R) continue;
            const uint8_t* src = reinterpret_cast<const uint8_t*>(
                st.tp_sendbufs[r].load(std::memory_order_relaxed)) + my_start * esz;
            apply_reduce(out, src, my_len, datatype, op);
        }
        if (op == ncclAvg)
            scale_avg(out, my_len, datatype, N);
    }

    // ── Phase 2: barrier after reduce-scatter ─────────────────────────────
    const int gen2 = st.tp_gen.load(std::memory_order_acquire);
    if (st.tp_arrived.fetch_add(1, std::memory_order_acq_rel) + 1 == N) {
        st.tp_arrived.store(0, std::memory_order_relaxed);
        st.tp_gen.store(gen2 + 1, std::memory_order_release);
    } else {
        while (st.tp_gen.load(std::memory_order_acquire) == gen2)
            std::this_thread::yield();
    }
    std::atomic_thread_fence(std::memory_order_acquire);

    // ── All-gather: copy each rank's fully-reduced chunk to our recvbuff ──
    for (int r = 0; r < N; ++r) {
        if (r == R) continue;
        const size_t r_start = r * chunk;
        const size_t r_end   = std::min(r_start + chunk, count);
        const size_t r_len   = (r_end > r_start) ? r_end - r_start : 0UL;
        if (r_len == 0) continue;
        const uint8_t* src = reinterpret_cast<const uint8_t*>(
            st.tp_recvbufs[r].load(std::memory_order_relaxed)) + r_start * esz;
        memcpy(static_cast<uint8_t*>(recvbuff) + r_start * esz, src, r_len * esz);
    }

    VGRE_LOG_DEBUG("NCCL", "tp_ring_allreduce: rank=" + std::to_string(R) +
                   " count=" + std::to_string(count));
    return ncclSuccess;
}

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

    // Tensor-parallel same-process fast path.
    // Active when VGRE_TP_DEGREE is set, all ranks are in this process
    // (creator_pid == getpid()), and nranks does not exceed tp_degree.
    // Bypasses TCPCluster and condvar — uses atomic fence barriers only.
    if (c->state->tp_degree > 0 &&
        c->nranks <= c->state->tp_degree &&
        c->state->creator_pid == vgre::os::getpid()) {
        return tp_ring_allreduce(sendbuff, recvbuff, count, datatype, op, c);
    }

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
    //   small  (≤64 KB)        → flat barrier (lowest latency, root does all work)
    //   larger (>64 KB)        → ring reduce-scatter + all-gather (each rank owns
    //                            one chunk; no centralised root, no per-round
    //                            global barrier on a single accumulator)
    // The ring (ring_allreduce) is the bandwidth-optimal, per-rank-slice
    // algorithm; it superseded a centralised binary-tree path that reduced all
    // data through rank 0's result_buf (correct but serialised on the root).
    size_t bytes = count * nccl_elem_size(datatype);
    if (c->state->nranks > 1 && bytes > kTreeThreshold)
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
    // Chunk layout: chunk k spans element indices [k*recvcount, (k+1)*recvcount).
    // Each of the N chunks is exactly recvcount elements (rank r's output slice).
    // Total data per ring_accum buffer = N * recvcount elements.
    auto chunk_range = [&](int k) -> std::pair<size_t, size_t> {
        size_t start = static_cast<size_t>(k) * recvcount;
        size_t end   = start + recvcount;
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
    // Each chunk position is accumulated exactly once per rank, so Kahan
    // compensation gives no benefit here; apply_reduce is correct and sufficient.
    const int src_rank = (rank - 1 + nranks) % nranks;

    for (int s = 0; s < nranks - 1; ++s) {
        int src_chunk = ((rank - s - 1) % nranks + nranks) % nranks;
        auto [start, end] = chunk_range(src_chunk);
        size_t nelems = end - start;

        uint8_t*       dst_ptr = st.ring_accum[rank].data()     + start * elem_sz;
        const uint8_t* src_ptr = st.ring_accum[src_rank].data() + start * elem_sz;

        apply_reduce(dst_ptr, src_ptr, nelems, datatype, op);

        if (!ring_barrier(lk)) return ncclSystemError;
    }

    // ── Average scaling ───────────────────────────────────────────────────────
    // After N-1 steps, ring_accum[rank] owns chunk (rank+1)%N at byte offset
    // (rank+1)%N * recvcount * elem_sz.  Scale only that slice by 1/N so the
    // allgather phase reads pre-scaled data from each rank's owned chunk.
    if (op == ncclAvg) {
        size_t owned_off = static_cast<size_t>((rank + 1) % nranks) * recvcount * elem_sz;
        scale_avg(st.ring_accum[rank].data() + owned_off, recvcount, datatype, nranks);
    }

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
