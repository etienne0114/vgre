#ifndef VGRE_CORE_CLUSTER_H
#define VGRE_CORE_CLUSTER_H

// Phase 3, Track P3-6 — Thread-block clusters + Distributed Shared Memory (DSMEM).
//
// Hopper (SM90) groups CTAs into a *cluster*: a fixed set of thread blocks that
// are co-scheduled and share a Distributed Shared Memory window — any CTA may
// read / write / atomically update a peer CTA's shared memory through
// `mapa.shared::cluster`, and the whole cluster rendezvouses at
// `barrier.cluster.arrive` / `barrier.cluster.wait` (a.k.a. `cluster.sync`).
//
// This models that exactly on VGRE's CPU substrate, where every CTA runs on its
// own OS thread with its own shared buffer (executor `getThreadSharedMem`). The
// Cluster keeps a rank→SMEM-base table, so `map_shared_rank` retargets an address
// to the SAME byte offset in a peer CTA's shared memory — a real cross-thread
// access — and `sync()` is a real sense-reversing barrier over the cluster's CTAs
// (the same algorithm the cooperative-grid barrier uses), hence safely reusable
// across multiple cluster phases.

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

namespace vgre {
namespace cluster {

// Cluster shape: CTAs along each axis (cudaLaunchAttributeClusterDimension).
struct ClusterDim {
    unsigned x = 1, y = 1, z = 1;
    unsigned size() const { return x * y * z; }
};

// Linear cluster rank ↔ 3-D cluster coordinate (x fastest, then y, then z) —
// the CTA-rank-in-cluster ordering of `%cluster_ctarank` / `cluster.map_shared`.
inline unsigned cluster_rank(const ClusterDim& d, unsigned cx, unsigned cy, unsigned cz) {
    return (cz * d.y + cy) * d.x + cx;
}
inline void cluster_coord(const ClusterDim& d, unsigned rank,
                          unsigned& cx, unsigned& cy, unsigned& cz) {
    cx = rank % d.x;
    cy = (rank / d.x) % d.y;
    cz = rank / (static_cast<std::uint64_t>(d.x) * d.y);
}

// One thread-block cluster: a rank→SMEM-base table plus a reusable cluster barrier.
class Cluster {
public:
    explicit Cluster(unsigned size) : size_(size), smemBase_(size) {
        for (auto& a : smemBase_) a.store(nullptr, std::memory_order_relaxed);
    }
    Cluster(const Cluster&) = delete;
    Cluster& operator=(const Cluster&) = delete;

    unsigned size() const { return size_; }

    // A CTA publishes its shared-memory base before any DSMEM access (thread-safe).
    void register_smem(unsigned rank, void* base) {
        smemBase_[rank].store(base, std::memory_order_release);
    }

    // mapa.shared::cluster — translate an address in CTA `myRank`'s shared memory
    // to the SAME byte offset in CTA `dstRank`'s shared memory (Distributed SMEM).
    // The offset is preserved exactly, so a peer's `smem[i]` maps to `smem[i]`.
    void* map_shared_rank(unsigned myRank, const void* myAddr, unsigned dstRank) const {
        const auto* myBase =
            static_cast<const std::uint8_t*>(smemBase_[myRank].load(std::memory_order_acquire));
        auto* dstBase =
            static_cast<std::uint8_t*>(smemBase_[dstRank].load(std::memory_order_acquire));
        const std::ptrdiff_t off = static_cast<const std::uint8_t*>(myAddr) - myBase;
        return dstBase + off;
    }

    // Read-only peer base (for raw DSMEM addressing / typed helpers).
    void* smem_base(unsigned rank) const {
        return smemBase_[rank].load(std::memory_order_acquire);
    }

    // barrier.cluster (cluster.sync) — all `size` CTAs rendezvous. Sense-reversing
    // (generation counter) so it is correct when reused across cluster phases.
    void sync() {
        if (size_ <= 1) return;
        const unsigned gen = generation_.load(std::memory_order_acquire);
        if (arrived_.fetch_add(1, std::memory_order_acq_rel) + 1 == size_) {
            arrived_.store(0, std::memory_order_release);   // last CTA: reset + wake
            {
                std::lock_guard<std::mutex> lk(mu_);
                generation_.fetch_add(1, std::memory_order_release);
            }
            cv_.notify_all();
        } else {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [&] { return generation_.load(std::memory_order_acquire) != gen; });
        }
    }

private:
    unsigned size_;
    std::vector<std::atomic<void*>> smemBase_;
    std::atomic<unsigned> arrived_{0};
    std::atomic<unsigned> generation_{0};
    std::mutex mu_;
    std::condition_variable cv_;
};

// Typed DSMEM access helpers (mapa + dereference), mirroring the PTX pattern of
// `mapa.shared::cluster.u32` followed by a `ld`/`st`/`atom` to the mapped address.
template <class T>
inline T dsmem_load(const Cluster& c, unsigned myRank, const T* myAddr, unsigned dstRank) {
    return *static_cast<const T*>(c.map_shared_rank(myRank, myAddr, dstRank));
}
template <class T>
inline void dsmem_store(const Cluster& c, unsigned myRank, T* myAddr, unsigned dstRank, T v) {
    *static_cast<T*>(c.map_shared_rank(myRank, myAddr, dstRank)) = v;
}

}  // namespace cluster
}  // namespace vgre

#endif  // VGRE_CORE_CLUSTER_H
