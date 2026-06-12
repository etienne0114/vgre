#ifndef VGRE_CORE_INTERCONNECT_H
#define VGRE_CORE_INTERCONNECT_H

// Phase 3, Track P3-15 — Virtual NVLink/NVSwitch topology + collective cost model.
//
// VGRE's multi-vGPU collectives compute correct results but, sharing host memory,
// model no interconnect — so collective *timing* is unrealistic for scaling
// studies. This module adds a configurable virtual topology (NVLink domains +
// NVSwitch all-to-all, PCIe fallback between domains) and the standard α-β cost
// model for ring and tree collectives:
//     T(n) = (#steps)·α  +  (bytes-per-step)·β,   β = 1/bottleneck_bandwidth.
// Ring AllReduce is bandwidth-optimal: 2(P−1) steps each moving n/P bytes →
//     T_ring = 2(P−1)·α + (2(P−1)/P)·n·β.
// Tree AllReduce is latency-optimal: 2·⌈log₂P⌉ steps each moving n bytes →
//     T_tree = 2·⌈log₂P⌉·(α + n·β).
// The numerics of a collective are unchanged; only the reported cost reflects
// the topology. All times in microseconds, bandwidth in GB/s.

#include <cstdint>

namespace vgre {
namespace dist {

struct Topology {
    int    numGpus    = 8;
    int    domainSize = 8;        // GPUs per full-bandwidth NVLink/NVSwitch domain
    double nvlinkGBps = 300.0;    // intra-domain bandwidth (e.g. NVLink/NVSwitch)
    double pcieGBps   = 32.0;     // inter-domain fallback bandwidth (PCIe)
    double latencyUs  = 1.5;      // per-hop latency

    // The collective's bottleneck bandwidth: PCIe when it must cross domains.
    double bottleneckGBps() const {
        return (numGpus > domainSize) ? pcieGBps : nvlinkGBps;
    }
};

// µs per byte at the given bandwidth (GB/s → bytes/µs is bw·1e3).
inline double beta_us_per_byte(double gbps) { return 1.0 / (gbps * 1.0e3); }

inline int ceil_log2(int p) { int s = 0; for (int x = 1; x < p; x <<= 1) ++s; return s; }

// Ring AllReduce cost (reduce-scatter + all-gather).
inline double ring_allreduce_us(const Topology& t, double bytes) {
    const int P = t.numGpus;
    if (P <= 1) return 0.0;
    const double beta = beta_us_per_byte(t.bottleneckGBps());
    return 2.0 * (P - 1) * t.latencyUs + (2.0 * (P - 1) / P) * bytes * beta;
}

// Tree (recursive-doubling-style) AllReduce cost.
inline double tree_allreduce_us(const Topology& t, double bytes) {
    const int P = t.numGpus;
    if (P <= 1) return 0.0;
    const int steps = ceil_log2(P);
    const double beta = beta_us_per_byte(t.bottleneckGBps());
    return 2.0 * steps * (t.latencyUs + bytes * beta);
}

// AllGather cost (ring): P−1 steps each moving n/P bytes.
inline double ring_allgather_us(const Topology& t, double bytes) {
    const int P = t.numGpus;
    if (P <= 1) return 0.0;
    const double beta = beta_us_per_byte(t.bottleneckGBps());
    return (P - 1) * t.latencyUs + ((double)(P - 1) / P) * bytes * beta;
}

// Pick the cheaper algorithm for a given message size (the real NCCL choice).
inline double best_allreduce_us(const Topology& t, double bytes) {
    const double r = ring_allreduce_us(t, bytes), tr = tree_allreduce_us(t, bytes);
    return (r < tr) ? r : tr;
}

} // namespace dist
} // namespace vgre

#endif // VGRE_CORE_INTERCONNECT_H
