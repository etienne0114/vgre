// Tensor- & pipeline-parallel execution for the HLO engine — milestone L5.
//
// Models that don't fit one box are split across ranks. This module is the
// sharding *math* — how a matmul decomposes across ranks and which collective
// recombines the pieces — i.e. the computational core of Megatron-LM tensor
// parallelism and GSPMD. The collective itself (all-gather / all-reduce) is
// provided here as an in-process Communicator for single-process TP and tests;
// the identical interface is what the cluster's real transport implements
// (collective_ops_manager.cpp's NCCL-style collectives, or the portable
// SoftwareRDMA), so the same decomposition runs across machines unchanged.
//
// Two canonical linear shardings (a transformer FFN/attention projection uses
// both, back to back, so only one collective fires per pair):
//   • Column-parallel: W[K,N] split along N. Each rank computes X·W_r = [M,N_r];
//     ALL-GATHER (concat along N) → [M,N].
//   • Row-parallel: W[K,N] split along K and X[M,K] along K. Each rank computes
//     X_r·W_r = [M,N] partial; ALL-REDUCE (sum) → [M,N].
#ifndef VGRE_XLA_PARALLEL_H
#define VGRE_XLA_PARALLEL_H

#include <utility>
#include <vector>

#include "vgre/xla/hlo.h"

namespace vgre {
namespace xla {

// ── Collectives over `ranks` per-rank buffers ────────────────────────────────
// In-process implementation; a real backend (RDMA/NCCL) implements the same two
// reductions across machines. Each rank ends up with the full result.
struct Communicator {
    // Element-wise sum of identically-shaped per-rank buffers (all-reduce SUM).
    static Literal allReduceSum(const std::vector<Literal>& perRank);
    // Concatenate per-rank [M, N_r] buffers along the last (N) axis (all-gather).
    static Literal allGatherConcatLastAxis(const std::vector<Literal>& perRank);
};

// ── Weight / activation sharding ─────────────────────────────────────────────
// W[K,N] → `ranks` shards [K, N_r] (split columns, near-even).
std::vector<Literal> shardColumns(const Literal& W, int ranks);
// W[K,N] → `ranks` shards [K_r, N] (split rows, near-even).
std::vector<Literal> shardRows(const Literal& W, int ranks);
// X[M,K] → `ranks` shards [M, K_r] (split the contracting dim, matches shardRows).
std::vector<Literal> shardContracting(const Literal& X, int ranks);

// ── Tensor-parallel matmul (result identical to single-node X·W) ─────────────
// Column-parallel: shard W along N, per-rank GEMM, all-gather. Exact.
Literal columnParallelMatmul(const Literal& X, const Literal& W, int ranks);
// Row-parallel: shard W along K and X along K, per-rank GEMM, all-reduce sum.
// Equal to single-node up to float reassociation.
Literal rowParallelMatmul(const Literal& X, const Literal& W, int ranks);

// ── Pipeline parallelism ─────────────────────────────────────────────────────
// Partition `numLayers` into `stages` balanced contiguous [begin,end) ranges
// (earlier stages get the +1 when it doesn't divide evenly).
std::vector<std::pair<int, int>> pipelinePartition(int numLayers, int stages);

}  // namespace xla
}  // namespace vgre

#endif  // VGRE_XLA_PARALLEL_H
