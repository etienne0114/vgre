// GSPMD-style automatic partitioning for the HLO engine — milestone L5.
//
// Given a single-program HloModule plus a sharding annotation on its parameters,
// this pass *propagates* sharding through the graph and *inserts the collectives*
// itself — exactly the job GSPMD / jax.shard_map do: the user says how inputs are
// laid out across ranks and the compiler figures out the communication.
//
// Propagation + collective rules (tensor parallelism):
//   • Parameter: sharded per annotation, else replicated.
//   • Elementwise (unary/binary, output-shape == operand-shape): sharding flows
//     through unchanged; computed per-rank on the local shard.
//   • Dot [M,K]·[K,N]:
//       rhs sharded on N, lhs replicated  → column-parallel; output sharded on N
//                                           (no collective yet).
//       lhs sharded on K and rhs on K     → row-parallel; ALL-REDUCE the partial
//                                           sums → output replicated.
//       both replicated                   → replicated.
//   • At the root, a still-sharded result is ALL-GATHERed to replicated.
//
// gspmdExecute runs the partitioned program across `ranks` and returns the
// (replicated) result — identical to single-node HloModule::evaluate.
#ifndef VGRE_XLA_GSPMD_H
#define VGRE_XLA_GSPMD_H

#include <map>
#include <vector>

#include "vgre/xla/hlo.h"

namespace vgre {
namespace xla {

struct Sharding {
    int dim = -1;  // -1 = replicated; otherwise sharded across the ranks on this dim
    bool replicated() const { return dim < 0; }
    static Sharding replicate() { return Sharding{-1}; }
    static Sharding on(int d) { return Sharding{d}; }
};

// Partition `m` under the given per-parameter sharding and execute across
// `ranks`, auto-inserting all-reduce / all-gather. Returns the replicated result.
Literal gspmdExecute(const HloModule& m,
                     const std::vector<Literal>& params,
                     const std::map<int, Sharding>& paramSharding,
                     int ranks);

}  // namespace xla
}  // namespace vgre

#endif  // VGRE_XLA_GSPMD_H
