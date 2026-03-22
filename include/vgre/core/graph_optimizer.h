#ifndef VGRE_CORE_GRAPH_OPTIMIZER_H
#define VGRE_CORE_GRAPH_OPTIMIZER_H

#include "vgre/common/error_codes.h"

namespace vgre {
namespace core {

struct Graph;
struct GraphNode;

/**
 * @brief Performs optimization passes on VGRE Graphs.
 * 
 * The primary optimization is Dynamic JIT Fusion, which merges sequential
 * kernels with identical dimensions into a single execution unit.
 */
class GraphOptimizer {
public:
    static VGREResult optimize(Graph& graph);

private:
    static bool areFusible(const GraphNode& a, const GraphNode& b);
    static VGREResult fuseNodes(Graph& graph, size_t idxA, size_t idxB);
};

} // namespace core
} // namespace vgre

#endif // VGRE_CORE_GRAPH_OPTIMIZER_H
