// Track 13 — graph optimizer node-set consistency after structural passes.
//
// GraphOptimizer::sortTopologically() reorders graph.nodes but previously left
// graph.nodeIndex (the O(1) nodeId→index map used by dependency resolution and
// execution) pointing at the PRE-sort slots — a stale, silently-wrong index.
// fuseNodes() had the same defect after erasing the fused nodes, plus it only
// rewired consumers of one of the two fused nodes (a fork A→{B,C} would leave C
// dangling). This test verifies, through the public API, that after a structural
// pass the graph is internally consistent: topological order, a rebuilt index,
// and no dangling dependency edges.

#include "vgre/core/graph_optimizer.h"
#include "vgre/core/graph_manager.h"

#include <cstdio>
#include <unordered_set>

using namespace vgre;
using namespace vgre::core;

static int g_pass = 0, g_total = 0;
static void check(const char *name, bool ok) {
    ++g_total;
    printf(ok ? "  PASS  %s\n" : "  FAIL  %s\n", name);
    if (ok) ++g_pass;
}

static GraphNode mk(uint64_t id, GraphNodeType t, std::vector<uint64_t> deps) {
    GraphNode n;
    n.type = t;
    n.nodeId = id;
    n.kernelName = "n" + std::to_string(id);
    n.gridDim = dim3(1, 1, 1);
    n.blockDim = dim3(1, 1, 1);
    n.deps = std::move(deps);
    return n;
}

// Assert internal consistency invariants of a graph.
static bool graphConsistent(const Graph &g) {
    std::unordered_set<uint64_t> ids;
    for (const auto &n : g.nodes) ids.insert(n.nodeId);
    // index map matches the vector exactly
    if (g.nodeIndex.size() != g.nodes.size()) return false;
    for (size_t i = 0; i < g.nodes.size(); ++i) {
        auto it = g.nodeIndex.find(g.nodes[i].nodeId);
        if (it == g.nodeIndex.end() || it->second != i) return false;
    }
    // no dangling edges
    for (const auto &n : g.nodes)
        for (uint64_t d : n.deps)
            if (ids.count(d) == 0) return false;
    return true;
}

// Verify every node appears AFTER all of its dependencies.
static bool isTopological(const Graph &g) {
    std::unordered_set<uint64_t> seen;
    for (const auto &n : g.nodes) {
        for (uint64_t d : n.deps)
            if (seen.count(d) == 0) return false; // dep came later → not topological
        seen.insert(n.nodeId);
    }
    return true;
}

int main() {
    printf("=== Graph Optimizer Consistency (Track 13) ===\n");

    // Build a graph in DELIBERATELY non-topological order: C(deps B), B(deps A), A.
    // A fork too: D depends on A as well, so a structural pass must keep D's edge
    // valid (not dangling) and rebuild the index.
    Graph g;
    g.id = 1;
    g.nodes.push_back(mk(3, GraphNodeType::KERNEL, {2}));   // C → B
    g.nodes.push_back(mk(4, GraphNodeType::KERNEL, {1}));   // D → A (the fork)
    g.nodes.push_back(mk(2, GraphNodeType::KERNEL, {1}));   // B → A
    g.nodes.push_back(mk(1, GraphNodeType::KERNEL, {}));    // A (root)
    g.nextNodeId = 5;
    g.nodeIndex.clear();
    for (size_t i = 0; i < g.nodes.size(); ++i)
        g.nodeIndex[g.nodes[i].nodeId] = i;

    check("pre-sort order is NOT topological (test premise)", !isTopological(g));

    VGREResult r = GraphOptimizer::sortTopologically(g);
    check("sortTopologically succeeded", r == VGREResult::SUCCESS);
    check("nodes now in topological order", isTopological(g));
    check("nodeIndex rebuilt & consistent, no dangling edges (the fix)",
          graphConsistent(g));

    // Spot-check: the index actually resolves a node correctly after the sort.
    auto it = g.nodeIndex.find(4);
    bool idxResolves = (it != g.nodeIndex.end() &&
                        it->second < g.nodes.size() &&
                        g.nodes[it->second].nodeId == 4 &&
                        g.nodes[it->second].deps.size() == 1 &&
                        g.nodes[it->second].deps[0] == 1);
    check("nodeIndex resolves node D→ its real slot (fork edge intact)", idxResolves);

    printf("\n%d / %d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
