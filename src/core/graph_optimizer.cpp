#include "vgre/core/graph_optimizer.h"
#include "vgre/core/graph_manager.h"
#include "vgre/core/runtime_engine.h"
#include "vgre/common/logger.h"
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <map>

namespace vgre {
namespace core {

VGREResult GraphOptimizer::optimize(Graph& graph) {
    if (graph.nodes.empty()) return VGREResult::SUCCESS;

    VGRE_LOG_INFO("GraphOptimizer", "Starting optimization pass on graph " + std::to_string(graph.id));

    bool changed = true;
    int fusionCount = 0;

    while (changed) {
        changed = false;
        
        // Build dependency map to find single-successor chains
        std::unordered_map<uint64_t, std::vector<size_t>> successors;
        for (size_t i = 0; i < graph.nodes.size(); ++i) {
            for (uint64_t depId : graph.nodes[i].deps) {
                successors[depId].push_back(i);
            }
        }

        for (size_t i = 0; i < graph.nodes.size(); ++i) {
            const auto& nodeA = graph.nodes[i];
            if (nodeA.type != GraphNodeType::KERNEL) continue;

            auto it = successors.find(nodeA.nodeId);
            if (it == successors.end() || it->second.size() != 1) continue;

            size_t nextIdx = it->second[0];
            const auto& nodeB = graph.nodes[nextIdx];

            if (nodeB.type == GraphNodeType::KERNEL && areFusible(nodeA, nodeB)) {
                VGRE_LOG_INFO("GraphOptimizer", "Fusing nodes: " + nodeA.kernelName + " (" + std::to_string(nodeA.nodeId) + 
                             ") -> " + nodeB.kernelName + " (" + std::to_string(nodeB.nodeId) + ")");
                
                if (fuseNodes(graph, i, nextIdx) == VGREResult::SUCCESS) {
                    fusionCount++;
                    changed = true;
                    break; // Restart pass to rebuild dependency map
                }
            }
        }
    }

    if (fusionCount > 0) {
        VGRE_LOG_INFO("GraphOptimizer", "Optimization complete. Fused " + std::to_string(fusionCount) + " kernel pairs.");
        // Re-sort graph to maintain topological order after node replacements
        return sortTopologically(graph);
    }
    return VGREResult::SUCCESS;
}

VGREResult GraphOptimizer::sortTopologically(Graph& graph) {
    if (graph.nodes.empty()) return VGREResult::SUCCESS;

    std::unordered_map<uint64_t, size_t> inDegree;
    std::unordered_map<uint64_t, std::vector<uint64_t>> adj;
    std::unordered_map<uint64_t, const GraphNode*> nodeMap;

    for (const auto& node : graph.nodes) {
        nodeMap[node.nodeId] = &node;
        if (inDegree.find(node.nodeId) == inDegree.end()) {
            inDegree[node.nodeId] = 0;
        }
        for (uint64_t dep : node.deps) {
            adj[dep].push_back(node.nodeId);
            inDegree[node.nodeId]++;
        }
    }

    std::queue<uint64_t> q;
    for (auto const& [id, degree] : inDegree) {
        if (degree == 0) {
            q.push(id);
        }
    }

    std::vector<GraphNode> sortedNodes;
    while (!q.empty()) {
        uint64_t u = q.front();
        q.pop();

        sortedNodes.push_back(*nodeMap[u]);

        for (uint64_t v : adj[u]) {
            if (--inDegree[v] == 0) {
                q.push(v);
            }
        }
    }

    if (sortedNodes.size() != graph.nodes.size()) {
        VGRE_LOG_ERROR("GraphOptimizer", "Topological sort failed: Cycle detected in graph " + std::to_string(graph.id));
        return VGREResult::ERR_INVALID_VALUE;
    }

    graph.nodes = std::move(sortedNodes);
    return VGREResult::SUCCESS;
}


bool GraphOptimizer::areFusible(const GraphNode& a, const GraphNode& b) {
    // 0. Stream-boundary check: only fuse nodes captured on the same stream.
    //    Nodes from different streams execute independently and may run
    //    concurrently; fusing them into one kernel would introduce an implicit
    //    serialisation that the original CUDA program did not intend.
    //    StreamId 0 (default stream) is treated as its own stream.
    if (a.streamId != b.streamId) {
        VGRE_LOG_DEBUG("GraphOptimizer",
            "Fusion blocked: stream-boundary — node " + std::to_string(a.nodeId) +
            " (stream " + std::to_string(a.streamId) + ") vs node " +
            std::to_string(b.nodeId) + " (stream " + std::to_string(b.streamId) + ")");
        return false;
    }

    // 1. Dimensions: identical grid+block OR compatible element count.
    // Identical dims: always fusible (pointwise fusion).
    bool identicalDims = (a.gridDim.x == b.gridDim.x && a.gridDim.y == b.gridDim.y &&
                          a.gridDim.z == b.gridDim.z &&
                          a.blockDim.x == b.blockDim.x && a.blockDim.y == b.blockDim.y &&
                          a.blockDim.z == b.blockDim.z);
    if (!identicalDims) {
        // Compatible element count: total threads (grid × block) must match.
        // Allows fusing kernels with different tile shapes over the same element count.
        uint64_t threadsA = static_cast<uint64_t>(a.gridDim.x) * a.gridDim.y * a.gridDim.z *
                            static_cast<uint64_t>(a.blockDim.x) * a.blockDim.y * a.blockDim.z;
        uint64_t threadsB = static_cast<uint64_t>(b.gridDim.x) * b.gridDim.y * b.gridDim.z *
                            static_cast<uint64_t>(b.blockDim.x) * b.blockDim.y * b.blockDim.z;
        if (threadsA == 0 || threadsA != threadsB) return false;
        // Use A's launch geometry for the fused kernel (fused body iterates all elements)
    }

    VGRE_LOG_INFO("GraphOptimizer", "areFusible check: NodeA=" + std::to_string(a.nodeId) + " (KernelID " + std::to_string(a.kernelId) + "), NodeB=" + std::to_string(b.nodeId) + " (KernelID " + std::to_string(b.kernelId) + ")");
    const auto* irA = RuntimeEngine::instance().getKernelIR(a.kernelId);
    const auto* irB = RuntimeEngine::instance().getKernelIR(b.kernelId);
    if (!irA || !irB) {
        VGRE_LOG_WARN("GraphOptimizer", "areFusible failed: Missing IR for IDs " + std::to_string(a.kernelId) + " or " + std::to_string(b.kernelId));
        return false;
    }

    // 2. Output-output hazard detection: if both kernels write to the same pointer,
    //    fusing them would create a WAW (write-after-write) race condition.
    //    `capturedWritePtrs` holds all pointer-type args (conservative: any pointer
    //    could be a write target).  Block fusion when A and B share a write target.
    // Build index sets for pointer hazard analysis.
    std::unordered_set<void*> writesA(a.capturedWritePtrs.begin(),
                                      a.capturedWritePtrs.end());
    std::unordered_set<void*> writesB(b.capturedWritePtrs.begin(),
                                      b.capturedWritePtrs.end());
    std::unordered_set<void*> readsA(a.capturedReadPtrs.begin(),
                                     a.capturedReadPtrs.end());
    std::unordered_set<void*> readsB(b.capturedReadPtrs.begin(),
                                     b.capturedReadPtrs.end());

    // WAW hazard: both kernels write the same buffer — fusing creates a race.
    for (void* p : writesB) {
        if (p != nullptr && writesA.count(p)) {
            VGRE_LOG_INFO("GraphOptimizer",
                "Fusion blocked: WAW hazard — nodes " +
                std::to_string(a.nodeId) + " and " + std::to_string(b.nodeId) +
                " share write target " + std::to_string(reinterpret_cast<uintptr_t>(p)));
            return false;
        }
    }

    // WAR hazard: A reads a buffer that B writes — fusing risks B overwriting
    // the value before A's fused thread reads it (within the fused kernel body).
    for (void* p : writesB) {
        if (p != nullptr && readsA.count(p)) {
            VGRE_LOG_INFO("GraphOptimizer",
                "Fusion blocked: WAR hazard — node " + std::to_string(b.nodeId) +
                " writes a buffer read by node " + std::to_string(a.nodeId) +
                " at " + std::to_string(reinterpret_cast<uintptr_t>(p)));
            return false;
        }
    }

    // Producer-consumer (RAW): A writes X, B reads X — fusion is BENEFICIAL
    // (the intermediate buffer is eliminated).  Log and allow.
    for (void* p : writesA) {
        if (p != nullptr && readsB.count(p)) {
            VGRE_LOG_DEBUG("GraphOptimizer",
                "RAW producer-consumer link detected: node " + std::to_string(a.nodeId) +
                " → node " + std::to_string(b.nodeId) +
                " via ptr " + std::to_string(reinterpret_cast<uintptr_t>(p)) +
                " — fusion eliminates intermediate buffer");
            break; // at least one RAW edge: fusion is known-beneficial
        }
    }

    // 3. Shared memory check (authoritative hardware limit).
    DeviceProperties props;
    if (RuntimeEngine::instance().getDeviceProperties(0, props) == VGREResult::SUCCESS) {
        if (irA->sharedMemSize + irB->sharedMemSize > props.sharedMemPerBlock) return false;
    } else {
        if (irA->sharedMemSize + irB->sharedMemSize > 48 * 1024) return false;
    }

    // 4. Register pressure estimation.
    // Heuristic: count scalar-typed arguments + local variable patterns in the
    // source. Each float/double ≈ 1-2 registers. Keep headroom below 200
    // (hardware limit 255, ~55 register overhead for the fused wrapper).
    // Over-counting is safe: it may reject fusible pairs but never allows invalid fusions.
    auto estimateRegs = [](const KernelIR* ir) -> int {
        int regs = static_cast<int>(ir->argSizes.size()); // one reg per arg
        const std::string& src = ir->source;
        // Count 'float ' and 'double ' local var declarations (rough register estimate)
        size_t pos = 0;
        while ((pos = src.find("float ", pos)) != std::string::npos) { ++regs; ++pos; }
        pos = 0;
        while ((pos = src.find("double ", pos)) != std::string::npos) { regs += 2; ++pos; }
        pos = 0;
        while ((pos = src.find("int ", pos)) != std::string::npos) { ++regs; ++pos; }
        return regs;
    };
    int combinedRegs = estimateRegs(irA) + estimateRegs(irB);
    if (combinedRegs > 200) {
        VGRE_LOG_DEBUG("GraphOptimizer",
            "Fusion blocked: estimated register pressure " + std::to_string(combinedRegs) +
            " > 200 for nodes " + std::to_string(a.nodeId) + " + " + std::to_string(b.nodeId));
        return false;
    }

    return true;
}

VGREResult GraphOptimizer::fuseNodes(Graph& graph, size_t idxA, size_t idxB) {
    const auto& nodeA = graph.nodes[idxA];
    const auto& nodeB = graph.nodes[idxB];
    
    std::vector<KernelId> ids = {nodeA.kernelId, nodeB.kernelId};
    KernelId fusedId = 0;
    std::string fusedName;
    auto r = RuntimeEngine::instance().fuseKernels(ids, fusedId, &fusedName);
    if (r != VGREResult::SUCCESS) return r;

    // Create a new fused node
    GraphNode fused;
    fused.type = GraphNodeType::KERNEL;
    fused.nodeId = graph.nextNodeId++;
    fused.kernelId = fusedId;
    fused.kernelName = fusedName;
    fused.gridDim = nodeA.gridDim;
    fused.blockDim = nodeA.blockDim;
    
    // Concatenate arguments
    fused.capturedArgs = nodeA.capturedArgs;
    fused.capturedArgs.insert(fused.capturedArgs.end(), nodeB.capturedArgs.begin(), nodeB.capturedArgs.end());
    
    // Dependencies: union of A's deps and B's deps (excluding A)
    fused.deps = nodeA.deps;
    for (auto d : nodeB.deps) {
        if (d != nodeA.nodeId) {
            bool found = false;
            for (auto ad : fused.deps) if (ad == d) { found = true; break; }
            if (!found) fused.deps.push_back(d);
        }
    }

    // Update successors of B to depend on the fused node
    uint64_t targetId = nodeB.nodeId;
    for (auto& n : graph.nodes) {
        for (auto& d : n.deps) {
            if (d == targetId) d = fused.nodeId;
        }
    }

    // Replace A and B with the fused node
    // To maintain topological integrity during the pass, we'll mark them for deletion
    // but for now we'll just insert and the caller will handle the "changed" loop.
    graph.nodes.push_back(std::move(fused));
    
    // Remove old nodes (this is tricky mid-iteration, we'll use a removal list if needed)
    // For simplicity in this v0.1.1 baseline, we'll just remove them
    graph.nodes.erase(graph.nodes.begin() + std::max(idxA, idxB));
    graph.nodes.erase(graph.nodes.begin() + std::min(idxA, idxB));

    return VGREResult::SUCCESS;
}

} // namespace core
} // namespace vgre
