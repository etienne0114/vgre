#include "vgre/core/graph_optimizer.h"
#include "vgre/core/graph_manager.h"
#include "vgre/core/runtime_engine.h"
#include "vgre/common/logger.h"
#include "vgre/common/os_backend.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <map>
#include <vector>

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

    // ── Populate GraphNode cost fields from KernelIR ──────────────────────────
    // KernelIR carries staticFlopCount (from LLVM IR analysis) and estimated
    // instruction count. Use these to populate the DAG weight fields so the
    // roofline cost-benefit check below has real data to work with.
    {
        auto& mA = const_cast<GraphNode&>(a);
        auto& mB = const_cast<GraphNode&>(b);

        // FLOPs: staticFlopCount × threads gives total floating-point operations.
        // Use per-block estimate scaled by total thread count.
        uint64_t threadsA = static_cast<uint64_t>(a.gridDim.x) * a.gridDim.y * a.gridDim.z *
                            static_cast<uint64_t>(a.blockDim.x) * a.blockDim.y * a.blockDim.z;
        uint64_t threadsB = static_cast<uint64_t>(b.gridDim.x) * b.gridDim.y * b.gridDim.z *
                            static_cast<uint64_t>(b.blockDim.x) * b.blockDim.y * b.blockDim.z;

        if (mA.computeFlops == 0.f && irA->staticFlopCount > 0)
            mA.computeFlops = static_cast<float>(irA->staticFlopCount * threadsA);

        if (mB.computeFlops == 0.f && irB->staticFlopCount > 0)
            mB.computeFlops = static_cast<float>(irB->staticFlopCount * threadsB);

        // Memory bandwidth: estimate from argument sizes × thread count.
        // Each pointer arg is read once and written once per thread (conservative).
        auto argBytesEstimate = [](const KernelIR* ir, uint64_t threads) -> float {
            size_t ptrArgBytes = 0;
            for (size_t i = 0; i < ir->argTypes.size(); ++i) {
                if (ir->argTypes[i] == ArgType::POINTER) {
                    size_t sz = (i < ir->argSizes.size()) ? ir->argSizes[i] : 8;
                    // Assume each thread accesses sz bytes from each pointer arg
                    ptrArgBytes += sz;
                }
            }
            // Total bytes = pointer args × element size × threads
            return static_cast<float>(ptrArgBytes * threads) / 1e9f; // GB
        };

        if (mA.memoryBandwidthGB == 0.f)
            mA.memoryBandwidthGB = argBytesEstimate(irA, threadsA);
        if (mB.memoryBandwidthGB == 0.f)
            mB.memoryBandwidthGB = argBytesEstimate(irB, threadsB);

        // Intermediate bytes: the output of A consumed by B (the RAW link).
        // Estimate as A's write-pointer argument size × thread count.
        if (mA.intermediateBytes == 0.f && !a.capturedWritePtrs.empty()) {
            // Each write pointer carries one output element per thread.
            // Use 4 bytes (float) as the default element size.
            size_t writeArgBytes = 4; // default float element
            for (size_t i = 0; i < irA->argTypes.size(); ++i) {
                if (irA->argTypes[i] == ArgType::POINTER && i < irA->argSizes.size()) {
                    writeArgBytes = irA->argSizes[i];
                    break;
                }
            }
            mA.intermediateBytes = static_cast<float>(writeArgBytes * threadsA) / 1e9f; // GB
        }

        // Arithmetic intensity for roofline classification
        if (mA.arithmeticIntensity == 0.f && mA.memoryBandwidthGB > 0.f)
            mA.arithmeticIntensity = mA.computeFlops / (mA.memoryBandwidthGB * 1e9f);
        if (mB.arithmeticIntensity == 0.f && mB.memoryBandwidthGB > 0.f)
            mB.arithmeticIntensity = mB.computeFlops / (mB.memoryBandwidthGB * 1e9f);
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

    // 4. Register pressure analysis — improved model.
    // Counts typed arguments + arithmetic temporaries estimated from operator density.
    // Each pointer arg = 1 reg; each scalar (float/double/int) = 1-2 regs.
    // Arithmetic temporaries: each binary operator (+,-,*,/) in a function body
    // requires one temporary register. This over-estimates for expression reuse
    // (which is safe — it may block some valid fusions but never permits unsafe ones).
    auto estimateRegs = [](const KernelIR* ir) -> int {
        int regs = 0;
        // Pointer args: each takes 1 register for the address
        for (size_t i = 0; i < ir->argTypes.size(); ++i)
            regs += (ir->argTypes[i] == ArgType::POINTER) ? 1 : 1;
        // Scan source for typed local declarations
        const std::string& src = ir->source;
        // Count each distinct typed declaration inside the function body
        // Use a simple state machine: after '{', count 'type name;' patterns
        auto countDecls = [&src](const char* typeName, int weight) {
            int n = 0; size_t pos = 0;
            const std::string t(typeName);
            while ((pos = src.find(t, pos)) != std::string::npos) {
                // Must be followed by space (not part of another identifier)
                if (pos + t.size() < src.size() && src[pos + t.size()] == ' ') ++n;
                ++pos;
            }
            return n * weight;
        };
        regs += countDecls("float ",  1);
        regs += countDecls("double ", 2); // doubles take 2 regs in VGRE's FP32/64 model
        regs += countDecls("int ",    1);
        regs += countDecls("uint32_t",1);
        // Add operator-density estimate: each +-*/ in the source body requires ~1 temp
        for (char op : {'+','-','*','/'}) {
            size_t p = 0;
            while ((p = src.find(op, p)) != std::string::npos) { ++regs; ++p; }
        }
        regs /= 3; // dampen operator over-count (operators share temporaries)
        return regs;
    };

    const int combinedRegs = estimateRegs(irA) + estimateRegs(irB);
    if (combinedRegs > 200) {
        VGRE_LOG_DEBUG("GraphOptimizer",
            "Fusion blocked: register pressure " + std::to_string(combinedRegs) +
            " > 200 for nodes " + std::to_string(a.nodeId) + "/" + std::to_string(b.nodeId));
        return false;
    }

    // 5. Cost-Benefit Kernel Fusion — roofline-based ROI.
    // Compute the estimated speedup from fusion by comparing:
    //   time_separate = T_A + T_intermediate_write + T_B
    //   time_fused    = T_fused (no intermediate write/read)
    //
    // Uses the Roofline model:
    //   T = max(FLOPs / peak_compute_GFLOPs_s,  memory_bytes / bandwidth_GB_s)
    //
    // Intermediate bytes eliminated = a.intermediateBytes (set by caller).
    // Only proceed if fusion_speedup > 1.05 (5% minimum benefit threshold).
    {
        // Compute peak GFLOP/s: cores × FMAs/cycle × FP32/FMA-width × GHz.
        // AVX2 FMA: 2 × 256-bit (8 FP32) per cycle = 16 FP32 FMAs/core/cycle.
        // Conservative 2.5 GHz base: 16 FP32/cycle × 2.5e9 = 40 GFLOPs/core/s.
        // Non-AVX2 fallback: 2 FP32/cycle × 2.5 GHz = 5 GFLOPs/core/s.
        static const float kPeakComputeGFLOPs = []() -> float {
            int cores = vgre::os::get_cpu_count();
            if (cores <= 0) cores = 1;
#if defined(__AVX512F__)
            return static_cast<float>(cores) * 64.0f;   // 2 × 512-bit = 32 FP32; ×2 FMAs
#elif defined(__AVX2__)
            return static_cast<float>(cores) * 40.0f;   // 16 FP32/cycle × ~2.5 GHz
#elif defined(__AVX__)
            return static_cast<float>(cores) * 20.0f;   // 8 FP32/cycle × ~2.5 GHz
#else
            return static_cast<float>(cores) * 5.0f;    // scalar/NEON conservative
#endif
        }();

        // Measure system memory bandwidth via a STREAM-style copy benchmark.
        static const float kBandwidthGBps = []() -> float {
            constexpr size_t kN = 8 * 1024 * 1024;  // 8M floats = 32 MB per array
            constexpr int    kR = 3;
            try {
                std::vector<float> src(kN, 1.0f), dst(kN, 0.0f);
                std::memcpy(dst.data(), src.data(), kN * sizeof(float));  // warmup
                auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < kR; ++i)
                    std::memcpy(dst.data(), src.data(), kN * sizeof(float));
                auto t1 = std::chrono::steady_clock::now();
                double ns = static_cast<double>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
                double gbps = (ns > 0.0) ? (2.0 * kN * sizeof(float) * kR / (ns * 1e-9) / 1e9) : 0.0;
                if (gbps > 0.5 && gbps < 5000.0) return static_cast<float>(gbps);
            } catch (...) {}
            return 40.0f;  // fallback: typical dual-channel DDR4
        }();

        auto rooflineMs = [&](float flops, float bytes) -> float {
            float t_compute  = flops  / (kPeakComputeGFLOPs * 1e9f) * 1e3f; // ms
            float t_memory   = bytes  / (kBandwidthGBps     * 1e9f) * 1e3f; // ms
            return std::max(t_compute, t_memory);
        };

        float bytesA    = a.memoryBandwidthGB * 1e9f;
        float bytesB    = b.memoryBandwidthGB * 1e9f;
        float intermed  = a.intermediateBytes;

        if (a.computeFlops > 0.f || bytesA > 0.f) {
            float t_sep = rooflineMs(a.computeFlops, bytesA)
                        + rooflineMs(0.f, intermed)      // write intermediate
                        + rooflineMs(b.computeFlops, bytesB);
            float t_fused = rooflineMs(a.computeFlops + b.computeFlops,
                                       bytesA + bytesB - intermed);
            if (t_sep > 0.f) {
                float speedup = t_sep / std::max(1e-6f, t_fused);
                const_cast<GraphNode&>(b).fusionBenefit = speedup;
                if (speedup < 1.05f) {
                    VGRE_LOG_DEBUG("GraphOptimizer",
                        "Fusion ROI too low: speedup=" + std::to_string(speedup) +
                        " for nodes " + std::to_string(a.nodeId) + "/" +
                        std::to_string(b.nodeId) + " (need >1.05)");
                    return false;
                }
            }
        }
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
