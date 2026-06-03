// QUEUE-27: cudaGraph full replay — topological sort + dependency-ordered
// kernel execution. Five tests covering the remaining gaps in topo-sort
// coverage identified in the audit.
//
// All tests use RuntimeEngine::graphAddHostNode to observe execution order
// without a GPU; the HOST node invokes a plain function pointer inline, in
// the order determined by Kahn's algorithm.

#include "vgre/core/runtime_engine.h"
#include "vgre/api/vgre_c_api.h"

#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>
#include <algorithm>

using namespace vgre;
using namespace vgre::core;

// ---------------------------------------------------------------------------
// Helper: append the integer tag stored in userData to a shared order vector.
// ---------------------------------------------------------------------------
struct HostCtx {
    std::vector<int> *order;
    int tag;
};

static void record_tag(void *ud) {
    auto *ctx = static_cast<HostCtx *>(ud);
    ctx->order->push_back(ctx->tag);
}

// ---------------------------------------------------------------------------
// Test 1: Diamond dependency  A→B, A→C, B→D, C→D
//
// Kahn invariant: in_degree[D] starts at 2; it is decremented to 1 when the
// first of {B,C} completes and to 0 when the other completes — only then is D
// enqueued and executed.  A must precede B and C; D must follow both.
// ---------------------------------------------------------------------------
static void test_graph_diamond_dependency() {
    std::cout << "--- Test: diamond dependency (A→B, A→C, B→D, C→D) ---\n";

    auto &engine = RuntimeEngine::instance();
    engine.initialize();

    std::vector<int> order;
    // Tags: A=0, B=1, C=2, D=3
    HostCtx ctxA{&order, 0};
    HostCtx ctxB{&order, 1};
    HostCtx ctxC{&order, 2};
    HostCtx ctxD{&order, 3};

    GraphId gid;
    engine.graphCreate(gid);

    // Kahn: in_degree[D] = 2; in_degree[B] = in_degree[C] = 1; in_degree[A] = 0
    uint64_t nA, nB, nC, nD;
    engine.graphAddHostNode(gid, record_tag, &ctxA, {}, nA);
    engine.graphAddHostNode(gid, record_tag, &ctxB, {nA}, nB);
    engine.graphAddHostNode(gid, record_tag, &ctxC, {nA}, nC);
    engine.graphAddHostNode(gid, record_tag, &ctxD, {nB, nC}, nD);

    GraphExecId exec;
    VGREResult res = engine.graphInstantiate(gid, exec);
    assert(res == VGREResult::SUCCESS && "diamond: instantiate must succeed");

    engine.graphLaunch(exec, 0);
    engine.streamSynchronize(0);

    // Verify ordering constraints:
    // Kahn: A before B and C; B and C both before D; all 4 nodes executed.
    assert(order.size() == 4 && "all 4 nodes must execute");

    auto pos = [&](int tag) -> int {
        for (int i = 0; i < (int)order.size(); ++i)
            if (order[i] == tag) return i;
        return -1;
    };
    // A must execute before B and C
    assert(pos(0) < pos(1) && "A must precede B");
    assert(pos(0) < pos(2) && "A must precede C");
    // B and C must both execute before D
    // Kahn: in_degree[D] goes 2→1→0 after B and C complete
    assert(pos(1) < pos(3) && "B must precede D");
    assert(pos(2) < pos(3) && "C must precede D");

    std::cout << "  Execution order:";
    for (int t : order) std::cout << " " << t;
    std::cout << "\n  [PASS] diamond dependency\n";

    engine.shutdown();
}

// ---------------------------------------------------------------------------
// Test 2: Multi-root — A, B, C independent; D depends on all three
//
// Kahn invariant: 3 nodes have in_degree=0 at start; all 3 are enqueued
// simultaneously into the ready queue before D (in_degree=3→2→1→0) is ever
// reached.  D must execute after all three roots have completed.
// ---------------------------------------------------------------------------
static void test_graph_multi_root() {
    std::cout << "--- Test: multi-root (A, B, C → D) ---\n";

    auto &engine = RuntimeEngine::instance();
    engine.initialize();

    std::vector<int> order;
    // Tags: A=10, B=11, C=12, D=13
    HostCtx ctxA{&order, 10};
    HostCtx ctxB{&order, 11};
    HostCtx ctxC{&order, 12};
    HostCtx ctxD{&order, 13};

    GraphId gid;
    engine.graphCreate(gid);

    // Kahn: in_degree[A]=in_degree[B]=in_degree[C]=0; in_degree[D]=3
    uint64_t nA, nB, nC, nD;
    engine.graphAddHostNode(gid, record_tag, &ctxA, {}, nA);
    engine.graphAddHostNode(gid, record_tag, &ctxB, {}, nB);
    engine.graphAddHostNode(gid, record_tag, &ctxC, {}, nC);
    engine.graphAddHostNode(gid, record_tag, &ctxD, {nA, nB, nC}, nD);

    GraphExecId exec;
    VGREResult res = engine.graphInstantiate(gid, exec);
    assert(res == VGREResult::SUCCESS && "multi-root: instantiate must succeed");

    engine.graphLaunch(exec, 0);
    engine.streamSynchronize(0);

    assert(order.size() == 4 && "all 4 nodes must execute");

    auto pos = [&](int tag) -> int {
        for (int i = 0; i < (int)order.size(); ++i)
            if (order[i] == tag) return i;
        return -1;
    };
    // Kahn: A, B, C all have in_degree 0 — each fires before D
    assert(pos(10) < pos(13) && "A must precede D");
    assert(pos(11) < pos(13) && "B must precede D");
    assert(pos(12) < pos(13) && "C must precede D");
    // D must be last
    assert(pos(13) == 3 && "D must be the final node");

    std::cout << "  Execution order:";
    for (int t : order) std::cout << " " << t;
    std::cout << "\n  [PASS] multi-root\n";

    engine.shutdown();
}

// ---------------------------------------------------------------------------
// Test 3: Re-launch — same exec launched twice, no residual state
//
// Re-launch invariant: graphLaunch rebuilds in-degree from scratch on every
// call; no mutable residual in_degree counters survive between launches.
// The second launch must produce an identical execution order.
// ---------------------------------------------------------------------------
static void test_graph_relaunch_identical() {
    std::cout << "--- Test: re-launch (same order on second launch) ---\n";

    auto &engine = RuntimeEngine::instance();
    engine.initialize();

    std::vector<int> run1, run2;
    // Tags: 20, 21, 22 for A→B→C chain
    HostCtx ctx1A{&run1, 20};
    HostCtx ctx1B{&run1, 21};
    HostCtx ctx1C{&run1, 22};
    HostCtx ctx2A{&run2, 20};
    HostCtx ctx2B{&run2, 21};
    HostCtx ctx2C{&run2, 22};

    GraphId gid;
    engine.graphCreate(gid);

    uint64_t nA, nB, nC;
    engine.graphAddHostNode(gid, record_tag, &ctx1A, {}, nA);
    engine.graphAddHostNode(gid, record_tag, &ctx1B, {nA}, nB);
    engine.graphAddHostNode(gid, record_tag, &ctx1C, {nB}, nC);

    GraphExecId exec;
    VGREResult res = engine.graphInstantiate(gid, exec);
    assert(res == VGREResult::SUCCESS && "relaunch: instantiate must succeed");

    // First launch — capture into run1 via the ctx1* contexts
    engine.graphLaunch(exec, 0);
    engine.streamSynchronize(0);
    assert(run1.size() == 3 && "first launch: all 3 nodes must execute");

    // Re-launch invariant: fresh in_degree rebuild; no residual state.
    // We cannot change the userData after instantiation, so we verify that
    // the same exec can be re-launched without error and that it processes
    // exactly 3 nodes again (the callbacks will append to run1 a second time).
    std::vector<int> run1_copy = run1;  // save first-run order
    engine.graphLaunch(exec, 0);
    engine.streamSynchronize(0);
    // After second launch run1 has 6 entries: first-run followed by second-run
    assert(run1.size() == 6 && "second launch: 3 more executions");

    // First-run order == second-run order (indices 0-2 == 3-5)
    for (int i = 0; i < 3; ++i) {
        assert(run1[i] == run1[i + 3] &&
               "Re-launch invariant: order must be identical on second launch");
    }

    std::cout << "  First launch order:";
    for (int i = 0; i < 3; ++i) std::cout << " " << run1[i];
    std::cout << "\n  Second launch order:";
    for (int i = 3; i < 6; ++i) std::cout << " " << run1[i];
    std::cout << "\n  [PASS] re-launch identical\n";

    engine.shutdown();
}

// ---------------------------------------------------------------------------
// Test 4: Memcpy node in chain — [H→D memcpy] → [host node that reads the
//         staged value] → [D→H memcpy back] — verifying dependency ordering.
//
// Dependency invariant: the memcpy node must complete before any dependent
// node executes.  We use a HOST node in the middle that reads the host-side
// staging buffer (written by the H→D → D→H round-trip) to confirm the
// memcpy nodes executed in the right order.
//
// Because the virtual device's "device memory" is host-allocated memory, a
// HOST node can read/write it directly.  We write 42 from host, copy H→D
// (which is a no-op in virtual memory — the pointer IS the device pointer),
// then a HOST node reads from the device buffer and verifies the value, then
// copies D→H back.  The final host read confirms the full chain.
// ---------------------------------------------------------------------------
static void test_graph_memcpy_in_chain() {
    std::cout << "--- Test: memcpy node in chain (H→D → host-verify → D→H) ---\n";

    auto &engine = RuntimeEngine::instance();
    engine.initialize();

    // Allocate "device" buffer (MemoryHandle == void* in VGRE virtual memory)
    MemoryHandle devBuf = nullptr;
    VGREResult alloc_res = engine.malloc(sizeof(int), devBuf);
    assert(alloc_res == VGREResult::SUCCESS && "malloc must succeed");

    // Host source and destination buffers
    int h_src = 42;
    int h_dst = 0;

    // A HOST node that verifies the device buffer holds the expected value
    // and sets a flag we can check after the launch.
    struct VerifyCtx {
        void *devPtr;   // device buffer pointer
        int   expected;
        bool  ok;
    };
    VerifyCtx vCtx{devBuf, 42, false};

    auto verify_fn = [](void *ud) {
        auto *vc = static_cast<VerifyCtx *>(ud);
        int val = *static_cast<int *>(vc->devPtr);
        vc->ok = (val == vc->expected);
    };

    GraphId gid;
    engine.graphCreate(gid);

    // Dependency invariant: memcpy (H→D) must complete before the host node
    // reads the device buffer; the host node must complete before D→H memcpy.
    uint64_t nCopy1, nVerify, nCopy2;
    // Node 1: copy h_src → devBuf  (kind=0 = VGRE_MEMCPY_HOST_TO_DEVICE)
    engine.graphAddMemcpyNode(gid, devBuf, &h_src, sizeof(int),
                              VGRE_MEMCPY_HOST_TO_DEVICE, {}, nCopy1);
    // Node 2: HOST node — reads devBuf, checks value
    engine.graphAddHostNode(gid, verify_fn, &vCtx, {nCopy1}, nVerify);
    // Node 3: copy devBuf → h_dst  (kind=1 = VGRE_MEMCPY_DEVICE_TO_HOST)
    engine.graphAddMemcpyNode(gid, &h_dst, devBuf, sizeof(int),
                              VGRE_MEMCPY_DEVICE_TO_HOST, {nVerify}, nCopy2);

    GraphExecId exec;
    VGREResult res = engine.graphInstantiate(gid, exec);
    assert(res == VGREResult::SUCCESS && "memcpy chain: instantiate must succeed");

    engine.graphLaunch(exec, 0);
    engine.streamSynchronize(0);

    // Dependency invariant: verify node ran after H→D memcpy
    assert(vCtx.ok && "HOST verify node must see value written by H→D memcpy");
    // Dependency invariant: D→H memcpy ran after verify node
    assert(h_dst == 42 && "final host buffer must match kernel output");

    std::cout << "  h_dst after D→H memcpy = " << h_dst << " (expected 42)\n";
    std::cout << "  [PASS] memcpy node in chain\n";

    engine.shutdown();
}

// ---------------------------------------------------------------------------
// Test 5: Disconnected components — two independent chains A→B and C→D
//
// Kahn invariant: disconnected DAG still fully sorts — all zero-in-degree
// nodes (A and C) are enqueued simultaneously at start; B and C/D are
// processed as their in_degree drops to zero.  cudaGraphInstantiate must
// succeed and all 4 nodes must execute.
// ---------------------------------------------------------------------------
static void test_graph_disconnected_components() {
    std::cout << "--- Test: disconnected components (A→B and C→D, no shared edges) ---\n";

    auto &engine = RuntimeEngine::instance();
    engine.initialize();

    std::vector<int> order;
    // Tags: A=30, B=31, C=32, D=33
    HostCtx ctxA{&order, 30};
    HostCtx ctxB{&order, 31};
    HostCtx ctxC{&order, 32};
    HostCtx ctxD{&order, 33};

    GraphId gid;
    engine.graphCreate(gid);

    // Kahn: in_degree[A]=0, in_degree[B]=1 (depends on A)
    //       in_degree[C]=0, in_degree[D]=1 (depends on C)
    // Two disconnected components — topoSort must still enumerate all 4 nodes.
    uint64_t nA, nB, nC, nD;
    engine.graphAddHostNode(gid, record_tag, &ctxA, {}, nA);
    engine.graphAddHostNode(gid, record_tag, &ctxB, {nA}, nB);
    engine.graphAddHostNode(gid, record_tag, &ctxC, {}, nC);
    engine.graphAddHostNode(gid, record_tag, &ctxD, {nC}, nD);

    GraphExecId exec;
    VGREResult res = engine.graphInstantiate(gid, exec);
    assert(res == VGREResult::SUCCESS &&
           "disconnected: instantiate must succeed (no cycle, valid DAG)");

    engine.graphLaunch(exec, 0);
    engine.streamSynchronize(0);

    // Kahn: all 4 nodes must appear in the topological order
    assert(order.size() == 4 && "all 4 nodes must execute");

    auto pos = [&](int tag) -> int {
        for (int i = 0; i < (int)order.size(); ++i)
            if (order[i] == tag) return i;
        return -1;
    };
    // Within each component the chain ordering must hold
    assert(pos(30) < pos(31) && "A must precede B in chain 1");
    assert(pos(32) < pos(33) && "C must precede D in chain 2");
    // Both components present
    assert(pos(30) >= 0 && pos(31) >= 0 && pos(32) >= 0 && pos(33) >= 0 &&
           "all 4 nodes must be present in output");

    std::cout << "  Execution order:";
    for (int t : order) std::cout << " " << t;
    std::cout << "\n  [PASS] disconnected components\n";

    engine.shutdown();
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    std::cout << "=====================================================\n";
    std::cout << "  VGRE Integration Test — Graph Topo Advanced       \n";
    std::cout << "  QUEUE-27: diamond, multi-root, relaunch, memcpy,  \n";
    std::cout << "            disconnected                             \n";
    std::cout << "=====================================================\n";

    test_graph_diamond_dependency();
    test_graph_multi_root();
    test_graph_relaunch_identical();
    test_graph_memcpy_in_chain();
    test_graph_disconnected_components();

    std::cout << "\n[ALL PASS] test_graph_topo_advanced\n";
    return 0;
}
