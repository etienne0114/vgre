#include "vgre/core/runtime_engine.h"
#include <iostream>
#include <cassert>

using namespace vgre;
using namespace vgre::core;

void test_graph_cycle() {
    auto& engine = RuntimeEngine::instance();
    engine.initialize();
    
    GraphId gid;
    engine.graphCreate(gid);
    
    // Create nodes
    uint64_t n1, n2;
    std::vector<ArgType> types;
    engine.graphAddKernelNode(gid, 0, "k1", {1,1,1}, {1,1,1}, nullptr, types, {}, n1);
    engine.graphAddKernelNode(gid, 0, "k2", {1,1,1}, {1,1,1}, nullptr, types, {}, n2);
    
    // Create a cycle: n1 -> n2 -> n1
    // n2 depends on n1
    engine.graphAddDependency(gid, n2, n1);
    
    std::cout << "--- Test: Graph Cycle Detection ---\n";
    
    GraphExecId execId;
    // This should pass (no cycle yet)
    VGREResult res = engine.graphInstantiate(gid, execId);
    assert(res == VGREResult::SUCCESS);
    std::cout << "  ✓ DAG instantiation passed (linear chain).\n";

    // Now introduce the cycle: n1 depends on n2
    engine.graphAddDependency(gid, n1, n2);
    
    // This should FAIL
    res = engine.graphInstantiate(gid, execId);
    if (res != VGREResult::SUCCESS) {
        std::cout << "  ✓ Cycle detected and instantiation rejected as expected.\n";
    } else {
        std::cout << "  ✗ FAILURE: Cycle was NOT detected!\n";
        exit(1);
    }

    std::cout << "[PASS] Graph Cycle Detection\n";
    engine.shutdown();
}

int main() {
    test_graph_cycle();
    return 0;
}
