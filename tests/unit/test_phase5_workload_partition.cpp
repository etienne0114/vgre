#include "vgre/advanced/workload_partitioner.h"
#include <iostream>
#include <vector>
#include <cassert>

using namespace vgre::advanced;

void test_basic_partitioning() {
    std::cout << "Testing Basic Workload Partitioning..." << std::endl;
    WorkloadPartitioner partitioner;
    
    std::vector<NodeCapability> nodes = {
        {100.0, 1.0, 4, -1, "10.0.0.1", false},
        {100.0, 1.0, 8, -1, "10.0.0.2", false},
        {100.0, 1.0, 4, -1, "local", true}
    };
    
    uint32_t gridDim[3] = {16, 1, 1};
    uint32_t blockDim[3] = {128, 1, 1};
    PartitionPlan plan;
    vgre::VGREResult r = partitioner.createPartitionPlan(gridDim, blockDim, nodes, plan);
    
    assert(r == vgre::VGREResult::SUCCESS);
    assert(plan.slices.size() == 3);
    
    // Total cores: 4 + 8 + 4 = 16
    // 16 blocks, 16 cores -> approx 1 block per core
    // 10.0.0.1 (4 cores) -> 4 blocks
    // 10.0.0.2 (8 cores) -> 8 blocks
    // local (4 cores)    -> 4 blocks
    
    // Node 1 (8 cores) has highest capacity, so it should be first in the plan
    assert(plan.slices[0].partition_grid_x == 8);
    assert(plan.slices[1].partition_grid_x == 4);
    assert(plan.slices[2].partition_grid_x == 4);
    
    assert(plan.slices[0].grid_x_start == 0);
    assert(plan.slices[1].grid_x_start == 8);
    assert(plan.slices[2].grid_x_start == 12);
    
    std::cout << "  Passed (3 nodes, 16 blocks)." << std::endl;
}

void test_proportional_imbalance() {
    std::cout << "Testing Proportional Imbalance..." << std::endl;
    WorkloadPartitioner partitioner;
    
    std::vector<NodeCapability> nodes = {
        {100.0, 1.0, 32, -1, "powerful", false},
        {100.0, 1.0, 4, -1, "weak", false}
    };
    
    uint32_t gridDim[3] = {100, 1, 1};
    uint32_t blockDim[3] = {128, 1, 1};
    PartitionPlan plan;
    partitioner.createPartitionPlan(gridDim, blockDim, nodes, plan);
    
    // Total 36 cores. 100 blocks.
    // Ratio: 32/36 (88.8%) and 4/36 (11.1%)
    // Powerful: 88.8 blocks -> ~89 blocks
    // Weak: 11.1 blocks -> ~11 blocks
    
    assert(plan.slices[0].partition_grid_x + plan.slices[1].partition_grid_x == 100);
    assert(plan.slices[0].partition_grid_x > 80);
    assert(plan.slices[1].partition_grid_x < 20);
    
    std::cout << "  Passed (Powerful vs Weak node)." << std::endl;
}

int main() {
    try {
        test_basic_partitioning();
        test_proportional_imbalance();
        std::cout << "\nALL WorkloadPartitioner tests PASSED." << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}
