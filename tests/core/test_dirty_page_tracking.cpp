#include "vgre/core/memory_manager.h"
#include "vgre/core/runtime_engine.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <cstring>
#include <thread>

using namespace vgre;
using namespace vgre::core;

void test_dirty_page_tracking() {
    std::cout << "[Test] Starting Dirty Page Tracking Verification..." << std::endl;

    // Use a small pool for testing
    size_t poolSize = 1024 * 1024 * 64; // 64MB
    MemoryManager mgr(poolSize);

    MemoryHandle handle = nullptr;
    // Allocate 1MB of managed memory
    size_t size = 1024 * 1024;
    VGREResult res = mgr.allocateManaged(size, handle, 0, 0); // flags=0 -> PROT_NONE initially
    assert(res == VGREResult::SUCCESS);
    assert(handle != nullptr);

    uint8_t* ptr = static_cast<uint8_t*>(handle);

    // 1. Initial state: no dirty pages
    std::vector<std::pair<size_t, size_t>> dirtyRanges;
    mgr.getDirtyPages(handle, dirtyRanges);
    assert(dirtyRanges.empty());
    std::cout << "[Pass] Initial state is clean." << std::endl;

    // 2. Write to the first page
    std::cout << "Writing to page 0..." << std::endl;
    ptr[0] = 0xAA;
    
    mgr.getDirtyPages(handle, dirtyRanges);
    assert(dirtyRanges.size() == 1);
    assert(dirtyRanges[0].first == 0);
    assert(dirtyRanges[0].second == 4096);
    std::cout << "[Pass] Page 0 correctly marked dirty." << std::endl;

    // 3. Write to a middle page (page 100)
    std::cout << "Writing to page 100..." << std::endl;
    ptr[100 * 4096] = 0xBB;
    
    mgr.getDirtyPages(handle, dirtyRanges);
    // Should have two ranges or one combined if they were adjacent (not in this case)
    assert(dirtyRanges.size() == 2);
    assert(dirtyRanges[1].first == 100 * 4096);
    assert(dirtyRanges[1].second == 101 * 4096);
    std::cout << "[Pass] Page 100 correctly marked dirty." << std::endl;

    // 4. Clear dirty pages
    std::cout << "Clearing dirty pages..." << std::endl;
    mgr.clearDirtyPages(handle);
    mgr.getDirtyPages(handle, dirtyRanges);
    assert(dirtyRanges.empty());
    std::cout << "[Pass] Dirty pages cleared successfully." << std::endl;

    // 5. Write again to verify re-protection works
    std::cout << "Writing to page 200 after clear..." << std::endl;
    ptr[200 * 4096] = 0xCC;
    
    mgr.getDirtyPages(handle, dirtyRanges);
    assert(dirtyRanges.size() == 1);
    assert(dirtyRanges[0].first == 200 * 4096);
    std::cout << "[Pass] Page 200 correctly marked dirty after clear." << std::endl;

    mgr.free(handle);
    std::cout << "[Test] Dirty Page Tracking Verification PASSED!" << std::endl;
}

void test_many_managed_regions() {
    std::cout << "[Test] Allocating and freeing many managed regions..." << std::endl;
    size_t poolSize = 1024 * 1024 * 128; // 128MB
    MemoryManager mgr(poolSize);
    const int N = 64;
    std::vector<MemoryHandle> handles;
    handles.reserve(N);

    // Allocate N small managed regions and touch them to trigger faults
    for (int i = 0; i < N; ++i) {
        MemoryHandle h = nullptr;
        VGREResult r = mgr.allocateManaged(16 * 1024, h, 0, 0);
        if (r != VGREResult::SUCCESS) {
            throw std::runtime_error("allocateManaged failed");
        }
        handles.push_back(h);
        uint8_t* p = static_cast<uint8_t*>(h);
        // Touch first and last byte of the region to generate page faults
        p[0] = static_cast<uint8_t>(i);
        p[16*1024 - 1] = static_cast<uint8_t>(i);
    }

    // Verify dirty pages reported for at least some regions
    int dirtyCount = 0;
    for (auto h : handles) {
        std::vector<std::pair<size_t,size_t>> dr;
        mgr.getDirtyPages(h, dr);
        if (!dr.empty()) ++dirtyCount;
    }
    if (dirtyCount == 0) throw std::runtime_error("No dirty pages detected across allocations");

    // Clear and free
    for (auto h : handles) {
        mgr.clearDirtyPages(h);
        mgr.free(h);
    }

    std::cout << "[Pass] Many region allocate/free cycle succeeded." << std::endl;
}

int main() {
    try {
        test_dirty_page_tracking();
        test_many_managed_regions();
    } catch (const std::exception& e) {
        std::cerr << "[Fail] Test threw exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
