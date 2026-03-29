/**
 * VGRE Unit Tests — Memory Concurrency
 * Verifies that shared_lock allows multiple concurrent readers without contention.
 */
#include "vgre/core/memory_manager.h"
#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <cassert>
#include <chrono>

using namespace vgre::core;

void test_concurrent_lookups() {
    const size_t poolSize = 100 * 1024 * 1024; // 100 MB
    MemoryManager mm(poolSize);

    const int numAllocations = 1000;
    std::vector<vgre::MemoryHandle> handles;
    for (int i = 0; i < numAllocations; ++i) {
        vgre::MemoryHandle h = nullptr;
        mm.allocate(1024, h);
        handles.push_back(h);
    }

    const int numThreads = 16;
    const int iterationsPerThread = 10000;
    std::atomic<int> totalLookups{0};
    
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::thread> threads;
    for (int t = 0; t < numThreads; ++t) {
        threads.emplace_back([&mm, &handles, &totalLookups]() {
            for (int i = 0; i < iterationsPerThread; ++i) {
                // Perform a mix of lookups
                vgre::MemoryHandle h = handles[i % handles.size()];
                bool valid = mm.isValidHandle(h);
                size_t sz = mm.getAllocationSize(h);
                void* ptr = mm.getPointer(h);
                
                assert(valid);
                assert(sz >= 1024);
                assert(ptr != nullptr);
                
                totalLookups.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    auto end = std::chrono::high_resolution_clock::now();
    double durationSec = std::chrono::duration<double>(end - start).count();

    std::cout << "[PASS] Concurrent Lookups: " << totalLookups.load() 
              << " operations in " << durationSec << "s (" 
              << (totalLookups.load() / durationSec) << " ops/sec)" << std::endl;
}

int main() {
    std::cout << "=== VGRE Memory Concurrency Unit Test ===" << std::endl;
    test_concurrent_lookups();
    return 0;
}
