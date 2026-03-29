#include "vgre/compiler/clang_kernel_parser.h"
#include "vgre/compiler/kernel_cache.h"
#include "vgre/common/error_codes.h"
#include <iostream>
#include <chrono>

using namespace vgre;
using namespace vgre::compiler;

int main() {
    // Initialize cache
    KernelCache::instance().initialize();
    
    std::string source = R"(
        __global__ void simpleKernel(float* a, float* b, int n) {
            int i = threadIdx.x;
            a[i] = b[i] * 2.0f;
        }
    )";
    
    ClangKernelParser parser;
    EnhancedKernelIR ir;
    
    std::cout << "=== FIRST PARSE (should be slow) ===" << std::endl;
    auto start1 = std::chrono::high_resolution_clock::now();
    VGREResult r1 = parser.parseEnhanced("simpleKernel", source, ir);
    auto end1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count();
    std::cout << "Result: " << (r1 == VGREResult::SUCCESS ? "SUCCESS" : "FAILED") << std::endl;
    std::cout << "Time: " << duration1 << " ms" << std::endl;
    
    std::cout << "\n=== SECOND PARSE (should be fast with cache) ===" << std::endl;
    auto start2 = std::chrono::high_resolution_clock::now();
    VGREResult r2 = parser.parseEnhanced("simpleKernel", source, ir);
    auto end2 = std::chrono::high_resolution_clock::now();;
    auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2).count();
    std::cout << "Result: " << (r2 == VGREResult::SUCCESS ? "SUCCESS" : "FAILED") << std::endl;
    std::cout << "Time: " << duration2 << " ms" << std::endl;
    
    std::cout << "\n=== PERFORMANCE ===" << std::endl;
    std::cout << "First parse:  " << duration1 << " ms" << std::endl;
    std::cout << "Second parse: " << duration2 << " ms" << std::endl;
    std::cout << "Speedup: " << (duration1 / (double)duration2) << "x" << std::endl;
    
    // Show cache stats
    auto stats = KernelCache::instance().getStats();
    std::cout << "\n=== CACHE STATS ===" << std::endl;
    std::cout << "Hits: " << stats.hits << std::endl;
    std::cout << "Misses: " << stats.misses << std::endl;
    std::cout << "Memory hits: " << stats.memoryHits << std::endl;
    std::cout << "Disk reads: " << stats.diskReads << std::endl;
    std::cout << "Disk writes: " << stats.diskWrites << std::endl;
    
    return 0;
}
