#include "vgre/advanced/memory_compression.h"
#include <iostream>
#include <vector>
#include <string>
#include <cassert>

using namespace vgre::advanced;

int main() {
    MemoryCompression& mc = MemoryCompression::instance();
    
    std::string testData = "VGRE VGRE VGRE VGRE VGRE VGRE VGRE VGRE VGRE VGRE VGRE VGRE VGRE VGRE VGRE VGRE";
    // Make it larger to trigger compression
    while (testData.size() < 10000) {
        testData += testData;
    }
    
    std::vector<uint8_t> compressed;
    vgre::VGREResult res = mc.compress(testData.data(), testData.size(), compressed);
    
    if (res != vgre::VGREResult::SUCCESS) {
        std::cerr << "Compression failed!" << std::endl;
        return 1;
    }
    
    std::cout << "Original size: " << testData.size() << " bytes" << std::endl;
    std::cout << "Compressed size: " << compressed.size() << " bytes" << std::endl;
    
    std::vector<char> decompressed(testData.size());
    size_t actualSize = 0;
    res = mc.decompress(compressed.data(), compressed.size(), decompressed.data(), decompressed.size(), actualSize);
    
    if (res != vgre::VGREResult::SUCCESS) {
        std::cerr << "Decompression failed!" << std::endl;
        return 1;
    }
    
    assert(actualSize == testData.size());
    assert(std::string(decompressed.data(), actualSize) == testData);
    
    std::cout << "Verification SUCCESSFUL!" << std::endl;
    
    CompressionStats stats = mc.getStats();
    std::cout << "Compression Ratio: " << stats.avgCompressionRatio << std::endl;
    
    return 0;
}
