#include "vgre/core/texture_manager.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace vgre;
using namespace vgre::core;

void test_format_reinterpretation() {
    // Create a UINT32 texture
    std::vector<uint32_t> data = {0x01020304, 0x05060708};
    TextureId baseId;
    TextureDescriptor baseDesc;
    baseDesc.elementType = TextureElementType::UINT32;
    baseDesc.filterMode = TextureFilterMode::POINT;
    
    auto r = TextureManager::instance().createTexture(baseId, data.data(), 2, 1, 4, baseDesc);
    assert(r == VGREResult::SUCCESS);
    
    // Create a UINT8 view of the same data
    TextureId viewId;
    ResourceViewDescriptor viewDesc;
    viewDesc.format = TextureElementType::UINT8;
    viewDesc.width = 8;
    viewDesc.height = 1;
    viewDesc.depth = 1;
    viewDesc.offsetInBytes = 0;
    
    TextureDescriptor texDesc = baseDesc;
    r = TextureManager::instance().createTextureView(viewId, baseId, viewDesc, texDesc);
    assert(r == VGREResult::SUCCESS);
    
    // Verify sampling (UINT8 view)
    // Little-endian: 0x04, 0x03, 0x02, 0x01, 0x08, 0x07, 0x06, 0x05
    assert(std::abs(TextureManager::instance().tex1Dfetch(viewId, 0) - 4.0f) < 1e-5);
    assert(std::abs(TextureManager::instance().tex1Dfetch(viewId, 1) - 3.0f) < 1e-5);
    assert(std::abs(TextureManager::instance().tex1Dfetch(viewId, 2) - 2.0f) < 1e-5);
    assert(std::abs(TextureManager::instance().tex1Dfetch(viewId, 3) - 1.0f) < 1e-5);
    
    std::cout << "[PASS] Format reinterpretation: UINT32 as UINT8" << std::endl;
}

void test_sub_resource_offset() {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    TextureId baseId;
    TextureDescriptor baseDesc;
    baseDesc.elementType = TextureElementType::FLOAT32;
    baseDesc.filterMode = TextureFilterMode::POINT;
    
    auto r = TextureManager::instance().createTexture(baseId, data.data(), 4, 1, 4, baseDesc);
    assert(r == VGREResult::SUCCESS);
    
    // Create a view starting at the second element (offset = 4 bytes)
    TextureId viewId;
    ResourceViewDescriptor viewDesc;
    viewDesc.format = TextureElementType::FLOAT32;
    viewDesc.width = 2;
    viewDesc.height = 1;
    viewDesc.depth = 1;
    viewDesc.offsetInBytes = 4;
    
    TextureDescriptor texDesc = baseDesc;
    r = TextureManager::instance().createTextureView(viewId, baseId, viewDesc, texDesc);
    assert(r == VGREResult::SUCCESS);
    
    // Verify sampling (view starts at 2.0f)
    assert(std::abs(TextureManager::instance().tex1Dfetch(viewId, 0) - 2.0f) < 1e-5);
    assert(std::abs(TextureManager::instance().tex1Dfetch(viewId, 1) - 3.0f) < 1e-5);
    
    std::cout << "[PASS] Sub-resource offset mapping" << std::endl;
}

int main() {
    std::cout << "=== VGRE Texture View Unit Tests ===" << std::endl;
    test_format_reinterpretation();
    test_sub_resource_offset();
    std::cout << "All texture view tests passed!" << std::endl;
    return 0;
}
