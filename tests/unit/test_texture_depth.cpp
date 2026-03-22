#include "vgre/core/texture_manager.h"
#include <cassert>
#include <iostream>
#include <vector>

using namespace vgre;
using namespace vgre::core;

void test_int16_sampling() {
    TextureManager& mgr = TextureManager::instance();
    std::vector<int16_t> data = {100, 200, 300, 400};
    
    TextureDescriptor desc;
    desc.elementType = TextureElementType::INT16;
    desc.filterMode = TextureFilterMode::POINT;
    
    TextureId tex;
    assert(mgr.createTexture(tex, data.data(), 2, 2, sizeof(int16_t), desc) == VGREResult::SUCCESS);
    
    // Test point sampling
    assert(mgr.tex2D(tex, 0.0f, 0.0f) == 100.0f);
    assert(mgr.tex2D(tex, 1.0f, 0.0f) == 200.0f);
    assert(mgr.tex2D(tex, 0.0f, 1.0f) == 300.0f);
    assert(mgr.tex2D(tex, 1.0f, 1.0f) == 400.0f);
    
    mgr.destroyTexture(tex);
    std::cout << "[PASS] Int16 Point Sampling" << std::endl;
}

void test_uint8_surface() {
    TextureManager& mgr = TextureManager::instance();
    std::vector<uint8_t> data(16, 0);
    
    SurfaceId surf;
    assert(mgr.createSurface(surf, data.data(), 4, 4, sizeof(uint8_t), TextureElementType::UINT8) == VGREResult::SUCCESS);
    
    mgr.surf2Dwrite(surf, 255.0f, 0, 0);
    mgr.surf2Dwrite(surf, 128.0f, 1, 1);
    
    float val1, val2;
    mgr.surf2Dread(surf, val1, 0, 0);
    mgr.surf2Dread(surf, val2, 1, 1);
    
    assert(val1 == 255.0f);
    assert(val2 == 128.0f);
    assert(data[0] == 255);
    assert(data[4+1] == 128);
    
    mgr.destroySurface(surf);
    std::cout << "[PASS] Uint8 Surface Read/Write" << std::endl;
}

void test_int32_linear_sampling() {
    TextureManager& mgr = TextureManager::instance();
    std::vector<int32_t> data = {10, 20, 30, 40};
    
    TextureDescriptor desc;
    desc.elementType = TextureElementType::INT32;
    desc.filterMode = TextureFilterMode::LINEAR;
    
    TextureId tex;
    assert(mgr.createTexture(tex, data.data(), 2, 2, sizeof(int32_t), desc) == VGREResult::SUCCESS);
    
    // Nearest to center (0.5, 0.5) => 10
    assert(mgr.tex2D(tex, 0.5f, 0.5f) == 10.0f);
    
    // Between (0.5, 0.5) and (1.5, 0.5) => (10 + 20) / 2 = 15
    assert(mgr.tex2D(tex, 1.0f, 0.5f) == 15.0f);
    
    mgr.destroyTexture(tex);
    std::cout << "[PASS] Int32 Linear Sampling" << std::endl;
}

int main() {
    std::cout << "=== VGRE Texture Depth Support Unit Tests ===" << std::endl;
    test_int16_sampling();
    test_uint8_surface();
    test_int32_linear_sampling();
    std::cout << "\nAll texture depth tests passed!" << std::endl;
    return 0;
}
