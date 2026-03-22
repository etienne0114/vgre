#include <cstdint>
#include "vgre/core/texture_manager.h"

using namespace vgre;
using namespace vgre::core;

extern "C" {
float vgre_tex1D_f32(uint64_t tex, float x);
float vgre_tex2D_f32(uint64_t tex, float x, float y);
float vgre_tex3D_f32(uint64_t tex, float x, float y, float z);
void vgre_surf2Dwrite_f32(uint64_t surf, float val, int x, int y);
void vgre_surf2Dread_f32(uint64_t surf, float* val, int x, int y);
}

extern "C" {

float vgre_tex1D_f32(uint64_t tex, float x) {
    return TextureManager::instance().tex1D(tex, x);
}

float vgre_tex2D_f32(uint64_t tex, float x, float y) {
    return TextureManager::instance().tex2D(tex, x, y);
}

float vgre_tex3D_f32(uint64_t tex, float x, float y, float z) {
    return TextureManager::instance().tex3D(tex, x, y, z);
}

void vgre_surf2Dwrite_f32(uint64_t surf, float val, int x, int y) {
    TextureManager::instance().surf2Dwrite(surf, val, x, y);
}

void vgre_surf2Dread_f32(uint64_t surf, float* val, int x, int y) {
    if (val) {
        TextureManager::instance().surf2Dread(surf, *val, x, y);
    }
}

} // extern "C"
