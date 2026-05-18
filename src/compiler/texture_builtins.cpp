#include <cstdint>
#include "vgre/core/texture_manager.h"

using namespace vgre;
using namespace vgre::core;

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

float vgre_tex1Dfetch_f32(uint64_t tex, int x) {
    return TextureManager::instance().tex1Dfetch(tex, x);
}

void vgre_surf2Dwrite_f32(uint64_t surf, float val, int x, int y) {
    TextureManager::instance().surf2Dwrite(surf, val, x, y);
}

void vgre_surf2Dread_f32(uint64_t surf, float* val, int x, int y) {
    if (val) {
        TextureManager::instance().surf2Dread(surf, *val, x, y);
    }
}

// ── Per-channel tex fetch (for PTX tex.*.v2/v4) ───────────────────────────
float vgre_tex1D_chan_f32(uint64_t tex, float x, unsigned channel) {
    return TextureManager::instance().tex1DChan(tex, x, channel);
}

float vgre_tex2D_chan_f32(uint64_t tex, float x, float y, unsigned channel) {
    return TextureManager::instance().tex2DChan(tex, x, y, channel);
}

float vgre_tex3D_chan_f32(uint64_t tex, float x, float y, float z, unsigned channel) {
    return TextureManager::instance().tex3DChan(tex, x, y, z, channel);
}

// ── Texture dimension queries (for PTX txq) ───────────────────────────────
uint32_t vgre_txq_width(uint64_t tex) {
    TextureManager::TextureInfo info;
    if (!TextureManager::instance().getTextureInfo(tex, info)) return 1u;
    return static_cast<uint32_t>(info.width ? info.width : 1);
}

uint32_t vgre_txq_height(uint64_t tex) {
    TextureManager::TextureInfo info;
    if (!TextureManager::instance().getTextureInfo(tex, info)) return 1u;
    return static_cast<uint32_t>(info.height ? info.height : 1);
}

uint32_t vgre_txq_depth(uint64_t tex) {
    TextureManager::TextureInfo info;
    if (!TextureManager::instance().getTextureInfo(tex, info)) return 1u;
    return static_cast<uint32_t>(info.depth ? info.depth : 1);
}

uint32_t vgre_txq_channels(uint64_t tex) {
    TextureManager::TextureInfo info;
    if (!TextureManager::instance().getTextureInfo(tex, info)) return 1u;
    return info.channels;
}

} // extern "C"
