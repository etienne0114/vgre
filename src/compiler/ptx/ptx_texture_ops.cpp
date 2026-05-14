// PTX texture / surface instruction translation map.
// Maps to existing VGRE texture builtins (declared in cpu_cuda_env.h).

#include "ptx_translator_internal.h"

namespace vgre {
namespace compiler {

const TranslateMap& getTextureMap() {
    static const TranslateMap kMap = {
        // ── Texture fetch (1D / 2D / 3D) ───────────────────────────────────
        // VGRE TextureManager currently supports single-channel float textures.
        // For v2/v4 vector returns we replicate the scalar sample to all channels.
        {"tex.1d.f32.s32", [](auto& o){
            return o[0]+" = vgre_tex1D_f32((uint64_t)"+o[1]+",(float)(int)"+o[2]+");";
        }},
        {"tex.1d.f32.f32", [](auto& o){
            return o[0]+" = vgre_tex1D_f32((uint64_t)"+o[1]+",(float)"+o[2]+");";
        }},
        {"tex.1d.v2.f32.s32", [](auto& o){
            return "{float _ts=vgre_tex1D_f32((uint64_t)"+o[2]+",(float)(int)"+o[3]+
                   "); "+o[0]+"=_ts; "+o[1]+"=_ts; }";
        }},
        {"tex.1d.v2.f32.f32", [](auto& o){
            return "{float _ts=vgre_tex1D_f32((uint64_t)"+o[2]+",(float)"+o[3]+
                   "); "+o[0]+"=_ts; "+o[1]+"=_ts; }";
        }},
        {"tex.1d.v4.f32.s32", [](auto& o){
            return "{float _ts=vgre_tex1D_f32((uint64_t)"+o[4]+",(float)(int)"+o[5]+
                   "); "+o[0]+"=_ts; "+o[1]+"=_ts; "+o[2]+"=_ts; "+o[3]+"=_ts; }";
        }},
        {"tex.1d.v4.f32.f32", [](auto& o){
            return "{float _ts=vgre_tex1D_f32((uint64_t)"+o[4]+",(float)"+o[5]+
                   "); "+o[0]+"=_ts; "+o[1]+"=_ts; "+o[2]+"=_ts; "+o[3]+"=_ts; }";
        }},
        {"tex.2d.f32.f32", [](auto& o){
            return o[0]+" = vgre_tex2D_f32((uint64_t)"+o[1]+",(float)"+o[2]+
                   ",(float)"+o[3]+");";
        }},
        {"tex.2d.v2.f32.f32", [](auto& o){
            return "{float _ts=vgre_tex2D_f32((uint64_t)"+o[2]+",(float)"+o[3]+
                   ",(float)"+o[4]+"); "+o[0]+"=_ts; "+o[1]+"=_ts; }";
        }},
        {"tex.2d.v4.f32.f32", [](auto& o){
            return "{float _ts=vgre_tex2D_f32((uint64_t)"+o[4]+",(float)"+o[5]+
                   ",(float)"+o[6]+"); "+o[0]+"=_ts; "+o[1]+"=_ts; "+
                   o[2]+"=_ts; "+o[3]+"=_ts; }";
        }},
        {"tex.3d.f32.f32", [](auto& o){
            return o[0]+" = vgre_tex3D_f32((uint64_t)"+o[1]+",(float)"+o[2]+
                   ",(float)"+o[3]+",(float)"+o[4]+");";
        }},
        {"tex.3d.v2.f32.f32", [](auto& o){
            return "{float _ts=vgre_tex3D_f32((uint64_t)"+o[2]+",(float)"+o[3]+
                   ",(float)"+o[4]+",(float)"+o[5]+"); "+o[0]+"=_ts; "+o[1]+"=_ts; }";
        }},
        {"tex.3d.v4.f32.f32", [](auto& o){
            return "{float _ts=vgre_tex3D_f32((uint64_t)"+o[4]+",(float)"+o[5]+
                   ",(float)"+o[6]+",(float)"+o[7]+
                   "); "+o[0]+"=_ts; "+o[1]+"=_ts; "+
                   o[2]+"=_ts; "+o[3]+"=_ts; }";
        }},
        // ── Texel gather (tld4) ────────────────────────────────────────────
        // tld4 gathers 4 texels from a 2×2 neighborhood. CPU emulation: sample once.
        {"tld4.2d.v4.f32.f32", [](auto& o){
            return "{float _ts=vgre_tex2D_f32((uint64_t)"+o[4]+",(float)"+o[5]+
                   ",(float)"+o[6]+"); "+o[0]+"=_ts; "+o[1]+"=_ts; "+
                   o[2]+"=_ts; "+o[3]+"=_ts; }";
        }},
        // ── Texture query (txq) ────────────────────────────────────────────
        // VGRE does not expose texture metadata queries; return conservative defaults.
        {"txq.width.u32",  [](auto& o){ return o[0]+" = 1024u; /* txq.width default */"; }},
        {"txq.height.u32", [](auto& o){ return o[0]+" = 1024u; /* txq.height default */"; }},
        {"txq.depth.u32",  [](auto& o){ return o[0]+" = 1u; /* txq.depth default */"; }},
        {"txq.channels.u32",[](auto& o){ return o[0]+" = 1u; /* txq.channels default */"; }},
        // ── Surface load / store (suld / sust) ─────────────────────────────
        {"suld.2d.f32", [](auto& o){
            return "{float _sv; vgre_surf2Dread_f32((uint64_t)"+o[3]+",&_sv,"+
                   o[1]+","+o[2]+"); "+o[0]+"=_sv; }";
        }},
        {"suld.2d.v2.f32", [](auto& o){
            return "{float _sv; vgre_surf2Dread_f32((uint64_t)"+o[4]+",&_sv,"+
                   o[2]+","+o[3]+"); "+o[0]+"=_sv; "+o[1]+"=_sv; }";
        }},
        {"suld.2d.v4.f32", [](auto& o){
            return "{float _sv; vgre_surf2Dread_f32((uint64_t)"+o[6]+",&_sv,"+
                   o[4]+","+o[5]+"); "+o[0]+"=_sv; "+o[1]+"=_sv; "+
                   o[2]+"=_sv; "+o[3]+"=_sv; }";
        }},
        {"sust.2d.f32", [](auto& o){
            return "vgre_surf2Dwrite_f32((uint64_t)"+o[3]+",(float)"+o[0]+","+
                   o[1]+","+o[2]+");";
        }},
        {"sust.2d.v2.f32", [](auto& o){
            return "{vgre_surf2Dwrite_f32((uint64_t)"+o[4]+",(float)"+o[0]+","+
                   o[2]+","+o[3]+"); vgre_surf2Dwrite_f32((uint64_t)"+o[4]+
                   ",(float)"+o[1]+","+o[2]+","+o[3]+"); }";
        }},
        {"sust.2d.v4.f32", [](auto& o){
            return "{vgre_surf2Dwrite_f32((uint64_t)"+o[6]+",(float)"+o[0]+","+
                   o[4]+","+o[5]+"); vgre_surf2Dwrite_f32((uint64_t)"+o[6]+
                   ",(float)"+o[1]+","+o[4]+","+o[5]+"); vgre_surf2Dwrite_f32((uint64_t)"+o[6]+
                   ",(float)"+o[2]+","+o[4]+","+o[5]+"); vgre_surf2Dwrite_f32((uint64_t)"+o[6]+
                   ",(float)"+o[3]+","+o[4]+","+o[5]+"); }";
        }},
    };
    return kMap;
}

} // namespace compiler
} // namespace vgre
