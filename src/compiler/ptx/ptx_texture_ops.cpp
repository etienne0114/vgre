// PTX texture / surface instruction translation map.
// Maps to VGRE texture builtins (declared in cpu_cuda_env.h).

#include "ptx_translator_internal.h"

namespace vgre {
namespace compiler {

const TranslateMap& getTextureMap() {
    static const TranslateMap kMap = {
        // ── Texture fetch scalar (1D / 2D / 3D) ───────────────────────────
        {"tex.1d.f32.s32", [](auto& o){
            return o[0]+" = vgre_tex1D_f32((uint64_t)"+o[1]+",(float)(int)"+o[2]+");";
        }},
        {"tex.1d.f32.f32", [](auto& o){
            return o[0]+" = vgre_tex1D_f32((uint64_t)"+o[1]+",(float)"+o[2]+");";
        }},
        // ── Texture fetch v2 (1D) — per-channel for packed float2 textures ─
        {"tex.1d.v2.f32.s32", [](auto& o){
            return "{uint64_t _th=(uint64_t)"+o[2]+"; float _tx=(float)(int)"+o[3]+"; "
                   +o[0]+"=vgre_tex1D_chan_f32(_th,_tx,0u); "
                   +o[1]+"=vgre_tex1D_chan_f32(_th,_tx,1u); }";
        }},
        {"tex.1d.v2.f32.f32", [](auto& o){
            return "{uint64_t _th=(uint64_t)"+o[2]+"; float _tx=(float)"+o[3]+"; "
                   +o[0]+"=vgre_tex1D_chan_f32(_th,_tx,0u); "
                   +o[1]+"=vgre_tex1D_chan_f32(_th,_tx,1u); }";
        }},
        // ── Texture fetch v4 (1D) — per-channel for packed float4 textures ─
        {"tex.1d.v4.f32.s32", [](auto& o){
            return "{uint64_t _th=(uint64_t)"+o[4]+"; float _tx=(float)(int)"+o[5]+"; "
                   +o[0]+"=vgre_tex1D_chan_f32(_th,_tx,0u); "
                   +o[1]+"=vgre_tex1D_chan_f32(_th,_tx,1u); "
                   +o[2]+"=vgre_tex1D_chan_f32(_th,_tx,2u); "
                   +o[3]+"=vgre_tex1D_chan_f32(_th,_tx,3u); }";
        }},
        {"tex.1d.v4.f32.f32", [](auto& o){
            return "{uint64_t _th=(uint64_t)"+o[4]+"; float _tx=(float)"+o[5]+"; "
                   +o[0]+"=vgre_tex1D_chan_f32(_th,_tx,0u); "
                   +o[1]+"=vgre_tex1D_chan_f32(_th,_tx,1u); "
                   +o[2]+"=vgre_tex1D_chan_f32(_th,_tx,2u); "
                   +o[3]+"=vgre_tex1D_chan_f32(_th,_tx,3u); }";
        }},
        // ── Texture fetch scalar (2D) ──────────────────────────────────────
        {"tex.2d.f32.f32", [](auto& o){
            return o[0]+" = vgre_tex2D_f32((uint64_t)"+o[1]+",(float)"+o[2]+
                   ",(float)"+o[3]+");";
        }},
        // ── Texture fetch v2 (2D) ──────────────────────────────────────────
        {"tex.2d.v2.f32.f32", [](auto& o){
            return "{uint64_t _th=(uint64_t)"+o[2]+"; float _tx=(float)"+o[3]+
                   ",_ty=(float)"+o[4]+"; "
                   +o[0]+"=vgre_tex2D_chan_f32(_th,_tx,_ty,0u); "
                   +o[1]+"=vgre_tex2D_chan_f32(_th,_tx,_ty,1u); }";
        }},
        // ── Texture fetch v4 (2D) ──────────────────────────────────────────
        {"tex.2d.v4.f32.f32", [](auto& o){
            return "{uint64_t _th=(uint64_t)"+o[4]+"; float _tx=(float)"+o[5]+
                   ",_ty=(float)"+o[6]+"; "
                   +o[0]+"=vgre_tex2D_chan_f32(_th,_tx,_ty,0u); "
                   +o[1]+"=vgre_tex2D_chan_f32(_th,_tx,_ty,1u); "
                   +o[2]+"=vgre_tex2D_chan_f32(_th,_tx,_ty,2u); "
                   +o[3]+"=vgre_tex2D_chan_f32(_th,_tx,_ty,3u); }";
        }},
        // ── Texture fetch scalar (3D) ──────────────────────────────────────
        {"tex.3d.f32.f32", [](auto& o){
            return o[0]+" = vgre_tex3D_f32((uint64_t)"+o[1]+",(float)"+o[2]+
                   ",(float)"+o[3]+",(float)"+o[4]+");";
        }},
        // ── Texture fetch v2/v4 (3D) ───────────────────────────────────────
        {"tex.3d.v2.f32.f32", [](auto& o){
            return "{uint64_t _th=(uint64_t)"+o[2]+"; float _tx=(float)"+o[3]+
                   ",_ty=(float)"+o[4]+",_tz=(float)"+o[5]+"; "
                   +o[0]+"=vgre_tex3D_chan_f32(_th,_tx,_ty,_tz,0u); "
                   +o[1]+"=vgre_tex3D_chan_f32(_th,_tx,_ty,_tz,1u); }";
        }},
        {"tex.3d.v4.f32.f32", [](auto& o){
            return "{uint64_t _th=(uint64_t)"+o[4]+"; float _tx=(float)"+o[5]+
                   ",_ty=(float)"+o[6]+",_tz=(float)"+o[7]+"; "
                   +o[0]+"=vgre_tex3D_chan_f32(_th,_tx,_ty,_tz,0u); "
                   +o[1]+"=vgre_tex3D_chan_f32(_th,_tx,_ty,_tz,1u); "
                   +o[2]+"=vgre_tex3D_chan_f32(_th,_tx,_ty,_tz,2u); "
                   +o[3]+"=vgre_tex3D_chan_f32(_th,_tx,_ty,_tz,3u); }";
        }},
        // ── Texel gather (tld4) ────────────────────────────────────────────
        // tld4 gathers 4 texels from a 2×2 footprint; emulate as 4 point samples
        // at the four integer-offset neighbours and return per-channel.
        {"tld4.2d.v4.f32.f32", [](auto& o){
            return "{uint64_t _th=(uint64_t)"+o[4]+"; float _tx=(float)"+o[5]+
                   ",_ty=(float)"+o[6]+"; "
                   +o[0]+"=vgre_tex2D_chan_f32(_th,_tx,     _ty,     0u); "
                   +o[1]+"=vgre_tex2D_chan_f32(_th,_tx+1.f, _ty,     0u); "
                   +o[2]+"=vgre_tex2D_chan_f32(_th,_tx,     _ty+1.f, 0u); "
                   +o[3]+"=vgre_tex2D_chan_f32(_th,_tx+1.f, _ty+1.f, 0u); }";
        }},
        // ── Texture query (txq) — returns real texture dimensions ──────────
        {"txq.width.u32",   [](auto& o){ return o[0]+" = vgre_txq_width((uint64_t)"+o[1]+");"; }},
        {"txq.height.u32",  [](auto& o){ return o[0]+" = vgre_txq_height((uint64_t)"+o[1]+");"; }},
        {"txq.depth.u32",   [](auto& o){ return o[0]+" = vgre_txq_depth((uint64_t)"+o[1]+");"; }},
        {"txq.channels.u32",[](auto& o){ return o[0]+" = vgre_txq_channels((uint64_t)"+o[1]+");"; }},
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
