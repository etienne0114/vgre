// PTX conversion, precision, and cooperative-group instruction translation map.

#include "ptx_translator_internal.h"

namespace vgre {
namespace compiler {

const TranslateMap& getConversionMap() {
    static const TranslateMap kMap = {
        // ── Missing cvt.* variants (float ↔ integer, signed/unsigned) ────────
        // f32 → s32
        {"cvt.rn.s32.f32", [](auto& o){ return o[0]+" = (int)("+o[1]+");"; }},
        {"cvt.rz.s32.f32", [](auto& o){ return o[0]+" = (int)("+o[1]+");"; }},
        {"cvt.rm.s32.f32", [](auto& o){ return o[0]+" = (int)__builtin_floorf("+o[1]+");"; }},
        {"cvt.rp.s32.f32", [](auto& o){ return o[0]+" = (int)__builtin_ceilf("+o[1]+");"; }},
        // f32 → u32
        {"cvt.rn.u32.f32", [](auto& o){ return o[0]+" = (unsigned)("+o[1]+");"; }},
        {"cvt.rz.u32.f32", [](auto& o){ return o[0]+" = (unsigned)("+o[1]+");"; }},
        {"cvt.rm.u32.f32", [](auto& o){ return o[0]+" = (unsigned)__builtin_floorf("+o[1]+");"; }},
        {"cvt.rp.u32.f32", [](auto& o){ return o[0]+" = (unsigned)__builtin_ceilf("+o[1]+");"; }},
        // f64 → s32
        {"cvt.rn.s32.f64", [](auto& o){ return o[0]+" = (int)("+o[1]+");"; }},
        {"cvt.rz.s32.f64", [](auto& o){ return o[0]+" = (int)("+o[1]+");"; }},
        {"cvt.rm.s32.f64", [](auto& o){ return o[0]+" = (int)__builtin_floor("+o[1]+");"; }},
        {"cvt.rp.s32.f64", [](auto& o){ return o[0]+" = (int)__builtin_ceil("+o[1]+");"; }},
        // f64 → u32
        {"cvt.rn.u32.f64", [](auto& o){ return o[0]+" = (unsigned)("+o[1]+");"; }},
        {"cvt.rz.u32.f64", [](auto& o){ return o[0]+" = (unsigned)("+o[1]+");"; }},
        {"cvt.rm.u32.f64", [](auto& o){ return o[0]+" = (unsigned)__builtin_floor("+o[1]+");"; }},
        {"cvt.rp.u32.f64", [](auto& o){ return o[0]+" = (unsigned)__builtin_ceil("+o[1]+");"; }},
        // s32 → f32 (round-nearest is default for integer→float)
        {"cvt.rn.f32.s32", [](auto& o){ return o[0]+" = (float)("+o[1]+");"; }},
        {"cvt.rn.f32.u32", [](auto& o){ return o[0]+" = (float)(unsigned)("+o[1]+");"; }},
        // s32 → f64
        {"cvt.rn.f64.s32", [](auto& o){ return o[0]+" = (double)("+o[1]+");"; }},
        {"cvt.rn.f64.u32", [](auto& o){ return o[0]+" = (double)(unsigned)("+o[1]+");"; }},
        // f32 → f16 / f16 → f32 (already have some; add more rounding modes)
        {"cvt.rn.f16.f32", [](auto& o){ return o[0]+" = (__half)("+o[1]+");"; }},
        {"cvt.rz.f16.f32", [](auto& o){ return o[0]+" = (__half)("+o[1]+");"; }},
        {"cvt.rn.f32.f16", [](auto& o){ return o[0]+" = (float)("+o[1]+");"; }},
        // f64 → f32 (already have rn; add others)
        {"cvt.rz.f32.f64", [](auto& o){ return o[0]+" = (float)("+o[1]+");"; }},
        {"cvt.rm.f32.f64", [](auto& o){ return o[0]+" = (float)__builtin_floor("+o[1]+");"; }},
        {"cvt.rp.f32.f64", [](auto& o){ return o[0]+" = (float)__builtin_ceil("+o[1]+");"; }},
        // f32 → f64 (already have rn)
        {"cvt.rz.f64.f32", [](auto& o){ return o[0]+" = (double)("+o[1]+");"; }},
        {"cvt.rm.f64.f32", [](auto& o){ return o[0]+" = (double)__builtin_floorf("+o[1]+");"; }},
        {"cvt.rp.f64.f32", [](auto& o){ return o[0]+" = (double)__builtin_ceilf("+o[1]+");"; }},
        // s64 / u64 ↔ float
        {"cvt.rn.f32.s64", [](auto& o){ return o[0]+" = (float)(long long)("+o[1]+");"; }},
        {"cvt.rn.f32.u64", [](auto& o){ return o[0]+" = (float)(unsigned long long)("+o[1]+");"; }},
        {"cvt.rn.f64.s64", [](auto& o){ return o[0]+" = (double)(long long)("+o[1]+");"; }},
        {"cvt.rn.f64.u64", [](auto& o){ return o[0]+" = (double)(unsigned long long)("+o[1]+");"; }},
        {"cvt.rn.s64.f32", [](auto& o){ return o[0]+" = (long long)("+o[1]+");"; }},
        {"cvt.rn.u64.f32", [](auto& o){ return o[0]+" = (unsigned long long)("+o[1]+");"; }},
        {"cvt.rn.s64.f64", [](auto& o){ return o[0]+" = (long long)("+o[1]+");"; }},
        {"cvt.rn.u64.f64", [](auto& o){ return o[0]+" = (unsigned long long)("+o[1]+");"; }},
        // saturating conversions
        {"cvt.sat.f32.f32",[](auto& o){ return o[0]+" = ("+o[1]+"<0.f?0.f:("+o[1]+">1.f?1.f:"+o[1]+"));"; }},
        {"cvt.sat.f16.f32",[](auto& o){ return o[0]+" = (__half)("+o[1]+"<0.f?0.f:("+o[1]+">1.f?1.f:"+o[1]+"));"; }},
        // ── sqrt.rn.f32 (missing from core map) ────────────────────────────
        {"sqrt.rn.f32", [](auto& o){ return o[0]+" = __builtin_sqrtf("+o[1]+");"; }},
        {"sqrt.rz.f32", [](auto& o){ return o[0]+" = __builtin_sqrtf("+o[1]+");"; }},
        {"sqrt.rm.f32", [](auto& o){ return o[0]+" = __builtin_sqrtf("+o[1]+");"; }},
        {"sqrt.rp.f32", [](auto& o){ return o[0]+" = __builtin_sqrtf("+o[1]+");"; }},
        // ── FP16 vector loads ────────────────────────────────────────────────
        {"ld.global.v2.f16", [](auto& o){
            return "{__half* _hp=(__half*)("+o[2]+"); "+o[0]+"=_hp[0]; "+o[1]+"=_hp[1]; }";
        }},
        {"ld.global.v4.f16", [](auto& o){
            return "{__half* _hp=(__half*)("+o[4]+"); "+o[0]+"=_hp[0]; "+o[1]+"=_hp[1]; "+
                   o[2]+"=_hp[2]; "+o[3]+"=_hp[3]; }";
        }},
        {"st.global.v2.f16", [](auto& o){
            return "{__half* _hp=(__half*)("+o[0]+"); _hp[0]="+o[1]+"; _hp[1]="+o[2]+"; }";
        }},
        {"st.global.v4.f16", [](auto& o){
            return "{__half* _hp=(__half*)("+o[0]+"); _hp[0]="+o[1]+"; _hp[1]="+o[2]+
                   "; _hp[2]="+o[3]+"; _hp[3]="+o[4]+"; }";
        }},
        // ── Cooperative group primitives (Hopper / Ampere) ─────────────────
        // match.sync: return a bitmask of threads in the warp with the same value.
        // In serial CPU model all threads appear identical; return full mask.
        {"match.sync.eq.b32", [](auto& o){ return o[0]+" = 0xFFFFFFFFu; /* match.sync serial */"; }},
        {"match.sync.eq.b64", [](auto& o){ return o[0]+" = 0xFFFFFFFFu; /* match.sync serial */"; }},
        {"match.sync.lt.b32", [](auto& o){ return o[0]+" = 0xFFFFFFFFu; /* match.sync serial */"; }},
        {"match.sync.lt.b64", [](auto& o){ return o[0]+" = 0xFFFFFFFFu; /* match.sync serial */"; }},
        // elect.sync: elect one thread as leader. Serial model → thread 0 is elected.
        {"elect.sync", [](auto& o){ return o[0]+" = 0; /* elect.sync serial → tid 0 */"; }},
        // grid.sync / griddepcontrol ────────────────────────────────────────
        {"grid.sync", [](auto&){ return "vgre_jit_syncgrid();"; }},
        {"griddepcontrol.launch_dependents", [](auto&){ return "/* griddepcontrol.launch_dependents */"; }},
        {"griddepcontrol.wait", [](auto&){ return "/* griddepcontrol.wait */"; }},
    };
    return kMap;
}

} // namespace compiler
} // namespace vgre
