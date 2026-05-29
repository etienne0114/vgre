// PTX opcode to C++ translation map.

#include "ptx_translator_internal.h"

namespace vgre {
namespace compiler {

const TranslateMap& getMap() {
    static const TranslateMap kMap = {
        // ── Integer arithmetic ─────────────────────────────────────────────
        {"add.s32",  [](auto& o){ return o[0]+" = "+o[1]+" + "+o[2]+";"; }},
        {"add.u32",  [](auto& o){ return o[0]+" = "+o[1]+" + "+o[2]+";"; }},
        {"add.s64",  [](auto& o){ return o[0]+" = "+o[1]+" + "+o[2]+";"; }},
        {"sub.s32",  [](auto& o){ return o[0]+" = "+o[1]+" - "+o[2]+";"; }},
        {"mul.lo.s32",[](auto& o){ return o[0]+" = (int)((int)"+o[1]+"*(int)"+o[2]+");"; }},
        {"mul.lo.u32",[](auto& o){ return o[0]+" = (unsigned)((unsigned)"+o[1]+"*(unsigned)"+o[2]+");"; }},
        {"mul.hi.u32",[](auto& o){
            return o[0]+" = (unsigned)(((unsigned long long)"+o[1]+"*(unsigned long long)"+o[2]+")>>32);";
        }},
        {"mad.lo.s32",[](auto& o){ return o[0]+" = "+o[1]+"*"+o[2]+"+"+o[3]+";"; }},
        {"mad.lo.u32",[](auto& o){ return o[0]+" = "+o[1]+"*"+o[2]+"+"+o[3]+";"; }},
        {"div.s32",  [](auto& o){ return o[0]+" = "+o[1]+" / "+o[2]+";"; }},
        {"rem.s32",  [](auto& o){ return o[0]+" = "+o[1]+" % "+o[2]+";"; }},
        {"neg.s32",  [](auto& o){ return o[0]+" = -("+o[1]+");"; }},
        // ── Carry-flag arithmetic (multi-word wide integer) ────────────────────
        // _cc models the CC.CF carry-flag register as a thread-local int.
        // Simplified: we propagate carry as part of the expression.
        {"add.cc.u32", [](auto& o){
            return "{ unsigned _s="+(o[1])+"+"+(o[2])+"; _cc=(_s<(unsigned)"+(o[1])+"); "+(o[0])+"=_s; }";
        }},
        {"add.cc.s32", [](auto& o){
            return "{ unsigned _s=(unsigned)"+(o[1])+"+(unsigned)"+(o[2])+"; _cc=(_s<(unsigned)"+(o[1])+"); "+(o[0])+"=(int)_s; }";
        }},
        {"addc.u32",   [](auto& o){
            return "{ unsigned _s="+(o[1])+"+"+(o[2])+"+(unsigned)_cc; "+(o[0])+"=_s; }";
        }},
        {"addc.cc.u32",[](auto& o){
            return "{ unsigned _t=(unsigned)"+(o[1])+"+(unsigned)"+(o[2])+"; unsigned _tc=_t+(unsigned)_cc; "
                   "_cc=(_tc<_t)||(_t<(unsigned)"+(o[1])+"); "+(o[0])+"=_tc; }";
        }},
        {"sub.cc.u32", [](auto& o){
            return "{ unsigned _s=(unsigned)"+(o[1])+"-(unsigned)"+(o[2])+"; _cc=((unsigned)"+(o[1])+"<(unsigned)"+(o[2])+"); "+(o[0])+"=_s; }";
        }},
        {"sub.cc.s32", [](auto& o){
            return "{ unsigned _s=(unsigned)"+(o[1])+"-(unsigned)"+(o[2])+"; _cc=((unsigned)"+(o[1])+"<(unsigned)"+(o[2])+"); "+(o[0])+"=(int)_s; }";
        }},
        {"subc.u32",   [](auto& o){
            return "{ unsigned _b=(unsigned)"+(o[2])+"+(unsigned)_cc; "+(o[0])+"=(unsigned)"+(o[1])+"-_b; }";
        }},
        {"subc.cc.u32",[](auto& o){
            return "{ unsigned _b=(unsigned)"+(o[2])+"+(unsigned)_cc; _cc=((unsigned)"+(o[1])+"<_b); "+(o[0])+"=(unsigned)"+(o[1])+"-_b; }";
        }},
        {"mad.hi.cc.u32",[](auto& o){
            return "{ unsigned long long _p=(unsigned long long)(unsigned)"+(o[1])+"*(unsigned long long)(unsigned)"+(o[2])+"; "
                   "unsigned _hi=(unsigned)(_p>>32); unsigned _acc=_hi+(unsigned)"+(o[3])+"; "
                   "_cc=(_acc<_hi); "+(o[0])+"=_acc; }";
        }},
        // ── 3-operand logic (lop3) ─────────────────────────────────────────────
        {"lop3.b32",   [](auto& o){
            return o[0]+" = vgre_lop3_b32((unsigned)"+o[1]+",(unsigned)"+o[2]+",(unsigned)"+o[3]+","
                   "(unsigned char)("+o[4]+"));";
        }},
        // ── Ampere mma.sync.aligned (m16n8k16 / m16n8k8 / m8n8k4) ─────────────
        // These are the low-level Ampere matrix multiply instructions used by
        // FlashAttention, CUTLASS 3.x, and Triton-generated PTX.
        // Emulated via the existing AVX-512 WMMA path from wmma_emulation.h.
        // Operand notation: D (out), A, B, C (in) — all fragmented across a warp.
        // In VGRE's serial CPU model, we treat each fragment as a complete tile.
        {"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32", [](auto& o){
            // d[0..3], a[0..7](f16 packed), b[0..3](f16 packed), c[0..3]
            return "vgre_mma_m16n8k16_f32_f16("+o[0]+","+o[1]+","+o[2]+","+o[3]+","
                   +o[4]+","+o[5]+","+o[6]+","+o[7]+","
                   +o[8]+","+o[9]+","
                   +o[10]+","+o[11]+","+o[12]+","+o[13]+");";
        }},
        {"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32", [](auto& o){
            return "vgre_mma_m16n8k8_tf32("+o[0]+","+o[1]+","+o[2]+","+o[3]+","
                   +o[4]+","+o[5]+","
                   +o[6]+","+o[7]+","+o[8]+","+o[9]+");";
        }},
        {"mma.sync.aligned.m8n8k4.row.col.f64", [](auto& o){
            return "vgre_mma_m8n8k4_f64("+o[0]+","+o[1]+","+o[2]+","+o[3]+");";
        }},
        // BF16 variants (used by Hopper/Ampere BF16 GEMMs)
        {"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32", [](auto& o){
            return "vgre_mma_m16n8k16_f32_bf16("+o[0]+","+o[1]+","+o[2]+","+o[3]+","
                   +o[4]+","+o[5]+","+o[6]+","+o[7]+","
                   +o[8]+","+o[9]+","
                   +o[10]+","+o[11]+","+o[12]+","+o[13]+");";
        }},
        // INT8 mma (INT8×INT8 → INT32 accumulate)
        {"mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32", [](auto& o){
            return "vgre_mma_m16n8k32_s8("+o[0]+","+o[1]+","+o[2]+","+o[3]+","
                   +o[4]+","+o[5]+","
                   +o[6]+","
                   +o[7]+","+o[8]+","+o[9]+","+o[10]+");";
        }},
        // INT4 mma (s4×s4 → s32 accumulate, m8n8k32, satfinite)  — 4.1.15
        {"mma.sync.aligned.m8n8k32.row.col.satfinite.s32.s4.s4.s32", [](auto& o){
            // o[0..1]=d0,d1; o[2]=a0; o[3]=b0; o[4..5]=c0,c1
            return "vgre_mma_m8n8k32_s4("+o[0]+","+o[1]+","
                   +o[2]+","
                   +o[3]+","
                   +o[4]+","+o[5]+");";
        }},
        // INT4 mma (u4×u4 → s32 accumulate, m8n8k32, satfinite)  — 4.1.15
        {"mma.sync.aligned.m8n8k32.row.col.satfinite.s32.u4.u4.s32", [](auto& o){
            return "vgre_mma_m8n8k32_u4("+o[0]+","+o[1]+","
                   +o[2]+","
                   +o[3]+","
                   +o[4]+","+o[5]+");";
        }},
        // Binary mma AND+POPC (b1×b1 → s32, m8n8k128)  — 4.1.15
        {"mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.and.popc", [](auto& o){
            // o[0..1]=d; o[2..5]=a0-a3; o[6..9]=b0-b3; o[10..11]=c0,c1
            return "vgre_mma_m8n8k128_b1_and("+o[0]+","+o[1]+","
                   +o[2]+","+o[3]+","+o[4]+","+o[5]+","
                   +o[6]+","+o[7]+","+o[8]+","+o[9]+","
                   +o[10]+","+o[11]+");";
        }},
        // Binary mma XOR+POPC (b1×b1 → s32, m8n8k128)  — 4.1.15
        {"mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.xor.popc", [](auto& o){
            return "vgre_mma_m8n8k128_b1_xor("+o[0]+","+o[1]+","
                   +o[2]+","+o[3]+","+o[4]+","+o[5]+","
                   +o[6]+","+o[7]+","+o[8]+","+o[9]+","
                   +o[10]+","+o[11]+");";
        }},
        // Warp-group MMA (Hopper) — full emulation via vgre_wgmma_* helpers.
        // operand layout: o[0]=descA, o[1]=descB, o[2]=d_ptr (FP32 accumulator)
        // The accumulator pointer is the output of wgmma; callers pass
        // reinterpret_cast<uint64_t>(d_array) as o[2].
        {"wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16", [](auto& o){
            auto d = o.size() > 2 ? o[2] : std::string("nullptr");
            return "vgre_wgmma_m64n256k16_bf16_f32((float*)(uintptr_t)("+d+"),"
                   "(uint64_t)("+o[0]+"),(uint64_t)("+o[1]+"));";
        }},
        {"wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16", [](auto& o){
            auto d = o.size() > 2 ? o[2] : std::string("nullptr");
            return "vgre_wgmma_m64n128k16_bf16_f32((float*)(uintptr_t)("+d+"),"
                   "(uint64_t)("+o[0]+"),(uint64_t)("+o[1]+"));";
        }},
        {"wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16", [](auto& o){
            auto d = o.size() > 2 ? o[2] : std::string("nullptr");
            return "vgre_wgmma_m64n64k16_bf16_f32((float*)(uintptr_t)("+d+"),"
                   "(uint64_t)("+o[0]+"),(uint64_t)("+o[1]+"));";
        }},
        {"wgmma.mma_async.sync.aligned.m64n256k16.f32.f16.f16", [](auto& o){
            auto d = o.size() > 2 ? o[2] : std::string("nullptr");
            return "vgre_wgmma_m64n256k16_f16_f32((float*)(uintptr_t)("+d+"),"
                   "(uint64_t)("+o[0]+"),(uint64_t)("+o[1]+"));";
        }},
        {"wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16", [](auto& o){
            auto d = o.size() > 2 ? o[2] : std::string("nullptr");
            return "vgre_wgmma_m64n128k16_f16_f32((float*)(uintptr_t)("+d+"),"
                   "(uint64_t)("+o[0]+"),(uint64_t)("+o[1]+"));";
        }},
        {"wgmma.mma_async.sync.aligned.m64n256k8.f32.tf32.tf32", [](auto& o){
            auto d = o.size() > 2 ? o[2] : std::string("nullptr");
            return "vgre_wgmma_m64n256k8_tf32_f32((float*)(uintptr_t)("+d+"),"
                   "(uint64_t)("+o[0]+"),(uint64_t)("+o[1]+"));";
        }},
        // TMA bulk async copy: cp.async.bulk copies from global→shared in one call.
        // CPU serial emulation: synchronous memcpy (no async staging needed).
        {"cp.async.bulk.tensor.1d.global.shared::cta.bulk_group", [](auto& o){
            return "vgre_cp_async_bulk((void*)(uintptr_t)("+o[0]+"),"
                   "(const void*)(uintptr_t)("+o[1]+"),"+(o.size()>2?o[2]:std::string("0"))+");";
        }},
        {"cp.async.bulk.tensor.2d.global.shared::cta.bulk_group", [](auto& o){
            // o[0]=dst, o[1]=tma_desc, o[2]=x, o[3]=y; tile dims come from desc->boxDim
            return "vgre_tma_load_2d_dispatch((void*)(uintptr_t)("+o[0]+"),"
                   "(const VgreTMADescriptor*)(uintptr_t)("+o[1]+"),"
                   +(o.size()>2?o[2]:std::string("0"))+","
                   +(o.size()>3?o[3]:std::string("0"))+");";
        }},
        {"cp.async.bulk.tensor.1d.shared::cta.global", [](auto& o){
            return "vgre_cp_async_bulk((void*)(uintptr_t)("+o[0]+"),"
                   "(const void*)(uintptr_t)("+o[1]+"),"+(o.size()>2?o[2]:std::string("0"))+");";
        }},
        // TMA fence/commit — no-ops in serial CPU model (no async pipeline)
        {"cp.async.bulk.commit_group",  [](auto&){ return "/* cp.async.bulk.commit_group */"; }},
        {"cp.async.bulk.wait_group",    [](auto&){ return "/* cp.async.bulk.wait_group */"; }},
        {"cp.async.bulk.wait_group.read",[](auto&){ return "/* cp.async.bulk.wait_group.read */"; }},
        // wgmma fence — serializes wgmma pipeline
        {"wgmma.fence.sync.aligned",    [](auto&){ return "__atomic_thread_fence(__ATOMIC_SEQ_CST);"; }},
        {"wgmma.commit_group.sync.aligned",[](auto&){ return "/* wgmma.commit_group */"; }},
        {"wgmma.wait_group.sync.aligned", [](auto&){
            return "__atomic_thread_fence(__ATOMIC_SEQ_CST);";
        }},
        // ── Warp reductions (redux.sync, bar.red) ─────────────────────────────
        // In serial CPU model, reduction is already complete per-thread.
        {"redux.sync.add.s32",  [](auto& o){ return o[0]+" = "+o[1]+";"; }},
        {"redux.sync.add.u32",  [](auto& o){ return o[0]+" = "+o[1]+";"; }},
        {"redux.sync.min.s32",  [](auto& o){ return o[0]+" = "+o[1]+";"; }},
        {"redux.sync.max.s32",  [](auto& o){ return o[0]+" = "+o[1]+";"; }},
        {"bar.red.popc.u32", [](auto& o){
            return o[0]+" = __syncthreads_count("+o[2]+");";
        }},
        {"bar.red.and.pred",  [](auto& o){
            return o[0]+" = __syncthreads_and("+o[2]+");";
        }},
        {"bar.red.or.pred",   [](auto& o){
            return o[0]+" = __syncthreads_or("+o[2]+");";
        }},
        {"abs.s32",  [](auto& o){ return o[0]+" = __builtin_abs("+o[1]+");"; }},
        {"min.s32",  [](auto& o){ return o[0]+" = ("+o[1]+"<"+o[2]+"?"+o[1]+":"+o[2]+");"; }},
        {"max.s32",  [](auto& o){ return o[0]+" = ("+o[1]+">"+o[2]+"?"+o[1]+":"+o[2]+");"; }},
        // ── Floating-point ─────────────────────────────────────────────────
        {"add.f32",  [](auto& o){ return o[0]+" = "+o[1]+" + "+o[2]+";"; }},
        {"add.rn.f32",[](auto& o){ return o[0]+" = "+o[1]+" + "+o[2]+";"; }},
        {"sub.f32",  [](auto& o){ return o[0]+" = "+o[1]+" - "+o[2]+";"; }},
        {"mul.f32",  [](auto& o){ return o[0]+" = "+o[1]+" * "+o[2]+";"; }},
        {"mul.rn.f32",[](auto& o){ return o[0]+" = "+o[1]+" * "+o[2]+";"; }},
        {"div.approx.f32",[](auto& o){ return o[0]+" = "+o[1]+" / "+o[2]+";"; }},
        {"div.rn.f32",[](auto& o){ return o[0]+" = "+o[1]+" / "+o[2]+";"; }},
        {"sqrt.approx.f32",[](auto& o){ return o[0]+" = __builtin_sqrtf("+o[1]+");"; }},
        {"rsqrt.approx.f32",[](auto& o){ return o[0]+" = 1.0f/__builtin_sqrtf("+o[1]+");"; }},
        {"fma.rn.f32",[](auto& o){ return o[0]+" = "+o[1]+"*"+o[2]+"+"+o[3]+";"; }},
        {"neg.f32",  [](auto& o){ return o[0]+" = -("+o[1]+");"; }},
        {"abs.f32",  [](auto& o){ return o[0]+" = __builtin_fabsf("+o[1]+");"; }},
        {"min.f32",  [](auto& o){ return o[0]+" = ("+o[1]+"<"+o[2]+"?"+o[1]+":"+o[2]+");"; }},
        {"max.f32",  [](auto& o){ return o[0]+" = ("+o[1]+">"+o[2]+"?"+o[1]+":"+o[2]+");"; }},
        // ── Bitwise ─────────────────────────────────────────────────────────
        {"and.b32",  [](auto& o){ return o[0]+" = "+o[1]+" & "+o[2]+";"; }},
        {"or.b32",   [](auto& o){ return o[0]+" = "+o[1]+" | "+o[2]+";"; }},
        {"xor.b32",  [](auto& o){ return o[0]+" = "+o[1]+" ^ "+o[2]+";"; }},
        {"not.b32",  [](auto& o){ return o[0]+" = ~"+o[1]+";"; }},
        {"shl.b32",  [](auto& o){ return o[0]+" = "+o[1]+" << "+o[2]+";"; }},
        {"shr.s32",  [](auto& o){ return o[0]+" = (int)"+o[1]+" >> "+o[2]+";"; }},
        {"shr.u32",  [](auto& o){ return o[0]+" = (unsigned)"+o[1]+" >> "+o[2]+";"; }},
        // ── Memory (simplistic: treat ptr operand as C pointer) ────────────
        {"ld.global.f32",[](auto& o){ return o[0]+" = *(float*)("+o[1]+");"; }},
        {"ld.global.u32",[](auto& o){ return o[0]+" = *(unsigned*)("+o[1]+");"; }},
        {"st.global.f32",[](auto& o){ return "*(float*)("+o[0]+") = "+o[1]+";"; }},
        {"st.global.u32",[](auto& o){ return "*(unsigned*)("+o[0]+") = "+o[1]+";"; }},
        // ── Move / conversion ───────────────────────────────────────────────
        {"mov.u32",  [](auto& o){ return o[0]+" = "+o[1]+";"; }},
        {"mov.s32",  [](auto& o){ return o[0]+" = "+o[1]+";"; }},
        {"mov.f32",  [](auto& o){ return o[0]+" = "+o[1]+";"; }},
        {"cvt.rn.f32.s32",[](auto& o){ return o[0]+" = (float)("+o[1]+");"; }},
        {"cvt.rn.f32.u32",[](auto& o){ return o[0]+" = (float)("+o[1]+");"; }},
        {"cvt.rz.s32.f32",[](auto& o){ return o[0]+" = (int)("+o[1]+");"; }},
        // ── Bit counting ─────────────────────────────────────────────────────
        {"popc.b32", [](auto& o){ return o[0]+" = __builtin_popcount("+o[1]+");"; }},
        {"clz.b32",  [](auto& o){ return o[0]+" = __builtin_clz("+o[1]+");"; }},
        // ── FP64 arithmetic ────────────────────────────────────────────────
        {"add.f64",   [](auto& o){ return o[0]+" = "+o[1]+" + "+o[2]+";"; }},
        {"add.rn.f64",[](auto& o){ return o[0]+" = "+o[1]+" + "+o[2]+";"; }},
        {"sub.f64",   [](auto& o){ return o[0]+" = "+o[1]+" - "+o[2]+";"; }},
        {"mul.f64",   [](auto& o){ return o[0]+" = "+o[1]+" * "+o[2]+";"; }},
        {"mul.rn.f64",[](auto& o){ return o[0]+" = "+o[1]+" * "+o[2]+";"; }},
        {"div.rn.f64",[](auto& o){ return o[0]+" = "+o[1]+" / "+o[2]+";"; }},
        {"fma.rn.f64",[](auto& o){ return o[0]+" = "+o[1]+"*"+o[2]+"+"+o[3]+";"; }},
        {"sqrt.rn.f64",[](auto& o){ return o[0]+" = __builtin_sqrt("+o[1]+");"; }},
        {"neg.f64",   [](auto& o){ return o[0]+" = -("+o[1]+");"; }},
        {"abs.f64",   [](auto& o){ return o[0]+" = __builtin_fabs("+o[1]+");"; }},
        {"min.f64",   [](auto& o){ return o[0]+" = ("+o[1]+"<"+o[2]+"?"+o[1]+":"+o[2]+");"; }},
        {"max.f64",   [](auto& o){ return o[0]+" = ("+o[1]+">"+o[2]+"?"+o[1]+":"+o[2]+");"; }},
        // ── 64-bit integer arithmetic ──────────────────────────────────────
        {"add.u64",   [](auto& o){ return o[0]+" = (unsigned long long)("+o[1]+"+("+o[2]+"));"; }},
        {"add.s64",   [](auto& o){ return o[0]+" = (long long)("+o[1]+"+("+o[2]+"));"; }},
        {"sub.u64",   [](auto& o){ return o[0]+" = (unsigned long long)("+o[1]+"-("+o[2]+"));"; }},
        {"mul.lo.u64",[](auto& o){ return o[0]+" = (unsigned long long)("+o[1]+"*("+o[2]+"));"; }},
        {"mul.lo.s64",[](auto& o){ return o[0]+" = (long long)("+o[1]+"*("+o[2]+"));"; }},
        {"mul.hi.u64",[](auto& o){
            return o[0]+" = (unsigned long long)(__uint128_t("+o[1]+")*__uint128_t("+o[2]+")>>64);";
        }},
        {"mad.lo.u64",[](auto& o){ return o[0]+" = (unsigned long long)("+o[1]+"*("+o[2]+")+"+o[3]+");"; }},
        {"neg.s64",   [](auto& o){ return o[0]+" = -(long long)("+o[1]+");"; }},
        {"min.u64",   [](auto& o){ return o[0]+" = ("+o[1]+"<(unsigned long long)"+o[2]+"?"+o[1]+":"+o[2]+");"; }},
        {"max.u64",   [](auto& o){ return o[0]+" = ("+o[1]+">(unsigned long long)"+o[2]+"?"+o[1]+":"+o[2]+");"; }},
        {"shr.u64",   [](auto& o){ return o[0]+" = (unsigned long long)"+o[1]+" >> "+o[2]+";"; }},
        {"shl.b64",   [](auto& o){ return o[0]+" = (unsigned long long)"+o[1]+" << "+o[2]+";"; }},
        {"and.b64",   [](auto& o){ return o[0]+" = (unsigned long long)"+o[1]+" & "+o[2]+";"; }},
        {"or.b64",    [](auto& o){ return o[0]+" = (unsigned long long)"+o[1]+" | "+o[2]+";"; }},
        {"xor.b64",   [](auto& o){ return o[0]+" = (unsigned long long)"+o[1]+" ^ "+o[2]+";"; }},
        // ── Memory — shared memory ─────────────────────────────────────────
        {"ld.shared.f32",[](auto& o){ return o[0]+" = *(float*)("+o[1]+");"; }},
        {"ld.shared.u32",[](auto& o){ return o[0]+" = *(unsigned*)("+o[1]+");"; }},
        {"ld.shared.s32",[](auto& o){ return o[0]+" = *(int*)("+o[1]+");"; }},
        {"ld.shared.f64",[](auto& o){ return o[0]+" = *(double*)("+o[1]+");"; }},
        {"ld.shared.u64",[](auto& o){ return o[0]+" = *(unsigned long long*)("+o[1]+");"; }},
        {"ld.shared.v2.f32",[](auto& o){
            return "{float* _vp=(float*)("+o[2]+"); "+o[0]+"=_vp[0]; "+o[1]+"=_vp[1];}";
        }},
        {"st.shared.f32",[](auto& o){ return "*(float*)("+o[0]+") = "+o[1]+";"; }},
        {"st.shared.u32",[](auto& o){ return "*(unsigned*)("+o[0]+") = "+o[1]+";"; }},
        {"st.shared.s32",[](auto& o){ return "*(int*)("+o[0]+") = "+o[1]+";"; }},
        {"st.shared.f64",[](auto& o){ return "*(double*)("+o[0]+") = "+o[1]+";"; }},
        {"st.shared.u64",[](auto& o){ return "*(unsigned long long*)("+o[0]+") = "+o[1]+";"; }},
        // ── Memory — global vectorized ─────────────────────────────────────
        {"ld.global.f64",[](auto& o){ return o[0]+" = *(double*)("+o[1]+");"; }},
        {"ld.global.s32",[](auto& o){ return o[0]+" = *(int*)("+o[1]+");"; }},
        {"ld.global.s64",[](auto& o){ return o[0]+" = *(long long*)("+o[1]+");"; }},
        {"ld.global.u64",[](auto& o){ return o[0]+" = *(unsigned long long*)("+o[1]+");"; }},
        {"ld.global.v2.f32",[](auto& o){
            return "{float* _vp=(float*)("+o[2]+"); "+o[0]+"=_vp[0]; "+o[1]+"=_vp[1];}";
        }},
        {"ld.global.v4.f32",[](auto& o){
            return "{float* _vp=(float*)("+o[4]+"); "+o[0]+"=_vp[0]; "+o[1]+"=_vp[1]; "+
                   o[2]+"=_vp[2]; "+o[3]+"=_vp[3];}";
        }},
        {"ld.global.nc.f32",[](auto& o){ return o[0]+" = *(const float*)("+o[1]+");"; }},
        {"ld.global.nc.f64",[](auto& o){ return o[0]+" = *(const double*)("+o[1]+");"; }},
        {"ld.global.cs.f32",[](auto& o){ return o[0]+" = *(const float*)("+o[1]+");"; }},
        {"st.global.f64",  [](auto& o){ return "*(double*)("+o[0]+") = "+o[1]+";"; }},
        {"st.global.s32",  [](auto& o){ return "*(int*)("+o[0]+") = "+o[1]+";"; }},
        {"st.global.u64",  [](auto& o){ return "*(unsigned long long*)("+o[0]+") = "+o[1]+";"; }},
        {"st.global.v2.f32",[](auto& o){
            return "{float* _vp=(float*)("+o[0]+"); _vp[0]="+o[1]+"; _vp[1]="+o[2]+";}";
        }},
        {"st.global.v4.f32",[](auto& o){
            return "{float* _vp=(float*)("+o[0]+"); _vp[0]="+o[1]+"; _vp[1]="+o[2]+
                   "; _vp[2]="+o[3]+"; _vp[3]="+o[4]+";}";
        }},
        // ── Local memory ───────────────────────────────────────────────────
        {"ld.local.f32", [](auto& o){ return o[0]+" = *(float*)("+o[1]+");"; }},
        {"ld.local.u32", [](auto& o){ return o[0]+" = *(unsigned*)("+o[1]+");"; }},
        {"st.local.f32", [](auto& o){ return "*(float*)("+o[0]+") = "+o[1]+";"; }},
        {"st.local.u32", [](auto& o){ return "*(unsigned*)("+o[0]+") = "+o[1]+";"; }},
        // ── Atomic operations ──────────────────────────────────────────────
        {"atom.global.add.f32",[](auto& o){
            return o[0]+" = __atomic_fetch_add((float*)("+o[1]+"), "+o[2]+", __ATOMIC_SEQ_CST);";
        }},
        {"atom.global.add.s32",[](auto& o){
            return o[0]+" = __atomic_fetch_add((int*)("+o[1]+"), "+o[2]+", __ATOMIC_SEQ_CST);";
        }},
        {"atom.global.add.u32",[](auto& o){
            return o[0]+" = __atomic_fetch_add((unsigned*)("+o[1]+"), "+o[2]+", __ATOMIC_SEQ_CST);";
        }},
        {"atom.global.add.u64",[](auto& o){
            return o[0]+" = __atomic_fetch_add((unsigned long long*)("+o[1]+"), "+o[2]+", __ATOMIC_SEQ_CST);";
        }},
        {"atom.global.max.s32",[](auto& o){
            return "{int _vv="+o[2]+"; int _cur=*(int*)("+o[1]+"); while(_vv>_cur && !__atomic_compare_exchange_n((int*)("+o[1]+"), &_cur, _vv, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)){} "+o[0]+"=_cur;}";
        }},
        {"atom.global.min.s32",[](auto& o){
            return "{int _vv="+o[2]+"; int _cur=*(int*)("+o[1]+"); while(_vv<_cur && !__atomic_compare_exchange_n((int*)("+o[1]+"), &_cur, _vv, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)){} "+o[0]+"=_cur;}";
        }},
        {"atom.global.cas.b32",[](auto& o){
            return "{unsigned _exp=(unsigned)"+o[2]+"; __atomic_compare_exchange_n((unsigned*)("+o[1]+"), &_exp, "+o[3]+", 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST); "+o[0]+"=_exp;}";
        }},
        {"atom.global.exch.b32",[](auto& o){
            return o[0]+" = __atomic_exchange_n((unsigned*)("+o[1]+"), "+o[2]+", __ATOMIC_SEQ_CST);";
        }},
        {"red.global.add.f32",[](auto& o){
            return "__atomic_fetch_add((float*)("+o[0]+"), "+o[1]+", __ATOMIC_SEQ_CST);";
        }},
        {"red.global.add.s32",[](auto& o){
            return "__atomic_fetch_add((int*)("+o[0]+"), "+o[1]+", __ATOMIC_SEQ_CST);";
        }},
        // ── Warp voting ────────────────────────────────────────────────────
        {"vote.sync.all.pred",[](auto& o){ return o[0]+" = __all_sync("+o[2]+", "+o[1]+");"; }},
        {"vote.sync.any.pred",[](auto& o){ return o[0]+" = __any_sync("+o[2]+", "+o[1]+");"; }},
        {"vote.sync.ballot.b32",[](auto& o){ return o[0]+" = __ballot_sync("+o[2]+", "+o[1]+");"; }},
        {"activemask.b32",[](auto& o){ return o[0]+" = 0xFFFFFFFFu;"; }},
        // ── Warp shuffle ───────────────────────────────────────────────────
        {"shfl.sync.idx.b32",[](auto& o){
            // shfl.sync.idx.b32 dst, src, laneSel, memberMask, mask (5 operands in PTX)
            return o[0]+" = __shfl_sync((unsigned)"+o[4]+", "+o[1]+", (int)"+o[2]+", "+o[3]+");";
        }},
        {"shfl.sync.down.b32",[](auto& o){
            return o[0]+" = __shfl_down_sync((unsigned)"+o[4]+", "+o[1]+", "+o[2]+", "+o[3]+");";
        }},
        {"shfl.sync.up.b32",[](auto& o){
            return o[0]+" = __shfl_up_sync((unsigned)"+o[4]+", "+o[1]+", "+o[2]+", "+o[3]+");";
        }},
        {"shfl.sync.bfly.b32",[](auto& o){
            return o[0]+" = __shfl_xor_sync((unsigned)"+o[4]+", "+o[1]+", "+o[2]+", "+o[3]+");";
        }},
        // ── Convert (more variants) ────────────────────────────────────────
        {"cvt.rn.f32.f16",[](auto& o){ return o[0]+" = (float)("+o[1]+");"; }},
        {"cvt.rn.f16.f32",[](auto& o){ return o[0]+" = (__half)("+o[1]+");"; }},
        {"cvt.rn.f64.f32",[](auto& o){ return o[0]+" = (double)("+o[1]+");"; }},
        {"cvt.rn.f32.f64",[](auto& o){ return o[0]+" = (float)("+o[1]+");"; }},
        {"cvt.rn.f64.s32",[](auto& o){ return o[0]+" = (double)("+o[1]+");"; }},
        {"cvt.rn.f64.u32",[](auto& o){ return o[0]+" = (double)(unsigned)("+o[1]+");"; }},
        {"cvt.rz.s32.f64",[](auto& o){ return o[0]+" = (int)("+o[1]+");"; }},
        {"cvt.rn.u32.f32",[](auto& o){ return o[0]+" = (unsigned)("+o[1]+");"; }},
        {"cvt.s32.f32",   [](auto& o){ return o[0]+" = (int)("+o[1]+");"; }},
        {"cvt.u32.s32",   [](auto& o){ return o[0]+" = (unsigned)("+o[1]+");"; }},
        {"cvt.s32.u32",   [](auto& o){ return o[0]+" = (int)("+o[1]+");"; }},
        {"cvt.u64.u32",   [](auto& o){ return o[0]+" = (unsigned long long)(unsigned)("+o[1]+");"; }},
        {"cvt.u64.s32",   [](auto& o){ return o[0]+" = (unsigned long long)(int)("+o[1]+");"; }},
        {"cvt.s64.s32",   [](auto& o){ return o[0]+" = (long long)("+o[1]+");"; }},
        {"cvt.sat.u8.f32",[](auto& o){
            return o[0]+" = (unsigned char)(("+o[1]+"<0.f?0.f:("+o[1]+">255.f?255.f:"+o[1]+"))+0.5f);";
        }},
        // ── Move — extra variants ──────────────────────────────────────────
        {"mov.f64",   [](auto& o){ return o[0]+" = "+o[1]+";"; }},
        {"mov.u64",   [](auto& o){ return o[0]+" = (unsigned long long)("+o[1]+");"; }},
        {"mov.s64",   [](auto& o){ return o[0]+" = (long long)("+o[1]+");"; }},
        {"mov.b32",   [](auto& o){ return o[0]+" = "+o[1]+";"; }},
        {"mov.b64",   [](auto& o){ return o[0]+" = "+o[1]+";"; }},
        // ── Predicate / select ─────────────────────────────────────────────
        {"setp.lt.f32",[](auto& o){ return o[0]+" = ((float)("+o[1]+")<(float)("+o[2]+"));"; }},
        {"setp.gt.f32",[](auto& o){ return o[0]+" = ((float)("+o[1]+")>(float)("+o[2]+"));"; }},
        {"setp.le.f32",[](auto& o){ return o[0]+" = ((float)("+o[1]+")<=(float)("+o[2]+"));"; }},
        {"setp.ge.f32",[](auto& o){ return o[0]+" = ((float)("+o[1]+")>=(float)("+o[2]+"));"; }},
        {"setp.eq.f32",[](auto& o){ return o[0]+" = ((float)("+o[1]+")==(float)("+o[2]+"));"; }},
        {"setp.ne.f32",[](auto& o){ return o[0]+" = ((float)("+o[1]+")!=(float)("+o[2]+"));"; }},
        {"setp.lt.s32",[](auto& o){ return o[0]+" = ((int)("+o[1]+")<(int)("+o[2]+"));"; }},
        {"setp.gt.s32",[](auto& o){ return o[0]+" = ((int)("+o[1]+")>(int)("+o[2]+"));"; }},
        {"setp.le.s32",[](auto& o){ return o[0]+" = ((int)("+o[1]+")<=(int)("+o[2]+"));"; }},
        {"setp.ge.s32",[](auto& o){ return o[0]+" = ((int)("+o[1]+")>=(int)("+o[2]+"));"; }},
        {"setp.eq.s32",[](auto& o){ return o[0]+" = ((int)("+o[1]+")==(int)("+o[2]+"));"; }},
        {"setp.ne.s32",[](auto& o){ return o[0]+" = ((int)("+o[1]+")!=(int)("+o[2]+"));"; }},
        {"setp.eq.u32",[](auto& o){ return o[0]+" = ((unsigned)("+o[1]+")==(unsigned)("+o[2]+"));"; }},
        {"setp.ne.u32",[](auto& o){ return o[0]+" = ((unsigned)("+o[1]+")!=(unsigned)("+o[2]+"));"; }},
        {"setp.lt.u32",[](auto& o){ return o[0]+" = ((unsigned)("+o[1]+")<(unsigned)("+o[2]+"));"; }},
        {"setp.ge.u32",[](auto& o){ return o[0]+" = ((unsigned)("+o[1]+")>=(unsigned)("+o[2]+"));"; }},
        {"selp.f32",  [](auto& o){ return o[0]+" = ("+o[3]+") ? ("+o[1]+") : ("+o[2]+");"; }},
        {"selp.s32",  [](auto& o){ return o[0]+" = ("+o[3]+") ? (int)("+o[1]+") : (int)("+o[2]+");"; }},
        {"selp.u32",  [](auto& o){ return o[0]+" = ("+o[3]+") ? (unsigned)("+o[1]+") : (unsigned)("+o[2]+");"; }},
        {"selp.b32",  [](auto& o){ return o[0]+" = ("+o[3]+") ? "+o[1]+" : "+o[2]+";"; }},
        // ── Control flow ───────────────────────────────────────────────────
        {"bar.sync",  [](auto&){ return "__syncthreads();"; }},
        {"bar.arrive",[](auto&){ return "__syncthreads();"; }},
        {"membar.gl", [](auto&){ return "__atomic_thread_fence(__ATOMIC_SEQ_CST);"; }},
        {"membar.sys",[](auto&){ return "__atomic_thread_fence(__ATOMIC_SEQ_CST);"; }},
        {"membar.cta",[](auto&){ return "__atomic_thread_fence(__ATOMIC_SEQ_CST);"; }},
        {"ret",       [](auto&){ return "return;"; }},
        // ── Type punning ───────────────────────────────────────────────────
        {"mov.b32",   [](auto& o){ return o[0]+" = "+o[1]+";"; }},
        // ── Special register reads ─────────────────────────────────────────
        {"mov.u32.tid.x",  [](auto& o){ return o[0]+" = threadIdx.x;"; }},
        {"mov.u32.tid.y",  [](auto& o){ return o[0]+" = threadIdx.y;"; }},
        {"mov.u32.ntid.x", [](auto& o){ return o[0]+" = blockDim.x;"; }},
        {"mov.u32.ctaid.x",[](auto& o){ return o[0]+" = blockIdx.x;"; }},
        {"mov.u32.nctaid.x",[](auto& o){ return o[0]+" = gridDim.x;"; }},
        // ── FP32 intrinsics ────────────────────────────────────────────────
        {"ex2.approx.f32",[](auto& o){ return o[0]+" = __builtin_exp2f("+o[1]+");"; }},
        {"lg2.approx.f32",[](auto& o){ return o[0]+" = __builtin_log2f("+o[1]+");"; }},
        {"sin.approx.f32",[](auto& o){ return o[0]+" = __builtin_sinf("+o[1]+");"; }},
        {"cos.approx.f32",[](auto& o){ return o[0]+" = __builtin_cosf("+o[1]+");"; }},
        {"rcp.approx.f32",[](auto& o){ return o[0]+" = 1.0f/"+o[1]+";"; }},
        {"rcp.rn.f32",    [](auto& o){ return o[0]+" = 1.0f/"+o[1]+";"; }},
        // ── Type-punning bitcast ───────────────────────────────────────────
        {"bitcast.b32.f32",[](auto& o){
            return "{unsigned _bt; __builtin_memcpy(&_bt, &("+o[1]+"), 4); "+o[0]+"=_bt;}";
        }},
        {"bitcast.f32.b32",[](auto& o){
            return "{float _bt; __builtin_memcpy(&_bt, &("+o[1]+"), 4); "+o[0]+"=_bt;}";
        }},

        // ── prmt.b32 — byte permutation ───────────────────────────────────────
        // prmt.b32 d, a, b, c;
        // Selects 4 bytes from the 8-byte concatenation [b:a] according to
        // the 16-bit selector c (4 nibbles, each selects one byte 0-7 with
        // optional sign-extension for nibble >= 8).
        {"prmt.b32",[](auto& o){
            return o[0]+" = vgre_prmt_b32((unsigned)"+o[1]+",(unsigned)"+o[2]+",(unsigned)"+o[3]+");";
        }},
        {"prmt.b32.f4e",[](auto& o){  // forward-4-extract (nibble=0123)
            return o[0]+" = vgre_prmt_b32((unsigned)"+o[1]+",(unsigned)"+o[2]+",(unsigned)"+o[3]+");";
        }},
        {"prmt.b32.b4e",[](auto& o){  // backward-4-extract (nibble=0123 from high)
            return o[0]+" = vgre_prmt_b32((unsigned)"+o[1]+",(unsigned)"+o[2]+",(unsigned)"+o[3]+");";
        }},
        {"prmt.b32.rc8",[](auto& o){  // replicate-8 (replicate byte 0 to all 4)
            return o[0]+" = vgre_prmt_b32((unsigned)"+o[1]+",(unsigned)"+o[2]+",(unsigned)"+o[3]+");";
        }},
        {"prmt.b32.ecl",[](auto& o){  // edge-clamp-left
            return o[0]+" = vgre_prmt_b32((unsigned)"+o[1]+",(unsigned)"+o[2]+",(unsigned)"+o[3]+");";
        }},
        {"prmt.b32.ecr",[](auto& o){  // edge-clamp-right
            return o[0]+" = vgre_prmt_b32((unsigned)"+o[1]+",(unsigned)"+o[2]+",(unsigned)"+o[3]+");";
        }},
        {"prmt.b32.rc16",[](auto& o){  // replicate-16
            return o[0]+" = vgre_prmt_b32((unsigned)"+o[1]+",(unsigned)"+o[2]+",(unsigned)"+o[3]+");";
        }},

        // ── sad / dsad — sum of absolute differences ──────────────────────────
        // sad.u32  d, a, b, c;  →  d = |a-b| + c
        // sad.s32  d, a, b, c;  →  d = |a-b| + c (signed)
        // dsad.u32 d, a, b, c;  →  sum of 4 unsigned byte differences + c
        {"sad.u32",[](auto& o){
            return "{unsigned _a=(unsigned)"+o[1]+",_b=(unsigned)"+o[2]+"; "
                   +o[0]+"=(_a>_b?_a-_b:_b-_a)+(unsigned)"+o[3]+";}";
        }},
        {"sad.s32",[](auto& o){
            return "{int _a=(int)"+o[1]+",_b=(int)"+o[2]+"; "
                   +o[0]+"=(_a>_b?_a-_b:_b-_a)+(int)"+o[3]+";}";
        }},
        {"dsad.u32",[](auto& o){
            // Sum of absolute differences of four byte pairs (SIMD-within-register)
            return "{unsigned _a=(unsigned)"+o[1]+",_b=(unsigned)"+o[2]+"; unsigned _s=(unsigned)"+o[3]+";"
                   " for(int _i=0;_i<4;++_i){"
                   "  unsigned _ba=(_a>>(8*_i))&0xFF, _bb=(_b>>(8*_i))&0xFF;"
                   "  _s+=(_ba>_bb?_ba-_bb:_bb-_ba);"
                   "} "+o[0]+"=_s;}";
        }},

        // ── mad.hi — multiply-add high-half (wide integer variants) ──────────
        // mad.hi.s32  d, a, b, c;  →  d = (s64(a)*s64(b) >> 32) + c
        // mad.hi.u64  d, a, b, c;  →  d = (u128(a)*u128(b) >> 64) + c
        // mad.lo.s64  d, a, b, c;  →  d = s64(a)*s64(b) + c
        // mad.hi.s64  d, a, b, c;  →  d = (__int128(a)*__int128(b) >> 64) + c
        {"mad.hi.s32",[](auto& o){
            return o[0]+" = (int)(((long long)"+o[1]+"*(long long)"+o[2]+")>>32)+(int)"+o[3]+";";
        }},
        {"mad.hi.u32",[](auto& o){
            return o[0]+" = (unsigned)(((unsigned long long)"+o[1]+"*(unsigned long long)"+o[2]+")>>32)+(unsigned)"+o[3]+";";
        }},
        {"mad.lo.s64",[](auto& o){
            return o[0]+" = (long long)"+o[1]+"*(long long)"+o[2]+"+(long long)"+o[3]+";";
        }},
        {"mad.hi.s64",[](auto& o){
            return o[0]+" = (long long)((__int128)(long long)"+o[1]+"*(__int128)(long long)"+o[2]+
                   ">>64)+(long long)"+o[3]+";";
        }},
        {"mad.hi.u64",[](auto& o){
            return o[0]+" = (unsigned long long)((__uint128_t)(unsigned long long)"+o[1]+
                   "*(__uint128_t)(unsigned long long)"+o[2]+
                   ">>64)+(unsigned long long)"+o[3]+";";
        }},
        // Wide-multiply without add (mul.hi.s32 / mul.hi.s64)
        {"mul.hi.s32",[](auto& o){
            return o[0]+" = (int)(((long long)"+o[1]+"*(long long)"+o[2]+")>>32);";
        }},
        {"mul.hi.s64",[](auto& o){
            return o[0]+" = (long long)((__int128)(long long)"+o[1]+"*(__int128)(long long)"+o[2]+">>64);";
        }},
        // div/rem 64-bit variants
        {"div.u64",[](auto& o){
            return o[0]+" = (unsigned long long)"+o[1]+"/(unsigned long long)"+o[2]+";";
        }},
        {"div.s64",[](auto& o){
            return o[0]+" = (long long)"+o[1]+"/(long long)"+o[2]+";";
        }},
        {"rem.u64",[](auto& o){
            return o[0]+" = (unsigned long long)"+o[1]+"%(unsigned long long)"+o[2]+";";
        }},
        {"rem.s64",[](auto& o){
            return o[0]+" = (long long)"+o[1]+"%(long long)"+o[2]+";";
        }},
        // Additional min/max signed variants
        {"min.s32",[](auto& o){ return o[0]+" = ((int)"+o[1]+"<(int)"+o[2]+"?(int)"+o[1]+":(int)"+o[2]+");"; }},
        {"max.s32",[](auto& o){ return o[0]+" = ((int)"+o[1]+">(int)"+o[2]+"?(int)"+o[1]+":(int)"+o[2]+");"; }},
        {"min.s64",[](auto& o){ return o[0]+" = ((long long)"+o[1]+"<(long long)"+o[2]+"?(long long)"+o[1]+":(long long)"+o[2]+");"; }},
        {"max.s64",[](auto& o){ return o[0]+" = ((long long)"+o[1]+">(long long)"+o[2]+"?(long long)"+o[1]+":(long long)"+o[2]+");"; }},
        {"min.u32",[](auto& o){ return o[0]+" = ((unsigned)"+o[1]+"<(unsigned)"+o[2]+"?(unsigned)"+o[1]+":(unsigned)"+o[2]+");"; }},
        {"max.u32",[](auto& o){ return o[0]+" = ((unsigned)"+o[1]+">(unsigned)"+o[2]+"?(unsigned)"+o[1]+":(unsigned)"+o[2]+");"; }},
        // abs for integer types
        {"abs.s32",[](auto& o){ return o[0]+" = ((int)"+o[1]+">=0?(int)"+o[1]+":-(int)"+o[1]+");"; }},
        {"abs.s64",[](auto& o){ return o[0]+" = ((long long)"+o[1]+">=0LL?(long long)"+o[1]+":-(long long)"+o[1]+");"; }},
    };
    return kMap;
}

// Split comma-separated operands respecting brackets
std::vector<std::string> splitOperands(const std::string& s) {
    std::vector<std::string> r;
    std::string cur;
    int depth = 0;
    for (char c : s) {
        if (c == '[' || c == '(' || c == '{') ++depth;
        else if (c == ']' || c == ')' || c == '}') --depth;
        if (c == ',' && depth == 0) { r.push_back(trim(cur)); cur.clear(); }
        else cur += c;
    }
    if (!cur.empty()) r.push_back(trim(cur));
    return r;
}

} // namespace compiler
} // namespace vgre
