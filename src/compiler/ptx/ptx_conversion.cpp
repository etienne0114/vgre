// PTX conversion, precision, and cooperative-group instruction translation map.

#include "ptx_translator_internal.h"

namespace vgre {
namespace compiler {

// ── Blackwell tcgen05 emitters (P3-7) ────────────────────────────────────────
// tcgen05.mma [d-tmem], a-desc, b-desc [, idesc, enable-input-d] accumulates
// D = A·B (+D when enable-input-d) into the TMEM address d-tmem. After
// splitOperands: o[0]=d-tmem addr, o[1]=descA, o[2]=descB, o[4]=enable (if any).
static std::string tcgen05_mma_emit(const std::vector<std::string>& o,
                                    int M, int N, int K, int kind) {
    const std::string acc = o.size() > 4 ? o[4] : std::string("1");  // enable_input_d
    return "vgre_jit_tcgen05_mma((uint32_t)(" + (o.size() > 0 ? o[0] : std::string("0")) +
           "),(uint64_t)(" + (o.size() > 1 ? o[1] : std::string("0")) +
           "),(uint64_t)(" + (o.size() > 2 ? o[2] : std::string("0")) + ")," +
           std::to_string(M) + "," + std::to_string(N) + "," + std::to_string(K) + "," +
           std::to_string(kind) + ",(int)(" + acc + "));";
}

// tcgen05.ld.sync.aligned.<shape>.xN.b32 {r0..r_{N-1}}, [taddr]  — load N words
// from TMEM into the register list. o[0]={regs}, o[1]=taddr (brackets stripped).
static std::string tcgen05_ld_emit(const std::vector<std::string>& o, int n) {
    if (o.size() < 2) return "/* tcgen05.ld: bad operands */";
    auto regs = splitOperands(o[0].size() >= 2 && o[0].front() == '{'
                              ? o[0].substr(1, o[0].size() - 2) : o[0]);
    std::string out = "{ uint32_t _tl[" + std::to_string(n) + "]; "
                      "vgre_jit_tcgen05_ld(_tl,(uint32_t)(" + o[1] + "),1," + std::to_string(n) + ");";
    for (int i = 0; i < n && i < static_cast<int>(regs.size()); ++i)
        out += " " + regs[i] + " = _tl[" + std::to_string(i) + "];";
    return out + " }";
}

// tcgen05.st.sync.aligned.<shape>.xN.b32 [taddr], {r0..r_{N-1}}  — store N words
// from the register list into TMEM. o[0]=taddr, o[1]={regs}.
static std::string tcgen05_st_emit(const std::vector<std::string>& o, int n) {
    if (o.size() < 2) return "/* tcgen05.st: bad operands */";
    auto regs = splitOperands(o[1].size() >= 2 && o[1].front() == '{'
                              ? o[1].substr(1, o[1].size() - 2) : o[1]);
    std::string out = "{ uint32_t _ts[" + std::to_string(n) + "] = {";
    for (int i = 0; i < n; ++i) { out += (i < (int)regs.size() ? regs[i] : "0"); if (i + 1 < n) out += ","; }
    out += "}; vgre_jit_tcgen05_st((uint32_t)(" + o[0] + "),_ts,1," + std::to_string(n) + "); }";
    return out;
}

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
        // ── TMA 3D / 4D / 5D bulk copy ─────────────────────────────────────
        // Tile dimensions come from desc->boxDim (set by cuTensorMapEncodeTiled),
        // NOT from PTX instruction operands.  Use the _b variants.
        {"cp.async.bulk.tensor.3d.global.shared::cta.bulk_group",
            [](auto& o){ return tma_store_emit(o, 3); }},
        {"cp.async.bulk.tensor.4d.global.shared::cta.bulk_group",
            [](auto& o){ return tma_store_emit(o, 4); }},
        {"cp.async.bulk.tensor.5d.global.shared::cta.bulk_group",
            [](auto& o){ return tma_store_emit(o, 5); }},
        // ── cp.reduce.async (shared → global atomic reduction) ─────────────
        {"cp.reduce.async.add.f32", [](auto& o){
            return "vgre_cp_reduce_async_add_f32((float*)("+o[0]+"),(const float*)("+o[1]+
                   "),"+(o.size()>2?o[2]:std::string("1"))+");";
        }},
        {"cp.reduce.async.add.f64", [](auto& o){
            return "vgre_cp_reduce_async_add_f64((double*)("+o[0]+"),(const double*)("+o[1]+
                   "),"+(o.size()>2?o[2]:std::string("1"))+");";
        }},
        {"cp.reduce.async.min.f32", [](auto& o){
            return "vgre_cp_reduce_async_min_f32((float*)("+o[0]+"),(const float*)("+o[1]+
                   "),"+(o.size()>2?o[2]:std::string("1"))+");";
        }},
        {"cp.reduce.async.max.f32", [](auto& o){
            return "vgre_cp_reduce_async_max_f32((float*)("+o[0]+"),(const float*)("+o[1]+
                   "),"+(o.size()>2?o[2]:std::string("1"))+");";
        }},
        // ── tcgen05.mma (Blackwell SM100) — accumulate into a TMEM address ─────
        // o[0] is the TMEM accumulator address (from tcgen05.alloc), not a raw
        // pointer; the result is read back with tcgen05.ld. kind: 0=bf16 1=f16
        // 2=tf32 3=e4m3 4=e5m2 5=e4m3e5m2.
        {"tcgen05.mma.cta_group::1.m64n256k16.f32.bf16.bf16",
            [](auto& o){ return tcgen05_mma_emit(o, 64, 256, 16, 0); }},
        {"tcgen05.mma.cta_group::1.m64n128k16.f32.bf16.bf16",
            [](auto& o){ return tcgen05_mma_emit(o, 64, 128, 16, 0); }},
        {"tcgen05.mma.cta_group::1.m64n256k16.f32.f16.f16",
            [](auto& o){ return tcgen05_mma_emit(o, 64, 256, 16, 1); }},
        {"tcgen05.mma.cta_group::1.m64n128k16.f32.f16.f16",
            [](auto& o){ return tcgen05_mma_emit(o, 64, 128, 16, 1); }},
        {"tcgen05.mma.cta_group::1.m64n256k8.f32.tf32.tf32",
            [](auto& o){ return tcgen05_mma_emit(o, 64, 256, 8, 2); }},
        {"tcgen05.mma.cta_group::1.m128n256k16.f32.bf16.bf16",
            [](auto& o){ return tcgen05_mma_emit(o, 128, 256, 16, 0); }},
        // FP8 (E4M3 / E5M2, K=32) tcgen05.mma into TMEM.
        {"tcgen05.mma.cta_group::1.m64n256k32.f32.e4m3.e4m3",
            [](auto& o){ return tcgen05_mma_emit(o, 64, 256, 32, 3); }},
        {"tcgen05.mma.cta_group::1.m64n128k32.f32.e4m3.e4m3",
            [](auto& o){ return tcgen05_mma_emit(o, 64, 128, 32, 3); }},
        {"tcgen05.mma.cta_group::1.m64n256k32.f32.e5m2.e5m2",
            [](auto& o){ return tcgen05_mma_emit(o, 64, 256, 32, 4); }},
        // ── tcgen05 TMEM data path: alloc / dealloc / ld / st / cp / sync ──────
        // tcgen05.alloc writes the allocated TMEM address into shared memory.
        {"tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32", [](auto& o){
            return "*(uint32_t*)(" + (o.size()>0?o[0]:std::string("0")) +
                   ") = vgre_jit_tmem_alloc((int)(" + (o.size()>1?o[1]:std::string("0")) + "));";
        }},
        {"tcgen05.dealloc.cta_group::1.sync.aligned.b32", [](auto& o){
            return "vgre_jit_tmem_dealloc((uint32_t)(" + (o.size()>0?o[0]:std::string("0")) +
                   "),(int)(" + (o.size()>1?o[1]:std::string("0")) + "));";
        }},
        {"tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned",
            [](auto&){ return "vgre_jit_tmem_relinquish();"; }},
        {"tcgen05.ld.sync.aligned.32x32b.x1.b32",  [](auto& o){ return tcgen05_ld_emit(o, 1); }},
        {"tcgen05.ld.sync.aligned.32x32b.x2.b32",  [](auto& o){ return tcgen05_ld_emit(o, 2); }},
        {"tcgen05.ld.sync.aligned.32x32b.x4.b32",  [](auto& o){ return tcgen05_ld_emit(o, 4); }},
        {"tcgen05.ld.sync.aligned.32x32b.x8.b32",  [](auto& o){ return tcgen05_ld_emit(o, 8); }},
        {"tcgen05.st.sync.aligned.32x32b.x1.b32",  [](auto& o){ return tcgen05_st_emit(o, 1); }},
        {"tcgen05.st.sync.aligned.32x32b.x2.b32",  [](auto& o){ return tcgen05_st_emit(o, 2); }},
        {"tcgen05.st.sync.aligned.32x32b.x4.b32",  [](auto& o){ return tcgen05_st_emit(o, 4); }},
        {"tcgen05.st.sync.aligned.32x32b.x8.b32",  [](auto& o){ return tcgen05_st_emit(o, 8); }},
        // Commit / wait / fence — synchronous on CPU (the MMA already completed).
        {"tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64", [](auto&){ return "/* tcgen05.commit serial */"; }},
        {"tcgen05.wait::ld.sync.aligned",  [](auto&){ return "/* tcgen05.wait::ld serial */"; }},
        {"tcgen05.wait::st.sync.aligned",  [](auto&){ return "/* tcgen05.wait::st serial */"; }},
        {"tcgen05.fence::before_thread_sync", [](auto&){ return "__atomic_thread_fence(__ATOMIC_SEQ_CST);"; }},
        {"tcgen05.fence::after_thread_sync",  [](auto&){ return "__atomic_thread_fence(__ATOMIC_SEQ_CST);"; }},
        // ── mbarrier (Hopper SM90 async pipeline synchronization) ──────────────
        // CPU serial emulation: all async copies complete synchronously, so
        // barriers are trivially satisfied.  init=noop, arrive=complete token,
        // test_wait/try_wait=1 (always done), wait=noop, inval=noop.
        {"mbarrier.init.shared.b64",       [](auto&){ return "/* mbarrier.init serial */"; }},
        {"mbarrier.init.shared::cta.b64",  [](auto&){ return "/* mbarrier.init serial */"; }},
        {"mbarrier.arrive.shared.b64", [](auto& o){
            // token = arrive(); in serial, token=0 (phase 0 immediately complete)
            return (o.size()>0 ? o[0]+" = 0ull; " : "")+"/* mbarrier.arrive serial */";
        }},
        {"mbarrier.arrive.shared::cta.b64", [](auto& o){
            return (o.size()>0 ? o[0]+" = 0ull; " : "")+"/* mbarrier.arrive serial */";
        }},
        {"mbarrier.arrive.noComplete.shared.b64", [](auto& o){
            return (o.size()>0 ? o[0]+" = 0ull; " : "")+"/* mbarrier.arrive.noComplete serial */";
        }},
        {"mbarrier.arrive.noComplete.shared::cta.b64", [](auto& o){
            return (o.size()>0 ? o[0]+" = 0ull; " : "")+"/* mbarrier.arrive.noComplete serial */";
        }},
        {"mbarrier.test_wait.shared.b64", [](auto& o){
            return (o.size()>0 ? o[0]+" = 1; " : "")+"/* mbarrier.test_wait always done in serial */";
        }},
        {"mbarrier.test_wait.shared::cta.b64", [](auto& o){
            return (o.size()>0 ? o[0]+" = 1; " : "")+"/* mbarrier.test_wait always done */";
        }},
        {"mbarrier.try_wait.shared.b64", [](auto& o){
            return (o.size()>0 ? o[0]+" = 1; " : "")+"/* mbarrier.try_wait always done in serial */";
        }},
        {"mbarrier.try_wait.shared::cta.b64", [](auto& o){
            return (o.size()>0 ? o[0]+" = 1; " : "")+"/* mbarrier.try_wait always done */";
        }},
        {"mbarrier.try_wait.parity.shared.b64", [](auto& o){
            return (o.size()>0 ? o[0]+" = 1; " : "")+"/* mbarrier.try_wait.parity always done */";
        }},
        {"mbarrier.try_wait.parity.shared::cta.b64", [](auto& o){
            return (o.size()>0 ? o[0]+" = 1; " : "")+"/* mbarrier.try_wait.parity always done */";
        }},
        {"mbarrier.wait.shared.b64",        [](auto&){ return "/* mbarrier.wait serial noop */"; }},
        {"mbarrier.wait.shared::cta.b64",   [](auto&){ return "/* mbarrier.wait serial noop */"; }},
        {"mbarrier.wait.parity.shared.b64", [](auto&){ return "/* mbarrier.wait.parity serial noop */"; }},
        {"mbarrier.wait.parity.shared::cta.b64",[](auto&){ return "/* mbarrier.wait.parity noop */"; }},
        {"mbarrier.inval.shared.b64",       [](auto&){ return "/* mbarrier.inval serial */"; }},
        {"mbarrier.inval.shared::cta.b64",  [](auto&){ return "/* mbarrier.inval serial */"; }},
        {"mbarrier.complete_tx.shared.b64", [](auto&){
            // In serial CPU model all async copies complete synchronously: noop.
            return "/* mbarrier.complete_tx serial (copy already synchronous) */";
        }},
        {"mbarrier.complete_tx.shared::cta.b64", [](auto&){
            return "/* mbarrier.complete_tx serial */";
        }},
        // ── fence.proxy (Hopper async-copy memory ordering) ────────────────────
        // On CPU, all operations are already sequentially consistent.
        {"fence.proxy.async",                          [](auto&){ return "__atomic_thread_fence(__ATOMIC_SEQ_CST);"; }},
        {"fence.proxy.async.shared::cta",              [](auto&){ return "__atomic_thread_fence(__ATOMIC_SEQ_CST);"; }},
        {"fence.proxy.async.global",                   [](auto&){ return "__atomic_thread_fence(__ATOMIC_SEQ_CST);"; }},
        {"fence.proxy.tensormap::generic.acquire.gpu", [](auto&){ return "/* fence.proxy.tensormap acquire */"; }},
        {"fence.proxy.tensormap::generic.release.cta", [](auto&){ return "/* fence.proxy.tensormap release */"; }},
        {"fence.proxy.tensormap::generic.acquire.cta", [](auto&){ return "/* fence.proxy.tensormap acquire.cta */"; }},
        {"fence.proxy.tensormap::generic.release.gpu", [](auto&){ return "/* fence.proxy.tensormap release.gpu */"; }},
        // ── tensormap.replace (update TMA descriptor address at runtime) ───────
        // The TMA descriptor is laid out as VgreTMADescriptor; baseAddr is at offset 0.
        {"tensormap.replace.tile.global_address.shared::cta.b1024.b64", [](auto& o){
            return "{VgreTMADescriptor* _td=(VgreTMADescriptor*)(uintptr_t)("+
                   (o.size()>0?o[0]:std::string("0"))+
                   "); if(_td) _td->baseAddr=(void*)(uintptr_t)("+
                   (o.size()>1?o[1]:std::string("0"))+"); } /* tensormap.replace.tile.global_address */";
        }},
        // NOTE: the 1d/2d cp.async.bulk.tensor STORE opcodes are handled in
        // getMap() (via tma_store_emit) — getMap is searched first, so duplicating
        // them here would be dead code. Only the 3d/4d/5d store forms live above.
        // ── Prefetch / cache hint variants (no-ops on CPU) ─────────────────────
        {"cp.async.bulk.prefetch.tensor.2d.l2.global",  [](auto&){ return "/* cp.async.bulk.prefetch 2D */"; }},
        {"cp.async.bulk.prefetch.tensor.3d.l2.global",  [](auto&){ return "/* cp.async.bulk.prefetch 3D */"; }},
        {"cp.async.bulk.prefetch.tensor.4d.l2.global",  [](auto&){ return "/* cp.async.bulk.prefetch 4D */"; }},
        {"cp.async.bulk.prefetch.tensor.5d.l2.global",  [](auto&){ return "/* cp.async.bulk.prefetch 5D */"; }},

        // ── elect.sync — cooperative group leader election (SM90+) ─────────────
        // In serial CPU model there is exactly one thread per warp, so it is
        // always "elected".  Output predicate is always true; membermask ignored.
        {"elect.sync", [](auto& o){
            return (o.size()>0 ? o[0]+" = 1;" : "")
                   + " /* elect.sync: always elected in serial model */";
        }},

        // ── griddepcontrol — CDP2 grid dependency (CUDA 12.0+) ────────────────
        // All streams execute serially in VGRE; dependencies are always satisfied.
        {"griddepcontrol.launch_dependents", [](auto&){
            return "/* griddepcontrol.launch_dependents: serial noop */";
        }},
        {"griddepcontrol.wait", [](auto&){
            return "/* griddepcontrol.wait: serial noop */";
        }},
        {"griddepcontrol.wait_ifnot_lbi", [](auto&){
            return "/* griddepcontrol.wait_ifnot_lbi: serial noop */";
        }},

        // ── setmaxnreg — register-file reconfiguration (Ada SM89+) ───────────
        // No register file on CPU; ignored.
        {"setmaxnreg.inc.sync.aligned.u32", [](auto&){
            return "/* setmaxnreg.inc: no-op on CPU */";
        }},
        {"setmaxnreg.dec.sync.aligned.u32", [](auto&){
            return "/* setmaxnreg.dec: no-op on CPU */";
        }},

        // ── cp.reduce.async.bulk — write-combining reduction to global ─────────
        // On CPU emulator all memory is directly accessible; perform the add
        // reduction immediately (no async needed).
        {"cp.reduce.async.bulk.tensor.1d.global.shared::cta.add.f32", [](auto& o){
            if (o.size() < 3) return std::string("/* cp.reduce.async.bulk.1d */");
            return "{ float* _dst=(float*)(uintptr_t)("+o[0]+");"
                   " const float* _src=(const float*)(uintptr_t)("+o[1]+");"
                   " size_t _n=(size_t)("+o[2]+")>>2;"
                   " for(size_t _i=0;_i<_n;_i++) _dst[_i]+=_src[_i]; }";
        }},
        {"cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.f32", [](auto& o){
            if (o.size() < 3) return std::string("/* cp.reduce.async.bulk.2d */");
            return "{ float* _dst=(float*)(uintptr_t)("+o[0]+");"
                   " const float* _src=(const float*)(uintptr_t)("+o[1]+");"
                   " size_t _n=(size_t)("+o[2]+")>>2;"
                   " for(size_t _i=0;_i<_n;_i++) _dst[_i]+=_src[_i]; }";
        }},
        // Integer accumulation variant
        {"cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.u32", [](auto& o){
            if (o.size() < 3) return std::string("/* cp.reduce.async.bulk.u32 */");
            return "{ uint32_t* _dst=(uint32_t*)(uintptr_t)("+o[0]+");"
                   " const uint32_t* _src=(const uint32_t*)(uintptr_t)("+o[1]+");"
                   " size_t _n=(size_t)("+o[2]+")>>2;"
                   " for(size_t _i=0;_i<_n;_i++) _dst[_i]+=_src[_i]; }";
        }},

        // ── FP8 cvt — scalar and packed E4M3/E5M2 conversions ─────────────────
        // Scalar: FP8 → f32 (byte in low 8 bits of source register)
        {"cvt.rn.f32.e4m3", [](auto& o){
            return o[0]+" = vgre_fp8e4m3_to_f32((uint8_t)((unsigned)("+o[1]+")&0xFFu));";
        }},
        {"cvt.rn.f32.e5m2", [](auto& o){
            return o[0]+" = vgre_fp8e5m2_to_f32((uint8_t)((unsigned)("+o[1]+")&0xFFu));";
        }},
        // Scalar: f32 → FP8 (result in low 8 bits of destination register)
        {"cvt.rn.e4m3.f32", [](auto& o){
            return o[0]+" = (unsigned)vgre_f32_to_fp8e4m3((float)("+o[1]+"));";
        }},
        {"cvt.rn.e5m2.f32", [](auto& o){
            return o[0]+" = (unsigned)vgre_f32_to_fp8e5m2((float)("+o[1]+"));";
        }},
        // Saturating variants (same semantics; satfinite clamps to FP8 range, same as our impl)
        {"cvt.rn.satfinite.e4m3.f32", [](auto& o){
            return o[0]+" = (unsigned)vgre_f32_to_fp8e4m3((float)("+o[1]+"));";
        }},
        {"cvt.rn.satfinite.e5m2.f32", [](auto& o){
            return o[0]+" = (unsigned)vgre_f32_to_fp8e5m2((float)("+o[1]+"));";
        }},
        // Packed: f32×2 → e4m3x2 (two FP8 bytes packed into one uint32)
        // PTX syntax: cvt.rn.satfinite.e4m3x2.f32 d, b, a
        //   d[7:0]  = f32_to_e4m3(a)   (first source = low byte)
        //   d[15:8] = f32_to_e4m3(b)   (second source = high byte)
        {"cvt.rn.satfinite.e4m3x2.f32", [](auto& o){
            auto a = o.size() > 2 ? o[2] : o[1];
            auto b = o.size() > 1 ? o[1] : o[1];
            return o[0]+" = (unsigned)vgre_f32_to_fp8e4m3((float)("+a+"))"
                   " | ((unsigned)vgre_f32_to_fp8e4m3((float)("+b+")))<<8;";
        }},
        {"cvt.rn.satfinite.e5m2x2.f32", [](auto& o){
            auto a = o.size() > 2 ? o[2] : o[1];
            auto b = o.size() > 1 ? o[1] : o[1];
            return o[0]+" = (unsigned)vgre_f32_to_fp8e5m2((float)("+a+"))"
                   " | ((unsigned)vgre_f32_to_fp8e5m2((float)("+b+")))<<8;";
        }},
        // Packed: e4m3x2 → f32×2 (unpack to two float variables)
        // PTX syntax: cvt.rn.f32x2.e4m3x2 {fa, fb}, d
        //   fa = e4m3_to_f32(d[7:0]), fb = e4m3_to_f32(d[15:8])
        {"cvt.rn.f32x2.e4m3x2", [](auto& o){
            if (o.size() < 3) return std::string("/* cvt.rn.f32x2.e4m3x2 */");
            return o[0]+" = vgre_fp8e4m3_to_f32((uint8_t)((unsigned)("+o[2]+")&0xFFu));"
                   " "+o[1]+" = vgre_fp8e4m3_to_f32((uint8_t)(((unsigned)("+o[2]+")>>8)&0xFFu));";
        }},
        {"cvt.rn.f32x2.e5m2x2", [](auto& o){
            if (o.size() < 3) return std::string("/* cvt.rn.f32x2.e5m2x2 */");
            return o[0]+" = vgre_fp8e5m2_to_f32((uint8_t)((unsigned)("+o[2]+")&0xFFu));"
                   " "+o[1]+" = vgre_fp8e5m2_to_f32((uint8_t)(((unsigned)("+o[2]+")>>8)&0xFFu));";
        }},
    };
    return kMap;
}

} // namespace compiler
} // namespace vgre
