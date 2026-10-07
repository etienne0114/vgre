/**
 * Test: FP8 PTX translation (mma.sync.aligned e4m3/e5m2, wgmma FP8, cvt FP8)
 *
 * Verifies that the VGRE PTX translator accepts and translates FP8 tensor-core
 * instructions without errors:
 *   - mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
 *   - mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32
 *   - wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3
 *   - cvt.rn.satfinite.e4m3x2.f32 and cvt.rn.satfinite.e5m2x2.f32
 *   - cvt.rn.f32.e4m3 and cvt.rn.f32.e5m2
 *
 * Each PTX module is loaded via cuModuleLoadData; a CUDA_SUCCESS return code
 * confirms that the translator handled the instructions.
 */

#include <iostream>
#include <cstring>

extern "C" {

typedef void*  CUmodule;
typedef void*  CUfunction;
typedef int    CUresult;

static constexpr CUresult CUDA_SUCCESS = 0;

CUresult cuInit(unsigned int flags);
CUresult cuModuleLoadData(CUmodule* module, const void* image);
CUresult cuModuleUnload(CUmodule hmod);

} // extern "C"

static int g_pass = 0, g_total = 0;

static bool check(const char* name, bool ok) {
    ++g_total;
    if (ok) {
        ++g_pass;
        std::cout << "PASS [" << g_total << "] " << name << "\n";
    } else {
        std::cerr << "FAIL [" << g_total << "] " << name << "\n";
    }
    return ok;
}

// Load a PTX string as a CUDA module; return CUDA_SUCCESS if loading succeeds.
static CUresult loadPtx(const char* ptx, CUmodule* mod) {
    return cuModuleLoadData(mod, ptx);
}

int main() {
    std::cout << "--- FP8 PTX Translation Test ---\n";

    CUresult r = cuInit(0);
    if (r != CUDA_SUCCESS) {
        std::cerr << "FAIL: cuInit returned " << r << "\n";
        return 1;
    }

    CUmodule mod = nullptr;

    // ── 1. mma.sync.aligned FP8 E4M3×E4M3→FP32 ──────────────────────────────
    // Operands: d[0..3], a[0..3](4 uint32), b[0..1](2 uint32), c[0..3]
    static const char kPtxMmaE4M3[] =
        ".version 8.0\n"
        ".target sm_89\n"
        ".address_size 64\n"
        ".visible .entry test_mma_e4m3(.param .u64 out_ptr) {\n"
        "    .reg .f32 %d<4>;\n"
        "    .reg .b32 %a<4>;\n"
        "    .reg .b32 %b<2>;\n"
        "    .reg .f32 %c<4>;\n"
        "    .reg .u64 %ptr;\n"
        "    ld.param.u64 %ptr, [out_ptr];\n"
        "    mov.b32 %a0, 0;\n"
        "    mov.b32 %a1, 0;\n"
        "    mov.b32 %a2, 0;\n"
        "    mov.b32 %a3, 0;\n"
        "    mov.b32 %b0, 0;\n"
        "    mov.b32 %b1, 0;\n"
        "    mov.f32 %c0, 0.0;\n"
        "    mov.f32 %c1, 0.0;\n"
        "    mov.f32 %c2, 0.0;\n"
        "    mov.f32 %c3, 0.0;\n"
        "    mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32\n"
        "        {%d0,%d1,%d2,%d3},\n"
        "        {%a0,%a1,%a2,%a3},\n"
        "        {%b0,%b1},\n"
        "        {%c0,%c1,%c2,%c3};\n"
        "    st.global.f32 [%ptr], %d0;\n"
        "    ret;\n"
        "}\n";

    r = loadPtx(kPtxMmaE4M3, &mod);
    check("cuModuleLoadData: mma.sync.aligned e4m3.e4m3.f32", r == CUDA_SUCCESS);
    if (mod) { cuModuleUnload(mod); mod = nullptr; }

    // ── 2. mma.sync.aligned FP8 E5M2×E5M2→FP32 ──────────────────────────────
    static const char kPtxMmaE5M2[] =
        ".version 8.0\n"
        ".target sm_89\n"
        ".address_size 64\n"
        ".visible .entry test_mma_e5m2(.param .u64 out_ptr) {\n"
        "    .reg .f32 %d<4>;\n"
        "    .reg .b32 %a<4>;\n"
        "    .reg .b32 %b<2>;\n"
        "    .reg .f32 %c<4>;\n"
        "    .reg .u64 %ptr;\n"
        "    ld.param.u64 %ptr, [out_ptr];\n"
        "    mov.b32 %a0, 0;  mov.b32 %a1, 0;\n"
        "    mov.b32 %a2, 0;  mov.b32 %a3, 0;\n"
        "    mov.b32 %b0, 0;  mov.b32 %b1, 0;\n"
        "    mov.f32 %c0, 0.0; mov.f32 %c1, 0.0;\n"
        "    mov.f32 %c2, 0.0; mov.f32 %c3, 0.0;\n"
        "    mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32\n"
        "        {%d0,%d1,%d2,%d3},\n"
        "        {%a0,%a1,%a2,%a3},\n"
        "        {%b0,%b1},\n"
        "        {%c0,%c1,%c2,%c3};\n"
        "    st.global.f32 [%ptr], %d0;\n"
        "    ret;\n"
        "}\n";

    r = loadPtx(kPtxMmaE5M2, &mod);
    check("cuModuleLoadData: mma.sync.aligned e5m2.e5m2.f32", r == CUDA_SUCCESS);
    if (mod) { cuModuleUnload(mod); mod = nullptr; }

    // ── BF16 mma.sync m16n8k16 and TF32 conversion + m16n8k8 ───────────────
    // These modules exercise the real PTX-to-C translation and helper signature
    // compilation, including b32 TF32 values produced by cvt.rn.tf32.f32.
    static const char kPtxBf16Tf32[] =
        ".version 8.0\n.target sm_90\n.address_size 64\n"
        ".visible .entry test_bf16_tf32(.param .u64 out_ptr) {\n"
        " .reg .u64 %ptr; .reg .b32 %ba<4>, %bb<2>, %ta<4>, %tb<2>;\n"
        " .reg .f32 %d<4>, %c<4>, %f<6>;\n"
        " ld.param.u64 %ptr, [out_ptr];\n"
        " mov.b32 %ba0, 0; mov.b32 %ba1, 0; mov.b32 %ba2, 0; mov.b32 %ba3, 0;\n"
        " mov.b32 %bb0, 0; mov.b32 %bb1, 0;\n"
        " mov.f32 %c0, 0.0; mov.f32 %c1, 0.0; mov.f32 %c2, 0.0; mov.f32 %c3, 0.0;\n"
        " mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32\n"
        "  {%d0,%d1,%d2,%d3}, {%ba0,%ba1,%ba2,%ba3}, {%bb0,%bb1}, {%c0,%c1,%c2,%c3};\n"
        " mov.f32 %f0, 1.0; mov.f32 %f1, 2.0; mov.f32 %f2, 3.0;\n"
        " mov.f32 %f3, 4.0; mov.f32 %f4, 5.0; mov.f32 %f5, 6.0;\n"
        " cvt.rn.tf32.f32 %ta0, %f0; cvt.rn.tf32.f32 %ta1, %f1;\n"
        " cvt.rn.tf32.f32 %ta2, %f2; cvt.rn.tf32.f32 %ta3, %f3;\n"
        " cvt.rn.tf32.f32 %tb0, %f4; cvt.rn.tf32.f32 %tb1, %f5;\n"
        " mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32\n"
        "  {%d0,%d1,%d2,%d3}, {%ta0,%ta1,%ta2,%ta3}, {%tb0,%tb1}, {%c0,%c1,%c2,%c3};\n"
        " st.global.f32 [%ptr], %d0; ret;\n}\n";
    r = loadPtx(kPtxBf16Tf32, &mod);
    check("cuModuleLoadData: BF16 MMA plus TF32 conversion and MMA", r == CUDA_SUCCESS);
    if (mod) { cuModuleUnload(mod); mod = nullptr; }

    // ── 3. mma.sync.aligned FP8 mixed E4M3×E5M2→FP32 ────────────────────────
    static const char kPtxMmaMixed[] =
        ".version 8.0\n"
        ".target sm_89\n"
        ".address_size 64\n"
        ".visible .entry test_mma_mixed(.param .u64 out_ptr) {\n"
        "    .reg .f32 %d<4>;\n"
        "    .reg .b32 %a<4>;\n"
        "    .reg .b32 %b<2>;\n"
        "    .reg .f32 %c<4>;\n"
        "    .reg .u64 %ptr;\n"
        "    ld.param.u64 %ptr, [out_ptr];\n"
        "    mov.b32 %a0, 0;  mov.b32 %a1, 0;\n"
        "    mov.b32 %a2, 0;  mov.b32 %a3, 0;\n"
        "    mov.b32 %b0, 0;  mov.b32 %b1, 0;\n"
        "    mov.f32 %c0, 0.0; mov.f32 %c1, 0.0;\n"
        "    mov.f32 %c2, 0.0; mov.f32 %c3, 0.0;\n"
        "    mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e5m2.f32\n"
        "        {%d0,%d1,%d2,%d3},\n"
        "        {%a0,%a1,%a2,%a3},\n"
        "        {%b0,%b1},\n"
        "        {%c0,%c1,%c2,%c3};\n"
        "    st.global.f32 [%ptr], %d0;\n"
        "    ret;\n"
        "}\n";

    r = loadPtx(kPtxMmaMixed, &mod);
    check("cuModuleLoadData: mma.sync.aligned e4m3.e5m2.f32", r == CUDA_SUCCESS);
    if (mod) { cuModuleUnload(mod); mod = nullptr; }

    // ── 4. wgmma.mma_async FP8 E4M3×E4M3→FP32 (m64n128k32) ─────────────────
    static const char kPtxWgmmaFp8[] =
        ".version 9.0\n"
        ".target sm_90\n"
        ".address_size 64\n"
        ".visible .entry test_wgmma_fp8(.param .u64 out_ptr) {\n"
        "    .reg .u64 %descA;\n"
        "    .reg .u64 %descB;\n"
        "    .reg .u64 %dptr;\n"
        "    .reg .u64 %out;\n"
        "    ld.param.u64 %out, [out_ptr];\n"
        "    mov.u64 %descA, 0;\n"
        "    mov.u64 %descB, 0;\n"
        "    mov.u64 %dptr, %out;\n"
        "    wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3\n"
        "        %descA, %descB, %dptr;\n"
        "    ret;\n"
        "}\n";

    r = loadPtx(kPtxWgmmaFp8, &mod);
    check("cuModuleLoadData: wgmma.mma_async m64n128k32.f32.e4m3.e4m3", r == CUDA_SUCCESS);
    if (mod) { cuModuleUnload(mod); mod = nullptr; }

    // ── 5. cvt.rn.satfinite.e4m3x2.f32 — pack two f32 → two FP8 E4M3 bytes ─
    static const char kPtxCvtPack[] =
        ".version 8.0\n"
        ".target sm_89\n"
        ".address_size 64\n"
        ".visible .entry test_cvt_pack(.param .u64 out_ptr) {\n"
        "    .reg .b32 %packed;\n"
        "    .reg .f32 %a;\n"
        "    .reg .f32 %b;\n"
        "    .reg .u64 %ptr;\n"
        "    ld.param.u64 %ptr, [out_ptr];\n"
        "    mov.f32 %a, 1.0;\n"
        "    mov.f32 %b, 2.0;\n"
        "    cvt.rn.satfinite.e4m3x2.f32 %packed, %b, %a;\n"
        "    st.global.u32 [%ptr], %packed;\n"
        "    ret;\n"
        "}\n";

    r = loadPtx(kPtxCvtPack, &mod);
    check("cuModuleLoadData: cvt.rn.satfinite.e4m3x2.f32", r == CUDA_SUCCESS);
    if (mod) { cuModuleUnload(mod); mod = nullptr; }

    // ── 6. cvt.rn.satfinite.e5m2x2.f32 ──────────────────────────────────────
    static const char kPtxCvtPackE5M2[] =
        ".version 8.0\n"
        ".target sm_89\n"
        ".address_size 64\n"
        ".visible .entry test_cvt_pack_e5m2(.param .u64 out_ptr) {\n"
        "    .reg .b32 %packed;\n"
        "    .reg .f32 %a;\n"
        "    .reg .f32 %b;\n"
        "    .reg .u64 %ptr;\n"
        "    ld.param.u64 %ptr, [out_ptr];\n"
        "    mov.f32 %a, 3.0;\n"
        "    mov.f32 %b, 4.0;\n"
        "    cvt.rn.satfinite.e5m2x2.f32 %packed, %b, %a;\n"
        "    st.global.u32 [%ptr], %packed;\n"
        "    ret;\n"
        "}\n";

    r = loadPtx(kPtxCvtPackE5M2, &mod);
    check("cuModuleLoadData: cvt.rn.satfinite.e5m2x2.f32", r == CUDA_SUCCESS);
    if (mod) { cuModuleUnload(mod); mod = nullptr; }

    // ── 7. cvt.rn.f32.e4m3 — scalar FP8 E4M3 → f32 ─────────────────────────
    static const char kPtxCvtScalar[] =
        ".version 8.0\n"
        ".target sm_89\n"
        ".address_size 64\n"
        ".visible .entry test_cvt_scalar(.param .u64 out_ptr) {\n"
        "    .reg .f32 %result;\n"
        "    .reg .b32 %fp8val;\n"
        "    .reg .u64 %ptr;\n"
        "    ld.param.u64 %ptr, [out_ptr];\n"
        "    mov.b32 %fp8val, 60;\n"  // 60 = 0x3C = some E4M3 value
        "    cvt.rn.f32.e4m3 %result, %fp8val;\n"
        "    st.global.f32 [%ptr], %result;\n"
        "    ret;\n"
        "}\n";

    r = loadPtx(kPtxCvtScalar, &mod);
    check("cuModuleLoadData: cvt.rn.f32.e4m3", r == CUDA_SUCCESS);
    if (mod) { cuModuleUnload(mod); mod = nullptr; }

    // ── 8. cvt.rn.f32.e5m2 — scalar FP8 E5M2 → f32 ─────────────────────────
    static const char kPtxCvtE5M2[] =
        ".version 8.0\n"
        ".target sm_89\n"
        ".address_size 64\n"
        ".visible .entry test_cvt_e5m2(.param .u64 out_ptr) {\n"
        "    .reg .f32 %result;\n"
        "    .reg .b32 %fp8val;\n"
        "    .reg .u64 %ptr;\n"
        "    ld.param.u64 %ptr, [out_ptr];\n"
        "    mov.b32 %fp8val, 48;\n"  // 0x30 = some E5M2 value
        "    cvt.rn.f32.e5m2 %result, %fp8val;\n"
        "    st.global.f32 [%ptr], %result;\n"
        "    ret;\n"
        "}\n";

    r = loadPtx(kPtxCvtE5M2, &mod);
    check("cuModuleLoadData: cvt.rn.f32.e5m2", r == CUDA_SUCCESS);
    if (mod) { cuModuleUnload(mod); mod = nullptr; }

    // These low-level shapes use one A and one B register per lane, with two
    // accumulator registers. Loading each module compiles the generated helper
    // call and catches PTX operand-count or C++ signature mismatches.
    static const char kPtxM8S4[] =
        ".version 8.0\n.target sm_75\n.address_size 64\n"
        ".visible .entry test_m8_s4(.param .u64 out_ptr) {\n"
        " .reg .u64 %ptr; .reg .b32 %a, %b; .reg .s32 %d<2>, %c<2>;\n"
        " ld.param.u64 %ptr, [out_ptr];\n"
        " mov.b32 %a, 0x88888888; mov.b32 %b, 0x77777777;\n"
        " mov.s32 %c0, 0; mov.s32 %c1, 0;\n"
        " mma.sync.aligned.m8n8k32.row.col.satfinite.s32.s4.s4.s32 "
        "{%d0,%d1}, {%a}, {%b}, {%c0,%c1};\n"
        " st.global.s32 [%ptr], %d0; ret;\n}\n";
    r = loadPtx(kPtxM8S4, &mod);
    check("cuModuleLoadData: full-fragment m8n8k32 signed INT4 MMA", r == CUDA_SUCCESS);
    if (mod) { cuModuleUnload(mod); mod = nullptr; }

    static const char kPtxM8U4[] =
        ".version 8.0\n.target sm_75\n.address_size 64\n"
        ".visible .entry test_m8_u4(.param .u64 out_ptr) {\n"
        " .reg .u64 %ptr; .reg .b32 %a, %b; .reg .s32 %d<2>, %c<2>;\n"
        " ld.param.u64 %ptr, [out_ptr];\n"
        " mov.b32 %a, 0xffffffff; mov.b32 %b, 0xffffffff;\n"
        " mov.s32 %c0, 0; mov.s32 %c1, 0;\n"
        " mma.sync.aligned.m8n8k32.row.col.satfinite.s32.u4.u4.s32 "
        "{%d0,%d1}, {%a}, {%b}, {%c0,%c1};\n"
        " st.global.s32 [%ptr], %d0; ret;\n}\n";
    r = loadPtx(kPtxM8U4, &mod);
    check("cuModuleLoadData: full-fragment m8n8k32 unsigned INT4 MMA", r == CUDA_SUCCESS);
    if (mod) { cuModuleUnload(mod); mod = nullptr; }

    static const char kPtxM8B1And[] =
        ".version 8.0\n.target sm_75\n.address_size 64\n"
        ".visible .entry test_m8_b1_and(.param .u64 out_ptr) {\n"
        " .reg .u64 %ptr; .reg .b32 %a, %b; .reg .s32 %d<2>, %c<2>;\n"
        " ld.param.u64 %ptr, [out_ptr];\n"
        " mov.b32 %a, 0xffffffff; mov.b32 %b, 0x55555555;\n"
        " mov.s32 %c0, 0; mov.s32 %c1, 0;\n"
        " mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.and.popc "
        "{%d0,%d1}, {%a}, {%b}, {%c0,%c1};\n"
        " st.global.s32 [%ptr], %d0; ret;\n}\n";
    r = loadPtx(kPtxM8B1And, &mod);
    check("cuModuleLoadData: full-fragment m8n8k128 AND+POPC MMA", r == CUDA_SUCCESS);
    if (mod) { cuModuleUnload(mod); mod = nullptr; }

    static const char kPtxM8B1Xor[] =
        ".version 8.0\n.target sm_75\n.address_size 64\n"
        ".visible .entry test_m8_b1_xor(.param .u64 out_ptr) {\n"
        " .reg .u64 %ptr; .reg .b32 %a, %b; .reg .s32 %d<2>, %c<2>;\n"
        " ld.param.u64 %ptr, [out_ptr];\n"
        " mov.b32 %a, 0xffffffff; mov.b32 %b, 0x55555555;\n"
        " mov.s32 %c0, 0; mov.s32 %c1, 0;\n"
        " mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.xor.popc "
        "{%d0,%d1}, {%a}, {%b}, {%c0,%c1};\n"
        " st.global.s32 [%ptr], %d0; ret;\n}\n";
    r = loadPtx(kPtxM8B1Xor, &mod);
    check("cuModuleLoadData: full-fragment m8n8k128 XOR+POPC MMA", r == CUDA_SUCCESS);
    if (mod) { cuModuleUnload(mod); mod = nullptr; }

    static const char kPtxM8F64[] =
        ".version 8.0\n.target sm_70\n.address_size 64\n"
        ".visible .entry test_m8_f64(.param .u64 out_ptr) {\n"
        " .reg .u64 %ptr; .reg .f64 %a, %b, %d<2>, %c<2>;\n"
        " ld.param.u64 %ptr, [out_ptr];\n"
        " mov.f64 %a, 2.0; mov.f64 %b, 3.0;\n"
        " mov.f64 %c0, 1.0; mov.f64 %c1, 1.0;\n"
        " mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
        "{%d0,%d1}, {%a}, {%b}, {%c0,%c1};\n"
        " st.global.f64 [%ptr], %d0; ret;\n}\n";
    r = loadPtx(kPtxM8F64, &mod);
    check("cuModuleLoadData: full-fragment m8n8k4 FP64 MMA", r == CUDA_SUCCESS);
    if (mod) { cuModuleUnload(mod); mod = nullptr; }

    // ── Summary ───────────────────────────────────────────────────────────────
    std::cout << "\n--- FP8 PTX Translation: " << g_pass << "/" << g_total << " passed ---\n";
    if (g_pass != g_total) {
        std::cerr << "FAIL: " << (g_total - g_pass) << " test(s) failed\n";
        return 1;
    }
    std::cout << "[PASS] All FP8 PTX translation tests passed!\n";
    return 0;
}
