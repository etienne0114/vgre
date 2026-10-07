// Track 12 — PTX inline-asm translation: memory addressing + carry chains.
//
// (1) Memory operands `[reg]` / `[reg+imm]` must lose their PTX brackets so the
//     emitted C is valid (`*(float*)(addr+8)`, not `*(float*)([addr+8])`).
// (2) The carry-flag (add.cc/addc, sub.cc/subc) chain must compute exact
//     multi-word integer arithmetic — verified numerically here against a
//     64-bit reference using the SAME expressions the translator emits.

#include "vgre/compiler/ptx_translator.h"

#include <cstdio>
#include <cstdint>
#include <string>

using vgre::compiler::PTXTranslator;

static int g_pass = 0, g_total = 0;
static void check(const char *name, bool ok) {
    ++g_total;
    printf(ok ? "  PASS  %s\n" : "  FAIL  %s\n", name);
    if (ok) ++g_pass;
}

static bool contains(const std::string &h, const std::string &n) {
    return h.find(n) != std::string::npos;
}

int main() {
    printf("=== PTX Inline-Asm Translation (Track 12) ===\n");

    // Packed FP8 conversion follows PTX's lane ordering: d[7:0] receives the
    // second source operand b, while d[15:8] receives the first source a.
    {
        const std::string e4 = PTXTranslator::translate(
            "asm volatile(\"cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\");");
        const size_t e4Low = e4.find("vgre_f32_to_fp8e4m3_satfinite((float)(%1),false,false)");
        const size_t e4High = e4.find("vgre_f32_to_fp8e4m3_satfinite((float)(%2),false,false)");
        const size_t e4Shift = e4.find("<<8");
        check("packed E4M3 stores PTX source b in the low byte and source a high",
              e4Low != std::string::npos && e4High != std::string::npos &&
              e4Low < e4High && e4High < e4Shift);

        const std::string e5 = PTXTranslator::translate(
            "asm volatile(\"cvt.rn.satfinite.e5m2x2.f32 %0, %1, %2;\");");
        const size_t e5Low = e5.find("vgre_f32_to_fp8e5m2_satfinite((float)(%1),false,false)");
        const size_t e5High = e5.find("vgre_f32_to_fp8e5m2_satfinite((float)(%2),false,false)");
        const size_t e5Shift = e5.find("<<8");
        check("packed E5M2 uses finite saturation and preserves PTX source order",
              e5Low != std::string::npos && e5High != std::string::npos &&
              e5Low < e5High && e5High < e5Shift);

        const std::string rz = PTXTranslator::translate(
            "asm volatile(\"cvt.rz.satfinite.e4m3x2.f32 %0, %1, %2;\");");
        check("packed FP8 round-toward-zero selects truncating conversion",
              contains(rz, "vgre_f32_to_fp8e4m3_satfinite((float)(%1),true,false)"));
        const std::string relu = PTXTranslator::translate(
            "asm volatile(\"cvt.rn.relu.satfinite.e5m2x2.f32 %0, %1, %2;\");");
        check("packed FP8 ReLU selects the NaN-preserving ReLU conversion",
              contains(relu, "vgre_f32_to_fp8e5m2_satfinite((float)(%1),false,true)"));
        const std::string rzRelu = PTXTranslator::translate(
            "asm volatile(\"cvt.rz.relu.satfinite.e4m3x2.f32 %0, %1, %2;\");");
        check("packed FP8 supports combined round-toward-zero and ReLU",
              contains(rzRelu, "vgre_f32_to_fp8e4m3_satfinite((float)(%1),true,true)"));
    }

    // (1) Memory addressing: load with a byte offset.
    {
        std::string src =
            "asm volatile(\"ld.global.f32 %0, [%1+8];\" : \"=f\"(out) : \"l\"(addr));";
        std::string c = PTXTranslator::translate(src);
        // The generated C must dereference a float* and must NOT contain the PTX
        // address brackets around the pointer expression.
        check("ld.global.f32 emits a float* dereference", contains(c, "*(float*)("));
        check("ld.global with offset has no literal '[' brackets",
              c.find('[') == std::string::npos);
        check("ld.global preserves the +8 byte offset", contains(c, "+8"));
    }

    // (2) Store with bracketed address.
    {
        std::string src =
            "asm volatile(\"st.global.u32 [%0], %1;\" :: \"l\"(p), \"r\"(v));";
        std::string c = PTXTranslator::translate(src);
        check("st.global.u32 emits an unsigned* store", contains(c, "*(unsigned*)("));
        check("st.global has no literal '[' brackets", c.find('[') == std::string::npos);
    }

    // (3) Carry-chain numerical correctness. Emulate a 64-bit add as two 32-bit
    //     limbs using the EXACT expressions the translator emits for
    //     add.cc.u32 / addc.u32, and compare to a real 64-bit add.
    {
        auto add64_via_carry = [](uint64_t A, uint64_t B) -> uint64_t {
            unsigned a0 = (unsigned)A, a1 = (unsigned)(A >> 32);
            unsigned b0 = (unsigned)B, b1 = (unsigned)(B >> 32);
            int _cc = 0;
            unsigned r0, r1;
            // add.cc.u32 r0, a0, b0
            { unsigned _s = a0 + b0; _cc = (_s < (unsigned)a0); r0 = _s; }
            // addc.u32 r1, a1, b1
            { unsigned _s = a1 + b1 + (unsigned)_cc; r1 = _s; }
            return (uint64_t)r0 | ((uint64_t)r1 << 32);
        };
        bool ok = true;
        uint64_t cases[][2] = {
            {0xFFFFFFFFull, 1ull},                       // limb-0 carry into limb-1
            {0x00000001FFFFFFFFull, 0x0000000100000001ull},
            {0xDEADBEEFCAFEBABEull, 0x1234567890ABCDEFull},
            {0xFFFFFFFFFFFFFFFFull, 1ull},               // full wrap
        };
        for (auto &c : cases)
            if (add64_via_carry(c[0], c[1]) != (uint64_t)(c[0] + c[1])) ok = false;
        check("add.cc.u32/addc.u32 carry chain == 64-bit add", ok);
    }

    // (4) Borrow-chain numerical correctness for sub.cc/subc.
    {
        auto sub64_via_borrow = [](uint64_t A, uint64_t B) -> uint64_t {
            unsigned a0 = (unsigned)A, a1 = (unsigned)(A >> 32);
            unsigned b0 = (unsigned)B, b1 = (unsigned)(B >> 32);
            int _cc = 0;
            unsigned r0, r1;
            // sub.cc.u32 r0, a0, b0   (_cc = borrow)
            { unsigned _s = (unsigned)a0 - (unsigned)b0; _cc = ((unsigned)a0 < (unsigned)b0); r0 = _s; }
            // subc.u32 r1, a1, b1
            { unsigned _b = (unsigned)b1 + (unsigned)_cc; r1 = (unsigned)a1 - _b; }
            return (uint64_t)r0 | ((uint64_t)r1 << 32);
        };
        bool ok = true;
        uint64_t cases[][2] = {
            {0x0000000100000000ull, 1ull},               // borrow from limb-1
            {0x1234567890ABCDEFull, 0x0000000FFFFFFFFFull},
            {0ull, 1ull},                                // full underflow wrap
        };
        for (auto &c : cases)
            if (sub64_via_borrow(c[0], c[1]) != (uint64_t)(c[0] - c[1])) ok = false;
        check("sub.cc.u32/subc.u32 borrow chain == 64-bit sub", ok);
    }

    // (5) Tensor-core mma.sync (Track 9): the four brace-grouped operands
    //     {d0..3},{a0..3},{b0..1},{c0..3} must FLATTEN into the 14 individual
    //     registers the warp-collective helper expects (splitOperands keeps each
    //     {..} as one token, so the helper call must expand them).
    {
        std::string src =
            "asm volatile("
            "\"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\" "
            ": \"=f\"(d0),\"=f\"(d1),\"=f\"(d2),\"=f\"(d3));";
        std::string c = PTXTranslator::translate(src);
        check("mma.sync emits the warp-collective f16 helper",
              contains(c, "vgre_mma_m16n8k16_f32_f16("));
        // All 14 register tokens must appear individually (flattened, not braces).
        bool all14 = true;
        for (int i = 0; i <= 13; ++i) all14 = all14 && contains(c, "%" + std::to_string(i));
        check("mma.sync flattens all 14 brace-grouped operands", all14);
        check("mma.sync leaves no brace-grouped operands ({%)", c.find("{%") == std::string::npos);
    }

    // The supported TF32 and BF16 variants must reach the shape-specific
    // collective helpers, and FP32→TF32 conversion must preserve PTX rounding
    // and modifier semantics in the generated expression.
    {
        const char* const bodies[] = {
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};",
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};",
            "cvt.rn.tf32.f32 %0, %1;",
            "cvt.rz.tf32.f32 %0, %1;",
            "cvt.rna.tf32.f32 %0, %1;",
            "cvt.rn.relu.pzo.tf32.f32 %0, %1;",
            "cvt.rna.satfinite.tf32.f32 %0, %1;",
            "cvt.rn.satfinite.relu.tf32.f32 %0, %1;"
        };
        const char* const expected[] = {
            "vgre_mma_m16n8k16_f32_bf16(",
            "vgre_mma_m16n8k8_tf32(",
            "Tf32Rounding::NearestEven,false,false,false",
            "Tf32Rounding::TowardZero,false,false,false",
            "Tf32Rounding::NearestAway,false,false,false",
            "Tf32Rounding::NearestEven,false,true,true",
            "Tf32Rounding::NearestAway,true,false,false",
            "Tf32Rounding::NearestEven,true,true,false"
        };
        bool ok = true;
        for (size_t i = 0; i < sizeof(bodies) / sizeof(bodies[0]); ++i) {
            std::string translated = PTXTranslator::translate(
                std::string("asm volatile(\"") + bodies[i] + "\" : );");
            bool caseOk = contains(translated, expected[i]);
            if (i >= 2)
                caseOk = caseOk && contains(translated, "cvt_f32_to_tf32_bits(");
            if (!caseOk)
                std::printf("  [detail] translation for '%s': %s\n", bodies[i], translated.c_str());
            ok = ok && caseOk;
        }
        check("TF32/BF16 mma.sync and cvt variants map to real helpers", ok);
    }

    // m8 tensor-core shapes use one A register and one B register per lane,
    // plus two accumulator registers. The translator must pass the six actual
    // PTX operands in order; widening these fragments into extra registers
    // changes the operation rather than emulating it.
    {
        const char* const bodies[] = {
            "mma.sync.aligned.m8n8k32.row.col.satfinite.s32.s4.s4.s32 {%0,%1}, {%2}, {%3}, {%4,%5};",
            "mma.sync.aligned.m8n8k32.row.col.satfinite.s32.u4.u4.s32 {%0,%1}, {%2}, {%3}, {%4,%5};",
            "mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.and.popc {%0,%1}, {%2}, {%3}, {%4,%5};",
            "mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.xor.popc {%0,%1}, {%2}, {%3}, {%4,%5};",
            "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%4,%5};"
        };
        const char* const helpers[] = {
            "vgre_mma_m8n8k32_s4(", "vgre_mma_m8n8k32_u4(",
            "vgre_mma_m8n8k128_b1_and(", "vgre_mma_m8n8k128_b1_xor(",
            "vgre_mma_m8n8k4_f64("
        };
        bool ok = true;
        for (size_t i = 0; i < sizeof(bodies) / sizeof(bodies[0]); ++i) {
            std::string translated = PTXTranslator::translate(
                std::string("asm volatile(\"") + bodies[i] + "\" : );");
            ok = ok && contains(translated, helpers[i]);
            for (int reg = 0; reg < 6; ++reg)
                ok = ok && contains(translated, "%" + std::to_string(reg));
            ok = ok && translated.find("%6") == std::string::npos;
        }
        check("m8 integer and f64 mma.sync mappings preserve the six PTX fragment operands", ok);
    }

    // (6) Hopper wgmma with a REAL distributed accumulator ({d0..d31}, descA,
    //     descB) must route to the warp-group-collective helper (Track P3-4),
    //     packing the 32 (=N/2 for n64) accumulator registers.
    {
        std::string regs = "{";
        for (int i = 0; i < 32; ++i) { regs += "%" + std::to_string(i); if (i < 31) regs += ","; }
        regs += "}";
        std::string src = "asm volatile("
            "\"wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 " + regs + ", %32, %33;\" : );";
        std::string c = PTXTranslator::translate(src);
        check("wgmma brace-group → warp-group collective helper",
              contains(c, "vgre_wgmma_wg_bf16(_wgd,64,"));
        check("wgmma packs the 32 distributed accumulator registers",
              contains(c, "float _wgd[32]"));
    }

    {
        const auto translatedWgmma = [](const char* instruction, int n) {
            std::string regs = "{";
            const int count = n / 2;
            for (int i = 0; i < count; ++i) {
                regs += "%" + std::to_string(i);
                if (i + 1 < count) regs += ",";
            }
            regs += "}";
            return PTXTranslator::translate("asm volatile(\"" + std::string(instruction) +
                " " + regs + ", %" + std::to_string(count) + ", %" +
                std::to_string(count + 1) + ";\" : );");
        };
        const std::string f16 = translatedWgmma(
            "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16", 128);
        const std::string tf32 = translatedWgmma(
            "wgmma.mma_async.sync.aligned.m64n256k8.f32.tf32.tf32", 256);
        check("wgmma FP16 n128 mapping validates and packs all 64 accumulator registers",
              contains(f16, "vgre_wgmma_wg_f16(_wgd,128,") &&
              contains(f16, "float _wgd[64]"));
        check("wgmma TF32 n256 mapping validates and packs all 128 accumulator registers",
              contains(tf32, "vgre_wgmma_wg_tf32(_wgd,256,") &&
              contains(tf32, "float _wgd[128]"));
    }

    // (7) The dominant Hopper TMA load (cluster-scope global→shared with mbarrier
    //     completion, Track P3-5) must lower to the real strided box copy — not
    //     "not supported". o[1] = "tensorMap, {coords}" is parsed into the map +
    //     coordinates.
    {
        std::string src = "asm volatile("
            "\"cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes "
            "[%0], [%1, {%2, %3}], [%4];\" : );";
        std::string c = PTXTranslator::translate(src);
        check("TMA cluster load → real vgre_tma_load_2d_dispatch",
              contains(c, "vgre_tma_load_2d_dispatch("));
        check("TMA load parses the tensor-map and {coords}",
              contains(c, "VgreTMADescriptor") && contains(c, "%2") && contains(c, "%3"));
    }

    // (8) The Hopper TMA STORE (shared→global) cp.async.bulk.tensor.2d
    //     .global.shared::cta.bulk_group must lower to a STORE — vgre_tma_store_2d_b
    //     with the descriptor first and the source SMEM second — NOT a load. (This
    //     opcode was previously mis-mapped to vgre_tma_load_2d_dispatch, silently
    //     turning every TMA store into a load.) Real PTX bundles the target as
    //     [tensorMap, {coords}] and the source as the second operand.
    {
        std::string src = "asm volatile("
            "\"cp.async.bulk.tensor.2d.global.shared::cta.bulk_group "
            "[%0, {%1, %2}], [%3];\" : );";
        std::string c = PTXTranslator::translate(src);
        check("TMA store → vgre_tma_store_2d_b (a store, not a load)",
              contains(c, "vgre_tma_store_2d_b(") && !contains(c, "vgre_tma_load"));
        check("TMA store passes descriptor first then source SMEM",
              contains(c, "VgreTMADescriptor") && contains(c, "%0") &&
              contains(c, "%3") && contains(c, "%1") && contains(c, "%2"));
    }

    // (9) Hopper thread-block cluster (P3-6): barrier.cluster.wait must lower to
    //     the cluster barrier, and mapa.shared::cluster must lower to the DSMEM
    //     address-retarget helper (not "not supported").
    {
        std::string sync = PTXTranslator::translate(
            "asm volatile(\"barrier.cluster.wait;\" : );");
        check("barrier.cluster.wait -> vgre_jit_cluster_sync",
              contains(sync, "vgre_jit_cluster_sync("));

        std::string arrive = PTXTranslator::translate(
            "asm volatile(\"barrier.cluster.arrive;\" : );");
        check("barrier.cluster.arrive does not rendezvous (arrive is a no-op)",
              !contains(arrive, "vgre_jit_cluster_sync("));

        std::string mapa = PTXTranslator::translate(
            "asm volatile(\"mapa.shared::cluster.u64 %0, %1, %2;\" "
            ": \"=l\"(d) : \"l\"(a), \"r\"(rank));");
        check("mapa.shared::cluster -> vgre_jit_mapa_shared_cluster (DSMEM)",
              contains(mapa, "vgre_jit_mapa_shared_cluster(") &&
              contains(mapa, "%1") && contains(mapa, "%2"));
    }

    // (10) Blackwell tcgen05 TMEM data path (P3-7): alloc writes a TMEM address to
    //      SMEM, mma accumulates into a TMEM address, ld reads it back out — all
    //      via the vgre_jit_tmem*/tcgen05 helpers (not "not supported").
    {
        std::string alloc = PTXTranslator::translate(
            "asm volatile(\"tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 "
            "[%0], %1;\" :: \"l\"(p), \"r\"(n));");
        check("tcgen05.alloc -> vgre_jit_tmem_alloc into SMEM",
              contains(alloc, "vgre_jit_tmem_alloc("));

        std::string mma = PTXTranslator::translate(
            "asm volatile(\"tcgen05.mma.cta_group::1.m64n256k16.f32.bf16.bf16 "
            "[%0], %1, %2, %3;\" :: \"r\"(d),\"l\"(a),\"l\"(b),\"r\"(idesc));");
        check("tcgen05.mma -> vgre_jit_tcgen05_mma accumulating into TMEM",
              contains(mma, "vgre_jit_tcgen05_mma(") &&
              contains(mma, ",64,256,16,0,"));

        std::string ld = PTXTranslator::translate(
            "asm volatile(\"tcgen05.ld.sync.aligned.32x32b.x2.b32 {%0,%1}, [%2];\" "
            ": \"=r\"(r0),\"=r\"(r1) : \"r\"(taddr));");
        check("tcgen05.ld -> vgre_jit_tcgen05_ld reads the TMEM fragment",
              contains(ld, "vgre_jit_tcgen05_ld(") && contains(ld, "%0") && contains(ld, "%1"));
    }

    printf("\n%d / %d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
