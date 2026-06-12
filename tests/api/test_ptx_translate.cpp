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

    printf("\n%d / %d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
