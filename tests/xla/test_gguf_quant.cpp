// L3 int4/GGUF: block-quant dequant kernels (Q8_0/Q4_0/Q4_1), native quantized
// Literal storage (~4.5 bits/weight resident), a memory-mapped GGUF loader on a
// synthesized file, and an int4 weight flowing through the engine — a matmul fed
// a Q4_0 parameter equals the matmul of its dequantized f32 (the engine
// dequantizes on use, so results are identical, not approximate).

#include "vgre/xla/gguf.h"
#include "vgre/xla/hlo.h"
#include "vgre/xla/quant.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

using namespace vgre::xla;

static int g_fail = 0;
#define CHECK(cond, msg)                                                   \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("FAIL: %s  (%s:%d)\n", (msg), __FILE__, __LINE__); \
            ++g_fail;                                                      \
        }                                                                  \
    } while (0)

// ── byte builders (little-endian) ───────────────────────────────────────────
static void u16(std::vector<uint8_t>& b, uint16_t v) { b.push_back(v & 0xFF); b.push_back(v >> 8); }
static void u32(std::vector<uint8_t>& b, uint32_t v) { for (int i = 0; i < 4; ++i) b.push_back((v >> (8 * i)) & 0xFF); }
static void u64(std::vector<uint8_t>& b, uint64_t v) { for (int i = 0; i < 8; ++i) b.push_back((v >> (8 * i)) & 0xFF); }
static void f32b(std::vector<uint8_t>& b, float f) { uint8_t t[4]; std::memcpy(t, &f, 4); b.insert(b.end(), t, t + 4); }
static void gstr(std::vector<uint8_t>& b, const std::string& s) { u64(b, s.size()); b.insert(b.end(), s.begin(), s.end()); }

// Build a Q4_0 block (18 B): f16 scale d, then 16 bytes packing 32 nibbles.
// element j (0..15) = lo nibble of qs[j]; element j+16 = hi nibble. value=d*(n-8).
static std::vector<uint8_t> makeQ40Block(uint16_t d_f16, const int* nibbles /*32*/) {
    std::vector<uint8_t> blk;
    u16(blk, d_f16);
    for (int j = 0; j < 16; ++j)
        blk.push_back((uint8_t)((nibbles[j] & 0xF) | ((nibbles[j + 16] & 0xF) << 4)));
    return blk;
}

int main() {
    // ── 1. Block dequant kernels ────────────────────────────────────────────
    {
        // Q8_0: d=0.5 (f16 0x3800), qs[i]=i-16 → x_i = 0.5*(i-16)
        std::vector<uint8_t> blk; u16(blk, 0x3800);
        for (int i = 0; i < 32; ++i) blk.push_back((uint8_t)(int8_t)(i - 16));
        float out[32];
        dequant_q8_0(blk.data(), 32, out);
        bool ok = true;
        for (int i = 0; i < 32; ++i) ok &= (out[i] == 0.5f * (i - 16));
        CHECK(ok, "Q8_0 dequant");
    }
    {
        int nib[32];
        for (int j = 0; j < 16; ++j) { nib[j] = j; nib[j + 16] = 15 - j; }
        auto blk = makeQ40Block(0x3C00 /*d=1.0*/, nib);  // value = 1.0*(n-8)
        float out[32];
        dequant_q4_0(blk.data(), 32, out);
        bool ok = true;
        for (int j = 0; j < 16; ++j) { ok &= (out[j] == j - 8); ok &= (out[j + 16] == (15 - j) - 8); }
        CHECK(ok, "Q4_0 dequant");
        CHECK(quantStorageBytes(2, 32) == 18 && quantBlockBytes(2) == 18, "Q4_0 block size");
    }

    // ── 2. Native quantized Literal storage + toF32 + footprint ─────────────
    {
        int nib[32];
        for (int j = 0; j < 32; ++j) nib[j] = (j * 5) & 0xF;
        auto blk = makeQ40Block(0x3C00, nib);
        Literal q; q.shape = Shape{{4, 8}}; q.dtype = DType::Q4_0; q.quant = blk;
        CHECK(q.numel() == 32, "quant numel from shape");
        CHECK(q.storageBytes() == 18, "Q4_0 resident = 18 B for 32 weights (~4.5 bit/wt)");
        CHECK(q.storageBytes() * 7 < 32 * 4, "int4 well under fp32 footprint");

        Literal f = q.toF32();
        float ref[32];
        dequant_q4_0(blk.data(), 32, ref);
        bool ok = (f.dtype == DType::F32 && f.data.size() == 32 && f.shape.dims == q.shape.dims);
        for (int i = 0; i < 32; ++i) ok &= (f.data[i] == ref[i]);
        CHECK(ok, "Q4_0 Literal toF32 matches kernel");
        CHECK(q.at(5) == ref[5] && q.at(20) == ref[20], "Q4_0 element accessor");
    }

    // ── 3. Memory-mapped GGUF loader on a synthesized file ──────────────────
    {
        namespace fs = std::filesystem;
        fs::path path = fs::temp_directory_path() / "vgre_test.gguf";

        // tensor data blob (tight offsets): w_f32[2,3]=24B, w_q8[32]=34B, w_q4[32]=18B
        std::vector<uint8_t> blob;
        for (float v : {1.f, 2.f, 3.f, 4.f, 5.f, 6.f}) f32b(blob, v);  // w_f32 @0
        u16(blob, 0x3800);                                             // w_q8 @24, d=0.5
        for (int i = 0; i < 32; ++i) blob.push_back((uint8_t)(int8_t)(i - 16));
        {                                                              // w_q4 @58, d=1.0
            int nib[32];
            for (int j = 0; j < 16; ++j) { nib[j] = j; nib[j + 16] = 15 - j; }
            auto blk = makeQ40Block(0x3C00, nib);
            blob.insert(blob.end(), blk.begin(), blk.end());
        }

        std::vector<uint8_t> h;
        h.insert(h.end(), {'G', 'G', 'U', 'F'});
        u32(h, 3);          // version
        u64(h, 3);          // tensor count
        u64(h, 3);          // kv count
        // kv: string, alignment(u32), array<u32>
        gstr(h, "general.architecture"); u32(h, 8 /*STRING*/); gstr(h, "llama");
        gstr(h, "general.alignment");    u32(h, 4 /*U32*/);    u32(h, 32);
        gstr(h, "test.arr");             u32(h, 9 /*ARRAY*/);  u32(h, 4 /*U32*/); u64(h, 3); u32(h, 1); u32(h, 2); u32(h, 3);
        // tensor infos
        gstr(h, "w_f32"); u32(h, 2); u64(h, 3); u64(h, 2); u32(h, 0 /*F32*/);  u64(h, 0);
        gstr(h, "w_q8");  u32(h, 1); u64(h, 32);            u32(h, 8 /*Q8_0*/); u64(h, 24);
        gstr(h, "w_q4");  u32(h, 1); u64(h, 32);            u32(h, 2 /*Q4_0*/); u64(h, 58);
        while (h.size() % 32 != 0) h.push_back(0);  // align data blob to 32

        {
            std::ofstream f(path, std::ios::binary | std::ios::trunc);
            f.write(reinterpret_cast<char*>(h.data()), (std::streamsize)h.size());
            f.write(reinterpret_cast<char*>(blob.data()), (std::streamsize)blob.size());
        }

        auto g = GGUF::open(path.string());
        CHECK(g != nullptr, "open synthesized GGUF");
        if (g) {
            CHECK(g->version() == 3, "GGUF version 3");
            CHECK(g->names().size() == 3, "3 tensors listed");
            CHECK(g->metadataString("general.architecture") == "llama", "string metadata");
            const auto* qi = g->info("w_q4");
            CHECK(qi && qi->ggml_type == 2 && qi->shape == std::vector<int64_t>{32}, "w_q4 info");

            Literal l;
            CHECK(g->load("w_f32", l) && l.shape.dims == (std::vector<int64_t>{2, 3}), "load f32 shape");
            CHECK(l.data == (std::vector<float>{1, 2, 3, 4, 5, 6}), "f32 values");

            CHECK(g->load("w_q8", l), "load q8 (dequant)");
            bool q8ok = (l.data.size() == 32);
            for (int i = 0; i < 32; ++i) q8ok &= (l.data[i] == 0.5f * (i - 16));
            CHECK(q8ok, "Q8_0 dequantized on load");

            // keepNative keeps Q4_0 at 18 B; toF32 reproduces it.
            Literal nat;
            CHECK(g->load("w_q4", nat, /*keepNative=*/true), "load q4 native");
            CHECK(nat.dtype == DType::Q4_0 && nat.storageBytes() == 18, "Q4_0 kept native (18 B)");
            Literal deq; CHECK(g->load("w_q4", deq), "load q4 dequant");
            CHECK(nat.toF32().data == deq.data, "native Q4_0 toF32 == dequant-on-load");

            CHECK(!g->load("nope", l), "missing tensor → false");
        }
        fs::remove(path);
        CHECK(GGUF::open("/no/such.gguf") == nullptr, "missing file → nullptr");
    }

    // ── 4. int4 weight through the engine: Q4_0 param == its dequant f32 ─────
    {
        int nib[32];
        for (int j = 0; j < 32; ++j) nib[j] = (j * 3 + 1) & 0xF;
        auto blk = makeQ40Block(0x3C00, nib);
        Literal Wq; Wq.shape = Shape{{4, 8}}; Wq.dtype = DType::Q4_0; Wq.quant = blk;  // [4,8] weight
        Literal Wf = Wq.toF32();

        std::vector<float> X;  // [8,3] activations
        for (int i = 0; i < 24; ++i) X.push_back((float)((i % 7) - 3) * 0.25f);
        Literal Xl = Literal::make({{8, 3}}, X);

        HloModule m;
        int w = m.parameter(0, Shape{{4, 8}});
        int x = m.parameter(1, Shape{{8, 3}});
        m.dot(w, x);
        Literal ref = m.evaluate({Wf, Xl});           // f32 weight
        Literal got = m.evaluate({Wq, Xl});           // Q4_0 weight (engine dequants on use)
        CHECK(got.data == ref.data, "Q4_0-param matmul identical to dequantized-f32 matmul");
        CHECK(Wq.storageBytes() < Wf.storageBytes(), "Q4_0 weight smaller than f32 weight");
    }

    if (g_fail == 0) { std::printf("test_gguf_quant: ALL CHECKS PASSED\n"); return 0; }
    std::printf("test_gguf_quant: %d FAILURE(S)\n", g_fail);
    return 1;
}
