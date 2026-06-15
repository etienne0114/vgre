// Memory-mapped safetensors loader: synthesize a real .safetensors file with
// mixed dtypes (F32 / BF16 / F16 incl. a subnormal / I8), load it back through
// the mmap path, and verify shapes + dequantized values. Plus error handling.

#include "vgre/xla/safetensors.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using vgre::xla::SafeTensors;
using vgre::xla::Literal;

static int g_fail = 0;
#define CHECK(cond, msg)                                                   \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("FAIL: %s  (%s:%d)\n", (msg), __FILE__, __LINE__); \
            ++g_fail;                                                      \
        }                                                                  \
    } while (0)

static void putF32(std::vector<uint8_t>& b, float f) {
    uint8_t t[4];
    std::memcpy(t, &f, 4);
    b.insert(b.end(), t, t + 4);  // x86/arm are little-endian, matching safetensors
}
static void putU16(std::vector<uint8_t>& b, uint16_t v) {
    b.push_back((uint8_t)(v & 0xFF));
    b.push_back((uint8_t)(v >> 8));
}

int main() {
    namespace fs = std::filesystem;
    fs::path path = fs::temp_directory_path() / "vgre_test_weights.safetensors";

    // ── Build the data blob (contiguous, row-major) ─────────────────────────
    std::vector<uint8_t> blob;
    // w_f32: [2,2]
    putF32(blob, 1.0f); putF32(blob, -2.5f); putF32(blob, 3.25f); putF32(blob, 4.0f);
    // w_bf16: [3] = 1.0, -2.0, 0.5  (exactly representable)
    putU16(blob, 0x3F80); putU16(blob, 0xC000); putU16(blob, 0x3F00);
    // w_f16: [3] = 1.0, 2.0, -0.5
    putU16(blob, 0x3C00); putU16(blob, 0x4000); putU16(blob, 0xB800);
    // w_i8: [4] = -128, -1, 0, 127
    blob.push_back(0x80); blob.push_back(0xFF); blob.push_back(0x00); blob.push_back(0x7F);
    // w_f16sub: [1] = smallest positive f16 subnormal = 2^-24
    putU16(blob, 0x0001);

    const std::string header =
        "{\"w_f32\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,16]},"
        "\"w_bf16\":{\"dtype\":\"BF16\",\"shape\":[3],\"data_offsets\":[16,22]},"
        "\"w_f16\":{\"dtype\":\"F16\",\"shape\":[3],\"data_offsets\":[22,28]},"
        "\"w_i8\":{\"dtype\":\"I8\",\"shape\":[4],\"data_offsets\":[28,32]},"
        "\"__metadata__\":{\"format\":\"pt\"},"
        "\"w_f16sub\":{\"dtype\":\"F16\",\"shape\":[1],\"data_offsets\":[32,34]}}";

    {
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        uint64_t hlen = header.size();
        uint8_t lb[8];
        for (int i = 0; i < 8; ++i) lb[i] = (uint8_t)(hlen >> (8 * i));  // u64 LE
        f.write(reinterpret_cast<char*>(lb), 8);
        f.write(header.data(), (std::streamsize)header.size());
        f.write(reinterpret_cast<char*>(blob.data()), (std::streamsize)blob.size());
    }

    auto st = SafeTensors::open(path.string());
    CHECK(st != nullptr, "open synthesized safetensors");
    if (!st) { fs::remove(path); return 1; }

    CHECK(st->mappedBytes() == 8 + header.size() + blob.size(), "mappedBytes = header+blob");
    CHECK(st->names().size() == 5, "5 tensors (metadata skipped)");
    CHECK(st->contains("w_f32") && st->contains("w_f16sub"), "contains by name");
    CHECK(!st->contains("nope"), "missing name not contained");

    const auto* bi = st->info("w_bf16");
    CHECK(bi && bi->dtype == SafeTensors::DType::BF16 && bi->shape == std::vector<int64_t>{3},
          "w_bf16 dtype+shape");

    Literal l;
    CHECK(st->load("w_f32", l), "load f32");
    CHECK(l.shape.dims == (std::vector<int64_t>{2, 2}), "f32 shape [2,2]");
    CHECK(l.data == (std::vector<float>{1.0f, -2.5f, 3.25f, 4.0f}), "f32 values");

    CHECK(st->load("w_bf16", l), "load bf16");
    CHECK(l.data == (std::vector<float>{1.0f, -2.0f, 0.5f}), "bf16 dequantized values");

    CHECK(st->load("w_f16", l), "load f16");
    CHECK(l.data == (std::vector<float>{1.0f, 2.0f, -0.5f}), "f16 dequantized values");

    CHECK(st->load("w_i8", l), "load i8");
    CHECK(l.data == (std::vector<float>{-128.0f, -1.0f, 0.0f, 127.0f}), "i8 widened values");

    CHECK(st->load("w_f16sub", l), "load f16 subnormal");
    CHECK(std::fabs(l.data[0] - std::ldexp(1.0f, -24)) < 1e-30f, "f16 subnormal == 2^-24");

    CHECK(!st->load("nope", l), "load missing returns false");

    // ── Error handling ──────────────────────────────────────────────────────
    CHECK(SafeTensors::open("/no/such/file.safetensors") == nullptr, "missing file → nullptr");
    {
        fs::path bad = fs::temp_directory_path() / "vgre_bad.safetensors";
        std::ofstream f(bad, std::ios::binary | std::ios::trunc);
        const char junk[] = "not a safetensors file at all";
        f.write(junk, sizeof(junk));
        f.close();
        CHECK(SafeTensors::open(bad.string()) == nullptr, "garbage file → nullptr");
        fs::remove(bad);
    }

    fs::remove(path);
    if (g_fail == 0) { std::printf("test_safetensors: ALL CHECKS PASSED\n"); return 0; }
    std::printf("test_safetensors: %d FAILURE(S)\n", g_fail);
    return 1;
}
