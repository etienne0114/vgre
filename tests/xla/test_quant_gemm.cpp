// Dequant-inside-the-GEMM: gemm_q must equal gemm_f32(X, dequant(W)) exactly,
// for every supported block-quant type, while never materializing the full f32
// weight (peak transient is one row slice).

#include "vgre/xla/quant_gemm.h"
#include "vgre/xla/blas_gemm.h"
#include "vgre/xla/quant.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
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

// Build a plausible quantized W blob of `K*N` weights (N a multiple of the block
// size) by quantizing then storing — but the gguf lib can't quantize K-quants,
// so we synthesize raw blocks with valid f16 scales and random payloads; the
// *reference* is our own dequantBlock, so the test checks gemm_q == gemm of that
// same dequant (i.e. the fused path matches the materialized path).
static std::vector<uint8_t> randBlocks(int ggml_type, int64_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> byte(0, 255);
    const int bb = quantBlockBytes(ggml_type), be = quantBlockElems(ggml_type);
    std::vector<uint8_t> buf((size_t)((n / be) * bb));
    for (auto& b : buf) b = (uint8_t)byte(rng);
    // Force a sane f16 super-block scale so values are finite.
    const uint16_t dpos = 0x3000;  // f16 ~0.125
    for (int64_t blk = 0; blk < n / be; ++blk) {
        uint8_t* p = buf.data() + blk * bb;
        if (ggml_type == 8 || ggml_type == 2 || ggml_type == 3) { p[0] = dpos & 0xFF; p[1] = dpos >> 8; }
        if (ggml_type == 3) { p[2] = 0; p[3] = 0; }      // Q4_1 min = 0
        if (ggml_type == 12) { p[0] = dpos & 0xFF; p[1] = dpos >> 8; p[2] = 0; p[3] = 0; }
        if (ggml_type == 14) { p[208] = dpos & 0xFF; p[209] = dpos >> 8; }
    }
    return buf;
}

static void testType(int ggml_type, const char* name, int64_t M, int64_t K, int64_t N) {
    std::vector<uint8_t> Wq = randBlocks(ggml_type, K * N, 100 + ggml_type);
    std::vector<float> Wf((size_t)(K * N));
    dequantBlock(ggml_type, Wq.data(), K * N, Wf.data());  // materialized reference weight

    std::mt19937 rng(7);
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    std::vector<float> X((size_t)(M * K));
    for (auto& v : X) v = d(rng);

    std::vector<float> ref((size_t)(M * N)), got((size_t)(M * N));
    gemm_f32(false, false, M, N, K, X.data(), Wf.data(), ref.data(), 1);  // materialize-then-GEMM
    CHECK(gemm_q(M, N, K, X.data(), Wq.data(), ggml_type, got.data()), "gemm_q runs");

    double mx = 0;
    for (size_t i = 0; i < ref.size(); ++i) mx = std::max(mx, std::fabs((double)ref[i] - got[i]));
    char m[96]; std::snprintf(m, sizeof(m), "%s gemm_q == gemm_f32(dequant) (maxdiff %.2e)", name, mx);
    CHECK(mx < 1e-3, m);
}

int main() {
    // Various shapes (N multiple of block size: 32 for legacy, 256 for K-quants).
    testType(8, "Q8_0", 4, 8, 64);
    testType(2, "Q4_0", 5, 16, 96);
    testType(3, "Q4_1", 3, 7, 128);
    testType(12, "Q4_K", 4, 6, 256);
    testType(14, "Q6_K", 2, 5, 512);
    testType(8, "Q8_0 single-row (M=1 decode)", 1, 32, 256);
    testType(2, "Q4_0 larger", 16, 64, 512);

    // Unsupported type / misaligned N → false.
    std::vector<float> X(8), C(8);
    std::vector<uint8_t> dummy(64);
    CHECK(!gemm_q(1, 30, 8, X.data(), dummy.data(), 2, C.data()), "misaligned N rejected");
    CHECK(!gemm_q(1, 32, 8, X.data(), dummy.data(), 999, C.data()), "unknown type rejected");

    if (g_fail == 0) { std::printf("test_quant_gemm: ALL CHECKS PASSED\n"); return 0; }
    std::printf("test_quant_gemm: %d FAILURE(S)\n", g_fail);
    return 1;
}
