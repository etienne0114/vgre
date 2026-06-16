// GPTQ / AWQ 4-bit dequant: pack random int4 weights the canonical way (the
// forward direction, written independently of the loader) and verify the
// dequantizers recover the reference s·(q − z). Covers GPTQ's +1 zero
// convention, act-order g_idx, and AWQ's [0,2,4,6,1,3,5,7] nibble order.

#include "vgre/xla/gptq.h"

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

static const int kAwqInv[8] = {0, 4, 1, 5, 2, 6, 3, 7};

static double maxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return 1e9;
    double m = 0; for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs((double)a[i] - b[i])); return m;
}

int main() {
    const int64_t IN = 16, OUT = 24, GS = 8;          // group_size 8 → 2 groups
    const int64_t G = IN / GS, OUT8 = OUT / 8;
    std::mt19937 rng(5);
    std::uniform_int_distribution<int> nibble(0, 15);
    std::uniform_real_distribution<float> sc(0.01f, 0.2f);

    // Random quantized data shared by both formats.
    std::vector<int> qw((size_t)(IN * OUT)), z((size_t)(G * OUT));
    std::vector<float> scales((size_t)(G * OUT));
    for (auto& v : qw) v = nibble(rng);
    for (auto& v : z) v = nibble(rng);
    for (auto& v : scales) v = sc(rng);

    // ── GPTQ ─────────────────────────────────────────────────────────────────
    {
        // Pack qweight [in/8, out] (8 nibbles along `in`) and qzeros [groups,out/8].
        GptqTensors q;
        q.in_features = IN; q.out_features = OUT; q.groups = G; q.bits = 4;
        q.qweight.assign((size_t)((IN / 8) * OUT), 0);
        q.qzeros.assign((size_t)(G * OUT8), 0);
        q.scales.assign(scales.begin(), scales.end());
        for (int64_t r = 0; r < IN / 8; ++r)
            for (int64_t j = 0; j < OUT; ++j) {
                int32_t p = 0;
                for (int t = 0; t < 8; ++t) p |= (qw[(size_t)((r * 8 + t) * OUT + j)] & 0xF) << (4 * t);
                q.qweight[(size_t)(r * OUT + j)] = p;
            }
        for (int64_t gg = 0; gg < G; ++gg)
            for (int64_t jj = 0; jj < OUT8; ++jj) {
                int32_t p = 0;
                for (int t = 0; t < 8; ++t) p |= (z[(size_t)(gg * OUT + jj * 8 + t)] & 0xF) << (4 * t);
                q.qzeros[(size_t)(gg * OUT8 + jj)] = p;
            }

        // Reference: W = s·(qw − (z+1)), group = i/group_size.
        std::vector<float> ref((size_t)(IN * OUT));
        for (int64_t i = 0; i < IN; ++i) {
            int64_t g = i / GS;
            for (int64_t j = 0; j < OUT; ++j)
                ref[(size_t)(i * OUT + j)] = scales[(size_t)(g * OUT + j)] *
                                             (float)(qw[(size_t)(i * OUT + j)] - (z[(size_t)(g * OUT + j)] + 1));
        }
        Literal W = gptqDequantize(q);
        CHECK(W.shape.dims == (std::vector<int64_t>{IN, OUT}), "GPTQ shape [in,out]");
        CHECK(maxAbsDiff(W.data, ref) < 1e-5, "GPTQ dequant == reference");

        // act-order: a non-trivial g_idx must be honored.
        q.g_idx.resize((size_t)IN);
        for (int64_t i = 0; i < IN; ++i) q.g_idx[(size_t)i] = (int)((IN - 1 - i) / GS);  // reversed groups
        std::vector<float> ref2((size_t)(IN * OUT));
        for (int64_t i = 0; i < IN; ++i) {
            int64_t g = (IN - 1 - i) / GS;
            for (int64_t j = 0; j < OUT; ++j)
                ref2[(size_t)(i * OUT + j)] = scales[(size_t)(g * OUT + j)] *
                                              (float)(qw[(size_t)(i * OUT + j)] - (z[(size_t)(g * OUT + j)] + 1));
        }
        CHECK(maxAbsDiff(gptqDequantize(q).data, ref2) < 1e-5, "GPTQ act-order g_idx honored");
    }

    // ── AWQ ──────────────────────────────────────────────────────────────────
    {
        // Pack qweight [in, out/8] and qzeros [groups, out/8] with AWQ nibble order.
        GptqTensors q;
        q.in_features = IN; q.out_features = OUT; q.groups = G; q.bits = 4;
        q.qweight.assign((size_t)(IN * OUT8), 0);
        q.qzeros.assign((size_t)(G * OUT8), 0);
        q.scales.assign(scales.begin(), scales.end());
        for (int64_t i = 0; i < IN; ++i)
            for (int64_t jj = 0; jj < OUT8; ++jj) {
                int32_t p = 0;
                for (int t = 0; t < 8; ++t) p |= (qw[(size_t)(i * OUT + jj * 8 + t)] & 0xF) << (4 * kAwqInv[t]);
                q.qweight[(size_t)(i * OUT8 + jj)] = p;
            }
        for (int64_t gg = 0; gg < G; ++gg)
            for (int64_t jj = 0; jj < OUT8; ++jj) {
                int32_t p = 0;
                for (int t = 0; t < 8; ++t) p |= (z[(size_t)(gg * OUT + jj * 8 + t)] & 0xF) << (4 * kAwqInv[t]);
                q.qzeros[(size_t)(gg * OUT8 + jj)] = p;
            }

        // Reference: W = (qw − z)·s, group = i/group_size (no +1).
        std::vector<float> ref((size_t)(IN * OUT));
        for (int64_t i = 0; i < IN; ++i) {
            int64_t g = i / GS;
            for (int64_t j = 0; j < OUT; ++j)
                ref[(size_t)(i * OUT + j)] = (float)(qw[(size_t)(i * OUT + j)] - z[(size_t)(g * OUT + j)]) *
                                             scales[(size_t)(g * OUT + j)];
        }
        Literal W = awqDequantize(q);
        CHECK(W.shape.dims == (std::vector<int64_t>{IN, OUT}), "AWQ shape [in,out]");
        CHECK(maxAbsDiff(W.data, ref) < 1e-5, "AWQ dequant (order permutation) == reference");
    }

    if (g_fail == 0) { std::printf("test_gptq: ALL CHECKS PASSED\n"); return 0; }
    std::printf("test_gptq: %d FAILURE(S)\n", g_fail);
    return 1;
}
