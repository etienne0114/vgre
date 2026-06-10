// Track 18 — paged KV-cache + paged attention.
//
// Verifies that attention computed over a sequence whose KV is scattered across
// non-contiguous physical blocks (via its block table) matches a contiguous
// reference, and that the block allocator's free list behaves correctly.

#include "vgre/core/kv_cache.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace vgre::core;

static int g_pass = 0, g_total = 0;
static void check(const char *name, bool ok) {
    ++g_total;
    printf(ok ? "  PASS  %s\n" : "  FAIL  %s\n", name);
    if (ok) ++g_pass;
}

// Contiguous reference: gather all K/V for `head` into arrays, full softmax.
static void refAttention(const float *q, const std::vector<float> &K,
                         const std::vector<float> &V, int len, int d, float scale,
                         int causalUpTo, std::vector<float> &out) {
    out.assign(d, 0.0f);
    int last = (causalUpTo < 0 || causalUpTo >= len) ? len - 1 : causalUpTo;
    std::vector<float> s(last + 1);
    float mx = -1e30f;
    for (int t = 0; t <= last; ++t) {
        float dot = 0; for (int i = 0; i < d; ++i) dot += q[i] * K[t * d + i];
        s[t] = dot * scale; if (s[t] > mx) mx = s[t];
    }
    float sum = 0; for (int t = 0; t <= last; ++t) { s[t] = std::exp(s[t] - mx); sum += s[t]; }
    for (int t = 0; t <= last; ++t)
        for (int i = 0; i < d; ++i) out[i] += (s[t] / sum) * V[t * d + i];
}

int main() {
    printf("=== Paged KV-cache + paged attention (Track 18) ===\n");

    const int numBlocks = 16, blockSize = 4, numHeads = 2, headDim = 8;
    KVCacheManager kv(numBlocks, blockSize, numHeads, headDim);
    check("pool starts fully free", kv.freeBlocks() == numBlocks);

    // Fragment the pool first: take 3 blocks and free the middle one, so the
    // sequence below will receive NON-contiguous physical blocks.
    int b0 = kv.allocateBlock(), b1 = kv.allocateBlock(), b2 = kv.allocateBlock();
    kv.freeBlock(b1);
    check("3 allocated, 1 freed → free count tracks", kv.freeBlocks() == numBlocks - 2);
    (void)b0; (void)b2;

    // Append 10 tokens to a sequence (needs ceil(10/4)=3 blocks, drawn from a
    // fragmented free list → not contiguous). Keep contiguous copies for the ref.
    const SeqId seq = 42;
    const int N = 10;
    std::mt19937 rng(3);
    std::normal_distribution<float> nd(0, 1);
    std::vector<float> Kref[numHeads], Vref[numHeads];
    for (int h = 0; h < numHeads; ++h) { Kref[h].reserve(N * headDim); Vref[h].reserve(N * headDim); }

    bool allAppended = true;
    for (int t = 0; t < N; ++t) {
        std::vector<float> kAll(numHeads * headDim), vAll(numHeads * headDim);
        for (int h = 0; h < numHeads; ++h)
            for (int i = 0; i < headDim; ++i) {
                float kv_ = nd(rng), vv_ = nd(rng);
                kAll[h * headDim + i] = kv_;
                vAll[h * headDim + i] = vv_;
                Kref[h].push_back(kv_);
                Vref[h].push_back(vv_);
            }
        if (!kv.appendToken(seq, kAll.data(), vAll.data())) allAppended = false;
    }
    check("all 10 tokens appended", allAppended);
    check("sequence length == 10", kv.seqLen(seq) == N);
    check("block table has 3 blocks", (int)kv.blockTable(seq).size() == 3);
    // Confirm the blocks really are non-contiguous (fragmented allocation).
    const auto &bt = kv.blockTable(seq);
    bool nonContig = false;
    for (size_t i = 1; i < bt.size(); ++i) if (bt[i] != bt[i - 1] + 1) nonContig = true;
    check("block table is non-contiguous (paged)", nonContig);

    // ── Paged attention over the cache matches the contiguous reference ──────
    const float scale = 1.0f / std::sqrt((float)headDim);
    double maxErr = 0.0;
    for (int h = 0; h < numHeads; ++h) {
        std::vector<float> q(headDim);
        for (auto &x : q) x = nd(rng);
        // full (non-causal)
        std::vector<float> paged(headDim), ref;
        pagedAttention(q.data(), h, seq, kv, scale, paged.data(), -1);
        refAttention(q.data(), Kref[h], Vref[h], N, headDim, scale, -1, ref);
        for (int i = 0; i < headDim; ++i) maxErr = std::fmax(maxErr, std::fabs(paged[i] - ref[i]));
        // causal up to position 6
        pagedAttention(q.data(), h, seq, kv, scale, paged.data(), 6);
        refAttention(q.data(), Kref[h], Vref[h], N, headDim, scale, 6, ref);
        for (int i = 0; i < headDim; ++i) maxErr = std::fmax(maxErr, std::fabs(paged[i] - ref[i]));
    }
    printf("  [info] paged vs contiguous attention max err = %.2e\n", maxErr);
    check("paged attention == contiguous reference (1e-5)", maxErr < 1e-5);

    // ── Freeing the sequence returns its blocks to the pool ──────────────────
    int before = kv.freeBlocks();
    kv.freeSequence(seq);
    check("freeSequence returns all 3 blocks", kv.freeBlocks() == before + 3);

    // ── Pool exhaustion is reported, not crashed ─────────────────────────────
    KVCacheManager tiny(2, 4, 1, 4);  // 2 blocks of 4 tokens = 8 tokens max
    std::vector<float> z(4, 0.0f);
    bool exhausted = false;
    for (int t = 0; t < 100; ++t)
        if (!tiny.appendToken(7, z.data(), z.data())) { exhausted = true; break; }
    check("pool exhaustion returns false (no crash)", exhausted && tiny.seqLen(7) == 8);

    printf("\n%d / %d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
