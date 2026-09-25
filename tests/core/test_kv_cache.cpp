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

    // ── Continuous-batching scheduler (Track 22) ─────────────────────────────
    {
        // Small pool so capacity is the binding constraint; maxBatch=2 so the 4
        // requests cannot all run at once — they must be batched continuously.
        KVCacheManager pool(8, 4, 1, 4);            // 8 blocks
        const int total0 = pool.freeBlocks();
        ContinuousBatchScheduler sched(pool, /*maxBatch=*/2);

        sched.addRequest(/*id=*/1, /*promptLen=*/3, /*maxNew=*/2);
        sched.addRequest(2, 2, 3);
        sched.addRequest(3, 4, 1);
        check("4 reqs queued (one added mid-run later)", sched.waitingCount() == 3);

        int maxConcurrent = 0, steps = 0;
        bool addedLate = false;
        while (!sched.allDone() && steps < 1000) {
            int running = sched.step();
            maxConcurrent = std::max(maxConcurrent, running);
            // Submit a 4th request mid-flight at step 2 — continuous batching must
            // still admit and complete it without a fresh batch.
            if (steps == 2 && !addedLate) { sched.addRequest(4, 2, 2); addedLate = true; }
            ++steps;
        }
        check("scheduler drains all requests", sched.allDone());
        check("all 4 requests finished (incl. the late one)", sched.finishedCount() == 4);
        check("never exceeded maxBatch=2 concurrently", maxConcurrent <= 2);
        check("continuous batching (>1 batch worth of steps)", steps > 3);
        check("KV pool fully reclaimed after all retire", pool.freeBlocks() == total0);
    }

    // ── Quantized KV-cache (int8 / int4) on the paged pools ──────────────────
    // The serving path stores K/V as symmetric-absmax int8 or packed int4 with a
    // per-head scale, cutting KV footprint ~3.8×/~7× while staying close to fp32.
    {
        const int nb = 16, bsz = 4, nh = 2, hd = 64, M = 10;   // hd=64: the real long-context regime where the scale amortizes
        KVCacheManager f32(nb, bsz, nh, hd, KVDType::F32);
        KVCacheManager i8(nb, bsz, nh, hd, KVDType::I8);
        KVCacheManager i4(nb, bsz, nh, hd, KVDType::I4);
        check("int8 dtype selected", i8.dtype() == KVDType::I8);
        check("int4 dtype selected (even headDim)", i4.dtype() == KVDType::I4);

        // Same random K/V into all three; keep contiguous fp32 copies for the ref.
        std::mt19937 rng2(11);
        std::normal_distribution<float> nd2(0, 1);
        std::vector<float> Kr[2], Vr[2];
        for (int t = 0; t < M; ++t) {
            std::vector<float> kAll(nh * hd), vAll(nh * hd);
            for (int h = 0; h < nh; ++h)
                for (int i = 0; i < hd; ++i) {
                    float kk = nd2(rng2), vv = nd2(rng2);
                    kAll[h * hd + i] = kk; vAll[h * hd + i] = vv;
                    Kr[h].push_back(kk);  Vr[h].push_back(vv);
                }
            f32.appendToken(1, kAll.data(), vAll.data());
            i8.appendToken(1, kAll.data(), vAll.data());
            i4.appendToken(1, kAll.data(), vAll.data());
        }

        // Round-trip: readKey/readVal in int8 mode should recover K/V within the
        // int8 step (scale = amax/127 ⇒ |err| ≤ scale/2 ≤ amax/254 per element).
        double rtErr = 0.0;
        for (int t = 0; t < M; ++t) {
            const int logical = t / bsz, slot = t % bsz;
            const int pb = i8.blockTable(1)[logical];
            std::vector<float> kk(hd), vv(hd);
            for (int h = 0; h < nh; ++h) {
                i8.readKey(pb, slot, h, kk.data());
                i8.readVal(pb, slot, h, vv.data());
                for (int i = 0; i < hd; ++i) {
                    rtErr = std::fmax(rtErr, std::fabs(kk[i] - Kr[h][t * hd + i]));
                    rtErr = std::fmax(rtErr, std::fabs(vv[i] - Vr[h][t * hd + i]));
                }
            }
        }
        printf("  [info] int8 KV round-trip max err = %.3e\n", rtErr);
        check("int8 round-trip within the quant step (<0.05)", rtErr < 0.05);

        // Paged attention: int8/int4 stay close to the fp32 reference.
        const float sc = 1.0f / std::sqrt((float)hd);
        double e8 = 0.0, e4 = 0.0;
        for (int h = 0; h < nh; ++h) {
            std::vector<float> q(hd); for (auto &x : q) x = nd2(rng2);
            std::vector<float> o8(hd), o4(hd), ref;
            refAttention(q.data(), Kr[h], Vr[h], M, hd, sc, -1, ref);
            pagedAttention(q.data(), h, 1, i8, sc, o8.data(), -1);
            pagedAttention(q.data(), h, 1, i4, sc, o4.data(), -1);
            for (int i = 0; i < hd; ++i) {
                e8 = std::fmax(e8, std::fabs(o8[i] - ref[i]));
                e4 = std::fmax(e4, std::fabs(o4[i] - ref[i]));
            }
        }
        printf("  [info] paged attention err vs fp32: int8=%.3e  int4=%.3e\n", e8, e4);
        check("int8 paged attention ~ fp32 (<0.05)", e8 < 0.05);
        check("int4 paged attention ~ fp32 (<0.30)", e4 < 0.30);

        // Memory: int8 ≈ 1/3.8, int4 ≈ 1/7 of fp32 bytes/token (+ per-head scale).
        const size_t bF = f32.bytesPerToken(), b8 = i8.bytesPerToken(), b4 = i4.bytesPerToken();
        printf("  [info] bytes/token: fp32=%zu int8=%zu int4=%zu\n", bF, b8, b4);
        check("int8 uses < 1/3 of fp32 KV bytes", b8 * 3 < bF);
        check("int4 uses < 1/5 of fp32 KV bytes", b4 * 5 < bF);

        // int4 with an odd headDim can't byte-align heads → downgrades to int8.
        KVCacheManager odd(4, 4, 1, 3, KVDType::I4);
        check("int4 + odd headDim downgrades to int8", odd.dtype() == KVDType::I8);

        // The continuous-batching scheduler runs unchanged over a quantized pool.
        KVCacheManager qpool(8, 4, 1, 4, KVDType::I8);
        const int q0 = qpool.freeBlocks();
        ContinuousBatchScheduler qsched(qpool, 2);
        qsched.addRequest(1, 3, 2); qsched.addRequest(2, 2, 3); qsched.addRequest(3, 4, 1);
        int qsteps = 0;
        while (!qsched.allDone() && qsteps < 1000) { qsched.step(); ++qsteps; }
        check("scheduler drains over an int8 pool", qsched.allDone() && qsched.finishedCount() == 3);
        check("int8 pool fully reclaimed", qpool.freeBlocks() == q0);
    }

    printf("\n%d / %d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
