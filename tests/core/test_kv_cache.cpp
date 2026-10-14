// Track 18 — paged KV-cache + paged attention.
//
// Verifies that attention computed over a sequence whose KV is scattered across
// non-contiguous physical blocks (via its block table) matches a contiguous
// reference, and that the block allocator's free list behaves correctly.

#include "vgre/core/kv_cache.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
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

    // Reserving the maximum sequence length is atomic and does not fabricate
    // logical tokens. Appends consume the reserved blocks as normal.
    {
        KVCacheManager reserved(2, 4, 1, 2);
        const int free0 = reserved.freeBlocks();
        check("sequence capacity reservation succeeds", reserved.reserveSequenceCapacity(8, 5));
        check("reservation allocates blocks but leaves length at zero",
              reserved.freeBlocks() == free0 - 2 && reserved.seqLen(8) == 0);
        check("capacity reservation rejects an impossible request atomically",
              !reserved.reserveSequenceCapacity(9, 9) && reserved.freeBlocks() == free0 - 2);
        std::vector<float> key{1.0f, 2.0f}, value{3.0f, 4.0f};
        bool appended = true;
        for (int i = 0; i < 5; ++i)
            appended = reserved.appendToken(8, key.data(), value.data()) && appended;
        check("appends consume pre-reserved blocks", appended && reserved.seqLen(8) == 5);
        reserved.freeSequence(8);
        check("freeSequence releases unused reserved capacity", reserved.freeBlocks() == free0);
    }

    // ── Pool exhaustion is reported, not crashed ─────────────────────────────
    KVCacheManager tiny(2, 4, 1, 4);  // 2 blocks of 4 tokens = 8 tokens max
    std::vector<float> z(4, 0.0f);
    bool exhausted = false;
    for (int t = 0; t < 100; ++t)
        if (!tiny.appendToken(7, z.data(), z.data())) { exhausted = true; break; }
    check("pool exhaustion returns false (no crash)", exhausted && tiny.seqLen(7) == 8);

    // ── Continuous-batching scheduler (Track 22) ─────────────────────────────
    {
        KVCacheManager smallPool(2, 4, 1, 2);
        ContinuousBatchScheduler sched(smallPool, 2);
        bool rejected = false;
        try { sched.addRequest(99, 1, 8); }
        catch (const std::length_error&) { rejected = true; }
        check("scheduler rejects a request that can never fit the KV pool",
              rejected && sched.waitingCount() == 0 && smallPool.freeBlocks() == 2);
    }

    {
        // Small pool so capacity is the binding constraint; maxBatch=2 so the 4
        // requests cannot all run at once — they must be batched continuously.
        KVCacheManager pool(8, 4, 1, 4);            // 8 blocks
        const int total0 = pool.freeBlocks();
        ContinuousBatchScheduler sched(pool, /*maxBatch=*/2);
        sched.setTokenExecutor([](const Request& request, ScheduledTokenPhase phase,
                                  int position, std::vector<float>& keys,
                                  std::vector<float>& values) {
            const float base = static_cast<float>(request.id) + position +
                (phase == ScheduledTokenPhase::PREFILL ? 1.0f : 2.0f);
            std::fill(keys.begin(), keys.end(), base);
            std::fill(values.begin(), values.end(), -base);
        });

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

    // The waiting index skips requests that do not fit, while retaining FIFO
    // order across all eligible capacity buckets.
    {
        KVCacheManager pool(6, 1, 1, 1);
        const int rawBlock = pool.allocateBlock();
        ContinuousBatchScheduler sched(pool, /*maxBatch=*/3);
        std::vector<SeqId> prefillOrder;
        bool enqueuedMidStep = false;
        sched.setTokenExecutor([&](const Request& request, ScheduledTokenPhase phase,
                                   int position, std::vector<float>& keys,
                                   std::vector<float>& values) {
            if (phase == ScheduledTokenPhase::PREFILL && position == 0) {
                prefillOrder.push_back(request.id);
                if (request.id == 2 && !enqueuedMidStep) {
                    enqueuedMidStep = true;
                    sched.addRequest(4, 1, 0);
                }
            }
            std::fill(keys.begin(), keys.end(), static_cast<float>(request.id));
            std::fill(values.begin(), values.end(), -static_cast<float>(request.id));
        });
        sched.addRequest(1, 6, 0);  // oldest request needs all six blocks
        sched.addRequest(2, 1, 0);  // fits first in the five-block remainder
        sched.addRequest(3, 2, 0);  // fits second; tests order across size buckets
        const int firstWave = sched.step();
        check("scheduler admits oldest eligible requests around an oversized waiter",
              rawBlock >= 0 && firstWave == 2 && prefillOrder.size() == 2 &&
              prefillOrder[0] == 2 && prefillOrder[1] == 3 &&
              enqueuedMidStep && sched.waitingCount() == 2 &&
              !pool.hasSequence(1) && !pool.hasSequence(4) &&
              pool.seqLen(2) == 1 && pool.seqLen(3) == 2);

        pool.freeBlock(rawBlock);
        const int secondWave = sched.step();
        check("older large request is admitted once its full capacity becomes free",
              secondWave == 1 && prefillOrder.size() == 3 && prefillOrder[2] == 1 &&
              sched.waitingCount() == 1 && pool.seqLen(1) == 6 &&
              !pool.hasSequence(4));
        const int thirdWave = sched.step();
        check("request enqueued during prefill joins at the next scheduler step",
              thirdWave == 1 && prefillOrder.size() == 4 && prefillOrder[3] == 4 &&
              sched.waitingCount() == 0 && pool.seqLen(4) == 1);
        sched.step();
        check("capacity-indexed waiting queue drains and reclaims all blocks",
              sched.allDone() && sched.finishedCount() == 4 && pool.freeBlocks() == 6);
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
        qsched.setTokenExecutor([](const Request& request, ScheduledTokenPhase phase,
                                   int position, std::vector<float>& keys,
                                   std::vector<float>& values) {
            const float base = static_cast<float>(request.id + position +
                (phase == ScheduledTokenPhase::PREFILL ? 1 : 2));
            std::fill(keys.begin(), keys.end(), base);
            std::fill(values.begin(), values.end(), -base);
        });
        qsched.addRequest(1, 3, 2); qsched.addRequest(2, 2, 3); qsched.addRequest(3, 4, 1);
        int qsteps = 0;
        while (!qsched.allDone() && qsteps < 1000) { qsched.step(); ++qsteps; }
        check("scheduler drains over an int8 pool", qsched.allDone() && qsched.finishedCount() == 3);
        check("int8 pool fully reclaimed", qpool.freeBlocks() == q0);
    }

    // The scheduler uses model-produced K/V, reports absent/incomplete
    // executors without crossing a shared-library exception boundary, and
    // preserves the request so a failed prefill can be retried.
    {
        KVCacheManager pool(4, 4, 1, 2);
        ContinuousBatchScheduler sched(pool, 1);
        sched.addRequest(55, 2, 1);
        const int missingExecutorResult = sched.step();
        check("scheduler refuses token work without a model executor",
              missingExecutorResult == -1 &&
              sched.lastError() == SchedulerError::MISSING_EXECUTOR &&
              !sched.lastErrorMessage().empty() && pool.freeBlocks() == 4);

        sched.setTokenExecutor([](const Request&, ScheduledTokenPhase, int,
                                  std::vector<float>&, std::vector<float>&) {
            // Deliberately leave the NaN-initialized output untouched.
        });
        const int incompleteOutputResult = sched.step();
        check("scheduler rejects unset K/V output and rolls prefill back",
              incompleteOutputResult == -1 &&
              sched.lastError() == SchedulerError::INVALID_KV_OUTPUT &&
              !sched.lastErrorMessage().empty() && pool.freeBlocks() == 4 &&
              pool.seqLen(55) == 0 && sched.waitingCount() == 1);

        int prefillCalls = 0, decodeCalls = 0;
        sched.setTokenExecutor([&](const Request&, ScheduledTokenPhase phase, int position,
                                   std::vector<float>& keys, std::vector<float>& values) {
            if (phase == ScheduledTokenPhase::PREFILL) {
                ++prefillCalls;
                std::fill(keys.begin(), keys.end(), 10.0f + position);
                std::fill(values.begin(), values.end(), 20.0f + position);
            } else {
                ++decodeCalls;
                std::fill(keys.begin(), keys.end(), 30.0f + position);
                std::fill(values.begin(), values.end(), 40.0f + position);
            }
        });
        sched.step();
        const auto& blocks = pool.blockTable(55);
        float firstKey[2] = {}, firstValue[2] = {}, decodeKey[2] = {}, decodeValue[2] = {};
        pool.readKey(blocks[0], 0, 0, firstKey);
        pool.readVal(blocks[0], 0, 0, firstValue);
        pool.readKey(blocks[0], 2, 0, decodeKey);
        pool.readVal(blocks[0], 2, 0, decodeValue);
        check("scheduler appends model-produced prompt and decode K/V",
              prefillCalls == 2 && decodeCalls == 1 && pool.seqLen(55) == 3 &&
              firstKey[0] == 10.0f && firstValue[0] == 20.0f &&
              decodeKey[0] == 32.0f && decodeValue[0] == 42.0f);
        sched.step();
        check("successful request releases its reserved KV blocks",
              sched.allDone() && pool.freeBlocks() == 4);
    }

    // A shared cache manager can be used by other components. If an external
    // caller claims a queued ID's sequence before admission, the scheduler must
    // report the ownership conflict and leave that external reservation intact.
    {
        KVCacheManager pool(2, 1, 1, 1);
        ContinuousBatchScheduler sched(pool, 1);
        sched.setTokenExecutor([](const Request&, ScheduledTokenPhase, int,
                                  std::vector<float>& keys, std::vector<float>& values) {
            std::fill(keys.begin(), keys.end(), 1.0f);
            std::fill(values.begin(), values.end(), 2.0f);
        });
        sched.addRequest(88, 1, 0);
        const bool externalReservation = pool.reserveSequenceCapacity(88, 1);
        const int conflictedStep = sched.step();
        check("scheduler reports external KV sequence ownership conflict",
              externalReservation && conflictedStep == -1 &&
              sched.lastError() == SchedulerError::CACHE_RESERVATION_FAILURE &&
              sched.waitingCount() == 1 && pool.hasSequence(88) &&
              pool.freeBlocks() == 1);
        pool.freeSequence(88);
        const int retryStep = sched.step();
        sched.step();
        check("queued request retries after external KV reservation is released",
              retryStep == 1 && sched.allDone() && pool.freeBlocks() == 2);
    }

    // A model failure during a later request's prefill must leave bucket order
    // intact and keep already-running requests' reservations.
    {
        KVCacheManager pool(8, 4, 1, 2);
        ContinuousBatchScheduler sched(pool, 2);
        sched.setTokenExecutor([](const Request& request, ScheduledTokenPhase phase,
                                  int position, std::vector<float>& keys,
                                  std::vector<float>& values) {
            if (request.id == 2 && phase == ScheduledTokenPhase::PREFILL)
                throw std::runtime_error("model prefill failed");
            std::fill(keys.begin(), keys.end(), static_cast<float>(request.id + position));
            std::fill(values.begin(), values.end(), static_cast<float>(request.id - position));
        });
        sched.addRequest(1, 1, 1);
        sched.addRequest(2, 2, 1);
        sched.addRequest(3, 1, 1);
        const int failedPrefillResult = sched.step();
        check("failed prefill reports an error without corrupting scheduler queues",
              failedPrefillResult == -1 &&
              sched.lastError() == SchedulerError::EXECUTOR_FAILURE &&
              sched.lastErrorMessage() == "model prefill failed" &&
              sched.runningCount() == 1 && sched.waitingCount() == 2 &&
              pool.seqLen(1) == 1 && !pool.hasSequence(2));
        sched.setTokenExecutor([](const Request& request, ScheduledTokenPhase,
                                  int position, std::vector<float>& keys,
                                  std::vector<float>& values) {
            std::fill(keys.begin(), keys.end(), static_cast<float>(request.id + position));
            std::fill(values.begin(), values.end(), static_cast<float>(request.id - position));
        });
        int steps = 0;
        while (!sched.allDone() && steps++ < 10) sched.step();
        check("remaining requests drain after a model callback failure",
              sched.allDone() && sched.finishedCount() == 3 && pool.freeBlocks() == 8);
    }

    printf("\n%d / %d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
