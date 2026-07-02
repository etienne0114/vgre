// Test: from-scratch HNSW vector index (src/xla/vector_index.cpp).
//
// Verifies against brute-force ground truth — the ANN graph must actually
// navigate, not just compile:
//   1. recall@10 ≥ 0.95 on 2000 random 64-d vectors (cosine AND squared-L2)
//   2. exact self-retrieval: querying an indexed vector returns its own id first
//   3. save → load → identical search results
//   4. error paths: bad args, missing file

#include "vgre/xla/vector_index.h"

// Tests build in Release (-DNDEBUG); keep asserts REAL — several perform the
// actual operations under test.
#undef NDEBUG
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

constexpr int kN = 2000, kDim = 64, kQueries = 50, kK = 10;

float refDist(const float* a, const float* b, int metric) {
    if (metric == 0) {
        double da = 0, db = 0, dot = 0;
        for (int i = 0; i < kDim; ++i) { da += (double)a[i]*a[i]; db += (double)b[i]*b[i]; dot += (double)a[i]*b[i]; }
        return (float)(1.0 - dot / (std::sqrt(da) * std::sqrt(db) + 1e-30));
    }
    float s = 0;
    for (int i = 0; i < kDim; ++i) { float d = a[i]-b[i]; s += d*d; }
    return s;
}

// Brute-force top-k ids for a query.
std::vector<int64_t> bruteTopK(const std::vector<float>& base, const float* q, int metric) {
    std::vector<std::pair<float,int64_t>> all(kN);
    for (int i = 0; i < kN; ++i)
        all[i] = {refDist(base.data() + (size_t)i*kDim, q, metric), i};
    std::partial_sort(all.begin(), all.begin()+kK, all.end());
    std::vector<int64_t> ids(kK);
    for (int i = 0; i < kK; ++i) ids[i] = all[i].second;
    return ids;
}

double runRecall(int metric, vgre_vindex* outIx, std::vector<float>* outBase,
                 std::vector<float>* outQueries) {
    std::mt19937 rng(metric == 0 ? 42 : 43);
    std::normal_distribution<float> nd(0.f, 1.f);

    std::vector<float> base((size_t)kN * kDim), queries((size_t)kQueries * kDim);
    for (auto& v : base) v = nd(rng);
    for (auto& v : queries) v = nd(rng);

    vgre_vindex ix = vgre_vindex_create(kDim, 16, 200, metric);
    assert(ix);
    for (int i = 0; i < kN; ++i)
        assert(vgre_vindex_add(ix, base.data() + (size_t)i*kDim, i) == 0);
    assert(vgre_vindex_size(ix) == kN);
    assert(vgre_vindex_dim(ix) == kDim);

    int64_t ids[kK]; float dists[kK];
    int hit = 0, total = 0;
    for (int q = 0; q < kQueries; ++q) {
        const float* qp = queries.data() + (size_t)q*kDim;
        int n = vgre_vindex_search(ix, qp, kK, 100, ids, dists);
        assert(n == kK);
        for (int i = 1; i < n; ++i) assert(dists[i] >= dists[i-1]); // sorted
        auto truth = bruteTopK(base, qp, metric);
        for (int i = 0; i < n; ++i)
            if (std::find(truth.begin(), truth.end(), ids[i]) != truth.end()) ++hit;
        total += kK;
    }
    *outIx = ix; *outBase = std::move(base); *outQueries = std::move(queries);
    return (double)hit / total;
}

}  // namespace

int main() {
    std::printf("\n--- Test: HNSW vector index ---\n");

    // 1. Recall vs brute force, both metrics.
    for (int metric = 0; metric <= 1; ++metric) {
        vgre_vindex ix; std::vector<float> base, queries;
        double recall = runRecall(metric, &ix, &base, &queries);
        std::printf("[%s] recall@%d = %.3f (n=%d, dim=%d)\n",
                    metric == 0 ? "cosine" : "l2", kK, recall, kN, kDim);
        if (recall < 0.95) { std::printf("FAIL: recall too low\n"); return 1; }

        // 2. Self-retrieval: an indexed vector's nearest neighbor is itself.
        int64_t ids[kK]; float dists[kK];
        for (int i = 0; i < 20; ++i) {
            int probe = i * 97 % kN;
            int n = vgre_vindex_search(ix, base.data() + (size_t)probe*kDim,
                                       kK, 100, ids, dists);
            assert(n == kK);
            assert(ids[0] == probe);
            assert(dists[0] < 1e-5f);
        }

        // 3. save → load → identical results on 10 queries.
        char path[256];
        std::snprintf(path, sizeof(path), "vgre_vindex_test_%d.bin", metric);
        assert(vgre_vindex_save(ix, path) == 0);
        vgre_vindex ld = vgre_vindex_load(path);
        assert(ld && vgre_vindex_size(ld) == kN);
        for (int q = 0; q < 10; ++q) {
            const float* qp = queries.data() + (size_t)q*kDim;
            int64_t idA[kK], idB[kK]; float dA[kK], dB[kK];
            int nA = vgre_vindex_search(ix, qp, kK, 100, idA, dA);
            int nB = vgre_vindex_search(ld, qp, kK, 100, idB, dB);
            assert(nA == nB);
            for (int i = 0; i < nA; ++i) { assert(idA[i] == idB[i]); assert(dA[i] == dB[i]); }
        }
        vgre_vindex_free(ld);
        vgre_vindex_free(ix);
        std::remove(path);
        std::printf("[PASS] %s: self-retrieval + save/load roundtrip\n",
                    metric == 0 ? "cosine" : "l2");
    }

    // 4. Error paths.
    assert(vgre_vindex_create(0, 16, 200, 0) == nullptr);
    assert(vgre_vindex_create(8, 16, 200, 7) == nullptr);
    assert(vgre_vindex_add(nullptr, nullptr, 0) != 0);
    assert(vgre_vindex_load("/nonexistent/vgre_vx.bin") == nullptr);
    assert(vgre_vindex_size(nullptr) == 0);
    std::printf("[PASS] error paths\n");

    std::printf("[PASS] HNSW vector index\n");
    return 0;
}
