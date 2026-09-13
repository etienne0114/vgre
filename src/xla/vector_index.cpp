// From-scratch HNSW (Hierarchical Navigable Small World) ANN index.
// See include/vgre/xla/vector_index.h for the API contract.
//
// This is the real Malkov–Yashunin algorithm (arXiv:1603.09320), not a wrapper:
//   • geometric level assignment  L = ⌊−ln(U)·mL⌋,  mL = 1/ln(M)
//   • best-first beam search per layer (Algorithm 2) with a visited set
//   • neighbor selection by the pruning heuristic (Algorithm 4): a candidate
//     is kept only if it is closer to the query than to every already-kept
//     neighbor — this is what creates the navigable long-range links
//   • bidirectional linking with degree caps M (upper layers) / 2M (layer 0)
//
// Single-writer thread safety via one mutex (searches also take it — CPU-local
// RAG workloads are read-mostly and the critical sections are short).

#include "vgre/xla/vector_index.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <queue>
#include <random>
#include <vector>

namespace {

struct Node {
    int64_t id;
    int     level;
    // links[l] = neighbor node indices at layer l (0..level)
    std::vector<std::vector<uint32_t>> links;
};

struct HnswIndex {
    std::mutex mu;
    int dim = 0;
    int M = 16;                 // degree target (upper layers)
    int Mmax0 = 32;             // degree cap at layer 0 (2M)
    int efC = 200;              // construction beam width
    int metric = 0;             // 0 = cosine (normalized, 1-dot), 1 = squared L2
    double mL = 0.0;            // level multiplier 1/ln(M)
    int maxLevel = -1;
    uint32_t entry = 0;         // entry point node index (valid when count > 0)
    std::vector<Node> nodes;
    std::vector<float> data;    // node i's vector at data[i*dim .. i*dim+dim)
    std::mt19937_64 rng{0x9E3779B97F4A7C15ull};

    const float* vec(uint32_t i) const { return data.data() + (size_t)i * dim; }

    float dist(const float* a, const float* b) const {
        if (metric == 0) {                       // cosine on normalized vectors
            float dot = 0.f;
            for (int i = 0; i < dim; ++i) dot += a[i] * b[i];
            return 1.0f - dot;
        }
        float s = 0.f;                           // squared L2
        for (int i = 0; i < dim; ++i) { float d = a[i] - b[i]; s += d * d; }
        return s;
    }

    int sampleLevel() {
        std::uniform_real_distribution<double> u(0.0, 1.0);
        double r = u(rng);
        if (r < 1e-300) r = 1e-300;
        return (int)(-std::log(r) * mL);
    }

    // ── Algorithm 2: best-first search on one layer ──────────────────────────
    // Returns up to `ef` (dist, node) pairs, unsorted (heap order).
    using DN = std::pair<float, uint32_t>;       // (distance, node index)

    std::vector<DN> searchLayer(const float* q, uint32_t ep, int ef, int layer,
                                std::vector<uint32_t>& visitedMark,
                                uint32_t visitEpoch) const {
        std::priority_queue<DN, std::vector<DN>, std::greater<DN>> cand; // min
        std::priority_queue<DN> best;                                     // max
        float d0 = dist(q, vec(ep));
        cand.push({d0, ep});
        best.push({d0, ep});
        visitedMark[ep] = visitEpoch;

        while (!cand.empty()) {
            auto [dc, c] = cand.top();
            if (dc > best.top().first && (int)best.size() >= ef) break;
            cand.pop();
            for (uint32_t nb : nodes[c].links[layer]) {
                if (visitedMark[nb] == visitEpoch) continue;
                visitedMark[nb] = visitEpoch;
                float d = dist(q, vec(nb));
                if ((int)best.size() < ef || d < best.top().first) {
                    cand.push({d, nb});
                    best.push({d, nb});
                    if ((int)best.size() > ef) best.pop();
                }
            }
        }
        std::vector<DN> out;
        out.reserve(best.size());
        while (!best.empty()) { out.push_back(best.top()); best.pop(); }
        return out;                              // farthest-first order
    }

    // ── Algorithm 4: neighbor-selection heuristic ────────────────────────────
    // `cands` sorted ascending by distance to the query point.
    std::vector<uint32_t> selectNeighbors(const std::vector<DN>& cands, int m) const {
        std::vector<uint32_t> sel;
        sel.reserve(m);
        for (const auto& [dq, c] : cands) {
            if ((int)sel.size() >= m) break;
            bool keep = true;
            for (uint32_t s : sel) {
                if (dist(vec(c), vec(s)) < dq) { keep = false; break; }
            }
            if (keep) sel.push_back(c);
        }
        // Backfill with nearest pruned candidates if the heuristic was too
        // aggressive (keepPrunedConnections in the paper).
        if ((int)sel.size() < m) {
            for (const auto& [dq, c] : cands) {
                (void)dq;
                if ((int)sel.size() >= m) break;
                if (std::find(sel.begin(), sel.end(), c) == sel.end())
                    sel.push_back(c);
            }
        }
        return sel;
    }

    void pruneNode(uint32_t n, int layer) {
        const int cap = (layer == 0) ? Mmax0 : M;
        auto& ls = nodes[n].links[layer];
        if ((int)ls.size() <= cap) return;
        std::vector<DN> cands;
        cands.reserve(ls.size());
        for (uint32_t nb : ls) cands.push_back({dist(vec(n), vec(nb)), nb});
        std::sort(cands.begin(), cands.end());
        ls = selectNeighbors(cands, cap);
    }

    void insert(const float* v, int64_t id) {
        uint32_t idx = (uint32_t)nodes.size();
        int level = sampleLevel();

        data.insert(data.end(), v, v + dim);
        Node node;
        node.id = id;
        node.level = level;
        node.links.resize(level + 1);
        nodes.push_back(std::move(node));

        if (idx == 0) { entry = 0; maxLevel = level; return; }

        std::vector<uint32_t> visited(nodes.size(), 0);
        uint32_t epoch = 0;
        uint32_t ep = entry;

        // Greedy descent through layers above the new node's level (ef = 1).
        for (int l = maxLevel; l > level; --l) {
            bool moved = true;
            float dEp = dist(v, vec(ep));
            while (moved) {
                moved = false;
                for (uint32_t nb : nodes[ep].links[l]) {
                    float d = dist(v, vec(nb));
                    if (d < dEp) { dEp = d; ep = nb; moved = true; }
                }
            }
        }

        // Beam search + heuristic linking on each layer ≤ min(level, maxLevel).
        for (int l = std::min(level, maxLevel); l >= 0; --l) {
            ++epoch;
            auto found = searchLayer(v, ep, efC, l, visited, epoch);
            std::sort(found.begin(), found.end());          // ascending distance
            auto neigh = selectNeighbors(found, M);
            for (uint32_t nb : neigh) {
                nodes[idx].links[l].push_back(nb);
                nodes[nb].links[l].push_back(idx);
                pruneNode(nb, l);
            }
            if (!found.empty()) ep = found.front().second;   // nearest as next entry
        }

        if (level > maxLevel) { maxLevel = level; entry = idx; }
    }

    int search(const float* q, int k, int ef, int64_t* outIds, float* outDists) {
        if (nodes.empty() || k <= 0) return 0;
        if (ef <= 0) ef = std::max(64, k);
        if (ef < k) ef = k;

        std::vector<uint32_t> visited(nodes.size(), 0);
        uint32_t ep = entry;
        for (int l = maxLevel; l > 0; --l) {                 // greedy descent
            bool moved = true;
            float dEp = dist(q, vec(ep));
            while (moved) {
                moved = false;
                for (uint32_t nb : nodes[ep].links[l]) {
                    float d = dist(q, vec(nb));
                    if (d < dEp) { dEp = d; ep = nb; moved = true; }
                }
            }
        }
        auto found = searchLayer(q, ep, ef, 0, visited, 1);
        std::sort(found.begin(), found.end());
        int n = std::min<int>(k, (int)found.size());
        for (int i = 0; i < n; ++i) {
            outIds[i] = nodes[found[i].second].id;
            if (outDists) outDists[i] = found[i].first;
        }
        return n;
    }
};

void normalize(float* v, int dim) {
    double s = 0.0;
    for (int i = 0; i < dim; ++i) s += (double)v[i] * v[i];
    if (s <= 0.0) return;
    float inv = (float)(1.0 / std::sqrt(s));
    for (int i = 0; i < dim; ++i) v[i] *= inv;
}

// ── Portable little-endian binary persistence ────────────────────────────────
constexpr uint32_t kMagic = 0x58564756u;  // "VGVX"
constexpr uint32_t kVersion = 1;

template <typename T>
bool wr(FILE* f, const T& v) { return std::fwrite(&v, sizeof(T), 1, f) == 1; }
template <typename T>
bool rd(FILE* f, T& v) { return std::fread(&v, sizeof(T), 1, f) == 1; }

}  // namespace

extern "C" {

vgre_vindex vgre_vindex_create(int dim, int M, int ef_construction, int metric) {
    if (dim <= 0 || metric < 0 || metric > 1) return nullptr;
    auto* ix = new HnswIndex();
    ix->dim = dim;
    ix->M = (M > 1) ? M : 16;
    ix->Mmax0 = 2 * ix->M;
    ix->efC = (ef_construction > 0) ? ef_construction : 200;
    ix->metric = metric;
    ix->mL = 1.0 / std::log((double)ix->M);
    return ix;
}

int vgre_vindex_add(vgre_vindex h, const float* vec, int64_t id) {
    if (!h || !vec) return -2;
    auto* ix = static_cast<HnswIndex*>(h);
    std::lock_guard<std::mutex> lk(ix->mu);
    std::vector<float> v(vec, vec + ix->dim);
    if (ix->metric == 0) normalize(v.data(), ix->dim);
    ix->insert(v.data(), id);
    return 0;
}

int vgre_vindex_search(vgre_vindex h, const float* query, int k, int ef,
                       int64_t* out_ids, float* out_dists) {
    if (!h || !query || !out_ids) return -2;
    auto* ix = static_cast<HnswIndex*>(h);
    std::lock_guard<std::mutex> lk(ix->mu);
    std::vector<float> q(query, query + ix->dim);
    if (ix->metric == 0) normalize(q.data(), ix->dim);
    return ix->search(q.data(), k, ef, out_ids, out_dists);
}

int64_t vgre_vindex_size(vgre_vindex h) {
    if (!h) return 0;
    auto* ix = static_cast<HnswIndex*>(h);
    std::lock_guard<std::mutex> lk(ix->mu);
    return (int64_t)ix->nodes.size();
}

int vgre_vindex_dim(vgre_vindex h) {
    return h ? static_cast<HnswIndex*>(h)->dim : 0;
}

int vgre_vindex_save(vgre_vindex h, const char* path) {
    if (!h || !path) return -2;
    auto* ix = static_cast<HnswIndex*>(h);
    std::lock_guard<std::mutex> lk(ix->mu);
    FILE* f = std::fopen(path, "wb");
    if (!f) return -6;
    bool ok = wr(f, kMagic) && wr(f, kVersion) && wr(f, (int32_t)ix->dim) &&
              wr(f, (int32_t)ix->M) && wr(f, (int32_t)ix->efC) &&
              wr(f, (int32_t)ix->metric) && wr(f, (int32_t)ix->maxLevel) &&
              wr(f, (uint32_t)ix->entry) && wr(f, (uint64_t)ix->nodes.size());
    for (size_t i = 0; ok && i < ix->nodes.size(); ++i) {
        const Node& n = ix->nodes[i];
        ok = wr(f, n.id) && wr(f, (int32_t)n.level) &&
             std::fwrite(ix->vec((uint32_t)i), sizeof(float), ix->dim, f) ==
                 (size_t)ix->dim;
        for (int l = 0; ok && l <= n.level; ++l) {
            ok = wr(f, (uint32_t)n.links[l].size()) &&
                 (n.links[l].empty() ||
                  std::fwrite(n.links[l].data(), sizeof(uint32_t),
                              n.links[l].size(), f) == n.links[l].size());
        }
    }
    std::fclose(f);
    return ok ? 0 : -6;
}

vgre_vindex vgre_vindex_load(const char* path) {
    if (!path) return nullptr;
    FILE* f = std::fopen(path, "rb");
    if (!f) return nullptr;
    uint32_t magic = 0, version = 0, entry = 0;
    int32_t dim = 0, M = 0, efC = 0, metric = 0, maxLevel = 0;
    uint64_t count = 0;
    bool ok = rd(f, magic) && magic == kMagic && rd(f, version) &&
              version == kVersion && rd(f, dim) && rd(f, M) && rd(f, efC) &&
              rd(f, metric) && rd(f, maxLevel) && rd(f, entry) && rd(f, count);
    if (!ok || dim <= 0 || M <= 1 || count > (uint64_t)1 << 40) {
        std::fclose(f);
        return nullptr;
    }
    auto* ix = static_cast<HnswIndex*>(vgre_vindex_create(dim, M, efC, metric));
    ix->maxLevel = maxLevel;
    ix->entry = entry;
    ix->nodes.reserve(count);
    ix->data.reserve(count * (size_t)dim);
    for (uint64_t i = 0; ok && i < count; ++i) {
        Node n;
        int32_t level = 0;
        ok = rd(f, n.id) && rd(f, level) && level >= 0 && level < 64;
        if (!ok) break;
        n.level = level;
        n.links.resize(level + 1);
        size_t off = ix->data.size();
        ix->data.resize(off + dim);
        ok = std::fread(ix->data.data() + off, sizeof(float), dim, f) ==
             (size_t)dim;
        for (int l = 0; ok && l <= level; ++l) {
            uint32_t cnt = 0;
            ok = rd(f, cnt) && cnt <= count;
            if (ok && cnt) {
                n.links[l].resize(cnt);
                ok = std::fread(n.links[l].data(), sizeof(uint32_t), cnt, f) == cnt;
            }
        }
        ix->nodes.push_back(std::move(n));
    }
    std::fclose(f);
    if (!ok || ix->nodes.size() != count ||
        (count > 0 && entry >= ix->nodes.size())) {
        delete ix;
        return nullptr;
    }
    return ix;
}

void vgre_vindex_free(vgre_vindex h) { delete static_cast<HnswIndex*>(h); }

}  // extern "C"
