// Tensor- & pipeline-parallel execution — see include/vgre/xla/parallel.h.

#include "vgre/xla/parallel.h"

#include <stdexcept>

#include "vgre/xla/blas_gemm.h"

namespace vgre {
namespace xla {

namespace {
// Even-as-possible split of `total` into `ranks` parts; earlier parts get the
// remainder. Returns the [begin,end) boundary of each part.
std::vector<std::pair<int64_t, int64_t>> splitRange(int64_t total, int ranks) {
    std::vector<std::pair<int64_t, int64_t>> r;
    const int64_t base = total / ranks, rem = total % ranks;
    int64_t b = 0;
    for (int i = 0; i < ranks; ++i) {
        int64_t sz = base + (i < rem ? 1 : 0);
        r.emplace_back(b, b + sz);
        b += sz;
    }
    return r;
}
}  // namespace

Literal Communicator::allReduceSum(const std::vector<Literal>& perRank) {
    if (perRank.empty()) return Literal{};
    Literal out = perRank[0];  // copies shape + data
    for (size_t r = 1; r < perRank.size(); ++r) {
        if (perRank[r].data.size() != out.data.size())
            throw std::runtime_error("allReduceSum: mismatched shard sizes");
        for (size_t i = 0; i < out.data.size(); ++i) out.data[i] += perRank[r].data[i];
    }
    return out;
}

Literal Communicator::allGatherConcatLastAxis(const std::vector<Literal>& perRank) {
    if (perRank.empty()) return Literal{};
    const auto& dims0 = perRank[0].shape.dims;
    if (dims0.size() != 2) throw std::runtime_error("allGather expects 2-D [M,N_r]");
    const int64_t M = dims0[0];
    int64_t Ntot = 0;
    for (const auto& p : perRank) Ntot += p.shape.dims[1];

    Literal out;
    out.shape.dims = {M, Ntot};
    out.data.assign((size_t)(M * Ntot), 0.0f);
    int64_t colOff = 0;
    for (const auto& p : perRank) {
        const int64_t Nr = p.shape.dims[1];
        for (int64_t m = 0; m < M; ++m)
            for (int64_t n = 0; n < Nr; ++n)
                out.data[(size_t)(m * Ntot + colOff + n)] = p.data[(size_t)(m * Nr + n)];
        colOff += Nr;
    }
    return out;
}

std::vector<Literal> shardColumns(const Literal& W, int ranks) {
    const int64_t K = W.shape.dims[0], N = W.shape.dims[1];
    std::vector<Literal> out;
    for (auto [n0, n1] : splitRange(N, ranks)) {
        const int64_t Nr = n1 - n0;
        Literal s; s.shape.dims = {K, Nr}; s.data.resize((size_t)(K * Nr));
        for (int64_t k = 0; k < K; ++k)
            for (int64_t n = 0; n < Nr; ++n)
                s.data[(size_t)(k * Nr + n)] = W.data[(size_t)(k * N + n0 + n)];
        out.push_back(std::move(s));
    }
    return out;
}

std::vector<Literal> shardRows(const Literal& W, int ranks) {
    const int64_t K = W.shape.dims[0], N = W.shape.dims[1];
    std::vector<Literal> out;
    for (auto [k0, k1] : splitRange(K, ranks)) {
        const int64_t Kr = k1 - k0;
        Literal s; s.shape.dims = {Kr, N};
        s.data.assign(W.data.begin() + (size_t)(k0 * N), W.data.begin() + (size_t)(k1 * N));
        out.push_back(std::move(s));
    }
    return out;
}

std::vector<Literal> shardContracting(const Literal& X, int ranks) {
    const int64_t M = X.shape.dims[0], K = X.shape.dims[1];
    std::vector<Literal> out;
    for (auto [k0, k1] : splitRange(K, ranks)) {
        const int64_t Kr = k1 - k0;
        Literal s; s.shape.dims = {M, Kr}; s.data.resize((size_t)(M * Kr));
        for (int64_t m = 0; m < M; ++m)
            for (int64_t k = 0; k < Kr; ++k)
                s.data[(size_t)(m * Kr + k)] = X.data[(size_t)(m * K + k0 + k)];
        out.push_back(std::move(s));
    }
    return out;
}

Literal columnParallelMatmul(const Literal& X, const Literal& W, int ranks) {
    const int64_t M = X.shape.dims[0], K = X.shape.dims[1];
    std::vector<Literal> Wsh = shardColumns(W, ranks);
    std::vector<Literal> local;
    for (const auto& Wr : Wsh) {                       // each rank: X · W_r → [M, N_r]
        const int64_t Nr = Wr.shape.dims[1];
        Literal y; y.shape.dims = {M, Nr}; y.data.resize((size_t)(M * Nr));
        gemm_f32(false, false, M, Nr, K, X.data.data(), Wr.data.data(), y.data.data(), 1);
        local.push_back(std::move(y));
    }
    return Communicator::allGatherConcatLastAxis(local);  // concat along N
}

Literal rowParallelMatmul(const Literal& X, const Literal& W, int ranks) {
    const int64_t M = X.shape.dims[0], N = W.shape.dims[1];
    std::vector<Literal> Xsh = shardContracting(X, ranks);
    std::vector<Literal> Wsh = shardRows(W, ranks);
    std::vector<Literal> partials;
    for (int r = 0; r < ranks; ++r) {                  // each rank: X_r · W_r → [M,N]
        const int64_t Kr = Xsh[r].shape.dims[1];
        Literal y; y.shape.dims = {M, N}; y.data.resize((size_t)(M * N));
        gemm_f32(false, false, M, N, Kr, Xsh[r].data.data(), Wsh[r].data.data(), y.data.data(), 1);
        partials.push_back(std::move(y));
    }
    return Communicator::allReduceSum(partials);          // sum partial products
}

std::vector<std::pair<int, int>> pipelinePartition(int numLayers, int stages) {
    std::vector<std::pair<int, int>> out;
    if (stages <= 0) return out;
    const int base = numLayers / stages, rem = numLayers % stages;
    int b = 0;
    for (int i = 0; i < stages; ++i) {
        int sz = base + (i < rem ? 1 : 0);
        out.emplace_back(b, b + sz);
        b += sz;
    }
    return out;
}

}  // namespace xla
}  // namespace vgre
