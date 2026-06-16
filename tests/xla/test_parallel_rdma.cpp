// L5 over the real transport: tensor-parallel matmul whose all-gather /
// all-reduce run over SoftwareRDMA one-sided reads (real sockets, loopback), not
// in-process memcpy. Result must equal single-node, and real bytes must move.

#include "vgre/xla/parallel.h"
#include "vgre/xla/blas_gemm.h"
#include "vgre/xla/hlo.h"

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

static Literal refDot(const Literal& X, const Literal& W) {
    int64_t M = X.shape.dims[0], K = X.shape.dims[1], N = W.shape.dims[1];
    Literal y; y.shape.dims = {M, N}; y.data.resize((size_t)(M * N));
    gemm_f32(false, false, M, N, K, X.data.data(), W.data.data(), y.data.data(), 1);
    return y;
}
static double maxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0; for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs((double)a[i] - b[i])); return m;
}
static Literal randMat(int64_t r, int64_t c, uint32_t seed) {
    std::mt19937 rng(seed); std::uniform_real_distribution<float> d(-1.f, 1.f);
    Literal m; m.shape.dims = {r, c}; m.data.resize((size_t)(r * c));
    for (auto& v : m.data) v = d(rng);
    return m;
}

int main() {
    Literal X = randMat(6, 12, 2);    // [M=6, K=12]
    Literal W = randMat(12, 10, 3);   // [K=12, N=10]
    Literal ref = refDot(X, W);

    for (int R = 2; R <= 4; ++R) {
        uint64_t cbytes = 0, rbytes = 0;
        Literal cp = columnParallelMatmulRdma(X, W, R, &cbytes);
        CHECK(cp.shape.dims == ref.shape.dims, "column-parallel/RDMA shape");
        CHECK(maxAbsDiff(cp.data, ref.data) < 1e-4, "column-parallel over RDMA == single node");
        CHECK(cbytes >= (uint64_t)(X.shape.dims[0] * W.shape.dims[1] * sizeof(float)),
              "column-parallel moved the full output over RDMA");

        Literal rp = rowParallelMatmulRdma(X, W, R, &rbytes);
        CHECK(maxAbsDiff(rp.data, ref.data) < 1e-3, "row-parallel over RDMA == single node");
        CHECK(rbytes > 0, "row-parallel moved bytes over RDMA");

        std::printf("R=%d: col-parallel/RDMA moved %llu B, row-parallel/RDMA moved %llu B\n",
                    R, (unsigned long long)cbytes, (unsigned long long)rbytes);
    }

    // Uneven split (N=10 over 3 ranks) stays exact through the real transport.
    Literal cp3 = columnParallelMatmulRdma(X, W, 3, nullptr);
    CHECK(maxAbsDiff(cp3.data, ref.data) < 1e-4, "uneven column-parallel/RDMA exact");

    if (g_fail == 0) { std::printf("test_parallel_rdma: ALL CHECKS PASSED\n"); return 0; }
    std::printf("test_parallel_rdma: %d FAILURE(S)\n", g_fail);
    return 1;
}
