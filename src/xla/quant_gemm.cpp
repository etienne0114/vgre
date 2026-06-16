// Dequant-inside-the-GEMM — see include/vgre/xla/quant_gemm.h.

#include "vgre/xla/quant_gemm.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <vector>

#include "vgre/xla/quant.h"
#include "vgre/xla/thread_pool.h"

namespace vgre {
namespace xla {

bool gemm_q(int64_t M, int64_t N, int64_t K, const float* X,
            const uint8_t* Wq, int ggml_type, float* C) {
    const int be = quantBlockElems(ggml_type);
    const int bb = quantBlockBytes(ggml_type);
    if (bb == 0 || N % be != 0) return false;

    const int64_t nblk = N / be;          // column-blocks per W row
    const int64_t rowBytes = nblk * bb;   // bytes per quantized W row
    std::memset(C, 0, sizeof(float) * (size_t)(M * N));
    if (M <= 0 || K <= 0) return true;

    // One unit of work = a contiguous range of column-blocks [b0,b1). For each
    // such range we dequantize that slice of each W row (k) once and apply the
    // rank-1 update to the disjoint C columns — no redundant dequant, no write
    // contention between units.
    auto runBlocks = [&](int64_t b0, int64_t b1) {
        const int64_t c0 = b0 * be, width = (b1 - b0) * be;
        std::vector<float> wrow((size_t)width);
        for (int64_t k = 0; k < K; ++k) {
            dequantBlock(ggml_type, Wq + k * rowBytes + b0 * bb, width, wrow.data());
            for (int64_t m = 0; m < M; ++m) {
                const float xmk = X[(size_t)(m * K + k)];
                if (xmk == 0.0f) continue;
                float* crow = C + (size_t)(m * N + c0);
                for (int64_t c = 0; c < width; ++c) crow[c] += xmk * wrow[(size_t)c];
            }
        }
    };

    ThreadPool& pool = ThreadPool::global();
    const int64_t P = (int64_t)pool.concurrency();
    const int64_t flops = M * N * K;
    if (P < 2 || nblk < 2 || flops < (1 << 16)) {
        runBlocks(0, nblk);
        return true;
    }
    // Split the column-blocks across the pool.
    const int64_t chunks = P * 4;
    const int64_t grain = (nblk + chunks - 1) / chunks;
    const int64_t nChunks = (nblk + grain - 1) / grain;
    std::function<void(int64_t)> body = [&](int64_t ci) {
        runBlocks(ci * grain, std::min(ci * grain + grain, nblk));
    };
    pool.parallelFor(nChunks, 1, body);
    return true;
}

}  // namespace xla
}  // namespace vgre
