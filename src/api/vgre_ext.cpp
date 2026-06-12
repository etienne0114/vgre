// VGRE Phase-3 extension API — compiles the Phase-3 capability modules into
// libvgre and exposes each as a stable public entry point (declared in
// vgre/api/vgre_ext.h). This is the production wiring for what were previously
// header-only modules exercised only by unit tests.

#include "vgre/api/vgre_ext.h"

#include "vgre/core/occupancy.h"
#include "vgre/core/interconnect.h"
#include "vgre/core/deterministic.h"
#include "vgre/core/ssm.h"
#include "vgre/core/moe.h"
#include "vgre/core/speculative.h"
#include "vgre/core/tensor_parallel.h"
#include "vgre/core/attention_masks.h"
#include "vgre/core/nvshmem.h"
#include "vgre/core/prefix_cache.h"
#include "vgre/core/gradient_checkpointing.h"
#include "vgre/core/tma.h"
#include "vgre/core/math/mx_quant.h"
#include "vgre/core/math/int4_weight_quant.h"

#include <cstdint>
#include <random>
#include <vector>

namespace vgre {
namespace ext {

int occupancy_max_active_blocks(int threadsPerBlock, int regsPerThread, int smemPerBlock) {
    perf::SMLimits sm;
    perf::KernelUsage k{threadsPerBlock, regsPerThread, smemPerBlock};
    return perf::compute_occupancy(sm, k).activeBlocksPerSM;
}
double roofline_gflops(double intensity, double peakGflops, double peakGBps) {
    return perf::roofline_gflops(intensity, peakGflops, peakGBps);
}

double allreduce_us(int numGpus, int domainSize, double nvlinkGBps, double pcieGBps, double bytes) {
    dist::Topology t;
    t.numGpus = numGpus; t.domainSize = domainSize;
    t.nvlinkGBps = nvlinkGBps; t.pcieGBps = pcieGBps;
    return dist::best_allreduce_us(t, bytes);
}

float deterministic_reduce(const float* x, size_t n) { return det::deterministic_reduce(x, n); }
float counter_rng_u01(uint64_t seed, uint64_t counter) { return det::counter_u01(seed, counter); }

void ssm_selective_scan(float* y, const float* Abar, const float* Bx, const float* C, int L, int N) {
    ssm::selective_scan_parallel(y, Abar, Bx, C, L, N);
}

void moe_forward(float* Y, const float* X, const float* logits, const float* We,
                 int T, int E, int k, int D, int H) {
    moe::Routing r = moe::moe_route(logits, T, E, k);
    moe::moe_forward_grouped(Y, X, We, r, D, H);
}

int speculative_decode(int* out, const float* targetProb, const float* draftProb,
                       int K, int V, uint64_t seed) {
    std::mt19937 rng(static_cast<uint32_t>(seed));
    return serving::speculative_decode(out, targetProb, draftProb, K, V, rng);
}

void tensor_parallel_column_linear(float* Y, const float* X, const float* W,
                                   int T, int Din, int Dout, int P) {
    dist::column_parallel_linear(Y, X, W, T, Din, Dout, P);
}

void w4a16_gemm(float* D, const float* A, const float* W, int M, int N, int K, int group) {
    quant::W4A16Weight w = quant::w4a16_quantize(W, K, N, group);
    quant::w4a16_gemm(D, A, w, M, N);
}

void mx_gemm(float* D, const float* A, const float* B, int M, int N, int K, int elem, int block) {
    quant::MXElem fmt = (elem == 1) ? quant::MXElem::FP8_E4M3
                      : (elem == 2) ? quant::MXElem::FP8_E5M2
                                    : quant::MXElem::FP4_E2M1;
    quant::mx_gemm(D, A, B, M, N, K, fmt, block);
}

void attention_masked(float* O, const float* Q, const float* K, const float* V,
                      int S, int d, float scale, int maskKind, int window, int sink) {
    attn::MaskConfig cfg;
    cfg.kind   = (maskKind == 1) ? attn::MaskKind::SLIDING_WINDOW
               : (maskKind == 2) ? attn::MaskKind::SINK_WINDOW
                                 : attn::MaskKind::CAUSAL;
    cfg.window = window;
    cfg.sink   = sink;
    attn::attention_masked(O, Q, K, V, S, d, scale, cfg);
}

void nvshmem_allreduce_sum(float* perPe, int nPes, int count) {
    if (nPes <= 0 || count <= 0) return;
    const size_t bytes = static_cast<size_t>(count) * sizeof(float);
    dist::SymmetricHeap sh(nPes, 2 * bytes);                 // [data | tmp] per PE
    for (int p = 0; p < nPes; ++p)
        sh.put(p, 0, &perPe[static_cast<size_t>(p) * count], bytes);
    dist::nvshmem_allreduce_sum(sh, /*dataOff=*/0, /*tmpOff=*/bytes, count);
    for (int p = 0; p < nPes; ++p)
        sh.get(&perPe[static_cast<size_t>(p) * count], p, 0, bytes);
}

int prefix_cache_block(const float* content, int floatsPerBlock, int* wasShared) {
    serving::PrefixCache cache(8, floatsPerBlock);
    const uint64_t h = serving::hash_block(content, floatsPerBlock);
    int b1 = cache.getOrCreate(h, content);
    int b2 = cache.getOrCreate(h, content);                  // identical content → shared block
    if (wasShared) *wasShared = (b1 == b2) ? 1 : 0;
    return b1;
}

bool gradient_checkpoint_roundtrip(const float* activation, unsigned numElements) {
    vgre::ActivationCheckpoint cp;
    cp.save(activation, numElements);
    std::vector<float> back(numElements, 0.0f);
    cp.load(back.data());
    for (unsigned i = 0; i < numElements; ++i)
        if (back[i] != activation[i]) return false;
    return true;
}

void tma_box_copy(float* dst, const float* tensor, int H, int W, int bh, int bw, int row, int col) {
    uint64_t dim[2]    = {static_cast<uint64_t>(H), static_cast<uint64_t>(W)};
    uint64_t stride[2] = {static_cast<uint64_t>(W) * sizeof(float), sizeof(float)};
    uint32_t box[2]    = {static_cast<uint32_t>(bh), static_cast<uint32_t>(bw)};
    tma::TensorMap tm = tma::tensor_map_encode_tiled(tensor, 2, dim, stride, box, sizeof(float));
    tma::Mbarrier mb; tma::mbarrier_init(mb); tma::mbarrier_expect_tx(mb, tma::box_bytes(tm));
    int32_t coord[2] = {row, col};
    tma::cp_async_bulk_tensor(dst, tm, coord, &mb);
}

}  // namespace ext
}  // namespace vgre
