// Track 7 — Flash Attention tiled online-softmax correctness.
//
// JIT-compiles and runs the corrected FlashAttention-2 online-softmax kernel
// (identical to what KernelFusionEngine::genFlashAttentionSource emits for the
// standard MHA path) and compares its output, element-by-element, against a
// naive full-softmax attention reference. The previous kernel failed to rescale
// the output accumulator on a running-max update (and shared softmax state
// across rows), so its result diverged from the reference.

#include "vgre/core/runtime_engine.h"
#include "vgre/core/memory_manager.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace vgre;
using namespace vgre::core;

static int g_pass = 0, g_total = 0;
static void check(const char *name, bool ok) {
    ++g_total;
    printf(ok ? "  PASS  %s\n" : "  FAIL  %s\n", name);
    if (ok) ++g_pass;
}

// The exact MHA body emitted by genFlashAttentionSource (causal toggled by the
// `MASK` macro the generator bakes in; here we parameterise two kernels).
static const char *kFlashSrc = R"SRC(
extern "C" __global__ void FLASH_FN(
    const float* __restrict__ Q, const float* __restrict__ K,
    const float* __restrict__ V, float* __restrict__ O,
    int seq_len, int d_head, float scale) {
    const int BK = 32;
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= seq_len) return;
    float acc[128];
    for (int d = 0; d < d_head && d < 128; ++d) acc[d] = 0.0f;
    float row_max = -3.402823466e+38f, row_sum = 0.0f;
    for (int kv_blk = 0; kv_blk < (seq_len + BK - 1) / BK; ++kv_blk) {
        const int kv_start = kv_blk * BK;
        float s_vals[32]; float local_max = -3.402823466e+38f;
        for (int k = 0; k < BK; ++k) {
            const int col = kv_start + k;
            if (col >= seq_len) { s_vals[k] = -3.402823466e+38f; continue; }
            if (CAUSAL_MASK) { s_vals[k] = -3.402823466e+38f; continue; }
            float dot = 0.0f;
            for (int d = 0; d < d_head && d < 128; ++d)
                dot += Q[row * d_head + d] * K[col * d_head + d];
            s_vals[k] = dot * scale;
            if (s_vals[k] > local_max) local_max = s_vals[k];
        }
        const float new_max = (row_max > local_max) ? row_max : local_max;
        const float exp_scale = expf(row_max - new_max);
        for (int d = 0; d < d_head && d < 128; ++d) acc[d] *= exp_scale;
        row_sum *= exp_scale;
        for (int k = 0; k < BK; ++k) {
            const int col = kv_start + k;
            if (col >= seq_len) continue;
            if (s_vals[k] <= -3.402823466e+38f / 2.0f) continue;
            const float e = expf(s_vals[k] - new_max);
            row_sum += e;
            for (int d = 0; d < d_head && d < 128; ++d)
                acc[d] += e * V[col * d_head + d];
        }
        row_max = new_max;
    }
    if (row_sum > 0.0f)
        for (int d = 0; d < d_head && d < 128; ++d)
            O[row * d_head + d] = acc[d] / row_sum;
}
)SRC";

// Naive reference: full softmax attention.
static std::vector<float> naiveAttention(const std::vector<float> &Q,
                                         const std::vector<float> &K,
                                         const std::vector<float> &V,
                                         int n, int d, float scale, bool causal) {
    std::vector<float> O(n * d, 0.0f);
    for (int i = 0; i < n; ++i) {
        std::vector<float> s(n, 0.0f);
        float mx = -1e30f;
        for (int j = 0; j < n; ++j) {
            if (causal && j > i) { s[j] = -1e30f; continue; }
            float dot = 0.0f;
            for (int t = 0; t < d; ++t) dot += Q[i * d + t] * K[j * d + t];
            s[j] = dot * scale;
            if (s[j] > mx) mx = s[j];
        }
        float sum = 0.0f;
        for (int j = 0; j < n; ++j) { if (s[j] > -1e29f) { s[j] = std::exp(s[j] - mx); sum += s[j]; } else s[j] = 0.0f; }
        for (int j = 0; j < n; ++j) {
            float w = sum > 0 ? s[j] / sum : 0.0f;
            for (int t = 0; t < d; ++t) O[i * d + t] += w * V[j * d + t];
        }
    }
    return O;
}

static bool runCase(bool causal) {
    const int n = 24, d = 16;          // seq_len, d_head
    const float scale = 1.0f / std::sqrt((float)d);
    std::mt19937 rng(causal ? 7 : 3);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> Q(n * d), K(n * d), V(n * d), O(n * d, -1.0f);
    for (auto &x : Q) x = nd(rng);
    for (auto &x : K) x = nd(rng);
    for (auto &x : V) x = nd(rng);

    std::string src = kFlashSrc;
    const std::string mask = causal ? "col > row" : "false";
    for (size_t p; (p = src.find("CAUSAL_MASK")) != std::string::npos;)
        src.replace(p, 11, mask);
    const std::string fn = causal ? "flash_attn_causal" : "flash_attn_mha";
    for (size_t p; (p = src.find("FLASH_FN")) != std::string::npos;)
        src.replace(p, 8, fn);  // function name must match the registration name

    KernelId k = 0;
    if (RuntimeEngine::instance().registerKernel(fn, src, k) !=
            VGREResult::SUCCESS || k == 0)
        return false;

    // Device buffers (VGRE maps kernel pointer args to MemoryHandles).
    auto &mm = RuntimeEngine::instance().getMemoryManager();
    MemoryHandle dQ = nullptr, dK = nullptr, dV = nullptr, dO = nullptr;
    const size_t bytes = (size_t)n * d * sizeof(float);
    mm.allocate(bytes, dQ); mm.allocate(bytes, dK);
    mm.allocate(bytes, dV); mm.allocate(bytes, dO);
    mm.copyHostToDevice(dQ, Q.data(), bytes);
    mm.copyHostToDevice(dK, K.data(), bytes);
    mm.copyHostToDevice(dV, V.data(), bytes);

    int seq = n, dh = d; float sc = scale;
    void *args[] = {&dQ, &dK, &dV, &dO, &seq, &dh, &sc};
    dim3 grid((unsigned)n, 1, 1), block(1, 1, 1);   // one query row per block
    RuntimeEngine::instance().launchKernel(k, grid, block, args, 0, 0, dim3(0, 0, 0));
    RuntimeEngine::instance().synchronize();
    mm.copyDeviceToHost(O.data(), dO, bytes);

    std::vector<float> ref = naiveAttention(Q, K, V, n, d, scale, causal);
    double maxErr = 0.0; int at = 0;
    for (int i = 0; i < n * d; ++i) {
        double e = std::fabs(O[i] - ref[i]);
        if (e > maxErr) { maxErr = e; at = i; }
    }
    printf("  [info] %s max abs err vs naive = %.2e at idx %d (row %d,d %d): O=%.4f ref=%.4f | O[0..2]=%.3f,%.3f,%.3f\n",
           causal ? "causal" : "mha", maxErr, at, at / d, at % d, O[at], ref[at], O[0], O[1], O[2]);
    return maxErr < 1e-4;
}

// ── GQA path (grouped-query attention: fewer K/V heads than Q heads) ─────────
// Corrected GQA kernel as emitted by genFlashAttentionSource(..., gqa=true).
static const char *kGqaSrc = R"SRC(
extern "C" __global__ void flash_attn_gqa(
    const float* __restrict__ Q, const float* __restrict__ K,
    const float* __restrict__ V, float* __restrict__ O,
    int batch, int num_heads, int num_kv_heads,
    int seq_len, int d_head, float scale) {
    const int BM = 64, BK = 32;
    const int tid = threadIdx.x;
    const int gid = blockIdx.x;
    const int num_q_blocks = (seq_len + BM - 1) / BM;
    const int total_heads_blocks = num_heads * num_q_blocks;
    const int b = gid / total_heads_blocks;
    const int rem = gid % total_heads_blocks;
    const int q_head = rem / num_q_blocks;
    const int q_blk = rem % num_q_blocks;
    if (b >= batch || q_head >= num_heads) return;
    const int groups = num_heads / num_kv_heads;
    const int kv_head = q_head / groups;
    const int q_start = q_blk * BM;
    const float* Qh = Q + (b * num_heads + q_head) * seq_len * d_head;
    const float* Kh = K + (b * num_kv_heads + kv_head) * seq_len * d_head;
    const float* Vh = V + (b * num_kv_heads + kv_head) * seq_len * d_head;
    float* Oh = O + (b * num_heads + q_head) * seq_len * d_head;
    const int row = q_start + tid;
    if (row >= seq_len) return;
    float acc[128];
    for (int d = 0; d < d_head && d < 128; ++d) acc[d] = 0.0f;
    float row_max = -3.402823466e+38f, row_sum = 0.0f;
    for (int kv_blk = 0; kv_blk < (seq_len + BK - 1) / BK; ++kv_blk) {
        const int kv_start = kv_blk * BK;
        float s_vals[32]; float local_max = -3.402823466e+38f;
        for (int k = 0; k < BK; ++k) {
            const int col = kv_start + k;
            if (col >= seq_len) { s_vals[k] = -3.402823466e+38f; continue; }
            float dot = 0.0f;
            for (int d = 0; d < d_head && d < 128; ++d)
                dot += Qh[row * d_head + d] * Kh[col * d_head + d];
            s_vals[k] = dot * scale;
            if (s_vals[k] > local_max) local_max = s_vals[k];
        }
        const float new_max = (row_max > local_max) ? row_max : local_max;
        const float exp_scale = expf(row_max - new_max);
        for (int d = 0; d < d_head && d < 128; ++d) acc[d] *= exp_scale;
        row_sum *= exp_scale;
        for (int k = 0; k < BK; ++k) {
            const int col = kv_start + k;
            if (col >= seq_len) continue;
            if (s_vals[k] > -3.402823466e+38f / 2.0f) {
                const float e = expf(s_vals[k] - new_max);
                row_sum += e;
                for (int d = 0; d < d_head && d < 128; ++d)
                    acc[d] += e * Vh[col * d_head + d];
            }
        }
        row_max = new_max;
    }
    if (row_sum > 0.0f)
        for (int d = 0; d < d_head && d < 128; ++d)
            Oh[row * d_head + d] = acc[d] / row_sum;
}
)SRC";

static bool runGqa() {
    const int batch = 1, num_heads = 4, num_kv_heads = 2, n = 12, d = 8;
    const float scale = 1.0f / std::sqrt((float)d);
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> Q((size_t)num_heads * n * d), K((size_t)num_kv_heads * n * d),
        V((size_t)num_kv_heads * n * d), O((size_t)num_heads * n * d, -1.0f);
    for (auto &x : Q) x = nd(rng);
    for (auto &x : K) x = nd(rng);
    for (auto &x : V) x = nd(rng);

    KernelId k = 0;
    if (RuntimeEngine::instance().registerKernel("flash_attn_gqa", kGqaSrc, k) !=
            VGREResult::SUCCESS || k == 0) return false;

    auto &mm = RuntimeEngine::instance().getMemoryManager();
    MemoryHandle dQ, dK, dV, dO;
    mm.allocate(Q.size() * 4, dQ); mm.allocate(K.size() * 4, dK);
    mm.allocate(V.size() * 4, dV); mm.allocate(O.size() * 4, dO);
    mm.copyHostToDevice(dQ, Q.data(), Q.size() * 4);
    mm.copyHostToDevice(dK, K.data(), K.size() * 4);
    mm.copyHostToDevice(dV, V.data(), V.size() * 4);
    int bb = batch, nh = num_heads, nkv = num_kv_heads, seq = n, dh = d; float sc = scale;
    void *args[] = {&dQ, &dK, &dV, &dO, &bb, &nh, &nkv, &seq, &dh, &sc};
    RuntimeEngine::instance().launchKernel(k, dim3((unsigned)num_heads, 1, 1),
                                           dim3(128, 1, 1), args, 0, 0, dim3(0, 0, 0));
    RuntimeEngine::instance().synchronize();
    mm.copyDeviceToHost(O.data(), dO, O.size() * 4);

    // Naive GQA reference.
    const int groups = num_heads / num_kv_heads;
    double maxErr = 0.0;
    for (int h = 0; h < num_heads; ++h) {
        int kvh = h / groups;
        const float *Qh = &Q[(size_t)h * n * d];
        const float *Kh = &K[(size_t)kvh * n * d];
        const float *Vh = &V[(size_t)kvh * n * d];
        const float *Oh = &O[(size_t)h * n * d];
        for (int i = 0; i < n; ++i) {
            std::vector<float> s(n); float mx = -1e30f;
            for (int j = 0; j < n; ++j) {
                float dot = 0; for (int t = 0; t < d; ++t) dot += Qh[i * d + t] * Kh[j * d + t];
                s[j] = dot * scale; if (s[j] > mx) mx = s[j];
            }
            float sum = 0; for (int j = 0; j < n; ++j) { s[j] = std::exp(s[j] - mx); sum += s[j]; }
            for (int t = 0; t < d; ++t) {
                float o = 0; for (int j = 0; j < n; ++j) o += (s[j] / sum) * Vh[j * d + t];
                maxErr = std::max(maxErr, (double)std::fabs(o - Oh[i * d + t]));
            }
        }
    }
    printf("  [info] gqa max abs err vs naive = %.2e\n", maxErr);
    return maxErr < 1e-4;
}

int main() {
    printf("=== Flash Attention online-softmax correctness (Track 7) ===\n");
    RuntimeEngine::instance().initialize();
    check("non-causal flash attention matches naive softmax (1e-4)", runCase(false));
    check("causal flash attention matches naive softmax (1e-4)", runCase(true));
    check("GQA flash attention matches naive grouped softmax (1e-4)", runGqa());
    printf("\n%d / %d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
