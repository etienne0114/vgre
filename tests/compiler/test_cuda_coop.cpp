// Cooperative kernels (`__shared__` + `__syncthreads`) on the Tier-1 compiled
// backend's fiber executor, checked against the Tier-0 PTX interpreter (the
// long-standing reference for barrier kernels). Both must agree bit-for-bit — the
// two tiers run the SAME reduction/tiling order, so equality is exact, not
// approximate. This proves block-cooperative kernels no longer need the slow
// interpreter: the classic block reduction and a shared-memory tiled matmul both
// run on the ~30-90× faster compiled tier.
//
// Tests build in Release (-DNDEBUG); asserts must stay real.
#undef NDEBUG

#include "vgre/compiler/backend/backend_registry.h"
#include "vgre/compiler/backend/execution_backend.h"
#include "vgre/compiler/frontend/codegen.h"
#include "vgre/compiler/frontend/compiled_kernel.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace fe = vgre::compiler::frontend;
namespace be = vgre::compiler::backend;

static uint64_t g_rng = 0x1234567 ^ 0x9e3779b97f4a7c15ull;
static uint32_t rnd() { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17; return (uint32_t)(g_rng >> 32); }
static float rf() { return (float)(int32_t)rnd() / (float)(1u << 22); }

static int g_fail = 0;

// Run `src` on both the compiled tier (fiber path) and the interpreter over the
// same 3-D launch + args; compare `outFloats` output elements bit-for-bit.
static void checkCoop(const char* label, const char* name, const std::string& src,
                      const uint32_t grid[3], const uint32_t block[3],
                      std::vector<void*> argsCompiled, std::vector<float>& outC,
                      std::vector<void*> argsInterp, std::vector<float>& outI,
                      int outN) {
    std::string err;
    auto ck = fe::CompiledKernel::compileSource(src, name, err);
    if (!ck) { std::printf("FAIL: %s not compiled-tier eligible: %s\n", label, err.c_str()); ++g_fail; return; }

    std::unique_ptr<be::ExecutionBackend> interp = be::makeBackend("interpreter");
    if (!interp) { std::printf("FAIL: %s no interpreter backend\n", label); ++g_fail; return; }
    auto cg = fe::compileToPtx(src, name);
    if (!cg.ok) { std::printf("FAIL: %s PTX codegen: %s\n", label, cg.error.c_str()); ++g_fail; return; }
    auto pk = interp->preparePtx(cg.ptx, name);
    if (!pk) { std::printf("FAIL: %s interpreter prepare\n", label); ++g_fail; return; }

    fe::Extent g{grid[0], grid[1], grid[2]}, b{block[0], block[1], block[2]};
    if (!ck->launch(g, b, argsCompiled.data(), (int)argsCompiled.size())) { std::printf("FAIL: %s compiled launch\n", label); ++g_fail; return; }
    be::LaunchConfig cfg;
    for (int i = 0; i < 3; ++i) { cfg.gridDim[i] = grid[i]; cfg.blockDim[i] = block[i]; }
    if (!interp->launch(*pk, cfg, argsInterp.data(), (int)argsInterp.size())) { std::printf("FAIL: %s interpreter launch\n", label); ++g_fail; return; }

    int bad = 0;
    for (int i = 0; i < outN; ++i) {
        uint32_t a, c; std::memcpy(&a, &outC[i], 4); std::memcpy(&c, &outI[i], 4);
        if (a != c) { if (bad < 4) std::printf("  %s diff i=%d compiled=%.9g(%08x) interp=%.9g(%08x)\n", label, i, (double)outC[i], a, (double)outI[i], c); ++bad; }
    }
    if (bad) { std::printf("FAIL: %s compiled(fiber) vs interpreter: %d/%d mismatches\n", label, bad, outN); ++g_fail; }
    else std::printf("  %s (compiled fiber) == interpreter (%d elems)\n", label, outN);
}

int main() {
    // 1) Block reduction: __shared__ tree sum with a barrier inside the loop.
    {
        const int block = 256, blocks = 8, N = block * blocks;
        const char* src = R"(
extern "C" __global__ void blocksum(const float* in, float* out, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (i < n) ? in[i] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s = s >> 1) {
        if (tid < s) { sdata[tid] = sdata[tid] + sdata[tid + s]; }
        __syncthreads();
    }
    if (tid == 0) { out[blockIdx.x] = sdata[0]; }
})";
        std::vector<float> in(N); for (int i = 0; i < N; ++i) in[i] = rf();
        std::vector<float> outC(blocks, -1.f), outI(blocks, -2.f);
        int n = N;
        float* ip = in.data(); float* ocp = outC.data(); float* oip = outI.data();
        uint32_t grid[3] = {(uint32_t)blocks, 1, 1}, blk[3] = {(uint32_t)block, 1, 1};
        checkCoop("blocksum", "blocksum", src, grid, blk,
                  {&ip, &ocp, &n}, outC, {&ip, &oip, &n}, outI, blocks);
    }

    // 2) Shared-memory TILED matmul: load A/B tiles into shared, barrier, multiply,
    //    barrier, advance — the canonical cooperative GEMM.
    {
        const int T = 16, M = 32, K = 32, Nn = 32;   // multiples of the tile
        const char* src = R"(
extern "C" __global__ void tiled(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[256];
    __shared__ float Bs[256];
    int ty = threadIdx.y, tx = threadIdx.x;
    int row = blockIdx.y * 16 + ty;
    int col = blockIdx.x * 16 + tx;
    float acc = 0.0f;
    for (int t = 0; t < K; t = t + 16) {
        As[ty * 16 + tx] = A[row * K + (t + tx)];
        Bs[ty * 16 + tx] = B[(t + ty) * N + col];
        __syncthreads();
        for (int k = 0; k < 16; k = k + 1) { acc = acc + As[ty * 16 + k] * Bs[k * 16 + tx]; }
        __syncthreads();
    }
    C[row * N + col] = acc;
})";
        std::vector<float> A(M * K), B(K * Nn); for (int i = 0; i < M * K; ++i) A[i] = rf(); for (int i = 0; i < K * Nn; ++i) B[i] = rf();
        std::vector<float> C1(M * Nn, -1.f), C2(M * Nn, -2.f);
        int m = M, nn = Nn, kk = K;
        float* ap = A.data(); float* bp = B.data(); float* c1 = C1.data(); float* c2 = C2.data();
        uint32_t grid[3] = {(uint32_t)(Nn / T), (uint32_t)(M / T), 1}, blk[3] = {(uint32_t)T, (uint32_t)T, 1};
        checkCoop("tiled-gemm", "tiled", src, grid, blk,
                  {&ap, &bp, &c1, &m, &nn, &kk}, C1, {&ap, &bp, &c2, &m, &nn, &kk}, C2, M * Nn);
    }

    // 3) Warp shuffle: a butterfly all-reduce (__shfl_xor_sync) + a broadcast
    //    (__shfl_sync from lane 0). Every lane ends with the warp sum, then lane 0's.
    {
        const int block = 64, blocks = 4, N = block * blocks;   // 8 warps
        const char* src = R"(
extern "C" __global__ void warpred(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = (i < n) ? in[i] : 0.0f;
    for (int o = 16; o > 0; o = o >> 1) { v = v + __shfl_xor_sync(-1, v, o); }
    float b = __shfl_sync(-1, v, 0);
    if (i < n) { out[i] = v + b; }
})";
        std::vector<float> in(N); for (int i = 0; i < N; ++i) in[i] = rf();
        std::vector<float> outC(N, -1.f), outI(N, -2.f);
        int n = N;
        float* ip = in.data(); float* ocp = outC.data(); float* oip = outI.data();
        uint32_t grid[3] = {(uint32_t)blocks, 1, 1}, blk[3] = {(uint32_t)block, 1, 1};
        checkCoop("warp-shuffle", "warpred", src, grid, blk,
                  {&ip, &ocp, &n}, outC, {&ip, &oip, &n}, outI, N);
    }

    // 4) Warp shuffle-down reduction (__shfl_down_sync) writing one result per warp.
    {
        const int block = 64, blocks = 4, N = block * blocks;
        const char* src = R"(
extern "C" __global__ void wdown(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = (i < n) ? in[i] : 0.0f;
    for (int o = 16; o > 0; o = o >> 1) { v = v + __shfl_down_sync(-1, v, o); }
    if ((threadIdx.x & 31) == 0) { out[i >> 5] = v; }
})";
        std::vector<float> in(N); for (int i = 0; i < N; ++i) in[i] = rf();
        std::vector<float> outC(N / 32, -1.f), outI(N / 32, -2.f);
        int n = N;
        float* ip = in.data(); float* ocp = outC.data(); float* oip = outI.data();
        uint32_t grid[3] = {(uint32_t)blocks, 1, 1}, blk[3] = {(uint32_t)block, 1, 1};
        checkCoop("warp-shfl-down", "wdown", src, grid, blk,
                  {&ip, &ocp, &n}, outC, {&ip, &oip, &n}, outI, N / 32);
    }

    // 5) Warp vote: __ballot_sync / __any_sync / __all_sync. (int I/O via float
    //    byte-buffers — checkCoop compares raw 32-bit words.)
    {
        const int block = 64, blocks = 4, N = block * blocks;
        const char* src = R"(
extern "C" __global__ void wvote(const int* in, int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int p = in[i] > 0;
    unsigned b = __ballot_sync(-1, p);
    int a = __any_sync(-1, p);
    int al = __all_sync(-1, p);
    out[i] = (int)b + a * 2 + al * 4;
})";
        std::vector<int> in(N); for (int i = 0; i < N; ++i) in[i] = (int)(rnd() % 3) - 1;
        std::vector<float> outC(N, -1.f), outI(N, -2.f);
        int n = N; int* ip = in.data();
        int* ocp = reinterpret_cast<int*>(outC.data()); int* oip = reinterpret_cast<int*>(outI.data());
        uint32_t grid[3] = {(uint32_t)blocks, 1, 1}, blk[3] = {(uint32_t)block, 1, 1};
        checkCoop("warp-vote", "wvote", src, grid, blk,
                  {&ip, &ocp, &n}, outC, {&ip, &oip, &n}, outI, N);
    }

    // 6) Warp reduce: __reduce_add_sync / __reduce_max_sync (compute capability 8.0).
    {
        const int block = 64, blocks = 4, N = block * blocks;
        const char* src = R"(
extern "C" __global__ void wreduce(const int* in, int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int v = in[i];
    int s = __reduce_add_sync(-1, v);
    int mx = __reduce_max_sync(-1, v);
    out[i] = s + mx;
})";
        std::vector<int> in(N); for (int i = 0; i < N; ++i) in[i] = (int)(rnd() % 200) - 100;
        std::vector<float> outC(N, -1.f), outI(N, -2.f);
        int n = N; int* ip = in.data();
        int* ocp = reinterpret_cast<int*>(outC.data()); int* oip = reinterpret_cast<int*>(outI.data());
        uint32_t grid[3] = {(uint32_t)blocks, 1, 1}, blk[3] = {(uint32_t)block, 1, 1};
        checkCoop("warp-reduce", "wreduce", src, grid, blk,
                  {&ip, &ocp, &n}, outC, {&ip, &oip, &n}, outI, N);
    }

    // 7) Block vote: __syncthreads_count / __syncthreads_and / __syncthreads_or.
    {
        const int block = 64, blocks = 4, N = block * blocks;
        const char* src = R"(
extern "C" __global__ void bvote(const int* in, int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int p = in[i] > 0;
    int c = __syncthreads_count(p);
    int a = __syncthreads_and(p);
    int o = __syncthreads_or(p);
    out[i] = c + a * 1000 + o * 100000;
})";
        std::vector<int> in(N); for (int i = 0; i < N; ++i) in[i] = (int)(rnd() % 3) - 1;
        std::vector<float> outC(N, -1.f), outI(N, -2.f);
        int n = N; int* ip = in.data();
        int* ocp = reinterpret_cast<int*>(outC.data()); int* oip = reinterpret_cast<int*>(outI.data());
        uint32_t grid[3] = {(uint32_t)blocks, 1, 1}, blk[3] = {(uint32_t)block, 1, 1};
        checkCoop("block-vote", "bvote", src, grid, blk,
                  {&ip, &ocp, &n}, outC, {&ip, &oip, &n}, outI, N);
    }

    if (g_fail == 0) std::printf("PASS: cooperative shared/barrier + warp shuffle/vote/reduce + block vote on the compiled fiber tier, == interpreter\n");
    else std::printf("FAILED: %d check(s)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
