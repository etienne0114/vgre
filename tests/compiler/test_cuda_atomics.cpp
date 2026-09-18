// Track Z / Stage 3.2: the atomic family beyond atomicAdd — atomicMin/Max/Exch/
// CAS/And/Or/Xor/Sub. The from-scratch front-end lowers them to atom.global.<op>
// (or ld/compute/st for __shared__/local), and the interpreter performs a real
// lock-free RMW so concurrent CTAs — run on parallel OS threads by the backend —
// stay correct. Each returns the OLD value. Verified under contention (many lanes
// racing one address) with order-independent references.
//
// Tests build in Release (-DNDEBUG); asserts must stay real.
#undef NDEBUG

#include "vgre/compiler/backend/backend_registry.h"
#include "vgre/compiler/backend/execution_backend.h"
#include "vgre/compiler/frontend/codegen.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

using namespace vgre::compiler::frontend;
namespace be = vgre::compiler::backend;

static int g_fail = 0;
#define CHECK(cond, msg)                                                   \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("FAIL: %s  (%s:%d)\n", (msg), __FILE__, __LINE__); \
            ++g_fail;                                                      \
        }                                                                  \
    } while (0)

static bool runInterp(const char* src, const char* name,
                      int grid, int block, void* const* args, int numArgs) {
    auto cg = compileToPtx(src, name);
    if (!cg.ok) { std::printf("  codegen %s: %s\n", name, cg.error.c_str()); return false; }
    auto beI = be::makeBackend("interpreter");
    auto k = beI->preparePtx(cg.ptx, name);
    if (!k) { std::printf("  prepare %s:\n%s\n", name, cg.ptx.c_str()); return false; }
    be::LaunchConfig lc; lc.gridDim[0] = (uint32_t)grid; lc.blockDim[0] = (uint32_t)block;
    return beI->launch(*k, lc, args, numArgs);
}

// out[0] folds in[] with the given atomic; runs across parallel CTAs.
static const char* kMax = R"(
extern "C" __global__ void amax(int* out, const int* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicMax(&out[0], in[i]);
})";
static const char* kMin = R"(
extern "C" __global__ void amin(int* out, const int* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicMin(&out[0], in[i]);
})";
static const char* kAnd = R"(
extern "C" __global__ void aand(int* out, const int* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAnd(&out[0], in[i]);
})";
static const char* kOr = R"(
extern "C" __global__ void aor(int* out, const int* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicOr(&out[0], in[i]);
})";
static const char* kXor = R"(
extern "C" __global__ void axor(int* out, const int* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicXor(&out[0], in[i]);
})";
static const char* kSub = R"(
extern "C" __global__ void asub(int* out, const int* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicSub(&out[0], in[i]);
})";
// Each lane swaps in its own tag; the olds + the final value form a permutation.
static const char* kExch = R"(
extern "C" __global__ void aexch(int* out, int* cell, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = atomicExch(&cell[0], i + 1);
})";
// CAS-loop increment: every lane increments cell[0] exactly once → final == n.
static const char* kCas = R"(
extern "C" __global__ void acas(int* cell, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int assumed = cell[0];
        int old = atomicCAS(&cell[0], assumed, assumed + 1);
        while (old != assumed) { assumed = old; old = atomicCAS(&cell[0], assumed, assumed + 1); }
    }
})";

static bool foldTest(const char* src, const char* name, int init,
                     const std::vector<int>& in, int ref) {
    std::vector<int> out(1, init);
    void* op = out.data(); void* ip = (void*)in.data(); int n = (int)in.size();
    void* args[] = {&op, &ip, &n};
    if (!runInterp(src, name, 4, 32, args, 3)) { std::printf("  %s did not run\n", name); return false; }
    if (out[0] != ref) { std::printf("  %s: got %d want %d\n", name, out[0], ref); return false; }
    return true;
}

int main() {
    const int N = 128;
    std::vector<int> in(N);
    for (int i = 0; i < N; ++i) in[i] = (i * 2654435761u) & 0x7fffffff;   // varied positives

    int mx = in[0], mn = in[0], aa = ~0, ao = 0, ax = 0; long sum = 0;
    for (int v : in) { mx = std::max(mx, v); mn = std::min(mn, v); aa &= v; ao |= v; ax ^= v; sum += v; }

    CHECK(foldTest(kMax, "amax", -1, in, mx), "atomicMax folds to max");
    CHECK(foldTest(kMin, "amin", 0x7fffffff, in, mn), "atomicMin folds to min");
    CHECK(foldTest(kAnd, "aand", ~0, in, aa), "atomicAnd folds to AND");
    CHECK(foldTest(kOr,  "aor", 0, in, ao), "atomicOr folds to OR");
    CHECK(foldTest(kXor, "axor", 0, in, ax), "atomicXor folds to XOR");
    CHECK(foldTest(kSub, "asub", (int)sum, in, 0), "atomicSub subtracts every input (init=sum -> 0)");

    // Exch permutation: cell starts 0; olds ∪ {final} == {0,1,…,N}.
    {
        std::vector<int> out(N, -1), cell(1, 0);
        void* op = out.data(); void* cp = cell.data(); int n = N;
        void* args[] = {&op, &cp, &n};
        CHECK(runInterp(kExch, "aexch", 4, 32, args, 3), "aexch runs");
        std::vector<int> seen = out; seen.push_back(cell[0]);
        std::sort(seen.begin(), seen.end());
        bool ok = ((int)seen.size() == N + 1);
        for (int i = 0; ok && i <= N; ++i) if (seen[i] != i) ok = false;
        CHECK(ok, "atomicExch: olds + final form the permutation {0..N}");
    }

    // CAS-loop increment under contention: final == N.
    {
        std::vector<int> cell(1, 0);
        void* cp = cell.data(); int n = N;
        void* args[] = {&cp, &n};
        CHECK(runInterp(kCas, "acas", 4, 32, args, 2), "acas runs");
        CHECK(cell[0] == N, "atomicCAS-loop increments exactly once per lane");
    }

    if (g_fail == 0)
        std::printf("PASS: atomics (min/max/and/or/xor/sub/exch/cas) under CTA contention\n");
    return g_fail ? 1 : 0;
}
