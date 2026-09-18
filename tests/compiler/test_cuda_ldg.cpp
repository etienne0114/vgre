// Track Z / Stage 3.2: __ldg and the cache-hinted load/store intrinsics. Real
// CUDA kernels use __ldg (read-only data cache) pervasively, plus __ldca/__ldcs/
// __ldcg/__ldlu/__ldcv and the __stwb/__stcg/__stcs/__stwt store hints. A CPU
// interpreter has no cache hierarchy, so each is a plain load/store of the pointed
// element — the front-end lowers them to ld.global / st.global. Both the
// __ldg(&a[i]) and __ldg(p) forms are supported. No LLVM.
//
// Tests build in Release (-DNDEBUG); asserts must stay real.
#undef NDEBUG

#include "vgre/compiler/backend/backend_registry.h"
#include "vgre/compiler/backend/execution_backend.h"
#include "vgre/compiler/frontend/codegen.h"

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

// __ldg(&a[i]) — the "&element" form.
static const char* kLdgAddr = R"(
extern "C" __global__ void ldg_addr(float* out, const float* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __ldg(&a[i]) * 2.0f;
})";
// __ldg(p) — the "pointer variable" form (advance the pointer, then deref).
static const char* kLdgPtr = R"(
extern "C" __global__ void ldg_ptr(int* out, const int* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { const int* p = a + i; out[i] = __ldg(p) + 7; }
})";
// The cache-hinted load family: each must behave like a plain load.
static const char* kLdVariants = R"(
extern "C" __global__ void ldvars(float* out, const float* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __ldca(&a[i]) + __ldcs(&a[i]) + __ldcg(&a[i]) + __ldlu(&a[i]);
})";
// Cache-hinted store: __stcg(&out[i], v) writes like a plain store.
static const char* kStcg = R"(
extern "C" __global__ void stcg(float* out, const float* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) __stcg(&out[i], __ldg(&a[i]) + 1.0f);
})";

int main() {
    const int N = 96;
    std::vector<float> a(N);
    for (int i = 0; i < N; ++i) a[i] = (i - 40) * 0.5f;

    // __ldg(&a[i]) * 2
    {
        std::vector<float> out(N, -1);
        void* op = out.data(); void* ap = a.data(); int n = N;
        void* args[] = {&op, &ap, &n};
        CHECK(runInterp(kLdgAddr, "ldg_addr", 3, 32, args, 3), "ldg_addr runs");
        bool ok = true;
        for (int i = 0; i < N; ++i) if (out[i] != a[i] * 2.0f) { ok = false; break; }
        CHECK(ok, "__ldg(&a[i]) == a[i]");
    }
    // __ldg(p) with p = a + i
    {
        std::vector<int> ai(N), out(N, -1);
        for (int i = 0; i < N; ++i) ai[i] = i * 3 - 10;
        void* op = out.data(); void* ap = ai.data(); int n = N;
        void* args[] = {&op, &ap, &n};
        CHECK(runInterp(kLdgPtr, "ldg_ptr", 3, 32, args, 3), "ldg_ptr runs");
        bool ok = true;
        for (int i = 0; i < N; ++i) if (out[i] != ai[i] + 7) { ok = false; break; }
        CHECK(ok, "__ldg(p) with p=a+i");
    }
    // __ldca/__ldcs/__ldcg/__ldlu all equal a plain load.
    {
        std::vector<float> out(N, -1);
        void* op = out.data(); void* ap = a.data(); int n = N;
        void* args[] = {&op, &ap, &n};
        CHECK(runInterp(kLdVariants, "ldvars", 3, 32, args, 3), "ldvars runs");
        bool ok = true;
        for (int i = 0; i < N; ++i) if (out[i] != a[i] * 4.0f) { ok = false; break; }
        CHECK(ok, "cache-hinted loads all == plain load");
    }
    // __stcg store hint.
    {
        std::vector<float> out(N, -1);
        void* op = out.data(); void* ap = a.data(); int n = N;
        void* args[] = {&op, &ap, &n};
        CHECK(runInterp(kStcg, "stcg", 3, 32, args, 3), "stcg runs");
        bool ok = true;
        for (int i = 0; i < N; ++i) if (out[i] != a[i] + 1.0f) { ok = false; break; }
        CHECK(ok, "__stcg(&out[i], v) stores v");
    }

    if (g_fail == 0)
        std::printf("PASS: __ldg + cache-hinted load/store intrinsics\n");
    return g_fail ? 1 : 0;
}
