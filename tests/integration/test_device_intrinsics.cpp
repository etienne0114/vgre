/**
 * VGRE Integration Test — Device Intrinsics (2026-06-22)
 *
 * Verifies that the device-side CUDA intrinsics added to the JIT environment
 * compile *and execute correctly* when real CUDA-C kernels are JIT-compiled and
 * launched through VGRE. These are the bread-and-butter intrinsics ML/DL kernels
 * depend on; before this work they failed JIT compilation with
 * "use of undeclared identifier".
 *
 * Coverage:
 *   1. Integer atomics      — atomicMax/Min/And/Or/Xor/Inc into shared counters
 *   2. __ldg + float4       — read-only cache load + vectorized vector type
 *   3. __syncthreads_count  — block-wide vote (real barrier reduction)
 *   4. __reduce_add_sync    — warp-level reduction (sm_80+ intrinsic)
 *   5. Bit / math intrinsics— __int_as_float/__float_as_int round-trip, rsqrtf
 *
 * Each kernel writes its results to a device buffer; the host checks exact
 * (integer / bit-exact) or tightly-toleranced (float) expected values.
 */

#include "vgre/core/runtime_engine.h"
#include "vgre/core/memory_manager.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

using namespace vgre;

static constexpr float EPS = 1e-5f;
static int g_failures = 0;

static void report(const char* name, bool ok) {
    std::cout << (ok ? "[PASS] " : "[FAIL] ") << name << "\n";
    if (!ok) ++g_failures;
}

// ── Test 1: integer atomics ─────────────────────────────────────────────────
// 64 threads. out[0]=atomicMax(lane), out[1]=atomicMin(lane), out[2]=OR of all
// lane bits, out[3]=AND, out[4]=XOR, out[5]=count via atomicInc.
static void test_atomics() {
    core::RuntimeEngine engine;
    if (engine.initialize() != VGREResult::SUCCESS) { report("atomics(init)", false); return; }

    static const char* src = R"(
extern "C" __global__ void atomics_k(int* out, unsigned* uout) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    atomicMax(&out[0], t);
    atomicMin(&out[1], t);
    atomicOr (&out[2], (1 << (t & 7)));
    atomicAnd(&out[3], t);
    atomicXor(&out[4], t);
    atomicInc(&uout[0], 1000000u);   // wrap threshold far above 64 → plain count
}
)";

    auto& mm = engine.getMemoryManager();
    MemoryHandle dInt = nullptr, dU = nullptr;
    mm.allocate(8 * sizeof(int), dInt);
    mm.allocate(1 * sizeof(unsigned), dU);
    int initI[8] = { -2000000000, 2000000000, 0, ~0, 0, 0, 0, 0 };
    unsigned initU = 0;
    std::memcpy(static_cast<void*>(dInt), initI, sizeof(initI));
    std::memcpy(static_cast<void*>(dU), &initU, sizeof(initU));

    void* args[2] = { &dInt, &dU };
    vgre::dim3 grid(2), block(32);   // 64 threads
    bool ok = engine.launchKernel("atomics_k", src, grid, block, args, 0) == VGREResult::SUCCESS;
    ok = ok && engine.synchronize() == VGREResult::SUCCESS;

    if (ok) {
        int* r = static_cast<int*>(dInt);
        unsigned* u = static_cast<unsigned*>(dU);
        // Expected over t = 0..63
        int expMax = 63, expMin = 0;
        int expOr = 0, expAnd = ~0, expXor = 0;
        for (int t = 0; t < 64; ++t) { expOr |= (1 << (t & 7)); expAnd &= t; expXor ^= t; }
        ok = (r[0] == expMax) && (r[1] == expMin) && (r[2] == expOr) &&
             (r[3] == expAnd) && (r[4] == expXor) && (u[0] == 64u);
        if (!ok)
            std::cerr << "  atomics got max=" << r[0] << " min=" << r[1] << " or=" << r[2]
                      << " and=" << r[3] << " xor=" << r[4] << " inc=" << u[0]
                      << " (exp 63,0," << expOr << "," << expAnd << "," << expXor << ",64)\n";
    }
    mm.free(dInt); mm.free(dU);
    engine.shutdown();
    report("integer atomics (Max/Min/And/Or/Xor/Inc)", ok);
}

// ── Test 2: __ldg + float4 vectorized load ──────────────────────────────────
static void test_ldg_float4() {
    core::RuntimeEngine engine;
    if (engine.initialize() != VGREResult::SUCCESS) { report("ldg(init)", false); return; }

    static const char* src = R"(
extern "C" __global__ void ldg_k(const float4* in, float* out) {
    int t = threadIdx.x;
    float4 v = __ldg(&in[t]);              // read-only cache load of a 16B vector
    out[t] = v.x + v.y + v.z + v.w;
}
)";
    const int N = 8;
    auto& mm = engine.getMemoryManager();
    MemoryHandle dIn = nullptr, dOut = nullptr;
    mm.allocate(N * 4 * sizeof(float), dIn);
    mm.allocate(N * sizeof(float), dOut);
    std::vector<float> in(N * 4);
    for (int i = 0; i < N * 4; ++i) in[i] = static_cast<float>(i);
    std::memcpy(static_cast<void*>(dIn), in.data(), in.size() * sizeof(float));

    void* args[2] = { &dIn, &dOut };
    vgre::dim3 grid(1), block(N);
    bool ok = engine.launchKernel("ldg_k", src, grid, block, args, 0) == VGREResult::SUCCESS;
    ok = ok && engine.synchronize() == VGREResult::SUCCESS;
    if (ok) {
        float* r = static_cast<float*>(dOut);
        for (int t = 0; t < N; ++t) {
            float exp = in[t*4] + in[t*4+1] + in[t*4+2] + in[t*4+3];
            if (std::abs(r[t] - exp) > EPS) { ok = false;
                std::cerr << "  ldg/float4 t=" << t << " got " << r[t] << " exp " << exp << "\n"; }
        }
    }
    mm.free(dIn); mm.free(dOut);
    engine.shutdown();
    report("__ldg + float4 vectorized load", ok);
}

// ── Test 3: __syncthreads_count (block-wide vote) ───────────────────────────
static void test_syncthreads_count() {
    core::RuntimeEngine engine;
    if (engine.initialize() != VGREResult::SUCCESS) { report("synccount(init)", false); return; }

    static const char* src = R"(
extern "C" __global__ void vote_k(int* out) {
    int t = threadIdx.x;
    // predicate true for even lanes → count == blockDim/2
    int c = __syncthreads_count((t & 1) == 0);
    int allPos = __syncthreads_and(t >= 0);     // every t >= 0  → nonzero
    int anyBig = __syncthreads_or(t == 100);    // none equal 100 → 0
    if (t == 0) { out[0] = c; out[1] = allPos; out[2] = anyBig; }
}
)";
    auto& mm = engine.getMemoryManager();
    MemoryHandle dOut = nullptr;
    mm.allocate(4 * sizeof(int), dOut);
    void* args[1] = { &dOut };
    vgre::dim3 grid(1), block(64);
    bool ok = engine.launchKernel("vote_k", src, grid, block, args, 0) == VGREResult::SUCCESS;
    ok = ok && engine.synchronize() == VGREResult::SUCCESS;
    if (ok) {
        int* r = static_cast<int*>(dOut);
        ok = (r[0] == 32) && (r[1] != 0) && (r[2] == 0);
        if (!ok) std::cerr << "  vote count=" << r[0] << " and=" << r[1]
                           << " or=" << r[2] << " (exp 32, nonzero, 0)\n";
    }
    mm.free(dOut);
    engine.shutdown();
    report("__syncthreads_count / _and / _or", ok);
}

// ── Test 4: __reduce_add_sync (warp reduction) ──────────────────────────────
static void test_warp_reduce_sync() {
    core::RuntimeEngine engine;
    if (engine.initialize() != VGREResult::SUCCESS) { report("reduce(init)", false); return; }

    static const char* src = R"(
extern "C" __global__ void reduce_k(int* out) {
    int t = threadIdx.x;
    int s = __reduce_add_sync(0xFFFFFFFFu, t);   // every lane gets sum 0..31 = 496
    out[t] = s;
}
)";
    auto& mm = engine.getMemoryManager();
    MemoryHandle dOut = nullptr;
    mm.allocate(32 * sizeof(int), dOut);
    void* args[1] = { &dOut };
    vgre::dim3 grid(1), block(32);
    bool ok = engine.launchKernel("reduce_k", src, grid, block, args, 0) == VGREResult::SUCCESS;
    ok = ok && engine.synchronize() == VGREResult::SUCCESS;
    if (ok) {
        int* r = static_cast<int*>(dOut);
        for (int t = 0; t < 32; ++t)
            if (r[t] != 496) { ok = false;
                std::cerr << "  reduce t=" << t << " got " << r[t] << " exp 496\n"; break; }
    }
    mm.free(dOut);
    engine.shutdown();
    report("__reduce_add_sync (warp reduce)", ok);
}

// ── Test 5: bit-reinterpret + rsqrtf ────────────────────────────────────────
static void test_bit_and_math() {
    core::RuntimeEngine engine;
    if (engine.initialize() != VGREResult::SUCCESS) { report("bitmath(init)", false); return; }

    static const char* src = R"(
extern "C" __global__ void bitmath_k(float* out) {
    float x = 3.14159f;
    int bits = __float_as_int(x);
    float back = __int_as_float(bits);     // round-trips exactly
    out[0] = back;
    out[1] = rsqrtf(4.0f);                 // 0.5
    out[2] = norm3df(2.0f, 3.0f, 6.0f);    // sqrt(4+9+36)=7
    out[3] = __saturatef(2.5f);            // clamp → 1
    out[4] = __expf(0.0f) + __sinf(0.0f);  // 1 + 0
}
)";
    auto& mm = engine.getMemoryManager();
    MemoryHandle dOut = nullptr;
    mm.allocate(8 * sizeof(float), dOut);
    void* args[1] = { &dOut };
    vgre::dim3 grid(1), block(1);
    bool ok = engine.launchKernel("bitmath_k", src, grid, block, args, 0) == VGREResult::SUCCESS;
    ok = ok && engine.synchronize() == VGREResult::SUCCESS;
    if (ok) {
        float* r = static_cast<float*>(dOut);
        ok = std::abs(r[0] - 3.14159f) < EPS && std::abs(r[1] - 0.5f) < EPS &&
             std::abs(r[2] - 7.0f) < EPS && std::abs(r[3] - 1.0f) < EPS &&
             std::abs(r[4] - 1.0f) < EPS;
        if (!ok) std::cerr << "  bitmath " << r[0] << "," << r[1] << "," << r[2]
                           << "," << r[3] << "," << r[4] << "\n";
    }
    mm.free(dOut);
    engine.shutdown();
    report("bit-reinterpret + rsqrtf/norm3df/__saturatef/__expf", ok);
}

int main() {
    std::cout << "=== VGRE Device-Intrinsics JIT Integration Test ===\n";
    test_atomics();
    test_ldg_float4();
    test_syncthreads_count();
    test_warp_reduce_sync();
    test_bit_and_math();
    std::cout << (g_failures == 0 ? "ALL PASSED\n" : "FAILURES PRESENT\n");
    return g_failures == 0 ? 0 : 1;
}
