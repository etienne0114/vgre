/**
 * VGRE Integration Test — Warp Shuffle Intrinsics (QUEUE-10)
 *
 * Verifies __shfl_sync, __shfl_down_sync, __shfl_up_sync, __shfl_xor_sync
 * produce correct results when kernels are JIT-compiled through VGRE.
 *
 * Math invariant: For width=32, each variant implements a deterministic
 * permutation network on 32 values.
 *   __shfl_sync:      result[t] = val[srcLane % width]  (broadcast/gather)
 *   __shfl_down_sync: result[t] = val[t + delta] if t+delta < groupEnd else val[t]
 *   __shfl_up_sync:   result[t] = val[t - delta] if t-delta >= groupBase else val[t]
 *   __shfl_xor_sync:  result[t] = val[t ^ laneMask]  (butterfly)
 *
 * Tests:
 *   1. Broadcast via __shfl_sync: all 32 threads get thread-0's value
 *   2. Warp sum-reduction via __shfl_down_sync: tree reduction, thread-0 = Σ val[i]
 *   3. Prefix shift via __shfl_up_sync: thread t gets val[t-1] (delta=1)
 *   4. Butterfly exchange via __shfl_xor_sync: lane_xor_1 pattern
 */

#include "vgre/core/runtime_engine.h"
#include "vgre/core/memory_manager.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

using namespace vgre;

// Maximum error tolerance (these are exact integer operations bit-cast through float)
static constexpr float EPSILON = 1e-5f;

// Helper: launch kernel, wait, copy result back
static bool run_kernel(core::RuntimeEngine& engine,
                        const char* name, const char* source,
                        vgre::dim3 grid, vgre::dim3 block,
                        void** args, size_t sharedBytes,
                        float* hostOut, int count)
{
    auto& mm  = engine.getMemoryManager();
    MemoryHandle devOut = nullptr;
    if (mm.allocate(static_cast<size_t>(count) * sizeof(float), devOut)
            != VGREResult::SUCCESS || !devOut) {
        std::cerr << "  allocate failed\n";
        return false;
    }
    // args[0] points to devOut — patch it in
    args[0] = &devOut;

    VGREResult r = engine.launchKernel(name, source, grid, block, args, sharedBytes);
    if (r != VGREResult::SUCCESS) {
        std::cerr << "  launchKernel failed: " << static_cast<int>(r) << "\n";
        mm.free(devOut);
        return false;
    }
    if (engine.synchronize() != VGREResult::SUCCESS) {
        std::cerr << "  synchronize failed\n";
        mm.free(devOut);
        return false;
    }
    // Copy device → host
    float* ptr = static_cast<float*>(devOut);
    for (int i = 0; i < count; ++i) hostOut[i] = ptr[i];
    mm.free(devOut);
    return true;
}

// ── Test 1: broadcast via __shfl_sync ────────────────────────────────────────
// Thread 0 has value 42; all others have 0.
// After shfl_sync(mask, val, 0), every thread should have 42.
static bool test_broadcast() {
    core::RuntimeEngine engine;
    if (engine.initialize() != VGREResult::SUCCESS) return false;

    static const char* src = R"(
__global__ void shfl_broadcast(float* out) {
    int t = threadIdx.x;
    float val = (t == 0) ? 42.f : 0.f;
    // Broadcast lane-0's value to all 32 lanes
    float bcast = __shfl_sync(0xFFFFFFFFu, val, 0, 32);
    out[t] = bcast;
}
)";

    std::vector<float> result(32, -1.f);
    void* args[1];
    float* dummy = nullptr;
    args[0] = &dummy;

    auto& mm = engine.getMemoryManager();
    MemoryHandle devOut = nullptr;
    if (mm.allocate(32 * sizeof(float), devOut) != VGREResult::SUCCESS) {
        engine.shutdown();
        return false;
    }
    args[0] = &devOut;

    vgre::dim3 grid(1), block(32);
    VGREResult r = engine.launchKernel("shfl_broadcast", src, grid, block, args, 0);
    if (r != VGREResult::SUCCESS) {
        std::cerr << "  [broadcast] launch failed: " << static_cast<int>(r) << "\n";
        mm.free(devOut);
        engine.shutdown();
        return false;
    }
    engine.synchronize();

    float* ptr = static_cast<float*>(devOut);
    bool ok = true;
    for (int i = 0; i < 32; ++i) {
        if (std::abs(ptr[i] - 42.f) > EPSILON) {
            std::cerr << "  [broadcast] thread " << i << ": expected 42, got " << ptr[i] << "\n";
            ok = false;
        }
    }
    mm.free(devOut);
    engine.shutdown();

    std::cout << (ok ? "[PASS]" : "[FAIL]") << " shfl_broadcast\n";
    return ok;
}

// ── Test 2: warp sum-reduction via __shfl_down_sync ──────────────────────────
// val[t] = t (0..31). Tree reduction: thread 0 accumulates sum = 0+1+...+31 = 496.
static bool test_warp_reduce() {
    core::RuntimeEngine engine;
    if (engine.initialize() != VGREResult::SUCCESS) return false;

    static const char* src = R"(
__global__ void warp_reduce(float* out) {
    int t = threadIdx.x;
    float val = (float)t;
    // Tree reduction: shfl_down_sync with offsets 16,8,4,2,1
    val += __shfl_down_sync(0xFFFFFFFFu, val, 16, 32);
    val += __shfl_down_sync(0xFFFFFFFFu, val,  8, 32);
    val += __shfl_down_sync(0xFFFFFFFFu, val,  4, 32);
    val += __shfl_down_sync(0xFFFFFFFFu, val,  2, 32);
    val += __shfl_down_sync(0xFFFFFFFFu, val,  1, 32);
    if (t == 0) out[0] = val;
}
)";

    auto& mm = engine.getMemoryManager();
    MemoryHandle devOut = nullptr;
    if (mm.allocate(sizeof(float), devOut) != VGREResult::SUCCESS) {
        engine.shutdown();
        return false;
    }
    void* args[1] = {&devOut};

    vgre::dim3 grid(1), block(32);
    VGREResult r = engine.launchKernel("warp_reduce", src, grid, block, args, 0);
    if (r != VGREResult::SUCCESS) {
        std::cerr << "  [warp_reduce] launch failed: " << static_cast<int>(r) << "\n";
        mm.free(devOut);
        engine.shutdown();
        return false;
    }
    engine.synchronize();

    float result = *static_cast<float*>(devOut);
    mm.free(devOut);
    engine.shutdown();

    // Expected: 0+1+...+31 = 31*32/2 = 496
    const float expected = 496.f;
    bool ok = (std::abs(result - expected) <= EPSILON);
    std::cout << (ok ? "[PASS]" : "[FAIL]")
              << " warp_reduce: result=" << result << " expected=" << expected << "\n";
    return ok;
}

// ── Test 3: prefix shift via __shfl_up_sync ──────────────────────────────────
// val[t] = t+1 (1..32). After shfl_up(mask, val, 1):
//   thread 0 gets own value (clamped at group start) = 1
//   thread t (t>0) gets val[t-1] = t
// So out[t] = t for all t in [0,31].
static bool test_shfl_up() {
    core::RuntimeEngine engine;
    if (engine.initialize() != VGREResult::SUCCESS) return false;

    static const char* src = R"(
__global__ void shfl_up_test(float* out) {
    int t = threadIdx.x;
    float val = (float)(t + 1);  // val[t] = t+1
    float shifted = __shfl_up_sync(0xFFFFFFFFu, val, 1u, 32);
    out[t] = shifted;
}
)";

    auto& mm = engine.getMemoryManager();
    MemoryHandle devOut = nullptr;
    if (mm.allocate(32 * sizeof(float), devOut) != VGREResult::SUCCESS) {
        engine.shutdown();
        return false;
    }
    void* args[1] = {&devOut};

    vgre::dim3 grid(1), block(32);
    VGREResult r = engine.launchKernel("shfl_up_test", src, grid, block, args, 0);
    if (r != VGREResult::SUCCESS) {
        std::cerr << "  [shfl_up] launch failed: " << static_cast<int>(r) << "\n";
        mm.free(devOut);
        engine.shutdown();
        return false;
    }
    engine.synchronize();

    float* ptr = static_cast<float*>(devOut);
    bool ok = true;
    for (int t = 0; t < 32; ++t) {
        float expected = (float)t;   // thread 0 clamped → val[0]=1, shifted = 1 ... wait
        // thread 0: t=0, src=0-1=-1 < groupBase=0 → src=0 → val[0]=1 → result=1
        // thread 1: src=0 → val[0]=1 → result=1 ... hmm
        // Actually: val[t] = t+1 so val[0]=1,val[1]=2,...
        // shfl_up(delta=1): thread t gets val[t-1] (clamped if t-1 < 0)
        // thread 0: clamped → gets val[0] = 1
        // thread 1: gets val[0] = 1
        // thread 2: gets val[1] = 2
        // ...
        // thread t (t>=1): gets val[t-1] = t
        // So expected[t] = (t == 0) ? 1.f : (float)t
        expected = (t == 0) ? 1.f : (float)t;
        if (std::abs(ptr[t] - expected) > EPSILON) {
            std::cerr << "  [shfl_up] thread " << t << ": expected " << expected
                      << " got " << ptr[t] << "\n";
            ok = false;
        }
    }
    mm.free(devOut);
    engine.shutdown();
    std::cout << (ok ? "[PASS]" : "[FAIL]") << " shfl_up\n";
    return ok;
}

// ── Test 4: butterfly exchange via __shfl_xor_sync ───────────────────────────
// val[t] = t. After shfl_xor(mask, val, 1):
//   thread 0 ↔ thread 1, thread 2 ↔ thread 3, ...
//   out[t] = t ^ 1  (i.e., neighbors swap)
static bool test_shfl_xor() {
    core::RuntimeEngine engine;
    if (engine.initialize() != VGREResult::SUCCESS) return false;

    static const char* src = R"(
__global__ void shfl_xor_test(float* out) {
    int t = threadIdx.x;
    float val = (float)t;
    float swapped = __shfl_xor_sync(0xFFFFFFFFu, val, 1, 32);
    out[t] = swapped;
}
)";

    auto& mm = engine.getMemoryManager();
    MemoryHandle devOut = nullptr;
    if (mm.allocate(32 * sizeof(float), devOut) != VGREResult::SUCCESS) {
        engine.shutdown();
        return false;
    }
    void* args[1] = {&devOut};

    vgre::dim3 grid(1), block(32);
    VGREResult r = engine.launchKernel("shfl_xor_test", src, grid, block, args, 0);
    if (r != VGREResult::SUCCESS) {
        std::cerr << "  [shfl_xor] launch failed: " << static_cast<int>(r) << "\n";
        mm.free(devOut);
        engine.shutdown();
        return false;
    }
    engine.synchronize();

    float* ptr = static_cast<float*>(devOut);
    bool ok = true;
    for (int t = 0; t < 32; ++t) {
        float expected = static_cast<float>(t ^ 1);
        if (std::abs(ptr[t] - expected) > EPSILON) {
            std::cerr << "  [shfl_xor] thread " << t << ": expected " << expected
                      << " got " << ptr[t] << "\n";
            ok = false;
        }
    }
    mm.free(devOut);
    engine.shutdown();
    std::cout << (ok ? "[PASS]" : "[FAIL]") << " shfl_xor\n";
    return ok;
}

int main() {
    bool all_pass = true;

    all_pass &= test_broadcast();
    all_pass &= test_warp_reduce();
    all_pass &= test_shfl_up();
    all_pass &= test_shfl_xor();

    return all_pass ? 0 : 1;
}
