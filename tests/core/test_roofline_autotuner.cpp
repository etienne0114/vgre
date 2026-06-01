// VGRE Test — Roofline-Driven Kernel Auto-Tuner (QUEUE-01)
//
// Verifies:
//   1. KernelTunerLRU get/put are O(1) and respect capacity-256 LRU eviction.
//   2. A 1-D kernel launched ≥5 times still produces bitwise-correct output
//      after the auto-tuner fires and potentially re-maps block geometry.
//   3. The LRU evicts the least-recently-used entry when capacity is exceeded.
//
// Compiled with -DNDEBUG in Release; all checks use explicit if+abort().
#include "vgre/core/kernel_tuner.h"
#include "vgre/core/memory_manager.h"
#include "vgre/core/runtime_engine.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using namespace vgre;
using namespace vgre::core;

// Abort with message — works even when assert() is compiled away by NDEBUG.
#define REQUIRE(cond) \
    do { if (!(cond)) { \
        std::cerr << "FAIL: " #cond " at " __FILE__ ":" << __LINE__ << "\n"; \
        std::abort(); \
    } } while(0)

// ── 1. KernelTunerLRU unit tests ──────────────────────────────────────────

void test_lru_basic_put_get() {
    KernelTunerLRU cache;

    cache.put("k1", TuneConfig{128, 1.0});
    TuneConfig* c = cache.get("k1");
    REQUIRE(c != nullptr);
    REQUIRE(c->blockSize == 128);
    REQUIRE(cache.size() == 1);
    REQUIRE(cache.get("missing") == nullptr);

    std::cout << "[PASS] LRU basic put/get\n";
}

void test_lru_update_existing() {
    KernelTunerLRU cache;
    cache.put("k1", TuneConfig{128, 2.0});
    cache.put("k1", TuneConfig{256, 1.0}); // update same key
    TuneConfig* c = cache.get("k1");
    REQUIRE(c != nullptr);
    REQUIRE(c->blockSize == 256);
    REQUIRE(c->bestMs == 1.0);
    REQUIRE(cache.size() == 1);
    std::cout << "[PASS] LRU update existing key\n";
}

void test_lru_capacity_eviction() {
    KernelTunerLRU cache;

    // Fill to capacity: insertion order 0,1,...,255 → LRU=0, MRU=255
    for (size_t i = 0; i < KernelTunerLRU::kCapacity; ++i)
        cache.put("kernel_" + std::to_string(i), TuneConfig{64, double(i)});
    REQUIRE(cache.size() == KernelTunerLRU::kCapacity);

    // Touch kernel_0: promotes it to MRU → new LRU = kernel_1
    REQUIRE(cache.get("kernel_0") != nullptr);

    // Insert one more: evicts LRU (kernel_1)
    cache.put("kernel_overflow", TuneConfig{512, 99.0});
    REQUIRE(cache.size() == KernelTunerLRU::kCapacity);

    REQUIRE(cache.get("kernel_0") != nullptr);       // survived (was touched)
    REQUIRE(cache.get("kernel_overflow") != nullptr); // just inserted
    REQUIRE(cache.get("kernel_1") == nullptr);        // evicted as LRU

    std::cout << "[PASS] LRU capacity eviction (256 entries, LRU evicted)\n";
}

void test_lru_mru_ordering() {
    KernelTunerLRU cache;
    // Insert A, B, C — insertion order: LRU=A, MRU=C
    cache.put("A", TuneConfig{64,  1.0});
    cache.put("B", TuneConfig{128, 2.0});
    cache.put("C", TuneConfig{256, 3.0});

    // Touch A — promotes A to MRU; LRU is now B
    REQUIRE(cache.get("A") != nullptr);

    // Fill remaining capacity: after 3 inserts above, add kCapacity-3 more
    // so total == kCapacity; next insert causes eviction of current LRU = B.
    for (size_t i = 0; i < KernelTunerLRU::kCapacity - 3; ++i)
        cache.put("fill_" + std::to_string(i), TuneConfig{64, 0.0});
    REQUIRE(cache.size() == KernelTunerLRU::kCapacity);

    // This insert must evict B (oldest untouched)
    cache.put("trigger_eviction", TuneConfig{64, 0.0});
    REQUIRE(cache.size() == KernelTunerLRU::kCapacity);

    REQUIRE(cache.get("A") != nullptr); // survived (touched/promoted)
    REQUIRE(cache.get("C") != nullptr); // survived (younger than B)
    REQUIRE(cache.get("B") == nullptr); // evicted

    std::cout << "[PASS] LRU MRU ordering: touched entry survives, LRU evicted\n";
}

// ── 2. Kernel auto-tuner integration test ─────────────────────────────────
// Launch a 1-D vector-add kernel 8 times (> kAutoTuneMinSamples=5).
// Verify the output is correct on every iteration — the tuner must not
// corrupt results by remapping block geometry.
void test_autotuner_correctness() {
    constexpr int N = 8192;
    RuntimeEngine engine;
    VGREResult r = engine.initialize();
    if (r != VGREResult::SUCCESS) {
        std::cerr << "FATAL: engine.initialize() failed: " << (int)r << "\n";
        std::abort();
    }

    auto& mm = engine.getMemoryManager();
    MemoryHandle devA = nullptr, devB = nullptr, devC = nullptr;
    r = mm.allocate(N * sizeof(float), devA);
    REQUIRE(r == VGREResult::SUCCESS && devA != nullptr);
    r = mm.allocate(N * sizeof(float), devB);
    REQUIRE(r == VGREResult::SUCCESS && devB != nullptr);
    r = mm.allocate(N * sizeof(float), devC);
    REQUIRE(r == VGREResult::SUCCESS && devC != nullptr);

    std::vector<float> hA(N), hB(N), hC(N, 0.0f);
    for (int i = 0; i < N; ++i) { hA[i] = float(i); hB[i] = float(i) * 2.0f; }
    mm.copyHostToDevice(devA, hA.data(), N * sizeof(float));
    mm.copyHostToDevice(devB, hB.data(), N * sizeof(float));

    // Simple 1-D kernel: idx = blockIdx.x * blockDim.x + threadIdx.x
    // Safe for block-size remapping since indices are linear.
    const char* src = R"(
        __global__ void vadd_roofline(float* A, float* B, float* C, int n) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) C[i] = A[i] + B[i];
        }
    )";

    dim3 grid((N + 255) / 256), block(256);
    int  n    = N;
    void* args[] = {&devA, &devB, &devC, &n};

    // Launch 8 times — auto-tuner fires at sample 5, then uses cached config.
    for (int iter = 0; iter < 8; ++iter) {
        r = engine.launchKernel("vadd_roofline", src, grid, block, args);
        if (r != VGREResult::SUCCESS) {
            std::cerr << "FATAL: launchKernel failed iter=" << iter
                      << " err=" << (int)r << "\n";
            std::abort();
        }
        engine.synchronize();

        r = mm.copyDeviceToHost(hC.data(), devC, N * sizeof(float));
        REQUIRE(r == VGREResult::SUCCESS);

        int errors = 0;
        for (int i = 0; i < N; ++i) {
            if (std::fabs(hC[i] - (hA[i] + hB[i])) >= 1e-5f) ++errors;
        }
        if (errors) {
            std::cerr << "FATAL: correctness failure at iter=" << iter
                      << " errors=" << errors << "\n";
            std::abort();
        }
    }

    mm.free(devA); mm.free(devB); mm.free(devC);
    engine.shutdown();
    std::cout << "[PASS] Auto-tuner correctness: 8 launches, "
              << N << " elements verified each time\n";
}

int main() {
    std::cout << "=== VGRE: Roofline Auto-Tuner Tests ===\n";

    test_lru_basic_put_get();
    test_lru_update_existing();
    test_lru_capacity_eviction();
    test_lru_mru_ordering();
    test_autotuner_correctness();

    std::cout << "\n[ALL PASS] Roofline Auto-Tuner (QUEUE-01)\n";
    return 0;
}
