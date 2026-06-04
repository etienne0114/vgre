// QUEUE-69: cudaStreamCreateWithPriority + cudaStreamGetPriority via public C API.
// Priority is stored in the shim's g_streamMeta map at create time and read back
// without going through the interceptor. O(1) per call.
// Tests:
//   1. Priority range: least >= greatest (CUDA convention: negative = high priority).
//   2. Create with greatest priority, retrieve it.
//   3. Create with least priority, retrieve it.
//   4. NULL stream always reports priority 0.
//   5. Destroy + re-create: meta cleared then repopulated correctly.

#include <cassert>
#include <cstring>
#include <iostream>

#define PASS(msg) std::cout << "[PASS] " << msg << "\n"
#define FAIL(msg) do { std::cerr << "[FAIL] " << msg << "\n"; return 1; } while(0)

typedef void* cudaStream_t;
typedef int cudaError_t;
static constexpr cudaError_t cudaSuccess = 0;
static constexpr cudaError_t cudaErrorInvalidValue = 1;

extern "C" {
cudaError_t cudaStreamCreateWithPriority(cudaStream_t*, unsigned int, int);
cudaError_t cudaStreamGetPriority(cudaStream_t, int*);
cudaError_t cudaDeviceGetStreamPriorityRange(int*, int*);
cudaError_t cudaStreamDestroy(cudaStream_t);
}

// ── 1. Priority range is well-formed ─────────────────────────────────────────
int test_priority_range() {
    int least = 999, greatest = 999;
    cudaError_t st = cudaDeviceGetStreamPriorityRange(&least, &greatest);
    if (st != cudaSuccess) FAIL("deviceGetStreamPriorityRange status");
    // VGRE: higher numeric value = higher urgency; greatest >= least.
    if (greatest < least) FAIL("greatest should be >= least");
    PASS("priority range: least=" + std::to_string(least) + " greatest=" + std::to_string(greatest));
    return 0;
}

// ── 2. Create with greatest (highest) priority, read it back ─────────────────
int test_create_greatest() {
    int least = 0, greatest = 0;
    cudaDeviceGetStreamPriorityRange(&least, &greatest);

    cudaStream_t s = nullptr;
    cudaError_t st = cudaStreamCreateWithPriority(&s, 0u, greatest);
    if (st != cudaSuccess) FAIL("create with greatest priority");
    if (!s) FAIL("stream handle is null");

    int got = 999;
    st = cudaStreamGetPriority(s, &got);
    if (st != cudaSuccess) FAIL("getpriority status");
    if (got != greatest) FAIL("got=" + std::to_string(got) + " want=" + std::to_string(greatest));

    cudaStreamDestroy(s);
    PASS("create+get greatest priority=" + std::to_string(greatest));
    return 0;
}

// ── 3. Create with least (lowest) priority, read it back ─────────────────────
int test_create_least() {
    int least = 0, greatest = 0;
    cudaDeviceGetStreamPriorityRange(&least, &greatest);

    cudaStream_t s = nullptr;
    cudaError_t st = cudaStreamCreateWithPriority(&s, 0u, least);
    if (st != cudaSuccess) FAIL("create with least priority");

    int got = 999;
    st = cudaStreamGetPriority(s, &got);
    if (st != cudaSuccess) FAIL("getpriority status");
    if (got != least) FAIL("got=" + std::to_string(got) + " want=" + std::to_string(least));

    cudaStreamDestroy(s);
    PASS("create+get least priority=" + std::to_string(least));
    return 0;
}

// ── 4. NULL stream has priority 0 ────────────────────────────────────────────
int test_null_stream_priority() {
    int got = 999;
    cudaError_t st = cudaStreamGetPriority(nullptr, &got);
    if (st != cudaSuccess) FAIL("null stream getpriority status");
    if (got != 0) FAIL("null stream priority should be 0, got " + std::to_string(got));
    PASS("NULL stream priority = 0");
    return 0;
}

// ── 5. Destroy then re-create: fresh priority stored ─────────────────────────
int test_destroy_recreate() {
    int least = 0, greatest = 0;
    cudaDeviceGetStreamPriorityRange(&least, &greatest);

    cudaStream_t s = nullptr;
    cudaStreamCreateWithPriority(&s, 0u, greatest);
    cudaStreamDestroy(s);

    // Re-create same handle slot with least priority
    cudaStream_t s2 = nullptr;
    cudaStreamCreateWithPriority(&s2, 0u, least);

    int got = 999;
    cudaStreamGetPriority(s2, &got);
    if (got != least) FAIL("after destroy+recreate got=" + std::to_string(got)
                           + " want least=" + std::to_string(least));
    cudaStreamDestroy(s2);
    PASS("destroy+recreate gives fresh priority");
    return 0;
}

// ── 6. NULL pointer guard ─────────────────────────────────────────────────────
int test_null_guard() {
    cudaError_t st = cudaStreamGetPriority(nullptr, nullptr);
    if (st != cudaErrorInvalidValue) FAIL("null priority pointer should return error");
    PASS("null priority pointer returns cudaErrorInvalidValue");
    return 0;
}

int main() {
    std::cout << "=== cudaStreamCreateWithPriority/GetPriority tests (QUEUE-69) ===\n";
    int fails = 0;
    fails += test_priority_range();
    fails += test_create_greatest();
    fails += test_create_least();
    fails += test_null_stream_priority();
    fails += test_destroy_recreate();
    fails += test_null_guard();
    if (fails == 0) std::cout << "All QUEUE-69 stream priority tests passed.\n";
    return fails;
}
