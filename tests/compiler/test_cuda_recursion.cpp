// Recursive __device__ functions on the from-scratch Tier-1 compiled backend
// (no LLVM). Recursive helpers cannot be inlined, so the compiler compiles each
// once into a shared body over a fixed slot frame and saves/restores that frame
// around every call — the native stack carries the nested invocations. Covers
// direct recursion (factorial, Fibonacci) and mutual recursion (isEven/isOdd).
//
// Tests build in Release (-DNDEBUG); asserts must stay real.
#undef NDEBUG

#include "vgre/compiler/frontend/compiled_kernel.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace vgre::compiler::frontend;

static int g_fail = 0;
#define CHECK(cond, msg)                                                   \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("FAIL: %s  (%s:%d)\n", (msg), __FILE__, __LINE__); \
            ++g_fail;                                                      \
        }                                                                  \
    } while (0)

// fact/fib are directly recursive; sumTo recurses with a different shape (fib calls
// itself twice → interleaved live frames). isEven/isOdd are MUTUALLY recursive —
// `isOdd` is forward-declared (a function prototype) so `isEven` can call it.
static const char* kSrc = R"(
__device__ int fact(int n)  { if (n <= 1) return 1; return n * fact(n - 1); }
__device__ int fib(int n)   { if (n < 2) return n; return fib(n - 1) + fib(n - 2); }
__device__ int sumTo(int n) { if (n <= 0) return 0; return n + sumTo(n - 1); }
__device__ int isOdd(int n);
__device__ int isEven(int n) { if (n == 0) return 1; return isOdd(n - 1); }
__device__ int isOdd(int n)  { if (n == 0) return 0; return isEven(n - 1); }

extern "C" __global__ void k(int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = fact(i % 8) + fib(i % 13) + sumTo(i % 20) * 1000 + isEven(i) * 1000000;
})";

static int hfact(int n)  { return n <= 1 ? 1 : n * hfact(n - 1); }
static int hfib(int n)   { return n < 2 ? n : hfib(n - 1) + hfib(n - 2); }
static int hsumTo(int n) { return n <= 0 ? 0 : n + hsumTo(n - 1); }

int main() {
    std::string err;
    auto ck = CompiledKernel::compileSource(kSrc, "k", err);
    CHECK(ck != nullptr, "recursive kernel compiles on the compiled tier");
    if (!ck) { std::printf("  err: %s\n", err.c_str()); std::printf("FAILED: %d\n", g_fail); return 1; }

    const int N = 64;
    std::vector<int> out(N, -1);
    int* op = out.data(); int n = N;
    void* args[] = {&op, &n};
    CHECK(ck->launch(Extent{1, 1, 1}, Extent{(uint32_t)N, 1, 1}, args, 2), "recursive kernel runs");

    int bad = 0;
    for (int i = 0; i < N; ++i) {
        int ref = hfact(i % 8) + hfib(i % 13) + hsumTo(i % 20) * 1000 + ((i % 2 == 0) ? 1 : 0) * 1000000;
        if (out[i] != ref) { if (bad < 4) std::printf("  mismatch i=%d got=%d exp=%d\n", i, out[i], ref); ++bad; }
    }
    CHECK(bad == 0, "recursive fact + fib + sumTo + mutual isEven/isOdd match the host reference");
    std::printf("  recursion: %d threads, fact/fib/sumTo + mutual isEven/isOdd exact (%d bad)\n", N, bad);

    if (g_fail == 0) std::printf("PASS: recursive __device__ functions on the compiled tier (no LLVM)\n");
    else std::printf("FAILED: %d\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
