// P0 #1: char/short/bool memory operations. A char* / short* / bool* access must
// load and store at the type's REAL width (1 or 2 bytes), not 4 — otherwise the
// address stride (elemBytes) and the ld/st width disagree and neighbouring
// elements are read/clobbered. Signed char/short must sign-extend on load; bool
// and unsigned narrow types zero-extend. This was silently wrong (memSuffix
// collapsed everything to u32) and untested. No LLVM.
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

static bool runInterp(const char* src, const char* name, int grid, int block,
                      void* const* args, int numArgs) {
    auto cg = compileToPtx(src, name);
    if (!cg.ok) { std::printf("  codegen %s: %s\n", name, cg.error.c_str()); return false; }
    auto beI = be::makeBackend("interpreter");
    auto k = beI->preparePtx(cg.ptx, name);
    if (!k) { std::printf("  prepare %s:\n%s\n", name, cg.ptx.c_str()); return false; }
    be::LaunchConfig lc; lc.gridDim[0] = (uint32_t)grid; lc.blockDim[0] = (uint32_t)block;
    return beI->launch(*k, lc, args, numArgs);
}

// Load a signed char, widen to int (must sign-extend, and read exactly 1 byte).
static const char* kCharLoad = R"(
extern "C" __global__ void cload(int* out, const char* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i];
})";
// Load an unsigned char → int (must zero-extend).
static const char* kUCharLoad = R"(
extern "C" __global__ void ucload(int* out, const unsigned char* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i];
})";
// Load a signed short → int (sign-extend, 2-byte read).
static const char* kShortLoad = R"(
extern "C" __global__ void sload(int* out, const short* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i];
})";
// Store int → char (truncate to low byte); each thread writes ONE byte, so
// neighbours must be untouched (proves the store width is 1, not 4).
static const char* kCharStore = R"(
extern "C" __global__ void cstore(char* out, const int* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (char)in[i];
})";
// Store int → short (2-byte writes, independent neighbours).
static const char* kShortStore = R"(
extern "C" __global__ void sstore(short* out, const int* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (short)in[i];
})";

int main() {
    const int block = 32, grid = 4, N = block * grid;

    // signed char load: values spanning negatives; out == (int)(signed char).
    {
        std::vector<signed char> in(N);
        for (int i = 0; i < N; ++i) in[i] = (signed char)(i * 7 - 130);
        std::vector<int> out(N, 12345);
        void* op = out.data(); void* ip = in.data(); int n = N;
        void* args[] = {&op, &ip, &n};
        CHECK(runInterp(kCharLoad, "cload", grid, block, args, 3), "cload runs");
        bool ok = true;
        for (int i = 0; i < N; ++i) if (out[i] != (int)in[i]) { ok = false;
            std::printf("  cload i=%d got=%d want=%d\n", i, out[i], (int)in[i]); break; }
        CHECK(ok, "signed char load sign-extends and reads 1 byte");
    }
    // unsigned char load: zero-extend (200 stays 200, not -56).
    {
        std::vector<unsigned char> in(N);
        for (int i = 0; i < N; ++i) in[i] = (unsigned char)(i * 9 + 100);
        std::vector<int> out(N, -1);
        void* op = out.data(); void* ip = in.data(); int n = N;
        void* args[] = {&op, &ip, &n};
        CHECK(runInterp(kUCharLoad, "ucload", grid, block, args, 3), "ucload runs");
        bool ok = true;
        for (int i = 0; i < N; ++i) if (out[i] != (int)in[i]) { ok = false;
            std::printf("  ucload i=%d got=%d want=%d\n", i, out[i], (int)in[i]); break; }
        CHECK(ok, "unsigned char load zero-extends");
    }
    // signed short load.
    {
        std::vector<short> in(N);
        for (int i = 0; i < N; ++i) in[i] = (short)(i * 617 - 20000);
        std::vector<int> out(N, -1);
        void* op = out.data(); void* ip = in.data(); int n = N;
        void* args[] = {&op, &ip, &n};
        CHECK(runInterp(kShortLoad, "sload", grid, block, args, 3), "sload runs");
        bool ok = true;
        for (int i = 0; i < N; ++i) if (out[i] != (int)in[i]) { ok = false;
            std::printf("  sload i=%d got=%d want=%d\n", i, out[i], (int)in[i]); break; }
        CHECK(ok, "signed short load sign-extends and reads 2 bytes");
    }
    // char store: truncation + independent 1-byte writes (poison neighbours first).
    {
        std::vector<int> in(N);
        for (int i = 0; i < N; ++i) in[i] = i * 13 - 400;   // exceeds char range → truncates
        std::vector<signed char> out(N, (signed char)0x5a);
        void* op = out.data(); void* ip = in.data(); int n = N;
        void* args[] = {&op, &ip, &n};
        CHECK(runInterp(kCharStore, "cstore", grid, block, args, 3), "cstore runs");
        bool ok = true;
        for (int i = 0; i < N; ++i) if (out[i] != (signed char)in[i]) { ok = false;
            std::printf("  cstore i=%d got=%d want=%d\n", i, (int)out[i], (int)(signed char)in[i]); break; }
        CHECK(ok, "char store writes 1 byte (truncate, no neighbour clobber)");
    }
    // short store.
    {
        std::vector<int> in(N);
        for (int i = 0; i < N; ++i) in[i] = i * 1234 - 50000;
        std::vector<short> out(N, (short)0x5a5a);
        void* op = out.data(); void* ip = in.data(); int n = N;
        void* args[] = {&op, &ip, &n};
        CHECK(runInterp(kShortStore, "sstore", grid, block, args, 3), "sstore runs");
        bool ok = true;
        for (int i = 0; i < N; ++i) if (out[i] != (short)in[i]) { ok = false;
            std::printf("  sstore i=%d got=%d want=%d\n", i, (int)out[i], (int)(short)in[i]); break; }
        CHECK(ok, "short store writes 2 bytes (truncate, no neighbour clobber)");
    }

    if (g_fail == 0)
        std::printf("PASS: narrow-type memory ops (char/short load+store, sign/zero extend)\n");
    return g_fail ? 1 : 0;
}
