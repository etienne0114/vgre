// Track Z / Stage 3.2: compound bitwise/shift assignments &= |= ^= <<= >>=.
// Previously only += -= *= /= %= were lexed, and the codegen took op[0] (so a
// multi-char op like <<= would have mis-lowered to '<'). The lexer now tokenizes
// &= |= ^= <<= >>= (the shift-assigns as 3-char tokens, before <<//>>), and the
// codegen strips the trailing '=' to get the real binary op. No LLVM.
//
// Tests build in Release (-DNDEBUG); asserts must stay real.
#undef NDEBUG

#include "vgre/compiler/backend/backend_registry.h"
#include "vgre/compiler/backend/execution_backend.h"
#include "vgre/compiler/frontend/codegen.h"

#include <cstdint>
#include <cstdio>
#include <functional>
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

static void check(const char* nm, const char* src, std::function<int(int)> ref) {
    const int block = 32, grid = 3, N = block * grid;
    std::vector<int> a(N), out(N, -777);
    for (int i = 0; i < N; ++i) a[i] = (i * 2654435761u) & 0x7fffffff;   // varied bit patterns
    void* op = out.data(); void* ap = a.data(); int n = N;
    void* args[] = {&op, &ap, &n};
    if (!runInterp(src, nm, grid, block, args, 3)) { std::printf("  %s did not run\n", nm); ++g_fail; return; }
    for (int i = 0; i < N; ++i) { int w = ref(a[i]);
        if (out[i] != w) { std::printf("  %s i=%d a=%d got=%d want=%d\n", nm, i, a[i], out[i], w); ++g_fail; return; } }
}

int main() {
    check("ca_and", R"(extern "C" __global__ void ca_and(int* out,const int* a,int n){
        int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){int x=a[i]; x&=0x0ff0; out[i]=x;}})",
        [](int a){ return a & 0x0ff0; });
    check("ca_or", R"(extern "C" __global__ void ca_or(int* out,const int* a,int n){
        int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){int x=a[i]; x|=0x1001; out[i]=x;}})",
        [](int a){ return a | 0x1001; });
    check("ca_xor", R"(extern "C" __global__ void ca_xor(int* out,const int* a,int n){
        int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){int x=a[i]; x^=0xabcd; out[i]=x;}})",
        [](int a){ return a ^ 0xabcd; });
    check("ca_shl", R"(extern "C" __global__ void ca_shl(int* out,const int* a,int n){
        int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){int x=a[i]; x<<=3; out[i]=x;}})",
        [](int a){ return a << 3; });
    check("ca_shr", R"(extern "C" __global__ void ca_shr(int* out,const int* a,int n){
        int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){int x=a[i]; x>>=4; out[i]=x;}})",
        [](int a){ return a >> 4; });
    // Chained compound ops (a common hashing idiom: mix bits).
    check("ca_chain", R"(extern "C" __global__ void ca_chain(int* out,const int* a,int n){
        int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){int h=a[i]; h^=h>>16; h*=0x45d9f3b; h&=0x7fffffff; out[i]=h;}})",
        [](int a){ int h=a; h^=h>>16; h*=0x45d9f3b; h&=0x7fffffff; return h; });

    if (g_fail == 0)
        std::printf("PASS: compound assignments (&= |= ^= <<= >>=)\n");
    return g_fail ? 1 : 0;
}
