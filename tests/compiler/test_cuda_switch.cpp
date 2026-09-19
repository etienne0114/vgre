// Track Z / Stage 3.2: switch statement. `switch (e) { case K: …; break; …;
// default: … }` — the front-end evaluates the selector once, jumps to the first
// matching case value (or default / past-the-end), and honors C fall-through
// (no implicit break). `break` inside a switch exits it (via the same break-target
// stack as loops); `continue` still targets the nearest enclosing loop. No LLVM.
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
    for (int i = 0; i < N; ++i) a[i] = i;   // selectors 0..N-1 hit every case + default
    void* op = out.data(); void* ap = a.data(); int n = N;
    void* args[] = {&op, &ap, &n};
    if (!runInterp(src, nm, grid, block, args, 3)) { std::printf("  %s did not run\n", nm); ++g_fail; return; }
    for (int i = 0; i < N; ++i) { int w = ref(a[i]);
        if (out[i] != w) { std::printf("  %s i=%d a=%d got=%d want=%d\n", nm, i, a[i], out[i], w); ++g_fail; return; } }
}

int main() {
    // Basic switch with break in each case + default.
    check("sw_basic", R"(extern "C" __global__ void sw_basic(int* out,const int* a,int n){
        int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){int r; switch(a[i]&3){
            case 0: r=10; break; case 1: r=20; break; case 2: r=30; break; default: r=99; } out[i]=r;}})",
        [](int a){ int m=a&3; return m==0?10:m==1?20:m==2?30:99; });

    // Fall-through: case 0 and 1 share the same body (no break after case 0).
    check("sw_fallthrough", R"(extern "C" __global__ void sw_fallthrough(int* out,const int* a,int n){
        int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){int r=0; switch(a[i]%4){
            case 0:
            case 1: r=5; break;
            case 2: r=7; break;
            default: r=1; } out[i]=r;}})",
        [](int a){ int m=a%4; return (m==0||m==1)?5:(m==2?7:1); });

    // No default: unmatched selector leaves r at its initial value.
    check("sw_nodefault", R"(extern "C" __global__ void sw_nodefault(int* out,const int* a,int n){
        int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){int r=-1; switch(a[i]%5){
            case 1: r=100; break; case 3: r=300; break; } out[i]=r;}})",
        [](int a){ int m=a%5; return m==1?100:m==3?300:-1; });

    // switch inside a loop: break exits the switch, loop continues.
    check("sw_in_loop", R"(extern "C" __global__ void sw_in_loop(int* out,const int* a,int n){
        int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){int s=0; for(int j=0;j<4;j++){
            switch(j){ case 0: s+=1; break; case 2: s+=10; break; default: s+=100; } } out[i]=s;}})",
        [](int){ return 1 + 100 + 10 + 100; });   // j=0:+1, j=1:+100, j=2:+10, j=3:+100

    if (g_fail == 0)
        std::printf("PASS: switch (basic / fall-through / no-default / in-loop)\n");
    return g_fail ? 1 : 0;
}
