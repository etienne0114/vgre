// Track Z / Stage 3.2: control flow — while / do-while loops, break / continue,
// and multi-variable declarations (`int a, b = 0;`). break/continue use a
// loop-context stack in codegen: `break` branches to the loop's exit, `continue`
// to its re-test (while/do-while) or increment (for). Nested loops break/continue
// only the innermost. No LLVM.
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

// Run kernel(out,a,n) over N ints; compare each to ref(i, a).
static void check(const char* nm, const char* src, std::function<int(int, std::vector<int>&)> ref) {
    const int block = 32, grid = 3, N = block * grid;
    std::vector<int> a(N), out(N, -777);
    for (int i = 0; i < N; ++i) a[i] = (i % 23) - 9;
    void* op = out.data(); void* ap = a.data(); int n = N;
    void* args[] = {&op, &ap, &n};
    if (!runInterp(src, nm, grid, block, args, 3)) { std::printf("  %s did not run\n", nm); ++g_fail; return; }
    for (int i = 0; i < N; ++i) { int w = ref(i, a);
        if (out[i] != w) { std::printf("  %s i=%d got=%d want=%d\n", nm, i, out[i], w); ++g_fail; return; } }
}

int main() {
    // while + multi-variable declaration: sum a[i] four times.
    check("cf_while", R"(extern "C" __global__ void cf_while(int* out,const int* a,int n){
        int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){int s=0,j=0; while(j<4){s+=a[i]; j++;} out[i]=s;}})",
        [](int i, std::vector<int>& a){ return 4 * a[i]; });

    // do-while runs the body at least once even if the condition is initially false.
    check("cf_dowhile", R"(extern "C" __global__ void cf_dowhile(int* out,const int* a,int n){
        int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){int s=0,j=10; do{s+=a[i]; j++;}while(j<3); out[i]=s;}})",
        [](int i, std::vector<int>& a){ return a[i]; });   // exactly one iteration

    // break: stop the loop early at j==5.
    check("cf_break", R"(extern "C" __global__ void cf_break(int* out,const int* a,int n){
        int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){int s=0; for(int j=0;j<100;j++){ if(j==5) break; s+=1;} out[i]=s;}})",
        [](int, std::vector<int>&){ return 5; });

    // continue: count only odd j in [0,10).
    check("cf_continue", R"(extern "C" __global__ void cf_continue(int* out,const int* a,int n){
        int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){int s=0; for(int j=0;j<10;j++){ if((j&1)==0) continue; s+=1;} out[i]=s;}})",
        [](int, std::vector<int>&){ return 5; });

    // continue in a while advances the counter before continuing (no infinite loop).
    check("cf_while_continue", R"(extern "C" __global__ void cf_while_continue(int* out,const int* a,int n){
        int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){int s=0,j=0; while(j<8){ j++; if(j>4) continue; s+=1;} out[i]=s;}})",
        [](int, std::vector<int>&){ return 4; });

    // Nested loops: break exits only the inner loop; outer runs fully.
    check("cf_nested_break", R"(extern "C" __global__ void cf_nested_break(int* out,const int* a,int n){
        int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){int s=0; for(int x=0;x<3;x++){ for(int y=0;y<10;y++){ if(y==2) break; s+=1;} s+=100;} out[i]=s;}})",
        [](int, std::vector<int>&){ return 3 * 2 + 3 * 100; });   // inner adds 2 per outer, +100 each

    if (g_fail == 0)
        std::printf("PASS: control flow (while / do-while / break / continue / multi-decl)\n");
    return g_fail ? 1 : 0;
}
