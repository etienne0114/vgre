// Phase 3, Track P3-13 — Triton IR (TTIR) frontend.
//
// Parses the textual Triton IR for two representative kernels and runs them
// directly from IR (no PTX round-trip), then checks the output matches an
// independent reference. The vector-add kernel exercises the canonical TTIR
// pattern (get_program_id → make_range → splat → addptr → masked load/store with
// a boundary mask, N not a multiple of BLOCK); the fused kernel adds a float
// constant + mulf. Correct output proves the parser + per-lane lowering, not
// just "it ran".

#include "vgre/compiler/triton/triton_frontend.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace vgre::triton;

static int g_pass = 0, g_total = 0;
static void check(const char* name, bool ok) {
    ++g_total;
    printf(ok ? "  PASS  %s\n" : "  FAIL  %s\n", name);
    if (ok) ++g_pass;
}

// Canonical Triton vector add: out[i] = x[i] + y[i], masked for i < n. BLOCK=256.
static const char* kVecAdd = R"MLIR(
module {
  tt.func public @add_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32) {
    %0 = tt.get_program_id x : i32
    %c256_i32 = arith.constant 256 : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %3 = tt.splat %1 : i32 -> tensor<256xi32>
    %4 = arith.addi %3, %2 : tensor<256xi32>
    %5 = tt.splat %arg3 : i32 -> tensor<256xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<256xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %8 = tt.addptr %7, %4 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    %9 = tt.load %8, %6 : tensor<256xf32>
    %10 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %11 = tt.addptr %10, %4 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    %12 = tt.load %11, %6 : tensor<256xf32>
    %13 = arith.addf %9, %12 : tensor<256xf32>
    %14 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %15 = tt.addptr %14, %4 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    tt.store %15, %13, %6 : tensor<256xf32>
    tt.return
  }
}
)MLIR";

// Fused: out[i] = 2.0 * x[i] + y[i], masked. Exercises a float constant + mulf.
static const char* kFused = R"MLIR(
module {
  tt.func public @fma_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32) {
    %0 = tt.get_program_id x : i32
    %c128_i32 = arith.constant 128 : i32
    %1 = arith.muli %0, %c128_i32 : i32
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %3 = tt.splat %1 : i32 -> tensor<128xi32>
    %4 = arith.addi %3, %2 : tensor<128xi32>
    %5 = tt.splat %arg3 : i32 -> tensor<128xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<128xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %8 = tt.addptr %7, %4 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %9 = tt.load %8, %6 : tensor<128xf32>
    %cst = arith.constant 2.000000e+00 : f32
    %10 = tt.splat %cst : f32 -> tensor<128xf32>
    %11 = arith.mulf %9, %10 : tensor<128xf32>
    %12 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %13 = tt.addptr %12, %4 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %14 = tt.load %13, %6 : tensor<128xf32>
    %15 = arith.addf %11, %14 : tensor<128xf32>
    %16 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %17 = tt.addptr %16, %4 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    tt.store %17, %15, %6 : tensor<128xf32>
    tt.return
  }
}
)MLIR";

static int ceilDiv(int a, int b) { return (a + b - 1) / b; }

int main() {
    printf("=== Triton IR (TTIR) frontend (P3-13) ===\n");

    std::mt19937 rng(13);
    std::uniform_real_distribution<float> ud(-5.0f, 5.0f);
    const int N = 1000;                                   // not a multiple of BLOCK
    std::vector<float> x(N), y(N);
    for (int i = 0; i < N; ++i) { x[i] = ud(rng); y[i] = ud(rng); }

    // (1) Parse + structure.
    Module m = parse(kVecAdd);
    bool structOk = m.funcs.size() == 1 && m.funcs[0].name == "add_kernel" &&
                    m.funcs[0].args.size() == 4 && !m.funcs[0].body.empty();
    check("parses tt.func signature + body from TTIR", structOk);

    // (2) Run vector-add from IR via the compiled library entry (run_kernel) and
    //     match x + y — exercises the shipped API, not just the header.
    {
        std::vector<float> out(N, -999.0f);
        std::vector<KernelArg> args = {
            {true, x.data(), 0}, {true, y.data(), 0}, {true, out.data(), 0}, {false, nullptr, N}};
        run_kernel(kVecAdd, "add_kernel", args, ceilDiv(N, 256));
        double maxErr = 0.0;
        for (int i = 0; i < N; ++i) maxErr = std::max(maxErr, (double)std::fabs(out[i] - (x[i] + y[i])));
        printf("  [info] vector-add max abs err = %.2e (grid=%d, BLOCK=256, N=%d)\n",
               maxErr, ceilDiv(N, 256), N);
        check("Triton vector-add runs from IR == reference x+y", maxErr == 0.0);
    }

    // (3) Run the fused kernel from IR and match 2*x + y.
    {
        Module mf = parse(kFused);
        std::vector<float> out(N, -999.0f);
        std::vector<KernelArg> args = {
            {true, x.data(), 0}, {true, y.data(), 0}, {true, out.data(), 0}, {false, nullptr, N}};
        Interpreter::run(mf.funcs[0], args, ceilDiv(N, 128));
        double maxErr = 0.0;
        for (int i = 0; i < N; ++i) maxErr = std::max(maxErr, (double)std::fabs(out[i] - (2.0f * x[i] + y[i])));
        printf("  [info] fused 2*x+y max abs err = %.2e\n", maxErr);
        check("Triton fused (2*x+y) runs from IR == reference", maxErr < 1e-6);
    }

    // (4) The boundary mask must leave out-of-range outputs unwritten (no OOB).
    {
        std::vector<float> out(N, 7.0f);
        // shrink the logical length so only the first 100 elements are active.
        std::vector<KernelArg> args = {
            {true, x.data(), 0}, {true, y.data(), 0}, {true, out.data(), 0}, {false, nullptr, 100}};
        Interpreter::run(m.funcs[0], args, ceilDiv(N, 256));
        bool ok = true;
        for (int i = 0; i < 100; ++i) if (out[i] != x[i] + y[i]) ok = false;
        for (int i = 100; i < N; ++i) if (out[i] != 7.0f) ok = false;   // masked → untouched
        check("boundary mask writes only active lanes (n=100)", ok);
    }

    printf("\n%d / %d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
