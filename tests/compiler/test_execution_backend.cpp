// Track Z (Zero-Burden Engine): the pluggable ExecutionBackend seam.
//
// Runs a real saxpy PTX kernel through the Tier-0 interpreter backend via the
// public ExecutionBackend interface + backend registry — the same path the
// runtime will take once kernel launch is routed through backends. Proves a
// kernel executes with NO LLVM/codegen involved.
//
// Tests build in Release (-DNDEBUG); asserts must stay real.
#undef NDEBUG

#include "vgre/compiler/backend/backend_registry.h"
#include "vgre/compiler/backend/execution_backend.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using vgre::compiler::backend::ExecutionBackend;
using vgre::compiler::backend::LaunchConfig;
using vgre::compiler::backend::makeBackend;
using vgre::compiler::backend::makeDefaultBackend;

static int g_fail = 0;
#define CHECK(cond, msg)                                                   \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("FAIL: %s  (%s:%d)\n", (msg), __FILE__, __LINE__); \
            ++g_fail;                                                      \
        }                                                                  \
    } while (0)

// nvcc -arch=sm_52 style saxpy: y[i] = a*x[i] + y[i] for i < n.
static const char* kSaxpy = R"(
.version 7.0
.target sm_52
.address_size 64

.visible .entry saxpy(
    .param .f32 saxpy_param_0,
    .param .u64 saxpy_param_1,
    .param .u64 saxpy_param_2,
    .param .u32 saxpy_param_3
)
{
    .reg .pred  %p<2>;
    .reg .f32   %f<5>;
    .reg .b32   %r<6>;
    .reg .b64   %rd<8>;

    ld.param.f32    %f1, [saxpy_param_0];
    ld.param.u64    %rd1, [saxpy_param_1];
    ld.param.u64    %rd2, [saxpy_param_2];
    ld.param.u32    %r2, [saxpy_param_3];
    cvta.to.global.u64  %rd3, %rd1;
    cvta.to.global.u64  %rd4, %rd2;
    mov.u32         %r3, %ctaid.x;
    mov.u32         %r4, %ntid.x;
    mov.u32         %r5, %tid.x;
    mad.lo.s32      %r1, %r3, %r4, %r5;
    setp.ge.s32     %p1, %r1, %r2;
    @%p1 bra        $L__BB0_2;
    mul.wide.s32    %rd5, %r1, 4;
    add.s64         %rd6, %rd3, %rd5;
    ld.global.f32   %f2, [%rd6];
    add.s64         %rd7, %rd4, %rd5;
    ld.global.f32   %f3, [%rd7];
    fma.rn.f32      %f4, %f2, %f1, %f3;
    st.global.f32   [%rd7], %f4;
$L__BB0_2:
    ret;
}
)";

// Run saxpy over gridX CTAs × blockX threads through `be` and verify the result.
static void run_saxpy(ExecutionBackend& be, const char* label) {
    const int N = 64;
    const float a = 2.5f;
    std::vector<float> x(N), y(N), y_ref(N);
    for (int i = 0; i < N; ++i) {
        x[i] = static_cast<float>(i) * 0.5f;
        y[i] = static_cast<float>(N - i);
        y_ref[i] = a * x[i] + y[i];
    }

    auto kernel = be.preparePtx(kSaxpy, "saxpy");
    CHECK(kernel != nullptr, "preparePtx(saxpy) succeeds");
    if (!kernel) return;
    CHECK(kernel->entry() == "saxpy", "prepared kernel reports its entry name");

    float* xp = x.data();
    float* yp = y.data();
    int n = N;
    float av = a;
    void* args[] = {&av, &xp, &yp, &n};

    LaunchConfig cfg;             // 2 CTAs × 32 threads = 64 elements
    cfg.gridDim[0] = 2;
    cfg.blockDim[0] = 32;

    bool ok = be.launch(*kernel, cfg, args, 4);
    CHECK(ok, "backend launch runs to completion");

    float max_err = 0.0f;
    for (int i = 0; i < N; ++i)
        max_err = std::fmax(max_err, std::fabs(y[i] - y_ref[i]));
    CHECK(max_err < 1e-5f, "saxpy result matches reference");
    std::printf("  [%s] saxpy max error = %.3e\n", label, max_err);
}

int main() {
    // 1) Registry: explicit interpreter backend.
    auto interp = makeBackend("interpreter");
    CHECK(interp != nullptr, "makeBackend(\"interpreter\") returns a backend");
    if (interp) {
        CHECK(std::string(interp->name()) == "interpreter", "backend name is 'interpreter'");
        CHECK(interp->acceptsPtx(), "interpreter backend accepts PTX");
        run_saxpy(*interp, "interpreter");
    }

    // 2) Unknown backend name returns nullptr (no crash).
    CHECK(makeBackend("does-not-exist") == nullptr, "unknown backend name -> nullptr");

    // 3) Default backend (honours $VGRE_EXEC_BACKEND, else interpreter).
    auto def = makeDefaultBackend();
    CHECK(def != nullptr, "makeDefaultBackend() returns a backend");
    if (def) run_saxpy(*def, "default");

    // 4) A bad kernel fails at prepare, not at launch.
    if (interp) {
        auto bad = interp->preparePtx("this is not ptx", "nope");
        CHECK(bad == nullptr, "preparePtx on malformed PTX -> nullptr");
    }

    if (g_fail == 0)
        std::printf("PASS: ExecutionBackend (Tier-0 interpreter) — all checks green\n");
    else
        std::printf("FAILED: %d check(s)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
