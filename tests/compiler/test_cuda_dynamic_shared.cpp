// Track Z / Stage 3.2: dynamic shared memory — `extern __shared__ T s[];`, whose
// byte size comes from the kernel launch (the 3rd launch parameter), not the
// source. The front-end lowers it to `.extern .shared` and the interpreter sizes
// the per-CTA shared arena from LaunchConfig.sharedBytes, aliasing the region
// after any static shared. A textbook block-sum reduction exercises it. No LLVM.
//
// Tests build in Release (-DNDEBUG); asserts must stay real.
#undef NDEBUG

#include "vgre/compiler/backend/backend_registry.h"
#include "vgre/compiler/backend/execution_backend.h"
#include "vgre/compiler/frontend/codegen.h"

#include <cmath>
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

// Block-sum reduction into out[blockIdx.x] using a launch-sized shared buffer.
static const char* kReduce = R"(
extern "C" __global__ void dynred(float* out, const float* in, int n) {
    extern __shared__ float s[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    s[tid] = (i < n) ? in[i] : 0.0f;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride = stride / 2) {
        if (tid < stride) s[tid] = s[tid] + s[tid + stride];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = s[0];
})";

int main() {
    const int W = 32, blocks = 4, N = W * blocks;   // block size is a power of two
    std::vector<float> in(N), out(blocks, -1);
    for (int i = 0; i < N; ++i) in[i] = (i % 9) * 0.5f - 2.0f;

    void* op = out.data(); void* ip = in.data(); int n = N;
    void* args[] = {&op, &ip, &n};

    auto cg = compileToPtx(kReduce, "dynred");
    CHECK(cg.ok, "extern __shared__ kernel compiles");
    if (cg.ok) {
        auto beI = be::makeBackend("interpreter");
        auto k = beI->preparePtx(cg.ptx, "dynred");
        CHECK(k != nullptr, "dynred prepares");
        if (k) {
            be::LaunchConfig lc;
            lc.gridDim[0] = (uint32_t)blocks;
            lc.blockDim[0] = (uint32_t)W;
            lc.sharedBytes = (size_t)W * sizeof(float);   // dynamic shared: one float per lane
            CHECK(beI->launch(*k, lc, args, 3), "dynred runs");
            bool ok = true;
            for (int b = 0; b < blocks; ++b) {
                float ref = 0; for (int j = 0; j < W; ++j) ref += in[b * W + j];
                if (std::fabs(out[b] - ref) > 1e-4f) { ok = false;
                    std::printf("  block %d: got %.5f want %.5f\n", b, out[b], ref); break; }
            }
            CHECK(ok, "block reduction via extern __shared__ == CPU sum");
        }
    } else {
        std::printf("  codegen error: %s\n", cg.error.c_str());
    }

    if (g_fail == 0)
        std::printf("PASS: dynamic shared memory (extern __shared__) block reduction\n");
    return g_fail ? 1 : 0;
}
