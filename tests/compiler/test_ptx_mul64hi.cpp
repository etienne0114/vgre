// 64-bit high multiply on the PTX interpreter: mul.hi.{s64,u64} and mad.hi.s64
// (previously threw "unsupported"; now compute the true high 64 bits of a 64×64
// product via the shared __mul64hi/__umul64hi helpers). Verified vs a host reference.
//
// Tests build in Release (-DNDEBUG); asserts must stay real.
#undef NDEBUG

#include "vgre/compiler/backend/backend_registry.h"
#include "vgre/compiler/backend/execution_backend.h"
#include "vgre/compiler/cpu_cuda_intrinsics.h"   // host reference: __mul64hi/__umul64hi

#include <cstdint>
#include <cstdio>
#include <vector>

namespace be = vgre::compiler::backend;

static int g_fail = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("FAIL: %s\n", (m)); ++g_fail; } } while (0)

// out[0] = mul.hi.s64(a,b), out[1] = mul.hi.u64(a,b), out[2] = mad.hi.s64(a,b,c).
static const char* kPtx = R"(
.version 7.0
.target sm_52
.address_size 64

.visible .entry mulhi64(
    .param .u64 p_a,
    .param .u64 p_b,
    .param .u64 p_c,
    .param .u64 p_out
)
{
    .reg .b64 %rd<12>;
    ld.param.u64    %rd1, [p_a];
    ld.param.u64    %rd2, [p_b];
    ld.param.u64    %rd3, [p_c];
    ld.param.u64    %rd4, [p_out];
    cvta.to.global.u64 %rd5, %rd4;
    mul.hi.s64      %rd6, %rd1, %rd2;
    st.global.u64   [%rd5], %rd6;
    mul.hi.u64      %rd7, %rd1, %rd2;
    st.global.u64   [%rd5+8], %rd7;
    mad.hi.s64      %rd8, %rd1, %rd2, %rd3;
    st.global.u64   [%rd5+16], %rd8;
    ret;
}
)";

int main() {
    auto backend = be::makeBackend("interpreter");
    auto k = backend->preparePtx(kPtx, "mulhi64");
    CHECK(k != nullptr, "mul.hi.64 PTX loads");
    if (!k) { std::printf("FAILED: %d\n", ++g_fail); return 1; }

    struct Case { uint64_t a, b, c; };
    const Case cases[] = {
        {0x123456789ABCDEF0ull, 0x0FEDCBA987654321ull, 42},
        {(uint64_t)-3, (uint64_t)5, 100},                       // signed: -3 * 5
        {(uint64_t)-7, (uint64_t)-11, 0},                       // signed: (-7)*(-11)
        {0xFFFFFFFFFFFFFFFFull, 0xFFFFFFFFFFFFFFFFull, 1},       // max unsigned
        {0x8000000000000000ull, 2, 7},                          // sign bit set
    };
    for (const auto& tc : cases) {
        std::vector<uint64_t> out(3, 0);
        uint64_t a = tc.a, b = tc.b, c = tc.c; uint64_t* op = out.data();
        void* args[] = {&a, &b, &c, &op};
        be::LaunchConfig lc; lc.gridDim[0] = 1; lc.blockDim[0] = 1;
        CHECK(backend->launch(*k, lc, args, 4), "mul.hi.64 kernel runs");

        const int64_t sHi  = vgre_cuda::__mul64hi((int64_t)a, (int64_t)b);
        const uint64_t uHi = vgre_cuda::__umul64hi(a, b);
        const int64_t madHi = sHi + (int64_t)c;
        CHECK((int64_t)out[0] == sHi,  "mul.hi.s64 == host __mul64hi");
        CHECK(out[1] == uHi,           "mul.hi.u64 == host __umul64hi");
        CHECK((int64_t)out[2] == madHi, "mad.hi.s64 == host __mul64hi + c");
        std::printf("  a=%016llx b=%016llx  s.hi=%016llx u.hi=%016llx mad=%016llx\n",
                    (unsigned long long)a, (unsigned long long)b,
                    (unsigned long long)out[0], (unsigned long long)out[1], (unsigned long long)out[2]);
    }

    if (g_fail == 0) std::printf("PASS: PTX mul.hi.64 / mad.hi.64 (interpreter == host high-multiply)\n");
    else std::printf("FAILED: %d\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
