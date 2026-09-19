// P2: atomic type/op validation. Atomics act on 32- or 64-bit types only, and
// min/max + the bitwise ops (and/or/xor) require an integer type. The old codegen
// would emit invalid PTX (e.g. atom.*.min.f32) or nonsensical atomics on floats /
// sub-32-bit types; these are now located errors, while the valid combinations
// still compile. No LLVM.
//
// Tests build in Release (-DNDEBUG); asserts must stay real.
#undef NDEBUG

#include "vgre/compiler/frontend/codegen.h"

#include <cstdio>
#include <string>

using namespace vgre::compiler::frontend;

static int g_fail = 0;
#define CHECK(cond, msg)                                                   \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("FAIL: %s  (%s:%d)\n", (msg), __FILE__, __LINE__); \
            ++g_fail;                                                      \
        }                                                                  \
    } while (0)

// Compile a kernel body that performs one atomic; return whether codegen succeeded.
static bool compiles(const std::string& sig, const std::string& stmt) {
    std::string src = "extern \"C\" __global__ void k(" + sig + ", int n) {\n"
                      "  int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
                      "  if (i < n) { " + stmt + " }\n}";
    auto cg = compileToPtx(src, "k");
    return cg.ok;
}

int main() {
    // Invalid — must be rejected.
    CHECK(!compiles("float* p", "atomicMin(&p[i], 1.0f);"), "atomicMin(float*) rejected");
    CHECK(!compiles("float* p", "atomicMax(&p[i], 1.0f);"), "atomicMax(float*) rejected");
    CHECK(!compiles("float* p", "atomicAnd(&p[i], 1.0f);"), "atomicAnd(float*) rejected");
    CHECK(!compiles("float* p", "atomicOr(&p[i], 1.0f);"),  "atomicOr(float*) rejected");
    CHECK(!compiles("float* p", "atomicXor(&p[i], 1.0f);"), "atomicXor(float*) rejected");
    CHECK(!compiles("char* p, const int* a", "atomicAdd(&p[i], (char)a[i]);"), "atomicAdd(char*) rejected (sub-32-bit)");
    CHECK(!compiles("short* p, const int* a", "atomicMin(&p[i], (short)a[i]);"), "atomicMin(short*) rejected (sub-32-bit)");

    // Valid — must still compile.
    CHECK(compiles("int* p", "atomicMin(&p[i], i);"),         "atomicMin(int*) compiles");
    CHECK(compiles("int* p", "atomicMax(&p[i], i);"),         "atomicMax(int*) compiles");
    CHECK(compiles("int* p", "atomicAnd(&p[i], i);"),         "atomicAnd(int*) compiles");
    CHECK(compiles("int* p", "atomicOr(&p[i], i);"),          "atomicOr(int*) compiles");
    CHECK(compiles("int* p", "atomicXor(&p[i], i);"),         "atomicXor(int*) compiles");
    CHECK(compiles("float* p", "atomicAdd(&p[i], 1.0f);"),    "atomicAdd(float*) compiles");
    CHECK(compiles("int* p", "atomicAdd(&p[i], i);"),         "atomicAdd(int*) compiles");
    CHECK(compiles("int* p", "atomicExch(&p[i], i);"),        "atomicExch(int*) compiles");
    CHECK(compiles("int* p", "atomicCAS(&p[i], 0, i);"),      "atomicCAS(int*) compiles");
    CHECK(compiles("unsigned* p", "atomicMin(&p[i], (unsigned)i);"), "atomicMin(unsigned*) compiles");

    if (g_fail == 0)
        std::printf("PASS: atomic type/op validation\n");
    return g_fail ? 1 : 0;
}
