// P2: cast validation. A C-style cast used to silently relabel a register for any
// target type. Now casts with no meaning in the subset are located errors: to/from
// a struct, and float<->pointer (which would need a bit reinterpret, not a value
// cast). Meaningful casts (numeric conversions, int<->pointer, pointer<->pointer)
// still work. No LLVM.
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

// Compile a kernel with the given signature + body; return codegen success.
static bool compiles(const std::string& sig, const std::string& body, const std::string& pre = "") {
    std::string src = pre + "\nextern \"C\" __global__ void k(" + sig + ", int n) {\n"
                      "  int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
                      "  if (i < n) { " + body + " }\n}";
    return compileToPtx(src, "k").ok;
}

int main() {
    // Valid casts — must compile.
    CHECK(compiles("int* out, const float* a", "out[i] = (int)a[i];"), "(int)float compiles");
    CHECK(compiles("float* out, const int* a", "out[i] = (float)a[i];"), "(float)int compiles");
    CHECK(compiles("int* out, const float* a", "double d = (double)a[i]; out[i] = (int)d;"), "(double)/(int) compiles");
    CHECK(compiles("int* out, const int* a", "out[i] = (char)a[i];"), "(char) compiles");
    CHECK(compiles("int* out, const int* a", "out[i] = (int)(unsigned)a[i];"), "(unsigned)/(int) compiles");
    CHECK(compiles("int* out, const int* a", "const void* p = (const void*)a; const int* q = (const int*)p; out[i] = q[i];"),
          "pointer<->pointer casts compile");

    // Invalid casts — must be rejected.
    CHECK(!compiles("int* out, const int* a", "out[i] = (int)(float)a;"),
          "(float) of a pointer rejected");
    CHECK(!compiles("float* out, const int* a", "const int* p = a; float f = (float)p; out[i] = f;"),
          "(float)pointer rejected");
    CHECK(!compiles("int* out, const float* a", "float f = a[i]; int* p = (int*)f; out[i] = 0;"),
          "(int*)float rejected");
    CHECK(!compiles("int* out, S s", "out[i] = (int)s;", "struct S { int a; };"),
          "(int) of a struct rejected");
    CHECK(!compiles("int* out, const int* a", "S z = (S)a[i]; out[i] = z.a;", "struct S { int a; };"),
          "(struct) of an int rejected");

    if (g_fail == 0)
        std::printf("PASS: cast validation\n");
    return g_fail ? 1 : 0;
}
