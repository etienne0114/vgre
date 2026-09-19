// P3: the structural PTX verifier. It self-checks the codegen's output — a
// failure means a codegen bug produced malformed PTX. This exercises the verifier
// directly: real generated PTX passes, and hand-crafted defects (undefined branch
// target, out-of-range register, empty operand) are caught. No LLVM.
//
// Tests build in Release (-DNDEBUG); asserts must stay real.
#undef NDEBUG

#include "vgre/compiler/frontend/codegen.h"
#include "vgre/compiler/frontend/ptx_verifier.h"

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

static const char* head =
    ".version 7.0\n.target sm_52\n.address_size 64\n\n"
    ".visible .entry k(\n\t.param .u64 out\n)\n{\n";

int main() {
    // Real generated PTX verifies.
    {
        auto cg = compileToPtx(
            "extern \"C\" __global__ void k(int* out, const int* a, int n) {\n"
            "  int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
            "  if (i < n) out[i] = a[i] * 2 + 1;\n}", "k");
        CHECK(cg.ok, "real kernel compiles (passes verification)");
        PtxVerifyResult v = verifyPtx(cg.ptx);
        CHECK(v.ok, "real generated PTX verifies clean");
    }

    // Well-formed minimal PTX.
    {
        std::string ptx = std::string(head) +
            "\t.reg .b32 %r<3>;\n"
            "\tmov.u32 %r0, 5;\n"
            "\tmov.u32 %r1, %r0;\n"
            "\tadd.s32 %r2, %r0, %r1;\n"
            "\tret;\n}\n";
        CHECK(verifyPtx(ptx).ok, "well-formed minimal PTX passes");
    }
    // Branch to an undefined label.
    {
        std::string ptx = std::string(head) +
            "\t.reg .b32 %r<1>;\n"
            "\tmov.u32 %r0, 5;\n"
            "\tbra $L9;\n"          // $L9 is never defined
            "\tret;\n}\n";
        CHECK(!verifyPtx(ptx).ok, "undefined branch label is rejected");
    }
    // Out-of-range register (%r5 with only %r<2>).
    {
        std::string ptx = std::string(head) +
            "\t.reg .b32 %r<2>;\n"
            "\tmov.u32 %r0, 5;\n"
            "\tmov.u32 %r5, %r0;\n"   // %r5 >= declared 2
            "\tret;\n}\n";
        CHECK(!verifyPtx(ptx).ok, "out-of-range register is rejected");
    }
    // Empty operand (the shape the old scalar-__shared__ ++ bug produced).
    {
        std::string ptx = std::string(head) +
            "\t.reg .b32 %r<2>;\n"
            "\tmov.u32 %r0, 5;\n"
            "\tadd.s32 %r1, , %r0;\n"   // empty middle operand
            "\tret;\n}\n";
        CHECK(!verifyPtx(ptx).ok, "empty operand is rejected");
    }
    // A register class used but never declared.
    {
        std::string ptx = std::string(head) +
            "\t.reg .b32 %r<2>;\n"
            "\tmov.f32 %f0, 0f3F800000;\n"   // %f never declared
            "\tret;\n}\n";
        CHECK(!verifyPtx(ptx).ok, "undeclared register class is rejected");
    }

    if (g_fail == 0)
        std::printf("PASS: PTX verifier (accepts valid, rejects malformed)\n");
    return g_fail ? 1 : 0;
}
