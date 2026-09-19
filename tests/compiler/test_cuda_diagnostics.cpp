// P3: source-snippet diagnostics. A located error ("line:col: message") is now
// enriched with the offending source line and a caret under the column, so the
// diagnostic points at the problem instead of just naming a position. No LLVM.
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

static bool has(const std::string& s, const std::string& sub) { return s.find(sub) != std::string::npos; }

int main() {
    // A codegen error (undeclared identifier) gets the source line + caret.
    {
        const char* src =
            "extern \"C\" __global__ void k(int* out, int n) {\n"
            "    int i = threadIdx.x;\n"
            "    if (i < n) out[i] = undefined_var;\n"
            "}";
        auto cg = compileToPtx(src, "k");
        CHECK(!cg.ok, "undeclared-identifier kernel fails");
        CHECK(has(cg.error, "undeclared identifier"), "message is present");
        CHECK(has(cg.error, "out[i] = undefined_var;"), "offending source line quoted");
        CHECK(has(cg.error, "^"), "caret present");
        CHECK(cg.error.find('\n') != std::string::npos, "diagnostic is multi-line");
        std::printf("  diagnostic:\n%s\n", cg.error.c_str());
    }
    // A parse error also gets the snippet.
    {
        const char* src =
            "extern \"C\" __global__ void k(int* out, int n) {\n"
            "    int i = ;\n"                    // missing expression
            "}";
        auto cg = compileToPtx(src, "k");
        CHECK(!cg.ok, "parse-error kernel fails");
        CHECK(has(cg.error, "^"), "parse-error diagnostic has a caret");
    }
    // A valid kernel still compiles (no diagnostic).
    {
        auto cg = compileToPtx(
            "extern \"C\" __global__ void k(int* out, const int* a, int n) {\n"
            "  int i = blockIdx.x*blockDim.x+threadIdx.x; if (i<n) out[i]=a[i];\n}", "k");
        CHECK(cg.ok, "valid kernel compiles clean");
        CHECK(cg.error.empty(), "no error on success");
    }

    if (g_fail == 0)
        std::printf("PASS: source-snippet diagnostics\n");
    return g_fail ? 1 : 0;
}
