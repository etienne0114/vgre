// Track Z (Zero-Burden Engine): the CUDA-C front-end parser -> AST.
// Parses a real vecAdd kernel and checks the AST shape, then checks that a
// malformed kernel reports a located error instead of crashing.
//
// Tests build in Release (-DNDEBUG); asserts must stay real.
#undef NDEBUG

#include "vgre/compiler/frontend/parser.h"

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

static const char* kVecAdd = R"(
extern "C" __global__ void vecAdd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
)";

int main() {
    ParseResult r = parse(kVecAdd);
    CHECK(r.ok, "vecAdd parses");
    if (!r.ok) { std::printf("  parse error: %s\n", r.error.c_str()); return 1; }

    CHECK(r.module->kernels.size() == 1, "one kernel");
    const Kernel& k = *r.module->kernels[0];
    CHECK(k.name == "vecAdd", "kernel name");
    CHECK(k.isGlobal, "__global__ recognized");

    // Parameters: const float* a, const float* b, float* c, int n
    CHECK(k.params.size() == 4, "four params");
    CHECK(k.params[0].type.base == Type::Float && k.params[0].type.ptr == 1 &&
          k.params[0].type.isConst && k.params[0].name == "a", "param0 = const float* a");
    CHECK(k.params[2].type.base == Type::Float && k.params[2].type.ptr == 1 &&
          !k.params[2].type.isConst && k.params[2].name == "c", "param2 = float* c");
    CHECK(k.params[3].type.base == Type::Int && k.params[3].type.ptr == 0 &&
          k.params[3].name == "n", "param3 = int n");

    // Body: VarDecl(i) then If.
    CHECK(k.body.size() == 2, "two top-level statements");
    const Stmt& decl = *k.body[0];
    CHECK(decl.kind == Stmt::VarDecl && decl.name == "i" && decl.type.base == Type::Int,
          "stmt0 = int i = ...");
    // init = ( (blockIdx.x * blockDim.x) + threadIdx.x )
    CHECK(decl.expr && decl.expr->kind == Expr::Binary && decl.expr->str == "+",
          "init is a '+' expression");
    const Expr& add = *decl.expr;
    CHECK(add.args[0]->kind == Expr::Binary && add.args[0]->str == "*",
          "left of '+' is a '*' (precedence: * binds tighter)");
    const Expr& mul = *add.args[0];
    CHECK(mul.args[0]->kind == Expr::Member && mul.args[0]->str == "x" &&
          mul.args[0]->args[0]->kind == Expr::Ident && mul.args[0]->args[0]->str == "blockIdx",
          "mul.lhs = blockIdx.x");
    CHECK(add.args[1]->kind == Expr::Member && add.args[1]->str == "x" &&
          add.args[1]->args[0]->str == "threadIdx", "add.rhs = threadIdx.x");

    // If: cond (i < n), then-block with one assignment.
    const Stmt& iff = *k.body[1];
    CHECK(iff.kind == Stmt::If, "stmt1 = if");
    CHECK(iff.expr && iff.expr->kind == Expr::Binary && iff.expr->str == "<", "if cond is '<'");
    CHECK(iff.body.size() == 1 && iff.body[0]->kind == Stmt::Block, "then is a block");
    CHECK(iff.elseBody.empty(), "no else");
    const Stmt& thenBlk = *iff.body[0];
    CHECK(thenBlk.body.size() == 1 && thenBlk.body[0]->kind == Stmt::ExprStmt,
          "then block has one expr-stmt");
    const Expr& asgn = *thenBlk.body[0]->expr;
    CHECK(asgn.kind == Expr::Assign && asgn.str == "=", "assignment '='");
    CHECK(asgn.args[0]->kind == Expr::Index && asgn.args[0]->args[0]->str == "c",
          "lhs = c[i]");
    CHECK(asgn.args[1]->kind == Expr::Binary && asgn.args[1]->str == "+" &&
          asgn.args[1]->args[0]->kind == Expr::Index &&
          asgn.args[1]->args[1]->kind == Expr::Index, "rhs = a[i] + b[i]");

    // A saxpy-style kernel with fma-able expression + a cast-free flow parses too.
    ParseResult r2 = parse(
        "__global__ void scale(float* y, float a, int n){ int i = threadIdx.x;"
        " if (i < n) y[i] = a * y[i]; }");
    CHECK(r2.ok, "second kernel parses");

    // Malformed: missing ')' -> a located error, not a crash.
    ParseResult bad = parse("__global__ void k(int a { }");
    CHECK(!bad.ok && !bad.error.empty(), "malformed kernel reports an error");
    std::printf("  (expected) error: %s\n", bad.error.c_str());

    if (g_fail == 0)
        std::printf("PASS: CUDA-C front-end parser — all checks green\n");
    else
        std::printf("FAILED: %d check(s)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
