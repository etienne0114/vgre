// Track Z (Zero-Burden Engine): the from-scratch CUDA-C front-end lexer.
// Tokenizes a real vecAdd kernel and checks the stream — keywords, builtins,
// operators, literals, string literal (extern "C"), and comment skipping.
//
// Tests build in Release (-DNDEBUG); asserts must stay real.
#undef NDEBUG

#include "vgre/compiler/frontend/lexer.h"

#include <cstdio>
#include <string>
#include <vector>

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
    // one element per thread
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];   /* fused? no, just add */
    }
}
)";

// Count tokens of a given kind.
static int count(const std::vector<Token>& t, TokenKind k) {
    int n = 0;
    for (const auto& tok : t) if (tok.kind == k) ++n;
    return n;
}

int main() {
    std::vector<Token> t = lex(kVecAdd);

    // Must be non-empty and end with exactly one End token.
    CHECK(!t.empty() && t.back().kind == TokenKind::End, "stream ends with End");

    // Header: extern "C" __global__ void vecAdd (
    CHECK(t[0].kind == TokenKind::KwExtern, "first token is extern");
    CHECK(t[1].kind == TokenKind::StringLiteral && t[1].text == "C", "extern \"C\" string literal");
    CHECK(t[2].kind == TokenKind::KwGlobal, "__global__ keyword");
    CHECK(t[3].kind == TokenKind::KwVoid, "void keyword");
    CHECK(t[4].kind == TokenKind::Identifier && t[4].text == "vecAdd", "kernel name identifier");
    CHECK(t[5].kind == TokenKind::LParen, "opening paren");

    // Builtins are plain identifiers; the '.' is a separate Dot token.
    bool sawBlockIdxDotX = false, sawThreadIdxDotX = false;
    for (size_t i = 0; i + 2 < t.size(); ++i) {
        if (t[i].kind == TokenKind::Identifier && t[i].text == "blockIdx" &&
            t[i + 1].kind == TokenKind::Dot &&
            t[i + 2].kind == TokenKind::Identifier && t[i + 2].text == "x")
            sawBlockIdxDotX = true;
        if (t[i].kind == TokenKind::Identifier && t[i].text == "threadIdx" &&
            t[i + 1].kind == TokenKind::Dot &&
            t[i + 2].kind == TokenKind::Identifier && t[i + 2].text == "x")
            sawThreadIdxDotX = true;
    }
    CHECK(sawBlockIdxDotX, "blockIdx . x tokenized");
    CHECK(sawThreadIdxDotX, "threadIdx . x tokenized");

    // Keywords / operators / punctuation present.
    CHECK(count(t, TokenKind::KwConst) == 2, "two const (a and b params)");
    CHECK(count(t, TokenKind::KwFloat) == 3, "three float types");
    CHECK(count(t, TokenKind::KwInt) == 2, "two int (param n, local i)");
    CHECK(count(t, TokenKind::KwIf) == 1, "one if");
    CHECK(count(t, TokenKind::Star) >= 3, "pointer stars + multiply present");
    CHECK(count(t, TokenKind::Lt) == 1, "one '<' (i < n)");
    CHECK(count(t, TokenKind::LBracket) == 3, "three '[' (c[i], a[i], b[i])");
    CHECK(count(t, TokenKind::Assign) == 2, "two '=' (int i =, c[i] =)");
    CHECK(count(t, TokenKind::Plus) == 2, "two '+' (index add, value add)");

    // Comments (// and /* */) produced no tokens (no Unknown either).
    CHECK(count(t, TokenKind::Unknown) == 0, "no Unknown tokens (all chars recognized)");

    // Number + operator + suffix lexing sanity on a separate snippet.
    std::vector<Token> n = lex("x = 0x1F + 3.5f - 10u * 2e-3;");
    CHECK(n[0].kind == TokenKind::Identifier, "n: identifier");
    CHECK(n[2].kind == TokenKind::IntLiteral && n[2].text == "0x1F", "hex int literal");
    CHECK(n[4].kind == TokenKind::FloatLiteral && n[4].text == "3.5f", "float literal with f suffix");
    CHECK(n[6].kind == TokenKind::IntLiteral && n[6].text == "10u", "int literal with u suffix");
    CHECK(n[8].kind == TokenKind::FloatLiteral && n[8].text == "2e-3", "exponent float literal");

    // Multi-char operators.
    std::vector<Token> ops = lex("a += b; c == d && e << f >= g;");
    bool sawPlusEq = false, sawEq = false, sawAndAnd = false, sawShl = false, sawGe = false;
    for (const auto& tok : ops) {
        if (tok.kind == TokenKind::PlusEq) sawPlusEq = true;
        if (tok.kind == TokenKind::Eq) sawEq = true;
        if (tok.kind == TokenKind::AndAnd) sawAndAnd = true;
        if (tok.kind == TokenKind::Shl) sawShl = true;
        if (tok.kind == TokenKind::Ge) sawGe = true;
    }
    CHECK(sawPlusEq && sawEq && sawAndAnd && sawShl && sawGe, "multi-char operators lexed");

    if (g_fail == 0)
        std::printf("PASS: CUDA-C front-end lexer — all checks green\n");
    else
        std::printf("FAILED: %d check(s)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
