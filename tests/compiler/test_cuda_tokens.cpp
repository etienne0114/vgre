// Track Z / Stage 3.2: token vocabulary completeness. The lexer now records a
// source byte span per token (in addition to line/column), recognizes character
// literals and true/false/nullptr, and draws its keywords from the centralized
// keywordLookup in token.cpp (no duplicated keyword chain). The parser lowers
// char literals to their integer value and true/false/nullptr to 1/0/0. This
// checks the runtime-visible behavior plus the lexer's byte spans and the
// classification helpers. No LLVM.
//
// Tests build in Release (-DNDEBUG); asserts must stay real.
#undef NDEBUG

#include "vgre/compiler/backend/backend_registry.h"
#include "vgre/compiler/backend/execution_backend.h"
#include "vgre/compiler/frontend/codegen.h"
#include "vgre/compiler/frontend/lexer.h"

#include <cstdint>
#include <cstdio>
#include <string>
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

static bool runInterp(const char* src, const char* name, int grid, int block,
                      void* const* args, int numArgs) {
    auto cg = compileToPtx(src, name);
    if (!cg.ok) { std::printf("  codegen %s: %s\n", name, cg.error.c_str()); return false; }
    auto beI = be::makeBackend("interpreter");
    auto k = beI->preparePtx(cg.ptx, name);
    if (!k) { std::printf("  prepare %s:\n%s\n", name, cg.ptx.c_str()); return false; }
    be::LaunchConfig lc; lc.gridDim[0] = (uint32_t)grid; lc.blockDim[0] = (uint32_t)block;
    return beI->launch(*k, lc, args, numArgs);
}

// out[i] packs: bit0=('A'==65), bit1=('\n'==10), bit2=('\x41'==65),
//               bit3=(true?), bit4=(!false), bit5=(nullptr==0 for a null ptr)
static const char* kLits = R"(
extern "C" __global__ void lits(int* out, const int* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int r = 0;
        if ('A' == 65) r |= 1;
        if ('\n' == 10) r |= 2;
        if ('\x41' == 65) r |= 4;
        if (true) r |= 8;
        if (!false) r |= 16;
        const int* p = nullptr;
        if (p == nullptr) r |= 32;
        out[i] = r;
    }
})";

int main() {
    // ── Runtime behavior of the new literal tokens ──────────────────────────────
    {
        const int block = 32, grid = 3, N = block * grid;
        std::vector<int> a(N, 0), out(N, -1);
        void* op = out.data(); void* ap = a.data(); int n = N;
        void* args[] = {&op, &ap, &n};
        CHECK(runInterp(kLits, "lits", grid, block, args, 3), "lits runs");
        bool ok = true;
        for (int i = 0; i < N; ++i) if (out[i] != 0x3f) { ok = false;
            std::printf("  lits i=%d got=0x%x want=0x3f\n", i, out[i]); break; }
        CHECK(ok, "char literals + true/false/nullptr evaluate correctly");
    }

    // ── Lexer byte spans (half-open [begin,end), including delimiters) ───────────
    {
        const std::string src = "int  x = 42;";   // note the double space
        std::vector<Token> t = lex(src);
        // Each non-End token's span must reproduce the exact source slice.
        bool ok = true;
        for (const Token& tk : t) {
            if (tk.kind == TokenKind::End) continue;
            if (tk.beginByte < 0 || tk.endByte > (int)src.size() || tk.beginByte >= tk.endByte) { ok = false; break; }
        }
        CHECK(ok, "byte spans are well-formed");
        // Spot-check specific spans.
        CHECK(t[0].kind == TokenKind::KwInt && t[0].beginByte == 0 && t[0].endByte == 3, "int span [0,3)");
        CHECK(t[1].text == "x" && t[1].beginByte == 5 && t[1].endByte == 6, "x span [5,6) (after 2 spaces)");
        CHECK(t[3].text == "42" && t[3].beginByte == 9 && t[3].endByte == 11, "42 span [9,11)");
    }

    // ── Centralized keyword lookup + classification helpers ─────────────────────
    {
        CHECK(keywordLookup("switch") == TokenKind::KwSwitch, "keywordLookup(switch)");
        CHECK(keywordLookup("__constant__") == TokenKind::KwCudaConstant, "keywordLookup(__constant__)");
        CHECK(keywordLookup("notakeyword") == TokenKind::Identifier, "keywordLookup(non-keyword) -> Identifier");
        CHECK(isKeyword(TokenKind::KwFor) && !isKeyword(TokenKind::Plus), "isKeyword");
        CHECK(isAssignmentOperator(TokenKind::ShlEq) && !isAssignmentOperator(TokenKind::Eq), "isAssignmentOperator");
        CHECK(isLiteral(TokenKind::CharLiteral) && isLiteral(TokenKind::KwTrue), "isLiteral");
        CHECK(isLaunchOperator(TokenKind::LaunchOpen) && !isOperator(TokenKind::LaunchOpen), "launch op is not a value operator");
        CHECK(std::string(tokenSpelling(TokenKind::ShlEq)) == "<<=", "tokenSpelling(<<=)");
        CHECK(tokenSpelling(TokenKind::Identifier) == nullptr, "tokenSpelling(Identifier) is null");
    }

    // ── The new operator tokens lex correctly ───────────────────────────────────
    {
        std::vector<Token> t = lex("a -> b :: c ... k<<<g,b>>>(x)");
        auto has = [&](TokenKind k) { for (auto& tk : t) if (tk.kind == k) return true; return false; };
        CHECK(has(TokenKind::Arrow), "-> lexed");
        CHECK(has(TokenKind::Scope), ":: lexed");
        CHECK(has(TokenKind::Ellipsis), "... lexed");
        CHECK(has(TokenKind::LaunchOpen) && has(TokenKind::LaunchClose), "<<< >>> lexed (not shifts)");
    }

    // ── Completeness: every kind is named; every keyword round-trips; every ──────
    // operator/punctuator has a spelling. The enum is contiguous [End, Spaceship].
    {
        int checked = 0, keywords = 0;
        for (int i = (int)TokenKind::End; i <= (int)TokenKind::Spaceship; ++i) {
            TokenKind k = (TokenKind)i;
            ++checked;
            const char* sp = tokenSpelling(k);
            // Only End/Unknown and the variable-text tokens (Identifier + literals)
            // may lack a fixed spelling; every keyword/operator/punctuator must have one.
            const bool variableText = (k == TokenKind::End || k == TokenKind::Unknown ||
                                       k == TokenKind::Identifier || k == TokenKind::IntLiteral ||
                                       k == TokenKind::FloatLiteral || k == TokenKind::CharLiteral ||
                                       k == TokenKind::StringLiteral);
            if (!variableText && !sp) { std::printf("  kind %d has no spelling\n", i); ++g_fail; }
            if (tokenKindName(k) == nullptr) { std::printf("  kind %d has no name\n", i); ++g_fail; }
            if (isKeyword(k)) {
                ++keywords;
                // Keyword ⇒ has a fixed spelling that maps back to the same kind.
                if (!sp) { std::printf("  keyword %d has no spelling\n", i); ++g_fail; continue; }
                if (keywordLookup(sp) != k) {
                    std::printf("  keyword '%s' does not round-trip (kind %d)\n", sp, i); ++g_fail;
                }
            }
        }
        CHECK(checked > 150, "enum fully walked");
        CHECK(keywords >= 90, "the full C++ + CUDA keyword set is present");
        // Aliased spelling still resolves.
        CHECK(keywordLookup("__restrict") == TokenKind::KwCudaRestrict, "__restrict alias");
        CHECK(keywordLookup("char16_t") == TokenKind::KwChar16, "char16_t present");
        CHECK(keywordLookup("co_await") == TokenKind::KwCoAwait, "C++20 co_await present");
        CHECK(keywordLookup("__cluster_dims__") == TokenKind::KwCudaClusterDims, "CUDA __cluster_dims__ present");
        CHECK(std::string(tokenSpelling(TokenKind::Spaceship)) == "<=>", "spaceship spelled");
        CHECK(std::string(tokenSpelling(TokenKind::ArrowStar)) == "->*", "->* spelled");
    }

    if (g_fail == 0)
        std::printf("PASS: token vocabulary (literals / byte spans / keyword lookup / new operators)\n");
    return g_fail ? 1 : 0;
}
