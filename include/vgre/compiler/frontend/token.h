// Tokens for VGRE's from-scratch CUDA-C front-end (Track Z, Zero-Burden Engine).
// The front-end lexes and parses the CUDA-C kernel subset VGRE supports and
// lowers it to PTX for the interpreter/codegen backends — no Clang/LLVM. See
// docs/zeroBurdenRoadmap.md.
//
// This header is the single source of truth for the token vocabulary: the kind
// enum, the Token record (with line/column AND a source byte span), and the
// centralized keyword lookup / fixed spelling / classification helpers. The lexer
// and parser use these rather than duplicating keyword tables or spelling chains.
#ifndef VGRE_COMPILER_FRONTEND_TOKEN_H
#define VGRE_COMPILER_FRONTEND_TOKEN_H

#include <string>

namespace vgre {
namespace compiler {
namespace frontend {

enum class TokenKind {
    End,            // end of input
    Identifier,     // foo, threadIdx, blockIdx, …

    // Literals.
    IntLiteral,     // 42, 0x1f, 0
    FloatLiteral,   // 1.5, 3.0f, 1e-3
    StringLiteral,  // "C" in extern "C" (text excludes the quotes)
    CharLiteral,    // 'a', '\n', '\x41' (text excludes the quotes)

    // ── Keywords (the supported CUDA-C subset) ──────────────────────────────────
    // Function / storage qualifiers.
    KwGlobal,       // __global__
    KwDevice,       // __device__
    KwHost,         // __host__
    KwShared,       // __shared__
    KwConstant,     // __constant__
    KwManaged,      // __managed__
    KwForceInline,  // __forceinline__
    KwLaunchBounds, // __launch_bounds__
    KwExtern,       // extern
    KwStatic,       // static
    KwInline,       // inline
    KwConst,        // const
    KwVolatile,     // volatile
    KwRestrict,     // __restrict__ / __restrict
    KwTypedef,      // typedef

    // Type keywords.
    KwVoid, KwBool, KwChar, KwShort, KwInt, KwLong, KwFloat, KwDouble, KwHalf, KwUnsigned, KwSigned,
    KwStruct,

    // Statement / expression keywords.
    KwIf, KwElse, KwFor, KwWhile, KwDo, KwBreak, KwContinue, KwReturn,
    KwSwitch, KwCase, KwDefault,
    KwSizeof,

    // Constant keywords.
    KwTrue, KwFalse, KwNullptr,

    // ── Punctuation / delimiters ────────────────────────────────────────────────
    LParen, RParen, LBrace, RBrace, LBracket, RBracket,
    Semicolon, Comma, Dot,
    Arrow,          // ->
    ColonColon,     // ::
    Ellipsis,       // ...

    // ── Operators (assignment, arithmetic, comparison, logical, bitwise) ────────
    Assign,                                   // =
    Plus, Minus, Star, Slash, Percent,        // + - * / %
    PlusEq, MinusEq, StarEq, SlashEq, PercentEq,
    AmpEq, PipeEq, CaretEq, ShlEq, ShrEq,     // &= |= ^= <<= >>=
    Inc, Dec,                                 // ++ --
    Eq, Ne, Lt, Le, Gt, Ge,                   // == != < <= > >=
    AndAnd, OrOr, Not,                        // && || !
    Amp, Pipe, Caret, Tilde, Shl, Shr,        // & | ^ ~ << >>
    Question, Colon,                          // ?:

    // CUDA kernel-launch operators — distinct from the shift operators, so a
    // launch `k<<<grid, block>>>(...)` is not confused with `a << b`.
    TripleLt,       // <<<
    TripleGt,       // >>>

    Unknown,        // an unrecognized character (lex error marker)
};

// A lexed token. Carries its exact source spelling, a 1-based line/column for
// human-facing diagnostics, and a half-open byte span [beginByte, endByte) into
// the original source for precise slicing (span includes any delimiters that the
// spelling omits, e.g. the quotes of a string/char literal).
struct Token {
    TokenKind   kind = TokenKind::End;
    std::string text;         // exact source spelling (literals: without delimiters)
    int         line = 1;     // 1-based
    int         col  = 1;     // 1-based
    int         beginByte = 0;  // byte offset of the first character
    int         endByte   = 0;  // byte offset just past the last character
};

// Human-readable NAME for a token kind (diagnostics/tests): a keyword/operator
// returns its spelling ("int", "<<"), a variable token returns a category name
// ("Identifier", "IntLiteral").
const char* tokenKindName(TokenKind k);

// Fixed source SPELLING for tokens that have one (keywords, operators,
// punctuation) — e.g. "<<=", "return". Returns nullptr for tokens whose text
// varies (Identifier, the literals, End, Unknown); use Token::text for those.
const char* tokenSpelling(TokenKind k);

// Centralized keyword lookup: the keyword kind for `id`, or TokenKind::Identifier
// if it is not a keyword. The one place the keyword vocabulary is defined.
TokenKind keywordLookup(const std::string& id);

// Classification helpers.
bool isKeyword(TokenKind k);
bool isLiteral(TokenKind k);              // Int/Float/String/Char + true/false/nullptr
bool isOperator(TokenKind k);             // any arithmetic/logical/bitwise/compare/assign operator
bool isAssignmentOperator(TokenKind k);   // = += -= *= /= %= &= |= ^= <<= >>=
bool isLaunchOperator(TokenKind k);       // <<< or >>>

}  // namespace frontend
}  // namespace compiler
}  // namespace vgre

#endif  // VGRE_COMPILER_FRONTEND_TOKEN_H
