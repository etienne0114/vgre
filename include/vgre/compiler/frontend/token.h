// Tokens for VGRE's from-scratch CUDA-C front-end (Track Z, Zero-Burden Engine).
// The front-end lexes and parses the CUDA-C kernel subset VGRE supports and
// lowers it to PTX for the interpreter/codegen backends — no Clang/LLVM. See
// docs/zeroBurdenRoadmap.md.
#ifndef VGRE_COMPILER_FRONTEND_TOKEN_H
#define VGRE_COMPILER_FRONTEND_TOKEN_H

#include <string>

namespace vgre {
namespace compiler {
namespace frontend {

enum class TokenKind {
    End,            // end of input
    Identifier,     // foo, threadIdx, blockIdx, …
    IntLiteral,     // 42, 0x1f, 0
    FloatLiteral,   // 1.5, 3.0f, 1e-3
    StringLiteral,  // "C" in extern "C" (text excludes the quotes)

    // Keywords (the supported CUDA-C subset).
    KwGlobal,       // __global__
    KwDevice,       // __device__
    KwShared,       // __shared__
    KwExtern,       // extern
    KwConst,        // const
    KwRestrict,     // __restrict__ / __restrict
    KwVoid, KwBool, KwChar, KwShort, KwInt, KwLong, KwFloat, KwDouble, KwUnsigned, KwSigned,
    KwStruct,
    KwIf, KwElse, KwFor, KwWhile, KwReturn,

    // Punctuation / delimiters.
    LParen, RParen, LBrace, RBrace, LBracket, RBracket,
    Semicolon, Comma, Dot,

    // Operators (assignment, arithmetic, comparison, logical, bitwise).
    Assign,                                   // =
    Plus, Minus, Star, Slash, Percent,        // + - * / %
    PlusEq, MinusEq, StarEq, SlashEq, PercentEq,
    Inc, Dec,                                 // ++ --
    Eq, Ne, Lt, Le, Gt, Ge,                   // == != < <= > >=
    AndAnd, OrOr, Not,                        // && || !
    Amp, Pipe, Caret, Tilde, Shl, Shr,        // & | ^ ~ << >>
    Question, Colon,                          // ?:

    Unknown,        // an unrecognized character (lex error marker)
};

struct Token {
    TokenKind   kind = TokenKind::End;
    std::string text;     // exact source spelling
    int         line = 1; // 1-based
    int         col  = 1; // 1-based
};

// Human-readable name for a token kind (diagnostics/tests).
const char* tokenKindName(TokenKind k);

}  // namespace frontend
}  // namespace compiler
}  // namespace vgre

#endif  // VGRE_COMPILER_FRONTEND_TOKEN_H
