// Tokens for VGRE's from-scratch CUDA-C front-end (Track Z, Zero-Burden Engine).
// The front-end lexes and parses the CUDA-C kernel subset VGRE supports and
// lowers it to PTX for the interpreter/codegen backends — no Clang/LLVM. See
// docs/zeroBurdenRoadmap.md.
//
// This header is the single source of truth for the token vocabulary: the kind
// enum (the full C++ reserved-keyword set plus CUDA qualifiers/annotations and
// every operator/punctuator), the Token record (line/column AND a source byte
// span), and the centralized keyword lookup / fixed spelling / classification
// helpers. The lexer and parser use these rather than duplicating tables. Not
// every kind is yet handled by the parser's grammar — kinds outside the supported
// subset lex cleanly and surface a *located* error if used, never wrong code.
#ifndef VGRE_COMPILER_FRONTEND_TOKEN_H
#define VGRE_COMPILER_FRONTEND_TOKEN_H

#include <string>

namespace vgre {
namespace compiler {
namespace frontend {

enum class TokenKind {
    // ── Special ─────────────────────────────────────────────────────────────────
    End,
    Unknown,

    // ── Identifiers / literals ──────────────────────────────────────────────────
    Identifier,
    IntLiteral,
    FloatLiteral,
    CharLiteral,
    StringLiteral,

    // ── C++ standard keywords (complete reserved set) ───────────────────────────
    KwAlignas,
    KwAlignof,
    KwAsm,
    KwAuto,
    KwBool,
    KwBreak,
    KwCase,
    KwCatch,
    KwChar,
    KwChar8,
    KwChar16,
    KwChar32,
    KwClass,
    KwConst,
    KwConstCast,
    KwContinue,
    KwDecltype,
    KwDefault,
    KwDelete,
    KwDo,
    KwDouble,
    KwDynamicCast,
    KwElse,
    KwEnum,
    KwExplicit,
    KwExport,
    KwExtern,
    KwFalse,
    KwFloat,
    KwFor,
    KwFriend,
    KwGoto,
    KwIf,
    KwInline,
    KwInt,
    KwLong,
    KwMutable,
    KwNamespace,
    KwNew,
    KwNoexcept,
    KwNullptr,
    KwOperator,
    KwPrivate,
    KwProtected,
    KwPublic,
    KwRegister,
    KwReinterpretCast,
    KwReturn,
    KwShort,
    KwSigned,
    KwSizeof,
    KwStatic,
    KwStaticAssert,
    KwStaticCast,
    KwStruct,
    KwSwitch,
    KwTemplate,
    KwThis,
    KwThreadLocal,
    KwThrow,
    KwTrue,
    KwTry,
    KwTypedef,
    KwTypeid,
    KwTypename,
    KwUnion,
    KwUnsigned,
    KwUsing,
    KwVirtual,
    KwVoid,
    KwVolatile,
    KwWchar,
    KwWhile,

    // C++ alternative operator representations.
    KwAnd,
    KwAndEq,
    KwBitand,
    KwBitor,
    KwCompl,
    KwNot,
    KwNotEq,
    KwOr,
    KwOrEq,
    KwXor,
    KwXorEq,

    // C++20.
    KwConcept,
    KwConsteval,
    KwConstexpr,
    KwConstinit,
    KwCoAwait,
    KwCoReturn,
    KwCoYield,
    KwRequires,

    // C++26.
    KwContractAssert,

    // ── CUDA-specific keywords / qualifiers / annotations ───────────────────────
    // Execution-space specifiers.
    KwCudaHost,          // __host__
    KwCudaDevice,        // __device__
    KwCudaGlobal,        // __global__
    KwCudaTile,          // __tile__
    KwCudaTileGlobal,    // __tile_global__

    // Memory-space specifiers.
    KwCudaShared,        // __shared__
    KwCudaConstant,      // __constant__
    KwCudaManaged,       // __managed__

    // CUDA type specifiers.
    KwCudaHalf,          // __half (VGRE fp16 storage type)

    // Function optimization / aliasing.
    KwCudaNoinline,      // __noinline__
    KwCudaForceinline,   // __forceinline__
    KwCudaInlineHint,    // __inline_hint__
    KwCudaRestrict,      // __restrict__ / __restrict

    // Kernel / function annotations.
    KwCudaGridConstant,  // __grid_constant__
    KwCudaLaunchBounds,  // __launch_bounds__
    KwCudaMaxNReg,       // __maxnreg__
    KwCudaClusterDims,   // __cluster_dims__

    // ── Punctuation ─────────────────────────────────────────────────────────────
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,

    Semicolon,
    Comma,
    Dot,

    Ellipsis,          // ...

    Arrow,             // ->
    Scope,             // ::

    DotStar,           // .*
    ArrowStar,         // ->*

    // ── Kernel launch ───────────────────────────────────────────────────────────
    LaunchOpen,        // <<<
    LaunchClose,       // >>>

    // ── Operators ───────────────────────────────────────────────────────────────
    Assign,            // =

    Plus,
    Minus,
    Star,
    Slash,
    Percent,

    PlusEq,
    MinusEq,
    StarEq,
    SlashEq,
    PercentEq,

    AmpEq,
    PipeEq,
    CaretEq,
    ShlEq,
    ShrEq,

    Inc,
    Dec,

    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,

    AndAnd,
    OrOr,
    Not,

    Amp,
    Pipe,
    Caret,
    Tilde,
    Shl,
    Shr,

    Question,
    Colon,

    Spaceship,         // <=>
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
bool isCudaKeyword(TokenKind k);          // the __host__/__device__/… family
bool isLiteral(TokenKind k);              // Int/Float/String/Char + true/false/nullptr
bool isOperator(TokenKind k);             // any arithmetic/logical/bitwise/compare/assign operator
bool isAssignmentOperator(TokenKind k);   // = += -= *= /= %= &= |= ^= <<= >>= (and and_eq/or_eq/xor_eq)
bool isLaunchOperator(TokenKind k);       // <<< or >>>

}  // namespace frontend
}  // namespace compiler
}  // namespace vgre

#endif  // VGRE_COMPILER_FRONTEND_TOKEN_H
