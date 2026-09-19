// Centralized token vocabulary — the single definition of the keyword table, the
// fixed token spellings, human-readable names, and classification predicates.
// The lexer and parser call into these instead of duplicating a keyword chain or
// a spelling switch. See include/vgre/compiler/frontend/token.h.

#include "vgre/compiler/frontend/token.h"

#include <unordered_map>

namespace vgre {
namespace compiler {
namespace frontend {

// The keyword vocabulary, defined once: the full C++ reserved set (including the
// alternative operator spellings and the C++20/26 additions) plus CUDA's
// qualifiers/annotations. Spellings that alias the same kind (`__restrict__` /
// `__restrict`) both appear here.
TokenKind keywordLookup(const std::string& id) {
    static const std::unordered_map<std::string, TokenKind> kw = {
        // ── C++ standard keywords ──
        {"alignas", TokenKind::KwAlignas}, {"alignof", TokenKind::KwAlignof},
        {"asm", TokenKind::KwAsm}, {"auto", TokenKind::KwAuto},
        {"bool", TokenKind::KwBool}, {"break", TokenKind::KwBreak},
        {"case", TokenKind::KwCase}, {"catch", TokenKind::KwCatch},
        {"char", TokenKind::KwChar}, {"char8_t", TokenKind::KwChar8},
        {"char16_t", TokenKind::KwChar16}, {"char32_t", TokenKind::KwChar32},
        {"class", TokenKind::KwClass}, {"const", TokenKind::KwConst},
        {"const_cast", TokenKind::KwConstCast}, {"continue", TokenKind::KwContinue},
        {"decltype", TokenKind::KwDecltype}, {"default", TokenKind::KwDefault},
        {"delete", TokenKind::KwDelete}, {"do", TokenKind::KwDo},
        {"double", TokenKind::KwDouble}, {"dynamic_cast", TokenKind::KwDynamicCast},
        {"else", TokenKind::KwElse}, {"enum", TokenKind::KwEnum},
        {"explicit", TokenKind::KwExplicit}, {"export", TokenKind::KwExport},
        {"extern", TokenKind::KwExtern}, {"false", TokenKind::KwFalse},
        {"float", TokenKind::KwFloat}, {"for", TokenKind::KwFor},
        {"friend", TokenKind::KwFriend}, {"goto", TokenKind::KwGoto},
        {"if", TokenKind::KwIf}, {"inline", TokenKind::KwInline},
        {"int", TokenKind::KwInt}, {"long", TokenKind::KwLong},
        {"mutable", TokenKind::KwMutable}, {"namespace", TokenKind::KwNamespace},
        {"new", TokenKind::KwNew}, {"noexcept", TokenKind::KwNoexcept},
        {"nullptr", TokenKind::KwNullptr}, {"operator", TokenKind::KwOperator},
        {"private", TokenKind::KwPrivate}, {"protected", TokenKind::KwProtected},
        {"public", TokenKind::KwPublic}, {"register", TokenKind::KwRegister},
        {"reinterpret_cast", TokenKind::KwReinterpretCast}, {"return", TokenKind::KwReturn},
        {"short", TokenKind::KwShort}, {"signed", TokenKind::KwSigned},
        {"sizeof", TokenKind::KwSizeof}, {"static", TokenKind::KwStatic},
        {"static_assert", TokenKind::KwStaticAssert}, {"static_cast", TokenKind::KwStaticCast},
        {"struct", TokenKind::KwStruct}, {"switch", TokenKind::KwSwitch},
        {"template", TokenKind::KwTemplate}, {"this", TokenKind::KwThis},
        {"thread_local", TokenKind::KwThreadLocal}, {"throw", TokenKind::KwThrow},
        {"true", TokenKind::KwTrue}, {"try", TokenKind::KwTry},
        {"typedef", TokenKind::KwTypedef}, {"typeid", TokenKind::KwTypeid},
        {"typename", TokenKind::KwTypename}, {"union", TokenKind::KwUnion},
        {"unsigned", TokenKind::KwUnsigned}, {"using", TokenKind::KwUsing},
        {"virtual", TokenKind::KwVirtual}, {"void", TokenKind::KwVoid},
        {"volatile", TokenKind::KwVolatile}, {"wchar_t", TokenKind::KwWchar},
        {"while", TokenKind::KwWhile},
        // ── C++ alternative operator representations ──
        {"and", TokenKind::KwAnd}, {"and_eq", TokenKind::KwAndEq},
        {"bitand", TokenKind::KwBitand}, {"bitor", TokenKind::KwBitor},
        {"compl", TokenKind::KwCompl}, {"not", TokenKind::KwNot},
        {"not_eq", TokenKind::KwNotEq}, {"or", TokenKind::KwOr},
        {"or_eq", TokenKind::KwOrEq}, {"xor", TokenKind::KwXor},
        {"xor_eq", TokenKind::KwXorEq},
        // ── C++20 ──
        {"concept", TokenKind::KwConcept}, {"consteval", TokenKind::KwConsteval},
        {"constexpr", TokenKind::KwConstexpr}, {"constinit", TokenKind::KwConstinit},
        {"co_await", TokenKind::KwCoAwait}, {"co_return", TokenKind::KwCoReturn},
        {"co_yield", TokenKind::KwCoYield}, {"requires", TokenKind::KwRequires},
        // ── C++26 ──
        {"contract_assert", TokenKind::KwContractAssert},
        // ── CUDA execution-space specifiers ──
        {"__host__", TokenKind::KwCudaHost}, {"__device__", TokenKind::KwCudaDevice},
        {"__global__", TokenKind::KwCudaGlobal}, {"__tile__", TokenKind::KwCudaTile},
        {"__tile_global__", TokenKind::KwCudaTileGlobal},
        // ── CUDA memory-space specifiers ──
        {"__shared__", TokenKind::KwCudaShared}, {"__constant__", TokenKind::KwCudaConstant},
        {"__managed__", TokenKind::KwCudaManaged},
        // ── CUDA type specifiers ──
        {"__half", TokenKind::KwCudaHalf},
        // ── CUDA function optimization / aliasing ──
        {"__noinline__", TokenKind::KwCudaNoinline}, {"__forceinline__", TokenKind::KwCudaForceinline},
        {"__inline_hint__", TokenKind::KwCudaInlineHint},
        {"__restrict__", TokenKind::KwCudaRestrict}, {"__restrict", TokenKind::KwCudaRestrict},
        // ── CUDA kernel/function annotations ──
        {"__grid_constant__", TokenKind::KwCudaGridConstant},
        {"__launch_bounds__", TokenKind::KwCudaLaunchBounds},
        {"__maxnreg__", TokenKind::KwCudaMaxNReg}, {"__cluster_dims__", TokenKind::KwCudaClusterDims},
    };
    auto it = kw.find(id);
    return it != kw.end() ? it->second : TokenKind::Identifier;
}

// Fixed spelling for tokens that have one; nullptr for variable-text tokens.
const char* tokenSpelling(TokenKind k) {
    switch (k) {
        // ── C++ keywords ──
        case TokenKind::KwAlignas: return "alignas";
        case TokenKind::KwAlignof: return "alignof";
        case TokenKind::KwAsm: return "asm";
        case TokenKind::KwAuto: return "auto";
        case TokenKind::KwBool: return "bool";
        case TokenKind::KwBreak: return "break";
        case TokenKind::KwCase: return "case";
        case TokenKind::KwCatch: return "catch";
        case TokenKind::KwChar: return "char";
        case TokenKind::KwChar8: return "char8_t";
        case TokenKind::KwChar16: return "char16_t";
        case TokenKind::KwChar32: return "char32_t";
        case TokenKind::KwClass: return "class";
        case TokenKind::KwConst: return "const";
        case TokenKind::KwConstCast: return "const_cast";
        case TokenKind::KwContinue: return "continue";
        case TokenKind::KwDecltype: return "decltype";
        case TokenKind::KwDefault: return "default";
        case TokenKind::KwDelete: return "delete";
        case TokenKind::KwDo: return "do";
        case TokenKind::KwDouble: return "double";
        case TokenKind::KwDynamicCast: return "dynamic_cast";
        case TokenKind::KwElse: return "else";
        case TokenKind::KwEnum: return "enum";
        case TokenKind::KwExplicit: return "explicit";
        case TokenKind::KwExport: return "export";
        case TokenKind::KwExtern: return "extern";
        case TokenKind::KwFalse: return "false";
        case TokenKind::KwFloat: return "float";
        case TokenKind::KwFor: return "for";
        case TokenKind::KwFriend: return "friend";
        case TokenKind::KwGoto: return "goto";
        case TokenKind::KwIf: return "if";
        case TokenKind::KwInline: return "inline";
        case TokenKind::KwInt: return "int";
        case TokenKind::KwLong: return "long";
        case TokenKind::KwMutable: return "mutable";
        case TokenKind::KwNamespace: return "namespace";
        case TokenKind::KwNew: return "new";
        case TokenKind::KwNoexcept: return "noexcept";
        case TokenKind::KwNullptr: return "nullptr";
        case TokenKind::KwOperator: return "operator";
        case TokenKind::KwPrivate: return "private";
        case TokenKind::KwProtected: return "protected";
        case TokenKind::KwPublic: return "public";
        case TokenKind::KwRegister: return "register";
        case TokenKind::KwReinterpretCast: return "reinterpret_cast";
        case TokenKind::KwReturn: return "return";
        case TokenKind::KwShort: return "short";
        case TokenKind::KwSigned: return "signed";
        case TokenKind::KwSizeof: return "sizeof";
        case TokenKind::KwStatic: return "static";
        case TokenKind::KwStaticAssert: return "static_assert";
        case TokenKind::KwStaticCast: return "static_cast";
        case TokenKind::KwStruct: return "struct";
        case TokenKind::KwSwitch: return "switch";
        case TokenKind::KwTemplate: return "template";
        case TokenKind::KwThis: return "this";
        case TokenKind::KwThreadLocal: return "thread_local";
        case TokenKind::KwThrow: return "throw";
        case TokenKind::KwTrue: return "true";
        case TokenKind::KwTry: return "try";
        case TokenKind::KwTypedef: return "typedef";
        case TokenKind::KwTypeid: return "typeid";
        case TokenKind::KwTypename: return "typename";
        case TokenKind::KwUnion: return "union";
        case TokenKind::KwUnsigned: return "unsigned";
        case TokenKind::KwUsing: return "using";
        case TokenKind::KwVirtual: return "virtual";
        case TokenKind::KwVoid: return "void";
        case TokenKind::KwVolatile: return "volatile";
        case TokenKind::KwWchar: return "wchar_t";
        case TokenKind::KwWhile: return "while";
        // ── Alternative operator spellings ──
        case TokenKind::KwAnd: return "and";
        case TokenKind::KwAndEq: return "and_eq";
        case TokenKind::KwBitand: return "bitand";
        case TokenKind::KwBitor: return "bitor";
        case TokenKind::KwCompl: return "compl";
        case TokenKind::KwNot: return "not";
        case TokenKind::KwNotEq: return "not_eq";
        case TokenKind::KwOr: return "or";
        case TokenKind::KwOrEq: return "or_eq";
        case TokenKind::KwXor: return "xor";
        case TokenKind::KwXorEq: return "xor_eq";
        // ── C++20 / C++26 ──
        case TokenKind::KwConcept: return "concept";
        case TokenKind::KwConsteval: return "consteval";
        case TokenKind::KwConstexpr: return "constexpr";
        case TokenKind::KwConstinit: return "constinit";
        case TokenKind::KwCoAwait: return "co_await";
        case TokenKind::KwCoReturn: return "co_return";
        case TokenKind::KwCoYield: return "co_yield";
        case TokenKind::KwRequires: return "requires";
        case TokenKind::KwContractAssert: return "contract_assert";
        // ── CUDA keywords ──
        case TokenKind::KwCudaHost: return "__host__";
        case TokenKind::KwCudaDevice: return "__device__";
        case TokenKind::KwCudaGlobal: return "__global__";
        case TokenKind::KwCudaTile: return "__tile__";
        case TokenKind::KwCudaTileGlobal: return "__tile_global__";
        case TokenKind::KwCudaShared: return "__shared__";
        case TokenKind::KwCudaConstant: return "__constant__";
        case TokenKind::KwCudaManaged: return "__managed__";
        case TokenKind::KwCudaHalf: return "__half";
        case TokenKind::KwCudaNoinline: return "__noinline__";
        case TokenKind::KwCudaForceinline: return "__forceinline__";
        case TokenKind::KwCudaInlineHint: return "__inline_hint__";
        case TokenKind::KwCudaRestrict: return "__restrict__";
        case TokenKind::KwCudaGridConstant: return "__grid_constant__";
        case TokenKind::KwCudaLaunchBounds: return "__launch_bounds__";
        case TokenKind::KwCudaMaxNReg: return "__maxnreg__";
        case TokenKind::KwCudaClusterDims: return "__cluster_dims__";
        // ── Punctuation ──
        case TokenKind::LParen: return "(";
        case TokenKind::RParen: return ")";
        case TokenKind::LBrace: return "{";
        case TokenKind::RBrace: return "}";
        case TokenKind::LBracket: return "[";
        case TokenKind::RBracket: return "]";
        case TokenKind::Semicolon: return ";";
        case TokenKind::Comma: return ",";
        case TokenKind::Dot: return ".";
        case TokenKind::Ellipsis: return "...";
        case TokenKind::Arrow: return "->";
        case TokenKind::Scope: return "::";
        case TokenKind::DotStar: return ".*";
        case TokenKind::ArrowStar: return "->*";
        case TokenKind::LaunchOpen: return "<<<";
        case TokenKind::LaunchClose: return ">>>";
        // ── Operators ──
        case TokenKind::Assign: return "=";
        case TokenKind::Plus: return "+";
        case TokenKind::Minus: return "-";
        case TokenKind::Star: return "*";
        case TokenKind::Slash: return "/";
        case TokenKind::Percent: return "%";
        case TokenKind::PlusEq: return "+=";
        case TokenKind::MinusEq: return "-=";
        case TokenKind::StarEq: return "*=";
        case TokenKind::SlashEq: return "/=";
        case TokenKind::PercentEq: return "%=";
        case TokenKind::AmpEq: return "&=";
        case TokenKind::PipeEq: return "|=";
        case TokenKind::CaretEq: return "^=";
        case TokenKind::ShlEq: return "<<=";
        case TokenKind::ShrEq: return ">>=";
        case TokenKind::Inc: return "++";
        case TokenKind::Dec: return "--";
        case TokenKind::Eq: return "==";
        case TokenKind::Ne: return "!=";
        case TokenKind::Lt: return "<";
        case TokenKind::Le: return "<=";
        case TokenKind::Gt: return ">";
        case TokenKind::Ge: return ">=";
        case TokenKind::AndAnd: return "&&";
        case TokenKind::OrOr: return "||";
        case TokenKind::Not: return "!";
        case TokenKind::Amp: return "&";
        case TokenKind::Pipe: return "|";
        case TokenKind::Caret: return "^";
        case TokenKind::Tilde: return "~";
        case TokenKind::Shl: return "<<";
        case TokenKind::Shr: return ">>";
        case TokenKind::Question: return "?";
        case TokenKind::Colon: return ":";
        case TokenKind::Spaceship: return "<=>";
        // ── Variable-text tokens have no fixed spelling ──
        case TokenKind::End: case TokenKind::Unknown: case TokenKind::Identifier:
        case TokenKind::IntLiteral: case TokenKind::FloatLiteral:
        case TokenKind::CharLiteral: case TokenKind::StringLiteral:
            return nullptr;
    }
    return nullptr;
}

const char* tokenKindName(TokenKind k) {
    switch (k) {
        case TokenKind::End:           return "End";
        case TokenKind::Unknown:       return "Unknown";
        case TokenKind::Identifier:    return "Identifier";
        case TokenKind::IntLiteral:    return "IntLiteral";
        case TokenKind::FloatLiteral:  return "FloatLiteral";
        case TokenKind::CharLiteral:   return "CharLiteral";
        case TokenKind::StringLiteral: return "StringLiteral";
        default: break;
    }
    const char* s = tokenSpelling(k);
    return s ? s : "?";
}

bool isKeyword(TokenKind k) {
    // The whole keyword block is contiguous: [KwAlignas, KwCudaClusterDims].
    return k >= TokenKind::KwAlignas && k <= TokenKind::KwCudaClusterDims;
}

bool isCudaKeyword(TokenKind k) {
    return k >= TokenKind::KwCudaHost && k <= TokenKind::KwCudaClusterDims;
}

bool isLiteral(TokenKind k) {
    return k == TokenKind::IntLiteral || k == TokenKind::FloatLiteral ||
           k == TokenKind::StringLiteral || k == TokenKind::CharLiteral ||
           k == TokenKind::KwTrue || k == TokenKind::KwFalse || k == TokenKind::KwNullptr;
}

bool isAssignmentOperator(TokenKind k) {
    switch (k) {
        case TokenKind::Assign: case TokenKind::PlusEq: case TokenKind::MinusEq:
        case TokenKind::StarEq: case TokenKind::SlashEq: case TokenKind::PercentEq:
        case TokenKind::AmpEq: case TokenKind::PipeEq: case TokenKind::CaretEq:
        case TokenKind::ShlEq: case TokenKind::ShrEq:
        case TokenKind::KwAndEq: case TokenKind::KwOrEq: case TokenKind::KwXorEq:
            return true;
        default: return false;
    }
}

bool isLaunchOperator(TokenKind k) {
    return k == TokenKind::LaunchOpen || k == TokenKind::LaunchClose;
}

bool isOperator(TokenKind k) {
    // The symbolic operator block is contiguous: [Assign, Spaceship]. The
    // alternative-spelling operator keywords (and, or, …) are also operators.
    if (k >= TokenKind::Assign && k <= TokenKind::Spaceship) return true;
    switch (k) {
        case TokenKind::KwAnd: case TokenKind::KwAndEq: case TokenKind::KwBitand:
        case TokenKind::KwBitor: case TokenKind::KwCompl: case TokenKind::KwNot:
        case TokenKind::KwNotEq: case TokenKind::KwOr: case TokenKind::KwOrEq:
        case TokenKind::KwXor: case TokenKind::KwXorEq:
            return true;
        default: return false;
    }
}

}  // namespace frontend
}  // namespace compiler
}  // namespace vgre
