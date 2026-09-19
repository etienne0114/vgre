// Centralized token vocabulary — the single definition of the keyword table, the
// fixed token spellings, human-readable names, and classification predicates.
// The lexer and parser call into these instead of duplicating a keyword chain or
// a spelling switch. See include/vgre/compiler/frontend/token.h.

#include "vgre/compiler/frontend/token.h"

#include <unordered_map>

namespace vgre {
namespace compiler {
namespace frontend {

// The keyword vocabulary, defined once. Spellings that alias the same kind
// (`__restrict__` / `__restrict`) both appear here.
TokenKind keywordLookup(const std::string& id) {
    static const std::unordered_map<std::string, TokenKind> kw = {
        // Function / storage qualifiers.
        {"__global__", TokenKind::KwGlobal}, {"__device__", TokenKind::KwDevice},
        {"__host__", TokenKind::KwHost}, {"__shared__", TokenKind::KwShared},
        {"__constant__", TokenKind::KwConstant}, {"__managed__", TokenKind::KwManaged},
        {"__forceinline__", TokenKind::KwForceInline}, {"__launch_bounds__", TokenKind::KwLaunchBounds},
        {"extern", TokenKind::KwExtern}, {"static", TokenKind::KwStatic},
        {"inline", TokenKind::KwInline}, {"const", TokenKind::KwConst},
        {"volatile", TokenKind::KwVolatile},
        {"__restrict__", TokenKind::KwRestrict}, {"__restrict", TokenKind::KwRestrict},
        {"typedef", TokenKind::KwTypedef},
        // Types.
        {"void", TokenKind::KwVoid}, {"bool", TokenKind::KwBool}, {"char", TokenKind::KwChar},
        {"short", TokenKind::KwShort}, {"int", TokenKind::KwInt}, {"long", TokenKind::KwLong},
        {"float", TokenKind::KwFloat}, {"double", TokenKind::KwDouble},
        {"__half", TokenKind::KwHalf},
        {"unsigned", TokenKind::KwUnsigned}, {"signed", TokenKind::KwSigned},
        {"struct", TokenKind::KwStruct},
        // Statements / expressions.
        {"if", TokenKind::KwIf}, {"else", TokenKind::KwElse}, {"for", TokenKind::KwFor},
        {"while", TokenKind::KwWhile}, {"do", TokenKind::KwDo},
        {"break", TokenKind::KwBreak}, {"continue", TokenKind::KwContinue},
        {"switch", TokenKind::KwSwitch}, {"case", TokenKind::KwCase},
        {"default", TokenKind::KwDefault}, {"return", TokenKind::KwReturn},
        {"sizeof", TokenKind::KwSizeof},
        // Constants.
        {"true", TokenKind::KwTrue}, {"false", TokenKind::KwFalse}, {"nullptr", TokenKind::KwNullptr},
    };
    auto it = kw.find(id);
    return it != kw.end() ? it->second : TokenKind::Identifier;
}

// Fixed spelling for tokens that have one; nullptr for variable-text tokens.
const char* tokenSpelling(TokenKind k) {
    switch (k) {
        // Keywords.
        case TokenKind::KwGlobal:     return "__global__";
        case TokenKind::KwDevice:     return "__device__";
        case TokenKind::KwHost:       return "__host__";
        case TokenKind::KwShared:     return "__shared__";
        case TokenKind::KwConstant:   return "__constant__";
        case TokenKind::KwManaged:    return "__managed__";
        case TokenKind::KwForceInline: return "__forceinline__";
        case TokenKind::KwLaunchBounds: return "__launch_bounds__";
        case TokenKind::KwExtern:     return "extern";
        case TokenKind::KwStatic:     return "static";
        case TokenKind::KwInline:     return "inline";
        case TokenKind::KwConst:      return "const";
        case TokenKind::KwVolatile:   return "volatile";
        case TokenKind::KwRestrict:   return "__restrict__";
        case TokenKind::KwTypedef:    return "typedef";
        case TokenKind::KwVoid:       return "void";
        case TokenKind::KwBool:       return "bool";
        case TokenKind::KwChar:       return "char";
        case TokenKind::KwShort:      return "short";
        case TokenKind::KwInt:        return "int";
        case TokenKind::KwLong:       return "long";
        case TokenKind::KwFloat:      return "float";
        case TokenKind::KwDouble:     return "double";
        case TokenKind::KwHalf:       return "__half";
        case TokenKind::KwUnsigned:   return "unsigned";
        case TokenKind::KwSigned:     return "signed";
        case TokenKind::KwStruct:     return "struct";
        case TokenKind::KwIf:         return "if";
        case TokenKind::KwElse:       return "else";
        case TokenKind::KwFor:        return "for";
        case TokenKind::KwWhile:      return "while";
        case TokenKind::KwDo:         return "do";
        case TokenKind::KwBreak:      return "break";
        case TokenKind::KwContinue:   return "continue";
        case TokenKind::KwSwitch:     return "switch";
        case TokenKind::KwCase:       return "case";
        case TokenKind::KwDefault:    return "default";
        case TokenKind::KwReturn:     return "return";
        case TokenKind::KwSizeof:     return "sizeof";
        case TokenKind::KwTrue:       return "true";
        case TokenKind::KwFalse:      return "false";
        case TokenKind::KwNullptr:    return "nullptr";
        // Punctuation.
        case TokenKind::LParen:       return "(";
        case TokenKind::RParen:       return ")";
        case TokenKind::LBrace:       return "{";
        case TokenKind::RBrace:       return "}";
        case TokenKind::LBracket:     return "[";
        case TokenKind::RBracket:     return "]";
        case TokenKind::Semicolon:    return ";";
        case TokenKind::Comma:        return ",";
        case TokenKind::Dot:          return ".";
        case TokenKind::Arrow:        return "->";
        case TokenKind::ColonColon:   return "::";
        case TokenKind::Ellipsis:     return "...";
        // Operators.
        case TokenKind::Assign:       return "=";
        case TokenKind::Plus:         return "+";
        case TokenKind::Minus:        return "-";
        case TokenKind::Star:         return "*";
        case TokenKind::Slash:        return "/";
        case TokenKind::Percent:      return "%";
        case TokenKind::PlusEq:       return "+=";
        case TokenKind::MinusEq:      return "-=";
        case TokenKind::StarEq:       return "*=";
        case TokenKind::SlashEq:      return "/=";
        case TokenKind::PercentEq:    return "%=";
        case TokenKind::AmpEq:        return "&=";
        case TokenKind::PipeEq:       return "|=";
        case TokenKind::CaretEq:      return "^=";
        case TokenKind::ShlEq:        return "<<=";
        case TokenKind::ShrEq:        return ">>=";
        case TokenKind::Inc:          return "++";
        case TokenKind::Dec:          return "--";
        case TokenKind::Eq:           return "==";
        case TokenKind::Ne:           return "!=";
        case TokenKind::Lt:           return "<";
        case TokenKind::Le:           return "<=";
        case TokenKind::Gt:           return ">";
        case TokenKind::Ge:           return ">=";
        case TokenKind::AndAnd:       return "&&";
        case TokenKind::OrOr:         return "||";
        case TokenKind::Not:          return "!";
        case TokenKind::Amp:          return "&";
        case TokenKind::Pipe:         return "|";
        case TokenKind::Caret:        return "^";
        case TokenKind::Tilde:        return "~";
        case TokenKind::Shl:          return "<<";
        case TokenKind::Shr:          return ">>";
        case TokenKind::Question:     return "?";
        case TokenKind::Colon:        return ":";
        case TokenKind::TripleLt:     return "<<<";
        case TokenKind::TripleGt:     return ">>>";
        // Variable-text tokens have no fixed spelling.
        case TokenKind::End: case TokenKind::Identifier: case TokenKind::IntLiteral:
        case TokenKind::FloatLiteral: case TokenKind::StringLiteral: case TokenKind::CharLiteral:
        case TokenKind::Unknown:
            return nullptr;
    }
    return nullptr;
}

const char* tokenKindName(TokenKind k) {
    switch (k) {
        case TokenKind::End:           return "End";
        case TokenKind::Identifier:    return "Identifier";
        case TokenKind::IntLiteral:    return "IntLiteral";
        case TokenKind::FloatLiteral:  return "FloatLiteral";
        case TokenKind::StringLiteral: return "StringLiteral";
        case TokenKind::CharLiteral:   return "CharLiteral";
        case TokenKind::Unknown:       return "Unknown";
        default: break;
    }
    const char* s = tokenSpelling(k);
    return s ? s : "?";
}

bool isKeyword(TokenKind k) {
    return k >= TokenKind::KwGlobal && k <= TokenKind::KwNullptr;
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
            return true;
        default: return false;
    }
}

bool isLaunchOperator(TokenKind k) {
    return k == TokenKind::TripleLt || k == TokenKind::TripleGt;
}

bool isOperator(TokenKind k) {
    // The contiguous operator block runs from Assign through Shr; launch operators
    // (<<< >>>) are launch punctuation, not value operators, and are excluded.
    return (k >= TokenKind::Assign && k <= TokenKind::Colon);
}

}  // namespace frontend
}  // namespace compiler
}  // namespace vgre
