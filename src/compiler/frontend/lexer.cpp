// Hand-written CUDA-C-subset lexer — see include/vgre/compiler/frontend/lexer.h.

#include "vgre/compiler/frontend/lexer.h"

#include <cctype>
#include <unordered_map>

namespace vgre {
namespace compiler {
namespace frontend {

const char* tokenKindName(TokenKind k) {
    switch (k) {
        case TokenKind::End:          return "End";
        case TokenKind::Identifier:   return "Identifier";
        case TokenKind::IntLiteral:   return "IntLiteral";
        case TokenKind::FloatLiteral: return "FloatLiteral";
        case TokenKind::StringLiteral: return "StringLiteral";
        case TokenKind::KwGlobal:     return "__global__";
        case TokenKind::KwDevice:     return "__device__";
        case TokenKind::KwShared:     return "__shared__";
        case TokenKind::KwExtern:     return "extern";
        case TokenKind::KwConst:      return "const";
        case TokenKind::KwRestrict:   return "__restrict__";
        case TokenKind::KwVoid:       return "void";
        case TokenKind::KwBool:       return "bool";
        case TokenKind::KwChar:       return "char";
        case TokenKind::KwShort:      return "short";
        case TokenKind::KwInt:        return "int";
        case TokenKind::KwLong:       return "long";
        case TokenKind::KwFloat:      return "float";
        case TokenKind::KwDouble:     return "double";
        case TokenKind::KwUnsigned:   return "unsigned";
        case TokenKind::KwSigned:     return "signed";
        case TokenKind::KwStruct:     return "struct";
        case TokenKind::KwIf:         return "if";
        case TokenKind::KwElse:       return "else";
        case TokenKind::KwFor:        return "for";
        case TokenKind::KwWhile:      return "while";
        case TokenKind::KwReturn:     return "return";
        case TokenKind::LParen:       return "(";
        case TokenKind::RParen:       return ")";
        case TokenKind::LBrace:       return "{";
        case TokenKind::RBrace:       return "}";
        case TokenKind::LBracket:     return "[";
        case TokenKind::RBracket:     return "]";
        case TokenKind::Semicolon:    return ";";
        case TokenKind::Comma:        return ",";
        case TokenKind::Dot:          return ".";
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
        case TokenKind::Unknown:      return "Unknown";
    }
    return "?";
}

namespace {

const std::unordered_map<std::string, TokenKind>& keywords() {
    static const std::unordered_map<std::string, TokenKind> kw = {
        {"__global__", TokenKind::KwGlobal}, {"__device__", TokenKind::KwDevice},
        {"__shared__", TokenKind::KwShared}, {"extern", TokenKind::KwExtern},
        {"const", TokenKind::KwConst},
        {"__restrict__", TokenKind::KwRestrict}, {"__restrict", TokenKind::KwRestrict},
        {"void", TokenKind::KwVoid}, {"bool", TokenKind::KwBool}, {"char", TokenKind::KwChar},
        {"short", TokenKind::KwShort}, {"int", TokenKind::KwInt}, {"long", TokenKind::KwLong},
        {"float", TokenKind::KwFloat}, {"double", TokenKind::KwDouble},
        {"unsigned", TokenKind::KwUnsigned}, {"signed", TokenKind::KwSigned},
        {"struct", TokenKind::KwStruct},
        {"if", TokenKind::KwIf}, {"else", TokenKind::KwElse}, {"for", TokenKind::KwFor},
        {"while", TokenKind::KwWhile}, {"return", TokenKind::KwReturn},
    };
    return kw;
}

bool isIdentStart(char c) { return std::isalpha((unsigned char)c) || c == '_'; }
bool isIdentCont(char c)  { return std::isalnum((unsigned char)c) || c == '_'; }

struct Lexer {
    const char* p;
    const char* end;
    int line = 1, col = 1;
    std::vector<Token> out;

    explicit Lexer(const std::string& s) : p(s.data()), end(s.data() + s.size()) {}

    void advance(int n = 1) {
        for (int i = 0; i < n && p < end; ++i) {
            if (*p == '\n') { ++line; col = 1; } else { ++col; }
            ++p;
        }
    }

    void push(TokenKind k, const std::string& text, int l, int c) {
        out.push_back(Token{k, text, l, c});
    }

    // Two-character operator if p[0]==a && p[1]==b; else fall back to `one`.
    bool two(char a, char b, TokenKind twoKind, const char* twoStr, int l, int c) {
        if (*p == a && p + 1 < end && p[1] == b) {
            push(twoKind, twoStr, l, c);
            advance(2);
            return true;
        }
        return false;
    }

    void skipTrivia() {
        for (;;) {
            if (p >= end) return;
            char c = *p;
            if (c == ' ' || c == '\t' || c == '\r' || c == '\n') { advance(); continue; }
            if (c == '/' && p + 1 < end && p[1] == '/') {          // line comment
                while (p < end && *p != '\n') advance();
                continue;
            }
            if (c == '/' && p + 1 < end && p[1] == '*') {          // block comment
                advance(2);
                while (p < end && !(*p == '*' && p + 1 < end && p[1] == '/')) advance();
                if (p < end) advance(2);
                continue;
            }
            if (c == '#' && (col == 1 || out.empty())) {           // preprocessor line
                while (p < end && *p != '\n') advance();
                continue;
            }
            return;
        }
    }

    void lexNumber() {
        const int l = line, c = col;
        const char* start = p;
        bool isFloat = false;
        if (*p == '0' && p + 1 < end && (p[1] == 'x' || p[1] == 'X')) {
            advance(2);
            while (p < end && std::isxdigit((unsigned char)*p)) advance();
        } else {
            while (p < end && std::isdigit((unsigned char)*p)) advance();
            if (p < end && *p == '.') { isFloat = true; advance();
                while (p < end && std::isdigit((unsigned char)*p)) advance(); }
            if (p < end && (*p == 'e' || *p == 'E')) {
                isFloat = true; advance();
                if (p < end && (*p == '+' || *p == '-')) advance();
                while (p < end && std::isdigit((unsigned char)*p)) advance();
            }
        }
        // Numeric suffixes (f/F float; u/U/l/L integer sizes).
        while (p < end && (*p == 'f' || *p == 'F' || *p == 'u' || *p == 'U' ||
                           *p == 'l' || *p == 'L')) {
            if (*p == 'f' || *p == 'F') isFloat = true;
            advance();
        }
        push(isFloat ? TokenKind::FloatLiteral : TokenKind::IntLiteral,
             std::string(start, p), l, c);
    }

    void run() {
        for (;;) {
            skipTrivia();
            if (p >= end) { push(TokenKind::End, "", line, col); return; }
            const int l = line, c = col;
            char ch = *p;

            if (isIdentStart(ch)) {
                const char* start = p;
                while (p < end && isIdentCont(*p)) advance();
                std::string id(start, p);
                auto it = keywords().find(id);
                push(it != keywords().end() ? it->second : TokenKind::Identifier, id, l, c);
                continue;
            }
            if (std::isdigit((unsigned char)ch) ||
                (ch == '.' && p + 1 < end && std::isdigit((unsigned char)p[1]))) {
                lexNumber();
                continue;
            }
            if (ch == '"') {                       // string literal (e.g. extern "C")
                advance();                         // opening quote
                std::string s;
                while (p < end && *p != '"') {
                    if (*p == '\\' && p + 1 < end) { s.push_back(*p); advance(); }
                    s.push_back(*p);
                    advance();
                }
                if (p < end) advance();            // closing quote
                push(TokenKind::StringLiteral, s, l, c);
                continue;
            }

            // Multi-char operators first.
            if (two('+', '+', TokenKind::Inc, "++", l, c)) continue;
            if (two('-', '-', TokenKind::Dec, "--", l, c)) continue;
            if (two('+', '=', TokenKind::PlusEq, "+=", l, c)) continue;
            if (two('-', '=', TokenKind::MinusEq, "-=", l, c)) continue;
            if (two('*', '=', TokenKind::StarEq, "*=", l, c)) continue;
            if (two('/', '=', TokenKind::SlashEq, "/=", l, c)) continue;
            if (two('%', '=', TokenKind::PercentEq, "%=", l, c)) continue;
            if (two('=', '=', TokenKind::Eq, "==", l, c)) continue;
            if (two('!', '=', TokenKind::Ne, "!=", l, c)) continue;
            if (two('<', '=', TokenKind::Le, "<=", l, c)) continue;
            if (two('>', '=', TokenKind::Ge, ">=", l, c)) continue;
            if (two('&', '&', TokenKind::AndAnd, "&&", l, c)) continue;
            if (two('|', '|', TokenKind::OrOr, "||", l, c)) continue;
            if (two('<', '<', TokenKind::Shl, "<<", l, c)) continue;
            if (two('>', '>', TokenKind::Shr, ">>", l, c)) continue;

            // Single-char tokens.
            TokenKind k = TokenKind::Unknown;
            switch (ch) {
                case '(': k = TokenKind::LParen; break;
                case ')': k = TokenKind::RParen; break;
                case '{': k = TokenKind::LBrace; break;
                case '}': k = TokenKind::RBrace; break;
                case '[': k = TokenKind::LBracket; break;
                case ']': k = TokenKind::RBracket; break;
                case ';': k = TokenKind::Semicolon; break;
                case ',': k = TokenKind::Comma; break;
                case '.': k = TokenKind::Dot; break;
                case '=': k = TokenKind::Assign; break;
                case '+': k = TokenKind::Plus; break;
                case '-': k = TokenKind::Minus; break;
                case '*': k = TokenKind::Star; break;
                case '/': k = TokenKind::Slash; break;
                case '%': k = TokenKind::Percent; break;
                case '<': k = TokenKind::Lt; break;
                case '>': k = TokenKind::Gt; break;
                case '!': k = TokenKind::Not; break;
                case '&': k = TokenKind::Amp; break;
                case '|': k = TokenKind::Pipe; break;
                case '^': k = TokenKind::Caret; break;
                case '~': k = TokenKind::Tilde; break;
                case '?': k = TokenKind::Question; break;
                case ':': k = TokenKind::Colon; break;
                default:  k = TokenKind::Unknown; break;
            }
            push(k, std::string(1, ch), l, c);
            advance();
        }
    }
};

}  // namespace

std::vector<Token> lex(const std::string& source) {
    Lexer lx(source);
    lx.run();
    return std::move(lx.out);
}

}  // namespace frontend
}  // namespace compiler
}  // namespace vgre
