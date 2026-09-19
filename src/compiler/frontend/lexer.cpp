// Hand-written CUDA-C-subset lexer — see include/vgre/compiler/frontend/lexer.h.

#include "vgre/compiler/frontend/lexer.h"

#include <cctype>

namespace vgre {
namespace compiler {
namespace frontend {

// tokenKindName / tokenSpelling / keywordLookup live in token.cpp (the single
// source of truth for the token vocabulary); this file only produces tokens.

namespace {

bool isIdentStart(char c) { return std::isalpha((unsigned char)c) || c == '_'; }
bool isIdentCont(char c)  { return std::isalnum((unsigned char)c) || c == '_'; }

struct Lexer {
    const char* base;   // first byte of the source (for byte-span offsets)
    const char* p;
    const char* end;
    int line = 1, col = 1;
    int tokBegin = 0;   // byte offset of the token currently being produced
    std::vector<Token> out;

    explicit Lexer(const std::string& s)
        : base(s.data()), p(s.data()), end(s.data() + s.size()) {}

    void advance(int n = 1) {
        for (int i = 0; i < n && p < end; ++i) {
            if (*p == '\n') { ++line; col = 1; } else { ++col; }
            ++p;
        }
    }

    // Push the token spanning [tokBegin, current p). Callers consume the token's
    // characters (advance) *before* pushing, so the byte span is exact.
    void push(TokenKind k, const std::string& text, int l, int c) {
        out.push_back(Token{k, text, l, c, tokBegin, int(p - base)});
    }

    // Fixed-length operator if p[0..n-1] match; consumes then pushes (span exact).
    bool two(char a, char b, TokenKind twoKind, const char* twoStr, int l, int c) {
        if (*p == a && p + 1 < end && p[1] == b) {
            advance(2);
            push(twoKind, twoStr, l, c);
            return true;
        }
        return false;
    }

    // Three-character operator (e.g. <<=, >>=); checked before its two-char prefix.
    bool three(char a, char b, char d, TokenKind kind, const char* str, int l, int c) {
        if (*p == a && p + 2 < end && p[1] == b && p[2] == d) {
            advance(3);
            push(kind, str, l, c);
            return true;
        }
        return false;
    }

    // Four-character operator prefix (the launch operators <<< and >>> are 3, but
    // this handles their disambiguation cleanly alongside <<= / >>=).
    bool threeSame(char a, TokenKind kind, const char* str, int l, int c) {
        if (*p == a && p + 2 < end && p[1] == a && p[2] == a) {
            advance(3);
            push(kind, str, l, c);
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
            tokBegin = int(p - base);              // byte offset of the next token
            if (p >= end) { push(TokenKind::End, "", line, col); return; }
            const int l = line, c = col;
            char ch = *p;

            if (isIdentStart(ch)) {
                const char* start = p;
                while (p < end && isIdentCont(*p)) advance();
                std::string id(start, p);
                push(keywordLookup(id), id, l, c);   // centralized keyword vocabulary
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
            if (ch == '\'') {                      // character literal 'a', '\n', '\x41'
                advance();                         // opening quote
                std::string s;
                while (p < end && *p != '\'') {
                    if (*p == '\\' && p + 1 < end) { s.push_back(*p); advance(); }
                    s.push_back(*p);
                    advance();
                }
                if (p < end) advance();            // closing quote
                push(TokenKind::CharLiteral, s, l, c);
                continue;
            }

            // Launch operators <<< / >>> before the (assign-)shift operators, so a
            // kernel launch is not mis-lexed as two shifts.
            if (threeSame('<', TokenKind::LaunchOpen, "<<<", l, c)) continue;
            if (threeSame('>', TokenKind::LaunchClose, ">>>", l, c)) continue;

            // Multi-char operators first (longest match wins — 3-char before 2).
            if (three('.', '.', '.', TokenKind::Ellipsis, "...", l, c)) continue;
            if (three('<', '=', '>', TokenKind::Spaceship, "<=>", l, c)) continue;
            if (three('-', '>', '*', TokenKind::ArrowStar, "->*", l, c)) continue;
            if (two('-', '>', TokenKind::Arrow, "->", l, c)) continue;
            if (two('.', '*', TokenKind::DotStar, ".*", l, c)) continue;
            if (two(':', ':', TokenKind::Scope, "::", l, c)) continue;
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
            // Shift-assign (3 chars) must be tried before the 2-char shift.
            if (three('<', '<', '=', TokenKind::ShlEq, "<<=", l, c)) continue;
            if (three('>', '>', '=', TokenKind::ShrEq, ">>=", l, c)) continue;
            if (two('<', '<', TokenKind::Shl, "<<", l, c)) continue;
            if (two('>', '>', TokenKind::Shr, ">>", l, c)) continue;
            if (two('&', '=', TokenKind::AmpEq, "&=", l, c)) continue;
            if (two('|', '=', TokenKind::PipeEq, "|=", l, c)) continue;
            if (two('^', '=', TokenKind::CaretEq, "^=", l, c)) continue;

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
            advance();                             // consume, then push (span exact)
            push(k, std::string(1, ch), l, c);
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
