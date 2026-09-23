// Recursive-descent parser — see include/vgre/compiler/frontend/parser.h.

#include "vgre/compiler/frontend/parser.h"
#include <algorithm>  // std::sort/min_element/find_if/... (don't rely on transitive includes)

#include "vgre/compiler/frontend/lexer.h"

#include <cstdlib>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace vgre {
namespace compiler {
namespace frontend {

namespace {

// Operator spelling stored on Binary/Assign/Unary nodes (for codegen dispatch).
const char* opSpelling(TokenKind k) {
    switch (k) {
        case TokenKind::Assign: return "=";
        case TokenKind::PlusEq: return "+="; case TokenKind::MinusEq: return "-=";
        case TokenKind::StarEq: return "*="; case TokenKind::SlashEq: return "/=";
        case TokenKind::PercentEq: return "%=";
        case TokenKind::AmpEq: return "&="; case TokenKind::PipeEq: return "|=";
        case TokenKind::CaretEq: return "^="; case TokenKind::ShlEq: return "<<=";
        case TokenKind::ShrEq: return ">>=";
        case TokenKind::Plus: return "+"; case TokenKind::Minus: return "-";
        case TokenKind::Star: return "*"; case TokenKind::Slash: return "/";
        case TokenKind::Percent: return "%";
        case TokenKind::Eq: return "=="; case TokenKind::Ne: return "!=";
        case TokenKind::Lt: return "<"; case TokenKind::Le: return "<=";
        case TokenKind::Gt: return ">"; case TokenKind::Ge: return ">=";
        case TokenKind::AndAnd: return "&&"; case TokenKind::OrOr: return "||";
        case TokenKind::Amp: return "&"; case TokenKind::Pipe: return "|";
        case TokenKind::Caret: return "^"; case TokenKind::Shl: return "<<";
        case TokenKind::Shr: return ">>"; case TokenKind::Not: return "!";
        case TokenKind::Tilde: return "~";
        default: return "?";
    }
}

// Binary-operator precedence (higher binds tighter); 0 = not a binary operator.
int binPrec(TokenKind k) {
    switch (k) {
        case TokenKind::OrOr:  return 1;
        case TokenKind::AndAnd: return 2;
        case TokenKind::Pipe:  return 3;
        case TokenKind::Caret: return 4;
        case TokenKind::Amp:   return 5;
        case TokenKind::Eq: case TokenKind::Ne: return 6;
        case TokenKind::Lt: case TokenKind::Le: case TokenKind::Gt: case TokenKind::Ge: return 7;
        case TokenKind::Shl: case TokenKind::Shr: return 8;
        case TokenKind::Plus: case TokenKind::Minus: return 9;
        case TokenKind::Star: case TokenKind::Slash: case TokenKind::Percent: return 10;
        default: return 0;
    }
}

// Decode the body of a character literal (quotes already stripped by the lexer):
// a plain char, or an escape \n \t \r \0 \\ \' \" \xHH \NNN(octal). Yields the
// character's integer value (C's `int` char-constant semantics).
int64_t decodeCharLiteral(const std::string& s) {
    if (s.empty()) return 0;
    if (s[0] != '\\') return (unsigned char)s[0];
    if (s.size() < 2) return '\\';
    char e = s[1];
    switch (e) {
        case 'n': return '\n'; case 't': return '\t'; case 'r': return '\r';
        case '0': return (s.size() == 2) ? 0 : (int64_t)std::strtoll(s.c_str() + 1, nullptr, 8);
        case '\\': return '\\'; case '\'': return '\''; case '"': return '"';
        case 'a': return '\a'; case 'b': return '\b'; case 'f': return '\f'; case 'v': return '\v';
        case 'x': return (int64_t)std::strtoll(s.c_str() + 2, nullptr, 16);   // \xHH
        default:
            if (e >= '1' && e <= '7') return (int64_t)std::strtoll(s.c_str() + 1, nullptr, 8);  // octal
            return (unsigned char)e;   // unknown escape → the literal char
    }
}

bool isAssignOp(TokenKind k) {
    return k == TokenKind::Assign || k == TokenKind::PlusEq || k == TokenKind::MinusEq ||
           k == TokenKind::StarEq || k == TokenKind::SlashEq || k == TokenKind::PercentEq ||
           k == TokenKind::AmpEq || k == TokenKind::PipeEq || k == TokenKind::CaretEq ||
           k == TokenKind::ShlEq || k == TokenKind::ShrEq;
}

bool isTypeStart(TokenKind k) {
    switch (k) {
        case TokenKind::KwConst: case TokenKind::KwVolatile:
        case TokenKind::KwUnsigned: case TokenKind::KwSigned:
        case TokenKind::KwVoid: case TokenKind::KwBool: case TokenKind::KwChar:
        case TokenKind::KwShort: case TokenKind::KwInt: case TokenKind::KwLong:
        case TokenKind::KwFloat: case TokenKind::KwDouble: case TokenKind::KwCudaHalf:
            return true;
        default: return false;
    }
}

// Byte size of a type (for sizeof): any pointer is 8; scalars use elemBytes().
int64_t sizeofType(const Type& t) { return t.ptr > 0 ? 8 : (int64_t)t.elemBytes(); }

struct Parser {
    std::vector<Token> toks;
    size_t pos = 0;
    bool failed = false;
    std::string err;
    Module* mod_ = nullptr;  // module being built (so parseType can resolve struct names)
    // Template state: the params of the function template currently being parsed
    // (so parseType sees `T` as a type), and the names of every template seen so far
    // (so `foo<int>(x)` is a template call, not the comparison chain `foo < int > (x)`).
    const std::vector<TemplateParam>* curTParams_ = nullptr;
    std::set<std::string> templateNames_;

    // Is `n` a type-parameter of the template currently being parsed?
    bool isCurTypeParam(const std::string& n) const {
        if (!curTParams_) return false;
        for (const auto& tp : *curTParams_) if (tp.isTypename && tp.name == n) return true;
        return false;
    }
    // A callee that `name<...>(...)` may legitimately template on: a user template
    // seen so far, or a texture/surface builtin (tex1D<T>/surf2Dread<T>/…).
    bool isTemplateCallee(const std::string& n) const {
        if (templateNames_.count(n)) return true;
        return n.compare(0, 3, "tex") == 0 || n.compare(0, 4, "surf") == 0;
    }

    const Token& cur() const { return toks[pos]; }
    TokenKind kind() const { return toks[pos].kind; }
    bool at(TokenKind k) const { return kind() == k; }

    void fail(const std::string& msg) {
        if (failed) return;
        failed = true;
        const Token& t = cur();
        err = std::to_string(t.line) + ":" + std::to_string(t.col) + ": " + msg;
    }

    const Token& advance() {
        const Token& t = toks[pos];
        if (pos + 1 < toks.size()) ++pos;
        return t;
    }
    bool accept(TokenKind k) { if (at(k)) { advance(); return true; } return false; }
    void expect(TokenKind k, const char* what) { if (!accept(k)) fail(std::string("expected ") + what); }

    // Consume a balanced parenthesized group (annotation args like __launch_bounds__(256)).
    void skipParenGroup() {
        if (!accept(TokenKind::LParen)) return;
        int depth = 1;
        while (depth > 0 && !at(TokenKind::End)) {
            if (at(TokenKind::LParen)) ++depth;
            else if (at(TokenKind::RParen)) --depth;
            advance();
        }
    }

    ExprPtr mkExpr(Expr::Kind k) {
        auto e = std::make_unique<Expr>();
        e->kind = k; e->line = cur().line; e->col = cur().col; return e;
    }

    // ── Types ─────────────────────────────────────────────────────────────────
    // CUDA built-in vector types (float4, int2, uchar3, …) are just structs with
    // components x/y/z/w. Map `name` → (component type, lane count); false if not one.
    static bool isVectorTypeName(const std::string& n) { Type c; int k; return vectorComponent(n, c, k); }
    static bool vectorComponent(const std::string& name, Type& comp, int& count) {
        if (name.size() < 2) return false;
        const char d = name.back();
        if (d < '1' || d > '4') return false;
        count = d - '0';
        const std::string base = name.substr(0, name.size() - 1);
        comp = Type{};
        if      (base == "float")  comp.base = Type::Float;
        else if (base == "double") comp.base = Type::Double;
        else if (base == "int")    comp.base = Type::Int;
        else if (base == "uint")  { comp.base = Type::Int;   comp.isUnsigned = true; }
        else if (base == "char")   comp.base = Type::Char;
        else if (base == "uchar") { comp.base = Type::Char;  comp.isUnsigned = true; }
        else if (base == "short")  comp.base = Type::Short;
        else if (base == "ushort"){ comp.base = Type::Short; comp.isUnsigned = true; }
        else if (base == "long" || base == "longlong")   comp.base = Type::Long;
        else if (base == "ulong" || base == "ulonglong"){ comp.base = Type::Long; comp.isUnsigned = true; }
        else return false;
        return true;
    }
    // Register a built-in vector type's StructDef in the module the first time it is
    // used, so all the existing struct machinery (member access, layout) applies.
    void ensureVectorStruct(const std::string& name) {
        if (!mod_ || mod_->findStruct(name)) return;
        Type comp; int count;
        if (!vectorComponent(name, comp, count)) return;
        StructDef def; def.name = name;
        static const char* kLanes[4] = {"x", "y", "z", "w"};
        const int esz = comp.elemBytes();
        int off = 0;
        for (int i = 0; i < count; ++i) {
            StructMember m; m.type = comp; m.name = kLanes[i];
            off = (off + esz - 1) / esz * esz; m.offset = off; off += esz;
            def.members.push_back(std::move(m));
        }
        def.size = (off + esz - 1) / esz * esz;   // natural alignment (matches CUDA sizeof)
        mod_->structs.push_back(std::move(def));
    }

    // Consume the trailing `*`/const/__restrict__ qualifiers of a pointer type.
    void parsePtrQualifiers(Type& out) {
        while (accept(TokenKind::Star)) {
            out.ptr++;
            while (accept(TokenKind::KwConst) || accept(TokenKind::KwVolatile) ||
                   accept(TokenKind::KwCudaRestrict)) { /* qualifier */ }
        }
    }

    bool parseType(Type& out) {
        // A template type-parameter in scope (`T`, `T*`, `const T*`) — a placeholder
        // resolved to a concrete type by instantiation before codegen.
        if (kind() == TokenKind::Identifier && isCurTypeParam(cur().text)) {
            out = Type{}; out.tparam = advance().text; parsePtrQualifiers(out); return true;
        }
        // Texture/surface object handles are opaque 64-bit ids (unsigned long long).
        if (kind() == TokenKind::Identifier &&
            (cur().text == "cudaTextureObject_t" || cur().text == "cudaSurfaceObject_t")) {
            out = Type{}; out.base = Type::Long; out.isUnsigned = true; advance();
            parsePtrQualifiers(out); return true;
        }
        // A CUDA built-in vector type (float4, int2, …): register it as a struct on
        // first use so the struct path below (and everywhere else) just works.
        if (kind() == TokenKind::Identifier) ensureVectorStruct(cur().text);
        // Struct type: a known struct name, optionally preceded by the 'struct'
        // keyword (`struct Foo` or bare `Foo`). Resolved against the module table.
        bool sawStructKw = (kind() == TokenKind::KwStruct);
        if (sawStructKw ||
            (kind() == TokenKind::Identifier && mod_ && mod_->findStruct(cur().text))) {
            out = Type{};
            if (sawStructKw) advance();
            if (kind() != TokenKind::Identifier || !mod_ || !mod_->findStruct(cur().text)) {
                fail("unknown struct type"); return false;
            }
            out.base = Type::Struct;
            out.structName = advance().text;
            while (accept(TokenKind::Star)) {
                out.ptr++;
                while (accept(TokenKind::KwConst) || accept(TokenKind::KwCudaRestrict)) { /* qualifier */ }
            }
            return true;
        }
        if (!isTypeStart(kind())) return false;
        out = Type{};
        // qualifiers / sign, in any leading order
        for (;;) {
            if (accept(TokenKind::KwConst)) { out.isConst = true; continue; }
            if (accept(TokenKind::KwVolatile)) { /* accepted, no effect on lowering */ continue; }
            if (accept(TokenKind::KwUnsigned)) { out.isUnsigned = true; continue; }
            if (accept(TokenKind::KwSigned)) { out.isUnsigned = false; continue; }
            break;
        }
        // `const T` / `const T*` — a template type-parameter after leading qualifiers.
        if (kind() == TokenKind::Identifier && isCurTypeParam(cur().text)) {
            out.tparam = advance().text; parsePtrQualifiers(out); return true;
        }
        // A struct type can follow leading qualifiers too (e.g. `const Vec*`,
        // `const float4*`): the base-type switch below only knows primitives, so catch
        // a struct / built-in vector name here (keeping the `isConst` we just parsed).
        if (kind() == TokenKind::Identifier) ensureVectorStruct(cur().text);
        if (kind() == TokenKind::KwStruct ||
            (kind() == TokenKind::Identifier && mod_ && mod_->findStruct(cur().text))) {
            if (kind() == TokenKind::KwStruct) advance();
            if (kind() != TokenKind::Identifier || !mod_ || !mod_->findStruct(cur().text)) {
                fail("unknown struct type"); return false;
            }
            out.base = Type::Struct;
            out.structName = advance().text;
            while (accept(TokenKind::Star)) {
                out.ptr++;
                while (accept(TokenKind::KwConst) || accept(TokenKind::KwCudaRestrict)) { /* qualifier */ }
            }
            return true;
        }
        switch (kind()) {
            case TokenKind::KwVoid:   out.base = Type::Void;  advance(); break;
            case TokenKind::KwBool:   out.base = Type::Bool;  advance(); break;
            case TokenKind::KwChar:   out.base = Type::Char;  advance(); break;
            case TokenKind::KwShort:  out.base = Type::Short; advance(); break;
            case TokenKind::KwInt:    out.base = Type::Int;   advance(); break;
            case TokenKind::KwLong:   out.base = Type::Long;  advance(); break;
            case TokenKind::KwFloat:  out.base = Type::Float; advance(); break;
            case TokenKind::KwDouble: out.base = Type::Double; advance(); break;
            case TokenKind::KwCudaHalf:   out.base = Type::Half;  advance(); break;
            default:
                // "unsigned"/"const" alone implies int.
                out.base = Type::Int; break;
        }
        // pointer stars with trailing const/volatile/__restrict__ qualifiers
        while (accept(TokenKind::Star)) {
            out.ptr++;
            while (accept(TokenKind::KwConst) || accept(TokenKind::KwVolatile) ||
                   accept(TokenKind::KwCudaRestrict)) { /* qualifier */ }
        }
        return true;
    }

    // ── Expressions (precedence climbing) ──────────────────────────────────────
    ExprPtr parseExpr() { return parseAssign(); }

    // Conditional operator: cond ? a : b (right-associative, below assignment).
    ExprPtr parseTernary() {
        ExprPtr cond = parseBinary(1);
        if (!cond || failed) return cond;
        if (at(TokenKind::Question)) {
            auto node = mkExpr(Expr::Ternary);
            advance();
            ExprPtr thenE = parseAssign();          // full expression between ? and :
            expect(TokenKind::Colon, "':'");
            if (failed || !thenE) return nullptr;
            ExprPtr elseE = parseTernary();          // right-associative
            if (!elseE) return nullptr;
            node->args.push_back(std::move(cond));
            node->args.push_back(std::move(thenE));
            node->args.push_back(std::move(elseE));
            return node;
        }
        return cond;
    }

    ExprPtr parseAssign() {
        ExprPtr lhs = parseTernary();
        if (!lhs || failed) return lhs;
        if (isAssignOp(kind())) {
            auto node = mkExpr(Expr::Assign);
            node->str = opSpelling(kind());
            advance();
            ExprPtr rhs = parseAssign();  // right-associative
            if (!rhs) return nullptr;
            node->args.push_back(std::move(lhs));
            node->args.push_back(std::move(rhs));
            return node;
        }
        return lhs;
    }

    ExprPtr parseBinary(int minPrec) {
        ExprPtr lhs = parseUnary();
        if (!lhs || failed) return lhs;
        for (;;) {
            int prec = binPrec(kind());
            if (prec == 0 || prec < minPrec) break;
            auto node = mkExpr(Expr::Binary);
            node->str = opSpelling(kind());
            advance();
            ExprPtr rhs = parseBinary(prec + 1);  // left-associative
            if (!rhs) return nullptr;
            node->args.push_back(std::move(lhs));
            node->args.push_back(std::move(rhs));
            lhs = std::move(node);
        }
        return lhs;
    }

    ExprPtr parseUnary() {
        TokenKind k = kind();
        // sizeof: `sizeof(type)` folds to the type's byte size at parse time (a
        // compile-time integer constant). `sizeof expr` is not supported (the
        // parser has no type inference) — a located error, never wrong code.
        if (k == TokenKind::KwSizeof) {
            advance();  // sizeof
            if (at(TokenKind::LParen) && pos + 1 < toks.size() && isTypeStart(toks[pos + 1].kind)) {
                advance();  // '('
                Type t;
                if (!parseType(t)) { fail("expected a type in sizeof"); return nullptr; }
                expect(TokenKind::RParen, "')'");
                if (failed) return nullptr;
                auto e = mkExpr(Expr::IntLit);
                e->ival = sizeofType(t);
                e->wide = false;
                return e;
            }
            fail("sizeof requires a parenthesized type, e.g. sizeof(float)");
            return nullptr;
        }
        // C-style cast: '(' <type> ')' <unary>. Distinguished from a parenthesized
        // expression by a type keyword right after '('.
        if (k == TokenKind::LParen && pos + 1 < toks.size() && isTypeStart(toks[pos + 1].kind)) {
            auto node = mkExpr(Expr::Cast);
            advance();  // '('
            if (!parseType(node->castType)) { fail("expected a type in cast"); return nullptr; }
            expect(TokenKind::RParen, "')'");
            if (failed) return nullptr;
            ExprPtr operand = parseUnary();
            if (!operand) return nullptr;
            node->args.push_back(std::move(operand));
            return node;
        }
        if (k == TokenKind::Inc || k == TokenKind::Dec) {          // prefix ++x / --x
            auto node = mkExpr(Expr::Unary);
            node->str = (k == TokenKind::Inc) ? "pre++" : "pre--";
            advance();
            ExprPtr operand = parseUnary();
            if (!operand) return nullptr;
            node->args.push_back(std::move(operand));
            return node;
        }
        if (k == TokenKind::Minus || k == TokenKind::Not || k == TokenKind::Tilde ||
            k == TokenKind::Star  || k == TokenKind::Amp || k == TokenKind::Plus) {
            auto node = mkExpr(Expr::Unary);
            node->str = opSpelling(k);
            advance();
            ExprPtr operand = parseUnary();
            if (!operand) return nullptr;
            node->args.push_back(std::move(operand));
            return node;
        }
        return parsePostfix();
    }

    // Try to parse an explicit template-argument list `< T, 4, … >` at a call site.
    // On success `pos` is left just after `>` and `out` holds the args; on failure
    // `pos` is fully restored (so the `<` can still parse as a comparison operator).
    // Nested angle args (`foo<bar<int>>`) aren't supported (`>>` won't split here).
    bool tryParseTemplateArgs(std::vector<TemplateArg>& out) {
        out.clear();
        size_t save = pos;
        if (!accept(TokenKind::Lt)) return false;
        if (!at(TokenKind::Gt)) {
            for (;;) {
                TemplateArg a;
                const bool typeAhead =
                    isTypeStart(kind()) || kind() == TokenKind::KwStruct ||
                    (kind() == TokenKind::Identifier &&
                     ((mod_ && mod_->findStruct(cur().text)) || isCurTypeParam(cur().text)));
                if (typeAhead) {
                    if (!parseType(a.type)) { pos = save; out.clear(); return false; }
                    a.isType = true;
                } else if (at(TokenKind::IntLiteral)) {           // non-type constant `foo<4>()`
                    a.isType = false;
                    a.value = (int64_t)std::strtoll(advance().text.c_str(), nullptr, 0);
                } else { pos = save; out.clear(); return false; }
                out.push_back(std::move(a));
                if (!accept(TokenKind::Comma)) break;
            }
        }
        if (!accept(TokenKind::Gt)) { pos = save; out.clear(); return false; }
        return true;
    }

    ExprPtr parsePostfix() {
        ExprPtr e = parsePrimary();
        if (!e || failed) return e;
        for (;;) {
            // Template call `callee<args>(…)` — only for a name we know can be a
            // template (a user template seen so far, or a tex/surf builtin), so an
            // ordinary comparison chain `a < b > (c)` is never misread as a call.
            if (e->kind == Expr::Ident && at(TokenKind::Lt) && isTemplateCallee(e->str)) {
                std::vector<TemplateArg> targs;
                size_t save = pos;
                if (tryParseTemplateArgs(targs) && at(TokenKind::LParen)) {
                    advance();                                    // consume '('
                    auto node = mkExpr(Expr::Call);
                    node->str = e->str;
                    node->targs = std::move(targs);
                    if (!at(TokenKind::RParen)) {
                        for (;;) {
                            ExprPtr a = parseAssign();
                            if (!a) return nullptr;
                            node->args.push_back(std::move(a));
                            if (!accept(TokenKind::Comma)) break;
                        }
                    }
                    expect(TokenKind::RParen, "')'");
                    e = std::move(node);
                    continue;
                }
                pos = save;   // not a template call → let the binary parser see '<'
            }
            if (accept(TokenKind::LBracket)) {          // e[index]
                auto node = mkExpr(Expr::Index);
                ExprPtr idx = parseExpr();
                expect(TokenKind::RBracket, "']'");
                if (!idx || failed) return nullptr;
                node->args.push_back(std::move(e));
                node->args.push_back(std::move(idx));
                e = std::move(node);
            } else if (accept(TokenKind::Dot)) {        // e.field
                auto node = mkExpr(Expr::Member);
                if (!at(TokenKind::Identifier)) { fail("expected member name after '.'"); return nullptr; }
                node->str = advance().text;
                node->args.push_back(std::move(e));
                e = std::move(node);
            } else if (accept(TokenKind::Arrow)) {      // e->field (member through a pointer)
                auto node = mkExpr(Expr::Member);
                if (!at(TokenKind::Identifier)) { fail("expected member name after '->'"); return nullptr; }
                node->str = advance().text;
                node->args.push_back(std::move(e));
                e = std::move(node);   // codegen dispatches on the object's type (struct value vs pointer)
            } else if (accept(TokenKind::LParen)) {     // callee(args…)  (e must be an Ident)
                auto node = mkExpr(Expr::Call);
                if (e->kind != Expr::Ident) { fail("call of non-function"); return nullptr; }
                node->str = e->str;
                if (!at(TokenKind::RParen)) {
                    for (;;) {
                        ExprPtr a = parseAssign();
                        if (!a) return nullptr;
                        node->args.push_back(std::move(a));
                        if (!accept(TokenKind::Comma)) break;
                    }
                }
                expect(TokenKind::RParen, "')'");
                e = std::move(node);
            } else if (at(TokenKind::Inc) || at(TokenKind::Dec)) {   // postfix x++ / x--
                auto node = mkExpr(Expr::Unary);
                node->str = at(TokenKind::Inc) ? "post++" : "post--";
                advance();
                node->args.push_back(std::move(e));
                e = std::move(node);
            } else {
                break;
            }
        }
        return e;
    }

    ExprPtr parsePrimary() {
        switch (kind()) {
            case TokenKind::IntLiteral: {
                auto e = mkExpr(Expr::IntLit);
                const std::string txt = advance().text;
                e->ival = static_cast<int64_t>(std::strtoll(txt.c_str(), nullptr, 0));
                // `long` if it carries an l/L suffix or doesn't fit in 32 bits.
                e->wide = txt.find_first_of("lL") != std::string::npos ||
                          e->ival > 2147483647LL || e->ival < -2147483648LL;
                return e;
            }
            case TokenKind::FloatLiteral: {
                auto e = mkExpr(Expr::FloatLit);
                const std::string txt = advance().text;
                e->fval = std::strtod(txt.c_str(), nullptr);
                // `double` unless it has an f/F suffix (C default is double).
                e->wide = txt.find_first_of("fF") == std::string::npos;
                return e;
            }
            case TokenKind::CharLiteral: {
                auto e = mkExpr(Expr::IntLit);
                e->ival = decodeCharLiteral(advance().text);   // 'a' -> 97, '\n' -> 10
                e->wide = false;
                return e;
            }
            case TokenKind::KwTrue: case TokenKind::KwFalse: {
                auto e = mkExpr(Expr::IntLit);
                e->ival = (kind() == TokenKind::KwTrue) ? 1 : 0;
                advance();
                e->wide = false;
                return e;
            }
            case TokenKind::KwNullptr: {                        // null pointer constant (value 0)
                auto e = mkExpr(Expr::IntLit);
                advance();
                e->ival = 0; e->wide = false;
                return e;
            }
            case TokenKind::Identifier: {
                auto e = mkExpr(Expr::Ident);
                e->str = advance().text;
                return e;
            }
            case TokenKind::LParen: {
                advance();
                ExprPtr e = parseExpr();
                expect(TokenKind::RParen, "')'");
                return e;
            }
            default:
                fail("expected an expression");
                return nullptr;
        }
    }

    // ── Statements ──────────────────────────────────────────────────────────────
    StmtPtr mkStmt(Stmt::Kind k) {
        auto s = std::make_unique<Stmt>();
        s->kind = k; s->line = cur().line; s->col = cur().col; return s;
    }

    StmtPtr parseStmt() {
        switch (kind()) {
            case TokenKind::LBrace:     return parseBlockStmt();
            case TokenKind::KwIf:       return parseIf();
            case TokenKind::KwFor:      return parseFor();
            case TokenKind::KwWhile:    return parseWhile();
            case TokenKind::KwDo:       return parseDoWhile();
            case TokenKind::KwSwitch:   return parseSwitch();
            case TokenKind::KwReturn:   return parseReturn();
            case TokenKind::KwBreak:    { auto s = mkStmt(Stmt::Break); advance(); expect(TokenKind::Semicolon, "';'"); return failed ? nullptr : std::move(s); }
            case TokenKind::KwContinue: { auto s = mkStmt(Stmt::Continue); advance(); expect(TokenKind::Semicolon, "';'"); return failed ? nullptr : std::move(s); }
            case TokenKind::Semicolon:  { auto s = mkStmt(Stmt::Empty); advance(); return s; }
            default: break;
        }
        if (isTypeStart(kind()) || at(TokenKind::KwCudaShared) || at(TokenKind::KwExtern) ||
            at(TokenKind::KwStatic) || at(TokenKind::KwInline) || at(TokenKind::KwStruct) ||
            (at(TokenKind::Identifier) && mod_ &&
             (mod_->findStruct(cur().text) || isVectorTypeName(cur().text))))
            return parseVarDecl();   // (static) (extern) __shared__ …, volatile T x, `Vec p;`, `float4 v;`, etc.
        // expression statement
        auto s = mkStmt(Stmt::ExprStmt);
        s->expr = parseExpr();
        expect(TokenKind::Semicolon, "';'");
        return failed ? nullptr : std::move(s);
    }

    StmtPtr parseBlockStmt() {
        auto s = mkStmt(Stmt::Block);
        expect(TokenKind::LBrace, "'{'");
        while (!at(TokenKind::RBrace) && !at(TokenKind::End) && !failed) {
            StmtPtr st = parseStmt();
            if (!st) return nullptr;
            s->body.push_back(std::move(st));
        }
        expect(TokenKind::RBrace, "'}'");
        return failed ? nullptr : std::move(s);
    }

    StmtPtr parseVarDecl() {
        bool isExternShared = false, isShared = false;
        // Leading storage-class qualifiers (accepted, no effect on lowering):
        // `static __shared__ T s[N];`, `inline`, etc.
        while (accept(TokenKind::KwStatic) || accept(TokenKind::KwInline)) { /* qualifier */ }
        // `extern __shared__ T name[];` — dynamic shared memory (size from launch).
        if (accept(TokenKind::KwExtern)) { isExternShared = true; isShared = true; }
        if (accept(TokenKind::KwCudaShared)) isShared = true;   // __shared__ [type] name[N];
        if (accept(TokenKind::KwStatic) || accept(TokenKind::KwInline)) { /* e.g. __shared__ static */ }
        Type base;
        if (!parseType(base)) { fail("expected a type"); return nullptr; }

        // Parse one declarator (name [dims] [= init]) sharing the base type. C allows
        // several comma-separated declarators in one statement (`int a, b = 1;`); we
        // emit one VarDecl each and wrap them in a Block when there is more than one.
        auto parseDeclarator = [&]() -> StmtPtr {
            auto s = mkStmt(Stmt::VarDecl);
            s->type = base; s->isExternShared = isExternShared; s->isShared = isShared;
            if (!at(TokenKind::Identifier)) { fail("expected a variable name"); return nullptr; }
            s->name = advance().text;
            while (accept(TokenKind::LBracket)) {              // array declarator name[N][M]…
                if (at(TokenKind::RBracket)) {                 // empty [] — dynamic extern shared
                    if (!s->isExternShared) { fail("only 'extern __shared__' may use an unsized []"); return nullptr; }
                    advance();  // ]
                    continue;
                }
                if (!at(TokenKind::IntLiteral)) { fail("expected an array size"); return nullptr; }
                int dim = static_cast<int>(std::strtoll(advance().text.c_str(), nullptr, 0));
                expect(TokenKind::RBracket, "']'");
                if (dim <= 0) { fail("array size must be positive"); return nullptr; }
                s->arrayDims.push_back(dim);
            }
            if (!s->arrayDims.empty()) {
                s->arraySize = 1;
                for (int d : s->arrayDims) s->arraySize *= d;  // total element count
            }
            if (accept(TokenKind::Assign)) {
                s->expr = parseExpr();
                if (!s->expr) return nullptr;
            }
            return s;
        };

        StmtPtr first = parseDeclarator();
        if (!first) return nullptr;
        if (!at(TokenKind::Comma)) {                           // single declarator (common case)
            expect(TokenKind::Semicolon, "';'");
            return failed ? nullptr : std::move(first);
        }
        auto blk = mkStmt(Stmt::Block);                        // `T a, b, …;` → a block of decls
        blk->body.push_back(std::move(first));
        while (accept(TokenKind::Comma)) {
            StmtPtr d = parseDeclarator();
            if (!d) return nullptr;
            blk->body.push_back(std::move(d));
        }
        expect(TokenKind::Semicolon, "';'");
        return failed ? nullptr : std::move(blk);
    }

    StmtPtr parseIf() {
        auto s = mkStmt(Stmt::If);
        advance();  // if
        expect(TokenKind::LParen, "'('");
        s->expr = parseExpr();
        expect(TokenKind::RParen, "')'");
        if (failed) return nullptr;
        StmtPtr thenS = parseStmt();
        if (!thenS) return nullptr;
        s->body.push_back(std::move(thenS));
        if (accept(TokenKind::KwElse)) {
            StmtPtr elseS = parseStmt();
            if (!elseS) return nullptr;
            s->elseBody.push_back(std::move(elseS));
        }
        return s;
    }

    StmtPtr parseWhile() {
        auto s = mkStmt(Stmt::While);
        advance();  // while
        expect(TokenKind::LParen, "'('");
        s->expr = parseExpr();
        expect(TokenKind::RParen, "')'");
        if (failed) return nullptr;
        StmtPtr body = parseStmt();
        if (!body) return nullptr;
        s->body.push_back(std::move(body));
        return s;
    }

    StmtPtr parseSwitch() {
        auto s = mkStmt(Stmt::Switch);
        advance();  // switch
        expect(TokenKind::LParen, "'('");
        s->expr = parseExpr();
        expect(TokenKind::RParen, "')'");
        expect(TokenKind::LBrace, "'{'");
        if (failed) return nullptr;
        // Body is a flat list of statements with `case N:` / `default:` markers
        // interleaved (C fall-through semantics; codegen wires the jumps).
        while (!at(TokenKind::RBrace) && !at(TokenKind::End) && !failed) {
            if (at(TokenKind::KwCase)) {
                auto c = mkStmt(Stmt::Case);
                advance();  // case
                c->expr = parseExpr();       // constant label expression
                expect(TokenKind::Colon, "':'");
                if (failed) return nullptr;
                s->body.push_back(std::move(c));
            } else if (at(TokenKind::KwDefault)) {
                auto d = mkStmt(Stmt::Default);
                advance();  // default
                expect(TokenKind::Colon, "':'");
                if (failed) return nullptr;
                s->body.push_back(std::move(d));
            } else {
                StmtPtr st = parseStmt();
                if (!st) return nullptr;
                s->body.push_back(std::move(st));
            }
        }
        expect(TokenKind::RBrace, "'}'");
        return failed ? nullptr : std::move(s);
    }

    StmtPtr parseDoWhile() {
        auto s = mkStmt(Stmt::DoWhile);
        advance();  // do
        StmtPtr body = parseStmt();
        if (!body) return nullptr;
        s->body.push_back(std::move(body));
        expect(TokenKind::KwWhile, "'while'");
        expect(TokenKind::LParen, "'('");
        s->expr = parseExpr();
        expect(TokenKind::RParen, "')'");
        expect(TokenKind::Semicolon, "';'");
        return failed ? nullptr : std::move(s);
    }

    StmtPtr parseFor() {
        auto s = mkStmt(Stmt::For);
        advance();  // for
        expect(TokenKind::LParen, "'('");
        // init: declaration, expr-stmt, or empty
        if (accept(TokenKind::Semicolon)) {
            // no init
        } else if (isTypeStart(kind()) ||
                   (at(TokenKind::Identifier) && mod_ &&
                    (mod_->findStruct(cur().text) || isVectorTypeName(cur().text)))) {
            s->forInit = parseVarDecl();   // consumes the ';'
            if (!s->forInit) return nullptr;
        } else {
            auto initS = mkStmt(Stmt::ExprStmt);
            initS->expr = parseExpr();
            expect(TokenKind::Semicolon, "';'");
            s->forInit = std::move(initS);
        }
        if (!at(TokenKind::Semicolon)) { s->forCond = parseExpr(); if (!s->forCond) return nullptr; }
        expect(TokenKind::Semicolon, "';'");
        if (!at(TokenKind::RParen)) { s->forIncr = parseExpr(); if (!s->forIncr) return nullptr; }
        expect(TokenKind::RParen, "')'");
        if (failed) return nullptr;
        StmtPtr body = parseStmt();
        if (!body) return nullptr;
        s->body.push_back(std::move(body));
        return s;
    }

    StmtPtr parseReturn() {
        auto s = mkStmt(Stmt::Return);
        advance();  // return
        if (!at(TokenKind::Semicolon)) { s->expr = parseExpr(); if (!s->expr) return nullptr; }
        expect(TokenKind::Semicolon, "';'");
        return failed ? nullptr : std::move(s);
    }

    // ── Kernel / module ─────────────────────────────────────────────────────────
    // template < typename T, int N, … >  — a function-template header. Type params
    // (`typename`/`class T`) and non-type params (`int N`). Returns false on error.
    bool parseTemplateHeader(std::vector<TemplateParam>& out) {
        advance();  // 'template'
        expect(TokenKind::Lt, "'<'");
        if (!at(TokenKind::Gt)) {
            for (;;) {
                TemplateParam tp;
                if (accept(TokenKind::KwTypename) || accept(TokenKind::KwClass)) {
                    tp.isTypename = true;
                    if (!at(TokenKind::Identifier)) { fail("expected a template type-parameter name"); return false; }
                    tp.name = advance().text;
                } else {                                   // non-type parameter: `int N`
                    tp.isTypename = false;
                    if (!parseType(tp.type)) { fail("expected a template parameter"); return false; }
                    if (!at(TokenKind::Identifier)) { fail("expected a non-type template parameter name"); return false; }
                    tp.name = advance().text;
                }
                out.push_back(std::move(tp));
                if (!accept(TokenKind::Comma)) break;
            }
        }
        expect(TokenKind::Gt, "'>'");
        return !failed;
    }

    std::unique_ptr<Kernel> parseKernel() {
        auto k = std::make_unique<Kernel>();
        // A function template: parse its header, then compile the body with its type
        // parameters in scope so `T`/`T*` parse as types. It is instantiated on demand.
        std::vector<TemplateParam> tparams;
        if (at(TokenKind::KwTemplate)) {
            if (!parseTemplateHeader(tparams)) return nullptr;
            k->tparams = tparams;
            curTParams_ = &tparams;
        }
        struct TParamGuard { const std::vector<TemplateParam>*& slot; ~TParamGuard() { slot = nullptr; } } guard{curTParams_};
        // Leading qualifiers in any order: extern "C", the execution-space specifiers,
        // and inline/storage hints. Only __global__ changes lowering (it marks the
        // entry); __device__ helpers are inlined; the rest are accepted and ignored.
        for (;;) {
            if (accept(TokenKind::KwExtern)) { accept(TokenKind::StringLiteral); continue; }
            if (accept(TokenKind::KwCudaGlobal)) { k->isGlobal = true; continue; }
            if (accept(TokenKind::KwCudaDevice)) { k->isDevice = true; continue; }
            if (accept(TokenKind::KwCudaHost)) { k->isHost = true; continue; }
            if (accept(TokenKind::KwCudaForceinline) || accept(TokenKind::KwCudaNoinline) ||
                accept(TokenKind::KwCudaInlineHint)) continue;
            if (accept(TokenKind::KwStatic) || accept(TokenKind::KwInline)) continue;
            if (accept(TokenKind::KwCudaLaunchBounds)) { skipParenGroup(); continue; }  // __launch_bounds__(...)
            break;
        }
        Type ret;
        if (!parseType(ret)) { fail("expected a return type"); return nullptr; }
        k->returnType = ret;
        if (!at(TokenKind::Identifier)) { fail("expected a kernel name"); return nullptr; }
        k->name = advance().text;
        if (!k->tparams.empty()) templateNames_.insert(k->name);   // so callers parse `name<…>(…)`
        expect(TokenKind::LParen, "'('");
        if (!at(TokenKind::RParen)) {
            for (;;) {
                Param p;
                if (!parseType(p.type)) { fail("expected a parameter type"); return nullptr; }
                if (at(TokenKind::Identifier)) p.name = advance().text;  // name optional
                k->params.push_back(std::move(p));
                if (!accept(TokenKind::Comma)) break;
            }
        }
        expect(TokenKind::RParen, "')'");
        if (failed) return nullptr;
        // Body block.
        StmtPtr block = parseBlockStmt();
        if (!block) return nullptr;
        k->body = std::move(block->body);
        return k;
    }

    // struct Name { type member; ... };  — scalar/pointer members, natural alignment.
    bool parseStructDef() {
        advance();  // 'struct'
        if (!at(TokenKind::Identifier)) { fail("expected a struct name"); return false; }
        StructDef def;
        def.name = advance().text;
        expect(TokenKind::LBrace, "'{'");
        int offset = 0, maxAlign = 1;
        while (!at(TokenKind::RBrace) && !at(TokenKind::End) && !failed) {
            StructMember mem;
            if (!parseType(mem.type)) { fail("expected a struct member type"); return false; }
            if (mem.type.isStruct()) { fail("nested struct members are unsupported"); return false; }
            if (!at(TokenKind::Identifier)) { fail("expected a struct member name"); return false; }
            mem.name = advance().text;
            expect(TokenKind::Semicolon, "';'");
            const int sz = mem.type.isPointer() ? 8 : mem.type.elemBytes();
            const int align = sz > 0 ? sz : 1;
            offset = (offset + align - 1) / align * align;   // natural alignment
            mem.offset = offset;
            offset += sz;
            if (align > maxAlign) maxAlign = align;
            def.members.push_back(std::move(mem));
        }
        expect(TokenKind::RBrace, "'}'");
        expect(TokenKind::Semicolon, "';'");
        if (failed) return false;
        def.size = (offset + maxAlign - 1) / maxAlign * maxAlign;
        mod_->structs.push_back(std::move(def));
        return true;
    }

    std::unique_ptr<Module> parseModule() {
        auto m = std::make_unique<Module>();
        mod_ = m.get();  // so parseType/parseKernel can resolve struct names
        while (!at(TokenKind::End) && !failed) {
            if (accept(TokenKind::Semicolon)) continue;
            if (at(TokenKind::KwStruct)) { if (!parseStructDef()) return nullptr; continue; }
            auto k = parseKernel();
            if (!k) return nullptr;
            m->kernels.push_back(std::move(k));
        }
        return failed ? nullptr : std::move(m);
    }
};

// ── Template instantiation (monomorphization) ────────────────────────────────
// After parsing, every `template<…>` function is a Kernel with non-empty tparams
// and every `foo<int>(…)`/`foo(x)` call carries explicit or deducible type args.
// This pass clones a concrete copy of each template per distinct argument set,
// substitutes the type/non-type parameters throughout, rewrites the calls to the
// mangled instance names, and drops the templates — so codegen and the compiled
// tier only ever see ordinary concrete functions. Chained templates (a template
// calling another) resolve via a worklist to a fixed point.

ExprPtr cloneExpr(const Expr& e) {
    auto n = std::make_unique<Expr>();
    n->kind = e.kind; n->line = e.line; n->col = e.col;
    n->ival = e.ival; n->fval = e.fval; n->wide = e.wide;
    n->str = e.str; n->castType = e.castType; n->targs = e.targs;
    for (const auto& a : e.args) n->args.push_back(cloneExpr(*a));
    return n;
}
StmtPtr cloneStmt(const Stmt& s) {
    auto n = std::make_unique<Stmt>();
    n->kind = s.kind; n->line = s.line; n->col = s.col;
    n->type = s.type; n->name = s.name; n->isShared = s.isShared;
    n->isExternShared = s.isExternShared; n->arraySize = s.arraySize; n->arrayDims = s.arrayDims;
    if (s.expr) n->expr = cloneExpr(*s.expr);
    for (const auto& b : s.body) n->body.push_back(cloneStmt(*b));
    for (const auto& b : s.elseBody) n->elseBody.push_back(cloneStmt(*b));
    if (s.forInit) n->forInit = cloneStmt(*s.forInit);
    if (s.forCond) n->forCond = cloneExpr(*s.forCond);
    if (s.forIncr) n->forIncr = cloneExpr(*s.forIncr);
    return n;
}

std::string mangleType(const Type& t) {
    std::string s;
    if (t.isConst) s += "K";
    if (t.isUnsigned) s += "u";
    switch (t.base) {
        case Type::Void: s += "v"; break; case Type::Bool: s += "b"; break;
        case Type::Char: s += "c"; break; case Type::Short: s += "s"; break;
        case Type::Int: s += "i"; break;  case Type::Long: s += "l"; break;
        case Type::Float: s += "f"; break; case Type::Double: s += "d"; break;
        case Type::Half: s += "h"; break;  case Type::Struct: s += "S" + t.structName; break;
    }
    for (int i = 0; i < t.ptr; ++i) s += "P";
    return s;
}

struct Instantiator {
    Module& m;
    std::string err;
    std::unordered_map<std::string, const Kernel*> templates;   // name → template
    std::unordered_map<std::string, bool> done;                 // mangled instance names
    std::vector<std::unique_ptr<Kernel>> instances;
    std::vector<Kernel*> worklist;

    explicit Instantiator(Module& mod) : m(mod) {}

    void fail(const std::string& e) { if (err.empty()) err = e; }

    void substType(Type& t, const std::unordered_map<std::string, Type>& tmap) {
        if (t.tparam.empty()) return;
        auto it = tmap.find(t.tparam);
        if (it == tmap.end()) return;
        const int extraPtr = t.ptr; const bool wasConst = t.isConst;
        t = it->second;                 // adopt the concrete base/struct/unsigned/ptr
        t.ptr += extraPtr;              // `T*` over T=float ⇒ float*
        t.isConst = t.isConst || wasConst;
        t.tparam.clear();
    }
    void substExpr(Expr& e, const std::unordered_map<std::string, Type>& tmap,
                   const std::unordered_map<std::string, int64_t>& vmap) {
        substType(e.castType, tmap);
        if (e.kind == Expr::Ident) {                     // non-type parameter → its constant
            auto it = vmap.find(e.str);
            if (it != vmap.end()) { e.kind = Expr::IntLit; e.ival = it->second; e.wide = false; e.str.clear(); }
        }
        for (auto& ta : e.targs) if (ta.isType) substType(ta.type, tmap);
        for (auto& a : e.args) substExpr(*a, tmap, vmap);
    }
    void substStmt(Stmt& s, const std::unordered_map<std::string, Type>& tmap,
                   const std::unordered_map<std::string, int64_t>& vmap) {
        substType(s.type, tmap);
        if (s.expr) substExpr(*s.expr, tmap, vmap);
        for (auto& b : s.body) substStmt(*b, tmap, vmap);
        for (auto& b : s.elseBody) substStmt(*b, tmap, vmap);
        if (s.forInit) substStmt(*s.forInit, tmap, vmap);
        if (s.forCond) substExpr(*s.forCond, tmap, vmap);
        if (s.forIncr) substExpr(*s.forIncr, tmap, vmap);
    }

    // Best-effort type of a caller argument expression, for template deduction.
    Type typeOfExpr(const Expr& e, const std::unordered_map<std::string, Type>& sym) {
        switch (e.kind) {
            case Expr::IntLit:   { Type t; t.base = e.wide ? Type::Long : Type::Int; return t; }
            case Expr::FloatLit: { Type t; t.base = e.wide ? Type::Double : Type::Float; return t; }
            case Expr::Ident:    { auto it = sym.find(e.str); return it != sym.end() ? it->second : Type{}; }
            case Expr::Index:    { Type b = e.args.empty() ? Type{} : typeOfExpr(*e.args[0], sym); if (b.ptr > 0) b.ptr--; return b; }
            case Expr::Cast:     return e.castType;
            case Expr::Unary:    return e.args.empty() ? Type{} : typeOfExpr(*e.args[0], sym);
            case Expr::Binary:   return e.args.size() == 2 ? typeOfExpr(*e.args[0], sym) : Type{};
            case Expr::Ternary:  return e.args.size() == 3 ? typeOfExpr(*e.args[1], sym) : Type{};
            default:             return Type{};
        }
    }

    // Resolve every template parameter of `tpl` for this call: explicit args first,
    // then deduce remaining type parameters from the call's argument types.
    bool resolveArgs(const Kernel& tpl, const Expr& call, const std::unordered_map<std::string, Type>& sym,
                     std::unordered_map<std::string, Type>& tmap, std::unordered_map<std::string, int64_t>& vmap) {
        if (call.targs.size() > tpl.tparams.size()) { fail("too many template arguments for '" + tpl.name + "'"); return false; }
        for (size_t i = 0; i < call.targs.size(); ++i) {
            const TemplateParam& tp = tpl.tparams[i];
            if (tp.isTypename) {
                if (!call.targs[i].isType) { fail("template parameter '" + tp.name + "' expects a type"); return false; }
                tmap[tp.name] = call.targs[i].type;
            } else {
                if (call.targs[i].isType) { fail("non-type template parameter '" + tp.name + "' expects a value"); return false; }
                vmap[tp.name] = call.targs[i].value;
            }
        }
        for (const auto& tp : tpl.tparams) {
            if (!tp.isTypename || tmap.count(tp.name)) continue;
            bool found = false;
            for (size_t pi = 0; pi < tpl.params.size() && pi < call.args.size(); ++pi) {
                if (tpl.params[pi].type.tparam != tp.name) continue;
                Type at = typeOfExpr(*call.args[pi], sym);
                at.ptr -= tpl.params[pi].type.ptr; if (at.ptr < 0) at.ptr = 0;
                at.isConst = false; at.tparam.clear();
                tmap[tp.name] = at; found = true; break;
            }
            if (!found) { fail("cannot deduce template parameter '" + tp.name + "' for '" + tpl.name + "'"); return false; }
        }
        for (const auto& tp : tpl.tparams)
            if (!tp.isTypename && !vmap.count(tp.name)) { fail("non-type template parameter '" + tp.name + "' needs an explicit argument"); return false; }
        return true;
    }

    // Rewrite one call to a template into a call to its (possibly new) instance.
    bool rewriteCall(Expr& call, const std::unordered_map<std::string, Type>& sym) {
        auto ti = templates.find(call.str);
        if (ti == templates.end()) return true;   // not a template call
        const Kernel& tpl = *ti->second;
        std::unordered_map<std::string, Type> tmap;
        std::unordered_map<std::string, int64_t> vmap;
        if (!resolveArgs(tpl, call, sym, tmap, vmap)) return false;
        std::string mangled = tpl.name;
        for (const auto& tp : tpl.tparams)
            mangled += "$" + (tp.isTypename ? mangleType(tmap[tp.name]) : std::to_string(vmap[tp.name]));
        if (!done.count(mangled)) {
            done[mangled] = true;
            auto inst = std::make_unique<Kernel>();
            inst->name = mangled;
            inst->returnType = tpl.returnType; substType(inst->returnType, tmap);
            inst->isGlobal = tpl.isGlobal; inst->isDevice = tpl.isDevice; inst->isHost = tpl.isHost;
            for (const auto& p : tpl.params) { Param np; np.name = p.name; np.type = p.type; substType(np.type, tmap); inst->params.push_back(std::move(np)); }
            for (const auto& s : tpl.body) { auto cs = cloneStmt(*s); substStmt(*cs, tmap, vmap); inst->body.push_back(std::move(cs)); }
            Kernel* raw = inst.get();
            instances.push_back(std::move(inst));
            worklist.push_back(raw);   // its body may call further templates
        }
        call.str = mangled; call.targs.clear();
        return true;
    }

    bool rewriteExpr(Expr& e, const std::unordered_map<std::string, Type>& sym) {
        for (auto& a : e.args) if (!rewriteExpr(*a, sym)) return false;
        if (e.kind == Expr::Call) return rewriteCall(e, sym);
        return true;
    }
    bool rewriteStmt(Stmt& s, std::unordered_map<std::string, Type>& sym) {
        if (s.expr && !rewriteExpr(*s.expr, sym)) return false;
        if (s.forInit && !rewriteStmt(*s.forInit, sym)) return false;
        if (s.forCond && !rewriteExpr(*s.forCond, sym)) return false;
        if (s.forIncr && !rewriteExpr(*s.forIncr, sym)) return false;
        for (auto& b : s.body) if (!rewriteStmt(*b, sym)) return false;
        for (auto& b : s.elseBody) if (!rewriteStmt(*b, sym)) return false;
        if (s.kind == Stmt::VarDecl && !s.name.empty()) sym[s.name] = s.type;   // track locals for deduction
        return true;
    }

    bool run() {
        for (auto& k : m.kernels) if (!k->tparams.empty()) templates[k->name] = k.get();
        if (templates.empty()) return true;
        for (auto& k : m.kernels) if (k->tparams.empty()) worklist.push_back(k.get());
        for (size_t i = 0; i < worklist.size(); ++i) {
            Kernel* caller = worklist[i];
            std::unordered_map<std::string, Type> sym;
            for (const auto& p : caller->params) sym[p.name] = p.type;
            for (auto& s : caller->body) if (!rewriteStmt(*s, sym)) return false;
        }
        for (auto& inst : instances) m.kernels.push_back(std::move(inst));
        m.kernels.erase(std::remove_if(m.kernels.begin(), m.kernels.end(),
                        [](const std::unique_ptr<Kernel>& k) { return !k->tparams.empty(); }),
                        m.kernels.end());
        return true;
    }
};

bool instantiateTemplates(Module& m, std::string& err) {
    Instantiator inst(m);
    if (!inst.run()) { err = inst.err.empty() ? "template instantiation failed" : inst.err; return false; }
    return true;
}

}  // namespace

ParseResult parse(const std::string& source) {
    ParseResult r;
    Parser p;
    p.toks = lex(source);
    r.module = p.parseModule();
    r.ok = !p.failed && r.module != nullptr;
    if (!r.ok) { r.error = p.err.empty() ? "parse failed" : p.err; return r; }
    std::string terr;
    if (!instantiateTemplates(*r.module, terr)) { r.ok = false; r.module.reset(); r.error = terr; }
    return r;
}

}  // namespace frontend
}  // namespace compiler
}  // namespace vgre
