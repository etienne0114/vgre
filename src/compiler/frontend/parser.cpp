// Recursive-descent parser — see include/vgre/compiler/frontend/parser.h.

#include "vgre/compiler/frontend/parser.h"

#include "vgre/compiler/frontend/lexer.h"

#include <cstdlib>
#include <string>
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

bool isAssignOp(TokenKind k) {
    return k == TokenKind::Assign || k == TokenKind::PlusEq || k == TokenKind::MinusEq ||
           k == TokenKind::StarEq || k == TokenKind::SlashEq || k == TokenKind::PercentEq;
}

bool isTypeStart(TokenKind k) {
    switch (k) {
        case TokenKind::KwConst: case TokenKind::KwUnsigned: case TokenKind::KwSigned:
        case TokenKind::KwVoid: case TokenKind::KwBool: case TokenKind::KwChar:
        case TokenKind::KwShort: case TokenKind::KwInt: case TokenKind::KwLong:
        case TokenKind::KwFloat: case TokenKind::KwDouble:
            return true;
        default: return false;
    }
}

struct Parser {
    std::vector<Token> toks;
    size_t pos = 0;
    bool failed = false;
    std::string err;
    Module* mod_ = nullptr;  // module being built (so parseType can resolve struct names)

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

    ExprPtr mkExpr(Expr::Kind k) {
        auto e = std::make_unique<Expr>();
        e->kind = k; e->line = cur().line; e->col = cur().col; return e;
    }

    // ── Types ─────────────────────────────────────────────────────────────────
    bool parseType(Type& out) {
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
                while (accept(TokenKind::KwConst) || accept(TokenKind::KwRestrict)) { /* qualifier */ }
            }
            return true;
        }
        if (!isTypeStart(kind())) return false;
        out = Type{};
        // qualifiers / sign, in any leading order
        for (;;) {
            if (accept(TokenKind::KwConst)) { out.isConst = true; continue; }
            if (accept(TokenKind::KwUnsigned)) { out.isUnsigned = true; continue; }
            if (accept(TokenKind::KwSigned)) { out.isUnsigned = false; continue; }
            break;
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
            default:
                // "unsigned"/"const" alone implies int.
                out.base = Type::Int; break;
        }
        // pointer stars with trailing const/__restrict__ qualifiers
        while (accept(TokenKind::Star)) {
            out.ptr++;
            while (accept(TokenKind::KwConst) || accept(TokenKind::KwRestrict)) { /* qualifier */ }
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

    ExprPtr parsePostfix() {
        ExprPtr e = parsePrimary();
        if (!e || failed) return e;
        for (;;) {
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
            case TokenKind::KwReturn:   return parseReturn();
            case TokenKind::Semicolon:  { auto s = mkStmt(Stmt::Empty); advance(); return s; }
            default: break;
        }
        if (isTypeStart(kind()) || at(TokenKind::KwShared)) return parseVarDecl();
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
        auto s = mkStmt(Stmt::VarDecl);
        if (accept(TokenKind::KwShared)) s->isShared = true;   // __shared__ [type] name[N];
        if (!parseType(s->type)) { fail("expected a type"); return nullptr; }
        if (!at(TokenKind::Identifier)) { fail("expected a variable name"); return nullptr; }
        s->name = advance().text;
        while (accept(TokenKind::LBracket)) {                  // array declarator name[N][M]…
            if (!at(TokenKind::IntLiteral)) { fail("expected an array size"); return nullptr; }
            int dim = static_cast<int>(std::strtoll(advance().text.c_str(), nullptr, 0));
            expect(TokenKind::RBracket, "']'");
            if (dim <= 0) { fail("array size must be positive"); return nullptr; }
            s->arrayDims.push_back(dim);
        }
        if (!s->arrayDims.empty()) {
            s->arraySize = 1;
            for (int d : s->arrayDims) s->arraySize *= d;      // total element count
        }
        if (accept(TokenKind::Assign)) {
            s->expr = parseExpr();
            if (!s->expr) return nullptr;
        }
        expect(TokenKind::Semicolon, "';'");
        return failed ? nullptr : std::move(s);
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

    StmtPtr parseFor() {
        auto s = mkStmt(Stmt::For);
        advance();  // for
        expect(TokenKind::LParen, "'('");
        // init: declaration, expr-stmt, or empty
        if (accept(TokenKind::Semicolon)) {
            // no init
        } else if (isTypeStart(kind())) {
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
    std::unique_ptr<Kernel> parseKernel() {
        auto k = std::make_unique<Kernel>();
        // Optional linkage: extern "C"
        if (accept(TokenKind::KwExtern)) accept(TokenKind::StringLiteral);
        // Qualifiers: __global__ / __device__ (order-insensitive with the return type).
        for (;;) {
            if (accept(TokenKind::KwGlobal)) { k->isGlobal = true; continue; }
            if (accept(TokenKind::KwDevice)) { continue; }
            break;
        }
        Type ret;
        if (!parseType(ret)) { fail("expected a return type"); return nullptr; }
        k->returnType = ret;
        if (!at(TokenKind::Identifier)) { fail("expected a kernel name"); return nullptr; }
        k->name = advance().text;
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

}  // namespace

ParseResult parse(const std::string& source) {
    ParseResult r;
    Parser p;
    p.toks = lex(source);
    r.module = p.parseModule();
    r.ok = !p.failed && r.module != nullptr;
    if (!r.ok) r.error = p.err.empty() ? "parse failed" : p.err;
    return r;
}

}  // namespace frontend
}  // namespace compiler
}  // namespace vgre
