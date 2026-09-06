// PTX code generation — see include/vgre/compiler/frontend/codegen.h.
//
// A single non-SSA pass: each local variable lives in one virtual register that
// is mutated in place (the interpreter's registers are mutable), so control flow
// needs no phi nodes. Types are inferred bottom-up as we emit. Only the
// supported subset is accepted; anything else fails with a located error.

#include "vgre/compiler/frontend/codegen.h"

#include "vgre/compiler/frontend/parser.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>

namespace vgre {
namespace compiler {
namespace frontend {

namespace {

// A computed value: the register holding it and its type.
struct Val {
    std::string reg;
    Type type;
};

// PTX register classes.
enum class RC { R32, F32, RD64, Pred };

RC classOf(const Type& t) {
    if (t.isPointer()) return RC::RD64;
    if (t.isFloating()) return RC::F32;
    return RC::R32;  // bool/char/short/int/long collapse to 32-bit here
}

struct Codegen {
    const Kernel& k;
    std::string body;                 // instruction stream (built first)
    int nR = 0, nF = 0, nRd = 0, nP = 0, nLbl = 0;
    std::unordered_map<std::string, Val> vars;  // name -> value (single mutable reg)
    bool failed = false;
    std::string err;
    int line = 0, col = 0;

    explicit Codegen(const Kernel& kernel) : k(kernel) {}

    void fail(const std::string& m) {
        if (failed) return;
        failed = true;
        err = std::to_string(line) + ":" + std::to_string(col) + ": " + m;
    }

    std::string fresh(RC rc) {
        switch (rc) {
            case RC::R32:  return "%r"  + std::to_string(nR++);
            case RC::F32:  return "%f"  + std::to_string(nF++);
            case RC::RD64: return "%rd" + std::to_string(nRd++);
            case RC::Pred: return "%p"  + std::to_string(nP++);
        }
        return "%r0";
    }
    std::string label() { return "$L" + std::to_string(nLbl++); }

    void emit(const std::string& s) { body += "\t"; body += s; body += "\n"; }
    void emitLabel(const std::string& l) { body += l; body += ":\n"; }

    static Type intType() { Type t; t.base = Type::Int; return t; }
    static Type floatType() { Type t; t.base = Type::Float; return t; }

    // PTX ld/st/param type suffix for a scalar (pointee) type.
    static const char* memSuffix(const Type& t) { return t.isFloating() ? "f32" : "u32"; }

    // Format a float32 immediate as PTX hex (0f%08X) — the interpreter's format.
    static std::string f32imm(double d) {
        float f = static_cast<float>(d);
        uint32_t bits;
        std::memcpy(&bits, &f, 4);
        char buf[16];
        std::snprintf(buf, sizeof(buf), "0f%08X", bits);
        return buf;
    }

    // Coerce `v` to `want` (int<->float only). Returns the coerced Val.
    Val coerce(const Val& v, const Type& want) {
        if (v.type.isFloating() == want.isFloating() && v.type.isPointer() == want.isPointer())
            return v;
        if (want.isFloating() && !v.type.isFloating()) {          // int -> float
            std::string d = fresh(RC::F32);
            emit("cvt.rn.f32.s32 " + d + ", " + v.reg + ";");
            return {d, floatType()};
        }
        if (!want.isFloating() && v.type.isFloating()) {          // float -> int
            std::string d = fresh(RC::R32);
            emit("cvt.rzi.s32.f32 " + d + ", " + v.reg + ";");
            return {d, intType()};
        }
        return v;
    }

    // ── Expressions ─────────────────────────────────────────────────────────────
    bool isBuiltinObj(const std::string& n) {
        return n == "threadIdx" || n == "blockIdx" || n == "blockDim" || n == "gridDim";
    }
    static const char* sregOf(const std::string& obj) {
        if (obj == "threadIdx") return "tid";
        if (obj == "blockIdx")  return "ctaid";
        if (obj == "blockDim")  return "ntid";
        return "nctaid";  // gridDim
    }

    Val emitExpr(const Expr& e) {
        line = e.line; col = e.col;
        switch (e.kind) {
            case Expr::IntLit: {
                std::string d = fresh(RC::R32);
                emit("mov.u32 " + d + ", " + std::to_string(e.ival) + ";");
                return {d, intType()};
            }
            case Expr::FloatLit: {
                std::string d = fresh(RC::F32);
                emit("mov.f32 " + d + ", " + f32imm(e.fval) + ";");
                return {d, floatType()};
            }
            case Expr::Ident: {
                auto it = vars.find(e.str);
                if (it == vars.end()) { fail("use of undeclared identifier '" + e.str + "'"); return {}; }
                return it->second;
            }
            case Expr::Member: {
                const Expr& obj = *e.args[0];
                if (obj.kind == Expr::Ident && isBuiltinObj(obj.str) &&
                    (e.str == "x" || e.str == "y" || e.str == "z")) {
                    std::string d = fresh(RC::R32);
                    emit("mov.u32 " + d + ", %" + sregOf(obj.str) + "." + e.str + ";");
                    return {d, intType()};
                }
                fail("unsupported member access '." + e.str + "'");
                return {};
            }
            case Expr::Index: return emitLoad(e);
            case Expr::Unary: return emitUnary(e);
            case Expr::Binary: return emitBinary(e);
            case Expr::Assign: return emitAssign(e);
            case Expr::Call: return emitCall(e);
        }
        fail("unsupported expression");
        return {};
    }

    // Compute the byte address of base[index] into a fresh %rd; sets pointee.
    std::string emitAddress(const Expr& index, Type& pointee) {
        const Expr& base = *index.args[0];
        Val b = emitExpr(base);
        if (failed) return "";
        if (!b.type.isPointer()) { fail("indexing a non-pointer"); return ""; }
        pointee = b.type; pointee.ptr -= 1;
        Val idx = emitExpr(*index.args[1]);
        if (failed) return "";
        idx = coerce(idx, intType());
        std::string off = fresh(RC::RD64);
        emit("mul.wide.s32 " + off + ", " + idx.reg + ", " + std::to_string(pointee.elemBytes()) + ";");
        std::string addr = fresh(RC::RD64);
        emit("add.s64 " + addr + ", " + b.reg + ", " + off + ";");
        return addr;
    }

    Val emitLoad(const Expr& index) {
        Type pointee;
        std::string addr = emitAddress(index, pointee);
        if (failed) return {};
        RC rc = classOf(pointee);
        std::string d = fresh(rc);
        emit(std::string("ld.global.") + memSuffix(pointee) + " " + d + ", [" + addr + "];");
        return {d, pointee};
    }

    Val emitUnary(const Expr& e) {
        if (e.str == "+") return emitExpr(*e.args[0]);
        Val v = emitExpr(*e.args[0]);
        if (failed) return {};
        if (e.str == "-") {
            std::string d = fresh(classOf(v.type));
            emit(std::string("neg.") + (v.type.isFloating() ? "f32 " : "s32 ") + d + ", " + v.reg + ";");
            return {d, v.type};
        }
        if (e.str == "!") {
            std::string p = fresh(RC::Pred), d = fresh(RC::R32);
            emit("setp.eq.s32 " + p + ", " + v.reg + ", 0;");
            emit("mov.u32 " + d + ", 0;");
            emit("@" + p + " mov.u32 " + d + ", 1;");
            return {d, intType()};
        }
        fail("unsupported unary operator '" + e.str + "'");
        return {};
    }

    // Materialize a comparison as an int 0/1 (predicated mov, no selp needed).
    Val emitCompare(const std::string& op, Val a, Val b) {
        bool fp = a.type.isFloating() || b.type.isFloating();
        Type ct = fp ? floatType() : intType();
        a = coerce(a, ct); b = coerce(b, ct);
        const char* cc = op == "<" ? "lt" : op == "<=" ? "le" : op == ">" ? "gt" :
                         op == ">=" ? "ge" : op == "==" ? "eq" : "ne";
        std::string p = fresh(RC::Pred), d = fresh(RC::R32);
        emit(std::string("setp.") + cc + (fp ? ".f32 " : ".s32 ") + p + ", " + a.reg + ", " + b.reg + ";");
        emit("mov.u32 " + d + ", 0;");
        emit("@" + p + " mov.u32 " + d + ", 1;");
        return {d, intType()};
    }

    Val emitBinary(const Expr& e) {
        const std::string& op = e.str;
        if (op == "<" || op == "<=" || op == ">" || op == ">=" || op == "==" || op == "!=") {
            Val a = emitExpr(*e.args[0]); if (failed) return {};
            Val b = emitExpr(*e.args[1]); if (failed) return {};
            return emitCompare(op, a, b);
        }
        Val a = emitExpr(*e.args[0]); if (failed) return {};
        Val b = emitExpr(*e.args[1]); if (failed) return {};

        if (op == "&&" || op == "||") {
            // Non-short-circuit (subset is side-effect-free in conditions).
            const char* ins = op == "&&" ? "and.b32 " : "or.b32 ";
            std::string d = fresh(RC::R32);
            emit(std::string(ins) + d + ", " + a.reg + ", " + b.reg + ";");
            return {d, intType()};
        }

        bool fp = a.type.isFloating() || b.type.isFloating();
        Type ct = fp ? floatType() : intType();
        a = coerce(a, ct); b = coerce(b, ct);
        std::string d = fresh(classOf(ct));
        std::string ins;
        if (op == "+") ins = fp ? "add.f32 " : "add.s32 ";
        else if (op == "-") ins = fp ? "sub.f32 " : "sub.s32 ";
        else if (op == "*") ins = fp ? "mul.f32 " : "mul.lo.s32 ";
        else if (op == "/") ins = fp ? "div.rn.f32 " : "div.s32 ";
        else if (op == "%") { if (fp) { fail("'%' on floating type"); return {}; } ins = "rem.s32 "; }
        else if (op == "&") ins = "and.b32 ";
        else if (op == "|") ins = "or.b32 ";
        else if (op == "^") ins = "xor.b32 ";
        else if (op == "<<") ins = "shl.b32 ";
        else if (op == ">>") ins = "shr.s32 ";
        else { fail("unsupported binary operator '" + op + "'"); return {}; }
        emit(ins + d + ", " + a.reg + ", " + b.reg + ";");
        return {d, ct};
    }

    Val emitAssign(const Expr& e) {
        const Expr& lhs = *e.args[0];
        const std::string& op = e.str;

        // Compute the value to store: rhs, or (lhs <binop> rhs) for compound.
        auto computeRhs = [&](const Val& lhsVal) -> Val {
            Val rhs = emitExpr(*e.args[1]);
            if (failed) return {};
            if (op == "=") return rhs;
            std::string bop(1, op[0]);  // "+=" -> "+"
            Expr fake; fake.kind = Expr::Binary; fake.str = bop;
            // Reuse emitBinary's arithmetic by hand to avoid re-emitting lhs load.
            bool fp = lhsVal.type.isFloating() || rhs.type.isFloating();
            Type ct = fp ? floatType() : intType();
            Val a = coerce(lhsVal, ct), b = coerce(rhs, ct);
            std::string d = fresh(classOf(ct)), ins;
            if (bop == "+") ins = fp ? "add.f32 " : "add.s32 ";
            else if (bop == "-") ins = fp ? "sub.f32 " : "sub.s32 ";
            else if (bop == "*") ins = fp ? "mul.f32 " : "mul.lo.s32 ";
            else if (bop == "/") ins = fp ? "div.rn.f32 " : "div.s32 ";
            else { ins = "rem.s32 "; }
            emit(ins + d + ", " + a.reg + ", " + b.reg + ";");
            return {d, ct};
        };

        if (lhs.kind == Expr::Ident) {
            auto it = vars.find(lhs.str);
            if (it == vars.end()) { line = lhs.line; col = lhs.col; fail("assignment to undeclared '" + lhs.str + "'"); return {}; }
            Val& var = it->second;
            Val rhs = computeRhs(var);
            if (failed) return {};
            rhs = coerce(rhs, var.type);
            const char* mov = var.type.isFloating() ? "mov.f32 " : (var.type.isPointer() ? "mov.u64 " : "mov.u32 ");
            emit(std::string(mov) + var.reg + ", " + rhs.reg + ";");
            return var;
        }
        if (lhs.kind == Expr::Index) {
            // For compound on memory we'd need the current value; support '=' now.
            Type pointee;
            if (op != "=") {
                // load current, combine, store
                Val cur = emitLoad(lhs);
                if (failed) return {};
                Val rhs = computeRhs(cur);
                if (failed) return {};
                std::string addr = emitAddress(lhs, pointee);
                if (failed) return {};
                rhs = coerce(rhs, pointee);
                emit(std::string("st.global.") + memSuffix(pointee) + " [" + addr + "], " + rhs.reg + ";");
                return rhs;
            }
            std::string addr = emitAddress(lhs, pointee);
            if (failed) return {};
            Val rhs = emitExpr(*e.args[1]);
            if (failed) return {};
            rhs = coerce(rhs, pointee);
            emit(std::string("st.global.") + memSuffix(pointee) + " [" + addr + "], " + rhs.reg + ";");
            return rhs;
        }
        fail("invalid assignment target");
        return {};
    }

    Val emitCall(const Expr& e) {
        if (e.str == "__syncthreads" && e.args.empty()) {
            emit("bar.sync 0;");
            return {};
        }
        // A few common device math intrinsics map to PTX approximations.
        if (e.args.size() == 1) {
            static const std::unordered_map<std::string, const char*> unary = {
                {"sqrtf", "sqrt.rn.f32"}, {"__expf", "ex2.approx.f32"},
            };
            auto it = unary.find(e.str);
            if (it != unary.end()) {
                Val a = coerce(emitExpr(*e.args[0]), floatType());
                if (failed) return {};
                std::string d = fresh(RC::F32);
                emit(std::string(it->second) + " " + d + ", " + a.reg + ";");
                return {d, floatType()};
            }
        }
        fail("unsupported call to '" + e.str + "'");
        return {};
    }

    // Branch to `lbl` when `cond` is false.
    void emitCondBranchFalse(const Expr& cond, const std::string& lbl) {
        line = cond.line; col = cond.col;
        if (cond.kind == Expr::Binary &&
            (cond.str == "<" || cond.str == "<=" || cond.str == ">" ||
             cond.str == ">=" || cond.str == "==" || cond.str == "!=")) {
            Val a = emitExpr(*cond.args[0]); if (failed) return;
            Val b = emitExpr(*cond.args[1]); if (failed) return;
            bool fp = a.type.isFloating() || b.type.isFloating();
            Type ct = fp ? floatType() : intType();
            a = coerce(a, ct); b = coerce(b, ct);
            // Inverted predicate → branch when the original condition is false.
            const std::string& o = cond.str;
            const char* inv = o == "<" ? "ge" : o == "<=" ? "gt" : o == ">" ? "le" :
                              o == ">=" ? "lt" : o == "==" ? "ne" : "eq";
            std::string p = fresh(RC::Pred);
            emit(std::string("setp.") + inv + (fp ? ".f32 " : ".s32 ") + p + ", " + a.reg + ", " + b.reg + ";");
            emit("@" + p + " bra " + lbl + ";");
            return;
        }
        Val c = emitExpr(cond);
        if (failed) return;
        std::string p = fresh(RC::Pred);
        emit("setp.eq.s32 " + p + ", " + c.reg + ", 0;");
        emit("@" + p + " bra " + lbl + ";");
    }

    // ── Statements ──────────────────────────────────────────────────────────────
    void emitStmt(const Stmt& s) {
        line = s.line; col = s.col;
        switch (s.kind) {
            case Stmt::VarDecl: {
                Val v; v.type = s.type; v.reg = fresh(classOf(s.type));
                vars[s.name] = v;
                if (s.expr) {
                    Val init = emitExpr(*s.expr);
                    if (failed) return;
                    init = coerce(init, s.type);
                    const char* mov = s.type.isFloating() ? "mov.f32 " : (s.type.isPointer() ? "mov.u64 " : "mov.u32 ");
                    emit(std::string(mov) + v.reg + ", " + init.reg + ";");
                }
                return;
            }
            case Stmt::ExprStmt: if (s.expr) emitExpr(*s.expr); return;
            case Stmt::Block: for (auto& st : s.body) { emitStmt(*st); if (failed) return; } return;
            case Stmt::Return: emit("ret;"); return;
            case Stmt::Empty: return;
            case Stmt::If: {
                std::string elseL = label(), endL = label();
                emitCondBranchFalse(*s.expr, s.elseBody.empty() ? endL : elseL);
                if (failed) return;
                for (auto& st : s.body) { emitStmt(*st); if (failed) return; }
                if (!s.elseBody.empty()) {
                    emit("bra " + endL + ";");
                    emitLabel(elseL);
                    for (auto& st : s.elseBody) { emitStmt(*st); if (failed) return; }
                }
                emitLabel(endL);
                return;
            }
            case Stmt::While: {
                std::string top = label(), end = label();
                emitLabel(top);
                emitCondBranchFalse(*s.expr, end);
                if (failed) return;
                for (auto& st : s.body) { emitStmt(*st); if (failed) return; }
                emit("bra " + top + ";");
                emitLabel(end);
                return;
            }
            case Stmt::For: {
                if (s.forInit) { emitStmt(*s.forInit); if (failed) return; }
                std::string top = label(), end = label();
                emitLabel(top);
                if (s.forCond) { emitCondBranchFalse(*s.forCond, end); if (failed) return; }
                for (auto& st : s.body) { emitStmt(*st); if (failed) return; }
                if (s.forIncr) { emitExpr(*s.forIncr); if (failed) return; }
                emit("bra " + top + ";");
                emitLabel(end);
                return;
            }
        }
    }

    // ── Assembly ────────────────────────────────────────────────────────────────
    static const char* paramSuffix(const Type& t) {
        if (t.isPointer()) return "u64";
        if (t.isFloating()) return "f32";
        return "u32";
    }

    std::string run() {
        // Load parameters into registers / global pointers.
        for (const Param& p : k.params) {
            if (p.name.empty()) continue;  // unnamed param: nothing binds to it
            if (p.type.isPointer()) {
                std::string raw = fresh(RC::RD64), gbl = fresh(RC::RD64);
                emit("ld.param.u64 " + raw + ", [" + p.name + "];");
                emit("cvta.to.global.u64 " + gbl + ", " + raw + ";");
                vars[p.name] = {gbl, p.type};
            } else if (p.type.isFloating()) {
                std::string r = fresh(RC::F32);
                emit("ld.param.f32 " + r + ", [" + p.name + "];");
                vars[p.name] = {r, p.type};
            } else {
                std::string r = fresh(RC::R32);
                emit("ld.param.u32 " + r + ", [" + p.name + "];");
                vars[p.name] = {r, p.type};
            }
        }
        for (auto& st : k.body) { emitStmt(*st); if (failed) return ""; }
        emit("ret;");

        // Assemble: header + signature + reg decls + body.
        std::string out;
        out += ".version 7.0\n.target sm_52\n.address_size 64\n\n";
        out += ".visible .entry " + k.name + "(\n";
        for (size_t i = 0; i < k.params.size(); ++i) {
            out += "\t.param ." + std::string(paramSuffix(k.params[i].type)) + " " +
                   (k.params[i].name.empty() ? ("_arg" + std::to_string(i)) : k.params[i].name);
            if (i + 1 < k.params.size()) out += ",";
            out += "\n";
        }
        out += ")\n{\n";
        if (nP  > 0) out += "\t.reg .pred %p<" + std::to_string(nP)  + ">;\n";
        if (nR  > 0) out += "\t.reg .b32 %r<"  + std::to_string(nR)  + ">;\n";
        if (nF  > 0) out += "\t.reg .f32 %f<"  + std::to_string(nF)  + ">;\n";
        if (nRd > 0) out += "\t.reg .b64 %rd<" + std::to_string(nRd) + ">;\n";
        out += body;
        out += "}\n";
        return out;
    }
};

}  // namespace

CodegenResult generatePtx(const Kernel& kernel) {
    CodegenResult r;
    Codegen cg(kernel);
    std::string ptx = cg.run();
    if (cg.failed) { r.ok = false; r.error = cg.err; return r; }
    r.ptx = std::move(ptx);
    r.ok = true;
    return r;
}

CodegenResult compileToPtx(const std::string& source, const std::string& name) {
    CodegenResult r;
    ParseResult pr = parse(source);
    if (!pr.ok) { r.error = pr.error; return r; }
    const Kernel* target = nullptr;
    for (auto& kp : pr.module->kernels) {
        if (name.empty() || kp->name == name) { target = kp.get(); break; }
    }
    if (!target) { r.error = "kernel not found: " + (name.empty() ? std::string("<first>") : name); return r; }
    return generatePtx(*target);
}

}  // namespace frontend
}  // namespace compiler
}  // namespace vgre
