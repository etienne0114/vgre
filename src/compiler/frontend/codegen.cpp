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
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace vgre {
namespace compiler {
namespace frontend {

namespace {

// Where a value/pointer lives. Value = a plain register; Global = a 64-bit
// global address (param pointer after cvta); Shared = a __shared__ array,
// addressed by its PTX name with 32-bit offsets (ld.shared/st.shared); Local = a
// per-thread array (register-scratch), same 32-bit symbol addressing but in the
// .local space (ld.local/st.local).
enum class Space { Value, Global, Shared, Local, ParamStruct };

// A computed value: the register holding it and its type (+ memory space for
// pointers/arrays). For a by-value struct param, space==ParamStruct and
// `paramName` is the PTX .param byte-array symbol (members read via ld.param).
struct Val {
    std::string reg;
    Type type;
    Space space = Space::Value;
    std::string sharedName;   // Shared: the PTX .shared symbol name
    std::string localName;    // Local: the PTX .local symbol name
    std::string paramName;    // ParamStruct: the .param symbol name
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
    std::string sharedDecls;          // .shared declarations (emitted in the decl section)
    std::string localDecls;           // .local declarations (emitted in the decl section)
    int nR = 0, nF = 0, nRd = 0, nP = 0, nLbl = 0;
    std::unordered_map<std::string, Val> vars;  // name -> value (single mutable reg)
    std::unordered_map<std::string, std::vector<int>> arrayDims_;  // declared array -> dim sizes
    bool failed = false;
    std::string err;
    int line = 0, col = 0;

    // __device__ helper functions available for inlining (name -> Kernel), and
    // the active inline-call context stack (return register + end label + return
    // type) so a `return` inside an inlined body branches to the call site.
    const std::unordered_map<std::string, const Kernel*>* deviceFns_ = nullptr;
    const Module* mod_ = nullptr;     // for struct layouts (findStruct)
    struct InlineCtx { std::string retReg; std::string endLabel; Type retType; };
    std::vector<InlineCtx> inlineCtx_;
    std::set<std::string> inlining_;  // recursion guard

    explicit Codegen(const Kernel& kernel,
                     const std::unordered_map<std::string, const Kernel*>* deviceFns = nullptr,
                     const Module* mod = nullptr)
        : k(kernel), deviceFns_(deviceFns), mod_(mod) {}

    // Struct byte-size from the module's struct table (0 if unknown).
    int structSize(const std::string& name) const {
        const StructDef* d = mod_ ? mod_->findStruct(name) : nullptr;
        return d ? d->size : 0;
    }

    static const char* movFor(const Type& t) {
        return t.isFloating() ? "mov.f32 " : (t.isPointer() ? "mov.u64 " : "mov.u32 ");
    }

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

    void emit(const std::string& s) {
        // Defensive cap: a codegen bug (e.g. runaway inlining) must fail cleanly,
        // never OOM. 16 MB of PTX text is far beyond any real kernel.
        if (body.size() > (16u << 20)) {
            if (!failed) fail("codegen output exceeded 16MB — aborting (last line: " + s + ")");
            return;
        }
        body += "\t"; body += s; body += "\n";
    }
    void emitLabel(const std::string& l) { body += l; body += ":\n"; }

    static Type intType() { Type t; t.base = Type::Int; return t; }
    static Type floatType() { Type t; t.base = Type::Float; return t; }

    // The PTX-interpreter codegen tier is 32-bit int / f32 float. Reject 64-bit
    // scalar types (double/long) rather than silently truncating them — such
    // kernels correctly use the compiled tier (int64/double) or the JIT.
    bool ensureSupported(const Type& t) {
        if (t.base == Type::Double) { fail("'double' unsupported by the interpreter codegen tier — use the compiled tier or the JIT"); return false; }
        if (t.base == Type::Long)   { fail("'long'/64-bit int unsupported by the interpreter codegen tier — use the compiled tier or the JIT"); return false; }
        return true;
    }

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
                // Struct member read: obj is a by-value struct param → load the
                // member scalar from its .param byte array at the member offset.
                if (obj.kind == Expr::Ident) {
                    auto it = vars.find(obj.str);
                    if (it != vars.end() && it->second.space == Space::ParamStruct) {
                        const StructDef* def = mod_ ? mod_->findStruct(it->second.type.structName) : nullptr;
                        const StructMember* m = def ? def->find(e.str) : nullptr;
                        if (!m) { fail("no member '." + e.str + "' in struct '" + it->second.type.structName + "'"); return {}; }
                        std::string d = fresh(classOf(m->type));
                        emit("ld.param." + std::string(memSuffix(m->type)) + " " + d + ", [" +
                             it->second.paramName + "+" + std::to_string(m->offset) + "];");
                        return {d, m->type};
                    }
                }
                fail("unsupported member access '." + e.str + "'");
                return {};
            }
            case Expr::Index: return emitLoad(e);
            case Expr::Unary: return emitUnary(e);
            case Expr::Binary: return emitBinary(e);
            case Expr::Assign: return emitAssign(e);
            case Expr::Call: return emitCall(e);
            case Expr::Cast: return emitCast(e);
            case Expr::Ternary: return emitTernary(e);
        }
        fail("unsupported expression");
        return {};
    }

    // Static "is this expression floating-typed?" — needed to pick the result
    // register class for a ternary before emitting its branches.
    bool isFloatExpr(const Expr& e) {
        switch (e.kind) {
            case Expr::FloatLit: return true;
            case Expr::IntLit:   return false;
            case Expr::Ident: { auto it = vars.find(e.str); return it != vars.end() && it->second.type.isFloating(); }
            case Expr::Member:   return false;  // threadIdx/blockIdx/… are integers
            case Expr::Index:    return const_cast<Codegen*>(this)->indexPointee(e).isFloating();
            case Expr::Cast:     return e.castType.isFloating();
            case Expr::Unary:    return e.str == "!" ? false : isFloatExpr(*e.args[0]);
            case Expr::Binary: {
                const std::string& o = e.str;
                if (o == "<" || o == "<=" || o == ">" || o == ">=" || o == "==" || o == "!=" ||
                    o == "&&" || o == "||" || o == "&" || o == "|" || o == "^" ||
                    o == "<<" || o == ">>" || o == "%") return false;
                return isFloatExpr(*e.args[0]) || isFloatExpr(*e.args[1]);
            }
            case Expr::Assign:   return isFloatExpr(*e.args[1]);
            case Expr::Ternary:  return isFloatExpr(*e.args[1]) || isFloatExpr(*e.args[2]);
            case Expr::Call: {
                const std::string& fn = e.str;
                if (fn == "min" || fn == "max") return !e.args.empty() && isFloatExpr(*e.args[0]);
                return true;  // the float math intrinsics (sqrtf, fmaf, …) return float
            }
        }
        return false;
    }

    Val emitCast(const Expr& e) {
        if (!ensureSupported(e.castType)) return {};
        Val v = emitExpr(*e.args[0]);
        if (failed) return {};
        Val r = coerce(v, e.castType);
        r.type = e.castType;
        if (e.castType.isPointer()) r.space = v.space;
        return r;
    }

    Val emitTernary(const Expr& e) {
        Type rt = (isFloatExpr(*e.args[1]) || isFloatExpr(*e.args[2])) ? floatType() : intType();
        std::string res = fresh(classOf(rt));
        std::string elseL = label(), endL = label();
        emitCondBranchFalse(*e.args[0], elseL);
        if (failed) return {};
        Val t = coerce(emitExpr(*e.args[1]), rt);
        if (failed) return {};
        emit(std::string(rt.isFloating() ? "mov.f32 " : "mov.u32 ") + res + ", " + t.reg + ";");
        emit("bra " + endL + ";");
        emitLabel(elseL);
        Val f = coerce(emitExpr(*e.args[2]), rt);
        if (failed) return {};
        emit(std::string(rt.isFloating() ? "mov.f32 " : "mov.u32 ") + res + ", " + f.reg + ";");
        emitLabel(endL);
        return {res, rt};
    }

    // A computed element address: the register holding it, the pointee type, and
    // whether it is in shared memory (32-bit addressing, ld/st.shared).
    struct Addr {
        std::string reg;
        Type pointee;
        bool shared = false;
        bool local = false;
    };

    // The PTX memory-space prefix for a computed address ("shared.", "local." or
    // "global.") — used to build ld./st. mnemonics.
    static std::string spacePrefix(const Addr& a) {
        return a.shared ? "shared." : a.local ? "local." : "global.";
    }

    // The element type of an index expression's base, without emitting code
    // (used to coerce the stored value before emitting the store). In the
    // supported subset the base is an Ident naming a pointer/array variable.
    // Peel nested Index nodes (`A[i][j]` → Index(Index(A,i),j)). Returns the
    // innermost base expr and fills `idxs` with the index expressions in
    // outer-dimension-first order ([i, j] for A[i][j]).
    static const Expr* peelIndex(const Expr& index, std::vector<const Expr*>& idxs) {
        std::vector<const Expr*> rev;
        const Expr* cur = &index;
        while (cur->kind == Expr::Index) { rev.push_back(cur->args[1].get()); cur = cur->args[0].get(); }
        for (auto it = rev.rbegin(); it != rev.rend(); ++it) idxs.push_back(*it);
        return cur;
    }

    Type indexPointee(const Expr& index) {
        std::vector<const Expr*> idxs;
        const Expr* root = peelIndex(index, idxs);
        if (root->kind == Expr::Ident) {
            auto it = vars.find(root->str);
            if (it != vars.end() && it->second.type.isPointer()) {
                Type t = it->second.type; t.ptr -= 1; return t;
            }
        }
        return intType();
    }

    // Compute the address of an index expression, flattening multi-dimensional
    // declared arrays (row-major: A[i][j] → A[i*cols + j]). Global uses 64-bit
    // addressing; __shared__/.local arrays use 32-bit offsets from the symbol.
    Addr emitAddress(const Expr& index) {
        Addr a;
        std::vector<const Expr*> idxs;
        const Expr* root = peelIndex(index, idxs);
        const bool isArray = root->kind == Expr::Ident && arrayDims_.count(root->str);

        Val b;
        if (isArray) {
            b = vars[root->str];
            const std::vector<int>& dims = arrayDims_[root->str];
            if (idxs.size() != dims.size()) {
                fail("array '" + root->str + "' expects " + std::to_string(dims.size()) +
                     " index(es), got " + std::to_string(idxs.size()));
                return a;
            }
        } else {
            if (idxs.size() != 1) { fail("multi-dimensional indexing requires a declared array"); return a; }
            b = emitExpr(*root);
            if (failed) return a;
            if (!b.type.isPointer()) { fail("indexing a non-pointer"); return a; }
        }
        a.pointee = b.type; a.pointee.ptr -= 1;

        // Fold the indices into one element offset, row-major.
        Val flat = coerce(emitExpr(*idxs[0]), intType());
        if (failed) return a;
        if (isArray) {
            const std::vector<int>& dims = arrayDims_[root->str];
            for (size_t d = 1; d < idxs.size(); ++d) {
                std::string m = fresh(RC::R32);
                emit("mul.lo.s32 " + m + ", " + flat.reg + ", " + std::to_string(dims[d]) + ";");
                Val ik = coerce(emitExpr(*idxs[d]), intType());
                if (failed) return a;
                std::string s = fresh(RC::R32);
                emit("add.s32 " + s + ", " + m + ", " + ik.reg + ";");
                flat = {s, intType()};
            }
        }

        const int elem = a.pointee.elemBytes();
        if (b.space == Space::Shared || b.space == Space::Local) {
            (b.space == Space::Shared ? a.shared : a.local) = true;
            const std::string& sym = (b.space == Space::Shared) ? b.sharedName : b.localName;
            std::string base = fresh(RC::R32), off = fresh(RC::R32), addr = fresh(RC::R32);
            emit("mov.u32 " + base + ", " + sym + ";");
            emit("mul.lo.s32 " + off + ", " + flat.reg + ", " + std::to_string(elem) + ";");
            emit("add.s32 " + addr + ", " + base + ", " + off + ";");
            a.reg = addr;
        } else {
            std::string off = fresh(RC::RD64), addr = fresh(RC::RD64);
            emit("mul.wide.s32 " + off + ", " + flat.reg + ", " + std::to_string(elem) + ";");
            emit("add.s64 " + addr + ", " + b.reg + ", " + off + ";");
            a.reg = addr;
        }
        return a;
    }

    Val emitLoad(const Expr& index) {
        Addr a = emitAddress(index);
        if (failed) return {};
        std::string d = fresh(classOf(a.pointee));
        emit("ld." + spacePrefix(a) + memSuffix(a.pointee) +
             " " + d + ", [" + a.reg + "];");
        return {d, a.pointee};
    }

    // Store `value` (already coerced) to the address of index-expr `lhs`.
    void emitStore(const Expr& lhs, const Val& value) {
        Addr a = emitAddress(lhs);
        if (failed) return;
        emit("st." + spacePrefix(a) + memSuffix(a.pointee) +
             " [" + a.reg + "], " + value.reg + ";");
    }

    // ++x / --x / x++ / x-- on a scalar variable (int or float).
    Val emitIncDec(const Expr& e) {
        const Expr& operand = *e.args[0];
        if (operand.kind != Expr::Ident) { fail("'++'/'--' requires a variable"); return {}; }
        auto it = vars.find(operand.str);
        if (it == vars.end()) { fail("'++'/'--' of undeclared '" + operand.str + "'"); return {}; }
        Val& var = it->second;
        if (var.type.isPointer()) { fail("'++'/'--' on a pointer is unsupported"); return {}; }
        const bool inc = e.str.find("++") != std::string::npos;
        const bool pre = e.str.compare(0, 3, "pre") == 0;
        Val old;
        if (!pre) {  // postfix: capture the value before the update
            old.type = var.type;
            old.reg = fresh(classOf(var.type));
            emit(std::string(var.type.isFloating() ? "mov.f32 " : "mov.u32 ") + old.reg + ", " + var.reg + ";");
        }
        if (var.type.isFloating())
            emit(std::string(inc ? "add.f32 " : "sub.f32 ") + var.reg + ", " + var.reg + ", 0f3F800000;");
        else
            emit(std::string(inc ? "add.s32 " : "sub.s32 ") + var.reg + ", " + var.reg + ", 1;");
        return pre ? var : old;
    }

    Val emitUnary(const Expr& e) {
        if (e.str == "pre++" || e.str == "pre--" || e.str == "post++" || e.str == "post--")
            return emitIncDec(e);
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
            const Type pointee = indexPointee(lhs);
            Val value;
            if (op == "=") {
                value = emitExpr(*e.args[1]);
            } else {                       // compound: load current, combine, store
                Val cur = emitLoad(lhs);
                if (failed) return {};
                value = computeRhs(cur);
            }
            if (failed) return {};
            value = coerce(value, pointee);
            emitStore(lhs, value);
            return value;
        }
        fail("invalid assignment target");
        return {};
    }

    // atomicAdd(&arr[i], val): read-modify-write returning the old value.
    //  * global: a real `atom.global.add`, so it stays correct when the backend
    //    runs a grid's CTAs on parallel interpreter instances.
    //  * shared: within one CTA the interpreter runs threads sequentially, so a
    //    plain ld/add/st is atomic (a CTA never spans instances).
    Val emitAtomicAdd(const Expr& e) {
        const Expr& a0 = *e.args[0];
        if (a0.kind != Expr::Unary || a0.str != "&" || a0.args.empty() || a0.args[0]->kind != Expr::Index) {
            fail("atomicAdd expects &array[index] as its first argument");
            return {};
        }
        Addr addr = emitAddress(*a0.args[0]);
        if (failed) return {};
        const Type pt = addr.pointee;
        Val val = coerce(emitExpr(*e.args[1]), pt);
        if (failed) return {};
        std::string oldv = fresh(classOf(pt));
        if (addr.shared || addr.local) {
            // Shared: one CTA runs sequentially in the interpreter. Local: private
            // to the thread. Either way a plain ld/add/st is already atomic.
            std::string sum = fresh(classOf(pt));
            emit("ld." + spacePrefix(addr) + memSuffix(pt) + " " + oldv + ", [" + addr.reg + "];");
            emit(std::string(pt.isFloating() ? "add.f32 " : "add.s32 ") + sum + ", " + oldv + ", " + val.reg + ";");
            emit("st." + spacePrefix(addr) + memSuffix(pt) + " [" + addr.reg + "], " + sum + ";");
        } else {
            emit(std::string("atom.global.add.") + memSuffix(pt) + " " + oldv + ", [" + addr.reg + "], " + val.reg + ";");
        }
        return {oldv, pt};
    }

    // Inline a call to a __device__ helper function: bind args to a fresh param
    // scope, emit the body (a `return` branches to the end label with the value
    // in retReg), then restore the caller's scope. Non-recursive only.
    Val emitInlineDeviceCall(const Kernel& fn, const Expr& call) {
        if (inlineCtx_.size() > 64) { fail("__device__ inline depth exceeded at '" + fn.name + "' (recursion?)"); return {}; }
        if (inlining_.count(fn.name)) { fail("recursive __device__ function '" + fn.name + "' is unsupported"); return {}; }
        if (call.args.size() != fn.params.size()) { fail("wrong argument count for '" + fn.name + "'"); return {}; }

        std::vector<Val> argVals;
        for (const auto& a : call.args) { argVals.push_back(emitExpr(*a)); if (failed) return {}; }

        // Fresh param scope: the body sees only its params + its own locals.
        std::unordered_map<std::string, Val> inlineScope;
        for (size_t i = 0; i < fn.params.size(); ++i) {
            const Param& p = fn.params[i];
            Val v = coerce(argVals[i], p.type);
            std::string reg = fresh(classOf(p.type));
            emit(std::string(movFor(p.type)) + reg + ", " + v.reg + ";");
            Val pv; pv.reg = reg; pv.type = p.type;
            if (p.type.isPointer()) pv.space = argVals[i].space;  // keep global/shared addressing
            inlineScope[p.name] = pv;
        }

        const Type rt = fn.returnType;
        std::string retReg = (rt.base == Type::Void) ? std::string() : fresh(classOf(rt));
        std::string endL = label();
        inlineCtx_.push_back({retReg, endL, rt});
        inlining_.insert(fn.name);

        std::unordered_map<std::string, Val> savedVars;
        savedVars.swap(vars);
        vars = std::move(inlineScope);
        for (const auto& st : fn.body) { emitStmt(*st); if (failed) break; }
        emitLabel(endL);
        vars = std::move(savedVars);

        inlining_.erase(fn.name);
        inlineCtx_.pop_back();
        if (failed) return {};
        if (rt.base == Type::Void) return {};
        return {retReg, rt};
    }

    Val emitCall(const Expr& e) {
        const std::string& fn = e.str;
        if (fn == "__syncthreads" && e.args.empty()) { emit("bar.sync 0;"); return {}; }
        if (fn == "atomicAdd" && e.args.size() == 2) return emitAtomicAdd(e);
        if (deviceFns_) {
            auto it = deviceFns_->find(fn);
            if (it != deviceFns_->end()) return emitInlineDeviceCall(*it->second, e);
        }

        // Unary intrinsics.
        if (e.args.size() == 1) {
            if (fn == "abs") {                          // integer or float abs (type-preserving)
                Val a = emitExpr(*e.args[0]); if (failed) return {};
                if (a.type.isFloating()) { std::string d = fresh(RC::F32); emit("abs.f32 " + d + ", " + a.reg + ";"); return {d, floatType()}; }
                std::string d = fresh(RC::R32); emit("abs.s32 " + d + ", " + a.reg + ";"); return {d, intType()};
            }
            Val a = coerce(emitExpr(*e.args[0]), floatType());  // f32 arg -> f32 result
            if (failed) return {};
            std::string d = fresh(RC::F32);
            if (fn == "sqrtf")  { emit("sqrt.rn.f32 " + d + ", " + a.reg + ";"); return {d, floatType()}; }
            if (fn == "fabsf")  { emit("abs.f32 "     + d + ", " + a.reg + ";"); return {d, floatType()}; }
            if (fn == "rsqrtf") { emit("rsqrt.approx.f32 " + d + ", " + a.reg + ";"); return {d, floatType()}; }
            if (fn == "sinf")   { emit("sin.approx.f32 " + d + ", " + a.reg + ";"); return {d, floatType()}; }
            if (fn == "cosf")   { emit("cos.approx.f32 " + d + ", " + a.reg + ";"); return {d, floatType()}; }
            if (fn == "__expf" || fn == "expf") {       // e^x = 2^(x*log2 e)
                std::string t = fresh(RC::F32);
                emit("mul.f32 " + t + ", " + a.reg + ", 0f3FB8AA3B;");  // log2(e)
                emit("ex2.approx.f32 " + d + ", " + t + ";");
                return {d, floatType()};
            }
            if (fn == "__logf" || fn == "logf") {       // ln x = log2(x)*ln 2
                std::string t = fresh(RC::F32);
                emit("lg2.approx.f32 " + t + ", " + a.reg + ";");
                emit("mul.f32 " + d + ", " + t + ", 0f3F317218;");     // ln(2)
                return {d, floatType()};
            }
            fail("unsupported call to '" + fn + "'");
            return {};
        }

        // Binary intrinsics: min/max (int or float).
        if (e.args.size() == 2) {
            Val a = emitExpr(*e.args[0]); if (failed) return {};
            Val b = emitExpr(*e.args[1]); if (failed) return {};
            if (fn == "fminf" || fn == "fmaxf") {
                a = coerce(a, floatType()); b = coerce(b, floatType());
                std::string d = fresh(RC::F32);
                emit(std::string(fn == "fminf" ? "min.f32 " : "max.f32 ") + d + ", " + a.reg + ", " + b.reg + ";");
                return {d, floatType()};
            }
            if (fn == "min" || fn == "max") {
                bool fp = a.type.isFloating() || b.type.isFloating();
                Type ct = fp ? floatType() : intType();
                a = coerce(a, ct); b = coerce(b, ct);
                std::string d = fresh(classOf(ct));
                std::string op = fn == "min" ? "min." : "max.";
                emit(op + (fp ? "f32 " : "s32 ") + d + ", " + a.reg + ", " + b.reg + ";");
                return {d, ct};
            }
            if (fn == "powf") {                         // x^y = 2^(y*log2 x)
                a = coerce(a, floatType()); b = coerce(b, floatType());
                std::string lg = fresh(RC::F32), mul = fresh(RC::F32), d = fresh(RC::F32);
                emit("lg2.approx.f32 " + lg + ", " + a.reg + ";");
                emit("mul.f32 " + mul + ", " + b.reg + ", " + lg + ";");
                emit("ex2.approx.f32 " + d + ", " + mul + ";");
                return {d, floatType()};
            }
            fail("unsupported call to '" + fn + "'");
            return {};
        }

        // Ternary intrinsic: fmaf(a,b,c) = a*b + c.
        if (e.args.size() == 3 && fn == "fmaf") {
            Val a = coerce(emitExpr(*e.args[0]), floatType()); if (failed) return {};
            Val b = coerce(emitExpr(*e.args[1]), floatType()); if (failed) return {};
            Val c = coerce(emitExpr(*e.args[2]), floatType()); if (failed) return {};
            std::string d = fresh(RC::F32);
            emit("fma.rn.f32 " + d + ", " + a.reg + ", " + b.reg + ", " + c.reg + ";");
            return {d, floatType()};
        }

        fail("unsupported call to '" + fn + "'");
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
                if (!ensureSupported(s.type)) return;
                if (s.arraySize > 0) {
                    int bytes = s.arraySize * s.type.elemBytes();
                    Type ptr = s.type; ptr.ptr = 1;   // the array decays to a pointer-to-element
                    Val v; v.type = ptr;
                    if (s.isShared) {
                        // __shared__ T name[N]  ->  .shared .align 4 .b8 name[N*sizeof(T)]
                        sharedDecls += "\t.shared .align 4 .b8 " + s.name + "[" + std::to_string(bytes) + "];\n";
                        v.space = Space::Shared; v.sharedName = s.name;
                    } else {
                        // T name[N]  ->  .local .align 4 .b8 name[N*sizeof(T)] (per-thread scratch)
                        localDecls += "\t.local .align 4 .b8 " + s.name + "[" + std::to_string(bytes) + "];\n";
                        v.space = Space::Local; v.localName = s.name;
                    }
                    vars[s.name] = v;
                    arrayDims_[s.name] = s.arrayDims.empty()
                                             ? std::vector<int>{s.arraySize} : s.arrayDims;
                    return;
                }
                Val v; v.type = s.type; v.reg = fresh(classOf(s.type));
                if (s.type.isPointer()) v.space = Space::Global;  // a plain pointer var, if assigned one
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
            case Stmt::Return:
                if (!inlineCtx_.empty()) {
                    // Inside an inlined __device__ function: stash the value and
                    // branch to the call-site continuation (not a kernel `ret`).
                    // Copy the context fields BY VALUE — emitExpr() below may inline
                    // a nested device call, push_back to inlineCtx_, and reallocate
                    // it, which would dangle a reference into the vector.
                    const std::string retReg = inlineCtx_.back().retReg;
                    const std::string endLabel = inlineCtx_.back().endLabel;
                    const Type retType = inlineCtx_.back().retType;
                    if (s.expr && !retReg.empty()) {
                        Val v = coerce(emitExpr(*s.expr), retType);
                        if (failed) return;
                        emit(std::string(movFor(retType)) + retReg + ", " + v.reg + ";");
                    }
                    emit("bra " + endLabel + ";");
                } else {
                    emit("ret;");
                }
                return;
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
            if (!ensureSupported(p.type)) return "";
            if (p.type.isStruct()) {
                // By-value struct: stays in .param space; members read on access.
                Val v; v.type = p.type; v.space = Space::ParamStruct; v.paramName = p.name;
                vars[p.name] = v;
            } else if (p.type.isPointer()) {
                std::string raw = fresh(RC::RD64), gbl = fresh(RC::RD64);
                emit("ld.param.u64 " + raw + ", [" + p.name + "];");
                emit("cvta.to.global.u64 " + gbl + ", " + raw + ";");
                vars[p.name] = {gbl, p.type, Space::Global, ""};
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
            const Type& pt = k.params[i].type;
            const std::string pname =
                k.params[i].name.empty() ? ("_arg" + std::to_string(i)) : k.params[i].name;
            if (pt.isStruct()) {
                int sz = structSize(pt.structName);
                if (sz <= 0) sz = 1;
                out += "\t.param .align 8 .b8 " + pname + "[" + std::to_string(sz) + "]";
            } else {
                out += "\t.param ." + std::string(paramSuffix(pt)) + " " + pname;
            }
            if (i + 1 < k.params.size()) out += ",";
            out += "\n";
        }
        out += ")\n{\n";
        if (nP  > 0) out += "\t.reg .pred %p<" + std::to_string(nP)  + ">;\n";
        if (nR  > 0) out += "\t.reg .b32 %r<"  + std::to_string(nR)  + ">;\n";
        if (nF  > 0) out += "\t.reg .f32 %f<"  + std::to_string(nF)  + ">;\n";
        if (nRd > 0) out += "\t.reg .b64 %rd<" + std::to_string(nRd) + ">;\n";
        out += sharedDecls;
        out += localDecls;
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
    // Collect __device__ helper functions (non-__global__ kernels) for inlining,
    // and pick the __global__ entry kernel `name` (first __global__ if empty).
    std::unordered_map<std::string, const Kernel*> deviceFns;
    const Kernel* target = nullptr;
    for (auto& kp : pr.module->kernels) {
        if (!kp->isGlobal) deviceFns[kp->name] = kp.get();
        if (!target && (name.empty() ? kp->isGlobal : kp->name == name)) target = kp.get();
    }
    if (!target) { r.error = "kernel not found: " + (name.empty() ? std::string("<first>") : name); return r; }
    Codegen cg(*target, &deviceFns, pr.module.get());
    std::string ptx = cg.run();
    if (cg.failed) { r.error = cg.err; return r; }
    r.ptx = std::move(ptx);
    r.ok = true;
    return r;
}

}  // namespace frontend
}  // namespace compiler
}  // namespace vgre
