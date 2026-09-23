// Tier-2 backend, increment 1 — see include/vgre/compiler/frontend/ssa_ir.h.
//
// VGRE-IR is a small typed SSA: a function is a list of basic blocks, each a list of
// instructions ending in a terminator (br/condbr/ret); every non-terminator defines
// one SSA value (its index in `vals`). This file lowers the AST to that IR for the
// scalar per-thread subset — including full structured control flow (`if`/`if-else`,
// `for`, `while`) with on-demand phi insertion (Braun et al.) — verifies
// well-formedness, and reference-evaluates it per thread, matching the Tier-1 compiled
// backend bit-for-bit (same promotion + width-narrowing rules), which is how the
// lowering is validated. Optimizer passes and native emission build on this IR next.

#include "vgre/compiler/frontend/ssa_ir.h"

#include "vgre/compiler/frontend/ast.h"
#include "vgre/compiler/frontend/parser.h"

#include <cmath>
#include <cstring>
#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace vgre {
namespace compiler {
namespace frontend {
namespace {

// ── Runtime value + numeric semantics (mirror the compiled tier for bit-exactness) ─
struct SVal { bool isF = false; int64_t i = 0; double d = 0; };
static SVal SI(int64_t v) { SVal s; s.isF = false; s.i = v; return s; }
static SVal SF(double v)  { SVal s; s.isF = true;  s.d = v; return s; }
static int64_t asI(const SVal& v) { return v.isF ? (int64_t)v.d : v.i; }
static double  asF(const SVal& v) { return v.isF ? v.d : (double)v.i; }

static int64_t satFloatToInt(double d, const Type& t) {
    const double r = std::trunc(d);
    if (std::isnan(r)) return 0;
    const int bits = t.elemBytes() * 8;
    if (!t.isUnsigned) {
        const int64_t hi = bits >= 64 ? INT64_MAX : (((int64_t)1 << (bits - 1)) - 1);
        const int64_t lo = bits >= 64 ? INT64_MIN : -((int64_t)1 << (bits - 1));
        return r >= (double)hi ? hi : r <= (double)lo ? lo : (int64_t)r;
    }
    const uint64_t hi = bits >= 64 ? UINT64_MAX : (((uint64_t)1 << bits) - 1);
    return (int64_t)(r <= 0.0 ? 0 : r >= (double)hi ? hi : (uint64_t)r);
}
static SVal coerce(const SVal& v, const Type& t) {
    if (t.isPointer()) return SI(asI(v));
    if (t.isFloating())
        return t.base == Type::Double ? SF(asF(v)) : SF((double)(float)asF(v));
    const int64_t iv = v.isF ? satFloatToInt(v.d, t) : v.i;
    switch (t.base) {
        case Type::Bool:  return SI(iv != 0 ? 1 : 0);
        case Type::Char:  return SI(t.isUnsigned ? (int64_t)(uint8_t)iv  : (int64_t)(int8_t)iv);
        case Type::Short: return SI(t.isUnsigned ? (int64_t)(uint16_t)iv : (int64_t)(int16_t)iv);
        case Type::Int:   return SI(t.isUnsigned ? (int64_t)(uint32_t)iv : (int64_t)(int32_t)iv);
        default:          return SI(iv);
    }
}
static Type scalar(Type::Base b) { Type t; t.base = b; return t; }
static Type promoteT(const Type& a, const Type& b) {
    if (a.isPointer()) return a;
    if (b.isPointer()) return b;
    Type r;
    if (a.base == Type::Double || b.base == Type::Double) { r.base = Type::Double; return r; }
    if (a.isFloating() || b.isFloating() || a.base == Type::Half || b.base == Type::Half) { r.base = Type::Float; return r; }
    const bool lng = (a.base == Type::Long || b.base == Type::Long);
    r.base = lng ? Type::Long : Type::Int;
    auto atRank = [&](const Type& t) { return (t.base == Type::Long) == lng; };
    if ((a.isUnsigned && atRank(a)) || (b.isUnsigned && atRank(b))) r.isUnsigned = true;
    return r;
}

// ── The IR ──────────────────────────────────────────────────────────────────────
enum class Op {
    ConstI, ConstF, Param, Tid, Ctaid, Ntid, Nctaid,  // leaves
    Phi, Bin, Un, Cmp, Sel, Cast, Load, CallMath,      // value-producing
    Store, CondBr, Br, Ret                             // effects / terminators
};
struct Inst {
    Op op;
    Type ty;                    // result type (void for Store/Br/CondBr/Ret)
    std::vector<int> a;         // operand value-ids (for Phi: one per predecessor)
    std::vector<int> phiPred;   // Phi: predecessor block for each operand in `a`
    int64_t ci = 0; double cf = 0;
    int dim = 0, paramIdx = 0;
    std::string s;              // operator spelling / math fn name
    int elemBytes = 0;          // Load / Store element size
    int bbT = -1, bbF = -1;     // Br target / CondBr true,false
    int block = -1;             // Phi: the block it belongs to
};
struct BB { std::vector<int> insts; };   // value-ids, in order; last is a terminator

struct Fn {
    std::vector<Type> ptypes;   // kernel parameter types (pointer or scalar)
    std::vector<Inst> vals;     // all instructions, indexed by SSA id
    std::vector<BB> bbs;
    std::vector<std::vector<int>> preds;   // per-block predecessor block ids
    int entry = 0;
    bool isTerminator(int id) const {
        Op o = vals[id].op; return o == Op::Br || o == Op::CondBr || o == Op::Ret;
    }
};

// ── AST → SSA lowering (increment-1 subset) ───────────────────────────────────────
// SSA construction follows Braun et al., "Simple and Efficient Construction of
// Static Single Assignment Form" (2013): per-block current definitions, phis placed
// on demand at merges, incomplete phis in not-yet-sealed (loop-header) blocks.
struct Lowerer {
    const Kernel& k;
    Fn fn;
    bool ok = true;
    std::string err;
    int cur = 0;                                  // current block id
    std::unordered_map<std::string, Type> vtype;  // name -> type (stable across SSA versions)
    // Braun state:
    std::unordered_map<std::string, std::unordered_map<int, int>> curDef;   // var -> block -> value
    std::vector<char> sealed;                                               // per block
    std::unordered_map<int, std::unordered_map<std::string, int>> incPhis;  // block -> var -> phi id

    explicit Lowerer(const Kernel& kern) : k(kern) {}
    void fail(const std::string& m) { if (ok) { ok = false; err = m; } }

    int newBlock(bool seal = false) {
        fn.bbs.push_back({}); fn.preds.push_back({}); sealed.push_back(seal ? 1 : 0);
        return (int)fn.bbs.size() - 1;
    }
    void addEdge(int from, int to) { (void)from; fn.preds[to].push_back(from); }
    int emit(Inst in) { int id = (int)fn.vals.size(); fn.vals.push_back(std::move(in)); fn.bbs[cur].insts.push_back(id); return id; }

    // ── Braun SSA variable access ────────────────────────────────────────────────
    void writeVar(const std::string& v, int block, int val) { curDef[v][block] = val; }
    int readVar(const std::string& v, int block) {
        auto& m = curDef[v];
        auto it = m.find(block);
        if (it != m.end()) return it->second;
        return readVarRec(v, block);
    }
    int newPhi(const std::string& v, int block) {
        Inst in; in.op = Op::Phi; in.ty = vtype.count(v) ? vtype[v] : scalar(Type::Int); in.block = block;
        int id = (int)fn.vals.size(); fn.vals.push_back(std::move(in));
        fn.bbs[block].insts.insert(fn.bbs[block].insts.begin(), id);   // phis lead the block
        return id;
    }
    int readVarRec(const std::string& v, int block) {
        int val;
        if (!sealed[block]) { val = newPhi(v, block); incPhis[block][v] = val; }
        else if (fn.preds[block].size() == 1) { val = readVar(v, fn.preds[block][0]); }
        else { val = newPhi(v, block); writeVar(v, block, val); return addPhiOperands(v, val); }
        writeVar(v, block, val);
        return val;
    }
    int addPhiOperands(const std::string& v, int phi) {
        int block = fn.vals[phi].block;
        for (int p : fn.preds[block]) { fn.vals[phi].a.push_back(readVar(v, p)); fn.vals[phi].phiPred.push_back(p); }
        return phi;
    }
    void sealBlock(int block) {
        auto it = incPhis.find(block);
        if (it != incPhis.end()) for (auto& kv : it->second) addPhiOperands(kv.first, kv.second);
        sealed[block] = 1;
    }

    static bool member(const Expr& e, const char* obj, int& dim) {
        if (e.kind != Expr::Member || e.args.size() != 1 || e.args[0]->kind != Expr::Ident || e.args[0]->str != obj) return false;
        dim = e.str == "x" ? 0 : e.str == "y" ? 1 : e.str == "z" ? 2 : -1; return dim >= 0;
    }

    // Type of an expression (mirrors the compiled tier's estimateType for the subset).
    Type typeOf(const Expr& e) {
        switch (e.kind) {
            case Expr::IntLit:   { Type t; t.base = e.wide ? Type::Long : Type::Int; return t; }
            case Expr::FloatLit: return scalar(e.wide ? Type::Double : Type::Float);
            case Expr::Ident: { auto it = vtype.find(e.str); return it != vtype.end() ? it->second : scalar(Type::Int); }
            case Expr::Index: { Type b = e.args.empty() ? Type{} : vtype.count(e.args[0]->str) ? vtype[e.args[0]->str] : Type{}; if (b.ptr > 0) b.ptr--; return b; }
            case Expr::Unary:  if (e.str == "!") return scalar(Type::Int);
                               return e.args.empty() ? scalar(Type::Int) : typeOf(*e.args[0]);
            case Expr::Cast:   return e.castType;
            case Expr::Binary: {
                const std::string& o = e.str;
                if (o=="<"||o=="<="||o==">"||o==">="||o=="=="||o=="!="||o=="&&"||o=="||") return scalar(Type::Int);
                return promoteT(typeOf(*e.args[0]), typeOf(*e.args[1]));
            }
            case Expr::Ternary: return promoteT(typeOf(*e.args[1]), typeOf(*e.args[2]));
            case Expr::Assign:  return typeOf(*e.args[0]);
            case Expr::Call: {
                const std::string& fnn = e.str;
                if (fnn == "min" || fnn == "max") return e.args.size() < 2 ? scalar(Type::Int) : promoteT(typeOf(*e.args[0]), typeOf(*e.args[1]));
                if (fnn == "abs") return e.args.empty() ? scalar(Type::Int) : typeOf(*e.args[0]);
                return scalar(!fnn.empty() && fnn.back() == 'f' ? Type::Float : Type::Double);
            }
            default: return scalar(Type::Int);
        }
    }

    int constI(int64_t v, Type t) { Inst in; in.op = Op::ConstI; in.ty = t; in.ci = v; return emit(std::move(in)); }
    int constF(double v, Type t)  { Inst in; in.op = Op::ConstF; in.ty = t; in.cf = v; return emit(std::move(in)); }

    // Lower an expression to an SSA value id (its result coerced to `typeOf`).
    int lowerExpr(const Expr& e) {
        if (!ok) return 0;
        switch (e.kind) {
            case Expr::IntLit:   return constI(e.ival, typeOf(e));
            case Expr::FloatLit: return constF(e.fval, typeOf(e));
            case Expr::Ident: {
                if (!vtype.count(e.str)) { fail("SSA: unknown identifier '" + e.str + "'"); return 0; }
                return readVar(e.str, cur);
            }
            case Expr::Member: {
                int dim;
                if (member(e, "threadIdx", dim)) { Inst in; in.op = Op::Tid;   in.ty = scalar(Type::Int); in.dim = dim; return emit(std::move(in)); }
                if (member(e, "blockIdx", dim))  { Inst in; in.op = Op::Ctaid; in.ty = scalar(Type::Int); in.dim = dim; return emit(std::move(in)); }
                if (member(e, "blockDim", dim))  { Inst in; in.op = Op::Ntid;  in.ty = scalar(Type::Int); in.dim = dim; return emit(std::move(in)); }
                if (member(e, "gridDim", dim))   { Inst in; in.op = Op::Nctaid;in.ty = scalar(Type::Int); in.dim = dim; return emit(std::move(in)); }
                fail("SSA: unsupported member access"); return 0;
            }
            case Expr::Index: {   // p[idx] — a global load
                if (e.args.size() != 2 || e.args[0]->kind != Expr::Ident) { fail("SSA: bad index"); return 0; }
                Type pt = vtype.count(e.args[0]->str) ? vtype[e.args[0]->str] : Type{};
                if (pt.ptr != 1) { fail("SSA: index base must be a pointer"); return 0; }
                Type elem = pt; elem.ptr = 0;
                int base = lowerExpr(*e.args[0]);
                int idx  = lowerExpr(*e.args[1]);
                if (!ok) return 0;
                Inst in; in.op = Op::Load; in.ty = elem; in.a = {base, idx}; in.elemBytes = elem.elemBytes();
                return emit(std::move(in));
            }
            case Expr::Unary: {
                if (e.str == "+") return lowerExpr(*e.args[0]);
                // ++x / x++ / --x / x-- on a local variable (SSA rebind; returns old or new).
                if (e.str == "pre++" || e.str == "post++" || e.str == "pre--" || e.str == "post--") {
                    if (e.args[0]->kind != Expr::Ident || !vtype.count(e.args[0]->str)) { fail("SSA: ++/-- on a non-local"); return 0; }
                    const std::string& nm = e.args[0]->str; Type vt = vtype[nm];
                    int old = readVar(nm, cur);
                    int one = vt.isFloating() ? constF(1.0, vt) : constI(1, vt);
                    Inst b; b.op = Op::Bin; b.s = (e.str[3] == '+' || e.str[4] == '+') ? "+" : "-"; b.ty = vt; b.a = {old, one};
                    int nv = emit(std::move(b));
                    Inst c; c.op = Op::Cast; c.ty = vt; c.a = {nv}; int cv = emit(std::move(c));
                    writeVar(nm, cur, cv);
                    return (e.str.rfind("pre", 0) == 0) ? cv : old;
                }
                if (e.str != "-" && e.str != "!" && e.str != "~") { fail("SSA: unary '" + e.str + "'"); return 0; }
                int a = lowerExpr(*e.args[0]); if (!ok) return 0;
                Inst in; in.op = Op::Un; in.s = e.str; in.ty = e.str == "!" ? scalar(Type::Int) : typeOf(*e.args[0]); in.a = {a};
                return emit(std::move(in));
            }
            case Expr::Cast: {
                int a = lowerExpr(*e.args[0]); if (!ok) return 0;
                Inst in; in.op = Op::Cast; in.ty = e.castType; in.a = {a};
                return emit(std::move(in));
            }
            case Expr::Ternary: {
                int c = lowerExpr(*e.args[0]);
                int a = lowerExpr(*e.args[1]);
                int b = lowerExpr(*e.args[2]);
                if (!ok) return 0;
                Inst in; in.op = Op::Sel; in.ty = typeOf(e); in.a = {c, a, b};
                return emit(std::move(in));
            }
            case Expr::Binary: {
                int a = lowerExpr(*e.args[0]);
                int b = lowerExpr(*e.args[1]);
                if (!ok) return 0;
                const std::string& o = e.str;
                const bool isCmp = (o=="<"||o=="<="||o==">"||o==">="||o=="=="||o=="!=");
                Inst in; in.s = o; in.a = {a, b};
                if (isCmp) { in.op = Op::Cmp; in.ty = scalar(Type::Int); }
                else       { in.op = Op::Bin; in.ty = promoteT(typeOf(*e.args[0]), typeOf(*e.args[1])); }
                return emit(std::move(in));
            }
            case Expr::Call: {
                const std::string& fnn = e.str;
                if (e.args.size() == 1 && (fnn=="sqrtf"||fnn=="fabsf"||fnn=="expf"||fnn=="logf"||fnn=="sinf"||fnn=="cosf"||fnn=="floorf"||fnn=="ceilf"||fnn=="tanhf"||fnn=="sqrt"||fnn=="fabs"||fnn=="exp"||fnn=="log"||fnn=="sin"||fnn=="cos"||fnn=="floor"||fnn=="ceil"||fnn=="tanh")) {
                    int a = lowerExpr(*e.args[0]); if (!ok) return 0;
                    Inst in; in.op = Op::CallMath; in.s = fnn; in.ty = typeOf(e); in.a = {a};
                    return emit(std::move(in));
                }
                if ((fnn=="min"||fnn=="max"||fnn=="fminf"||fnn=="fmaxf"||fnn=="fmin"||fnn=="fmax") && e.args.size()==2) {
                    int a = lowerExpr(*e.args[0]); int b = lowerExpr(*e.args[1]); if (!ok) return 0;
                    Inst in; in.op = Op::CallMath; in.s = fnn; in.ty = typeOf(e); in.a = {a, b};
                    return emit(std::move(in));
                }
                fail("SSA: unsupported call '" + fnn + "'"); return 0;
            }
            case Expr::Assign: {
                lowerAssign(e);
                if (!ok) return 0;
                // The assignment expression's value is the new value of the target.
                if (e.args[0]->kind == Expr::Ident) return readVar(e.args[0]->str, cur);
                return lowerExpr(*e.args[0]);   // p[i] = v used as a value: re-load
            }
            default: fail("SSA: unsupported expression"); return 0;
        }
    }

    // `x [op]= rhs` (local, straight-line SSA rebind) or `p[i] [op]= rhs` (global store).
    void lowerAssign(const Expr& e) {
        const Expr& lhs = *e.args[0];
        const std::string& op = e.str;
        if (lhs.kind == Expr::Ident) {
            if (!vtype.count(lhs.str)) { fail("SSA: assign to unknown '" + lhs.str + "'"); return; }
            Type vt = vtype[lhs.str];
            int rhs = lowerExpr(*e.args[1]); if (!ok) return;
            int val = rhs;
            if (op != "=") { int old = readVar(lhs.str, cur); Inst in; in.op = Op::Bin; in.s = op.substr(0, op.size() - 1); in.ty = promoteT(vt, typeOf(*e.args[1])); in.a = {old, rhs}; val = emit(std::move(in)); }
            Inst c; c.op = Op::Cast; c.ty = vt; c.a = {val};      // narrow to the variable's type
            writeVar(lhs.str, cur, emit(std::move(c)));
            return;
        }
        if (lhs.kind == Expr::Index) {
            if (lhs.args.size() != 2 || lhs.args[0]->kind != Expr::Ident) { fail("SSA: bad store index"); return; }
            Type pt = vtype.count(lhs.args[0]->str) ? vtype[lhs.args[0]->str] : Type{};
            if (pt.ptr != 1) { fail("SSA: store base must be a pointer"); return; }
            Type elem = pt; elem.ptr = 0;
            int base = lowerExpr(*lhs.args[0]);
            int idx  = lowerExpr(*lhs.args[1]);
            int rhs  = lowerExpr(*e.args[1]);
            if (!ok) return;
            int val = rhs;
            if (op != "=") { Inst ld; ld.op = Op::Load; ld.ty = elem; ld.a = {base, idx}; ld.elemBytes = elem.elemBytes(); int cur_ = emit(std::move(ld));
                             Inst in; in.op = Op::Bin; in.s = op.substr(0, op.size() - 1); in.ty = promoteT(elem, typeOf(*e.args[1])); in.a = {cur_, rhs}; val = emit(std::move(in)); }
            Inst c; c.op = Op::Cast; c.ty = elem; c.a = {val}; int cv = emit(std::move(c));
            Inst st; st.op = Op::Store; st.a = {base, idx, cv}; st.elemBytes = elem.elemBytes(); st.ty = elem; emit(std::move(st));
            return;
        }
        fail("SSA: unsupported assignment target");
    }

    void lowerStmt(const Stmt& s) {
        if (!ok) return;
        switch (s.kind) {
            case Stmt::VarDecl: {
                if (s.arraySize > 0 || s.type.isStruct()) { fail("SSA: local arrays/structs unsupported on this tier"); return; }
                vtype[s.name] = s.type;
                int v;
                if (s.expr) { int r = lowerExpr(*s.expr); if (!ok) return; Inst c; c.op = Op::Cast; c.ty = s.type; c.a = {r}; v = emit(std::move(c)); }
                else        { v = s.type.isFloating() ? constF(0.0, s.type) : constI(0, s.type); }
                writeVar(s.name, cur, v);
                return;
            }
            case Stmt::ExprStmt: if (s.expr) { if (s.expr->kind == Expr::Assign) lowerAssign(*s.expr); else lowerExpr(*s.expr); } return;
            case Stmt::Block: for (auto& st : s.body) { lowerStmt(*st); if (!ok) return; } return;
            case Stmt::Return: { Inst in; in.op = Op::Ret; emit(std::move(in)); return; }
            case Stmt::Empty: return;
            case Stmt::If: {
                int c = lowerExpr(*s.expr); if (!ok) return;
                const bool hasElse = !s.elseBody.empty();
                int thenB = newBlock(), elseB = hasElse ? newBlock() : -1, merge = newBlock();
                int falseTgt = hasElse ? elseB : merge;
                Inst cb; cb.op = Op::CondBr; cb.a = {c}; cb.bbT = thenB; cb.bbF = falseTgt; emit(std::move(cb));
                addEdge(cur, thenB); addEdge(cur, falseTgt);
                sealBlock(thenB); if (hasElse) sealBlock(elseB);   // preds (the condbr block) are known
                cur = thenB;
                for (auto& st : s.body) { lowerStmt(*st); if (!ok) return; }
                Inst b1; b1.op = Op::Br; b1.bbT = merge; emit(std::move(b1)); addEdge(cur, merge);
                if (hasElse) {
                    cur = elseB;
                    for (auto& st : s.elseBody) { lowerStmt(*st); if (!ok) return; }
                    Inst b2; b2.op = Op::Br; b2.bbT = merge; emit(std::move(b2)); addEdge(cur, merge);
                }
                sealBlock(merge);
                cur = merge;
                return;
            }
            case Stmt::For:
            case Stmt::While: {
                if (s.kind == Stmt::For && s.forInit) { lowerStmt(*s.forInit); if (!ok) return; }
                int header = newBlock();   // unsealed: the back-edge pred is still pending
                Inst brh; brh.op = Op::Br; brh.bbT = header; emit(std::move(brh)); addEdge(cur, header);
                cur = header;
                const Expr* cond = (s.kind == Stmt::For) ? s.forCond.get() : s.expr.get();
                int c = cond ? lowerExpr(*cond) : constI(1, scalar(Type::Int));
                if (!ok) return;
                int body = newBlock(), exit = newBlock();
                Inst cb; cb.op = Op::CondBr; cb.a = {c}; cb.bbT = body; cb.bbF = exit; emit(std::move(cb));
                addEdge(header, body); addEdge(header, exit);
                sealBlock(body);
                cur = body;
                const std::vector<StmtPtr>& loopBody = (s.kind == Stmt::For) ? s.body : s.body;
                for (auto& st : loopBody) { lowerStmt(*st); if (!ok) return; }
                if (s.kind == Stmt::For && s.forIncr) { lowerExpr(*s.forIncr); if (!ok) return; }
                Inst bb; bb.op = Op::Br; bb.bbT = header; emit(std::move(bb)); addEdge(cur, header);   // back-edge
                sealBlock(header);   // all header preds known now → fill its incomplete phis
                sealBlock(exit);
                cur = exit;
                return;
            }
            default: fail("SSA: unsupported statement (do-while/switch/break/continue → later increment)"); return;
        }
    }

    bool run() {
        // Entry block (sealed: it has no predecessors), params.
        cur = newBlock(/*seal=*/true); fn.entry = 0;
        for (size_t i = 0; i < k.params.size(); ++i) {
            const Param& p = k.params[i];
            fn.ptypes.push_back(p.type);
            if (p.type.isStruct()) { fail("SSA: struct params unsupported on this tier"); return false; }
            Inst in; in.op = Op::Param; in.ty = p.type; in.paramIdx = (int)i;
            int v = emit(std::move(in));
            if (!p.name.empty()) { vtype[p.name] = p.type; writeVar(p.name, cur, v); }
        }
        for (auto& st : k.body) { lowerStmt(*st); if (!ok) return false; }
        // Ensure every block ends in a terminator (append an implicit ret).
        for (auto& bb : fn.bbs) {
            if (bb.insts.empty() || !fn.isTerminator(bb.insts.back())) {
                Inst in; in.op = Op::Ret; int id = (int)fn.vals.size(); fn.vals.push_back(std::move(in)); bb.insts.push_back(id);
            }
        }
        return ok;
    }
};

// ── Verifier: each value defined once (by construction), operands defined, every
// block terminated, branch targets in range. ─────────────────────────────────────
static bool verify(const Fn& fn, std::string& err) {
    if (fn.bbs.empty()) { err = "SSA verify: no blocks"; return false; }
    for (size_t b = 0; b < fn.bbs.size(); ++b) {
        const BB& bb = fn.bbs[b];
        if (bb.insts.empty()) { err = "SSA verify: empty block"; return false; }
        for (size_t j = 0; j < bb.insts.size(); ++j) {
            int id = bb.insts[j];
            if (id < 0 || id >= (int)fn.vals.size()) { err = "SSA verify: bad value id"; return false; }
            const Inst& in = fn.vals[id];
            const bool term = fn.isTerminator(id);
            if (term != (j + 1 == bb.insts.size())) { err = "SSA verify: terminator not at block end"; return false; }
            for (int op : in.a) if (op < 0 || op >= (int)fn.vals.size()) { err = "SSA verify: operand out of range"; return false; }
            if (in.op == Op::Br && (in.bbT < 0 || in.bbT >= (int)fn.bbs.size())) { err = "SSA verify: bad br target"; return false; }
            if (in.op == Op::CondBr && (in.bbT < 0 || in.bbT >= (int)fn.bbs.size() || in.bbF < 0 || in.bbF >= (int)fn.bbs.size())) { err = "SSA verify: bad condbr target"; return false; }
        }
    }
    return true;
}

// ── Reference evaluator (one thread) ──────────────────────────────────────────────
static SVal binop(const std::string& o, const SVal& A, const SVal& B, const Type& rt) {
    if (rt.isFloating() || A.isF || B.isF) {
        double a = asF(A), b = asF(B), r = 0;
        if (o == "+") r = a + b; else if (o == "-") r = a - b; else if (o == "*") r = a * b; else if (o == "/") r = a / b;
        else return SF(0);
        return coerce(SF(r), rt);
    }
    int64_t a = A.i, b = B.i, r = 0;
    const bool u = rt.isUnsigned;
    if (o == "+") r = a + b; else if (o == "-") r = a - b; else if (o == "*") r = a * b;
    else if (o == "/") r = b ? (u ? (int64_t)((uint64_t)a / (uint64_t)b) : a / b) : 0;
    else if (o == "%") r = b ? (u ? (int64_t)((uint64_t)a % (uint64_t)b) : a % b) : 0;
    else if (o == "&") r = a & b; else if (o == "|") r = a | b; else if (o == "^") r = a ^ b;
    else if (o == "<<") r = a << (b & 63); else if (o == ">>") r = u ? (int64_t)((uint64_t)a >> (b & 63)) : a >> (b & 63);
    else if (o == "&&") r = (a != 0 && b != 0); else if (o == "||") r = (a != 0 || b != 0);
    return coerce(SI(r), rt);
}
static SVal cmpop(const std::string& o, const SVal& A, const SVal& B) {
    bool r;
    if (A.isF || B.isF) { double a = asF(A), b = asF(B);
        r = o=="<"?a<b:o=="<="?a<=b:o==">"?a>b:o==">="?a>=b:o=="=="?a==b:a!=b; }
    else { int64_t a = A.i, b = B.i;
        r = o=="<"?a<b:o=="<="?a<=b:o==">"?a>b:o==">="?a>=b:o=="=="?a==b:a!=b; }
    return SI(r ? 1 : 0);
}
static SVal mathfn(const std::string& fn, const SVal& a, const SVal* b, const Type& rt) {
    if (fn == "min" || fn == "max") {
        if (rt.isFloating()) { double x = asF(a), y = asF(*b); return coerce(SF(fn=="min"?std::fmin(x,y):std::fmax(x,y)), rt); }
        int64_t x = a.i, y = b->i; return coerce(SI(fn=="min"?(x<y?x:y):(x>y?x:y)), rt);
    }
    if (fn == "fminf" || fn == "fmin") return coerce(SF(std::fmin(asF(a), asF(*b))), rt);
    if (fn == "fmaxf" || fn == "fmax") return coerce(SF(std::fmax(asF(a), asF(*b))), rt);
    double x = asF(a), r = x;
    const std::string g = (!fn.empty() && fn.back() == 'f') ? fn.substr(0, fn.size() - 1) : fn;
    if (g == "sqrt") r = std::sqrt(x); else if (g == "fabs") r = std::fabs(x);
    else if (g == "exp") r = std::exp(x); else if (g == "log") r = std::log(x);
    else if (g == "sin") r = std::sin(x); else if (g == "cos") r = std::cos(x);
    else if (g == "floor") r = std::floor(x); else if (g == "ceil") r = std::ceil(x);
    else if (g == "tanh") r = std::tanh(x);
    return coerce(SF(r), rt);
}
static SVal memLoad(int64_t addr, const Type& t) {
    void* p = reinterpret_cast<void*>(addr);
    if (t.isPointer()) { int64_t v; std::memcpy(&v, p, 8); return SI(v); }
    if (t.isFloating()) { if (t.elemBytes() == 8) { double d; std::memcpy(&d, p, 8); return SF(d); } float f; std::memcpy(&f, p, 4); return SF((double)f); }
    switch (t.elemBytes()) {
        case 1: { if (t.isUnsigned) { uint8_t v; std::memcpy(&v, p, 1); return SI(v); } int8_t v; std::memcpy(&v, p, 1); return SI(v); }
        case 2: { if (t.isUnsigned) { uint16_t v; std::memcpy(&v, p, 2); return SI(v); } int16_t v; std::memcpy(&v, p, 2); return SI(v); }
        case 8: { int64_t v; std::memcpy(&v, p, 8); return SI(v); }
        default: { if (t.isUnsigned) { uint32_t v; std::memcpy(&v, p, 4); return SI(v); } int32_t v; std::memcpy(&v, p, 4); return SI((int64_t)v); }
    }
}
static void memStore(int64_t addr, const Type& t, const SVal& v) {
    void* p = reinterpret_cast<void*>(addr);
    if (t.isPointer()) { int64_t x = asI(v); std::memcpy(p, &x, 8); return; }
    if (t.isFloating()) { if (t.elemBytes() == 8) { double d = asF(v); std::memcpy(p, &d, 8); } else { float f = (float)asF(v); std::memcpy(p, &f, 4); } return; }
    int64_t x = asI(v);
    switch (t.elemBytes()) {
        case 1: { uint8_t b = (uint8_t)x; std::memcpy(p, &b, 1); break; }
        case 2: { uint16_t b = (uint16_t)x; std::memcpy(p, &b, 2); break; }
        case 8: { std::memcpy(p, &x, 8); break; }
        default: { int32_t b = (int32_t)x; std::memcpy(p, &b, 4); break; }
    }
}

// ── Optimizer passes (increment 3): pure, semantics-preserving IR→IR ──────────────
static bool isConst(const Inst& in) { return in.op == Op::ConstI || in.op == Op::ConstF; }

// Constant folding: rewrite pure ops with all-constant operands to a constant, in
// place (same value id, so uses need no rewrite). Runs to a fixpoint.
static void constFold(Fn& fn) {
    auto cst = [&](int id, SVal& out) -> bool {
        const Inst& o = fn.vals[id];
        if (o.op == Op::ConstI) { out = coerce(SI(o.ci), o.ty); return true; }
        if (o.op == Op::ConstF) { out = coerce(SF(o.cf), o.ty); return true; }
        return false;
    };
    auto setConst = [&](Inst& in, SVal r) {
        if (in.ty.isFloating()) { in.op = Op::ConstF; in.cf = r.d; } else { in.op = Op::ConstI; in.ci = coerce(r, in.ty).i; }
        in.a.clear();
    };
    bool changed = true;
    while (changed) {
        changed = false;
        for (auto& in : fn.vals) {
            if (isConst(in)) continue;
            SVal a, b, c;
            if (in.op == Op::Bin && in.a.size() == 2 && cst(in.a[0], a) && cst(in.a[1], b)) { setConst(in, binop(in.s, a, b, in.ty)); changed = true; }
            else if (in.op == Op::Cmp && in.a.size() == 2 && cst(in.a[0], a) && cst(in.a[1], b)) { in.op = Op::ConstI; in.ci = cmpop(in.s, a, b).i; in.a.clear(); changed = true; }
            else if (in.op == Op::Cast && in.a.size() == 1 && cst(in.a[0], a)) { setConst(in, coerce(a, in.ty)); changed = true; }
            else if (in.op == Op::Un && in.a.size() == 1 && cst(in.a[0], a)) {
                SVal r = in.s == "-" ? (in.ty.isFloating() ? SF(-asF(a)) : SI(-a.i)) : in.s == "~" ? SI(~a.i) : SI(asI(a) == 0 ? 1 : 0);
                setConst(in, r); changed = true;
            } else if (in.op == Op::Sel && in.a.size() == 3 && cst(in.a[0], c)) {
                in.op = Op::Cast; in.a = {(asI(c) != 0) ? in.a[1] : in.a[2]}; changed = true;   // fold to a copy of the taken arm
            }
        }
    }
}

// Local value numbering (safe GVN within a block): dedupe identical pure instructions;
// the canonical precedes the duplicate in the same block, so it dominates all uses.
static void localGvn(Fn& fn) {
    std::vector<int> remap(fn.vals.size());
    for (size_t i = 0; i < remap.size(); ++i) remap[i] = (int)i;
    std::function<int(int)> resolve = [&](int id) { while (remap[id] != id) id = remap[id]; return id; };
    for (auto& bb : fn.bbs) {
        std::unordered_map<std::string, int> seen;
        for (int id : bb.insts) {
            Inst& in = fn.vals[id];
            const bool pure = in.op == Op::ConstI || in.op == Op::ConstF || in.op == Op::Bin ||
                              in.op == Op::Un || in.op == Op::Cmp || in.op == Op::Cast || in.op == Op::Sel;
            for (int& op : in.a) op = resolve(op);
            if (!pure) continue;
            uint64_t fb; std::memcpy(&fb, &in.cf, 8);
            std::string key = std::to_string((int)in.op) + "|" + in.s + "|" + std::to_string(in.ci) + "|" +
                              std::to_string(fb) + "|" + std::to_string((int)in.ty.base) + std::to_string(in.ty.ptr) +
                              std::to_string(in.ty.isUnsigned ? 1 : 0);
            for (int op : in.a) key += "," + std::to_string(op);
            auto it = seen.find(key);
            if (it != seen.end()) remap[id] = it->second; else seen[key] = id;
        }
    }
    for (auto& in : fn.vals) for (int& op : in.a) op = resolve(op);
    for (auto& bb : fn.bbs) {
        std::vector<int> keep;
        for (int id : bb.insts) if (resolve(id) == id) keep.push_back(id);
        bb.insts = std::move(keep);
    }
}

// Dead-code elimination: keep effects (stores, terminators) + everything transitively
// used by them; drop the rest from the block lists.
static void dce(Fn& fn) {
    std::vector<char> live(fn.vals.size(), 0);
    std::vector<int> work;
    for (auto& bb : fn.bbs) for (int id : bb.insts) {
        const Inst& in = fn.vals[id];
        if (in.op == Op::Store || in.op == Op::Ret || in.op == Op::Br || in.op == Op::CondBr) {
            if (!live[id]) { live[id] = 1; work.push_back(id); }
        }
    }
    while (!work.empty()) { int id = work.back(); work.pop_back(); for (int op : fn.vals[id].a) if (!live[op]) { live[op] = 1; work.push_back(op); } }
    for (auto& bb : fn.bbs) {
        std::vector<int> keep;
        for (int id : bb.insts) {
            const Inst& in = fn.vals[id];
            const bool effect = in.op == Op::Store || in.op == Op::Ret || in.op == Op::Br || in.op == Op::CondBr;
            if (effect || live[id]) keep.push_back(id);
        }
        bb.insts = std::move(keep);
    }
}

static void runOpt(Fn& fn) { constFold(fn); localGvn(fn); constFold(fn); dce(fn); }

}  // namespace

// ── SsaProgram wrapper ────────────────────────────────────────────────────────────
struct SsaProgram::Impl {
    Fn fn;
    std::vector<std::string> pnames;   // (unused at runtime; args are positional)
};

SsaProgram::SsaProgram() : p_(new Impl) {}
SsaProgram::~SsaProgram() = default;
int SsaProgram::numBlocks() const { return (int)p_->fn.bbs.size(); }
int SsaProgram::numValues() const { return (int)p_->fn.vals.size(); }
int SsaProgram::liveInsts() const { int n = 0; for (const auto& bb : p_->fn.bbs) n += (int)bb.insts.size(); return n; }

std::unique_ptr<SsaProgram> SsaProgram::compile(const std::string& source, const std::string& name, std::string& err, bool optimize) {
    ParseResult pr = parse(source);
    if (!pr.ok || !pr.module) { err = pr.error; return nullptr; }
    const Kernel* target = nullptr;
    for (auto& kp : pr.module->kernels)
        if (kp->isGlobal && (name.empty() || kp->name == name)) { target = kp.get(); break; }
    if (!target) { err = "SSA: kernel '" + name + "' not found"; return nullptr; }
    Lowerer lo(*target);
    if (!lo.run()) { err = lo.err; return nullptr; }
    if (!verify(lo.fn, err)) return nullptr;
    if (optimize) { runOpt(lo.fn); if (!verify(lo.fn, err)) return nullptr; }
    std::unique_ptr<SsaProgram> prog(new SsaProgram());
    prog->p_->fn = std::move(lo.fn);
    return prog;
}

bool SsaProgram::launch(Extent grid, Extent block, void* const* args, int numArgs) {
    const Fn& fn = p_->fn;
    if (numArgs < (int)fn.ptypes.size()) return false;
    const uint32_t bx = block.x, by = block.y, bz = block.z;
    for (uint32_t gz = 0; gz < grid.z; ++gz)
    for (uint32_t gy = 0; gy < grid.y; ++gy)
    for (uint32_t gx = 0; gx < grid.x; ++gx)
    for (uint32_t tz = 0; tz < bz; ++tz)
    for (uint32_t ty = 0; ty < by; ++ty)
    for (uint32_t tx = 0; tx < bx; ++tx) {
        std::vector<SVal> v(fn.vals.size());
        std::vector<char> done(fn.vals.size(), 0);
        const uint32_t tid[3] = {tx, ty, tz}, ctaid[3] = {gx, gy, gz};
        const uint32_t ntid[3] = {bx, by, bz}, nctaid[3] = {grid.x, grid.y, grid.z};
        int bb = fn.entry, prevBB = -1;
        for (long guard = 0; guard < (1L << 34); ++guard) {   // bounded against runaway loops
            bool ret = false;
            // Phis (which lead a block) resolve against the predecessor we arrived from,
            // read as a parallel copy of the pre-block state.
            {
                std::vector<std::pair<int, SVal>> phiSet;
                for (int id : fn.bbs[bb].insts) {
                    const Inst& in = fn.vals[id];
                    if (in.op != Op::Phi) break;
                    SVal sel{};
                    for (size_t kk = 0; kk < in.phiPred.size(); ++kk)
                        if (in.phiPred[kk] == prevBB) { sel = v[in.a[kk]]; break; }
                    phiSet.push_back({id, sel});
                }
                for (auto& ps : phiSet) v[ps.first] = ps.second;
            }
            for (int id : fn.bbs[bb].insts) {
                const Inst& in = fn.vals[id];
                switch (in.op) {
                    case Op::Phi: break;   // already resolved above
                    case Op::ConstI: v[id] = coerce(SI(in.ci), in.ty); break;
                    case Op::ConstF: v[id] = coerce(SF(in.cf), in.ty); break;
                    case Op::Param: {
                        const Type& t = fn.ptypes[in.paramIdx];
                        v[id] = memLoad(reinterpret_cast<int64_t>(args[in.paramIdx]), t);
                        break;
                    }
                    case Op::Tid:    v[id] = SI(tid[in.dim]); break;
                    case Op::Ctaid:  v[id] = SI(ctaid[in.dim]); break;
                    case Op::Ntid:   v[id] = SI(ntid[in.dim]); break;
                    case Op::Nctaid: v[id] = SI(nctaid[in.dim]); break;
                    case Op::Bin:  v[id] = binop(in.s, v[in.a[0]], v[in.a[1]], in.ty); break;
                    case Op::Cmp:  v[id] = cmpop(in.s, v[in.a[0]], v[in.a[1]]); break;
                    case Op::Un:
                        if (in.s == "-") v[id] = in.ty.isFloating() ? coerce(SF(-asF(v[in.a[0]])), in.ty) : coerce(SI(-v[in.a[0]].i), in.ty);
                        else if (in.s == "~") v[id] = coerce(SI(~v[in.a[0]].i), in.ty);
                        else v[id] = SI(asI(v[in.a[0]]) == 0 ? 1 : 0);   // !
                        break;
                    case Op::Sel:  v[id] = (asI(v[in.a[0]]) != 0) ? v[in.a[1]] : v[in.a[2]]; break;
                    case Op::Cast: v[id] = coerce(v[in.a[0]], in.ty); break;
                    case Op::CallMath: v[id] = mathfn(in.s, v[in.a[0]], in.a.size() > 1 ? &v[in.a[1]] : nullptr, in.ty); break;
                    case Op::Load: {
                        int64_t addr = asI(v[in.a[0]]) + asI(v[in.a[1]]) * in.elemBytes;
                        v[id] = memLoad(addr, in.ty); break;
                    }
                    case Op::Store: {
                        int64_t addr = asI(v[in.a[0]]) + asI(v[in.a[1]]) * in.elemBytes;
                        memStore(addr, in.ty, v[in.a[2]]); break;
                    }
                    case Op::Br: prevBB = bb; bb = in.bbT; goto nextblock;
                    case Op::CondBr: prevBB = bb; bb = (asI(v[in.a[0]]) != 0) ? in.bbT : in.bbF; goto nextblock;
                    case Op::Ret: ret = true; goto nextblock;
                }
                done[id] = 1;
            }
            nextblock:
            if (ret) break;
        }
    }
    return true;
}

std::string SsaProgram::dump() const {
    const Fn& fn = p_->fn;
    std::string out;
    for (size_t b = 0; b < fn.bbs.size(); ++b) {
        out += "bb" + std::to_string(b) + ":\n";
        for (int id : fn.bbs[b].insts) out += "  %" + std::to_string(id) + " op=" + std::to_string((int)fn.vals[id].op) + "\n";
    }
    return out;
}

}  // namespace frontend
}  // namespace compiler
}  // namespace vgre
