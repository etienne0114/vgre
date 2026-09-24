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

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Tier-2 native machine-code emission is available on x86-64 Linux (SysV ABI, mmap
// W^X). Everywhere else the portable reference evaluator runs the SSA, so Tier-2
// still works — just not as native code.
#if defined(__x86_64__) && defined(__linux__)
#include <sys/mman.h>
#define VGRE_SSA_X64 1
#else
#define VGRE_SSA_X64 0
#endif

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
    std::vector<std::pair<int, int>> loops;   // (breakTarget, continueTarget) per enclosing loop

    explicit Lowerer(const Kernel& kern) : k(kern) {}
    void fail(const std::string& m) { if (ok) { ok = false; err = m; } }

    int newBlock(bool seal = false) {
        fn.bbs.push_back({}); fn.preds.push_back({}); sealed.push_back(seal ? 1 : 0);
        return (int)fn.bbs.size() - 1;
    }
    void addEdge(int from, int to) { (void)from; fn.preds[to].push_back(from); }
    int emit(Inst in) { int id = (int)fn.vals.size(); fn.vals.push_back(std::move(in)); fn.bbs[cur].insts.push_back(id); return id; }
    void emitBr(int target) { Inst in; in.op = Op::Br; in.bbT = target; emit(std::move(in)); }

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
                // header(cond) → body → latch(incr) → back to header; break→exit, continue→latch.
                if (s.kind == Stmt::For && s.forInit) { lowerStmt(*s.forInit); if (!ok) return; }
                int header = newBlock(), body = newBlock(), latch = newBlock(), exit = newBlock();
                emitBr(header); addEdge(cur, header);
                cur = header;                              // unsealed until the back-edge
                const Expr* cond = (s.kind == Stmt::For) ? s.forCond.get() : s.expr.get();
                int c = cond ? lowerExpr(*cond) : constI(1, scalar(Type::Int));
                if (!ok) return;
                Inst cb; cb.op = Op::CondBr; cb.a = {c}; cb.bbT = body; cb.bbF = exit; emit(std::move(cb));
                addEdge(header, body); addEdge(header, exit);
                sealBlock(body);                           // body's only pred is header
                cur = body;
                loops.push_back({exit, latch});
                for (auto& st : s.body) { lowerStmt(*st); if (!ok) return; }
                loops.pop_back();
                emitBr(latch); addEdge(cur, latch);        // normal fall-through to the latch
                sealBlock(latch);                          // preds: body fall-through + any `continue`s
                cur = latch;
                if (s.kind == Stmt::For && s.forIncr) { lowerExpr(*s.forIncr); if (!ok) return; }
                emitBr(header); addEdge(cur, header);       // back-edge
                sealBlock(header);                          // preds complete → fill header phis
                sealBlock(exit);                            // preds: header false-edge + any `break`s
                cur = exit;
                return;
            }
            case Stmt::DoWhile: {
                // body → cond; cond true → body (back-edge); break→exit, continue→cond.
                int body = newBlock(), condB = newBlock(), exit = newBlock();
                emitBr(body); addEdge(cur, body);
                cur = body;                                // unsealed until the back-edge
                loops.push_back({exit, condB});
                for (auto& st : s.body) { lowerStmt(*st); if (!ok) return; }
                loops.pop_back();
                emitBr(condB); addEdge(cur, condB);
                sealBlock(condB);                          // preds: body fall-through + continues
                cur = condB;
                int c = s.expr ? lowerExpr(*s.expr) : constI(1, scalar(Type::Int));
                if (!ok) return;
                Inst cb; cb.op = Op::CondBr; cb.a = {c}; cb.bbT = body; cb.bbF = exit; emit(std::move(cb));
                addEdge(condB, body); addEdge(condB, exit);
                sealBlock(body);                           // preds: pre-loop + condB back-edge
                sealBlock(exit);                           // preds: condB + breaks
                cur = exit;
                return;
            }
            case Stmt::Break:
            case Stmt::Continue: {
                if (loops.empty()) { fail("SSA: break/continue outside a loop"); return; }
                int tgt = (s.kind == Stmt::Break) ? loops.back().first : loops.back().second;
                emitBr(tgt); addEdge(cur, tgt);
                cur = newBlock(/*seal=*/true);             // fresh unreachable block for any trailing stmts
                return;
            }
            default: fail("SSA: unsupported statement (switch → later increment)"); return;
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

// Dominator sets by iterative dataflow (small CFGs): dom[b][d] ⇔ d dominates b.
static std::vector<std::vector<char>> computeDom(const Fn& fn) {
    const int n = (int)fn.bbs.size();
    std::vector<std::vector<char>> dom(n, std::vector<char>(n, 1));
    for (int d = 0; d < n; ++d) dom[fn.entry][d] = (d == fn.entry) ? 1 : 0;
    bool changed = true;
    while (changed) {
        changed = false;
        for (int b = 0; b < n; ++b) {
            if (b == fn.entry) continue;
            std::vector<char> inter(n, 1); bool any = false;
            for (int p : fn.preds[b]) { any = true; for (int d = 0; d < n; ++d) inter[d] = inter[d] && dom[p][d]; }
            if (!any) std::fill(inter.begin(), inter.end(), 0);
            inter[b] = 1;
            if (inter != dom[b]) { dom[b] = std::move(inter); changed = true; }
        }
    }
    return dom;
}

// Loop-invariant code motion: hoist pure instructions whose operands are all defined
// outside the loop (or are themselves invariant) into the loop's preheader. Speculative
// execution of pure ops is safe (no traps: /0 yields 0, float ops don't fault).
static void licm(Fn& fn) {
    const int n = (int)fn.bbs.size();
    auto dom = computeDom(fn);
    std::vector<int> blockOf(fn.vals.size(), -1);
    for (int b = 0; b < n; ++b) for (int id : fn.bbs[b].insts) blockOf[id] = b;
    for (int h = 0; h < n; ++h) {
        std::vector<int> latches;                       // preds of h dominated by h ⇒ back-edges
        for (int p : fn.preds[h]) if (dom[p][h]) latches.push_back(p);
        if (latches.empty()) continue;
        std::vector<char> inLoop(n, 0); inLoop[h] = 1;
        std::vector<int> stk;
        for (int l : latches) if (!inLoop[l]) { inLoop[l] = 1; stk.push_back(l); }
        while (!stk.empty()) { int b = stk.back(); stk.pop_back(); for (int p : fn.preds[b]) if (!inLoop[p]) { inLoop[p] = 1; stk.push_back(p); } }
        int preheader = -1, phCount = 0;                // h's unique out-of-loop pred
        for (int p : fn.preds[h]) if (!inLoop[p]) { preheader = p; ++phCount; }
        if (phCount != 1) continue;                     // not a simple structured loop
        auto inLoopDef = [&](int id) { return blockOf[id] >= 0 && inLoop[blockOf[id]]; };
        std::vector<char> inv(fn.vals.size(), 0);
        bool changed = true;
        while (changed) {
            changed = false;
            for (int b = 0; b < n; ++b) {
                if (!inLoop[b]) continue;
                for (int id : fn.bbs[b].insts) {
                    const Inst& in = fn.vals[id];
                    const bool pure = in.op == Op::Bin || in.op == Op::Un || in.op == Op::Cmp ||
                                      in.op == Op::Cast || in.op == Op::Sel || in.op == Op::ConstI || in.op == Op::ConstF;
                    if (!pure || inv[id]) continue;
                    bool allInv = true;
                    for (int op : in.a) if (inLoopDef(op) && !inv[op]) { allInv = false; break; }
                    if (allInv) { inv[id] = 1; changed = true; }
                }
            }
        }
        std::vector<int> hoist;
        for (int b = 0; b < n; ++b) {
            if (!inLoop[b]) continue;
            std::vector<int> keep;
            for (int id : fn.bbs[b].insts) { if (inv[id]) { hoist.push_back(id); blockOf[id] = preheader; } else keep.push_back(id); }
            fn.bbs[b].insts = std::move(keep);
        }
        std::sort(hoist.begin(), hoist.end());          // SSA ids increase with def order ⇒ defs precede uses
        auto& ph = fn.bbs[preheader].insts;
        size_t at = ph.empty() ? 0 : ph.size() - 1;     // before the preheader's terminator
        ph.insert(ph.begin() + at, hoist.begin(), hoist.end());
    }
}

static void runOpt(Fn& fn) { constFold(fn); localGvn(fn); licm(fn); constFold(fn); dce(fn); }

// ── Tier-2 native machine-code emission (x86-64 / Linux) ──────────────────────────
#if VGRE_SSA_X64
// Per-thread launch context. `pvals` holds one 8-byte value per kernel parameter,
// pre-decoded by the launcher (pointer as-is; int sign/zero-extended; float/double as
// the double's bit pattern) so the emitted code reads a uniform slot. `idx` is
// tid[3],ctaid[3],ntid[3],nctaid[3].
struct ThreadCtx { const int64_t* pvals; uint32_t idx[12]; };

// Bit-exact C-ABI helpers for the ops the emitter delegates (same math the evaluator
// uses), so the native result matches the reference tier exactly.
extern "C" {
int64_t vgre_ssa_sat(double d, int base, int uns) { Type t; t.base = (Type::Base)base; t.isUnsigned = uns != 0; return satFloatToInt(d, t); }
int64_t vgre_ssa_idiv(int64_t a, int64_t bb, int op, int uns) {   // op 0=div 1=mod
    if (bb == 0) return 0;
    if (uns) return (int64_t)(op ? (uint64_t)a % (uint64_t)bb : (uint64_t)a / (uint64_t)bb);
    if (bb == -1 && a == INT64_MIN) return op ? 0 : INT64_MIN;
    return op ? a % bb : a / bb;
}
int vgre_ssa_fcmp(int pred, double a, double b) {
    switch (pred) { case 0: return a < b; case 1: return a <= b; case 2: return a > b;
                    case 3: return a >= b; case 4: return a == b; default: return a != b; }
}
double vgre_ssa_m1(int fn, double x) {
    switch (fn) { case 0: return std::sqrt(x); case 1: return std::fabs(x); case 2: return std::exp(x);
                  case 3: return std::log(x); case 4: return std::sin(x); case 5: return std::cos(x);
                  case 6: return std::floor(x); case 7: return std::ceil(x); case 8: return std::tanh(x); }
    return x;
}
double vgre_ssa_m2(int fn, double x, double y) { return fn ? std::fmax(x, y) : std::fmin(x, y); }
}

using SsaFn = void (*)(ThreadCtx*);

// A minimal x86-64 encoder + slot-based SSA→machine-code lowering. Every SSA value
// lives in an 8-byte rbp-relative slot; each instruction loads operands into fixed
// scratch registers (rax/rcx, xmm0/xmm1), computes, and stores its result. Hot ops
// are inlined; the tricky ones call the helpers above via the SysV ABI.
struct X64Asm {
    std::vector<uint8_t> c;
    const Fn& fn;
    int nvals, frame = 0;
    std::vector<size_t> off;                       // block start offsets (filled while emitting)
    std::vector<std::pair<size_t, int>> jpatch;    // (rel32 site, target block)
    bool ok = true;
    // Redundant-load elimination: the SSA value currently known to be live in rax /
    // xmm0 (from the immediately-preceding op's result), or -1. A first-operand load
    // of that value can be skipped. Invalidated at every block boundary and consumed
    // (set to -1) the moment it is used, so it can never elide a load unsafely.
    int cRax = -1, cXmm = -1;
    void loadG0(int id) { if (cRax == id) { cRax = -1; return; } ldG(0, id); cRax = -1; }
    void loadX0(int id) { if (cXmm == id) { cXmm = -1; return; } ldX(0, id); cXmm = -1; }
    // Record where an op left its result — precisely, per op (some "float" ops leave
    // the value in rax as bits, not xmm0, so the cache must not claim xmm0 for those).
    void setResultCache(const Inst& in, int id) {
        auto inRax = [&] { cRax = id; cXmm = -1; };
        auto inXmm = [&] { cXmm = id; cRax = -1; };
        auto none  = [&] { cRax = cXmm = -1; };
        switch (in.op) {
            case Op::ConstI: case Op::Tid: case Op::Ctaid: case Op::Ntid: case Op::Nctaid: case Op::Cmp: inRax(); break;
            case Op::Param: in.ty.isFloating() ? none() : inRax(); break;
            case Op::Bin:   in.ty.isFloating() ? inXmm() : inRax(); break;
            case Op::Un:    (in.s == "!" || !in.ty.isFloating()) ? inRax() : none(); break;
            case Op::Sel:   in.ty.isFloating() ? none() : inRax(); break;
            case Op::Cast:  case Op::Load: in.ty.isFloating() ? inXmm() : inRax(); break;
            case Op::CallMath: inXmm(); break;
            default: none(); break;   // ConstF (bits in rax), Store, Phi
        }
    }

    explicit X64Asm(const Fn& f) : fn(f), nvals((int)f.vals.size()), vreg((size_t)f.vals.size(), -1) {}
    void bad() { ok = false; }

    // Global linear-scan register allocation (Poletto & Sarkar). Integer/pointer
    // values get a callee-saved register r12–r15 for their whole live range, ACROSS
    // blocks and loop iterations (e.g. a pointer param stays in a register through the
    // loop instead of being reloaded each iteration); the rest use memory slots.
    // Callee-saved ⇒ safe across the emitter's helper calls (no spill-around-call). The
    // emitter accesses every value through ldG/stG (register or slot), so this pass
    // alone enables it. Live intervals are conservative (a register is never freed
    // before the value's last live position). Phi values are kept in slots for now —
    // allocating them to registers needs the interval to also cover the edge-copy
    // writes at every predecessor terminator, which has an unresolved overlap case on
    // nested loop+conditional CFGs; a correct phi allocation is the next increment.
    void allocateRegs() {
        const int n = (int)fn.bbs.size();
        std::vector<int> pos(nvals, -1), firstPos(n, 0), lastPos(n, 0);
        int p = 0;
        for (int b = 0; b < n; ++b) { firstPos[b] = p; for (int id : fn.bbs[b].insts) pos[id] = p++; lastPos[b] = p - 1; }
        std::vector<int> defBlk(nvals, -1);
        for (int b = 0; b < n; ++b) for (int id : fn.bbs[b].insts) defBlk[id] = b;
        auto succOf = [&](int b) {
            std::vector<int> s; const Inst& t = fn.vals[fn.bbs[b].insts.back()];
            if (t.op == Op::Br) s.push_back(t.bbT);
            else if (t.op == Op::CondBr) { s.push_back(t.bbT); s.push_back(t.bbF); }
            return s;
        };
        // Backward liveness dataflow (per-block live-in / live-out value sets).
        std::vector<std::unordered_set<int>> liveIn(n), liveOut(n);
        for (bool changed = true; changed;) {
            changed = false;
            for (int b = n - 1; b >= 0; --b) {
                std::unordered_set<int> out;
                for (int s : succOf(b)) {
                    for (int v : liveIn[s]) if (!(fn.vals[v].op == Op::Phi && defBlk[v] == s)) out.insert(v);   // phi defs aren't live-in from here
                    for (int id : fn.bbs[s].insts) {                                                          // phi operands from b are live-out of b
                        const Inst& in = fn.vals[id];
                        if (in.op != Op::Phi) break;
                        for (size_t k = 0; k < in.phiPred.size(); ++k) if (in.phiPred[k] == b) out.insert(in.a[k]);
                    }
                }
                std::unordered_set<int> live = out;
                const auto& insts = fn.bbs[b].insts;
                for (int i = (int)insts.size() - 1; i >= 0; --i) {                        // walk the block backward
                    const Inst& in = fn.vals[insts[i]];
                    live.erase(insts[i]);                                                 // def
                    if (in.op != Op::Phi) for (int op : in.a) live.insert(op);            // uses (phi operands are edge uses, not here)
                }
                if (live != liveIn[b] || out != liveOut[b]) { liveIn[b] = std::move(live); liveOut[b] = std::move(out); changed = true; }
            }
        }
        // Live intervals: start = def; end = max over all uses + all blocks the value
        // is live-out of (their last position). Conservative single interval per value.
        std::vector<int> start(nvals, 0), end(nvals, 0);
        for (int id = 0; id < nvals; ++id) { start[id] = pos[id] < 0 ? 0 : pos[id]; end[id] = start[id]; }
        for (int b = 0; b < n; ++b) for (int id : fn.bbs[b].insts) {
            const Inst& in = fn.vals[id];
            if (in.op == Op::Phi) {
                // A phi is written by its edge-copies at EVERY predecessor's terminator
                // (the back-edge latch comes after the phi's own position), and read at
                // its uses. Its register must stay reserved across all of that, so its
                // interval spans from the earliest to the latest predecessor terminator.
                for (size_t k = 0; k < in.phiPred.size(); ++k) {
                    int op = in.a[k], predEnd = lastPos[in.phiPred[k]];
                    end[op]  = std::max(end[op], predEnd);          // operand: live until the copy
                    start[id] = std::min(start[id], predEnd);       // phi: written at the copy
                    end[id]   = std::max(end[id], predEnd);
                }
            } else for (int op : in.a) end[op] = std::max(end[op], pos[id]);
        }
        for (int b = 0; b < n; ++b) for (int v : liveOut[b]) end[v] = std::max(end[v], lastPos[b]);
        // Linear scan over intervals sorted by start.
        std::vector<int> order;
        for (int id = 0; id < nvals; ++id) {
            const Inst& in = fn.vals[id];
            const bool intType = in.ty.base == Type::Int || in.ty.base == Type::Long || in.ty.base == Type::Char ||
                                 in.ty.base == Type::Short || in.ty.base == Type::Bool || in.ty.isPointer();
            if (intType && in.op != Op::Phi && pos[id] >= 0 && end[id] > start[id]) order.push_back(id);   // has a real interval
        }
        std::sort(order.begin(), order.end(), [&](int a, int b) { return start[a] < start[b]; });
        std::vector<int> freeRegs = {15, 14, 13, 12};
        std::vector<std::pair<int, int>> active;   // (end, id)
        for (int id : order) {
            for (size_t a = 0; a < active.size();) {
                if (active[a].first < start[id]) { freeRegs.push_back(vreg[active[a].second]); active.erase(active.begin() + a); }
                else ++a;
            }
            if (!freeRegs.empty()) { vreg[id] = freeRegs.back(); freeRegs.pop_back(); active.push_back({end[id], id}); }
        }
    }
    void b(uint8_t x) { c.push_back(x); }
    void d32(uint32_t v) { for (int i = 0; i < 4; ++i) b((uint8_t)(v >> (8 * i))); }
    void d64(uint64_t v) { for (int i = 0; i < 8; ++i) b((uint8_t)(v >> (8 * i))); }

    // Slots sit below the 4 callee-saved GPRs (r12–r15) and rbx pushed in the prologue.
    int slot(int id) const { return -(48 + id * 8); }
    int temp(int i) const { return -(48 + (nvals + i) * 8); }
    void modRbp(int rg, int disp) { b((uint8_t)(0x80 | ((rg & 7) << 3) | 5)); d32((uint32_t)disp); }

    // mov dst, src (64-bit), for registers 0..15 (REX.R for a high src, REX.B for a high dst).
    void movReg(int dst, int src) {
        b((uint8_t)(0x48 | ((src >= 8) ? 0x04 : 0) | ((dst >= 8) ? 0x01 : 0)));
        b(0x89); b((uint8_t)(0xC0 | ((src & 7) << 3) | (dst & 7)));
    }
    std::vector<int> vreg;   // per value: a register 12..15 if allocated, else -1 (slot)

    // GPR (64-bit) and XMM (double) load/store to an rbp-relative displacement.
    void ldGd(int rg, int disp) { b(0x48); b(0x8B); modRbp(rg, disp); }
    void stGd(int rg, int disp) { b(0x48); b(0x89); modRbp(rg, disp); }
    void ldXd(int x, int disp)  { b(0xF2); b(0x0F); b(0x10); modRbp(x, disp); }
    void stXd(int x, int disp)  { b(0xF2); b(0x0F); b(0x11); modRbp(x, disp); }
    // A register-resident value is copied to/from scratch; otherwise it uses its slot.
    void ldG(int rg, int id) { if (vreg[id] >= 0) movReg(rg, vreg[id]); else ldGd(rg, slot(id)); }
    void stG(int rg, int id) { if (vreg[id] >= 0) movReg(vreg[id], rg); else stGd(rg, slot(id)); }
    void ldX(int x, int id)  { ldXd(x, slot(id)); }   // floats are never register-allocated
    void stX(int x, int id)  { stXd(x, slot(id)); }

    void movImm(int rg, uint64_t v) { b(0x48); b((uint8_t)(0xB8 | (rg & 7))); d64(v); }
    void movImm32(int rg, uint32_t v) { b(0x48); b(0xC7); b((uint8_t)(0xC0 | (rg & 7))); d32(v); }
    void rr(uint8_t op, int dst, int src) { b(0x48); b(op); b((uint8_t)(0xC0 | ((src & 7) << 3) | (dst & 7))); }
    void addRR(int d, int s) { rr(0x01, d, s); }
    void subRR(int d, int s) { rr(0x29, d, s); }
    void andRR(int d, int s) { rr(0x21, d, s); }
    void orRR(int d, int s)  { rr(0x09, d, s); }
    void xorRR(int d, int s) { rr(0x31, d, s); }
    void movRR(int d, int s) { rr(0x89, d, s); }
    void cmpRR(int d, int s) { rr(0x39, d, s); }
    void testRR(int d, int s) { rr(0x85, d, s); }
    void imulRR(int d, int s) { b(0x48); b(0x0F); b(0xAF); b((uint8_t)(0xC0 | ((d & 7) << 3) | (s & 7))); }
    void negR(int r) { b(0x48); b(0xF7); b((uint8_t)(0xD8 | (r & 7))); }
    void notR(int r) { b(0x48); b(0xF7); b((uint8_t)(0xD0 | (r & 7))); }
    void shlCl(int r) { b(0x48); b(0xD3); b((uint8_t)(0xE0 | (r & 7))); }
    void sarCl(int r) { b(0x48); b(0xD3); b((uint8_t)(0xF8 | (r & 7))); }
    void shrCl(int r) { b(0x48); b(0xD3); b((uint8_t)(0xE8 | (r & 7))); }
    void movRAXtoRCX() { movRR(1, 0); }
    // signed/unsigned extend eax's low `bytes` into rax
    void extRax(int bytes, bool uns) {
        if (bytes >= 8) return;
        if (bytes == 4) { if (uns) { b(0x89); b(0xC0); } else { b(0x48); b(0x63); b(0xC0); } return; }
        b(0x0F); b((uint8_t)(bytes == 1 ? (uns ? 0xB6 : 0xBE) : (uns ? 0xB7 : 0xBF))); b(0xC0);
    }
    // float: xmm0 op= xmm1 (add 58 / sub 5C / mul 59 / div 5E)
    void fArith0(uint8_t op) { b(0xF2); b(0x0F); b(op); b(0xC1); }
    void cvtsi2sd0() { b(0xF2); b(0x48); b(0x0F); b(0x2A); b(0xC0); }   // xmm0 = (double)rax
    void cvtsd2ss0() { b(0xF2); b(0x0F); b(0x5A); b(0xC0); }            // xmm0 = (float)xmm0
    void cvtss2sd0() { b(0xF3); b(0x0F); b(0x5A); b(0xC0); }
    void movqX0R(int r) { b(0x66); b(0x48); b(0x0F); b(0x6E); b((uint8_t)(0xC0 | r)); }   // xmm0 = rax bits (r=0)
    void movqRX0() { b(0x66); b(0x48); b(0x0F); b(0x7E); b(0xC0); }     // rax = xmm0 bits
    // set rax = (rax <cc> 0)?1:0 after a cmp (cc byte for setcc)
    void setcc(uint8_t cc) { b(0x0F); b(cc); b(0xC0); b(0x0F); b(0xB6); b(0xC0); }   // setcc al; movzx eax,al
    void call(uint64_t addr) { movImm(0, addr); b(0xFF); b(0xD0); }    // mov rax,addr; call rax
    // mov reg32, imm32 (zero-extends into reg64) — for small integer helper args.
    void movImmReg(int rg, uint32_t v) { if (rg >= 8) b(0x41); b((uint8_t)(0xB8 | (rg & 7))); d32(v); }

    bool isFloatVal(int id) const { return fn.vals[id].ty.isFloating(); }
    void jmpBlock(int blk) { b(0xE9); jpatch.push_back({c.size(), blk}); d32(0); }

    // Phi edge-copies pred→succ: two-phase (operands→temps, temps→phi slots).
    void phiCopies(int pred, int succ) {
        std::vector<std::pair<int, int>> pr;   // (phiId, operandId)
        for (int id : fn.bbs[succ].insts) {
            const Inst& in = fn.vals[id];
            if (in.op != Op::Phi) break;
            for (size_t k = 0; k < in.phiPred.size(); ++k) if (in.phiPred[k] == pred) { pr.push_back({id, in.a[k]}); break; }
        }
        for (size_t i = 0; i < pr.size(); ++i) { ldG(0, pr[i].second); stGd(0, temp((int)i)); }
        for (size_t i = 0; i < pr.size(); ++i) { ldGd(0, temp((int)i)); stG(0, pr[i].first); }
    }

    int mathId(const std::string& s) const {
        std::string g = (!s.empty() && s.back() == 'f') ? s.substr(0, s.size() - 1) : s;
        if (g == "sqrt") return 0; if (g == "fabs") return 1; if (g == "exp") return 2; if (g == "log") return 3;
        if (g == "sin") return 4; if (g == "cos") return 5; if (g == "floor") return 6; if (g == "ceil") return 7;
        if (g == "tanh") return 8; return -1;
    }
    void narrowIfFloat(const Type& t) { if (t.base == Type::Float) { cvtsd2ss0(); cvtss2sd0(); } }   // xmm0 → float32-rounded

    void emitInst(int blk, int id) {
        (void)blk;
        const Inst& in = fn.vals[id];
        switch (in.op) {
            case Op::Phi: return;
            case Op::ConstI: { movImm(0, (uint64_t)coerce(SI(in.ci), in.ty).i); stG(0, id); return; }
            case Op::ConstF: { double d = coerce(SF(in.cf), in.ty).d; uint64_t bits; std::memcpy(&bits, &d, 8); movImm(0, bits); stG(0, id); return; }
            case Op::Param: { b(0x48); b(0x8B); b(0x03);                          // mov rax,[rbx]  (pvals)
                              b(0x48); b(0x8B); b(0x80); d32((uint32_t)(in.paramIdx * 8));   // mov rax,[rax+p*8]
                              stG(0, id); return; }
            case Op::Tid: case Op::Ctaid: case Op::Ntid: case Op::Nctaid: {
                int base = in.op == Op::Tid ? 0 : in.op == Op::Ctaid ? 3 : in.op == Op::Ntid ? 6 : 9;
                b(0x8B); b(0x83); d32((uint32_t)(8 + (base + in.dim) * 4));       // mov eax,[rbx+off] (zero-extends)
                stG(0, id); return;
            }
            case Op::Bin: {
                if (in.ty.isFloating()) {
                    loadX0(in.a[0]); ldX(1, in.a[1]);
                    uint8_t op = in.s == "+" ? 0x58 : in.s == "-" ? 0x5C : in.s == "*" ? 0x59 : in.s == "/" ? 0x5E : 0;
                    if (!op) { bad(); return; }
                    fArith0(op); narrowIfFloat(in.ty); stX(0, id); return;
                }
                const std::string& o = in.s;
                if (o == "/" || o == "%") {
                    ldG(7, in.a[0]); ldG(6, in.a[1]);                             // rdi=a, rsi=b
                    movImmReg(2, o == "%" ? 1 : 0); movImmReg(1, in.ty.isUnsigned ? 1 : 0);   // edx=op, ecx=uns
                    call((uint64_t)&vgre_ssa_idiv); stG(0, id); return;
                }
                if (o == "&&" || o == "||") {
                    ldG(0, in.a[0]); testRR(0, 0); setcc(0x95); stGd(0, temp(0));  // (a!=0)
                    ldG(0, in.a[1]); testRR(0, 0); setcc(0x95); ldGd(1, temp(0));  // rax=(b!=0), rcx=(a!=0)
                    if (o == "&&") andRR(0, 1); else orRR(0, 1); stG(0, id); return;
                }
                loadG0(in.a[0]); ldG(1, in.a[1]);
                if (o == "+") addRR(0, 1); else if (o == "-") subRR(0, 1); else if (o == "*") imulRR(0, 1);
                else if (o == "&") andRR(0, 1); else if (o == "|") orRR(0, 1); else if (o == "^") xorRR(0, 1);
                else if (o == "<<") shlCl(0); else if (o == ">>") { if (in.ty.isUnsigned) shrCl(0); else sarCl(0); }
                else { bad(); return; }
                extRax(in.ty.elemBytes(), in.ty.isUnsigned); stG(0, id); return;
            }
            case Op::Cmp: {
                bool fl = isFloatVal(in.a[0]) || isFloatVal(in.a[1]);
                const std::string& o = in.s;
                int pred = o == "<" ? 0 : o == "<=" ? 1 : o == ">" ? 2 : o == ">=" ? 3 : o == "==" ? 4 : 5;
                if (fl) {
                    movImmReg(7, (uint32_t)pred); loadX0(in.a[0]); ldX(1, in.a[1]);
                    call((uint64_t)&vgre_ssa_fcmp); b(0x48); b(0x63); b(0xC0); stG(0, id); return;   // movsxd rax,eax
                }
                loadG0(in.a[0]); ldG(1, in.a[1]); cmpRR(0, 1);
                uint8_t cc = o == "<" ? 0x9C : o == "<=" ? 0x9E : o == ">" ? 0x9F : o == ">=" ? 0x9D : o == "==" ? 0x94 : 0x95;
                setcc(cc); stG(0, id); return;
            }
            case Op::Un: {
                if (in.s == "!") { ldG(0, in.a[0]); testRR(0, 0); setcc(0x94); stG(0, id); return; }
                if (in.ty.isFloating()) { ldG(0, in.a[0]); movImm(1, 0x8000000000000000ull); xorRR(0, 1); stG(0, id); return; }  // sign flip
                ldG(0, in.a[0]); if (in.s == "-") negR(0); else notR(0); extRax(in.ty.elemBytes(), in.ty.isUnsigned); stG(0, id); return;
            }
            case Op::Sel: {
                ldG(0, in.a[2]); stG(0, id);                                     // res = else
                ldG(0, in.a[0]); testRR(0, 0);
                b(0x0F); b(0x84); size_t js = c.size(); d32(0);                  // jz skip
                ldG(0, in.a[1]); stG(0, id);                                     // res = then
                uint32_t rel = (uint32_t)(c.size() - (js + 4)); std::memcpy(&c[js], &rel, 4);
                return;
            }
            case Op::Cast: {
                const Type& t = in.ty; int op0 = in.a[0];
                if (t.isFloating()) {
                    if (isFloatVal(op0)) { ldX(0, op0); narrowIfFloat(t); } else { ldG(0, op0); cvtsi2sd0(); narrowIfFloat(t); }
                    stX(0, id); return;
                }
                if (t.isPointer()) { ldG(0, op0); stG(0, id); return; }
                if (isFloatVal(op0)) {                                           // float → int (saturating helper)
                    ldX(0, op0); movImmReg(7, (uint32_t)t.base); movImmReg(6, t.isUnsigned ? 1 : 0);
                    call((uint64_t)&vgre_ssa_sat); stG(0, id); return;
                }
                ldG(0, op0); extRax(t.elemBytes(), t.isUnsigned); stG(0, id); return;   // int → int (width wrap)
            }
            case Op::Load: {
                loadG0(in.a[0]); ldG(1, in.a[1]); movImm(2, (uint64_t)in.elemBytes); imulRR(1, 2); addRR(0, 1);  // rax = base + idx*bytes
                const Type& t = in.ty;
                if (t.isFloating()) {
                    if (t.elemBytes() == 8) { b(0xF2); b(0x0F); b(0x10); b(0x00); } else { b(0xF3); b(0x0F); b(0x10); b(0x00); cvtss2sd0(); }
                    stX(0, id); return;
                }
                if (t.isPointer() || t.elemBytes() == 8) { b(0x48); b(0x8B); b(0x00); }
                else if (t.elemBytes() == 4) { if (t.isUnsigned) { b(0x8B); b(0x00); } else { b(0x48); b(0x63); b(0x00); } }
                else if (t.elemBytes() == 2) { b(0x0F); b(t.isUnsigned ? 0xB7 : 0xBF); b(0x00); }
                else { b(0x0F); b(t.isUnsigned ? 0xB6 : 0xBE); b(0x00); }
                stG(0, id); return;
            }
            case Op::Store: {
                ldG(2, in.a[0]); ldG(1, in.a[1]); movImm(0, (uint64_t)in.elemBytes); imulRR(1, 0); addRR(2, 1);  // rdx = base + idx*bytes
                const Type& t = in.ty;
                if (t.isFloating()) {
                    ldX(0, in.a[2]);
                    if (t.elemBytes() == 8) { b(0xF2); b(0x0F); b(0x11); b(0x02); } else { cvtsd2ss0(); b(0xF3); b(0x0F); b(0x11); b(0x02); }
                    return;
                }
                ldG(0, in.a[2]);
                if (t.isPointer() || t.elemBytes() == 8) { b(0x48); b(0x89); b(0x02); }
                else if (t.elemBytes() == 4) { b(0x89); b(0x02); }
                else if (t.elemBytes() == 2) { b(0x66); b(0x89); b(0x02); }
                else { b(0x88); b(0x02); }
                return;
            }
            case Op::CallMath: {
                if (in.a.size() == 1) { ldX(0, in.a[0]); movImmReg(7, (uint32_t)mathId(in.s)); call((uint64_t)&vgre_ssa_m1); }
                else { ldX(0, in.a[0]); ldX(1, in.a[1]); movImmReg(7, (in.s == "fmax" || in.s == "fmaxf") ? 1u : 0u); call((uint64_t)&vgre_ssa_m2); }
                narrowIfFloat(in.ty); stX(0, id); return;
            }
            default: return;   // terminators handled by the block loop
        }
    }

    void emitEpilogue() {
        b(0x48); b(0x81); b(0xC4); d32((uint32_t)frame);                        // add rsp,frame
        b(0x41); b(0x5F); b(0x41); b(0x5E); b(0x41); b(0x5D); b(0x41); b(0x5C);  // pop r15; r14; r13; r12
        b(0x5B); b(0x5D); b(0xC3);                                              // pop rbx; pop rbp; ret
    }

    bool build() {
        allocateRegs();
        const int n = (int)fn.bbs.size();
        int maxTemps = 0;
        for (auto& bb : fn.bbs) { int p = 0; for (int id : bb.insts) { if (fn.vals[id].op == Op::Phi) ++p; else break; } if (p > maxTemps) maxTemps = p; }
        const int need = (nvals + maxTemps) * 8;
        frame = ((need + 15) / 16) * 16 + 8;         // keep rsp 16-aligned at calls
        b(0x55); b(0x48); b(0x89); b(0xE5); b(0x53);  // push rbp; mov rbp,rsp; push rbx
        b(0x41); b(0x54); b(0x41); b(0x55); b(0x41); b(0x56); b(0x41); b(0x57);   // push r12; r13; r14; r15
        b(0x48); b(0x81); b(0xEC); d32((uint32_t)frame);   // sub rsp,frame
        b(0x48); b(0x89); b(0xFB);                    // mov rbx,rdi
        off.assign(n, 0);
        for (int blk = 0; blk < n; ++blk) {
            off[blk] = c.size();
            cRax = cXmm = -1;                                     // register cache is per-block
            const BB& bb = fn.bbs[blk];
            for (size_t j = 0; j < bb.insts.size(); ++j) {
                int id = bb.insts[j];
                const Inst& in = fn.vals[id];
                if (in.op == Op::Ret) { emitEpilogue(); break; }
                if (in.op == Op::Br) { phiCopies(blk, in.bbT); jmpBlock(in.bbT); break; }
                if (in.op == Op::CondBr) {
                    loadG0(in.a[0]); testRR(0, 0);
                    b(0x0F); b(0x84); size_t js = c.size(); d32(0);            // jz to false edge
                    phiCopies(blk, in.bbT); jmpBlock(in.bbT);
                    uint32_t rel = (uint32_t)(c.size() - (js + 4)); std::memcpy(&c[js], &rel, 4);
                    phiCopies(blk, in.bbF); jmpBlock(in.bbF);
                    break;
                }
                emitInst(blk, id);
                if (!ok) return false;
                setResultCache(in, id);       // update the register cache for the next op
            }
        }
        for (auto& jp : jpatch) { uint32_t rel = (uint32_t)(off[jp.second] - (jp.first + 4)); std::memcpy(&c[jp.first], &rel, 4); }
        return ok;
    }
};
#endif  // VGRE_SSA_X64

}  // namespace

// ── SsaProgram wrapper ────────────────────────────────────────────────────────────
struct SsaProgram::Impl {
    Fn fn;
#if VGRE_SSA_X64
    void* code = nullptr;      // mmap'd W^X machine code (null ⇒ use the evaluator)
    size_t codeSize = 0;
    SsaFn nativeFn = nullptr;
    ~Impl() { if (code) munmap(code, codeSize); }
#endif
};

SsaProgram::SsaProgram() : p_(new Impl) {}
SsaProgram::~SsaProgram() = default;
bool SsaProgram::usedNative() const {
#if VGRE_SSA_X64
    return p_->nativeFn != nullptr;
#else
    return false;
#endif
}
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
#if VGRE_SSA_X64
    // Best-effort: emit native machine code. On any unsupported op the emitter bails
    // and launch() falls back to the (identical-semantics) evaluator.
    {
        X64Asm asmb(prog->p_->fn);
        if (asmb.build() && !asmb.c.empty()) {
            size_t sz = ((asmb.c.size() + 4095) / 4096) * 4096;
            void* mem = mmap(nullptr, sz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (mem != MAP_FAILED) {
                std::memcpy(mem, asmb.c.data(), asmb.c.size());
                if (mprotect(mem, sz, PROT_READ | PROT_EXEC) == 0) {
                    prog->p_->code = mem; prog->p_->codeSize = sz;
                    prog->p_->nativeFn = reinterpret_cast<SsaFn>(mem);
                } else { munmap(mem, sz); }
            }
        }
    }
#endif
    return prog;
}

bool SsaProgram::launch(Extent grid, Extent block, void* const* args, int numArgs) {
    const Fn& fn = p_->fn;
    if (numArgs < (int)fn.ptypes.size()) return false;
    const uint32_t bx = block.x, by = block.y, bz = block.z;
#if VGRE_SSA_X64
    if (p_->nativeFn) {
        // Pre-decode each parameter into a uniform 8-byte value (shared by all threads).
        std::vector<int64_t> pvals(fn.ptypes.size(), 0);
        for (size_t i = 0; i < fn.ptypes.size(); ++i) {
            SVal pv = memLoad(reinterpret_cast<int64_t>(args[i]), fn.ptypes[i]);
            if (fn.ptypes[i].isFloating()) { double d = pv.d; std::memcpy(&pvals[i], &d, 8); }
            else pvals[i] = pv.i;
        }
        ThreadCtx ctx; ctx.pvals = pvals.data();
        for (uint32_t gz = 0; gz < grid.z; ++gz)
        for (uint32_t gy = 0; gy < grid.y; ++gy)
        for (uint32_t gx = 0; gx < grid.x; ++gx)
        for (uint32_t tz = 0; tz < bz; ++tz)
        for (uint32_t ty = 0; ty < by; ++ty)
        for (uint32_t tx = 0; tx < bx; ++tx) {
            ctx.idx[0] = tx; ctx.idx[1] = ty; ctx.idx[2] = tz;
            ctx.idx[3] = gx; ctx.idx[4] = gy; ctx.idx[5] = gz;
            ctx.idx[6] = bx; ctx.idx[7] = by; ctx.idx[8] = bz;
            ctx.idx[9] = grid.x; ctx.idx[10] = grid.y; ctx.idx[11] = grid.z;
            p_->nativeFn(&ctx);
        }
        return true;
    }
#endif
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
        for (long long guard = 0; guard < (1LL << 34); ++guard) {   // bounded against runaway loops (long is 32-bit on Windows)
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
