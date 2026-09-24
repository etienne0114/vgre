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

// macOS gates the POSIX ucontext routines (used by the native cooperative fiber
// scheduler for __shared__/__syncthreads) behind _XOPEN_SOURCE; _DARWIN_C_SOURCE
// re-enables the BSD extensions it would otherwise hide. Both must be set before any
// system header is included (mirrors the compiled tier, which is CI-proven on macos-arm64).
#if defined(__APPLE__)
#  ifndef _XOPEN_SOURCE
#    define _XOPEN_SOURCE 700
#  endif
#  ifndef _DARWIN_C_SOURCE
#    define _DARWIN_C_SOURCE 1
#  endif
#endif

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
#if defined(__x86_64__) && defined(__linux__) && !defined(VGRE_SSA_NO_NATIVE)
#include <sys/mman.h>
#include <ucontext.h>   // ucontext fibers for native __shared__/__syncthreads scheduling
#define VGRE_SSA_X64 1
#else
// VGRE_SSA_NO_NATIVE forces the portable evaluator even on x86-64/Linux — used
// to exercise the evaluator under sanitizers (the mmap'd native code is opaque
// to ASan/UBSan and its call rel32 range interacts with ASan's address layout).
#define VGRE_SSA_X64 0
#endif

// AArch64 native emitter (Apple Silicon + ARM Linux). Its ENCODINGS are unit-tested
// locally against llvm-mc (test_ssa_arm64_enc); EXECUTION is validated on real ARM
// (the macos-arm64 CI job's SsaIr differential test). Because that execution path
// cannot be exercised on an x86-64 dev host, native activation is OPT-IN via
// VGRE_SSA_ARM_NATIVE=1 (see SsaProgram::compile) — off by default so a latent codegen
// bug can never regress the runtime; the portable evaluator runs the SSA otherwise.
#if defined(__aarch64__) && !defined(VGRE_SSA_NO_NATIVE)
#include <sys/mman.h>
#include <cstdlib>
#include <ucontext.h>   // ucontext fibers for native __shared__/__syncthreads scheduling
#if defined(__APPLE__)
#include <pthread.h>                 // pthread_jit_write_protect_np (Apple W^X toggle)
#include <libkern/OSCacheControl.h>  // sys_icache_invalidate (ARM I-cache flush)
#endif
#define VGRE_SSA_ARM64 1
#else
#define VGRE_SSA_ARM64 0
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
    Phi, Bin, Un, Cmp, Sel, Cast, Load, CallMath, LoadL, LoadS,   // value-producing (LoadL/LoadS = local/shared-array read)
    WarpShfl, WarpVote, WarpReduce, WarpMatch,        // value-producing, warp-cooperative (__shfl_*_sync / __ballot|any|all_sync / __reduce_*_sync / __match_any_sync)
    WarpActive,                                       // value-producing, per-thread (__activemask — mask of the warp's in-range lanes)
    Store, StoreL, StoreS, Barrier, CondBr, Br, Ret   // effects / terminators (StoreL/StoreS write; Barrier = __syncthreads/__syncwarp)
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
    int arrId = -1;             // LoadL / StoreL: local-array index (into Fn::localArrays)
};
struct BB { std::vector<int> insts; };   // value-ids, in order; last is a terminator

struct Fn {
    std::vector<Type> ptypes;   // kernel parameter types (pointer or scalar)
    std::vector<Inst> vals;     // all instructions, indexed by SSA id
    std::vector<BB> bbs;
    std::vector<std::vector<int>> preds;   // per-block predecessor block ids
    std::vector<std::pair<Type, int>> localArrays;    // per-thread scratch arrays: (element type, count)
    std::vector<std::pair<Type, int>> sharedArrays;   // block-shared arrays (__shared__): (element type, count)
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

    // __device__ helper inlining: callee locals are alpha-renamed with `mpfx` so they
    // never collide with the caller's names in curDef/vtype; a callee `return` writes
    // the current inline's result var and branches to its exit block (inlineCtx.back()).
    std::unordered_map<std::string, const Kernel*> deviceFns;   // name -> __device__ definition
    std::string mpfx;                          // active mangling prefix ("" at top level)
    std::unordered_map<std::string, std::pair<int, Type>> localArr;    // mangled name -> (localArrays id, element type)
    std::unordered_map<std::string, std::pair<int, Type>> sharedArr;   // name -> (sharedArrays id, element type) — block scope
    int inlineUid = 0, inlineDepth = 0;
    struct InlineCtx { int exitB; std::string retVar; Type rt; };
    std::vector<InlineCtx> inlineCtx;
    std::string mangle(const std::string& n) const { return mpfx.empty() ? n : mpfx + n; }

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
        // readVar() may recursively create phis and grow fn.vals, invalidating any
        // reference into it. Snapshot the preds, then for each: compute the operand
        // FIRST (into a local), and only AFTER that re-index fn.vals[phi] to append.
        // Never write `fn.vals[phi].a.push_back(readVar(...))`: the object glvalue is
        // sequenced before the argument, so a reallocation inside readVar leaves it
        // dangling — benign on clang (materializes `this` after the arg), a
        // use-after-free that corrupts the IR on MSVC (the Windows CI segfault).
        std::vector<int> preds = fn.preds[block];
        for (int p : preds) {
            int operand = readVar(v, p);
            fn.vals[phi].a.push_back(operand);
            fn.vals[phi].phiPred.push_back(p);
        }
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
            case Expr::Ident: { auto it = vtype.find(mangle(e.str)); return it != vtype.end() ? it->second : scalar(Type::Int); }
            case Expr::Index: { std::string bn = e.args.empty() ? std::string() : mangle(e.args[0]->str); auto la = localArr.find(bn); if (la != localArr.end()) return la->second.second; auto sa = sharedArr.find(bn); if (sa != sharedArr.end()) return sa->second.second; Type b = vtype.count(bn) ? vtype[bn] : Type{}; if (b.ptr > 0) b.ptr--; return b; }
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
                std::string nm = mangle(e.str);
                if (!vtype.count(nm)) { fail("SSA: unknown identifier '" + e.str + "'"); return 0; }
                return readVar(nm, cur);
            }
            case Expr::Member: {
                int dim;
                if (member(e, "threadIdx", dim)) { Inst in; in.op = Op::Tid;   in.ty = scalar(Type::Int); in.dim = dim; return emit(std::move(in)); }
                if (member(e, "blockIdx", dim))  { Inst in; in.op = Op::Ctaid; in.ty = scalar(Type::Int); in.dim = dim; return emit(std::move(in)); }
                if (member(e, "blockDim", dim))  { Inst in; in.op = Op::Ntid;  in.ty = scalar(Type::Int); in.dim = dim; return emit(std::move(in)); }
                if (member(e, "gridDim", dim))   { Inst in; in.op = Op::Nctaid;in.ty = scalar(Type::Int); in.dim = dim; return emit(std::move(in)); }
                fail("SSA: unsupported member access"); return 0;
            }
            case Expr::Index: {   // p[idx] — a global load, or a[idx] — a local-array read
                if (e.args.size() != 2 || e.args[0]->kind != Expr::Ident) { fail("SSA: bad index"); return 0; }
                auto la = localArr.find(mangle(e.args[0]->str));
                if (la != localArr.end()) {
                    int idx = lowerExpr(*e.args[1]); if (!ok) return 0;
                    Inst in; in.op = Op::LoadL; in.ty = la->second.second; in.arrId = la->second.first;
                    in.elemBytes = la->second.second.elemBytes(); in.a = {idx};
                    return emit(std::move(in));
                }
                auto sa = sharedArr.find(mangle(e.args[0]->str));
                if (sa != sharedArr.end()) {
                    int idx = lowerExpr(*e.args[1]); if (!ok) return 0;
                    Inst in; in.op = Op::LoadS; in.ty = sa->second.second; in.arrId = sa->second.first;
                    in.elemBytes = sa->second.second.elemBytes(); in.a = {idx};
                    return emit(std::move(in));
                }
                Type pt = vtype.count(mangle(e.args[0]->str)) ? vtype[mangle(e.args[0]->str)] : Type{};
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
                    if (e.args[0]->kind != Expr::Ident || !vtype.count(mangle(e.args[0]->str))) { fail("SSA: ++/-- on a non-local"); return 0; }
                    const std::string nm = mangle(e.args[0]->str); Type vt = vtype[nm];
                    int old = readVar(nm, cur);
                    int one = vt.isFloating() ? constF(1.0, vt) : constI(1, vt);
                    Inst b; b.op = Op::Bin; b.s = (e.str[3] == '+' || e.str[4] == '+') ? "+" : "-"; b.ty = vt; b.a = {old, one};
                    int nv = emit(std::move(b));
                    Inst c; c.op = Op::Cast; c.ty = vt; c.a = {nv}; int cv = emit(std::move(c));
                    writeVar(nm, cur, cv);
                    return (e.str.rfind("pre", 0) == 0) ? cv : old;
                }
                if (e.str != "-" && e.str != "!" && e.str != "~") { fail("SSA: unary '" + e.str + "'"); return 0; }
                if (e.str == "~" && typeOf(*e.args[0]).isFloating()) { fail("SSA: '~' on a floating operand"); return 0; }
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
                Type rt = typeOf(e);   // promoteT of the two arms
                // Coerce BOTH arms to the result type before selecting, so the chosen
                // value is stored/loaded as that type (else e.g. `cond ? intE : floatE`
                // would select an int value that the float Sel path reads as a double).
                Inst ca; ca.op = Op::Cast; ca.ty = rt; ca.a = {a}; a = emit(std::move(ca));
                Inst cb; cb.op = Op::Cast; cb.ty = rt; cb.a = {b}; b = emit(std::move(cb));
                Inst in; in.op = Op::Sel; in.ty = rt; in.a = {c, a, b};
                return emit(std::move(in));
            }
            case Expr::Binary: {
                const std::string& o = e.str;
                // `&&` / `||` yield an int and test each operand's truthiness — which is
                // type-dependent (float 0.0 vs int 0). Lower them as (a!=0) &/| (b!=0) so
                // the comparison handles the operand type correctly (a plain Bin "&&" would
                // be typed by promoteT and mis-evaluated for float operands). Non-short-
                // circuit, but equivalent for the side-effect-free kernel expression subset.
                if (o == "&&" || o == "||") {
                    auto truth = [&](const Expr& e2) -> int {
                        int v = lowerExpr(e2); if (!ok) return 0;
                        Type t = typeOf(e2);
                        int z = t.isFloating() ? constF(0.0, t) : constI(0, t);
                        Inst cm; cm.op = Op::Cmp; cm.s = "!="; cm.ty = scalar(Type::Int); cm.a = {v, z};
                        return emit(std::move(cm));
                    };
                    int ta = truth(*e.args[0]); int tb = truth(*e.args[1]); if (!ok) return 0;
                    Inst in; in.op = Op::Bin; in.s = (o == "&&") ? "&" : "|"; in.ty = scalar(Type::Int); in.a = {ta, tb};
                    return emit(std::move(in));
                }
                int a = lowerExpr(*e.args[0]);
                int b = lowerExpr(*e.args[1]);
                if (!ok) return 0;
                const bool isCmp = (o=="<"||o=="<="||o==">"||o==">="||o=="=="||o=="!=");
                const bool arith = (o=="+"||o=="-"||o=="*"||o=="/");
                const bool intBit = (o=="%"||o=="&"||o=="|"||o=="^"||o=="<<"||o==">>");
                Inst in; in.s = o; in.a = {a, b};
                if (isCmp) { in.op = Op::Cmp; in.ty = scalar(Type::Int); }
                else if (arith || intBit) {
                    in.op = Op::Bin; in.ty = promoteT(typeOf(*e.args[0]), typeOf(*e.args[1]));
                    if (intBit && in.ty.isFloating()) { fail("SSA: bitwise/'%' operator '" + o + "' on floating operands"); return 0; }
                } else { fail("SSA: unsupported binary operator '" + o + "'"); return 0; }   // never emit an op the tiers can't evaluate
                return emit(std::move(in));
            }
            case Expr::Call: {
                const std::string& fnn = e.str;
                if (fnn == "__syncthreads" && e.args.empty()) {   // block barrier
                    Inst in; in.op = Op::Barrier; in.ty = scalar(Type::Int); emit(std::move(in));
                    return constI(0, scalar(Type::Int));   // void; the value is unused (statement context)
                }
                if (e.args.size() == 1 && (fnn=="sqrtf"||fnn=="fabsf"||fnn=="expf"||fnn=="logf"||fnn=="sinf"||fnn=="cosf"||fnn=="floorf"||fnn=="ceilf"||fnn=="tanhf"||fnn=="exp2f"||fnn=="log2f"||fnn=="rsqrtf"||fnn=="erff"||
                                           fnn=="sqrt"||fnn=="fabs"||fnn=="exp"||fnn=="log"||fnn=="sin"||fnn=="cos"||fnn=="floor"||fnn=="ceil"||fnn=="tanh"||fnn=="exp2"||fnn=="log2"||fnn=="rsqrt"||fnn=="erf")) {
                    int a = lowerExpr(*e.args[0]); if (!ok) return 0;
                    Inst in; in.op = Op::CallMath; in.s = fnn; in.ty = typeOf(e); in.a = {a};
                    return emit(std::move(in));
                }
                if ((fnn=="min"||fnn=="max"||fnn=="fminf"||fnn=="fmaxf"||fnn=="fmin"||fnn=="fmax"||fnn=="powf"||fnn=="pow") && e.args.size()==2) {
                    int a = lowerExpr(*e.args[0]); int b = lowerExpr(*e.args[1]); if (!ok) return 0;
                    Inst in; in.op = Op::CallMath; in.s = fnn; in.ty = typeOf(e); in.a = {a, b};
                    return emit(std::move(in));
                }
                if ((fnn=="fmaf"||fnn=="fma") && e.args.size()==3) {
                    int a = lowerExpr(*e.args[0]); int b = lowerExpr(*e.args[1]); int c = lowerExpr(*e.args[2]); if (!ok) return 0;
                    Inst in; in.op = Op::CallMath; in.s = fnn; in.ty = typeOf(e); in.a = {a, b, c};
                    return emit(std::move(in));
                }
                // __syncwarp(): a warp-scoped barrier. A block-wide barrier is a correct
                // superset (it also synchronizes the warp) and exchanges no data, so the
                // evaluator/interpreter results are identical — lower it to Barrier.
                if (fnn == "__syncwarp" && e.args.size() <= 1) {
                    Inst in; in.op = Op::Barrier; in.ty = scalar(Type::Int); emit(std::move(in));
                    return constI(0, scalar(Type::Int));
                }
                // Warp shuffle: __shfl[_up|_down|_xor]_sync(mask, var, laneArg[, width]).
                // mode 0=idx 1=up 2=down 3=xor. The `mask` arg (participation) is honored by
                // the rendezvous of the lanes that actually reach the op, matching the tiers.
                if ((fnn=="__shfl_sync"||fnn=="__shfl_up_sync"||fnn=="__shfl_down_sync"||fnn=="__shfl_xor_sync") && e.args.size() >= 3) {
                    int var = lowerExpr(*e.args[1]); int lane = lowerExpr(*e.args[2]); if (!ok) return 0;
                    Inst in; in.op = Op::WarpShfl; in.ty = typeOf(*e.args[1]);
                    in.dim = fnn=="__shfl_sync"?0 : fnn=="__shfl_up_sync"?1 : fnn=="__shfl_down_sync"?2 : 3;
                    in.a = {var, lane};
                    if (e.args.size() >= 4) { int w = lowerExpr(*e.args[3]); if (!ok) return 0; in.a.push_back(w); }
                    return emit(std::move(in));
                }
                // Warp vote: __ballot_sync/__any_sync/__all_sync(mask, pred). op 0=ballot 1=any 2=all.
                if ((fnn=="__ballot_sync"||fnn=="__any_sync"||fnn=="__all_sync") && e.args.size() == 2) {
                    int mask = lowerExpr(*e.args[0]); int pred = lowerExpr(*e.args[1]); if (!ok) return 0;
                    Inst in; in.op = Op::WarpVote; in.ty = scalar(Type::Int);
                    in.dim = fnn=="__ballot_sync"?0 : fnn=="__any_sync"?1 : 2;
                    in.a = {mask, pred};
                    return emit(std::move(in));
                }
                // Warp reduce: __reduce_{add,min,max,and,or,xor}_sync(mask, value). op 0-5.
                // Folds `value` over the warp's participating lanes; signedness from `value`.
                if ((fnn=="__reduce_add_sync"||fnn=="__reduce_min_sync"||fnn=="__reduce_max_sync"||
                     fnn=="__reduce_and_sync"||fnn=="__reduce_or_sync"||fnn=="__reduce_xor_sync") && e.args.size() == 2) {
                    int mask = lowerExpr(*e.args[0]); int val = lowerExpr(*e.args[1]); if (!ok) return 0;
                    Inst in; in.op = Op::WarpReduce; in.ty = typeOf(*e.args[1]);
                    in.dim = fnn=="__reduce_add_sync"?0 : fnn=="__reduce_min_sync"?1 : fnn=="__reduce_max_sync"?2
                           : fnn=="__reduce_and_sync"?3 : fnn=="__reduce_or_sync"?4 : 5;
                    in.ci = typeOf(*e.args[1]).isUnsigned ? 0 : 1;   // sgn flag for min/max fold
                    in.a = {mask, val};
                    return emit(std::move(in));
                }
                // Warp match: __match_any_sync(mask, value) → mask of same-valued lanes.
                if (fnn=="__match_any_sync" && e.args.size() == 2) {
                    int mask = lowerExpr(*e.args[0]); int val = lowerExpr(*e.args[1]); if (!ok) return 0;
                    Inst in; in.op = Op::WarpMatch; { Type u = scalar(Type::Int); u.isUnsigned = true; in.ty = u; }
                    in.dim = 0;   // any
                    in.a = {mask, val};
                    return emit(std::move(in));
                }
                // __match_all_sync(mask, value, &pred): the participant mask if EVERY
                // participating lane agrees (else 0), and *pred = agreed?1:0. The interpreter
                // tier restricts &pred to a global element (&out[i]); we also accept &localInt.
                // pred is derived as (result != 0) — bit-exact vs the tiers whenever there is
                // at least one participant (the well-formed case; participants==0 is UB).
                if (fnn=="__match_all_sync" && e.args.size() == 3) {
                    const Expr& pe = *e.args[2];
                    if (pe.kind != Expr::Unary || pe.str != "&" || pe.args.empty()) { fail("SSA: __match_all_sync expects &pred as its 3rd argument"); return 0; }
                    int mask = lowerExpr(*e.args[0]); int val = lowerExpr(*e.args[1]); if (!ok) return 0;
                    Inst in; in.op = Op::WarpMatch; { Type u = scalar(Type::Int); u.isUnsigned = true; in.ty = u; }
                    in.dim = 1;   // all
                    in.a = {mask, val};
                    int result = emit(std::move(in));
                    int zero = constI(0, scalar(Type::Int));
                    Inst cm; cm.op = Op::Cmp; cm.s = "!="; cm.ty = scalar(Type::Int); cm.a = {result, zero};
                    int pred = emit(std::move(cm));
                    const Expr& tgt = *pe.args[0];
                    if (tgt.kind == Expr::Ident) {                                     // &localInt
                        std::string nm = mangle(tgt.str);
                        Type vt = vtype.count(nm) ? vtype[nm] : scalar(Type::Int);
                        Inst c; c.op = Op::Cast; c.ty = vt; c.a = {pred}; writeVar(nm, cur, emit(std::move(c)));
                    } else if (tgt.kind == Expr::Index && tgt.args.size() == 2 && tgt.args[0]->kind == Expr::Ident &&
                               !localArr.count(mangle(tgt.args[0]->str)) && !sharedArr.count(mangle(tgt.args[0]->str))) {   // &global[i]
                        std::string bn = mangle(tgt.args[0]->str);
                        Type pt = vtype.count(bn) ? vtype[bn] : Type{};
                        if (pt.ptr != 1) { fail("SSA: __match_all_sync &pred base must be a pointer"); return 0; }
                        Type elem = pt; elem.ptr = 0;
                        int base = lowerExpr(*tgt.args[0]); int idx = lowerExpr(*tgt.args[1]); if (!ok) return 0;
                        Inst c; c.op = Op::Cast; c.ty = elem; c.a = {pred}; int cv = emit(std::move(c));
                        Inst st; st.op = Op::Store; st.a = {base, idx, cv}; st.elemBytes = elem.elemBytes(); st.ty = elem; emit(std::move(st));
                    } else { fail("SSA: __match_all_sync &pred must be &localInt or &global[i]"); return 0; }
                    return result;
                }
                // __activemask(): the warp's in-range lane mask (per-thread, no rendezvous —
                // matches the interpreter's non-exited-lane query for convergent code).
                if (fnn=="__activemask" && e.args.empty()) {
                    Inst in; in.op = Op::WarpActive; { Type u = scalar(Type::Int); u.isUnsigned = true; in.ty = u; }
                    return emit(std::move(in));
                }
                auto dit = deviceFns.find(fnn);
                if (dit != deviceFns.end()) return lowerInlineCall(*dit->second, e);
                fail("SSA: unsupported call '" + fnn + "'"); return 0;
            }
            case Expr::Assign: {
                lowerAssign(e);
                if (!ok) return 0;
                // The assignment expression's value is the new value of the target.
                if (e.args[0]->kind == Expr::Ident) return readVar(mangle(e.args[0]->str), cur);
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
            std::string nm = mangle(lhs.str);
            if (!vtype.count(nm)) { fail("SSA: assign to unknown '" + lhs.str + "'"); return; }
            Type vt = vtype[nm];
            int rhs = lowerExpr(*e.args[1]); if (!ok) return;
            int val = rhs;
            if (op != "=") { int old = readVar(nm, cur); Inst in; in.op = Op::Bin; in.s = op.substr(0, op.size() - 1); in.ty = promoteT(vt, typeOf(*e.args[1])); in.a = {old, rhs}; val = emit(std::move(in)); }
            Inst c; c.op = Op::Cast; c.ty = vt; c.a = {val};      // narrow to the variable's type
            writeVar(nm, cur, emit(std::move(c)));
            return;
        }
        if (lhs.kind == Expr::Index) {
            if (lhs.args.size() != 2 || lhs.args[0]->kind != Expr::Ident) { fail("SSA: bad store index"); return; }
            auto la = localArr.find(mangle(lhs.args[0]->str));
            auto sa = sharedArr.find(mangle(lhs.args[0]->str));
            if (la != localArr.end() || sa != sharedArr.end()) {   // a[idx] [op]= rhs — local- or shared-array write
                const bool shared = (sa != sharedArr.end());
                Type elem = shared ? sa->second.second : la->second.second;
                int arrId = shared ? sa->second.first : la->second.first;
                const Op ldOp = shared ? Op::LoadS : Op::LoadL, stOp = shared ? Op::StoreS : Op::StoreL;
                int idx = lowerExpr(*lhs.args[1]);
                int rhs = lowerExpr(*e.args[1]); if (!ok) return;
                int val = rhs;
                if (op != "=") { Inst ld; ld.op = ldOp; ld.ty = elem; ld.arrId = arrId; ld.elemBytes = elem.elemBytes(); ld.a = {idx}; int old = emit(std::move(ld));
                                 Inst in; in.op = Op::Bin; in.s = op.substr(0, op.size() - 1); in.ty = promoteT(elem, typeOf(*e.args[1])); in.a = {old, rhs}; val = emit(std::move(in)); }
                Inst c; c.op = Op::Cast; c.ty = elem; c.a = {val}; int cv = emit(std::move(c));
                Inst st; st.op = stOp; st.ty = elem; st.arrId = arrId; st.elemBytes = elem.elemBytes(); st.a = {idx, cv}; emit(std::move(st));
                return;
            }
            Type pt = vtype.count(mangle(lhs.args[0]->str)) ? vtype[mangle(lhs.args[0]->str)] : Type{};
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

    // Inline a __device__ helper call. The callee's params/locals are alpha-renamed
    // (mpfx) so they never collide with the caller's; each `return` in the body writes
    // this call's result var and branches to a fresh exit block; the call value is the
    // result var read at the exit (a phi merging every return + a 0 fall-through). Args
    // are lowered in the CALLER scope; recursion is bounded by inlineDepth.
    int lowerInlineCall(const Kernel& F, const Expr& call) {
        if (call.args.size() != F.params.size()) { fail("SSA: wrong argument count for '" + F.name + "'"); return 0; }
        if (inlineDepth > 32) { fail("SSA: __device__ inline too deep (recursion?)"); return 0; }
        std::vector<int> argv(F.params.size());
        for (size_t i = 0; i < F.params.size(); ++i) { argv[i] = lowerExpr(*call.args[i]); if (!ok) return 0; }

        const std::string savedPfx = mpfx;
        mpfx = savedPfx + "$" + std::to_string(inlineUid++) + "@";   // fresh, unique scope
        ++inlineDepth;
        const Type rt = F.returnType;
        const std::string retVar = mpfx + "#ret";                    // '#' can't occur in a source name
        vtype[retVar] = (rt.base == Type::Void) ? scalar(Type::Int) : rt;
        writeVar(retVar, cur, rt.isFloating() ? constF(0.0, vtype[retVar]) : constI(0, vtype[retVar]));
        for (size_t i = 0; i < F.params.size(); ++i) {
            const std::string pn = mangle(F.params[i].name);
            vtype[pn] = F.params[i].type;
            Inst c; c.op = Op::Cast; c.ty = F.params[i].type; c.a = {argv[i]}; writeVar(pn, cur, emit(std::move(c)));
        }
        int exitB = newBlock();
        inlineCtx.push_back({exitB, retVar, rt});
        for (auto& st : F.body) { lowerStmt(*st); if (!ok) { inlineCtx.pop_back(); mpfx = savedPfx; --inlineDepth; return 0; } }
        emitBr(exitB); addEdge(cur, exitB);         // fall-through (no return on this path)
        inlineCtx.pop_back();
        sealBlock(exitB);
        cur = exitB;
        int result = readVar(retVar, cur);          // phi over all returns + the fall-through
        mpfx = savedPfx; --inlineDepth;
        return result;
    }

    void lowerStmt(const Stmt& s) {
        if (!ok) return;
        switch (s.kind) {
            case Stmt::VarDecl: {
                if (s.type.isStruct()) { fail("SSA: local structs unsupported on this tier"); return; }
                if (s.isShared) {   // __shared__ array (block scope, 1-D, static size)
                    if (s.isExternShared) { fail("SSA: dynamic extern __shared__ unsupported on this tier"); return; }
                    if (s.arraySize <= 0) { fail("SSA: scalar __shared__ unsupported on this tier"); return; }
                    if (!mpfx.empty()) { fail("SSA: __shared__ inside an inlined __device__ function unsupported"); return; }
                    int arrId = (int)fn.sharedArrays.size();
                    fn.sharedArrays.push_back({s.type, s.arraySize});
                    sharedArr[s.name] = {arrId, s.type};
                    return;
                }
                if (s.arraySize > 0) {   // per-thread scratch array (register/stack backed, 1-D flattened)
                    if (s.expr) { fail("SSA: local array initializers unsupported on this tier"); return; }
                    int arrId = (int)fn.localArrays.size();
                    fn.localArrays.push_back({s.type, s.arraySize});
                    localArr[mangle(s.name)] = {arrId, s.type};
                    return;
                }
                std::string nm = mangle(s.name);
                vtype[nm] = s.type;
                int v;
                if (s.expr) { int r = lowerExpr(*s.expr); if (!ok) return; Inst c; c.op = Op::Cast; c.ty = s.type; c.a = {r}; v = emit(std::move(c)); }
                else        { v = s.type.isFloating() ? constF(0.0, s.type) : constI(0, s.type); }
                writeVar(nm, cur, v);
                return;
            }
            case Stmt::ExprStmt: if (s.expr) { if (s.expr->kind == Expr::Assign) lowerAssign(*s.expr); else lowerExpr(*s.expr); } return;
            case Stmt::Block: for (auto& st : s.body) { lowerStmt(*st); if (!ok) return; } return;
            case Stmt::Return: {
                if (!inlineCtx.empty()) {                       // return inside an inlined __device__ body
                    const InlineCtx ic = inlineCtx.back();      // (innermost callee) — copy: writeVar may realloc
                    if (s.expr && ic.rt.base != Type::Void) {
                        int rv = lowerExpr(*s.expr); if (!ok) return;
                        Inst c; c.op = Op::Cast; c.ty = ic.rt; c.a = {rv}; int cv = emit(std::move(c));
                        writeVar(ic.retVar, cur, cv);
                    }
                    emitBr(ic.exitB); addEdge(cur, ic.exitB);
                    cur = newBlock(/*seal=*/true);              // trailing stmts after return are unreachable
                    return;
                }
                Inst in; in.op = Op::Ret; emit(std::move(in)); return;
            }
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
                // `loops` holds (breakTgt, continueTgt) for each enclosing loop AND switch;
                // a switch is a break target but forwards the enclosing loop's continue
                // (or -1 when there is none — `continue` inside a switch with no loop).
                if (loops.empty()) { fail("SSA: break/continue outside a loop or switch"); return; }
                int tgt = (s.kind == Stmt::Break) ? loops.back().first : loops.back().second;
                if (tgt < 0) { fail("SSA: continue not inside a loop"); return; }
                emitBr(tgt); addEdge(cur, tgt);
                cur = newBlock(/*seal=*/true);             // fresh unreachable block for any trailing stmts
                return;
            }
            case Stmt::Switch: {
                // switch(sw){ case L: ... default: ... } — C fall-through. The body is a
                // flat list of stmts with Case/Default label markers; segment i is the
                // stmts after label i up to the next label. Lowering: a dispatch chain of
                // (sw == Li) → segBlk[i] tests, ending in a jump to the default segment
                // (or the exit); each segment falls through to the next (unless it breaks),
                // and `break` jumps to the exit. Braun seals each segment once its preds
                // (its dispatch edge + the fall-through from the previous segment) exist.
                int sw = lowerExpr(*s.expr); if (!ok) return;
                struct Seg { bool isDefault; const Expr* label; std::vector<const Stmt*> stmts; };
                std::vector<Seg> segs;
                for (auto& st : s.body) {
                    if (st->kind == Stmt::Case)         segs.push_back({false, st->expr.get(), {}});
                    else if (st->kind == Stmt::Default) segs.push_back({true, nullptr, {}});
                    else if (!segs.empty())             segs.back().stmts.push_back(st.get());
                    // stmts before the first label are unreachable in C — dropped.
                }
                const int nseg = (int)segs.size();
                int exitB = newBlock();
                std::vector<int> segBlk(nseg);
                for (int i = 0; i < nseg; ++i) segBlk[i] = newBlock();
                int defaultIdx = -1;
                for (int i = 0; i < nseg; ++i) if (segs[i].isDefault) { defaultIdx = i; break; }
                // Dispatch chain (each test in its own block; single-pred blocks, sealed at once).
                for (int i = 0; i < nseg; ++i) {
                    if (segs[i].isDefault) continue;
                    int lab = lowerExpr(*segs[i].label); if (!ok) return;
                    Inst cmp; cmp.op = Op::Cmp; cmp.s = "=="; cmp.ty = scalar(Type::Int); cmp.a = {sw, lab};
                    int c = emit(std::move(cmp));
                    int next = newBlock();
                    Inst cb; cb.op = Op::CondBr; cb.a = {c}; cb.bbT = segBlk[i]; cb.bbF = next; emit(std::move(cb));
                    addEdge(cur, segBlk[i]); addEdge(cur, next);
                    sealBlock(next);
                    cur = next;
                }
                int dflt = (defaultIdx >= 0) ? segBlk[defaultIdx] : exitB;
                emitBr(dflt); addEdge(cur, dflt);
                const int contTgt = loops.empty() ? -1 : loops.back().second;   // forward enclosing continue
                loops.push_back({exitB, contTgt});
                for (int i = 0; i < nseg; ++i) {
                    sealBlock(segBlk[i]);                  // preds: dispatch edge + fall-through from i-1
                    cur = segBlk[i];
                    for (const Stmt* st : segs[i].stmts) { lowerStmt(*st); if (!ok) { loops.pop_back(); return; } }
                    int fall = (i + 1 < nseg) ? segBlk[i + 1] : exitB;
                    emitBr(fall); addEdge(cur, fall);      // C fall-through (dead edge if the segment broke)
                }
                loops.pop_back();
                sealBlock(exitB);
                cur = exitB;
                return;
            }
            default: fail("SSA: unsupported statement"); return;
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
static SVal mathfn(const std::string& fn, const SVal& a, const SVal* b, const SVal* c, const Type& rt) {
    if (fn == "min" || fn == "max") {
        if (rt.isFloating()) { double x = asF(a), y = asF(*b); return coerce(SF(fn=="min"?std::fmin(x,y):std::fmax(x,y)), rt); }
        int64_t x = a.i, y = b->i; return coerce(SI(fn=="min"?(x<y?x:y):(x>y?x:y)), rt);
    }
    if (fn == "fminf" || fn == "fmin") return coerce(SF(std::fmin(asF(a), asF(*b))), rt);
    if (fn == "fmaxf" || fn == "fmax") return coerce(SF(std::fmax(asF(a), asF(*b))), rt);
    if (fn == "powf"  || fn == "pow")  return coerce(SF(std::pow(asF(a), asF(*b))), rt);
    if (fn == "fmaf"  || fn == "fma")  return coerce(SF(std::fma(asF(a), asF(*b), asF(*c))), rt);
    if (fn == "erff"  || fn == "erf")  return coerce(SF(std::erf(asF(a))), rt);   // base name ends in 'f' — no suffix strip
    double x = asF(a), r = x;
    const std::string g = (!fn.empty() && fn.back() == 'f') ? fn.substr(0, fn.size() - 1) : fn;
    if (g == "sqrt") r = std::sqrt(x); else if (g == "fabs") r = std::fabs(x);
    else if (g == "exp") r = std::exp(x); else if (g == "log") r = std::log(x);
    else if (g == "sin") r = std::sin(x); else if (g == "cos") r = std::cos(x);
    else if (g == "floor") r = std::floor(x); else if (g == "ceil") r = std::ceil(x);
    else if (g == "tanh") r = std::tanh(x);
    else if (g == "exp2") r = std::exp2(x); else if (g == "log2") r = std::log2(x);
    else if (g == "rsqrt") r = 1.0 / std::sqrt(x);
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

static std::vector<std::vector<char>> computeDom(const Fn& fn);   // defined below (used by gvn + licm)

// Global value numbering (dominator-scoped CSE): dedupe identical pure instructions
// across the whole CFG, not just within a block. Blocks are walked in dominator-tree
// preorder carrying a scoped key→value table: a pure op whose (opcode, type, resolved
// operands) key already appears — in this block or any DOMINATOR — is replaced by that
// earlier value. Because every visible entry is from the current block or an ancestor,
// the canonical definition always dominates the use, so the rewrite is legal (and its
// value is available: SSA defs dominate their uses, and dominance is transitive). Phis
// and effectful ops are never value-numbered. This subsumes the old per-block CSE.
static void gvn(Fn& fn) {
    const int n = (int)fn.bbs.size();
    std::vector<int> remap(fn.vals.size());
    for (size_t i = 0; i < remap.size(); ++i) remap[i] = (int)i;
    std::function<int(int)> resolve = [&](int id) { while (remap[id] != id) id = remap[id]; return id; };

    // Immediate dominators → dominator-tree children. idom[b] is b's deepest strict
    // dominator (the strict dominator with the most dominators; dominators are totally
    // ordered along any path, so this is unique).
    auto dom = computeDom(fn);
    std::vector<int> idom(n, -1), domCount(n, 0);
    for (int b = 0; b < n; ++b) for (int d = 0; d < n; ++d) if (dom[b][d]) ++domCount[b];
    std::vector<std::vector<int>> children(n);
    for (int b = 0; b < n; ++b) {
        if (b == fn.entry) continue;
        int best = -1;
        for (int d = 0; d < n; ++d) if (d != b && dom[b][d] && (best < 0 || domCount[d] > domCount[best])) best = d;
        idom[b] = best;
        if (best >= 0) children[best].push_back(b);
    }

    auto keyOf = [&](const Inst& in) {
        uint64_t fb; std::memcpy(&fb, &in.cf, 8);
        std::string key = std::to_string((int)in.op) + "|" + in.s + "|" + std::to_string(in.ci) + "|" +
                          std::to_string(fb) + "|" + std::to_string((int)in.ty.base) + std::to_string(in.ty.ptr) +
                          std::to_string(in.ty.isUnsigned ? 1 : 0);
        for (int op : in.a) key += "," + std::to_string(op);
        return key;
    };

    std::unordered_map<std::string, int> table;   // scoped: entries live only for the current dom-tree path
    // Iterative dom-tree DFS with a scope stack: ENTER a block numbers its pure insts,
    // LEAVE undoes exactly the table changes it made (restoring any shadowed ancestor entry).
    struct Frame { int b; bool entered; };
    std::vector<Frame> stack{{fn.entry, false}};
    std::vector<std::vector<std::pair<std::string, int>>> undo;   // per active block: (key, prevValue or -1)
    while (!stack.empty()) {
        Frame& f = stack.back();
        if (!f.entered) {
            f.entered = true;
            undo.emplace_back();
            for (int id : fn.bbs[f.b].insts) {
                Inst& in = fn.vals[id];
                for (int& op : in.a) op = resolve(op);
                const bool pure = in.op == Op::ConstI || in.op == Op::ConstF || in.op == Op::Bin ||
                                  in.op == Op::Un || in.op == Op::Cmp || in.op == Op::Cast || in.op == Op::Sel;
                if (!pure) continue;
                std::string key = keyOf(in);
                auto it = table.find(key);
                if (it != table.end()) { remap[id] = it->second; }
                else { undo.back().push_back({key, -1}); table[key] = id; }
            }
            int b = f.b;
            for (int ch : children[b]) stack.push_back({ch, false});
        } else {
            for (auto it = undo.back().rbegin(); it != undo.back().rend(); ++it) {
                if (it->second == -1) table.erase(it->first); else table[it->first] = it->second;
            }
            undo.pop_back();
            stack.pop_back();
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
        if (in.op == Op::Store || in.op == Op::StoreL || in.op == Op::StoreS || in.op == Op::Barrier ||
            in.op == Op::WarpShfl || in.op == Op::WarpVote ||   // cross-lane rendezvous — keep (every lane executes it)
            in.op == Op::WarpReduce || in.op == Op::WarpMatch ||
            in.op == Op::Ret || in.op == Op::Br || in.op == Op::CondBr) {
            if (!live[id]) { live[id] = 1; work.push_back(id); }
        }
    }
    while (!work.empty()) { int id = work.back(); work.pop_back(); for (int op : fn.vals[id].a) if (!live[op]) { live[op] = 1; work.push_back(op); } }
    for (auto& bb : fn.bbs) {
        std::vector<int> keep;
        for (int id : bb.insts) {
            const Inst& in = fn.vals[id];
            const bool effect = in.op == Op::Store || in.op == Op::StoreL || in.op == Op::StoreS || in.op == Op::Barrier ||
                                in.op == Op::WarpShfl || in.op == Op::WarpVote ||
                                in.op == Op::WarpReduce || in.op == Op::WarpMatch ||
                                in.op == Op::Ret || in.op == Op::Br || in.op == Op::CondBr;
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

static void runOpt(Fn& fn) { constFold(fn); gvn(fn); licm(fn); constFold(fn); gvn(fn); dce(fn); }

// ── Shared global linear-scan register allocation (used by both native emitters) ──
#if VGRE_SSA_X64 || VGRE_SSA_ARM64
// Poletto & Sarkar over the SSA, backend-agnostic: hands out physical registers from
// `gprPool` (int/pointer values, incl. loop-carried phis) and `fpPool` (float values)
// into vreg[]/vxmm[] (-1 = memory slot). Both pools must be callee-saved on the target
// so no spill-around-call is needed. Live intervals are widened over every block a
// value is live across (loops aren't in execution order, so a loop-carried value is
// live in a back-edge block placed BEFORE its def — start is lowered to cover that).
// Float values whose interval spans a helper call (fcmp/sat/idiv/mathfn) stay in slots
// (conservative: correct on any target, incl. ones whose FP regs are caller-saved).
// Registers are popped from the BACK of each pool, so pass them highest-first to match.
static void computeRegAlloc(const Fn& fn, int nvals,
                            const std::vector<int>& gprPool, const std::vector<int>& fpPool,
                            std::vector<int>& vreg, std::vector<int>& vxmm) {
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
    std::vector<std::unordered_set<int>> liveIn(n), liveOut(n);
    for (bool changed = true; changed;) {
        changed = false;
        for (int b = n - 1; b >= 0; --b) {
            std::unordered_set<int> out;
            for (int s : succOf(b)) {
                for (int v : liveIn[s]) if (!(fn.vals[v].op == Op::Phi && defBlk[v] == s)) out.insert(v);
                for (int id : fn.bbs[s].insts) {
                    const Inst& in = fn.vals[id];
                    if (in.op != Op::Phi) break;
                    for (size_t k = 0; k < in.phiPred.size(); ++k) if (in.phiPred[k] == b) out.insert(in.a[k]);
                }
            }
            std::unordered_set<int> live = out;
            const auto& insts = fn.bbs[b].insts;
            for (int i = (int)insts.size() - 1; i >= 0; --i) {
                const Inst& in = fn.vals[insts[i]];
                live.erase(insts[i]);
                if (in.op != Op::Phi) for (int op : in.a) live.insert(op);
            }
            if (live != liveIn[b] || out != liveOut[b]) { liveIn[b] = std::move(live); liveOut[b] = std::move(out); changed = true; }
        }
    }
    std::vector<int> start(nvals, 0), end(nvals, 0);
    for (int id = 0; id < nvals; ++id) { start[id] = pos[id] < 0 ? 0 : pos[id]; end[id] = start[id]; }
    for (int b = 0; b < n; ++b) for (int id : fn.bbs[b].insts) {
        const Inst& in = fn.vals[id];
        if (in.op == Op::Phi) {
            for (size_t k = 0; k < in.phiPred.size(); ++k) {
                int op = in.a[k], predEnd = lastPos[in.phiPred[k]];
                end[op]  = std::max(end[op], predEnd);
                start[id] = std::min(start[id], predEnd);
                end[id]   = std::max(end[id], predEnd);
            }
        } else for (int op : in.a) end[op] = std::max(end[op], pos[id]);
    }
    for (int b = 0; b < n; ++b) {
        for (int v : liveIn[b])  start[v] = std::min(start[v], firstPos[b]);
        for (int v : liveOut[b]) end[v]   = std::max(end[v], lastPos[b]);
    }
    std::vector<int> order;
    for (int id = 0; id < nvals; ++id) {
        const Inst& in = fn.vals[id];
        const bool intType = in.ty.base == Type::Int || in.ty.base == Type::Long || in.ty.base == Type::Char ||
                             in.ty.base == Type::Short || in.ty.base == Type::Bool || in.ty.isPointer();
        if (intType && pos[id] >= 0 && end[id] > start[id]) order.push_back(id);
    }
    std::sort(order.begin(), order.end(), [&](int a, int b) { return start[a] < start[b]; });
    std::vector<int> freeRegs = gprPool;
    std::vector<std::pair<int, int>> active;
    for (int id : order) {
        for (size_t a = 0; a < active.size();) {
            if (active[a].first < start[id]) { freeRegs.push_back(vreg[active[a].second]); active.erase(active.begin() + a); }
            else ++a;
        }
        if (!freeRegs.empty()) { vreg[id] = freeRegs.back(); freeRegs.pop_back(); active.push_back({end[id], id}); }
    }
    auto emitsCall = [&](const Inst& in) {
        if (in.op == Op::CallMath || in.op == Op::Barrier) return true;   // Barrier = call vgre_ssa_barrier (clobbers caller-saved XMM)
        if (in.op == Op::Cmp) return fn.vals[in.a[0]].ty.isFloating() || fn.vals[in.a[1]].ty.isFloating();
        if (in.op == Op::Cast) return !in.ty.isFloating() && !in.ty.isPointer() && fn.vals[in.a[0]].ty.isFloating();
        if (in.op == Op::Bin)  return (in.s == "/" || in.s == "%") && !in.ty.isFloating();
        return false;
    };
    const int P = p;
    std::vector<int> callPre(P + 1, 0);
    for (int b = 0; b < n; ++b) for (int id : fn.bbs[b].insts) if (pos[id] >= 0 && emitsCall(fn.vals[id])) callPre[pos[id] + 1] = 1;
    for (int i = 0; i < P; ++i) callPre[i + 1] += callPre[i];
    std::vector<int> forder;
    for (int id = 0; id < nvals; ++id) {
        const Inst& in = fn.vals[id];
        if (!in.ty.isFloating() || pos[id] < 0 || end[id] <= start[id]) continue;
        if (callPre[end[id] + 1] - callPre[start[id]] != 0) continue;
        forder.push_back(id);
    }
    std::sort(forder.begin(), forder.end(), [&](int a, int b) { return start[a] < start[b]; });
    std::vector<int> freeX = fpPool;
    std::vector<std::pair<int, int>> activeX;
    for (int id : forder) {
        for (size_t a = 0; a < activeX.size();) {
            if (activeX[a].first < start[id]) { freeX.push_back(vxmm[activeX[a].second]); activeX.erase(activeX.begin() + a); }
            else ++a;
        }
        if (!freeX.empty()) { vxmm[id] = freeX.back(); freeX.pop_back(); activeX.push_back({end[id], id}); }
    }
}
#endif  // VGRE_SSA_X64 || VGRE_SSA_ARM64

// ── Tier-2 native machine-code emission (x86-64 / Linux) ──────────────────────────
#if VGRE_SSA_X64 || VGRE_SSA_ARM64
// Per-thread launch context. `pvals` holds one 8-byte value per kernel parameter,
// pre-decoded by the launcher (pointer as-is; int sign/zero-extended; float/double as
// the double's bit pattern) so the emitted code reads a uniform slot. `idx` is
// tid[3],ctaid[3],ntid[3],nctaid[3]. Shared by both native emitters (x86-64 + AArch64).
// `shared` points to the block's __shared__ buffer (set per block by the coop launcher;
// null for barrier-free kernels). It sits at a fixed offset the emitter hardcodes.
struct ThreadCtx { const int64_t* pvals; uint32_t idx[12]; void* shared; };

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
                  case 6: return std::floor(x); case 7: return std::ceil(x); case 8: return std::tanh(x);
                  case 9: return std::exp2(x); case 10: return std::log2(x);
                  case 11: return 1.0 / std::sqrt(x); case 12: return std::erf(x); }
    return x;
}
double vgre_ssa_m2(int op, double x, double y) {   // 0=fmin 1=fmax 2=pow
    switch (op) { case 1: return std::fmax(x, y); case 2: return std::pow(x, y); default: return std::fmin(x, y); }
}
double vgre_ssa_m3(double x, double y, double z) { return std::fma(x, y, z); }   // fma(x,y,z)=x*y+z
}

using SsaFn = void (*)(ThreadCtx*);

// Shared by both native emitters (ISA-independent). mathId maps a unary math intrinsic
// to the vgre_ssa_m1 selector (erf's name ends in 'f', so it precedes the suffix strip).
static int mathId(const std::string& s) {
    if (s == "erf" || s == "erff") return 12;
    std::string g = (!s.empty() && s.back() == 'f') ? s.substr(0, s.size() - 1) : s;
    if (g == "sqrt") return 0; if (g == "fabs") return 1; if (g == "exp") return 2; if (g == "log") return 3;
    if (g == "sin") return 4; if (g == "cos") return 5; if (g == "floor") return 6; if (g == "ceil") return 7;
    if (g == "tanh") return 8; if (g == "exp2") return 9; if (g == "log2") return 10; if (g == "rsqrt") return 11;
    return -1;
}
// Where an op leaves its result, for the redundant-load peephole: 0 = the integer
// accumulator (rax / x0), 1 = the fp accumulator (xmm0 / d0), 2 = neither (reload needed).
static int resultCacheKind(const Inst& in) {
    switch (in.op) {
        case Op::ConstI: case Op::Tid: case Op::Ctaid: case Op::Ntid: case Op::Nctaid: case Op::Cmp: return 0;
        case Op::Param: return in.ty.isFloating() ? 2 : 0;
        case Op::Bin:   return in.ty.isFloating() ? 1 : 0;
        case Op::Un:    return (in.s == "!" || !in.ty.isFloating()) ? 0 : 2;
        case Op::Sel:   return in.ty.isFloating() ? 2 : 0;
        case Op::Cast: case Op::Load: case Op::LoadL: return in.ty.isFloating() ? 1 : 0;
        case Op::CallMath: return 1;
        default: return 2;
    }
}

// ── Native cooperative execution for __shared__/__syncthreads (ucontext fibers) ───
// Each CUDA thread runs its emitted machine code on its own ucontext fiber. At a
// __syncthreads the JIT code `call`s vgre_ssa_barrier, which swapcontexts back to the
// per-block scheduler; the scheduler resumes all fibers once every live one is parked
// at the barrier. swapcontext/getcontext save/restore the callee-saved registers, and
// values live on the fiber stack — so a barrier is transparent to the emitted code (no
// state machine) and register allocation keeps working across it. Shared by both native
// emitters; the launcher runs the scheduler (x86-64 + AArch64, POSIX ucontext).
struct NFiber {
    ucontext_t ctx;
    ucontext_t* sched = nullptr;   // the block scheduler's context
    std::vector<char> stack;
    SsaFn fn = nullptr;
    ThreadCtx* tctx = nullptr;
    bool done = false, atBarrier = false;
};
thread_local NFiber* g_nfCur = nullptr;   // fiber currently running (set before each resume)
inline void nfTrampoline() { NFiber* f = g_nfCur; f->fn(f->tctx); f->done = true; }   // return → uc_link (scheduler)
extern "C" void vgre_ssa_barrier() { NFiber* f = g_nfCur; f->atBarrier = true; swapcontext(&f->ctx, f->sched); }
#endif  // VGRE_SSA_X64 || VGRE_SSA_ARM64

#if VGRE_SSA_X64
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
        int k = resultCacheKind(in);   // 0 rax / 1 xmm0 / 2 neither
        cRax = (k == 0) ? id : -1;
        cXmm = (k == 1) ? id : -1;
    }

    explicit X64Asm(const Fn& f) : fn(f), nvals((int)f.vals.size()), vreg((size_t)f.vals.size(), -1), vxmm((size_t)f.vals.size(), -1) {}
    void bad() { ok = false; }

    // x86-64 pools: int/pointer → callee-saved r12–r15; float → xmm2–xmm7 (xmm0/xmm1
    // stay scratch, and XMM is caller-saved so call-spanning floats keep to slots).
    // The shared linear-scan does the work (see computeRegAlloc).
    void allocateRegs() { computeRegAlloc(fn, nvals, {15, 14, 13, 12}, {7, 6, 5, 4, 3, 2}, vreg, vxmm); }

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
    std::vector<int> vreg;   // per value: a GPR 12..15 if allocated, else -1 (slot)
    std::vector<int> vxmm;   // per float value: an XMM reg 2..7 if allocated, else -1 (slot)
    std::vector<int> arrBase;   // per local array: its first element's slot index (element k = arrBase+k)
    std::vector<int> sharedOff; // per __shared__ array: its byte offset in the block shared buffer

    // GPR (64-bit) and XMM (double) load/store to an rbp-relative displacement.
    void ldGd(int rg, int disp) { b(0x48); b(0x8B); modRbp(rg, disp); }
    void stGd(int rg, int disp) { b(0x48); b(0x89); modRbp(rg, disp); }
    void leaRbp(int rg, int disp) { b(0x48); b(0x8D); modRbp(rg, disp); }   // lea Rreg,[rbp+disp]
    void ldRbxOfs(int rg, int disp) { b(0x48); b(0x8B); b((uint8_t)(0x80 | ((rg & 7) << 3) | 3)); d32((uint32_t)disp); }  // mov Rreg,[rbx+disp]
    void ldXd(int x, int disp)  { b(0xF2); b(0x0F); b(0x10); modRbp(x, disp); }
    void stXd(int x, int disp)  { b(0xF2); b(0x0F); b(0x11); modRbp(x, disp); }
    // A register-resident value is copied to/from scratch; otherwise it uses its slot.
    void ldG(int rg, int id) { if (vreg[id] >= 0) movReg(rg, vreg[id]); else ldGd(rg, slot(id)); }
    void stG(int rg, int id) { if (vreg[id] >= 0) movReg(vreg[id], rg); else stGd(rg, slot(id)); }
    // Float values live in an XMM register (vxmm 2..7) if allocated, else a memory slot.
    // Every float access goes through these two (movsd for reg↔reg), so the allocator
    // pass alone enables XMM residency. stFloatBits/ldFloatBits move a value whose bit
    // pattern is currently in rax to/from its home (used by ConstF/Param/Un, which build
    // the float as integer bits): movq to the register, or a plain store/load to the slot.
    void ldX(int x, int id)  { if (vxmm[id] >= 0) movXmm(x, vxmm[id]); else ldXd(x, slot(id)); }
    void stX(int x, int id)  { if (vxmm[id] >= 0) movXmm(vxmm[id], x); else stXd(x, slot(id)); }
    void stFloatBits(int id) { if (vxmm[id] >= 0) movqXR(vxmm[id]); else stGd(0, slot(id)); }   // rax bits → home
    void ldFloatBits(int id) { if (vxmm[id] >= 0) movqRX(vxmm[id]); else ldGd(0, slot(id)); }   // home → rax bits

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
    void cmov(uint8_t cc, int dst, int src) { b(0x48); b(0x0F); b(cc); b((uint8_t)(0xC0 | ((dst & 7) << 3) | (src & 7))); }  // cmovCC dst,src (64-bit)
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
    // movsd xmm dst,src and movq xmm<->rax for the XMM register allocator (xmm0..7, no REX).
    void movXmm(int d, int s) { b(0xF2); b(0x0F); b(0x10); b((uint8_t)(0xC0 | ((d & 7) << 3) | (s & 7))); }
    void movqXR(int x) { b(0x66); b(0x48); b(0x0F); b(0x6E); b((uint8_t)(0xC0 | ((x & 7) << 3))); }   // xmm x = rax bits
    void movqRX(int x) { b(0x66); b(0x48); b(0x0F); b(0x7E); b((uint8_t)(0xC0 | ((x & 7) << 3))); }   // rax = xmm x bits
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
        // Phase 1 reads every operand into a temp slot, phase 2 writes every phi from its
        // temp — a parallel copy (breaks cycles). Float phis move via xmm (movsd), int/
        // pointer phis via rax; each side is register- or slot-resident per ldX/stX/ldG/stG.
        for (size_t i = 0; i < pr.size(); ++i) {
            if (fn.vals[pr[i].first].ty.isFloating()) { ldX(0, pr[i].second); stXd(0, temp((int)i)); }
            else                                       { ldG(0, pr[i].second); stGd(0, temp((int)i)); }
        }
        for (size_t i = 0; i < pr.size(); ++i) {
            if (fn.vals[pr[i].first].ty.isFloating()) { ldXd(0, temp((int)i)); stX(0, pr[i].first); }
            else                                       { ldGd(0, temp((int)i)); stG(0, pr[i].first); }
        }
    }

    void narrowIfFloat(const Type& t) { if (t.base == Type::Float) { cvtsd2ss0(); cvtss2sd0(); } }   // xmm0 → float32-rounded
    // Type-width memory access shared by global Load/Store and local-array LoadL/StoreL:
    // load reads [rax] → rax (int) / xmm0 (float); store writes the int in rax or the
    // float in xmm0 to [rdx]. The caller computes the address into rax (load) / rdx (store).
    void loadFromRax(const Type& t) {
        if (t.isFloating()) { if (t.elemBytes() == 8) { b(0xF2); b(0x0F); b(0x10); b(0x00); } else { b(0xF3); b(0x0F); b(0x10); b(0x00); cvtss2sd0(); } return; }
        if (t.isPointer() || t.elemBytes() == 8) { b(0x48); b(0x8B); b(0x00); }
        else if (t.elemBytes() == 4) { if (t.isUnsigned) { b(0x8B); b(0x00); } else { b(0x48); b(0x63); b(0x00); } }
        else if (t.elemBytes() == 2) { b(0x0F); b(t.isUnsigned ? 0xB7 : 0xBF); b(0x00); }
        else { b(0x0F); b(t.isUnsigned ? 0xB6 : 0xBE); b(0x00); }
    }
    void storeToRdx(const Type& t) {
        if (t.isFloating()) { if (t.elemBytes() == 8) { b(0xF2); b(0x0F); b(0x11); b(0x02); } else { cvtsd2ss0(); b(0xF3); b(0x0F); b(0x11); b(0x02); } return; }
        if (t.isPointer() || t.elemBytes() == 8) { b(0x48); b(0x89); b(0x02); }
        else if (t.elemBytes() == 4) { b(0x89); b(0x02); }
        else if (t.elemBytes() == 2) { b(0x66); b(0x89); b(0x02); }
        else { b(0x88); b(0x02); }
    }

    void emitInst(int blk, int id) {
        (void)blk;
        const Inst& in = fn.vals[id];
        switch (in.op) {
            case Op::Phi: return;
            case Op::ConstI: { movImm(0, (uint64_t)coerce(SI(in.ci), in.ty).i); stG(0, id); return; }
            case Op::ConstF: { double d = coerce(SF(in.cf), in.ty).d; uint64_t bits; std::memcpy(&bits, &d, 8); movImm(0, bits); stFloatBits(id); return; }
            case Op::Param: { b(0x48); b(0x8B); b(0x03);                          // mov rax,[rbx]  (pvals)
                              b(0x48); b(0x8B); b(0x80); d32((uint32_t)(in.paramIdx * 8));   // mov rax,[rax+p*8]
                              if (in.ty.isFloating()) stFloatBits(id); else stG(0, id); return; }
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
                if (in.ty.isFloating()) { ldFloatBits(in.a[0]); movImm(1, 0x8000000000000000ull); xorRR(0, 1); stFloatBits(id); return; }  // sign flip (bits in rax)
                ldG(0, in.a[0]); if (in.s == "-") negR(0); else notR(0); extRax(in.ty.elemBytes(), in.ty.isUnsigned); stG(0, id); return;
            }
            case Op::Sel: {
                const bool fl = in.ty.isFloating();                              // arms are float ⇒ move via xmm
                if (fl) { ldX(0, in.a[2]); stX(0, id); } else { ldG(0, in.a[2]); stG(0, id); }   // res = else
                ldG(0, in.a[0]); testRR(0, 0);                                    // condition is always int
                b(0x0F); b(0x84); size_t js = c.size(); d32(0);                  // jz skip
                if (fl) { ldX(0, in.a[1]); stX(0, id); } else { ldG(0, in.a[1]); stG(0, id); }   // res = then
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
            case Op::Load: {   // rax = base + idx*bytes
                loadG0(in.a[0]); ldG(1, in.a[1]); movImm(2, (uint64_t)in.elemBytes); imulRR(1, 2); addRR(0, 1);
                loadFromRax(in.ty); if (in.ty.isFloating()) stX(0, id); else stG(0, id); return;
            }
            case Op::Store: {   // rdx = base + idx*bytes
                ldG(2, in.a[0]); ldG(1, in.a[1]); movImm(0, (uint64_t)in.elemBytes); imulRR(1, 0); addRR(2, 1);
                if (in.ty.isFloating()) ldX(0, in.a[2]); else ldG(0, in.a[2]);
                storeToRdx(in.ty); return;
            }
            case Op::Barrier: call((uint64_t)&vgre_ssa_barrier); return;   // __syncthreads → yield the fiber to the block scheduler
            case Op::WarpShfl: case Op::WarpVote: case Op::WarpReduce:
            case Op::WarpMatch: case Op::WarpActive: bad(); return;       // warp intrinsics run on the cooperative evaluator
            case Op::LoadS: {   // rax = ctx.shared + sharedOff[arrId] + idx*elemBytes
                ldRbxOfs(0, (int)offsetof(ThreadCtx, shared)); ldG(1, in.a[0]); movImm(2, (uint64_t)in.elemBytes); imulRR(1, 2); addRR(0, 1);
                movImm(1, (uint64_t)sharedOff[in.arrId]); addRR(0, 1);
                loadFromRax(in.ty); if (in.ty.isFloating()) stX(0, id); else stG(0, id); return;
            }
            case Op::StoreS: {   // rdx = ctx.shared + sharedOff[arrId] + idx*elemBytes; write in.a[1]
                ldRbxOfs(2, (int)offsetof(ThreadCtx, shared)); ldG(1, in.a[0]); movImm(0, (uint64_t)in.elemBytes); imulRR(1, 0); addRR(2, 1);
                movImm(0, (uint64_t)sharedOff[in.arrId]); addRR(2, 0);
                if (in.ty.isFloating()) ldX(0, in.a[1]); else ldG(0, in.a[1]);
                storeToRdx(in.ty); return;
            }
            case Op::LoadL: {   // rax = &arr[0] - idx*8 (elements are 8-byte slots, descending)
                leaRbp(0, slot(arrBase[in.arrId])); ldG(1, in.a[0]); movImm(2, 8); imulRR(1, 2); subRR(0, 1);
                loadFromRax(in.ty); if (in.ty.isFloating()) stX(0, id); else stG(0, id); return;
            }
            case Op::StoreL: {   // rdx = &arr[0] - idx*8; write in.a[1]
                leaRbp(2, slot(arrBase[in.arrId])); ldG(1, in.a[0]); movImm(0, 8); imulRR(1, 0); subRR(2, 1);
                if (in.ty.isFloating()) ldX(0, in.a[1]); else ldG(0, in.a[1]);
                storeToRdx(in.ty); return;
            }
            case Op::CallMath: {
                if (in.a.size() == 1) {
                    int mid = mathId(in.s); if (mid < 0) { bad(); return; }   // unknown intrinsic → evaluator fallback
                    ldX(0, in.a[0]); movImmReg(7, (uint32_t)mid); call((uint64_t)&vgre_ssa_m1);
                } else if (in.a.size() == 2) {
                    if (!in.ty.isFloating()) {   // integer min/max via cmov (signed, matching the evaluator's x<y?x:y)
                        ldG(0, in.a[0]); ldG(1, in.a[1]); cmpRR(0, 1);              // rax=a, rcx=b; flags = a-b
                        cmov(in.s == "max" ? 0x4C : 0x4F, 0, 1);                   // max: cmovl (a<b→b); min: cmovg (a>b→b)
                        extRax(in.ty.elemBytes(), in.ty.isUnsigned); stG(0, id); return;
                    }
                    uint32_t op = (in.s == "fmax" || in.s == "fmaxf") ? 1u : (in.s == "pow" || in.s == "powf") ? 2u : 0u;
                    ldX(0, in.a[0]); ldX(1, in.a[1]); movImmReg(7, op); call((uint64_t)&vgre_ssa_m2);
                } else {   // 3-arg: fma(x,y,z) → xmm0,xmm1,xmm2 (no int selector)
                    ldX(0, in.a[0]); ldX(1, in.a[1]); ldX(2, in.a[2]); call((uint64_t)&vgre_ssa_m3);
                }
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
        // Local arrays get 8-byte slots after the value + phi-temp slots (element k of
        // array a is slot arrBase[a]+k), so all the existing slot addressing applies.
        arrBase.assign(fn.localArrays.size(), 0);
        int arrElems = 0;
        for (size_t i = 0; i < fn.localArrays.size(); ++i) { arrBase[i] = nvals + maxTemps + arrElems; arrElems += fn.localArrays[i].second; }
        // __shared__ arrays live in the block-shared buffer (ctx.shared), 8-byte aligned.
        sharedOff.assign(fn.sharedArrays.size(), 0);
        int so = 0;
        for (size_t i = 0; i < fn.sharedArrays.size(); ++i) { sharedOff[i] = so; so += ((fn.sharedArrays[i].second * fn.sharedArrays[i].first.elemBytes() + 7) / 8) * 8; }
        const int need = (nvals + maxTemps + arrElems) * 8;
        frame = ((need + 15) / 16) * 16 + 8;         // keep rsp 16-aligned at calls
        b(0x55); b(0x48); b(0x89); b(0xE5); b(0x53);  // push rbp; mov rbp,rsp; push rbx
        b(0x41); b(0x54); b(0x41); b(0x55); b(0x41); b(0x56); b(0x41); b(0x57);   // push r12; r13; r14; r15
        b(0x48); b(0x81); b(0xEC); d32((uint32_t)frame);   // sub rsp,frame
        b(0x48); b(0x89); b(0xFB);                    // mov rbx,rdi
        if (arrElems) { xorRR(0, 0); for (int i = 0; i < arrElems; ++i) stGd(0, slot(nvals + maxTemps + i)); }  // zero-init scratch arrays
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

// ── Tier-2 native machine-code emission (AArch64 — Apple Silicon + ARM Linux) ─────
// Compiled whenever EITHER native backend is on (the struct is pure, arch-independent
// byte generation), so its encodings can be unit-tested on an x86-64 host against
// llvm-mc (arm64EncSelfTest / test_ssa_arm64_enc). It is only INSTANTIATED and EXECUTED
// on an actual aarch64 host (VGRE_SSA_ARM64), where compile() mmaps + runs it.
#if VGRE_SSA_X64 || VGRE_SSA_ARM64
// Mirrors X64Asm exactly — same IR, same slot model, same helper delegation, same
// linear-scan reg alloc and two-phase phi copies — differing only in encodings and
// the AAPCS64 calling convention. Fixed-width 32-bit instructions. Registers: x19 =
// ThreadCtx*, x0/x1/x2/x9 = int scratch, d0/d1/d2 = fp scratch, x20–x27 = allocatable
// GPRs (callee-saved), d8–d15 = allocatable FP (callee-saved). SSA values live in
// sp-relative slots (positive offsets). Every instruction is a single 32-bit word.
struct Arm64Asm {
    std::vector<uint8_t> c;
    const Fn& fn;
    int nvals, frame = 0;
    std::vector<size_t> off;                       // block start byte offsets
    std::vector<std::pair<size_t, int>> jpatch;    // (b-instruction site, target block)
    bool ok = true;
    int cX0 = -1, cD0 = -1;                        // value currently in x0 / d0 (peephole), or -1
    std::vector<int> vreg, vxmm;                   // physical reg per value (x20–27 / d8–15), or -1
    std::vector<int> arrBase;                      // per local array: first element's slot index (element k = arrBase+k)
    std::vector<int> sharedOff;                    // per __shared__ array: byte offset in the block shared buffer

    explicit Arm64Asm(const Fn& f) : fn(f), nvals((int)f.vals.size()),
        vreg((size_t)f.vals.size(), -1), vxmm((size_t)f.vals.size(), -1) {}
    void bad() { ok = false; }
    void allocateRegs() { computeRegAlloc(fn, nvals, {27, 26, 25, 24, 23, 22, 21, 20}, {15, 14, 13, 12, 11, 10, 9, 8}, vreg, vxmm); }

    void w(uint32_t x) { for (int i = 0; i < 4; ++i) c.push_back((uint8_t)(x >> (8 * i))); }

    // ── encoders (all verified against llvm-mc; see test_ssa_arm64_enc) ──
    void movRR(int d, int n)  { w(0xaa0003e0u | ((uint32_t)n << 16) | (uint32_t)d); }               // mov Xd,Xn
    void add3(int d, int n, int m) { w(0x8b000000u | ((uint32_t)m << 16) | ((uint32_t)n << 5) | d); }
    void sub3(int d, int n, int m) { w(0xcb000000u | ((uint32_t)m << 16) | ((uint32_t)n << 5) | d); }
    void mul3(int d, int n, int m) { w(0x9b007c00u | ((uint32_t)m << 16) | ((uint32_t)n << 5) | d); }
    void and3(int d, int n, int m) { w(0x8a000000u | ((uint32_t)m << 16) | ((uint32_t)n << 5) | d); }
    void orr3(int d, int n, int m) { w(0xaa000000u | ((uint32_t)m << 16) | ((uint32_t)n << 5) | d); }
    void eor3(int d, int n, int m) { w(0xca000000u | ((uint32_t)m << 16) | ((uint32_t)n << 5) | d); }
    void lslv(int d, int n, int m) { w(0x9ac02000u | ((uint32_t)m << 16) | ((uint32_t)n << 5) | d); }
    void lsrv(int d, int n, int m) { w(0x9ac02400u | ((uint32_t)m << 16) | ((uint32_t)n << 5) | d); }
    void asrv(int d, int n, int m) { w(0x9ac02800u | ((uint32_t)m << 16) | ((uint32_t)n << 5) | d); }
    void cmpRR(int n, int m)  { w(0xeb00001fu | ((uint32_t)m << 16) | ((uint32_t)n << 5)); }         // subs xzr,Xn,Xm
    void csetF(int d, int f)  { w(0x9a9f07e0u | ((uint32_t)f << 12) | (uint32_t)d); }                // cset Xd,cond(field)
    void cselC(int d, int n, int m, int cond) { w(0x9a800000u | ((uint32_t)m << 16) | ((uint32_t)cond << 12) | ((uint32_t)n << 5) | d); }  // csel Xd,Xn,Xm,cond
    void negR(int d, int n)   { w(0xcb0003e0u | ((uint32_t)n << 16) | (uint32_t)d); }                // sub Xd,xzr,Xn
    void notR(int d, int n)   { w(0xaa2003e0u | ((uint32_t)n << 16) | (uint32_t)d); }                // orn Xd,xzr,Xn
    void sxtw(int d, int n)   { w(0x93407c00u | ((uint32_t)n << 5) | d); }
    void sxth(int d, int n)   { w(0x93403c00u | ((uint32_t)n << 5) | d); }
    void sxtb(int d, int n)   { w(0x93401c00u | ((uint32_t)n << 5) | d); }
    void uxtw(int d, int n)   { w(0x2a0003e0u | ((uint32_t)n << 16) | (uint32_t)d); }                // mov Wd,Wn
    void uxth(int d, int n)   { w(0x53003c00u | ((uint32_t)n << 5) | d); }
    void uxtb(int d, int n)   { w(0x53001c00u | ((uint32_t)n << 5) | d); }
    void movz(int d, uint32_t imm16, int s) { w(0xd2800000u | ((uint32_t)s << 21) | ((imm16 & 0xffff) << 5) | d); }
    void movk(int d, uint32_t imm16, int s) { w(0xf2800000u | ((uint32_t)s << 21) | ((imm16 & 0xffff) << 5) | d); }
    void movImm(int d, uint64_t v) { movz(d, (uint32_t)(v & 0xffff), 0);
        movk(d, (uint32_t)((v >> 16) & 0xffff), 1); movk(d, (uint32_t)((v >> 32) & 0xffff), 2); movk(d, (uint32_t)((v >> 48) & 0xffff), 3); }
    // loads/stores through a base register, offset 0 (address already computed in Rn)
    void ldrX(int rt, int rn)  { w(0xf9400000u | ((uint32_t)rn << 5) | rt); }
    void ldrW(int rt, int rn)  { w(0xb9400000u | ((uint32_t)rn << 5) | rt); }
    void ldrSW(int rt, int rn) { w(0xb9800000u | ((uint32_t)rn << 5) | rt); }
    void ldrH(int rt, int rn)  { w(0x79400000u | ((uint32_t)rn << 5) | rt); }
    void ldrSH(int rt, int rn) { w(0x79c00000u | ((uint32_t)rn << 5) | rt); }
    void ldrB(int rt, int rn)  { w(0x39400000u | ((uint32_t)rn << 5) | rt); }
    void ldrSB(int rt, int rn) { w(0x39c00000u | ((uint32_t)rn << 5) | rt); }
    void strX(int rt, int rn)  { w(0xf9000000u | ((uint32_t)rn << 5) | rt); }
    void strW(int rt, int rn)  { w(0xb9000000u | ((uint32_t)rn << 5) | rt); }
    void strH(int rt, int rn)  { w(0x79000000u | ((uint32_t)rn << 5) | rt); }
    void strB(int rt, int rn)  { w(0x39000000u | ((uint32_t)rn << 5) | rt); }
    void ldrDreg(int dt, int rn) { w(0xfd400000u | ((uint32_t)rn << 5) | dt); }
    void strDreg(int dt, int rn) { w(0xfd000000u | ((uint32_t)rn << 5) | dt); }
    void ldrSreg(int st, int rn) { w(0xbd400000u | ((uint32_t)rn << 5) | st); }
    void strSreg(int st, int rn) { w(0xbd000000u | ((uint32_t)rn << 5) | st); }
    // sp-relative slot access (scaled immediate). off is a byte offset, multiple of 8.
    void ldrXslot(int rt, int off) { w(0xf9400000u | ((uint32_t)(off / 8) << 10) | (31u << 5) | rt); }
    void strXslot(int rt, int off) { w(0xf9000000u | ((uint32_t)(off / 8) << 10) | (31u << 5) | rt); }
    void ldrDslot(int dt, int off) { w(0xfd400000u | ((uint32_t)(off / 8) << 10) | (31u << 5) | dt); }
    void strDslot(int dt, int off) { w(0xfd000000u | ((uint32_t)(off / 8) << 10) | (31u << 5) | dt); }
    void ldrXofs(int rt, int rn, int off) { w(0xf9400000u | ((uint32_t)(off / 8) << 10) | ((uint32_t)rn << 5) | rt); }
    void ldrWofs(int rt, int rn, int off) { w(0xb9400000u | ((uint32_t)(off / 4) << 10) | ((uint32_t)rn << 5) | rt); }
    void fmovDX(int dd, int xn) { w(0x9e670000u | ((uint32_t)xn << 5) | dd); }   // Dd = Xn bits
    void fmovXD(int xd, int dn) { w(0x9e660000u | ((uint32_t)dn << 5) | xd); }   // Xd = Dn bits
    void fmovDD(int dd, int dn) { w(0x1e604000u | ((uint32_t)dn << 5) | dd); }
    void faddD(int dd, int dn, int dm) { w(0x1e602800u | ((uint32_t)dm << 16) | ((uint32_t)dn << 5) | dd); }
    void fsubD(int dd, int dn, int dm) { w(0x1e603800u | ((uint32_t)dm << 16) | ((uint32_t)dn << 5) | dd); }
    void fmulD(int dd, int dn, int dm) { w(0x1e600800u | ((uint32_t)dm << 16) | ((uint32_t)dn << 5) | dd); }
    void fdivD(int dd, int dn, int dm) { w(0x1e601800u | ((uint32_t)dm << 16) | ((uint32_t)dn << 5) | dd); }
    void fcvtSD(int sd, int dn) { w(0x1e624000u | ((uint32_t)dn << 5) | sd); }   // Sd = (float)Dn
    void fcvtDS(int dd, int sn) { w(0x1e22c000u | ((uint32_t)sn << 5) | dd); }   // Dd = (double)Sn
    void scvtfD(int dd, int xn) { w(0x9e620000u | ((uint32_t)xn << 5) | dd); }   // Dd = (double)Xn (signed)
    void blr(int xn) { w(0xd63f0000u | ((uint32_t)xn << 5)); }
    void retI()      { w(0xd65f03c0u); }
    // sub/add sp,sp,#v — decomposed into an optional #hi,LSL#12 part + a #lo part so any
    // frame up to (4095<<12)+4095 is reachable (a single #imm12 only covers 0..4095).
    void subSpImm(int v) {
        int hi = (v >> 12) & 0xfff, lo = v & 0xfff;
        if (hi) w(0xd1400000u | ((uint32_t)hi << 10) | (31u << 5) | 31u);   // sub sp,sp,#hi,LSL#12
        if (lo || !hi) w(0xd1000000u | ((uint32_t)lo << 10) | (31u << 5) | 31u);
    }
    void addSpImm(int v) {
        int hi = (v >> 12) & 0xfff, lo = v & 0xfff;
        if (lo || !hi) w(0x91000000u | ((uint32_t)lo << 10) | (31u << 5) | 31u);
        if (hi) w(0x91400000u | ((uint32_t)hi << 10) | (31u << 5) | 31u);   // add sp,sp,#hi,LSL#12
    }
    void addFromSp(int d, int imm) { w(0x91000000u | ((uint32_t)imm << 10) | (31u << 5) | d); }   // add Xd, sp, #imm
    void stpPreX(int t1, int t2) { w(0xa9800000u | (0x7eu << 15) | ((uint32_t)t2 << 10) | (31u << 5) | t1); }  // stp Xt1,Xt2,[sp,#-16]!
    void ldpPostX(int t1, int t2){ w(0xa8c00000u | (0x02u << 15) | ((uint32_t)t2 << 10) | (31u << 5) | t1); }  // ldp Xt1,Xt2,[sp],#16
    void stpPreD(int t1, int t2) { w(0x6d800000u | (0x7eu << 15) | ((uint32_t)t2 << 10) | (31u << 5) | t1); }
    void ldpPostD(int t1, int t2){ w(0x6cc00000u | (0x02u << 15) | ((uint32_t)t2 << 10) | (31u << 5) | t1); }
    void call(uint64_t addr) { movImm(9, addr); blr(9); }
    void bBlock(int blk) { jpatch.push_back({c.size(), blk}); w(0x14000000u); }   // b <blk> (patched)
    size_t cbz0() { size_t at = c.size(); w(0xb4000000u); return at; }             // cbz x0, . (patched)
    void patchCbz(size_t at, size_t target) {
        uint32_t word; std::memcpy(&word, &c[at], 4);
        uint32_t imm19 = (uint32_t)(((int64_t)target - (int64_t)at) / 4) & 0x7ffff;
        word |= (imm19 << 5); std::memcpy(&c[at], &word, 4);
    }
    static int csetField(const std::string& o) {   // signed cmp → cset condition field
        return o == "==" ? 1 : o == "!=" ? 0 : o == "<" ? 10 : o == "<=" ? 12 : o == ">" ? 13 : /*>=*/ 11;
    }

    int slot(int id) const { return id * 8; }
    int temp(int i) const { return (nvals + i) * 8; }
    bool isFloatVal(int id) const { return fn.vals[id].ty.isFloating(); }

    // value access: register (movReg) or sp slot. Mirrors x86 ldG/stG/ldX/stX.
    void ldG(int rg, int id) { if (vreg[id] >= 0) movRR(rg, vreg[id]); else ldrXslot(rg, slot(id)); }
    void stG(int rg, int id) { if (vreg[id] >= 0) movRR(vreg[id], rg); else strXslot(rg, slot(id)); }
    void ldX(int dg, int id) { if (vxmm[id] >= 0) fmovDD(dg, vxmm[id]); else ldrDslot(dg, slot(id)); }
    void stX(int dg, int id) { if (vxmm[id] >= 0) fmovDD(vxmm[id], dg); else strDslot(dg, slot(id)); }
    void stFloatBits(int id) { if (vxmm[id] >= 0) fmovDX(vxmm[id], 0); else strXslot(0, slot(id)); }  // x0 bits → home
    void ldFloatBits(int id) { if (vxmm[id] >= 0) fmovXD(0, vxmm[id]); else ldrXslot(0, slot(id)); }  // home → x0 bits
    void loadG0(int id) { if (cX0 == id) { cX0 = -1; return; } ldG(0, id); cX0 = -1; }
    void loadX0(int id) { if (cD0 == id) { cD0 = -1; return; } ldX(0, id); cD0 = -1; }
    void setResultCache(const Inst& in, int id) {
        int k = resultCacheKind(in);   // 0 x0 / 1 d0 / 2 neither
        cX0 = (k == 0) ? id : -1;
        cD0 = (k == 1) ? id : -1;
    }
    void narrowIfFloat(const Type& t) { if (t.base == Type::Float) { fcvtSD(0, 0); fcvtDS(0, 0); } }  // round d0 to float32
    void extX0(int bytes, bool uns) {
        if (bytes >= 8) return;
        if (bytes == 4) { uns ? uxtw(0, 0) : sxtw(0, 0); return; }
        if (bytes == 2) { uns ? uxth(0, 0) : sxth(0, 0); return; }
        uns ? uxtb(0, 0) : sxtb(0, 0);
    }
    // Type-width memory access shared by global Load/Store and local-array LoadL/StoreL:
    // load reads [base] → x0 (int) / d0 (float); store writes x0/d0 to [base].
    void loadFromBase(const Type& t, int base) {
        if (t.isFloating()) { if (t.elemBytes() == 8) ldrDreg(0, base); else { ldrSreg(0, base); fcvtDS(0, 0); } return; }
        if (t.isPointer() || t.elemBytes() == 8) ldrX(0, base);
        else if (t.elemBytes() == 4) { t.isUnsigned ? ldrW(0, base) : ldrSW(0, base); }
        else if (t.elemBytes() == 2) { t.isUnsigned ? ldrH(0, base) : ldrSH(0, base); }
        else                         { t.isUnsigned ? ldrB(0, base) : ldrSB(0, base); }
    }
    void storeToBase(const Type& t, int base) {
        if (t.isFloating()) { if (t.elemBytes() == 8) strDreg(0, base); else { fcvtSD(0, 0); strSreg(0, base); } return; }
        if (t.isPointer() || t.elemBytes() == 8) strX(0, base);
        else if (t.elemBytes() == 4) strW(0, base);
        else if (t.elemBytes() == 2) strH(0, base);
        else                         strB(0, base);
    }

    // Phi edge-copies pred→succ: two-phase (operands→temps, temps→phi), like x86.
    void phiCopies(int pred, int succ) {
        std::vector<std::pair<int, int>> pr;
        for (int id : fn.bbs[succ].insts) {
            const Inst& in = fn.vals[id];
            if (in.op != Op::Phi) break;
            for (size_t k = 0; k < in.phiPred.size(); ++k) if (in.phiPred[k] == pred) { pr.push_back({id, in.a[k]}); break; }
        }
        for (size_t i = 0; i < pr.size(); ++i) {
            if (fn.vals[pr[i].first].ty.isFloating()) { ldX(0, pr[i].second); strDslot(0, temp((int)i)); }
            else                                       { ldG(0, pr[i].second); strXslot(0, temp((int)i)); }
        }
        for (size_t i = 0; i < pr.size(); ++i) {
            if (fn.vals[pr[i].first].ty.isFloating()) { ldrDslot(0, temp((int)i)); stX(0, pr[i].first); }
            else                                       { ldrXslot(0, temp((int)i)); stG(0, pr[i].first); }
        }
    }

    void emitInst(int id) {
        const Inst& in = fn.vals[id];
        switch (in.op) {
            case Op::Phi: return;
            case Op::ConstI: { movImm(0, (uint64_t)coerce(SI(in.ci), in.ty).i); stG(0, id); return; }
            case Op::ConstF: { double d = coerce(SF(in.cf), in.ty).d; uint64_t bits; std::memcpy(&bits, &d, 8); movImm(0, bits); stFloatBits(id); return; }
            case Op::Param: { ldrX(9, 19); ldrXofs(0, 9, (int)(in.paramIdx * 8));      // x9=pvals; x0=pvals[i]
                              if (in.ty.isFloating()) stFloatBits(id); else stG(0, id); return; }
            case Op::Tid: case Op::Ctaid: case Op::Ntid: case Op::Nctaid: {
                int base = in.op == Op::Tid ? 0 : in.op == Op::Ctaid ? 3 : in.op == Op::Ntid ? 6 : 9;
                ldrWofs(0, 19, 8 + (base + in.dim) * 4); stG(0, id); return;           // w load zero-extends
            }
            case Op::Bin: {
                if (in.ty.isFloating()) {
                    loadX0(in.a[0]); ldX(1, in.a[1]);
                    if (in.s == "+") faddD(0, 0, 1); else if (in.s == "-") fsubD(0, 0, 1);
                    else if (in.s == "*") fmulD(0, 0, 1); else if (in.s == "/") fdivD(0, 0, 1);
                    else { bad(); return; }
                    narrowIfFloat(in.ty); stX(0, id); return;
                }
                const std::string& o = in.s;
                if (o == "/" || o == "%") {
                    ldG(0, in.a[0]); ldG(1, in.a[1]);                                   // x0=a, x1=b
                    movImm(2, o == "%" ? 1 : 0); movImm(3, in.ty.isUnsigned ? 1 : 0);
                    call((uint64_t)&vgre_ssa_idiv); stG(0, id); return;
                }
                if (o == "&&" || o == "||") {
                    ldG(0, in.a[0]); cmpRR(0, 31); csetF(0, 0); strXslot(0, temp(0));   // (a!=0)
                    ldG(0, in.a[1]); cmpRR(0, 31); csetF(0, 0); ldrXslot(1, temp(0));   // x0=(b!=0), x1=(a!=0)
                    if (o == "&&") and3(0, 0, 1); else orr3(0, 0, 1); stG(0, id); return;
                }
                loadG0(in.a[0]); ldG(1, in.a[1]);
                if (o == "+") add3(0, 0, 1); else if (o == "-") sub3(0, 0, 1); else if (o == "*") mul3(0, 0, 1);
                else if (o == "&") and3(0, 0, 1); else if (o == "|") orr3(0, 0, 1); else if (o == "^") eor3(0, 0, 1);
                else if (o == "<<") lslv(0, 0, 1); else if (o == ">>") { in.ty.isUnsigned ? lsrv(0, 0, 1) : asrv(0, 0, 1); }
                else { bad(); return; }
                extX0(in.ty.elemBytes(), in.ty.isUnsigned); stG(0, id); return;
            }
            case Op::Cmp: {
                bool fl = isFloatVal(in.a[0]) || isFloatVal(in.a[1]);
                const std::string& o = in.s;
                if (fl) {
                    int pred = o == "<" ? 0 : o == "<=" ? 1 : o == ">" ? 2 : o == ">=" ? 3 : o == "==" ? 4 : 5;
                    ldX(0, in.a[0]); ldX(1, in.a[1]); movImm(0, (uint64_t)pred);        // d0,d1 args; w0=pred
                    call((uint64_t)&vgre_ssa_fcmp); sxtw(0, 0); stG(0, id); return;
                }
                loadG0(in.a[0]); ldG(1, in.a[1]); cmpRR(0, 1); csetF(0, csetField(o)); stG(0, id); return;
            }
            case Op::Un: {
                if (in.s == "!") { ldG(0, in.a[0]); cmpRR(0, 31); csetF(0, 1); stG(0, id); return; }   // ==0
                if (in.ty.isFloating()) { ldFloatBits(in.a[0]); movImm(1, 0x8000000000000000ull); eor3(0, 0, 1); stFloatBits(id); return; }
                ldG(0, in.a[0]); if (in.s == "-") negR(0, 0); else notR(0, 0); extX0(in.ty.elemBytes(), in.ty.isUnsigned); stG(0, id); return;
            }
            case Op::Sel: {
                const bool fl = in.ty.isFloating();
                if (fl) { ldX(0, in.a[2]); stX(0, id); } else { ldG(0, in.a[2]); stG(0, id); }          // res = else
                ldG(1, in.a[0]); size_t j = c.size(); w(0xb4000000u | 1u);                              // cbz x1, skip (patched)
                if (fl) { ldX(0, in.a[1]); stX(0, id); } else { ldG(0, in.a[1]); stG(0, id); }          // res = then
                patchCbz(j, c.size());
                return;
            }
            case Op::Cast: {
                const Type& t = in.ty; int op0 = in.a[0];
                if (t.isFloating()) {
                    if (isFloatVal(op0)) { ldX(0, op0); narrowIfFloat(t); } else { ldG(0, op0); scvtfD(0, 0); narrowIfFloat(t); }
                    stX(0, id); return;
                }
                if (t.isPointer()) { ldG(0, op0); stG(0, id); return; }
                if (isFloatVal(op0)) { ldX(0, op0); movImm(0, (uint64_t)t.base); movImm(1, t.isUnsigned ? 1 : 0);
                                       call((uint64_t)&vgre_ssa_sat); stG(0, id); return; }
                ldG(0, op0); extX0(t.elemBytes(), t.isUnsigned); stG(0, id); return;
            }
            case Op::Load: {   // x0 = base + idx*bytes
                loadG0(in.a[0]); ldG(1, in.a[1]); movImm(2, (uint64_t)in.elemBytes); mul3(1, 1, 2); add3(0, 0, 1);
                loadFromBase(in.ty, 0); if (in.ty.isFloating()) stX(0, id); else stG(0, id); return;
            }
            case Op::Store: {   // x2 = base + idx*bytes
                ldG(2, in.a[0]); ldG(1, in.a[1]); movImm(0, (uint64_t)in.elemBytes); mul3(1, 1, 0); add3(2, 2, 1);
                if (in.ty.isFloating()) ldX(0, in.a[2]); else ldG(0, in.a[2]);
                storeToBase(in.ty, 2); return;
            }
            case Op::CallMath: {
                if (in.a.size() == 1) {
                    int mid = mathId(in.s); if (mid < 0) { bad(); return; }
                    ldX(0, in.a[0]); movImm(0, (uint64_t)mid); call((uint64_t)&vgre_ssa_m1);
                } else if (in.a.size() == 2) {
                    if (!in.ty.isFloating()) {   // integer min/max via csel (signed, matching the evaluator)
                        ldG(0, in.a[0]); ldG(1, in.a[1]); cmpRR(0, 1);
                        cselC(0, 0, 1, in.s == "max" ? 10 : 13);   // max: ge (a>=b?a:b); min: le (a<=b?a:b)
                        extX0(in.ty.elemBytes(), in.ty.isUnsigned); stG(0, id); return;
                    }
                    uint64_t op = (in.s == "fmax" || in.s == "fmaxf") ? 1u : (in.s == "pow" || in.s == "powf") ? 2u : 0u;
                    ldX(0, in.a[0]); ldX(1, in.a[1]); movImm(0, op); call((uint64_t)&vgre_ssa_m2);
                } else {
                    ldX(0, in.a[0]); ldX(1, in.a[1]); ldX(2, in.a[2]); call((uint64_t)&vgre_ssa_m3);
                }
                narrowIfFloat(in.ty); stX(0, id); return;
            }
            case Op::LoadL: {   // x2 = sp + (arrBase+idx)*8 (elements are 8-byte slots)
                ldG(1, in.a[0]); movImm(2, 8); mul3(1, 1, 2);
                movImm(2, (uint64_t)(arrBase[in.arrId] * 8)); add3(1, 1, 2);
                addFromSp(2, 0); add3(2, 2, 1);
                loadFromBase(in.ty, 2); if (in.ty.isFloating()) stX(0, id); else stG(0, id); return;
            }
            case Op::StoreL: {  // x2 = sp + (arrBase+idx)*8; write in.a[1]
                ldG(1, in.a[0]); movImm(2, 8); mul3(1, 1, 2);
                movImm(2, (uint64_t)(arrBase[in.arrId] * 8)); add3(1, 1, 2);
                addFromSp(2, 0); add3(2, 2, 1);
                if (in.ty.isFloating()) ldX(0, in.a[1]); else ldG(0, in.a[1]);
                storeToBase(in.ty, 2); return;
            }
            case Op::Barrier: call((uint64_t)&vgre_ssa_barrier); return;   // __syncthreads → yield the fiber to the block scheduler
            case Op::WarpShfl: case Op::WarpVote: case Op::WarpReduce:
            case Op::WarpMatch: case Op::WarpActive: bad(); return;       // warp intrinsics run on the cooperative evaluator
            case Op::LoadS: {   // x0 = ctx.shared + sharedOff[arrId] + idx*elemBytes
                ldrXofs(0, 19, (int)offsetof(ThreadCtx, shared)); ldG(1, in.a[0]); movImm(2, (uint64_t)in.elemBytes); mul3(1, 1, 2);
                movImm(2, (uint64_t)sharedOff[in.arrId]); add3(1, 1, 2); add3(0, 0, 1);
                loadFromBase(in.ty, 0); if (in.ty.isFloating()) stX(0, id); else stG(0, id); return;
            }
            case Op::StoreS: {   // x2 = ctx.shared + sharedOff[arrId] + idx*elemBytes; write in.a[1]
                ldrXofs(2, 19, (int)offsetof(ThreadCtx, shared)); ldG(1, in.a[0]); movImm(0, (uint64_t)in.elemBytes); mul3(1, 1, 0);
                movImm(0, (uint64_t)sharedOff[in.arrId]); add3(1, 1, 0); add3(2, 2, 1);
                if (in.ty.isFloating()) ldX(0, in.a[1]); else ldG(0, in.a[1]);
                storeToBase(in.ty, 2); return;
            }
            default: return;
        }
    }

    void prologue() {
        stpPreX(29, 30); stpPreX(19, 20); stpPreX(21, 22); stpPreX(23, 24); stpPreX(25, 26); stpPreX(27, 28);
        stpPreD(8, 9); stpPreD(10, 11); stpPreD(12, 13); stpPreD(14, 15);
        if (frame) subSpImm(frame);
        movRR(19, 0);   // x19 = ThreadCtx* (first arg)
    }
    void epilogue() {
        if (frame) addSpImm(frame);
        ldpPostD(14, 15); ldpPostD(12, 13); ldpPostD(10, 11); ldpPostD(8, 9);
        ldpPostX(27, 28); ldpPostX(25, 26); ldpPostX(23, 24); ldpPostX(21, 22); ldpPostX(19, 20); ldpPostX(29, 30);
        retI();
    }

    bool build() {
        allocateRegs();
        const int n = (int)fn.bbs.size();
        int maxTemps = 0;
        for (auto& bb : fn.bbs) { int p = 0; for (int id : bb.insts) { if (fn.vals[id].op == Op::Phi) ++p; else break; } if (p > maxTemps) maxTemps = p; }
        // Local arrays get 8-byte slots after the value + phi-temp slots (element k of
        // array a is slot arrBase[a]+k), sp-relative like every other slot.
        arrBase.assign(fn.localArrays.size(), 0);
        int arrElems = 0;
        for (size_t i = 0; i < fn.localArrays.size(); ++i) { arrBase[i] = nvals + maxTemps + arrElems; arrElems += fn.localArrays[i].second; }
        // __shared__ arrays live in the block-shared buffer (ctx.shared), 8-byte aligned.
        sharedOff.assign(fn.sharedArrays.size(), 0);
        int so = 0;
        for (size_t i = 0; i < fn.sharedArrays.size(); ++i) { sharedOff[i] = so; so += ((fn.sharedArrays[i].second * fn.sharedArrays[i].first.elemBytes() + 7) / 8) * 8; }
        frame = (((nvals + maxTemps + arrElems) * 8 + 15) / 16) * 16;
        if (frame > 32760) return false;   // beyond the scaled slot-offset imm range — fall back to the evaluator
        prologue();
        if (arrElems) for (int i = 0; i < arrElems; ++i) strXslot(31, (nvals + maxTemps + i) * 8);  // zero-init scratch arrays (str xzr)
        off.assign(n, 0);
        for (int blk = 0; blk < n; ++blk) {
            off[blk] = c.size();
            cX0 = cD0 = -1;
            const BB& bb = fn.bbs[blk];
            for (int id : bb.insts) {
                const Inst& in = fn.vals[id];
                if (in.op == Op::Ret) { epilogue(); break; }
                if (in.op == Op::Br) { phiCopies(blk, in.bbT); bBlock(in.bbT); break; }
                if (in.op == Op::CondBr) {
                    loadG0(in.a[0]); size_t j = cbz0();                // cbz x0, <false> (patched)
                    phiCopies(blk, in.bbT); bBlock(in.bbT);
                    patchCbz(j, c.size());
                    phiCopies(blk, in.bbF); bBlock(in.bbF);
                    break;
                }
                emitInst(id);
                if (!ok) return false;
                setResultCache(in, id);
            }
        }
        for (auto& jp : jpatch) {
            uint32_t word; std::memcpy(&word, &c[jp.first], 4);
            uint32_t imm26 = (uint32_t)(((int64_t)off[jp.second] - (int64_t)jp.first) / 4) & 0x3ffffff;
            word |= imm26; std::memcpy(&c[jp.first], &word, 4);
        }
        return ok;
    }
};
#endif  // VGRE_SSA_X64 || VGRE_SSA_ARM64

}  // namespace

// Encoder self-test: every AArch64 instruction the emitter produces, checked against
// the exact bytes llvm-mc assembles for the corresponding mnemonic (baked in, so the
// test needs no external assembler). This validates the ENCODING layer on any host
// (incl. x86-64 dev/CI), independent of ARM execution. Returns true (all match) or
// sets err. On a host where neither native backend is compiled it is a no-op pass.
bool arm64EncSelfTest(std::string& err) {
#if VGRE_SSA_X64 || VGRE_SSA_ARM64
    Fn dummy;
    Arm64Asm a(dummy);
    bool ok = true;
    auto chk = [&](const char* name, std::initializer_list<uint8_t> exp) {
        std::vector<uint8_t> e(exp);
        if (a.c != e) { if (ok) err = std::string("arm64 enc: ") + name; ok = false; }
        a.c.clear();
    };
    a.movRR(0, 1);        chk("mov x0,x1",   {0xe0, 0x03, 0x01, 0xaa});
    a.movRR(19, 0);       chk("mov x19,x0",  {0xf3, 0x03, 0x00, 0xaa});
    a.add3(0, 1, 2);      chk("add",         {0x20, 0x00, 0x02, 0x8b});
    a.sub3(0, 1, 2);      chk("sub",         {0x20, 0x00, 0x02, 0xcb});
    a.mul3(0, 1, 2);      chk("mul",         {0x20, 0x7c, 0x02, 0x9b});
    a.and3(0, 1, 2);      chk("and",         {0x20, 0x00, 0x02, 0x8a});
    a.orr3(0, 1, 2);      chk("orr",         {0x20, 0x00, 0x02, 0xaa});
    a.eor3(0, 1, 2);      chk("eor",         {0x20, 0x00, 0x02, 0xca});
    a.lslv(0, 1, 2);      chk("lsl",         {0x20, 0x20, 0xc2, 0x9a});
    a.lsrv(0, 1, 2);      chk("lsr",         {0x20, 0x24, 0xc2, 0x9a});
    a.asrv(0, 1, 2);      chk("asr",         {0x20, 0x28, 0xc2, 0x9a});
    a.cmpRR(1, 2);        chk("cmp",         {0x3f, 0x00, 0x02, 0xeb});
    a.csetF(0, 1);        chk("cset eq",     {0xe0, 0x17, 0x9f, 0x9a});
    a.csetF(0, 0);        chk("cset ne",     {0xe0, 0x07, 0x9f, 0x9a});
    a.csetF(0, 10);       chk("cset lt",     {0xe0, 0xa7, 0x9f, 0x9a});
    a.csetF(0, 11);       chk("cset ge",     {0xe0, 0xb7, 0x9f, 0x9a});
    a.csetF(0, 12);       chk("cset le",     {0xe0, 0xc7, 0x9f, 0x9a});
    a.csetF(0, 13);       chk("cset gt",     {0xe0, 0xd7, 0x9f, 0x9a});
    a.negR(0, 1);         chk("neg",         {0xe0, 0x03, 0x01, 0xcb});
    a.notR(0, 1);         chk("mvn",         {0xe0, 0x03, 0x21, 0xaa});
    a.sxtw(0, 1);         chk("sxtw",        {0x20, 0x7c, 0x40, 0x93});
    a.sxth(0, 1);         chk("sxth",        {0x20, 0x3c, 0x40, 0x93});
    a.sxtb(0, 1);         chk("sxtb",        {0x20, 0x1c, 0x40, 0x93});
    a.uxtw(0, 1);         chk("uxtw",        {0xe0, 0x03, 0x01, 0x2a});
    a.uxth(0, 1);         chk("uxth",        {0x20, 0x3c, 0x00, 0x53});
    a.uxtb(0, 1);         chk("uxtb",        {0x20, 0x1c, 0x00, 0x53});
    a.movz(0, 0, 0);      chk("movz",        {0x00, 0x00, 0x80, 0xd2});
    a.movk(0, 0, 1);      chk("movk lsl16",  {0x00, 0x00, 0xa0, 0xf2});
    a.ldrXslot(0, 0);     chk("ldr x,[sp]",  {0xe0, 0x03, 0x40, 0xf9});
    a.strXslot(0, 0);     chk("str x,[sp]",  {0xe0, 0x03, 0x00, 0xf9});
    a.strXslot(0, 4088);  chk("str x,[sp,#4088]", {0xe0, 0xff, 0x07, 0xf9});
    a.ldrDslot(0, 0);     chk("ldr d,[sp]",  {0xe0, 0x03, 0x40, 0xfd});
    a.strDslot(0, 0);     chk("str d,[sp]",  {0xe0, 0x03, 0x00, 0xfd});
    a.ldrXofs(0, 9, 8);   chk("ldr x0,[x9,#8]",  {0x20, 0x05, 0x40, 0xf9});
    a.ldrWofs(0, 19, 8);  chk("ldr w0,[x19,#8]", {0x60, 0x0a, 0x40, 0xb9});
    a.ldrX(0, 0);         chk("ldr x,[x0]",  {0x00, 0x00, 0x40, 0xf9});
    a.ldrW(0, 0);         chk("ldr w,[x0]",  {0x00, 0x00, 0x40, 0xb9});
    a.ldrSW(0, 0);        chk("ldrsw",       {0x00, 0x00, 0x80, 0xb9});
    a.ldrH(0, 0);         chk("ldrh",        {0x00, 0x00, 0x40, 0x79});
    a.ldrSH(0, 0);        chk("ldrsh",       {0x00, 0x00, 0xc0, 0x79});
    a.ldrB(0, 0);         chk("ldrb",        {0x00, 0x00, 0x40, 0x39});
    a.ldrSB(0, 0);        chk("ldrsb",       {0x00, 0x00, 0xc0, 0x39});
    a.strX(0, 2);         chk("str x,[x2]",  {0x40, 0x00, 0x00, 0xf9});
    a.strW(0, 2);         chk("str w,[x2]",  {0x40, 0x00, 0x00, 0xb9});
    a.strH(0, 2);         chk("strh",        {0x40, 0x00, 0x00, 0x79});
    a.strB(0, 2);         chk("strb",        {0x40, 0x00, 0x00, 0x39});
    a.ldrDreg(0, 0);      chk("ldr d,[x0]",  {0x00, 0x00, 0x40, 0xfd});
    a.strDreg(0, 2);      chk("str d,[x2]",  {0x40, 0x00, 0x00, 0xfd});
    a.ldrSreg(0, 0);      chk("ldr s,[x0]",  {0x00, 0x00, 0x40, 0xbd});
    a.strSreg(0, 2);      chk("str s,[x2]",  {0x40, 0x00, 0x00, 0xbd});
    a.fmovDX(0, 0);       chk("fmov d0,x0",  {0x00, 0x00, 0x67, 0x9e});
    a.fmovXD(0, 0);       chk("fmov x0,d0",  {0x00, 0x00, 0x66, 0x9e});
    a.fmovDD(0, 1);       chk("fmov d0,d1",  {0x20, 0x40, 0x60, 0x1e});
    a.faddD(0, 0, 1);     chk("fadd",        {0x00, 0x28, 0x61, 0x1e});
    a.fsubD(0, 0, 1);     chk("fsub",        {0x00, 0x38, 0x61, 0x1e});
    a.fmulD(0, 0, 1);     chk("fmul",        {0x00, 0x08, 0x61, 0x1e});
    a.fdivD(0, 0, 1);     chk("fdiv",        {0x00, 0x18, 0x61, 0x1e});
    a.fcvtSD(0, 0);       chk("fcvt s,d",    {0x00, 0x40, 0x62, 0x1e});
    a.fcvtDS(0, 0);       chk("fcvt d,s",    {0x00, 0xc0, 0x22, 0x1e});
    a.scvtfD(0, 0);       chk("scvtf",       {0x00, 0x00, 0x62, 0x9e});
    a.blr(9);             chk("blr x9",      {0x20, 0x01, 0x3f, 0xd6});
    a.retI();             chk("ret",         {0xc0, 0x03, 0x5f, 0xd6});
    a.subSpImm(16);       chk("sub sp,#16",  {0xff, 0x43, 0x00, 0xd1});
    a.addSpImm(16);       chk("add sp,#16",  {0xff, 0x43, 0x00, 0x91});
    a.subSpImm(4080);     chk("sub sp,#4080",{0xff, 0xc3, 0x3f, 0xd1});
    a.stpPreX(29, 30);    chk("stp x29,x30", {0xfd, 0x7b, 0xbf, 0xa9});
    a.stpPreX(19, 20);    chk("stp x19,x20", {0xf3, 0x53, 0xbf, 0xa9});
    a.stpPreD(8, 9);      chk("stp d8,d9",   {0xe8, 0x27, 0xbf, 0x6d});
    a.ldpPostX(29, 30);   chk("ldp x29,x30", {0xfd, 0x7b, 0xc1, 0xa8});
    a.ldpPostD(8, 9);     chk("ldp d8,d9",   {0xe8, 0x27, 0xc1, 0x6c});
    a.cselC(0, 0, 1, 13); chk("csel le",     {0x00, 0xd0, 0x81, 0x9a});
    a.cselC(0, 0, 1, 10); chk("csel ge",     {0x00, 0xa0, 0x81, 0x9a});
    a.addFromSp(2, 0);    chk("add x2,sp,#0",  {0xe2, 0x03, 0x00, 0x91});
    a.addFromSp(2, 16);   chk("add x2,sp,#16", {0xe2, 0x43, 0x00, 0x91});
    if (ok) err.clear();
    return ok;
#else
    (void)err; return true;   // encoders not compiled on this host
#endif
}

#if VGRE_SSA_X64 || VGRE_SSA_ARM64   // only the native coop path uses these; the evaluator is always cooperative
// Total bytes of the block-shared buffer (matching the emitter's sharedOff layout).
static size_t ssaSharedBytes(const Fn& fn) {
    size_t s = 0;
    for (auto& sa : fn.sharedArrays) s += (((size_t)sa.second * sa.first.elemBytes() + 7) / 8) * 8;
    return s;
}
// A kernel is cooperative (needs the per-block fiber scheduler) if it uses __shared__
// memory or a __syncthreads barrier.
static bool ssaIsCoop(const Fn& fn) {
    if (!fn.sharedArrays.empty()) return true;
    for (const auto& in : fn.vals)
        if (in.op == Op::Barrier || in.op == Op::WarpShfl || in.op == Op::WarpVote ||
            in.op == Op::WarpReduce || in.op == Op::WarpMatch) return true;
    return false;
}
#endif

// ── SsaProgram wrapper ────────────────────────────────────────────────────────────
struct SsaProgram::Impl {
    Fn fn;
    bool coop = false;         // uses __shared__/__syncthreads → per-block fiber scheduler
#if VGRE_SSA_X64 || VGRE_SSA_ARM64
    void* code = nullptr;      // mmap'd W^X machine code (null ⇒ use the evaluator)
    size_t codeSize = 0;
    SsaFn nativeFn = nullptr;
    ~Impl() { if (code) munmap(code, codeSize); }
#endif
};

SsaProgram::SsaProgram() : p_(new Impl) {}
SsaProgram::~SsaProgram() = default;
bool SsaProgram::usedNative() const {
#if VGRE_SSA_X64 || VGRE_SSA_ARM64
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
    for (auto& kp : pr.module->kernels)     // __device__ helpers available for inlining
        if (kp->isDevice && !kp->isGlobal) lo.deviceFns[kp->name] = kp.get();
    if (!lo.run()) { err = lo.err; return nullptr; }
    if (!verify(lo.fn, err)) return nullptr;
    if (optimize) { runOpt(lo.fn); if (!verify(lo.fn, err)) return nullptr; }
    std::unique_ptr<SsaProgram> prog(new SsaProgram());
    prog->p_->fn = std::move(lo.fn);
#if VGRE_SSA_X64 || VGRE_SSA_ARM64
    prog->p_->coop = ssaIsCoop(prog->p_->fn);   // routes the native launch to the fiber scheduler
#endif
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
#if VGRE_SSA_ARM64
    // AArch64 native emission is OPT-IN (VGRE_SSA_ARM_NATIVE=1) until validated on real
    // ARM hardware — see the guard comment up top. Same best-effort contract: on any
    // unsupported op build() bails and launch() uses the (identical) evaluator.
    if (std::getenv("VGRE_SSA_ARM_NATIVE")) {
        Arm64Asm asmb(prog->p_->fn);
        if (asmb.build() && !asmb.c.empty()) {
            size_t sz = ((asmb.c.size() + 4095) / 4096) * 4096;
#if defined(__APPLE__)
            // Apple Silicon: MAP_JIT memory, per-thread W^X toggle, and an explicit
            // I-cache flush (mandatory for self-modifying code on ARM).
            void* mem = mmap(nullptr, sz, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANON | MAP_JIT, -1, 0);
            if (mem != MAP_FAILED) {
                pthread_jit_write_protect_np(0);
                std::memcpy(mem, asmb.c.data(), asmb.c.size());
                pthread_jit_write_protect_np(1);
                sys_icache_invalidate(mem, asmb.c.size());
                prog->p_->code = mem; prog->p_->codeSize = sz;
                prog->p_->nativeFn = reinterpret_cast<SsaFn>(mem);
            }
#else
            // ARM Linux: RW → RX via mprotect, then flush the I-cache.
            void* mem = mmap(nullptr, sz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (mem != MAP_FAILED) {
                std::memcpy(mem, asmb.c.data(), asmb.c.size());
                if (mprotect(mem, sz, PROT_READ | PROT_EXEC) == 0) {
                    __builtin___clear_cache(reinterpret_cast<char*>(mem), reinterpret_cast<char*>(mem) + asmb.c.size());
                    prog->p_->code = mem; prog->p_->codeSize = sz;
                    prog->p_->nativeFn = reinterpret_cast<SsaFn>(mem);
                } else { munmap(mem, sz); }
            }
#endif
        }
    }
#endif
    return prog;
}

bool SsaProgram::launch(Extent grid, Extent block, void* const* args, int numArgs) {
    const Fn& fn = p_->fn;
    if (numArgs < (int)fn.ptypes.size()) return false;
    const uint32_t bx = block.x, by = block.y, bz = block.z;
#if VGRE_SSA_X64 || VGRE_SSA_ARM64
    if (p_->nativeFn) {
        // Pre-decode each parameter into a uniform 8-byte value (shared by all threads).
        std::vector<int64_t> pvals(fn.ptypes.size(), 0);
        for (size_t i = 0; i < fn.ptypes.size(); ++i) {
            SVal pv = memLoad(reinterpret_cast<int64_t>(args[i]), fn.ptypes[i]);
            if (fn.ptypes[i].isFloating()) { double d = pv.d; std::memcpy(&pvals[i], &d, 8); }
            else pvals[i] = pv.i;
        }
#if VGRE_SSA_X64 || VGRE_SSA_ARM64
        if (p_->coop) {
            // __shared__/__syncthreads: run each thread's machine code on its own ucontext
            // fiber; a per-block scheduler resumes every runnable fiber to its next barrier
            // (or to completion), then loops — so all threads reach barrier N before any
            // crosses it. Barriers `call vgre_ssa_barrier`, which swapcontexts back here.
            const uint32_t nT = bx * by * bz;
            const size_t shBytes = ssaSharedBytes(fn);
            for (uint32_t gz = 0; gz < grid.z; ++gz)
            for (uint32_t gy = 0; gy < grid.y; ++gy)
            for (uint32_t gx = 0; gx < grid.x; ++gx) {
                std::vector<char> shared(shBytes, 0);            // one block-shared buffer
                std::vector<ThreadCtx> tc(nT);
                std::vector<NFiber> fib(nT);
                ucontext_t sched;
                for (uint32_t t = 0; t < nT; ++t) {
                    ThreadCtx& c = tc[t];
                    c.pvals = pvals.data(); c.shared = shBytes ? shared.data() : nullptr;
                    c.idx[0] = t % bx; c.idx[1] = (t / bx) % by; c.idx[2] = t / (bx * by);
                    c.idx[3] = gx; c.idx[4] = gy; c.idx[5] = gz;
                    c.idx[6] = bx; c.idx[7] = by; c.idx[8] = bz;
                    c.idx[9] = grid.x; c.idx[10] = grid.y; c.idx[11] = grid.z;
                    NFiber& f = fib[t];
                    f.fn = p_->nativeFn; f.tctx = &c; f.sched = &sched; f.stack.resize(1 << 17);   // 128 KiB
                    getcontext(&f.ctx);
                    f.ctx.uc_stack.ss_sp = f.stack.data(); f.ctx.uc_stack.ss_size = f.stack.size();
                    f.ctx.uc_link = &sched;                       // returning from the body lands back here
                    makecontext(&f.ctx, nfTrampoline, 0);
                }
                for (long long round = 0; round < (1LL << 34); ++round) {
                    bool allDone = true;
                    for (uint32_t t = 0; t < nT; ++t) {
                        if (fib[t].done) continue;
                        allDone = false;
                        g_nfCur = &fib[t];
                        swapcontext(&sched, &fib[t].ctx);         // runs until barrier or done
                    }
                    if (allDone) break;
                }
            }
            return true;
        }
#endif
        ThreadCtx ctx; ctx.pvals = pvals.data(); ctx.shared = nullptr;
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
    // Cooperative per-block execution. All threads of a block run in lock-step across
    // __syncthreads barriers and share the block's __shared__ arrays. Each thread keeps
    // resumable state (v, local arrays, PC = block+inst-index); the scheduler runs every
    // thread to its next barrier (or to Ret), then repeats — so no thread crosses barrier
    // N until all threads have reached it. Barrier-free kernels just run to Ret in round 0.
    struct TState {
        std::vector<SVal> v;
        std::vector<std::vector<SVal>> larr;
        int bb = 0, j = 0, prevBB = -1;
        bool entered = false, finished = false;
        // Warp-cooperative rendezvous (WarpShfl/WarpVote): when a thread reaches a warp
        // op it publishes its operand in `pub`, records the op id in `parkOp`, and
        // suspends; the scheduler resolves the whole warp, writes each lane's result into
        // v[parkOp], and sets `parkResolved` so the resumed thread consumes it.
        SVal pub{};
        int parkOp = -1;
        bool parkResolved = false;
    };
    // Run one thread from its current PC until it hits a Barrier (suspend, PC past it) or
    // Ret (finish). `shared` is the block's __shared__ storage (written across threads).
    auto runSegment = [&](TState& t, std::vector<std::vector<SVal>>& shared,
                          const uint32_t tid[3], const uint32_t ctaid[3],
                          const uint32_t ntid[3], const uint32_t nctaid[3]) {
        auto& v = t.v; auto& larr = t.larr;
        for (long long guard = 0; guard < (1LL << 34); ++guard) {
            if (!t.entered) {   // resolve phis on first entry to a block (parallel copy)
                std::vector<std::pair<int, SVal>> phiSet;
                for (int id : fn.bbs[t.bb].insts) {
                    const Inst& in = fn.vals[id]; if (in.op != Op::Phi) break;
                    SVal sel{};
                    for (size_t kk = 0; kk < in.phiPred.size(); ++kk)
                        if (in.phiPred[kk] == t.prevBB) { sel = v[in.a[kk]]; break; }
                    phiSet.push_back({id, sel});
                }
                for (auto& ps : phiSet) v[ps.first] = ps.second;
                t.entered = true;
            }
            const auto& insts = fn.bbs[t.bb].insts;
            for (; t.j < (int)insts.size(); ++t.j) {
                int id = insts[t.j]; const Inst& in = fn.vals[id];
                switch (in.op) {
                    case Op::Phi: break;   // resolved on block entry
                    case Op::ConstI: v[id] = coerce(SI(in.ci), in.ty); break;
                    case Op::ConstF: v[id] = coerce(SF(in.cf), in.ty); break;
                    case Op::Param: v[id] = memLoad(reinterpret_cast<int64_t>(args[in.paramIdx]), fn.ptypes[in.paramIdx]); break;
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
                    case Op::CallMath: v[id] = mathfn(in.s, v[in.a[0]], in.a.size() > 1 ? &v[in.a[1]] : nullptr, in.a.size() > 2 ? &v[in.a[2]] : nullptr, in.ty); break;
                    case Op::Load:  { int64_t addr = asI(v[in.a[0]]) + asI(v[in.a[1]]) * in.elemBytes; v[id] = memLoad(addr, in.ty); break; }
                    case Op::Store: { int64_t addr = asI(v[in.a[0]]) + asI(v[in.a[1]]) * in.elemBytes; memStore(addr, in.ty, v[in.a[2]]); break; }
                    case Op::LoadL: { int64_t k = asI(v[in.a[0]]); auto& a = larr[in.arrId]; v[id] = (k >= 0 && k < (int64_t)a.size()) ? coerce(a[(size_t)k], in.ty) : coerce(SI(0), in.ty); break; }
                    case Op::StoreL:{ int64_t k = asI(v[in.a[0]]); auto& a = larr[in.arrId]; if (k >= 0 && k < (int64_t)a.size()) a[(size_t)k] = coerce(v[in.a[1]], in.ty); break; }
                    case Op::LoadS: { int64_t k = asI(v[in.a[0]]); auto& a = shared[in.arrId]; v[id] = (k >= 0 && k < (int64_t)a.size()) ? coerce(a[(size_t)k], in.ty) : coerce(SI(0), in.ty); break; }
                    case Op::StoreS:{ int64_t k = asI(v[in.a[0]]); auto& a = shared[in.arrId]; if (k >= 0 && k < (int64_t)a.size()) a[(size_t)k] = coerce(v[in.a[1]], in.ty); break; }
                    case Op::Barrier: ++t.j; return;   // suspend; resume at the next instruction
                    case Op::WarpShfl:
                    case Op::WarpVote:
                    case Op::WarpReduce:
                    case Op::WarpMatch:
                        if (t.parkOp == id && t.parkResolved) {   // scheduler resolved the warp → v[id] is set
                            t.parkOp = -1; t.parkResolved = false; break;   // consume, fall through to ++t.j
                        }
                        // publish var (shfl a[0]) / predicate|value (vote/reduce/match a[1])
                        t.pub = (in.op == Op::WarpShfl) ? v[in.a[0]] : v[in.a[1]];
                        t.parkOp = id; t.parkResolved = false;
                        return;   // suspend WITHOUT advancing j — re-enter this op once the warp is resolved
                    case Op::WarpActive: {
                        // __activemask() = the warp's non-exited lanes. Both this scheduler and
                        // the interpreter retire lanes in strictly increasing lane order (each
                        // runs to Ret before the next starts, absent a rendezvous), so when
                        // lane L reads activemask the lower lanes 0..L-1 have already retired:
                        // mask = fullWarpMask with bits [0,lane) cleared. Bit-exact vs the
                        // interpreter/compiled tiers for convergent code (their documented
                        // sequential-done behavior; real HW returns the full mask).
                        uint32_t lin = tid[0] + tid[1] * bx + tid[2] * bx * by, ntot = bx * by * bz;
                        uint32_t wbase = (lin / 32u) * 32u, lane = lin & 31u;
                        uint32_t cnt = ntot - wbase; if (cnt > 32u) cnt = 32u;
                        uint32_t full = (cnt >= 32u) ? 0xFFFFFFFFu : ((1u << cnt) - 1u);
                        uint32_t mask = full & ~((1u << lane) - 1u);
                        v[id] = coerce(SI((int64_t)mask), in.ty);
                        break;
                    }
                    case Op::Br:     t.prevBB = t.bb; t.bb = in.bbT; t.j = 0; t.entered = false; goto nextblock;
                    case Op::CondBr: t.prevBB = t.bb; t.bb = (asI(v[in.a[0]]) != 0) ? in.bbT : in.bbF; t.j = 0; t.entered = false; goto nextblock;
                    case Op::Ret:    t.finished = true; return;
                }
            }
            nextblock:;
        }
    };

    const uint32_t nT = bx * by * bz;
    for (uint32_t gz = 0; gz < grid.z; ++gz)
    for (uint32_t gy = 0; gy < grid.y; ++gy)
    for (uint32_t gx = 0; gx < grid.x; ++gx) {
        std::vector<std::vector<SVal>> shared(fn.sharedArrays.size());   // one instance per block
        for (size_t si = 0; si < fn.sharedArrays.size(); ++si)
            shared[si].assign(fn.sharedArrays[si].second, coerce(SI(0), fn.sharedArrays[si].first));
        std::vector<TState> th(nT);
        for (uint32_t lin = 0; lin < nT; ++lin) {
            TState& t = th[lin];
            t.v.assign(fn.vals.size(), SVal{});
            t.larr.resize(fn.localArrays.size());
            for (size_t ai = 0; ai < fn.localArrays.size(); ++ai)
                t.larr[ai].assign(fn.localArrays[ai].second, coerce(SI(0), fn.localArrays[ai].first));
            t.bb = fn.entry;
        }
        const uint32_t ctaid[3] = {gx, gy, gz}, nctaid[3] = {grid.x, grid.y, grid.z}, ntid[3] = {bx, by, bz};
        const uint32_t nwarps = (nT + 31) / 32;
        for (long long round = 0; round < (1LL << 34); ++round) {   // one round advances every thread by one barrier segment
            bool allDone = true;
            for (uint32_t lin = 0; lin < nT; ++lin) {
                if (th[lin].finished) continue;
                const uint32_t tid[3] = {lin % bx, (lin / bx) % by, lin / (bx * by)};
                runSegment(th[lin], shared, tid, ctaid, ntid, nctaid);
                if (!th[lin].finished) allDone = false;   // suspended at a barrier / warp op
            }
            if (allDone) break;

            // Warp rendezvous: release each warp whose live lanes are all parked at a
            // warp op (__shfl_*_sync / __ballot|any|all_sync). Compute the active mask
            // (parked lanes), snapshot the published operands, then write each lane's
            // result into v[parkOp]. Mirrors the compiled tier's coopShuffle/coopVote.
            bool resolvedAny = false;
            for (uint32_t w = 0; w < nwarps; ++w) {
                const uint32_t base = w * 32;
                bool anyLive = false, allParked = true;
                for (uint32_t l = 0; l < 32 && base + l < nT; ++l) {
                    if (th[base + l].finished) continue;
                    anyLive = true;
                    if (th[base + l].parkOp < 0) { allParked = false; break; }
                }
                if (!anyLive || !allParked) continue;
                uint32_t active = 0; SVal snap[32] = {};
                for (uint32_t l = 0; l < 32 && base + l < nT; ++l)
                    if (!th[base + l].finished && th[base + l].parkOp >= 0) { active |= (1u << l); snap[l] = th[base + l].pub; }
                for (uint32_t l = 0; l < 32 && base + l < nT; ++l) {
                    TState& t = th[base + l];
                    if (t.finished || t.parkOp < 0) continue;
                    const Inst& in = fn.vals[t.parkOp];
                    if (in.op == Op::WarpShfl) {
                        int mode = in.dim;
                        int width = in.a.size() > 2 ? (int)asI(t.v[in.a[2]]) : 32;
                        if (width <= 0 || width > 32) width = 32;
                        int laneArg = (int)asI(t.v[in.a[1]]);
                        int laneInSub = (int)l % width, subBase = ((int)l / width) * width;
                        int srcSub = laneInSub; bool own = false;
                        if (mode == 0)      srcSub = laneArg % width;
                        else if (mode == 1) { srcSub = laneInSub - laneArg; if (srcSub < 0) own = true; }
                        else if (mode == 2) { srcSub = laneInSub + laneArg; if (srcSub >= width) own = true; }
                        else                { srcSub = laneInSub ^ laneArg; if (srcSub >= width) own = true; }
                        int src = subBase + srcSub;
                        SVal res = (own || src < 0 || src >= 32 || !(active & (1u << src))) ? t.pub : snap[src];
                        t.v[t.parkOp] = coerce(res, in.ty);
                    } else if (in.op == Op::WarpVote) {   // op 0=ballot 1=any 2=all
                        uint32_t memMask = (uint32_t)asI(t.v[in.a[0]]);
                        uint32_t ballot = 0;
                        for (uint32_t k = 0; k < 32; ++k)
                            if ((active & (1u << k)) && asI(snap[k]) != 0) ballot |= (1u << k);
                        uint32_t masked = ballot & memMask;
                        int64_t r = in.dim == 0 ? (int64_t)masked
                                  : in.dim == 1 ? (masked != 0 ? 1 : 0)
                                  : (masked == (memMask & active) ? 1 : 0);
                        t.v[t.parkOp] = SI(r);
                    } else if (in.op == Op::WarpReduce) {   // op 0=add 1=min 2=max 3=and 4=or 5=xor
                        const int rop = in.dim; const bool sgn = in.ci != 0;
                        uint32_t memMask = (uint32_t)asI(t.v[in.a[0]]);
                        bool first = true; int64_t acc = 0;
                        for (uint32_t k = 0; k < 32; ++k) {
                            if (!(active & (1u << k)) || !(memMask & (1u << k))) continue;
                            uint32_t raw = (uint32_t)asI(snap[k]);
                            int64_t vv = sgn ? (int64_t)(int32_t)raw : (int64_t)raw;
                            if (first) { acc = vv; first = false; continue; }
                            if (rop == 0) acc = vv + acc; else if (rop == 1) acc = acc < vv ? acc : vv;
                            else if (rop == 2) acc = acc > vv ? acc : vv; else if (rop == 3) acc &= vv;
                            else if (rop == 4) acc |= vv; else acc ^= vv;
                        }
                        t.v[t.parkOp] = coerce(SI((int64_t)(uint32_t)acc), in.ty);
                    } else {   // WarpMatch — dim 0 = __match_any_sync, dim 1 = __match_all_sync
                        uint32_t memMask = (uint32_t)asI(t.v[in.a[0]]);
                        uint32_t part = 0;
                        for (uint32_t k = 0; k < 32; ++k) if ((active & (1u << k)) && (memMask & (1u << k))) part |= (1u << k);
                        uint32_t mine = (uint32_t)asI(t.pub), same = 0;
                        for (uint32_t k = 0; k < 32; ++k)
                            if ((part & (1u << k)) && (uint32_t)asI(snap[k]) == mine) same |= (1u << k);
                        uint32_t r = (in.dim == 0) ? same : ((same == part) ? part : 0u);   // any → same; all → participants iff unanimous
                        t.v[t.parkOp] = coerce(SI((int64_t)r), in.ty);
                    }
                    t.parkResolved = true; resolvedAny = true;
                }
            }
            // No warp advanced and no thread finished, yet threads remain — if every
            // remaining thread is stuck at a warp op, the rendezvous can never complete
            // (divergent/ill-formed). Break instead of spinning to the round cap.
            if (!resolvedAny) {
                bool anyNF = false, allParkedGlobal = true;
                for (uint32_t lin = 0; lin < nT; ++lin) {
                    if (th[lin].finished) continue;
                    anyNF = true;
                    if (th[lin].parkOp < 0) { allParkedGlobal = false; break; }
                }
                if (anyNF && allParkedGlobal) break;
            }
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
