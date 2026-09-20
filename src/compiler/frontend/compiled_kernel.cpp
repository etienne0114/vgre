// Tier-1 compiled execution — see include/vgre/compiler/frontend/compiled_kernel.h.
//
// One compile pass turns the AST into std::function closures bound to register
// slots; launch() runs them per-thread with no string parsing. Portable, no LLVM.

#include "vgre/compiler/frontend/compiled_kernel.h"

#include "vgre/compiler/frontend/parser.h"

#include "vgre/common/atomic_rmw.h"
#include "vgre/xla/thread_pool.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace vgre {
namespace compiler {
namespace frontend {

namespace {

// A runtime value: an int/pointer (i) or a float (f).
struct Cell {
    bool isFloat = false;
    int64_t i = 0;
    double f = 0.0;
    static Cell I(int64_t v) { Cell c; c.i = v; return c; }
    static Cell F(double v) { Cell c; c.isFloat = true; c.f = v; return c; }
    double  asF() const { return isFloat ? f : static_cast<double>(i); }
    int64_t asI() const { return isFloat ? static_cast<int64_t>(f) : i; }
};

// Per-thread execution state.
struct TS {
    std::vector<Cell> regs;
    uint32_t tid[3] = {0, 0, 0}, ctaid[3] = {0, 0, 0};
    uint32_t ntid[3] = {1, 1, 1}, nctaid[3] = {1, 1, 1};
    void* const* args = nullptr;
    bool returned = false;
};

using ExprFn = std::function<Cell(TS&)>;
using StmtFn = std::function<void(TS&)>;

// atomicAdd over the Cell/byte-width convention, returning the OLD value. The
// compiled tier runs a grid's CTAs on parallel OS threads (see
// CompiledKernelImpl::launch), so these dispatch to the shared lock-free RMW
// primitives in <vgre/common/atomic_rmw.h>.
inline int64_t atomicAddInt(void* addr, int bytes, int64_t v) {
    if (bytes == 8) return static_cast<int64_t>(common::atomicAddU64(addr, static_cast<uint64_t>(v)));
    return static_cast<int64_t>(static_cast<int32_t>(
        common::atomicAddU32(addr, static_cast<uint32_t>(v))));
}
inline double atomicAddFloat(void* addr, int bytes, double v) {
    if (bytes == 8) return common::atomicAddF64(addr, v);
    return static_cast<double>(common::atomicAddF32(addr, static_cast<float>(v)));
}

// PTX float→int (cvt.rzi): round toward zero, then SATURATE to the destination
// range, NaN → 0 — not a plain C cast (which is undefined out of range). Mirrors
// the Tier-0 interpreter so the two tiers agree.
int64_t satFloatToInt(double d, const Type& t) {
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

// Narrow a value to the exact scalar type `t` — the compiled tier holds every
// float in a double Cell and every int in an int64 Cell, so without this an
// intermediate `float` op would keep double precision and a 32-bit int op would
// keep 64 bits (both diverging from the PTX interpreter / real hardware, which
// round/wrap at the operand width). Pointers keep the full 64 bits.
Cell coerce(const Cell& v, const Type& t) {
    if (t.isPointer()) return Cell::I(v.asI());
    if (t.isFloating())
        return t.base == Type::Double ? Cell::F(v.asF())
                                      : Cell::F(static_cast<double>(static_cast<float>(v.asF())));
    // float → int saturates (PTX semantics); int → int wraps to the target width.
    const int64_t iv = v.isFloat ? satFloatToInt(v.f, t) : v.i;
    switch (t.base) {
        case Type::Bool:  return Cell::I(iv != 0 ? 1 : 0);
        case Type::Char:  return Cell::I(t.isUnsigned ? (int64_t)(uint8_t)iv  : (int64_t)(int8_t)iv);
        case Type::Short: return Cell::I(t.isUnsigned ? (int64_t)(uint16_t)iv : (int64_t)(int16_t)iv);
        case Type::Int:   return Cell::I(t.isUnsigned ? (int64_t)(uint32_t)iv : (int64_t)(int32_t)iv);
        default:          return Cell::I(iv);   // Long / 64-bit / other: full width
    }
}

// Usual arithmetic conversions for the compiled tier's scalar subset — mirrors
// the codegen/interpreter promote(): pointer wins, else double, else float, else
// long/int with the unsigned-at-result-rank rule.
Type promoteT(const Type& a, const Type& b) {
    if (a.isPointer()) return a;
    if (b.isPointer()) return b;
    Type r;
    if (a.base == Type::Double || b.base == Type::Double) { r.base = Type::Double; return r; }
    if (a.isFloating() || b.isFloating() || a.base == Type::Half || b.base == Type::Half) {
        r.base = Type::Float; return r;
    }
    const bool resultLong = (a.base == Type::Long || b.base == Type::Long);
    r.base = resultLong ? Type::Long : Type::Int;
    auto atRank = [&](const Type& t) { return (t.base == Type::Long) == resultLong; };
    if ((a.isUnsigned && atRank(a)) || (b.isUnsigned && atRank(b))) r.isUnsigned = true;
    return r;
}

class CompiledKernelImpl : public CompiledKernel {
public:
    int numParams() const override { return static_cast<int>(paramTypes_.size()); }

    bool launch(const Extent& grid, const Extent& block, void* const* args, int numArgs) override {
        if (numArgs != numParams()) return false;
        const uint32_t gt = grid.x * grid.y * grid.z;
        const uint32_t bt = block.x * block.y * block.z;

        // One CTA's worth of work: run all its threads sequentially. CTAs on the
        // compiled tier are barrier-free (kernels that need __syncthreads() defer
        // to the Tier-0 interpreter), so distinct CTAs are independent and safe to
        // run concurrently — the only cross-CTA sharing is atomicAdd, which is a
        // real atomic RMW (see atomicAddInt/atomicAddFloat).
        auto runCTA = [&](uint32_t cta) {
            uint32_t cx = cta % grid.x, cy = (cta / grid.x) % grid.y, cz = cta / (grid.x * grid.y);
            for (uint32_t t = 0; t < bt; ++t) {
                TS ts;
                ts.regs.resize(numSlots_);
                ts.args = args;
                ts.tid[0] = t % block.x; ts.tid[1] = (t / block.x) % block.y; ts.tid[2] = t / (block.x * block.y);
                ts.ctaid[0] = cx; ts.ctaid[1] = cy; ts.ctaid[2] = cz;
                ts.ntid[0] = block.x; ts.ntid[1] = block.y; ts.ntid[2] = block.z;
                ts.nctaid[0] = grid.x; ts.nctaid[1] = grid.y; ts.nctaid[2] = grid.z;
                loadParams(ts);
                for (auto& s : body_) { s(ts); if (ts.returned) break; }
            }
        };

        auto& pool = xla::ThreadPool::global();
        if (gt <= 1 || pool.concurrency() <= 1) {
            for (uint32_t cta = 0; cta < gt; ++cta) runCTA(cta);
        } else {
            // Chunk the grid so each worker gets a few CTAs — amortises task
            // overhead while keeping the load balanced across cores.
            int64_t grain = std::max<int64_t>(1, static_cast<int64_t>(gt) /
                                                     (static_cast<int64_t>(pool.concurrency()) * 4));
            pool.parallelFor(static_cast<int64_t>(gt), grain,
                             [&](int64_t cta) { runCTA(static_cast<uint32_t>(cta)); });
        }
        return true;
    }

    // Populated by Compiler (friend-like via public setters kept minimal).
    std::vector<Type> paramTypes_;
    std::vector<StmtFn> body_;
    size_t numSlots_ = 0;

    void loadParams(TS& ts) const {
        for (size_t k = 0; k < paramTypes_.size(); ++k) {
            const void* ap = ts.args[k];
            const Type& t = paramTypes_[k];
            if (t.isPointer()) {
                void* p = nullptr; std::memcpy(&p, ap, sizeof(void*));
                ts.regs[k] = Cell::I(reinterpret_cast<int64_t>(p));
            } else if (t.isFloating()) {
                if (t.elemBytes() == 8) { double d; std::memcpy(&d, ap, 8); ts.regs[k] = Cell::F(d); }
                else { float f; std::memcpy(&f, ap, 4); ts.regs[k] = Cell::F(f); }
            } else {
                if (t.elemBytes() == 8) { int64_t v; std::memcpy(&v, ap, 8); ts.regs[k] = Cell::I(v); }
                else { int32_t v; std::memcpy(&v, ap, 4); ts.regs[k] = Cell::I(static_cast<int64_t>(v)); }
            }
        }
    }
};

// Lowers the AST to closures. Fails (sets err) on unsupported constructs.
struct Compiler {
    std::unordered_map<std::string, size_t> slot;   // var name -> register slot
    std::unordered_map<std::string, Type> vtype;    // var name -> type
    // Per-thread local arrays (`float acc[8];`, `float t[4][4];`): a contiguous
    // run of Cell slots [base, base+size). Indexed by slot, not by memory address
    // — private to the thread, so no barriers/atomics are needed and these stay
    // on the fast tier. `dims` are the per-dimension sizes for row-major flatten.
    struct LocalArr { size_t base; Type elem; int size; std::vector<int> dims; };
    std::unordered_map<std::string, LocalArr> arrays;
    size_t nextSlot = 0;
    bool failed = false;
    std::string err;
    int line = 0, col = 0;

    void fail(const std::string& m) {
        if (failed) return;
        failed = true;
        err = std::to_string(line) + ":" + std::to_string(col) + ": " + m;
    }

    size_t declare(const std::string& name, const Type& t) {
        auto it = slot.find(name);
        if (it != slot.end()) { vtype[name] = t; return it->second; }
        size_t s = nextSlot++;
        slot[name] = s; vtype[name] = t;
        return s;
    }

    // Pre-scan: reject barriers/shared, and allocate a slot for every local.
    void scan(const Stmt& s) {
        if (failed) return;
        if (s.kind == Stmt::VarDecl) {
            if (s.isShared) { fail("__shared__ needs the interpreter tier"); return; }
            if (s.arraySize > 0) {
                // Reserve a contiguous slot run for the per-thread local array.
                size_t base = nextSlot;
                nextSlot += (size_t)s.arraySize;
                std::vector<int> dims = s.arrayDims.empty()
                                            ? std::vector<int>{s.arraySize} : s.arrayDims;
                arrays[s.name] = LocalArr{base, s.type, s.arraySize, std::move(dims)};
                return;
            }
            declare(s.name, s.type);
        }
        scanExprBarriers(s.expr.get());
        for (auto& c : s.body) scan(*c);
        for (auto& c : s.elseBody) scan(*c);
        if (s.forInit) scan(*s.forInit);
        scanExprBarriers(s.forCond.get());
        scanExprBarriers(s.forIncr.get());
    }
    void scanExprBarriers(const Expr* e) {
        if (!e || failed) return;
        if (e->kind == Expr::Call &&
            (e->str == "__syncthreads" || e->str == "__shfl_sync" || e->str == "__shfl_up_sync" ||
             e->str == "__shfl_down_sync" || e->str == "__shfl_xor_sync"))
            fail("warp-cooperative op needs the interpreter tier");   // __syncthreads / __shfl_*
        for (auto& a : e->args) scanExprBarriers(a.get());
    }

    static const char* sreg(const std::string& obj) { return obj.c_str(); }

    ExprFn compileExpr(const Expr& e) {
        line = e.line; col = e.col;
        switch (e.kind) {
            case Expr::IntLit: { int64_t v = e.ival; return [v](TS&) { return Cell::I(v); }; }
            // A float literal (`1.5f`, no `wide`) holds its float-rounded value, not
            // the full double text — matching the constant the codegen emits.
            case Expr::FloatLit: {
                double v = e.wide ? e.fval : static_cast<double>(static_cast<float>(e.fval));
                return [v](TS&) { return Cell::F(v); };
            }
            case Expr::Ident: {
                auto it = slot.find(e.str);
                if (it == slot.end()) { fail("undeclared identifier '" + e.str + "'"); return {}; }
                size_t s = it->second;
                return [s](TS& ts) { return ts.regs[s]; };
            }
            case Expr::Member: return compileMember(e);
            case Expr::Index:  return compileLoad(e);
            // Narrow compound-op results to their static type so intermediate
            // float ops round to float and 32-bit int ops wrap at 32 bits — the
            // Cell otherwise carries double/int64 and would diverge from the
            // interpreter. (Cast/Assign already narrow via coerce.)
            case Expr::Unary:  return narrowResult(compileUnary(e), e);
            case Expr::Binary: return narrowResult(compileBinary(e), e);
            case Expr::Assign: return compileAssign(e);
            case Expr::Call:   return narrowResult(compileCall(e), e);
            case Expr::Cast:   return compileCast(e);
            case Expr::Ternary: return narrowResult(compileTernary(e), e);
        }
        fail("unsupported expression");
        return {};
    }

    ExprFn compileCast(const Expr& e) {
        ExprFn a = compileExpr(*e.args[0]);
        if (failed) return {};
        Type t = e.castType;
        return [a, t](TS& ts) { return coerce(a(ts), t); };
    }

    ExprFn compileTernary(const Expr& e) {
        ExprFn c = compileExpr(*e.args[0]);
        ExprFn t = compileExpr(*e.args[1]);
        ExprFn f = compileExpr(*e.args[2]);
        if (failed) return {};
        return [c, t, f](TS& ts) -> Cell { return c(ts).asI() != 0 ? t(ts) : f(ts); };
    }

    ExprFn compileMember(const Expr& e) {
        const Expr& obj = *e.args[0];
        int comp = e.str == "x" ? 0 : e.str == "y" ? 1 : e.str == "z" ? 2 : -1;
        if (obj.kind == Expr::Ident && comp >= 0) {
            const std::string& o = obj.str;
            if (o == "threadIdx") return [comp](TS& ts) { return Cell::I(ts.tid[comp]); };
            if (o == "blockIdx")  return [comp](TS& ts) { return Cell::I(ts.ctaid[comp]); };
            if (o == "blockDim")  return [comp](TS& ts) { return Cell::I(ts.ntid[comp]); };
            if (o == "gridDim")   return [comp](TS& ts) { return Cell::I(ts.nctaid[comp]); };
        }
        fail("unsupported member access");
        return {};
    }

    // Peel nested Index nodes (`A[i][j]` → Index(Index(A,i),j)). Returns the
    // innermost base expr; fills `idxs` outer-dimension-first ([i, j]).
    static const Expr* peelIndex(const Expr& index, std::vector<const Expr*>& idxs) {
        std::vector<const Expr*> rev;
        const Expr* cur = &index;
        while (cur->kind == Expr::Index) { rev.push_back(cur->args[1].get()); cur = cur->args[0].get(); }
        for (auto it = rev.rbegin(); it != rev.rend(); ++it) idxs.push_back(*it);
        return cur;
    }

    // If `index` (possibly nested A[i][j]) targets a local array, returns it and
    // compiles the row-major flattened element index into `flatOut`; else null.
    const LocalArr* localArrayAccess(const Expr& index, ExprFn& flatOut) {
        std::vector<const Expr*> idxs;
        const Expr* root = peelIndex(index, idxs);
        if (root->kind != Expr::Ident) return nullptr;
        auto it = arrays.find(root->str);
        if (it == arrays.end()) return nullptr;
        const LocalArr& la = it->second;
        if (idxs.size() != la.dims.size()) {
            fail("array '" + root->str + "' expects " + std::to_string(la.dims.size()) +
                 " index(es), got " + std::to_string(idxs.size()));
            return nullptr;
        }
        ExprFn flat = compileExpr(*idxs[0]);
        if (failed) return nullptr;
        for (size_t d = 1; d < idxs.size(); ++d) {
            int dim = la.dims[d];
            ExprFn prev = flat, ik = compileExpr(*idxs[d]);
            if (failed) return nullptr;
            flat = [prev, dim, ik](TS& ts) { return Cell::I(prev(ts).asI() * dim + ik(ts).asI()); };
        }
        flatOut = std::move(flat);
        return &it->second;
    }

    // The pointee (element) type of an index base — for load/store width.
    Type pointee(const Expr& index) {
        std::vector<const Expr*> idxs;
        const Expr* root = peelIndex(index, idxs);
        if (root->kind == Expr::Ident) {
            auto la = arrays.find(root->str);
            if (la != arrays.end()) return la->second.elem;
            auto it = vtype.find(root->str);
            if (it != vtype.end() && it->second.isPointer()) { Type t = it->second; t.ptr -= 1; return t; }
        }
        Type t; t.base = Type::Int; return t;
    }

    static Type scalar(Type::Base b) { Type t; t.base = b; return t; }

    // Static type of an expression — enough to know the width/precision each
    // result must be narrowed to. Mirrors the codegen's estimateType over the
    // subset the compiled tier accepts (no structs/half beyond pass-through).
    Type estimateType(const Expr& e) {
        switch (e.kind) {
            case Expr::FloatLit: return scalar(e.wide ? Type::Double : Type::Float);
            case Expr::IntLit:
                return scalar((e.wide || e.ival > 2147483647LL || e.ival < -2147483648LL) ? Type::Long : Type::Int);
            case Expr::Ident: {
                auto it = vtype.find(e.str);
                if (it != vtype.end()) return it->second;
                auto ar = arrays.find(e.str);
                if (ar != arrays.end()) { Type t = ar->second.elem; t.ptr += 1; return t; }
                return scalar(Type::Int);
            }
            case Expr::Member:  return scalar(Type::Int);   // threadIdx/blockIdx/… builtins
            case Expr::Index:   return pointee(e);
            case Expr::Cast:    return e.castType;
            case Expr::Unary:
                if (e.str == "!") return scalar(Type::Int);
                if (e.str == "*") { Type t = estimateType(*e.args[0]); if (t.ptr > 0) t.ptr--; return t; }
                if (e.str == "&") { Type t = estimateType(*e.args[0]); t.ptr++; return t; }
                return estimateType(*e.args[0]);
            case Expr::Binary: {
                const std::string& o = e.str;
                if (o == "<" || o == "<=" || o == ">" || o == ">=" || o == "==" || o == "!=" ||
                    o == "&&" || o == "||") return scalar(Type::Int);
                return promoteT(estimateType(*e.args[0]), estimateType(*e.args[1]));
            }
            case Expr::Assign:  return estimateType(*e.args[0]);
            case Expr::Ternary: return promoteT(estimateType(*e.args[1]), estimateType(*e.args[2]));
            case Expr::Call: {
                const std::string& fn = e.str;
                if (fn == "min" || fn == "max")
                    return e.args.size() < 2 ? scalar(Type::Int) : promoteT(estimateType(*e.args[0]), estimateType(*e.args[1]));
                if (fn == "abs")       return e.args.empty() ? scalar(Type::Int) : estimateType(*e.args[0]);
                if (fn == "atomicAdd") return e.args.size() < 2 ? scalar(Type::Int) : estimateType(*e.args[1]);
                // Math intrinsic: the `f`-suffixed spelling returns float, else double.
                return scalar(!fn.empty() && fn.back() == 'f' ? Type::Float : Type::Double);
            }
        }
        return scalar(Type::Int);
    }

    // Wrap a compound-op closure so its result is narrowed to the op's static
    // type. Only floats and sub-64-bit ints need it (double/long/pointer are
    // already the Cell's full width), so we skip the rest to avoid overhead.
    ExprFn narrowResult(ExprFn fn, const Expr& e) {
        if (failed || !fn) return fn;
        Type t = estimateType(e);
        const bool needs = t.ptr == 0 &&
            (t.base == Type::Float || t.base == Type::Bool || t.base == Type::Char ||
             t.base == Type::Short || t.base == Type::Int);
        if (!needs) return fn;
        return [fn, t](TS& ts) { return coerce(fn(ts), t); };
    }

    ExprFn compileLoad(const Expr& e) {
        ExprFn flat;
        if (const LocalArr* la = localArrayAccess(e, flat)) {  // acc[i] / t[i][j] — slot access
            if (failed) return {};
            size_t base = la->base; int sz = la->size;
            return [base, flat, sz](TS& ts) -> Cell {
                int64_t i = flat(ts).asI();
                if (i < 0 || i >= sz) return Cell::I(0);     // OOB → 0 (no fault)
                return ts.regs[base + (size_t)i];
            };
        }
        ExprFn base = compileExpr(*e.args[0]);
        ExprFn idx  = compileExpr(*e.args[1]);
        if (failed) return {};
        Type pt = pointee(e);
        int bytes = pt.elemBytes();
        bool fp = pt.isFloating();
        return [base, idx, bytes, fp](TS& ts) -> Cell {
            int64_t addr = base(ts).asI() + idx(ts).asI() * bytes;
            if (fp) {
                if (bytes == 8) { double d; std::memcpy(&d, reinterpret_cast<void*>(addr), 8); return Cell::F(d); }
                float f; std::memcpy(&f, reinterpret_cast<void*>(addr), 4); return Cell::F(f);
            }
            if (bytes == 8) { int64_t v; std::memcpy(&v, reinterpret_cast<void*>(addr), 8); return Cell::I(v); }
            int32_t v; std::memcpy(&v, reinterpret_cast<void*>(addr), 4); return Cell::I(static_cast<int64_t>(v));
        };
    }

    // Store `value` to index-expr `lhs`.
    StmtFn compileStore(const Expr& lhs, ExprFn value) {
        ExprFn flat;
        if (const LocalArr* la = localArrayAccess(lhs, flat)) {  // acc[i] / t[i][j] = v — slot access
            if (failed) return {};
            size_t base = la->base; int sz = la->size; Type pt = la->elem;
            return [base, flat, value, sz, pt](TS& ts) {
                int64_t i = flat(ts).asI();
                if (i < 0 || i >= sz) return;                // OOB → drop (no fault)
                ts.regs[base + (size_t)i] = coerce(value(ts), pt);
            };
        }
        ExprFn base = compileExpr(*lhs.args[0]);
        ExprFn idx  = compileExpr(*lhs.args[1]);
        if (failed) return {};
        Type pt = pointee(lhs);
        int bytes = pt.elemBytes();
        bool fp = pt.isFloating();
        return [base, idx, value, bytes, fp](TS& ts) {
            int64_t addr = base(ts).asI() + idx(ts).asI() * bytes;
            Cell v = value(ts);
            if (fp) {
                if (bytes == 8) { double d = v.asF(); std::memcpy(reinterpret_cast<void*>(addr), &d, 8); }
                else { float f = static_cast<float>(v.asF()); std::memcpy(reinterpret_cast<void*>(addr), &f, 4); }
            } else {
                if (bytes == 8) { int64_t x = v.asI(); std::memcpy(reinterpret_cast<void*>(addr), &x, 8); }
                else { int32_t x = static_cast<int32_t>(v.asI()); std::memcpy(reinterpret_cast<void*>(addr), &x, 4); }
            }
        };
    }

    ExprFn compileUnary(const Expr& e) {
        if (e.str == "pre++" || e.str == "pre--" || e.str == "post++" || e.str == "post--")
            return compileIncDec(e);
        ExprFn a = compileExpr(*e.args[0]);
        if (failed) return {};
        if (e.str == "+") return a;
        std::string op = e.str;
        return [a, op](TS& ts) -> Cell {
            Cell v = a(ts);
            if (op == "-") return v.isFloat ? Cell::F(-v.f) : Cell::I(-v.i);
            if (op == "!") return Cell::I(v.asI() == 0 ? 1 : 0);
            if (op == "~") return Cell::I(~v.asI());
            return v;
        };
    }

    ExprFn compileIncDec(const Expr& e) {
        const Expr& operand = *e.args[0];
        if (operand.kind != Expr::Ident) { fail("'++'/'--' requires a variable"); return {}; }
        auto it = slot.find(operand.str);
        if (it == slot.end()) { fail("'++'/'--' of undeclared variable"); return {}; }
        size_t s = it->second;
        bool inc = e.str.find("++") != std::string::npos;
        bool pre = e.str.compare(0, 3, "pre") == 0;
        return [s, inc, pre](TS& ts) -> Cell {
            Cell before = ts.regs[s];
            Cell& r = ts.regs[s];
            if (r.isFloat) r.f += inc ? 1.0 : -1.0; else r.i += inc ? 1 : -1;
            return pre ? r : before;
        };
    }

    ExprFn compileBinary(const Expr& e) {
        const std::string& op = e.str;
        ExprFn a = compileExpr(*e.args[0]);
        ExprFn b = compileExpr(*e.args[1]);
        if (failed) return {};
        // Logical &&/|| test each side's truthiness — no arithmetic coercion.
        if (op == "&&" || op == "||") {
            const bool isAnd = (op == "&&");
            return [a, b, isAnd](TS& ts) -> Cell {
                return Cell::I((isAnd ? (a(ts).asI() != 0 && b(ts).asI() != 0)
                                      : (a(ts).asI() != 0 || b(ts).asI() != 0)) ? 1 : 0);
            };
        }
        // Compute at the operands' common type (comparisons compare there but
        // yield int, handled by binop); narrowResult wraps the result to its own
        // static type.
        const Type ct = promoteT(estimateType(*e.args[0]), estimateType(*e.args[1]));
        return makeBinary(a, b, op, ct);
    }

    // Assignment as an expression (also used by ExprStmt); returns the stored value.
    ExprFn compileAssign(const Expr& e) {
        const Expr& lhs = *e.args[0];
        const std::string& op = e.str;
        ExprFn rhs = compileExpr(*e.args[1]);
        if (failed) return {};

        if (lhs.kind == Expr::Ident) {
            auto it = slot.find(lhs.str);
            if (it == slot.end()) { fail("assignment to undeclared '" + lhs.str + "'"); return {}; }
            size_t s = it->second;
            Type vt = vtype[lhs.str];
            // `x op= rhs` is `x = (x op rhs)` computed at the common type of x and
            // rhs (usual arithmetic conversions), then narrowed back to x's type.
            ExprFn value = rhs;
            if (op != "=") {
                std::string bop = op.substr(0, op.size() - 1);   // "+=" -> "+", "<<=" -> "<<"
                Type ct = promoteT(vt, estimateType(*e.args[1]));
                ExprFn cur = [s](TS& ts) { return ts.regs[s]; };
                value = makeBinary(cur, rhs, bop, ct);
            }
            return [s, value, vt](TS& ts) -> Cell { ts.regs[s] = coerce(value(ts), vt); return ts.regs[s]; };
        }
        if (lhs.kind == Expr::Index) {
            Type pt = pointee(lhs);
            ExprFn value;
            if (op == "=") value = rhs;
            else {
                std::string bop = op.substr(0, op.size() - 1);
                Type ct = promoteT(pt, estimateType(*e.args[1]));
                value = makeBinary(compileLoad(lhs), rhs, bop, ct);
            }
            StmtFn st = compileStore(lhs, [value, pt](TS& ts) { return coerce(value(ts), pt); });
            if (failed) return {};
            // Return the stored value as an expression result.
            ExprFn stored = value;
            return [st, stored, pt](TS& ts) -> Cell { Cell v = coerce(stored(ts), pt); st(ts); return v; };
        }
        fail("invalid assignment target");
        return {};
    }

    // Build a closure applying `op` to operands `a`,`b` at their common type `ct`
    // — the usual-arithmetic-conversion rule the interpreter follows: coerce both
    // operands to `ct` first (so an int operand of a float op rounds to float),
    // and compute float32 arithmetic in float (single rounding). Shared by
    // compileBinary and compound assignment so both convert identically.
    static ExprFn makeBinary(ExprFn a, ExprFn b, const std::string& op, const Type& ct) {
        const bool arith = (op == "+" || op == "-" || op == "*" || op == "/");
        if (arith && !ct.isPointer() && ct.base == Type::Float) {
            return [a, b, op](TS& ts) -> Cell {
                float x = static_cast<float>(a(ts).asF()), y = static_cast<float>(b(ts).asF()), r;
                if (op == "+") r = x + y; else if (op == "-") r = x - y;
                else if (op == "*") r = x * y; else r = x / y;
                return Cell::F(static_cast<double>(r));
            };
        }
        if (ct.isPointer()) return [a, b, op](TS& ts) -> Cell { return binop(op, a(ts), b(ts)); };
        return [a, b, op, ct](TS& ts) -> Cell { return binop(op, coerce(a(ts), ct), coerce(b(ts), ct)); };
    }

    // Apply `op` to two already-coerced operands (callers narrow to the common
    // type first). Floating ops run in double here; the caller's float32 path
    // computes those in float, and narrowResult wraps every result to its width.
    static Cell binop(const std::string& op, const Cell& x, const Cell& y) {
        const bool fp = x.isFloat || y.isFloat;
        if (op == "+") return fp ? Cell::F(x.asF() + y.asF()) : Cell::I(x.asI() + y.asI());
        if (op == "-") return fp ? Cell::F(x.asF() - y.asF()) : Cell::I(x.asI() - y.asI());
        if (op == "*") return fp ? Cell::F(x.asF() * y.asF()) : Cell::I(x.asI() * y.asI());
        if (op == "/") return fp ? Cell::F(x.asF() / y.asF()) : Cell::I(y.asI() ? x.asI() / y.asI() : 0);
        if (op == "%") return Cell::I(y.asI() ? x.asI() % y.asI() : 0);
        if (op == "<")  return Cell::I((fp ? x.asF() <  y.asF() : x.asI() <  y.asI()) ? 1 : 0);
        if (op == "<=") return Cell::I((fp ? x.asF() <= y.asF() : x.asI() <= y.asI()) ? 1 : 0);
        if (op == ">")  return Cell::I((fp ? x.asF() >  y.asF() : x.asI() >  y.asI()) ? 1 : 0);
        if (op == ">=") return Cell::I((fp ? x.asF() >= y.asF() : x.asI() >= y.asI()) ? 1 : 0);
        if (op == "==") return Cell::I((fp ? x.asF() == y.asF() : x.asI() == y.asI()) ? 1 : 0);
        if (op == "!=") return Cell::I((fp ? x.asF() != y.asF() : x.asI() != y.asI()) ? 1 : 0);
        if (op == "&&") return Cell::I((x.asI() != 0 && y.asI() != 0) ? 1 : 0);
        if (op == "||") return Cell::I((x.asI() != 0 || y.asI() != 0) ? 1 : 0);
        if (op == "&")  return Cell::I(x.asI() & y.asI());
        if (op == "|")  return Cell::I(x.asI() | y.asI());
        if (op == "^")  return Cell::I(x.asI() ^ y.asI());
        if (op == "<<") return Cell::I(x.asI() << y.asI());
        if (op == ">>") return Cell::I(x.asI() >> y.asI());
        return Cell::I(0);
    }

    // atomicAdd(&arr[i], val): read-modify-write returning the old value.
    ExprFn compileAtomicAdd(const Expr& e) {
        const Expr& a0 = *e.args[0];
        if (a0.kind != Expr::Unary || a0.str != "&" || a0.args.empty() || a0.args[0]->kind != Expr::Index) {
            fail("atomicAdd expects &array[index] as its first argument");
            return {};
        }
        const Expr& index = *a0.args[0];
        ExprFn flat;
        if (const LocalArr* la = localArrayAccess(index, flat)) {  // atomicAdd(&acc[i], v) — slot RMW
            ExprFn val = compileExpr(*e.args[1]);
            if (failed) return {};
            size_t base = la->base; int sz = la->size; Type pt = la->elem;
            return [base, flat, val, sz, pt](TS& ts) -> Cell {
                int64_t i = flat(ts).asI();
                if (i < 0 || i >= sz) return Cell::I(0);
                Cell old = ts.regs[base + (size_t)i];        // private to the thread → plain RMW
                ts.regs[base + (size_t)i] = coerce(binop("+", old, val(ts)), pt);
                return old;
            };
        }
        ExprFn base = compileExpr(*index.args[0]);
        ExprFn idx  = compileExpr(*index.args[1]);
        ExprFn val  = compileExpr(*e.args[1]);
        if (failed) return {};
        Type pt = pointee(index);
        int bytes = pt.elemBytes();
        bool fp = pt.isFloating();
        return [base, idx, val, bytes, fp](TS& ts) -> Cell {
            void* addr = reinterpret_cast<void*>(base(ts).asI() + idx(ts).asI() * bytes);
            Cell v = val(ts);
            if (fp) return Cell::F(atomicAddFloat(addr, bytes, v.asF()));
            return Cell::I(atomicAddInt(addr, bytes, v.asI()));
        };
    }

    ExprFn compileCall(const Expr& e) {
        const std::string& fn = e.str;
        if (fn == "atomicAdd" && e.args.size() == 2) return compileAtomicAdd(e);
        if (e.args.size() == 1) {
            ExprFn a = compileExpr(*e.args[0]); if (failed) return {};
            // Intrinsics compute in double; narrowResult rounds the f32 spellings
            // (sqrtf/fabsf/…) back to float via estimateType, so f32 and f64 differ
            // only in the final rounding, as they should.
            if (fn == "abs")   return [a](TS& ts) { Cell v = a(ts); return v.isFloat ? Cell::F(std::fabs(v.f)) : Cell::I(std::llabs((long long)v.i)); };
            if (fn == "sqrtf"  || fn == "sqrt")  return [a](TS& ts) { return Cell::F(std::sqrt(a(ts).asF())); };
            if (fn == "fabsf"  || fn == "fabs")  return [a](TS& ts) { return Cell::F(std::fabs(a(ts).asF())); };
            if (fn == "rsqrtf" || fn == "rsqrt") return [a](TS& ts) { return Cell::F(1.0 / std::sqrt(a(ts).asF())); };
            if (fn == "sinf"   || fn == "sin")   return [a](TS& ts) { return Cell::F(std::sin(a(ts).asF())); };
            if (fn == "cosf"   || fn == "cos")   return [a](TS& ts) { return Cell::F(std::cos(a(ts).asF())); };
            if (fn == "floorf" || fn == "floor") return [a](TS& ts) { return Cell::F(std::floor(a(ts).asF())); };
            if (fn == "ceilf"  || fn == "ceil")  return [a](TS& ts) { return Cell::F(std::ceil(a(ts).asF())); };
            if (fn == "__expf" || fn == "expf" || fn == "exp") return [a](TS& ts) { return Cell::F(std::exp(a(ts).asF())); };
            if (fn == "__logf" || fn == "logf" || fn == "log") return [a](TS& ts) { return Cell::F(std::log(a(ts).asF())); };
        } else if (e.args.size() == 2) {
            ExprFn a = compileExpr(*e.args[0]); ExprFn b = compileExpr(*e.args[1]); if (failed) return {};
            if (fn == "fminf" || fn == "fmin") return [a, b](TS& ts) { return Cell::F(std::fmin(a(ts).asF(), b(ts).asF())); };
            if (fn == "fmaxf" || fn == "fmax") return [a, b](TS& ts) { return Cell::F(std::fmax(a(ts).asF(), b(ts).asF())); };
            if (fn == "powf"  || fn == "pow")  return [a, b](TS& ts) { return Cell::F(std::pow(a(ts).asF(), b(ts).asF())); };
            if (fn == "min") return [a, b](TS& ts) { Cell x = a(ts), y = b(ts); bool fp = x.isFloat || y.isFloat;
                return fp ? Cell::F(std::fmin(x.asF(), y.asF())) : Cell::I(std::min(x.asI(), y.asI())); };
            if (fn == "max") return [a, b](TS& ts) { Cell x = a(ts), y = b(ts); bool fp = x.isFloat || y.isFloat;
                return fp ? Cell::F(std::fmax(x.asF(), y.asF())) : Cell::I(std::max(x.asI(), y.asI())); };
        } else if (e.args.size() == 3 && (fn == "fmaf" || fn == "fma")) {
            ExprFn a = compileExpr(*e.args[0]); ExprFn b = compileExpr(*e.args[1]); ExprFn c = compileExpr(*e.args[2]);
            if (failed) return {};
            return [a, b, c](TS& ts) { return Cell::F(std::fma(a(ts).asF(), b(ts).asF(), c(ts).asF())); };
        }
        fail("unsupported call to '" + fn + "'");
        return {};
    }

    StmtFn compileStmt(const Stmt& s) {
        line = s.line; col = s.col;
        switch (s.kind) {
            case Stmt::VarDecl: {
                if (s.arraySize > 0) {
                    // Local array: slots reserved in scan(), zeroed by regs.resize().
                    return [](TS&) {};
                }
                size_t sl = slot[s.name];
                Type vt = s.type;
                if (s.expr) {
                    ExprFn init = compileExpr(*s.expr);
                    if (failed) return {};
                    return [sl, init, vt](TS& ts) { ts.regs[sl] = coerce(init(ts), vt); };
                }
                return [sl](TS& ts) { ts.regs[sl] = Cell::I(0); };
            }
            case Stmt::ExprStmt: {
                if (!s.expr) return [](TS&) {};
                ExprFn ex = compileExpr(*s.expr);
                if (failed) return {};
                return [ex](TS& ts) { ex(ts); };
            }
            case Stmt::Block: return compileBody(s.body);
            case Stmt::Return: return [](TS& ts) { ts.returned = true; };
            case Stmt::Empty: return [](TS&) {};
            case Stmt::If: {
                ExprFn cond = compileExpr(*s.expr);
                StmtFn then_ = compileBody(s.body);
                StmtFn else_ = s.elseBody.empty() ? StmtFn() : compileBody(s.elseBody);
                if (failed) return {};
                return [cond, then_, else_](TS& ts) {
                    if (cond(ts).asI() != 0) then_(ts);
                    else if (else_) else_(ts);
                };
            }
            case Stmt::While: {
                ExprFn cond = compileExpr(*s.expr);
                StmtFn body = compileBody(s.body);
                if (failed) return {};
                return [cond, body](TS& ts) {
                    while (!ts.returned && cond(ts).asI() != 0) body(ts);
                };
            }
            case Stmt::For: {
                StmtFn init = s.forInit ? compileStmt(*s.forInit) : StmtFn();
                ExprFn cond = s.forCond ? compileExpr(*s.forCond) : ExprFn();
                ExprFn incr = s.forIncr ? compileExpr(*s.forIncr) : ExprFn();
                StmtFn body = compileBody(s.body);
                if (failed) return {};
                return [init, cond, incr, body](TS& ts) {
                    if (init) init(ts);
                    while (!ts.returned && (!cond || cond(ts).asI() != 0)) {
                        body(ts);
                        if (incr) incr(ts);
                    }
                };
            }
        }
        fail("unsupported statement");
        return {};
    }

    StmtFn compileBody(const std::vector<StmtPtr>& stmts) {
        std::vector<StmtFn> fns;
        for (auto& s : stmts) { StmtFn f = compileStmt(*s); if (failed) return {}; fns.push_back(std::move(f)); }
        return [fns](TS& ts) { for (auto& f : fns) { f(ts); if (ts.returned) return; } };
    }
};

}  // namespace

std::unique_ptr<CompiledKernel> CompiledKernel::compile(const Kernel& k, std::string& err) {
    Compiler c;
    // Params occupy the first slots (slot index == param index).
    for (const Param& p : k.params) c.declare(p.name.empty() ? ("__arg" + std::to_string(c.nextSlot)) : p.name, p.type);
    for (auto& s : k.body) c.scan(*s);
    if (c.failed) { err = c.err; return nullptr; }

    auto impl = std::unique_ptr<CompiledKernelImpl>(new CompiledKernelImpl());
    for (const Param& p : k.params) impl->paramTypes_.push_back(p.type);
    for (auto& s : k.body) {
        StmtFn f = c.compileStmt(*s);
        if (c.failed) { err = c.err; return nullptr; }
        impl->body_.push_back(std::move(f));
    }
    impl->numSlots_ = c.nextSlot;
    return impl;
}

std::unique_ptr<CompiledKernel> CompiledKernel::compileSource(const std::string& source,
                                                             const std::string& name,
                                                             std::string& err) {
    ParseResult pr = parse(source);
    if (!pr.ok) { err = pr.error; return nullptr; }
    const Kernel* target = nullptr;
    for (auto& kp : pr.module->kernels)
        if (name.empty() || kp->name == name) { target = kp.get(); break; }
    if (!target) { err = "kernel not found"; return nullptr; }
    return compile(*target, err);
}

}  // namespace frontend
}  // namespace compiler
}  // namespace vgre
