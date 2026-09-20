// PTX code generation — see include/vgre/compiler/frontend/codegen.h.
//
// A single non-SSA pass: each local variable lives in one virtual register that
// is mutated in place (the interpreter's registers are mutable), so control flow
// needs no phi nodes. Types are inferred bottom-up as we emit. Only the
// supported subset is accepted; anything else fails with a located error.

#include "vgre/compiler/frontend/codegen.h"

#include "vgre/compiler/frontend/parser.h"
#include "vgre/compiler/frontend/ptx_verifier.h"

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
enum class Space { Value, Global, Shared, Local, ParamStruct, LocalStruct };

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
enum class RC { R32, F32, RD64, F64, Pred };

// A 64-bit scalar: `long`/`unsigned long` (int) or `double` (float). Pointers are
// 64-bit too but handled via RD64 separately.
inline bool is64BitScalar(const Type& t) {
    return t.ptr == 0 && (t.base == Type::Long || t.base == Type::Double);
}

RC classOf(const Type& t) {
    if (t.isPointer()) return RC::RD64;
    if (t.base == Type::Double) return RC::F64;
    if (t.isFloating()) return RC::F32;              // float
    if (t.base == Type::Long) return RC::RD64;       // 64-bit int
    return RC::R32;  // bool/char/short/int collapse to 32-bit
}

struct Codegen {
    const Kernel& k;
    std::string body;                 // instruction stream (built first)
    std::string sharedDecls;          // .shared declarations (emitted in the decl section)
    std::string localDecls;           // .local declarations (emitted in the decl section)
    int nR = 0, nF = 0, nRd = 0, nFd = 0, nP = 0, nLbl = 0;
    std::unordered_map<std::string, Val> vars;  // name -> value (single mutable reg)
    std::unordered_map<std::string, std::vector<int>> arrayDims_;  // declared array -> dim sizes
    // Local struct variables: each scalar member lives in its own register, keyed
    // by struct-var name → member name. (Struct params stay in .param; see ParamStruct.)
    std::unordered_map<std::string, std::unordered_map<std::string, Val>> localStructMembers_;
    // Enclosing loops for break/continue: (continueTarget, breakTarget) labels.
    // `continue` branches to the first, `break` to the second, of the innermost.
    std::vector<std::pair<std::string, std::string>> loopCtx_;
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

    CodegenOptions opts_;             // target/version for the PTX header

    explicit Codegen(const Kernel& kernel,
                     const std::unordered_map<std::string, const Kernel*>* deviceFns = nullptr,
                     const Module* mod = nullptr,
                     const CodegenOptions& opts = {})
        : k(kernel), deviceFns_(deviceFns), mod_(mod), opts_(opts) {}

    // Struct byte-size from the module's struct table (0 if unknown).
    int structSize(const std::string& name) const {
        const StructDef* d = mod_ ? mod_->findStruct(name) : nullptr;
        return d ? d->size : 0;
    }

    // Element size of a pointee type for addressing/pointer-arithmetic strides.
    // A struct pointee's size comes from the struct table (Type::elemBytes() can't
    // know it and returns 0), so struct arrays and `struct*` arithmetic stride
    // correctly.
    int pointeeBytes(const Type& pointee) const {
        if (pointee.base == Type::Struct && pointee.ptr == 0) {
            int s = structSize(pointee.structName);
            return s > 0 ? s : 1;
        }
        int e = pointee.elemBytes();
        return e > 0 ? e : 1;
    }

    static const char* movFor(const Type& t) {
        if (t.base == Type::Double) return "mov.f64 ";
        if (t.isFloating()) return "mov.f32 ";
        if (t.isPointer() || t.base == Type::Long) return "mov.u64 ";
        return "mov.u32 ";
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
            case RC::F64:  return "%fd" + std::to_string(nFd++);
            case RC::Pred: return "%p"  + std::to_string(nP++);
        }
        return "%r0";
    }
    std::string label() { return "$L" + std::to_string(nLbl++); }

    // A globally-unique PTX symbol for a named local/shared array. Two inlinings of
    // the same __device__ helper (or a caller and helper that share an array name)
    // must not emit the same .local/.shared symbol, or the declarations collide.
    int symSeq_ = 0;
    std::string uniqueSym(const std::string& base) { return base + "$" + std::to_string(symSeq_++); }

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
    static Type longType() { Type t; t.base = Type::Long; return t; }
    static Type doubleType() { Type t; t.base = Type::Double; return t; }
    static Type halfType() { Type t; t.base = Type::Half; return t; }

    // 64-bit int/double and 32-bit int/float are all supported now; nothing to
    // reject here (kept as a hook for genuinely unsupported types).
    bool ensureSupported(const Type&) { return true; }

    // PTX ld/st/param type suffix for a scalar (pointee) type — width-aware.
    // NOTE: this collapses all sub-32-bit ints to u32; use ldSuffix/stSuffix for
    // real memory accesses so char/short honor their true 1/2-byte width.
    static std::string memSuffix(const Type& t) {
        if (t.base == Type::Double) return "f64";
        if (t.base == Type::Half) return "b16";   // __half: 16-bit raw storage
        if (t.isFloating()) return "f32";
        if (t.base == Type::Long) return "u64";
        return "u32";
    }

    // Load suffix: a narrow integer loads at its real width and extends into the
    // 32-bit register — signed char/short SIGN-extend (s8/s16), unsigned and bool
    // ZERO-extend (u8/u16). 32/64-bit and float fall back to memSuffix. This fixes
    // char*/short* accesses, which otherwise read 4 bytes at a 1/2-byte stride.
    static std::string ldSuffix(const Type& t) {
        if (t.isPointer()) return "u64";
        if (t.base == Type::Bool) return "u8";
        if (t.base == Type::Char) return t.isUnsigned ? "u8" : "s8";
        if (t.base == Type::Short) return t.isUnsigned ? "u16" : "s16";
        return memSuffix(t);
    }
    // Store suffix: a narrow integer stores only its low byte(s); signedness is
    // irrelevant on a store, so bool/char use u8 and short uses u16.
    static std::string stSuffix(const Type& t) {
        if (t.isPointer()) return "u64";
        if (t.base == Type::Bool || t.base == Type::Char) return "u8";
        if (t.base == Type::Short) return "u16";
        return memSuffix(t);
    }

    // Signed-int PTX suffix for cvt / arithmetic (s32 or s64).
    static std::string intSuffix(const Type& t) { return is64BitScalar(t) || t.base == Type::Long ? "s64" : "s32"; }
    // Signedness-aware int suffix: u32/u64 for unsigned types, else s32/s64. Used
    // for the ops whose result differs by signedness — ordered compares, division,
    // remainder, right shift (logical vs arithmetic), integer min/max.
    static std::string uIntSuffix(const Type& t) {
        const bool w64 = is64BitScalar(t) || t.base == Type::Long;
        return std::string(t.isUnsigned ? "u" : "s") + (w64 ? "64" : "32");
    }
    // Float PTX suffix (f32 or f64).
    static std::string floatSuffix(const Type& t) { return t.base == Type::Double ? "f64" : "f32"; }
    // Arithmetic op suffix: f64/f32 for floats, s64/s32 for ints.
    static std::string arithSuffix(const Type& t) { return t.isFloating() ? floatSuffix(t) : intSuffix(t); }
    // Bitwise op suffix: b64 for 64-bit ints, else b32.
    static std::string bitSuffix(const Type& t) { return t.base == Type::Long ? "b64" : "b32"; }

    // Format a float32 immediate as PTX hex (0f%08X) — the interpreter's format.
    static std::string f32imm(double d) {
        float f = static_cast<float>(d);
        uint32_t bits;
        std::memcpy(&bits, &f, 4);
        char buf[16];
        std::snprintf(buf, sizeof(buf), "0f%08X", bits);
        return buf;
    }
    // Format a float64 immediate as PTX hex (0d%016llX).
    static std::string f64imm(double d) {
        uint64_t bits;
        std::memcpy(&bits, &d, 8);
        char buf[24];
        std::snprintf(buf, sizeof(buf), "0d%016llX", static_cast<unsigned long long>(bits));
        return buf;
    }

    // Coerce `v` to `want`, emitting the right cvt for any int/float/width change
    // (int32/int64/f32/f64). Pointers pass through unchanged.
    Val coerce(const Val& v, const Type& want) {
        if (want.isPointer()) {
            if (v.type.isPointer()) { Val r = v; r.type = want; return r; }  // ptr → ptr: relabel
            // integer → pointer (a null constant `0`, or a `(T*)intExpr` cast):
            // materialize a full 64-bit address register so pointer compares and
            // stores use the whole width, not a stray 32-bit register.
            std::string d = fresh(RC::RD64);
            if (is64BitScalar(v.type) || v.type.base == Type::Long)
                emit("mov.u64 " + d + ", " + v.reg + ";");
            else
                emit("cvt.u64.u32 " + d + ", " + v.reg + ";");   // zero-extend 32-bit
            Val r; r.reg = d; r.type = want; r.space = v.space; return r;
        }
        if (v.type.isPointer()) { Val r = v; r.type = want; return r; }  // ptr → int: relabel

        // __half is a 16-bit float kept in a 32-bit register; isFloating() excludes
        // it, so the int/float paths below would reinterpret its bits. Convert via
        // f32: half→T promotes to float first; T→half narrows from float.
        const bool vHalf = v.type.base == Type::Half && v.type.ptr == 0;
        const bool wHalf = want.base == Type::Half && want.ptr == 0;
        if (vHalf && wHalf) { Val r = v; r.type = want; return r; }   // half → half: relabel
        if (vHalf) return coerce(h2f(v), want);                       // half → T
        if (wHalf) { Val r = f2h(coerce(v, floatType())); r.type = want; return r; }  // T → half

        const bool vf = v.type.isFloating(), wf = want.isFloating();
        const bool v64 = is64BitScalar(v.type), w64 = is64BitScalar(want);
        if (vf == wf && v64 == w64) { Val r = v; r.type = want; return r; }  // same kind+width
        std::string d = fresh(classOf(want));
        if (!vf && !wf) {                                         // int -> int (width)
            // Signedness of the SOURCE controls extension (u32->u64 zero-extends,
            // s32->s64 sign-extends), so an unsigned value widens without spurious
            // sign bits — e.g. (unsigned)x promoted to 64-bit against a large literal.
            emit("cvt." + uIntSuffix(want) + "." + uIntSuffix(v.type) + " " + d + ", " + v.reg + ";");
        } else if (!vf && wf) {                                   // int -> float
            emit("cvt.rn." + floatSuffix(want) + "." + uIntSuffix(v.type) + " " + d + ", " + v.reg + ";");
        } else if (vf && !wf) {                                   // float -> int
            emit("cvt.rzi." + uIntSuffix(want) + "." + floatSuffix(v.type) + " " + d + ", " + v.reg + ";");
        } else {                                                  // float -> float (width)
            // Widening f32->f64 is exact; narrowing f64->f32 needs a rounding mode.
            const std::string rnd = w64 ? "" : "rn.";
            emit("cvt." + rnd + floatSuffix(want) + "." + floatSuffix(v.type) + " " + d + ", " + v.reg + ";");
        }
        return {d, want};
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

    // Emit an integer constant of type `t` as a single immediate mov.
    Val emitConstInt(int64_t v, const Type& t) {
        if (is64BitScalar(t) || t.base == Type::Long) {
            std::string d = fresh(RC::RD64);
            emit("mov.u64 " + d + ", " + std::to_string(v) + ";");
            return {d, t};
        }
        std::string d = fresh(RC::R32);
        emit("mov.u32 " + d + ", " + std::to_string((int32_t)v) + ";");
        return {d, t};
    }

    Val emitExpr(const Expr& e) {
        line = e.line; col = e.col;
        // Constant folding: an integer constant *expression* (side-effect-free by
        // construction) collapses to one immediate — smaller PTX, fewer registers.
        // Only for compound nodes (a lone literal/ident is already minimal) and only
        // when the result is integer-typed (so a float/pointer isn't mistyped).
        if (e.kind == Expr::Binary || e.kind == Expr::Unary ||
            e.kind == Expr::Cast || e.kind == Expr::Ternary) {
            Type t = estimateType(e);
            // Only fold signed integer expressions: constEval uses signed semantics,
            // so an unsigned-typed result (differing only for /, %, >>, comparisons)
            // is left to normal codegen rather than mis-folded.
            if (!t.isFloating() && !t.isPointer() && t.base != Type::Half &&
                t.base != Type::Struct && !t.isUnsigned) {
                int64_t cv; bool cw;
                if (constEval(e, cv, cw)) return emitConstInt(cv, t);
            }
        }
        switch (e.kind) {
            case Expr::IntLit: {
                if (e.wide || e.ival > 2147483647LL || e.ival < -2147483648LL) {
                    std::string d = fresh(RC::RD64);
                    emit("mov.u64 " + d + ", " + std::to_string(e.ival) + ";");
                    return {d, longType()};
                }
                std::string d = fresh(RC::R32);
                emit("mov.u32 " + d + ", " + std::to_string(e.ival) + ";");
                return {d, intType()};
            }
            case Expr::FloatLit: {
                if (e.wide) {
                    std::string d = fresh(RC::F64);
                    emit("mov.f64 " + d + ", " + f64imm(e.fval) + ";");
                    return {d, doubleType()};
                }
                std::string d = fresh(RC::F32);
                emit("mov.f32 " + d + ", " + f32imm(e.fval) + ";");
                return {d, floatType()};
            }
            case Expr::Ident: {
                auto it = vars.find(e.str);
                if (it == vars.end()) { fail("use of undeclared identifier '" + e.str + "'"); return {}; }
                if (isScalarShared(it->second)) return loadSharedScalar(it->second);
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
                // `p->field` — member through a pointer-to-struct: load from memory.
                {
                    MemberAddr ma = structPtrMember(obj, e.str);
                    if (failed) return {};
                    if (ma.ok) {
                        std::string d = fresh(classOf(ma.type));
                        emit("ld." + std::string(ma.space == Space::Shared ? "shared." : "global.") +
                             ldSuffix(ma.type) + " " + d + ", [" + ma.reg + "];");
                        return {d, ma.type};
                    }
                }
                // `arr[i].field` — member of a struct-array element in memory.
                {
                    MemberAddr ma = structArrayMember(obj, e.str);
                    if (failed) return {};
                    if (ma.ok) {
                        std::string d = fresh(classOf(ma.type));
                        emit("ld." + std::string(ma.space == Space::Shared ? "shared." : "global.") +
                             ldSuffix(ma.type) + " " + d + ", [" + ma.reg + "];");
                        return {d, ma.type};
                    }
                }
                // Struct member read.
                if (obj.kind == Expr::Ident) {
                    auto it = vars.find(obj.str);
                    // Local struct: each member is a register.
                    if (it != vars.end() && it->second.space == Space::LocalStruct) {
                        auto& members = localStructMembers_[obj.str];
                        auto mit = members.find(e.str);
                        if (mit == members.end()) { fail("no member '." + e.str + "' in struct '" + it->second.type.structName + "'"); return {}; }
                        return mit->second;
                    }
                    // By-value struct param → load the member from its .param bytes.
                    if (it != vars.end() && it->second.space == Space::ParamStruct) {
                        const StructDef* def = mod_ ? mod_->findStruct(it->second.type.structName) : nullptr;
                        const StructMember* m = def ? def->find(e.str) : nullptr;
                        if (!m) { fail("no member '." + e.str + "' in struct '" + it->second.type.structName + "'"); return {}; }
                        std::string d = fresh(classOf(m->type));
                        emit("ld.param." + std::string(ldSuffix(m->type)) + " " + d + ", [" +
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

    // Statically estimate an expression's type (kind + width) without emitting —
    // needed to pick the result register class for a ternary before emitting its
    // branches, and to know a value's width for promotion.
    Type estimateType(const Expr& e) {
        switch (e.kind) {
            case Expr::FloatLit: return e.wide ? doubleType() : floatType();
            case Expr::IntLit:
                return (e.wide || e.ival > 2147483647LL || e.ival < -2147483648LL) ? longType() : intType();
            case Expr::Ident: { auto it = vars.find(e.str); return it != vars.end() ? it->second.type : intType(); }
            case Expr::Member: {
                const Expr& obj = *e.args[0];
                {   // p->field: member of a pointer-to-struct.
                    Type ot = estimateType(obj);
                    if (ot.isPointer() && ot.base == Type::Struct && ot.ptr == 1) {
                        const StructDef* def = mod_ ? mod_->findStruct(ot.structName) : nullptr;
                        const StructMember* m = def ? def->find(e.str) : nullptr;
                        if (m) return m->type;
                    }
                }
                if (obj.kind == Expr::Index) {   // arr[i].field: member of a struct-array element
                    Type et = indexPointee(obj);
                    if (et.base == Type::Struct && et.ptr == 0) {
                        const StructDef* def = mod_ ? mod_->findStruct(et.structName) : nullptr;
                        const StructMember* m = def ? def->find(e.str) : nullptr;
                        if (m) return m->type;
                    }
                }
                if (obj.kind == Expr::Ident) {
                    auto it = vars.find(obj.str);
                    if (it != vars.end() && it->second.space == Space::LocalStruct) {
                        auto mit = localStructMembers_.find(obj.str);
                        if (mit != localStructMembers_.end()) {
                            auto m = mit->second.find(e.str);
                            if (m != mit->second.end()) return m->second.type;
                        }
                    }
                    if (it != vars.end() && it->second.space == Space::ParamStruct) {
                        const StructDef* def = mod_ ? mod_->findStruct(it->second.type.structName) : nullptr;
                        const StructMember* m = def ? def->find(e.str) : nullptr;
                        if (m) return m->type;
                    }
                }
                return intType();  // threadIdx/blockIdx/… are integers
            }
            case Expr::Index: return indexPointee(e);
            case Expr::Cast:  return e.castType;
            case Expr::Unary:
                if (e.str == "!") return intType();
                if (e.str == "*") { Type t = estimateType(*e.args[0]); if (t.ptr > 0) t.ptr--; return t; }
                if (e.str == "&") { Type t = estimateType(*e.args[0]); t.ptr++; return t; }
                return estimateType(*e.args[0]);
            case Expr::Binary: {
                const std::string& o = e.str;
                if (o == "<" || o == "<=" || o == ">" || o == ">=" || o == "==" || o == "!=" ||
                    o == "&&" || o == "||") return intType();
                return promote(estimateType(*e.args[0]), estimateType(*e.args[1]));
            }
            case Expr::Assign:  return estimateType(*e.args[0]);
            case Expr::Ternary: return promote(estimateType(*e.args[1]), estimateType(*e.args[2]));
            case Expr::Call: {
                const std::string& fn = e.str;
                if (fn == "min" || fn == "max" || fn == "abs")
                    return e.args.empty() ? intType() : estimateType(*e.args[0]);
                if (fn == "atomicAdd")
                    return e.args.size() < 2 ? intType() : estimateType(*e.args[1]);
                return floatType();  // sqrtf/expf/fmaf/… return float
            }
        }
        return intType();
    }
    bool isFloatExpr(const Expr& e) { return estimateType(e).isFloating(); }

    // Wrap a value to a 32- or 64-bit two's-complement result (`wide` = 64-bit).
    static int64_t wrapTo(int64_t v, bool wide) { return wide ? v : (int64_t)(int32_t)v; }

    // Fold a *signed* integer constant expression. Returns true and sets `out` and
    // `wide` (result is 64-bit long vs 32-bit int), or false if `e` isn't a constant
    // integer. Critically, every operation wraps to its result width — a 32-bit
    // subexpression wraps at 32 bits — so the folded value matches the runtime's
    // 32-bit ops (e.g. `(big << 3) >> 10` overflows int32 exactly as at runtime).
    static bool constEval(const Expr& e, int64_t& out, bool& wide) {
        switch (e.kind) {
            case Expr::IntLit:
                out = e.ival;
                wide = e.wide || e.ival > 2147483647LL || e.ival < -2147483648LL;
                return true;
            case Expr::Cast: {
                const Type& ct = e.castType;
                if (ct.isFloating() || ct.base == Type::Half || ct.isPointer() || ct.isStruct())
                    return false;   // not an integer constant
                int64_t v; bool vw; if (!constEval(*e.args[0], v, vw)) return false;
                switch (ct.base) {   // narrow to the cast's integer width
                    case Type::Bool:  out = (v != 0); wide = false; break;
                    case Type::Char:  out = ct.isUnsigned ? (int64_t)(uint8_t)v  : (int64_t)(int8_t)v;  wide = false; break;
                    case Type::Short: out = ct.isUnsigned ? (int64_t)(uint16_t)v : (int64_t)(int16_t)v; wide = false; break;
                    case Type::Int:   out = ct.isUnsigned ? (int64_t)(uint32_t)v : (int64_t)(int32_t)v; wide = false; break;
                    default:          out = v; wide = true; break;   // long
                }
                return true;
            }
            case Expr::Ternary: {
                int64_t c; bool cw; if (!constEval(*e.args[0], c, cw)) return false;
                return constEval(c ? *e.args[1] : *e.args[2], out, wide);
            }
            case Expr::Unary: {
                int64_t v; bool vw; if (!constEval(*e.args[0], v, vw)) return false;
                if (e.str == "!") { out = !v; wide = false; return true; }
                wide = vw;
                if (e.str == "-")      out = wrapTo((int64_t)(0u - (uint64_t)v), wide);
                else if (e.str == "+") out = v;
                else if (e.str == "~") out = wrapTo((int64_t)~(uint64_t)v, wide);
                else return false;
                return true;
            }
            case Expr::Binary: {
                int64_t a, b; bool aw, bw;
                if (!constEval(*e.args[0], a, aw) || !constEval(*e.args[1], b, bw)) return false;
                const std::string& o = e.str;
                // Comparisons and logical operators produce an int 0/1.
                wide = false;
                if (o == "==") { out = (a == b); return true; }
                if (o == "!=") { out = (a != b); return true; }
                if (o == "<")  { out = (a < b);  return true; }
                if (o == "<=") { out = (a <= b); return true; }
                if (o == ">")  { out = (a > b);  return true; }
                if (o == ">=") { out = (a >= b); return true; }
                if (o == "&&") { out = (a != 0) && (b != 0); return true; }
                if (o == "||") { out = (a != 0) || (b != 0); return true; }
                // Shifts: the result type is the (promoted) left operand's.
                if (o == "<<") { wide = aw; int s = (int)(b & (wide ? 63 : 31)); out = wrapTo((int64_t)((uint64_t)a << s), wide); return true; }
                if (o == ">>") { wide = aw; int s = (int)(b & (wide ? 63 : 31)); out = wide ? (a >> s) : (int64_t)((int32_t)a >> s); return true; }
                // Arithmetic / bitwise: promote to 64-bit if either operand is.
                wide = aw || bw;
                int64_t res;
                if (o == "+")      res = (int64_t)((uint64_t)a + (uint64_t)b);
                else if (o == "-") res = (int64_t)((uint64_t)a - (uint64_t)b);
                else if (o == "*") res = (int64_t)((uint64_t)a * (uint64_t)b);
                else if (o == "/") { if (!b) return false; res = a / b; }
                else if (o == "%") { if (!b) return false; res = a % b; }
                else if (o == "&") res = a & b;
                else if (o == "|") res = a | b;
                else if (o == "^") res = a ^ b;
                else return false;
                out = wrapTo(res, wide);
                return true;
            }
            default: return false;
        }
    }

    Val emitCast(const Expr& e) {
        if (!ensureSupported(e.castType)) return {};
        Val v = emitExpr(*e.args[0]);
        if (failed) return {};
        const Type& to = e.castType;
        // Reject casts with no meaning in the subset (rather than silently
        // relabeling registers): structs aren't scalar-convertible, and a floating
        // value and a pointer can't be cast to each other (that needs a bit
        // reinterpret, e.g. __float_as_int, not a value cast).
        if (to.isStruct() || v.type.isStruct()) { fail("cannot cast to or from a struct type"); return {}; }
        const bool toFloat = to.isFloating() || to.base == Type::Half;
        const bool fromFloat = v.type.isFloating() || v.type.base == Type::Half;
        if ((toFloat && v.type.isPointer()) || (to.isPointer() && fromFloat)) {
            fail("cannot cast between a floating-point type and a pointer"); return {};
        }
        Val r = coerce(v, to);
        r.type = to;
        if (to.isPointer()) r.space = v.space;
        return r;
    }

    Val emitTernary(const Expr& e) {
        Type rt = promote(estimateType(*e.args[1]), estimateType(*e.args[2]));
        std::string res = fresh(classOf(rt));
        std::string elseL = label(), endL = label();
        emitCondBranchFalse(*e.args[0], elseL);
        if (failed) return {};
        Val t = coerce(emitExpr(*e.args[1]), rt);
        if (failed) return {};
        emit(std::string(movFor(rt)) + res + ", " + t.reg + ";");
        emit("bra " + endL + ";");
        emitLabel(elseL);
        Val f = coerce(emitExpr(*e.args[2]), rt);
        if (failed) return {};
        emit(std::string(movFor(rt)) + res + ", " + f.reg + ";");
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

        const int elem = pointeeBytes(a.pointee);
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
        emit("ld." + spacePrefix(a) + ldSuffix(a.pointee) +
             " " + d + ", [" + a.reg + "];");
        return {d, a.pointee};
    }

    // A scalar `__shared__` variable (e.g. `__shared__ int flag;`): lives in
    // shared memory as a 1-element cell, addressed by its PTX symbol, so a write
    // by one thread is seen block-wide — unlike a register, which is per-thread.
    // Shared *arrays* decay to a pointer (isPointer), so a non-pointer Shared Val
    // is exactly a scalar shared variable.
    static bool isScalarShared(const Val& v) { return v.space == Space::Shared && !v.type.isPointer(); }
    Val loadSharedScalar(const Val& v) {
        std::string addr = fresh(RC::R32), d = fresh(classOf(v.type));
        emit("mov.u32 " + addr + ", " + v.sharedName + ";");
        emit("ld.shared." + std::string(memSuffix(v.type)) + " " + d + ", [" + addr + "];");
        return {d, v.type};
    }
    void storeSharedScalar(const Val& v, const Val& val) {
        std::string addr = fresh(RC::R32);
        emit("mov.u32 " + addr + ", " + v.sharedName + ";");
        emit("st.shared." + std::string(memSuffix(v.type)) + " [" + addr + "], " + val.reg + ";");
    }

    // Store `value` (already coerced) to the address of index-expr `lhs`.
    void emitStore(const Expr& lhs, const Val& value) {
        Addr a = emitAddress(lhs);
        if (failed) return;
        emit("st." + spacePrefix(a) + stSuffix(a.pointee) +
             " [" + a.reg + "], " + value.reg + ";");
    }

    // ++x / --x / x++ / x-- on a scalar variable (int or float).
    Val emitIncDec(const Expr& e) {
        const Expr& operand = *e.args[0];
        if (operand.kind != Expr::Ident) { fail("'++'/'--' requires a variable"); return {}; }
        auto it = vars.find(operand.str);
        if (it == vars.end()) { fail("'++'/'--' of undeclared '" + operand.str + "'"); return {}; }
        Val& var = it->second;
        const bool inc = e.str.find("++") != std::string::npos;
        const bool pre = e.str.compare(0, 3, "pre") == 0;

        // Pointer ++ / -- advances/retreats by one element (pointee size), 64-bit.
        // Arrays are not modifiable lvalues, so reject those.
        if (var.type.isPointer()) {
            if (arrayDims_.count(operand.str)) { fail("cannot '++'/'--' an array"); return {}; }
            Type pointee = var.type; pointee.ptr -= 1;
            const int step = pointeeBytes(pointee);
            Val old;
            if (!pre) { old.type = var.type; old.reg = fresh(RC::RD64); emit("mov.u64 " + old.reg + ", " + var.reg + ";"); }
            emit(std::string(inc ? "add.s64 " : "sub.s64 ") + var.reg + ", " + var.reg + ", " + std::to_string(step) + ";");
            return pre ? var : old;
        }
        const std::string one = !var.type.isFloating() ? "1"
                              : (var.type.base == Type::Double ? f64imm(1.0) : "0f3F800000");
        const std::string addsub = std::string(inc ? "add." : "sub.") + arithSuffix(var.type);

        // A scalar __shared__ variable has no register — read/modify/write it
        // through ld.shared/st.shared (a register RMW would emit `add , , 1`).
        if (isScalarShared(var)) {
            Val cur = loadSharedScalar(var);
            std::string nv = fresh(classOf(var.type));
            emit(addsub + " " + nv + ", " + cur.reg + ", " + one + ";");
            storeSharedScalar(var, {nv, var.type});
            return pre ? Val{nv, var.type} : cur;   // prefix → new value, postfix → old
        }

        Val old;
        if (!pre) {  // postfix: capture the value before the update
            old.type = var.type;
            old.reg = fresh(classOf(var.type));
            emit(std::string(movFor(var.type)) + old.reg + ", " + var.reg + ";");
        }
        emit(addsub + " " + var.reg + ", " + var.reg + ", " + one + ";");
        return pre ? var : old;
    }

    // PTX space prefix for a load/store through a pointer value (shared vs global).
    static std::string ptrSpacePrefix(const Val& p) {
        return p.space == Space::Shared ? "shared." : p.space == Space::Local ? "local." : "global.";
    }

    // Dereference a pointer value: load the pointed-to element (`*p`, `*(a+i)`).
    Val emitDeref(const Val& p) {
        if (!p.type.isPointer()) { fail("cannot dereference a non-pointer"); return {}; }
        Type pointee = p.type; pointee.ptr -= 1;
        std::string d = fresh(classOf(pointee));
        emit("ld." + ptrSpacePrefix(p) + ldSuffix(pointee) + " " + d + ", [" + p.reg + "];");
        return {d, pointee};
    }

    Val emitUnary(const Expr& e) {
        if (e.str == "pre++" || e.str == "pre--" || e.str == "post++" || e.str == "post--")
            return emitIncDec(e);
        if (e.str == "+") return emitExpr(*e.args[0]);

        // Dereference: `*p` / `*(a+i)`.
        if (e.str == "*") {
            Val p = emitExpr(*e.args[0]);
            if (failed) return {};
            return emitDeref(p);
        }
        // Address-of an array element: `&a[i]` → a pointer to the element. (A plain
        // scalar local has no memory address in this register model, so `&scalar`
        // is a located error; `&global[i]` yields a 64-bit global address.)
        if (e.str == "&") {
            const Expr& operand = *e.args[0];
            if (operand.kind != Expr::Index) { fail("'&' requires an array element (e.g. &a[i])"); return {}; }
            Addr a = emitAddress(operand);
            if (failed) return {};
            if (a.shared || a.local) { fail("'&' of a __shared__/local element is unsupported"); return {}; }
            Type ptr = a.pointee; ptr.ptr += 1;
            Val r; r.reg = a.reg; r.type = ptr; r.space = Space::Global;
            return r;
        }

        Val v = emitExpr(*e.args[0]);
        if (failed) return {};
        if (e.str == "-") {
            std::string d = fresh(classOf(v.type));
            emit("neg." + arithSuffix(v.type) + " " + d + ", " + v.reg + ";");
            return {d, v.type};
        }
        if (e.str == "!") {
            std::string p = fresh(RC::Pred), d = fresh(RC::R32);
            std::string zero = !v.type.isFloating() ? "0"
                             : (v.type.base == Type::Double ? f64imm(0.0) : f32imm(0.0));
            emit("setp.eq." + arithSuffix(v.type) + " " + p + ", " + v.reg + ", " + zero + ";");
            emit("mov.u32 " + d + ", 0;");
            emit("@" + p + " mov.u32 " + d + ", 1;");
            return {d, intType()};
        }
        fail("unsupported unary operator '" + e.str + "'");
        return {};
    }

    // C-style usual arithmetic conversions across the supported scalar types.
    static Type promote(const Type& a, const Type& b) {
        // A pointer participates as itself (64-bit addressing): this is what makes
        // pointer comparisons and `cond ? p : q` use the full 64-bit width and the
        // pointer register class, rather than collapsing to a 32-bit int.
        if (a.isPointer()) return a;
        if (b.isPointer()) return b;
        if (a.base == Type::Double || b.base == Type::Double) return doubleType();
        // __half promotes to float for arithmetic (as C++ half operators do), so
        // `h1 + h2` computes in f32 rather than integer-adding the raw fp16 bits;
        // assigning the result back to a __half narrows it via coerce.
        if (a.isFloating() || b.isFloating() ||
            a.base == Type::Half || b.base == Type::Half) return floatType();
        // Integer operands smaller than int (char/short/bool) undergo the integer
        // promotion to int; the common type is int, or long if either operand is
        // long-rank. (Values already sit sign/zero-extended in 32-bit registers.)
        const bool resultLong = (a.base == Type::Long || b.base == Type::Long);
        Type r = resultLong ? longType() : intType();
        // Unsigned propagates only from an operand AT the result's rank: e.g.
        // `unsigned int + long` (64-bit) is *signed* long (long represents every
        // unsigned int), while `unsigned long + int` is unsigned long.
        auto atResultRank = [&](const Type& t) { return (t.base == Type::Long) == resultLong; };
        if ((a.isUnsigned && atResultRank(a)) || (b.isUnsigned && atResultRank(b)))
            r.isUnsigned = true;
        return r;
    }

    // setp suffix for comparing two values of common type `ct` with operator `o`:
    // pointers compare as unsigned 64-bit addresses; ==/!= are sign-agnostic;
    // ordered integer compares honor unsignedness.
    static std::string cmpSuffix(const Type& ct, const std::string& o) {
        if (ct.isPointer()) return "u64";
        if (ct.isFloating()) return floatSuffix(ct);
        if (o == "==" || o == "!=") return intSuffix(ct);
        return uIntSuffix(ct);
    }

    // Emit a binary arithmetic/bitwise op, promoting both operands to their
    // common type (width-aware: f64/f32/s64/s32/b64/b32). Shared by emitBinary
    // and compound assignment so the op-suffix logic lives in one place.
    Val emitArith(const std::string& op, Val a, Val b) {
        Type ct = promote(a.type, b.type);
        const bool fp = ct.isFloating();
        a = coerce(a, ct); b = coerce(b, ct);
        if (failed) return {};
        const std::string suf = arithSuffix(ct);
        std::string ins;
        if (op == "+") ins = "add." + suf;
        else if (op == "-") ins = "sub." + suf;
        else if (op == "*") ins = fp ? ("mul." + suf) : ("mul.lo." + suf);
        else if (op == "/") ins = fp ? ("div.rn." + suf) : ("div." + uIntSuffix(ct));
        else if (op == "%") { if (fp) { fail("'%' on a floating type"); return {}; } ins = "rem." + uIntSuffix(ct); }
        else if (op == "&") ins = "and." + bitSuffix(ct);
        else if (op == "|") ins = "or." + bitSuffix(ct);
        else if (op == "^") ins = "xor." + bitSuffix(ct);
        else if (op == "<<") ins = "shl." + bitSuffix(ct);
        else if (op == ">>") ins = "shr." + uIntSuffix(ct);   // logical for unsigned, arithmetic for signed
        else { fail("unsupported binary operator '" + op + "'"); return {}; }
        std::string d = fresh(classOf(ct));
        emit(ins + " " + d + ", " + a.reg + ", " + b.reg + ";");
        return {d, ct};
    }

    // A predicate that is true iff `v` is non-zero (C truthiness), handling every
    // scalar kind: int/long (`!= 0`), pointer (`!= 0` as u64), float/double
    // (`!= 0.0`), and __half (promoted to float first so -0.0 is correctly falsy).
    std::string emitToPred(const Val& v) {
        std::string p = fresh(RC::Pred);
        if (v.type.base == Type::Half) {
            Val f = h2f(v);
            emit("setp.ne.f32 " + p + ", " + f.reg + ", " + f32imm(0.0) + ";");
        } else if (v.type.isPointer()) {
            emit("setp.ne.u64 " + p + ", " + v.reg + ", 0;");
        } else if (v.type.isFloating()) {
            std::string z = v.type.base == Type::Double ? f64imm(0.0) : f32imm(0.0);
            emit("setp.ne." + floatSuffix(v.type) + " " + p + ", " + v.reg + ", " + z + ";");
        } else {
            emit("setp.ne." + intSuffix(v.type) + " " + p + ", " + v.reg + ", 0;");
        }
        return p;
    }

    // Materialize a comparison as an int 0/1 (predicated mov, no selp needed).
    Val emitCompare(const std::string& op, Val a, Val b) {
        Type ct = promote(a.type, b.type);
        a = coerce(a, ct); b = coerce(b, ct);
        if (failed) return {};
        const char* cc = op == "<" ? "lt" : op == "<=" ? "le" : op == ">" ? "gt" :
                         op == ">=" ? "ge" : op == "==" ? "eq" : "ne";
        const std::string suf = cmpSuffix(ct, op);
        std::string p = fresh(RC::Pred), d = fresh(RC::R32);
        emit("setp." + std::string(cc) + "." + suf + " " + p + ", " + a.reg + ", " + b.reg + ";");
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

        // Logical && / || — real C semantics: each side is tested for truthiness
        // (not bit-and'd), the result is 0/1, and the RHS is short-circuited (so
        // `p && p->x` never dereferences a null `p`). The LHS is evaluated first;
        // the RHS only on the path where it can change the result.
        if (op == "&&" || op == "||") {
            std::string d = fresh(RC::R32), endL = label();
            Val a = emitExpr(*e.args[0]); if (failed) return {};
            std::string pa = emitToPred(a);
            emit("selp.b32 " + d + ", 1, 0, " + pa + ";");        // d = (a != 0)
            std::string skip = fresh(RC::Pred);
            // &&: LHS false ⇒ result 0, skip RHS. ||: LHS true ⇒ result 1, skip RHS.
            emit("setp." + std::string(op == "&&" ? "eq" : "ne") + ".s32 " + skip + ", " + d + ", 0;");
            emit("@" + skip + " bra " + endL + ";");
            Val b = emitExpr(*e.args[1]); if (failed) return {};
            std::string pb = emitToPred(b);
            emit("selp.b32 " + d + ", 1, 0, " + pb + ";");        // d = (b != 0)
            emitLabel(endL);
            return {d, intType()};
        }

        Val a = emitExpr(*e.args[0]); if (failed) return {};
        Val b = emitExpr(*e.args[1]); if (failed) return {};

        // Pointer arithmetic: p + i / i + p / p - i scale the integer index by
        // the pointee size and keep 64-bit addressing (and the pointer's memory
        // space), matching C pointer semantics. This mirrors the address folding
        // that a[i] does, but for an explicit pointer value.
        if ((op == "+" || op == "-") && (a.type.isPointer() || b.type.isPointer())) {
            // pointer - pointer → element-count difference (ptrdiff): byte diff / size.
            if (a.type.isPointer() && b.type.isPointer()) {
                if (op != "-") { fail("cannot add two pointers"); return {}; }
                Type pointee = a.type; pointee.ptr -= 1;
                const int elem = pointeeBytes(pointee);
                std::string bd = fresh(RC::RD64), res = fresh(RC::RD64);
                emit("sub.s64 " + bd + ", " + a.reg + ", " + b.reg + ";");
                emit("div.s64 " + res + ", " + bd + ", " + std::to_string(elem) + ";");
                return {res, longType()};
            }
            const bool aPtr = a.type.isPointer();
            if (op == "-" && !aPtr) { fail("cannot subtract a pointer from an integer"); return {}; }
            const Val& ptr = aPtr ? a : b;
            const Val& idx = aPtr ? b : a;
            Type pointee = ptr.type; pointee.ptr -= 1;
            Val i32 = coerce(idx, intType()); if (failed) return {};
            std::string off = fresh(RC::RD64), res = fresh(RC::RD64);
            emit("mul.wide.s32 " + off + ", " + i32.reg + ", " + std::to_string(pointeeBytes(pointee)) + ";");
            emit(std::string(op == "+" ? "add.s64 " : "sub.s64 ") + res + ", " + ptr.reg + ", " + off + ";");
            Val r; r.reg = res; r.type = ptr.type; r.space = ptr.space;
            return r;
        }
        return emitArith(op, a, b);
    }

    // For `p->field` (p a pointer-to-struct in memory): compute the member's
    // address (ptr + byte offset) and its type/space. `.ok` is false (with no code
    // emitted) when `obj` is not a struct pointer, so callers can fall through to
    // the struct-value handling.
    struct MemberAddr { std::string reg; Type type; Space space = Space::Global; bool ok = false; };
    MemberAddr structPtrMember(const Expr& obj, const std::string& field) {
        MemberAddr r;
        Type ot = estimateType(obj);
        if (!(ot.isPointer() && ot.base == Type::Struct && ot.ptr == 1)) return r;  // not a struct ptr
        Val pv = emitExpr(obj);
        if (failed) return r;
        const StructDef* def = mod_ ? mod_->findStruct(ot.structName) : nullptr;
        const StructMember* m = def ? def->find(field) : nullptr;
        if (!m) { fail("no member '->" + field + "' in struct '" + ot.structName + "'"); return r; }
        std::string addr = fresh(RC::RD64);
        emit("add.s64 " + addr + ", " + pv.reg + ", " + std::to_string(m->offset) + ";");
        r.reg = addr; r.type = m->type; r.space = pv.space; r.ok = true;
        return r;
    }

    // For `arr[i].field` (arr a struct array in memory): the member's address is
    // the struct element's address (arr + i*sizeof(struct), from emitAddress) plus
    // the member's byte offset. `.ok` is false (no code emitted) when `obj` isn't
    // an index into a struct array, so callers fall through.
    MemberAddr structArrayMember(const Expr& obj, const std::string& field) {
        MemberAddr r;
        if (obj.kind != Expr::Index) return r;
        Type et = indexPointee(obj);
        if (!(et.base == Type::Struct && et.ptr == 0)) return r;   // element isn't a struct
        const StructDef* def = mod_ ? mod_->findStruct(et.structName) : nullptr;
        const StructMember* m = def ? def->find(field) : nullptr;
        if (!m) { fail("no member '." + field + "' in struct '" + et.structName + "'"); return r; }
        Addr a = emitAddress(obj);   // address of the struct element arr[i]
        if (failed) return r;
        if (a.local) { fail("member access on a local struct array is unsupported"); return r; }
        if (a.shared) {
            std::string addr = fresh(RC::R32);
            emit("add.s32 " + addr + ", " + a.reg + ", " + std::to_string(m->offset) + ";");
            r.reg = addr; r.space = Space::Shared;
        } else {
            std::string addr = fresh(RC::RD64);
            emit("add.s64 " + addr + ", " + a.reg + ", " + std::to_string(m->offset) + ";");
            r.reg = addr; r.space = Space::Global;
        }
        r.type = m->type; r.ok = true;
        return r;
    }

    // Copy every member of a struct value (a local-struct or by-value param
    // struct named by `srcExpr`) into the local struct `dst`.
    void emitStructCopy(const std::string& dst, const Expr& srcExpr) {
        if (srcExpr.kind != Expr::Ident) { fail("a struct can only be copied from a struct variable"); return; }
        auto sit = vars.find(srcExpr.str);
        if (sit == vars.end()) { fail("use of undeclared identifier '" + srcExpr.str + "'"); return; }
        const Val src = sit->second;
        auto& dmembers = localStructMembers_[dst];
        if (src.space == Space::LocalStruct) {
            auto smembers = localStructMembers_[srcExpr.str];   // by value (avoid rehash aliasing)
            for (auto& kv : dmembers) {
                auto s = smembers.find(kv.first);
                if (s == smembers.end()) { fail("struct member mismatch copying '" + srcExpr.str + "'"); return; }
                emit(std::string(movFor(kv.second.type)) + kv.second.reg + ", " + s->second.reg + ";");
            }
        } else if (src.space == Space::ParamStruct) {
            const StructDef* def = mod_ ? mod_->findStruct(src.type.structName) : nullptr;
            for (auto& kv : dmembers) {
                const StructMember* m = def ? def->find(kv.first) : nullptr;
                if (!m) { fail("struct member mismatch copying param '" + srcExpr.str + "'"); return; }
                emit("ld.param." + std::string(ldSuffix(m->type)) + " " + kv.second.reg + ", [" +
                     src.paramName + "+" + std::to_string(m->offset) + "];");
            }
        } else {
            fail("cannot initialize a struct from a non-struct value");
        }
    }

    Val emitAssign(const Expr& e) {
        const Expr& lhs = *e.args[0];
        const std::string& op = e.str;

        // Compute the value to store: rhs, or (lhs <binop> rhs) for compound.
        auto computeRhs = [&](const Val& lhsVal) -> Val {
            Val rhs = emitExpr(*e.args[1]);
            if (failed) return {};
            if (op == "=") return rhs;
            std::string bop = op.substr(0, op.size() - 1);  // "+="->"+", "<<="->"<<", "&="->"&"
            return emitArith(bop, lhsVal, rhs);   // shared width-aware arithmetic
        };

        if (lhs.kind == Expr::Ident) {
            auto it = vars.find(lhs.str);
            if (it == vars.end()) { line = lhs.line; col = lhs.col; fail("assignment to undeclared '" + lhs.str + "'"); return {}; }
            Val& var = it->second;
            if (var.space == Space::LocalStruct) {   // whole-struct copy: s1 = s2
                if (op != "=") { fail("compound assignment on a struct is unsupported"); return {}; }
                emitStructCopy(lhs.str, *e.args[1]);
                return var;
            }
            if (isScalarShared(var)) {
                // Compound (+= etc.) reads the current shared value first.
                Val cur = (op == "=") ? Val{} : loadSharedScalar(var);
                Val rhs = computeRhs(cur);
                if (failed) return {};
                rhs = coerce(rhs, var.type);
                storeSharedScalar(var, rhs);
                return rhs;
            }
            Val rhs = computeRhs(var);
            if (failed) return {};
            rhs = coerce(rhs, var.type);
            emit(std::string(movFor(var.type)) + var.reg + ", " + rhs.reg + ";");
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
        // Store through a dereferenced pointer: `*p = v` (and compound `*p += v`).
        if (lhs.kind == Expr::Unary && lhs.str == "*") {
            Val p = emitExpr(*lhs.args[0]);
            if (failed) return {};
            if (!p.type.isPointer()) { fail("cannot dereference a non-pointer"); return {}; }
            Type pointee = p.type; pointee.ptr -= 1;
            Val value;
            if (op == "=") {
                value = emitExpr(*e.args[1]);
            } else {                       // compound: load current through p, combine
                std::string cur = fresh(classOf(pointee));
                emit("ld." + ptrSpacePrefix(p) + ldSuffix(pointee) + " " + cur + ", [" + p.reg + "];");
                value = computeRhs({cur, pointee});
            }
            if (failed) return {};
            value = coerce(value, pointee);
            emit("st." + ptrSpacePrefix(p) + stSuffix(pointee) + " [" + p.reg + "], " + value.reg + ";");
            return value;
        }
        // Write through a struct pointer: `p->field = v` (and compound `p->field += v`).
        if (lhs.kind == Expr::Member) {
            MemberAddr ma = structPtrMember(*lhs.args[0], lhs.str);
            if (failed) return {};
            if (ma.ok) {
                const std::string sp = ma.space == Space::Shared ? "shared." : "global.";
                Val value;
                if (op == "=") {
                    value = emitExpr(*e.args[1]);
                } else {
                    std::string cur = fresh(classOf(ma.type));
                    emit("ld." + sp + ldSuffix(ma.type) + " " + cur + ", [" + ma.reg + "];");
                    value = computeRhs({cur, ma.type});
                }
                if (failed) return {};
                value = coerce(value, ma.type);
                emit("st." + sp + stSuffix(ma.type) + " [" + ma.reg + "], " + value.reg + ";");
                return value;
            }
        }
        // Write a struct-array member: `arr[i].field = v` (and compound `+=`).
        if (lhs.kind == Expr::Member) {
            MemberAddr ma = structArrayMember(*lhs.args[0], lhs.str);
            if (failed) return {};
            if (ma.ok) {
                const std::string sp = ma.space == Space::Shared ? "shared." : "global.";
                Val value;
                if (op == "=") {
                    value = emitExpr(*e.args[1]);
                } else {
                    std::string cur = fresh(classOf(ma.type));
                    emit("ld." + sp + ldSuffix(ma.type) + " " + cur + ", [" + ma.reg + "];");
                    value = computeRhs({cur, ma.type});
                }
                if (failed) return {};
                value = coerce(value, ma.type);
                emit("st." + sp + stSuffix(ma.type) + " [" + ma.reg + "], " + value.reg + ";");
                return value;
            }
        }
        // Write a local-struct member: `s.field = v` (and compound `s.field += v`).
        if (lhs.kind == Expr::Member && lhs.args[0]->kind == Expr::Ident) {
            auto it = vars.find(lhs.args[0]->str);
            if (it != vars.end() && it->second.space == Space::LocalStruct) {
                auto& members = localStructMembers_[lhs.args[0]->str];
                auto mit = members.find(lhs.str);
                if (mit == members.end()) { fail("no member '." + lhs.str + "' in struct '" + it->second.type.structName + "'"); return {}; }
                const Val member = mit->second;   // reg name is stable; copy to avoid rehash aliasing
                Val rhs = computeRhs(member);
                if (failed) return {};
                rhs = coerce(rhs, member.type);
                emit(std::string(movFor(member.type)) + member.reg + ", " + rhs.reg + ";");
                return member;
            }
        }
        fail("invalid assignment target");
        return {};
    }

    // atomicAdd(&arr[i], val): read-modify-write returning the old value.
    //  * global: a real `atom.global.add`, so it stays correct when the backend
    //    runs a grid's CTAs on parallel interpreter instances.
    //  * shared: within one CTA the interpreter runs threads sequentially, so a
    //    plain ld/add/st is atomic (a CTA never spans instances).
    // atomicAdd/Sub/Min/Max/Exch/And/Or/Xor(&p[i], v) and atomicCAS(&p[i], cmp, v).
    // All return the OLD value. `op` is the CUDA op sans the "atomic" prefix, lower-
    // cased: add/sub/min/max/exch/and/or/xor/cas.
    Val emitAtomic(const Expr& e, const std::string& op) {
        const bool isCas = (op == "cas");
        if (e.args.size() != (isCas ? 3u : 2u)) {
            fail("atomic" + op + " has the wrong number of arguments"); return {};
        }
        const Expr& a0 = *e.args[0];
        if (a0.kind != Expr::Unary || a0.str != "&" || a0.args.empty() || a0.args[0]->kind != Expr::Index) {
            fail("atomic" + op + " expects &array[index] as its first argument");
            return {};
        }
        Addr addr = emitAddress(*a0.args[0]);
        if (failed) return {};
        const Type pt = addr.pointee;

        // Type/op validation — reject combinations that have no CUDA atomic (and
        // would otherwise emit invalid or nonsensical PTX). Atomics act on 32- or
        // 64-bit types only; min/max and the bitwise ops require an integer type.
        const int atomBytes = pt.isPointer() ? 8 : pt.elemBytes();
        if (atomBytes != 4 && atomBytes != 8) {
            fail("atomic" + op + " requires a 32- or 64-bit type (got a " +
                 std::to_string(atomBytes * 8) + "-bit type)"); return {};
        }
        if (pt.isFloating() && (op == "min" || op == "max" || op == "and" ||
                                op == "or" || op == "xor")) {
            fail("atomic" + op + " is not defined for floating-point types"); return {};
        }

        const bool w64 = is64BitScalar(pt) || pt.base == Type::Long || pt.isPointer();
        const bool bitOp = (op == "exch" || op == "and" || op == "or" || op == "xor" || isCas);
        const std::string bitSuf = w64 ? "b64" : "b32";
        // atom type suffix: bit ops → bXX; min/max → signed/unsigned int; add → mem.
        std::string atomSuf;
        if (bitOp) atomSuf = bitSuf;
        else if (op == "min" || op == "max")
            atomSuf = w64 ? (pt.isUnsigned ? "u64" : "s64") : (pt.isUnsigned ? "u32" : "s32");
        else atomSuf = memSuffix(pt);            // add / sub (int or float)

        Val cmp;
        if (isCas) { cmp = coerce(emitExpr(*e.args[1]), pt); if (failed) return {}; }
        Val val = coerce(emitExpr(*e.args[isCas ? 2 : 1]), pt);
        if (failed) return {};
        if (op == "sub") {                       // atomicSub(p,v) == atomicAdd(p,-v)
            std::string neg = fresh(classOf(pt));
            emit("neg." + arithSuffix(pt) + " " + neg + ", " + val.reg + ";");
            val = {neg, pt};
        }
        const std::string ptxOp = (op == "sub") ? "add" : op;

        std::string oldv = fresh(classOf(pt));
        if (addr.shared || addr.local) {
            // Shared: one CTA runs sequentially in the interpreter. Local: private
            // to the thread. Either way a plain ld / compute / st is atomic here.
            const std::string sp = spacePrefix(addr), ms = memSuffix(pt);
            emit("ld." + sp + ms + " " + oldv + ", [" + addr.reg + "];");
            std::string nv;
            if (op == "exch") {
                nv = val.reg;
            } else if (isCas) {
                std::string pc = fresh(RC::Pred), sel = fresh(classOf(pt));
                emit("setp.eq." + arithSuffix(pt) + " " + pc + ", " + oldv + ", " + cmp.reg + ";");
                emit("selp." + bitSuf + " " + sel + ", " + val.reg + ", " + oldv + ", " + pc + ";");
                nv = sel;
            } else {                             // add/min/max/and/or/xor
                nv = fresh(classOf(pt));
                const std::string s = (ptxOp == "add") ? arithSuffix(pt) : atomSuf;
                emit(ptxOp + "." + s + " " + nv + ", " + oldv + ", " + val.reg + ";");
            }
            emit("st." + sp + ms + " [" + addr.reg + "], " + nv + ";");
        } else if (isCas) {
            emit("atom.global.cas." + atomSuf + " " + oldv + ", [" + addr.reg + "], " +
                 cmp.reg + ", " + val.reg + ";");
        } else {
            emit("atom.global." + ptxOp + "." + atomSuf + " " + oldv + ", [" + addr.reg +
                 "], " + val.reg + ";");
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
        // Define retReg up front: if the body falls through without a `return`
        // (a missing return on some path — UB in C++), the caller still reads a
        // defined 0 rather than an uninitialized register.
        if (rt.base != Type::Void) {
            std::string z = rt.isFloating() ? (rt.base == Type::Double ? f64imm(0.0) : f32imm(0.0)) : "0";
            emit(std::string(movFor(rt)) + retReg + ", " + z + ";");
        }
        std::string endL = label();
        inlineCtx_.push_back({retReg, endL, rt});
        inlining_.insert(fn.name);

        // The inlined body gets its own variable scope AND its own array/struct
        // symbol tables, so its locals never collide with the caller's (or with a
        // second inlining of the same helper). Local/shared array PTX symbols are
        // uniquified at declaration (uniqueSym), so the .local/.shared decls differ.
        std::unordered_map<std::string, Val> savedVars;
        savedVars.swap(vars);
        vars = std::move(inlineScope);
        auto savedDims = std::move(arrayDims_); arrayDims_.clear();
        auto savedStructs = std::move(localStructMembers_); localStructMembers_.clear();
        for (const auto& st : fn.body) { emitStmt(*st); if (failed) break; }
        emitLabel(endL);
        vars = std::move(savedVars);
        arrayDims_ = std::move(savedDims);
        localStructMembers_ = std::move(savedStructs);

        inlining_.erase(fn.name);
        inlineCtx_.pop_back();
        if (failed) return {};
        if (rt.base == Type::Void) return {};
        return {retReg, rt};
    }

    // Map a unary float-math intrinsic name to its canonical op + precision.
    // The `f`-suffixed spelling is f32; the bare C name is f64 (double).
    // `__expf`/`__logf` are the f32 fast variants.
    static bool recognizeUnaryMath(const std::string& fn, std::string& canon, bool& dbl) {
        if (fn == "__expf") { canon = "exp"; dbl = false; return true; }
        if (fn == "__logf") { canon = "log"; dbl = false; return true; }
        dbl = fn.empty() || fn.back() != 'f';
        canon = dbl ? fn : fn.substr(0, fn.size() - 1);   // strip trailing 'f' for f32
        return canon == "sqrt" || canon == "fabs" || canon == "rsqrt" ||
               canon == "sin" || canon == "cos" || canon == "exp" || canon == "log" ||
               canon == "exp2" || canon == "log2" || canon == "tanh";
    }

    // Recognize an explicit-rounding arithmetic intrinsic:
    //   __f{add,sub,mul,div}_{rn,rz,ru,rd}  (f32)   __d{add,sub,mul,div}_{...} (f64)
    //   __fmaf_{...}/__fma_{...}                     __f/drcp_{...}, __f/dsqrt_{...}
    //   __frsqrt_rn
    // Fills the PTX op, precision, arity, and rounding mode. These name the
    // rounding a plain operator leaves to the compiler; we lower to the
    // rounding-tagged PTX op (rcp/sqrt/div carry a mode; rsqrt is .approx).
    static bool recognizeRoundedArith(const std::string& fn, std::string& op,
                                      bool& dbl, int& arity, std::string& mode) {
        if (fn.size() < 8 || fn.compare(0, 2, "__") != 0) return false;
        const size_t us = fn.rfind('_');
        if (us == std::string::npos || us + 3 != fn.size()) return false;   // mode is 2 chars, trailing
        mode = fn.substr(us + 1);
        if (mode != "rn" && mode != "rz" && mode != "ru" && mode != "rd") return false;
        const std::string b = fn.substr(2, us - 2);   // between "__" and "_<mode>"
        struct E { const char* name; const char* op; bool dbl; int arity; };
        static const E tbl[] = {
            {"fadd","add",false,2},{"fsub","sub",false,2},{"fmul","mul",false,2},{"fdiv","div",false,2},
            {"dadd","add",true, 2},{"dsub","sub",true, 2},{"dmul","mul",true, 2},{"ddiv","div",true, 2},
            {"fmaf","fma",false,3},{"fma","fma",true,3},
            {"frcp","rcp",false,1},{"drcp","rcp",true,1},
            {"fsqrt","sqrt",false,1},{"dsqrt","sqrt",true,1},
            {"frsqrt","rsqrt",false,1},
        };
        for (const E& e : tbl) if (b == e.name) { op = e.op; dbl = e.dbl; arity = e.arity; return true; }
        return false;
    }

    // Warp shuffle: __shfl[_up|_down|_xor]_sync(mask, var, lane [, width]).
    // Lowers to `shfl.sync.<mode>.b32 d, var, lane, <width>, mask` — the c operand
    // carries the subwarp width (our interpreter's encoding). 32-bit values shuffle
    // directly; 64-bit values (double / long) split into two .b32 shuffles.
    Val emitShfl(const Expr& e) {
        const std::string& fn = e.str;
        if (e.args.size() < 3 || e.args.size() > 4) { fail("'" + fn + "' expects (mask, var, lane[, width])"); return {}; }
        Val mask = coerce(emitExpr(*e.args[0]), intType()); if (failed) return {};
        Val var  = emitExpr(*e.args[1]);                    if (failed) return {};
        Val lane = coerce(emitExpr(*e.args[2]), intType()); if (failed) return {};
        int width = 32;
        if (e.args.size() == 4) {
            if (e.args[3]->kind != Expr::IntLit) { fail("shuffle width must be a constant"); return {}; }
            width = static_cast<int>(e.args[3]->ival);
            if (width <= 0 || width > 32 || (width & (width - 1))) { fail("shuffle width must be a power of two in [1,32]"); return {}; }
        }
        const char* mode = fn == "__shfl_sync" ? "idx" : fn == "__shfl_up_sync" ? "up" :
                           fn == "__shfl_down_sync" ? "down" : "bfly";
        const std::string wc = std::to_string(width);
        auto shfl32 = [&](const std::string& dst, const std::string& srcReg) {
            emit("shfl.sync." + std::string(mode) + ".b32 " + dst + ", " + srcReg + ", " +
                 lane.reg + ", " + wc + ", " + mask.reg + ";");
        };

        if (!is64BitScalar(var.type)) {                     // 32-bit: shuffle directly
            std::string d = fresh(classOf(var.type));
            shfl32(d, var.reg);
            return {d, var.type};
        }

        // 64-bit shuffle: shfl is a .b32 operation on real hardware, so shuffle the
        // low and high 32-bit words separately (same mask/lane/width) and recombine.
        // The result is bit-exact for any 64-bit type; a double is moved through an
        // integer register (mov.b64 is a bit reinterpret) so the two halves are its
        // raw IEEE-754 bits.
        const bool isF64 = (var.type.base == Type::Double);
        std::string bits = var.reg;
        if (isF64) { bits = fresh(RC::RD64); emit("mov.b64 " + bits + ", " + var.reg + ";"); }

        std::string lo = fresh(RC::R32), hitmp = fresh(RC::RD64), hi = fresh(RC::R32);
        emit("cvt.u32.u64 " + lo + ", " + bits + ";");           // low 32 bits (truncate)
        emit("shr.u64 " + hitmp + ", " + bits + ", 32;");
        emit("cvt.u32.u64 " + hi + ", " + hitmp + ";");          // high 32 bits

        std::string los = fresh(RC::R32), his = fresh(RC::R32);
        shfl32(los, lo);
        shfl32(his, hi);

        std::string lo64 = fresh(RC::RD64), hi64 = fresh(RC::RD64),
                    hish = fresh(RC::RD64), res = fresh(RC::RD64);
        emit("cvt.u64.u32 " + lo64 + ", " + los + ";");          // zero-extend
        emit("cvt.u64.u32 " + hi64 + ", " + his + ";");
        emit("shl.b64 " + hish + ", " + hi64 + ", 32;");
        emit("or.b64 " + res + ", " + lo64 + ", " + hish + ";");

        if (isF64) { std::string fres = fresh(RC::F64); emit("mov.b64 " + fres + ", " + res + ";"); return {fres, var.type}; }
        return {res, var.type};
    }

    // Warp vote: __ballot_sync / __any_sync / __all_sync(mask, predicate).
    // The predicate (any int expr) becomes a real .pred via `setp.ne`, then:
    //   ballot → vote.sync.ballot.b32 (32-bit lane mask, returned as int)
    //   any/all → vote.sync.{any,all}.pred, then selp to an int 0/1.
    // Warp-cooperative, so (like shuffle) these run on the Tier-0 interpreter.
    Val emitVote(const Expr& e) {
        const std::string& fn = e.str;
        if (e.args.size() != 2) { fail("'" + fn + "' expects (mask, predicate)"); return {}; }
        Val mask = coerce(emitExpr(*e.args[0]), intType()); if (failed) return {};
        Val pv   = coerce(emitExpr(*e.args[1]), intType()); if (failed) return {};
        std::string p = fresh(RC::Pred);
        emit("setp.ne.s32 " + p + ", " + pv.reg + ", 0;");
        if (fn == "__ballot_sync") {
            std::string d = fresh(RC::R32);
            emit("vote.sync.ballot.b32 " + d + ", " + p + ", " + mask.reg + ";");
            return {d, intType()};                       // 32-bit lane mask
        }
        const char* vop = (fn == "__all_sync") ? "all" : "any";
        std::string pd = fresh(RC::Pred);
        emit("vote.sync." + std::string(vop) + ".pred " + pd + ", " + p + ", " + mask.reg + ";");
        std::string d = fresh(RC::R32);
        emit("selp.b32 " + d + ", 1, 0, " + pd + ";");   // pred → int 0/1
        return {d, intType()};
    }

    // __half <-> float helpers (used by the half arithmetic intrinsics, which all
    // promote to float, compute, then narrow back to __half).
    Val h2f(const Val& h) { std::string d = fresh(RC::F32); emit("cvt.f32.f16 " + d + ", " + h.reg + ";"); return {d, floatType()}; }
    Val f2h(const Val& f) { std::string d = fresh(RC::R32); emit("cvt.rn.f16.f32 " + d + ", " + f.reg + ";"); return {d, halfType()}; }

    // __ldg / cache-hinted loads (__ldca/__ldcs/__ldcg/__ldlu/__ldcv/__ldg_nc):
    // read-only / cache-hinted global loads. A CPU interpreter has no cache
    // hierarchy, so the hint is a no-op and each is a plain load of the pointed-to
    // element. Common forms: __ldg(&a[i]) and __ldg(p) (p a pointer variable).
    Val emitCachedLoad(const Expr& e) {
        const std::string& fn = e.str;
        if (e.args.size() != 1) { fail("'" + fn + "' expects one pointer argument"); return {}; }
        const Expr& arg = *e.args[0];
        if (arg.kind == Expr::Unary && arg.str == "&" && !arg.args.empty())
            return emitLoad(*arg.args[0]);            // __ldg(&a[i]) → load a[i]
        Val p = emitExpr(arg);                        // __ldg(p): p is a pointer
        if (failed) return {};
        if (!p.type.isPointer()) { fail("'" + fn + "' requires a pointer argument"); return {}; }
        Type pointee = p.type; pointee.ptr -= 1;
        std::string d = fresh(classOf(pointee));
        emit("ld." + std::string(p.space == Space::Shared ? "shared." : "global.") +
             ldSuffix(pointee) + " " + d + ", [" + p.reg + "];");
        return {d, pointee};
    }

    // Cache-hinted stores (__stwb/__stcg/__stcs/__stwt): plain stores here (the
    // write-back/streaming hints have no meaning without a cache). __stcg(&a[i], v).
    Val emitCachedStore(const Expr& e) {
        const std::string& fn = e.str;
        if (e.args.size() != 2) { fail("'" + fn + "' expects (pointer, value)"); return {}; }
        const Expr& ptr = *e.args[0];
        if (ptr.kind == Expr::Unary && ptr.str == "&" && !ptr.args.empty()) {
            const Expr& idx = *ptr.args[0];
            Val v = coerce(emitExpr(*e.args[1]), indexPointee(idx));  // __stcg(&a[i], v)
            if (failed) return {};
            emitStore(idx, v);
            return {};
        }
        Val p = emitExpr(ptr);
        if (failed) return {};
        if (!p.type.isPointer()) { fail("'" + fn + "' requires a pointer argument"); return {}; }
        Type pointee = p.type; pointee.ptr -= 1;
        Val v = coerce(emitExpr(*e.args[1]), pointee);
        if (failed) return {};
        emit("st." + std::string(p.space == Space::Shared ? "shared." : "global.") +
             stSuffix(pointee) + " [" + p.reg + "], " + v.reg + ";");
        return {};
    }

    Val emitCall(const Expr& e) {
        const std::string& fn = e.str;
        // ── Cache-hinted global loads/stores: plain load/store on the CPU ──
        if (fn == "__ldg" || fn == "__ldca" || fn == "__ldcs" || fn == "__ldcg" ||
            fn == "__ldlu" || fn == "__ldcv" || fn == "__ldg_nc")
            return emitCachedLoad(e);
        if (fn == "__stwb" || fn == "__stcg" || fn == "__stcs" || fn == "__stwt")
            return emitCachedStore(e);
        // ── Explicit IEEE-rounding arithmetic: __fadd_rn/__fmaf_rn/__fdiv_rz/… ──
        {
            std::string rop, rmode; bool rdbl = false; int rarity = 0;
            if (recognizeRoundedArith(fn, rop, rdbl, rarity, rmode)) {
                if ((int)e.args.size() != rarity) {
                    fail("'" + fn + "' expects " + std::to_string(rarity) + " argument(s)"); return {};
                }
                const Type ft = rdbl ? doubleType() : floatType();
                const std::string suf = rdbl ? "f64" : "f32";
                Val a = coerce(emitExpr(*e.args[0]), ft); if (failed) return {};
                std::string d = fresh(classOf(ft));
                if (rarity == 1) {                       // rcp / sqrt / rsqrt
                    if (rop == "rsqrt") emit("rsqrt.approx." + suf + " " + d + ", " + a.reg + ";");
                    else emit(rop + "." + rmode + "." + suf + " " + d + ", " + a.reg + ";");
                    return {d, ft};
                }
                Val b = coerce(emitExpr(*e.args[1]), ft); if (failed) return {};
                if (rarity == 2) {                       // add / sub / mul / div
                    emit(rop + "." + rmode + "." + suf + " " + d + ", " + a.reg + ", " + b.reg + ";");
                    return {d, ft};
                }
                Val c = coerce(emitExpr(*e.args[2]), ft); if (failed) return {};   // fma
                emit("fma." + rmode + "." + suf + " " + d + ", " + a.reg + ", " + b.reg + ", " + c.reg + ";");
                return {d, ft};
            }
        }
        // ── Half arithmetic: compute in float, result __half (or bool for cmp) ──
        if (e.args.size() == 2 &&
            (fn == "__hadd" || fn == "__hsub" || fn == "__hmul" || fn == "__hdiv" ||
             fn == "__hmax" || fn == "__hmin")) {
            Val a = h2f(emitExpr(*e.args[0])); if (failed) return {};
            Val b = h2f(emitExpr(*e.args[1])); if (failed) return {};
            const std::string op = fn.substr(3);          // __h(add/sub/mul/div/max/min)
            std::string rf = fresh(RC::F32);
            emit((op == "div" ? "div.rn.f32 " : op + ".f32 ") + rf + ", " + a.reg + ", " + b.reg + ";");
            return f2h({rf, floatType()});
        }
        if (e.args.size() == 2 &&
            (fn == "__heq" || fn == "__hne" || fn == "__hlt" || fn == "__hle" ||
             fn == "__hgt" || fn == "__hge")) {
            Val a = h2f(emitExpr(*e.args[0])); if (failed) return {};
            Val b = h2f(emitExpr(*e.args[1])); if (failed) return {};
            std::string p = fresh(RC::Pred), d = fresh(RC::R32);
            emit("setp." + fn.substr(3) + ".f32 " + p + ", " + a.reg + ", " + b.reg + ";");
            emit("selp.b32 " + d + ", 1, 0, " + p + ";");   // __half compare → int 0/1
            return {d, intType()};
        }
        if (e.args.size() == 3 && fn == "__hfma") {
            Val a = h2f(emitExpr(*e.args[0])); if (failed) return {};
            Val b = h2f(emitExpr(*e.args[1])); if (failed) return {};
            Val c = h2f(emitExpr(*e.args[2])); if (failed) return {};
            std::string rf = fresh(RC::F32);
            emit("fma.rn.f32 " + rf + ", " + a.reg + ", " + b.reg + ", " + c.reg + ";");
            return f2h({rf, floatType()});
        }
        if (e.args.size() == 1 && fn == "__hneg") {
            Val a = h2f(emitExpr(*e.args[0])); if (failed) return {};
            std::string rf = fresh(RC::F32);
            emit("neg.f32 " + rf + ", " + a.reg + ";");
            return f2h({rf, floatType()});
        }
        // ── Half unary math: promote to float, apply the f32 op, narrow to __half ──
        // These mirror the f32 math intrinsics (same PTX the float family emits), so
        // the result is the fp16-rounded value of the correctly computed float. All
        // run on the Tier-0 interpreter with no LLVM.
        if (e.args.size() == 1 &&
            (fn == "hsqrt" || fn == "hrsqrt" || fn == "hrcp" || fn == "__habs" ||
             fn == "hceil" || fn == "hfloor" || fn == "htrunc" || fn == "hrint" ||
             fn == "hexp" || fn == "hexp2" || fn == "hexp10" ||
             fn == "hlog" || fn == "hlog2" || fn == "hlog10" ||
             fn == "hsin" || fn == "hcos")) {
            Val a = h2f(emitExpr(*e.args[0])); if (failed) return {};
            std::string rf = fresh(RC::F32);
            if (fn == "hsqrt")       emit("sqrt.rn.f32 " + rf + ", " + a.reg + ";");
            else if (fn == "hrsqrt") emit("rsqrt.approx.f32 " + rf + ", " + a.reg + ";");
            else if (fn == "hrcp")   emit("rcp.approx.f32 " + rf + ", " + a.reg + ";");
            else if (fn == "__habs") emit("abs.f32 " + rf + ", " + a.reg + ";");
            else if (fn == "hceil")  emit("cvt.rpi.f32.f32 " + rf + ", " + a.reg + ";");
            else if (fn == "hfloor") emit("cvt.rmi.f32.f32 " + rf + ", " + a.reg + ";");
            else if (fn == "htrunc") emit("cvt.rzi.f32.f32 " + rf + ", " + a.reg + ";");
            else if (fn == "hrint")  emit("cvt.rni.f32.f32 " + rf + ", " + a.reg + ";");
            else if (fn == "hexp2")  emit("ex2.approx.f32 " + rf + ", " + a.reg + ";");
            else if (fn == "hlog2")  emit("lg2.approx.f32 " + rf + ", " + a.reg + ";");
            else if (fn == "hsin")   emit("sin.approx.f32 " + rf + ", " + a.reg + ";");
            else if (fn == "hcos")   emit("cos.approx.f32 " + rf + ", " + a.reg + ";");
            else if (fn == "hexp" || fn == "hexp10") {      // a^x = 2^(x*log2 a)
                std::string t = fresh(RC::F32);
                emit("mul.f32 " + t + ", " + a.reg + ", " +
                     f32imm(fn == "hexp" ? 1.4426950408889634 /*log2 e*/
                                         : 3.3219280948873623 /*log2 10*/) + ";");
                emit("ex2.approx.f32 " + rf + ", " + t + ";");
            } else {                                        // hlog/hlog10: log2(x)*k
                std::string t = fresh(RC::F32);
                emit("lg2.approx.f32 " + t + ", " + a.reg + ";");
                emit("mul.f32 " + rf + ", " + t + ", " +
                     f32imm(fn == "hlog" ? 0.6931471805599453 /*ln 2*/
                                         : 0.3010299956639812 /*log10 2*/) + ";");
            }
            return f2h({rf, floatType()});
        }
        if (fn == "__syncthreads" && e.args.empty()) { emit("bar.sync 0;"); return {}; }
        // Memory fences: order this thread's memory ops at block / device / system
        // scope. Lowered to real PTX membar; the interpreter issues an actual CPU
        // barrier (device/system scope orders global memory across the concurrently
        // scheduled CTAs), so producer/consumer and grid-reduction kernels are
        // correct, not just accepted.
        if (fn == "__threadfence_block"  && e.args.empty()) { emit("membar.cta;"); return {}; }
        if (fn == "__threadfence"        && e.args.empty()) { emit("membar.gl;");  return {}; }
        if (fn == "__threadfence_system" && e.args.empty()) { emit("membar.sys;"); return {}; }
        // Barrier + block-wide predicate reduction: __syncthreads_count returns the
        // number of threads with a nonzero predicate; __syncthreads_and/_or return
        // nonzero iff the predicate holds for all / any thread. bar.red rendezvous
        // the whole CTA, so these run on the Tier-0 interpreter.
        if (e.args.size() == 1 &&
            (fn == "__syncthreads_count" || fn == "__syncthreads_and" || fn == "__syncthreads_or")) {
            Val pv = coerce(emitExpr(*e.args[0]), intType()); if (failed) return {};
            std::string p = fresh(RC::Pred);
            emit("setp.ne.s32 " + p + ", " + pv.reg + ", 0;");
            if (fn == "__syncthreads_count") {
                std::string d = fresh(RC::R32);
                emit("bar.red.popc.u32 " + d + ", 0, " + p + ";");   // count of set predicates
                return {d, intType()};
            }
            const char* rop = (fn == "__syncthreads_and") ? "and" : "or";
            std::string pd = fresh(RC::Pred), d = fresh(RC::R32);
            emit("bar.red." + std::string(rop) + ".pred " + pd + ", 0, " + p + ";");
            emit("selp.b32 " + d + ", 1, 0, " + pd + ";");           // block-vote → int 0/1
            return {d, intType()};
        }
        if (fn == "__syncwarp" && e.args.size() <= 1) {   // warp barrier (default: full mask)
            std::string mask = "0xffffffff";
            if (e.args.size() == 1) { Val m = coerce(emitExpr(*e.args[0]), intType()); if (failed) return {}; mask = m.reg; }
            emit("bar.warp.sync " + mask + ";");
            return {};
        }
        if (fn == "__activemask" && e.args.empty()) {     // bitmask of active warp lanes
            std::string d = fresh(RC::R32);
            emit("activemask.b32 " + d + ";");
            return {d, intType()};
        }
        if (fn == "atomicAdd"  && e.args.size() == 2) return emitAtomic(e, "add");
        if (fn == "atomicSub"  && e.args.size() == 2) return emitAtomic(e, "sub");
        if (fn == "atomicMin"  && e.args.size() == 2) return emitAtomic(e, "min");
        if (fn == "atomicMax"  && e.args.size() == 2) return emitAtomic(e, "max");
        if (fn == "atomicExch" && e.args.size() == 2) return emitAtomic(e, "exch");
        if (fn == "atomicAnd"  && e.args.size() == 2) return emitAtomic(e, "and");
        if (fn == "atomicOr"   && e.args.size() == 2) return emitAtomic(e, "or");
        if (fn == "atomicXor"  && e.args.size() == 2) return emitAtomic(e, "xor");
        if (fn == "atomicCAS"  && e.args.size() == 3) return emitAtomic(e, "cas");
        if (fn == "__shfl_sync" || fn == "__shfl_up_sync" ||
            fn == "__shfl_down_sync" || fn == "__shfl_xor_sync") return emitShfl(e);
        if (fn == "__ballot_sync" || fn == "__any_sync" || fn == "__all_sync") return emitVote(e);
        if (deviceFns_) {
            auto it = deviceFns_->find(fn);
            if (it != deviceFns_->end()) return emitInlineDeviceCall(*it->second, e);
        }

        // Unary intrinsics.
        if (e.args.size() == 1) {
            if (fn == "abs") {                          // integer or float abs (width-preserving)
                Val a = emitExpr(*e.args[0]); if (failed) return {};
                std::string d = fresh(classOf(a.type));
                emit("abs." + (a.type.isFloating() ? floatSuffix(a.type) : intSuffix(a.type)) +
                     " " + d + ", " + a.reg + ";");
                return {d, a.type};
            }
            // Bit-count intrinsics — population count / count-leading-zeros. The
            // `ll` forms take a 64-bit operand; all return a 32-bit int (as in CUDA).
            // __popc(__ballot_sync(...)) is the canonical active-lane count.
            if (fn == "__popc" || fn == "__popcll" || fn == "__clz" || fn == "__clzll") {
                const bool ll = (fn == "__popcll" || fn == "__clzll");
                const std::string op = (fn == "__popc" || fn == "__popcll") ? "popc" : "clz";
                Val a = coerce(emitExpr(*e.args[0]), ll ? longType() : intType()); if (failed) return {};
                std::string d = fresh(RC::R32);
                emit(op + (ll ? ".b64 " : ".b32 ") + d + ", " + a.reg + ";");
                return {d, intType()};
            }
            // Bit reversal — width-preserving (__brev: unsigned; __brevll: 64-bit).
            if (fn == "__brev" || fn == "__brevll") {
                const bool ll = (fn == "__brevll");
                const Type rt = ll ? longType() : intType();
                Val a = coerce(emitExpr(*e.args[0]), rt); if (failed) return {};
                std::string d = fresh(classOf(rt));
                emit(std::string("brev.") + (ll ? "b64 " : "b32 ") + d + ", " + a.reg + ";");
                return {d, rt};
            }
            // Find-first-set: 1-indexed position of the least-significant set bit, or
            // 0 if the operand is 0. Composed as clz(brev(x)) + 1, guarded for x == 0.
            if (fn == "__ffs" || fn == "__ffsll") {
                const bool ll = (fn == "__ffsll");
                Val a = coerce(emitExpr(*e.args[0]), ll ? longType() : intType()); if (failed) return {};
                std::string rev = fresh(ll ? RC::RD64 : RC::R32);
                emit(std::string("brev.") + (ll ? "b64 " : "b32 ") + rev + ", " + a.reg + ";");
                std::string clz = fresh(RC::R32);
                emit(std::string("clz.") + (ll ? "b64 " : "b32 ") + clz + ", " + rev + ";");
                std::string tmp = fresh(RC::R32);
                emit("add.s32 " + tmp + ", " + clz + ", 1;");
                std::string p = fresh(RC::Pred);
                emit(std::string("setp.eq.") + (ll ? "s64 " : "s32 ") + p + ", " + a.reg + ", 0;");
                std::string d = fresh(RC::R32);
                emit("selp.b32 " + d + ", 0, " + tmp + ", " + p + ";");   // x==0 ? 0 : clz+1
                return {d, intType()};
            }
            // Type-punning reinterprets: copy the raw bits between a float and an
            // integer register of the same width (mov.b{32,64} is a bit cast). Used
            // for fast-math bit tricks and float atomics via CAS.
            if (fn == "__float_as_int" || fn == "__float_as_uint") {
                Val a = coerce(emitExpr(*e.args[0]), floatType()); if (failed) return {};
                std::string d = fresh(RC::R32);
                emit("mov.b32 " + d + ", " + a.reg + ";");
                return {d, intType()};
            }
            if (fn == "__int_as_float" || fn == "__uint_as_float") {
                Val a = coerce(emitExpr(*e.args[0]), intType()); if (failed) return {};
                std::string d = fresh(RC::F32);
                emit("mov.b32 " + d + ", " + a.reg + ";");
                return {d, floatType()};
            }
            if (fn == "__double_as_longlong") {
                Val a = coerce(emitExpr(*e.args[0]), doubleType()); if (failed) return {};
                std::string d = fresh(RC::RD64);
                emit("mov.b64 " + d + ", " + a.reg + ";");
                return {d, longType()};
            }
            if (fn == "__longlong_as_double") {
                Val a = coerce(emitExpr(*e.args[0]), longType()); if (failed) return {};
                std::string d = fresh(RC::F64);
                emit("mov.b64 " + d + ", " + a.reg + ";");
                return {d, doubleType()};
            }
            // Half precision (__half). Values are the raw 16-bit f16 bit pattern,
            // carried in a 32-bit register; only conversions to/from float are
            // provided (arithmetic is done in float via __half2float / __float2half).
            if (fn == "__float2half" || fn == "__float2half_rn") {
                Val a = coerce(emitExpr(*e.args[0]), floatType()); if (failed) return {};
                std::string d = fresh(RC::R32);
                emit("cvt.rn.f16.f32 " + d + ", " + a.reg + ";");
                return {d, halfType()};
            }
            if (fn == "__half2float") {
                Val a = emitExpr(*e.args[0]); if (failed) return {};   // operand is __half
                std::string d = fresh(RC::F32);
                emit("cvt.f32.f16 " + d + ", " + a.reg + ";");
                return {d, floatType()};
            }
            // Rounding-mode conversions f32 -> int32 (rn=nearest, rz=toward-zero,
            // ru=+inf, rd=-inf) — these VALUE conversions differ from the bit
            // reinterprets above.
            if (fn == "__float2int_rn"  || fn == "__float2int_rz"  ||
                fn == "__float2int_ru"  || fn == "__float2int_rd"  ||
                fn == "__float2uint_rn" || fn == "__float2uint_rz" ||
                fn == "__float2uint_ru" || fn == "__float2uint_rd") {
                const bool uns = (fn.find("uint") != std::string::npos);
                const char m = fn.back();       // n / z / u / d
                const std::string rnd = (m == 'n') ? "rni" : (m == 'z') ? "rzi"
                                      : (m == 'u') ? "rpi" : "rmi";
                Val a = coerce(emitExpr(*e.args[0]), floatType()); if (failed) return {};
                std::string d = fresh(RC::R32);
                emit("cvt." + rnd + "." + std::string(uns ? "u32" : "s32") + ".f32 " + d + ", " + a.reg + ";");
                return {d, intType()};
            }
            // int32/uint32 -> f32 (round to nearest even).
            if (fn == "__int2float_rn" || fn == "__uint2float_rn") {
                const bool uns = (fn.find("uint") != std::string::npos);
                Val a = coerce(emitExpr(*e.args[0]), intType()); if (failed) return {};
                std::string d = fresh(RC::F32);
                emit("cvt.rn.f32." + std::string(uns ? "u32" : "s32") + " " + d + ", " + a.reg + ";");
                return {d, floatType()};
            }
            // Round-to-integer-in-float. cvt rounding modes: .rmi=floor, .rpi=ceil,
            // .rzi=trunc (toward zero), .rni=rint/nearbyint (nearest, ties to even).
            // (CUDA's round() — ties away from zero — is intentionally NOT mapped
            // here; it differs from rni and would need a separate lowering.)
            if (fn == "floorf" || fn == "floor" || fn == "ceilf" || fn == "ceil" ||
                fn == "truncf" || fn == "trunc" || fn == "rintf" || fn == "rint" ||
                fn == "nearbyintf" || fn == "nearbyint") {
                const bool dbl = (fn.back() != 'f');     // f32 spellings end in 'f'
                const std::string suf = dbl ? "f64" : "f32";
                const Type ft = dbl ? doubleType() : floatType();
                Val a = coerce(emitExpr(*e.args[0]), ft); if (failed) return {};
                std::string d = fresh(classOf(ft));
                std::string rnd;
                if (fn == "floorf" || fn == "floor")      rnd = "rmi";
                else if (fn == "ceilf" || fn == "ceil")   rnd = "rpi";
                else if (fn == "truncf" || fn == "trunc") rnd = "rzi";
                else                                      rnd = "rni";  // rint/nearbyint
                emit("cvt." + rnd + "." + suf + "." + suf + " " + d + ", " + a.reg + ";");
                return {d, ft};
            }
            if (fn == "__saturatef") {                  // clamp to [0, 1] (f32 only)
                Val a = coerce(emitExpr(*e.args[0]), floatType()); if (failed) return {};
                std::string t = fresh(RC::F32), d = fresh(RC::F32);
                emit("max.f32 " + t + ", " + a.reg + ", 0f00000000;");   // max(x, 0.0f)
                emit("min.f32 " + d + ", " + t + ", 0f3F800000;");       // min(_, 1.0f)
                return {d, floatType()};
            }
            // Composite unary math built from ex2/lg2 (both tiers support these):
            //   exp10, log10, sinh, cosh, expm1, log1p — f32 (`…f`) and f64 (bare C).
            {
                const bool cdbl = fn.empty() || fn.back() != 'f';
                const std::string cc = cdbl ? fn : fn.substr(0, fn.size() - 1);
                static const std::set<std::string> comp =
                    {"exp10", "log10", "sinh", "cosh", "expm1", "log1p"};
                if (comp.count(cc)) {
                    const Type ft = cdbl ? doubleType() : floatType();
                    const std::string suf = cdbl ? "f64" : "f32";
                    auto lit = [&](double v) { return cdbl ? f64imm(v) : f32imm(v); };
                    Val a = coerce(emitExpr(*e.args[0]), ft); if (failed) return {};
                    auto expOf = [&](const std::string& x) {                 // e^x = 2^(x*log2 e)
                        std::string t = fresh(classOf(ft)), d = fresh(classOf(ft));
                        emit("mul." + suf + " " + t + ", " + x + ", " + lit(1.4426950408889634) + ";");
                        emit("ex2.approx." + suf + " " + d + ", " + t + ";");
                        return d;
                    };
                    std::string d = fresh(classOf(ft));
                    if (cc == "exp10") {                                      // 10^x = 2^(x*log2 10)
                        std::string t = fresh(classOf(ft));
                        emit("mul." + suf + " " + t + ", " + a.reg + ", " + lit(3.3219280948873623) + ";");
                        emit("ex2.approx." + suf + " " + d + ", " + t + ";");
                    } else if (cc == "log10") {                              // log2(x)*log10(2)
                        std::string t = fresh(classOf(ft));
                        emit("lg2.approx." + suf + " " + t + ", " + a.reg + ";");
                        emit("mul." + suf + " " + d + ", " + t + ", " + lit(0.3010299956639812) + ";");
                    } else if (cc == "expm1") {                              // e^x - 1
                        std::string e = expOf(a.reg);
                        emit("sub." + suf + " " + d + ", " + e + ", " + lit(1.0) + ";");
                    } else if (cc == "log1p") {                              // log(1+x) = lg2(1+x)*ln2
                        std::string opx = fresh(classOf(ft)), lg = fresh(classOf(ft));
                        emit("add." + suf + " " + opx + ", " + a.reg + ", " + lit(1.0) + ";");
                        emit("lg2.approx." + suf + " " + lg + ", " + opx + ";");
                        emit("mul." + suf + " " + d + ", " + lg + ", " + lit(0.6931471805599453) + ";");
                    } else {                                                 // sinh/cosh = (e^x ± e^-x)/2
                        std::string nx = fresh(classOf(ft));
                        emit("neg." + suf + " " + nx + ", " + a.reg + ";");
                        std::string ep = expOf(a.reg), en = expOf(nx), s = fresh(classOf(ft));
                        emit(std::string(cc == "sinh" ? "sub." : "add.") + suf + " " + s + ", " + ep + ", " + en + ";");
                        emit("mul." + suf + " " + d + ", " + s + ", " + lit(0.5) + ";");
                    }
                    return {d, ft};
                }
            }

            // Interpreter-tier unary math with no PTX approx op — inverse trig,
            // cbrt, erf. Emitted as `<op>.approx.<ty>`; the Tier-0 interpreter
            // computes them via libm (so these are interpreter-tier only).
            {
                const bool idbl = fn.empty() || fn.back() != 'f';
                const std::string ic = idbl ? fn : fn.substr(0, fn.size() - 1);
                static const std::set<std::string> itr = {"atan", "asin", "acos", "cbrt", "erf"};
                if (itr.count(ic)) {
                    const Type ft = idbl ? doubleType() : floatType();
                    Val a = coerce(emitExpr(*e.args[0]), ft); if (failed) return {};
                    std::string d = fresh(classOf(ft));
                    emit(ic + ".approx." + std::string(idbl ? "f64" : "f32") + " " + d + ", " + a.reg + ";");
                    return {d, ft};
                }
            }

            // Float math intrinsic: the `f`-suffixed name is f32, the bare C name
            // is f64 (double). `__expf`/`__logf` are the f32 fast variants.
            std::string canon; bool dbl;
            if (!recognizeUnaryMath(fn, canon, dbl)) { fail("unsupported call to '" + fn + "'"); return {}; }
            const Type ft = dbl ? doubleType() : floatType();
            const std::string suf = dbl ? "f64" : "f32";
            Val a = coerce(emitExpr(*e.args[0]), ft);
            if (failed) return {};
            std::string d = fresh(classOf(ft));
            if (canon == "sqrt")  { emit("sqrt.rn." + suf + " " + d + ", " + a.reg + ";"); return {d, ft}; }
            if (canon == "fabs")  { emit("abs." + suf + " " + d + ", " + a.reg + ";"); return {d, ft}; }
            if (canon == "rsqrt") { emit("rsqrt.approx." + suf + " " + d + ", " + a.reg + ";"); return {d, ft}; }
            if (canon == "sin")   { emit("sin.approx." + suf + " " + d + ", " + a.reg + ";"); return {d, ft}; }
            if (canon == "cos")   { emit("cos.approx." + suf + " " + d + ", " + a.reg + ";"); return {d, ft}; }
            if (canon == "exp2")  { emit("ex2.approx." + suf + " " + d + ", " + a.reg + ";"); return {d, ft}; }
            if (canon == "log2")  { emit("lg2.approx." + suf + " " + d + ", " + a.reg + ";"); return {d, ft}; }
            if (canon == "tanh")  { emit("tanh.approx." + suf + " " + d + ", " + a.reg + ";"); return {d, ft}; }
            if (canon == "exp") {                       // e^x = 2^(x*log2 e)
                std::string t = fresh(classOf(ft));
                std::string log2e = dbl ? f64imm(1.4426950408889634) : "0f3FB8AA3B";
                emit("mul." + suf + " " + t + ", " + a.reg + ", " + log2e + ";");
                emit("ex2.approx." + suf + " " + d + ", " + t + ";");
                return {d, ft};
            }
            if (canon == "log") {                       // ln x = log2(x)*ln 2
                std::string t = fresh(classOf(ft));
                std::string ln2 = dbl ? f64imm(0.6931471805599453) : "0f3F317218";
                emit("lg2.approx." + suf + " " + t + ", " + a.reg + ";");
                emit("mul." + suf + " " + d + ", " + t + ", " + ln2 + ";");
                return {d, ft};
            }
            fail("unsupported call to '" + fn + "'");
            return {};
        }

        // Binary intrinsics: min/max (int or float).
        if (e.args.size() == 2) {
            Val a = emitExpr(*e.args[0]); if (failed) return {};
            Val b = emitExpr(*e.args[1]); if (failed) return {};
            if (fn == "fminf" || fn == "fmaxf" || fn == "fmin" || fn == "fmax") {
                const bool dbl = (fn == "fmin" || fn == "fmax");
                const bool isMin = (fn == "fminf" || fn == "fmin");
                const Type ft = dbl ? doubleType() : floatType();
                const std::string suf = dbl ? "f64" : "f32";
                a = coerce(a, ft); b = coerce(b, ft);
                std::string d = fresh(classOf(ft));
                emit(std::string(isMin ? "min." : "max.") + suf + " " + d + ", " + a.reg + ", " + b.reg + ";");
                return {d, ft};
            }
            if (fn == "min" || fn == "max") {           // int or float, width-preserving
                const bool fp = a.type.isFloating() || b.type.isFloating();
                Type ct = promote(a.type, b.type);
                a = coerce(a, ct); b = coerce(b, ct);
                std::string d = fresh(classOf(ct));
                emit(std::string(fn == "min" ? "min." : "max.") + (fp ? floatSuffix(ct) : uIntSuffix(ct)) +
                     " " + d + ", " + a.reg + ", " + b.reg + ";");
                return {d, ct};
            }
            if (fn == "powf" || fn == "pow") {          // x^y = 2^(y*log2 x)
                const bool dbl = (fn == "pow");
                const Type ft = dbl ? doubleType() : floatType();
                const std::string suf = dbl ? "f64" : "f32";
                a = coerce(a, ft); b = coerce(b, ft);
                std::string lg = fresh(classOf(ft)), mul = fresh(classOf(ft)), d = fresh(classOf(ft));
                emit("lg2.approx." + suf + " " + lg + ", " + a.reg + ";");
                emit("mul." + suf + " " + mul + ", " + b.reg + ", " + lg + ";");
                emit("ex2.approx." + suf + " " + d + ", " + mul + ";");
                return {d, ft};
            }
            if (fn == "atan2f" || fn == "atan2") {      // atan2(y, x) — interpreter-tier
                const bool dbl = (fn == "atan2");
                const Type ft = dbl ? doubleType() : floatType();
                const std::string suf = dbl ? "f64" : "f32";
                a = coerce(a, ft); b = coerce(b, ft);
                std::string d = fresh(classOf(ft));
                emit("atan2.approx." + suf + " " + d + ", " + a.reg + ", " + b.reg + ";");
                return {d, ft};
            }
            if (fn == "hypotf" || fn == "hypot") {      // sqrt(x*x + y*y)
                const bool dbl = (fn == "hypot");
                const Type ft = dbl ? doubleType() : floatType();
                const std::string suf = dbl ? "f64" : "f32";
                a = coerce(a, ft); b = coerce(b, ft);
                std::string x2 = fresh(classOf(ft)), s = fresh(classOf(ft)), d = fresh(classOf(ft));
                emit("mul." + suf + " " + x2 + ", " + a.reg + ", " + a.reg + ";");         // x*x
                emit("fma.rn." + suf + " " + s + ", " + b.reg + ", " + b.reg + ", " + x2 + ";");  // y*y + x*x
                emit("sqrt.rn." + suf + " " + d + ", " + s + ";");
                return {d, ft};
            }
            if (fn == "fdividef") {                     // fast x/y (f32)
                a = coerce(a, floatType()); b = coerce(b, floatType());
                std::string d = fresh(RC::F32);
                emit("div.rn.f32 " + d + ", " + a.reg + ", " + b.reg + ";");
                return {d, floatType()};
            }
            if (fn == "fmodf" || fn == "fmod") {        // x - trunc(x/y)*y
                const bool dbl = (fn == "fmod");
                const Type ft = dbl ? doubleType() : floatType();
                const std::string suf = dbl ? "f64" : "f32";
                a = coerce(a, ft); b = coerce(b, ft);
                std::string q = fresh(classOf(ft)), tq = fresh(classOf(ft)),
                            m = fresh(classOf(ft)), d = fresh(classOf(ft));
                emit("div.rn." + suf + " " + q + ", " + a.reg + ", " + b.reg + ";");
                emit("cvt.rzi." + suf + "." + suf + " " + tq + ", " + q + ";");       // trunc(x/y)
                emit("mul." + suf + " " + m + ", " + tq + ", " + b.reg + ";");
                emit("sub." + suf + " " + d + ", " + a.reg + ", " + m + ";");
                return {d, ft};
            }
            if (fn == "copysignf" || fn == "copysign") {  // |x| with the sign of y
                const bool dbl = (fn == "copysign");
                const Type ft = dbl ? doubleType() : floatType();
                a = coerce(a, ft); b = coerce(b, ft);
                const std::string bt = dbl ? "b64" : "b32";
                const std::string absMask  = dbl ? "0x7fffffffffffffff" : "0x7fffffff";
                const std::string signMask = dbl ? "0x8000000000000000" : "0x80000000";
                const RC ic = dbl ? RC::RD64 : RC::R32;
                std::string xb = fresh(ic), yb = fresh(ic), xa = fresh(ic),
                            ys = fresh(ic), rb = fresh(ic), d = fresh(classOf(ft));
                emit("mov." + bt + " " + xb + ", " + a.reg + ";");           // bit-reinterpret x
                emit("mov." + bt + " " + yb + ", " + b.reg + ";");           // bit-reinterpret y
                emit("and." + bt + " " + xa + ", " + xb + ", " + absMask + ";");   // |x| bits
                emit("and." + bt + " " + ys + ", " + yb + ", " + signMask + ";");  // sign(y) bit
                emit("or."  + bt + " " + rb + ", " + xa + ", " + ys + ";");
                emit("mov." + bt + " " + d + ", " + rb + ";");               // back to float
                return {d, ft};
            }
            fail("unsupported call to '" + fn + "'");
            return {};
        }

        // Ternary intrinsic: fmaf/fma(a,b,c) = a*b + c (f32 / f64).
        if (e.args.size() == 3 && (fn == "fmaf" || fn == "fma")) {
            const bool dbl = (fn == "fma");
            const Type ft = dbl ? doubleType() : floatType();
            Val a = coerce(emitExpr(*e.args[0]), ft); if (failed) return {};
            Val b = coerce(emitExpr(*e.args[1]), ft); if (failed) return {};
            Val c = coerce(emitExpr(*e.args[2]), ft); if (failed) return {};
            std::string d = fresh(classOf(ft));
            emit("fma.rn." + std::string(dbl ? "f64" : "f32") + " " + d + ", " + a.reg + ", " + b.reg + ", " + c.reg + ";");
            return {d, ft};
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
            Type ct = promote(a.type, b.type);
            a = coerce(a, ct); b = coerce(b, ct);
            // Inverted predicate → branch when the original condition is false.
            const std::string& o = cond.str;
            const char* inv = o == "<" ? "ge" : o == "<=" ? "gt" : o == ">" ? "le" :
                              o == ">=" ? "lt" : o == "==" ? "ne" : "eq";
            const std::string suf = cmpSuffix(ct, o);
            std::string p = fresh(RC::Pred);
            emit("setp." + std::string(inv) + "." + suf + " " + p + ", " + a.reg + ", " + b.reg + ";");
            emit("@" + p + " bra " + lbl + ";");
            return;
        }
        Val c = emitExpr(cond);
        if (failed) return;
        // `if (x)` / `while (x)` etc. — branch when x is NOT truthy. Reuse the
        // truthiness predicate (handles pointers/float/half correctly) and invert.
        std::string p = emitToPred(c);
        emit("@!" + p + " bra " + lbl + ";");
    }

    // ── Statements ──────────────────────────────────────────────────────────────
    void emitStmt(const Stmt& s) {
        line = s.line; col = s.col;
        switch (s.kind) {
            case Stmt::VarDecl: {
                if (!ensureSupported(s.type)) return;
                if (s.isExternShared) {
                    // extern __shared__ T name[]  ->  .extern .shared .align 16 .b8 name[];
                    // A dynamically-sized shared array (byte size comes from the launch);
                    // all such arrays alias the region after the static shared area.
                    Type ptr = s.type; ptr.ptr = 1;   // decays to a pointer-to-element
                    Val v; v.type = ptr; v.space = Space::Shared; v.sharedName = s.name;
                    sharedDecls += "\t.extern .shared .align 16 .b8 " + s.name + "[];\n";
                    vars[s.name] = v;
                    arrayDims_[s.name] = std::vector<int>{1};   // 1D; only element stride matters
                    return;
                }
                if (s.arraySize > 0) {
                    int bytes = s.arraySize * s.type.elemBytes();
                    Type ptr = s.type; ptr.ptr = 1;   // the array decays to a pointer-to-element
                    const std::string sym = uniqueSym(s.name);   // hygienic PTX symbol
                    Val v; v.type = ptr;
                    if (s.isShared) {
                        // __shared__ T name[N]  ->  .shared .align 4 .b8 sym[N*sizeof(T)]
                        sharedDecls += "\t.shared .align 4 .b8 " + sym + "[" + std::to_string(bytes) + "];\n";
                        v.space = Space::Shared; v.sharedName = sym;
                    } else {
                        // T name[N]  ->  .local .align 4 .b8 sym[N*sizeof(T)] (per-thread scratch)
                        localDecls += "\t.local .align 4 .b8 " + sym + "[" + std::to_string(bytes) + "];\n";
                        v.space = Space::Local; v.localName = sym;
                    }
                    vars[s.name] = v;
                    arrayDims_[s.name] = s.arrayDims.empty()
                                             ? std::vector<int>{s.arraySize} : s.arrayDims;
                    return;
                }
                if (s.isShared) {
                    // Scalar __shared__ T name;  ->  .shared .align 4 .b8 name[sizeof(T)].
                    // One block-wide cell (not a per-thread register): a write by any
                    // thread is visible to the whole CTA. CUDA forbids an initializer
                    // on a shared variable, so there is none to emit.
                    if (s.expr) { fail("a __shared__ variable cannot have an initializer"); return; }
                    int bytes = s.type.elemBytes();
                    const std::string sym = uniqueSym(s.name);   // hygienic PTX symbol
                    sharedDecls += "\t.shared .align 4 .b8 " + sym + "[" + std::to_string(bytes) + "];\n";
                    Val v; v.type = s.type; v.space = Space::Shared; v.sharedName = sym;
                    vars[s.name] = v;
                    return;
                }
                // Local struct: each scalar member lives in its own register
                // (register-per-field). Members are read/written as `s.field`.
                if (s.type.isStruct()) {
                    const StructDef* def = mod_ ? mod_->findStruct(s.type.structName) : nullptr;
                    if (!def) { fail("unknown struct type '" + s.type.structName + "'"); return; }
                    auto& members = localStructMembers_[s.name];
                    members.clear();
                    for (const auto& m : def->members) {
                        Val mv; mv.type = m.type; mv.reg = fresh(classOf(m.type));
                        if (m.type.isPointer()) mv.space = Space::Global;
                        members[m.name] = mv;
                    }
                    Val v; v.type = s.type; v.space = Space::LocalStruct;
                    vars[s.name] = v;
                    if (s.expr) {   // struct copy-init: `Foo b = a;`
                        emitStructCopy(s.name, *s.expr);
                        if (failed) return;
                    }
                    return;
                }
                Val v; v.type = s.type; v.reg = fresh(classOf(s.type));
                if (s.type.isPointer()) v.space = Space::Global;  // a plain pointer var, if assigned one
                vars[s.name] = v;
                if (s.expr) {
                    Val init = emitExpr(*s.expr);
                    if (failed) return;
                    init = coerce(init, s.type);
                    emit(std::string(movFor(s.type)) + v.reg + ", " + init.reg + ";");
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
                // continue → re-test the condition (top); break → end.
                std::string top = label(), end = label();
                emitLabel(top);
                emitCondBranchFalse(*s.expr, end);
                if (failed) return;
                loopCtx_.push_back({top, end});
                for (auto& st : s.body) { emitStmt(*st); if (failed) { loopCtx_.pop_back(); return; } }
                loopCtx_.pop_back();
                emit("bra " + top + ";");
                emitLabel(end);
                return;
            }
            case Stmt::DoWhile: {
                // Body runs once before the test. continue → the test (cont); break → end.
                std::string top = label(), cont = label(), end = label();
                emitLabel(top);
                loopCtx_.push_back({cont, end});
                for (auto& st : s.body) { emitStmt(*st); if (failed) { loopCtx_.pop_back(); return; } }
                loopCtx_.pop_back();
                emitLabel(cont);
                emitCondBranchFalse(*s.expr, end);   // !cond → exit
                if (failed) return;
                emit("bra " + top + ";");            // cond → loop again
                emitLabel(end);
                return;
            }
            case Stmt::For: {
                if (s.forInit) { emitStmt(*s.forInit); if (failed) return; }
                // continue → the increment (cont), not the top, so the loop advances.
                std::string top = label(), cont = label(), end = label();
                emitLabel(top);
                if (s.forCond) { emitCondBranchFalse(*s.forCond, end); if (failed) return; }
                loopCtx_.push_back({cont, end});
                for (auto& st : s.body) { emitStmt(*st); if (failed) { loopCtx_.pop_back(); return; } }
                loopCtx_.pop_back();
                emitLabel(cont);
                if (s.forIncr) { emitExpr(*s.forIncr); if (failed) return; }
                emit("bra " + top + ";");
                emitLabel(end);
                return;
            }
            case Stmt::Break: {
                // Nearest enclosing loop OR switch (both set a break target).
                if (loopCtx_.empty()) { line = s.line; col = s.col; fail("'break' outside a loop or switch"); return; }
                emit("bra " + loopCtx_.back().second + ";");
                return;
            }
            case Stmt::Continue: {
                // Nearest enclosing LOOP: switch entries carry an empty continue label.
                for (auto it = loopCtx_.rbegin(); it != loopCtx_.rend(); ++it)
                    if (!it->first.empty()) { emit("bra " + it->first + ";"); return; }
                line = s.line; col = s.col; fail("'continue' outside a loop");
                return;
            }
            case Stmt::Switch: {
                Val c = coerce(emitExpr(*s.expr), intType());
                if (failed) return;
                std::string end = label();
                // Label each case/default marker, folding case labels to constants
                // and validating: labels must be constant integers, no duplicate
                // case value, at most one default (C rules).
                std::vector<std::string> labels(s.body.size());
                std::vector<int64_t> caseVal(s.body.size(), 0);
                std::string defaultLabel;
                std::set<int64_t> seen;
                int defaults = 0;
                for (size_t i = 0; i < s.body.size(); ++i) {
                    if (s.body[i]->kind == Stmt::Case) {
                        labels[i] = label();
                        line = s.body[i]->line; col = s.body[i]->col;
                        bool cw;
                        if (!constEval(*s.body[i]->expr, caseVal[i], cw)) { fail("case label must be a constant integer"); return; }
                        if (!seen.insert(caseVal[i]).second) { fail("duplicate case label '" + std::to_string(caseVal[i]) + "'"); return; }
                    } else if (s.body[i]->kind == Stmt::Default) {
                        labels[i] = label();
                        if (++defaults > 1) { line = s.body[i]->line; col = s.body[i]->col; fail("multiple 'default' labels in one switch"); return; }
                        defaultLabel = labels[i];
                    }
                }
                // Comparison chain: jump to the first matching (constant) case value.
                for (size_t i = 0; i < s.body.size(); ++i) {
                    if (s.body[i]->kind != Stmt::Case) continue;
                    std::string p = fresh(RC::Pred);
                    emit("setp.eq.s32 " + p + ", " + c.reg + ", " + std::to_string((int32_t)caseVal[i]) + ";");
                    emit("@" + p + " bra " + labels[i] + ";");
                }
                emit("bra " + (defaultLabel.empty() ? end : defaultLabel) + ";");
                // Bodies in order with C fall-through; break → end.
                loopCtx_.push_back({"", end});
                for (size_t i = 0; i < s.body.size(); ++i) {
                    if (s.body[i]->kind == Stmt::Case || s.body[i]->kind == Stmt::Default)
                        emitLabel(labels[i]);
                    else { emitStmt(*s.body[i]); if (failed) { loopCtx_.pop_back(); return; } }
                }
                loopCtx_.pop_back();
                emitLabel(end);
                return;
            }
            case Stmt::Case: case Stmt::Default: return;   // only meaningful inside a Switch
        }
    }

    // ── Assembly ────────────────────────────────────────────────────────────────
    static std::string paramSuffix(const Type& t) {
        return t.isPointer() ? "u64" : memSuffix(t);
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
            } else {
                // Scalar param (int/long/float/double): load into its register class.
                std::string r = fresh(classOf(p.type));
                emit("ld.param." + memSuffix(p.type) + " " + r + ", [" + p.name + "];");
                vars[p.name] = {r, p.type};
            }
        }
        for (auto& st : k.body) { emitStmt(*st); if (failed) return ""; }
        emit("ret;");

        // Assemble: header + signature + reg decls + body.
        std::string out;
        out += ".version " + opts_.ptxVersion + "\n.target " + opts_.target +
               "\n.address_size " + std::to_string(opts_.addressSize) + "\n\n";
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
        if (nFd > 0) out += "\t.reg .f64 %fd<" + std::to_string(nFd) + ">;\n";
        out += sharedDecls;
        out += localDecls;
        out += body;
        out += "}\n";
        return out;
    }
};

}  // namespace

// With a module (so struct types resolve). This is the complete path for a
// single kernel; the no-module overload below forwards here with an empty module.
CodegenResult generatePtx(const Kernel& kernel, const Module& module, const CodegenOptions& opts) {
    CodegenResult r;
    Codegen cg(kernel, nullptr, &module, opts);
    std::string ptx = cg.run();
    if (cg.failed) { r.ok = false; r.error = cg.err; return r; }
    PtxVerifyResult v = verifyPtx(ptx);   // self-check: a failure is a codegen bug
    if (!v.ok) { r.ok = false; r.error = v.error; return r; }
    r.ptx = std::move(ptx);
    r.ok = true;
    return r;
}

CodegenResult generatePtx(const Kernel& kernel, const CodegenOptions& opts) {
    // No struct table is available here, so a struct parameter (or return) cannot
    // be laid out — reject it with a clear pointer to the module-aware path rather
    // than emitting wrong PTX (a struct param would otherwise get a 1-byte slot).
    for (const Param& p : kernel.params)
        if (p.type.isStruct()) {
            CodegenResult r;
            r.error = "kernel '" + kernel.name + "' has a struct parameter; use "
                      "generatePtx(kernel, module) or compileToPtx() so struct layouts resolve";
            return r;
        }
    Module empty;
    return generatePtx(kernel, empty, opts);
}

// Enrich a "line:col: message" diagnostic with the offending source line and a
// caret under the column, e.g.
//     3:14: use of undeclared identifier 'foo'
//         out[i] = foo(i);
//                  ^
// Returns the error unchanged if it has no parseable location.
static std::string withSourceSnippet(const std::string& source, const std::string& err) {
    size_t c1 = err.find(':');
    if (c1 == std::string::npos || c1 == 0) return err;
    size_t c2 = err.find(':', c1 + 1);
    if (c2 == std::string::npos) return err;
    long ln = std::strtol(err.substr(0, c1).c_str(), nullptr, 10);
    long col = std::strtol(err.substr(c1 + 1, c2 - c1 - 1).c_str(), nullptr, 10);
    if (ln <= 0 || col <= 0) return err;
    // Locate 1-based line `ln`.
    size_t pos = 0; long cur = 1;
    while (cur < ln && pos < source.size()) { if (source[pos] == '\n') ++cur; ++pos; }
    if (cur != ln) return err;
    size_t eol = source.find('\n', pos);
    std::string srcLine = source.substr(pos, (eol == std::string::npos ? source.size() : eol) - pos);
    // Caret line: copy the source's leading whitespace so tabs stay aligned.
    std::string caret;
    for (long i = 0; i + 1 < col && i < (long)srcLine.size(); ++i)
        caret.push_back(srcLine[i] == '\t' ? '\t' : ' ');
    caret.push_back('^');
    return err + "\n    " + srcLine + "\n    " + caret;
}

CodegenResult compileToPtx(const std::string& source, const std::string& name,
                           const CodegenOptions& opts) {
    CodegenResult r;
    ParseResult pr = parse(source);
    if (!pr.ok) { r.error = withSourceSnippet(source, pr.error); return r; }
    // Collect device-callable helpers (not __global__, not __host__-only) for
    // inlining, and pick the __global__ entry `name` (first __global__ if empty).
    std::unordered_map<std::string, const Kernel*> deviceFns;
    const Kernel* target = nullptr;
    for (auto& kp : pr.module->kernels) {
        if (kp->isDeviceCallable()) deviceFns[kp->name] = kp.get();
        // The entry must be a __global__ kernel; a named __device__/__host__
        // helper is never selected as the launch target.
        if (!target && kp->isGlobal && (name.empty() || kp->name == name)) target = kp.get();
    }
    if (!target) {
        r.error = name.empty() ? "no __global__ kernel found"
                               : ("__global__ kernel not found: " + name);
        return r;
    }
    Codegen cg(*target, &deviceFns, pr.module.get(), opts);
    std::string ptx = cg.run();
    if (cg.failed) { r.error = withSourceSnippet(source, cg.err); return r; }
    PtxVerifyResult v = verifyPtx(ptx);   // self-check: a failure is a codegen bug
    if (!v.ok) { r.error = v.error; return r; }
    r.ptx = std::move(ptx);
    r.ok = true;
    return r;
}

}  // namespace frontend
}  // namespace compiler
}  // namespace vgre
