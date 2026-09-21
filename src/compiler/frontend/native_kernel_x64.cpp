// Native execution tier — a from-scratch, LLVM-free x86-64 (System V / Linux) JIT
// for the elementwise kernel idiom. See include/vgre/compiler/frontend/native_kernel.h.
//
// The emitted function has signature `void fn(void* const* args, uint32_t i)`:
// rdi = args (one pointer per kernel param, pointing at its value), esi = the flat
// global thread index. The kernel's `blockIdx.x*blockDim.x+threadIdx.x` is that
// index. Float expression values live in xmm0 (accumulator) / xmm1 (temp); nested
// operands spill to the red zone below rsp (the function is a leaf, so that's
// safe). Anything outside the subset returns nullptr → the caller falls back.

#include "vgre/compiler/frontend/native_kernel.h"

#include "vgre/compiler/frontend/parser.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(__x86_64__) && defined(__linux__)
#include "vgre/xla/thread_pool.h"
#include <sys/mman.h>
#endif

namespace vgre {
namespace compiler {
namespace frontend {

#if defined(__x86_64__) && defined(__linux__)

namespace {

// ── x86-64 byte emitter (only the instructions this JIT needs) ───────────────
// GPRs: rax=0, rcx=1, rdx=2 (all we scratch). xmm regs by number.
struct X64 {
    std::vector<uint8_t> code;
    void u8(uint8_t b) { code.push_back(b); }
    void u32(uint32_t v) { for (int i = 0; i < 4; ++i) u8((uint8_t)(v >> (8 * i))); }

    // mov r64, [rdi + disp8]   (load the k-th arg pointer; disp = 8*k < 128)
    void movArg(int reg, int k) { u8(0x48); u8(0x8B); u8((uint8_t)(0x40 | (reg << 3) | 7)); u8((uint8_t)(8 * k)); }
    // mov r64, [r64]           (dereference)
    void deref(int reg) { u8(0x48); u8(0x8B); u8((uint8_t)((reg << 3) | reg)); }
    // mov eax, [rax]           (load a 32-bit int through rax)
    void loadInt32Rax() { u8(0x8B); u8(0x00); }
    void movEsiEsi() { u8(0x89); u8(0xF6); }              // zero-extend esi into rsi
    void cmpEsiEax() { u8(0x39); u8(0xC6); }              // cmp esi, eax
    void cmpEcxEax() { u8(0x39); u8(0xC1); }              // cmp ecx, eax  (flags = ecx - eax)
    size_t jgePlaceholder() { u8(0x0F); u8(0x8D); size_t at = code.size(); u32(0); return at; }  // jge rel32
    size_t jgPlaceholder()  { u8(0x0F); u8(0x8F); size_t at = code.size(); u32(0); return at; }  // jg  rel32
    void jmpBackTo(size_t target) { u8(0xE9); size_t at = code.size(); u32(0); patchRel32(at, target); }  // jmp target
    void ret() { u8(0xC3); }

    // movss xmm, [base + rsi*4]   /   movss [base + rsi*4], xmm   (base in rcx/rdx)
    void movssLoadIdx(int xmm, int base) { u8(0xF3); u8(0x0F); u8(0x10); u8((uint8_t)((xmm << 3) | 4)); u8((uint8_t)(0x80 | (6 << 3) | base)); }
    void movssStoreIdx(int base, int xmm) { u8(0xF3); u8(0x0F); u8(0x11); u8((uint8_t)((xmm << 3) | 4)); u8((uint8_t)(0x80 | (6 << 3) | base)); }
    // movss xmm, [base]           (scalar param, base in rcx)
    void movssLoad(int xmm, int base) { u8(0xF3); u8(0x0F); u8(0x10); u8((uint8_t)((xmm << 3) | base)); }
    // general indexed loads: [base + rax*4], the index precomputed (zero-extended) in rax
    void movssLoadIdxRax(int xmm, int base) { u8(0xF3); u8(0x0F); u8(0x10); u8((uint8_t)((xmm << 3) | 4)); u8((uint8_t)(0x80 | base)); }
    void movEaxLoadIdxRax(int base) { u8(0x8B); u8(0x04); u8((uint8_t)(0x80 | base)); }  // mov eax, [base + rax*4]
    // spill/reload via the red zone: movss [rsp-off], xmm0 ; movss xmm, [rsp-off]
    void spillXmm(int xmm, int off) { u8(0xF3); u8(0x0F); u8(0x11); u8((uint8_t)(0x44 | (xmm << 3))); u8(0x24); u8((uint8_t)(-off)); }
    void reloadXmm(int xmm, int off) { u8(0xF3); u8(0x0F); u8(0x10); u8((uint8_t)(0x44 | (xmm << 3))); u8(0x24); u8((uint8_t)(-off)); }

    void movImmEax(uint32_t bits) { u8(0xB8); u32(bits); }        // mov eax, imm32

    // ── 32-bit integer path (values in eax; temp in ecx; spill to the red zone) ──
    void movEaxMem(int base) { u8(0x8B); u8((uint8_t)(0x00 | base)); }        // mov eax, [base]   (base rax/rcx/rdx)
    void movEaxIdx(int base) { u8(0x8B); u8(0x04); u8((uint8_t)(0x80 | (6 << 3) | base)); }  // mov eax, [base + rsi*4]
    void movStoreIdxEax(int base) { u8(0x89); u8(0x04); u8((uint8_t)(0x80 | (6 << 3) | base)); } // mov [base + rsi*4], eax
    void spillEax(int off) { u8(0x89); u8(0x44); u8(0x24); u8((uint8_t)(-off)); }   // mov [rsp-off], eax
    void reloadEcx(int off) { u8(0x8B); u8(0x4C); u8(0x24); u8((uint8_t)(-off)); }  // mov ecx, [rsp-off]
    void movEaxFromSlot(int off) { u8(0x8B); u8(0x44); u8(0x24); u8((uint8_t)(-off)); }  // mov eax, [rsp-off]
    void addEaxEcx() { u8(0x01); u8(0xC8); }                       // add eax, ecx   (eax = eax + ecx)
    void subEcxEaxToEax() { u8(0x29); u8(0xC1); u8(0x89); u8(0xC8); } // sub ecx,eax ; mov eax,ecx  (eax = ecx - eax)
    void subEaxEcx() { u8(0x29); u8(0xC8); }                       // sub eax, ecx   (eax = eax - ecx)
    void movEcxEax() { u8(0x89); u8(0xC1); }                       // mov ecx, eax
    void addEaxImm(int32_t v) { u8(0x05); u32((uint32_t)v); }      // add eax, imm32
    void setccAl(uint8_t cc) { u8(0x0F); u8(cc); u8(0xC0); }       // setcc al   (cc: l 9C le 9E g 9F ge 9D e 94 ne 95)
    void movzxEaxAl() { u8(0x0F); u8(0xB6); u8(0xC0); }            // movzx eax, al  (0/1 → 32-bit)
    void imulEaxEcx() { u8(0x0F); u8(0xAF); u8(0xC1); }            // imul eax, ecx  (eax = eax * ecx)
    void negEax() { u8(0xF7); u8(0xD8); }                          // neg eax
    void movEaxEsi() { u8(0x89); u8(0xF0); }                       // mov eax, esi   (the induction var i)
    void cvtsi2ssFromEax(int xmm) { u8(0xF3); u8(0x0F); u8(0x2A); u8((uint8_t)(0xC0 | (xmm << 3))); }  // xmm = (float)eax
    // float→int32 saturating cast helpers (match the compiled tier's satFloatToInt).
    void cvttss2siEax(int xmm) { u8(0xF3); u8(0x0F); u8(0x2C); u8((uint8_t)(0xC0 | xmm)); }  // eax = (int)trunc(xmm), 0x80000000 on overflow/NaN
    // Scratch for the saturating cast is ECX, not EDX: the store keeps its base
    // pointer in RDX across the whole expression, so RDX must stay untouched.
    void movImmEcx(uint32_t v) { u8(0xB9); u32(v); }              // mov ecx, imm32
    void xorEcxEcx() { u8(0x31); u8(0xC9); }                       // ecx = 0
    void ucomiss(int a, int b) { u8(0x0F); u8(0x2E); u8((uint8_t)(0xC0 | (a << 3) | b)); }  // compare xmm_a, xmm_b → EFLAGS
    void cmovaeEaxEcx() { u8(0x0F); u8(0x43); u8(0xC1); }          // if CF=0 (x >= bound, ordered) eax = ecx
    void cmovpEaxEcx() { u8(0x0F); u8(0x4A); u8(0xC1); }           // if PF=1 (NaN)             eax = ecx

    void movdXmmFromEax(int xmm) { u8(0x66); u8(0x0F); u8(0x6E); u8((uint8_t)(0xC0 | (xmm << 3))); }
    void movdEaxFromXmm(int xmm) { u8(0x66); u8(0x0F); u8(0x7E); u8((uint8_t)(0xC0 | (xmm << 3))); }
    void xorEaxImm(uint32_t v) { u8(0x35); u32(v); }
    // <op>ss xmm1, xmm0  (dst op= src) — add 58 / sub 5C / mul 59 / div 5E
    void arithXmm(uint8_t opc, int dst, int src) { u8(0xF3); u8(0x0F); u8(opc); u8((uint8_t)(0xC0 | (dst << 3) | src)); }
    void sqrtssXmm(int dst, int src) { u8(0xF3); u8(0x0F); u8(0x51); u8((uint8_t)(0xC0 | (dst << 3) | src)); }  // dst = sqrt(src), correctly rounded
    void maxssXmm(int dst, int src) { u8(0xF3); u8(0x0F); u8(0x5F); u8((uint8_t)(0xC0 | (dst << 3) | src)); }   // dst = (dst > src) ? dst : src
    void minssXmm(int dst, int src) { u8(0xF3); u8(0x0F); u8(0x5D); u8((uint8_t)(0xC0 | (dst << 3) | src)); }   // dst = (dst < src) ? dst : src
    void movapsXmm(int dst, int src) { u8(0x0F); u8(0x28); u8((uint8_t)(0xC0 | (dst << 3) | src)); }
    // cmpss xmm_dst, xmm_src, imm8  → dst = all-ones/zero mask per the ordered predicate
    // (imm: 0 EQ, 1 LT, 2 LE, 4 NEQ — all matching C's NaN behaviour).
    void cmpss(int dst, int src, uint8_t pred) { u8(0xF3); u8(0x0F); u8(0xC2); u8((uint8_t)(0xC0 | (dst << 3) | src)); u8(pred); }
    void andps(int dst, int src) { u8(0x0F); u8(0x54); u8((uint8_t)(0xC0 | (dst << 3) | src)); }   // dst &= src
    void andnps(int dst, int src) { u8(0x0F); u8(0x55); u8((uint8_t)(0xC0 | (dst << 3) | src)); }  // dst = ~dst & src
    void orps(int dst, int src) { u8(0x0F); u8(0x56); u8((uint8_t)(0xC0 | (dst << 3) | src)); }    // dst |= src

    void patchRel32(size_t at, size_t target) {
        int32_t rel = (int32_t)(target - (at + 4));
        for (int i = 0; i < 4; ++i) code[at + i] = (uint8_t)((uint32_t)rel >> (8 * i));
    }
};

// ── The elementwise-subset compiler ──────────────────────────────────────────
struct Lowerer {
    const Kernel& k;
    X64 asm_;
    std::string idxVar;                 // the `int i = …` induction variable
    bool ok = true;
    std::string err;

    // Per-thread scalar locals live in the red zone at [rsp - off]; expression
    // scratch spills sit above them (offsets ≥ scratchBase), so the two never alias.
    struct Local { bool isFloat; int off; };
    std::unordered_map<std::string, Local> locals;
    int scratchBase = 0;                 // = 8 * (number of locals)
    int evalSlot(int depth) const { return scratchBase + 8 * (depth + 1); }

    Lowerer(const Kernel& kern) : k(kern) {}
    void fail(const std::string& m) { if (ok) { ok = false; err = m; } }

    int paramIndex(const std::string& name) const {
        for (size_t i = 0; i < k.params.size(); ++i) if (k.params[i].name == name) return (int)i;
        return -1;
    }
    const Param* param(const std::string& name) const {
        int i = paramIndex(name); return i < 0 ? nullptr : &k.params[i];
    }
    static bool member(const Expr& e, const char* obj, const char* field) {
        return e.kind == Expr::Member && e.args.size() == 1 && e.args[0]->kind == Expr::Ident &&
               e.args[0]->str == obj && e.str == field;
    }
    // blockIdx.x*blockDim.x + threadIdx.x  (either operand order for + and *).
    bool isFlatIndex(const Expr& e) const {
        if (e.kind != Expr::Binary || e.str != "+" || e.args.size() != 2) return false;
        const Expr* mul = e.args[0]->kind == Expr::Binary && e.args[0]->str == "*" ? e.args[0].get()
                        : e.args[1]->kind == Expr::Binary && e.args[1]->str == "*" ? e.args[1].get() : nullptr;
        const Expr* tid = member(*e.args[0], "threadIdx", "x") ? e.args[0].get()
                        : member(*e.args[1], "threadIdx", "x") ? e.args[1].get() : nullptr;
        if (!mul || !tid) return false;
        const Expr& a = *mul->args[0]; const Expr& b = *mul->args[1];
        return (member(a, "blockIdx", "x") && member(b, "blockDim", "x")) ||
               (member(a, "blockDim", "x") && member(b, "blockIdx", "x"));
    }

    // Map a comparison operator to an SSE cmpss ordered predicate (0 EQ, 1 LT,
    // 2 LE, 4 NEQ — all matching C's NaN behaviour); `>`/`>=` are the swapped
    // `<`/`<=`. Returns false for non-comparison operators.
    static bool cmpPredicate(const std::string& op, uint8_t& pred, bool& swap) {
        swap = false;
        if (op == "<") pred = 1;
        else if (op == "<=") pred = 2;
        else if (op == ">") { pred = 1; swap = true; }
        else if (op == ">=") { pred = 2; swap = true; }
        else if (op == "==") pred = 0;
        else if (op == "!=") pred = 4;
        else return false;
        return true;
    }

    // Is `e` an integer-typed expression in the native subset? (Picks the eval
    // path for a cast operand; comparisons are int, arithmetic is float if any
    // operand is float.)
    bool isIntExpr(const Expr& e) const {
        switch (e.kind) {
            case Expr::IntLit:   return true;
            case Expr::FloatLit: return false;
            case Expr::Ident: {
                if (e.str == idxVar) return true;
                auto it = locals.find(e.str);
                if (it != locals.end()) return !it->second.isFloat;
                const Param* p = param(e.str);
                return p && !p->type.isPointer() && p->type.base == Type::Int;
            }
            case Expr::Index: {
                const Param* p = (e.args.size() == 2 && e.args[0]->kind == Expr::Ident) ? param(e.args[0]->str) : nullptr;
                return p && p->type.ptr == 1 && p->type.base == Type::Int;
            }
            case Expr::Unary: return e.args.size() == 1 && isIntExpr(*e.args[0]);
            case Expr::Cast:  return e.castType.ptr == 0 && e.castType.base == Type::Int;
            case Expr::Binary: {
                if (e.str == "<" || e.str == "<=" || e.str == ">" || e.str == ">=" || e.str == "==" || e.str == "!=") return true;
                return e.args.size() == 2 && isIntExpr(*e.args[0]) && isIntExpr(*e.args[1]);
            }
            case Expr::Ternary: return e.args.size() == 3 && isIntExpr(*e.args[1]) && isIntExpr(*e.args[2]);
            default: return false;
        }
    }

    // Emit a float expression into xmm0. `depth` picks the red-zone spill slot.
    void emitFloat(const Expr& e, int depth) {
        if (!ok) return;
        if (evalSlot(depth) > 120) { fail("native: expression too deep"); return; }
        switch (e.kind) {
            case Expr::FloatLit: {
                float f = (float)e.fval; uint32_t bits; std::memcpy(&bits, &f, 4);
                asm_.movImmEax(bits); asm_.movdXmmFromEax(0); return;
            }
            case Expr::IntLit: {   // an int literal used in a float context → its float value
                float f = (float)e.ival; uint32_t bits; std::memcpy(&bits, &f, 4);
                asm_.movImmEax(bits); asm_.movdXmmFromEax(0); return;
            }
            case Expr::Ident: {    // a scalar `float` param or a float local
                auto it = locals.find(e.str);
                if (it != locals.end()) {
                    if (!it->second.isFloat) { fail("native: int local '" + e.str + "' in float context (needs a cast)"); return; }
                    asm_.reloadXmm(0, it->second.off); return;   // xmm0 = [rsp - off]
                }
                const Param* p = param(e.str);
                if (!p || p->type.isPointer() || p->type.base != Type::Float) { fail("native: unsupported identifier '" + e.str + "'"); return; }
                asm_.movArg(1, paramIndex(e.str)); asm_.movssLoad(0, 1);   // rcx=&value; xmm0=[rcx]
                return;
            }
            case Expr::Index: {    // q[<int expr>] — a `float*` param, general integer index
                if (e.args.size() != 2 || e.args[0]->kind != Expr::Ident) { fail("native: bad index expression"); return; }
                const Param* p = param(e.args[0]->str);
                if (!p || !(p->type.base == Type::Float && p->type.ptr == 1)) { fail("native: index base must be float*"); return; }
                if (e.args[1]->kind == Expr::Ident && e.args[1]->str == idxVar) {      // q[i] fast path (index already in rsi)
                    asm_.movArg(1, paramIndex(e.args[0]->str)); asm_.deref(1);         // rcx = q pointer
                    asm_.movssLoadIdx(0, 1);                                           // xmm0 = [rcx + rsi*4]
                } else {                                                              // q[expr]: compute index → rax, then [base + rax*4]
                    emitInt(*e.args[1], depth);                                        // eax = index (rax zero-extended)
                    if (!ok) return;
                    asm_.movArg(1, paramIndex(e.args[0]->str)); asm_.deref(1);         // rcx = q pointer (rax preserved)
                    asm_.movssLoadIdxRax(0, 1);                                        // xmm0 = [rcx + rax*4]
                }
                return;
            }
            case Expr::Unary: {
                if (e.str == "+") { emitFloat(*e.args[0], depth); return; }
                if (e.str != "-") { fail("native: unary '" + e.str + "'"); return; }
                emitFloat(*e.args[0], depth);
                asm_.movdEaxFromXmm(0); asm_.xorEaxImm(0x80000000u); asm_.movdXmmFromEax(0);   // negate
                return;
            }
            case Expr::Call: {     // float math intrinsics that are bit-exact here
                if (e.args.size() == 1 && (e.str == "sqrtf" || e.str == "fabsf")) {
                    emitFloat(*e.args[0], depth);
                    if (e.str == "sqrtf") {
                        asm_.sqrtssXmm(0, 0);   // SSE sqrt is IEEE correctly-rounded, == C sqrtf
                    } else {                    // fabsf: clear the sign bit
                        asm_.movImmEax(0x7fffffffu); asm_.movdXmmFromEax(1); asm_.andps(0, 1);
                    }
                    return;
                }
                if (e.args.size() == 2 && (e.str == "fmaxf" || e.str == "fminf")) {
                    // C fmaxf/fminf, bit-exact: (a==b || isnan(b)) ? a : {max,min}ss(a,b).
                    // The a==b arm reproduces the ±0 rule (fmax(+0,-0) keeps a's sign);
                    // the isnan(b) arm reproduces "NaN operand → the other" for b.
                    const bool isMax = e.str == "fmaxf";
                    const int off = evalSlot(depth);
                    emitFloat(*e.args[0], depth); asm_.spillXmm(0, off);        // [off] = a
                    emitFloat(*e.args[1], depth + 1); asm_.reloadXmm(1, off);   // xmm1 = a, xmm0 = b
                    asm_.movapsXmm(2, 1);                                       // xmm2 = a
                    if (isMax) asm_.maxssXmm(2, 0); else asm_.minssXmm(2, 0);   // xmm2 = m = {max,min}ss(a,b)
                    asm_.movapsXmm(3, 1);                                       // xmm3 = a
                    asm_.cmpss(3, 0, 0);                                        // xmm3 = (a == b) mask (ordered EQ)
                    asm_.cmpss(0, 0, 3);                                        // xmm0 = (b is NaN) mask (UNORD)
                    asm_.orps(3, 0);                                            // xmm3 = keepA = (a==b) | isnan(b)
                    asm_.andps(1, 3);                                           // xmm1 = a & keepA
                    asm_.andnps(3, 2);                                          // xmm3 = ~keepA & m
                    asm_.orps(1, 3);                                            // xmm1 = result
                    asm_.movapsXmm(0, 1);
                    return;
                }
                fail("native: unsupported call '" + e.str + "'"); return;
            }
            case Expr::Cast: {     // (float)x
                if (e.castType.base == Type::Float && e.castType.ptr == 0) {
                    if (isIntExpr(*e.args[0])) { emitInt(*e.args[0], depth); asm_.cvtsi2ssFromEax(0); }  // int→float, round-to-nearest
                    else emitFloat(*e.args[0], depth);   // already float — a no-op fold
                    return;
                }
                fail("native: unsupported cast"); return;
            }
            case Expr::Binary: {
                // A bare float comparison yields 1.0f / 0.0f (a predicate mask AND 1.0f).
                { uint8_t pred; bool swap;
                  if (cmpPredicate(e.str, pred, swap)) {
                    const int off = evalSlot(depth);
                    emitFloat(*e.args[0], depth); asm_.spillXmm(0, off);       // [off] = L
                    emitFloat(*e.args[1], depth + 1); asm_.reloadXmm(1, off);  // xmm1 = L, xmm0 = R
                    if (swap) { asm_.cmpss(0, 1, pred); asm_.movapsXmm(1, 0); } // mask → xmm1
                    else       asm_.cmpss(1, 0, pred);                          // mask → xmm1
                    asm_.movImmEax(0x3f800000u); asm_.movdXmmFromEax(0);        // xmm0 = 1.0f
                    asm_.andps(0, 1);                                           // xmm0 = mask ? 1.0f : 0.0f
                    return;
                  } }
                uint8_t opc = e.str == "+" ? 0x58 : e.str == "-" ? 0x5C : e.str == "*" ? 0x59 : e.str == "/" ? 0x5E : 0;
                if (!opc) { fail("native: operator '" + e.str + "'"); return; }
                const int off = evalSlot(depth);
                emitFloat(*e.args[0], depth);       // L → xmm0
                asm_.spillXmm(0, off);              // [rsp-off] = L
                emitFloat(*e.args[1], depth + 1);   // R → xmm0
                asm_.reloadXmm(1, off);            // xmm1 = L
                asm_.arithXmm(opc, 1, 0);          // xmm1 = L op R
                asm_.movapsXmm(0, 1);              // xmm0 = result
                return;
            }
            case Expr::Ternary: {   // `(<cmp>) ? t : e` — branchless SSE mask blend
                const Expr& c = *e.args[0];
                uint8_t pred; bool swap;
                if (c.kind != Expr::Binary || c.args.size() != 2 || !cmpPredicate(c.str, pred, swap)) {
                    fail("native: ternary condition must be a float comparison"); return;
                }

                // Reserve two red-zone slots for this level; sub-exprs use deeper slots.
                const int slotMask = evalSlot(depth);
                const int slotThen = evalSlot(depth + 1);
                const int cd = depth + 2;
                if (evalSlot(cd) > 120) { fail("native: expression too deep"); return; }

                emitFloat(*c.args[0], cd); asm_.spillXmm(0, slotMask);   // [mask] = cl
                emitFloat(*c.args[1], cd); asm_.reloadXmm(1, slotMask);  // xmm1 = cl, xmm0 = cr
                if (swap) { asm_.cmpss(0, 1, pred); asm_.movapsXmm(1, 0); }  // xmm1 = (cr ? cl) i.e. cl > / >= cr
                else       asm_.cmpss(1, 0, pred);                          // xmm1 = cl < / <= / == / != cr
                asm_.spillXmm(1, slotMask);                                 // [mask] = compare result

                emitFloat(*e.args[1], cd); asm_.spillXmm(0, slotThen);   // [then] = t
                emitFloat(*e.args[2], cd);                               // xmm0 = e (else)
                if (!ok) return;
                asm_.reloadXmm(1, slotMask);   // xmm1 = mask
                asm_.reloadXmm(2, slotThen);   // xmm2 = t
                asm_.andps(2, 1);              // xmm2 = t & mask
                asm_.andnps(1, 0);            // xmm1 = ~mask & e
                asm_.orps(2, 1);             // xmm2 = (t & mask) | (~mask & e)
                asm_.movapsXmm(0, 2);        // xmm0 = result
                return;
            }
            default: fail("native: unsupported expression"); return;
        }
    }

    // Saturating float→int32 of the value in xmm0, result in eax — matches the
    // compiled tier's satFloatToInt (trunc; NaN→0; clamp to [INT_MIN, INT_MAX]).
    // cvttss2si already yields the right answer for in-range values, genuine
    // INT_MIN, and everything ≤ INT_MIN incl. -inf (all → 0x80000000). Only two
    // cases need fixing: x ≥ 2^31 (incl +inf) → INT_MAX, and NaN → 0.
    void emitSatF2I() {
        asm_.movImmEax(0x4F000000u); asm_.movdXmmFromEax(1);   // xmm1 = 2^31 as float
        asm_.cvttss2siEax(0);                                  // eax = trunc(x) or 0x80000000
        asm_.movImmEcx(0x7FFFFFFFu);                           // ecx = INT_MAX
        asm_.ucomiss(0, 1); asm_.cmovaeEaxEcx();               // x ≥ 2^31 (ordered) → INT_MAX
        asm_.xorEcxEcx();                                      // ecx = 0
        asm_.ucomiss(0, 0); asm_.cmovpEaxEcx();                // x is NaN → 0
    }

    // Emit a 32-bit integer expression into eax. All arithmetic wraps mod 2^32
    // (x86 add/sub/imul on r32), matching the compiled tier's int32 semantics.
    // Division/modulo are deferred (they trap on /0 and INT_MIN/-1).
    void emitInt(const Expr& e, int depth) {
        if (!ok) return;
        // Int and float share one depth→red-zone-offset map (8-byte stride) so a
        // cast that nests int eval inside float eval never aliases a live slot.
        if (evalSlot(depth) > 120) { fail("native: expression too deep"); return; }
        switch (e.kind) {
            case Expr::IntLit: {
                asm_.movImmEax((uint32_t)(int32_t)e.ival); return;
            }
            case Expr::Ident: {    // the induction var `i`, an int local, or a scalar `int` param
                if (e.str == idxVar) { asm_.movEaxEsi(); return; }
                auto it = locals.find(e.str);
                if (it != locals.end()) {
                    if (it->second.isFloat) { fail("native: float local '" + e.str + "' in int context (needs a cast)"); return; }
                    asm_.movEaxFromSlot(it->second.off); return;   // eax = [rsp - off]
                }
                const Param* p = param(e.str);
                if (!p || p->type.isPointer() || p->type.base != Type::Int) { fail("native: unsupported int identifier '" + e.str + "'"); return; }
                asm_.movArg(1, paramIndex(e.str)); asm_.movEaxMem(1);   // rcx=&value; eax=[rcx]
                return;
            }
            case Expr::Index: {    // q[<int expr>] — an `int*` param, general integer index
                if (e.args.size() != 2 || e.args[0]->kind != Expr::Ident) { fail("native: bad index expression"); return; }
                const Param* p = param(e.args[0]->str);
                if (!p || !(p->type.base == Type::Int && p->type.ptr == 1)) { fail("native: index base must be int*"); return; }
                if (e.args[1]->kind == Expr::Ident && e.args[1]->str == idxVar) {      // q[i] fast path
                    asm_.movArg(1, paramIndex(e.args[0]->str)); asm_.deref(1);         // rcx = q pointer
                    asm_.movEaxIdx(1);                                                 // eax = [rcx + rsi*4]
                } else {                                                              // q[expr]: index → rax, then [base + rax*4]
                    emitInt(*e.args[1], depth);                                        // eax = index (rax zero-extended)
                    if (!ok) return;
                    asm_.movArg(1, paramIndex(e.args[0]->str)); asm_.deref(1);         // rcx = q pointer (rax preserved)
                    asm_.movEaxLoadIdxRax(1);                                          // eax = [rcx + rax*4]
                }
                return;
            }
            case Expr::Unary: {
                if (e.str == "+") { emitInt(*e.args[0], depth); return; }
                if (e.str != "-") { fail("native: int unary '" + e.str + "'"); return; }
                emitInt(*e.args[0], depth); asm_.negEax(); return;
            }
            case Expr::Cast: {     // (int)x
                if (e.castType.base == Type::Int && e.castType.ptr == 0) {
                    if (isIntExpr(*e.args[0])) emitInt(*e.args[0], depth);      // int→int32 (already int)
                    else { emitFloat(*e.args[0], depth); emitSatF2I(); }        // float→int32, saturating
                    return;
                }
                fail("native: unsupported int cast"); return;
            }
            case Expr::Binary: {
                // A bare int comparison yields 0 / 1 (signed cmp + setcc + zero-extend).
                { uint8_t cc = e.str == "<" ? 0x9C : e.str == "<=" ? 0x9E : e.str == ">" ? 0x9F
                             : e.str == ">=" ? 0x9D : e.str == "==" ? 0x94 : e.str == "!=" ? 0x95 : 0;
                  if (cc) {
                    const int off = evalSlot(depth);
                    emitInt(*e.args[0], depth); asm_.spillEax(off);       // [off] = L
                    emitInt(*e.args[1], depth + 1); asm_.reloadEcx(off);  // ecx = L, eax = R
                    asm_.cmpEcxEax();                                     // flags = L - R
                    asm_.setccAl(cc); asm_.movzxEaxAl();                  // eax = (L cmp R) ? 1 : 0
                    return;
                  } }
                const bool add = e.str == "+", sub = e.str == "-", mul = e.str == "*";
                if (!add && !sub && !mul) { fail("native: int operator '" + e.str + "'"); return; }
                const int off = evalSlot(depth);
                emitInt(*e.args[0], depth);        // L → eax
                asm_.spillEax(off);               // [rsp-off] = L
                emitInt(*e.args[1], depth + 1);    // R → eax
                asm_.reloadEcx(off);              // ecx = L
                if (add) asm_.addEaxEcx();         // eax = R + L
                else if (mul) asm_.imulEaxEcx();   // eax = R * L
                else asm_.subEcxEaxToEax();        // eax = L - R
                return;
            }
            default: fail("native: unsupported int expression"); return;
        }
    }

    // ── statement emitters (the top-level body and loop bodies share these) ──────

    // `p[i] = <expr>;` store — the element type picks the evaluation path.
    bool emitStore(const Expr& lhs, const Expr& rhs) {
        if (lhs.args.size() != 2 || lhs.args[0]->kind != Expr::Ident ||
            lhs.args[1]->kind != Expr::Ident || lhs.args[1]->str != idxVar) { err = "native: store target must be p[i]"; return false; }
        const Param* p = param(lhs.args[0]->str);
        if (!p || p->type.ptr != 1 || (p->type.base != Type::Float && p->type.base != Type::Int)) {
            err = "native: store base must be float* or int*"; return false;
        }
        asm_.movArg(2, paramIndex(lhs.args[0]->str)); asm_.deref(2);   // rdx = p pointer (survives expr eval)
        if (p->type.base == Type::Int) {
            emitInt(rhs, 0); if (!ok) { err = err.empty() ? "native: unsupported store expression" : err; return false; }
            asm_.movStoreIdxEax(2);                                    // [rdx + rsi*4] = eax
        } else {
            emitFloat(rhs, 0); if (!ok) { err = err.empty() ? "native: unsupported store expression" : err; return false; }
            asm_.movssStoreIdx(2, 0);                                  // [rdx + rsi*4] = xmm0
        }
        return true;
    }

    // An assignment / compound-assignment / ++/-- used as a statement (this also
    // handles the loop-increment expression).
    bool emitAssignExpr(const Expr& e) {
        if (e.kind == Expr::Unary && (e.str == "pre++" || e.str == "post++" || e.str == "pre--" || e.str == "post--")) {
            auto it = e.args[0]->kind == Expr::Ident ? locals.find(e.args[0]->str) : locals.end();
            if (it == locals.end() || it->second.isFloat) { err = "native: ++/-- needs an int local"; return false; }
            asm_.movEaxFromSlot(it->second.off);
            asm_.addEaxImm((e.str == "pre++" || e.str == "post++") ? 1 : -1);
            asm_.spillEax(it->second.off);
            return true;
        }
        if (e.kind != Expr::Assign || e.args.size() != 2) { err = "native: statement must be an assignment or ++/--"; return false; }
        const Expr& lhs = *e.args[0];
        const Expr& rhs = *e.args[1];
        const std::string& op = e.str;

        if (lhs.kind == Expr::Index) {   // p[i] = expr  (compound stores unsupported)
            if (op != "=") { err = "native: compound store not supported"; return false; }
            return emitStore(lhs, rhs);
        }
        if (lhs.kind != Expr::Ident) { err = "native: unsupported assignment target"; return false; }
        auto it = locals.find(lhs.str);
        if (it == locals.end()) { err = "native: assignment to non-local '" + lhs.str + "'"; return false; }
        const Local& L = it->second;
        if (L.isFloat) {
            if (op == "=") { emitFloat(rhs, 0); if (!ok) return false; asm_.spillXmm(0, L.off); return true; }
            uint8_t opc = op == "+=" ? 0x58 : op == "-=" ? 0x5C : op == "*=" ? 0x59 : op == "/=" ? 0x5E : 0;
            if (!opc) { err = "native: unsupported float compound op"; return false; }
            emitFloat(rhs, 0); if (!ok) return false;   // xmm0 = rhs
            asm_.reloadXmm(1, L.off);                    // xmm1 = local
            asm_.arithXmm(opc, 1, 0);                    // xmm1 = local op rhs
            asm_.spillXmm(1, L.off);
            return true;
        }
        if (op == "=") { emitInt(rhs, 0); if (!ok) return false; asm_.spillEax(L.off); return true; }
        emitInt(rhs, 0); if (!ok) return false;          // eax = rhs
        asm_.movEcxEax();                                // ecx = rhs
        asm_.movEaxFromSlot(L.off);                      // eax = local
        if (op == "+=") asm_.addEaxEcx();                // eax = local + rhs
        else if (op == "-=") asm_.subEaxEcx();           // eax = local - rhs
        else if (op == "*=") asm_.imulEaxEcx();          // eax = local * rhs
        else { err = "native: unsupported int compound op"; return false; }
        asm_.spillEax(L.off);
        return true;
    }

    // A single body statement (recursively used for loop bodies).
    bool emitStmt(const Stmt& s) {
        switch (s.kind) {
            case Stmt::VarDecl: {
                auto it = locals.find(s.name);
                if (it == locals.end()) { err = "native: unexpected local declaration"; return false; }
                const Local& L = it->second;
                if (L.isFloat) { emitFloat(*s.expr, 0); if (!ok) { err = err.empty() ? "native: bad local init" : err; return false; } asm_.spillXmm(0, L.off); }
                else           { emitInt(*s.expr, 0);   if (!ok) { err = err.empty() ? "native: bad local init" : err; return false; } asm_.spillEax(L.off); }
                return true;
            }
            case Stmt::ExprStmt:
                if (!s.expr) { err = "native: empty statement"; return false; }
                return emitAssignExpr(*s.expr);
            case Stmt::For: return emitFor(s);
            default: err = "native: unsupported statement in body"; return false;
        }
    }

    // A bounded for-loop: `for (int j = init; j < bound; ++j) { … }` (also `<=`).
    // The loop var and any accumulators live in red-zone slots, so nothing needs
    // to survive in a register across iterations.
    bool emitFor(const Stmt& s) {
        if (!s.forInit || s.forInit->kind != Stmt::VarDecl) { err = "native: for-init must declare an int loop var"; return false; }
        const Stmt& init = *s.forInit;
        auto it = locals.find(init.name);
        if (it == locals.end() || it->second.isFloat || !init.expr) { err = "native: for loop var must be an int local"; return false; }
        const int jOff = it->second.off;
        emitInt(*init.expr, 0); if (!ok) return false; asm_.spillEax(jOff);   // j = init

        if (!s.forCond || s.forCond->kind != Expr::Binary || s.forCond->args.size() != 2) { err = "native: for-cond must be a comparison"; return false; }
        bool le;
        if (s.forCond->str == "<") le = false;
        else if (s.forCond->str == "<=") le = true;
        else { err = "native: for-cond must be `<` or `<=`"; return false; }

        const size_t top = asm_.code.size();
        emitInt(*s.forCond->args[0], 0); if (!ok) return false;   // eax = L
        asm_.spillEax(evalSlot(0));
        emitInt(*s.forCond->args[1], 1); if (!ok) return false;   // eax = R
        asm_.reloadEcx(evalSlot(0));                              // ecx = L
        asm_.cmpEcxEax();                                         // flags = L - R
        const size_t exitJmp = le ? asm_.jgPlaceholder() : asm_.jgePlaceholder();  // exit when !(L < / <= R)

        const std::vector<StmtPtr>& bodyStmts =
            (s.body.size() == 1 && s.body[0]->kind == Stmt::Block) ? s.body[0]->body : s.body;
        for (const StmtPtr& bs : bodyStmts) { if (!emitStmt(*bs)) return false; if (!ok) return false; }

        if (s.forIncr) { if (!emitAssignExpr(*s.forIncr)) return false; }
        asm_.jmpBackTo(top);
        asm_.patchRel32(exitJmp, asm_.code.size());
        return true;
    }

    // Compile the whole kernel; returns false (with err) if it isn't in the subset.
    bool compile() {
        // Only float*/int* pointers and float/int scalars — the elementwise subset.
        for (const Param& p : k.params) {
            const bool okPtr = p.type.ptr == 1 && (p.type.base == Type::Float || p.type.base == Type::Int);
            const bool okScalar = p.type.ptr == 0 && (p.type.base == Type::Float || p.type.base == Type::Int);
            if (!okPtr && !okScalar) { err = "native: unsupported param type"; return false; }
        }
        if (k.body.size() != 2 || k.body[0]->kind != Stmt::VarDecl || k.body[1]->kind != Stmt::If) {
            err = "native: not the `int i = …; if (i<n) {…}` shape"; return false;
        }
        const Stmt& decl = *k.body[0];
        if (decl.type.base != Type::Int || decl.type.ptr != 0 || decl.arraySize != 0 || !decl.expr || !isFlatIndex(*decl.expr)) {
            err = "native: first stmt must be `int i = blockIdx.x*blockDim.x+threadIdx.x`"; return false;
        }
        idxVar = decl.name;

        const Stmt& gate = *k.body[1];
        if (!gate.elseBody.empty() || !gate.expr || gate.expr->kind != Expr::Binary || gate.expr->str != "<" ||
            gate.expr->args[0]->kind != Expr::Ident || gate.expr->args[0]->str != idxVar ||
            gate.expr->args[1]->kind != Expr::Ident) { err = "native: guard must be `if (i < n)`"; return false; }
        const Param* nP = param(gate.expr->args[1]->str);
        if (!nP || nP->type.ptr != 0 || nP->type.base != Type::Int) { err = "native: bound `n` must be an int param"; return false; }
        const int nIdx = paramIndex(gate.expr->args[1]->str);

        // Prologue: i is in esi; zero-extend so [base + rsi*4] is valid.
        asm_.movEsiEsi();
        // Guard: load n, compare, skip the body if i >= n.
        asm_.movArg(0, nIdx); asm_.loadInt32Rax();   // rax=&n ; eax=n
        asm_.cmpEsiEax();
        size_t jmp = asm_.jgePlaceholder();

        // Body: scalar-local declarations interleaved with `p[i] = <expr>;` stores.
        // A braced `{ … }` body parses as a single Block wrapping the statements.
        const std::vector<StmtPtr>& body =
            (gate.body.size() == 1 && gate.body[0]->kind == Stmt::Block) ? gate.body[0]->body : gate.body;

        // Pre-pass: lay out per-thread scalar locals in the red zone (each an
        // 8-byte slot) so their types/offsets are known before any expression is
        // emitted; expression scratch then sits above them at ≥ scratchBase. This
        // also registers for-loop induction vars (declared in the loop's init).
        int nLocals = 0;
        auto registerLocal = [&](const Stmt& d) -> bool {
            if (d.type.ptr != 0 || d.arraySize != 0 || !d.expr ||
                (d.type.base != Type::Float && d.type.base != Type::Int)) {
                err = "native: only scalar float/int locals with an initializer"; return false;
            }
            if (nLocals >= 12) { err = "native: too many locals"; return false; }
            locals[d.name] = Local{d.type.base == Type::Float, 8 * (nLocals + 1)};
            ++nLocals;
            return true;
        };
        for (const StmtPtr& s : body) {
            if (s->kind == Stmt::VarDecl) { if (!registerLocal(*s)) return false; }
            else if (s->kind == Stmt::For && s->forInit && s->forInit->kind == Stmt::VarDecl) {
                if (!registerLocal(*s->forInit)) return false;
            }
        }
        scratchBase = 8 * nLocals;

        // Emit pass — each body statement (locals, stores, loops).
        for (const StmtPtr& s : body) {
            if (!emitStmt(*s)) return false;
            if (!ok) { if (err.empty()) err = "native: unsupported body"; return false; }
        }
        // Epilogue.
        asm_.patchRel32(jmp, asm_.code.size());
        asm_.ret();
        return true;
    }
};

using KernelFn = void (*)(void* const*, uint32_t);

class NativeKernelImpl : public NativeKernel {
public:
    NativeKernelImpl(std::vector<uint8_t> bytes, int nparams) : nparams_(nparams) {
        size_ = ((bytes.size() + 4095) / 4096) * 4096;
        mem_ = mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        std::memcpy(mem_, bytes.data(), bytes.size());
        mprotect(mem_, size_, PROT_READ | PROT_EXEC);   // W^X
        fn_ = reinterpret_cast<KernelFn>(mem_);
    }
    ~NativeKernelImpl() override { if (mem_ && mem_ != MAP_FAILED) munmap(mem_, size_); }

    int numParams() const override { return nparams_; }

    bool launch(const Extent& grid, const Extent& block, void* const* args, int numArgs) override {
        if (numArgs != nparams_ || mem_ == MAP_FAILED) return false;
        const uint32_t total = grid.x * block.x;   // 1-D elementwise
        KernelFn fn = fn_;
        auto& pool = xla::ThreadPool::global();
        if (total <= 1024 || pool.concurrency() <= 1) {
            for (uint32_t i = 0; i < total; ++i) fn(args, i);
        } else {
            int64_t grain = 256;
            pool.parallelFor((int64_t)total, grain, [&](int64_t i) { fn(args, (uint32_t)i); });
        }
        return true;
    }

private:
    void* mem_ = MAP_FAILED;
    size_t size_ = 0;
    KernelFn fn_ = nullptr;
    int nparams_ = 0;
};

}  // namespace

std::unique_ptr<NativeKernel> NativeKernel::compileSource(const std::string& source,
                                                         const std::string& name,
                                                         std::string& err) {
    ParseResult pr = parse(source);
    if (!pr.ok) { err = pr.error; return nullptr; }
    const Kernel* target = nullptr;
    if (!name.empty()) { for (auto& kp : pr.module->kernels) if (kp->name == name) { target = kp.get(); break; } }
    else { for (auto& kp : pr.module->kernels) if (kp->isGlobal) { target = kp.get(); break; } }
    if (!target) { err = "kernel not found"; return nullptr; }

    Lowerer low(*target);
    if (!low.compile()) { err = low.err; return nullptr; }
    return std::unique_ptr<NativeKernel>(new NativeKernelImpl(std::move(low.asm_.code), (int)target->params.size()));
}

#else   // not x86-64 Linux — no native tier; callers fall back.

std::unique_ptr<NativeKernel> NativeKernel::compileSource(const std::string&, const std::string&, std::string& err) {
    err = "native tier: only x86-64 Linux is supported";
    return nullptr;
}

#endif

}  // namespace frontend
}  // namespace compiler
}  // namespace vgre
