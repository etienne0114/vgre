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
    size_t jgePlaceholder() { u8(0x0F); u8(0x8D); size_t at = code.size(); u32(0); return at; }  // jge rel32
    void ret() { u8(0xC3); }

    // movss xmm, [base + rsi*4]   /   movss [base + rsi*4], xmm   (base in rcx/rdx)
    void movssLoadIdx(int xmm, int base) { u8(0xF3); u8(0x0F); u8(0x10); u8((uint8_t)((xmm << 3) | 4)); u8((uint8_t)(0x80 | (6 << 3) | base)); }
    void movssStoreIdx(int base, int xmm) { u8(0xF3); u8(0x0F); u8(0x11); u8((uint8_t)((xmm << 3) | 4)); u8((uint8_t)(0x80 | (6 << 3) | base)); }
    // movss xmm, [base]           (scalar param, base in rcx)
    void movssLoad(int xmm, int base) { u8(0xF3); u8(0x0F); u8(0x10); u8((uint8_t)((xmm << 3) | base)); }
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
    void addEaxEcx() { u8(0x01); u8(0xC8); }                       // add eax, ecx   (eax = eax + ecx)
    void subEcxEaxToEax() { u8(0x29); u8(0xC1); u8(0x89); u8(0xC8); } // sub ecx,eax ; mov eax,ecx  (eax = ecx - eax)
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

    // Is `e` an integer-typed expression in the native subset? (Picks the eval
    // path for a cast operand; comparisons are int, arithmetic is float if any
    // operand is float.)
    bool isIntExpr(const Expr& e) const {
        switch (e.kind) {
            case Expr::IntLit:   return true;
            case Expr::FloatLit: return false;
            case Expr::Ident: {
                if (e.str == idxVar) return true;
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
        if (depth > 14) { fail("native: expression too deep"); return; }
        switch (e.kind) {
            case Expr::FloatLit: {
                float f = (float)e.fval; uint32_t bits; std::memcpy(&bits, &f, 4);
                asm_.movImmEax(bits); asm_.movdXmmFromEax(0); return;
            }
            case Expr::IntLit: {   // an int literal used in a float context → its float value
                float f = (float)e.ival; uint32_t bits; std::memcpy(&bits, &f, 4);
                asm_.movImmEax(bits); asm_.movdXmmFromEax(0); return;
            }
            case Expr::Ident: {    // a scalar `float` param
                const Param* p = param(e.str);
                if (!p || p->type.isPointer() || p->type.base != Type::Float) { fail("native: unsupported identifier '" + e.str + "'"); return; }
                asm_.movArg(1, paramIndex(e.str)); asm_.movssLoad(0, 1);   // rcx=&value; xmm0=[rcx]
                return;
            }
            case Expr::Index: {    // q[i] — a `float*` param indexed by the induction var
                if (e.args.size() != 2 || e.args[0]->kind != Expr::Ident ||
                    e.args[1]->kind != Expr::Ident || e.args[1]->str != idxVar) { fail("native: only p[i] indexing"); return; }
                const Param* p = param(e.args[0]->str);
                if (!p || !(p->type.base == Type::Float && p->type.ptr == 1)) { fail("native: index base must be float*"); return; }
                asm_.movArg(1, paramIndex(e.args[0]->str)); asm_.deref(1);  // rcx = q pointer
                asm_.movssLoadIdx(0, 1);                                    // xmm0 = [rcx + rsi*4]
                return;
            }
            case Expr::Unary: {
                if (e.str == "+") { emitFloat(*e.args[0], depth); return; }
                if (e.str != "-") { fail("native: unary '" + e.str + "'"); return; }
                emitFloat(*e.args[0], depth);
                asm_.movdEaxFromXmm(0); asm_.xorEaxImm(0x80000000u); asm_.movdXmmFromEax(0);   // negate
                return;
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
                uint8_t opc = e.str == "+" ? 0x58 : e.str == "-" ? 0x5C : e.str == "*" ? 0x59 : e.str == "/" ? 0x5E : 0;
                if (!opc) { fail("native: operator '" + e.str + "'"); return; }
                const int off = 8 * (depth + 1);
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
                if (c.kind != Expr::Binary || c.args.size() != 2) { fail("native: ternary condition must be a float comparison"); return; }
                // Pick the ordered predicate; `>`/`>=` are the swapped `<`/`<=`.
                uint8_t pred; bool swap = false;
                if (c.str == "<") pred = 1;
                else if (c.str == "<=") pred = 2;
                else if (c.str == ">") { pred = 1; swap = true; }
                else if (c.str == ">=") { pred = 2; swap = true; }
                else if (c.str == "==") pred = 0;
                else if (c.str == "!=") pred = 4;
                else { fail("native: ternary needs a comparison condition"); return; }

                // Reserve two red-zone slots for this level; sub-exprs use deeper slots.
                const int slotMask = 8 * (depth + 1);
                const int slotThen = 8 * (depth + 2);
                const int cd = depth + 2;
                if (cd > 14) { fail("native: expression too deep"); return; }

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
        if (depth > 14) { fail("native: expression too deep"); return; }
        switch (e.kind) {
            case Expr::IntLit: {
                asm_.movImmEax((uint32_t)(int32_t)e.ival); return;
            }
            case Expr::Ident: {    // the induction var `i`, or a scalar `int` param
                if (e.str == idxVar) { asm_.movEaxEsi(); return; }
                const Param* p = param(e.str);
                if (!p || p->type.isPointer() || p->type.base != Type::Int) { fail("native: unsupported int identifier '" + e.str + "'"); return; }
                asm_.movArg(1, paramIndex(e.str)); asm_.movEaxMem(1);   // rcx=&value; eax=[rcx]
                return;
            }
            case Expr::Index: {    // q[i] — an `int*` param indexed by the induction var
                if (e.args.size() != 2 || e.args[0]->kind != Expr::Ident ||
                    e.args[1]->kind != Expr::Ident || e.args[1]->str != idxVar) { fail("native: only p[i] indexing"); return; }
                const Param* p = param(e.args[0]->str);
                if (!p || !(p->type.base == Type::Int && p->type.ptr == 1)) { fail("native: index base must be int*"); return; }
                asm_.movArg(1, paramIndex(e.args[0]->str)); asm_.deref(1);  // rcx = q pointer
                asm_.movEaxIdx(1);                                         // eax = [rcx + rsi*4]
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
                const bool add = e.str == "+", sub = e.str == "-", mul = e.str == "*";
                if (!add && !sub && !mul) { fail("native: int operator '" + e.str + "'"); return; }
                const int off = 8 * (depth + 1);
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

        // Body: a sequence of `p[i] = <float expr>;` stores. A braced `{ … }` body
        // parses as a single Block wrapping the statements — unwrap it.
        const std::vector<StmtPtr>& stores =
            (gate.body.size() == 1 && gate.body[0]->kind == Stmt::Block) ? gate.body[0]->body : gate.body;
        for (const StmtPtr& s : stores) {
            if (s->kind != Stmt::ExprStmt || !s->expr || s->expr->kind != Expr::Assign || s->expr->str != "=") {
                err = "native: only `p[i] = expr;` stores in the body"; return false;
            }
            const Expr& lhs = *s->expr->args[0];
            if (lhs.kind != Expr::Index || lhs.args.size() != 2 || lhs.args[0]->kind != Expr::Ident ||
                lhs.args[1]->kind != Expr::Ident || lhs.args[1]->str != idxVar) { err = "native: store target must be p[i]"; return false; }
            const Param* p = param(lhs.args[0]->str);
            if (!p || p->type.ptr != 1 || (p->type.base != Type::Float && p->type.base != Type::Int)) {
                err = "native: store base must be float* or int*"; return false;
            }
            asm_.movArg(2, paramIndex(lhs.args[0]->str)); asm_.deref(2);   // rdx = p pointer (survives expr eval)
            // The store's element type picks the evaluation path — an int* store
            // evaluates an int32 expression, a float* store a float expression.
            if (p->type.base == Type::Int) {
                emitInt(*s->expr->args[1], 0);                            // eax = value
                if (!ok) { err = err.empty() ? "native: unsupported store expression" : err; return false; }
                asm_.movStoreIdxEax(2);                                   // [rdx + rsi*4] = eax
            } else {
                emitFloat(*s->expr->args[1], 0);                          // xmm0 = value
                if (!ok) { err = err.empty() ? "native: unsupported store expression" : err; return false; }
                asm_.movssStoreIdx(2, 0);                                 // [rdx + rsi*4] = xmm0
            }
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
