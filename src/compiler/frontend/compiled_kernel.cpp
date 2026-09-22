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

// Cooperative (`__shared__`/`__syncthreads`) kernels run each CUDA thread as a
// fiber (ucontext) and a block scheduler releases the barrier once every thread
// reaches it — see the cooperative launch path below. POSIX-only; on other hosts
// such kernels fall back to the interpreter tier as before.
#if defined(__unix__) || defined(__APPLE__)
#define VGRE_COOP_FIBERS 1
#include <ucontext.h>
#else
#define VGRE_COOP_FIBERS 0
#endif

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
    // C truthiness: a float is true iff != 0.0 (NOT its int truncation — 0<|x|<1
    // is true), an int/pointer iff != 0.
    bool truthy() const { return isFloat ? f != 0.0 : i != 0; }
};

// Per-thread execution state.
struct TS {
    std::vector<Cell> regs;
    uint32_t tid[3] = {0, 0, 0}, ctaid[3] = {0, 0, 0};
    uint32_t ntid[3] = {1, 1, 1}, nctaid[3] = {1, 1, 1};
    void* const* args = nullptr;
    bool returned = false;
    Cell* shared = nullptr;    // block-wide `__shared__` storage (cooperative kernels)
    void* fiber = nullptr;     // Fiber* while running cooperatively; __syncthreads yields it
    uint32_t lintid = 0;       // linear thread id in the block (for warp = /32, lane = %32)
    Cell* shflPub = nullptr;   // per-CTA warp-shuffle publish buffer (numWarps*32 Cells)
    Cell* shflSnap = nullptr;  // its snapshot, taken at each warp release
    uint32_t* shflActive = nullptr;  // per-warp mask of lanes that reached this shuffle
};

using ExprFn = std::function<Cell(TS&)>;
using StmtFn = std::function<void(TS&)>;

#if VGRE_COOP_FIBERS
// One CUDA thread as a fiber: its own stack + context; the block scheduler round-
// robins them, releasing __syncthreads once every live fiber has reached it.
// wait kinds a fiber parks with: NONE=runnable, BLOCK=__syncthreads (block-wide),
// WARP=a warp-shuffle (released per 32-lane warp).
enum { WAIT_NONE = 0, WAIT_BLOCK = 1, WAIT_WARP = 2 };
struct Fiber {
    ucontext_t ctx;
    ucontext_t* sched = nullptr;
    std::vector<char> stack;
    const std::vector<StmtFn>* body = nullptr;
    TS* ts = nullptr;
    bool done = false;
    int wait = WAIT_NONE;
};
// Set by the scheduler just before a fiber's FIRST resume so the trampoline (which
// takes no args under makecontext) can find its work. Thread-local: one worker per CTA.
thread_local Fiber* g_startFiber = nullptr;
inline void fiberTrampoline() {
    Fiber* f = g_startFiber;
    for (const StmtFn& s : *f->body) { s(*f->ts); if (f->ts->returned) break; }
    f->done = true;   // returns to uc_link (the scheduler)
}
inline void fiberYield(TS& ts, int kind) {
    Fiber* f = static_cast<Fiber*>(ts.fiber);
    f->wait = kind;
    swapcontext(&f->ctx, f->sched);
}
// __syncthreads(): block-wide barrier.
inline void coopBarrier(TS& ts) { if (ts.fiber) fiberYield(ts, WAIT_BLOCK); }

// Warp shuffle: publish `var`, warp-barrier-yield, then read the source lane's
// value from the snapshot. mode 0=idx 1=up 2=down 3=xor; matches the interpreter
// (subgroup `width`, up/down/xor out-of-range → own value, inactive src → own).
inline Cell coopShuffle(TS& ts, Cell var, int mode, int laneArg, int width) {
    if (!ts.fiber) return var;
    if (width <= 0 || width > 32) width = 32;
    const uint32_t warp = ts.lintid >> 5, lane = ts.lintid & 31u;
    ts.shflPub[warp * 32 + lane] = var;
    fiberYield(ts, WAIT_WARP);                 // released once every warp lane arrives
    const int laneInSub = (int)lane % width;
    const int subBase = ((int)lane / width) * width;
    int srcSub = laneInSub; bool own = false;
    if (mode == 0)      srcSub = width ? (laneArg % width) : 0;
    else if (mode == 1) { srcSub = laneInSub - laneArg; if (srcSub < 0) own = true; }
    else if (mode == 2) { srcSub = laneInSub + laneArg; if (srcSub >= width) own = true; }
    else                { srcSub = laneInSub ^ laneArg; if (srcSub >= width) own = true; }
    const int src = subBase + srcSub;
    if (own || src < 0 || src >= 32 || !(ts.shflActive[warp] & (1u << src))) return var;
    return ts.shflSnap[warp * 32 + (uint32_t)src];
}

// __syncwarp(): a warp-scoped barrier (no data exchanged).
inline void coopSyncwarp(TS& ts) { if (ts.fiber) fiberYield(ts, WAIT_WARP); }

// Warp vote — op 0=ballot 1=any 2=all. Each lane publishes its predicate; after the
// warp rendezvous, every lane reads the whole ballot (matches the interpreter).
inline Cell coopVote(TS& ts, bool pred, uint32_t memMask, int op) {
    if (!ts.fiber) return Cell::I(0);
    const uint32_t warp = ts.lintid >> 5, lane = ts.lintid & 31u;
    ts.shflPub[warp * 32 + lane] = Cell::I(pred ? 1 : 0);
    fiberYield(ts, WAIT_WARP);
    const uint32_t active = ts.shflActive[warp];
    uint32_t ballot = 0;
    for (uint32_t l = 0; l < 32; ++l)
        if ((active & (1u << l)) && ts.shflSnap[warp * 32 + l].asI() != 0) ballot |= (1u << l);
    const uint32_t masked = ballot & memMask;
    if (op == 0) return Cell::I((int64_t)masked);                       // ballot
    if (op == 1) return Cell::I(masked != 0 ? 1 : 0);                   // any
    return Cell::I(masked == (memMask & active) ? 1 : 0);              // all
}

// Warp reduce (__reduce_<op>_sync) — op 0=add 1=min 2=max 3=and 4=or 5=xor. Folds
// `value` over the warp's participating lanes (active ∩ memMask); every lane gets it.
inline Cell coopReduce(TS& ts, Cell value, uint32_t memMask, int op, bool sgn) {
    if (!ts.fiber) return value;
    const uint32_t warp = ts.lintid >> 5, lane = ts.lintid & 31u;
    ts.shflPub[warp * 32 + lane] = value;
    fiberYield(ts, WAIT_WARP);
    const uint32_t active = ts.shflActive[warp];
    bool first = true; int64_t acc = 0;
    for (uint32_t l = 0; l < 32; ++l) {
        if (!(active & (1u << l)) || !(memMask & (1u << l))) continue;
        const uint32_t raw = (uint32_t)ts.shflSnap[warp * 32 + l].asI();
        const int64_t v = sgn ? (int64_t)(int32_t)raw : (int64_t)raw;
        if (first) { acc = v; first = false; continue; }
        if (op == 0) acc = v + acc; else if (op == 1) acc = acc < v ? acc : v;
        else if (op == 2) acc = acc > v ? acc : v; else if (op == 3) acc &= v;
        else if (op == 4) acc |= v; else acc ^= v;
    }
    return Cell::I((int64_t)(uint32_t)acc);
}
#else
inline void coopBarrier(TS&) {}
inline void coopSyncwarp(TS&) {}
inline Cell coopShuffle(TS&, Cell var, int, int, int) { return var; }
inline Cell coopVote(TS&, bool, uint32_t, int) { return Cell::I(0); }
inline Cell coopReduce(TS&, Cell value, uint32_t, int, bool) { return value; }
#endif

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

// Scalar load/store of a value of type `t` at a raw host address — used for
// struct-pointer members (`p->field`), width- and sign-correct (1/2/4/8 bytes).
Cell memLoad(int64_t addr, const Type& t) {
    void* p = reinterpret_cast<void*>(addr);
    if (t.isPointer()) { int64_t v; std::memcpy(&v, p, 8); return Cell::I(v); }
    if (t.isFloating()) {
        if (t.elemBytes() == 8) { double d; std::memcpy(&d, p, 8); return Cell::F(d); }
        float f; std::memcpy(&f, p, 4); return Cell::F(static_cast<double>(f));
    }
    switch (t.elemBytes()) {
        case 1: if (t.isUnsigned) { uint8_t v; std::memcpy(&v, p, 1); return Cell::I(v); }
                { int8_t v;  std::memcpy(&v, p, 1); return Cell::I(v); }
        case 2: if (t.isUnsigned) { uint16_t v; std::memcpy(&v, p, 2); return Cell::I(v); }
                { int16_t v; std::memcpy(&v, p, 2); return Cell::I(v); }
        case 8: { int64_t v; std::memcpy(&v, p, 8); return Cell::I(v); }
        default: if (t.isUnsigned) { uint32_t v; std::memcpy(&v, p, 4); return Cell::I(v); }
                { int32_t v; std::memcpy(&v, p, 4); return Cell::I(static_cast<int64_t>(v)); }
    }
}
void memStore(int64_t addr, const Type& t, const Cell& v) {
    void* p = reinterpret_cast<void*>(addr);
    if (t.isPointer()) { int64_t x = v.asI(); std::memcpy(p, &x, 8); return; }
    if (t.isFloating()) {
        if (t.elemBytes() == 8) { double d = v.asF(); std::memcpy(p, &d, 8); }
        else { float f = static_cast<float>(v.asF()); std::memcpy(p, &f, 4); }
        return;
    }
    const int64_t x = v.asI();
    switch (t.elemBytes()) {
        case 1: { uint8_t b = (uint8_t)x;  std::memcpy(p, &b, 1); break; }
        case 2: { uint16_t b = (uint16_t)x; std::memcpy(p, &b, 2); break; }
        case 8: { std::memcpy(p, &x, 8); break; }
        default: { int32_t b = (int32_t)x; std::memcpy(p, &b, 4); break; }
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

// One kernel parameter's slot layout. A scalar/pointer param occupies one slot
// (`base`); a by-value struct param occupies one slot per member starting at
// `base`, its bytes read at each member's offset from the arg pointer. The member
// list is COPIED (not a StructDef* into the parsed module, which is freed once
// compilation finishes) so loadParams stays valid for the kernel's lifetime.
struct ParamInfo { Type type; size_t base = 0; std::vector<StructMember> members; };

class CompiledKernelImpl : public CompiledKernel {
public:
    int numParams() const override { return static_cast<int>(params_.size()); }

    bool launch(const Extent& grid, const Extent& block, void* const* args, int numArgs) override {
        if (numArgs != numParams()) return false;
        const uint32_t gt = grid.x * grid.y * grid.z;
        const uint32_t bt = block.x * block.y * block.z;

        auto initTS = [&](TS& ts, uint32_t t, uint32_t cx, uint32_t cy, uint32_t cz, Cell* shared) {
            ts.regs.resize(numSlots_);
            ts.args = args;
            ts.tid[0] = t % block.x; ts.tid[1] = (t / block.x) % block.y; ts.tid[2] = t / (block.x * block.y);
            ts.ctaid[0] = cx; ts.ctaid[1] = cy; ts.ctaid[2] = cz;
            ts.ntid[0] = block.x; ts.ntid[1] = block.y; ts.ntid[2] = block.z;
            ts.nctaid[0] = grid.x; ts.nctaid[1] = grid.y; ts.nctaid[2] = grid.z;
            ts.shared = shared;
            loadParams(ts);
        };

        // One CTA's worth of work. Barrier-free kernels run each thread to
        // completion in turn (independent, so distinct CTAs run concurrently — the
        // only cross-CTA sharing is atomicAdd, a real atomic RMW). Cooperative
        // kernels (`__shared__`/`__syncthreads`) instead run each thread as a fiber
        // that yields at the barrier; a block scheduler releases them together.
        auto runCTA = [&](uint32_t cta) {
            uint32_t cx = cta % grid.x, cy = (cta / grid.x) % grid.y, cz = cta / (grid.x * grid.y);
            if (!coop_) {
                for (uint32_t t = 0; t < bt; ++t) {
                    TS ts; initTS(ts, t, cx, cy, cz, nullptr);
                    for (auto& s : body_) { s(ts); if (ts.returned) break; }
                }
                return;
            }
            runCoopCTA(cta, cx, cy, cz, bt, initTS);
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

    // Cooperative CTA execution: one fiber per thread; a round-robin scheduler
    // resumes each fiber until it reaches __syncthreads() (or finishes), then
    // releases them all together — correct for barriers anywhere (loops, ifs),
    // no kernel transformation. Threads share one block-wide `shared` buffer.
    template <class Init>
    void runCoopCTA(uint32_t /*cta*/, uint32_t cx, uint32_t cy, uint32_t cz, uint32_t bt, Init&& initTS) {
#if VGRE_COOP_FIBERS
        const uint32_t nwarps = (bt + 31) / 32;
        std::vector<Cell> shared(sharedSlots_ ? sharedSlots_ : 1);
        std::vector<Cell> shflPub(nwarps * 32), shflSnap(nwarps * 32);
        std::vector<uint32_t> shflActive(nwarps, 0);
        std::vector<TS> ts(bt);
        std::vector<Fiber> fibers(bt);
        ucontext_t sched;
        for (uint32_t t = 0; t < bt; ++t) {
            initTS(ts[t], t, cx, cy, cz, shared.data());
            ts[t].lintid = t; ts[t].shflPub = shflPub.data(); ts[t].shflSnap = shflSnap.data(); ts[t].shflActive = shflActive.data();
            Fiber& f = fibers[t];
            f.sched = &sched; f.body = &body_; f.ts = &ts[t];
            f.stack.resize(1 << 17);   // 128 KiB per fiber (kernels are shallow)
            getcontext(&f.ctx);
            f.ctx.uc_stack.ss_sp = f.stack.data();
            f.ctx.uc_stack.ss_size = f.stack.size();
            f.ctx.uc_link = &sched;    // returning from the body lands back in the scheduler
            makecontext(&f.ctx, fiberTrampoline, 0);
            ts[t].fiber = &f;
        }
        // Drive to completion: resume every runnable fiber (to its next barrier or
        // finish), then release whatever is satisfied — the block barrier when all
        // live fibers are at __syncthreads, and each warp independently when all its
        // live lanes are at a shuffle (snapshotting the published values first so a
        // fast lane racing ahead can't clobber a value another lane hasn't read).
        auto laneLive = [&](uint32_t w, uint32_t l) {
            uint32_t t = w * 32 + l; return t < bt && !fibers[t].done;
        };
        for (;;) {
            bool ranAny = false, allDone = true;
            for (uint32_t t = 0; t < bt; ++t) {
                if (fibers[t].done) continue;
                allDone = false;
                if (fibers[t].wait != WAIT_NONE) continue;
                ranAny = true;
                g_startFiber = &fibers[t];
                swapcontext(&sched, &fibers[t].ctx);   // runs until barrier/shuffle or done
            }
            if (allDone) break;

            bool released = false;
            // Block barrier: every live fiber parked at __syncthreads.
            bool allBlock = true;
            for (uint32_t t = 0; t < bt; ++t) if (!fibers[t].done && fibers[t].wait != WAIT_BLOCK) { allBlock = false; break; }
            if (allBlock) {
                for (uint32_t t = 0; t < bt; ++t) if (!fibers[t].done) fibers[t].wait = WAIT_NONE;
                released = true;
            } else {
                // Per-warp shuffle: release each warp whose live lanes are all at a shuffle.
                for (uint32_t w = 0; w < nwarps; ++w) {
                    bool allWarp = true, anyLive = false;
                    for (uint32_t l = 0; l < 32; ++l) if (laneLive(w, l)) { anyLive = true; if (fibers[w * 32 + l].wait != WAIT_WARP) { allWarp = false; break; } }
                    if (!anyLive || !allWarp) continue;
                    uint32_t mask = 0;
                    for (uint32_t l = 0; l < 32; ++l) if (laneLive(w, l) && fibers[w * 32 + l].wait == WAIT_WARP) {
                        mask |= (1u << l);
                        shflSnap[w * 32 + l] = shflPub[w * 32 + l];   // snapshot before any reads
                    }
                    shflActive[w] = mask;
                    for (uint32_t l = 0; l < 32; ++l) if (laneLive(w, l)) fibers[w * 32 + l].wait = WAIT_NONE;
                    released = true;
                }
            }
            if (!released && !ranAny) break;   // no progress possible (divergent/UB) — avoid a hang
        }
#else
        (void)cx; (void)cy; (void)cz; (void)bt; (void)initTS;   // no fibers → cooperative path unused
#endif
    }

    // Populated by Compiler (friend-like via public setters kept minimal).
    std::vector<ParamInfo> params_;
    std::vector<StmtFn> body_;
    size_t numSlots_ = 0;
    size_t sharedSlots_ = 0;   // block-wide __shared__ Cell count
    bool coop_ = false;        // uses __shared__/__syncthreads → fiber scheduler

    void loadParams(TS& ts) const {
        for (size_t k = 0; k < params_.size(); ++k) {
            const ParamInfo& pi = params_[k];
            const char* ap = reinterpret_cast<const char*>(ts.args[k]);
            if (!pi.members.empty()) {   // by-value struct: load each member at its offset
                for (size_t i = 0; i < pi.members.size(); ++i)
                    ts.regs[pi.base + i] = memLoad(reinterpret_cast<int64_t>(ap + pi.members[i].offset), pi.members[i].type);
            } else {                     // scalar / pointer: the arg points at the value
                ts.regs[pi.base] = memLoad(reinterpret_cast<int64_t>(ap), pi.type);
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
    struct LocalArr { size_t base; Type elem; int size; std::vector<int> dims; bool isShared = false; };
    std::unordered_map<std::string, LocalArr> arrays;
    size_t nextSlot = 0;
    size_t sharedNext = 0;        // block-wide `__shared__` element count (separate address space)
    bool hasBarrier = false;      // kernel calls __syncthreads() → needs cooperative execution
    bool failed = false;
    std::string err;
    int line = 0, col = 0;

    // `__device__` helper functions callable from this kernel (name -> definition),
    // inlined at each call site (see inlineDeviceCall). `retCtx` is the per-inline
    // return context (the slot the value lands in + its type); `inlining` guards
    // against recursion (unsupported when inlining).
    std::unordered_map<std::string, const Kernel*> deviceFns;
    struct RetCtx { size_t retSlot; Type retType; };
    std::vector<RetCtx> retCtx;
    std::unordered_map<std::string, int> inlining;

    // Struct layouts (member offsets/types) come from the parsed module.
    const Module* mod = nullptr;
    const StructDef* findStruct(const std::string& n) { return mod ? mod->findStruct(n) : nullptr; }

    // A struct **value** (a local `V v;` or a by-value struct param): a contiguous
    // run of Cell slots, one per member, like a small named array.
    struct StructVar { size_t base; const StructDef* def; };
    std::unordered_map<std::string, StructVar> structVars;

    // Reserve one slot per member for a struct variable/param.
    void declareStructVar(const std::string& name, const Type& t) {
        const StructDef* def = findStruct(t.structName);
        if (!def) { fail("unknown struct '" + t.structName + "'"); return; }
        size_t base = nextSlot;
        nextSlot += def->members.size();
        structVars[name] = {base, def};
        vtype[name] = t;   // so estimateType(Ident) sees the struct type
    }

    // `v.field` on a struct value → the member's slot and type. `.ok` false (no
    // fail) when `obj` isn't a known struct value.
    struct ValMember { size_t slot; Type type; bool ok = false; };
    ValMember structValueMember(const Expr& obj, const std::string& field) {
        if (obj.kind != Expr::Ident) return {};
        auto it = structVars.find(obj.str);
        if (it == structVars.end()) return {};
        const StructDef* def = it->second.def;
        for (size_t i = 0; i < def->members.size(); ++i)
            if (def->members[i].name == field) return { it->second.base + i, def->members[i].type, true };
        fail("no member '." + field + "' in struct '" + def->name + "'");
        return {};
    }

    // A struct member accessed in memory. Two forms, matching the interpreter:
    //   `p->field`     — obj a struct-pointer expression; struct is at p.
    //   `arr[i].field` — obj is `arr[i]` with arr a struct array (struct*); the
    //                    element is at arr + i*sizeof(struct).
    // Returns the member's address closure + type; `.ok` false (no fail) when obj
    // isn't a struct member reference in memory.
    struct MemAcc { ExprFn addr; Type type; bool ok = false; };
    MemAcc structPtrMember(const Expr& member) {
        const Expr& obj = *member.args[0];
        std::function<int64_t(TS&)> base;
        const StructDef* def = nullptr;
        Type ot = estimateType(obj);
        if (ot.isPointer() && ot.base == Type::Struct && ot.ptr == 1) {   // p->field
            def = findStruct(ot.structName);
            ExprFn pv = compileExpr(obj); if (failed) return {};
            base = [pv](TS& ts) { return pv(ts).asI(); };
        } else if (obj.kind == Expr::Index) {                            // arr[i].field
            Type at = estimateType(*obj.args[0]);
            if (!(at.isPointer() && at.base == Type::Struct && at.ptr == 1)) return {};
            def = findStruct(at.structName);
            if (!def) { fail("unknown struct '" + at.structName + "'"); return {}; }
            const int64_t stride = def->size;
            ExprFn arr = compileExpr(*obj.args[0]);
            ExprFn idx = compileExpr(*obj.args[1]);
            if (failed) return {};
            base = [arr, idx, stride](TS& ts) { return arr(ts).asI() + idx(ts).asI() * stride; };
        } else {
            return {};
        }
        const StructMember* m = def ? def->find(member.str) : nullptr;
        if (!m) { fail("no member '" + member.str + "' in struct '" + (def ? def->name : "?") + "'"); return {}; }
        const int off = m->offset;
        MemAcc r; r.type = m->type; r.ok = true;
        r.addr = [base, off](TS& ts) { return Cell::I(base(ts) + off); };
        return r;
    }

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
            if (s.isShared) {
                // `__shared__` array → block-wide storage (its own address space).
                // A dynamic `extern __shared__` (arraySize 0) or scalar shared still
                // defers to the interpreter.
                if (s.arraySize <= 0 || s.isExternShared) { fail("__shared__ (dynamic/scalar) needs the interpreter tier"); return; }
                size_t base = sharedNext;
                sharedNext += (size_t)s.arraySize;
                std::vector<int> dims = s.arrayDims.empty()
                                            ? std::vector<int>{s.arraySize} : s.arrayDims;
                arrays[s.name] = LocalArr{base, s.type, s.arraySize, std::move(dims), true};
                return;
            }
            if (s.arraySize > 0) {
                // Reserve a contiguous slot run for the per-thread local array.
                size_t base = nextSlot;
                nextSlot += (size_t)s.arraySize;
                std::vector<int> dims = s.arrayDims.empty()
                                            ? std::vector<int>{s.arraySize} : s.arrayDims;
                arrays[s.name] = LocalArr{base, s.type, s.arraySize, std::move(dims), false};
                return;
            }
            if (s.type.isStruct()) { declareStructVar(s.name, s.type); return; }   // local struct value
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
            (e->str == "__syncthreads" || e->str == "__syncwarp" ||
             e->str == "__shfl_sync" || e->str == "__shfl_up_sync" ||
             e->str == "__shfl_down_sync" || e->str == "__shfl_xor_sync" ||
             e->str == "__ballot_sync" || e->str == "__any_sync" || e->str == "__all_sync" ||
             e->str == "__reduce_add_sync" || e->str == "__reduce_min_sync" || e->str == "__reduce_max_sync" ||
             e->str == "__reduce_and_sync" || e->str == "__reduce_or_sync" || e->str == "__reduce_xor_sync")) {
            hasBarrier = true;   // cooperative — run each thread as a fiber, released at the barrier/warp op
        } else if (e->kind == Expr::Call &&
                   (e->str == "__syncthreads_count" || e->str == "__syncthreads_and" ||
                    e->str == "__syncthreads_or" || e->str == "__activemask")) {
            fail("block-vote / activemask op needs the interpreter tier");
        }
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
        return [c, t, f](TS& ts) -> Cell { return c(ts).truthy() ? t(ts) : f(ts); };
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
        // v.field on a struct value (local / by-value param): read the member slot.
        ValMember vm = structValueMember(*e.args[0], e.str);
        if (failed) return {};
        if (vm.ok) { size_t sl = vm.slot; return [sl](TS& ts) { return ts.regs[sl]; }; }
        // p->field on a pointer-to-struct: load the member from memory.
        MemAcc ma = structPtrMember(e);
        if (failed) return {};
        if (ma.ok) {
            ExprFn addr = std::move(ma.addr); Type mt = ma.type;
            return [addr, mt](TS& ts) { return memLoad(addr(ts).asI(), mt); };
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
            case Expr::Member: {
                const Expr& obj = *e.args[0];
                if (obj.kind == Expr::Ident) {                       // v.field on a struct value
                    auto sv = structVars.find(obj.str);
                    if (sv != structVars.end()) {
                        const StructMember* m = sv->second.def->find(e.str);
                        if (m) return m->type;
                    }
                }
                Type ot = estimateType(obj);                          // p->field on a struct pointer
                if (ot.isPointer() && ot.base == Type::Struct && ot.ptr == 1) {
                    const StructDef* def = findStruct(ot.structName);
                    const StructMember* m = def ? def->find(e.str) : nullptr;
                    if (m) return m->type;
                }
                if (obj.kind == Expr::Index) {                         // arr[i].field on a struct array
                    Type at = estimateType(*obj.args[0]);
                    if (at.isPointer() && at.base == Type::Struct && at.ptr == 1) {
                        const StructDef* def = findStruct(at.structName);
                        const StructMember* m = def ? def->find(e.str) : nullptr;
                        if (m) return m->type;
                    }
                }
                return scalar(Type::Int);   // threadIdx/blockIdx/… builtins
            }
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
                auto df = deviceFns.find(fn);
                if (df != deviceFns.end()) return df->second->returnType;   // user __device__ helper
                if (fn == "min" || fn == "max")
                    return e.args.size() < 2 ? scalar(Type::Int) : promoteT(estimateType(*e.args[0]), estimateType(*e.args[1]));
                if (fn == "abs")       return e.args.empty() ? scalar(Type::Int) : estimateType(*e.args[0]);
                if (fn == "atomicAdd") return e.args.size() < 2 ? scalar(Type::Int) : estimateType(*e.args[1]);
                // Warp shuffle returns the shuffled variable's type; vote/reduce/count
                // return int (not the default-double for a non-`f` name).
                if ((fn == "__shfl_sync" || fn == "__shfl_up_sync" || fn == "__shfl_down_sync" ||
                     fn == "__shfl_xor_sync") && e.args.size() >= 2) return estimateType(*e.args[1]);
                if (fn == "__ballot_sync" || fn == "__any_sync" || fn == "__all_sync" ||
                    fn == "__reduce_add_sync" || fn == "__reduce_min_sync" || fn == "__reduce_max_sync" ||
                    fn == "__reduce_and_sync" || fn == "__reduce_or_sync" || fn == "__reduce_xor_sync")
                    return scalar(Type::Int);
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
            size_t base = la->base; int sz = la->size; bool sh = la->isShared;
            return [base, flat, sz, sh](TS& ts) -> Cell {
                int64_t i = flat(ts).asI();
                if (i < 0 || i >= sz) return Cell::I(0);     // OOB → 0 (no fault)
                return (sh ? ts.shared : ts.regs.data())[base + (size_t)i];
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
            size_t base = la->base; int sz = la->size; Type pt = la->elem; bool sh = la->isShared;
            return [base, flat, value, sz, pt, sh](TS& ts) {
                int64_t i = flat(ts).asI();
                if (i < 0 || i >= sz) return;                // OOB → drop (no fault)
                (sh ? ts.shared : ts.regs.data())[base + (size_t)i] = coerce(value(ts), pt);
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

    // Element size a pointer to `pointee` strides by (struct size from the layout
    // table; the scalar/pointer width otherwise).
    int64_t strideOf(const Type& pointee) {
        if (pointee.base == Type::Struct && pointee.ptr == 0) {
            const StructDef* def = findStruct(pointee.structName);
            return def ? def->size : 1;
        }
        return pointee.elemBytes();
    }

    ExprFn compileUnary(const Expr& e) {
        if (e.str == "pre++" || e.str == "pre--" || e.str == "post++" || e.str == "post--")
            return compileIncDec(e);
        // `*p` — dereference: load the pointee value at the pointer's address.
        if (e.str == "*") {
            Type pt = estimateType(*e.args[0]);
            if (!pt.isPointer()) { fail("cannot dereference a non-pointer"); return {}; }
            ExprFn a = compileExpr(*e.args[0]);
            if (failed) return {};
            Type pointee = pt; pointee.ptr -= 1;
            return [a, pointee](TS& ts) { return memLoad(a(ts).asI(), pointee); };
        }
        ExprFn a = compileExpr(*e.args[0]);
        if (failed) return {};
        if (e.str == "+") return a;
        std::string op = e.str;
        return [a, op](TS& ts) -> Cell {
            Cell v = a(ts);
            if (op == "-") return v.isFloat ? Cell::F(-v.f) : Cell::I(-v.i);
            if (op == "!") return Cell::I(v.truthy() ? 0 : 1);
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
                return Cell::I((isAnd ? (a(ts).truthy() && b(ts).truthy())
                                      : (a(ts).truthy() || b(ts).truthy())) ? 1 : 0);
            };
        }
        // Pointer arithmetic: `p ± i` scales the integer index by the pointee size
        // (element-scaled 64-bit addressing), and `p - q` is the element-count
        // difference — matching C and the interpreter (a raw byte add would be
        // silently wrong).
        Type ta = estimateType(*e.args[0]), tb = estimateType(*e.args[1]);
        if ((op == "+" || op == "-") && (ta.isPointer() || tb.isPointer())) {
            if (op == "-" && ta.isPointer() && tb.isPointer()) {          // p - q → ptrdiff
                Type pointee = ta; pointee.ptr -= 1;
                const int64_t stride = strideOf(pointee);
                return [a, b, stride](TS& ts) -> Cell { return Cell::I((a(ts).asI() - b(ts).asI()) / stride); };
            }
            const bool aPtr = ta.isPointer();
            if (op == "-" && !aPtr) { fail("cannot subtract a pointer from an integer"); return {}; }
            ExprFn ptr = aPtr ? a : b, idx = aPtr ? b : a;              // i + p is commutative
            Type pointee = (aPtr ? ta : tb); pointee.ptr -= 1;
            const int64_t stride = strideOf(pointee);
            const bool sub = (op == "-");
            return [ptr, idx, stride, sub](TS& ts) -> Cell {
                int64_t off = (int64_t)(int32_t)idx(ts).asI() * stride;   // index is 32-bit (mul.wide.s32)
                return Cell::I(sub ? ptr(ts).asI() - off : ptr(ts).asI() + off);
            };
        }
        // Compute at the operands' common type (comparisons compare there but
        // yield int, handled by binop); narrowResult wraps the result to its own
        // static type.
        const Type ct = promoteT(ta, tb);
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
        if (lhs.kind == Expr::Member) {
            // `v.field = …` on a struct value → write the member slot.
            ValMember vm = structValueMember(*lhs.args[0], lhs.str);
            if (failed) return {};
            if (vm.ok) {
                size_t sl = vm.slot; Type mt = vm.type;
                ExprFn value = rhs;
                if (op != "=") {
                    std::string bop = op.substr(0, op.size() - 1);
                    Type ct = promoteT(mt, estimateType(*e.args[1]));
                    ExprFn cur = [sl](TS& ts) { return ts.regs[sl]; };
                    value = makeBinary(cur, rhs, bop, ct);
                }
                return [sl, mt, value](TS& ts) -> Cell { ts.regs[sl] = coerce(value(ts), mt); return ts.regs[sl]; };
            }
            // `p->field = …` on a struct pointer → store the member at (pointer + offset).
            MemAcc ma = structPtrMember(lhs);
            if (failed) return {};
            if (!ma.ok) { fail("unsupported assignment target (member)"); return {}; }
            ExprFn addr = std::move(ma.addr); Type mt = ma.type;
            ExprFn value = rhs;
            if (op != "=") {
                std::string bop = op.substr(0, op.size() - 1);
                Type ct = promoteT(mt, estimateType(*e.args[1]));
                ExprFn cur = [addr, mt](TS& ts) { return memLoad(addr(ts).asI(), mt); };
                value = makeBinary(cur, rhs, bop, ct);
            }
            return [addr, mt, value](TS& ts) -> Cell {
                Cell v = coerce(value(ts), mt);
                memStore(addr(ts).asI(), mt, v);
                return v;
            };
        }
        if (lhs.kind == Expr::Unary && lhs.str == "*") {
            // `*p = …` (and compound) — store the pointee value at p's address.
            Type pt = estimateType(*lhs.args[0]);
            if (!pt.isPointer()) { fail("cannot dereference a non-pointer"); return {}; }
            Type pointee = pt; pointee.ptr -= 1;
            ExprFn pv = compileExpr(*lhs.args[0]);
            if (failed) return {};
            ExprFn value = rhs;
            if (op != "=") {
                std::string bop = op.substr(0, op.size() - 1);
                Type ct = promoteT(pointee, estimateType(*e.args[1]));
                ExprFn cur = [pv, pointee](TS& ts) { return memLoad(pv(ts).asI(), pointee); };
                value = makeBinary(cur, rhs, bop, ct);
            }
            return [pv, pointee, value](TS& ts) -> Cell {
                Cell v = coerce(value(ts), pointee);
                memStore(pv(ts).asI(), pointee, v);
                return v;
            };
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
        if (op == "&&") return Cell::I((x.truthy() && y.truthy()) ? 1 : 0);
        if (op == "||") return Cell::I((x.truthy() || y.truthy()) ? 1 : 0);
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
            size_t base = la->base; int sz = la->size; Type pt = la->elem; bool sh = la->isShared;
            return [base, flat, val, sz, pt, sh](TS& ts) -> Cell {
                int64_t i = flat(ts).asI();
                if (i < 0 || i >= sz) return Cell::I(0);
                // Per-thread slot, or block-shared: either way fibers are cooperative
                // (one runs at a time, never mid-RMW), so a plain RMW is atomic here.
                Cell* store = sh ? ts.shared : ts.regs.data();
                Cell old = store[base + (size_t)i];
                store[base + (size_t)i] = coerce(binop("+", old, val(ts)), pt);
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

    // Inline a call to a __device__ helper: bind arg closures to fresh param slots,
    // compile the body in a fresh name scope (disjoint slot range) with a return
    // context, then run it with ts.returned save/restored so the callee's `return`
    // unwinds only its own body — not the caller's control flow. Non-recursive.
    ExprFn inlineDeviceCall(const Kernel& fn, const Expr& call) {
        if (inlining.count(fn.name)) { fail("recursive __device__ function '" + fn.name + "' unsupported on the compiled tier"); return {}; }
        if (inlining.size() > 64) { fail("__device__ inline depth exceeded (recursion?)"); return {}; }
        if (call.args.size() != fn.params.size()) { fail("wrong argument count for '" + fn.name + "'"); return {}; }

        // Args are compiled in the CALLER's scope.
        std::vector<ExprFn> args;
        for (auto& a : call.args) { args.push_back(compileExpr(*a)); if (failed) return {}; }
        std::vector<Type> ptypes; for (auto& p : fn.params) ptypes.push_back(p.type);

        // Fresh name scope for the callee's params + locals (their slots are a fresh,
        // disjoint range in the same per-thread register file).
        auto savedSlot = std::move(slot);     slot.clear();
        auto savedVtype = std::move(vtype);   vtype.clear();
        auto savedArrays = std::move(arrays); arrays.clear();

        std::vector<size_t> pslots;
        for (auto& p : fn.params) pslots.push_back(declare(p.name, p.type));
        for (auto& s : fn.body) { scan(*s); if (failed) break; }   // reserve callee local/array slots; rejects __shared__/barriers

        const size_t retSlot = nextSlot++;
        const Type rt = fn.returnType;
        StmtFn bodyFn;
        if (!failed) {
            retCtx.push_back({retSlot, rt});
            inlining[fn.name] = 1;
            bodyFn = compileBody(fn.body);
            inlining.erase(fn.name);
            retCtx.pop_back();
        }

        // Restore the caller's scope.
        slot = std::move(savedSlot);
        vtype = std::move(savedVtype);
        arrays = std::move(savedArrays);
        if (failed) return {};

        const bool isVoid = (rt.base == Type::Void);
        const Cell zero = rt.isFloating() ? Cell::F(0.0) : Cell::I(0);
        return [args, pslots, ptypes, bodyFn, retSlot, rt, isVoid, zero](TS& ts) -> Cell {
            std::vector<Cell> av; av.reserve(args.size());
            for (auto& a : args) av.push_back(a(ts));
            for (size_t i = 0; i < pslots.size(); ++i) ts.regs[pslots[i]] = coerce(av[i], ptypes[i]);
            ts.regs[retSlot] = zero;                     // defined value if the body falls through
            const bool saved = ts.returned; ts.returned = false;
            if (bodyFn) bodyFn(ts);
            ts.returned = saved;
            return isVoid ? Cell::I(0) : ts.regs[retSlot];
        };
    }

    ExprFn compileCall(const Expr& e) {
        const std::string& fn = e.str;
        if (fn == "__syncthreads") return [](TS& ts) -> Cell { coopBarrier(ts); return Cell::I(0); };
        // Warp shuffle: __shfl[_up|_down|_xor]_sync(mask, var, laneArg[, width]).
        if ((fn == "__shfl_sync" || fn == "__shfl_up_sync" || fn == "__shfl_down_sync" || fn == "__shfl_xor_sync") &&
            (e.args.size() == 3 || e.args.size() == 4)) {
            int mode = fn == "__shfl_sync" ? 0 : fn == "__shfl_up_sync" ? 1 : fn == "__shfl_down_sync" ? 2 : 3;
            ExprFn var = compileExpr(*e.args[1]);
            ExprFn laneArg = compileExpr(*e.args[2]);
            ExprFn widthE = e.args.size() == 4 ? compileExpr(*e.args[3]) : ExprFn{};
            if (failed) return {};
            return [var, laneArg, widthE, mode](TS& ts) -> Cell {
                Cell v = var(ts);
                int la = static_cast<int>(laneArg(ts).asI());
                int w = widthE ? static_cast<int>(widthE(ts).asI()) : 32;
                return coopShuffle(ts, v, mode, la, w);
            };
        }
        if (fn == "__syncwarp") return [](TS& ts) -> Cell { coopSyncwarp(ts); return Cell::I(0); };
        // Warp vote: __ballot_sync / __any_sync / __all_sync(mask, pred).
        if ((fn == "__ballot_sync" || fn == "__any_sync" || fn == "__all_sync") && e.args.size() == 2) {
            int op = fn == "__ballot_sync" ? 0 : fn == "__any_sync" ? 1 : 2;
            ExprFn mask = compileExpr(*e.args[0]);
            ExprFn pred = compileExpr(*e.args[1]);
            if (failed) return {};
            return [mask, pred, op](TS& ts) -> Cell {
                return coopVote(ts, pred(ts).truthy(), (uint32_t)mask(ts).asI(), op);
            };
        }
        // Warp reduce: __reduce_<op>_sync(mask, value).
        if ((fn == "__reduce_add_sync" || fn == "__reduce_min_sync" || fn == "__reduce_max_sync" ||
             fn == "__reduce_and_sync" || fn == "__reduce_or_sync" || fn == "__reduce_xor_sync") && e.args.size() == 2) {
            int op = fn == "__reduce_add_sync" ? 0 : fn == "__reduce_min_sync" ? 1 : fn == "__reduce_max_sync" ? 2
                   : fn == "__reduce_and_sync" ? 3 : fn == "__reduce_or_sync" ? 4 : 5;
            const bool sgn = !estimateType(*e.args[1]).isUnsigned;
            ExprFn mask = compileExpr(*e.args[0]);
            ExprFn val = compileExpr(*e.args[1]);
            if (failed) return {};
            return [mask, val, op, sgn](TS& ts) -> Cell {
                return coopReduce(ts, val(ts), (uint32_t)mask(ts).asI(), op, sgn);
            };
        }
        if (fn == "atomicAdd" && e.args.size() == 2) return compileAtomicAdd(e);
        auto dfit = deviceFns.find(fn);
        if (dfit != deviceFns.end()) return inlineDeviceCall(*dfit->second, e);   // user __device__ helper
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
            if (fn == "exp2f"  || fn == "exp2")  return [a](TS& ts) { return Cell::F(std::exp2(a(ts).asF())); };
            if (fn == "log2f"  || fn == "log2")  return [a](TS& ts) { return Cell::F(std::log2(a(ts).asF())); };
            if (fn == "tanhf"  || fn == "tanh")  return [a](TS& ts) { return Cell::F(std::tanh(a(ts).asF())); };
            if (fn == "erff"   || fn == "erf")   return [a](TS& ts) { return Cell::F(std::erf(a(ts).asF())); };
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
                if (s.type.isStruct()) {
                    // `V v;` — members default-zero (regs are zero-initialised).
                    // `V q = v;` — copy each member slot from the source struct.
                    auto dit = structVars.find(s.name);
                    if (dit == structVars.end()) { fail("struct variable not reserved"); return {}; }
                    if (!s.expr) return [](TS&) {};
                    if (s.expr->kind != Expr::Ident) { fail("a struct can only be copy-initialised from a struct variable"); return {}; }
                    auto sit = structVars.find(s.expr->str);
                    if (sit == structVars.end()) { fail("use of undeclared struct '" + s.expr->str + "'"); return {}; }
                    if (sit->second.def != dit->second.def) { fail("struct copy type mismatch"); return {}; }
                    size_t db = dit->second.base, sb = sit->second.base, cnt = dit->second.def->members.size();
                    return [db, sb, cnt](TS& ts) { for (size_t i = 0; i < cnt; ++i) ts.regs[db + i] = ts.regs[sb + i]; };
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
            case Stmt::Return: {
                // At kernel level, `return` just stops the thread. Inside an inlined
                // __device__ body, stash the returned value in the call's slot first
                // (the call closure saves/restores ts.returned so only the callee's
                // body unwinds, not the caller's control flow).
                if (retCtx.empty()) return [](TS& ts) { ts.returned = true; };
                RetCtx ctx = retCtx.back();
                ExprFn re = s.expr ? compileExpr(*s.expr) : ExprFn();
                if (failed) return {};
                return [ctx, re](TS& ts) {
                    if (re) ts.regs[ctx.retSlot] = coerce(re(ts), ctx.retType);
                    ts.returned = true;
                };
            }
            case Stmt::Empty: return [](TS&) {};
            case Stmt::If: {
                ExprFn cond = compileExpr(*s.expr);
                StmtFn then_ = compileBody(s.body);
                StmtFn else_ = s.elseBody.empty() ? StmtFn() : compileBody(s.elseBody);
                if (failed) return {};
                return [cond, then_, else_](TS& ts) {
                    if (cond(ts).truthy()) then_(ts);
                    else if (else_) else_(ts);
                };
            }
            case Stmt::While: {
                ExprFn cond = compileExpr(*s.expr);
                StmtFn body = compileBody(s.body);
                if (failed) return {};
                return [cond, body](TS& ts) {
                    while (!ts.returned && cond(ts).truthy()) body(ts);
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
                    while (!ts.returned && (!cond || cond(ts).truthy())) {
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

// Shared driver: params occupy the first slots, scan reserves local/array slots,
// then each top-level statement is lowered to a closure.
static std::unique_ptr<CompiledKernel> finishCompile(Compiler& c, const Kernel& k, std::string& err) {
    auto impl = std::unique_ptr<CompiledKernelImpl>(new CompiledKernelImpl());
    for (const Param& p : k.params) {
        std::string nm = p.name.empty() ? ("__arg" + std::to_string(c.nextSlot)) : p.name;
        ParamInfo pi; pi.type = p.type;
        if (p.type.isStruct()) {                    // by-value struct param → member slots
            c.declareStructVar(nm, p.type);
            if (c.failed) { err = c.err; return nullptr; }
            pi.base = c.structVars[nm].base;
            pi.members = c.structVars[nm].def->members;   // copy (module is freed after compile)
        } else {
            pi.base = c.declare(nm, p.type);
        }
        impl->params_.push_back(pi);
    }
    for (auto& s : k.body) c.scan(*s);
    if (c.failed) { err = c.err; return nullptr; }

    for (auto& s : k.body) {
        StmtFn f = c.compileStmt(*s);
        if (c.failed) { err = c.err; return nullptr; }
        impl->body_.push_back(std::move(f));
    }
    impl->numSlots_ = c.nextSlot;
    impl->sharedSlots_ = c.sharedNext;
    impl->coop_ = (c.sharedNext > 0 || c.hasBarrier);
#if !VGRE_COOP_FIBERS
    if (impl->coop_) { err = "__shared__/__syncthreads needs the interpreter tier on this host"; return nullptr; }
#endif
    return impl;
}

}  // namespace

std::unique_ptr<CompiledKernel> CompiledKernel::compile(const Kernel& k, std::string& err) {
    Compiler c;   // no module context: calls to __device__ helpers won't resolve
    return finishCompile(c, k, err);
}

std::unique_ptr<CompiledKernel> CompiledKernel::compileSource(const std::string& source,
                                                             const std::string& name,
                                                             std::string& err) {
    ParseResult pr = parse(source);
    if (!pr.ok) { err = pr.error; return nullptr; }
    const Kernel* target = nullptr;
    if (!name.empty()) {
        for (auto& kp : pr.module->kernels) if (kp->name == name) { target = kp.get(); break; }
    } else {
        for (auto& kp : pr.module->kernels) if (kp->isGlobal) { target = kp.get(); break; }  // entry point
        if (!target && !pr.module->kernels.empty()) target = pr.module->kernels.front().get();
    }
    if (!target) { err = "kernel not found"; return nullptr; }

    Compiler c;
    c.mod = pr.module.get();              // struct layouts for p->field
    for (auto& kp : pr.module->kernels)   // __device__ helpers this kernel may call
        if (kp->isDeviceCallable() && kp.get() != target) c.deviceFns[kp->name] = kp.get();
    return finishCompile(c, *target, err);
}

}  // namespace frontend
}  // namespace compiler
}  // namespace vgre
