// PTX single-step interpreter — the execution core of VGRE's CUDA-GDB support
// (missingFeatures §6.1). Unlike the JIT (which compiles kernels to native code
// and therefore cannot stop between PTX instructions), this interpreter executes
// a kernel's PTX **one instruction at a time**, per thread, with:
//
//   • per-thread virtual register files (%r/%rd/%f/%fd/%p…) and a per-thread PC
//   • real global memory (kernel pointer args are host addresses), per-CTA
//     .shared memory, and kernel .param space
//   • bar.sync-aware cooperative scheduling of the CTA's threads
//   • instruction-granular breakpoints, single-step, and register/memory
//     inspection — the primitives a GDB stub needs
//
// Coverage is the arithmetic/control/memory core emitted by nvcc for compute
// kernels (mov/ld/st/cvta/cvt, add/sub/mul/mad/fma/div/rem/min/max/neg/abs,
// and/or/xor/not/shl/shr, setp/selp/set, predicated bra, bar.sync, ret,
// %tid/%ntid/%ctaid/%nctaid/%laneid, mul.wide, sqrt/rsqrt/ex2/lg2/sin/cos).
// Anything outside it throws — no silent skips.
//
// Grid semantics: CTAs are executed sequentially (this is a debugger, not a
// throughput engine); threads within the running CTA are all live and
// individually addressable, which is what cuda-gdb models too.
#ifndef VGRE_DEBUG_PTX_INTERPRETER_H
#define VGRE_DEBUG_PTX_INTERPRETER_H

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace vgre {
namespace debug {

// One parsed PTX instruction (post-tokenization).
struct PtxInstr {
    std::string pred;               // predicate register ("" = none), e.g. "%p1"
    bool        predNeg = false;    // @!%p
    std::string op;                 // full dotted opcode, e.g. "mad.lo.s32"
    std::vector<std::string> args;  // raw operand strings
    int line = 0;                   // 1-based line in the original PTX (for listings)
    std::string text;               // original source text (trimmed)
};

struct PtxParam {
    std::string name;
    int sizeBytes = 0;              // 4 or 8
    int offset = 0;                 // byte offset in the param block
};

struct PtxKernel {
    std::string name;
    std::vector<PtxParam> params;
    std::vector<PtxInstr> code;                 // PC = index into this vector
    std::map<std::string, int> labels;          // label -> PC
    std::map<std::string, int> sharedVars;      // .shared name -> arena offset
    int sharedBytes = 0;
    std::map<std::string, int> localVars;       // .local name -> per-thread arena offset
    int localBytes = 0;
};

// 64-bit raw register value; interpretation (s32/u64/f32/…) is per-instruction.
struct RegVal {
    uint64_t u = 0;
};

// Why the machine stopped.
enum class StopReason { Running, Breakpoint, Step, Exited, Fault };

// A CUDA-style 3D extent (grid or block). Components default to 1 so a plain
// 1D launch is {n,1,1}.
struct Dim3 {
    int x = 1, y = 1, z = 1;
    int total() const { return x * y * z; }
};

class PtxInterpreter {
public:
    // Parses `ptx` and prepares kernel `entry` for execution. Throws
    // std::runtime_error on parse failure or if the kernel is absent.
    PtxInterpreter(const std::string& ptx, const std::string& entry);

    // Default-constructed: no kernel yet. Used by the non-throwing canParse(),
    // which parses into a probe via tryParse() without going through the
    // throwing constructor (see the macOS note on the static entry points).
    PtxInterpreter() = default;

    // Launch configuration + kernel arguments (CUDA-style: one pointer per
    // .param, dereferenced to the param's size). Resets all execution state.
    // The 1D overload is a convenience for {gridX,1,1} × {blockX,1,1}; the 3D
    // overload exposes the full grid/block extents (real %tid/%ctaid .y/.z).
    void launch(int gridX, int blockX, void* const* args, int numArgs);
    void launch(const Dim3& grid, const Dim3& block, void* const* args, int numArgs);

    // ── Non-throwing entry points for other components ───────────────────────
    // These handle every exception INSIDE this translation unit (returning a
    // bool) so nothing propagates across the library boundary to a caller in
    // another component. That matters on macOS, where an exception thrown here
    // and caught in a different library fails to unwind at all (even catch(...)),
    // aborting the process — so the backend calls these instead of constructing a
    // PtxInterpreter + catching.
    //   canParse:       is `entry` a parseable kernel in `ptx`? (used at prepare)
    //   runKernel:      launch + run the whole grid to exit; true on success
    //   runKernelRange: run CTAs [begin,end) to exit (the parallel backend path)
    static bool canParse(const std::string& ptx, const std::string& entry) noexcept;
    static bool runKernel(const std::string& ptx, const std::string& entry,
                          const Dim3& grid, const Dim3& block,
                          void* const* args, int numArgs) noexcept;
    static bool runKernelRange(const std::string& ptx, const std::string& entry,
                               const Dim3& grid, const Dim3& block,
                               void* const* args, int numArgs,
                               int ctaBegin, int ctaEnd) noexcept;

    // ── Bulk (non-debug) execution ───────────────────────────────────────────
    // Set up launch state, then run the CTA sub-range [ctaBegin, ctaEnd) to
    // completion (no breakpoints). Distinct CTAs are independent, so several
    // instances can each run a disjoint range on different threads for a
    // multi-core speedup — this is how the backend parallelises the interpreter
    // tier. Completely separate from resume()/stepThread() (the debugger surface),
    // which stay single-instance and sequential. Returns true if the range ran to
    // exit; throws on a malformed launch or a barrier deadlock.
    bool runCtaRange(const Dim3& grid, const Dim3& block, void* const* args,
                     int numArgs, int ctaBegin, int ctaEnd);

    // ── Execution (debugger surface) ─────────────────────────────────────────
    // Run until a breakpoint fires on any thread, or the whole grid exits.
    StopReason resume();
    // Single-step one instruction on `thread` (global thread id within the
    // current CTA). Other threads do not advance.
    StopReason stepThread(int thread);

    void setBreakpoint(int pc)   { breakpoints_.insert(pc); }
    void clearBreakpoint(int pc) { breakpoints_.erase(pc); }
    bool hasBreakpoint(int pc) const { return breakpoints_.count(pc) != 0; }

    // ── Inspection ───────────────────────────────────────────────────────────
    const PtxKernel& kernel() const { return kernel_; }
    int  numThreads() const { return blockTotal_; }  // threads per CTA (x*y*z)
    int  currentCta() const { return cta_; }         // linear CTA index
    bool exited() const { return exited_; }
    int  stoppedThread() const { return stoppedThread_; }  // thread that hit the bp/step
    int  pcOf(int thread) const;
    bool threadExited(int thread) const;

    // Ordered register names of `thread` (deterministic: declaration order).
    std::vector<std::string> registerNames() const { return regNames_; }
    bool     readRegister(int thread, const std::string& name, uint64_t& out) const;
    bool     writeRegister(int thread, const std::string& name, uint64_t v);

    // Global memory = this process's memory; bounds are the debuggee's problem,
    // exactly like a real gdbserver. Returns false on a null/unmapped-page probe.
    static bool readGlobal(uint64_t addr, void* dst, size_t n);
    static bool writeGlobal(uint64_t addr, const void* src, size_t n);

private:
    struct Thread {
        int pc = 0;
        bool done = false;
        bool atBarrier = false;
        bool atShfl = false;       // parked at a warp-shuffle rendezvous
        uint64_t shflVal = 0;      // the value this lane offers to the shuffle
        std::map<std::string, RegVal> regs;
        std::map<std::string, bool>   preds;
        std::vector<char> local;   // per-thread .local scratch (sized to localBytes)
    };

    // A thread that can still execute an instruction right now (not exited, not
    // parked at a barrier or a warp-shuffle rendezvous).
    static bool runnable(const Thread& t) { return !t.done && !t.atBarrier && !t.atShfl; }

    // Non-throwing parse: fills kernel_/regNames_ and returns true, or leaves a
    // human-readable reason in `err` and returns false. Reports every malformed
    // -PTX condition by return value rather than by throwing, so the parse path
    // never raises an exception that must unwind to a caller — the mechanism the
    // static canParse() relies on (a thrown-then-caught exception aborts the
    // process on macOS when two libc++abi runtimes are present, defeating even
    // catch(...)). parse() is the throwing wrapper kept for the debugger API.
    bool tryParse(const std::string& ptx, const std::string& entry, std::string& err);
    void parse(const std::string& ptx, const std::string& entry);
    // Validate the config and load grid/block extents + the packed param block.
    // Shared by launch() (debugger entry) and runCtaRange() (parallel backend).
    void setupLaunch(const Dim3& grid, const Dim3& block, void* const* args, int numArgs);
    // Executes exactly one instruction on t; returns false if t blocked at a
    // barrier without advancing.
    bool execOne(Thread& t, int tid);
    void releaseBarrierIfReady();
    // When every active lane of `anyTid`'s warp has reached the shuffle, perform
    // the cross-lane exchange and advance them all (a warp-wide rendezvous).
    void releaseShflIfReady(int anyTid);
    void startCta(int cta);
    bool ctaFinished() const;

    uint64_t evalOperand(Thread& t, int tid, const std::string& s, int sizeBytes) const;
    RegVal&  reg(Thread& t, const std::string& name);
    uint64_t loadFrom(Thread& t, int tid, const std::string& memRef, const std::string& space,
                      int size);
    void storeTo(Thread& t, int tid, const std::string& memRef, const std::string& space,
                 int size, uint64_t val);

    PtxKernel kernel_;
    std::vector<std::string> regNames_;

    // 3D launch geometry. gridDim_ = %nctaid.{x,y,z}, blockDim_ = %ntid.{x,y,z}.
    // cta_ is the linear CTA counter (0..gridTotal_); ctaIdx_ is its %ctaid.{x,y,z}.
    int gridDim_[3]  = {1, 1, 1};
    int blockDim_[3] = {1, 1, 1};
    int gridTotal_ = 0;   // gridDim_ x*y*z
    int blockTotal_ = 0;  // blockDim_ x*y*z
    int cta_ = 0;
    int ctaIdx_[3] = {0, 0, 0};
    std::vector<uint8_t> paramBlock_;
    std::vector<uint8_t> shared_;
    std::vector<Thread> threads_;
    std::set<int> breakpoints_;
    bool exited_ = true;
    int stoppedThread_ = 0;
};

}  // namespace debug
}  // namespace vgre

#endif  // VGRE_DEBUG_PTX_INTERPRETER_H
