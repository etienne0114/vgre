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

    // Launch configuration + kernel arguments (CUDA-style: one pointer per
    // .param, dereferenced to the param's size). Resets all execution state.
    // The 1D overload is a convenience for {gridX,1,1} × {blockX,1,1}; the 3D
    // overload exposes the full grid/block extents (real %tid/%ctaid .y/.z).
    void launch(int gridX, int blockX, void* const* args, int numArgs);
    void launch(const Dim3& grid, const Dim3& block, void* const* args, int numArgs);

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
        std::map<std::string, RegVal> regs;
        std::map<std::string, bool>   preds;
    };

    void parse(const std::string& ptx, const std::string& entry);
    // Executes exactly one instruction on t; returns false if t blocked at a
    // barrier without advancing.
    bool execOne(Thread& t, int tid);
    void releaseBarrierIfReady();
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
