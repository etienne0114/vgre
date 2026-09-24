// Tier-2 backend — VGRE-IR (SSA), increment 1: the IR data model + AST→SSA lowering
// + a well-formedness verifier + a reference evaluator. This is the foundation the
// optimizer passes (const-fold/DCE/GVN/LICM), linear-scan register allocation, and
// the x86-64/AArch64 machine-code emitter layer onto in later increments. Built from
// the AST (like the Tier-1 compiled backend), not from PTX. See docs/zeroBurdenRoadmap.md.
//
// Subset: the scalar per-thread kernel shape — int/float/double scalars and pointers,
// threadIdx/blockIdx/blockDim, arithmetic/bitwise/compare, ternary (→ select), inc/dec,
// indexed load/store, casts, unary + binary math intrinsics, and FULL control flow —
// `if`/`if-else`, `for`, `while`, `do-while`, `break`, `continue`, and `switch`
// (C fall-through) — with proper SSA phi insertion (Braun et al.). compile() also runs
// the optimizer (const-fold / global dominator-scoped GVN / LICM / DCE) unless
// optimize=false, then emits register-allocated x86-64 machine code on Linux/x86-64
// (the portable evaluator runs it everywhere else). Anything outside the subset makes
// compile() return null (callers fall back to another tier).
#ifndef VGRE_COMPILER_FRONTEND_SSA_IR_H
#define VGRE_COMPILER_FRONTEND_SSA_IR_H

#include "vgre/compiler/frontend/compiled_kernel.h"   // Extent

#include <memory>
#include <string>

namespace vgre {
namespace compiler {
namespace frontend {

// An SSA program for one kernel: parsed, lowered to VGRE-IR, and verified.
class SsaProgram {
public:
    ~SsaProgram();

    // Parse `source`, lower the named kernel to SSA, verify it, and (by default) run
    // the optimizer passes (const-fold / local GVN / DCE). Returns null (with `err`
    // set) on a parse error or an out-of-subset construct. `optimize=false` keeps the
    // raw lowering (used by tests to check the passes preserve semantics + shrink IR).
    static std::unique_ptr<SsaProgram> compile(const std::string& source,
                                               const std::string& name, std::string& err,
                                               bool optimize = true);

    // Reference-evaluate the SSA over a CUDA-style launch, one thread at a time
    // (the correctness oracle for the IR + lowering until native emission lands).
    bool launch(Extent grid, Extent block, void* const* args, int numArgs);

    // True if the last compile produced native machine code (x86-64/Linux) that
    // launch() will run; false means launch() uses the portable evaluator.
    bool usedNative() const;

    // Human-readable IR dump (for tests / debugging).
    std::string dump() const;

    // Counts, for tests: basic blocks, total SSA values, and live instructions
    // (those still referenced by a block after optimization).
    int numBlocks() const;
    int numValues() const;
    int liveInsts() const;

private:
    SsaProgram();
    struct Impl;
    std::unique_ptr<Impl> p_;
};

// Validate the AArch64 native emitter's instruction ENCODINGS against known-good bytes
// (assembled by llvm-mc, baked in). Runs on any host — the encoder is arch-independent
// byte generation — so ARM codegen is checked even on an x86-64 dev/CI machine. Returns
// true if every encoding matches, else sets `err`. (Execution is validated separately
// on real ARM: the macos-arm64 CI job's differential SsaIr test with VGRE_SSA_ARM_NATIVE.)
bool arm64EncSelfTest(std::string& err);

}  // namespace frontend
}  // namespace compiler
}  // namespace vgre

#endif  // VGRE_COMPILER_FRONTEND_SSA_IR_H
