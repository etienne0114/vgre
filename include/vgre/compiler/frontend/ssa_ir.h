// Tier-2 backend — VGRE-IR (SSA), increment 1: the IR data model + AST→SSA lowering
// + a well-formedness verifier + a reference evaluator. This is the foundation the
// optimizer passes (const-fold/DCE/GVN/LICM), linear-scan register allocation, and
// the x86-64/AArch64 machine-code emitter layer onto in later increments. Built from
// the AST (like the Tier-1 compiled backend), not from PTX. See docs/zeroBurdenRoadmap.md.
//
// Subset so far (increments 1–2): the scalar per-thread kernel shape — int/float/
// double scalars and pointers, threadIdx/blockIdx/blockDim, arithmetic/bitwise/
// compare, ternary (→ select), inc/dec, indexed load/store, casts, unary math
// intrinsics, AND full structured control flow — `if`/`if-else`, `for`, `while` —
// with proper SSA phi insertion (Braun et al.) at merges and loop headers. Still to
// come: do-while/switch/break/continue, then the optimizer passes and native
// emission. Anything outside the subset makes compile() return null (callers fall back).
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

    // Parse `source`, lower the named kernel to SSA, and verify it. Returns null
    // (with `err` set) on a parse error or an out-of-subset construct.
    static std::unique_ptr<SsaProgram> compile(const std::string& source,
                                               const std::string& name, std::string& err);

    // Reference-evaluate the SSA over a CUDA-style launch, one thread at a time
    // (the correctness oracle for the IR + lowering until native emission lands).
    bool launch(Extent grid, Extent block, void* const* args, int numArgs);

    // Human-readable IR dump (for tests / debugging).
    std::string dump() const;

    // Counts, for tests: basic blocks and SSA values in the lowered function.
    int numBlocks() const;
    int numValues() const;

private:
    SsaProgram();
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace frontend
}  // namespace compiler
}  // namespace vgre

#endif  // VGRE_COMPILER_FRONTEND_SSA_IR_H
