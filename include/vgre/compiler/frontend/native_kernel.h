#ifndef VGRE_COMPILER_FRONTEND_NATIVE_KERNEL_H
#define VGRE_COMPILER_FRONTEND_NATIVE_KERNEL_H

// Native execution tier (the zero-burden "speed" step): a from-scratch, LLVM-free
// machine-code JIT. Unlike the Tier-1 compiled backend (which runs bound closures
// per thread), this emits real host machine code for a kernel and executes it —
// the foundation of the copy-and-patch / SSA backends on the roadmap.
//
// This first cut is a direct x86-64 (System V, Linux) emitter for the elementwise
// idiom that dominates real compute kernels:
//
//     int i = blockIdx.x*blockDim.x + threadIdx.x;
//     if (i < n) { p[i] = <float expression over q[i], scalars, literals>; ... }
//
// Anything outside that subset — or any non-x86-64/Linux host — makes
// compileSource return nullptr, so callers fall back to the compiled/interpreter
// tiers. It shares the parser + AST with the other tiers, so correctness is
// checked by differential tests against them.

#include "vgre/compiler/frontend/ast.h"
#include "vgre/compiler/frontend/compiled_kernel.h"   // Extent

#include <memory>
#include <string>

namespace vgre {
namespace compiler {
namespace frontend {

class NativeKernel {
public:
    virtual ~NativeKernel() = default;

    // Compile the elementwise subset of kernel `name` (first `__global__` if empty)
    // to native code. Returns nullptr (and sets `err`) when the kernel isn't in the
    // supported subset or the host isn't a supported target — callers then fall
    // back to a slower tier.
    static std::unique_ptr<NativeKernel> compileSource(const std::string& source,
                                                       const std::string& name,
                                                       std::string& err);

    // Execute over a 1-D grid/block (grid.x * block.x threads). `args` is
    // CUDA-style: one pointer per kernel parameter, dereferenced to its value.
    virtual bool launch(const Extent& grid, const Extent& block,
                        void* const* args, int numArgs) = 0;

    virtual int numParams() const = 0;
};

}  // namespace frontend
}  // namespace compiler
}  // namespace vgre

#endif  // VGRE_COMPILER_FRONTEND_NATIVE_KERNEL_H
