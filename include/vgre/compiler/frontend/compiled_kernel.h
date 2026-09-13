// Tier-1 compiled execution (Track Z). The CUDA-C AST is lowered ONCE to
// slot-based bound operations (closures over resolved register indices), then
// executed per-thread with no string parsing — much faster than the PTX
// interpreter's per-instruction string dispatch, while staying portable
// (pure C++, every arch) and LLVM-free.
//
// It covers the barrier-free subset (elementwise / index-mapped kernels).
// Kernels using __shared__/__syncthreads need cooperative scheduling and are
// rejected here (compile() returns nullptr) so the caller can fall back to the
// interpreter tier. See docs/zeroBurdenRoadmap.md.
#ifndef VGRE_COMPILER_FRONTEND_COMPILED_KERNEL_H
#define VGRE_COMPILER_FRONTEND_COMPILED_KERNEL_H

#include "vgre/compiler/frontend/ast.h"

#include <cstdint>
#include <memory>
#include <string>

namespace vgre {
namespace compiler {
namespace frontend {

// A 3D extent (grid or block); components default to 1.
struct Extent {
    uint32_t x = 1, y = 1, z = 1;
};

class CompiledKernel {
public:
    virtual ~CompiledKernel() = default;

    // Compile `k`. Returns nullptr and sets `err` if the kernel uses a construct
    // the compiled tier does not support (barriers, shared memory, …).
    static std::unique_ptr<CompiledKernel> compile(const Kernel& k, std::string& err);

    // Compile from CUDA-C source (lex + parse + compile) for kernel `name`
    // (first kernel if empty).
    static std::unique_ptr<CompiledKernel> compileSource(const std::string& source,
                                                         const std::string& name,
                                                         std::string& err);

    // Execute over a 3D grid/block. `args` is CUDA-style: one pointer per kernel
    // parameter, dereferenced to the parameter's value. Returns false on a fault.
    virtual bool launch(const Extent& grid, const Extent& block,
                        void* const* args, int numArgs) = 0;

    // The number of kernel parameters (== the required numArgs).
    virtual int numParams() const = 0;
};

}  // namespace frontend
}  // namespace compiler
}  // namespace vgre

#endif  // VGRE_COMPILER_FRONTEND_COMPILED_KERNEL_H
