// Abstract execution backend — the seam that lets VGRE run GPU kernels through
// interchangeable engines behind one interface: Tier 0 (PTX interpreter, no
// codegen), Tier 1 (copy-and-patch), Tier 2 (own SSA backend), or the legacy
// LLVM ORC JIT. Part of Track Z (Zero-Burden Engine) — the goal is that VGRE
// depends on no heavyweight toolchain. See docs/zeroBurdenRoadmap.md.
#ifndef VGRE_COMPILER_BACKEND_EXECUTION_BACKEND_H
#define VGRE_COMPILER_BACKEND_EXECUTION_BACKEND_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace vgre {
namespace compiler {
namespace backend {

// CUDA-style 3D launch geometry plus dynamic shared memory (bytes).
struct LaunchConfig {
    uint32_t gridDim[3]  = {1, 1, 1};
    uint32_t blockDim[3] = {1, 1, 1};
    size_t   sharedBytes = 0;
};

// A kernel already prepared by a specific backend. Opaque to callers; owned by
// the caller (returned as a unique_ptr) and only meaningful to the backend that
// produced it.
class PreparedKernel {
public:
    virtual ~PreparedKernel() = default;
    // The entry-point name this handle executes.
    virtual const std::string& entry() const = 0;
};

// One execution strategy. Concrete backends: InterpreterBackend (Tier 0) today;
// CopyPatchBackend (Tier 1) and JitBackend (LLVM) in later Track Z increments.
class ExecutionBackend {
public:
    virtual ~ExecutionBackend() = default;

    // Stable identifier: "interpreter", "copy-patch", "jit", …
    virtual const char* name() const = 0;

    // True if this backend can prepare a kernel directly from PTX text.
    virtual bool acceptsPtx() const = 0;

    // Prepare `entry` from PTX source. Returns nullptr on parse/lookup failure.
    virtual std::unique_ptr<PreparedKernel> preparePtx(
        const std::string& ptx, const std::string& entry) = 0;

    // Execute a prepared kernel. `args` is CUDA-style: one pointer per .param,
    // dereferenced to the param's size. Returns false on a launch/exec fault.
    virtual bool launch(PreparedKernel& kernel, const LaunchConfig& cfg,
                        void* const* args, int numArgs) = 0;
};

}  // namespace backend
}  // namespace compiler
}  // namespace vgre

#endif  // VGRE_COMPILER_BACKEND_EXECUTION_BACKEND_H
