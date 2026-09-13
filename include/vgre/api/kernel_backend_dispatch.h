// Track Z: routes kernel register/launch through the pluggable ExecutionBackend
// when VGRE_EXEC_BACKEND selects a non-JIT backend (e.g. "interpreter"). In that
// mode a kernel is compiled by VGRE's own CUDA-C front-end to PTX and executed
// by the interpreter tier — no LLVM/Clang on the path. Kernels the subset cannot
// compile fall back to the LLVM JIT, so this is purely additive.
#ifndef VGRE_API_KERNEL_BACKEND_DISPATCH_H
#define VGRE_API_KERNEL_BACKEND_DISPATCH_H

#include <cstddef>
#include <cstdint>
#include <string>

namespace vgre {
namespace api {

// True when VGRE_EXEC_BACKEND names a non-JIT backend (kernels should try the
// execution-backend path first).
bool backendModeActive();

// Try to register `source` through the execution backend. On success assigns a
// backend kernel id to `outId` and returns true. Returns false when backend mode
// is off or the kernel is outside the supported CUDA-C subset (caller then uses
// the JIT path).
bool tryRegisterBackendKernel(const std::string& name, const std::string& source, uint64_t& outId);

// Launch dispatch for a kernel id. Returns:
//   -1  not a backend kernel (caller should use the engine/JIT launch)
//    0  handled, launch succeeded
//    1  handled, launch failed
int tryLaunchBackendKernel(uint64_t kid, const uint32_t grid[3], const uint32_t block[3],
                           void** args, int num_args, size_t shared_mem);

}  // namespace api
}  // namespace vgre

#endif  // VGRE_API_KERNEL_BACKEND_DISPATCH_H
