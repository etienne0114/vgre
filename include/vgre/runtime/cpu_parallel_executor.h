#ifndef VGRE_RUNTIME_CPU_PARALLEL_EXECUTOR_H
#define VGRE_RUNTIME_CPU_PARALLEL_EXECUTOR_H

#include "vgre/common/types.h"
#include "vgre/common/error_codes.h"

#include <atomic>

namespace vgre {
namespace runtime {

// ── CPU Parallel Executor ──────────────────────────────────────────────────
// Maps CUDA grid/block execution model to OpenMP parallel loops.
// Each block is a parallel work unit; threads within a block are
// simulated via inner loops with optional SIMD vectorization.
class CPUParallelExecutor {
public:
    explicit CPUParallelExecutor(int maxThreads = 0);
    ~CPUParallelExecutor();

    // Execute a compiled kernel across the full grid
    VGREResult execute(const CompiledKernelFn& fn,
                       const dim3& gridDim,
                       const dim3& blockDim,
                       void** args);

    // Setters
    void setMaxThreads(int n);
    int  getMaxThreads() const;

    // Statistics
    uint64_t getTotalKernelLaunches() const;
    uint64_t getTotalBlocksExecuted() const;

private:
    int                    maxThreads_;
    std::atomic<uint64_t>  totalLaunches_{0};
    std::atomic<uint64_t>  totalBlocks_{0};
};

} // namespace runtime
} // namespace vgre

#endif // VGRE_RUNTIME_CPU_PARALLEL_EXECUTOR_H
