// Zero-Burden build (no LLVM): RuntimeEngine kernel execution via the
// from-scratch CUDA-C front-end + interpreter/compiled backends. Compiled ONLY
// when VGRE_ENABLE_JIT is OFF (see src/core/CMakeLists.txt); in a JIT build the
// engine uses the LLVM ORC path and this file is not built.
//
// This is what lets the whole engine API (cudaLaunchKernel, the C++ RuntimeEngine
// interface, graph/cooperative launches that route through launchKernel) run
// CUDA kernels with no LLVM — not just the direct C-ABI dispatch.

#ifndef VGRE_ENABLE_JIT

#include "vgre/core/runtime_engine.h"

#include "vgre/common/logger.h"
#include "vgre/compiler/backend/backend_registry.h"
#include "vgre/compiler/backend/execution_backend.h"
#include "vgre/compiler/frontend/codegen.h"
#include "vgre/compiler/frontend/compiled_kernel.h"
#include "vgre/compiler/frontend/parser.h"

#include <memory>
#include <string>

namespace vgre {
namespace core {

namespace fe = vgre::compiler::frontend;
namespace be = vgre::compiler::backend;

// A registered backend kernel: a Tier-1 compiled kernel (barrier-free, fast) OR
// a Tier-0 interpreter kernel (shared-memory / __syncthreads cooperative path).
struct RuntimeEngine::BackendKernel {
    int numArgs = 0;
    std::unique_ptr<fe::CompiledKernel>   compiled;   // Tier-1
    std::unique_ptr<be::PreparedKernel>   prepared;   // Tier-0 (interpreter)
};

// Process-wide interpreter backend (the Tier-0 fallback), constructed once.
static be::ExecutionBackend *interpBackend() {
    static std::unique_ptr<be::ExecutionBackend> inst = be::makeBackend("interpreter");
    return inst.get();
}

std::shared_ptr<RuntimeEngine::BackendKernel>
RuntimeEngine::makeBackendKernel(const std::string &name, const std::string &source) {
    auto bk = std::make_shared<BackendKernel>();

    // Tier-1 compiled tier first (fast; barrier-free kernels).
    std::string err;
    if (auto ck = fe::CompiledKernel::compileSource(source, name, err)) {
        bk->numArgs = ck->numParams();
        bk->compiled = std::move(ck);
        VGRE_LOG_INFO("RuntimeEngine", "backend kernel '" + name + "' on the compiled tier (no LLVM)");
        return bk;
    }

    // Tier-0 interpreter fallback (e.g. __shared__ / __syncthreads kernels).
    auto cg = fe::compileToPtx(source, name);
    if (!cg.ok) {
        VGRE_LOG_INFO("RuntimeEngine",
                      "kernel '" + name + "' not backend-eligible: " + cg.error);
        return nullptr;
    }
    be::ExecutionBackend *b = interpBackend();
    if (!b) return nullptr;
    auto prep = b->preparePtx(cg.ptx, name);
    if (!prep) return nullptr;

    // Parameter count from a front-end parse (the interpreter validates arg count).
    if (auto pr = fe::parse(source); pr.ok && pr.module) {
        for (const auto &k : pr.module->kernels)
            if (k->name == name) { bk->numArgs = static_cast<int>(k->params.size()); break; }
    }
    bk->prepared = std::move(prep);
    VGRE_LOG_INFO("RuntimeEngine", "backend kernel '" + name + "' on the interpreter tier (no LLVM)");
    return bk;
}

VGREResult RuntimeEngine::launchBackendKernel(const std::shared_ptr<BackendKernel> &bk,
                                              const dim3 &gridDim, const dim3 &blockDim,
                                              void **args, size_t sharedMem) {
    if (bk->compiled) {
        fe::Extent g{gridDim.x, gridDim.y, gridDim.z};
        fe::Extent b{blockDim.x, blockDim.y, blockDim.z};
        return bk->compiled->launch(g, b, args, bk->numArgs)
                   ? VGREResult::SUCCESS : VGREResult::ERR_LAUNCH_FAILURE;
    }
    if (bk->prepared) {
        be::ExecutionBackend *b = interpBackend();
        if (!b) return VGREResult::ERR_NOT_SUPPORTED;
        be::LaunchConfig cfg;
        cfg.gridDim[0] = gridDim.x; cfg.gridDim[1] = gridDim.y; cfg.gridDim[2] = gridDim.z;
        cfg.blockDim[0] = blockDim.x; cfg.blockDim[1] = blockDim.y; cfg.blockDim[2] = blockDim.z;
        cfg.sharedBytes = sharedMem;
        return b->launch(*bk->prepared, cfg, args, bk->numArgs)
                   ? VGREResult::SUCCESS : VGREResult::ERR_LAUNCH_FAILURE;
    }
    return VGREResult::ERR_NOT_SUPPORTED;
}

VGREResult RuntimeEngine::launchBackendByName(const std::string &name, const dim3 &gridDim,
                                              const dim3 &blockDim, void **args, size_t sharedMem) {
    std::shared_ptr<BackendKernel> bk;
    {
        std::lock_guard<std::recursive_mutex> lock(mutex_);
        auto nit = kernelNames_.find(name);
        if (nit != kernelNames_.end()) {
            auto bit = backendKernels_.find(nit->second);
            if (bit != backendKernels_.end()) bk = bit->second;
        }
    }
    if (!bk) return VGREResult::ERR_INVALID_KERNEL;
    return launchBackendKernel(bk, gridDim, blockDim, args, sharedMem);
}

}  // namespace core
}  // namespace vgre

#endif  // !VGRE_ENABLE_JIT
