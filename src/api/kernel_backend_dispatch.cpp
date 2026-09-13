// See include/vgre/api/kernel_backend_dispatch.h.

#include "vgre/api/kernel_backend_dispatch.h"

#include "vgre/compiler/backend/backend_registry.h"
#include "vgre/compiler/backend/execution_backend.h"
#include "vgre/compiler/frontend/codegen.h"
#include "vgre/compiler/frontend/compiled_kernel.h"
#include "vgre/common/logger.h"

#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace vgre {
namespace api {

namespace {

namespace be = vgre::compiler::backend;
namespace fe = vgre::compiler::frontend;

// A registered backend kernel is either a Tier-1 compiled kernel or a Tier-0
// interpreter kernel (the fallback for barrier/shared kernels).
struct Entry {
    std::unique_ptr<fe::CompiledKernel> compiled;   // Tier-1
    std::unique_ptr<be::PreparedKernel> ptx;        // Tier-0 (interpreter)
};

std::mutex g_mu;
std::unordered_map<uint64_t, Entry> g_kernels;
uint64_t g_nextId = 0x4000000000000000ULL;  // high range: never collides with engine ids

be::ExecutionBackend* interpreter() {
    static std::unique_ptr<be::ExecutionBackend> inst = be::makeBackend("interpreter");
    return inst.get();
}

std::string modeName() {
    const char* e = std::getenv("VGRE_EXEC_BACKEND");
    return (e && e[0]) ? std::string(e) : std::string();
}

uint32_t nz(uint32_t v) { return v ? v : 1u; }

}  // namespace

bool backendModeActive() {
#ifndef VGRE_ENABLE_JIT
    // The LLVM JIT is compiled out — the execution backend is the only path, so
    // every kernel routes through the from-scratch front-end regardless of the
    // (now moot) VGRE_EXEC_BACKEND selector.
    return true;
#else
    std::string m = modeName();
    return !m.empty() && m != "jit" && m != "llvm";
#endif
}

bool tryRegisterBackendKernel(const std::string& name, const std::string& source, uint64_t& outId) {
    if (!backendModeActive()) return false;
    const std::string mode = modeName();
    Entry entry;

    // Tier-1: compiled backend when explicitly selected.
    if (mode == "compiled" || mode == "cp") {
        std::string err;
        auto ck = fe::CompiledKernel::compileSource(source, name, err);
        if (ck) {
            entry.compiled = std::move(ck);
            VGRE_LOG_INFO("BackendDispatch", "kernel '" + name + "' on the compiled tier (no LLVM)");
        } else {
            VGRE_LOG_INFO("BackendDispatch",
                          "kernel '" + name + "' not compiled-tier eligible (" + err +
                          ") — trying the interpreter tier");
        }
    }

    // Tier-0 interpreter: the default backend mode, and the fallback for kernels
    // the compiled tier rejects (e.g. __syncthreads).
    if (!entry.compiled) {
        be::ExecutionBackend* b = interpreter();
        if (!b) return false;
        auto cg = fe::compileToPtx(source, name);
        if (!cg.ok) {
            VGRE_LOG_INFO("BackendDispatch",
                          "kernel '" + name + "' outside the CUDA-C subset (" + cg.error +
                          ") — falling back to the JIT");
            return false;
        }
        auto k = b->preparePtx(cg.ptx, name);
        if (!k) return false;
        entry.ptx = std::move(k);
        VGRE_LOG_INFO("BackendDispatch", "kernel '" + name + "' on the interpreter tier (no LLVM)");
    }

    std::lock_guard<std::mutex> lock(g_mu);
    outId = g_nextId++;
    g_kernels[outId] = std::move(entry);
    return true;
}

int tryLaunchBackendKernel(uint64_t kid, const uint32_t grid[3], const uint32_t block[3],
                           void** args, int num_args, size_t shared_mem) {
    fe::CompiledKernel* ck = nullptr;
    be::PreparedKernel* pk = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_mu);
        auto it = g_kernels.find(kid);
        if (it == g_kernels.end()) return -1;  // not a backend kernel
        ck = it->second.compiled.get();
        pk = it->second.ptx.get();
    }
    if (ck) {
        fe::Extent g{nz(grid[0]), nz(grid[1]), nz(grid[2])};
        fe::Extent b{nz(block[0]), nz(block[1]), nz(block[2])};
        return ck->launch(g, b, args, num_args) ? 0 : 1;
    }
    if (pk) {
        be::ExecutionBackend* b = interpreter();
        if (!b) return 1;
        be::LaunchConfig cfg;
        cfg.gridDim[0] = nz(grid[0]);  cfg.gridDim[1] = nz(grid[1]);  cfg.gridDim[2] = nz(grid[2]);
        cfg.blockDim[0] = nz(block[0]); cfg.blockDim[1] = nz(block[1]); cfg.blockDim[2] = nz(block[2]);
        cfg.sharedBytes = shared_mem;
        return b->launch(*pk, cfg, args, num_args) ? 0 : 1;
    }
    return 1;
}

}  // namespace api
}  // namespace vgre
