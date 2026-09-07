// See include/vgre/api/kernel_backend_dispatch.h.

#include "vgre/api/kernel_backend_dispatch.h"

#include "vgre/compiler/backend/backend_registry.h"
#include "vgre/compiler/backend/execution_backend.h"
#include "vgre/compiler/frontend/codegen.h"
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

struct Entry {
    std::unique_ptr<be::PreparedKernel> kernel;
    std::string name;
};

std::mutex g_mu;
std::unordered_map<uint64_t, Entry> g_kernels;
// Backend kernel ids live in a high range so they never collide with the
// engine's small sequential ids.
uint64_t g_nextId = 0x4000000000000000ULL;

// One process-wide backend, built from VGRE_EXEC_BACKEND on first use.
be::ExecutionBackend* backend() {
    static std::unique_ptr<be::ExecutionBackend> inst = be::makeDefaultBackend();
    return inst.get();
}

uint32_t nz(uint32_t v) { return v ? v : 1u; }  // 0 dim -> 1

}  // namespace

bool backendModeActive() {
    const char* e = std::getenv("VGRE_EXEC_BACKEND");
    if (!e || !e[0]) return false;
    std::string s(e);
    return s != "jit" && s != "llvm";  // active for interpreter / cp / ssa / …
}

bool tryRegisterBackendKernel(const std::string& name, const std::string& source, uint64_t& outId) {
    if (!backendModeActive()) return false;
    be::ExecutionBackend* b = backend();
    if (!b) return false;

    auto cg = vgre::compiler::frontend::compileToPtx(source, name);
    if (!cg.ok) {
        VGRE_LOG_INFO("BackendDispatch",
                      "kernel '" + name + "' outside the CUDA-C subset (" + cg.error +
                      ") — falling back to the JIT");
        return false;
    }
    auto k = b->preparePtx(cg.ptx, name);
    if (!k) return false;

    std::lock_guard<std::mutex> lock(g_mu);
    outId = g_nextId++;
    g_kernels[outId] = Entry{std::move(k), name};
    VGRE_LOG_INFO("BackendDispatch",
                  "kernel '" + name + "' registered on the '" + b->name() +
                  "' backend (id=" + std::to_string(outId) + ", no LLVM)");
    return true;
}

int tryLaunchBackendKernel(uint64_t kid, const uint32_t grid[3], const uint32_t block[3],
                           void** args, int num_args, size_t shared_mem) {
    be::PreparedKernel* kernel = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_mu);
        auto it = g_kernels.find(kid);
        if (it == g_kernels.end()) return -1;  // not ours
        kernel = it->second.kernel.get();
    }
    be::ExecutionBackend* b = backend();
    if (!b || !kernel) return 1;

    be::LaunchConfig cfg;
    cfg.gridDim[0] = nz(grid[0]);  cfg.gridDim[1] = nz(grid[1]);  cfg.gridDim[2] = nz(grid[2]);
    cfg.blockDim[0] = nz(block[0]); cfg.blockDim[1] = nz(block[1]); cfg.blockDim[2] = nz(block[2]);
    cfg.sharedBytes = shared_mem;
    return b->launch(*kernel, cfg, args, num_args) ? 0 : 1;
}

}  // namespace api
}  // namespace vgre
