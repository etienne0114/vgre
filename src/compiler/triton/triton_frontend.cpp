// Phase 3, Track P3-13 — Triton IR (TTIR) frontend: compiled library entry.
//
// Keeps the parser + interpreter (triton_frontend.h) reachable as a shipped-library
// API rather than a header-only definition, so the frontend is wired into the
// build and callable by external code and future Triton-runtime integration.

#include "vgre/compiler/triton/triton_frontend.h"

#include <stdexcept>

namespace vgre {
namespace triton {

void run_kernel(const std::string& ttir, const std::string& kernelName,
                const std::vector<KernelArg>& args, int gridX) {
    Module m = parse(ttir);
    const Func* fn = nullptr;
    for (const auto& f : m.funcs) {
        if (kernelName.empty() || f.name == kernelName) { fn = &f; break; }
    }
    if (!fn)
        throw std::runtime_error("triton: kernel '" + kernelName + "' not found in TTIR module");
    Interpreter::run(*fn, args, gridX);
}

}  // namespace triton
}  // namespace vgre
