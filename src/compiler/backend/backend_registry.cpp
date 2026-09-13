#include "vgre/compiler/backend/backend_registry.h"

#include "interpreter/interpreter_backend.h"
#include "vgre/common/logger.h"

#include <cstdlib>
#include <string>

namespace vgre {
namespace compiler {
namespace backend {

std::unique_ptr<ExecutionBackend> makeBackend(const std::string& name) {
    if (name == "interpreter") {
        return std::unique_ptr<ExecutionBackend>(new InterpreterBackend());
    }
    // "copy-patch" (Tier 1) and "jit" (LLVM ORC) are registered here in later
    // Track Z increments.
    VGRE_LOG_WARN("BackendRegistry", "unknown execution backend '" + name + "'");
    return nullptr;
}

std::unique_ptr<ExecutionBackend> makeDefaultBackend() {
    const char* env = std::getenv("VGRE_EXEC_BACKEND");
    const std::string name = (env && env[0]) ? std::string(env) : std::string("interpreter");
    auto backend = makeBackend(name);
    if (!backend && name != "interpreter") {
        VGRE_LOG_WARN("BackendRegistry",
                      "unknown VGRE_EXEC_BACKEND='" + name +
                      "' — falling back to the interpreter tier");
        backend = makeBackend("interpreter");
    }
    return backend;
}

}  // namespace backend
}  // namespace compiler
}  // namespace vgre
