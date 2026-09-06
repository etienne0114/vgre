// Execution-backend factory. Callers obtain a backend by name or take the
// process default (honouring $VGRE_EXEC_BACKEND). Part of Track Z. See
// docs/zeroBurdenRoadmap.md.
#ifndef VGRE_COMPILER_BACKEND_BACKEND_REGISTRY_H
#define VGRE_COMPILER_BACKEND_BACKEND_REGISTRY_H

#include "vgre/compiler/backend/execution_backend.h"

#include <memory>
#include <string>

namespace vgre {
namespace compiler {
namespace backend {

// Create a backend by name ("interpreter" today). Returns nullptr if unknown.
std::unique_ptr<ExecutionBackend> makeBackend(const std::string& name);

// Create the default backend: honours $VGRE_EXEC_BACKEND, else "interpreter"
// (the Tier-0 fallback that needs no code generation and no dependencies).
std::unique_ptr<ExecutionBackend> makeDefaultBackend();

}  // namespace backend
}  // namespace compiler
}  // namespace vgre

#endif  // VGRE_COMPILER_BACKEND_BACKEND_REGISTRY_H
