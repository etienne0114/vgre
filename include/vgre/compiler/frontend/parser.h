// Recursive-descent parser for the CUDA-C kernel subset → AST (Track Z).
// No Clang: this is VGRE's own front-end. Errors are returned, never thrown.
#ifndef VGRE_COMPILER_FRONTEND_PARSER_H
#define VGRE_COMPILER_FRONTEND_PARSER_H

#include "vgre/compiler/frontend/ast.h"

#include <memory>
#include <string>

namespace vgre {
namespace compiler {
namespace frontend {

struct ParseResult {
    std::unique_ptr<Module> module;  // null on failure
    bool ok = false;
    std::string error;               // "line:col: message" on failure
};

// Lex + parse `source` into a Module of kernels.
ParseResult parse(const std::string& source);

}  // namespace frontend
}  // namespace compiler
}  // namespace vgre

#endif  // VGRE_COMPILER_FRONTEND_PARSER_H
