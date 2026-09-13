// Hand-written lexer for the CUDA-C kernel subset. Turns source text into a
// token stream; comments and whitespace are skipped. Errors surface as an
// Unknown token (never throws) so the parser can report a precise location.
// Track Z — see include/vgre/compiler/frontend/token.h.
#ifndef VGRE_COMPILER_FRONTEND_LEXER_H
#define VGRE_COMPILER_FRONTEND_LEXER_H

#include "vgre/compiler/frontend/token.h"

#include <string>
#include <vector>

namespace vgre {
namespace compiler {
namespace frontend {

// Tokenize `source`. The returned vector always ends with a single End token.
// Line comments (//…), block comments (/* … */), and preprocessor lines
// starting with '#' are skipped (a kernel's `#include`/`#define` are ignored;
// the supported subset needs no macro expansion).
std::vector<Token> lex(const std::string& source);

}  // namespace frontend
}  // namespace compiler
}  // namespace vgre

#endif  // VGRE_COMPILER_FRONTEND_LEXER_H
