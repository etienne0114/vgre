#ifndef VGRE_COMPILER_KERNEL_PARSER_H
#define VGRE_COMPILER_KERNEL_PARSER_H

#include "vgre/common/types.h"
#include "vgre/common/error_codes.h"

#include <string>
#include <vector>
#include <unordered_map>

namespace vgre {
namespace compiler {

// ── Token types for kernel source ──────────────────────────────────────────
enum class TokenType : uint8_t {
    KEYWORD,
    IDENTIFIER,
    TYPE,
    OPERATOR,
    LITERAL_INT,
    LITERAL_FLOAT,
    LPAREN,
    RPAREN,
    LBRACE,
    RBRACE,
    LBRACKET,
    RBRACKET,
    SEMICOLON,
    COMMA,
    BUILTIN_VAR,    // threadIdx, blockIdx, blockDim, gridDim
    END_OF_SOURCE
};

struct Token {
    TokenType   type;
    std::string value;
    int         line = 0;
    int         col  = 0;
};

// ── Parsed parameter descriptor ────────────────────────────────────────────
struct ParsedParam {
    std::string typeName;     // e.g. "float*", "int"
    std::string name;
    ArgType     argType;
    bool        isPointer = false;
};

// ── Kernel Parser ──────────────────────────────────────────────────────────
class KernelParser {
public:
    KernelParser();
    ~KernelParser();

    // Parse a CUDA-like kernel source to KernelIR
    VGREResult parse(const std::string& name,
                     const std::string& source,
                     KernelIR& outIR);

    // Extract parameter list from a function signature
    VGREResult parseParameters(const std::string& signature,
                               std::vector<ParsedParam>& outParams);

    // Identify CUDA built-in variables in the body
    std::vector<std::string> findBuiltinVars(const std::string& body);

private:
    // Tokenizer
    std::vector<Token> tokenize(const std::string& source);

    // Extract function signature, body, and qualifiers
    VGREResult extractFunction(const std::vector<Token>& tokens,
                               std::string& outName,
                               std::string& outSignature,
                               std::string& outBody,
                               bool& outIsGlobal);

    // Map C type string to ArgType
    ArgType mapType(const std::string& typeName, bool isPointer);

    // Built-in variable set
    static const std::unordered_map<std::string, std::string> builtinVars_;
};

} // namespace compiler
} // namespace vgre

#endif // VGRE_COMPILER_KERNEL_PARSER_H
