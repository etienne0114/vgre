#include "vgre/compiler/kernel_parser.h"
#include "vgre/common/logger.h"

#include <algorithm>
#include <cctype>
#include <regex>

namespace vgre {
namespace compiler {

namespace {
std::string normalizeTypeName(std::string typeName) {
    typeName = std::regex_replace(typeName, std::regex("\\b(const|volatile|restrict|__restrict__|__restrict)\\b"), "");
    typeName = std::regex_replace(typeName, std::regex("[\\t\\n\\r]+"), " ");
    typeName = std::regex_replace(typeName, std::regex("\\s+"), " ");
    if (!typeName.empty() && typeName.front() == ' ') {
        typeName.erase(0, typeName.find_first_not_of(' '));
    }
    if (!typeName.empty() && typeName.back() == ' ') {
        typeName.erase(typeName.find_last_not_of(' ') + 1);
    }
    return typeName;
}
} // namespace

// ── Built-in CUDA variables ────────────────────────────────────────────────
const std::unordered_map<std::string, std::string> KernelParser::builtinVars_ = {
    {"threadIdx",  "threadIdx"},
    {"blockIdx",   "blockIdx"},
    {"blockDim",   "blockDim"},
    {"gridDim",    "gridDim"},
    {"warpSize",   "warpSize"}
};

KernelParser::KernelParser()  = default;
KernelParser::~KernelParser() = default;

// ── Tokenizer ──────────────────────────────────────────────────────────────
std::vector<Token> KernelParser::tokenize(const std::string& source) {
    std::vector<Token> tokens;
    int line = 1, col = 0;
    size_t i = 0;

    auto isIdStart = [](char c) { return std::isalpha(c) || c == '_'; };
    auto isIdCont  = [](char c) { return std::isalnum(c) || c == '_'; };

    while (i < source.size()) {
        char c = source[i];

        // Skip whitespace
        if (std::isspace(c)) {
            if (c == '\n') { ++line; col = 0; }
            else { ++col; }
            ++i;
            continue;
        }

        // Skip line comments
        if (c == '/' && i + 1 < source.size() && source[i+1] == '/') {
            while (i < source.size() && source[i] != '\n') ++i;
            continue;
        }

        // Skip block comments
        if (c == '/' && i + 1 < source.size() && source[i+1] == '*') {
            i += 2;
            while (i + 1 < source.size() &&
                   !(source[i] == '*' && source[i+1] == '/')) {
                if (source[i] == '\n') { ++line; col = 0; }
                ++i;
            }
            i += 2;
            continue;
        }

        Token tok;
        tok.line = line;
        tok.col  = col;

        // Identifiers / keywords
        if (isIdStart(c)) {
            std::string word;
            while (i < source.size() && isIdCont(source[i])) {
                word += source[i++];
            }
            col += static_cast<int>(word.size());

            if (builtinVars_.count(word)) {
                tok.type = TokenType::BUILTIN_VAR;
            } else if (word == "__global__" || word == "__device__" ||
                       word == "__host__"   || word == "void"    ||
                       word == "return"     || word == "if"      ||
                       word == "else"       || word == "for"     ||
                       word == "while"      || word == "const"   ||
                       word == "kernel") {
                tok.type = TokenType::KEYWORD;
            } else if (word == "float"  || word == "double" ||
                       word == "int"    || word == "long"   ||
                       word == "unsigned" || word == "char"  ||
                       word == "size_t" || word == "uint32_t" ||
                       word == "int32_t"|| word == "int64_t" ||
                       word == "uint64_t") {
                tok.type = TokenType::TYPE;
            } else {
                tok.type = TokenType::IDENTIFIER;
            }
            tok.value = word;
            tokens.push_back(tok);
            continue;
        }

        // Numbers
        if (std::isdigit(c) || (c == '.' && i + 1 < source.size() &&
                                std::isdigit(source[i+1]))) {
            std::string num;
            bool isFloat = false;
            while (i < source.size() && (std::isdigit(source[i]) ||
                   source[i] == '.' || source[i] == 'f' ||
                   source[i] == 'e' || source[i] == 'E')) {
                if (source[i] == '.' || source[i] == 'f' ||
                    source[i] == 'e' || source[i] == 'E')
                    isFloat = true;
                num += source[i++];
            }
            tok.type  = isFloat ? TokenType::LITERAL_FLOAT : TokenType::LITERAL_INT;
            tok.value = num;
            col += static_cast<int>(num.size());
            tokens.push_back(tok);
            continue;
        }

        // Single-character tokens
        tok.value = std::string(1, c);
        ++col; ++i;
        switch (c) {
            case '(': tok.type = TokenType::LPAREN;    break;
            case ')': tok.type = TokenType::RPAREN;    break;
            case '{': tok.type = TokenType::LBRACE;    break;
            case '}': tok.type = TokenType::RBRACE;    break;
            case '[': tok.type = TokenType::LBRACKET;  break;
            case ']': tok.type = TokenType::RBRACKET;  break;
            case ';': tok.type = TokenType::SEMICOLON; break;
            case ',': tok.type = TokenType::COMMA;     break;
            default:  tok.type = TokenType::OPERATOR;  break;
        }
        tokens.push_back(tok);
    }

    Token eof;
    eof.type  = TokenType::END_OF_SOURCE;
    eof.value = "";
    eof.line  = line;
    eof.col   = col;
    tokens.push_back(eof);

    return tokens;
}

// ── Extract function ───────────────────────────────────────────────────────
VGREResult KernelParser::extractFunction(const std::vector<Token>& tokens,
                                          std::string& outName,
                                          size_t& outParamStart,
                                          size_t& outParamEnd,
                                          std::string& outBody,
                                          bool& outIsGlobal) {
    outIsGlobal = false;
    size_t i = 0;

    // Look for __global__ or kernel qualifier
    while (i < tokens.size()) {
        if (tokens[i].type == TokenType::KEYWORD &&
            (tokens[i].value == "__global__" || tokens[i].value == "kernel")) {
            outIsGlobal = true;
            ++i;
            break;
        }
        ++i;
    }

    if (!outIsGlobal) {
        i = 0;
    }

    // Skip return type
    while (i < tokens.size() &&
           (tokens[i].type == TokenType::TYPE ||
            tokens[i].type == TokenType::KEYWORD ||
            tokens[i].value == "::")) {
        ++i;
    }

    // Function name
    if (i < tokens.size() && tokens[i].type == TokenType::IDENTIFIER) {
        outName = tokens[i].value;
        ++i;
    } else {
        return VGREResult::ERROR_INVALID_KERNEL;
    }

    // Parameter list
    if (i < tokens.size() && tokens[i].type == TokenType::LPAREN) {
        outParamStart = i + 1;
        int depth = 1;
        ++i;
        while (i < tokens.size() && depth > 0) {
            if (tokens[i].type == TokenType::LPAREN) ++depth;
            if (tokens[i].type == TokenType::RPAREN) --depth;
            if (depth == 0) {
                outParamEnd = i;
                break;
            }
            ++i;
        }
        if (i < tokens.size()) ++i;
    } else {
        return VGREResult::ERROR_INVALID_KERNEL;
    }

    // Body
    if (i < tokens.size() && tokens[i].type == TokenType::LBRACE) {
        int depth = 1;
        ++i;
        while (i < tokens.size() && depth > 0) {
            if (tokens[i].type == TokenType::LBRACE) ++depth;
            if (tokens[i].type == TokenType::RBRACE) --depth;
            if (depth > 0) {
                outBody += tokens[i].value;
                if (tokens[i].type != TokenType::SEMICOLON &&
                    tokens[i].type != TokenType::LBRACE &&
                    tokens[i].type != TokenType::RBRACE &&
                    tokens[i].type != TokenType::OPERATOR) {
                    outBody += " ";
                }
            }
            ++i;
        }
    }

    return VGREResult::SUCCESS;
}

// ── Parse parameters ───────────────────────────────────────────────────────
VGREResult KernelParser::parseParameters(const std::vector<Token>& tokens,
                                         size_t startIndex, size_t endIndex,
                                         std::vector<ParsedParam>& outParams) {
    if (startIndex >= endIndex || endIndex > tokens.size()) return VGREResult::SUCCESS;
    
    std::vector<Token> currentParam;
    int templateDepth = 0;
    int parenDepth = 0;

    auto processParam = [&](const std::vector<Token>& pTokens) -> VGREResult {
        if (pTokens.empty()) return VGREResult::SUCCESS;
        ParsedParam pp;
        pp.isPointer = false;
        
        std::string typeStr;
        std::string nameStr;
        
        int nameIdx = static_cast<int>(pTokens.size()) - 1;
        
        while (nameIdx >= 0 && 
               (pTokens[nameIdx].type == TokenType::RBRACKET || 
                pTokens[nameIdx].type == TokenType::LBRACKET || 
                pTokens[nameIdx].value == "restrict" || 
                pTokens[nameIdx].value == "__restrict__")) {
            nameIdx--;
        }
        
        if (nameIdx >= 0 && 
           (pTokens[nameIdx].type == TokenType::IDENTIFIER || pTokens[nameIdx].type == TokenType::KEYWORD)) {
            nameStr = pTokens[nameIdx].value;
        } else {
            nameStr = "arg" + std::to_string(outParams.size());
            nameIdx = static_cast<int>(pTokens.size());
        }
        
        for (int j = 0; j < nameIdx; ++j) {
            if (pTokens[j].value == "*") pp.isPointer = true;
            if (pTokens[j].type != TokenType::KEYWORD || 
               (pTokens[j].value != "const" && pTokens[j].value != "restrict" && pTokens[j].value != "volatile")) {
                typeStr += pTokens[j].value;
            }
        }
        
        typeStr.erase(std::remove(typeStr.begin(), typeStr.end(), '*'), typeStr.end());
        nameStr.erase(std::remove(nameStr.begin(), nameStr.end(), '*'), nameStr.end());
        if (typeStr.empty()) typeStr = nameStr;
        
        pp.typeName = typeStr;
        pp.name = nameStr;
        bool recognized = false;
        pp.argType = mapType(pp.typeName, pp.isPointer, recognized);
        
        if (!recognized) {
             VGRE_LOG_ERROR("KernelParser", "Unsupported type: " + pp.typeName);
             return VGREResult::ERROR_INVALID_KERNEL;
        }
        outParams.push_back(pp);
        return VGREResult::SUCCESS;
    };

    for (size_t i = startIndex; i < endIndex; ++i) {
        const auto& tok = tokens[i];
        if (tok.value == "<") templateDepth++;
        else if (tok.value == ">") templateDepth--;
        else if (tok.type == TokenType::LPAREN) parenDepth++;
        else if (tok.type == TokenType::RPAREN) parenDepth--;
        
        if (tok.type == TokenType::COMMA && templateDepth == 0 && parenDepth == 0) {
            auto r = processParam(currentParam);
            if (r != VGREResult::SUCCESS) return r;
            currentParam.clear();
        } else {
            currentParam.push_back(tok);
        }
    }
    
    if (!currentParam.empty()) {
        auto r = processParam(currentParam);
        if (r != VGREResult::SUCCESS) return r;
    }
    
    return VGREResult::SUCCESS;
}

// ── Map C type to ArgType ──────────────────────────────────────────────────
ArgType KernelParser::mapType(const std::string& typeName, bool isPointer,
                              bool& recognized) {
    if (isPointer) {
        recognized = true;
        return ArgType::POINTER;
    }

    const std::string norm = normalizeTypeName(typeName);

    if (norm == "float") {
        recognized = true;
        return ArgType::FLOAT32;
    }
    if (norm == "double") {
        recognized = true;
        return ArgType::FLOAT64;
    }
    if (norm == "int" || norm == "int32_t" || norm == "signed" ||
        norm == "signed int" || norm == "char" || norm == "signed char" ||
        norm == "short" || norm == "short int" || norm == "int16_t" ||
        norm == "int8_t" || norm == "bool") {
        recognized = true;
        return ArgType::INT32;
    }
    if (norm == "unsigned" || norm == "unsigned int" || norm == "uint32_t" ||
        norm == "unsigned short" || norm == "unsigned short int" ||
        norm == "uint16_t" || norm == "unsigned char" || norm == "uint8_t") {
        recognized = true;
        return ArgType::UINT32;
    }
    if (norm == "int64_t" || norm == "long long" || norm == "long long int" ||
        norm == "signed long long" || norm == "signed long long int") {
        recognized = true;
        return ArgType::INT64;
    }
    if (norm == "uint64_t" || norm == "unsigned long long" ||
        norm == "unsigned long long int") {
        recognized = true;
        return ArgType::UINT64;
    }
    if (norm == "long" || norm == "long int" || norm == "signed long" ||
        norm == "signed long int") {
        recognized = true;
        return (sizeof(long) >= 8) ? ArgType::INT64 : ArgType::INT32;
    }
    if (norm == "unsigned long" || norm == "unsigned long int") {
        recognized = true;
        return (sizeof(unsigned long) >= 8) ? ArgType::UINT64 : ArgType::UINT32;
    }
    if (norm == "size_t") {
        recognized = true;
        return (sizeof(size_t) >= 8) ? ArgType::UINT64 : ArgType::UINT32;
    }

    // Handle compact signatures where extractor concatenates type+name
    // (e.g., "intN", "floatx", "unsignedCount").
    const std::regex intNamePattern(R"(^int[A-Za-z_][A-Za-z0-9_]*$)");
    const std::regex floatNamePattern(R"(^float[A-Za-z_][A-Za-z0-9_]*$)");
    const std::regex doubleNamePattern(R"(^double[A-Za-z_][A-Za-z0-9_]*$)");
    const std::regex longNamePattern(R"(^long[A-Za-z_][A-Za-z0-9_]*$)");
    const std::regex ulongNamePattern(R"(^unsigned ?long[A-Za-z_][A-Za-z0-9_]*$)");
    const std::regex unsignedNamePattern(R"(^unsigned[A-Za-z_][A-Za-z0-9_]*$)");
    if (std::regex_match(norm, intNamePattern)) {
        recognized = true;
        return ArgType::INT32;
    }
    if (std::regex_match(norm, floatNamePattern)) {
        recognized = true;
        return ArgType::FLOAT32;
    }
    if (std::regex_match(norm, doubleNamePattern)) {
        recognized = true;
        return ArgType::FLOAT64;
    }
    if (std::regex_match(norm, longNamePattern)) {
        recognized = true;
        return (sizeof(long) >= 8) ? ArgType::INT64 : ArgType::INT32;
    }
    if (std::regex_match(norm, ulongNamePattern) ||
        std::regex_match(norm, unsignedNamePattern)) {
        recognized = true;
        return ArgType::UINT32;
    }

    recognized = false;
    return ArgType::INT32;
}

// ── Find built-in variables ────────────────────────────────────────────────
std::vector<std::string> KernelParser::findBuiltinVars(const std::string& body) {
    std::vector<std::string> found;
    for (const auto& [name, _] : builtinVars_) {
        if (body.find(name) != std::string::npos) {
            found.push_back(name);
        }
    }
    return found;
}

// ── Main parse ─────────────────────────────────────────────────────────────
VGREResult KernelParser::parse(const std::string& name,
                                const std::string& source,
                                KernelIR& outIR) {
    VGRE_LOG_INFO("KernelParser", "Parsing kernel AST: " + name);

    auto tokens = tokenize(source);
    if (tokens.empty()) {
        return VGREResult::ERROR_INVALID_KERNEL;
    }

    std::string funcName, body;
    size_t paramStart = 0, paramEnd = 0;
    bool isGlobal = false;

    auto r = extractFunction(tokens, funcName, paramStart, paramEnd, body, isGlobal);
    if (r != VGREResult::SUCCESS) {
        VGRE_LOG_ERROR("KernelParser",
                       "Failed to extract function from kernel source");
        return r;
    }

    // Parse parameters
    std::vector<ParsedParam> params;
    auto paramResult = parseParameters(tokens, paramStart, paramEnd, params);
    if (paramResult != VGREResult::SUCCESS) {
        return paramResult;
    }

    // Build KernelIR
    outIR.name   = name.empty() ? funcName : name;
    outIR.source = source;
    outIR.irCode = "";  // Will be filled by LLVMTranslationEngine

    outIR.argTypes.clear();
    for (const auto& p : params) {
        outIR.argTypes.push_back(p.argType);
    }

    // Check for shared memory usage
    outIR.usesSharedMem = (source.find("__shared__") != std::string::npos);

    auto builtins = findBuiltinVars(body);
    VGRE_LOG_INFO("KernelParser",
                  "Parsed AST for kernel '" + outIR.name + "' with " +
                  std::to_string(params.size()) + " params, " +
                  std::to_string(builtins.size()) + " built-in vars");

    return VGREResult::SUCCESS;
}

} // namespace compiler
} // namespace vgre
