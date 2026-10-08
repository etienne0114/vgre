// PTX translator entry-point methods.

#include "ptx_translator_internal.h"
#include <algorithm>  // std::find/sort/... (don't rely on transitive includes)
#include <cctype>
#include <limits>
#include <stdexcept>
#include <vector>

namespace vgre {
namespace compiler {

namespace {

static std::vector<std::string> splitAsmFields(const std::string& text, char delimiter) {
    std::vector<std::string> fields;
    std::string field;
    int parens = 0, brackets = 0, braces = 0;
    char quote = '\0';
    bool escaped = false;
    for (char ch : text) {
        if (quote != '\0') {
            field += ch;
            if (escaped) escaped = false;
            else if (ch == '\\') escaped = true;
            else if (ch == quote) quote = '\0';
            continue;
        }
        if (ch == '\"' || ch == '\'') { quote = ch; field += ch; }
        else if (ch == '(') { ++parens; field += ch; }
        else if (ch == ')') { --parens; field += ch; }
        else if (ch == '[') { ++brackets; field += ch; }
        else if (ch == ']') { --brackets; field += ch; }
        else if (ch == '{') { ++braces; field += ch; }
        else if (ch == '}') { --braces; field += ch; }
        else if (ch == delimiter && parens == 0 && brackets == 0 && braces == 0) {
            fields.push_back(trim(field));
            field.clear();
        } else field += ch;
    }
    fields.push_back(trim(field));
    return fields;
}

struct AsmOperand {
    std::string expression;
    std::string name;
};

static std::vector<AsmOperand> parseAsmOperandSection(const std::string& section) {
    std::vector<AsmOperand> result;
    for (const std::string& field : splitAsmFields(section, ',')) {
        const size_t quoteStart = field.find('\"');
        if (quoteStart == std::string::npos) continue;
        std::string name = trim(field.substr(0, quoteStart));
        if (name.size() >= 2 && name.front() == '[' && name.back() == ']')
            name = trim(name.substr(1, name.size() - 2));

        bool escaped = false;
        size_t quoteEnd = quoteStart + 1;
        for (; quoteEnd < field.size(); ++quoteEnd) {
            if (escaped) escaped = false;
            else if (field[quoteEnd] == '\\') escaped = true;
            else if (field[quoteEnd] == '\"') break;
        }
        if (quoteEnd == field.size())
            throw std::runtime_error("unterminated inline PTX constraint string");

        const size_t open = field.find('(', quoteEnd + 1);
        if (open == std::string::npos) continue;
        int depth = 0;
        char quote = '\0';
        escaped = false;
        size_t close = open;
        for (; close < field.size(); ++close) {
            const char ch = field[close];
            if (quote != '\0') {
                if (escaped) escaped = false;
                else if (ch == '\\') escaped = true;
                else if (ch == quote) quote = '\0';
                continue;
            }
            if (ch == '\"' || ch == '\'') quote = ch;
            else if (ch == '(') ++depth;
            else if (ch == ')' && --depth == 0) break;
        }
        if (close == field.size() || depth != 0)
            throw std::runtime_error("unbalanced inline PTX constraint expression");
        std::string expression = trim(field.substr(open + 1, close - open - 1));
        if (expression.empty())
            throw std::runtime_error("empty inline PTX constraint expression");
        result.push_back({expression, name});
    }
    return result;
}

static std::vector<AsmOperand> parseAsmOperands(const std::string& constraints) {
    const std::string trimmed = trim(constraints);
    if (trimmed.empty() || trimmed.front() != ':') return {};
    const auto sections = splitAsmFields(trimmed, ':');
    std::vector<AsmOperand> result;
    if (sections.size() > 1) result = parseAsmOperandSection(sections[1]);
    if (sections.size() > 2) {
        auto inputs = parseAsmOperandSection(sections[2]);
        result.insert(result.end(), inputs.begin(), inputs.end());
    }
    return result;
}

static std::string substituteAsmOperands(const std::string& body,
                                         const std::string& constraints) {
    const auto operands = parseAsmOperands(constraints);
    std::unordered_map<std::string, std::string> named;
    for (const auto& operand : operands)
        if (!operand.name.empty()) named[operand.name] = operand.expression;

    std::string result;
    result.reserve(body.size());
    for (size_t i = 0; i < body.size();) {
        if (body[i] != '%' || i + 1 >= body.size()) {
            result += body[i++];
            continue;
        }
        if (body[i + 1] == '[') {
            const size_t close = body.find(']', i + 2);
            if (close != std::string::npos) {
                const std::string name = body.substr(i + 2, close - i - 2);
                const auto it = named.find(name);
                if (it == named.end())
                    throw std::runtime_error("unknown named inline PTX operand %" + name);
                result += "(" + it->second + ")";
                i = close + 1;
                continue;
            }
        }
        if (std::isdigit(static_cast<unsigned char>(body[i + 1]))) {
            size_t end = i + 1, index = 0;
            while (end < body.size() &&
                   std::isdigit(static_cast<unsigned char>(body[end]))) {
                const size_t digit = static_cast<size_t>(body[end] - '0');
                if (index > (std::numeric_limits<size_t>::max() - digit) / 10)
                    throw std::runtime_error("inline PTX operand index overflow");
                index = index * 10 + digit;
                ++end;
            }
            if (index >= operands.size())
                throw std::runtime_error("inline PTX operand %" + std::to_string(index) +
                                         " has no matching constraint expression");
            result += "(" + operands[index].expression + ")";
            i = end;
            continue;
        }
        result += body[i++];
    }
    return result;
}

static std::string decodeAsmStringSequence(const std::string& sequence) {
    std::string result;
    for (size_t i = 0; i < sequence.size();) {
        if (std::isspace(static_cast<unsigned char>(sequence[i]))) { ++i; continue; }
        if (sequence[i++] != '\"')
            throw std::runtime_error("invalid inline PTX string literal sequence");
        while (i < sequence.size() && sequence[i] != '\"') {
            if (sequence[i] != '\\' || i + 1 == sequence.size()) {
                result += sequence[i++];
                continue;
            }
            const char escaped = sequence[i + 1];
            switch (escaped) {
            case 'n': result += '\n'; break;
            case 't': result += '\t'; break;
            case 'r': result += '\r'; break;
            case '\\': result += '\\'; break;
            case '\"': result += '\"'; break;
            default: result += escaped; break;
            }
            i += 2;
        }
        if (i == sequence.size())
            throw std::runtime_error("unterminated inline PTX string literal");
        ++i;
    }
    return result;
}

static size_t findAsmCallEnd(const std::string& source, size_t from) {
    int depth = 0;
    char quote = '\0';
    bool escaped = false;
    for (size_t i = from; i < source.size(); ++i) {
        const char ch = source[i];
        if (quote != '\0') {
            if (escaped) escaped = false;
            else if (ch == '\\') escaped = true;
            else if (ch == quote) quote = '\0';
            continue;
        }
        if (ch == '\"' || ch == '\'') quote = ch;
        else if (ch == '(') ++depth;
        else if (ch == ')') {
            if (depth == 0) return i;
            --depth;
        }
    }
    throw std::runtime_error("unterminated inline PTX asm expression");
}

} // namespace

std::string PTXTranslator::translateInstruction(
    const std::string& instr, const std::string& operands)
{
    const TranslateMap* maps[] = { &getMap(), &getTextureMap(),
                                   &getSharedAtomicMap(), &getConversionMap() };
    for (const auto* m : maps) {
        auto it = m->find(instr);
        if (it != m->end()) {
            auto ops = splitOperands(operands);
            try { return it->second(ops); }
            catch (...) {
                VGRE_LOG_ERROR("PTXTranslator",
                    "PTX operand error for instruction '" + instr + "'");
                throw std::runtime_error("PTX operand error: " + instr);
            }
        }
    }
    VGRE_LOG_ERROR("PTXTranslator",
        "Unrecognized PTX instruction '" + instr + "' — not supported for production");
    throw std::runtime_error("PTX instruction not supported: " + instr);
}

std::string PTXTranslator::translateBlock(
    const std::string& ptxBody,
    const std::string& constraints,
    const std::string& /*clobbers*/)
{
    const std::string resolvedBody = substituteAsmOperands(ptxBody, constraints);
    // Declare CC register as a local variable if any carry instructions are present.
    bool needsCC = resolvedBody.find("add.cc") != std::string::npos ||
                   resolvedBody.find("sub.cc") != std::string::npos ||
                   resolvedBody.find("addc")   != std::string::npos ||
                   resolvedBody.find("subc")   != std::string::npos ||
                   resolvedBody.find("mad.hi.cc") != std::string::npos;

    std::ostringstream out;
    if (needsCC) out << "  int _cc = 0; /* PTX carry-flag register */\n";

    std::istringstream lines(resolvedBody);
    std::string line;
    while (std::getline(lines, line)) {
        std::string t = trim(line);
        if (t.empty() || t[0] == '/') {
            out << "  " << "/* " << t << " */\n";
            continue;
        }
        if (t[0] == '@')
            throw std::runtime_error("predicated inline PTX is not supported: " + t);
        // Remove trailing semicolon
        if (!t.empty() && t.back() == ';') t.pop_back();

        // Split: first token is opcode, rest is operands
        size_t sp = t.find(' ');
        std::string opcode  = (sp == std::string::npos) ? t : t.substr(0, sp);
        std::string operStr = (sp == std::string::npos) ? "" : trim(t.substr(sp + 1));
        std::transform(opcode.begin(), opcode.end(), opcode.begin(), ::tolower);

        out << "  " << translateInstruction(opcode, operStr) << "\n";
    }
    return out.str();
}

std::string PTXTranslator::translate(const std::string& source) {
    // Match the complete adjacent string-literal sequence. The closing asm
    // parenthesis is parsed separately because constraint expressions contain
    // their own nested parentheses.
    // Leaky static (never destroyed): the JIT background worker may run this
    // translation while the process is exiting and main-thread __cxa_atexit
    // handlers are destroying static locals.  A by-value static std::regex would
    // be freed out from under the worker (use-after-free / data race) and
    // corrupt the generated wrapper, which then crashes deep in LLVM codegen.
    // Heap-allocating once with no destructor eliminates that teardown race.
    static const std::regex *kAsmRe = new std::regex(
        "\\b(?:__asm__|asm)\\s*(?:(?:volatile|__volatile__)\\s*)?\\(\\s*((?:\"(?:[^\"\\\\]|\\\\.)*\"\\s*)+)",
        std::regex::ECMAScript);

    std::string out;
    out.reserve(source.size());
    size_t pos = 0;
    size_t searchPos = 0;
    while (searchPos < source.size()) {
        std::smatch match;
        const std::string remaining = source.substr(searchPos);
        if (!std::regex_search(remaining, match, *kAsmRe)) break;
        const size_t start = searchPos + static_cast<size_t>(match.position());
        const size_t afterLiterals = searchPos +
            static_cast<size_t>(match.position() + match.length());
        const size_t callEnd = findAsmCallEnd(source, afterLiterals);
        const std::string body = decodeAsmStringSequence(match[1].str());
        const std::string constraints = source.substr(afterLiterals,
                                                       callEnd - afterLiterals);
        out += source.substr(pos, start - pos);

        out += "/* PTX begin */\n";
        out += "{\n";
        out += translateBlock(body, constraints, "");
        out += "}\n";
        out += "/* PTX end */";

        pos = callEnd + 1;
        searchPos = pos;
    }
    out += source.substr(pos);

    if (pos > 0)
        VGRE_LOG_DEBUG("PTXTranslator",
            "Translated inline PTX assembly (" + std::to_string(pos) + " chars processed)");
    return out;
}

bool PTXTranslator::tryTranslate(const std::string& source,
                                 std::string& translated,
                                 std::string& error) {
    translated.clear();
    error.clear();
    try {
        std::string result = translate(source);
        translated.swap(result);
        return true;
    } catch (const std::exception& exception) {
        error = exception.what();
    } catch (...) {
        error = "unknown PTX translation failure";
    }
    return false;
}

} // namespace compiler
} // namespace vgre
