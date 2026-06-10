// PTX translator entry-point methods.

#include "ptx_translator_internal.h"

namespace vgre {
namespace compiler {

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
    const std::string& /*constraints*/,
    const std::string& /*clobbers*/)
{
    // Declare CC register as a local variable if any carry instructions are present.
    bool needsCC = ptxBody.find("add.cc") != std::string::npos ||
                   ptxBody.find("sub.cc") != std::string::npos ||
                   ptxBody.find("addc")   != std::string::npos ||
                   ptxBody.find("subc")   != std::string::npos ||
                   ptxBody.find("mad.hi.cc") != std::string::npos;

    std::ostringstream out;
    if (needsCC) out << "  int _cc = 0; /* PTX carry-flag register */\n";

    std::istringstream lines(ptxBody);
    std::string line;
    while (std::getline(lines, line)) {
        std::string t = trim(line);
        if (t.empty() || t[0] == '/' || t[0] == '@') {
            out << "  " << "/* " << t << " */\n";
            continue;
        }
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
    // Pattern: asm [volatile] ("ptx_body" : constraints...)
    // We match both single-string and multi-line forms.
    // Match: asm [volatile] ("body" : constraints)
    // Leaky static (never destroyed): the JIT background worker may run this
    // translation while the process is exiting and main-thread __cxa_atexit
    // handlers are destroying static locals.  A by-value static std::regex would
    // be freed out from under the worker (use-after-free / data race) and
    // corrupt the generated wrapper, which then crashes deep in LLVM codegen.
    // Heap-allocating once with no destructor eliminates that teardown race.
    static const std::regex *kAsmRe = new std::regex(
        "\\b(?:__asm__|asm)\\s*(?:volatile\\s*)?\\(\\s*\"([^\"\\\\]*(?:\\\\.[^\"\\\\]*)*)\"\\s*(:[^)]*)?\\s*\\)",
        std::regex::ECMAScript);

    std::string result = source;
    std::string out;
    out.reserve(source.size());

    auto it  = std::sregex_iterator(result.begin(), result.end(), *kAsmRe);
    auto end = std::sregex_iterator();
    size_t pos = 0;

    for (; it != end; ++it) {
        const auto& m = *it;
        out += result.substr(pos, m.position() - pos);

        std::string ptxBody = m[1].str();
        std::string rest    = m.size() > 2 ? m[2].str() : "";

        // Replace escaped newlines (\n\t etc.) with real newlines
        std::string body;
        for (size_t i = 0; i < ptxBody.size(); ++i) {
            if (ptxBody[i] == '\\' && i+1 < ptxBody.size()) {
                switch (ptxBody[i+1]) {
                    case 'n': body += '\n'; ++i; break;
                    case 't': body += '\t'; ++i; break;
                    default:  body += ptxBody[i]; break;
                }
            } else body += ptxBody[i];
        }

        out += "/* PTX begin */\n";
        out += "{\n";
        out += translateBlock(body, rest, "");
        out += "}\n";
        out += "/* PTX end */";

        pos = m.position() + m.length();
    }
    out += result.substr(pos);

    if (pos > 0)
        VGRE_LOG_DEBUG("PTXTranslator",
            "Translated inline PTX assembly (" + std::to_string(pos) + " chars processed)");
    return out;
}

} // namespace compiler
} // namespace vgre

