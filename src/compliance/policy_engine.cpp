// VGRE Compliance — classification + policy engine (impl).  See policy_engine.h.

#include "vgre/compliance/policy_engine.h"
#include "vgre/compliance/audit_log.h"

#include <algorithm>
#include <cctype>
#include <sstream>

namespace vgre {
namespace compliance {

// ── Data classification ─────────────────────────────────────────────────────
bool DataClassifier::luhnValid(const std::string& digits) {
    int n = 0;
    for (char c : digits) if (std::isdigit((unsigned char)c)) ++n;
    if (n < 13 || n > 19) return false;
    int sum = 0;
    bool dbl = false;
    // Walk right-to-left over digit characters only.
    for (auto it = digits.rbegin(); it != digits.rend(); ++it) {
        if (!std::isdigit((unsigned char)*it)) continue;
        int d = *it - '0';
        if (dbl) { d *= 2; if (d > 9) d -= 9; }
        sum += d;
        dbl = !dbl;
    }
    return (sum % 10) == 0;
}

namespace {
bool looksLikeSSN(const std::string& s, size_t i) {
    // ddd-dd-dddd
    auto dig = [&](size_t k) { return i + k < s.size() && std::isdigit((unsigned char)s[i + k]); };
    auto dash = [&](size_t k) { return i + k < s.size() && s[i + k] == '-'; };
    return dig(0) && dig(1) && dig(2) && dash(3) && dig(4) && dig(5) && dash(6) &&
           dig(7) && dig(8) && dig(9) && dig(10) &&
           (i + 11 >= s.size() || !std::isdigit((unsigned char)s[i + 11]));
}
std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char)std::tolower(c); });
    return s;
}
} // namespace

Classification DataClassifier::classify(const std::string& content) {
    Classification c;

    // PCI: scan runs of digits (with spaces/dashes) of length 13-19, Luhn-check.
    for (size_t i = 0; i < content.size();) {
        if (!std::isdigit((unsigned char)content[i])) { ++i; continue; }
        size_t j = i;
        std::string run;
        while (j < content.size() &&
               (std::isdigit((unsigned char)content[j]) || content[j] == ' ' || content[j] == '-')) {
            if (std::isdigit((unsigned char)content[j])) run.push_back(content[j]);
            ++j;
        }
        if (luhnValid(run)) ++c.pciCount;
        i = j;
    }

    // PII: SSN patterns + emails.
    for (size_t i = 0; i < content.size(); ++i)
        if (looksLikeSSN(content, i)) ++c.piiCount;
    for (size_t i = 0; i < content.size(); ++i) {
        if (content[i] == '@' && i > 0 && i + 2 < content.size()) {
            bool localOk = std::isalnum((unsigned char)content[i - 1]);
            bool hasDot = content.find('.', i) != std::string::npos;
            if (localOk && hasDot) ++c.piiCount;
        }
    }

    // PHI: keyword scan (HIPAA protected-health indicators).
    std::string lc = lower(content);
    static const char* phiKw[] = {"diagnosis", "patient", "medical record",
                                  "icd-10", "prescription", "health record", "mrn"};
    for (const char* kw : phiKw)
        if (lc.find(kw) != std::string::npos) ++c.phiCount;

    // Level: RESTRICTED if PCI/PHI, CONFIDENTIAL if PII, else PUBLIC.
    if (c.pciCount > 0 || c.phiCount > 0) c.level = DataClass::RESTRICTED;
    else if (c.piiCount > 0) c.level = DataClass::CONFIDENTIAL;
    else c.level = DataClass::PUBLIC;
    return c;
}

// ── Policy engine ───────────────────────────────────────────────────────────
void CompliancePolicyEngine::enableFramework(const std::string& name) {
    if (!frameworkEnabled(name)) frameworks_.push_back(name);
}
bool CompliancePolicyEngine::frameworkEnabled(const std::string& name) const {
    return std::find(frameworks_.begin(), frameworks_.end(), name) != frameworks_.end();
}

PolicyDecision CompliancePolicyEngine::evaluate(const PolicyContext& ctx) const {
    PolicyDecision dec; // default ALLOW
    bool isExport = ctx.action.find("export") != std::string::npos ||
                    ctx.action.find("transmit") != std::string::npos ||
                    ctx.action.find("share") != std::string::npos;
    bool isModify = ctx.action.find("modify") != std::string::npos ||
                    ctx.action.find("write") != std::string::npos ||
                    ctx.action.find("delete") != std::string::npos;
    bool restricted = ctx.classification == DataClass::RESTRICTED;
    bool confidential = ctx.classification >= DataClass::CONFIDENTIAL;

    auto deny = [&](const char* id, const char* why) {
        dec.effect = PolicyEffect::DENY; dec.ruleId = id; dec.reason = why;
    };

    // PCI DSS: cardholder data must be encrypted in transit/at rest.
    if (frameworkEnabled("PCI_DSS") && restricted) {
        dec.obligations.requireEncryption = true;
        dec.obligations.requireAudit = true;
        if (isExport && !ctx.encryptedInTransit) deny("PCI-DSS-4.1", "cardholder data exported unencrypted");
    }
    // HIPAA: PHI access requires audit; export requires encryption.
    if (frameworkEnabled("HIPAA") && restricted) {
        dec.obligations.requireAudit = true;
        if (isExport) {
            dec.obligations.requireEncryption = true;
            if (!ctx.encryptedInTransit && dec.effect == PolicyEffect::ALLOW)
                deny("HIPAA-164.312(e)", "ePHI transmitted without encryption");
        }
    }
    // GDPR: personal-data export requires audit + MFA on the actor.
    if (frameworkEnabled("GDPR") && confidential && isExport) {
        dec.obligations.requireAudit = true;
        dec.obligations.requireMFA = true;
    }
    // SOX: modification of financial records must be audited.
    if (frameworkEnabled("SOX") && isModify) {
        dec.obligations.requireAudit = true;
    }

    // Mirror the decision into the tamper-evident audit trail.
    AuditLog::global().emit(
        ctx.principal.empty() ? "system" : ctx.principal,
        std::string("compliance.") + ctx.action, ctx.resource,
        dec.effect == PolicyEffect::DENY ? AuditOutcome::Denied : AuditOutcome::Ok,
        dec.effect == PolicyEffect::DENY ? AuditSeverity::Critical : AuditSeverity::Notice,
        {{"rule", dec.ruleId}, {"class", std::to_string((int)ctx.classification)}});
    return dec;
}

std::string CompliancePolicyEngine::report(const std::vector<PolicyContext>& contexts) const {
    int allowed = 0, denied = 0, encObl = 0, auditObl = 0, mfaObl = 0;
    for (const auto& ctx : contexts) {
        PolicyDecision d = evaluate(ctx);
        if (d.effect == PolicyEffect::DENY) ++denied; else ++allowed;
        if (d.obligations.requireEncryption) ++encObl;
        if (d.obligations.requireAudit) ++auditObl;
        if (d.obligations.requireMFA) ++mfaObl;
    }
    std::ostringstream os;
    os << "{\"frameworks\":[";
    for (size_t i = 0; i < frameworks_.size(); ++i) os << (i ? "," : "") << "\"" << frameworks_[i] << "\"";
    os << "],\"evaluated\":" << contexts.size()
       << ",\"allowed\":" << allowed << ",\"denied\":" << denied
       << ",\"obligations\":{\"encryption\":" << encObl << ",\"audit\":" << auditObl
       << ",\"mfa\":" << mfaObl << "}}";
    return os.str();
}

} // namespace compliance
} // namespace vgre
