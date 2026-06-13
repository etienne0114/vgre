// VGRE Compliance — data classification + policy-as-code engine.
// docs/missingFeatures.md §11.3.  Real, self-contained: a content classifier
// (Luhn-validated PAN detection, SSN/email PII, PHI keyword scan) that assigns a
// sensitivity label, and a rule engine that evaluates operations against built-in
// PCI-DSS / HIPAA / GDPR / SOX rule sets, emitting allow/deny + obligations
// (encryption / audit / MFA). Decisions are mirrored into the §1.3 audit log.
// Framework *certification* remains an external attestation against this evidence.
#ifndef VGRE_COMPLIANCE_POLICY_ENGINE_H
#define VGRE_COMPLIANCE_POLICY_ENGINE_H

#include <cstdint>
#include <string>
#include <vector>

namespace vgre {
namespace compliance {

enum class DataClass : uint8_t { PUBLIC = 0, INTERNAL = 1, CONFIDENTIAL = 2, RESTRICTED = 3 };

struct Classification {
    DataClass level = DataClass::PUBLIC;
    int  piiCount = 0;  // SSN / email / phone
    int  pciCount = 0;  // Luhn-valid card numbers
    int  phiCount = 0;  // protected health info keywords
    bool hasPII() const { return piiCount > 0; }
    bool hasPCI() const { return pciCount > 0; }
    bool hasPHI() const { return phiCount > 0; }
};

class DataClassifier {
public:
    // Scan `content` and assign a sensitivity label + category counts.
    static Classification classify(const std::string& content);
    // Luhn checksum (exposed for testing / PAN validation).
    static bool luhnValid(const std::string& digits);
};

enum class PolicyEffect : uint8_t { ALLOW, DENY };

struct Obligations {
    bool requireEncryption = false;
    bool requireAudit = false;
    bool requireMFA = false;
};

// The operation being checked.
struct PolicyContext {
    std::string action;                 // e.g. "data.export", "record.modify"
    std::string resource;               // e.g. "dataset/patients"
    DataClass   classification = DataClass::PUBLIC;
    bool        encryptedInTransit = false;
    bool        encryptedAtRest = false;
    std::string principal;              // for audit
};

struct PolicyDecision {
    PolicyEffect effect = PolicyEffect::ALLOW;
    Obligations  obligations;
    std::string  ruleId;   // the deciding rule (empty = default allow)
    std::string  reason;
};

class CompliancePolicyEngine {
public:
    // Built-in rule sets: "PCI_DSS", "HIPAA", "GDPR", "SOX".  Multiple may be on.
    void enableFramework(const std::string& name);
    bool frameworkEnabled(const std::string& name) const;

    // Evaluate; the most restrictive matching rule wins (any DENY → DENY), and
    // obligations from all matching rules are unioned. Audited via the §1.3 log.
    PolicyDecision evaluate(const PolicyContext& ctx) const;

    // JSON compliance report over a batch of evaluated contexts (pass/deny counts
    // per framework + obligation summary).
    std::string report(const std::vector<PolicyContext>& contexts) const;

private:
    std::vector<std::string> frameworks_;
};

} // namespace compliance
} // namespace vgre

#endif // VGRE_COMPLIANCE_POLICY_ENGINE_H
