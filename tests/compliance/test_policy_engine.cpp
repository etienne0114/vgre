// Compliance — data classification (Luhn PAN, SSN/email PII, PHI) + policy engine.

#include "vgre/compliance/policy_engine.h"

#include <cstdio>
#include <string>

using namespace vgre::compliance;

static int g_pass = 0, g_total = 0;
static void check(const char* name, bool ok) {
    ++g_total;
    std::printf(ok ? "  PASS  %s\n" : "  FAIL  %s\n", name);
    if (ok) ++g_pass;
}

int main() {
    std::printf("=== Compliance policy engine + data classification ===\n");

    // ── Luhn ─────────────────────────────────────────────────────────────
    check("Luhn valid test PAN (4242 4242 4242 4242)", DataClassifier::luhnValid("4242424242424242"));
    check("Luhn rejects bad PAN", !DataClassifier::luhnValid("4242424242424241"));

    // ── Classification ───────────────────────────────────────────────────
    {
        Classification c = DataClassifier::classify("card on file: 4242 4242 4242 4242 thanks");
        check("PAN detected → PCI + RESTRICTED", c.hasPCI() && c.level == DataClass::RESTRICTED);
    }
    {
        Classification c = DataClassifier::classify("contact john@example.com, SSN 123-45-6789");
        check("email + SSN → PII", c.piiCount >= 2);
        check("PII without PCI/PHI → CONFIDENTIAL", c.level == DataClass::CONFIDENTIAL);
    }
    {
        Classification c = DataClassifier::classify("Patient diagnosis recorded in medical record MRN 5");
        check("PHI keywords → RESTRICTED", c.hasPHI() && c.level == DataClass::RESTRICTED);
    }
    {
        Classification c = DataClassifier::classify("just a normal log line, nothing sensitive");
        check("benign content → PUBLIC", c.level == DataClass::PUBLIC && !c.hasPII() && !c.hasPCI());
    }

    // ── Policy engine ────────────────────────────────────────────────────
    {
        CompliancePolicyEngine eng;
        eng.enableFramework("PCI_DSS");
        eng.enableFramework("HIPAA");
        eng.enableFramework("GDPR");
        eng.enableFramework("SOX");

        // Exporting RESTRICTED (card) data unencrypted → DENY + encryption obligation.
        PolicyContext exp;
        exp.action = "data.export"; exp.resource = "dataset/cards";
        exp.classification = DataClass::RESTRICTED; exp.encryptedInTransit = false;
        exp.principal = "alice";
        PolicyDecision d = eng.evaluate(exp);
        check("unencrypted RESTRICTED export DENIED", d.effect == PolicyEffect::DENY);
        check("DENY carries a rule id", !d.ruleId.empty());

        // Same but encrypted → ALLOW with obligations.
        exp.encryptedInTransit = true;
        PolicyDecision d2 = eng.evaluate(exp);
        check("encrypted RESTRICTED export ALLOWED", d2.effect == PolicyEffect::ALLOW);
        check("encryption + audit obligations set", d2.obligations.requireEncryption && d2.obligations.requireAudit);

        // GDPR: exporting CONFIDENTIAL PII requires MFA.
        PolicyContext pii;
        pii.action = "data.export"; pii.classification = DataClass::CONFIDENTIAL; pii.encryptedInTransit = true;
        check("GDPR PII export requires MFA", eng.evaluate(pii).obligations.requireMFA);

        // SOX: modifying records must be audited; benign public read is allowed clean.
        PolicyContext mod; mod.action = "record.modify"; mod.classification = DataClass::INTERNAL;
        check("SOX modify requires audit", eng.evaluate(mod).obligations.requireAudit);

        PolicyContext pub; pub.action = "data.read"; pub.classification = DataClass::PUBLIC;
        PolicyDecision dp = eng.evaluate(pub);
        check("public read allowed without obligations",
              dp.effect == PolicyEffect::ALLOW && !dp.obligations.requireEncryption && !dp.obligations.requireMFA);

        // ── Report ───────────────────────────────────────────────────────
        std::string rep = eng.report({exp, pii, mod, pub});
        check("report counts frameworks + decisions",
              rep.find("PCI_DSS") != std::string::npos && rep.find("\"evaluated\":4") != std::string::npos);
    }

    std::printf("\n%d / %d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
