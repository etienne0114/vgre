// VGRE Compliance — Cryptographically-Integrity-Protected Audit Log
// ---------------------------------------------------------------------------
// An append-only, tamper-evident audit trail for privileged operations, built
// on the project's own crypto primitives (vgre::advanced::crypto).  This is the
// compliance foundation referenced by docs/missingFeatures.md §1.3 (SOC 2 / ISO
// 27001 / FedRAMP audit logging) and §1.3 GDPR "right to be forgotten".
//
// Integrity model (real, not a placeholder):
//   * Every record is chained with a keyed MAC:
//         mac_i = HMAC-SHA256( chainKey, mac_{i-1} || canonical(record_i) )
//     with mac_0 = HMAC-SHA256( chainKey, "VGRE-AUDIT-GENESIS-v1" ).
//     Because the chain is keyed, an attacker who cannot recover chainKey can
//     neither forge, reorder, truncate, nor edit any record without the
//     recomputed head diverging — i.e. the log is tamper-EVIDENT and (given a
//     secret key) tamper-PROOF.  verify() re-reads the on-disk file and walks
//     the whole chain, reporting the first record whose stored MAC disagrees.
//
// GDPR crypto-erasure (real):
//   * PII never touches the chain in cleartext.  When a record carries a
//     subjectId + piiPlaintext, the plaintext is sealed with a per-subject
//     random key (AES-256-GCM) and only the ciphertext is hashed/persisted.
//     The per-subject keys live in a separate, mutable key table wrapped under
//     a master key.  forget(subjectId) destroys that subject's key (crypto-
//     shredding): the immutable chain stays intact and verifiable, but the
//     sealed PII becomes permanently unrecoverable.  A signed Deletion
//     Certificate is emitted as machine-checkable proof of erasure.
//
// Thread-safe.  Cross-platform (std::fstream + the portable crypto layer).

#ifndef VGRE_COMPLIANCE_AUDIT_LOG_H
#define VGRE_COMPLIANCE_AUDIT_LOG_H

#include "vgre/common/error_codes.h"

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace vgre {
namespace compliance {

// PascalCase values avoid Windows SDK macro collisions:
//   <wingdi.h>: #define ERROR 0
//   <winnt.h>:  no WARNING, but naming is kept consistent with GuardStatus
enum class AuditOutcome  : uint8_t { Ok = 0, Denied = 1, Err = 2 };
enum class AuditSeverity : uint8_t { Info = 0, Notice = 1, Warn = 2, Critical = 3 };

// One audit event.  `subjectId` + `piiPlaintext` are optional; when present the
// plaintext is crypto-sealed (see header note) and `piiPlaintext` is cleared
// before anything is persisted.  `mac`/`redacted` are filled by the log.
struct AuditRecord {
    uint64_t      seq = 0;            // monotonic, assigned by append()
    uint64_t      timestamp_ns = 0;   // wall-clock ns, assigned by append()
    std::string   actor;              // who performed the action
    std::string   action;            // e.g. "auth.login", "secret.read", "kernel.load"
    std::string   resource;          // e.g. "device:0", "cluster:node-3"
    AuditOutcome  outcome = AuditOutcome::Ok;
    AuditSeverity severity = AuditSeverity::Info;
    std::map<std::string, std::string> attributes;

    // Optional crypto-shreddable PII payload (pseudonymised: subjectId should be
    // a stable pseudonym, not the raw PII; the raw PII goes in piiPlaintext).
    std::string   subjectId;          // empty = no PII
    std::string   piiPlaintext;       // cleared after sealing; never persisted in clear

    std::array<uint8_t, 32> mac{};    // chained MAC head at this record
    bool          redacted = false;   // true in query results when PII was crypto-erased
};

// Machine-checkable proof that a subject's PII was crypto-erased.
struct DeletionCertificate {
    std::array<uint8_t, 32> subjectIdHash{}; // SHA-256(subjectId)
    std::vector<uint64_t>   affectedSeqs;     // records whose PII was shredded
    uint64_t                timestamp_ns = 0;
    uint64_t                tombstoneSeq = 0;  // seq of the appended erasure record
    std::array<uint8_t, 32> proof{};           // HMAC(chainKey, certificate body)
};

struct AuditQuery {
    std::string actorEquals;     // empty = any
    std::string actionEquals;    // empty = any
    std::string resourceEquals;  // empty = any
    uint64_t    sinceNs = 0;      // inclusive lower bound (0 = from start)
    uint64_t    untilNs = UINT64_MAX; // inclusive upper bound
    AuditSeverity minSeverity = AuditSeverity::Info;
    size_t      limit = 0;        // 0 = unlimited
};

class AuditLog {
public:
    ~AuditLog();

    // Open (creating if needed) the audit log at `path`.  `chainKey` seals the
    // integrity chain; `piiMasterKey` wraps per-subject PII keys.  On open the
    // existing chain is loaded and fully verified — open() fails with ERR_CRYPTO
    // if the on-disk chain has been tampered with.
    static vgre::VGREResult open(const std::string& path,
                                 const uint8_t chainKey[32],
                                 const uint8_t piiMasterKey[32],
                                 std::unique_ptr<AuditLog>& out);

    // Append an event.  Assigns seq/timestamp, seals any PII, extends the MAC
    // chain, and durably writes the record.  Returns the assigned seq.
    vgre::VGREResult append(AuditRecord rec, uint64_t& outSeq);

    // Convenience producer used by instrumented call sites.
    vgre::VGREResult emit(const std::string& actor, const std::string& action,
                          const std::string& resource, AuditOutcome outcome,
                          AuditSeverity severity,
                          std::map<std::string, std::string> attributes = {});

    // Re-read the on-disk file and walk the entire chain.  Returns SUCCESS and
    // sets firstBadSeq = UINT64_MAX when intact; otherwise ERR_CRYPTO with
    // firstBadSeq = the seq of the first record that fails verification.
    vgre::VGREResult verify(uint64_t& firstBadSeq) const;

    // Filtered read-back.  PII is decrypted when the subject key still exists;
    // crypto-erased records come back with redacted=true and empty PII.
    std::vector<AuditRecord> query(const AuditQuery& q) const;

    // GDPR Art. 17 crypto-erasure: destroy the subject's PII key, append a
    // tamper-evident tombstone, and produce a verifiable DeletionCertificate.
    vgre::VGREResult forget(const std::string& subjectId, DeletionCertificate& outCert);

    // Verify a previously issued DeletionCertificate against this log's key.
    bool verifyDeletionCertificate(const DeletionCertificate& cert) const;

    uint64_t count() const;                 // number of records
    std::array<uint8_t, 32> head() const;   // current chain head (publishable anchor)

    // Process-wide instance.  Active iff VGRE_AUDIT_LOG (a file path) is set;
    // otherwise a disabled no-op sink (enabled()==false) so instrumented call
    // sites pay nothing when auditing is off.  Keys are derived from
    // VGRE_AUDIT_KEY (hex) when present, else from a 0600 sidecar key file
    // created on first use.
    static AuditLog& global();
    bool enabled() const;

private:
    AuditLog();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace compliance
} // namespace vgre

#endif // VGRE_COMPLIANCE_AUDIT_LOG_H
