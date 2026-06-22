// VGRE Secrets — Sealed, Versioned Secret Store with Rotation & Access Policy
// ---------------------------------------------------------------------------
// docs/missingFeatures.md §9.3 (Advanced Secrets Management).  A self-contained,
// cross-platform secret store: secrets are sealed at rest with AES-256-GCM under
// a master key that is itself rooted in the OS/TPM credential store (via the
// existing HardwareTokenManager) so the master key never lives in cleartext on
// disk.  Each secret is versioned (zero-downtime rotation with a readable grace
// window), guarded by a per-secret access policy, and every access — allowed or
// denied — is mirrored into the tamper-evident compliance audit log.
//
// This is the local, real core.  "Vault/AWS-Secrets-Manager integration" is a
// remote backend bolted onto this same interface; the sealing, versioning,
// policy and audit semantics here are genuine and fully functional offline.

#ifndef VGRE_SECRETS_SECRET_STORE_H
#define VGRE_SECRETS_SECRET_STORE_H

#include "vgre/common/error_codes.h"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace vgre {
namespace secrets {

enum class SecretAction : uint8_t {
    READ   = 0,  // get / getVersion
    WRITE  = 1,  // put (create)
    ROTATE = 2,  // rotate (new version)
    Remove = 3,  // remove — 'DELETE' is #define DELETE 0x00010000L in <winnt.h>
    LIST   = 4,  // list / metadata
    ADMIN  = 5,  // setPolicy / prune
};

// Per-secret authorization: the owner (the put() principal) may do anything;
// other principals need an explicit grant.  Principal "*" grants to everyone.
struct AccessPolicy {
    std::map<std::string, uint32_t> grants; // principal -> bitmask over SecretAction
    void grant(const std::string& principal, SecretAction a) {
        grants[principal] |= (1u << static_cast<uint32_t>(a));
    }
    bool allows(const std::string& principal, SecretAction a) const {
        uint32_t bit = 1u << static_cast<uint32_t>(a);
        auto it = grants.find(principal);
        if (it != grants.end() && (it->second & bit)) return true;
        auto star = grants.find("*");
        return star != grants.end() && (star->second & bit);
    }
};

struct SecretMetadata {
    std::string name;
    std::string owner;
    uint32_t    currentVersion = 0;
    uint32_t    versionCount = 0;
    uint64_t    created_ns = 0;
    uint64_t    rotated_ns = 0;
};

class SecretStore {
public:
    ~SecretStore();

    // Open/create a store sealed under an explicit 32-byte master key.
    static vgre::VGREResult open(const std::string& path, const uint8_t masterKey[32],
                                 std::unique_ptr<SecretStore>& out);

    // Open/create a store whose master key is rooted in the OS/TPM credential
    // store (HardwareTokenManager service "vgre-secrets-master"), falling back
    // to VGRE_SECRETS_KEY (hex) or a 0600 sidecar.  This is the production path.
    static vgre::VGREResult openDefault(const std::string& path,
                                        std::unique_ptr<SecretStore>& out);

    // Create a secret at version 1, owned by `principal`.  ERR_ALREADY_EXISTS if
    // it exists (use rotate() to add versions).
    vgre::VGREResult put(const std::string& principal, const std::string& name,
                         const std::string& value);

    // Read the current (or a specific) version's plaintext.
    vgre::VGREResult get(const std::string& principal, const std::string& name,
                         std::string& outValue) const;
    vgre::VGREResult getVersion(const std::string& principal, const std::string& name,
                                uint32_t version, std::string& outValue) const;

    // Add a new version and make it current.  Old versions remain readable
    // (zero-downtime rotation) until pruned.  Returns the new version number.
    vgre::VGREResult rotate(const std::string& principal, const std::string& name,
                            const std::string& newValue, uint32_t& newVersion);

    // Delete a secret and securely overwrite its sealed material.
    vgre::VGREResult remove(const std::string& principal, const std::string& name);

    // Drop all but the newest `keepLast` versions (rotation grace cleanup).
    vgre::VGREResult pruneOldVersions(const std::string& principal, const std::string& name,
                                      uint32_t keepLast, size_t& outPruned);

    // List metadata for secrets the principal may LIST.
    std::vector<SecretMetadata> list(const std::string& principal) const;

    // Owner/ADMIN-only policy management.
    vgre::VGREResult setPolicy(const std::string& principal, const std::string& name,
                               const AccessPolicy& policy);
    vgre::VGREResult getPolicy(const std::string& principal, const std::string& name,
                               AccessPolicy& out) const;

    size_t count() const;

    // Process-wide store, active iff VGRE_SECRETS_STORE (a path) is set.
    static SecretStore& global();
    bool enabled() const;

private:
    SecretStore();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace secrets
} // namespace vgre

#endif // VGRE_SECRETS_SECRET_STORE_H
