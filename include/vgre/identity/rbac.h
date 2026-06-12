// VGRE Identity — Role-Based Access Control engine.
// Maps roles → permissions (with wildcard + role inheritance) and authorizes a
// verified token's roles/scopes against a requested permission.  Every decision
// is mirrored into the compliance audit log.
#ifndef VGRE_IDENTITY_RBAC_H
#define VGRE_IDENTITY_RBAC_H

#include "vgre/common/error_codes.h"
#include "vgre/identity/jwt_verifier.h"

#include <map>
#include <set>
#include <string>
#include <vector>

namespace vgre {
namespace identity {

// Permission strings are colon-scoped (e.g. "device:read", "cluster:admin").
// A granted "x:*" matches any "x:<verb>"; a granted "*" matches everything.
class RbacPolicy {
public:
    // Grant `permission` to `role`.
    void grant(const std::string& role, const std::string& permission);
    // `role` additionally inherits every permission of `parentRole`.
    void inherit(const std::string& role, const std::string& parentRole);

    // True iff any of `roles` (with inheritance + wildcards) grants `permission`.
    bool can(const std::vector<std::string>& roles, const std::string& permission) const;

    // Authorize a verified token.  Emits an audit record (actor = claims.sub,
    // action "authz.check", resource = permission) and returns SUCCESS when
    // allowed, ERR_AUTH_FAILED when denied.  `scopesAsRoles` also treats OAuth
    // scopes as grantable roles (common for client-credentials tokens).
    vgre::VGREResult authorize(const JwtClaims& claims, const std::string& permission,
                               bool scopesAsRoles = true) const;

    size_t roleCount() const { return rolePermissions_.size(); }

private:
    // Expand a role to itself + all ancestors (cycle-safe).
    void expandRoles(const std::string& role, std::set<std::string>& out) const;

    std::map<std::string, std::set<std::string>> rolePermissions_;
    std::map<std::string, std::set<std::string>> roleParents_;
};

} // namespace identity
} // namespace vgre

#endif // VGRE_IDENTITY_RBAC_H
