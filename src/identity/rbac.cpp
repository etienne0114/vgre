// VGRE Identity — RBAC engine (impl).  See rbac.h.

#include "vgre/identity/rbac.h"
#include "vgre/compliance/audit_log.h"

namespace vgre {
namespace identity {

namespace {
// Does a granted permission pattern match a requested permission?
// "*" matches anything; "x:*" matches "x:<verb>"; otherwise exact match.
bool permMatches(const std::string& granted, const std::string& requested) {
    if (granted == "*" || granted == requested) return true;
    if (granted.size() >= 2 && granted.compare(granted.size() - 2, 2, ":*") == 0) {
        std::string prefix = granted.substr(0, granted.size() - 1); // keep trailing ':'
        return requested.size() >= prefix.size() &&
               requested.compare(0, prefix.size(), prefix) == 0;
    }
    return false;
}
} // namespace

void RbacPolicy::grant(const std::string& role, const std::string& permission) {
    rolePermissions_[role].insert(permission);
}

void RbacPolicy::inherit(const std::string& role, const std::string& parentRole) {
    roleParents_[role].insert(parentRole);
}

void RbacPolicy::expandRoles(const std::string& role, std::set<std::string>& out) const {
    if (out.count(role)) return; // cycle-safe
    out.insert(role);
    auto it = roleParents_.find(role);
    if (it != roleParents_.end())
        for (const auto& parent : it->second)
            expandRoles(parent, out);
}

bool RbacPolicy::can(const std::vector<std::string>& roles, const std::string& permission) const {
    std::set<std::string> effective;
    for (const auto& r : roles) expandRoles(r, effective);
    for (const auto& r : effective) {
        auto it = rolePermissions_.find(r);
        if (it == rolePermissions_.end()) continue;
        for (const auto& granted : it->second)
            if (permMatches(granted, permission)) return true;
    }
    return false;
}

vgre::VGREResult RbacPolicy::authorize(const JwtClaims& claims, const std::string& permission,
                                       bool scopesAsRoles) const {
    std::vector<std::string> roles = claims.roles;
    if (scopesAsRoles)
        roles.insert(roles.end(), claims.scopes.begin(), claims.scopes.end());
    bool allowed = can(roles, permission);

    vgre::compliance::AuditLog::global().emit(
        claims.sub.empty() ? "anonymous" : claims.sub, "authz.check", permission,
        allowed ? vgre::compliance::AuditOutcome::SUCCESS
                : vgre::compliance::AuditOutcome::DENIED,
        allowed ? vgre::compliance::AuditSeverity::INFO
                : vgre::compliance::AuditSeverity::WARNING,
        {{"issuer", claims.iss}});

    return allowed ? vgre::VGREResult::SUCCESS : vgre::VGREResult::ERR_AUTH_FAILED;
}

} // namespace identity
} // namespace vgre
