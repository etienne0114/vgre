// Bridge between the TCP-cluster security manager and the Phase-4 compliance /
// secrets subsystems.  Kept in its own translation unit so security_manager.cpp
// stays within the module-size budget enforced by the architecture test.
#ifndef VGRE_ADVANCED_TCP_CLUSTER_SECURITY_AUDIT_BRIDGE_H
#define VGRE_ADVANCED_TCP_CLUSTER_SECURITY_AUDIT_BRIDGE_H

#include <string>

namespace vgre {
namespace advanced {
namespace detail {

// Mirror a cluster security event into the tamper-evident compliance audit log
// (no-op unless VGRE_AUDIT_LOG is configured).  Auth/violation failures are
// recorded as DENIED/WARNING; everything else as an INFO record.
void mirror_security_event_to_audit(const std::string& event, const std::string& ip,
                                    const std::string& details);

// Resolve the cluster auth token from the sealed secret store when configured
// (VGRE_SECRETS_STORE → secret "cluster-auth-token").  Returns true and sets
// `out` when a non-empty token was found; false otherwise.
bool resolve_auth_token_from_secret_store(std::string& out);

} // namespace detail
} // namespace advanced
} // namespace vgre

#endif // VGRE_ADVANCED_TCP_CLUSTER_SECURITY_AUDIT_BRIDGE_H
