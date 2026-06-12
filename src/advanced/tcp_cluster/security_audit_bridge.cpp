// Bridge: cluster security events → compliance audit log; cluster auth token ←
// sealed secret store.  See internal/security_audit_bridge.h.

#include "vgre/advanced/tcp_cluster/internal/security_audit_bridge.h"

#include "vgre/compliance/audit_log.h"
#include "vgre/secrets/secret_store.h"

#include <algorithm>
#include <cctype>

namespace vgre {
namespace advanced {
namespace detail {

void mirror_security_event_to_audit(const std::string& event, const std::string& ip,
                                    const std::string& details) {
    std::string lc = event;
    std::transform(lc.begin(), lc.end(), lc.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    bool failure = lc.find("fail") != std::string::npos ||
                   lc.find("violation") != std::string::npos ||
                   lc.find("denied") != std::string::npos ||
                   lc.find("reject") != std::string::npos ||
                   lc.find("unauthorized") != std::string::npos;
    vgre::compliance::AuditLog::global().emit(
        ip.empty() ? "unknown" : ip, "cluster." + event, "cluster",
        failure ? vgre::compliance::AuditOutcome::DENIED : vgre::compliance::AuditOutcome::SUCCESS,
        failure ? vgre::compliance::AuditSeverity::WARNING : vgre::compliance::AuditSeverity::INFO,
        {{"details", details}});
}

bool resolve_auth_token_from_secret_store(std::string& out) {
    auto& store = vgre::secrets::SecretStore::global();
    if (!store.enabled()) return false;
    std::string token;
    if (store.get("cluster", "cluster-auth-token", token) == vgre::VGREResult::SUCCESS &&
        !token.empty()) {
        out = token;
        return true;
    }
    return false;
}

} // namespace detail
} // namespace advanced
} // namespace vgre
