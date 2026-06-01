/**
 * Windows Socket Manager - Recovery and Diagnostics
 * 
 * Provides mechanisms for Winsock recovery and failure diagnostics.
 */

#include "vgre/advanced/tcp_cluster/internal/windows_socket_manager.h"
#include "vgre/common/logger.h"
#include <thread>
#include <chrono>

#include "vgre/common/os_backend.h"

namespace vgre {
namespace advanced {

#ifdef _WIN32

VGREResult WindowsSocketManager::attemptRecovery(int max_retries) {
    // Caller (initialize() / forceReinitialize()) already holds mutex_.
    if (getInitialized()) { WSACleanup(); getInitialized() = false; }
    getRefCount() = 0;

    for (int attempt = 1; attempt <= max_retries; ++attempt) {
        if (attempt > 1) {
            // Cap shift at 30 to prevent overflow: 100 * 2^30 = ~107 s max
            int backoff_ms = static_cast<int>(
                100LL * (1LL << std::min(attempt - 2, 30)));
            std::mutex wsa_retry_mutex;
            std::condition_variable wsa_retry_cv;
            std::unique_lock<std::mutex> lk(wsa_retry_mutex);
            wsa_retry_cv.wait_for(lk, std::chrono::milliseconds(backoff_ms));
        }
        int result = WSAStartup(MAKEWORD(2, 2), &getWsaData());
        if (result == 0) {
            getInitialized() = true;
            getRefCount() = 1;
            return VGREResult::SUCCESS;
        }
    }
    return VGREResult::ERR_IO;
}

std::string WindowsSocketManager::getWSAStartupDiagnostics(int error_code) {
    switch (error_code) {
        case WSASYSNOTREADY: return "Network subsystem not ready";
        case WSAVERNOTSUPPORTED: return "Winsock version not supported";
        case WSAEINPROGRESS: return "Blocking Winsock operation in progress";
        case WSAEPROCLIM: return "Process limit reached";
        default: return "Unknown WSAStartup error " + std::to_string(error_code);
    }
}

bool WindowsSocketManager::needsReinitialization() {
    if (!getInitialized().load()) return true;
    SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (s == INVALID_SOCKET) return true;
    closesocket(s);
    return false;
}

VGREResult WindowsSocketManager::forceReinitialize() {
    return attemptRecovery(5);
}

VGREResult WindowsSocketManager::validateWinsockFunctionality() {
    SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (s == INVALID_SOCKET) return VGREResult::ERR_PLATFORM_SPECIFIC;
    closesocket(s);
    return VGREResult::SUCCESS;
}

#else

VGREResult WindowsSocketManager::attemptRecovery(int max_retries) { return VGREResult::SUCCESS; }
std::string WindowsSocketManager::getWSAStartupDiagnostics(int error_code) { return "N/A"; }
bool WindowsSocketManager::needsReinitialization() { return false; }
VGREResult WindowsSocketManager::forceReinitialize() { return VGREResult::SUCCESS; }
VGREResult WindowsSocketManager::validateWinsockFunctionality() { return VGREResult::SUCCESS; }

#endif

} // namespace advanced
} // namespace vgre
