/**
 * Windows Socket Manager - Lifecycle Implementation
 * 
 * Manages WSAStartup and WSACleanup with proper reference counting.
 */

#include "vgre/advanced/tcp_cluster/internal/windows_socket_manager.h"
#include "vgre/common/logger.h"
#include <thread>
#include <chrono>

#include "vgre/common/os_backend.h"

namespace vgre {
namespace advanced {

#ifdef _WIN32

// Static members
#ifdef _WIN32
std::mutex& WindowsSocketManager::getMutex() {
    static std::mutex m;
    return m;
}
std::atomic<int>& WindowsSocketManager::getRefCount() {
    static std::atomic<int> rc{0};
    return rc;
}
std::atomic<bool>& WindowsSocketManager::getInitialized() {
    static std::atomic<bool> init{false};
    return init;
}
WSADATA& WindowsSocketManager::getWsaData() {
    static WSADATA wd{};
    return wd;
}
#endif

WindowsSocketManager::WindowsSocketManager() {
    // Do NOT call initialize() here.
    // WSAStartup is invoked explicitly by TCPClusterManager::initialize()
    // via windows_socket_manager_->initialize().  Calling it from the constructor
    // means it runs inside TCPClusterManager::instance() (static singleton init),
    // before any diagnostic output, and the nested-mutex recovery path in
    // initialize() can deadlock if WSAStartup returns an error.
}

WindowsSocketManager::~WindowsSocketManager() {
    cleanup();
}

VGREResult WindowsSocketManager::initialize() {
    std::lock_guard<std::mutex> lock(getMutex());

    int current_refs = getRefCount().fetch_add(1);

    if (current_refs == 0) {
        VGRE_LOG_DEBUG("WindowsSocketManager", "Initializing Winsock (first reference)");

        WSASetLastError(0);

        int result = WSAStartup(MAKEWORD(2, 2), &getWsaData());
        if (result != 0) {
            getRefCount().fetch_sub(1);

            VGRE_LOG_ERROR("WindowsSocketManager", 
                "WSAStartup failed with error code: " + std::to_string(result));
            VGRE_LOG_ERROR("WindowsSocketManager", getWSAStartupDiagnostics(result));

            if (result == WSASYSNOTREADY) {
                VGRE_LOG_ERROR("WindowsSocketManager", 
                    "Network subsystem not ready - checking services...");
                getRefCount() = 0;
                VGREResult recovery_result = attemptRecovery(5);  
                if (recovery_result == VGREResult::SUCCESS) return VGREResult::SUCCESS;
            } else if (result == WSAEINPROGRESS) {
                VGRE_LOG_WARN("WindowsSocketManager", "WSAStartup already in progress");
                {
                    std::mutex wsa_cv_mutex;
                    std::condition_variable wsa_cv;
                    std::unique_lock<std::mutex> lk(wsa_cv_mutex);
                    wsa_cv.wait_for(lk, std::chrono::milliseconds(500));
                }
                getRefCount() = 0;
                VGREResult recovery_result = attemptRecovery(3);
                if (recovery_result == VGREResult::SUCCESS) return VGREResult::SUCCESS;
            } else if (result == WSAVERNOTSUPPORTED) {
                VGRE_LOG_ERROR("WindowsSocketManager", "Winsock 2.2 not supported, falling back to 2.0");
                getRefCount() = 0;
                int fallback_result = WSAStartup(MAKEWORD(2, 0), &getWsaData());
                if (fallback_result == 0) {
                    getInitialized() = true;
                    getRefCount() = 1;
                    return VGREResult::SUCCESS;
                }
            }
            return VGREResult::ERR_IO;
        }

        if (LOBYTE(getWsaData().wVersion) != 2 || HIBYTE(getWsaData().wVersion) < 2) {
            VGRE_LOG_ERROR("WindowsSocketManager", "Winsock version incompatible");
            WSACleanup();
            getRefCount().fetch_sub(1);
            return VGREResult::ERR_PLATFORM_SPECIFIC;
        }

        VGREResult validation_result = validateWinsockFunctionality();
        if (validation_result != VGREResult::SUCCESS) {
            WSACleanup();
            getRefCount().fetch_sub(1);
            return validation_result;
        }

        getInitialized() = true;
        VGRE_LOG_INFO("WindowsSocketManager", "Winsock 2.2 initialized successfully");
    } else {
        if (needsReinitialization()) {
            getRefCount().fetch_sub(1);
            VGREResult recovery_result = forceReinitialize();
            if (recovery_result != VGREResult::SUCCESS) return recovery_result;
            getRefCount().fetch_add(1);
        }
    }

    return VGREResult::SUCCESS;
}

void WindowsSocketManager::cleanup() {
    std::lock_guard<std::mutex> lock(getMutex());

    int remaining_refs = getRefCount().fetch_sub(1);

    if (remaining_refs == 1) {  
        if (getInitialized()) {
            VGRE_LOG_DEBUG("WindowsSocketManager", "Cleaning up Winsock (last reference)");
            WSACleanup();
            getInitialized() = false;
        }
    } else if (remaining_refs <= 0) {
        VGRE_LOG_WARN("WindowsSocketManager", "Reference count underflow detected");
        getRefCount() = 0;  
    }
}

bool WindowsSocketManager::isInitialized() {
    return getInitialized().load();
}

int WindowsSocketManager::getReferenceCount() {
    return getRefCount().load();
}

#else

WindowsSocketManager::WindowsSocketManager() {}
WindowsSocketManager::~WindowsSocketManager() {}
VGREResult WindowsSocketManager::initialize() { return VGREResult::SUCCESS; }
void WindowsSocketManager::cleanup() {}
bool WindowsSocketManager::isInitialized() { return true; }
int WindowsSocketManager::getReferenceCount() { return 1; }

#endif

} // namespace advanced
} // namespace vgre
