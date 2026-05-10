/**
 * Windows Socket Manager - Lifecycle Implementation
 * 
 * Manages WSAStartup and WSACleanup with proper reference counting.
 */

#include "vgre/advanced/tcp_cluster/internal/windows_socket_manager.h"
#include "vgre/common/logger.h"
#include <thread>
#include <chrono>

#ifdef _WIN32
#include <winsock2.h>
#endif

namespace vgre {
namespace advanced {

#ifdef _WIN32

// Static members
std::mutex WindowsSocketManager::mutex_;
std::atomic<int> WindowsSocketManager::ref_count_{0};
std::atomic<bool> WindowsSocketManager::initialized_{false};
WSADATA WindowsSocketManager::wsa_data_{};

WindowsSocketManager::WindowsSocketManager() {
    initialize();
}

WindowsSocketManager::~WindowsSocketManager() {
    cleanup();
}

VGREResult WindowsSocketManager::initialize() {
    std::lock_guard<std::mutex> lock(mutex_);

    int current_refs = ref_count_.fetch_add(1);

    if (current_refs == 0) {
        VGRE_LOG_DEBUG("WindowsSocketManager", "Initializing Winsock (first reference)");

        WSASetLastError(0);

        int result = WSAStartup(MAKEWORD(2, 2), &wsa_data_);
        if (result != 0) {
            ref_count_.fetch_sub(1);

            VGRE_LOG_ERROR("WindowsSocketManager", 
                "WSAStartup failed with error code: " + std::to_string(result));
            VGRE_LOG_ERROR("WindowsSocketManager", getWSAStartupDiagnostics(result));

            if (result == WSASYSNOTREADY) {
                VGRE_LOG_ERROR("WindowsSocketManager", 
                    "Network subsystem not ready - checking services...");
                ref_count_ = 0;
                VGREResult recovery_result = attemptRecovery(5);  
                if (recovery_result == VGREResult::SUCCESS) return VGREResult::SUCCESS;
            } else if (result == WSAEINPROGRESS) {
                VGRE_LOG_WARN("WindowsSocketManager", "WSAStartup already in progress");
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                ref_count_ = 0;
                VGREResult recovery_result = attemptRecovery(3);
                if (recovery_result == VGREResult::SUCCESS) return VGREResult::SUCCESS;
            } else if (result == WSAVERNOTSUPPORTED) {
                VGRE_LOG_ERROR("WindowsSocketManager", "Winsock 2.2 not supported, falling back to 2.0");
                ref_count_ = 0;
                int fallback_result = WSAStartup(MAKEWORD(2, 0), &wsa_data_);
                if (fallback_result == 0) {
                    initialized_ = true;
                    ref_count_ = 1;
                    return VGREResult::SUCCESS;
                }
            }
            return VGREResult::ERR_IO;
        }

        if (LOBYTE(wsa_data_.wVersion) != 2 || HIBYTE(wsa_data_.wVersion) < 2) {
            VGRE_LOG_ERROR("WindowsSocketManager", "Winsock version incompatible");
            WSACleanup();
            ref_count_.fetch_sub(1);
            return VGREResult::ERR_PLATFORM_SPECIFIC;
        }

        VGREResult validation_result = validateWinsockFunctionality();
        if (validation_result != VGREResult::SUCCESS) {
            WSACleanup();
            ref_count_.fetch_sub(1);
            return validation_result;
        }

        initialized_ = true;
        VGRE_LOG_INFO("WindowsSocketManager", "Winsock 2.2 initialized successfully");
    } else {
        if (needsReinitialization()) {
            ref_count_.fetch_sub(1);
            VGREResult recovery_result = forceReinitialize();
            if (recovery_result != VGREResult::SUCCESS) return recovery_result;
            ref_count_.fetch_add(1);
        }
    }

    return VGREResult::SUCCESS;
}

void WindowsSocketManager::cleanup() {
    std::lock_guard<std::mutex> lock(mutex_);

    int remaining_refs = ref_count_.fetch_sub(1);

    if (remaining_refs == 1) {  
        if (initialized_) {
            VGRE_LOG_DEBUG("WindowsSocketManager", "Cleaning up Winsock (last reference)");
            WSACleanup();
            initialized_ = false;
        }
    } else if (remaining_refs <= 0) {
        VGRE_LOG_WARN("WindowsSocketManager", "Reference count underflow detected");
        ref_count_ = 0;  
    }
}

bool WindowsSocketManager::isInitialized() {
    return initialized_.load();
}

int WindowsSocketManager::getReferenceCount() {
    return ref_count_.load();
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
