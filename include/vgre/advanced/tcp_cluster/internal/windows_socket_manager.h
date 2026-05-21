/**
 * Windows Socket Manager Header
 * 
 * Provides proper WSAStartup/WSACleanup management for cross-platform
 * TCP cluster socket operations.
 */

#pragma once

#include "vgre/common/error_codes.h"
#include "vgre/common/sockets.h"  // Include for vgre_socket_t
#include <atomic>
#include <mutex>
#include <string>

#include "vgre/common/os_backend.h"

#ifdef _WIN32
#include <mstcpip.h>
#endif

namespace vgre {
namespace advanced {

/**
 * @brief Manages Windows socket initialization and cleanup
 * 
 * This class ensures that WSAStartup/WSACleanup are called exactly once
 * per process, with proper reference counting for multiple TCP cluster
 * instances. This fixes the ACCESS_VIOLATION crashes that occur when
 * Windows workers connect to Linux masters.
 * 
 * Key features:
 * - Thread-safe initialization and cleanup
 * - Reference counting for multiple instances
 * - Proper error handling and logging
 * - Cross-platform compatibility (no-op on non-Windows)
 */
class WindowsSocketManager {
public:
    WindowsSocketManager();
    ~WindowsSocketManager();
    
    // Non-copyable, non-movable
    WindowsSocketManager(const WindowsSocketManager&) = delete;
    WindowsSocketManager& operator=(const WindowsSocketManager&) = delete;
    WindowsSocketManager(WindowsSocketManager&&) = delete;
    WindowsSocketManager& operator=(WindowsSocketManager&&) = delete;
    
    /**
     * @brief Initialize Winsock (Windows only)
     * @return SUCCESS if initialization succeeded, ERR_IO if failed
     */
    VGREResult initialize();
    
    /**
     * @brief Cleanup Winsock resources (Windows only)
     */
    void cleanup();
    
    /**
     * @brief Check if Winsock is initialized
     * @return true if initialized, false otherwise
     */
    static bool isInitialized();
    
    /**
     * @brief Get current reference count
     * @return Number of active WindowsSocketManager instances
     */
    static int getReferenceCount();
    
    /**
     * @brief Get the last socket error as a VGREResult
     * @return Mapped error code
     */
    static VGREResult getLastSocketError();
    
    /**
     * @brief Get human-readable socket error string
     * @param error_code Platform-specific error code
     * @return Error description string
     */
    static std::string getSocketErrorString(int error_code);
    
    /**
     * @brief Configure socket with Windows-specific options
     * @param fd Socket file descriptor
     * @return SUCCESS if configuration succeeded, error code otherwise
     */
    static VGREResult configureSocket(vgre::common::vgre_socket_t fd);
    
    /**
     * @brief Attempt WSAStartup recovery after failure
     * @param max_retries Maximum number of retry attempts
     * @return SUCCESS if recovery succeeded, error code otherwise
     */
    VGREResult attemptRecovery(int max_retries = 3);
    
    /**
     * @brief Get detailed WSAStartup failure diagnostics
     * @param error_code WSAStartup error code
     * @return Detailed diagnostic information
     */
    static std::string getWSAStartupDiagnostics(int error_code);
    
    /**
     * @brief Check if Winsock needs reinitialization
     * @return true if reinitialization is needed
     */
    static bool needsReinitialization();
    
    /**
     * @brief Force Winsock reinitialization (for recovery scenarios)
     * @return SUCCESS if reinitialization succeeded
     */
    VGREResult forceReinitialize();
    
    /**
     * @brief Validate Winsock functionality to prevent ACCESS_VIOLATION
     * @return SUCCESS if validation passed, error code otherwise
     */
    VGREResult validateWinsockFunctionality();

private:
#ifdef _WIN32
    static std::mutex& getMutex();
    static std::atomic<int>& getRefCount();
    static std::atomic<bool>& getInitialized();
    static WSADATA& getWsaData();
#endif
};

} // namespace advanced
} // namespace vgre