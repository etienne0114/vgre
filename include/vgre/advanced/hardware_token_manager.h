#ifndef VGRE_ADVANCED_HARDWARE_TOKEN_MANAGER_H
#define VGRE_ADVANCED_HARDWARE_TOKEN_MANAGER_H

#include "vgre/common/error_codes.h"
#include <array>
#include <string>
#include <mutex>

// Forward declare TPM types if TPM support is enabled
#if defined(VGRE_HAS_TPM2)
#include <tss2/tss2_esys.h>
#include <tss2/tss2_mu.h>
#include <tss2/tss2_tctildr.h>
#endif

namespace vgre {
namespace advanced {

/**
 * @brief Hardware-backed secure token storage manager
 * 
 * Provides secure storage and retrieval of authentication tokens using
 * platform-specific secure storage mechanisms:
 * - Linux: libsecret (GNOME Keyring) or kernel keyring
 * - macOS: Keychain Services
 * - Windows: Credential Manager (CredWrite/CredRead)
 * - Fallback: TPM 2.0 NV storage (if available)
 * 
 * This replaces the insecure environment variable fallback.
 */
class HardwareTokenManager {
public:
    /**
     * @brief Token storage backend type
     */
    enum class BackendType {
        NONE = 0,
        LINUX_KEYRING,      // Linux kernel keyring
        LINUX_LIBSECRET,    // GNOME Keyring via libsecret
        MACOS_KEYCHAIN,     // macOS Keychain Services
        WINDOWS_CREDMAN,    // Windows Credential Manager
        TPM_2_0,            // TPM 2.0 NV storage
        FALLBACK_ENCRYPTED  // Encrypted file (last resort)
    };

    /**
     * @brief Get singleton instance
     */
    static HardwareTokenManager& instance();

    /**
     * @brief Initialize the token manager
     * 
     * Detects available secure storage backends and selects the best one.
     * 
     * @return SUCCESS if initialization succeeded
     */
    VGREResult initialize();

    /**
     * @brief Store authentication token securely
     * 
     * @param service Service name (e.g., "vgre_tcp_cluster")
     * @param token Token string to store
     * @return SUCCESS if token was stored successfully
     */
    VGREResult storeToken(const std::string& service, const std::string& token);

    /**
     * @brief Retrieve authentication token
     * 
     * @param service Service name
     * @param outToken Output token string
     * @return SUCCESS if token was retrieved successfully
     */
    VGREResult getToken(const std::string& service, std::string& outToken);

    /**
     * @brief Delete authentication token
     * 
     * @param service Service name
     * @return SUCCESS if token was deleted successfully
     */
    VGREResult deleteToken(const std::string& service);

    /**
     * @brief Check if a token exists
     * 
     * @param service Service name
     * @return true if token exists
     */
    bool hasToken(const std::string& service);

    /**
     * @brief Get current backend type
     */
    BackendType getBackendType() const { return backend_; }

    /**
     * @brief Get backend name as string
     */
    std::string getBackendName() const;

    /**
     * @brief Generate a new random token
     * 
     * @param length Token length in bytes (default: 32)
     * @return Random token as hex string
     */
    static std::string generateToken(size_t length = 32);

    /**
     * @brief Rotate token (generate new and store)
     * 
     * @param service Service name
     * @param outNewToken Output new token
     * @return SUCCESS if rotation succeeded
     */
    VGREResult rotateToken(const std::string& service, std::string& outNewToken);

    // Legacy compatibility (deprecated)
    VGREResult getAuthToken(std::string& outToken) {
        return getToken("vgre_tcp_cluster", outToken);
    }

private:
    HardwareTokenManager();
    ~HardwareTokenManager();

    // Disable copy/move
    HardwareTokenManager(const HardwareTokenManager&) = delete;
    HardwareTokenManager& operator=(const HardwareTokenManager&) = delete;

    // Platform-specific implementations
    VGREResult initLinuxKeyring();
    VGREResult initLinuxLibsecret();
    VGREResult initMacOSKeychain();
    VGREResult initWindowsCredMan();
    VGREResult initTPM();
    VGREResult initFallbackEncrypted();

    VGREResult storeLinuxKeyring(const std::string& service, const std::string& token);
    VGREResult storeLinuxLibsecret(const std::string& service, const std::string& token);
    VGREResult storeMacOSKeychain(const std::string& service, const std::string& token);
    VGREResult storeWindowsCredMan(const std::string& service, const std::string& token);
    VGREResult storeTPM(const std::string& service, const std::string& token);
    VGREResult storeFallbackEncrypted(const std::string& service, const std::string& token);

    VGREResult getLinuxKeyring(const std::string& service, std::string& outToken);
    VGREResult getLinuxLibsecret(const std::string& service, std::string& outToken);
    VGREResult getMacOSKeychain(const std::string& service, std::string& outToken);
    VGREResult getWindowsCredMan(const std::string& service, std::string& outToken);
    VGREResult getTPM(const std::string& service, std::string& outToken);
    VGREResult getFallbackEncrypted(const std::string& service, std::string& outToken);

    VGREResult deleteLinuxKeyring(const std::string& service);
    VGREResult deleteLinuxLibsecret(const std::string& service);
    VGREResult deleteMacOSKeychain(const std::string& service);
    VGREResult deleteWindowsCredMan(const std::string& service);
    VGREResult deleteTPM(const std::string& service);
    VGREResult deleteFallbackEncrypted(const std::string& service);

    // Authenticated encryption for fallback storage (PBKDF2 + AES-256-CTR + HMAC-SHA256)
    std::string encryptToken(const std::string& plaintext);
    std::string decryptToken(const std::string& ciphertext);
    std::array<uint8_t, 32> getMachineKey();

    BackendType backend_;
    bool initialized_;
    mutable std::mutex mutex_;
    std::string fallback_path_;
    
#if defined(VGRE_HAS_TPM2)
    ESYS_CONTEXT* tpm_context_;
    TPM2_HANDLE tpm_nv_handle_;
#endif
};

} // namespace advanced
} // namespace vgre

#endif // VGRE_ADVANCED_HARDWARE_TOKEN_MANAGER_H
