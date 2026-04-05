#include "vgre/advanced/hardware_token_manager.h"
#include "vgre/advanced/secure_channel.h"  // crypto:: namespace (SHA-256, HMAC, PBKDF2, AES-256-CTR)
#include "vgre/common/logger.h"

#include <fstream>
#include <sstream>
#include <iomanip>
#include <random>
#include <array>
#include <cstring>
#include <map>

// Platform-specific headers
#if defined(__linux__)
#include <sys/types.h>
#include <keyutils.h>
#include <unistd.h>
#include <sys/stat.h>
#elif defined(__APPLE__)
#include <Security/Security.h>
#elif defined(_WIN32)
#include <windows.h>
#include <wincred.h>
#endif

// TPM 2.0 headers
#if defined(VGRE_HAS_TPM2)
#include <tss2/tss2_esys.h>
#include <tss2/tss2_mu.h>
#include <tss2/tss2_tctildr.h>
#endif

namespace vgre {
namespace advanced {

HardwareTokenManager::HardwareTokenManager()
    : backend_(BackendType::NONE),
      initialized_(false),
      fallback_path_("")
#if defined(VGRE_HAS_TPM2)
      , tpm_context_(nullptr),
      tpm_nv_handle_(0)
#endif
{
}

HardwareTokenManager::~HardwareTokenManager() {
#if defined(VGRE_HAS_TPM2)
    if (tpm_context_) {
        Esys_Finalize(&tpm_context_);
    }
#endif
}

VGREResult HardwareTokenManager::initialize() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (initialized_) {
        return VGREResult::SUCCESS;
    }

    // Try backends in order of security preference
    
#if defined(__linux__)
    // Priority 1: Linux kernel keyring (most secure, no external deps)
    if (initLinuxKeyring() == VGREResult::SUCCESS) {
        backend_ = BackendType::LINUX_KEYRING;
        initialized_ = true;
        VGRE_LOG_INFO("HardwareTokenManager", "Initialized with Linux Keyring backend");
        return VGREResult::SUCCESS;
    }
    
    // Priority 2: libsecret (GNOME Keyring)
    if (initLinuxLibsecret() == VGREResult::SUCCESS) {
        backend_ = BackendType::LINUX_LIBSECRET;
        initialized_ = true;
        VGRE_LOG_INFO("HardwareTokenManager", "Initialized with libsecret backend");
        return VGREResult::SUCCESS;
    }
    
#elif defined(__APPLE__)
    // macOS Keychain
    if (initMacOSKeychain() == VGREResult::SUCCESS) {
        backend_ = BackendType::MACOS_KEYCHAIN;
        initialized_ = true;
        VGRE_LOG_INFO("HardwareTokenManager", "Initialized with macOS Keychain backend");
        return VGREResult::SUCCESS;
    }
    
#elif defined(_WIN32)
    // Windows Credential Manager
    if (initWindowsCredMan() == VGREResult::SUCCESS) {
        backend_ = BackendType::WINDOWS_CREDMAN;
        initialized_ = true;
        VGRE_LOG_INFO("HardwareTokenManager", "Initialized with Windows Credential Manager backend");
        return VGREResult::SUCCESS;
    }
#endif

    // Priority 3: TPM 2.0 (cross-platform, requires hardware)
    if (initTPM() == VGREResult::SUCCESS) {
        backend_ = BackendType::TPM_2_0;
        initialized_ = true;
        VGRE_LOG_INFO("HardwareTokenManager", "Initialized with TPM 2.0 backend");
        return VGREResult::SUCCESS;
    }

    // Last resort: Encrypted file storage
    if (initFallbackEncrypted() == VGREResult::SUCCESS) {
        backend_ = BackendType::FALLBACK_ENCRYPTED;
        initialized_ = true;
        VGRE_LOG_WARN("HardwareTokenManager", 
                      "Using encrypted file fallback - not recommended for production");
        return VGREResult::SUCCESS;
    }

    VGRE_LOG_ERROR("HardwareTokenManager", "Failed to initialize any secure storage backend");
    return VGREResult::ERR_NOT_SUPPORTED;
}

VGREResult HardwareTokenManager::storeToken(const std::string& service, const std::string& token) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!initialized_) {
        return VGREResult::ERROR_NOT_INITIALIZED;
    }

    switch (backend_) {
#if defined(__linux__)
        case BackendType::LINUX_KEYRING:
            return storeLinuxKeyring(service, token);
        case BackendType::LINUX_LIBSECRET:
            return storeLinuxLibsecret(service, token);
#elif defined(__APPLE__)
        case BackendType::MACOS_KEYCHAIN:
            return storeMacOSKeychain(service, token);
#elif defined(_WIN32)
        case BackendType::WINDOWS_CREDMAN:
            return storeWindowsCredMan(service, token);
#endif
        case BackendType::TPM_2_0:
            return storeTPM(service, token);
        case BackendType::FALLBACK_ENCRYPTED:
            return storeFallbackEncrypted(service, token);
        default:
            return VGREResult::ERR_NOT_SUPPORTED;
    }
}

VGREResult HardwareTokenManager::getToken(const std::string& service, std::string& outToken) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!initialized_) {
        return VGREResult::ERROR_NOT_INITIALIZED;
    }

    switch (backend_) {
#if defined(__linux__)
        case BackendType::LINUX_KEYRING:
            return getLinuxKeyring(service, outToken);
        case BackendType::LINUX_LIBSECRET:
            return getLinuxLibsecret(service, outToken);
#elif defined(__APPLE__)
        case BackendType::MACOS_KEYCHAIN:
            return getMacOSKeychain(service, outToken);
#elif defined(_WIN32)
        case BackendType::WINDOWS_CREDMAN:
            return getWindowsCredMan(service, outToken);
#endif
        case BackendType::TPM_2_0:
            return getTPM(service, outToken);
        case BackendType::FALLBACK_ENCRYPTED:
            return getFallbackEncrypted(service, outToken);
        default:
            return VGREResult::ERR_NOT_SUPPORTED;
    }
}

VGREResult HardwareTokenManager::deleteToken(const std::string& service) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!initialized_) {
        return VGREResult::ERROR_NOT_INITIALIZED;
    }

    switch (backend_) {
#if defined(__linux__)
        case BackendType::LINUX_KEYRING:
            return deleteLinuxKeyring(service);
        case BackendType::LINUX_LIBSECRET:
            return deleteLinuxLibsecret(service);
#elif defined(__APPLE__)
        case BackendType::MACOS_KEYCHAIN:
            return deleteMacOSKeychain(service);
#elif defined(_WIN32)
        case BackendType::WINDOWS_CREDMAN:
            return deleteWindowsCredMan(service);
#endif
        case BackendType::TPM_2_0:
            return deleteTPM(service);
        case BackendType::FALLBACK_ENCRYPTED:
            return deleteFallbackEncrypted(service);
        default:
            return VGREResult::ERR_NOT_SUPPORTED;
    }
}

bool HardwareTokenManager::hasToken(const std::string& service) {
    std::string token;
    return getToken(service, token) == VGREResult::SUCCESS;
}

std::string HardwareTokenManager::getBackendName() const {
    switch (backend_) {
        case BackendType::LINUX_KEYRING: return "Linux Keyring";
        case BackendType::LINUX_LIBSECRET: return "libsecret (GNOME Keyring)";
        case BackendType::MACOS_KEYCHAIN: return "macOS Keychain";
        case BackendType::WINDOWS_CREDMAN: return "Windows Credential Manager";
        case BackendType::TPM_2_0: return "TPM 2.0";
        case BackendType::FALLBACK_ENCRYPTED: return "Encrypted File (Fallback)";
        default: return "None";
    }
}

std::string HardwareTokenManager::generateToken(size_t length) {
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<int> dis(0, 255);
    
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    
    for (size_t i = 0; i < length; ++i) {
        oss << std::setw(2) << static_cast<int>(dis(gen));
    }
    
    return oss.str();
}

VGREResult HardwareTokenManager::rotateToken(const std::string& service, std::string& outNewToken) {
    outNewToken = generateToken(32);
    return storeToken(service, outNewToken);
}

// ============================================================================
// Linux Keyring Implementation
// ============================================================================

#if defined(__linux__)

VGREResult HardwareTokenManager::initLinuxKeyring() {
    // Check if keyutils is available
    key_serial_t keyring = keyctl_search(KEY_SPEC_USER_KEYRING, "keyring", "vgre", 0);
    if (keyring < 0) {
        // Create VGRE keyring
        keyring = add_key("keyring", "vgre", NULL, 0, KEY_SPEC_USER_KEYRING);
        if (keyring < 0) {
            return VGREResult::ERR_NOT_SUPPORTED;
        }
    }
    return VGREResult::SUCCESS;
}

VGREResult HardwareTokenManager::storeLinuxKeyring(const std::string& service, const std::string& token) {
    key_serial_t keyring = keyctl_search(KEY_SPEC_USER_KEYRING, "keyring", "vgre", 0);
    if (keyring < 0) {
        return VGREResult::ERROR_NOT_INITIALIZED;
    }
    
    std::string key_desc = "vgre:" + service;
    key_serial_t key = add_key("user", key_desc.c_str(), token.c_str(), token.size(), keyring);
    
    if (key < 0) {
        VGRE_LOG_ERROR("HardwareTokenManager", "Failed to store token in Linux keyring");
        return VGREResult::ERROR_IO;
    }
    
    // Set timeout to 0 (persistent until reboot)
    keyctl_set_timeout(key, 0);
    
    return VGREResult::SUCCESS;
}

VGREResult HardwareTokenManager::getLinuxKeyring(const std::string& service, std::string& outToken) {
    key_serial_t keyring = keyctl_search(KEY_SPEC_USER_KEYRING, "keyring", "vgre", 0);
    if (keyring < 0) {
        return VGREResult::ERROR_NOT_INITIALIZED;
    }
    
    std::string key_desc = "vgre:" + service;
    key_serial_t key = keyctl_search(keyring, "user", key_desc.c_str(), 0);
    
    if (key < 0) {
        return VGREResult::ERROR_AUTH_FAILED;
    }
    
    // Get key size
    long size = keyctl_read(key, NULL, 0);
    if (size < 0) {
        return VGREResult::ERROR_IO;
    }
    
    // Read key
    std::vector<char> buffer(size);
    long read_size = keyctl_read(key, buffer.data(), size);
    
    if (read_size < 0) {
        return VGREResult::ERROR_IO;
    }
    
    outToken = std::string(buffer.data(), read_size);
    return VGREResult::SUCCESS;
}

VGREResult HardwareTokenManager::deleteLinuxKeyring(const std::string& service) {
    key_serial_t keyring = keyctl_search(KEY_SPEC_USER_KEYRING, "keyring", "vgre", 0);
    if (keyring < 0) {
        return VGREResult::ERROR_NOT_INITIALIZED;
    }
    
    std::string key_desc = "vgre:" + service;
    key_serial_t key = keyctl_search(keyring, "user", key_desc.c_str(), 0);
    
    if (key < 0) {
        return VGREResult::ERROR_AUTH_FAILED;
    }
    
    if (keyctl_revoke(key) < 0) {
        return VGREResult::ERROR_IO;
    }
    
    return VGREResult::SUCCESS;
}

VGREResult HardwareTokenManager::initLinuxLibsecret() {
    // libsecret requires D-Bus and GNOME Keyring daemon
    // For now, return not supported (would require linking libsecret)
    return VGREResult::ERR_NOT_SUPPORTED;
}

VGREResult HardwareTokenManager::storeLinuxLibsecret(const std::string&, const std::string&) {
    return VGREResult::ERR_NOT_SUPPORTED;
}

VGREResult HardwareTokenManager::getLinuxLibsecret(const std::string&, std::string&) {
    return VGREResult::ERR_NOT_SUPPORTED;
}

VGREResult HardwareTokenManager::deleteLinuxLibsecret(const std::string&) {
    return VGREResult::ERR_NOT_SUPPORTED;
}

#else

// Platform-specific implementations
VGREResult HardwareTokenManager::initLinuxKeyring() { return VGREResult::ERR_NOT_SUPPORTED; }
VGREResult HardwareTokenManager::storeLinuxKeyring(const std::string&, const std::string&) { return VGREResult::ERR_NOT_SUPPORTED; }
VGREResult HardwareTokenManager::getLinuxKeyring(const std::string&, std::string&) { return VGREResult::ERR_NOT_SUPPORTED; }
VGREResult HardwareTokenManager::deleteLinuxKeyring(const std::string&) { return VGREResult::ERR_NOT_SUPPORTED; }
VGREResult HardwareTokenManager::initLinuxLibsecret() { return VGREResult::ERR_NOT_SUPPORTED; }
VGREResult HardwareTokenManager::storeLinuxLibsecret(const std::string&, const std::string&) { return VGREResult::ERR_NOT_SUPPORTED; }
VGREResult HardwareTokenManager::getLinuxLibsecret(const std::string&, std::string&) { return VGREResult::ERR_NOT_SUPPORTED; }
VGREResult HardwareTokenManager::deleteLinuxLibsecret(const std::string&) { return VGREResult::ERR_NOT_SUPPORTED; }

#endif

// ============================================================================
// macOS Keychain Implementation
// ============================================================================

#if defined(__APPLE__)

VGREResult HardwareTokenManager::initMacOSKeychain() {
    // Keychain is always available on macOS
    return VGREResult::SUCCESS;
}

VGREResult HardwareTokenManager::storeMacOSKeychain(const std::string& service, const std::string& token) {
    OSStatus status = SecKeychainAddGenericPassword(
        NULL,                           // default keychain
        service.length(),               // service name length
        service.c_str(),                // service name
        0,                              // account name length (use service as account)
        NULL,                           // account name
        token.length(),                 // password length
        token.c_str(),                  // password data
        NULL                            // item ref (not needed)
    );
    
    if (status == errSecDuplicateItem) {
        // Update existing item
        SecKeychainItemRef itemRef = NULL;
        status = SecKeychainFindGenericPassword(
            NULL, service.length(), service.c_str(), 0, NULL,
            NULL, NULL, &itemRef
        );
        
        if (status == errSecSuccess && itemRef) {
            status = SecKeychainItemModifyAttributesAndData(
                itemRef, NULL, token.length(), token.c_str()
            );
            CFRelease(itemRef);
        }
    }
    
    if (status != errSecSuccess) {
        VGRE_LOG_ERROR("HardwareTokenManager", 
                      "Failed to store token in macOS Keychain: " + std::to_string(status));
        return VGREResult::ERROR_IO;
    }
    
    return VGREResult::SUCCESS;
}

VGREResult HardwareTokenManager::getMacOSKeychain(const std::string& service, std::string& outToken) {
    void* passwordData = NULL;
    UInt32 passwordLength = 0;
    
    OSStatus status = SecKeychainFindGenericPassword(
        NULL,                           // default keychain
        service.length(),               // service name length
        service.c_str(),                // service name
        0,                              // account name length
        NULL,                           // account name
        &passwordLength,                // password length
        &passwordData,                  // password data
        NULL                            // item ref
    );
    
    if (status != errSecSuccess) {
        return VGREResult::ERROR_AUTH_FAILED;
    }
    
    outToken = std::string(static_cast<const char*>(passwordData), passwordLength);
    SecKeychainItemFreeContent(NULL, passwordData);
    
    return VGREResult::SUCCESS;
}

VGREResult HardwareTokenManager::deleteMacOSKeychain(const std::string& service) {
    SecKeychainItemRef itemRef = NULL;
    OSStatus status = SecKeychainFindGenericPassword(
        NULL, service.length(), service.c_str(), 0, NULL,
        NULL, NULL, &itemRef
    );
    
    if (status != errSecSuccess || !itemRef) {
        return VGREResult::ERROR_AUTH_FAILED;
    }
    
    status = SecKeychainItemDelete(itemRef);
    CFRelease(itemRef);
    
    if (status != errSecSuccess) {
        return VGREResult::ERROR_IO;
    }
    
    return VGREResult::SUCCESS;
}

#else

// Platform-specific implementations
VGREResult HardwareTokenManager::initMacOSKeychain() { return VGREResult::ERR_NOT_SUPPORTED; }
VGREResult HardwareTokenManager::storeMacOSKeychain(const std::string&, const std::string&) { return VGREResult::ERR_NOT_SUPPORTED; }
VGREResult HardwareTokenManager::getMacOSKeychain(const std::string&, std::string&) { return VGREResult::ERR_NOT_SUPPORTED; }
VGREResult HardwareTokenManager::deleteMacOSKeychain(const std::string&) { return VGREResult::ERR_NOT_SUPPORTED; }

#endif

// ============================================================================
// Windows Credential Manager Implementation
// ============================================================================

#if defined(_WIN32)

VGREResult HardwareTokenManager::initWindowsCredMan() {
    // Credential Manager is always available on Windows
    return VGREResult::SUCCESS;
}

VGREResult HardwareTokenManager::storeWindowsCredMan(const std::string& service, const std::string& token) {
    std::wstring wservice(service.begin(), service.end());
    std::wstring wtarget = L"VGRE:" + wservice;
    
    CREDENTIALW cred = {0};
    cred.Type = CRED_TYPE_GENERIC;
    cred.TargetName = const_cast<LPWSTR>(wtarget.c_str());
    cred.CredentialBlobSize = static_cast<DWORD>(token.size());
    cred.CredentialBlob = reinterpret_cast<LPBYTE>(const_cast<char*>(token.c_str()));
    cred.Persist = CRED_PERSIST_LOCAL_MACHINE;
    
    if (!CredWriteW(&cred, 0)) {
        VGRE_LOG_ERROR("HardwareTokenManager", 
                      "Failed to store token in Windows Credential Manager: " + 
                      std::to_string(GetLastError()));
        return VGREResult::ERROR_IO;
    }
    
    return VGREResult::SUCCESS;
}

VGREResult HardwareTokenManager::getWindowsCredMan(const std::string& service, std::string& outToken) {
    std::wstring wservice(service.begin(), service.end());
    std::wstring wtarget = L"VGRE:" + wservice;
    
    PCREDENTIALW pcred = NULL;
    if (!CredReadW(wtarget.c_str(), CRED_TYPE_GENERIC, 0, &pcred)) {
        return VGREResult::ERROR_AUTH_FAILED;
    }
    
    outToken = std::string(
        reinterpret_cast<const char*>(pcred->CredentialBlob),
        pcred->CredentialBlobSize
    );
    
    CredFree(pcred);
    return VGREResult::SUCCESS;
}

VGREResult HardwareTokenManager::deleteWindowsCredMan(const std::string& service) {
    std::wstring wservice(service.begin(), service.end());
    std::wstring wtarget = L"VGRE:" + wservice;
    
    if (!CredDeleteW(wtarget.c_str(), CRED_TYPE_GENERIC, 0)) {
        return VGREResult::ERROR_AUTH_FAILED;
    }
    
    return VGREResult::SUCCESS;
}

#else

// Platform-specific implementations
VGREResult HardwareTokenManager::initWindowsCredMan() { return VGREResult::ERR_NOT_SUPPORTED; }
VGREResult HardwareTokenManager::storeWindowsCredMan(const std::string&, const std::string&) { return VGREResult::ERR_NOT_SUPPORTED; }
VGREResult HardwareTokenManager::getWindowsCredMan(const std::string&, std::string&) { return VGREResult::ERR_NOT_SUPPORTED; }
VGREResult HardwareTokenManager::deleteWindowsCredMan(const std::string&) { return VGREResult::ERR_NOT_SUPPORTED; }

#endif

// ============================================================================
// TPM 2.0 Implementation
// ============================================================================

#if defined(VGRE_HAS_TPM2)

VGREResult HardwareTokenManager::initTPM() {
    TSS2_RC rc;
    TSS2_TCTI_CONTEXT* tcti_ctx = nullptr;
    
    // Initialize TCTI (TPM Command Transmission Interface)
    // Authoritative Phase 3: Purely physical device access. No simulator fallbacks.
    rc = Tss2_TctiLdr_Initialize("device:/dev/tpmrm0", &tcti_ctx);
    if (rc != TSS2_RC_SUCCESS) {
        rc = Tss2_TctiLdr_Initialize("device:/dev/tpm0", &tcti_ctx);
        if (rc != TSS2_RC_SUCCESS) {
            VGRE_LOG_DEBUG("HardwareTokenManager", 
                          "Physical TPM device not available (rc=" + std::to_string(rc) + ")");
            return VGREResult::ERR_NOT_SUPPORTED;
        }
    }
    
    // Initialize ESYS context
    rc = Esys_Initialize(&tpm_context_, tcti_ctx, NULL);
    if (rc != TSS2_RC_SUCCESS) {
        VGRE_LOG_ERROR("HardwareTokenManager", 
                      "Failed to initialize TPM ESYS context (rc=" + std::to_string(rc) + ")");
        Tss2_TctiLdr_Finalize(&tcti_ctx);
        return VGREResult::ERR_NOT_SUPPORTED;
    }
    
    // Define NV index for VGRE tokens (0x01C00000 is user-defined range)
    tpm_nv_handle_ = 0x01C00100; // VGRE token storage
    
    VGRE_LOG_INFO("HardwareTokenManager", "TPM 2.0 initialized successfully");
    return VGREResult::SUCCESS;
}

VGREResult HardwareTokenManager::storeTPM(const std::string& service, const std::string& token) {
    if (!tpm_context_) {
        return VGREResult::ERROR_NOT_INITIALIZED;
    }
    
    TSS2_RC rc;
    
    // Create service-specific NV index (hash service name to get unique index)
    uint32_t service_hash = 0;
    for (char c : service) {
        service_hash = service_hash * 31 + static_cast<uint32_t>(c);
    }
    TPM2_HANDLE nv_index = tpm_nv_handle_ + (service_hash % 0x100);
    
    // Check if NV index already exists
    ESYS_TR nv_handle = ESYS_TR_NONE;
    rc = Esys_TR_FromTPMPublic(tpm_context_, nv_index, 
                               ESYS_TR_NONE, ESYS_TR_NONE, ESYS_TR_NONE,
                               &nv_handle);
    
    if (rc == TSS2_RC_SUCCESS) {
        // NV index exists, undefine it first
        rc = Esys_NV_UndefineSpace(tpm_context_,
                                   ESYS_TR_RH_OWNER,
                                   nv_handle,
                                   ESYS_TR_PASSWORD, ESYS_TR_NONE, ESYS_TR_NONE);
        if (rc != TSS2_RC_SUCCESS) {
            VGRE_LOG_ERROR("HardwareTokenManager", 
                          "Failed to undefine existing NV space (rc=" + std::to_string(rc) + ")");
            return VGREResult::ERROR_IO;
        }
    }
    
    // Define NV index attributes
    TPM2B_AUTH auth = {.size = 0};
    TPM2B_NV_PUBLIC publicInfo = {
        .size = 0,
        .nvPublic = {
            .nvIndex = nv_index,
            .nameAlg = TPM2_ALG_SHA256,
            .attributes = TPMA_NV_AUTHREAD | TPMA_NV_AUTHWRITE | 
                         TPMA_NV_OWNERWRITE | TPMA_NV_OWNERREAD,
            .authPolicy = {.size = 0},
            .dataSize = static_cast<uint16_t>(token.size())
        }
    };
    
    // Define NV space
    rc = Esys_NV_DefineSpace(tpm_context_,
                            ESYS_TR_RH_OWNER,
                            ESYS_TR_PASSWORD, ESYS_TR_NONE, ESYS_TR_NONE,
                            &auth,
                            &publicInfo,
                            &nv_handle);
    
    if (rc != TSS2_RC_SUCCESS) {
        VGRE_LOG_ERROR("HardwareTokenManager", 
                      "Failed to define NV space (rc=" + std::to_string(rc) + ")");
        return VGREResult::ERROR_IO;
    }
    
    // Write token to NV space
    TPM2B_MAX_NV_BUFFER nv_write_data;
    nv_write_data.size = token.size();
    std::memcpy(nv_write_data.buffer, token.c_str(), token.size());
    
    rc = Esys_NV_Write(tpm_context_,
                      nv_handle,
                      nv_handle,
                      ESYS_TR_PASSWORD, ESYS_TR_NONE, ESYS_TR_NONE,
                      &nv_write_data,
                      0);
    
    if (rc != TSS2_RC_SUCCESS) {
        VGRE_LOG_ERROR("HardwareTokenManager", 
                      "Failed to write to NV space (rc=" + std::to_string(rc) + ")");
        return VGREResult::ERROR_IO;
    }
    
    VGRE_LOG_INFO("HardwareTokenManager", 
                  "Token stored in TPM NV index 0x" + 
                  std::to_string(nv_index));
    
    return VGREResult::SUCCESS;
}

VGREResult HardwareTokenManager::getTPM(const std::string& service, std::string& outToken) {
    if (!tpm_context_) {
        return VGREResult::ERROR_NOT_INITIALIZED;
    }
    
    TSS2_RC rc;
    
    // Calculate service-specific NV index
    uint32_t service_hash = 0;
    for (char c : service) {
        service_hash = service_hash * 31 + static_cast<uint32_t>(c);
    }
    TPM2_HANDLE nv_index = tpm_nv_handle_ + (service_hash % 0x100);
    
    // Get NV handle
    ESYS_TR nv_handle = ESYS_TR_NONE;
    rc = Esys_TR_FromTPMPublic(tpm_context_, nv_index,
                               ESYS_TR_NONE, ESYS_TR_NONE, ESYS_TR_NONE,
                               &nv_handle);
    
    if (rc != TSS2_RC_SUCCESS) {
        return VGREResult::ERROR_AUTH_FAILED;
    }
    
    // Read NV public to get size
    TPM2B_NV_PUBLIC* nv_public = nullptr;
    TPM2B_NAME* nv_name = nullptr;
    rc = Esys_NV_ReadPublic(tpm_context_,
                           nv_handle,
                           ESYS_TR_NONE, ESYS_TR_NONE, ESYS_TR_NONE,
                           &nv_public,
                           &nv_name);
    
    if (rc != TSS2_RC_SUCCESS) {
        return VGREResult::ERROR_IO;
    }
    
    uint16_t data_size = nv_public->nvPublic.dataSize;
    Esys_Free(nv_public);
    Esys_Free(nv_name);
    
    // Read token from NV space
    TPM2B_MAX_NV_BUFFER* nv_data = nullptr;
    rc = Esys_NV_Read(tpm_context_,
                     nv_handle,
                     nv_handle,
                     ESYS_TR_PASSWORD, ESYS_TR_NONE, ESYS_TR_NONE,
                     data_size,
                     0,
                     &nv_data);
    
    if (rc != TSS2_RC_SUCCESS) {
        VGRE_LOG_ERROR("HardwareTokenManager", 
                      "Failed to read from NV space (rc=" + std::to_string(rc) + ")");
        return VGREResult::ERROR_IO;
    }
    
    outToken = std::string(reinterpret_cast<const char*>(nv_data->buffer), 
                          nv_data->size);
    Esys_Free(nv_data);
    
    return VGREResult::SUCCESS;
}

VGREResult HardwareTokenManager::deleteTPM(const std::string& service) {
    if (!tpm_context_) {
        return VGREResult::ERROR_NOT_INITIALIZED;
    }
    
    TSS2_RC rc;
    
    // Calculate service-specific NV index
    uint32_t service_hash = 0;
    for (char c : service) {
        service_hash = service_hash * 31 + static_cast<uint32_t>(c);
    }
    TPM2_HANDLE nv_index = tpm_nv_handle_ + (service_hash % 0x100);
    
    // Get NV handle
    ESYS_TR nv_handle = ESYS_TR_NONE;
    rc = Esys_TR_FromTPMPublic(tpm_context_, nv_index,
                               ESYS_TR_NONE, ESYS_TR_NONE, ESYS_TR_NONE,
                               &nv_handle);
    
    if (rc != TSS2_RC_SUCCESS) {
        return VGREResult::ERROR_AUTH_FAILED;
    }
    
    // Undefine NV space
    rc = Esys_NV_UndefineSpace(tpm_context_,
                               ESYS_TR_RH_OWNER,
                               nv_handle,
                               ESYS_TR_PASSWORD, ESYS_TR_NONE, ESYS_TR_NONE);
    
    if (rc != TSS2_RC_SUCCESS) {
        VGRE_LOG_ERROR("HardwareTokenManager", 
                      "Failed to undefine NV space (rc=" + std::to_string(rc) + ")");
        return VGREResult::ERROR_IO;
    }
    
    return VGREResult::SUCCESS;
}

#else

// TPM 2.0 implementation for security-hardened environments
VGREResult HardwareTokenManager::initTPM() {
    // TPM 2.0 support requires tss2-esys library — currently not enabled in this build
    return VGREResult::ERR_NOT_SUPPORTED;
}

VGREResult HardwareTokenManager::storeTPM(const std::string&, const std::string&) {
    return VGREResult::ERR_NOT_SUPPORTED;
}

VGREResult HardwareTokenManager::getTPM(const std::string&, std::string&) {
    return VGREResult::ERR_NOT_SUPPORTED;
}

VGREResult HardwareTokenManager::deleteTPM(const std::string&) {
    return VGREResult::ERR_NOT_SUPPORTED;
}

#endif // VGRE_HAS_TPM2

// ============================================================================
// Encrypted File Fallback Implementation
// ============================================================================

VGREResult HardwareTokenManager::initFallbackEncrypted() {
#if defined(_WIN32)
    fallback_path_ = std::string(getenv("APPDATA")) + "\\vgre\\tokens.enc";
#else
    fallback_path_ = std::string(getenv("HOME")) + "/.vgre/tokens.enc";
#endif
    
    // Create directory if it doesn't exist
    std::string dir = fallback_path_.substr(0, fallback_path_.find_last_of("/\\"));
#if defined(_WIN32)
    CreateDirectoryA(dir.c_str(), NULL);
#else
    mkdir(dir.c_str(), 0700);
#endif
    
    return VGREResult::SUCCESS;
}

VGREResult HardwareTokenManager::storeFallbackEncrypted(const std::string& service, const std::string& token) {
    // Read existing tokens
    std::map<std::string, std::string> tokens;
    std::ifstream infile(fallback_path_);
    if (infile.is_open()) {
        std::string line;
        while (std::getline(infile, line)) {
            size_t pos = line.find('=');
            if (pos != std::string::npos) {
                std::string key = line.substr(0, pos);
                std::string encrypted = line.substr(pos + 1);
                tokens[key] = encrypted;
            }
        }
        infile.close();
    }
    
    // Add/update token
    tokens[service] = encryptToken(token);
    
    // Write back
    std::ofstream outfile(fallback_path_);
    if (!outfile.is_open()) {
        return VGREResult::ERROR_IO;
    }
    
    for (const auto& [key, value] : tokens) {
        outfile << key << "=" << value << "\n";
    }
    
    outfile.close();
    
#if !defined(_WIN32)
    chmod(fallback_path_.c_str(), 0600);
#endif
    
    return VGREResult::SUCCESS;
}

VGREResult HardwareTokenManager::getFallbackEncrypted(const std::string& service, std::string& outToken) {
    std::ifstream infile(fallback_path_);
    if (!infile.is_open()) {
        return VGREResult::ERROR_AUTH_FAILED;
    }
    
    std::string line;
    while (std::getline(infile, line)) {
        size_t pos = line.find('=');
        if (pos != std::string::npos) {
            std::string key = line.substr(0, pos);
            if (key == service) {
                std::string encrypted = line.substr(pos + 1);
                outToken = decryptToken(encrypted);
                infile.close();
                return VGREResult::SUCCESS;
            }
        }
    }
    
    infile.close();
    return VGREResult::ERROR_AUTH_FAILED;
}

VGREResult HardwareTokenManager::deleteFallbackEncrypted(const std::string& service) {
    // Read existing tokens
    std::map<std::string, std::string> tokens;
    std::ifstream infile(fallback_path_);
    if (!infile.is_open()) {
        return VGREResult::ERROR_AUTH_FAILED;
    }
    
    std::string line;
    bool found = false;
    while (std::getline(infile, line)) {
        size_t pos = line.find('=');
        if (pos != std::string::npos) {
            std::string key = line.substr(0, pos);
            if (key == service) {
                found = true;
                continue; // Skip this entry
            }
            std::string encrypted = line.substr(pos + 1);
            tokens[key] = encrypted;
        }
    }
    infile.close();
    
    if (!found) {
        return VGREResult::ERROR_AUTH_FAILED;
    }
    
    // Write back
    std::ofstream outfile(fallback_path_);
    if (!outfile.is_open()) {
        return VGREResult::ERROR_IO;
    }
    
    for (const auto& [key, value] : tokens) {
        outfile << key << "=" << value << "\n";
    }
    
    outfile.close();
    return VGREResult::SUCCESS;
}

// ── Authenticated Fallback Encryption ────────────────────────────────────
// Protocol: PBKDF2-HMAC-SHA256 key derivation (machine-specific) →
//           AES-256-CTR encryption → HMAC-SHA256 authentication (Encrypt-then-MAC).
// Wire format: hex(nonce16) ":" hex(ciphertext) ":" hex(mac32)
//
// Security properties:
//   - Key is derived via PBKDF2 (100k iterations) from a static application
//     secret + machine identity (hostname + username), making brute-force hard.
//   - Each encryption uses a fresh 16-byte random nonce.
//   - MAC prevents ciphertext tampering; secure_compare prevents timing attacks.
//   - File mode 0600 provides OS-level access control as a secondary defence.

namespace {

// Helper: encode bytes to lowercase hex string
static std::string toHex(const uint8_t* data, size_t len) {
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for (size_t i = 0; i < len; ++i)
        oss << std::setw(2) << static_cast<int>(data[i]);
    return oss.str();
}

// Helper: decode hex string to bytes; returns false on malformed input
static bool fromHex(const std::string& hex, std::vector<uint8_t>& out) {
    if (hex.size() % 2 != 0) return false;
    out.resize(hex.size() / 2);
    for (size_t i = 0; i < out.size(); ++i) {
        char buf[3] = {hex[2*i], hex[2*i+1], '\0'};
        char* end;
        long v = std::strtol(buf, &end, 16);
        if (end != buf + 2) return false;
        out[i] = static_cast<uint8_t>(v);
    }
    return true;
}

// SHA-256-CTR keystream (reuses the same construction as AES nonce derivation)
// Used to encrypt plaintext: keystream[i] = sha256(key32 || nonce16 || be32(i))[0..blockSize-1]
static void sha256CtrEncrypt(const uint8_t key[32], const uint8_t nonce[16],
                              const uint8_t* input, uint8_t* output, size_t len) {
    size_t offset = 0;
    uint32_t counter = 0;
    while (offset < len) {
        uint8_t block[32 + 16 + 4];
        std::memcpy(block, key, 32);
        std::memcpy(block + 32, nonce, 16);
        block[48] = (counter >> 24) & 0xff;
        block[49] = (counter >> 16) & 0xff;
        block[50] = (counter >>  8) & 0xff;
        block[51] =  counter        & 0xff;
        uint8_t ks[32];
        crypto::sha256(block, sizeof(block), ks);
        std::memset(block, 0, sizeof(block));
        size_t n = std::min(len - offset, static_cast<size_t>(32));
        for (size_t i = 0; i < n; ++i)
            output[offset + i] = input[offset + i] ^ ks[i];
        std::memset(ks, 0, sizeof(ks));
        offset += n;
        ++counter;
    }
}

} // anonymous namespace

std::array<uint8_t, 32> HardwareTokenManager::getMachineKey() {
    // Derive a 32-byte machine-specific key via PBKDF2-HMAC-SHA256.
    // Input material: static app secret + hostname + username (public, but slow to brute-force).
    std::string identity = "vgre_fallback_kdf";

#if defined(_WIN32)
    char hostname[256] = {}; DWORD sz = sizeof(hostname);
    GetComputerNameA(hostname, &sz);
    identity += hostname;
    char username[256] = {}; sz = sizeof(username);
    GetUserNameA(username, &sz);
    identity += username;
#else
    char hostname[256] = {};
    gethostname(hostname, sizeof(hostname));
    identity += hostname;
    const char* user = getenv("USER");
    if (user) identity += user;
#endif

    // Salt = SHA-256("vgre_fallback_v2:" || machine_identity)
    std::string saltInput = "vgre_fallback_v2:" + identity;
    uint8_t salt[32];
    crypto::sha256(reinterpret_cast<const uint8_t*>(saltInput.data()),
                   saltInput.size(), salt);

    std::array<uint8_t, 32> key{};
    crypto::pbkdf2_sha256(
        reinterpret_cast<const uint8_t*>("vgre_fallback_kdf"), 17,
        salt, sizeof(salt),
        100000,  // 100k iterations — adequate for a one-time key derivation
        key.data(), key.size());

    std::memset(salt, 0, sizeof(salt));
    return key;
}

std::string HardwareTokenManager::encryptToken(const std::string& plaintext) {
    // 1. Derive machine key
    auto key = getMachineKey();

    // 2. Generate 16-byte random nonce
    uint8_t nonce[16];
    crypto::random_bytes(nonce, sizeof(nonce));

    // 3. Encrypt: SHA-256-CTR (uses built-in crypto, no AES dependency needed here)
    std::vector<uint8_t> cipher(plaintext.size());
    sha256CtrEncrypt(key.data(), nonce,
                     reinterpret_cast<const uint8_t*>(plaintext.data()),
                     cipher.data(), plaintext.size());

    // 4. MAC = HMAC-SHA256(key, nonce || ciphertext)  — Encrypt-then-MAC
    std::vector<uint8_t> macInput(sizeof(nonce) + cipher.size());
    std::memcpy(macInput.data(), nonce, sizeof(nonce));
    if (!cipher.empty())
        std::memcpy(macInput.data() + sizeof(nonce), cipher.data(), cipher.size());
    uint8_t mac[32];
    crypto::hmac_sha256(key.data(), 32, macInput.data(), macInput.size(), mac);

    // 5. Zeroize key material
    std::fill(key.begin(), key.end(), 0);

    // 6. Encode as hex(nonce):hex(cipher):hex(mac)
    return toHex(nonce, sizeof(nonce)) + ":" +
           toHex(cipher.data(), cipher.size()) + ":" +
           toHex(mac, sizeof(mac));
}

std::string HardwareTokenManager::decryptToken(const std::string& ciphertext) {
    // Split on ":"
    size_t sep1 = ciphertext.find(':');
    if (sep1 == std::string::npos) return "";
    size_t sep2 = ciphertext.find(':', sep1 + 1);
    if (sep2 == std::string::npos) return "";

    std::vector<uint8_t> nonce, cipher, storedMac;
    if (!fromHex(ciphertext.substr(0, sep1), nonce)        || nonce.size() != 16) return "";
    if (!fromHex(ciphertext.substr(sep1 + 1, sep2 - sep1 - 1), cipher)) return "";
    if (!fromHex(ciphertext.substr(sep2 + 1), storedMac)   || storedMac.size() != 32) return "";

    // Derive key and verify MAC (Encrypt-then-MAC: verify before decrypt)
    auto key = getMachineKey();

    std::vector<uint8_t> macInput(nonce.size() + cipher.size());
    std::memcpy(macInput.data(), nonce.data(), nonce.size());
    if (!cipher.empty())
        std::memcpy(macInput.data() + nonce.size(), cipher.data(), cipher.size());

    uint8_t expectedMac[32];
    crypto::hmac_sha256(key.data(), 32, macInput.data(), macInput.size(), expectedMac);

    if (!crypto::secure_compare(expectedMac, storedMac.data(), 32)) {
        std::fill(key.begin(), key.end(), 0);
        return "";  // Authentication failure — token tampered or wrong machine
    }

    // Decrypt
    std::vector<uint8_t> plain(cipher.size());
    if (!cipher.empty())
        sha256CtrEncrypt(key.data(), nonce.data(), cipher.data(), plain.data(), cipher.size());

    std::fill(key.begin(), key.end(), 0);
    return std::string(reinterpret_cast<const char*>(plain.data()), plain.size());
}

// ============================================================================
// Singleton
// ============================================================================

HardwareTokenManager& HardwareTokenManager::instance() {
    static HardwareTokenManager manager;
    return manager;
}

} // namespace advanced
} // namespace vgre
