#include "vgre/advanced/hardware_token_manager.h"
#include "vgre/common/logger.h"

#include <random>
#include <iomanip>
#include <sstream>

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

    // Backend priority order is configurable via VGRE_TOKEN_BACKEND env var.
    // Valid values: "keyring", "libsecret", "keychain", "credman", "tpm", "file".
    // Default: platform-native backend first, TPM second, encrypted file last.
    const char *preferredBackend = std::getenv("VGRE_TOKEN_BACKEND");

    // Helper lambda: attempt a backend by name string.
    auto tryNamed = [&](const char *name) -> bool {
        if (!name) return false;
#if defined(__linux__)
        if (std::string(name) == "keyring"   && initLinuxKeyring()    == VGREResult::SUCCESS) { backend_ = BackendType::LINUX_KEYRING;   initialized_ = true; VGRE_LOG_INFO("HardwareTokenManager", "Initialized with Linux Keyring (env override)");   return true; }
        if (std::string(name) == "libsecret" && initLinuxLibsecret()  == VGREResult::SUCCESS) { backend_ = BackendType::LINUX_LIBSECRET; initialized_ = true; VGRE_LOG_INFO("HardwareTokenManager", "Initialized with libsecret (env override)");        return true; }
#endif
#if defined(__APPLE__)
        if (std::string(name) == "keychain"  && initMacOSKeychain()   == VGREResult::SUCCESS) { backend_ = BackendType::MACOS_KEYCHAIN;  initialized_ = true; VGRE_LOG_INFO("HardwareTokenManager", "Initialized with macOS Keychain (env override)");   return true; }
#endif
#if defined(_WIN32)
        if (std::string(name) == "credman"   && initWindowsCredMan()  == VGREResult::SUCCESS) { backend_ = BackendType::WINDOWS_CREDMAN; initialized_ = true; VGRE_LOG_INFO("HardwareTokenManager", "Initialized with Windows CredMan (env override)");  return true; }
#endif
        if (std::string(name) == "tpm"       && initTPM()             == VGREResult::SUCCESS) { backend_ = BackendType::TPM_2_0;         initialized_ = true; VGRE_LOG_INFO("HardwareTokenManager", "Initialized with TPM 2.0 (env override)");           return true; }
        if (std::string(name) == "file"      && initFallbackEncrypted() == VGREResult::SUCCESS) { backend_ = BackendType::FALLBACK_ENCRYPTED; initialized_ = true; VGRE_LOG_WARN("HardwareTokenManager", "Initialized with encrypted file (env override)"); return true; }
        return false;
    };

    // If a preferred backend is specified, try it first.
    if (preferredBackend) {
        VGRE_LOG_INFO("HardwareTokenManager",
                      "VGRE_TOKEN_BACKEND override: trying '" +
                          std::string(preferredBackend) + "' first");
        if (tryNamed(preferredBackend)) return VGREResult::SUCCESS;
        VGRE_LOG_WARN("HardwareTokenManager",
                      "Preferred backend '" + std::string(preferredBackend) +
                          "' unavailable; falling back to default order");
    }

    // Default priority: platform-native → TPM → encrypted file.
#if defined(__linux__)
    if (initLinuxKeyring() == VGREResult::SUCCESS) {
        backend_ = BackendType::LINUX_KEYRING;
        initialized_ = true;
        VGRE_LOG_INFO("HardwareTokenManager", "Initialized with Linux Keyring backend");
        return VGREResult::SUCCESS;
    }

    if (initLinuxLibsecret() == VGREResult::SUCCESS) {
        backend_ = BackendType::LINUX_LIBSECRET;
        initialized_ = true;
        VGRE_LOG_INFO("HardwareTokenManager", "Initialized with libsecret backend");
        return VGREResult::SUCCESS;
    }

#elif defined(__APPLE__)
    if (initMacOSKeychain() == VGREResult::SUCCESS) {
        backend_ = BackendType::MACOS_KEYCHAIN;
        initialized_ = true;
        VGRE_LOG_INFO("HardwareTokenManager", "Initialized with macOS Keychain backend");
        return VGREResult::SUCCESS;
    }

#elif defined(_WIN32)
    if (initWindowsCredMan() == VGREResult::SUCCESS) {
        backend_ = BackendType::WINDOWS_CREDMAN;
        initialized_ = true;
        VGRE_LOG_INFO("HardwareTokenManager", "Initialized with Windows Credential Manager backend");
        return VGREResult::SUCCESS;
    }
#endif

    if (initTPM() == VGREResult::SUCCESS) {
        backend_ = BackendType::TPM_2_0;
        initialized_ = true;
        VGRE_LOG_INFO("HardwareTokenManager", "Initialized with TPM 2.0 backend");
        return VGREResult::SUCCESS;
    }

    if (initFallbackEncrypted() == VGREResult::SUCCESS) {
        backend_ = BackendType::FALLBACK_ENCRYPTED;
        initialized_ = true;
        VGRE_LOG_WARN("HardwareTokenManager", "Using encrypted file fallback - not recommended for production");
        return VGREResult::SUCCESS;
    }

    VGRE_LOG_ERROR("HardwareTokenManager", "Failed to initialize any secure storage backend");
    return VGREResult::ERR_NOT_SUPPORTED;
}

VGREResult HardwareTokenManager::storeToken(const std::string& service, const std::string& token) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!initialized_) {
        return VGREResult::ERR_NOT_INITIALIZED;
    }

    auto storeOn = [&](BackendType b) -> VGREResult {
        switch (b) {
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
    };

    VGREResult r = storeOn(backend_);
    if (r == VGREResult::SUCCESS) {
        return r;
    }

    // Operational failover: if the selected backend initializes but cannot
    // perform store operations in this runtime (common in CI/sandboxed
    // keyring sessions), downgrade to the next secure backend automatically.
#if defined(__linux__)
    const BackendType fallbackOrder[] = {
        BackendType::LINUX_LIBSECRET,
        BackendType::TPM_2_0,
        BackendType::FALLBACK_ENCRYPTED
    };
    for (BackendType candidate : fallbackOrder) {
        if (candidate == backend_) continue;
        VGREResult initRes = VGREResult::ERR_NOT_SUPPORTED;
        switch (candidate) {
            case BackendType::LINUX_LIBSECRET: initRes = initLinuxLibsecret(); break;
            case BackendType::TPM_2_0: initRes = initTPM(); break;
            case BackendType::FALLBACK_ENCRYPTED: initRes = initFallbackEncrypted(); break;
            default: break;
        }
        if (initRes != VGREResult::SUCCESS) continue;
        VGREResult tryStore = storeOn(candidate);
        if (tryStore == VGREResult::SUCCESS) {
            backend_ = candidate;
            VGRE_LOG_WARN("HardwareTokenManager", "Store failover to backend: " + getBackendName());
            return VGREResult::SUCCESS;
        }
    }
#endif

    return r;
}

VGREResult HardwareTokenManager::getToken(const std::string& service, std::string& outToken) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!initialized_) {
        return VGREResult::ERR_NOT_INITIALIZED;
    }

    auto getOn = [&](BackendType b) -> VGREResult {
        switch (b) {
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
    };

    VGREResult r = getOn(backend_);
    if (r == VGREResult::SUCCESS || r == VGREResult::ERR_AUTH_FAILED) {
        return r;
    }

#if defined(__linux__)
    const BackendType fallbackOrder[] = {
        BackendType::LINUX_LIBSECRET,
        BackendType::TPM_2_0,
        BackendType::FALLBACK_ENCRYPTED
    };
    for (BackendType candidate : fallbackOrder) {
        if (candidate == backend_) continue;
        VGREResult initRes = VGREResult::ERR_NOT_SUPPORTED;
        switch (candidate) {
            case BackendType::LINUX_LIBSECRET: initRes = initLinuxLibsecret(); break;
            case BackendType::TPM_2_0: initRes = initTPM(); break;
            case BackendType::FALLBACK_ENCRYPTED: initRes = initFallbackEncrypted(); break;
            default: break;
        }
        if (initRes != VGREResult::SUCCESS) continue;
        VGREResult tryGet = getOn(candidate);
        if (tryGet == VGREResult::SUCCESS || tryGet == VGREResult::ERR_AUTH_FAILED) {
            backend_ = candidate;
            VGRE_LOG_WARN("HardwareTokenManager", "Get failover to backend: " + getBackendName());
            return tryGet;
        }
    }
#endif

    return r;
}

VGREResult HardwareTokenManager::deleteToken(const std::string& service) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!initialized_) {
        return VGREResult::ERR_NOT_INITIALIZED;
    }

    auto delOn = [&](BackendType b) -> VGREResult {
        switch (b) {
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
    };

    VGREResult r = delOn(backend_);
    if (r == VGREResult::SUCCESS || r == VGREResult::ERR_AUTH_FAILED) {
        return r;
    }

#if defined(__linux__)
    const BackendType fallbackOrder[] = {
        BackendType::LINUX_LIBSECRET,
        BackendType::TPM_2_0,
        BackendType::FALLBACK_ENCRYPTED
    };
    for (BackendType candidate : fallbackOrder) {
        if (candidate == backend_) continue;
        VGREResult initRes = VGREResult::ERR_NOT_SUPPORTED;
        switch (candidate) {
            case BackendType::LINUX_LIBSECRET: initRes = initLinuxLibsecret(); break;
            case BackendType::TPM_2_0: initRes = initTPM(); break;
            case BackendType::FALLBACK_ENCRYPTED: initRes = initFallbackEncrypted(); break;
            default: break;
        }
        if (initRes != VGREResult::SUCCESS) continue;
        VGREResult tryDel = delOn(candidate);
        if (tryDel == VGREResult::SUCCESS || tryDel == VGREResult::ERR_AUTH_FAILED) {
            backend_ = candidate;
            VGRE_LOG_WARN("HardwareTokenManager", "Delete failover to backend: " + getBackendName());
            return tryDel;
        }
    }
#endif

    return r;
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

HardwareTokenManager& HardwareTokenManager::instance() {
    static HardwareTokenManager inst;
    return inst;
}

} // namespace advanced
} // namespace vgre
