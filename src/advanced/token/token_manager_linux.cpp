#include "vgre/advanced/hardware_token_manager.h"
#include "vgre/common/logger.h"

#if defined(__linux__)
#include "vgre/common/os_backend.h"
#include <keyutils.h>  // Linux kernel keyring API
#include <vector>

#if defined(VGRE_HAS_LIBSECRET)
#include <libsecret/secret.h>
#endif

namespace vgre {
namespace advanced {

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
        return VGREResult::ERR_NOT_INITIALIZED;
    }
    
    std::string key_desc = "vgre:" + service;

    // Replace existing key atomically from caller perspective.
    // add_key("user", ...) fails when a key with the same description exists
    // in this keyring on many kernels, which caused store/update tests to fail.
    key_serial_t existing = keyctl_search(keyring, "user", key_desc.c_str(), 0);
    if (existing >= 0) {
        (void)keyctl_revoke(existing);
    }

    key_serial_t key = add_key("user", key_desc.c_str(), token.c_str(), token.size(), keyring);
    
    if (key < 0) {
        VGRE_LOG_ERROR("HardwareTokenManager", "Failed to store token in Linux keyring");
        return VGREResult::ERR_IO;
    }
    
    // Set timeout to 0 (persistent until reboot)
    keyctl_set_timeout(key, 0);
    
    return VGREResult::SUCCESS;
}

VGREResult HardwareTokenManager::getLinuxKeyring(const std::string& service, std::string& outToken) {
    key_serial_t keyring = keyctl_search(KEY_SPEC_USER_KEYRING, "keyring", "vgre", 0);
    if (keyring < 0) {
        return VGREResult::ERR_NOT_INITIALIZED;
    }
    
    std::string key_desc = "vgre:" + service;
    key_serial_t key = keyctl_search(keyring, "user", key_desc.c_str(), 0);
    
    if (key < 0) {
        return VGREResult::ERR_AUTH_FAILED;
    }
    
    // Get key size
    long size = keyctl_read(key, NULL, 0);
    if (size < 0) {
        return VGREResult::ERR_IO;
    }
    
    // Read key
    std::vector<char> buffer(size);
    long read_size = keyctl_read(key, buffer.data(), size);
    
    if (read_size < 0) {
        return VGREResult::ERR_IO;
    }
    
    outToken = std::string(buffer.data(), read_size);
    return VGREResult::SUCCESS;
}

VGREResult HardwareTokenManager::deleteLinuxKeyring(const std::string& service) {
    key_serial_t keyring = keyctl_search(KEY_SPEC_USER_KEYRING, "keyring", "vgre", 0);
    if (keyring < 0) {
        return VGREResult::ERR_NOT_INITIALIZED;
    }
    
    std::string key_desc = "vgre:" + service;
    key_serial_t key = keyctl_search(keyring, "user", key_desc.c_str(), 0);
    
    if (key < 0) {
        return VGREResult::ERR_AUTH_FAILED;
    }
    
    if (keyctl_revoke(key) < 0) {
        return VGREResult::ERR_IO;
    }
    
    return VGREResult::SUCCESS;
}

#if defined(VGRE_HAS_LIBSECRET)

// Schema that identifies VGRE tokens by service name.
static const SecretSchema VGRE_SECRET_SCHEMA = {
    "io.vgre.Token",
    SECRET_SCHEMA_NONE,
    {
        { "service", SECRET_SCHEMA_ATTRIBUTE_STRING },
        { nullptr, static_cast<SecretSchemaAttributeType>(0) }  // sentinel
    },
    0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr
};

VGREResult HardwareTokenManager::initLinuxLibsecret() {
    GError* error = nullptr;
    SecretService* svc = secret_service_get_sync(SECRET_SERVICE_NONE, nullptr, &error);
    if (error || !svc) {
        if (error) {
            VGRE_LOG_DEBUG("HardwareTokenManager", "libsecret unavailable: " + std::string(error->message));
            g_error_free(error);
        }
        return VGREResult::ERR_NOT_SUPPORTED;
    }
    g_object_unref(svc);
    VGRE_LOG_INFO("HardwareTokenManager", "libsecret service reachable (GNOME Keyring / KWallet)");
    return VGREResult::SUCCESS;
}

VGREResult HardwareTokenManager::storeLinuxLibsecret(const std::string& service, const std::string& token) {
    GError* error = nullptr;
    std::string label = "VGRE Token: " + service;

    gboolean ok = secret_password_store_sync(
        &VGRE_SECRET_SCHEMA,
        SECRET_COLLECTION_DEFAULT,
        label.c_str(),
        token.c_str(),
        nullptr,
        &error,
        "service", service.c_str(),
        nullptr);

    if (!ok || error) {
        if (error) {
            VGRE_LOG_ERROR("HardwareTokenManager", "libsecret store failed: " + std::string(error->message));
            g_error_free(error);
        }
        return VGREResult::ERR_IO;
    }
    return VGREResult::SUCCESS;
}

VGREResult HardwareTokenManager::getLinuxLibsecret(const std::string& service, std::string& outToken) {
    GError* error = nullptr;

    gchar* password = secret_password_lookup_sync(
        &VGRE_SECRET_SCHEMA,
        nullptr,
        &error,
        "service", service.c_str(),
        nullptr);

    if (error) {
        VGRE_LOG_ERROR("HardwareTokenManager", "libsecret lookup failed: " + std::string(error->message));
        g_error_free(error);
        return VGREResult::ERR_IO;
    }
    if (!password) {
        return VGREResult::ERR_AUTH_FAILED;
    }
    outToken = std::string(password);
    secret_password_free(password);
    return VGREResult::SUCCESS;
}

VGREResult HardwareTokenManager::deleteLinuxLibsecret(const std::string& service) {
    GError* error = nullptr;

    gboolean cleared = secret_password_clear_sync(
        &VGRE_SECRET_SCHEMA,
        nullptr,
        &error,
        "service", service.c_str(),
        nullptr);

    if (error) {
        VGRE_LOG_ERROR("HardwareTokenManager", "libsecret delete failed: " + std::string(error->message));
        g_error_free(error);
        return VGREResult::ERR_IO;
    }
    return VGREResult::SUCCESS;
}

#else  // !VGRE_HAS_LIBSECRET

VGREResult HardwareTokenManager::initLinuxLibsecret() { return VGREResult::ERR_NOT_SUPPORTED; }
VGREResult HardwareTokenManager::storeLinuxLibsecret(const std::string&, const std::string&) { return VGREResult::ERR_NOT_SUPPORTED; }
VGREResult HardwareTokenManager::getLinuxLibsecret(const std::string&, std::string&) { return VGREResult::ERR_NOT_SUPPORTED; }
VGREResult HardwareTokenManager::deleteLinuxLibsecret(const std::string&) { return VGREResult::ERR_NOT_SUPPORTED; }

#endif  // VGRE_HAS_LIBSECRET

} // namespace advanced
} // namespace vgre

#else  // !__linux__

namespace vgre {
namespace advanced {
VGREResult HardwareTokenManager::initLinuxKeyring() { return VGREResult::ERR_NOT_SUPPORTED; }
VGREResult HardwareTokenManager::storeLinuxKeyring(const std::string&, const std::string&) { return VGREResult::ERR_NOT_SUPPORTED; }
VGREResult HardwareTokenManager::getLinuxKeyring(const std::string&, std::string&) { return VGREResult::ERR_NOT_SUPPORTED; }
VGREResult HardwareTokenManager::deleteLinuxKeyring(const std::string&) { return VGREResult::ERR_NOT_SUPPORTED; }
VGREResult HardwareTokenManager::initLinuxLibsecret() { return VGREResult::ERR_NOT_SUPPORTED; }
VGREResult HardwareTokenManager::storeLinuxLibsecret(const std::string&, const std::string&) { return VGREResult::ERR_NOT_SUPPORTED; }
VGREResult HardwareTokenManager::getLinuxLibsecret(const std::string&, std::string&) { return VGREResult::ERR_NOT_SUPPORTED; }
VGREResult HardwareTokenManager::deleteLinuxLibsecret(const std::string&) { return VGREResult::ERR_NOT_SUPPORTED; }
} // namespace advanced
} // namespace vgre

#endif
