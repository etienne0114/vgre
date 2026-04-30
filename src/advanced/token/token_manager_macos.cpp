#include "vgre/advanced/hardware_token_manager.h"
#include "vgre/common/logger.h"

#if defined(__APPLE__)
#include <Security/Security.h>

namespace vgre {
namespace advanced {

VGREResult HardwareTokenManager::initMacOSKeychain() {
    return VGREResult::SUCCESS;
}

VGREResult HardwareTokenManager::storeMacOSKeychain(const std::string& service, const std::string& token) {
    OSStatus status = SecKeychainAddGenericPassword(
        NULL,                           // default keychain
        service.length(),               // service name length
        service.c_str(),                // service name
        0,                              // account name length
        NULL,                           // account name
        token.length(),                 // password length
        token.c_str(),                  // password data
        NULL                            // item ref
    );
    
    if (status == errSecDuplicateItem) {
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
        VGRE_LOG_ERROR("HardwareTokenManager", "Failed to store token in macOS Keychain: " + std::to_string(status));
        return VGREResult::ERR_IO;
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
        return VGREResult::ERR_AUTH_FAILED;
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
        return VGREResult::ERR_AUTH_FAILED;
    }
    
    status = SecKeychainItemDelete(itemRef);
    CFRelease(itemRef);
    
    if (status != errSecSuccess) {
        return VGREResult::ERR_IO;
    }
    
    return VGREResult::SUCCESS;
}

} // namespace advanced
} // namespace vgre

#else

namespace vgre {
namespace advanced {
VGREResult HardwareTokenManager::initMacOSKeychain() { return VGREResult::ERR_NOT_SUPPORTED; }
VGREResult HardwareTokenManager::storeMacOSKeychain(const std::string&, const std::string&) { return VGREResult::ERR_NOT_SUPPORTED; }
VGREResult HardwareTokenManager::getMacOSKeychain(const std::string&, std::string&) { return VGREResult::ERR_NOT_SUPPORTED; }
VGREResult HardwareTokenManager::deleteMacOSKeychain(const std::string&) { return VGREResult::ERR_NOT_SUPPORTED; }
} // namespace advanced
} // namespace vgre

#endif
