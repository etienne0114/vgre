#include "vgre/advanced/hardware_token_manager.h"
#include "vgre/common/logger.h"

#if defined(__APPLE__)
#include <Security/Security.h>

namespace vgre {
namespace advanced {

namespace {

constexpr const char* kKeychainAccount = "VGRE";

CFStringRef makeCFString(const std::string& s) {
    return CFStringCreateWithCString(kCFAllocatorDefault, s.c_str(),
                                     kCFStringEncodingUTF8);
}

CFDictionaryRef makeLookupQuery(CFStringRef service, bool returnData) {
    const void* keys[] = {
        kSecClass, kSecAttrService, kSecAttrAccount,
        returnData ? kSecReturnData : nullptr,
        returnData ? kSecMatchLimit : nullptr,
    };
    const void* vals[] = {
        kSecClassGenericPassword,
        service,
        CFStringCreateWithCString(kCFAllocatorDefault, kKeychainAccount,
                                  kCFStringEncodingUTF8),
        returnData ? kCFBooleanTrue : nullptr,
        returnData ? kSecMatchLimitOne : nullptr,
    };
    const CFIndex n = returnData ? 5 : 3;
    return CFDictionaryCreate(kCFAllocatorDefault, keys, vals, n,
                              &kCFTypeDictionaryKeyCallBacks,
                              &kCFTypeDictionaryValueCallBacks);
}

void releaseQuery(CFDictionaryRef query) {
    if (!query) return;
    // Release the account CFString we created inside makeLookupQuery.
    CFTypeRef acct = CFDictionaryGetValue(query, kSecAttrAccount);
    if (acct) CFRelease(acct);
    CFRelease(query);
}

} // namespace

VGREResult HardwareTokenManager::initMacOSKeychain() {
    return VGREResult::SUCCESS;
}

VGREResult HardwareTokenManager::storeMacOSKeychain(const std::string& service,
                                                     const std::string& token) {
    CFStringRef svc = makeCFString(service);
    if (!svc) return VGREResult::ERR_IO;

    CFDataRef data = CFDataCreate(kCFAllocatorDefault,
                                  reinterpret_cast<const UInt8*>(token.data()),
                                  static_cast<CFIndex>(token.size()));
    if (!data) {
        CFRelease(svc);
        return VGREResult::ERR_IO;
    }

    CFStringRef acct = makeCFString(kKeychainAccount);
    const void* addKeys[] = {
        kSecClass, kSecAttrService, kSecAttrAccount, kSecValueData, kSecAttrAccessible,
    };
    const void* addVals[] = {
        kSecClassGenericPassword, svc, acct, data, kSecAttrAccessibleAfterFirstUnlock,
    };
    CFDictionaryRef addQuery = CFDictionaryCreate(
        kCFAllocatorDefault, addKeys, addVals, 5,
        &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);

    OSStatus status = SecItemAdd(addQuery, nullptr);
    CFRelease(addQuery);

    if (status == errSecDuplicateItem) {
        CFDictionaryRef lookup = makeLookupQuery(svc, false);
        const void* updateKeys[] = { kSecValueData };
        const void* updateVals[] = { data };
        CFDictionaryRef update = CFDictionaryCreate(
            kCFAllocatorDefault, updateKeys, updateVals, 1,
            &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
        status = SecItemUpdate(lookup, update);
        releaseQuery(lookup);
        CFRelease(update);
    }

    CFRelease(acct);
    CFRelease(data);
    CFRelease(svc);

    if (status != errSecSuccess) {
        VGRE_LOG_ERROR("HardwareTokenManager",
                       "Failed to store token in macOS Keychain: " + std::to_string(status));
        return VGREResult::ERR_IO;
    }
    return VGREResult::SUCCESS;
}

VGREResult HardwareTokenManager::getMacOSKeychain(const std::string& service,
                                                  std::string& outToken) {
    CFStringRef svc = makeCFString(service);
    if (!svc) return VGREResult::ERR_IO;

    CFDictionaryRef lookup = makeLookupQuery(svc, true);
    CFTypeRef result = nullptr;
    OSStatus status = SecItemCopyMatching(lookup, &result);
    releaseQuery(lookup);
    CFRelease(svc);

    if (status != errSecSuccess) return VGREResult::ERR_AUTH_FAILED;

    if (!result || CFGetTypeID(result) != CFDataGetTypeID()) {
        if (result) CFRelease(result);
        return VGREResult::ERR_IO;
    }

    auto* data = static_cast<CFDataRef>(result);
    const auto* bytes = CFDataGetBytePtr(data);
    CFIndex len = CFDataGetLength(data);
    if (!bytes || len < 0) {
        CFRelease(result);
        return VGREResult::ERR_IO;
    }
    outToken.assign(reinterpret_cast<const char*>(bytes), static_cast<size_t>(len));
    CFRelease(result);
    return VGREResult::SUCCESS;
}

VGREResult HardwareTokenManager::deleteMacOSKeychain(const std::string& service) {
    CFStringRef svc = makeCFString(service);
    if (!svc) return VGREResult::ERR_IO;

    CFDictionaryRef lookup = makeLookupQuery(svc, false);
    OSStatus status = SecItemDelete(lookup);
    releaseQuery(lookup);
    CFRelease(svc);

    if (status != errSecSuccess && status != errSecItemNotFound) {
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
VGREResult HardwareTokenManager::storeMacOSKeychain(const std::string&, const std::string&) {
    return VGREResult::ERR_NOT_SUPPORTED;
}
VGREResult HardwareTokenManager::getMacOSKeychain(const std::string&, std::string&) {
    return VGREResult::ERR_NOT_SUPPORTED;
}
VGREResult HardwareTokenManager::deleteMacOSKeychain(const std::string&) {
    return VGREResult::ERR_NOT_SUPPORTED;
}
} // namespace advanced
} // namespace vgre

#endif
