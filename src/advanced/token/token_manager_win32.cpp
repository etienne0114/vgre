#include "vgre/advanced/hardware_token_manager.h"
#include "vgre/common/logger.h"

#if defined(_WIN32)
#include <windows.h>
#include <wincred.h>

namespace vgre {
namespace advanced {

VGREResult HardwareTokenManager::initWindowsCredMan() {
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
        VGRE_LOG_ERROR("HardwareTokenManager", "Failed to store token in Windows Credential Manager: " + std::to_string(GetLastError()));
        return VGREResult::ERR_IO;
    }
    
    return VGREResult::SUCCESS;
}

VGREResult HardwareTokenManager::getWindowsCredMan(const std::string& service, std::string& outToken) {
    std::wstring wservice(service.begin(), service.end());
    std::wstring wtarget = L"VGRE:" + wservice;
    
    PCREDENTIALW pcred = NULL;
    if (!CredReadW(wtarget.c_str(), CRED_TYPE_GENERIC, 0, &pcred)) {
        return VGREResult::ERR_AUTH_FAILED;
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
        return VGREResult::ERR_AUTH_FAILED;
    }
    
    return VGREResult::SUCCESS;
}

} // namespace advanced
} // namespace vgre

#else

namespace vgre {
namespace advanced {
VGREResult HardwareTokenManager::initWindowsCredMan() { return VGREResult::ERR_NOT_SUPPORTED; }
VGREResult HardwareTokenManager::storeWindowsCredMan(const std::string&, const std::string&) { return VGREResult::ERR_NOT_SUPPORTED; }
VGREResult HardwareTokenManager::getWindowsCredMan(const std::string&, std::string&) { return VGREResult::ERR_NOT_SUPPORTED; }
VGREResult HardwareTokenManager::deleteWindowsCredMan(const std::string&) { return VGREResult::ERR_NOT_SUPPORTED; }
} // namespace advanced
} // namespace vgre

#endif
