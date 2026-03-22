#ifndef VGRE_COMMON_SYSTEM_UTILS_H
#define VGRE_COMMON_SYSTEM_UTILS_H

#include "vgre/common/error_codes.h"
#include <string>
#include <filesystem>
#include <cstdlib>

#ifndef _WIN32
#include <dlfcn.h>
#else
#include <windows.h>
#endif

namespace vgre {
namespace common {

/**
 * @brief Get the user's home directory in a cross-platform way.
 */
inline std::string getHomeDirectory() {
#ifndef _WIN32
    const char* home = std::getenv("HOME");
    return home ? std::string(home) : "";
#else
    const char* userProfile = std::getenv("USERPROFILE");
    if (userProfile) return std::string(userProfile);
    const char* homeDrive = std::getenv("HOMEDRIVE");
    const char* homePath = std::getenv("HOMEPATH");
    if (homeDrive && homePath) return std::string(homeDrive) + std::string(homePath);
    return "";
#endif
}

/**
 * @brief Get the platform-appropriate application data or cache directory.
 */
inline std::string getCacheRoot() {
    std::string home = getHomeDirectory();
#ifndef _WIN32
    return home.empty() ? ".vgre_cache" : home + "/.vgre_cache";
#else
    const char* localAppData = std::getenv("LOCALAPPDATA");
    if (localAppData) return std::string(localAppData) + "/VGRE";
    return home.empty() ? ".vgre_cache" : home + "/AppData/Local/VGRE";
#endif
}

/**
 * @brief Get the absolute path to the currently running executable or shared library.
 */
inline std::filesystem::path getBinaryPath() {
    std::filesystem::path path;
#ifndef _WIN32
    Dl_info info;
    static int dummy = 0;
    if (dladdr((void*)&dummy, &info) && info.dli_fname) {
        path = std::filesystem::path(info.dli_fname).parent_path();
    }
#else
    char buffer[MAX_PATH];
    DWORD size = GetModuleFileNameA(NULL, buffer, MAX_PATH);
    if (size != 0) {
        path = std::filesystem::path(buffer).parent_path();
    }
#endif
    return path;
}

/**
 * @brief Robustly find the VGRE include directory.
 */
inline std::string findIncludeDir() {
    // 0. Manual override
    const char* envPath = std::getenv("VGRE_INCLUDE_DIR");
    if (envPath) return envPath;

    // 1. Check relative to binary
    std::filesystem::path binPath = getBinaryPath();
    if (!binPath.empty()) {
        // Standard layout: bin/libvgre.so -> include/vgre/
        if (std::filesystem::exists(binPath / "include/vgre/common/types.h")) {
            return (binPath / "include").string();
        }
        // Dev layout: build/src/api/libvgre.so -> ../../include/vgre/
        std::filesystem::path p = binPath;
        for (int i = 0; i < 3; ++i) {
            if (p.has_parent_path()) {
                p = p.parent_path();
                if (std::filesystem::exists(p / "include/vgre/common/types.h")) {
                    return (p / "include").string();
                }
            }
        }
    }

    // 2. Fallback: Search upwards from CWD
    auto cur = std::filesystem::current_path();
    for (int i = 0; i < 5; ++i) {
        if (std::filesystem::exists(cur / "include/vgre/common/types.h")) {
            return (cur / "include").string();
        }
        if (cur.has_parent_path()) cur = cur.parent_path();
    }

    // 3. Absolute fallback (likely will fail if not using standard install locations)
    return "/usr/local/include";
}

} // namespace common
} // namespace vgre

#endif // VGRE_COMMON_SYSTEM_UTILS_H
