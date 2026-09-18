#ifndef VGRE_COMMON_SYSTEM_UTILS_H
#define VGRE_COMMON_SYSTEM_UTILS_H

#include "vgre/common/error_codes.h"
#include <string>
#include <filesystem>
#include <cstdlib>
#include <array>
#include <memory>
#include <vector>
#include <cctype>

#include "vgre/common/os_backend.h"

namespace vgre {
namespace common {

namespace detail {

inline std::string trimTrailingWhitespace(std::string s) {
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' '))
        s.pop_back();
    return s;
}

inline std::string runCommandCapture(const char* cmd) {
    std::array<char, 512> buf{};
    std::string out;
#if defined(_WIN32)
    FILE* pipe = _popen(cmd, "r");
#else
    FILE* pipe = popen(cmd, "r");
#endif
    if (!pipe) return out;
    while (fgets(buf.data(), static_cast<int>(buf.size()), pipe))
        out += buf.data();
#if defined(_WIN32)
    _pclose(pipe);
#else
    pclose(pipe);
#endif
    return trimTrailingWhitespace(out);
}

inline bool pathExists(const std::filesystem::path& p) {
    std::error_code ec;
    return std::filesystem::exists(p, ec);
}

} // namespace detail

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

    // 1. Check relative to binary (libvgre.so / libvgre.dylib / vgre.dll)
    std::filesystem::path binPath = getBinaryPath();
    if (!binPath.empty()) {
        if (detail::pathExists(binPath / "include/vgre/common/types.h")) {
            return (binPath / "include").string();
        }
        std::filesystem::path p = binPath;
        for (int i = 0; i < 4; ++i) {
            if (p.has_parent_path()) {
                p = p.parent_path();
                if (detail::pathExists(p / "include/vgre/common/types.h")) {
                    return (p / "include").string();
                }
            }
        }
    }

    // 2. Fallback: Search upwards from CWD
    auto cur = std::filesystem::current_path();
    for (int i = 0; i < 5; ++i) {
        if (detail::pathExists(cur / "include/vgre/common/types.h")) {
            return (cur / "include").string();
        }
        if (cur.has_parent_path()) cur = cur.parent_path();
    }

    // 3. Platform-specific install roots (existence-checked; no hardcoded prefixes)
#ifdef _WIN32
    const char* localAppData = std::getenv("LOCALAPPDATA");
    if (localAppData) {
        std::filesystem::path p(localAppData);
        p /= "VGRE/include";
        if (detail::pathExists(p / "vgre/common/types.h")) return p.string();
    }
    const char* progFiles = std::getenv("PROGRAMFILES");
    if (progFiles) {
        std::filesystem::path p(progFiles);
        p /= "VGRE/include";
        if (detail::pathExists(p / "vgre/common/types.h")) return p.string();
    }
#else
    const char* install = std::getenv("VGRE_INSTALL_DIR");
    if (install) {
        std::filesystem::path p(install);
        p /= "include";
        if (detail::pathExists(p / "vgre/common/types.h")) return p.string();
    }
    std::string home = getHomeDirectory();
    if (!home.empty()) {
        std::filesystem::path p(home);
        p /= ".local/share/VGRE/include";
        if (detail::pathExists(p / "vgre/common/types.h")) return p.string();
    }
#if defined(__APPLE__)
    std::string brewPrefix = detail::runCommandCapture("brew --prefix vgre 2>/dev/null");
    if (!brewPrefix.empty()) {
        std::filesystem::path p(brewPrefix);
        p /= "include";
        if (detail::pathExists(p / "vgre/common/types.h")) return p.string();
    }
#endif
#endif
    return "";
}

/**
 * @brief Robustly find the Clang compiler path for JIT kernel compilation.
 */
inline std::string findCompilerPath() {
    const char* envPath = std::getenv("VGRE_CLANG_PATH");
    if (envPath && envPath[0]) return envPath;

    auto tryPath = [](const std::filesystem::path& p) -> std::string {
        return detail::pathExists(p) ? p.string() : std::string();
    };

    std::filesystem::path binPath = getBinaryPath();
    if (!binPath.empty()) {
#ifdef _WIN32
        for (const auto& rel : {"BuildTools/llvm/bin/clang++.exe", "llvm/bin/clang++.exe"}) {
            if (auto s = tryPath(binPath / rel); !s.empty()) return s;
        }
        auto parent = binPath;
        for (int i = 0; i < 3; ++i) {
            if (auto s = tryPath(parent / "BuildTools/llvm/bin/clang++.exe"); !s.empty())
                return s;
            if (parent.has_parent_path()) parent = parent.parent_path();
        }
        // Install-BuildTools.ps1's actual install location: it has no fixed
        // relation to wherever the running binary happens to live (a dev
        // build under the source tree, an installed app under Program
        // Files, ...), so the binary-relative search above never finds it.
        if (const char* installDir = std::getenv("VGRE_INSTALL_DIR")) {
            if (auto s = tryPath(std::filesystem::path(installDir) / "BuildTools/llvm/bin/clang++.exe"); !s.empty())
                return s;
        }
        if (const char* localAppData = std::getenv("LOCALAPPDATA")) {
            if (auto s = tryPath(std::filesystem::path(localAppData) / "VGRE/BuildTools/llvm/bin/clang++.exe"); !s.empty())
                return s;
        }
#else
        for (const auto& rel : {"clang++", "llvm/bin/clang++", "../llvm@18/bin/clang++",
                                "../../llvm/bin/clang++"}) {
            if (auto s = tryPath(binPath / rel); !s.empty()) return s;
        }
        for (const char* cfg : {"llvm-config-18", "llvm-config"}) {
            std::string bindir = detail::runCommandCapture(
                (std::string(cfg) + " --bindir 2>/dev/null").c_str());
            if (!bindir.empty()) {
                if (auto s = tryPath(std::filesystem::path(bindir) / "clang++"); !s.empty())
                    return s;
            }
        }
#if defined(__APPLE__)
        std::string llvm18 = detail::runCommandCapture("brew --prefix llvm@18 2>/dev/null");
        if (!llvm18.empty()) {
            if (auto s = tryPath(std::filesystem::path(llvm18) / "bin/clang++"); !s.empty())
                return s;
        }
#endif
        std::string which = detail::runCommandCapture("command -v clang++ 2>/dev/null");
        if (!which.empty() && detail::pathExists(which)) return which;
        which = detail::runCommandCapture("command -v clang-18 2>/dev/null");
        if (!which.empty() && detail::pathExists(which)) return which;
#endif
    }

    return "clang++";
}

#if defined(_WIN32)
/**
 * @brief Find an MSVC toolset directory (VC/Tools/MSVC/<version>) for the
 * runtime JIT's clang subprocess to borrow STL/CRT headers from via
 * -vctoolsdir, pinning it explicitly instead of leaving clang to
 * auto-detect. On a machine with multiple side-by-side VS installs (e.g. a
 * preview/Insiders build alongside the stable one), auto-detection can pick
 * a too-new toolset whose STL uses syntax the pinned clang version can't
 * parse at all — not a version-gate check, a real parse failure.
 *
 * Returns an empty string if nothing can be found; callers should skip
 * -vctoolsdir entirely in that case and let clang fall back to its own
 * auto-detection (unchanged behavior for machines where only one VS install
 * exists, e.g. most CI runners).
 */
inline std::string findMSVCToolsDir() {
    if (const char* envOverride = std::getenv("VGRE_MSVC_TOOLS_DIR")) {
        if (envOverride[0]) return envOverride;
    }

    // A vcvars-initialized shell already names the exact toolset the caller
    // intends to build with — prefer it over any filesystem probing.
    if (const char* vcTools = std::getenv("VCToolsInstallDir")) {
        if (vcTools[0] && detail::pathExists(vcTools)) {
            std::string s(vcTools);
            while (!s.empty() && (s.back() == '\\' || s.back() == '/')) s.pop_back();
            return s;
        }
    }

    // No override and no active dev-shell context: probe the standard VS
    // install roots. Best-effort default, not exhaustive — covers the
    // layout every VS 2022+ edition (Community/Professional/Enterprise/
    // Insiders/BuildTools) installs into.
    static const char* kRoots[] = {
        "C:\\Program Files\\Microsoft Visual Studio",
        "C:\\Program Files (x86)\\Microsoft Visual Studio",
    };
    std::vector<std::string> candidates;
    std::error_code ec;
    for (const char* root : kRoots) {
        if (!std::filesystem::is_directory(root, ec)) continue;
        for (auto& yearEntry : std::filesystem::directory_iterator(root, ec)) {
            if (!yearEntry.is_directory(ec)) continue;
            for (auto& editionEntry : std::filesystem::directory_iterator(yearEntry.path(), ec)) {
                if (!editionEntry.is_directory(ec)) continue;
                auto toolsRoot = editionEntry.path() / "VC" / "Tools" / "MSVC";
                if (!std::filesystem::is_directory(toolsRoot, ec)) continue;
                for (auto& verEntry : std::filesystem::directory_iterator(toolsRoot, ec)) {
                    if (verEntry.is_directory(ec)) candidates.push_back(verEntry.path().string());
                }
            }
        }
    }
    if (candidates.empty()) return "";

    // Prefer a non-preview toolset when more than one is installed.
    for (auto& c : candidates) {
        std::string lower = c;
        for (auto& ch : lower) ch = static_cast<char>(::tolower(static_cast<unsigned char>(ch)));
        if (lower.find("insiders") == std::string::npos && lower.find("preview") == std::string::npos)
            return c;
    }
    return candidates.front();
}
#endif

} // namespace common
} // namespace vgre

#endif // VGRE_COMMON_SYSTEM_UTILS_H
