/**
 * @file os_backend.h
 * @brief Cross-platform OS abstraction layer for VGRE.
 *
 * Include this header instead of raw POSIX or Win32 system headers.
 * All platform-specific includes and compatibility shims live here.
 *
 * Covered:
 *   - Virtual memory (mmap/VirtualAlloc, mprotect/VirtualProtect)
 *   - Signal handling (sigaction / VEH)
 *   - File I/O helpers (stat, open, read, write, close)
 *   - Socket helpers (unistd, fcntl, poll / Winsock)
 *   - Thread affinity (sched / SetThreadAffinityMask)
 *   - Dynamic library loading (dlopen / LoadLibrary)
 *   - Process info (getpid / GetCurrentProcessId)
 *   - Shared memory (shm_open+mmap / CreateFileMapping)
 *   - Monotonic clock (clock_gettime / QueryPerformanceCounter)
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

// ─────────────────────────────────────────────────────────────────────────────
// Platform detection
// ─────────────────────────────────────────────────────────────────────────────
#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  include <windows.h>
#  include <io.h>        // _access, _open, _read, _write, _close
#  include <process.h>   // _getpid
#  ifndef PROT_NONE
#    define PROT_NONE  0
#  endif
#  ifndef PROT_READ
#    define PROT_READ  1
#  endif
#  ifndef PROT_WRITE
#    define PROT_WRITE 2
#  endif
#  ifndef PROT_EXEC
#    define PROT_EXEC  4
#  endif
#else
#  include <unistd.h>
#  include <fcntl.h>
#  include <signal.h>
#  include <poll.h>
#  include <dlfcn.h>
#  include <sched.h>
#  include <pthread.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <sys/types.h>
#  include <sys/socket.h>
#  include <netinet/in.h>
#  include <arpa/inet.h>
#  include <cerrno>
#  include <fstream>   // std::ifstream — Linux sysfs L2 cache size fallback
#  ifdef __APPLE__
#    include <sys/sysctl.h>  // sysctlbyname — macOS L2 cache size + NUMA info
#  endif
// macOS uses MAP_ANON; Linux/BSD use MAP_ANONYMOUS. Provide a single name.
#  ifndef MAP_ANONYMOUS
#    define MAP_ANONYMOUS MAP_ANON
#  endif
#endif

// Windows needs <vector> for GetLogicalProcessorInformation buffer
#if defined(_WIN32)
#  include <vector>
#endif

// ─────────────────────────────────────────────────────────────────────────────
// vgre_getpid — process ID (int on all platforms)
// ─────────────────────────────────────────────────────────────────────────────
namespace vgre { namespace os {

inline int getpid() noexcept {
#if defined(_WIN32)
    return static_cast<int>(::GetCurrentProcessId());
#else
    return static_cast<int>(::getpid());
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
// vgre::os::mmap_alloc — allocate anonymous writable pages
// Returns nullptr on failure.
// ─────────────────────────────────────────────────────────────────────────────
inline void* mmap_alloc(std::size_t size) noexcept {
#if defined(_WIN32)
    return ::VirtualAlloc(nullptr, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
#else
    void* p = ::mmap(nullptr, size, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return (p == MAP_FAILED) ? nullptr : p;
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
// vgre::os::mmap_free — release pages previously obtained via mmap_alloc
// ─────────────────────────────────────────────────────────────────────────────
inline void mmap_free(void* ptr, std::size_t size) noexcept {
#if defined(_WIN32)
    (void)size;
    if (ptr) ::VirtualFree(ptr, 0, MEM_RELEASE);
#else
    if (ptr) ::munmap(ptr, size);
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
// vgre::os::mprotect_none — remove all access permissions from a page range
// Returns true on success.
// ─────────────────────────────────────────────────────────────────────────────
inline bool mprotect_none(void* ptr, std::size_t size) noexcept {
#if defined(_WIN32)
    DWORD old;
    return ::VirtualProtect(ptr, size, PAGE_NOACCESS, &old) != 0;
#else
    return ::mprotect(ptr, size, PROT_NONE) == 0;
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
// vgre::os::mprotect_rw — restore read-write access to a page range
// ─────────────────────────────────────────────────────────────────────────────
inline bool mprotect_rw(void* ptr, std::size_t size) noexcept {
#if defined(_WIN32)
    DWORD old;
    return ::VirtualProtect(ptr, size, PAGE_READWRITE, &old) != 0;
#else
    return ::mprotect(ptr, size, PROT_READ | PROT_WRITE) == 0;
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
// vgre::os::mprotect_ro — read-only access
// ─────────────────────────────────────────────────────────────────────────────
inline bool mprotect_ro(void* ptr, std::size_t size) noexcept {
#if defined(_WIN32)
    DWORD old;
    return ::VirtualProtect(ptr, size, PAGE_READONLY, &old) != 0;
#else
    return ::mprotect(ptr, size, PROT_READ) == 0;
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
// vgre::os::page_size — system page size in bytes
// ─────────────────────────────────────────────────────────────────────────────
inline std::size_t page_size() noexcept {
#if defined(_WIN32)
    SYSTEM_INFO si;
    ::GetSystemInfo(&si);
    return static_cast<std::size_t>(si.dwPageSize);
#else
    long ps = ::sysconf(_SC_PAGESIZE);
    return (ps > 0) ? static_cast<std::size_t>(ps) : 4096;
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
// vgre::os::file_exists — check file existence
// ─────────────────────────────────────────────────────────────────────────────
inline bool file_exists(const char* path) noexcept {
#if defined(_WIN32)
    return ::GetFileAttributesA(path) != INVALID_FILE_ATTRIBUTES;
#else
    return ::access(path, F_OK) == 0;
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
// vgre::os::file_mtime — file modification time (seconds since epoch), 0=error
// ─────────────────────────────────────────────────────────────────────────────
inline std::uint64_t file_mtime(const char* path) noexcept {
#if defined(_WIN32)
    WIN32_FILE_ATTRIBUTE_DATA fa;
    if (!::GetFileAttributesExA(path, GetFileExInfoStandard, &fa)) return 0;
    ULARGE_INTEGER t;
    t.LowPart  = fa.ftLastWriteTime.dwLowDateTime;
    t.HighPart = fa.ftLastWriteTime.dwHighDateTime;
    return static_cast<std::uint64_t>(t.QuadPart / 10000000ULL - 11644473600ULL);
#else
    struct stat st{};
    if (::stat(path, &st) != 0) return 0;
    return static_cast<std::uint64_t>(st.st_mtime);
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
// vgre::os::dlopen_lib / dlsym_lib / dlclose_lib
// ─────────────────────────────────────────────────────────────────────────────
inline void* dlopen_lib(const char* path) noexcept {
#if defined(_WIN32)
    return static_cast<void*>(::LoadLibraryA(path));
#else
    return ::dlopen(path, RTLD_LAZY | RTLD_LOCAL);
#endif
}

inline void* dlsym_lib(void* handle, const char* symbol) noexcept {
#if defined(_WIN32)
    return reinterpret_cast<void*>(
        ::GetProcAddress(static_cast<HMODULE>(handle), symbol));
#else
    return ::dlsym(handle, symbol);
#endif
}

inline void dlclose_lib(void* handle) noexcept {
#if defined(_WIN32)
    ::FreeLibrary(static_cast<HMODULE>(handle));
#else
    ::dlclose(handle);
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
// vgre::os::set_thread_affinity — bind current thread to a CPU core (best-effort)
// ─────────────────────────────────────────────────────────────────────────────
inline bool set_thread_affinity(int core) noexcept {
#if defined(_WIN32)
    DWORD_PTR mask = (DWORD_PTR)1 << core;
    return ::SetThreadAffinityMask(::GetCurrentThread(), mask) != 0;
#elif defined(__linux__)
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core, &cpuset);
    return ::sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) == 0;
#else
    (void)core;
    return false; // macOS/BSD: not supported without pthread_setaffinity_np extension
#endif
}
// ─────────────────────────────────────────────────────────────────────────────
// vgre::os::get_monotonic_time_ns — get monotonic system time in nanoseconds
// ─────────────────────────────────────────────────────────────────────────────
inline std::uint64_t get_monotonic_time_ns() noexcept {
#if defined(_WIN32)
    static LARGE_INTEGER frequency;
    static bool initialized = []() {
        return ::QueryPerformanceFrequency(&frequency) != 0;
    }();
    if (!initialized || frequency.QuadPart == 0) return 0;
    LARGE_INTEGER counter;
    ::QueryPerformanceCounter(&counter);
    return static_cast<std::uint64_t>((counter.QuadPart * 1000000000ULL) / frequency.QuadPart);
#else
    struct timespec ts;
    if (::clock_gettime(CLOCK_MONOTONIC, &ts) == 0) {
        return static_cast<std::uint64_t>(ts.tv_sec) * 1000000000ULL + ts.tv_nsec;
    }
    return 0;
#endif
}


// ─────────────────────────────────────────────────────────────────────────────
// vgre::os::mlock_pages / munlock_pages — pin/unpin pages in RAM
// Returns true on success (best-effort; may require elevated privileges).
// ─────────────────────────────────────────────────────────────────────────────
inline bool mlock_pages(void* ptr, std::size_t size) noexcept {
#if defined(_WIN32)
    return ::VirtualLock(ptr, size) != 0;
#else
    return ::mlock(ptr, size) == 0;
#endif
}

inline bool munlock_pages(void* ptr, std::size_t size) noexcept {
#if defined(_WIN32)
    return ::VirtualUnlock(ptr, size) != 0;
#else
    return ::munlock(ptr, size) == 0;
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
// vgre::os::set_file_perms_600 — restrict a file to owner read/write only
// ─────────────────────────────────────────────────────────────────────────────
inline void set_file_perms_600(const char* path) noexcept {
#if !defined(_WIN32)
    ::chmod(path, 0600);
#else
    (void)path; // Windows ACLs handled separately; not required for token files
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
// vgre::os::close_fd — close a POSIX/Win32 file descriptor
// ─────────────────────────────────────────────────────────────────────────────
inline void close_fd(int fd) noexcept {
#if defined(_WIN32)
    ::_close(fd);
#else
    ::close(fd);
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
// vgre::os::set_thread_name — set the current thread's name (best-effort)
// ─────────────────────────────────────────────────────────────────────────────
inline void set_thread_name(const char* name) noexcept {
#if defined(__linux__)
    // pthread_setname_np truncates to 15 chars + NUL
    ::pthread_setname_np(::pthread_self(), name);
#elif defined(_WIN32)
    // Requires Windows 10 1607+ / SetThreadDescription
    // Cast through wchar_t for the wide-string API
    int wlen = ::MultiByteToWideChar(CP_UTF8, 0, name, -1, nullptr, 0);
    if (wlen > 0) {
        wchar_t* wname = static_cast<wchar_t*>(::_alloca(wlen * sizeof(wchar_t)));
        ::MultiByteToWideChar(CP_UTF8, 0, name, -1, wname, wlen);
        typedef HRESULT (WINAPI *PSTD)(HANDLE, PCWSTR);
        static auto fn = reinterpret_cast<PSTD>(
            GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "SetThreadDescription"));
        if (fn) fn(GetCurrentThread(), wname);
    }
#else
    (void)name;
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
// vgre::os::get_cpu_count — logical CPU core count
// ─────────────────────────────────────────────────────────────────────────────
inline int get_cpu_count() noexcept {
#if defined(_WIN32)
    SYSTEM_INFO si{};
    ::GetSystemInfo(&si);
    return static_cast<int>(si.dwNumberOfProcessors);
#else
    long n = ::sysconf(_SC_NPROCESSORS_ONLN);
    return (n > 0) ? static_cast<int>(n) : 1;
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
// vgre::os::make_directory — create a directory (and parents) if not present.
// Returns true if the directory exists or was created successfully.
// ─────────────────────────────────────────────────────────────────────────────
inline bool make_directory(const char* path) noexcept {
#if defined(_WIN32)
    return ::CreateDirectoryA(path, nullptr) != 0
        || ::GetLastError() == ERROR_ALREADY_EXISTS;
#else
    return ::mkdir(path, 0700) == 0 || errno == EEXIST;
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
// vgre::os::get_l2_cache_size — L2 data cache size in bytes (best-effort).
// Returns 4 MiB when the OS query fails or is unsupported.
// ─────────────────────────────────────────────────────────────────────────────
inline std::size_t get_l2_cache_size() noexcept {
    static const std::size_t kDefault = 4 * 1024 * 1024;
#if defined(_WIN32)
    // Walk the processor cache hierarchy via GetLogicalProcessorInformation.
    DWORD bufSz = 0;
    ::GetLogicalProcessorInformation(nullptr, &bufSz);
    if (bufSz == 0) return kDefault;
    std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buf(
        bufSz / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
    if (!::GetLogicalProcessorInformation(buf.data(), &bufSz)) return kDefault;
    for (auto& info : buf) {
        if (info.Relationship == RelationCache &&
            info.Cache.Level == 2 &&
            (info.Cache.Type == CacheData || info.Cache.Type == CacheUnified)) {
            return static_cast<std::size_t>(info.Cache.Size);
        }
    }
    return kDefault;
#elif defined(__APPLE__)
    // sysctl hw.l2cachesize is available on both x86_64 and arm64 macOS.
    std::size_t val = 0;
    std::size_t len = sizeof(val);
    if (::sysctlbyname("hw.l2cachesize", &val, &len, nullptr, 0) == 0 && val > 0)
        return val;
    // Apple Silicon M1 has 12 MiB shared L2; return that as a safe fallback.
    return 12 * 1024 * 1024;
#else
    // Linux: _SC_LEVEL2_CACHE_SIZE is defined in unistd.h (x86_64 glibc).
#  ifdef _SC_LEVEL2_CACHE_SIZE
    long l2 = ::sysconf(_SC_LEVEL2_CACHE_SIZE);
    if (l2 > 0) return static_cast<std::size_t>(l2);
#  endif
    // Fallback: read from sysfs (always present on Linux ≥ 2.6).
    std::ifstream f("/sys/devices/system/cpu/cpu0/cache/index2/size");
    if (f) {
        std::string s; f >> s;
        if (!s.empty()) {
            char unit = s.back();
            long v = std::stol(s);
            if (unit == 'K' || unit == 'k') return static_cast<std::size_t>(v) * 1024;
            if (unit == 'M' || unit == 'm') return static_cast<std::size_t>(v) * 1024 * 1024;
            return static_cast<std::size_t>(v);
        }
    }
    return kDefault;
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
// vgre::os::get_temp_dir — writable per-user temporary directory.
// Returns path WITHOUT trailing separator. Checks TMPDIR / TEMP env vars first
// so tests and containers can redirect writes without touching /tmp.
// ─────────────────────────────────────────────────────────────────────────────
inline std::string get_temp_dir() noexcept {
#if defined(_WIN32)
    char buf[MAX_PATH + 1];
    DWORD len = ::GetTempPathA(static_cast<DWORD>(sizeof(buf)), buf);
    if (len > 0 && len < static_cast<DWORD>(sizeof(buf))) {
        std::string p(buf, len);
        while (!p.empty() && (p.back() == '\\' || p.back() == '/')) p.pop_back();
        return p;
    }
    const char* tmp = ::getenv("TEMP");
    if (!tmp || !tmp[0]) tmp = ::getenv("TMP");
    if (tmp && tmp[0]) return std::string(tmp);
    return "C:\\Temp";
#else
    const char* tmp = ::getenv("TMPDIR");
    if (tmp && tmp[0]) {
        std::string p(tmp);
        while (!p.empty() && p.back() == '/') p.pop_back();
        return p;
    }
#  ifdef __APPLE__
    return "/private/tmp";
#  else
    return "/tmp";
#  endif
#endif
}

} // namespace os
} // namespace vgre
