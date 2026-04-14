#include "vgre/core/shm_manager.h"
#include "vgre/common/logger.h"

#include <cstring>
#include <cerrno>

#if defined(_WIN32)
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace vgre {
namespace core {

ShmManager::ShmManager() {}

ShmManager::~ShmManager() {
    close();
    if (!name_.empty()) {
        unlink(name_);
    }
}

VGREResult ShmManager::open(const std::string& name, size_t size, bool create) {
    close();

#if defined(_WIN32)
    // Windows implementation using CreateFileMapping/MapViewOfFile
    
    // Ensure name is valid for Windows (no leading slash)
    std::string mappingName = name;
    if (!mappingName.empty() && mappingName[0] == '/') {
        mappingName = mappingName.substr(1);
    }
    if (mappingName.empty()) {
        VGRE_LOG_ERROR("ShmManager", "Invalid shared memory name (empty)");
        return VGREResult::ERR_INVALID_VALUE;
    }
    
    // Prefix with "Local\\" for local session scope
    std::string fullName = "Local\\vgre_shm_" + mappingName;
    
    DWORD dwCreationDisposition = create ? CREATE_ALWAYS : OPEN_EXISTING;
    DWORD flProtect = PAGE_READWRITE;
    DWORD dwDesiredAccess = FILE_MAP_ALL_ACCESS;
    
    // Create or open file mapping
    HANDLE hMapFile = CreateFileMappingA(
        INVALID_HANDLE_VALUE,    // Use paging file
        NULL,                    // Default security
        flProtect,               // Read/write access
        0,                       // High-order DWORD of size
        static_cast<DWORD>(size), // Low-order DWORD of size
        fullName.c_str()         // Name of mapping object
    );
    
    if (hMapFile == NULL) {
        DWORD err = GetLastError();
        if (!create && err == ERROR_FILE_NOT_FOUND) {
            VGRE_LOG_ERROR("ShmManager", "Shared memory segment not found: " + fullName);
            return VGREResult::ERR_NOT_FOUND;
        }
        VGRE_LOG_ERROR("ShmManager", "CreateFileMapping failed for " + fullName + 
                       " (error: " + std::to_string(err) + ")");
        return VGREResult::ERR_IO;
    }
    
    // Check if we opened an existing mapping when we wanted to create
    if (create && GetLastError() == ERROR_ALREADY_EXISTS) {
        VGRE_LOG_WARN("ShmManager", "Shared memory segment already exists: " + fullName);
    }
    
    // Map view of file
    void* pBuf = MapViewOfFile(
        hMapFile,           // Handle to map object
        dwDesiredAccess,    // Read/write permission
        0,                  // High-order DWORD of offset
        0,                  // Low-order DWORD of offset
        size                // Number of bytes to map
    );
    
    if (pBuf == NULL) {
        DWORD err = GetLastError();
        VGRE_LOG_ERROR("ShmManager", "MapViewOfFile failed (error: " + std::to_string(err) + ")");
        CloseHandle(hMapFile);
        return VGREResult::ERR_IO;
    }
    
    // Store handles
    fd_ = reinterpret_cast<intptr_t>(hMapFile); // Store HANDLE as intptr_t (platform-specific)
    basePtr_ = pBuf;
    name_ = fullName;
    size_ = size;
    
    VGRE_LOG_DEBUG("ShmManager", "Opened Windows shared memory segment " + fullName + 
                   " (" + std::to_string(size) + " bytes)");
    return VGREResult::SUCCESS;
    
#else
    // Linux/macOS implementation using POSIX shared memory
    int flags = O_RDWR;
    if (create) {
        flags |= O_CREAT;
    }

    // Ensure name starts with /
    std::string shmName = name;
    if (shmName.empty() || shmName[0] != '/') {
        shmName = "/" + shmName;
    }

    fd_ = shm_open(shmName.c_str(), flags, 0666);
    if (fd_ == -1) {
        VGRE_LOG_ERROR("ShmManager", "shm_open failed for " + shmName + ": " + std::strerror(errno));
        return VGREResult::ERR_IO;
    }

    if (create) {
        if (ftruncate(fd_, size) == -1) {
            VGRE_LOG_ERROR("ShmManager", "ftruncate failed: " + std::string(std::strerror(errno)));
            ::close(fd_);
            fd_ = -1;
            return VGREResult::ERR_IO;
        }
    }

    basePtr_ = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (basePtr_ == MAP_FAILED) {
        VGRE_LOG_ERROR("ShmManager", "mmap failed: " + std::string(std::strerror(errno)));
        ::close(fd_);
        fd_ = -1;
        basePtr_ = nullptr;
        return VGREResult::ERR_IO;
    }

    name_ = shmName;
    size_ = size;
    VGRE_LOG_DEBUG("ShmManager", "Opened segment " + shmName + " (" + std::to_string(size) + " bytes)");
    return VGREResult::SUCCESS;
#endif
}

void ShmManager::close() {
#if defined(_WIN32)
    if (basePtr_) {
        UnmapViewOfFile(basePtr_);
        basePtr_ = nullptr;
    }
    if (fd_ != -1) {
        HANDLE hMapFile = reinterpret_cast<HANDLE>(fd_);
        CloseHandle(hMapFile);
        fd_ = -1;
    }
    name_.clear();
    size_ = 0;
#else
    if (basePtr_) {
        munmap(basePtr_, size_);
        basePtr_ = nullptr;
    }
    if (fd_ != -1) {
        ::close(fd_);
        fd_ = -1;
    }
    name_.clear();
    size_ = 0;
#endif
}

void ShmManager::unlink(const std::string& name) {
#if defined(_WIN32)
    // On Windows, shared memory is automatically cleaned up when all handles are closed
    // No explicit unlink needed, but we log for consistency
    VGRE_LOG_DEBUG("ShmManager", "Windows shared memory cleanup (automatic): " + name);
    (void)name;
#else
    std::string shmName = name;
    if (shmName.empty() || shmName[0] != '/') {
        shmName = "/" + shmName;
    }
    shm_unlink(shmName.c_str());
#endif
}

} // namespace core
} // namespace vgre
