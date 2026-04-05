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
    (void)name;
    (void)size;
    (void)create;
    VGRE_LOG_WARN("ShmManager",
                  "Shared memory manager is not implemented on Windows yet");
    return VGREResult::ERR_NOT_SUPPORTED;
#else
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
    basePtr_ = nullptr;
    fd_ = -1;
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
