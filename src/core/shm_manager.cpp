#include "vgre/core/shm_manager.h"
#include "vgre/common/logger.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>
#include <cerrno>

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
        return VGREResult::ERROR_IO;
    }

    if (create) {
        if (ftruncate(fd_, size) == -1) {
            VGRE_LOG_ERROR("ShmManager", "ftruncate failed: " + std::string(std::strerror(errno)));
            ::close(fd_);
            fd_ = -1;
            return VGREResult::ERROR_IO;
        }
    }

    basePtr_ = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (basePtr_ == MAP_FAILED) {
        VGRE_LOG_ERROR("ShmManager", "mmap failed: " + std::string(std::strerror(errno)));
        ::close(fd_);
        fd_ = -1;
        basePtr_ = nullptr;
        return VGREResult::ERROR_IO;
    }

    name_ = shmName;
    size_ = size;
    VGRE_LOG_DEBUG("ShmManager", "Opened segment " + shmName + " (" + std::to_string(size) + " bytes)");
    return VGREResult::SUCCESS;
}

void ShmManager::close() {
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
}

void ShmManager::unlink(const std::string& name) {
    std::string shmName = name;
    if (shmName.empty() || shmName[0] != '/') {
        shmName = "/" + shmName;
    }
    shm_unlink(shmName.c_str());
}

} // namespace core
} // namespace vgre
