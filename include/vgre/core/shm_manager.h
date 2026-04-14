#pragma once

#include "vgre/common/error_codes.h"
#include <string>
#include <cstdint>

namespace vgre {
namespace core {

/**
 * @brief Manages POSIX Shared Memory segments for inter-process communication.
 */
class ShmManager {
public:
    ShmManager();
    ~ShmManager();

    /**
     * @brief Creates or opens a shared memory segment.
     */
    VGREResult open(const std::string& name, size_t size, bool create = true);

    /**
     * @brief Closes and unmaps the current segment.
     */
    void close();

    /**
     * @brief Unlinks the segment from the system (deletes it).
     */
    static void unlink(const std::string& name);

    /**
     * @brief Returns the base pointer of the mapped segment.
     */
    void* getBasePtr() const { return basePtr_; }

    /**
     * @brief Returns the size of the segment.
     */
    size_t getSize() const { return size_; }

    /**
     * @brief Returns the name of the segment.
     */
    const std::string& getName() const { return name_; }

private:
    std::string name_;
    size_t size_ = 0;
    intptr_t fd_ = -1;
    void* basePtr_ = nullptr;
};

} // namespace core
} // namespace vgre
