/**
 * @file library_loader.h
 * @brief Cross-platform dynamic library loader RAII wrapper.
 *
 * Wraps platform specific dlopen/LoadLibrary calls with deterministic RAII lifecycles,
 * type-safe symbol retrievals, and support for multi-path probing.
 */

#pragma once

#include "vgre/common/os_backend.h"
#include <initializer_list>
#include <utility>

namespace vgre {
namespace common {

class LibraryLoader {
public:
    LibraryLoader() noexcept : handle_(nullptr) {}

    explicit LibraryLoader(const char* path) noexcept 
        : handle_(vgre::os::dlopen_lib(path)) {}

    ~LibraryLoader() noexcept {
        if (handle_) {
            vgre::os::dlclose_lib(handle_);
        }
    }

    // Non-copyable
    LibraryLoader(const LibraryLoader&) = delete;
    LibraryLoader& operator=(const LibraryLoader&) = delete;

    // Move constructible
    LibraryLoader(LibraryLoader&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }

    // Move assignable
    LibraryLoader& operator=(LibraryLoader&& other) noexcept {
        if (this != &other) {
            if (handle_) {
                vgre::os::dlclose_lib(handle_);
            }
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    bool isLoaded() const noexcept {
        return handle_ != nullptr;
    }

    void* getSymbol(const char* name) const noexcept {
        if (!handle_) return nullptr;
        return vgre::os::dlsym_lib(handle_, name);
    }

    template<typename Func>
    Func getFunc(const char* name) const noexcept {
        return reinterpret_cast<Func>(getSymbol(name));
    }

    // Probe multiple paths, returning the first successfully loaded library
    static LibraryLoader tryLoad(std::initializer_list<const char*> paths) noexcept {
        for (const char* path : paths) {
            LibraryLoader loader(path);
            if (loader.isLoaded()) {
                return loader;
            }
        }
        return LibraryLoader();
    }

private:
    void* handle_ = nullptr;
};

} // namespace common
} // namespace vgre
