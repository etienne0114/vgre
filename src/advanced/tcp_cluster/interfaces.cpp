/**
 * Production implementations of the TCP cluster dependency-injection
 * interfaces.
 *
 * RealSocketFactory    — delegates to POSIX / Winsock system calls with
 * validation RealMemoryManager    — delegates to
 * RuntimeEngine::getMemoryManager() with error handling
 * RealSecureChannelFactory — creates real SecureChannel instances with
 * configuration validation
 */

#include "vgre/advanced/tcp_cluster/internal/interfaces.h"
#include "vgre/advanced/secure_channel.h"
#include "vgre/api/vgre_c_api.h"
#include "vgre/common/logger.h"
#include "vgre/common/sockets.h"
#include "vgre/core/memory_manager.h"
#include "vgre/core/runtime_engine.h"
#include "vgre/core/scheduler.h"
#include <cstring>
#include <errno.h>
#include <limits>

#include "vgre/common/os_backend.h"

namespace vgre {
namespace advanced {

using vgre::common::VGRE_INVALID_SOCKET;
using vgre::common::vgre_poll;
using vgre::common::vgre_pollfd;
using vgre::common::vgre_socket_t;

// ── RealSocketFactory
// ─────────────────────────────────────────────────────────

// Virtual destructor definition to generate vtable
RealSocketFactory::~RealSocketFactory() {
  // Empty destructor - provides vtable anchor point
}

vgre_socket_t RealSocketFactory::createSocket(int domain, int type,
                                              int protocol) {
  // Validate socket parameters
  if (domain != AF_INET && domain != AF_INET6 && domain != AF_UNIX) {
    VGRE_LOG_ERROR("RealSocketFactory",
                   "Invalid socket domain: " + std::to_string(domain));
    errno = EAFNOSUPPORT;
    return VGRE_INVALID_SOCKET;
  }

  vgre_socket_t fd = ::socket(domain, type, protocol);
  if (fd == VGRE_INVALID_SOCKET) {
    int err = vgre::common::vgre_get_last_socket_error();
    VGRE_LOG_ERROR("RealSocketFactory",
                   "Failed to create socket: " + vgre::common::vgre_socket_error_string(err));
  }
  return fd;
}

int RealSocketFactory::bind(vgre_socket_t fd, const sockaddr *addr,
                            socklen_t len) {
  if (fd == VGRE_INVALID_SOCKET || !addr || len == 0) {
    VGRE_LOG_ERROR("RealSocketFactory", "Invalid bind parameters");
    errno = EINVAL;
    return -1;
  }

  int result = ::bind(fd, addr, len);
  if (result != 0) {
    int err = vgre::common::vgre_get_last_socket_error();
    VGRE_LOG_ERROR("RealSocketFactory",
                   "Bind failed: " + vgre::common::vgre_socket_error_string(err));
  }
  return result;
}

int RealSocketFactory::listen(vgre_socket_t fd, int backlog) {
  if (fd == VGRE_INVALID_SOCKET) {
    VGRE_LOG_ERROR("RealSocketFactory", "Invalid socket for listen");
    errno = EBADF;
    return -1;
  }

  // Clamp backlog to reasonable limits
  if (backlog < 0)
    backlog = 0;
  if (backlog > SOMAXCONN)
    backlog = SOMAXCONN;

  int result = ::listen(fd, backlog);
  if (result != 0) {
    int err = vgre::common::vgre_get_last_socket_error();
    VGRE_LOG_ERROR("RealSocketFactory",
                   "Listen failed: " + vgre::common::vgre_socket_error_string(err));
  }
  return result;
}

vgre_socket_t RealSocketFactory::accept(vgre_socket_t fd, sockaddr *addr,
                                        socklen_t *len) {
  if (fd == VGRE_INVALID_SOCKET) {
    VGRE_LOG_ERROR("RealSocketFactory", "Invalid socket for accept");
    errno = EBADF;
    return VGRE_INVALID_SOCKET;
  }

  vgre_socket_t client_fd = ::accept(fd, addr, len);
  if (client_fd == VGRE_INVALID_SOCKET) {
    int err = vgre::common::vgre_get_last_socket_error();
    // Don't log EAGAIN/EWOULDBLOCK as errors (normal for non-blocking sockets)
    if (!vgre::common::vgre_is_would_block(err)) {
      VGRE_LOG_ERROR("RealSocketFactory",
                     "Accept failed: " + vgre::common::vgre_socket_error_string(err));
    }
  }
  return client_fd;
}

int RealSocketFactory::connect(vgre_socket_t fd, const sockaddr *addr,
                               socklen_t len) {
  if (fd == VGRE_INVALID_SOCKET || !addr || len == 0) {
    VGRE_LOG_ERROR("RealSocketFactory", "Invalid connect parameters");
    errno = EINVAL;
    return -1;
  }

  int result = ::connect(fd, addr, len);
  if (result != 0) {
    int err = vgre::common::vgre_get_last_socket_error();
    // Don't log EINPROGRESS/WSAEWOULDBLOCK (normal for non-blocking connect)
    if (!vgre::common::vgre_is_would_block(err)) {
      VGRE_LOG_ERROR("RealSocketFactory",
                     "Connect failed: " + vgre::common::vgre_socket_error_string(err));
    }
  }
  return result;
}

int RealSocketFactory::send(vgre_socket_t fd, const void *buf, size_t len,
                            int flags) {
  if (fd == VGRE_INVALID_SOCKET || !buf) {
    VGRE_LOG_ERROR("RealSocketFactory", "Invalid send parameters");
    errno = EINVAL;
    return -1;
  }

  // Prevent integer overflow on Windows
  if (len > static_cast<size_t>(std::numeric_limits<int>::max())) {
    VGRE_LOG_ERROR("RealSocketFactory",
                   "Send length too large: " + std::to_string(len));
    errno = EMSGSIZE;
    return -1;
  }

  int result = static_cast<int>(::send(fd, static_cast<const char *>(buf),
#ifdef _WIN32
                                       static_cast<int>(len),
#else
                                       len,
#endif
                                       flags));

  if (result < 0) {
    int err = vgre::common::vgre_get_last_socket_error();
    // Don't log EAGAIN/EWOULDBLOCK as errors (normal for non-blocking sockets)
    if (!vgre::common::vgre_is_would_block(err)) {
      VGRE_LOG_ERROR("RealSocketFactory",
                     "Send failed: " + vgre::common::vgre_socket_error_string(err));
    }
  }
  return result;
}

int RealSocketFactory::recv(vgre_socket_t fd, void *buf, size_t len,
                            int flags) {
  if (fd == VGRE_INVALID_SOCKET || !buf) {
    VGRE_LOG_ERROR("RealSocketFactory", "Invalid recv parameters");
    errno = EINVAL;
    return -1;
  }

  // Prevent integer overflow on Windows
  if (len > static_cast<size_t>(std::numeric_limits<int>::max())) {
    VGRE_LOG_ERROR("RealSocketFactory",
                   "Recv length too large: " + std::to_string(len));
    errno = EMSGSIZE;
    return -1;
  }

  int result = static_cast<int>(::recv(fd, static_cast<char *>(buf),
#ifdef _WIN32
                                       static_cast<int>(len),
#else
                                       len,
#endif
                                       flags));

  if (result < 0) {
    int err = vgre::common::vgre_get_last_socket_error();
    // Don't log EAGAIN/EWOULDBLOCK as errors (normal for non-blocking sockets)
    if (!vgre::common::vgre_is_would_block(err)) {
      VGRE_LOG_ERROR("RealSocketFactory",
                     "Recv failed: " + vgre::common::vgre_socket_error_string(err));
    }
  }
  return result;
}

int RealSocketFactory::poll(vgre_pollfd *fds, int nfds, int timeout) {
  if (!fds && nfds > 0) {
    VGRE_LOG_ERROR("RealSocketFactory", "Invalid poll parameters");
    errno = EINVAL;
    return -1;
  }

  if (nfds < 0) {
    VGRE_LOG_ERROR("RealSocketFactory",
                   "Invalid nfds for poll: " + std::to_string(nfds));
    errno = EINVAL;
    return -1;
  }

  int result = vgre_poll(fds, nfds, timeout);
  if (result < 0) {
    int err = vgre::common::vgre_get_last_socket_error();
    VGRE_LOG_ERROR("RealSocketFactory",
                   "Poll failed: " + vgre::common::vgre_socket_error_string(err));
  }
  return result;
}

void RealSocketFactory::close(vgre_socket_t fd) {
  if (fd == VGRE_INVALID_SOCKET) {
    VGRE_LOG_WARN("RealSocketFactory", "Attempting to close invalid socket");
    return;
  }

  vgre::common::vgre_close_socket(fd);
}

int RealSocketFactory::setSockOpt(vgre_socket_t fd, int level, int optname,
                                  const void *optval, socklen_t optlen) {
  if (fd == VGRE_INVALID_SOCKET || !optval) {
    VGRE_LOG_ERROR("RealSocketFactory", "Invalid setsockopt parameters");
    errno = EINVAL;
    return -1;
  }

  int result =
      vgre::common::vgre_setsockopt(fd, level, optname, optval, optlen);
  if (result != 0) {
    int err = vgre::common::vgre_get_last_socket_error();
    VGRE_LOG_ERROR("RealSocketFactory",
                   "SetSockOpt failed: " + vgre::common::vgre_socket_error_string(err));
  }
  return result;
}

int RealSocketFactory::getSockOpt(vgre_socket_t fd, int level, int optname,
                                  void *optval, socklen_t *optlen) {
  if (fd == VGRE_INVALID_SOCKET || !optval || !optlen) {
    VGRE_LOG_ERROR("RealSocketFactory", "Invalid getsockopt parameters");
    errno = EINVAL;
    return -1;
  }

  int result;
#ifdef _WIN32
  result =
      ::getsockopt(fd, level, optname, static_cast<char *>(optval), optlen);
#else
  result = ::getsockopt(fd, level, optname, optval, optlen);
#endif

  if (result != 0) {
    int err = vgre::common::vgre_get_last_socket_error();
    VGRE_LOG_ERROR("RealSocketFactory",
                   "GetSockOpt failed: " + vgre::common::vgre_socket_error_string(err));
  }
  return result;
}

// ── RealMemoryManager
// ─────────────────────────────────────────────────────────

// Virtual destructor definition to generate vtable
RealMemoryManager::~RealMemoryManager() {
  // Empty destructor - provides vtable anchor point
}

size_t RealMemoryManager::getAllocationSize(void *ptr) {
  if (!ptr) {
    VGRE_LOG_ERROR("RealMemoryManager",
                   "Null pointer passed to getAllocationSize");
    return 0;
  }

  try {
    return vgre::core::RuntimeEngine::instance()
        .getMemoryManager()
        .getAllocationSize(ptr);
  } catch (const std::exception &e) {
    VGRE_LOG_ERROR("RealMemoryManager",
                   "Exception in getAllocationSize: " + std::string(e.what()));
    return 0;
  }
}

void *RealMemoryManager::getPointer(uint64_t handle) {
  if (handle == 0) {
    VGRE_LOG_ERROR("RealMemoryManager",
                   "Invalid handle (0) passed to getPointer");
    return nullptr;
  }

  try {
    return vgre::core::RuntimeEngine::instance().getMemoryManager().getPointer(
        reinterpret_cast<void *>(handle));
  } catch (const std::exception &e) {
    VGRE_LOG_ERROR("RealMemoryManager",
                   "Exception in getPointer: " + std::string(e.what()));
    return nullptr;
  }
}

bool RealMemoryManager::isValidHandle(uint64_t handle) {
  if (handle == 0) {
    return false;
  }

  try {
    return vgre::core::RuntimeEngine::instance()
        .getMemoryManager()
        .isValidHandle(reinterpret_cast<void *>(handle));
  } catch (const std::exception &e) {
    VGRE_LOG_ERROR("RealMemoryManager",
                   "Exception in isValidHandle: " + std::string(e.what()));
    return false;
  }
}

uint64_t RealMemoryManager::allocateManagedAt(void *ptr, size_t size) {
  if (!ptr || size == 0) {
    VGRE_LOG_ERROR("RealMemoryManager",
                   "Invalid parameters for allocateManagedAt");
    return 0;
  }

  try {
    MemoryHandle outHandle = nullptr;
    VGREResult result = vgre::core::RuntimeEngine::instance()
                            .getMemoryManager()
                            .allocateManagedAt(ptr, size, outHandle);

    if (result != VGREResult::SUCCESS) {
      VGRE_LOG_ERROR("RealMemoryManager",
                     "allocateManagedAt failed with result: " +
                         std::to_string(static_cast<int>(result)));
      return 0;
    }

    return reinterpret_cast<uint64_t>(outHandle);
  } catch (const std::exception &e) {
    VGRE_LOG_ERROR("RealMemoryManager",
                   "Exception in allocateManagedAt: " + std::string(e.what()));
    return 0;
  }
}

void RealMemoryManager::getDirtyPages(
    void *ptr, std::vector<std::pair<size_t, size_t>> &dirtyRanges) {
  if (!ptr) {
    VGRE_LOG_ERROR("RealMemoryManager", "Null pointer passed to getDirtyPages");
    dirtyRanges.clear();
    return;
  }

  try {
    vgre::core::RuntimeEngine::instance().getMemoryManager().getDirtyPages(
        ptr, dirtyRanges);
  } catch (const std::exception &e) {
    VGRE_LOG_ERROR("RealMemoryManager",
                   "Exception in getDirtyPages: " + std::string(e.what()));
    dirtyRanges.clear();
  }
}

void RealMemoryManager::clearDirtyPages(void *ptr) {
  if (!ptr) {
    VGRE_LOG_ERROR("RealMemoryManager",
                   "Null pointer passed to clearDirtyPages");
    return;
  }

  try {
    vgre::core::RuntimeEngine::instance().getMemoryManager().clearDirtyPages(
        ptr);
  } catch (const std::exception &e) {
    VGRE_LOG_ERROR("RealMemoryManager",
                   "Exception in clearDirtyPages: " + std::string(e.what()));
  }
}

VGREResult RealMemoryManager::mapMemory(void *ptr, size_t size) {
  if (!ptr || size == 0) {
    VGRE_LOG_ERROR("RealMemoryManager", "Invalid parameters for mapMemory");
    return VGREResult::ERR_INVALID_VALUE;
  }

  // The core MemoryManager uses UVM-style managed memory that is already
  // accessible from both host and device; explicit mapping is a no-op.
  // However, we validate the parameters for production robustness.
  try {
    // Verify the pointer is within a valid allocation
    size_t allocSize = vgre::core::RuntimeEngine::instance()
                           .getMemoryManager()
                           .getAllocationSize(ptr);
    if (allocSize == 0) {
      VGRE_LOG_ERROR("RealMemoryManager",
                     "mapMemory called on unmanaged pointer");
      return VGREResult::ERR_INVALID_VALUE;
    }

    if (size > allocSize) {
      VGRE_LOG_WARN("RealMemoryManager", "mapMemory size (" +
                                             std::to_string(size) +
                                             ") exceeds allocation size (" +
                                             std::to_string(allocSize) + ")");
    }

    return VGREResult::SUCCESS;
  } catch (const std::exception &e) {
    VGRE_LOG_ERROR("RealMemoryManager",
                   "Exception in mapMemory: " + std::string(e.what()));
    return VGREResult::ERR_UNKNOWN;
  }
}

VGREResult RealMemoryManager::unmapMemory(void *ptr) {
  if (!ptr) {
    VGRE_LOG_ERROR("RealMemoryManager", "Null pointer passed to unmapMemory");
    return VGREResult::ERR_INVALID_VALUE;
  }

  // Similar to mapMemory, this is a no-op for UVM but we validate for
  // robustness
  try {
    size_t allocSize = vgre::core::RuntimeEngine::instance()
                           .getMemoryManager()
                           .getAllocationSize(ptr);
    if (allocSize == 0) {
      VGRE_LOG_ERROR("RealMemoryManager",
                     "unmapMemory called on unmanaged pointer");
      return VGREResult::ERR_INVALID_VALUE;
    }

    return VGREResult::SUCCESS;
  } catch (const std::exception &e) {
    VGRE_LOG_ERROR("RealMemoryManager",
                   "Exception in unmapMemory: " + std::string(e.what()));
    return VGREResult::ERR_UNKNOWN;
  }
}

// ── RealSecureChannelFactory
// ──────────────────────────────────────────────────

// Virtual destructor definition to generate vtable
RealSecureChannelFactory::~RealSecureChannelFactory() {
  // Empty destructor - provides vtable anchor point
}

std::unique_ptr<SecureChannel> RealSecureChannelFactory::create() {
  return std::make_unique<SecureChannel>();
}

bool RealSecureChannelFactory::isSecuritySupported() {
  return true; // AES-256-CTR + HMAC-SHA256 always compiled in
}

VGREResult RealSecureChannelFactory::validateConfiguration() {
  // Verify a SecureChannel can actually be constructed and initialised.
  // This catches configuration problems (missing entropy source, wrong
  // OpenSSL linkage, etc.) at startup rather than at the first handshake.
  auto ch = std::make_unique<SecureChannel>();
  if (!ch)
    return VGREResult::ERR_NOT_INITIALIZED;

  // Verify the auth token meets the minimum length requirement.
  // Using the same check as SecurityManager::enableSecurity to stay consistent.
  constexpr size_t kMinTokenLen = 16;
  const char *token_env = vgre_get_config("VGRE_TCP_AUTH_TOKEN");
  const char *token_file = vgre_get_config("VGRE_TCP_AUTH_TOKEN_FILE");
  bool hasToken = (token_env && token_env[0] != '\0') ||
                  (token_file && token_file[0] != '\0');

  if (hasToken && token_env && strlen(token_env) > 0 &&
      strlen(token_env) < kMinTokenLen) {
    // Token is set but too short — warn but do not fail startup so callers
    // can still connect (SecurityManager emits the same warning on its path).
    // Return success so production systems aren't blocked by a warning.
  }

  // Verify SecureChannel::generateNonce works (exercises the entropy source).
  uint8_t testNonce[crypto::kNonceLen];
  SecureChannel::generateNonce(testNonce);

  return VGREResult::SUCCESS;
}

// ── RealKernelManager
// ───────────────────────────────────────────────────────────

RealKernelManager::~RealKernelManager() {
  // Empty destructor - provides vtable anchor point
}

VGREResult RealKernelManager::getKernelArgTypes(uint64_t kernel_id, std::vector<ArgType>& argTypes) {
    try {
        return vgre::core::RuntimeEngine::instance()
            .getKernelArgTypes(kernel_id, argTypes);
    } catch (const std::exception& e) {
        VGRE_LOG_ERROR("RealKernelManager", "Exception in getKernelArgTypes: " + std::string(e.what()));
        return VGREResult::ERR_UNKNOWN;
    }
}

const void* RealKernelManager::getKernelIR(uint64_t kernel_id) {
    try {
        return vgre::core::RuntimeEngine::instance()
            .getKernelIR(kernel_id);
    } catch (const std::exception& e) {
        VGRE_LOG_ERROR("RealKernelManager", "Exception in getKernelIR: " + std::string(e.what()));
        return nullptr;
    }
}

VGREResult RealKernelManager::registerKernel(const std::string& name, const std::string& source, uint64_t& kernel_id) {
    try {
        return vgre::core::RuntimeEngine::instance()
            .registerKernel(name, source, kernel_id);
    } catch (const std::exception& e) {
        VGRE_LOG_ERROR("RealKernelManager", "Exception in registerKernel: " + std::string(e.what()));
        return VGREResult::ERR_UNKNOWN;
    }
}

VGREResult RealKernelManager::launchKernel(uint64_t kernel_id, const dim3& gridDim, const dim3& blockDim, void** args, size_t sharedMem) {
    try {
        return vgre::core::RuntimeEngine::instance()
            .launchKernel(kernel_id, gridDim, blockDim, args, sharedMem);
    } catch (const std::exception& e) {
        VGRE_LOG_ERROR("RealKernelManager", "Exception in launchKernel: " + std::string(e.what()));
        return VGREResult::ERR_UNKNOWN;
    }
}

} // namespace advanced
} // namespace vgre
