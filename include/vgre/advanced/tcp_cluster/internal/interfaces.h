#pragma once

#include "vgre/advanced/secure_channel.h"
#include "vgre/common/sockets.h"
#include "vgre/common/types.h"
#include <cstdint>
#include <memory>
#include <vector>

namespace vgre {
namespace advanced {

/**
 * @brief Interface abstraction for socket operations to enable dependency
 * injection and testing.
 *
 * This interface abstracts all socket operations used by the TCP cluster
 * manager, allowing for mock implementations in unit tests and fake
 * implementations in integration tests.
 */
class ISocketFactory {
public:
  virtual ~ISocketFactory() = default;

  // Socket creation and management
  virtual vgre_socket_t createSocket(int domain, int type, int protocol) = 0;
  virtual int bind(vgre_socket_t fd, const sockaddr *addr, socklen_t len) = 0;
  virtual int listen(vgre_socket_t fd, int backlog) = 0;
  virtual vgre_socket_t accept(vgre_socket_t fd, sockaddr *addr,
                               socklen_t *len) = 0;
  virtual int connect(vgre_socket_t fd, const sockaddr *addr,
                      socklen_t len) = 0;

  // Data transfer
  virtual int send(vgre_socket_t fd, const void *buf, size_t len,
                   int flags) = 0;
  virtual int recv(vgre_socket_t fd, void *buf, size_t len, int flags) = 0;

  // Socket control
  virtual int poll(vgre::common::vgre_pollfd *fds, int nfds, int timeout) = 0;
  virtual void close(vgre_socket_t fd) = 0;

  // Socket options
  virtual int setSockOpt(vgre_socket_t fd, int level, int optname,
                         const void *optval, socklen_t optlen) = 0;
  virtual int getSockOpt(vgre_socket_t fd, int level, int optname, void *optval,
                         socklen_t *optlen) = 0;
};

/**
 * @brief Interface abstraction for memory management operations to enable
 * dependency injection and testing.
 *
 * This interface abstracts memory management operations used by the TCP cluster
 * manager, allowing for mock implementations that can simulate different memory
 * scenarios in tests.
 */
class IMemoryManager {
public:
  virtual ~IMemoryManager() = default;

  // Memory allocation information
  virtual size_t getAllocationSize(void *ptr) = 0;
  virtual void *getPointer(uint64_t handle) = 0;
  virtual bool isValidHandle(uint64_t handle) = 0;
  virtual uint64_t allocateManagedAt(void *ptr, size_t size) = 0;

  // Dirty page tracking for delta-sync
  virtual void
  getDirtyPages(void *ptr,
                std::vector<std::pair<size_t, size_t>> &dirtyRanges) = 0;
  virtual void clearDirtyPages(void *ptr) = 0;

  // Memory mapping and synchronization
  virtual VGREResult mapMemory(void *ptr, size_t size) = 0;
  virtual VGREResult unmapMemory(void *ptr) = 0;
};

/**
 * @brief Interface abstraction for secure channel creation to enable dependency
 * injection and testing.
 *
 * This interface abstracts secure channel creation used by the TCP cluster
 * manager, allowing for mock implementations that can simulate different
 * security scenarios in tests.
 */
class ISecureChannelFactory {
public:
  virtual ~ISecureChannelFactory() = default;

  // Secure channel creation
  virtual std::unique_ptr<SecureChannel> create() = 0;

  // Security configuration
  virtual bool isSecuritySupported() = 0;
  virtual VGREResult validateConfiguration() = 0;
};

/**
 * @brief Interface abstraction for kernel management operations to enable
 * dependency injection and testing.
 */
class IKernelManager {
public:
  virtual ~IKernelManager() = default;

  virtual VGREResult getKernelArgTypes(uint64_t kernel_id, std::vector<ArgType>& argTypes) = 0;
  virtual const void* getKernelIR(uint64_t kernel_id) = 0;
  virtual VGREResult registerKernel(const std::string& name, const std::string& source, uint64_t& kernel_id) = 0;
  virtual VGREResult launchKernel(uint64_t kernel_id, const dim3& gridDim, const dim3& blockDim, void** args, size_t sharedMem) = 0;
};

/**
 * @brief Real implementation of ISocketFactory using actual system calls.
 *
 * This is the production implementation that delegates to actual socket system
 * calls. Used by the singleton instance() method for normal operation.
 */
class RealSocketFactory : public ISocketFactory {
public:
  RealSocketFactory() = default;
  virtual ~RealSocketFactory();

  // Socket creation and management
  vgre_socket_t createSocket(int domain, int type, int protocol) override;
  int bind(vgre_socket_t fd, const sockaddr *addr, socklen_t len) override;
  int listen(vgre_socket_t fd, int backlog) override;
  vgre_socket_t accept(vgre_socket_t fd, sockaddr *addr,
                       socklen_t *len) override;
  int connect(vgre_socket_t fd, const sockaddr *addr, socklen_t len) override;

  // Data transfer
  int send(vgre_socket_t fd, const void *buf, size_t len, int flags) override;
  int recv(vgre_socket_t fd, void *buf, size_t len, int flags) override;

  // Socket control
  int poll(vgre::common::vgre_pollfd *fds, int nfds, int timeout) override;
  void close(vgre_socket_t fd) override;

  // Socket options
  int setSockOpt(vgre_socket_t fd, int level, int optname, const void *optval,
                 socklen_t optlen) override;
  int getSockOpt(vgre_socket_t fd, int level, int optname, void *optval,
                 socklen_t *optlen) override;
};

/**
 * @brief Real implementation of IMemoryManager delegating to RuntimeEngine.
 *
 * This is the production implementation that delegates to the actual memory
 * manager from the RuntimeEngine. Used by the singleton instance() method for
 * normal operation.
 */
class RealMemoryManager : public IMemoryManager {
public:
  RealMemoryManager() = default;
  virtual ~RealMemoryManager();

  // Memory allocation information
  size_t getAllocationSize(void *ptr) override;
  void *getPointer(uint64_t handle) override;
  bool isValidHandle(uint64_t handle) override;
  uint64_t allocateManagedAt(void *ptr, size_t size) override;

  // Dirty page tracking for delta-sync
  void
  getDirtyPages(void *ptr,
                std::vector<std::pair<size_t, size_t>> &dirtyRanges) override;
  void clearDirtyPages(void *ptr) override;

  // Memory mapping and synchronization
  VGREResult mapMemory(void *ptr, size_t size) override;
  VGREResult unmapMemory(void *ptr) override;
};

/**
 * @brief Real implementation of ISecureChannelFactory creating actual
 * SecureChannel instances.
 *
 * This is the production implementation that creates real SecureChannel
 * instances. Used by the singleton instance() method for normal operation.
 */
class RealSecureChannelFactory : public ISecureChannelFactory {
public:
  RealSecureChannelFactory() = default;
  virtual ~RealSecureChannelFactory();

  // Secure channel creation
  std::unique_ptr<SecureChannel> create() override;

  // Security configuration
  bool isSecuritySupported() override;
  VGREResult validateConfiguration() override;
};

/**
 * @brief Real implementation of IKernelManager delegating to RuntimeEngine.
 */
class RealKernelManager : public IKernelManager {
public:
  RealKernelManager() = default;
  virtual ~RealKernelManager();

  VGREResult getKernelArgTypes(uint64_t kernel_id, std::vector<ArgType>& argTypes) override;
  const void* getKernelIR(uint64_t kernel_id) override;
  VGREResult registerKernel(const std::string& name, const std::string& source, uint64_t& kernel_id) override;
  VGREResult launchKernel(uint64_t kernel_id, const dim3& gridDim, const dim3& blockDim, void** args, size_t sharedMem) override;
};

} // namespace advanced
} // namespace vgre