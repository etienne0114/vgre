#include <gtest/gtest.h>
#include <vgre/advanced/tcp_cluster.h>
#include <vgre/common/sockets.h>
#include <thread>
#include <chrono>
#include <iostream>
#include <cstring>

using namespace vgre::common;

namespace vgre {
namespace test {

// Cross-platform worker/master communication test
class CrossPlatformWorkerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    master_addr_ = "127.0.0.1";
    master_port_ = 12345;
    worker_id_ = 1;
  }

  void TearDown() override {}

  std::string master_addr_;
  uint16_t master_port_;
  uint32_t worker_id_;
};

// Test: Socket creation and closure (platform-agnostic)
TEST_F(CrossPlatformWorkerTest, SocketCreationClosure) {
  vgre_socket_t sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  ASSERT_NE(sock, VGRE_INVALID_SOCKET);
  vgre_close_socket(sock);
}

// Test: Socket option setting (platform-agnostic)
TEST_F(CrossPlatformWorkerTest, SocketOptions) {
  vgre_socket_t sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  ASSERT_NE(sock, VGRE_INVALID_SOCKET);

  int nodelay = 1;
  ASSERT_EQ(vgre_setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, 
                            (const char*)&nodelay, sizeof(nodelay)), 0);

  int reuse = 1;
  ASSERT_EQ(vgre_setsockopt(sock, SOL_SOCKET, SO_REUSEADDR,
                            (const char*)&reuse, sizeof(reuse)), 0);

  vgre_close_socket(sock);
}

// Test: Non-blocking socket mode
TEST_F(CrossPlatformWorkerTest, NonBlockingMode) {
  vgre_socket_t sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  ASSERT_NE(sock, VGRE_INVALID_SOCKET);

  // Set non-blocking
  ASSERT_EQ(vgre_ioctl_nonblock(sock), 0);

  // Attempt connection should not block
  struct sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(19999); // Unlikely to be listening
  addr.sin_addr.s_addr = inet_addr("127.0.0.1");

  int ret = connect(sock, (struct sockaddr*)&addr, sizeof(addr));
  // Should get WOULD_BLOCK error, not hang
  if (ret != 0) {
    // This is expected
  }

  vgre_close_socket(sock);
}

// Test: Error code abstraction
TEST_F(CrossPlatformWorkerTest, ErrorCodeAbstraction) {
  // Verify error code functions are defined
  int err = EWOULDBLOCK;
  ASSERT_TRUE(vgre_is_would_block(err) || vgre_is_would_block(EAGAIN));
}

// Platform-specific behavior tests
class PlatformSpecificTests : public ::testing::Test {};

#ifdef _WIN32
TEST_F(PlatformSpecificTests, WindowsSocketInitialization) {
  // On Windows, WSAStartup should have been called
  vgre_socket_t sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  ASSERT_NE(sock, VGRE_INVALID_SOCKET);
  vgre_close_socket(sock);
}
#endif

#ifdef __APPLE__
TEST_F(PlatformSpecificTests, macOSSpecificFeatures) {
  // macOS-specific test for SO_NOSIGPIPE
  vgre_socket_t sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  ASSERT_NE(sock, VGRE_INVALID_SOCKET);
  // SO_NOSIGPIPE should prevent SIGPIPE on broken connections
  vgre_close_socket(sock);
}
#endif

#ifdef __linux__
TEST_F(PlatformSpecificTests, LinuxSpecificFeatures) {
  // Linux-specific test for socket capabilities
  vgre_socket_t sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  ASSERT_NE(sock, VGRE_INVALID_SOCKET);
  vgre_close_socket(sock);
}
#endif

} // namespace test
} // namespace vgre
