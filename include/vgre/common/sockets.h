#ifndef VGRE_COMMON_SOCKETS_H
#define VGRE_COMMON_SOCKETS_H

#if defined(_WIN32)
#include <winsock2.h>
#include <ws2tcpip.h>
#include <mstcpip.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <errno.h>
#endif

#include <vector>

namespace vgre {
namespace common {

#if defined(_WIN32)
typedef SOCKET vgre_socket_t;
typedef int socklen_t;
constexpr vgre_socket_t VGRE_INVALID_SOCKET = INVALID_SOCKET;
constexpr int VGRE_SOCKET_ERROR = SOCKET_ERROR;

#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif

#else
typedef int vgre_socket_t;
constexpr vgre_socket_t VGRE_INVALID_SOCKET = -1;
constexpr int VGRE_SOCKET_ERROR = -1;

#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif
#endif

// ── Platform-Agnostic Socket Helpers ─────────────────────────────────────

inline void vgre_close_socket(vgre_socket_t s) {
#if defined(_WIN32)
  closesocket(s);
#else
  close(s);
#endif
}

inline int vgre_get_last_socket_error() {
#if defined(_WIN32)
  return WSAGetLastError();
#else
  return errno;
#endif
}

inline bool vgre_is_would_block(int error) {
#if defined(_WIN32)
  return error == WSAEWOULDBLOCK || error == WSAEINPROGRESS;
#else
  return error == EAGAIN || error == EWOULDBLOCK || error == EINPROGRESS;
#endif
}

inline int vgre_setsockopt(vgre_socket_t s, int level, int optname,
                           const void *optval, int optlen) {
#if defined(_WIN32)
  return setsockopt(s, level, optname, static_cast<const char *>(optval), optlen);
#else
  return setsockopt(s, level, optname, optval, optlen);
#endif
}

inline int vgre_set_recv_timeout(vgre_socket_t s, int timeoutMs) {
#if defined(_WIN32)
  DWORD timeout = static_cast<DWORD>(timeoutMs);
  return vgre_setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
#else
  struct timeval tv;
  tv.tv_sec = timeoutMs / 1000;
  tv.tv_usec = (timeoutMs % 1000) * 1000;
  return vgre_setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
#endif
}

inline int vgre_ioctl_nonblock(vgre_socket_t s) {
#if defined(_WIN32)
  u_long mode = 1;
  return ioctlsocket(s, FIONBIO, &mode);
#else
  int flags = fcntl(s, F_GETFL, 0);
  if (flags == -1) return -1;
  return fcntl(s, F_SETFL, flags | O_NONBLOCK);
#endif
}

inline int vgre_set_tcp_keepalive(vgre_socket_t s, int idleS, int intvlS, int cnt) {
#if defined(_WIN32)
  tcp_keepalive alive;
  alive.onoff = 1;
  alive.keepalivetime = idleS * 1000;
  alive.keepaliveinterval = intvlS * 1000;
  DWORD bytesReturned = 0;
  return WSAIoctl(s, SIO_KEEPALIVE_VALS, &alive, sizeof(alive), NULL, 0, &bytesReturned, NULL, NULL);
#else
  int opt = 1;
  int res = setsockopt(s, SOL_SOCKET, SO_KEEPALIVE, &opt, sizeof(opt));
  if (res < 0) return res;
#ifdef TCP_KEEPIDLE
  setsockopt(s, IPPROTO_TCP, TCP_KEEPIDLE, &idleS, sizeof(idleS));
#endif
#ifdef TCP_KEEPINTVL
  setsockopt(s, IPPROTO_TCP, TCP_KEEPINTVL, &intvlS, sizeof(intvlS));
#endif
#ifdef TCP_KEEPCNT
  setsockopt(s, IPPROTO_TCP, TCP_KEEPCNT, &cnt, sizeof(cnt));
#endif
  return 0;
#endif
}

struct vgre_pollfd {
  vgre_socket_t fd;
  short events;
  short revents;
};

inline int vgre_poll(vgre_pollfd *fds, size_t count, int timeoutMs) {
#if defined(_WIN32)
  if (count == 0) return 0;
  std::vector<WSAPOLLFD> nativeFds(count);
  for (size_t i = 0; i < count; ++i) {
    nativeFds[i].fd = fds[i].fd;
    nativeFds[i].events = fds[i].events;
    nativeFds[i].revents = 0;
  }
  int res = WSAPoll(nativeFds.data(), static_cast<ULONG>(count), timeoutMs);
  if (res >= 0) {
    for (size_t i = 0; i < count; ++i) {
      fds[i].revents = nativeFds[i].revents;
    }
  }
  return res;
#else
  if (count == 0) return 0;
  std::vector<pollfd> nativeFds(count);
  for (size_t i = 0; i < count; ++i) {
    nativeFds[i].fd = fds[i].fd;
    nativeFds[i].events = fds[i].events;
    nativeFds[i].revents = 0;
  }
  int res = poll(nativeFds.data(), count, timeoutMs);
  if (res >= 0) {
    for (size_t i = 0; i < count; ++i) {
      fds[i].revents = nativeFds[i].revents;
    }
  }
  return res;
#endif
}

} // namespace common
} // namespace vgre

#endif // VGRE_COMMON_SOCKETS_H
