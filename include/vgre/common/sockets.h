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

#include <atomic>
#include <chrono>
#include <cstdio>
#include <string>
#include <thread>
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

// MSG_NOSIGNAL suppresses SIGPIPE on broken-pipe send() — Linux only.
// macOS uses the SO_NOSIGPIPE socket option instead (see vgre_set_nosigpipe).
// Define as 0 so call-sites compile; the SIGPIPE suppression comes from
// vgre_set_nosigpipe() applied once at socket creation time on macOS.
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
#if defined(__APPLE__)
  // macOS: TCP_KEEPALIVE sets the idle time before keepalives begin.
  // TCP_KEEPINTVL and TCP_KEEPCNT are also available on macOS.
  setsockopt(s, IPPROTO_TCP, TCP_KEEPALIVE, &idleS, sizeof(idleS));
#else
  // Linux: TCP_KEEPIDLE is the equivalent of macOS TCP_KEEPALIVE.
#ifdef TCP_KEEPIDLE
  setsockopt(s, IPPROTO_TCP, TCP_KEEPIDLE, &idleS, sizeof(idleS));
#endif
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

// ── SO_NOSIGPIPE (macOS only) ─────────────────────────────────────────────
// On macOS, MSG_NOSIGNAL is not a valid flag for send(). The equivalent
// protection against SIGPIPE is SO_NOSIGPIPE, set once at socket creation.
// This is a no-op on Linux (where MSG_NOSIGNAL works) and Windows (no SIGPIPE).
inline int vgre_set_nosigpipe(vgre_socket_t s) {
#if defined(__APPLE__)
  int opt = 1;
  return setsockopt(s, SOL_SOCKET, SO_NOSIGPIPE, &opt, sizeof(opt));
#else
  (void)s;
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

// ── RAII Socket Guard ─────────────────────────────────────────────────────
struct VgreSocketGuard {
  explicit VgreSocketGuard(vgre_socket_t fd = VGRE_INVALID_SOCKET) : fd_(fd) {}
  ~VgreSocketGuard() { if (fd_ != VGRE_INVALID_SOCKET) vgre_close_socket(fd_); }
  VgreSocketGuard(const VgreSocketGuard&) = delete;
  VgreSocketGuard& operator=(const VgreSocketGuard&) = delete;
  VgreSocketGuard(VgreSocketGuard&& o) noexcept : fd_(o.fd_) { o.fd_ = VGRE_INVALID_SOCKET; }
  VgreSocketGuard& operator=(VgreSocketGuard&& o) noexcept {
    if (this != &o) {
      if (fd_ != VGRE_INVALID_SOCKET) vgre_close_socket(fd_);
      fd_ = o.fd_;
      o.fd_ = VGRE_INVALID_SOCKET;
    }
    return *this;
  }
  vgre_socket_t get() const { return fd_; }
  vgre_socket_t release() { vgre_socket_t t = fd_; fd_ = VGRE_INVALID_SOCKET; return t; }
private:
  vgre_socket_t fd_;
};

// ── IPv6 helper: dual-stack TCP connect ──────────────────────────────────
// Resolves `host` (IPv4 dotted-decimal OR IPv6 literal OR hostname) and
// connects a TCP socket to `host:port`.  Returns the connected socket fd on
// success or VGRE_INVALID_SOCKET on failure.
// IPv6 literals must be passed without brackets, e.g. "fe80::1%eth0".
// Caller is responsible for calling vgre_close_socket() on the returned fd.
inline vgre_socket_t vgre_connect_tcp(const char* host, int port,
                                       int timeoutMs = 5000) {
  char portStr[8];
  ::snprintf(portStr, sizeof(portStr), "%d", port);

  struct addrinfo hints{};
  hints.ai_family   = AF_UNSPEC;    // accept IPv4 and IPv6
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_protocol = IPPROTO_TCP;

  struct addrinfo* res = nullptr;
  if (getaddrinfo(host, portStr, &hints, &res) != 0 || !res) {
    return VGRE_INVALID_SOCKET;
  }

  vgre_socket_t fd = VGRE_INVALID_SOCKET;
  for (struct addrinfo* ai = res; ai; ai = ai->ai_next) {
    fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
    if (fd == VGRE_INVALID_SOCKET) continue;

    // Non-blocking connect so we can apply a timeout.
    vgre_ioctl_nonblock(fd);

    // SO_NOSIGPIPE on macOS — suppresses SIGPIPE on broken-pipe writes.
    vgre_set_nosigpipe(fd);

    int cr = connect(fd, ai->ai_addr, static_cast<int>(ai->ai_addrlen));
    bool connected = (cr == 0);

    if (!connected) {
      int err = vgre_get_last_socket_error();
      if (vgre_is_would_block(err) || err == 0) {
        // Wait for connection to complete with poll/select.
        vgre_pollfd pfd{fd, POLLOUT, 0};
        int pr = vgre_poll(&pfd, 1, timeoutMs);
        if (pr > 0 && (pfd.revents & POLLOUT)) {
          // Verify no asynchronous error.
          int sockErr = 0;
#if defined(_WIN32)
          int optLen = sizeof(sockErr);
          getsockopt(fd, SOL_SOCKET, SO_ERROR,
                     reinterpret_cast<char*>(&sockErr), &optLen);
#else
          socklen_t optLen = sizeof(sockErr);
          getsockopt(fd, SOL_SOCKET, SO_ERROR, &sockErr, &optLen);
#endif
          connected = (sockErr == 0);
        }
      }
    }

    if (connected) {
      // Restore blocking mode for subsequent I/O.
#if defined(_WIN32)
      u_long mode = 0;
      ioctlsocket(fd, FIONBIO, &mode);
#else
      int flags = fcntl(fd, F_GETFL, 0);
      if (flags >= 0) fcntl(fd, F_SETFL, flags & ~O_NONBLOCK);
#endif
      // Enable TCP keepalive (30 s idle → 5 s probes → 3 retries = ~45 s dead detection).
      vgre_set_tcp_keepalive(fd, 30, 5, 3);
      freeaddrinfo(res);
      return fd;
    }

    vgre_close_socket(fd);
    fd = VGRE_INVALID_SOCKET;
  }

  freeaddrinfo(res);
  return VGRE_INVALID_SOCKET;
}

// ── IPv6 helper: dual-stack TCP listen socket ─────────────────────────────
// Creates a TCP server socket bound to `port` on all interfaces.
// On systems that support it, uses IPV6_V6ONLY=0 so a single socket accepts
// both IPv4 and IPv6 connections (dual-stack).  Falls back to IPv4-only if
// IPv6 is unavailable.
inline vgre_socket_t vgre_listen_tcp(int port, int backlog = 32) {
  // Try IPv6 dual-stack first.
  vgre_socket_t fd = socket(AF_INET6, SOCK_STREAM, IPPROTO_TCP);
  if (fd != VGRE_INVALID_SOCKET) {
    // IPV6_V6ONLY = 0 → accept IPv4-mapped addresses too (dual-stack).
    int v6only = 0;
    vgre_setsockopt(fd, IPPROTO_IPV6, IPV6_V6ONLY, &v6only, sizeof(v6only));

    int reuse = 1;
    vgre_setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    vgre_set_nosigpipe(fd);

    sockaddr_in6 addr{};
    addr.sin6_family = AF_INET6;
    addr.sin6_port   = htons(static_cast<uint16_t>(port));
    addr.sin6_addr   = in6addr_any;

    if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0 &&
        listen(fd, backlog) == 0) {
      vgre_set_tcp_keepalive(fd, 30, 5, 3);
      return fd;
    }
    vgre_close_socket(fd);
  }

  // Fallback: IPv4-only.
  fd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  if (fd == VGRE_INVALID_SOCKET) return VGRE_INVALID_SOCKET;

  int reuse = 1;
  vgre_setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
  vgre_set_nosigpipe(fd);

  sockaddr_in addr4{};
  addr4.sin_family      = AF_INET;
  addr4.sin_port        = htons(static_cast<uint16_t>(port));
  addr4.sin_addr.s_addr = INADDR_ANY;

  if (bind(fd, reinterpret_cast<sockaddr*>(&addr4), sizeof(addr4)) == 0 &&
      listen(fd, backlog) == 0) {
    vgre_set_tcp_keepalive(fd, 30, 5, 3);
    return fd;
  }
  vgre_close_socket(fd);
  return VGRE_INVALID_SOCKET;
}

// ── Peer address extraction (IPv4 or IPv6) ────────────────────────────────
// Returns a human-readable string "a.b.c.d" or "[::1]" for the connected peer.
inline std::string vgre_peer_address(vgre_socket_t fd) {
  sockaddr_storage ss{};
  socklen_t slen = sizeof(ss);
  if (getpeername(fd, reinterpret_cast<sockaddr*>(&ss), &slen) != 0)
    return "<unknown>";

  char buf[INET6_ADDRSTRLEN] = {};
  if (ss.ss_family == AF_INET6) {
    inet_ntop(AF_INET6,
              &reinterpret_cast<sockaddr_in6*>(&ss)->sin6_addr,
              buf, sizeof(buf));
    return std::string("[") + buf + "]";
  } else {
    inet_ntop(AF_INET,
              &reinterpret_cast<sockaddr_in*>(&ss)->sin_addr,
              buf, sizeof(buf));
    return buf;
  }
}

// ── Blocking send with optional shutdown-flag ─────────────────────────────
// Sends all `len` bytes; returns false on timeout (5s), error, or if
// `enabled` is non-null and cleared (shutdown signal).
inline bool vgre_send_all(vgre_socket_t sock, const void* buf, size_t len,
                          const std::atomic<bool>* enabled = nullptr) {
  const char* p = static_cast<const char*>(buf);
  size_t sent = 0;
  auto start = std::chrono::steady_clock::now();
  while (sent < len) {
    if (enabled && !enabled->load()) return false;
    if (std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - start).count() > 5)
      return false;
    int n = send(sock, p + sent, static_cast<int>(len - sent), MSG_NOSIGNAL);
    if (n <= 0) {
      if (n < 0 && vgre_is_would_block(vgre_get_last_socket_error())) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        continue;
      }
      return false;
    }
    sent += static_cast<size_t>(n);
  }
  return true;
}

} // namespace common
} // namespace vgre

#endif // VGRE_COMMON_SOCKETS_H
