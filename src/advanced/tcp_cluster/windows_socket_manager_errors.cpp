/**
 * Windows Socket Manager - Error Handling and Configuration
 * 
 * Provides unified error mapping and socket option configuration.
 */

#include "vgre/advanced/tcp_cluster/internal/windows_socket_manager.h"
#include "vgre/common/logger.h"

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#include <mstcpip.h>
#else
#include <cerrno>
#include <cstring>
#endif

namespace vgre {
namespace advanced {

VGREResult WindowsSocketManager::getLastSocketError() {
#ifdef _WIN32
    int error = WSAGetLastError();
    switch (error) {
        case WSAEWOULDBLOCK:
        case WSAEINPROGRESS:
        case WSAEALREADY:    return VGREResult::ERR_BUSY;
        case WSAECONNRESET:
        case WSAECONNABORTED:
        case WSAENETDOWN:
        case WSAENETUNREACH:
        case WSAEHOSTDOWN:
        case WSAEHOSTUNREACH:
        case WSAENOTCONN:
        case WSAECONNREFUSED:
        case WSAESHUTDOWN:   return VGREResult::ERR_NETWORK;
        case WSAENOTSOCK:
        case WSAEBADF:
        case WSAEINVAL:
        case WSAEFAULT:
        case WSAEOPNOTSUPP:
        case WSAEPROTONOSUPPORT:
        case WSAEMSGSIZE:
        case WSAENOPROTOOPT: return VGREResult::ERR_INVALID_VALUE;
        case WSAEACCES:      return VGREResult::ERR_AUTH_FAILED;
        case WSAEADDRINUSE:
        case WSAEISCONN:     return VGREResult::ERR_ALREADY_EXISTS;
        case WSAEADDRNOTAVAIL: return VGREResult::ERR_NOT_FOUND;
        case WSAETIMEDOUT:   return VGREResult::ERR_TIMEOUT;
        case WSAENOBUFS:
        case WSAENOMORE:
        case WSAEMFILE:
        case WSAETOOMANYREFS: return VGREResult::ERR_OUT_OF_MEMORY;
        default:             return VGREResult::ERR_IO;
    }
#else
    switch (errno) {
        case EAGAIN:
#if EAGAIN != EWOULDBLOCK
        case EWOULDBLOCK:
#endif
            return VGREResult::ERR_BUSY;
        case ECONNRESET:
        case ECONNABORTED:
        case ENETDOWN:
        case ENETUNREACH:
        case EHOSTDOWN:
        case EHOSTUNREACH:
        case ECONNREFUSED:   return VGREResult::ERR_NETWORK;
        case EBADF:
        case ENOTSOCK:
        case EINVAL:
        case EFAULT:         return VGREResult::ERR_INVALID_VALUE;
        case EACCES:         return VGREResult::ERR_AUTH_FAILED;
        case EADDRINUSE:     return VGREResult::ERR_ALREADY_EXISTS;
        case EADDRNOTAVAIL:  return VGREResult::ERR_NOT_FOUND;
        case ETIMEDOUT:      return VGREResult::ERR_TIMEOUT;
        default:             return VGREResult::ERR_IO;
    }
#endif
}

std::string WindowsSocketManager::getSocketErrorString(int error_code) {
#ifdef _WIN32
    switch (error_code) {
        case WSAEWOULDBLOCK: return "Operation would block";
        case WSAECONNRESET:  return "Connection reset by peer";
        case WSAECONNABORTED: return "Connection aborted";
        case WSAENETDOWN:    return "Network is down";
        case WSAETIMEDOUT:   return "Connection timed out";
        case WSANOTINITIALISED: return "Winsock not initialized";
        default:             return "Socket error " + std::to_string(error_code);
    }
#else
    return std::string(strerror(error_code)) + " (" + std::to_string(error_code) + ")";
#endif
}

VGREResult WindowsSocketManager::configureSocket(vgre::common::vgre_socket_t fd) {
    if (fd == vgre::common::VGRE_INVALID_SOCKET) return VGREResult::ERR_INVALID_VALUE;
#ifdef _WIN32
    int nodelay = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, (const char*)&nodelay, sizeof(nodelay));
    int buf_size = 131072;
    setsockopt(fd, SOL_SOCKET, SO_SNDBUF, (const char*)&buf_size, sizeof(buf_size));
    setsockopt(fd, SOL_SOCKET, SO_RCVBUF, (const char*)&buf_size, sizeof(buf_size));
    u_long mode = 1;
    ioctlsocket(fd, FIONBIO, &mode);
#endif
    return VGREResult::SUCCESS;
}

} // namespace advanced
} // namespace vgre
