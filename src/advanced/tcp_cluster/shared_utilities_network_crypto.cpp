/**
 * VGRE Shared Utilities - Network and Crypto
 */

#include "vgre/advanced/secure_channel.h"
#include "vgre/advanced/tcp_cluster/internal/shared_utilities.h"
#include "vgre/advanced/tcp_cluster/internal/windows_socket_manager.h"
#include "vgre/common/sockets.h"
#include <random>
#include <chrono>

namespace vgre {
namespace advanced {

VGREResult NetworkUtils::configureSocket(vgre::common::vgre_socket_t fd) {
#ifdef _WIN32
  return WindowsSocketManager::configureSocket(fd);
#else
  if (fd == vgre::common::VGRE_INVALID_SOCKET)
    return VGREResult::ERR_INVALID_VALUE;
  vgre::common::vgre_set_tcp_nodelay(fd);
  vgre::common::vgre_ioctl_nonblock(fd);
  return VGREResult::SUCCESS;
#endif
}

std::string CryptoUtils::computeHmacHex(const std::string &key,
                                        const std::string &msg) {
  uint8_t mac[32];
  vgre::advanced::crypto::hmac_sha256((const uint8_t *)key.data(), key.size(),
                                      (const uint8_t *)msg.data(), msg.size(),
                                      mac);
  char hex[65];
  for (int i = 0; i < 32; ++i)
    sprintf(hex + i * 2, "%02x", mac[i]);
  return std::string(hex);
}

bool CryptoUtils::verifyHmacHex(const std::string &key, const std::string &msg,
                                const std::string &hexTag) {
  std::string expected = computeHmacHex(key, msg);
  return expected == hexTag;
}

std::string CryptoUtils::computeTokenFingerprint(const std::string &token) {
  uint8_t hash[32];
  vgre::advanced::crypto::sha256((const uint8_t *)token.data(), token.size(),
                                 hash);
  char hex[65];
  for (int i = 0; i < 32; ++i)
    sprintf(hex + i * 2, "%02x", hash[i]);
  return std::string(hex);
}

std::string CryptoUtils::generateSecureToken(size_t length) {
  static const char alphabet[] =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  const size_t alphabetLen = sizeof(alphabet) - 1;
  std::string token;
  token.reserve(length);
  std::mt19937_64 rng(static_cast<uint64_t>(
      std::chrono::steady_clock::now().time_since_epoch().count()));
  std::uniform_int_distribution<size_t> dist(0, alphabetLen - 1);
  for (size_t i = 0; i < length; ++i) {
    token.push_back(alphabet[dist(rng)]);
  }
  return token;
}

} // namespace advanced
} // namespace vgre
