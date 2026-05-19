/**
 * VGRE Shared Utilities - Packets and Error Patterns
 */

#include "vgre/advanced/tcp_cluster/internal/shared_utilities.h"
#include <cstring>

namespace vgre {
namespace advanced {

std::vector<uint8_t> PacketUtils::constructVSBPPacket(PacketType type,
                                                      const void *payload,
                                                      size_t payload_size,
                                                      uint32_t sequence) {
  VSBPHeader header{
      VSBP_MAGIC,
      VSBP_VERSION,
      static_cast<uint16_t>(type),
      sequence,
      static_cast<uint32_t>(payload_size),
      static_cast<uint8_t>(PlatformDetection::getCurrentPlatform().type),
      {0, 0, 0}};
  std::vector<uint8_t> packet(sizeof(VSBPHeader) + payload_size);
  memcpy(packet.data(), &header, sizeof(VSBPHeader));
  if (payload && payload_size > 0)
    memcpy(packet.data() + sizeof(VSBPHeader), payload, payload_size);
  return packet;
}

bool PacketUtils::validateVSBPHeader(const VSBPHeader &header) {
  return header.magic == VSBP_MAGIC && header.version == VSBP_VERSION &&
         header.payloadSize <= 64 * 1024 * 1024;
}

std::string PacketUtils::packetTypeToString(PacketType type) {
  switch (type) {
  case PacketType::TELEMETRY:
    return "TELEMETRY";
  case PacketType::LAUNCH_KERNEL:
    return "LAUNCH_KERNEL";
  case PacketType::RESPONSE:
    return "RESPONSE";
  case PacketType::DATA_HEADER:
    return "DATA_HEADER";
  case PacketType::DATA_BODY:
    return "DATA_BODY";
  case PacketType::STRUCT_DATA:
    return "STRUCT_DATA";
  case PacketType::ARG_SCALAR:
    return "ARG_SCALAR";
  case PacketType::ARG_POINTER:
    return "ARG_POINTER";
  case PacketType::CAPABILITY:
    return "CAPABILITY";
  case PacketType::REGISTER_KERNEL:
    return "REGISTER_KERNEL";
  case PacketType::SECURE_HANDSHAKE:
    return "SECURE_HANDSHAKE";
  case PacketType::SECURE_HANDSHAKE_ACK:
    return "SECURE_HANDSHAKE_ACK";
  case PacketType::PARTITION_DISPATCH:
    return "PARTITION_DISPATCH";
  case PacketType::PARTITION_RESULT:
    return "PARTITION_RESULT";
  case PacketType::CREDIT_REPORT:
    return "CREDIT_REPORT";
  case PacketType::ROTATE_KEY:
    return "ROTATE_KEY";
  case PacketType::SHM_INIT:
    return "SHM_INIT";
  case PacketType::DATA_SHM:
    return "DATA_SHM";
  case PacketType::DATA_HEADER_DIRTY:
    return "DATA_HEADER_DIRTY";
  case PacketType::DATA_SHM_DIRTY:
    return "DATA_SHM_DIRTY";
  case PacketType::DIRTY_RANGE:
    return "DIRTY_RANGE";
  case PacketType::COOP_BARRIER_SYNC:
    return "COOP_BARRIER_SYNC";
  case PacketType::COOP_BARRIER_RESUME:
    return "COOP_BARRIER_RESUME";
  case PacketType::RAW_DATA:
    return "RAW_DATA";
  case PacketType::COLLECTIVE_OP:
    return "COLLECTIVE_OP";
  case PacketType::COLLECTIVE_COMPLETE:
    return "COLLECTIVE_COMPLETE";
  case PacketType::BANDWIDTH_PROBE:
    return "BANDWIDTH_PROBE";
  case PacketType::BANDWIDTH_ACK:
    return "BANDWIDTH_ACK";
  case PacketType::DATA_HEADER_RDMA:
    return "DATA_HEADER_RDMA";
  case PacketType::SECURE_READY:
    return "SECURE_READY";
  case PacketType::CLOCK_SYNC:
    return "CLOCK_SYNC";
  case PacketType::CLOCK_SYNC_REPLY:
    return "CLOCK_SYNC_REPLY";
  default:
    return "UNKNOWN(" + std::to_string(static_cast<uint32_t>(type)) + ")";
  }
}

std::string ErrorHandlingPatterns::translateVGREResult(VGREResult result) {
  switch (result) {
  case VGREResult::SUCCESS:
    return "SUCCESS";
  case VGREResult::ERR_NOT_INITIALIZED:
    return "NOT_INITIALIZED";
  case VGREResult::ERR_INVALID_DEVICE:
    return "INVALID_DEVICE";
  case VGREResult::ERR_OUT_OF_MEMORY:
    return "OUT_OF_MEMORY";
  case VGREResult::ERR_INVALID_VALUE:
    return "INVALID_VALUE";
  case VGREResult::ERR_INVALID_KERNEL:
    return "INVALID_KERNEL";
  case VGREResult::ERR_LAUNCH_FAILURE:
    return "LAUNCH_FAILURE";
  case VGREResult::ERR_COMPILATION:
    return "COMPILATION_ERROR";
  case VGREResult::ERR_NOT_SUPPORTED:
    return "NOT_SUPPORTED";
  case VGREResult::ERR_STREAM:
    return "STREAM_ERROR";
  case VGREResult::ERR_TIMEOUT:
    return "TIMEOUT";
  case VGREResult::ERR_ALREADY_EXISTS:
    return "ALREADY_EXISTS";
  case VGREResult::ERR_IO:
    return "IO_ERROR";
  case VGREResult::ERR_COMPRESSION:
    return "COMPRESSION_ERROR";
  case VGREResult::ERR_NETWORK:
    return "NETWORK_ERROR";
  case VGREResult::ERR_AUTH_FAILED:
    return "AUTH_FAILED";
  case VGREResult::ERR_CRYPTO:
    return "CRYPTO_ERROR";
  case VGREResult::ERR_BUSY:
    return "BUSY";
  case VGREResult::ERR_NOT_FOUND:
    return "NOT_FOUND";
  // ERR_AUTH_RETRY is not used in production (fail-closed auth policy).
  case VGREResult::ERR_UNKNOWN:
    return "UNKNOWN_ERROR";
  default:
    return "UNRECOGNIZED_ERROR";
  }
}

bool ErrorHandlingPatterns::isSocketError(int error_code) {
#ifdef _WIN32
  return error_code == WSAECONNRESET || error_code == WSAETIMEDOUT ||
         error_code == WSAECONNABORTED;
#else
  return error_code == ECONNRESET || error_code == ETIMEDOUT ||
         error_code == ECONNABORTED;
#endif
}

std::string ErrorHandlingPatterns::getSocketErrorString(int error_code) {
#ifdef _WIN32
  char buffer[256];
  if (FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                     NULL, error_code,
                     MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), buffer,
                     sizeof(buffer), NULL)) {
    std::string s(buffer);
    while (!s.empty() &&
           (s.back() == '\n' || s.back() == '\r' || s.back() == ' '))
      s.pop_back();
    return s + " (" + std::to_string(error_code) + ")";
  }
  return "Windows Socket Error " + std::to_string(error_code);
#else
  return std::string(strerror(error_code)) + " (" + std::to_string(error_code) +
         ")";
#endif
}

void ErrorHandlingPatterns::logErrorWithContext(
    VGREResult error, const std::string &operation,
    const std::string &component, const std::string &additional_context) {
  std::string error_msg = translateVGREResult(error);
  std::string full_msg =
      "[" + component + "] " + operation + " failed: " + error_msg;
  if (!additional_context.empty()) {
    full_msg += " - " + additional_context;
  }
}

bool ErrorHandlingPatterns::isRecoverableSocketError(int error_code) {
#ifdef _WIN32
  return error_code == WSAETIMEDOUT || error_code == WSAEWOULDBLOCK;
#else
  return error_code == ETIMEDOUT || error_code == EAGAIN ||
         error_code == EWOULDBLOCK;
#endif
}

bool ErrorHandlingPatterns::isRetryableError(VGREResult error) {
  return error == VGREResult::ERR_TIMEOUT || error == VGREResult::ERR_BUSY ||
         false;
}

bool ErrorHandlingPatterns::isCriticalError(VGREResult error) {
  return error == VGREResult::ERR_OUT_OF_MEMORY ||
         error == VGREResult::ERR_CRYPTO ||
         error == VGREResult::ERR_AUTH_FAILED;
}

std::string
ErrorHandlingPatterns::generateRecoverySuggestion(VGREResult error,
                                                  const std::string &context) {
  switch (error) {
  case VGREResult::ERR_TIMEOUT:
    return "Retry operation after timeout. Check network connectivity.";
  case VGREResult::ERR_AUTH_FAILED:
    return "Verify authentication credentials and retry.";
  case VGREResult::ERR_NETWORK:
    return "Check network connection and retry.";
  case VGREResult::ERR_BUSY:
    return "Wait and retry when system is less busy.";
  default:
    return "Contact system administrator with error details.";
  }
}

} // namespace advanced
} // namespace vgre
