#include "vgre/advanced/tcp_cluster/internal/packet_handler.h"
#include "vgre/advanced/tcp_cluster/internal/memory_sync_manager.h"
#include "vgre/advanced/secure_channel.h"
#include "vgre/common/logger.h"
#include "vgre/common/input_validation.h"
#include "vgre/common/sockets.h"
#include "vgre/advanced/tcp_cluster/internal/shared_utilities.h"
#include "vgre/core/runtime_engine.h"
#include "vgre/common/types.h"
#include "vgre/core/memory_manager.h"
#include <cstring>
#include <algorithm>
#include <vector>

namespace vgre {
namespace advanced {

// Using vgre::common types and helpers
using vgre::common::vgre_socket_t;
using vgre::common::VGRE_INVALID_SOCKET;
using vgre::common::vgre_get_last_socket_error;
using vgre::common::vgre_is_would_block;
using vgre::common::vgre_send_all;

PacketHandler::PacketHandler() {}

// ── Unified Packet Construction ──────────────────────────────────────────
std::vector<uint8_t> PacketHandler::constructPacket(PacketType type, const void* payload, size_t payloadLen) {
  uint32_t seq = sequence_counter_.fetch_add(1, std::memory_order_relaxed);
  return PacketUtils::constructVSBPPacket(type, payload, payloadLen, seq);
}

// ── Packet Sending (Queued) ──────────────────────────────────────────────
VGREResult PacketHandler::sendPacket(vgre_socket_t fd, PacketType type, 
                                     const void* payload, size_t payloadLen, 
                                     SecureChannel* sc) {
  // Validate socket
  if (fd == VGRE_INVALID_SOCKET) {
    return VGREResult::ERR_INVALID_VALUE;
  }

  // CRITICAL: Validate packet size to prevent buffer overflow
  if (common::InputValidator::validatePacketSize(payloadLen) != VGREResult::SUCCESS) {
    VGRE_LOG_ERROR("PacketHandler", "Packet size validation failed: " + std::to_string(payloadLen));
    return VGREResult::ERR_INVALID_VALUE;
  }

  // Additional check for total packet size
  size_t totalLen = sizeof(VSBPHeader) + payloadLen;
  if (common::InputValidator::validatePacketSize(totalLen) != VGREResult::SUCCESS) {
    VGRE_LOG_ERROR("PacketHandler", "Total packet size validation failed: " + std::to_string(totalLen));
    return VGREResult::ERR_INVALID_VALUE;
  }
  
  std::vector<uint8_t> staging = constructPacket(type, payload, payloadLen);

  packets_sent_.fetch_add(1, std::memory_order_relaxed);
  bytes_sent_.fetch_add(staging.size(), std::memory_order_relaxed);

  if (sc && sc->isInitialized()) {
    return sc->sendSecure(fd, staging.data(), staging.size());
  }
  bool ok = vgre_send_all(fd, staging.data(), staging.size());
  return ok ? VGREResult::SUCCESS : VGREResult::ERR_IO;
}

// ── Packet Sending (Direct) ──────────────────────────────────────────────
VGREResult PacketHandler::sendPacketDirect(vgre_socket_t fd, PacketType type, 
                                           const void* payload, size_t payloadLen, 
                                           SecureChannel* sc) {
  // Validate socket
  if (fd == VGRE_INVALID_SOCKET) {
    return VGREResult::ERR_INVALID_VALUE;
  }

  // Use unified packet construction
  std::vector<uint8_t> staging = constructPacket(type, payload, payloadLen);

  // Update statistics
  packets_sent_.fetch_add(1, std::memory_order_relaxed);
  bytes_sent_.fetch_add(staging.size(), std::memory_order_relaxed);

  // Send directly (bypass queue)
  if (sc && sc->isInitialized()) {
    return sc->sendSecure(fd, staging.data(), staging.size());
  } else {
    // Fallback to direct send without encryption
    bool success = vgre_send_all(fd, staging.data(), staging.size());
    return success ? VGREResult::SUCCESS : VGREResult::ERR_IO;
  }
}

// ── Packet Receiving ─────────────────────────────────────────────────────
int PacketHandler::recvPacket(vgre_socket_t fd, std::vector<uint8_t>& outBuffer, 
                              SecureChannel* sc) {
  if (sc && sc->isInitialized()) {
    std::vector<uint8_t> payload;
    VGREResult r = sc->recvSecure(fd, payload);
    if (r == VGREResult::SUCCESS) {
      packets_received_.fetch_add(1, std::memory_order_relaxed);
      // Total wire size includes secure packet header
      bytes_received_.fetch_add(static_cast<uint64_t>(payload.size() + sizeof(SecurePacketHeader)), 
                                std::memory_order_relaxed);
      outBuffer.insert(outBuffer.end(), payload.begin(), payload.end());
      return static_cast<int>(payload.size());
    } else if (r == VGREResult::ERR_TIMEOUT) {
      return 0;
    }
    return -1; // Error
  } else {
    char temp[8192];
    int n = recv(fd, temp, sizeof(temp), 0);
    if (n > 0) {
      packets_received_.fetch_add(1, std::memory_order_relaxed);
      bytes_received_.fetch_add(static_cast<uint64_t>(n), std::memory_order_relaxed);
      outBuffer.insert(outBuffer.end(), temp, temp + n);
      return n;
    } else if (n == 0) {
      return -1; // Disconnected
    } else {
      if (vgre_is_would_block(vgre_get_last_socket_error())) return 0;
      return -1;
    }
  }
}

// ── VSBP Header Parsing ──────────────────────────────────────────────────
bool PacketHandler::parseVSBPHeader(const uint8_t* data, size_t dataSize, 
                                    VSBPHeader& header) {
  // Validate input
  if (!data || dataSize < sizeof(VSBPHeader)) return false;

  // Copy header
  memcpy(&header, data, sizeof(VSBPHeader));

  // Use shared validation logic
  if (!PacketUtils::validateVSBPHeader(header)) {
    VGRE_LOG_ERROR("PacketHandler", 
                   "VSBP protocol violation: invalid magic (0x" + 
                   std::to_string(header.magic) + ") or version (" + 
                   std::to_string(header.version) + ")");
    return false;
  }

  // Validate payload size doesn't exceed buffer
  if (dataSize < sizeof(VSBPHeader) + header.payloadSize) {
    VGRE_LOG_DEBUG("PacketHandler", 
                   "Incomplete packet: need " + 
                   std::to_string(sizeof(VSBPHeader) + header.payloadSize) + 
                   " bytes, have " + std::to_string(dataSize));
    return false;
  }

  return true;
}

// ── MemorySyncManager Argument Marshalling ──────────────────────────────────

VGREResult MemorySyncManager::streamArgumentsToWorker(
    void **args, int num_args, uint64_t kernel_id,
    std::shared_ptr<TCPClusterManager::ClientConnection> client) {
  if (!client) {
    return VGREResult::ERR_INVALID_VALUE;
  }
  if (!args || num_args <= 0) {
    return VGREResult::SUCCESS;
  }

  std::vector<ArgType> argTypes;
  if (core::RuntimeEngine::instance().getKernelArgTypes(kernel_id, argTypes) != VGREResult::SUCCESS) {
    return VGREResult::ERR_INVALID_KERNEL;
  }

  // Ensure type info is complete
  if (static_cast<int>(argTypes.size()) < num_args) {
    VGRE_LOG_ERROR("TCPCluster", "streamArgumentsToWorker: kernel " + std::to_string(kernel_id) +
                                     " has " + std::to_string(argTypes.size()) +
                                     " registered arg types but " + std::to_string(num_args) +
                                     " args were supplied.");
    return VGREResult::ERR_INVALID_KERNEL;
  }

  for (int i = 0; i < num_args; ++i) {
    VGREResult r = VGREResult::SUCCESS;
    if (argTypes[i] == ArgType::STRUCT) {
      r = sendStructArg(args[i], i, kernel_id, client);
    } else if (argTypes[i] == ArgType::POINTER) {
      r = sendPointerArg(args[i], i, client);
    } else {
      r = sendScalarArg(args[i], i, argTypes[i], client);
    }

    if (r != VGREResult::SUCCESS) {
      return r;
    }
  }
  return VGREResult::SUCCESS;
}

VGREResult MemorySyncManager::sendScalarArg(
    void *arg, int arg_index, ArgType type,
    std::shared_ptr<TCPClusterManager::ClientConnection> client) {
  size_t arg_size = vgre_get_type_size(static_cast<int>(type));

  ArgScalarPacket apkt{};
  apkt.arg_index = static_cast<uint32_t>(arg_index);
  apkt.arg_type = static_cast<uint8_t>(type);
  memset(&apkt.value, 0, 8);
  memcpy(&apkt.value, arg, arg_size);

  if (parent_->send_packet(client->socket_fd, PacketType::ARG_SCALAR, &apkt, 
                           sizeof(ArgScalarPacket), client->secure_channel.get()) != VGREResult::SUCCESS) {
    return VGREResult::ERR_IO;
  }
  return VGREResult::SUCCESS;
}

VGREResult MemorySyncManager::sendPointerArg(
    void *arg, int arg_index,
    std::shared_ptr<TCPClusterManager::ClientConnection> client) {
  uint64_t handle = 0;

  if (arg) {
    void *ptr = *static_cast<void **>(arg);
    handle = reinterpret_cast<uint64_t>(ptr);

    if (ptr) {
      size_t size = core::RuntimeEngine::instance().getMemoryManager().getAllocationSize(ptr);
      if (size == 0) {
        VGRE_LOG_WARN("TCPCluster", "sendPointerArg[" + std::to_string(arg_index) + "]: pointer " +
                                        std::to_string(handle) + " has size=0 — not a VGRE-managed allocation. "
                                        "Use vgre_malloc/cudaMalloc to ensure the buffer is tracked.");
      } else {
        VGREResult syncResult = syncPointerToWorker(ptr, handle, client);
        if (syncResult != VGREResult::SUCCESS) {
          return syncResult;
        }
      }
    }
  }

  ArgScalarPacket apkt{};
  apkt.arg_index = static_cast<uint32_t>(arg_index);
  apkt.arg_type = static_cast<uint8_t>(ArgType::POINTER);
  apkt.value = handle;

  if (parent_->send_packet(client->socket_fd, PacketType::ARG_POINTER, &apkt, 
                           sizeof(ArgScalarPacket), client->secure_channel.get()) != VGREResult::SUCCESS) {
    return VGREResult::ERR_IO;
  }
  return VGREResult::SUCCESS;
}

VGREResult MemorySyncManager::sendStructArg(
    void *arg, int arg_index, uint64_t kernel_id,
    std::shared_ptr<TCPClusterManager::ClientConnection> client) {
  const auto *ir = core::RuntimeEngine::instance().getKernelIR(kernel_id);
  if (!ir || arg_index >= static_cast<int>(ir->argSizes.size())) {
    return VGREResult::ERR_INVALID_VALUE;
  }

  size_t size = ir->argSizes[arg_index];
  StructDataPacket spkt{};
  spkt.arg_index = static_cast<uint32_t>(arg_index);
  spkt.size = static_cast<uint32_t>(size);

  if (parent_->send_packet(client->socket_fd, PacketType::STRUCT_DATA, &spkt, 
                           sizeof(StructDataPacket), client->secure_channel.get()) != VGREResult::SUCCESS) {
    return VGREResult::ERR_IO;
  }

  if (parent_->send_packet(client->socket_fd, PacketType::DATA_BODY, arg, size, 
                           client->secure_channel.get()) != VGREResult::SUCCESS) {
    return VGREResult::ERR_IO;
  }
  return VGREResult::SUCCESS;
}
} // namespace advanced
} // namespace vgre
