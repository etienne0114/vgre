#pragma once

#include "vgre/api/vgre_c_api.h"
#include "vgre/common/error_codes.h"
#include "vgre/advanced/secure_channel.h"
#include "vgre/advanced/tcp_cluster_validation.h"
#include <cstdint>
#include <cstddef>

namespace vgre {
namespace advanced {

// ── VGRE Structured Binary Protocol (VSBP) v0.1.2 ─────────────────────────
constexpr uint32_t VSBP_MAGIC = 0x56475245; // 'VGRE'
constexpr uint16_t VSBP_VERSION = 0x0102;   // v0.1.2

enum class PacketType : uint32_t {
  TELEMETRY = 1,
  LAUNCH_KERNEL = 2,
  RESPONSE = 3,
  DATA_HEADER = 4,   // size and target pointer
  DATA_BODY = 5,      // raw bytes
  STRUCT_DATA = 6,    // size and arg index
  ARG_SCALAR = 7,     // scalar arg (8 bytes)
  ARG_POINTER = 8,    // pointer arg (handle)
  CAPABILITY = 9,     // per-node hardware info
  REGISTER_KERNEL = 10,
  // Phase 5: Global Compute Network
  SECURE_HANDSHAKE = 11,      // nonce exchange for key derivation
  SECURE_HANDSHAKE_ACK = 12,  // handshake acknowledgment
  PARTITION_DISPATCH = 13,    // sub-grid dispatch for partitioned kernel
  PARTITION_RESULT = 14,      // result from a partition execution
  CREDIT_REPORT = 15,         // compute-unit-seconds billing report
  ROTATE_KEY = 16,            // Phase 10: dynamic session key rotation
  SHM_INIT = 17,              // Phase 11: negotiate SHM segment
  DATA_SHM = 18,              // Phase 11: data in SHM
  DATA_HEADER_DIRTY = 19,     // Phase 11: dirty ranges header (TCP)
  DATA_SHM_DIRTY = 20,        // Phase 11: dirty ranges header (SHM)
  DIRTY_RANGE = 21,           // Phase 11: offset/size range packet
  COOP_BARRIER_SYNC = 22,     // Phase 13: Decentralized Grid Barrier
  COOP_BARRIER_RESUME = 23,   // Phase 13: Master resume signal
  RAW_DATA = 24,              // Generic unstructured payload
  COLLECTIVE_OP = 25,         // Phase 14: Collective Ops (all_reduce, etc.)
  COLLECTIVE_COMPLETE = 26,   // Phase 14: Master signal that reduction is ready
  BANDWIDTH_PROBE = 27,       // Master→Worker: 64 KB payload + 8-byte timestamp header
  BANDWIDTH_ACK = 28,         // Worker→Master: echo timestamp + worker send time
  DATA_HEADER_RDMA = 29,      // RDMA zero-copy notification: data written to bounce buffer
  SECURE_READY = 30,          // Master→Worker: post-handshake readiness barrier (encrypted)
  CLOCK_SYNC = 31,            // MT.6: master→worker: T1 timestamp for NTP-style calibration
  CLOCK_SYNC_REPLY = 32,      // MT.6: worker→master: T1+T2+T3 for offset computation
};

// MT.6: clock synchronization payloads (NTP-style 3-timestamp exchange)
#pragma pack(push, 1)
struct ClockSyncPayload {
  int64_t t1_us; // master's send time (microseconds since epoch)
};
struct ClockSyncReplyPayload {
  int64_t t1_us; // echo of master's T1
  int64_t t2_us; // worker's receive time
  int64_t t3_us; // worker's reply-send time
};
#pragma pack(pop)

#pragma pack(push, 1)
struct VSBPHeader {
  uint32_t magic;      // Explicit 32-bit magic
  uint16_t version;    // Platform-independent versioning
  uint16_t type;       // PacketType as uint16
  uint32_t sequence;   // Message sequencing
  uint64_t payloadSize; // 64-bit size for massive transfers
  uint8_t platform;    // Source platform for mesh topology validation
  uint8_t reserved[3]; // Padding for 8-byte alignment
};
#pragma pack(pop)

// Cross-platform memory alignment validation for VSBPHeader
static_assert(detail::validate_struct_size<VSBPHeader>(24), 
              "VSBPHeader must be exactly 24 bytes across all platforms for mesh topology");
static_assert(detail::validate_struct_alignment<VSBPHeader>(8),
              "VSBPHeader alignment must not exceed 8 bytes for cross-platform compatibility");
// Note: Endianness validation is performed at runtime in MeshTopologyManager::validateCrossPlatformSupport()

// Mesh topology platform compatibility validation
static_assert(detail::detect_platform() != detail::PlatformType::UNKNOWN,
              "Platform must be Windows, Linux, or macOS for mesh topology support");

enum class ReceiveState : uint8_t {
  IDLE = 0,
  EXPECTING_RANGES_TCP = 1,
  EXPECTING_RANGES_SHM = 2,
  EXPECTING_BODY = 3,
  EXPECTING_KERNEL_SOURCE = 4,
  EXPECTING_STRUCT_BODY = 5
};

#pragma pack(push, 1)
struct ShmInitPacket {
  char shm_name[64];
  uint64_t shm_size;
};
#pragma pack(pop)

// Cross-platform validation for ShmInitPacket
static_assert(detail::validate_struct_size<ShmInitPacket>(72),
              "ShmInitPacket must be exactly 72 bytes across all platforms");

#pragma pack(push, 1)
struct DataShmPacket {
  uint64_t target_ptr;
  uint64_t shm_offset;
  uint64_t size;
};
#pragma pack(pop)

// Cross-platform validation for DataShmPacket
static_assert(detail::validate_struct_size<DataShmPacket>(24),
              "DataShmPacket must be exactly 24 bytes across all platforms");

#pragma pack(push, 1)
struct DataHeaderDirtyPacket {
  uint64_t target_ptr;
  uint32_t num_ranges;
};
#pragma pack(pop)

// Cross-platform validation for DataHeaderDirtyPacket
static_assert(detail::validate_struct_size<DataHeaderDirtyPacket>(12),
              "DataHeaderDirtyPacket must be exactly 12 bytes across all platforms");

#pragma pack(push, 1)
struct DataShmDirtyPacket {
  uint64_t target_ptr;
  uint32_t num_ranges;
  uint64_t shm_offset;
};
#pragma pack(pop)

// Cross-platform validation for DataShmDirtyPacket
static_assert(detail::validate_struct_size<DataShmDirtyPacket>(20),
              "DataShmDirtyPacket must be exactly 20 bytes across all platforms");

#pragma pack(push, 1)
struct DirtyRangePacket {
  uint64_t offset;
  uint64_t size;
};
#pragma pack(pop)

// Cross-platform validation for DirtyRangePacket
static_assert(detail::validate_struct_size<DirtyRangePacket>(16),
              "DirtyRangePacket must be exactly 16 bytes across all platforms");

#pragma pack(push, 1)
struct KernelRegisterPacket {
  uint64_t auth_token;
  uint64_t kernel_id;
  char name[64];
  uint32_t source_len;
};
#pragma pack(pop)

// Cross-platform validation for KernelRegisterPacket
static_assert(detail::validate_struct_size<KernelRegisterPacket>(84),
              "KernelRegisterPacket must be exactly 84 bytes across all platforms");

#pragma pack(push, 1)
struct TelemetryPacket {
  vgre_telemetry_t telemetry;
};
#pragma pack(pop)

// Note: TelemetryPacket size depends on vgre_telemetry_t which is defined in C API
// Validation is performed at runtime in packet handler

#pragma pack(push, 1)
struct RemoteCommandPacket {
  uint64_t auth_token;
  uint64_t kernel_id;
  uint32_t grid_dim[3];
  uint32_t block_dim[3];
  size_t shared_mem;
  int num_args;
};
#pragma pack(pop)

// Cross-platform validation for RemoteCommandPacket
// Note: size_t and int may vary by platform, so we validate at runtime

#pragma pack(push, 1)
struct ArgScalarPacket {
  uint32_t arg_index;
  uint8_t arg_type;
  uint64_t value;
};
#pragma pack(pop)

// Cross-platform validation for ArgScalarPacket
static_assert(detail::validate_struct_size<ArgScalarPacket>(13),
              "ArgScalarPacket must be exactly 13 bytes across all platforms");

#pragma pack(push, 1)
struct StructDataPacket {
  uint32_t arg_index;
  uint32_t size;
};
#pragma pack(pop)

// Cross-platform validation for StructDataPacket
static_assert(detail::validate_struct_size<StructDataPacket>(8),
              "StructDataPacket must be exactly 8 bytes across all platforms");

#pragma pack(push, 1)
struct DataHeaderPacket {
  uint64_t target_ptr;
  uint64_t size;
};
#pragma pack(pop)

// Cross-platform validation for DataHeaderPacket
static_assert(detail::validate_struct_size<DataHeaderPacket>(16),
              "DataHeaderPacket must be exactly 16 bytes across all platforms");

// Sent after an RDMA WRITE completes: notifies the receiver that `size` bytes
// are now available at the base of its pre-registered bounce buffer and should
// be copied to the allocation identified by `target_ptr`.
// The receiver must NOT expect a DATA_BODY packet after this.
#pragma pack(push, 1)
struct DataHeaderRDMAPacket {
  uint64_t target_ptr;   // VGRE allocation handle on the receiver
  uint64_t size;         // bytes already written into bounce buffer[0..size)
};
#pragma pack(pop)

// Cross-platform validation for DataHeaderRDMAPacket
static_assert(detail::validate_struct_size<DataHeaderRDMAPacket>(16),
              "DataHeaderRDMAPacket must be exactly 16 bytes across all platforms");

#pragma pack(push, 1)
struct ResponsePacket {
  uint64_t kernel_id;
  VGREResult result;
};
#pragma pack(pop)

// Note: ResponsePacket size depends on VGREResult enum size, validated at runtime

#pragma pack(push, 1)
struct CapabilityPacket {
  // CPU
  int cpu_cores;
  uint64_t cpu_memory;           // bytes of RAM
  // iGPU (OpenCL / integrated)
  bool has_igpu;
  char igpu_name[64];
  // Discrete NVIDIA GPU (via GPUPassthrough dlopen probe)
  int  gpu_count;                // 0 = CPU-only worker
  char gpu_name[128];            // primary GPU name
  uint64_t gpu_memory_bytes;     // primary GPU VRAM in bytes (0 = unknown)
  int  gpu_compute_major;        // CUDA compute capability major
  int  gpu_compute_minor;        // CUDA compute capability minor
  int  gpu_sm_count;             // streaming multiprocessor count
};
#pragma pack(pop)

// Note: CapabilityPacket contains platform-dependent int and bool sizes, validated at runtime

#pragma pack(push, 1)
struct SecureHandshakePacket {
  uint8_t nonce[crypto::kNonceLen];
  uint8_t key_verification[crypto::kSHA256DigestLen];
};
#pragma pack(pop)

// Cross-platform validation for SecureHandshakePacket
// Note: Size depends on crypto constants, validated at runtime

// Bandwidth probe: master sends 64 KB payload + 8-byte timestamp prefix.
// Worker echoes back BandwidthAckPacket so master can measure round-trip.
#pragma pack(push, 1)
struct BandwidthAckPacket {
  uint64_t probe_sent_ms;  // echoed timestamp from the master's BANDWIDTH_PROBE
  uint64_t ack_sent_ms;    // worker's wall-clock ms when it sent this ACK
};
#pragma pack(pop)

// Cross-platform validation for BandwidthAckPacket
static_assert(detail::validate_struct_size<BandwidthAckPacket>(16),
              "BandwidthAckPacket must be exactly 16 bytes across all platforms");

#pragma pack(push, 1)
struct PartitionDispatchPacket {
  uint64_t auth_token;
  uint64_t kernel_id;
  uint32_t full_grid_dim[3];
  uint32_t partition_grid_dim[3];
  uint32_t block_dim[3];
  uint32_t grid_start[3];
  uint64_t shared_mem;
  int num_args;
  uint32_t partition_id;
  uint32_t total_partitions;
};
#pragma pack(pop)

// Note: PartitionDispatchPacket contains platform-dependent int size, validated at runtime

#pragma pack(push, 1)
struct PartitionResultPacket {
  uint64_t kernel_id;
  uint32_t partition_id;
  VGREResult result;
  double execution_time_ms;
};
#pragma pack(pop)

// Note: PartitionResultPacket size depends on VGREResult and double alignment, validated at runtime

#pragma pack(push, 1)
struct CreditReportPacket {
  double compute_seconds;
  int cpu_cores;
  uint64_t kernel_id;
  uint64_t timestamp;
};
#pragma pack(pop)

// Note: CreditReportPacket contains platform-dependent int and double sizes, validated at runtime

// Reduction operations for collective ops (encoded in upper 16 bits of op_type)
enum class ReductionOp : uint16_t {
    Sum  = 0,
    Prod = 1,
    Max  = 2,
    Min  = 3,
    Avg  = 4
};

inline uint32_t encodeCollectiveOpType(uint16_t collectiveType, ReductionOp redOp) {
    return (static_cast<uint32_t>(static_cast<uint16_t>(redOp)) << 16) | collectiveType;
}
inline uint16_t decodeCollectiveType(uint32_t opType) { return static_cast<uint16_t>(opType & 0xFFFF); }
inline ReductionOp decodeReductionOp(uint32_t opType) { return static_cast<ReductionOp>((opType >> 16) & 0xFFFF); }

#pragma pack(push, 1)
struct CollectiveOpPacket {
  uint32_t op_type;    // lower 16 bits = collective type (0 = all_reduce), upper 16 bits = ReductionOp
  uint32_t datatype;   // VGRE_ARG_...
  uint64_t count;
  uint64_t sequence;   // sync ID
};
#pragma pack(pop)

// Cross-platform validation for CollectiveOpPacket
static_assert(detail::validate_struct_size<CollectiveOpPacket>(24),
              "CollectiveOpPacket must be exactly 24 bytes across all platforms");

} // namespace advanced
} // namespace vgre
