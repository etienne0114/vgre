#ifndef VGRE_ADVANCED_RDMA_TRANSPORT_H
#define VGRE_ADVANCED_RDMA_TRANSPORT_H

// RDMA / RoCE zero-copy cluster transport.
//
// Build with: cmake -DVGRE_ENABLE_RDMA=ON (requires libibverbs-dev)
// At runtime, tryCreate() returns nullptr if no RDMA-capable NIC is present;
// callers must fall back to the existing TCP path.
//
// Architecture:
//   RDMAContext    — one per process: owns ibv_context, protection domain, CQ
//   RDMARegion     — one per registered buffer: ibv_mr* + rkey for remote access
//   RDMAConnection — one per peer: QP in RTS state + pre-registered bounce buffer
//
// Bounce buffer design:
//   Each RDMAConnection allocates a large pre-pinned bounce buffer during
//   connect().  The local buffer receives data written by the remote peer.
//   The remote peer's bounce buffer rkey/addr are exchanged during the QP
//   handshake so each side can RDMA-WRITE directly into the other's buffer.
//
// Usage (master → worker data transfer):
//   conn.connect(secureChannel, *ctx);       // QP handshake + bounce buffer exchange
//   conn.rdmaWriteToRemote(ctx, ptr, size);  // write into worker's bounce buffer
//   send_packet(fd, DATA_HEADER_RDMA, ...);  // TCP notification: data ready at bounce[0]
//   // Worker side: on DATA_HEADER_RDMA, memcpy from conn.bounceBuf() to allocation

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <string>

#include "vgre/common/sockets.h"   // vgre_socket_t

#ifdef VGRE_HAS_RDMA
#include <infiniband/verbs.h>
#endif

namespace vgre {

// Forward declaration so the header compiles without SecureChannel include
namespace advanced { class SecureChannel; }

namespace advanced {

// ── Registered memory region ───────────────────────────────────────────────────
struct RDMARegion {
#ifdef VGRE_HAS_RDMA
    ibv_mr*  mr     = nullptr;
#endif
    void*    addr   = nullptr;
    size_t   length = 0;
    uint32_t lkey   = 0;  // local key (for local RDMA sends)
    uint32_t rkey   = 0;  // remote key (shared with peer for remote writes)
};

// ── Peer QP address info (exchanged via TCP control channel) ───────────────────
struct RDMAQPInfo {
    uint16_t lid;       // InfiniBand LID (0 for RoCE)
    uint32_t qpn;       // Queue Pair Number
    uint32_t psn;       // Packet Sequence Number
    uint8_t  gid[16];   // GID (for RoCE / GRH routing)
    uint32_t rkey;      // rkey of the remote target buffer
    uint64_t remoteAddr;// VA of remote registered buffer
};

// ── RDMA context ───────────────────────────────────────────────────────────────
class RDMAContext {
public:
    ~RDMAContext();

    // Try to open the first available RDMA device.
    // Returns nullptr if no device/libibverbs available; caller falls back to TCP.
    static RDMAContext* tryCreate();

    // Register a host buffer for RDMA.  The registration pins the pages and
    // grants the NIC DMA access.  Must be deregistered before the buffer is freed.
    RDMARegion* registerMemory(void* ptr, size_t size);
    void        deregisterMemory(RDMARegion* region);

    // ── Zero-copy pre-registration for cudaMalloc blocks ─────────────────────
    // Called at cudaMalloc time to pin memory immediately.
    // Math: ibv_reg_mr cost = O(n/4K) page-table walks (blocked syscall).
    //       Pre-registering at alloc time amortises this cost; rdmaWriteToRemote
    //       then uses the cached region in O(1) instead of O(n/4K).
    // Invariant: each (ptr,size) pair maps to at most one live RDMARegion*.
    void        preRegisterAllocation(void* ptr, size_t size);
    void        deregisterAllocation(void* ptr);

    // Look up a previously pre-registered region.  Returns nullptr if the
    // pointer was not pre-registered (caller falls back to on-demand reg_mr).
    RDMARegion* lookupCachedRegion(void* ptr);

    // Accessors for RDMAConnection
#ifdef VGRE_HAS_RDMA
    ibv_context* ctx() const { return ctx_; }
    ibv_pd*      pd()  const { return pd_;  }
    ibv_cq*      cq()  const { return cq_;  }
#endif

    std::string deviceName() const { return deviceName_; }

private:
    RDMAContext() = default;

#ifdef VGRE_HAS_RDMA
    ibv_context* ctx_ = nullptr;
    ibv_pd*      pd_  = nullptr;
    ibv_cq*      cq_  = nullptr;
#endif
    std::string deviceName_;

    // Zero-copy allocation cache: ptr → pre-registered RDMARegion*.
    // Protected by allocCacheMu_. O(1) lookup via hash map.
    std::mutex                          allocCacheMu_;
    std::unordered_map<void*, RDMARegion*> allocCache_;
};

// ── RDMA connection (one per peer) ────────────────────────────────────────────
class RDMAConnection {
public:
    RDMAConnection()  = default;
    ~RDMAConnection();

    // Exchange QP info and bounce buffer rkey/addr with a peer via the existing
    // authenticated TCP channel (fd), then transition: RESET → INIT → RTR → RTS.
    // Allocates a pre-pinned bounce buffer (size controlled by VGRE_RDMA_BOUNCE_SIZE,
    // default 256 MB) and includes its rkey/addr in the QP info exchange so the
    // remote peer can RDMA-WRITE directly into it.
    // fd must be the socket that backs ctrl; sendSecure/recvSecure use it for I/O.
    // Returns false on any failure; caller should continue with TCP.
    bool connect(SecureChannel& ctrl, vgre::common::vgre_socket_t fd, RDMAContext& ctx);

    // Write src[0..size) into the remote peer's bounce buffer using RDMA WRITE.
    // Registers src temporarily, posts the WR, spin-polls until complete, then
    // deregisters. Thread-safe: serialized internally by writeMtx_.
    // Returns false if RDMA is unavailable, size exceeds the remote bounce capacity,
    // or the NIC reports an error; callers must fall back to TCP.
    bool rdmaWriteToRemote(RDMAContext& ctx, const void* src, size_t size);

    // Post an RDMA WRITE of src_mr->addr[0..size) to an explicit remote address.
    // Low-level primitive used by rdmaWriteToRemote; callers prefer that function.
    bool rdmaWrite(const RDMARegion* src, uint64_t remoteAddr,
                   uint32_t rkey, size_t size);

    // Spin-poll the completion queue for up to timeoutMs milliseconds.
    // Returns bytes transferred (size of the last posted WR), or 0 on error/timeout.
    size_t pollCompletion(int timeoutMs = 5000);

    bool   isConnected()    const { return connected_;   }
    void*  bounceBuf()      const { return bouncePtr_;   }
    size_t bounceCapacity() const { return bounceSize_;  }

private:
#ifdef VGRE_HAS_RDMA
    bool createQP(RDMAContext& ctx);
    bool transitionToInit();
    bool transitionToRTR(const RDMAQPInfo& remote);
    bool transitionToRTS();

    ibv_qp*  qp_  = nullptr;
    ibv_cq*  cq_  = nullptr;    // borrowed from RDMAContext
    uint32_t psn_ = 0;

    // Bounce buffer: pre-registered MR that the remote peer RDMA-WRITEs into.
    // Allocated and registered in connect(); freed in ~RDMAConnection.
    RDMAContext* ctxPtr_     = nullptr;  // borrowed reference for deregistration
    void*        bouncePtr_  = nullptr;
    size_t       bounceSize_ = 0;
    RDMARegion*  bounceMR_   = nullptr;

    // Remote peer's bounce buffer info (received during QP info exchange).
    uint64_t     remoteAddr_ = 0;
    uint32_t     remoteRkey_ = 0;

    // Serialize RDMA writes to the remote bounce buffer — prevents two
    // concurrent rdmaWriteToRemote() calls from corrupting each other.
    std::mutex   writeMtx_;
#else
    void*  bouncePtr_  = nullptr;
    size_t bounceSize_ = 0;
#endif
    bool     connected_  = false;
    size_t   lastWRSize_ = 0;   // size of the last posted WR (for pollCompletion)
};

} // namespace advanced
} // namespace vgre

#endif // VGRE_ADVANCED_RDMA_TRANSPORT_H
