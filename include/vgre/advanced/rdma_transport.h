#ifndef VGRE_ADVANCED_RDMA_TRANSPORT_H
#define VGRE_ADVANCED_RDMA_TRANSPORT_H

// RDMA / RoCE zero-copy cluster transport.
//
// Build with: cmake -DVGRE_ENABLE_RDMA=ON (requires libibverbs-dev)
// At runtime, tryCreate() returns nullptr if no RDMA-capable NIC is present;
// callers must fall back to the existing TCP path.
//
// Architecture:
//   RDMAContext  — one per process: owns ibv_context, protection domain, CQ
//   RDMARegion   — one per registered buffer: ibv_mr* + rkey for remote access
//   RDMAConnection — one per peer: Queue Pair (QP) in RTS state after connect()
//
// Usage:
//   auto* ctx = RDMAContext::tryCreate();
//   if (!ctx) { /* use TCP */ }
//   auto* mr = ctx->registerMemory(buf, size);
//   RDMAConnection conn;
//   conn.connect(existingSecureChannel, *ctx);  // exchange QP info via TCP
//   conn.rdmaWrite(mr, remoteMr.addr, remoteMr.rkey, size);
//   conn.pollCompletion();
//   ctx->deregisterMemory(mr);

#include <cstddef>
#include <cstdint>
#include <string>

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
};

// ── RDMA connection (one per peer) ────────────────────────────────────────────
class RDMAConnection {
public:
    RDMAConnection()  = default;
    ~RDMAConnection();

    // Exchange QP info with a peer via the existing authenticated TCP channel,
    // then transition the local QP: RESET → INIT → RTR → RTS.
    // Returns false if RDMA negotiation fails (peer has no RDMA, or hardware
    // mismatch); caller should continue with TCP for this peer.
    bool connect(SecureChannel& ctrl, RDMAContext& ctx);

    // Post an RDMA WRITE of src_mr->addr[0..size) to the remote peer's buffer.
    // Non-blocking: posts the WR to the send queue and returns immediately.
    // Call pollCompletion() to wait for the NIC to confirm delivery.
    bool rdmaWrite(const RDMARegion* src, uint64_t remoteAddr,
                   uint32_t rkey, size_t size);

    // Spin-poll the completion queue for up to timeoutMs milliseconds.
    // Returns the number of bytes transferred (size from the WR), or 0 on timeout/error.
    size_t pollCompletion(int timeoutMs = 5000);

    bool isConnected() const { return connected_; }

private:
#ifdef VGRE_HAS_RDMA
    bool createQP(RDMAContext& ctx);
    bool transitionToInit();
    bool transitionToRTR(const RDMAQPInfo& remote);
    bool transitionToRTS();

    ibv_qp*  qp_  = nullptr;
    ibv_cq*  cq_  = nullptr;    // borrowed from RDMAContext
    uint32_t psn_ = 0;
#endif
    bool     connected_  = false;
    size_t   lastWRSize_ = 0;   // size of the last posted WR (for pollCompletion)
};

} // namespace advanced
} // namespace vgre

#endif // VGRE_ADVANCED_RDMA_TRANSPORT_H
