#include "vgre/advanced/rdma_transport.h"
#include "vgre/advanced/secure_channel.h"
#include "vgre/common/logger.h"

#include <chrono>
#include <cstring>
#include <cstdlib>
#include <random>

#ifdef VGRE_HAS_RDMA
#include <infiniband/verbs.h>
#include <sys/mman.h>   // mmap, munmap

namespace {
// Bounce buffer size: controls the maximum single RDMA transfer size.
// Configurable via VGRE_RDMA_BOUNCE_SIZE (bytes). Default 256 MB.
size_t getRdmaBounceBufSize() {
    const char* env = std::getenv("VGRE_RDMA_BOUNCE_SIZE");
    if (env) {
        try {
            long long v = std::stoll(env);
            // Minimum 4 MB, maximum 4 GB (practical NIC limit)
            if (v >= 4 * 1024 * 1024LL && v <= 4LL * 1024 * 1024 * 1024LL)
                return static_cast<size_t>(v);
        } catch (...) {}
    }
    return 256ULL * 1024 * 1024;  // 256 MB
}
} // namespace

// ─────────────────────────────────────────────────────────────────────────────
// Helper: serialize / deserialize RDMAQPInfo over the SecureChannel
// ─────────────────────────────────────────────────────────────────────────────
namespace {

bool sendQPInfo(vgre::advanced::SecureChannel& ch,
                const vgre::advanced::RDMAQPInfo& info)
{
    return ch.send(reinterpret_cast<const uint8_t*>(&info), sizeof(info));
}

bool recvQPInfo(vgre::advanced::SecureChannel& ch,
                vgre::advanced::RDMAQPInfo& info)
{
    return ch.receive(reinterpret_cast<uint8_t*>(&info), sizeof(info)) == sizeof(info);
}

// Fill GID for the first active port on the device.
bool queryGID(ibv_context* ctx, uint8_t portNum, uint8_t gid[16]) {
    ibv_gid g{};
    if (ibv_query_gid(ctx, portNum, 0, &g) != 0) return false;
    std::memcpy(gid, g.raw, 16);
    return true;
}

} // namespace

// ─────────────────────────────────────────────────────────────────────────────
// RDMAContext
// ─────────────────────────────────────────────────────────────────────────────

namespace vgre {
namespace advanced {

RDMAContext::~RDMAContext() {
    if (cq_)  { ibv_destroy_cq(cq_);            cq_  = nullptr; }
    if (pd_)  { ibv_dealloc_pd(pd_);            pd_  = nullptr; }
    if (ctx_) { ibv_close_device(ctx_);         ctx_ = nullptr; }
}

RDMAContext* RDMAContext::tryCreate() {
    // Enumerate available RDMA devices
    int numDevices = 0;
    ibv_device** devList = ibv_get_device_list(&numDevices);
    if (!devList || numDevices == 0) {
        VGRE_LOG_INFO("RDMATransport", "No RDMA devices found — TCP fallback active");
        if (devList) ibv_free_device_list(devList);
        return nullptr;
    }

    auto* rdma = new RDMAContext();

    // Use the first device
    rdma->deviceName_ = ibv_get_device_name(devList[0]);
    rdma->ctx_ = ibv_open_device(devList[0]);
    ibv_free_device_list(devList);

    if (!rdma->ctx_) {
        VGRE_LOG_WARN("RDMATransport", "Failed to open RDMA device — TCP fallback active");
        delete rdma;
        return nullptr;
    }

    // Allocate protection domain
    rdma->pd_ = ibv_alloc_pd(rdma->ctx_);
    if (!rdma->pd_) {
        VGRE_LOG_WARN("RDMATransport", "ibv_alloc_pd failed — TCP fallback active");
        delete rdma;
        return nullptr;
    }

    // Create completion queue (shared across all connections on this context)
    rdma->cq_ = ibv_create_cq(rdma->ctx_,
                                /*cqe=*/256,
                                /*cq_context=*/nullptr,
                                /*channel=*/nullptr,
                                /*comp_vector=*/0);
    if (!rdma->cq_) {
        VGRE_LOG_WARN("RDMATransport", "ibv_create_cq failed — TCP fallback active");
        delete rdma;
        return nullptr;
    }

    VGRE_LOG_INFO("RDMATransport",
                  "RDMA context opened on device: " + rdma->deviceName_);
    return rdma;
}

RDMARegion* RDMAContext::registerMemory(void* ptr, size_t size) {
    if (!pd_ || !ptr || size == 0) return nullptr;

    int access = IBV_ACCESS_LOCAL_WRITE
               | IBV_ACCESS_REMOTE_WRITE
               | IBV_ACCESS_REMOTE_READ;

    ibv_mr* mr = ibv_reg_mr(pd_, ptr, size, access);
    if (!mr) {
        VGRE_LOG_ERROR("RDMATransport",
                       "ibv_reg_mr failed for " + std::to_string(size) + " bytes");
        return nullptr;
    }

    auto* region   = new RDMARegion();
    region->mr     = mr;
    region->addr   = ptr;
    region->length = size;
    region->lkey   = mr->lkey;
    region->rkey   = mr->rkey;

    VGRE_LOG_DEBUG("RDMATransport",
                   "Registered " + std::to_string(size) + " bytes, rkey=" +
                   std::to_string(mr->rkey));
    return region;
}

void RDMAContext::deregisterMemory(RDMARegion* region) {
    if (!region) return;
    if (region->mr) ibv_dereg_mr(region->mr);
    delete region;
}

// ─────────────────────────────────────────────────────────────────────────────
// RDMAConnection
// ─────────────────────────────────────────────────────────────────────────────

RDMAConnection::~RDMAConnection() {
    if (qp_) { ibv_destroy_qp(qp_); qp_ = nullptr; }
    if (bounceMR_ && ctxPtr_) { ctxPtr_->deregisterMemory(bounceMR_); bounceMR_ = nullptr; }
    if (bouncePtr_) { munmap(bouncePtr_, bounceSize_); bouncePtr_ = nullptr; bounceSize_ = 0; }
}

bool RDMAConnection::createQP(RDMAContext& ctx) {
    ibv_qp_init_attr attr{};
    attr.send_cq          = ctx.cq();
    attr.recv_cq          = ctx.cq();
    attr.qp_type          = IBV_QPT_RC;   // Reliable Connected (matches RoCE)
    attr.cap.max_send_wr  = 64;
    attr.cap.max_recv_wr  = 64;
    attr.cap.max_send_sge = 1;
    attr.cap.max_recv_sge = 1;
    attr.sq_sig_all       = 1;           // signal completion for every send

    cq_ = ctx.cq();
    qp_ = ibv_create_qp(ctx.pd(), &attr);
    if (!qp_) {
        VGRE_LOG_ERROR("RDMATransport", "ibv_create_qp failed");
        return false;
    }
    // Cryptographically-seeded PSN — std::random_device uses /dev/urandom on Linux
    std::random_device rd;
    psn_ = rd() & 0xFFFFFF;
    return true;
}

bool RDMAConnection::transitionToInit() {
    ibv_qp_attr attr{};
    attr.qp_state   = IBV_QPS_INIT;
    attr.pkey_index = 0;
    attr.port_num   = 1;   // first port
    attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ
                         | IBV_ACCESS_LOCAL_WRITE;

    int flags = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
    if (ibv_modify_qp(qp_, &attr, flags) != 0) {
        VGRE_LOG_ERROR("RDMATransport", "QP RESET→INIT failed");
        return false;
    }
    return true;
}

bool RDMAConnection::transitionToRTR(const RDMAQPInfo& remote) {
    ibv_qp_attr attr{};
    attr.qp_state              = IBV_QPS_RTR;
    attr.path_mtu              = IBV_MTU_4096;
    attr.dest_qp_num           = remote.qpn;
    attr.rq_psn                = remote.psn;
    attr.max_dest_rd_atomic    = 1;
    attr.min_rnr_timer         = 12;

    attr.ah_attr.is_global     = 0;
    attr.ah_attr.dlid          = remote.lid;
    attr.ah_attr.sl            = 0;
    attr.ah_attr.src_path_bits = 0;
    attr.ah_attr.port_num      = 1;

    // RoCE uses GRH (global routing header) when lid == 0
    if (remote.lid == 0) {
        attr.ah_attr.is_global = 1;
        std::memcpy(attr.ah_attr.grh.dgid.raw, remote.gid, 16);
        attr.ah_attr.grh.sgid_index    = 0;
        attr.ah_attr.grh.hop_limit     = 64;
        attr.ah_attr.grh.traffic_class = 0;
        attr.ah_attr.grh.flow_label    = 0;
    }

    int flags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN
              | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;

    if (ibv_modify_qp(qp_, &attr, flags) != 0) {
        VGRE_LOG_ERROR("RDMATransport", "QP INIT→RTR failed");
        return false;
    }
    return true;
}

bool RDMAConnection::transitionToRTS() {
    ibv_qp_attr attr{};
    attr.qp_state      = IBV_QPS_RTS;
    attr.sq_psn        = psn_;
    attr.timeout       = 14;    // ~67 ms local timeout
    attr.retry_cnt     = 7;
    attr.rnr_retry     = 7;     // infinite RNR retry
    attr.max_rd_atomic = 1;

    int flags = IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT
              | IBV_QP_RNR_RETRY | IBV_QP_MAX_QP_RD_ATOMIC;

    if (ibv_modify_qp(qp_, &attr, flags) != 0) {
        VGRE_LOG_ERROR("RDMATransport", "QP RTR→RTS failed");
        return false;
    }
    return true;
}

bool RDMAConnection::connect(SecureChannel& ctrl, RDMAContext& ctx) {
    if (!createQP(ctx))      return false;
    if (!transitionToInit()) return false;

    // ── Allocate and register bounce buffer ──────────────────────────────────
    // The bounce buffer is a large pre-pinned anonymous memory region that the
    // remote peer RDMA-WRITEs directly into. Its rkey and virtual address are
    // included in the QP info exchange so the remote knows where to write.
    const size_t bsz = getRdmaBounceBufSize();
    void* bptr = mmap(nullptr, bsz, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (bptr == MAP_FAILED) {
        VGRE_LOG_ERROR("RDMATransport",
                       "mmap failed for " + std::to_string(bsz) + "-byte bounce buffer");
        return false;
    }
    RDMARegion* bmr = ctx.registerMemory(bptr, bsz);
    if (!bmr) {
        munmap(bptr, bsz);
        VGRE_LOG_ERROR("RDMATransport", "ibv_reg_mr failed for bounce buffer");
        return false;
    }
    bouncePtr_  = bptr;
    bounceSize_ = bsz;
    bounceMR_   = bmr;
    ctxPtr_     = &ctx;

    // ── Query local port info for LID and GID ────────────────────────────────
    ibv_port_attr portAttr{};
    if (ibv_query_port(ctx.ctx(), 1, &portAttr) != 0) {
        VGRE_LOG_ERROR("RDMATransport", "ibv_query_port failed");
        return false;
    }

    RDMAQPInfo local{};
    local.lid        = portAttr.lid;
    local.qpn        = qp_->qp_num;
    local.psn        = psn_;
    local.rkey       = bounceMR_->rkey;
    local.remoteAddr = reinterpret_cast<uint64_t>(bouncePtr_);
    queryGID(ctx.ctx(), 1, local.gid);

    // Exchange QP info (including bounce buffer rkey/addr) over the existing
    // authenticated TCP/SecureChannel. Both sides send first, then receive,
    // to avoid a deadlock when both peers call connect() simultaneously.
    if (!sendQPInfo(ctrl, local)) {
        VGRE_LOG_ERROR("RDMATransport", "Failed to send local QP info");
        return false;
    }

    RDMAQPInfo remote{};
    if (!recvQPInfo(ctrl, remote)) {
        VGRE_LOG_ERROR("RDMATransport", "Failed to receive remote QP info");
        return false;
    }

    // Store the remote peer's bounce buffer so rdmaWriteToRemote can write into it.
    remoteAddr_ = remote.remoteAddr;
    remoteRkey_ = remote.rkey;

    if (!transitionToRTR(remote)) return false;
    if (!transitionToRTS())       return false;

    connected_ = true;
    VGRE_LOG_INFO("RDMATransport",
                  "QP connected: local qpn=" + std::to_string(local.qpn) +
                  " remote qpn=" + std::to_string(remote.qpn) +
                  " bounce=" + std::to_string(bsz / (1024*1024)) + " MB");
    return true;
}

bool RDMAConnection::rdmaWrite(const RDMARegion* src, uint64_t remoteAddr,
                                uint32_t rkey, size_t size)
{
    if (!connected_ || !qp_ || !src || size == 0) return false;

    ibv_sge sge{};
    sge.addr   = reinterpret_cast<uint64_t>(src->addr);
    sge.length = static_cast<uint32_t>(size);
    sge.lkey   = src->lkey;

    ibv_send_wr wr{};
    wr.wr_id                  = 1;
    wr.sg_list                = &sge;
    wr.num_sge                = 1;
    wr.opcode                 = IBV_WR_RDMA_WRITE;
    wr.send_flags             = IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr    = remoteAddr;
    wr.wr.rdma.rkey           = rkey;

    ibv_send_wr* bad_wr = nullptr;
    if (ibv_post_send(qp_, &wr, &bad_wr) != 0) {
        VGRE_LOG_ERROR("RDMATransport", "ibv_post_send failed");
        return false;
    }
    lastWRSize_ = size;
    return true;
}

size_t RDMAConnection::pollCompletion(int timeoutMs) {
    if (!cq_) return 0;

    auto deadline = std::chrono::steady_clock::now()
                  + std::chrono::milliseconds(timeoutMs);

    ibv_wc wc{};
    while (std::chrono::steady_clock::now() < deadline) {
        int n = ibv_poll_cq(cq_, 1, &wc);
        if (n < 0) {
            VGRE_LOG_ERROR("RDMATransport", "ibv_poll_cq error");
            return 0;
        }
        if (n == 1) {
            if (wc.status != IBV_WC_SUCCESS) {
                VGRE_LOG_ERROR("RDMATransport",
                               "WC error: " + std::string(ibv_wc_status_str(wc.status)));
                return 0;
            }
            return lastWRSize_;
        }
        // Spin a bit before polling again — avoids kernel involvement
        for (volatile int i = 0; i < 1000; ++i) {}
    }

    VGRE_LOG_WARN("RDMATransport",
                  "pollCompletion timed out after " + std::to_string(timeoutMs) + " ms");
    return 0;
}

bool RDMAConnection::rdmaWriteToRemote(RDMAContext& ctx, const void* src, size_t size) {
    if (!connected_ || remoteAddr_ == 0 || remoteRkey_ == 0 || !src || size == 0)
        return false;

    if (size > bounceSize_) {
        VGRE_LOG_WARN("RDMATransport",
                      "rdmaWriteToRemote: size " + std::to_string(size) +
                      " exceeds remote bounce capacity " + std::to_string(bounceSize_) +
                      " — falling back to TCP");
        return false;
    }

    // One RDMA write at a time — prevents concurrent calls from overlapping
    // writes into the same remote bounce buffer region.
    std::lock_guard<std::mutex> lk(writeMtx_);

    // Register the source buffer for this transfer.
    // ibv_reg_mr pins the pages and gives the NIC DMA access; ibv_dereg_mr
    // unpins them. This is intentionally per-call for correctness — caching
    // MRs across calls would require tracking buffer lifetimes.
    RDMARegion* srcMR = ctx.registerMemory(const_cast<void*>(src), size);
    if (!srcMR) {
        VGRE_LOG_ERROR("RDMATransport",
                       "rdmaWriteToRemote: ibv_reg_mr failed for " +
                       std::to_string(size) + " bytes");
        return false;
    }

    // Post RDMA WRITE WR: write srcMR → remote bounce buffer at base address.
    bool ok = rdmaWrite(srcMR, remoteAddr_, remoteRkey_, size);
    if (ok) {
        size_t transferred = pollCompletion(5000);
        ok = (transferred == size);
        if (!ok) {
            VGRE_LOG_ERROR("RDMATransport",
                           "rdmaWriteToRemote: completion poll returned " +
                           std::to_string(transferred) + " expected " +
                           std::to_string(size));
        }
    }

    ctx.deregisterMemory(srcMR);
    return ok;
}

} // namespace advanced
} // namespace vgre

#else // !VGRE_HAS_RDMA ─────────────────────────────────────────────────────────
// Fallback implementations when RDMA support is not compiled in.
// RDMAContext::tryCreate() returns nullptr; all operations are no-ops that
// return false/nullptr so callers transparently fall back to TCP.

namespace vgre {
namespace advanced {

RDMAContext::~RDMAContext() {}

RDMAContext* RDMAContext::tryCreate() {
    VGRE_LOG_DEBUG("RDMATransport",
                   "RDMA support not compiled (build with -DVGRE_ENABLE_RDMA=ON)");
    return nullptr;
}

RDMARegion* RDMAContext::registerMemory(void*, size_t)   { return nullptr; }
void        RDMAContext::deregisterMemory(RDMARegion* r)  { delete r; }

RDMAConnection::~RDMAConnection() {}
bool   RDMAConnection::connect(SecureChannel&, RDMAContext&)                      { return false; }
bool   RDMAConnection::rdmaWrite(const RDMARegion*, uint64_t, uint32_t, size_t)   { return false; }
bool   RDMAConnection::rdmaWriteToRemote(RDMAContext&, const void*, size_t)       { return false; }
size_t RDMAConnection::pollCompletion(int)                                        { return 0; }

} // namespace advanced
} // namespace vgre

#endif // VGRE_HAS_RDMA
