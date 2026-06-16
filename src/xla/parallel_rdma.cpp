// Tensor-parallel collectives over the real SoftwareRDMA transport — L5.
//
// The in-process Communicator (parallel.cpp) combines per-rank buffers with
// memcpy. Here each rank instead registers its partial result as an RDMA region
// and the gathering rank pulls every partial with one-sided RDMA *reads* over
// real sockets (loopback in-process, the same code across machines). This is the
// binding from the sharding executor to the portable cluster transport.

#include "vgre/xla/parallel.h"

#include <cstdint>
#include <memory>
#include <vector>

#include "vgre/advanced/software_rdma.h"
#include "vgre/xla/blas_gemm.h"

namespace vgre {
namespace xla {

namespace {

using vgre::advanced::SoftwareRDMA;

// Gather every rank's flat partial over real RDMA reads. Each rank stands up an
// endpoint and registers its buffer; one hub endpoint reads them all. Returns
// the R buffers (as transferred over the wire) and the bytes moved.
std::vector<std::vector<float>> rdmaGather(std::vector<std::vector<float>>& parts,
                                           uint64_t* bytesOut) {
    const int R = (int)parts.size();
    std::vector<std::unique_ptr<SoftwareRDMA>> eps(R);
    std::vector<uint32_t> region(R);
    for (int r = 0; r < R; ++r) {
        eps[r] = SoftwareRDMA::listen(0);
        region[r] = eps[r]->registerRegion(parts[r].data(), parts[r].size() * sizeof(float));
    }
    SoftwareRDMA& hub = *eps[0];
    std::vector<std::vector<float>> out(R);
    for (int p = 0; p < R; ++p) {
        out[p].resize(parts[p].size());
        uint64_t conn = hub.connect("127.0.0.1", eps[p]->port());
        hub.read(conn, out[p].data(), region[p], 0, parts[p].size() * sizeof(float));
        hub.disconnect(conn);
    }
    // Total bytes served across all endpoints' daemons = total moved over RDMA
    // (each endpoint counts the read of its own region in bytes_read_).
    if (bytesOut) { uint64_t tot = 0; for (auto& e : eps) tot += e->bytesRead(); *bytesOut = tot; }
    return out;  // eps torn down here; buffers already copied out
}

}  // namespace

Literal columnParallelMatmulRdma(const Literal& X, const Literal& W, int ranks,
                                 uint64_t* rdmaBytesMoved) {
    const int64_t M = X.shape.dims[0], K = X.shape.dims[1];
    std::vector<Literal> Wsh = shardColumns(W, ranks);

    std::vector<std::vector<float>> parts(ranks);
    std::vector<int64_t> nr(ranks);
    for (int r = 0; r < ranks; ++r) {                 // each rank: X · W_r → [M, N_r]
        nr[r] = Wsh[r].shape.dims[1];
        parts[r].resize((size_t)(M * nr[r]));
        gemm_f32(false, false, M, nr[r], K, X.data.data(), Wsh[r].data.data(), parts[r].data(), 1);
    }
    std::vector<std::vector<float>> gathered = rdmaGather(parts, rdmaBytesMoved);

    // all-gather concat along N (rebuild [M,N] from the [M,N_r] partials).
    int64_t Ntot = 0; for (int64_t n : nr) Ntot += n;
    Literal out; out.shape.dims = {M, Ntot}; out.data.assign((size_t)(M * Ntot), 0.0f);
    int64_t colOff = 0;
    for (int r = 0; r < ranks; ++r) {
        for (int64_t m = 0; m < M; ++m)
            for (int64_t n = 0; n < nr[r]; ++n)
                out.data[(size_t)(m * Ntot + colOff + n)] = gathered[r][(size_t)(m * nr[r] + n)];
        colOff += nr[r];
    }
    return out;
}

Literal rowParallelMatmulRdma(const Literal& X, const Literal& W, int ranks,
                              uint64_t* rdmaBytesMoved) {
    const int64_t M = X.shape.dims[0], N = W.shape.dims[1];
    std::vector<Literal> Xsh = shardContracting(X, ranks);
    std::vector<Literal> Wsh = shardRows(W, ranks);

    std::vector<std::vector<float>> parts(ranks);
    for (int r = 0; r < ranks; ++r) {                 // each rank: X_r · W_r → [M,N] partial
        const int64_t Kr = Xsh[r].shape.dims[1];
        parts[r].resize((size_t)(M * N));
        gemm_f32(false, false, M, N, Kr, Xsh[r].data.data(), Wsh[r].data.data(), parts[r].data(), 1);
    }
    std::vector<std::vector<float>> gathered = rdmaGather(parts, rdmaBytesMoved);

    // all-reduce: sum the partials.
    Literal out; out.shape.dims = {M, N}; out.data.assign((size_t)(M * N), 0.0f);
    for (int r = 0; r < ranks; ++r)
        for (size_t i = 0; i < out.data.size(); ++i) out.data[i] += gathered[r][i];
    return out;
}

}  // namespace xla
}  // namespace vgre
