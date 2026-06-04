// QUEUE-61: cudaMemcpyPeerAsync — cross-device memcpy submitted to scheduler.
// In VGRE single-process model all devices share host memory; peer copy = memcpy.
// Math: peer bandwidth model: same-node = host BW (1/hops = 1).
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>

#define PASS(msg) std::cout << "[PASS] " << msg << "\n"
#define FAIL(msg) do { std::cerr << "[FAIL] " << msg << "\n"; return 1; } while(0)

typedef void* cudaStream_t;
typedef int   cudaError_t;
static constexpr cudaError_t cudaSuccess           = 0;
static constexpr cudaError_t cudaErrorInvalidValue  = 1;
static constexpr cudaError_t cudaErrorInvalidDevice = 101;

extern "C" {
cudaError_t cudaMalloc(void **devPtr, size_t size);
cudaError_t cudaMallocManaged(void **devPtr, size_t size, unsigned int flags);
cudaError_t cudaFree(void *devPtr);
cudaError_t cudaStreamCreate(cudaStream_t *pStream);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaStreamDestroy(cudaStream_t stream);
cudaError_t cudaMemcpyPeer(void *dst, int dstDevice, const void *src, int srcDevice, size_t count);
cudaError_t cudaMemcpyPeerAsync(void *dst, int dstDevice, const void *src,
                                int srcDevice, size_t count, cudaStream_t stream);
}

// ── 1. Basic peer copy: two managed buffers, device 0→device 0 ───────────
int test_peer_copy_same_device() {
    const std::size_t N = 256;
    float *src = nullptr, *dst = nullptr;
    if (cudaMallocManaged((void**)&src, N * sizeof(float), 0) != cudaSuccess) FAIL("cudaMallocManaged src");
    if (cudaMallocManaged((void**)&dst, N * sizeof(float), 0) != cudaSuccess) FAIL("cudaMallocManaged dst");
    for (std::size_t i = 0; i < N; ++i) src[i] = (float)i;
    std::memset(dst, 0, N * sizeof(float));

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    cudaError_t st = cudaMemcpyPeerAsync(dst, 0, src, 0, N * sizeof(float), stream);
    if (st != cudaSuccess) FAIL("cudaMemcpyPeerAsync return");
    cudaStreamSynchronize(stream);

    for (std::size_t i = 0; i < N; ++i)
        if (dst[i] != src[i]) FAIL("peer copy data mismatch at i=" + std::to_string(i));

    cudaStreamDestroy(stream);
    cudaFree(src);
    cudaFree(dst);
    PASS("cudaMemcpyPeerAsync: 256 floats device0→device0 managed buffers");
    return 0;
}

// ── 2. cudaMalloc (device) buffers: verify copy completes correctly ────────
int test_peer_copy_device_buffers() {
    const std::size_t N = 1024;
    uint8_t *devSrc = nullptr, *devDst = nullptr;
    if (cudaMalloc((void**)&devSrc, N) != cudaSuccess) FAIL("cudaMalloc src");
    if (cudaMalloc((void**)&devDst, N) != cudaSuccess) FAIL("cudaMalloc dst");
    // Fill src with pattern 0xAB..
    std::memset(devSrc, 0xAB, N);
    std::memset(devDst, 0, N);

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    cudaError_t st = cudaMemcpyPeerAsync(devDst, 0, devSrc, 0, N, stream);
    if (st != cudaSuccess) FAIL("cudaMemcpyPeerAsync device buffers status");
    cudaStreamSynchronize(stream);

    for (std::size_t i = 0; i < N; ++i)
        if (devDst[i] != 0xAB) FAIL("device buffer peer copy mismatch");

    cudaStreamDestroy(stream);
    cudaFree(devSrc);
    cudaFree(devDst);
    PASS("cudaMemcpyPeerAsync: 1024 bytes device buffers 0xAB pattern");
    return 0;
}

// ── 3. Synchronous cudaMemcpyPeer: verify it also works ────────────────────
int test_peer_copy_sync() {
    const std::size_t N = 64;
    int32_t *src = nullptr, *dst = nullptr;
    cudaMallocManaged((void**)&src, N * sizeof(int32_t), 0);
    cudaMallocManaged((void**)&dst, N * sizeof(int32_t), 0);
    for (std::size_t i = 0; i < N; ++i) src[i] = (int32_t)(i * 3);
    std::memset(dst, 0, N * sizeof(int32_t));

    cudaError_t st = cudaMemcpyPeer(dst, 0, src, 0, N * sizeof(int32_t));
    if (st != cudaSuccess) FAIL("cudaMemcpyPeer status");
    for (std::size_t i = 0; i < N; ++i)
        if (dst[i] != src[i]) FAIL("cudaMemcpyPeer data mismatch at i=" + std::to_string(i));

    cudaFree(src);
    cudaFree(dst);
    PASS("cudaMemcpyPeer (sync): 64 int32 elements");
    return 0;
}

// ── 4. Invalid device rejects ──────────────────────────────────────────────
int test_peer_invalid_device() {
    float buf[4] = {1, 2, 3, 4};
    float out[4] = {};
    cudaStream_t s = nullptr;
    cudaStreamCreate(&s);
    // Device 999 is out of range
    cudaError_t st = cudaMemcpyPeerAsync(out, 999, buf, 0, sizeof(buf), s);
    if (st == cudaSuccess) FAIL("should reject invalid device");
    cudaStreamDestroy(s);
    PASS("cudaMemcpyPeerAsync rejects invalid dstDevice");
    return 0;
}

// ── 5. Null pointer rejects ───────────────────────────────────────────────
int test_peer_null_ptr() {
    float buf[4] = {};
    cudaStream_t s = nullptr;
    cudaStreamCreate(&s);
    cudaError_t st = cudaMemcpyPeerAsync(nullptr, 0, buf, 0, sizeof(buf), s);
    if (st == cudaSuccess) FAIL("should reject null dst");
    cudaStreamDestroy(s);
    PASS("cudaMemcpyPeerAsync rejects null dst pointer");
    return 0;
}

int main() {
    int fails = 0;
    fails += test_peer_copy_same_device();
    fails += test_peer_copy_device_buffers();
    fails += test_peer_copy_sync();
    fails += test_peer_invalid_device();
    fails += test_peer_null_ptr();
    if (fails == 0) std::cout << "All QUEUE-61 cudaMemcpyPeerAsync tests passed.\n";
    return fails;
}
