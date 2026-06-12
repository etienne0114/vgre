// Phase 3, Track P3-14 — CUDA-graph capture-from-stream fidelity.
//
// Regression for the gap where cudaMemsetAsync issued on a capturing stream
// executed eagerly instead of being recorded as a graph node — so the memset ran
// once at capture time and was MISSING from the replayed graph (replay != eager).
// The test captures a two-op graph (memset → dependent device-to-device memcpy)
// and verifies the three fidelity properties: (1) the memset is deferred during
// capture (not executed), (2) it replays on every graph launch, and (3) the
// dependent memcpy observes the memset result (the captured stream order is
// preserved), reproduced identically across repeated launches.

#include "vgre/api/cuda_interceptor.h"
#include "vgre/core/runtime_engine.h"

#include <cstdio>
#include <vector>

using namespace vgre::api;

static int g_pass = 0, g_total = 0;
static void check(const char* name, bool ok) {
    ++g_total;
    printf(ok ? "  PASS  %s\n" : "  FAIL  %s\n", name);
    if (ok) ++g_pass;
}

static bool allEqual(const std::vector<unsigned char>& v, unsigned char x) {
    for (unsigned char b : v) if (b != x) return false;
    return true;
}

int main() {
    printf("=== CUDA-graph capture-from-stream fidelity (P3-14) ===\n");
    CUDAInterceptor& cuda = CUDAInterceptor::instance();
    cuda.init();

    const size_t N = 4096;
    const unsigned char V = 0xAB;
    unsigned char *d_buf = nullptr, *d_out = nullptr;
    cuda.malloc((void**)&d_buf, N);
    cuda.malloc((void**)&d_out, N);
    std::vector<unsigned char> host(N);

    // Eager baseline: known, distinct initial contents.
    cuda.memset(d_buf, 0x00, N);
    cuda.memset(d_out, 0x11, N);

    cudaStream_t stream;
    cuda.streamCreate(&stream);

    cudaGraph_t graph;
    cuda.streamBeginCapture(stream);
    cuda.memsetAsync(d_buf, V, N, stream);                                   // node 1: memset
    cuda.memcpyAsync(d_out, d_buf, N, cudaMemcpyDeviceToDevice, stream);     // node 2: dep memcpy

    // (1) The memset must be DEFERRED (recorded), not executed during capture.
    cuda.memcpy(host.data(), d_buf, N, cudaMemcpyDeviceToHost);
    check("memsetAsync on a capturing stream is deferred (recorded, not executed)",
          allEqual(host, 0x00));

    cuda.streamEndCapture(stream, &graph);

    cudaGraphExec_t exec;
    cuda.graphInstantiate(&exec, graph);

    // Corrupt both buffers; a faithful replay must rebuild them from the nodes.
    cuda.memset(d_buf, 0x22, N);
    cuda.memset(d_out, 0x33, N);

    cuda.graphLaunch(exec, stream);
    cuda.streamSynchronize(stream);

    // (2) The captured memset replays on launch.
    cuda.memcpy(host.data(), d_buf, N, cudaMemcpyDeviceToHost);
    check("captured memset replays on graph launch", allEqual(host, V));

    // (3) The dependent memcpy observed the memset result (captured order preserved).
    cuda.memcpy(host.data(), d_out, N, cudaMemcpyDeviceToHost);
    check("dependent memcpy sees the memset result (ordered replay == eager)",
          allEqual(host, V));

    // Repeated launch reproduces the same result (graphs are replayable).
    cuda.memset(d_buf, 0x44, N);
    cuda.memset(d_out, 0x55, N);
    cuda.graphLaunch(exec, stream);
    cuda.streamSynchronize(stream);
    cuda.memcpy(host.data(), d_out, N, cudaMemcpyDeviceToHost);
    check("second graph launch reproduces the same result", allEqual(host, V));

    cuda.free(d_buf);
    cuda.free(d_out);

    printf("\n%d / %d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
