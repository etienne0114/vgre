/**
 * Test: cudaMemset2D/3D + cudaGraphAddKernelNode
 *
 * Covers P1.9 (2D/3D pitched memset) and P1.10 (graph kernel node CUDART shim).
 */

#include <iostream>
#include <cstring>
#include <atomic>
#include <unistd.h>

extern "C" {
typedef int cudaError_t;
#define cudaSuccess 0
#define cudaErrorInvalidValue 1

struct cudaPitchedPtr { void *ptr; size_t pitch; size_t xsize; size_t ysize; };
struct cudaExtent { size_t width; size_t height; size_t depth; };

typedef struct CUstream_st *cudaStream_t;
typedef unsigned long long cudaGraph_t;
typedef unsigned long long cudaGraphExec_t;
typedef unsigned long long cudaGraphNode_t;

struct dim3 { unsigned int x, y, z; };

cudaError_t cudaMalloc(void **devPtr, size_t size);
cudaError_t cudaFree(void *devPtr);
cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, int kind);
cudaError_t cudaMemset(void *devPtr, int value, size_t count);
cudaError_t cudaMemset2D(void *devPtr, size_t pitch, int value,
                          size_t width, size_t height);
cudaError_t cudaMemset3D(struct cudaPitchedPtr pitchedDevPtr, int value,
                          struct cudaExtent extent);
cudaError_t cudaMemset2DAsync(void *devPtr, size_t pitch, int value,
                               size_t width, size_t height, cudaStream_t stream);
cudaError_t cudaStreamCreate(cudaStream_t *stream);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaStreamDestroy(cudaStream_t stream);

cudaError_t cudaGraphCreate(cudaGraph_t *graph, unsigned int flags);
cudaError_t cudaGraphDestroy(cudaGraph_t graph);
cudaError_t cudaGraphAddEmptyNode(cudaGraphNode_t *pNode, cudaGraph_t graph,
                                   const cudaGraphNode_t *deps, size_t numDeps);
struct cudaHostNodeParams {
    void (*fn)(void *);
    void *userData;
};
cudaError_t cudaGraphAddHostNode(cudaGraphNode_t *pNode, cudaGraph_t graph,
                                  const cudaGraphNode_t *deps, size_t numDeps,
                                  const cudaHostNodeParams *params);
}

struct cudaHostNodeParams_local {
    void (*fn)(void *);
    void *userData;
};

static std::atomic<int> g_hostCallCount{0};

static void hostCallback(void *userData) {
    int *p = static_cast<int *>(userData);
    *p = 999;
    g_hostCallCount.fetch_add(1, std::memory_order_relaxed);
}

int main() {
    std::cout << "=== Test: cudaMemset2D / cudaMemset3D / cudaGraphAddHostNode ===\n";
    int pass = 0, total = 0;

    // ── 1. cudaMemset2D basic ──────────────────────────────────────────────
    ++total;
    {
        constexpr size_t W = 8, H = 4, PITCH = 16;  // pad to 16 bytes
        void *buf = nullptr;
        cudaMalloc(&buf, PITCH * H);
        memset(buf, 0xAA, PITCH * H);

        cudaError_t e = cudaMemset2D(buf, PITCH, 0x5C, W, H);
        if (e != cudaSuccess) {
            std::cerr << "FAIL [1] cudaMemset2D returned " << e << "\n";
        } else {
            auto *p = static_cast<unsigned char *>(buf);
            bool ok = true;
            for (size_t row = 0; row < H && ok; ++row) {
                for (size_t col = 0; col < W; ++col)
                    if (p[row * PITCH + col] != 0x5C) { ok = false; break; }
                // Padding bytes after W should still be 0xAA
                for (size_t col = W; col < PITCH; ++col)
                    if (p[row * PITCH + col] != 0xAA) { ok = false; break; }
            }
            if (ok) { std::cout << "PASS [1] cudaMemset2D basic\n"; ++pass; }
            else     std::cerr << "FAIL [1] cudaMemset2D data mismatch\n";
        }
        cudaFree(buf);
    }

    // ── 2. cudaMemset2D null dst rejected ────────────────────────────────
    ++total;
    {
        cudaError_t e = cudaMemset2D(nullptr, 8, 0, 4, 2);
        if (e == cudaErrorInvalidValue) {
            std::cout << "PASS [2] cudaMemset2D null rejected\n"; ++pass;
        } else {
            std::cerr << "FAIL [2] cudaMemset2D should reject null dst\n";
        }
    }

    // ── 3. cudaMemset2D zero size succeeds immediately ────────────────────
    ++total;
    {
        char dummy[16] = {0x11};
        cudaError_t e = cudaMemset2D(dummy, 8, 0xFF, 0, 2);
        if (e == cudaSuccess) {
            std::cout << "PASS [3] cudaMemset2D zero-width no-op\n"; ++pass;
        } else {
            std::cerr << "FAIL [3] cudaMemset2D zero-width should succeed\n";
        }
    }

    // ── 4. cudaMemset3D basic ─────────────────────────────────────────────
    ++total;
    {
        constexpr size_t W = 4, H = 3, D = 2, PITCH = 8;
        void *buf = nullptr;
        cudaMalloc(&buf, PITCH * H * D);
        memset(buf, 0xBB, PITCH * H * D);

        cudaPitchedPtr pp{buf, PITCH, W, H};
        cudaExtent ext{W, H, D};
        cudaError_t e = cudaMemset3D(pp, 0x7E, ext);
        if (e != cudaSuccess) {
            std::cerr << "FAIL [4] cudaMemset3D returned " << e << "\n";
        } else {
            auto *p = static_cast<unsigned char *>(buf);
            bool ok = true;
            for (size_t d = 0; d < D && ok; ++d)
                for (size_t h = 0; h < H && ok; ++h)
                    for (size_t w = 0; w < W; ++w)
                        if (p[d * PITCH * H + h * PITCH + w] != 0x7E)
                            { ok = false; break; }
            if (ok) { std::cout << "PASS [4] cudaMemset3D basic\n"; ++pass; }
            else     std::cerr << "FAIL [4] cudaMemset3D data mismatch\n";
        }
        cudaFree(buf);
    }

    // ── 5. cudaMemset2DAsync ──────────────────────────────────────────────
    ++total;
    {
        constexpr size_t W = 6, H = 4, PITCH = 8;
        void *buf = nullptr;
        cudaMalloc(&buf, PITCH * H);
        memset(buf, 0, PITCH * H);

        cudaStream_t stream;
        cudaStreamCreate(&stream);
        cudaError_t e = cudaMemset2DAsync(buf, PITCH, 0xCC, W, H, stream);
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);

        if (e != cudaSuccess) {
            std::cerr << "FAIL [5] cudaMemset2DAsync returned " << e << "\n";
        } else {
            auto *p = static_cast<unsigned char *>(buf);
            bool ok = true;
            for (size_t r = 0; r < H && ok; ++r)
                for (size_t c = 0; c < W; ++c)
                    if (p[r * PITCH + c] != 0xCC) { ok = false; break; }
            if (ok) { std::cout << "PASS [5] cudaMemset2DAsync\n"; ++pass; }
            else     std::cerr << "FAIL [5] cudaMemset2DAsync data mismatch\n";
        }
        cudaFree(buf);
    }

    // ── 6. cudaGraphAddEmptyNode ──────────────────────────────────────────
    ++total;
    {
        cudaGraph_t graph = 0;
        cudaGraphCreate(&graph, 0);
        cudaGraphNode_t n1 = 0, n2 = 0;

        cudaError_t e1 = cudaGraphAddEmptyNode(&n1, graph, nullptr, 0);
        cudaError_t e2 = cudaGraphAddEmptyNode(&n2, graph, &n1, 1);

        if (e1 == cudaSuccess && e2 == cudaSuccess && n1 != 0 && n2 != 0 && n1 != n2) {
            std::cout << "PASS [6] cudaGraphAddEmptyNode (dependency chain)\n"; ++pass;
        } else {
            std::cerr << "FAIL [6] e1=" << e1 << " e2=" << e2
                      << " n1=" << n1 << " n2=" << n2 << "\n";
        }
        cudaGraphDestroy(graph);
    }

    // ── 7. cudaGraphAddHostNode ───────────────────────────────────────────
    ++total;
    {
        cudaGraph_t graph = 0;
        cudaGraphCreate(&graph, 0);
        cudaGraphNode_t hNode = 0;

        int sentinel = 0;
        cudaHostNodeParams_local params{hostCallback, &sentinel};
        cudaError_t e = cudaGraphAddHostNode(
            &hNode, graph, nullptr, 0,
            reinterpret_cast<const cudaHostNodeParams *>(&params));

        if (e == cudaSuccess && hNode != 0) {
            std::cout << "PASS [7] cudaGraphAddHostNode\n"; ++pass;
        } else {
            std::cerr << "FAIL [7] e=" << e << " hNode=" << hNode << "\n";
        }
        cudaGraphDestroy(graph);
    }

    // ── Summary ───────────────────────────────────────────────────────────
    std::cout << "\n" << pass << "/" << total << " tests passed.\n";
    return (pass == total) ? 0 : 1;
}
