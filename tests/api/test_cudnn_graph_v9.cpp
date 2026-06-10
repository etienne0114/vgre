/**
 * Test: cuDNN v9 Graph API
 *
 * Creates a graph, adds a POINTWISE_ADD op descriptor with two 4-element float
 * tensors, builds it, executes it, and verifies output = input1 + input2.
 */

#include <iostream>
#include <cmath>
#include <cstring>
#include <vector>
#include <cstdint>
#include <cassert>

// ── Type stubs (matching cudnn_internal.h / cudnn_backend_internal.h) ─────────
extern "C" {

typedef int    cudnnStatus_t;
typedef void*  cudnnHandle_t;

#define CUDNN_STATUS_SUCCESS       0
#define CUDNN_STATUS_INVALID_VALUE 8
#define CUDNN_STATUS_NOT_SUPPORTED 9

enum cudnnDataType_t     { CUDNN_DATA_FLOAT = 0 };
enum cudnnTensorFormat_t { CUDNN_TENSOR_NCHW = 0 };

// ── Backend types (from cudnn_backend_internal.h) ─────────────────────────────
typedef int cudnnBackendDescriptorType_t;
typedef int cudnnBackendAttributeName_t;

// Descriptor type constants
#define CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR 37
#define CUDNN_BACKEND_POINTWISE_DESCRIPTOR           22
#define CUDNN_BACKEND_TENSOR_DESCRIPTOR              20
#define CUDNN_BACKEND_OPERATIONSET_DESCRIPTOR        27
#define CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR        33

// Attribute name constants
#define CUDNN_ATTR_TENSOR_DIMENSIONS                 30
#define CUDNN_ATTR_TENSOR_DATA_TYPE                  32
#define CUDNN_ATTR_POINTWISE_MODE                    0
#define CUDNN_ATTR_OPERATION_POINTWISE_XDESC         600
#define CUDNN_ATTR_OPERATION_POINTWISE_BDESC         601
#define CUDNN_ATTR_OPERATION_POINTWISE_YDESC         602
#define CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR 603
#define CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS           100
#define CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS        101

// cudnnPointwiseMode_t — must match values in cudnn_internal.h
typedef int cudnnPointwiseMode_t;
#define CUDNN_POINTWISE_ADD 0

// Backend API
cudnnStatus_t cudnnBackendCreateDescriptor(cudnnBackendDescriptorType_t type, void** descriptor);
cudnnStatus_t cudnnBackendDestroyDescriptor(void* descriptor);
cudnnStatus_t cudnnBackendSetAttribute(void* descriptor,
                                       cudnnBackendAttributeName_t attrName,
                                       int attributeType,
                                       int64_t elementCount,
                                       const void* arrayOfElements);
cudnnStatus_t cudnnBackendFinalize(void* descriptor);

// cuDNN v9 Graph API
struct CudnnGraph;
cudnnStatus_t cudnnGraphCreate(CudnnGraph** graph, cudnnHandle_t handle);
cudnnStatus_t cudnnGraphDestroy(CudnnGraph* graph);
cudnnStatus_t cudnnGraphAddNode(CudnnGraph* graph, void* opDescriptor);
cudnnStatus_t cudnnGraphBuildAndCheck(CudnnGraph* graph, cudnnHandle_t handle);
cudnnStatus_t cudnnGraphExecute(cudnnHandle_t handle, CudnnGraph* graph, void* variantPack);
cudnnStatus_t cudnnGraphGetAttribute(CudnnGraph* graph, int attr, void* value, size_t* size);

} // extern "C"

// ── Helper macros ─────────────────────────────────────────────────────────────
#define CHECKDNN(expr) do { \
    cudnnStatus_t _s = (expr); \
    if (_s != CUDNN_STATUS_SUCCESS) { \
        std::cerr << "cuDNN error " << _s << " at " << #expr << "\n"; \
        return 1; \
    } \
} while(0)

int main() {
    // ── Build a POINTWISE_ADD descriptor ──────────────────────────────────────

    const int N = 4;

    // 1a. Create x tensor descriptor (ID will be its backend node pointer)
    void* xDesc = nullptr;
    CHECKDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &xDesc));
    {
        int64_t dims[1] = { N };
        CHECKDNN(cudnnBackendSetAttribute(xDesc, CUDNN_ATTR_TENSOR_DIMENSIONS, 4 /*int64*/, 1, dims));
        int64_t dtype = CUDNN_DATA_FLOAT;
        CHECKDNN(cudnnBackendSetAttribute(xDesc, CUDNN_ATTR_TENSOR_DATA_TYPE, 4 /*int64*/, 1, &dtype));
        CHECKDNN(cudnnBackendFinalize(xDesc));
    }

    // 1b. Create b tensor descriptor (bias/second input)
    void* bDesc = nullptr;
    CHECKDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &bDesc));
    {
        int64_t dims[1] = { N };
        CHECKDNN(cudnnBackendSetAttribute(bDesc, CUDNN_ATTR_TENSOR_DIMENSIONS, 4 /*int64*/, 1, dims));
        int64_t dtype = CUDNN_DATA_FLOAT;
        CHECKDNN(cudnnBackendSetAttribute(bDesc, CUDNN_ATTR_TENSOR_DATA_TYPE, 4 /*int64*/, 1, &dtype));
        CHECKDNN(cudnnBackendFinalize(bDesc));
    }

    // 1c. Create y tensor descriptor (output)
    void* yDesc = nullptr;
    CHECKDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &yDesc));
    {
        int64_t dims[1] = { N };
        CHECKDNN(cudnnBackendSetAttribute(yDesc, CUDNN_ATTR_TENSOR_DIMENSIONS, 4 /*int64*/, 1, dims));
        int64_t dtype = CUDNN_DATA_FLOAT;
        CHECKDNN(cudnnBackendSetAttribute(yDesc, CUDNN_ATTR_TENSOR_DATA_TYPE, 4 /*int64*/, 1, &dtype));
        CHECKDNN(cudnnBackendFinalize(yDesc));
    }

    // 2. Create pointwise descriptor (mode = ADD)
    void* pwDesc = nullptr;
    CHECKDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_POINTWISE_DESCRIPTOR, &pwDesc));
    {
        int64_t mode = CUDNN_POINTWISE_ADD;
        CHECKDNN(cudnnBackendSetAttribute(pwDesc, CUDNN_ATTR_POINTWISE_MODE, 4 /*int64*/, 1, &mode));
        CHECKDNN(cudnnBackendFinalize(pwDesc));
    }

    // 3. Create pointwise operation descriptor
    void* opDesc = nullptr;
    CHECKDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR, &opDesc));
    {
        uintptr_t xId = reinterpret_cast<uintptr_t>(xDesc);
        uintptr_t bId = reinterpret_cast<uintptr_t>(bDesc);
        uintptr_t yId = reinterpret_cast<uintptr_t>(yDesc);
        uintptr_t pwId = reinterpret_cast<uintptr_t>(pwDesc);
        // attributeType=5 = uint64 for descriptor handles
        CHECKDNN(cudnnBackendSetAttribute(opDesc, CUDNN_ATTR_OPERATION_POINTWISE_XDESC, 5, 1, &xId));
        CHECKDNN(cudnnBackendSetAttribute(opDesc, CUDNN_ATTR_OPERATION_POINTWISE_BDESC, 5, 1, &bId));
        CHECKDNN(cudnnBackendSetAttribute(opDesc, CUDNN_ATTR_OPERATION_POINTWISE_YDESC, 5, 1, &yId));
        CHECKDNN(cudnnBackendSetAttribute(opDesc, CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR, 5, 1, &pwId));
        CHECKDNN(cudnnBackendFinalize(opDesc));
    }

    // 4. Create the v9 graph
    CudnnGraph* graph = nullptr;
    CHECKDNN(cudnnGraphCreate(&graph, nullptr));
    CHECKDNN(cudnnGraphAddNode(graph, opDesc));

    // 5. Verify GetAttribute before build
    {
        int64_t nodeCount = 0;
        size_t sz = 0;
        CHECKDNN(cudnnGraphGetAttribute(graph, 0, &nodeCount, &sz));
        if (nodeCount != 1) {
            std::cerr << "Expected 1 node, got " << nodeCount << "\n";
            return 1;
        }
        int32_t built = 0;
        CHECKDNN(cudnnGraphGetAttribute(graph, 1, &built, &sz));
        if (built != 0) {
            std::cerr << "Graph should not be built yet\n";
            return 1;
        }
    }

    // 6. Build
    CHECKDNN(cudnnGraphBuildAndCheck(graph, nullptr));

    // 7. Verify built flag
    {
        int32_t built = 0;
        size_t sz = 0;
        CHECKDNN(cudnnGraphGetAttribute(graph, 1, &built, &sz));
        if (built != 1) {
            std::cerr << "Graph should be built\n";
            return 1;
        }
    }

    // 8. Create variant pack and execute
    float x[N]   = {1.0f, 2.0f, 3.0f, 4.0f};
    float b[N]   = {10.0f, 20.0f, 30.0f, 40.0f};
    float y[N]   = {0.0f, 0.0f, 0.0f, 0.0f};

    void* vpDesc = nullptr;
    CHECKDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &vpDesc));
    {
        uintptr_t xId  = reinterpret_cast<uintptr_t>(xDesc);
        uintptr_t bId  = reinterpret_cast<uintptr_t>(bDesc);
        uintptr_t yId  = reinterpret_cast<uintptr_t>(yDesc);
        uintptr_t ids[3]  = { xId, bId, yId };
        uintptr_t ptrs[3] = {
            reinterpret_cast<uintptr_t>(x),
            reinterpret_cast<uintptr_t>(b),
            reinterpret_cast<uintptr_t>(y)
        };
        CHECKDNN(cudnnBackendSetAttribute(vpDesc, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,    5, 3, ids));
        CHECKDNN(cudnnBackendSetAttribute(vpDesc, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, 5, 3, ptrs));
        CHECKDNN(cudnnBackendFinalize(vpDesc));
    }

    CHECKDNN(cudnnGraphExecute(nullptr, graph, vpDesc));

    // 9. Verify output = x + b
    float expected[N] = {11.0f, 22.0f, 33.0f, 44.0f};
    for (int i = 0; i < N; ++i) {
        if (std::fabs(y[i] - expected[i]) > 1e-4f) {
            std::cerr << "Output mismatch at [" << i << "]: got " << y[i]
                      << ", expected " << expected[i] << "\n";
            return 1;
        }
    }

    // ── 11. Unary ERF pointwise op (Track E: extended pointwise modes) ────────
    // Reuses x/y tensor descriptors; applies y = erf(x) via a unary pointwise op.
    #define CUDNN_POINTWISE_ERF 42
    {
        void* pwErf = nullptr;
        CHECKDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_POINTWISE_DESCRIPTOR, &pwErf));
        int64_t mode = CUDNN_POINTWISE_ERF;
        CHECKDNN(cudnnBackendSetAttribute(pwErf, CUDNN_ATTR_POINTWISE_MODE, 4, 1, &mode));
        CHECKDNN(cudnnBackendFinalize(pwErf));

        void* opErf = nullptr;
        CHECKDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR, &opErf));
        uintptr_t xId = reinterpret_cast<uintptr_t>(xDesc);
        uintptr_t yId = reinterpret_cast<uintptr_t>(yDesc);
        uintptr_t pwId = reinterpret_cast<uintptr_t>(pwErf);
        CHECKDNN(cudnnBackendSetAttribute(opErf, CUDNN_ATTR_OPERATION_POINTWISE_XDESC, 5, 1, &xId));
        CHECKDNN(cudnnBackendSetAttribute(opErf, CUDNN_ATTR_OPERATION_POINTWISE_YDESC, 5, 1, &yId));
        CHECKDNN(cudnnBackendSetAttribute(opErf, CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR, 5, 1, &pwId));
        CHECKDNN(cudnnBackendFinalize(opErf));

        CudnnGraph* g2 = nullptr;
        CHECKDNN(cudnnGraphCreate(&g2, nullptr));
        CHECKDNN(cudnnGraphAddNode(g2, opErf));
        CHECKDNN(cudnnGraphBuildAndCheck(g2, nullptr));

        float xe[N] = {0.0f, 0.5f, 1.0f, -1.0f};
        float ye[N] = {0.0f, 0.0f, 0.0f, 0.0f};
        void* vp2 = nullptr;
        CHECKDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &vp2));
        uintptr_t ids2[2]  = { xId, yId };
        uintptr_t ptrs2[2] = { reinterpret_cast<uintptr_t>(xe),
                               reinterpret_cast<uintptr_t>(ye) };
        CHECKDNN(cudnnBackendSetAttribute(vp2, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,    5, 2, ids2));
        CHECKDNN(cudnnBackendSetAttribute(vp2, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, 5, 2, ptrs2));
        CHECKDNN(cudnnBackendFinalize(vp2));

        CHECKDNN(cudnnGraphExecute(nullptr, g2, vp2));

        for (int i = 0; i < N; ++i) {
            float want = std::erf(xe[i]);
            if (std::fabs(ye[i] - want) > 1e-5f) {
                std::cerr << "ERF mismatch at [" << i << "]: got " << ye[i]
                          << ", expected " << want << "\n";
                return 1;
            }
        }
        cudnnGraphDestroy(g2);
        cudnnBackendDestroyDescriptor(vp2);
        cudnnBackendDestroyDescriptor(opErf);
        cudnnBackendDestroyDescriptor(pwErf);
    }

    // 12. Cleanup
    cudnnGraphDestroy(graph);
    cudnnBackendDestroyDescriptor(vpDesc);
    cudnnBackendDestroyDescriptor(opDesc);
    cudnnBackendDestroyDescriptor(pwDesc);
    cudnnBackendDestroyDescriptor(yDesc);
    cudnnBackendDestroyDescriptor(bDesc);
    cudnnBackendDestroyDescriptor(xDesc);

    std::cout << "PASSED: cuDNN v9 Graph API — pointwise ADD + unary ERF verified\n";
    return 0;
}
