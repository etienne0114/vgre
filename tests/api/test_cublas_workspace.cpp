// QUEUE-36: cuBLAS workspace allocator set/get tests.
// Verifies cublasSetWorkspace_v2 and cublasGetWorkspaceSize contract:
//   1. Default workspace size = 0
//   2. Set/get round-trip
//   3. null+0 clears workspace
//   4. null+size>0 → INVALID_VALUE
//   5. Null handle → NOT_INITIALIZED
//   6. GetWorkspaceSize null size_ptr → INVALID_VALUE
//   7. Large (16 MiB) workspace round-trip
#include <cstddef>
#include <cstdio>

extern "C" {
typedef int    cublasStatus_t;
#define CUBLAS_STATUS_SUCCESS         0
#define CUBLAS_STATUS_NOT_INITIALIZED 1
#define CUBLAS_STATUS_INVALID_VALUE   7
typedef void*  cublasHandle_t;

cublasStatus_t cublasCreate_v2(cublasHandle_t* handle);
cublasStatus_t cublasDestroy_v2(cublasHandle_t handle);
cublasStatus_t cublasSetWorkspace_v2(cublasHandle_t handle,
                                      void* workspace, size_t workspaceSizeInBytes);
cublasStatus_t cublasGetWorkspaceSize(cublasHandle_t handle, size_t* workspaceSizeInBytes);
}

#define PASS(msg) do { printf("PASS: %s\n", (msg)); } while(0)
#define FAIL(msg) do { fprintf(stderr, "FAIL: %s\n", (msg)); return 1; } while(0)

// 16 MiB static buffer for the large-workspace test (in BSS, not stack)
static char g_bigBuf[16 * 1024 * 1024];

int main() {
    cublasHandle_t h = nullptr;
    if (cublasCreate_v2(&h) != CUBLAS_STATUS_SUCCESS || !h) FAIL("cublasCreate");

    // 1. Default workspace size = 0
    size_t sz = 99;
    if (cublasGetWorkspaceSize(h, &sz) != CUBLAS_STATUS_SUCCESS) FAIL("GetWorkspaceSize default");
    if (sz != 0) FAIL("default workspace size must be 0");
    PASS("default workspace size = 0");

    // 2. Set workspace: round-trip
    char buf[1024];
    if (cublasSetWorkspace_v2(h, buf, sizeof(buf)) != CUBLAS_STATUS_SUCCESS)
        FAIL("SetWorkspace with valid buffer");
    sz = 0;
    if (cublasGetWorkspaceSize(h, &sz) != CUBLAS_STATUS_SUCCESS) FAIL("GetWorkspaceSize after set");
    if (sz != sizeof(buf)) FAIL("workspace size after set must match buffer size");
    PASS("workspace set/get round-trip (1024 bytes)");

    // 3. null+0 clears workspace
    if (cublasSetWorkspace_v2(h, nullptr, 0) != CUBLAS_STATUS_SUCCESS)
        FAIL("SetWorkspace null+0 must succeed (clear)");
    sz = 99;
    if (cublasGetWorkspaceSize(h, &sz) != CUBLAS_STATUS_SUCCESS) FAIL("GetWorkspaceSize after clear");
    if (sz != 0) FAIL("workspace size after clear must be 0");
    PASS("null+0 clears workspace → size 0");

    // 4. null buffer + size > 0 → INVALID_VALUE
    if (cublasSetWorkspace_v2(h, nullptr, 64) != CUBLAS_STATUS_INVALID_VALUE)
        FAIL("SetWorkspace null+size>0 must return INVALID_VALUE");
    PASS("null buffer + size > 0 → INVALID_VALUE");

    // 5. Null handle → NOT_INITIALIZED
    if (cublasSetWorkspace_v2(nullptr, buf, 128) != CUBLAS_STATUS_NOT_INITIALIZED)
        FAIL("SetWorkspace null handle must return NOT_INITIALIZED");
    if (cublasGetWorkspaceSize(nullptr, &sz) != CUBLAS_STATUS_NOT_INITIALIZED)
        FAIL("GetWorkspaceSize null handle must return NOT_INITIALIZED");
    PASS("null handle → NOT_INITIALIZED");

    // 6. GetWorkspaceSize null size_ptr → INVALID_VALUE
    if (cublasGetWorkspaceSize(h, nullptr) != CUBLAS_STATUS_INVALID_VALUE)
        FAIL("GetWorkspaceSize null size_ptr must return INVALID_VALUE");
    PASS("GetWorkspaceSize null size_ptr → INVALID_VALUE");

    // 7. Large workspace round-trip (16 MiB — common cuBLAS default)
    const size_t bigSize = 16u * 1024u * 1024u;
    if (cublasSetWorkspace_v2(h, g_bigBuf, bigSize) != CUBLAS_STATUS_SUCCESS)
        FAIL("SetWorkspace 16 MiB");
    sz = 0;
    if (cublasGetWorkspaceSize(h, &sz) != CUBLAS_STATUS_SUCCESS) FAIL("GetWorkspaceSize 16 MiB");
    if (sz != bigSize) FAIL("workspace size 16 MiB round-trip");
    PASS("large workspace 16 MiB round-trip");

    cublasDestroy_v2(h);
    printf("All cuBLAS workspace QUEUE-36 tests passed.\n");
    return 0;
}
