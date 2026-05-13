/**
 * Test: cuBLAS Pointer Mode and Atomics Mode APIs (P2.2)
 *
 * Verifies:
 *   1. cublasSetPointerMode / cublasGetPointerMode round-trip
 *   2. cublasSetAtomicsMode / cublasGetAtomicsMode round-trip
 *   3. Default values (HOST, ALLOWED)
 *   4. Null pointer rejection
 */

#include <iostream>

extern "C" {
typedef int cublasStatus_t;
#define CUBLAS_STATUS_SUCCESS 0
#define CUBLAS_STATUS_INVALID_VALUE 7
#define CUBLAS_STATUS_NOT_INITIALIZED 1

typedef void* cublasHandle_t;

cublasStatus_t cublasCreate_v2(cublasHandle_t* handle);
cublasStatus_t cublasDestroy_v2(cublasHandle_t handle);

cublasStatus_t cublasSetPointerMode(cublasHandle_t handle, int mode);
cublasStatus_t cublasGetPointerMode(cublasHandle_t handle, int* mode);
cublasStatus_t cublasSetAtomicsMode(cublasHandle_t handle, int mode);
cublasStatus_t cublasGetAtomicsMode(cublasHandle_t handle, int* mode);
}

int main() {
    std::cout << "=== Test: cuBLAS Pointer/Atomics Mode (P2.2) ===\n";
    int pass = 0, total = 0;

    cublasHandle_t handle;
    if (cublasCreate_v2(&handle) != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "FAIL: cublasCreate\n";
        return 1;
    }

    // ── 1. Default pointer mode should be HOST (0) ────────────────────────
    ++total;
    {
        int mode = -1;
        auto r = cublasGetPointerMode(handle, &mode);
        if (r == CUBLAS_STATUS_SUCCESS && mode == 0) {
            std::cout << "PASS [1] Default pointer mode = HOST\n";
            ++pass;
        } else {
            std::cerr << "FAIL [1] Default pointer mode=" << mode << " r=" << r << "\n";
        }
    }

    // ── 2. Set/get pointer mode round-trip (HOST -> DEVICE -> HOST) ────────
    ++total;
    {
        auto r1 = cublasSetPointerMode(handle, 1); // DEVICE
        int mode = -1;
        auto r2 = cublasGetPointerMode(handle, &mode);
        if (r1 == CUBLAS_STATUS_SUCCESS && r2 == CUBLAS_STATUS_SUCCESS && mode == 1) {
            std::cout << "PASS [2] Pointer mode round-trip (HOST->DEVICE)\n";
            ++pass;
        } else {
            std::cerr << "FAIL [2] Pointer mode=" << mode << "\n";
        }
    }

    // ── 3. Set/get back to HOST ────────────────────────────────────────────
    ++total;
    {
        auto r1 = cublasSetPointerMode(handle, 0);
        int mode = -1;
        auto r2 = cublasGetPointerMode(handle, &mode);
        if (r1 == CUBLAS_STATUS_SUCCESS && r2 == CUBLAS_STATUS_SUCCESS && mode == 0) {
            std::cout << "PASS [3] Pointer mode back to HOST\n";
            ++pass;
        } else {
            std::cerr << "FAIL [3] Pointer mode=" << mode << "\n";
        }
    }

    // ── 4. Default atomics mode should be ALLOWED (1) ──────────────────────
    ++total;
    {
        int mode = -1;
        auto r = cublasGetAtomicsMode(handle, &mode);
        if (r == CUBLAS_STATUS_SUCCESS && mode == 1) {
            std::cout << "PASS [4] Default atomics mode = ALLOWED\n";
            ++pass;
        } else {
            std::cerr << "FAIL [4] Default atomics mode=" << mode << "\n";
        }
    }

    // ── 5. Atomics mode round-trip (ALLOWED -> NOT_ALLOWED -> ALLOWED) ────
    ++total;
    {
        auto r1 = cublasSetAtomicsMode(handle, 0); // NOT_ALLOWED
        int mode = -1;
        auto r2 = cublasGetAtomicsMode(handle, &mode);
        if (r1 == CUBLAS_STATUS_SUCCESS && r2 == CUBLAS_STATUS_SUCCESS && mode == 0) {
            std::cout << "PASS [5] Atomics mode round-trip (ALLOWED->NOT_ALLOWED)\n";
            ++pass;
        } else {
            std::cerr << "FAIL [5] Atomics mode=" << mode << "\n";
        }
    }

    ++total;
    {
        auto r1 = cublasSetAtomicsMode(handle, 1);
        int mode = -1;
        auto r2 = cublasGetAtomicsMode(handle, &mode);
        if (r1 == CUBLAS_STATUS_SUCCESS && r2 == CUBLAS_STATUS_SUCCESS && mode == 1) {
            std::cout << "PASS [6] Atomics mode back to ALLOWED\n";
            ++pass;
        } else {
            std::cerr << "FAIL [6] Atomics mode=" << mode << "\n";
        }
    }

    // ── 6. Null pointer rejection ──────────────────────────────────────────
    ++total;
    {
        int mode = 0;
        auto r = cublasGetPointerMode(handle, nullptr);
        if (r == CUBLAS_STATUS_INVALID_VALUE) {
            std::cout << "PASS [7] Null mode pointer rejected\n";
            ++pass;
        } else {
            std::cerr << "FAIL [7] Should reject null mode pointer\n";
        }
    }

    ++total;
    {
        auto r = cublasSetPointerMode(nullptr, 0);
        if (r == CUBLAS_STATUS_NOT_INITIALIZED) {
            std::cout << "PASS [8] Null handle rejected for SetPointerMode\n";
            ++pass;
        } else {
            std::cerr << "FAIL [8] Should reject null handle\n";
        }
    }

    cublasDestroy_v2(handle);

    std::cout << "\n" << pass << "/" << total << " tests passed.\n";
    return (pass == total) ? 0 : 1;
}
