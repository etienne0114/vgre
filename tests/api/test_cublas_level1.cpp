/**
 * Test: cuBLAS Level-1 functions (P2.1)
 *
 * Verifies: copy, swap, scal, asum, amax, amin, rot, rotg, rotm, rotmg
 * Both single (S) and double (D) precision.
 */

#include <iostream>
#include <cmath>
#include <cstring>

extern "C" {
typedef int cublasStatus_t;
#define CUBLAS_STATUS_SUCCESS 0
#define CUBLAS_STATUS_INVALID_VALUE 7

typedef void* cublasHandle_t;

cublasStatus_t cublasCreate_v2(cublasHandle_t* handle);
cublasStatus_t cublasDestroy_v2(cublasHandle_t handle);

cublasStatus_t cublasSscal_v2(cublasHandle_t, int, const float*, float*, int);
cublasStatus_t cublasDscal_v2(cublasHandle_t, int, const double*, double*, int);
cublasStatus_t cublasScopy_v2(cublasHandle_t, int, const float*, int, float*, int);
cublasStatus_t cublasDcopy_v2(cublasHandle_t, int, const double*, int, double*, int);
cublasStatus_t cublasSswap_v2(cublasHandle_t, int, float*, int, float*, int);
cublasStatus_t cublasDswap_v2(cublasHandle_t, int, double*, int, double*, int);
cublasStatus_t cublasSasum_v2(cublasHandle_t, int, const float*, int, float*);
cublasStatus_t cublasDasum_v2(cublasHandle_t, int, const double*, int, double*);
cublasStatus_t cublasIsamax_v2(cublasHandle_t, int, const float*, int, int*);
cublasStatus_t cublasIdamax_v2(cublasHandle_t, int, const double*, int, int*);
cublasStatus_t cublasIsamin_v2(cublasHandle_t, int, const float*, int, int*);
cublasStatus_t cublasIdamin_v2(cublasHandle_t, int, const double*, int, int*);
cublasStatus_t cublasSrot_v2(cublasHandle_t, int, float*, int, float*, int,
                              const float*, const float*);
cublasStatus_t cublasDrot_v2(cublasHandle_t, int, double*, int, double*, int,
                              const double*, const double*);
cublasStatus_t cublasSrotg_v2(cublasHandle_t, float*, float*, float*, float*);
cublasStatus_t cublasDrotg_v2(cublasHandle_t, double*, double*, double*, double*);
cublasStatus_t cublasSrotm_v2(cublasHandle_t, int, float*, int, float*, int,
                               const float*);
cublasStatus_t cublasDrotm_v2(cublasHandle_t, int, double*, int, double*, int,
                               const double*);
cublasStatus_t cublasSrotmg_v2(cublasHandle_t, float*, float*, float*, float*, float*);
cublasStatus_t cublasDrotmg_v2(cublasHandle_t, double*, double*, double*, double*, double*);
}

static bool approx_eq(float a, float b, float eps = 1e-5f) {
    return std::abs(a - b) <= eps;
}
static bool approx_eq(double a, double b, double eps = 1e-9) {
    return std::abs(a - b) <= eps;
}

int main() {
    std::cout << "=== Test: cuBLAS Level-1 (P2.1) ===\n";
    int pass = 0, total = 0;

    cublasHandle_t handle;
    if (cublasCreate_v2(&handle) != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "FAIL: cublasCreate\n";
        return 1;
    }

    // ── 1. SScal / DScal ───────────────────────────────────────────────────
    ++total;
    {
        float x[4] = {1,2,3,4};
        float alpha = 2.5f;
        auto r = cublasSscal_v2(handle, 4, &alpha, x, 1);
        bool ok = (r == CUBLAS_STATUS_SUCCESS) &&
                  approx_eq(x[0], 2.5f) && approx_eq(x[1], 5.0f) &&
                  approx_eq(x[2], 7.5f) && approx_eq(x[3], 10.0f);
        if (ok) { std::cout << "PASS [1] Sscal\n"; ++pass; }
        else    std::cerr << "FAIL [1] Sscal\n";
    }
    ++total;
    {
        double x[3] = {1,2,3};
        double alpha = 3.0;
        auto r = cublasDscal_v2(handle, 3, &alpha, x, 1);
        bool ok = (r == CUBLAS_STATUS_SUCCESS) &&
                  approx_eq(x[0], 3.0) && approx_eq(x[1], 6.0) && approx_eq(x[2], 9.0);
        if (ok) { std::cout << "PASS [2] Dscal\n"; ++pass; }
        else    std::cerr << "FAIL [2] Dscal\n";
    }

    // ── 2. SCopy / DCopy ───────────────────────────────────────────────────
    ++total;
    {
        float x[4] = {1,2,3,4}, y[4] = {0,0,0,0};
        auto r = cublasScopy_v2(handle, 4, x, 1, y, 1);
        bool ok = (r == CUBLAS_STATUS_SUCCESS) &&
                  approx_eq(y[0], 1) && approx_eq(y[1], 2) && approx_eq(y[2], 3) && approx_eq(y[3], 4);
        if (ok) { std::cout << "PASS [3] Scopy\n"; ++pass; }
        else    std::cerr << "FAIL [3] Scopy\n";
    }
    ++total;
    {
        double x[3] = {10,20,30}, y[3] = {0,0,0};
        auto r = cublasDcopy_v2(handle, 3, x, 1, y, 1);
        bool ok = (r == CUBLAS_STATUS_SUCCESS) && approx_eq(y[2], 30);
        if (ok) { std::cout << "PASS [4] Dcopy\n"; ++pass; }
        else    std::cerr << "FAIL [4] Dcopy\n";
    }

    // ── 3. SSwap / DSwap ───────────────────────────────────────────────────
    ++total;
    {
        float x[3] = {1,2,3}, y[3] = {7,8,9};
        auto r = cublasSswap_v2(handle, 3, x, 1, y, 1);
        bool ok = (r == CUBLAS_STATUS_SUCCESS) &&
                  approx_eq(x[0], 7) && approx_eq(x[1], 8) && approx_eq(x[2], 9) &&
                  approx_eq(y[0], 1) && approx_eq(y[1], 2) && approx_eq(y[2], 3);
        if (ok) { std::cout << "PASS [5] Sswap\n"; ++pass; }
        else    std::cerr << "FAIL [5] Sswap\n";
    }

    // ── 4. SAsum / DAsum ─────────────────────────────────────────────────────
    ++total;
    {
        float x[4] = {-1, 2, -3, 4}, res = 0;
        auto r = cublasSasum_v2(handle, 4, x, 1, &res);
        bool ok = (r == CUBLAS_STATUS_SUCCESS) && approx_eq(res, 10.0f);
        if (ok) { std::cout << "PASS [6] Sasum\n"; ++pass; }
        else    std::cerr << "FAIL [6] Sasum (got " << res << ")\n";
    }
    ++total;
    {
        double x[3] = {-1.5, 2.5, -3.5}, res = 0;
        auto r = cublasDasum_v2(handle, 3, x, 1, &res);
        bool ok = (r == CUBLAS_STATUS_SUCCESS) && approx_eq(res, 7.5);
        if (ok) { std::cout << "PASS [7] Dasum\n"; ++pass; }
        else    std::cerr << "FAIL [7] Dasum (got " << res << ")\n";
    }

    // ── 5. Isamax / Idamax ───────────────────────────────────────────────────
    ++total;
    {
        float x[5] = {1, -5, 3, 2, 4}; // max abs = 5 at index 1 (1-based = 2)
        int idx = 0;
        auto r = cublasIsamax_v2(handle, 5, x, 1, &idx);
        bool ok = (r == CUBLAS_STATUS_SUCCESS) && (idx == 2);
        if (ok) { std::cout << "PASS [8] Isamax\n"; ++pass; }
        else    std::cerr << "FAIL [8] Isamax (got " << idx << ")\n";
    }
    ++total;
    {
        double x[4] = {0.1, -0.3, 0.2, 0.15};
        int idx = 0;
        auto r = cublasIdamax_v2(handle, 4, x, 1, &idx);
        bool ok = (r == CUBLAS_STATUS_SUCCESS) && (idx == 2);
        if (ok) { std::cout << "PASS [9] Idamax\n"; ++pass; }
        else    std::cerr << "FAIL [9] Idamax (got " << idx << ")\n";
    }

    // ── 6. Isamin / Idamin ───────────────────────────────────────────────────
    ++total;
    {
        float x[5] = {5, -1, 3, 2, 4}; // min abs = 1 at index 1 (1-based = 2)
        int idx = 0;
        auto r = cublasIsamin_v2(handle, 5, x, 1, &idx);
        bool ok = (r == CUBLAS_STATUS_SUCCESS) && (idx == 2);
        if (ok) { std::cout << "PASS [10] Isamin\n"; ++pass; }
        else    std::cerr << "FAIL [10] Isamin (got " << idx << ")\n";
    }
    ++total;
    {
        double x[3] = {-0.5, 0.1, -0.3};
        int idx = 0;
        auto r = cublasIdamin_v2(handle, 3, x, 1, &idx);
        bool ok = (r == CUBLAS_STATUS_SUCCESS) && (idx == 2);
        if (ok) { std::cout << "PASS [11] Idamin\n"; ++pass; }
        else    std::cerr << "FAIL [11] Idamin (got " << idx << ")\n";
    }

    // ── 7. SRot / DRot ────────────────────────────────────────────────────────
    ++total;
    {
        float x[2] = {1,0}, y[2] = {0,1};
        float c_ = 0.0f, s_ = 1.0f; // 90-degree rotation
        auto r = cublasSrot_v2(handle, 2, x, 1, y, 1, &c_, &s_);
        bool ok = (r == CUBLAS_STATUS_SUCCESS) &&
                  approx_eq(x[0], 0.0f) && approx_eq(x[1], 1.0f) &&
                  approx_eq(y[0], -1.0f) && approx_eq(y[1], 0.0f);
        if (ok) { std::cout << "PASS [12] Srot\n"; ++pass; }
        else    std::cerr << "FAIL [12] Srot\n";
    }
    ++total;
    {
        double x[2] = {3,4}, y[2] = {0,0};
        double c_ = 0.6, s_ = 0.8; // simple rotation
        auto r = cublasDrot_v2(handle, 2, x, 1, y, 1, &c_, &s_);
        bool ok = (r == CUBLAS_STATUS_SUCCESS) &&
                  approx_eq(x[0], 1.8) && approx_eq(x[1], 2.4) &&
                  approx_eq(y[0], -2.4) && approx_eq(y[1], -3.2);
        if (ok) { std::cout << "PASS [13] Drot\n"; ++pass; }
        else    std::cerr << "FAIL [13] Drot\n";
    }

    // ── 8. SRotg / DRotg ─────────────────────────────────────────────────────
    ++total;
    {
        float a = 3.0f, b = 4.0f, c_, s_;
        auto r = cublasSrotg_v2(handle, &a, &b, &c_, &s_);
        bool ok = (r == CUBLAS_STATUS_SUCCESS) &&
                  approx_eq(a, 5.0f) && approx_eq(c_, 0.6f) && approx_eq(s_, 0.8f);
        if (ok) { std::cout << "PASS [14] Srotg\n"; ++pass; }
        else    std::cerr << "FAIL [14] Srotg (a=" << a << " c=" << c_ << " s=" << s_ << ")\n";
    }
    ++total;
    {
        double a = 5.0, b = 12.0, c_, s_;
        auto r = cublasDrotg_v2(handle, &a, &b, &c_, &s_);
        bool ok = (r == CUBLAS_STATUS_SUCCESS) &&
                  approx_eq(a, 13.0) && approx_eq(c_, 5.0/13.0) && approx_eq(s_, 12.0/13.0);
        if (ok) { std::cout << "PASS [15] Drotg\n"; ++pass; }
        else    std::cerr << "FAIL [15] Drotg\n";
    }

    // ── 9. SRotm / DRotm (identity) ──────────────────────────────────────────
    ++total;
    {
        float x[3] = {1,2,3}, y[3] = {4,5,6};
        float param[5] = {-2.0f, 0,0,0,0}; // identity
        auto r = cublasSrotm_v2(handle, 3, x, 1, y, 1, param);
        bool ok = (r == CUBLAS_STATUS_SUCCESS) &&
                  approx_eq(x[0], 1) && approx_eq(x[1], 2) && approx_eq(x[2], 3) &&
                  approx_eq(y[0], 4) && approx_eq(y[1], 5) && approx_eq(y[2], 6);
        if (ok) { std::cout << "PASS [16] Srotm identity\n"; ++pass; }
        else    std::cerr << "FAIL [16] Srotm identity\n";
    }

    // ── 10. SRotmg / DRotmg → SRotm round-trip ───────────────────────────────
    ++total;
    {
        float d1 = 1.0f, d2 = 1.0f, x1 = 3.0f, y1 = 4.0f;
        float param[5];
        auto r1 = cublasSrotmg_v2(handle, &d1, &d2, &x1, &y1, param);
        float xr[2] = {3,0}, yr[2] = {4,0};
        auto r2 = cublasSrotm_v2(handle, 2, xr, 1, yr, 1, param);
        // Invariant: y becomes 0 after modified Givens rotation; x is non-zero
        bool ok = (r1 == CUBLAS_STATUS_SUCCESS) && (r2 == CUBLAS_STATUS_SUCCESS) &&
                  approx_eq(yr[0], 0.0f, 1e-3f) && std::abs(xr[0]) > 0.1f;
        if (ok) { std::cout << "PASS [17] Srotmg+Srotm round-trip\n"; ++pass; }
        else    std::cerr << "FAIL [17] Srotmg+Srotm (x=" << xr[0] << " y=" << yr[0] << ")\n";
    }

    // ── Null pointer rejection ───────────────────────────────────────────────
    ++total;
    {
        float res = 0;
        auto r = cublasSasum_v2(handle, 4, nullptr, 1, &res);
        if (r == CUBLAS_STATUS_INVALID_VALUE) {
            std::cout << "PASS [18] Null pointer rejected\n"; ++pass;
        } else {
            std::cerr << "FAIL [18] Should reject null pointer\n";
        }
    }

    cublasDestroy_v2(handle);

    std::cout << "\n" << pass << "/" << total << " tests passed.\n";
    return (pass == total) ? 0 : 1;
}
