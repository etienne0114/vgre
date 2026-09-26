// Proves the previously-orphaned vgre::math kernels are now wired and working
// through the public C API (the surface host code, device kernels and cluster
// workers share): accelerated INT8 tensor-core GEMM, block-sparse SpMV, and
// cache-oblivious dense GEMM. Each is checked against an independent reference.

#include "vgre/api/vgre_c_api.h"

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>

static int g_fail = 0;
#define CHECK(cond, msg)                                                   \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("FAIL: %s  (%s:%d)\n", (msg), __FILE__, __LINE__); \
            ++g_fail;                                                      \
        }                                                                  \
    } while (0)

int main() {
    // ── 1. tensor-core capability query never errors, returns booleans ──────
    {
        int vnni = -1, bf16 = -1, amx = -1;
        CHECK(vgre_tensor_core_caps(&vnni, &bf16, &amx) == 0, "vgre_tensor_core_caps");
        CHECK((vnni == 0 || vnni == 1) && (bf16 == 0 || bf16 == 1) && (amx == 0 || amx == 1),
              "caps are 0/1");
        std::printf("tensor-core caps: VNNI=%d BF16=%d AMX=%d\n", vnni, bf16, amx);
    }

    // ── 2. INT8 tensor-core GEMM vs int32 scalar reference ──────────────────
    {
        const size_t m = 5, n = 6, k = 7;  // C[m×n] = A[m×k]·B[k×n]
        std::vector<int8_t> A(m * k), B(k * n);
        std::vector<int32_t> C(m * n, 0), Ref(m * n, 0);
        for (size_t i = 0; i < A.size(); ++i) A[i] = (int8_t)((int)(i * 3 % 31) - 15);
        for (size_t i = 0; i < B.size(); ++i) B[i] = (int8_t)((int)(i * 5 % 29) - 14);
        CHECK(vgre_tensor_core_gemm_int8(A.data(), B.data(), C.data(), m, n, k) == 0,
              "vgre_tensor_core_gemm_int8");
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < n; ++j) {
                int32_t acc = 0;
                for (size_t p = 0; p < k; ++p) acc += (int32_t)A[i * k + p] * (int32_t)B[p * n + j];
                Ref[i * n + j] = acc;
            }
        for (size_t i = 0; i < C.size(); ++i)
            CHECK(C[i] == Ref[i], "INT8 tensor-core GEMM bit-exact vs scalar");
    }

    // ── 3. Block-sparse SpMV vs dense reference ─────────────────────────────
    {
        // 4×4 matrix, block_size 2, in CSR. Dense form:
        //   [ 1 2 0 0 ]
        //   [ 3 4 0 0 ]
        //   [ 0 0 5 6 ]
        //   [ 0 0 7 8 ]
        const int32_t num_rows = 4, num_cols = 4, block_size = 2;
        std::vector<float>   values      = {1, 2, 3, 4, 5, 6, 7, 8};
        std::vector<int32_t> col_indices = {0, 1, 0, 1, 2, 3, 2, 3};
        std::vector<int32_t> row_offsets = {0, 2, 4, 6, 8};
        std::vector<float>   x           = {1, 1, 1, 1};
        std::vector<float>   y(num_rows, 0.f);
        CHECK(vgre_block_sparse_spmv(values.data(), col_indices.data(), row_offsets.data(),
                                     num_rows, num_cols, block_size, x.data(), y.data()) == 0,
              "vgre_block_sparse_spmv");
        // Dense reference y = A·x
        float dense[4][4] = {{1,2,0,0},{3,4,0,0},{0,0,5,6},{0,0,7,8}};
        for (int r = 0; r < 4; ++r) {
            float acc = 0.f;
            for (int c = 0; c < 4; ++c) acc += dense[r][c] * x[c];
            CHECK(std::fabs(y[r] - acc) < 1e-5f, "block-sparse SpMV vs dense");
        }
    }

    // ── 4. Cache-oblivious dense GEMM vs scalar reference ───────────────────
    {
        const size_t m = 6, n = 5, p = 7;  // C[m×p] = A[m×n]·B[n×p]
        std::vector<float> A(m * n), B(n * p), C(m * p, 0.f), Ref(m * p, 0.f);
        for (size_t i = 0; i < A.size(); ++i) A[i] = 0.25f * (float)((int)(i % 9) - 4);
        for (size_t i = 0; i < B.size(); ++i) B[i] = 0.5f * (float)((int)(i % 7) - 3);
        CHECK(vgre_cache_oblivious_matmul(A.data(), B.data(), C.data(), m, n, p) == 0,
              "vgre_cache_oblivious_matmul");
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < p; ++j) {
                float acc = 0.f;
                for (size_t q = 0; q < n; ++q) acc += A[i * n + q] * B[q * p + j];
                Ref[i * p + j] = acc;
            }
        for (size_t i = 0; i < C.size(); ++i)
            CHECK(std::fabs(C[i] - Ref[i]) < 1e-4f, "cache-oblivious GEMM vs scalar");
    }

    if (g_fail == 0) std::printf("test_math_kernels_capi: ALL PASS\n");
    else             std::printf("test_math_kernels_capi: %d FAILURE(S)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
