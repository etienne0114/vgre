// Track 26 — golden numerical-vector suite.
//
// A consolidated set of fixed inputs checked against INDEPENDENT, hand/NumPy-
// derived golden outputs (not VGRE's own naive reference), at tight tolerance:
//   - cuFFT C2C forward vs the exact DFT of known signals,
//   - cuBLAS Sgemm vs a layout-independent diagonal product,
//   - cuRAND uniform/normal vs theoretical distribution moments.

#include "vgre/api/cufft_shim.h"
#include "vgre/api/curand_shim.h"

#include <cmath>
#include <complex>
#include <cstdio>
#include <vector>

// ── cuBLAS (declared locally, like the existing cuBLAS tests) ────────────────
extern "C" {
typedef void *cublasHandle_t;
typedef int cublasOperation_t;
typedef int cublasStatus_t;
cublasStatus_t cublasCreate_v2(cublasHandle_t *handle);
cublasStatus_t cublasDestroy_v2(cublasHandle_t handle);
cublasStatus_t cublasSgemm_v2(cublasHandle_t handle, cublasOperation_t transa,
                              cublasOperation_t transb, int m, int n, int k,
                              const float *alpha, const float *A, int lda,
                              const float *B, int ldb, const float *beta,
                              float *C, int ldc);
}
#define CUBLAS_OP_N 0
#define CUBLAS_OK 0

using cfloat = std::complex<float>;

static int g_pass = 0, g_total = 0;
static void check(const char *name, bool ok) {
    ++g_total;
    printf(ok ? "  PASS  %s\n" : "  FAIL  %s\n", name);
    if (ok) ++g_pass;
}

static bool cmatch(const std::vector<cfloat> &got, const std::vector<cfloat> &gold,
                   float tol) {
    if (got.size() != gold.size()) return false;
    for (size_t i = 0; i < got.size(); ++i)
        if (std::abs(got[i] - gold[i]) > tol) return false;
    return true;
}

int main() {
    printf("=== Golden numerical-vector suite (Track 26) ===\n");

    // ── cuFFT C2C forward: exact DFT golden ──────────────────────────────────
    {
        // δ[0] = {1,0,0,0}  →  DFT = {1,1,1,1}  (a delta has a flat spectrum)
        std::vector<cfloat> in = {{1, 0}, {0, 0}, {0, 0}, {0, 0}}, out(4);
        cufftHandle p; cufftPlan1d(&p, 4, CUFFT_C2C, 1);
        cufftExecC2C(p, in.data(), out.data(), CUFFT_FORWARD);
        cufftDestroy(p);
        check("cuFFT DFT(delta) == [1,1,1,1] (1e-4)",
              cmatch(out, {{1, 0}, {1, 0}, {1, 0}, {1, 0}}, 1e-4f));
    }
    {
        // DC = {1,1,1,1}  →  DFT = {4,0,0,0}
        std::vector<cfloat> in = {{1, 0}, {1, 0}, {1, 0}, {1, 0}}, out(4);
        cufftHandle p; cufftPlan1d(&p, 4, CUFFT_C2C, 1);
        cufftExecC2C(p, in.data(), out.data(), CUFFT_FORWARD);
        cufftDestroy(p);
        check("cuFFT DFT([1,1,1,1]) == [4,0,0,0] (1e-4)",
              cmatch(out, {{4, 0}, {0, 0}, {0, 0}, {0, 0}}, 1e-4f));
    }
    {
        // A single cosine at bin 1, N=8: cos(2πk/8) → DFT peaks 4 at bins 1 and 7.
        const int N = 8;
        std::vector<cfloat> in(N), out(N);
        for (int k = 0; k < N; ++k) in[k] = cfloat(std::cos(2.0f * float(M_PI) * k / N), 0);
        cufftHandle p; cufftPlan1d(&p, N, CUFFT_C2C, 1);
        cufftExecC2C(p, in.data(), out.data(), CUFFT_FORWARD);
        cufftDestroy(p);
        bool ok = std::abs(std::abs(out[1]) - 4.0f) < 1e-3f &&
                  std::abs(std::abs(out[7]) - 4.0f) < 1e-3f &&
                  std::abs(out[0]) < 1e-3f && std::abs(out[2]) < 1e-3f;
        check("cuFFT DFT(cos@bin1, N=8) peaks 4 at bins 1 and 7", ok);
    }

    // ── cuBLAS Sgemm: diagonal product (layout-independent golden) ───────────
    {
        cublasHandle_t h = nullptr;
        bool ok = (cublasCreate_v2(&h) == CUBLAS_OK);
        // A=diag(2,3), B=diag(5,7) (column-major == row-major for diagonals).
        float A[4] = {2, 0, 0, 3}, B[4] = {5, 0, 0, 7}, C[4] = {0, 0, 0, 0};
        float alpha = 1.0f, beta = 0.0f;
        cublasSgemm_v2(h, CUBLAS_OP_N, CUBLAS_OP_N, 2, 2, 2, &alpha, A, 2, B, 2, &beta, C, 2);
        cublasDestroy_v2(h);
        // diag(2,3)·diag(5,7) = diag(10,21) → column-major {10,0,0,21}.
        ok = ok && std::fabs(C[0] - 10) < 1e-4f && std::fabs(C[1]) < 1e-4f &&
                   std::fabs(C[2]) < 1e-4f && std::fabs(C[3] - 21) < 1e-4f;
        check("cuBLAS Sgemm diag(2,3)·diag(5,7) == diag(10,21)", ok);
    }

    // ── cuRAND distribution moments vs theory ────────────────────────────────
    {
        const size_t N = 200000;
        curandGenerator_t g = nullptr;
        curandCreateGenerator(&g, CURAND_RNG_PSEUDO_XORWOW);
        curandSetPseudoRandomGeneratorSeed(g, 12345ULL);

        std::vector<float> u(N);
        curandGenerateUniform(g, u.data(), N);
        double mean = 0, var = 0;
        for (float x : u) mean += x; mean /= N;
        for (float x : u) var += (x - mean) * (x - mean); var /= N;
        printf("  [info] uniform mean=%.4f (exp 0.5)  var=%.4f (exp %.4f)\n", mean, var, 1.0 / 12);
        // In [0,1): all samples in range, mean≈0.5, var≈1/12≈0.0833.
        bool inRange = true; for (float x : u) if (x < 0.0f || x >= 1.0f) inRange = false;
        check("cuRAND uniform in [0,1)", inRange);
        check("cuRAND uniform mean≈0.5, var≈1/12", std::fabs(mean - 0.5) < 0.01 &&
                                                   std::fabs(var - 1.0 / 12) < 0.01);

        std::vector<float> nrm(N);
        curandGenerateNormal(g, nrm.data(), N, /*mean=*/0.0f, /*stddev=*/1.0f);
        double nm = 0, nv = 0;
        for (float x : nrm) nm += x; nm /= N;
        for (float x : nrm) nv += (x - nm) * (x - nm); nv /= N;
        printf("  [info] normal mean=%.4f (exp 0)  var=%.4f (exp 1)\n", nm, nv);
        check("cuRAND normal(0,1) mean≈0, var≈1", std::fabs(nm) < 0.02 && std::fabs(nv - 1.0) < 0.05);

        curandDestroyGenerator(g);
    }

    printf("\n%d / %d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
