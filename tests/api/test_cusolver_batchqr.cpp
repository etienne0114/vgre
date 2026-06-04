// QUEUE-40: cuSolver batch QR factorization tests.
// Verifies cusolverDnS/DgeqrfBatched correctness by checking that Q*R == A_original
// for each matrix in the batch, where Q is reconstructed from Householder reflectors.
//
// Math: A = Q * R (Householder QR), so ||A - Q*R||_F / ||A||_F < epsilon.
// Q is formed as Q = H_1 * H_2 * ... * H_k where H_j = I - tau_j * v_j * v_j^T.
//
// Convention: matrices are stored column-major (LAPACK/cuSOLVER convention).
//   lda >= m; element (r, c) is at A[c * lda + r].
//
// Tests:
//   1. Float batch B=4, m=6, n=4  — tall matrices, standard case
//   2. Float batch B=2, m=4, n=4  — square matrices
//   3. Double batch B=3, m=5, n=5 — square matrices
//   4. Double batch B=1, m=4, n=4 — batch size 1 (matches non-batch geqrf)
//   5. Double batch B=5, m=8, n=3 — tall wide batch
//   6. Null pointer inputs → INVALID_VALUE
//   7. Bad parameters (m=0, n=0, lda<m) → INVALID_VALUE

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>

extern "C" {
typedef int    cusolverStatus_t;
typedef void*  cusolverDnHandle_t;
#define CUSOLVER_STATUS_SUCCESS       0
#define CUSOLVER_STATUS_INVALID_VALUE 3

cusolverStatus_t cusolverDnCreate(cusolverDnHandle_t*);
cusolverStatus_t cusolverDnDestroy(cusolverDnHandle_t);
cusolverStatus_t cusolverDnSgeqrfBatched(cusolverDnHandle_t, int m, int n,
    float **Aarray, int lda, float **TauArray, int *info, int batchSize);
cusolverStatus_t cusolverDnDgeqrfBatched(cusolverDnHandle_t, int m, int n,
    double **Aarray, int lda, double **TauArray, int *info, int batchSize);
}

#define PASS(msg) do { printf("PASS: %s\n", (msg)); } while(0)
#define FAIL(msg) do { fprintf(stderr, "FAIL: %s\n", (msg)); return 1; } while(0)
#define CHECK(call, msg) do { if ((call) != CUSOLVER_STATUS_SUCCESS) FAIL(msg); } while(0)

// Seeded LCG
static float randf(unsigned& seed) {
    seed = seed * 1664525u + 1013904223u;
    return (float)(seed >> 16) / 65536.f - 0.5f;
}
static double randd(unsigned& seed) {
    seed = seed * 1664525u + 1013904223u;
    unsigned s2 = seed * 1664525u + 1013904223u; seed = s2;
    return (double)(seed >> 8) / (double)(1u << 24) - 0.5;
}

// Apply a single Householder reflector H = I - tau*v*v^T to matrix C on the left.
// C is m×n_cols stored column-major with leading dimension ldc: C[r,c] = C[c*ldc+r].
// v[j..m-1] with v[j]=1 (implicit); v[r] provided for r > j.
template<typename T>
static void apply_householder_left(int m, int n_cols, int j,
    const T* v, T tau, T* C, int ldc)
{
    // w[c] = tau * sum_{r=j}^{m-1} v[r] * C[r,c]
    std::vector<T> w(n_cols, T(0));
    for (int r = j; r < m; ++r) {
        T vr = (r == j) ? T(1) : v[r];
        for (int c = 0; c < n_cols; ++c)
            w[c] += tau * vr * C[c * ldc + r];
    }
    // C[r,c] -= v[r] * w[c]
    for (int r = j; r < m; ++r) {
        T vr = (r == j) ? T(1) : v[r];
        for (int c = 0; c < n_cols; ++c)
            C[c * ldc + r] -= vr * w[c];
    }
}

// Reconstruct Q (m×m column-major, ldq=m) from Householder vectors in A (column-major, lda).
// After geqrf, the Householder vector for reflector j is in A[j*lda+r] for r > j.
template<typename T>
static void form_Q(int m, int n, const T* A, int lda, const T* tau, T* Q, int ldq) {
    // Initialize Q = I (column-major: Q[r,c] = Q[c*ldq+r])
    for (int r = 0; r < m; ++r)
        for (int c = 0; c < m; ++c)
            Q[c * ldq + r] = (r == c) ? T(1) : T(0);
    // Apply reflectors in reverse order: H_{k-1} ... H_0
    int k = std::min(m, n);
    for (int j = k - 1; j >= 0; --j) {
        // Householder vector v: v[r] = A[j*lda+r] for r > j; v[j] = 1 implicitly
        std::vector<T> v(m, T(0));
        for (int r = j + 1; r < m; ++r) v[r] = A[j * lda + r];
        apply_householder_left(m, m, j, v.data(), tau[j], Q, ldq);
    }
}

// Check Q*R == A_orig within relative tolerance.
// All matrices are column-major: element (r,c) at A[c*lda+r].
template<typename T>
static bool check_qr(int m, int n, const T* A_fac, int lda, const T* tau,
                     const T* A_orig, double tol, const char* label)
{
    // Form Q (m×m column-major, ldq=m)
    std::vector<T> Q(m * m);
    form_Q(m, n, A_fac, lda, tau, Q.data(), m);

    // Extract R: upper triangle of A_fac (column-major m×n, ldR=m)
    std::vector<T> R(m * n, T(0));
    for (int r = 0; r < std::min(m,n); ++r)
        for (int c = r; c < n; ++c)
            R[c * m + r] = A_fac[c * lda + r];

    // Compute QR = Q(m×m) * R(m×n), both column-major
    std::vector<T> QR(m * n, T(0));
    for (int r = 0; r < m; ++r)
        for (int c = 0; c < n; ++c)
            for (int kk = 0; kk < m; ++kk)
                QR[c * m + r] += Q[kk * m + r] * R[c * m + kk];

    // Compute ||QR - A_orig||_F / ||A_orig||_F
    T err = T(0), nrm = T(0);
    for (int r = 0; r < m; ++r)
        for (int c = 0; c < n; ++c) {
            T d = QR[c * m + r] - A_orig[c * lda + r];
            err += d * d;
            T a = A_orig[c * lda + r]; nrm += a * a;
        }
    double rel = (nrm > 0) ? std::sqrt((double)err / (double)nrm) : std::sqrt((double)err);
    if (rel > tol) {
        fprintf(stderr, "  %s: rel_err=%g (tol=%g)\n", label, rel, tol);
        return false;
    }
    return true;
}

static int test_sgeqrf_batched(int m, int n, int batchSize, const char* label) {
    cusolverDnHandle_t h = nullptr;
    CHECK(cusolverDnCreate(&h), "cusolverDnCreate");

    unsigned seed = 0xDEAD0000u ^ (unsigned)(m * n * batchSize);
    int lda = m;  // column-major: lda = m
    std::vector<std::vector<float>> mats(batchSize, std::vector<float>((size_t)lda * n));
    std::vector<std::vector<float>> taus(batchSize, std::vector<float>(std::min(m, n)));
    for (int b = 0; b < batchSize; ++b)
        for (auto& v : mats[b]) v = randf(seed);

    // Save originals before factorization
    std::vector<std::vector<float>> orig(batchSize);
    for (int b = 0; b < batchSize; ++b) orig[b] = mats[b];

    // Build pointer arrays
    std::vector<float*> Aarray(batchSize), TauArray(batchSize);
    for (int b = 0; b < batchSize; ++b) {
        Aarray[b] = mats[b].data();
        TauArray[b] = taus[b].data();
    }
    int info = -99;
    CHECK(cusolverDnSgeqrfBatched(h, m, n, Aarray.data(), lda, TauArray.data(), &info, batchSize),
          "SgeqrfBatched");
    if (info != 0) FAIL("SgeqrfBatched returned non-zero info");

    // Verify each batch element
    for (int b = 0; b < batchSize; ++b) {
        char buf[64]; snprintf(buf, sizeof(buf), "float b=%d", b);
        if (!check_qr(m, n, mats[b].data(), lda, taus[b].data(), orig[b].data(), 1e-4, buf)) {
            FAIL("SgeqrfBatched QR residual too large");
        }
    }
    cusolverDnDestroy(h);
    printf("PASS: %s\n", label);
    return 0;
}

static int test_dgeqrf_batched(int m, int n, int batchSize, const char* label) {
    cusolverDnHandle_t h = nullptr;
    CHECK(cusolverDnCreate(&h), "cusolverDnCreate");

    unsigned seed = 0xCAFEBABEu ^ (unsigned)(m * n * batchSize);
    int lda = m;  // column-major: lda = m
    std::vector<std::vector<double>> mats(batchSize, std::vector<double>((size_t)lda * n));
    std::vector<std::vector<double>> taus(batchSize, std::vector<double>(std::min(m, n)));
    for (int b = 0; b < batchSize; ++b)
        for (auto& v : mats[b]) v = randd(seed);

    std::vector<std::vector<double>> orig(batchSize);
    for (int b = 0; b < batchSize; ++b) orig[b] = mats[b];

    std::vector<double*> Aarray(batchSize), TauArray(batchSize);
    for (int b = 0; b < batchSize; ++b) {
        Aarray[b] = mats[b].data();
        TauArray[b] = taus[b].data();
    }
    int info = -99;
    CHECK(cusolverDnDgeqrfBatched(h, m, n, Aarray.data(), lda, TauArray.data(), &info, batchSize),
          "DgeqrfBatched");
    if (info != 0) FAIL("DgeqrfBatched returned non-zero info");

    for (int b = 0; b < batchSize; ++b) {
        char buf[64]; snprintf(buf, sizeof(buf), "double b=%d", b);
        if (!check_qr(m, n, mats[b].data(), lda, taus[b].data(), orig[b].data(), 1e-10, buf)) {
            FAIL("DgeqrfBatched QR residual too large");
        }
    }
    cusolverDnDestroy(h);
    printf("PASS: %s\n", label);
    return 0;
}

static int test_invalid_params() {
    cusolverDnHandle_t h = nullptr;
    CHECK(cusolverDnCreate(&h), "cusolverDnCreate");

    float  dummy_f  = 0.f;
    double dummy_d  = 0.0;
    float  *pf  = &dummy_f;
    double *pd  = &dummy_d;
    float  *pft = &dummy_f;
    double *pdt = &dummy_d;
    int info = 0;

    // Null Aarray
    if (cusolverDnSgeqrfBatched(h, 4, 4, nullptr, 4, &pft, &info, 1) != CUSOLVER_STATUS_INVALID_VALUE)
        FAIL("null Aarray must return INVALID_VALUE");
    // Null TauArray
    if (cusolverDnSgeqrfBatched(h, 4, 4, &pf, 4, nullptr, &info, 1) != CUSOLVER_STATUS_INVALID_VALUE)
        FAIL("null TauArray must return INVALID_VALUE");
    // Null info
    if (cusolverDnSgeqrfBatched(h, 4, 4, &pf, 4, &pft, nullptr, 1) != CUSOLVER_STATUS_INVALID_VALUE)
        FAIL("null info must return INVALID_VALUE");
    // m=0
    if (cusolverDnSgeqrfBatched(h, 0, 4, &pf, 4, &pft, &info, 1) != CUSOLVER_STATUS_INVALID_VALUE)
        FAIL("m=0 must return INVALID_VALUE");
    // n=0
    if (cusolverDnSgeqrfBatched(h, 4, 0, &pf, 4, &pft, &info, 1) != CUSOLVER_STATUS_INVALID_VALUE)
        FAIL("n=0 must return INVALID_VALUE");
    // lda < m
    if (cusolverDnSgeqrfBatched(h, 4, 4, &pf, 3, &pft, &info, 1) != CUSOLVER_STATUS_INVALID_VALUE)
        FAIL("lda<m must return INVALID_VALUE");
    // batchSize=0
    if (cusolverDnSgeqrfBatched(h, 4, 4, &pf, 4, &pft, &info, 0) != CUSOLVER_STATUS_INVALID_VALUE)
        FAIL("batchSize=0 must return INVALID_VALUE");
    // double variants
    if (cusolverDnDgeqrfBatched(h, 4, 4, nullptr, 4, &pdt, &info, 1) != CUSOLVER_STATUS_INVALID_VALUE)
        FAIL("double null Aarray must return INVALID_VALUE");
    if (cusolverDnDgeqrfBatched(h, 0, 4, &pd, 4, &pdt, &info, 1) != CUSOLVER_STATUS_INVALID_VALUE)
        FAIL("double m=0 must return INVALID_VALUE");

    cusolverDnDestroy(h);
    PASS("invalid parameter checks (INVALID_VALUE)");
    return 0;
}

int main() {
    int rc = 0;
    rc |= test_sgeqrf_batched(6, 4, 4, "float  m=6, n=4, batch=4 (tall)");
    rc |= test_sgeqrf_batched(4, 4, 2, "float  m=4, n=4, batch=2 (square)");
    rc |= test_dgeqrf_batched(5, 5, 3, "double m=5, n=5, batch=3 (square)");
    rc |= test_dgeqrf_batched(4, 4, 1, "double m=4, n=4, batch=1 (single)");
    rc |= test_dgeqrf_batched(8, 3, 5, "double m=8, n=3, batch=5 (tall wide batch)");
    rc |= test_invalid_params();
    if (rc == 0)
        printf("All cuSolver batch QR QUEUE-40 tests passed.\n");
    return rc;
}
