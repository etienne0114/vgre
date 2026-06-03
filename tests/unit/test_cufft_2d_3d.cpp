// cuFFT 2D/3D test coverage — QUEUE-26
// Validates cufftPlan2d, cufftPlan3d, batched 2D plans, R2C/C2R 2D roundtrip,
// and the pure-frequency DFT identity for 2D complex exponentials.

#include "vgre/api/cufft_shim.h"

#include <cmath>
#include <complex>
#include <cstdio>
#include <iostream>
#include <vector>

using cfloat = std::complex<float>;
using cReal  = float;

static int g_pass = 0, g_total = 0;

static void check(const char *name, bool ok) {
    ++g_total;
    if (ok) { ++g_pass; std::cout << "PASS [" << g_total << "] " << name << "\n"; }
    else    { std::cerr << "FAIL [" << g_total << "] " << name << "\n"; }
}

// ── Test 1: 2D C2C forward + inverse roundtrip ───────────────────────────────
// Math invariant: IFFT(FFT(x)) = x for any x (up to floating-point precision).
// cufftPlan2d(nx=4, ny=4) with CUFFT_C2C.  Data layout: element [row][col] is
// at index row*ny + col (ny = 4).  Forward then inverse.  The implementation
// normalises each 1D pass internally (fft1d divides by length on INVERSE), so
// the two-pass (row then column) inverse yields the original without any
// additional caller-side division.
// Verify: |reconstructed[i] - original[i]| < 1e-4 for all 16 elements.
static void test_cufft_2d_c2c_roundtrip() {
    const int NX = 4, NY = 4;
    const int N  = NX * NY;
    std::vector<cfloat> in(N), fwd(N), inv(N);

    for (int r = 0; r < NX; ++r)
        for (int c = 0; c < NY; ++c)
            in[r * NY + c] = cfloat(
                std::sin(2.0f * 3.14159265f * r / NX + 0.3f * c),
                std::cos(1.5f * 3.14159265f * c / NY - 0.7f * r));

    cufftHandle plan;
    bool ok = (cufftPlan2d(&plan, NX, NY, CUFFT_C2C) == CUFFT_SUCCESS);
    ok &= (cufftExecC2C(plan, in.data(), fwd.data(), CUFFT_FORWARD)  == CUFFT_SUCCESS);
    ok &= (cufftExecC2C(plan, fwd.data(), inv.data(), CUFFT_INVERSE) == CUFFT_SUCCESS);
    cufftDestroy(plan);

    for (int i = 0; i < N && ok; ++i)
        ok &= (std::abs(in[i] - inv[i]) < 1e-4f);

    check("2D C2C forward+inverse roundtrip (4x4)", ok);
}

// ── Test 2: 2D R2C + C2R roundtrip ──────────────────────────────────────────
// Math invariant: IFFT(FFT(x_real)) = x_real for a real-valued signal.
// 8x8 real input (64 floats).  R2C produces the full NX*NY complex spectrum
// internally (the shim promotes real→complex before calling fft2d).  C2R then
// calls fft2d inverse and takes the real part; fft1d normalises each 1D pass
// internally, so no additional division is needed by the caller.
// Verify: |out[i] - in[i]| < 1e-4 for all 64 elements.
static void test_cufft_2d_r2c_c2r_roundtrip() {
    const int NX = 8, NY = 8;
    const int N  = NX * NY;
    std::vector<cReal>  in(N), out(N);
    std::vector<cfloat> freq(N);   // full spectrum (shim stores all NX*NY bins)

    for (int r = 0; r < NX; ++r)
        for (int c = 0; c < NY; ++c)
            in[r * NY + c] = std::sin(2.0f * 3.14159265f * r / NX)
                           + 0.5f * std::cos(2.0f * 3.14159265f * c / NY);

    cufftHandle plan_r2c, plan_c2r;
    bool ok = (cufftPlan2d(&plan_r2c, NX, NY, CUFFT_R2C) == CUFFT_SUCCESS);
    ok &= (cufftPlan2d(&plan_c2r, NX, NY, CUFFT_C2R) == CUFFT_SUCCESS);
    ok &= (cufftExecR2C(plan_r2c, in.data(),   freq.data()) == CUFFT_SUCCESS);
    ok &= (cufftExecC2R(plan_c2r, freq.data(), out.data())  == CUFFT_SUCCESS);
    cufftDestroy(plan_r2c);
    cufftDestroy(plan_c2r);

    for (int i = 0; i < N && ok; ++i)
        ok &= (std::fabs(out[i] - in[i]) < 1e-4f);

    check("2D R2C+C2R roundtrip (8x8)", ok);
}

// ── Test 3: 3D C2C forward + inverse roundtrip ───────────────────────────────
// Math invariant: 3D DFT separability — FFT over dim0 (nx), dim1 (ny), dim2 (nz)
// independently.  IFFT(FFT(x)) = x for any input x.
// Layout: element [z][y][x] at index (z*ny + y)*nx + x.
// 2x4x4 complex (32 elements).  Each 1D pass normalises internally on INVERSE.
// Verify reconstruction within 1e-4.
static void test_cufft_3d_c2c_roundtrip() {
    const int NX = 4, NY = 4, NZ = 2;
    const int N  = NX * NY * NZ;
    std::vector<cfloat> in(N), fwd(N), inv(N);

    for (int z = 0; z < NZ; ++z)
        for (int y = 0; y < NY; ++y)
            for (int x = 0; x < NX; ++x) {
                int idx = (z * NY + y) * NX + x;
                in[idx] = cfloat(
                    std::cos(2.0f * 3.14159265f * x / NX + 0.5f * y),
                    std::sin(2.0f * 3.14159265f * y / NY + 0.3f * z));
            }

    cufftHandle plan;
    bool ok = (cufftPlan3d(&plan, NX, NY, NZ, CUFFT_C2C) == CUFFT_SUCCESS);
    ok &= (cufftExecC2C(plan, in.data(), fwd.data(), CUFFT_FORWARD)  == CUFFT_SUCCESS);
    ok &= (cufftExecC2C(plan, fwd.data(), inv.data(), CUFFT_INVERSE) == CUFFT_SUCCESS);
    cufftDestroy(plan);

    for (int i = 0; i < N && ok; ++i)
        ok &= (std::abs(in[i] - inv[i]) < 1e-4f);

    check("3D C2C forward+inverse roundtrip (4x4x2)", ok);
}

// ── Test 4: 2D batched C2C — independence of batch elements ─────────────────
// Math invariant: batch elements are independent — the transform applied to
// batch b equals cufftPlan2d applied solely to the sub-array at offset b*nx*ny.
//
// Batch 0: DC impulse — x[0,0]=1, all others 0.
//   DFT of a DC impulse is a constant: X[k1,k2] = 1 for all (k1,k2).
//
// Batch 1: impulse at [1,0] — x[1,0]=1, all others 0.
//   DFT: X[k1,k2] = exp(-2πi * k1/N1)  →  |X[k1,k2]| = 1 for all bins.
//   Crucially, X[0,0] for batch 1 is exp(-2πi*0/4)=1, same magnitude but
//   the spectrum of batch 1 must NOT contaminate batch 0's output.
//
// Verification:
//   - All 16 bins of batch-0 output have |X| ≈ 1 (DC impulse → flat spectrum).
//   - All 16 bins of batch-1 output have |X| ≈ 1 (single-point impulse, unit energy).
//   - Batch 0 and batch 1 outputs differ (non-contamination): batch-1 bin [1,0]
//     has phase exp(-2πi/4) = -j, whereas batch-0 bin [1,0] is 1+0j.
static void test_cufft_2d_batch() {
    const int NX = 4, NY = 4;
    const int N  = NX * NY;   // elements per batch
    const int B  = 2;

    std::vector<cfloat> in(N * B, cfloat(0.0f, 0.0f));
    std::vector<cfloat> out(N * B);

    // Batch 0: DC impulse at [row=0, col=0]
    in[0 * N + 0 * NY + 0] = cfloat(1.0f, 0.0f);

    // Batch 1: impulse at [row=1, col=0]
    in[1 * N + 1 * NY + 0] = cfloat(1.0f, 0.0f);

    // cufftPlanMany for batched 2D C2C:
    //   rank=2, n={NX,NY}, simple contiguous layout.
    //   istride=1, idist=N, ostride=1, odist=N.
    int dims[2] = {NX, NY};
    cufftHandle plan;
    bool ok = (cufftPlanMany(&plan, 2, dims,
                             nullptr, 1, N,
                             nullptr, 1, N,
                             CUFFT_C2C, B) == CUFFT_SUCCESS);
    ok &= (cufftExecC2C(plan, in.data(), out.data(), CUFFT_FORWARD) == CUFFT_SUCCESS);
    cufftDestroy(plan);

    // Batch 0: all bins should have magnitude 1 (flat spectrum of DC impulse)
    for (int i = 0; i < N && ok; ++i)
        ok &= (std::fabs(std::abs(out[0 * N + i]) - 1.0f) < 1e-4f);

    // Batch 1: all bins should have magnitude 1 (single-point impulse)
    for (int i = 0; i < N && ok; ++i)
        ok &= (std::fabs(std::abs(out[1 * N + i]) - 1.0f) < 1e-4f);

    // Non-contamination: batch-0 bin at [row=1,col=0] should be 1+0j (real=1, imag=0)
    // while batch-1 bin at [row=1,col=0] has imaginary part ~= -1 (phase = exp(-2πi/4) = -j)
    if (ok) {
        cfloat b0_bin_1_0 = out[0 * N + 1 * NY + 0];
        cfloat b1_bin_1_0 = out[1 * N + 1 * NY + 0];
        // Batch 0 [1,0] = 1+0j
        ok &= (std::fabs(b0_bin_1_0.real() - 1.0f) < 1e-4f);
        ok &= (std::fabs(b0_bin_1_0.imag())         < 1e-4f);
        // Batch 1 [1,0] = exp(-2πi*1/4) = 0 - 1j  (imag ≈ -1)
        ok &= (std::fabs(b1_bin_1_0.real())          < 1e-4f);
        ok &= (std::fabs(b1_bin_1_0.imag() + 1.0f)  < 1e-4f);
    }

    check("2D batched C2C independence (batch=2, 4x4)", ok);
}

// ── Test 5: 2D single-frequency DFT identity ─────────────────────────────────
// Math invariant: DFT of a pure 2D complex exponential
//   x[n1,n2] = exp(2πi*(f1*n1/N1 + f2*n2/N2))
// has exactly one nonzero bin at (f1, f2) with magnitude N1*N2.
//
// Parameters: N1=N2=8, f1=2, f2=3.
// After forward FFT:
//   |X[2,3]| ≈ 64 (= N1*N2 = 8*8)
//   All other |X[k1,k2]| < 1e-3
//
// Layout: element [n1][n2] at index n1*N2 + n2 (row=n1, col=n2).
static void test_cufft_2d_single_frequency() {
    const int N1 = 8, N2 = 8;
    const int N  = N1 * N2;
    const int f1 = 2, f2 = 3;
    const float pi2 = 2.0f * 3.14159265358979f;

    std::vector<cfloat> in(N), out(N);

    for (int n1 = 0; n1 < N1; ++n1)
        for (int n2 = 0; n2 < N2; ++n2) {
            float angle = pi2 * (static_cast<float>(f1 * n1) / N1
                                + static_cast<float>(f2 * n2) / N2);
            in[n1 * N2 + n2] = cfloat(std::cos(angle), std::sin(angle));
        }

    cufftHandle plan;
    bool ok = (cufftPlan2d(&plan, N1, N2, CUFFT_C2C) == CUFFT_SUCCESS);
    ok &= (cufftExecC2C(plan, in.data(), out.data(), CUFFT_FORWARD) == CUFFT_SUCCESS);
    cufftDestroy(plan);

    // Peak bin at (f1, f2) should have magnitude N1*N2 = 64
    float peak = std::abs(out[f1 * N2 + f2]);
    ok &= (std::fabs(peak - static_cast<float>(N)) < 1e-2f);

    // All other bins should be near zero
    for (int k1 = 0; k1 < N1 && ok; ++k1)
        for (int k2 = 0; k2 < N2 && ok; ++k2)
            if (k1 != f1 || k2 != f2)
                ok &= (std::abs(out[k1 * N2 + k2]) < 1e-3f);

    check("2D single-frequency identity (N=8x8, f1=2, f2=3)", ok);
}

int main() {
    std::cout << "=== Test: cuFFT 2D/3D coverage (QUEUE-26) ===\n";

    test_cufft_2d_c2c_roundtrip();
    test_cufft_2d_r2c_c2r_roundtrip();
    test_cufft_3d_c2c_roundtrip();
    test_cufft_2d_batch();
    test_cufft_2d_single_frequency();

    std::cout << "\n" << g_pass << "/" << g_total << " tests passed.\n";
    return (g_pass == g_total) ? 0 : 1;
}
