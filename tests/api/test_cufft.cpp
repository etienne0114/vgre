// cuFFT test — validates O(n log n) FFT for power-of-2, non-power-of-2,
// 2D, R2C/C2R, batched, and Z2Z (double) transforms.

#include "vgre/api/cufft_shim.h"

#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>

using cfloat  = std::complex<float>;
using cdouble = std::complex<double>;

static int g_pass = 0, g_total = 0;

static uint16_t f2h(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000u);
    int exp = static_cast<int>((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = bits & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10) return sign;
        mant = (mant | 0x800000u) >> (1 - exp);
        return static_cast<uint16_t>(sign | ((mant + 0x1000u) >> 13));
    }
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    return static_cast<uint16_t>(sign | (static_cast<uint16_t>(exp) << 10) |
                                 ((mant + 0x1000u) >> 13));
}

static float h2f(uint16_t h) {
    uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x03ffu;
    uint32_t bits = 0;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) { mant <<= 1; --exp; }
            mant &= 0x03ffu;
            bits = sign | ((exp + 112u) << 23) | (mant << 13);
        }
    } else if (exp == 0x1fu) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

static void check(const char *name, bool ok) {
    ++g_total;
    if (ok) { ++g_pass; std::cout << "PASS [" << g_total << "] " << name << "\n"; }
    else    { std::cerr << "FAIL [" << g_total << "] " << name << "\n"; }
}

// Compare two complex arrays with tolerance
template<typename T>
static bool approx_eq(const std::complex<T> *a, const std::complex<T> *b, int n, T tol) {
    for (int i = 0; i < n; ++i)
        if (std::abs(a[i] - b[i]) > tol) return false;
    return true;
}

template<typename T>
static bool approx_eq_real(const T *a, const T *b, int n, T tol) {
    for (int i = 0; i < n; ++i)
        if (std::fabs(a[i] - b[i]) > tol) return false;
    return true;
}

// ── Test: C2C forward+inverse roundtrip (power-of-2) ─────────────────────────
static void test_c2c_pow2_roundtrip() {
    const int N = 256;
    std::vector<cfloat> in(N), fwd(N), inv(N);
    for (int i = 0; i < N; ++i) in[i] = cfloat(std::sin(2.0f * 3.14159f * i / N), 0);

    cufftHandle plan;
    cufftPlan1d(&plan, N, CUFFT_C2C, 1);
    cufftExecC2C(plan, in.data(), fwd.data(), CUFFT_FORWARD);
    cufftExecC2C(plan, fwd.data(), inv.data(), CUFFT_INVERSE);
    cufftDestroy(plan);

    check("C2C pow2 roundtrip (N=256)", approx_eq(in.data(), inv.data(), N, 1e-4f));
}

// ── Test: C2C forward+inverse roundtrip (non-power-of-2, Bluestein) ──────────
static void test_c2c_nonpow2_roundtrip() {
    const int N = 100;
    std::vector<cfloat> in(N), fwd(N), inv(N);
    for (int i = 0; i < N; ++i) in[i] = cfloat(static_cast<float>(i) / N, 0);

    cufftHandle plan;
    cufftPlan1d(&plan, N, CUFFT_C2C, 1);
    cufftExecC2C(plan, in.data(), fwd.data(), CUFFT_FORWARD);
    cufftExecC2C(plan, fwd.data(), inv.data(), CUFFT_INVERSE);
    cufftDestroy(plan);

    check("C2C non-pow2 roundtrip (N=100)", approx_eq(in.data(), inv.data(), N, 1e-4f));
}

// ── Test: Known single-frequency DFT result ──────────────────────────────────
static void test_c2c_single_freq() {
    const int N = 64;
    std::vector<cfloat> in(N), out(N);
    // Pure cosine at frequency k=3 => DFT should have peaks at bin 3 and N-3
    for (int i = 0; i < N; ++i)
        in[i] = cfloat(std::cos(2.0f * 3.14159265f * 3.0f * i / N), 0);

    cufftHandle plan;
    cufftPlan1d(&plan, N, CUFFT_C2C, 1);
    cufftExecC2C(plan, in.data(), out.data(), CUFFT_FORWARD);
    cufftDestroy(plan);

    // Bin 3 and bin N-3 should have magnitude ~N/2 = 32
    float mag3    = std::abs(out[3]);
    float magN3   = std::abs(out[N - 3]);
    float magOther = 0;
    for (int i = 0; i < N; ++i)
        if (i != 3 && i != N - 3) magOther = std::max(magOther, std::abs(out[i]));

    bool ok = (mag3 > 30.0f && magN3 > 30.0f && magOther < 0.1f);
    check("C2C single-frequency (k=3, N=64)", ok);
}

// ── Test: Z2Z (double precision) roundtrip ───────────────────────────────────
static void test_z2z_roundtrip() {
    const int N = 128;
    std::vector<cdouble> in(N), fwd(N), inv(N);
    for (int i = 0; i < N; ++i) in[i] = cdouble(std::sin(2.0 * 3.14159265 * i / N), 0);

    cufftHandle plan;
    cufftPlan1d(&plan, N, CUFFT_Z2Z, 1);
    cufftExecZ2Z(plan, in.data(), fwd.data(), CUFFT_FORWARD);
    cufftExecZ2Z(plan, fwd.data(), inv.data(), CUFFT_INVERSE);
    cufftDestroy(plan);

    check("Z2Z roundtrip (N=128)", approx_eq(in.data(), inv.data(), N, 1e-10));
}

// ── Test: R2C + C2R roundtrip ────────────────────────────────────────────────
static void test_r2c_c2r_roundtrip() {
    const int N = 64;
    std::vector<float> in(N), recovered(N);
    std::vector<cfloat> freq(N); // only N/2+1 are meaningful, but allocate N
    for (int i = 0; i < N; ++i) in[i] = std::sin(2.0f * 3.14159f * i / N);

    cufftHandle plan_r2c, plan_c2r;
    cufftPlan1d(&plan_r2c, N, CUFFT_R2C, 1);
    cufftPlan1d(&plan_c2r, N, CUFFT_C2R, 1);
    cufftExecR2C(plan_r2c, in.data(), freq.data());
    cufftExecC2R(plan_c2r, freq.data(), recovered.data());
    cufftDestroy(plan_r2c);
    cufftDestroy(plan_c2r);

    check("R2C+C2R roundtrip (N=64)", approx_eq_real(in.data(), recovered.data(), N, 1e-4f));
}

// ── Test: D2Z + Z2D roundtrip (double) ───────────────────────────────────────
static void test_d2z_z2d_roundtrip() {
    const int N = 128;
    std::vector<double> in(N), recovered(N);
    std::vector<cdouble> freq(N);
    for (int i = 0; i < N; ++i) in[i] = std::cos(2.0 * 3.14159265 * 5.0 * i / N);

    cufftHandle plan_d2z, plan_z2d;
    cufftPlan1d(&plan_d2z, N, CUFFT_D2Z, 1);
    cufftPlan1d(&plan_z2d, N, CUFFT_Z2D, 1);
    cufftExecD2Z(plan_d2z, in.data(), freq.data());
    cufftExecZ2D(plan_z2d, freq.data(), recovered.data());
    cufftDestroy(plan_d2z);
    cufftDestroy(plan_z2d);

    check("D2Z+Z2D roundtrip (N=128)", approx_eq_real(in.data(), recovered.data(), N, 1e-10));
}

// ── Test: Batched C2C ─────────────────────────────────────────────────────────
static void test_batched_c2c() {
    const int N = 32, B = 4;
    std::vector<cfloat> in(N * B), fwd(N * B), inv(N * B);
    for (int b = 0; b < B; ++b)
        for (int i = 0; i < N; ++i)
            in[b * N + i] = cfloat(static_cast<float>(b + 1) * std::sin(2.0f * 3.14159f * i / N), 0);

    cufftHandle plan;
    cufftPlan1d(&plan, N, CUFFT_C2C, B);
    cufftExecC2C(plan, in.data(), fwd.data(), CUFFT_FORWARD);
    cufftExecC2C(plan, fwd.data(), inv.data(), CUFFT_INVERSE);
    cufftDestroy(plan);

    check("Batched C2C (N=32, B=4)", approx_eq(in.data(), inv.data(), N * B, 1e-4f));
}

// ── Test: 2D C2C roundtrip ───────────────────────────────────────────────────
static void test_2d_c2c_roundtrip() {
    const int NX = 16, NY = 16;
    std::vector<cfloat> in(NX * NY), fwd(NX * NY), inv(NX * NY);
    for (int y = 0; y < NY; ++y)
        for (int x = 0; x < NX; ++x)
            in[y * NX + x] = cfloat(std::sin(2.0f * 3.14159f * x / NX) +
                                     std::cos(2.0f * 3.14159f * y / NY), 0);

    cufftHandle plan;
    cufftPlan2d(&plan, NX, NY, CUFFT_C2C);
    cufftExecC2C(plan, in.data(), fwd.data(), CUFFT_FORWARD);
    cufftExecC2C(plan, fwd.data(), inv.data(), CUFFT_INVERSE);
    cufftDestroy(plan);

    check("2D C2C roundtrip (16x16)", approx_eq(in.data(), inv.data(), NX * NY, 1e-3f));
}

// ── Test: Non-pow2 prime-size FFT ────────────────────────────────────────────
static void test_c2c_prime_size() {
    const int N = 97; // prime
    std::vector<cfloat> in(N), fwd(N), inv(N);
    for (int i = 0; i < N; ++i) in[i] = cfloat(static_cast<float>(i % 7), static_cast<float>(i % 3));

    cufftHandle plan;
    cufftPlan1d(&plan, N, CUFFT_C2C, 1);
    cufftExecC2C(plan, in.data(), fwd.data(), CUFFT_FORWARD);
    cufftExecC2C(plan, fwd.data(), inv.data(), CUFFT_INVERSE);
    cufftDestroy(plan);

    check("C2C prime-size roundtrip (N=97)", approx_eq(in.data(), inv.data(), N, 1e-3f));
}

// ── Test: Parseval's theorem (energy conservation) ───────────────────────────
static void test_parseval() {
    const int N = 512;
    std::vector<cfloat> in(N), out(N);
    for (int i = 0; i < N; ++i) in[i] = cfloat(std::sin(0.1f * i), std::cos(0.3f * i));

    cufftHandle plan;
    cufftPlan1d(&plan, N, CUFFT_C2C, 1);
    cufftExecC2C(plan, in.data(), out.data(), CUFFT_FORWARD);
    cufftDestroy(plan);

    double energy_time = 0, energy_freq = 0;
    for (int i = 0; i < N; ++i) {
        energy_time += static_cast<double>(std::norm(in[i]));
        energy_freq += static_cast<double>(std::norm(out[i]));
    }
    energy_freq /= N; // Parseval: sum|X[k]|^2 / N = sum|x[n]|^2

    bool ok = (std::fabs(energy_time - energy_freq) / energy_time < 1e-4);
    check("Parseval's theorem (N=512)", ok);
}

// ── Test: Plan management (create, destroy, invalid) ─────────────────────────
static void test_plan_management() {
    cufftHandle plan;
    bool ok = true;
    ok &= (cufftPlan1d(&plan, 64, CUFFT_C2C, 1) == CUFFT_SUCCESS);
    ok &= (cufftDestroy(plan) == CUFFT_SUCCESS);
    ok &= (cufftPlan1d(nullptr, 64, CUFFT_C2C, 1) != CUFFT_SUCCESS);
    ok &= (cufftPlan1d(&plan, -1, CUFFT_C2C, 1) != CUFFT_SUCCESS);
    check("Plan management (create/destroy/reject invalid)", ok);
}

// ── Test: FP16 complex roundtrip ─────────────────────────────────────────────
static void test_fp16_c2c_roundtrip() {
    const int N = 32;
    std::vector<cufftHalfComplex> in(N), fwd(N), inv(N);
    for (int i = 0; i < N; ++i) {
        in[i].x = f2h(std::sin(2.0f * 3.14159f * i / N));
        in[i].y = f2h(0.25f * std::cos(2.0f * 3.14159f * i / N));
    }

    cufftHandle plan;
    bool ok = (cufftPlan1d(&plan, N, CUFFT_C16C, 1) == CUFFT_SUCCESS);
    ok &= (cufftExecC16C(plan, in.data(), fwd.data(), CUFFT_FORWARD) == CUFFT_SUCCESS);
    ok &= (cufftExecC16C(plan, fwd.data(), inv.data(), CUFFT_INVERSE) == CUFFT_SUCCESS);
    cufftDestroy(plan);

    for (int i = 0; i < N && ok; ++i) {
        ok &= std::fabs(h2f(in[i].x) - h2f(inv[i].x)) < 2e-3f;
        ok &= std::fabs(h2f(in[i].y) - h2f(inv[i].y)) < 2e-3f;
    }
    check("FP16 C16C roundtrip (N=32)", ok);
}

// ── Test: PlanMany with strided batched 1D layout ───────────────────────────
static void test_plan_many_strided_batch() {
    const int N = 16, B = 2, stride = 2, dist = N * stride;
    std::vector<cfloat> in(dist * B), fwd(dist * B), inv(dist * B);
    for (int b = 0; b < B; ++b)
        for (int i = 0; i < N; ++i)
            in[b * dist + i * stride] = cfloat(static_cast<float>(b + 1) * std::sin(0.2f * i), 0.1f * i);

    int dims[1] = {N};
    cufftHandle plan;
    size_t work = 0;
    bool ok = (cufftPlanMany(&plan, 1, dims, nullptr, stride, dist, nullptr,
                             stride, dist, CUFFT_C2C, B) == CUFFT_SUCCESS);
    ok &= (cufftGetSizeMany(plan, 1, dims, nullptr, stride, dist, nullptr,
                            stride, dist, CUFFT_C2C, B, &work) == CUFFT_SUCCESS);
    ok &= (cufftExecC2C(plan, in.data(), fwd.data(), CUFFT_FORWARD) == CUFFT_SUCCESS);
    ok &= (cufftExecC2C(plan, fwd.data(), inv.data(), CUFFT_INVERSE) == CUFFT_SUCCESS);
    cufftDestroy(plan);

    for (int b = 0; b < B && ok; ++b)
        for (int i = 0; i < N && ok; ++i)
            ok &= std::abs(in[b * dist + i * stride] - inv[b * dist + i * stride]) < 1e-4f;
    check("PlanMany strided batched C2C", ok);
}

// ── Test: cufftGetVersion ─────────────────────────────────────────────────────
static void test_get_version() {
    int version = 0;
    bool ok = (cufftGetVersion(&version) == CUFFT_SUCCESS);
    ok &= (version > 0);
    ok &= (cufftGetVersion(nullptr) != CUFFT_SUCCESS);
    check("cufftGetVersion returns positive version", ok);
}

// ── Test: cufftSetStream / cufftGetStream ─────────────────────────────────────
static void test_stream_association() {
    cufftHandle plan;
    cufftPlan1d(&plan, 64, CUFFT_C2C, 1);

    // Sentinel pointer as stream (non-null)
    void *fake_stream = reinterpret_cast<void*>(static_cast<uintptr_t>(0xDEADBEEF));
    bool ok = (cufftSetStream(plan, fake_stream) == CUFFT_SUCCESS);

    void *out_stream = nullptr;
    ok &= (cufftGetStream(plan, &out_stream) == CUFFT_SUCCESS);
    ok &= (out_stream == fake_stream);

    // Null pointer sentinel clears stream
    ok &= (cufftSetStream(plan, nullptr) == CUFFT_SUCCESS);
    ok &= (cufftGetStream(plan, &out_stream) == CUFFT_SUCCESS);
    ok &= (out_stream == nullptr);

    // Invalid plan
    ok &= (cufftSetStream(9999999, fake_stream) == CUFFT_INVALID_PLAN);
    ok &= (cufftGetStream(9999999, &out_stream) == CUFFT_INVALID_PLAN);
    ok &= (cufftGetStream(plan, nullptr) == CUFFT_INVALID_VALUE);

    cufftDestroy(plan);
    check("cufftSetStream / cufftGetStream round-trip", ok);
}

// ── Test: stream does not break execution ────────────────────────────────────
static void test_exec_with_stream() {
    const int N = 32;
    std::vector<cfloat> in(N), fwd(N), inv(N);
    for (int i = 0; i < N; ++i) in[i] = cfloat(std::sin(0.2f * i), 0.0f);

    cufftHandle plan;
    cufftPlan1d(&plan, N, CUFFT_C2C, 1);
    void *fake_stream = reinterpret_cast<void*>(static_cast<uintptr_t>(42));
    cufftSetStream(plan, fake_stream);

    bool ok = (cufftExecC2C(plan, in.data(), fwd.data(), CUFFT_FORWARD) == CUFFT_SUCCESS);
    ok &= (cufftExecC2C(plan, fwd.data(), inv.data(), CUFFT_INVERSE) == CUFFT_SUCCESS);
    ok &= approx_eq(in.data(), inv.data(), N, 1e-4f);
    cufftDestroy(plan);
    check("Exec with associated stream gives same result (N=32)", ok);
}

int main() {
    std::cout << "=== Test: cuFFT O(n log n) FFT ===\n";

    test_c2c_pow2_roundtrip();
    test_c2c_nonpow2_roundtrip();
    test_c2c_single_freq();
    test_z2z_roundtrip();
    test_r2c_c2r_roundtrip();
    test_d2z_z2d_roundtrip();
    test_batched_c2c();
    test_2d_c2c_roundtrip();
    test_c2c_prime_size();
    test_parseval();
    test_plan_management();
    test_fp16_c2c_roundtrip();
    test_plan_many_strided_batch();
    test_get_version();
    test_stream_association();
    test_exec_with_stream();

    std::cout << "\n" << g_pass << "/" << g_total << " tests passed.\n";
    return (g_pass == g_total) ? 0 : 1;
}
