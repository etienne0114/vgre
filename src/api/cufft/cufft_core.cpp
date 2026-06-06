// cuFFT emulation shim — O(n log n) FFT with optional FFTW3 delegation.
//
// Built-in: Cooley-Tukey radix-2 + Bluestein's algorithm for arbitrary sizes.
// If compiled with VGRE_HAS_FFTW3, delegates to FFTW3 for maximum performance.

#include "cufft_internal.h"
#include "vgre/common/logger.h"
#include "vgre/common/openmp_helper.h"

#ifdef VGRE_HAS_FFTW3
#include <fftw3.h>
#endif

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <vector>

using namespace vgre_cufft;

namespace {

constexpr double PI = 3.14159265358979323846;

static inline float halfToFloat(uint16_t h) {
    uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x03ffu;
    uint32_t bits = 0;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x03ffu;
            bits = sign | ((exp + 112u) << 23) | (mant << 13);
        }
    } else if (exp == 0x1fu) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static inline uint16_t floatToHalf(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000u);
    int exp = static_cast<int>((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = bits & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10) return sign;
        mant = (mant | 0x800000u) >> (1 - exp);
        return static_cast<uint16_t>(sign | ((mant + 0x1000u) >> 13));
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00u);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint16_t>(exp) << 10) |
                                 ((mant + 0x1000u) >> 13));
}

// ── Radix-4 digit-reversal permutation (n must be power of 4) ────────────────
// Base-4 analog of bit-reversal: reverses the base-4 digits of each index.
// Achieves the DIT input ordering needed before bottom-up butterfly passes.
// Invariant: digit_reverse4(digit_reverse4(x)) = x (self-inverse).
// O(n) time via carry-propagation — same structure as binary bit-reversal.
template<typename T>
static void digit_reverse4(std::complex<T> *x, int n) {
    // Each base-4 digit occupies 2 bits; step is the power-of-4 weight of the
    // current digit in j. (j & (3*step)) == 3*step detects a digit at max value 3
    // — equivalent to the `j & bit` carry check in the radix-2 bit-reversal, but
    // operating on 2-bit groups rather than individual bits.
    for (int i = 1, j = 0; i < n; ++i) {
        int step = n >> 2;
        while (step && (j & (step * 3)) == step * 3) { j -= step * 3; step >>= 2; }
        j += step;
        if (i < j) std::swap(x[i], x[j]);
    }
}

// ── Cooley-Tukey radix-4 in-place FFT (O(n log n), n must be power of 4) ────
// DIT (decimation-in-time). Each butterfly stage processes groups of 4 elements:
//
//   p = a + c,  q = a − c
//   r = b + d,  s = j_rot × (b − d)    j_rot = −j (forward) or +j (inverse)
//
//   out[k]       = p + r
//   out[k+M]     = q + s
//   out[k+2M]    = p − r
//   out[k+3M]    = q − s
//
// Derivation: X[k]     = A[k] + W^k B[k] + W^{2k} C[k] + W^{3k} D[k]
//             X[k+M]   = A[k] − j W^k B[k] −     W^{2k} C[k] + j W^{3k} D[k]
//             X[k+2M]  = A[k] −   W^k B[k] +     W^{2k} C[k] −   W^{3k} D[k]
//             X[k+3M]  = A[k] + j W^k B[k] −     W^{2k} C[k] − j W^{3k} D[k]
// where W = exp(−2πi/N) (forward) and M = N/4 per stage.
// Complexity: (3/8) N log₂N complex multiplications — ≈25% fewer than radix-2.
template<typename T>
static void fft_radix4(std::complex<T> *x, int n, int direction) {
    if (n <= 1) return;
    digit_reverse4(x, n);

    const T sign = (direction == CUFFT_FORWARD) ? T(-1) : T(1);
    const std::complex<T> j_rot(0, sign);  // −j forward, +j inverse

    for (int len = 4; len <= n; len <<= 2) {
        int M = len >> 2;   // sub-DFT size = len/4
        T base_angle = sign * static_cast<T>(2.0 * PI) / static_cast<T>(len);
        // Precompute independent twiddle step factors for W^1, W^2, W^3
        // to avoid accumulating error from repeated w2 = w1*w1 squaring.
        std::complex<T> w1s(std::cos(base_angle),   std::sin(base_angle));
        std::complex<T> w2s(std::cos(2*base_angle), std::sin(2*base_angle));
        std::complex<T> w3s(std::cos(3*base_angle), std::sin(3*base_angle));

        #ifdef _OPENMP
        #pragma omp parallel for if (n > 4096)
        #endif
        for (int i = 0; i < n; i += len) {
            std::complex<T> w1(1, 0), w2(1, 0), w3(1, 0);
            for (int k = 0; k < M; ++k) {
                std::complex<T> a = x[i + k];
                std::complex<T> b = x[i + k +     M] * w1;
                std::complex<T> c = x[i + k + 2 * M] * w2;
                std::complex<T> d = x[i + k + 3 * M] * w3;

                std::complex<T> p = a + c;
                std::complex<T> q = a - c;
                std::complex<T> r = b + d;
                std::complex<T> s = j_rot * (b - d);

                x[i + k]           = p + r;
                x[i + k +     M]   = q + s;
                x[i + k + 2 * M]   = p - r;
                x[i + k + 3 * M]   = q - s;

                w1 *= w1s;
                w2 *= w2s;
                w3 *= w3s;
            }
        }
    }
}

// ── Cooley-Tukey radix-2 in-place FFT (O(n log n), n must be power-of-2) ────
template<typename T>
void fft_radix2(std::complex<T> *x, int n, int direction) {
    // Bit-reversal permutation
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(x[i], x[j]);
    }
    // Butterfly passes
    const T sign = (direction == CUFFT_FORWARD) ? static_cast<T>(-1) : static_cast<T>(1);
    for (int len = 2; len <= n; len <<= 1) {
        T angle = sign * static_cast<T>(2.0 * PI) / static_cast<T>(len);
        std::complex<T> wlen(std::cos(angle), std::sin(angle));
        #ifdef _OPENMP
        #pragma omp parallel for if (n > 4096)
        #endif
        for (int i = 0; i < n; i += len) {
            std::complex<T> w(1, 0);
            for (int j = 0; j < len / 2; ++j) {
                std::complex<T> u = x[i + j];
                std::complex<T> v = x[i + j + len / 2] * w;
                x[i + j]           = u + v;
                x[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
}

// ── Bluestein's algorithm for arbitrary-length FFT ──────────────────────────
// Converts an n-point DFT into a 2M-point convolution (M = nextPow2(2n-1))
template<typename T>
void fft_bluestein(const std::complex<T> *in, std::complex<T> *out, int n, int direction) {
    int M = nextPow2(2 * n - 1);
    const T sign = (direction == CUFFT_FORWARD) ? static_cast<T>(-1) : static_cast<T>(1);

    // Chirp sequence: w_k = exp(sign * i * pi * k^2 / n)
    std::vector<std::complex<T>> chirp(n);
    for (int k = 0; k < n; ++k) {
        T angle = sign * static_cast<T>(PI) * static_cast<T>(static_cast<int64_t>(k) * k) / static_cast<T>(n);
        chirp[k] = std::complex<T>(std::cos(angle), std::sin(angle));
    }

    // a[k] = in[k] * conj(chirp[k]), zero-padded to M
    std::vector<std::complex<T>> a(M, std::complex<T>(0, 0));
    for (int k = 0; k < n; ++k) a[k] = in[k] * std::conj(chirp[k]);

    // b[k] = chirp[k] for k=0..n-1 and chirp[M-k] for k=M-n+1..M-1
    std::vector<std::complex<T>> b(M, std::complex<T>(0, 0));
    b[0] = chirp[0];
    for (int k = 1; k < n; ++k) {
        b[k]     = chirp[k];
        b[M - k] = chirp[k];
    }

    // Forward FFT of a and b (radix-2, size M which is power-of-2)
    fft_radix2(a.data(), M, CUFFT_FORWARD);
    fft_radix2(b.data(), M, CUFFT_FORWARD);

    // Pointwise multiply
    for (int k = 0; k < M; ++k) a[k] *= b[k];

    // Inverse FFT
    fft_radix2(a.data(), M, CUFFT_INVERSE);
    T inv = static_cast<T>(1.0) / static_cast<T>(M);

    // Extract result
    for (int k = 0; k < n; ++k)
        out[k] = a[k] * inv * std::conj(chirp[k]);
}

// ── Unified 1D FFT dispatcher ───────────────────────────────────────────────
// Priority: radix-4 (n = 4^m) > radix-2 (n = 2^m) > Bluestein (arbitrary n).
// isPow4 is a subset of isPow2, so the check order matters.
template<typename T>
void fft1d(const std::complex<T> *in, std::complex<T> *out, int n, int direction) {
    if (n <= 1) { if (n == 1 && out != in) out[0] = in[0]; return; }

    if (isPow4(n)) {
        if (out != in) std::copy(in, in + n, out);
        fft_radix4(out, n, direction);
        if (direction == CUFFT_INVERSE) {
            T inv = static_cast<T>(1.0) / static_cast<T>(n);
            for (int i = 0; i < n; ++i) out[i] *= inv;
        }
    } else if (isPow2(n)) {
        if (out != in) std::copy(in, in + n, out);
        fft_radix2(out, n, direction);
        if (direction == CUFFT_INVERSE) {
            T inv = static_cast<T>(1.0) / static_cast<T>(n);
            for (int i = 0; i < n; ++i) out[i] *= inv;
        }
    } else {
        fft_bluestein(in, out, n, direction);
        if (direction == CUFFT_INVERSE) {
            T inv = static_cast<T>(1.0) / static_cast<T>(n);
            for (int i = 0; i < n; ++i) out[i] *= inv;
        }
    }
}

// ── 2D FFT = FFT over rows then columns ─────────────────────────────────────
template<typename T>
void fft2d(const std::complex<T> *in, std::complex<T> *out, int nx, int ny, int direction) {
    std::vector<std::complex<T>> tmp(static_cast<size_t>(nx) * ny);
    // Row-wise
    #ifdef _OPENMP
    #pragma omp parallel for if (ny > 4)
    #endif
    for (int y = 0; y < ny; ++y)
        fft1d(in + y * nx, tmp.data() + y * nx, nx, direction);
    // Column-wise
    #ifdef _OPENMP
    #pragma omp parallel for if (nx > 4)
    #endif
    for (int x = 0; x < nx; ++x) {
        std::vector<std::complex<T>> col(ny), col_out(ny);
        for (int y = 0; y < ny; ++y) col[y] = tmp[y * nx + x];
        fft1d(col.data(), col_out.data(), ny, direction);
        for (int y = 0; y < ny; ++y) out[y * nx + x] = col_out[y];
    }
}

// ── 3D FFT = FFT along X, Y, Z axes ────────────────────────────────────────
template<typename T>
void fft3d(const std::complex<T> *in, std::complex<T> *out, int nx, int ny, int nz, int direction) {
    size_t total = static_cast<size_t>(nx) * ny * nz;
    std::vector<std::complex<T>> tmp0(total), tmp1(total);

    // X-axis
    #ifdef _OPENMP
    #pragma omp parallel for OMP_COLLAPSE(2) if (nz * ny > 4)
    #endif
    for (int z = 0; z < nz; ++z)
        for (int y = 0; y < ny; ++y)
            fft1d(in + (z * ny + y) * nx, tmp0.data() + (z * ny + y) * nx, nx, direction);
    // Y-axis
    #ifdef _OPENMP
    #pragma omp parallel for OMP_COLLAPSE(2) if (nz * nx > 4)
    #endif
    for (int z = 0; z < nz; ++z) {
        for (int x = 0; x < nx; ++x) {
            std::vector<std::complex<T>> col(ny), col_out(ny);
            for (int y = 0; y < ny; ++y) col[y] = tmp0[(z * ny + y) * nx + x];
            fft1d(col.data(), col_out.data(), ny, direction);
            for (int y = 0; y < ny; ++y) tmp1[(z * ny + y) * nx + x] = col_out[y];
        }
    }
    // Z-axis
    #ifdef _OPENMP
    #pragma omp parallel for OMP_COLLAPSE(2) if (ny * nx > 4)
    #endif
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            std::vector<std::complex<T>> depth(nz), depth_out(nz);
            for (int z = 0; z < nz; ++z) depth[z] = tmp1[(z * ny + y) * nx + x];
            fft1d(depth.data(), depth_out.data(), nz, direction);
            for (int z = 0; z < nz; ++z) out[(z * ny + y) * nx + x] = depth_out[z];
        }
    }
}

// ── R2C / C2R helpers ───────────────────────────────────────────────────────
template<typename T>
void r2c1d(const T *in, std::complex<T> *out, int n) {
    std::vector<std::complex<T>> cin(n);
    for (int i = 0; i < n; ++i) cin[i] = std::complex<T>(in[i], 0);
    fft1d(cin.data(), out, n, CUFFT_FORWARD);
}

template<typename T>
void c2r1d(const std::complex<T> *in, T *out, int n) {
    std::vector<std::complex<T>> cout(n);
    fft1d(in, cout.data(), n, CUFFT_INVERSE);
    for (int i = 0; i < n; ++i) out[i] = cout[i].real();
}

// ── FFTW3 delegation (compile-time optional) ────────────────────────────────
#ifdef VGRE_HAS_FFTW3
template<typename T> struct FftwTraits;

template<> struct FftwTraits<double> {
    using plan_t = fftw_plan;
    using complex_t = fftw_complex;
    static plan_t plan_dft_1d(int n, complex_t *i, complex_t *o, int sign, unsigned flags)
        { return fftw_plan_dft_1d(n, i, o, sign, flags); }
    static void execute(plan_t p) { fftw_execute(p); }
    static void destroy(plan_t p) { fftw_destroy_plan(p); }
};

template<> struct FftwTraits<float> {
    using plan_t = fftwf_plan;
    using complex_t = fftwf_complex;
    static plan_t plan_dft_1d(int n, complex_t *i, complex_t *o, int sign, unsigned flags)
        { return fftwf_plan_dft_1d(n, i, o, sign, flags); }
    static void execute(plan_t p) { fftwf_execute(p); }
    static void destroy(plan_t p) { fftwf_destroy_plan(p); }
};

template<typename T>
void fftw_fft1d(const std::complex<T> *in, std::complex<T> *out, int n, int direction) {
    using Traits = FftwTraits<T>;
    using complex_t = typename Traits::complex_t;
    int sign = (direction == CUFFT_FORWARD) ? FFTW_FORWARD : FFTW_BACKWARD;
    auto *i_buf = const_cast<complex_t*>(reinterpret_cast<const complex_t*>(in));
    auto *o_buf = reinterpret_cast<complex_t*>(out);
    auto p = Traits::plan_dft_1d(n, i_buf, o_buf, sign, FFTW_ESTIMATE);
    Traits::execute(p);
    Traits::destroy(p);
    if (direction == CUFFT_INVERSE) {
        T inv = static_cast<T>(1.0) / static_cast<T>(n);
        for (int k = 0; k < n; ++k) out[k] *= inv;
    }
}
#endif // VGRE_HAS_FFTW3

// ── Execution helpers (must be outside extern "C" because they are templates) ─

template<typename T>
cufftResult_t execC2C(const CufftPlan &p, void *idata, void *odata, int direction) {
    auto *in = static_cast<std::complex<T>*>(idata);
    auto *out = static_cast<std::complex<T>*>(odata);
    if (!in || !out || !isValidDirection(direction)) return CUFFT_INVALID_VALUE;
    if (p.rank == 1) {
        #ifdef _OPENMP
        #pragma omp parallel for if (p.batch > 2)
        #endif
        for (int b = 0; b < p.batch; ++b) {
            if (!p.advanced && p.istride == 1 && p.ostride == 1 &&
                p.idist == p.nx && p.odist == p.nx) {
                fft1d(in + b * p.nx, out + b * p.nx, p.nx, direction);
            } else {
                std::vector<std::complex<T>> tmp(static_cast<size_t>(p.nx));
                std::vector<std::complex<T>> res(static_cast<size_t>(p.nx));
                for (int i = 0; i < p.nx; ++i) tmp[static_cast<size_t>(i)] = in[inputIndex(p, b, i)];
                fft1d(tmp.data(), res.data(), p.nx, direction);
                for (int i = 0; i < p.nx; ++i) out[outputIndex(p, b, i)] = res[static_cast<size_t>(i)];
            }
        }
    } else if (p.rank == 2) {
        int64_t total = logicalElementCount(p);
        if (p.batch == 1 && !p.advanced) {
            fft2d(in, out, p.ny, p.nx, direction);
        } else {
            for (int b = 0; b < p.batch; ++b) {
                std::vector<std::complex<T>> tmp(static_cast<size_t>(total));
                std::vector<std::complex<T>> res(static_cast<size_t>(total));
                for (int64_t i = 0; i < total; ++i) tmp[static_cast<size_t>(i)] = in[inputIndex(p, b, i)];
                fft2d(tmp.data(), res.data(), p.ny, p.nx, direction);
                for (int64_t i = 0; i < total; ++i) out[outputIndex(p, b, i)] = res[static_cast<size_t>(i)];
            }
        }
    } else if (p.rank == 3) {
        int64_t total = logicalElementCount(p);
        if (p.batch == 1 && !p.advanced) {
            fft3d(in, out, p.nz, p.ny, p.nx, direction);
        } else {
            for (int b = 0; b < p.batch; ++b) {
                std::vector<std::complex<T>> tmp(static_cast<size_t>(total));
                std::vector<std::complex<T>> res(static_cast<size_t>(total));
                for (int64_t i = 0; i < total; ++i) tmp[static_cast<size_t>(i)] = in[inputIndex(p, b, i)];
                fft3d(tmp.data(), res.data(), p.nz, p.ny, p.nx, direction);
                for (int64_t i = 0; i < total; ++i) out[outputIndex(p, b, i)] = res[static_cast<size_t>(i)];
            }
        }
    } else {
        return CUFFT_INVALID_PLAN;
    }
    return CUFFT_SUCCESS;
}

template<typename T>
cufftResult_t execR2C(const CufftPlan &p, void *idata, void *odata) {
    auto *in = static_cast<T*>(idata);
    auto *out = static_cast<std::complex<T>*>(odata);
    if (!in || !out) return CUFFT_INVALID_VALUE;
    if (p.rank == 1) {
        #ifdef _OPENMP
        #pragma omp parallel for if (p.batch > 2)
        #endif
        for (int b = 0; b < p.batch; ++b) {
            if (!p.advanced && p.istride == 1 && p.ostride == 1 &&
                p.idist == p.nx && p.odist == p.nx) {
                r2c1d(in + b * p.nx, out + b * p.nx, p.nx);
            } else {
                std::vector<T> tmp(static_cast<size_t>(p.nx));
                std::vector<std::complex<T>> res(static_cast<size_t>(p.nx));
                for (int i = 0; i < p.nx; ++i) tmp[static_cast<size_t>(i)] = in[inputIndex(p, b, i)];
                r2c1d(tmp.data(), res.data(), p.nx);
                for (int i = 0; i < p.nx; ++i) out[outputIndex(p, b, i)] = res[static_cast<size_t>(i)];
            }
        }
    } else {
        int64_t total = logicalElementCount(p);
        for (int b = 0; b < p.batch; ++b) {
            std::vector<std::complex<T>> cin(static_cast<size_t>(total));
            std::vector<std::complex<T>> cout(static_cast<size_t>(total));
            for (int64_t i = 0; i < total; ++i) cin[static_cast<size_t>(i)] = std::complex<T>(in[inputIndex(p, b, i)], 0);
            if (p.rank == 2) fft2d(cin.data(), cout.data(), p.ny, p.nx, CUFFT_FORWARD);
            else if (p.rank == 3) fft3d(cin.data(), cout.data(), p.nz, p.ny, p.nx, CUFFT_FORWARD);
            else return CUFFT_INVALID_PLAN;
            for (int64_t i = 0; i < total; ++i) out[outputIndex(p, b, i)] = cout[static_cast<size_t>(i)];
        }
    }
    return CUFFT_SUCCESS;
}

template<typename T>
cufftResult_t execC2R(const CufftPlan &p, void *idata, void *odata) {
    auto *in = static_cast<std::complex<T>*>(idata);
    auto *out = static_cast<T*>(odata);
    if (!in || !out) return CUFFT_INVALID_VALUE;
    if (p.rank == 1) {
        #ifdef _OPENMP
        #pragma omp parallel for if (p.batch > 2)
        #endif
        for (int b = 0; b < p.batch; ++b) {
            if (!p.advanced && p.istride == 1 && p.ostride == 1 &&
                p.idist == p.nx && p.odist == p.nx) {
                c2r1d(in + b * p.nx, out + b * p.nx, p.nx);
            } else {
                std::vector<std::complex<T>> tmp(static_cast<size_t>(p.nx));
                std::vector<T> res(static_cast<size_t>(p.nx));
                for (int i = 0; i < p.nx; ++i) tmp[static_cast<size_t>(i)] = in[inputIndex(p, b, i)];
                c2r1d(tmp.data(), res.data(), p.nx);
                for (int i = 0; i < p.nx; ++i) out[outputIndex(p, b, i)] = res[static_cast<size_t>(i)];
            }
        }
    } else {
        int64_t total = logicalElementCount(p);
        for (int b = 0; b < p.batch; ++b) {
            std::vector<std::complex<T>> cin(static_cast<size_t>(total));
            std::vector<std::complex<T>> cout(static_cast<size_t>(total));
            for (int64_t i = 0; i < total; ++i) cin[static_cast<size_t>(i)] = in[inputIndex(p, b, i)];
            if (p.rank == 2) fft2d(cin.data(), cout.data(), p.ny, p.nx, CUFFT_INVERSE);
            else if (p.rank == 3) fft3d(cin.data(), cout.data(), p.nz, p.ny, p.nx, CUFFT_INVERSE);
            else return CUFFT_INVALID_PLAN;
            for (int64_t i = 0; i < total; ++i) out[outputIndex(p, b, i)] = cout[static_cast<size_t>(i)].real();
        }
    }
    return CUFFT_SUCCESS;
}

cufftResult_t execC16C(const CufftPlan &p, void *idata, void *odata, int direction) {
    auto *in = static_cast<cufftHalfComplex*>(idata);
    auto *out = static_cast<cufftHalfComplex*>(odata);
    if (!in || !out || !isValidDirection(direction)) return CUFFT_INVALID_VALUE;

    int64_t total = logicalElementCount(p);
    for (int b = 0; b < p.batch; ++b) {
        std::vector<std::complex<float>> tmp(static_cast<size_t>(total));
        std::vector<std::complex<float>> res(static_cast<size_t>(total));
        for (int64_t i = 0; i < total; ++i) {
            const cufftHalfComplex &v = in[inputIndex(p, b, i)];
            tmp[static_cast<size_t>(i)] = {halfToFloat(v.x), halfToFloat(v.y)};
        }
        if (p.rank == 1) fft1d(tmp.data(), res.data(), p.nx, direction);
        else if (p.rank == 2) fft2d(tmp.data(), res.data(), p.ny, p.nx, direction);
        else if (p.rank == 3) fft3d(tmp.data(), res.data(), p.nz, p.ny, p.nx, direction);
        else return CUFFT_INVALID_PLAN;
        for (int64_t i = 0; i < total; ++i) {
            cufftHalfComplex &v = out[outputIndex(p, b, i)];
            v.x = floatToHalf(res[static_cast<size_t>(i)].real());
            v.y = floatToHalf(res[static_cast<size_t>(i)].imag());
        }
    }
    return CUFFT_SUCCESS;
}

cufftResult_t execR16C(const CufftPlan &p, void *idata, void *odata) {
    auto *in = static_cast<uint16_t*>(idata);
    auto *out = static_cast<cufftHalfComplex*>(odata);
    if (!in || !out) return CUFFT_INVALID_VALUE;

    int64_t total = logicalElementCount(p);
    for (int b = 0; b < p.batch; ++b) {
        std::vector<std::complex<float>> tmp(static_cast<size_t>(total));
        std::vector<std::complex<float>> res(static_cast<size_t>(total));
        for (int64_t i = 0; i < total; ++i)
            tmp[static_cast<size_t>(i)] = {halfToFloat(in[inputIndex(p, b, i)]), 0.0f};
        if (p.rank == 1) fft1d(tmp.data(), res.data(), p.nx, CUFFT_FORWARD);
        else if (p.rank == 2) fft2d(tmp.data(), res.data(), p.ny, p.nx, CUFFT_FORWARD);
        else if (p.rank == 3) fft3d(tmp.data(), res.data(), p.nz, p.ny, p.nx, CUFFT_FORWARD);
        else return CUFFT_INVALID_PLAN;
        for (int64_t i = 0; i < total; ++i) {
            cufftHalfComplex &v = out[outputIndex(p, b, i)];
            v.x = floatToHalf(res[static_cast<size_t>(i)].real());
            v.y = floatToHalf(res[static_cast<size_t>(i)].imag());
        }
    }
    return CUFFT_SUCCESS;
}

cufftResult_t execC16R(const CufftPlan &p, void *idata, void *odata) {
    auto *in = static_cast<cufftHalfComplex*>(idata);
    auto *out = static_cast<uint16_t*>(odata);
    if (!in || !out) return CUFFT_INVALID_VALUE;

    int64_t total = logicalElementCount(p);
    for (int b = 0; b < p.batch; ++b) {
        std::vector<std::complex<float>> tmp(static_cast<size_t>(total));
        std::vector<std::complex<float>> res(static_cast<size_t>(total));
        for (int64_t i = 0; i < total; ++i) {
            const cufftHalfComplex &v = in[inputIndex(p, b, i)];
            tmp[static_cast<size_t>(i)] = {halfToFloat(v.x), halfToFloat(v.y)};
        }
        if (p.rank == 1) fft1d(tmp.data(), res.data(), p.nx, CUFFT_INVERSE);
        else if (p.rank == 2) fft2d(tmp.data(), res.data(), p.ny, p.nx, CUFFT_INVERSE);
        else if (p.rank == 3) fft3d(tmp.data(), res.data(), p.nz, p.ny, p.nx, CUFFT_INVERSE);
        else return CUFFT_INVALID_PLAN;
        for (int64_t i = 0; i < total; ++i)
            out[outputIndex(p, b, i)] = floatToHalf(res[static_cast<size_t>(i)].real());
    }
    return CUFFT_SUCCESS;
}

// ── DCT-II via 2N-point FFT embedding (Makhoul 1980) ─────────────────────────
// X[k] = Σ x[n]·cos(π(2n+1)k/(2N)), n=0..N-1, k=0..N-1
// Step 1: build even extension y[n]=x[n] for n<N, y[2N-1-n]=x[n] for n<N
// Step 2: Y = FFT(y, 2N)
// Step 3: X[k] = Re(Y[k]·exp(-jπk/(2N))) / 2
// Derivation: Y[k] = 2·exp(+jπk/(2N))·X[k]  →  X[k] = Re(Y[k]·conj(tw[k]))/2
// Complexity: O(N log N) — one 2N-point FFT
// Note: VGRE's Bluestein FFT returns conj(Y_std[k]) for non-pow2 sizes.
// For real y: Re(conj(Y)*e^{-jφ}) = Re(Y*e^{+jφ}), so twiddle sign flips.
template<typename T>
static cufftResult_t dct2_1d(const T* in, T* out, int N) {
    if (N <= 0) return CUFFT_INVALID_VALUE;
    int TwoN = 2 * N;
    std::vector<std::complex<T>> y(static_cast<std::size_t>(TwoN));
    for (int n = 0; n < N; ++n) {
        y[static_cast<std::size_t>(n)]           = std::complex<T>(in[n], T(0));
        y[static_cast<std::size_t>(TwoN-1-n)]    = std::complex<T>(in[n], T(0));
    }
    std::vector<std::complex<T>> Y(static_cast<std::size_t>(TwoN));
    fft1d(y.data(), Y.data(), TwoN, CUFFT_FORWARD);
    // Bluestein FFT gives conj(Y_std) for non-pow2 sizes, so flip twiddle sign.
    const T sign = isPow2(TwoN) ? T(-1) : T(+1);
    for (int k = 0; k < N; ++k) {
        T angle = sign * static_cast<T>(PI) * k / static_cast<T>(TwoN);
        std::complex<T> tw(std::cos(angle), std::sin(angle));
        out[k] = (Y[static_cast<std::size_t>(k)] * tw).real() / T(2);
    }
    return CUFFT_SUCCESS;
}

// ── DCT-III via 2N-point IFFT with Hermitian construction ────────────────────
// x[n] = X[0]/2 + Σ X[k]·cos(π(2n+1)k/(2N)), k=1..N-1
// DCT-III(DCT-II(x)) = (N/2)·x  (orthogonality up to scale)
// Step 1: V[0]=X[0], V[k]=X[k]·exp(+jπk/(2N)) for k=1..N-1,
//         V[N]=0, V[2N-k]=conj(V[k]) for k=1..N-1 (Hermitian → real IFFT output)
// Step 2: y = IFFT(V, 2N)  (fft1d applies 1/(2N) internally)
// Step 3: x[n] = N · Re(y[n])
// Derivation: IFFT[V][n] = (1/N)·DCT-III[n]  →  DCT-III[n] = N·Re(IFFT[V][n])
// Complexity: O(N log N) — one 2N-point IFFT
// Note: VGRE's Bluestein IFFT computes exp(-j2πnk/N) instead of exp(+j2πnk/N),
// so the twiddle in V must use the negative angle to compensate.
template<typename T>
static cufftResult_t dct3_1d(const T* in, T* out, int N) {
    if (N <= 0) return CUFFT_INVALID_VALUE;
    int TwoN = 2 * N;
    // Bluestein IFFT uses exp(-j2πnk/N) instead of standard exp(+j2πnk/N),
    // so flip the twiddle sign for non-pow2 sizes.
    const T sign3 = isPow2(TwoN) ? T(+1) : T(-1);
    std::vector<std::complex<T>> V(static_cast<std::size_t>(TwoN), std::complex<T>(T(0), T(0)));
    V[0] = std::complex<T>(in[0], T(0));
    for (int k = 1; k < N; ++k) {
        T angle = sign3 * static_cast<T>(PI) * k / static_cast<T>(TwoN);
        std::complex<T> tw(std::cos(angle), std::sin(angle));
        V[static_cast<std::size_t>(k)]       = std::complex<T>(in[k], T(0)) * tw;
        V[static_cast<std::size_t>(TwoN-k)]  = std::conj(V[static_cast<std::size_t>(k)]);
    }
    // V[N] stays 0 (already initialized)
    std::vector<std::complex<T>> y(static_cast<std::size_t>(TwoN));
    fft1d(V.data(), y.data(), TwoN, CUFFT_INVERSE);
    const T scale = static_cast<T>(N);
    for (int n = 0; n < N; ++n)
        out[n] = y[static_cast<std::size_t>(n)].real() * scale;
    return CUFFT_SUCCESS;
}

} // namespace

// ── Public execution entry points ────────────────────────────────────────────

extern "C" {

cufftResult_t cufftExecC2C(cufftHandle plan, void *idata, void *odata, int direction) {
    CufftPlan p;
    if (!lookupPlan(plan, p)) return CUFFT_INVALID_PLAN;
    if (p.type != CUFFT_C2C) return CUFFT_INVALID_TYPE;
    return execC2C<float>(p, idata, odata, direction);
}

cufftResult_t cufftExecZ2Z(cufftHandle plan, void *idata, void *odata, int direction) {
    CufftPlan p;
    if (!lookupPlan(plan, p)) return CUFFT_INVALID_PLAN;
    if (p.type != CUFFT_Z2Z) return CUFFT_INVALID_TYPE;
    return execC2C<double>(p, idata, odata, direction);
}

cufftResult_t cufftExecR2C(cufftHandle plan, void *idata, void *odata) {
    CufftPlan p;
    if (!lookupPlan(plan, p)) return CUFFT_INVALID_PLAN;
    if (p.type != CUFFT_R2C) return CUFFT_INVALID_TYPE;
    return execR2C<float>(p, idata, odata);
}

cufftResult_t cufftExecC2R(cufftHandle plan, void *idata, void *odata) {
    CufftPlan p;
    if (!lookupPlan(plan, p)) return CUFFT_INVALID_PLAN;
    if (p.type != CUFFT_C2R) return CUFFT_INVALID_TYPE;
    return execC2R<float>(p, idata, odata);
}

cufftResult_t cufftExecD2Z(cufftHandle plan, void *idata, void *odata) {
    CufftPlan p;
    if (!lookupPlan(plan, p)) return CUFFT_INVALID_PLAN;
    if (p.type != CUFFT_D2Z) return CUFFT_INVALID_TYPE;
    return execR2C<double>(p, idata, odata);
}

cufftResult_t cufftExecZ2D(cufftHandle plan, void *idata, void *odata) {
    CufftPlan p;
    if (!lookupPlan(plan, p)) return CUFFT_INVALID_PLAN;
    if (p.type != CUFFT_Z2D) return CUFFT_INVALID_TYPE;
    return execC2R<double>(p, idata, odata);
}

cufftResult_t cufftExecR16C(cufftHandle plan, void *idata, void *odata) {
    CufftPlan p;
    if (!lookupPlan(plan, p)) return CUFFT_INVALID_PLAN;
    if (p.type != CUFFT_R16C) return CUFFT_INVALID_TYPE;
    return execR16C(p, idata, odata);
}

cufftResult_t cufftExecC16R(cufftHandle plan, void *idata, void *odata) {
    CufftPlan p;
    if (!lookupPlan(plan, p)) return CUFFT_INVALID_PLAN;
    if (p.type != CUFFT_C16R) return CUFFT_INVALID_TYPE;
    return execC16R(p, idata, odata);
}

cufftResult_t cufftExecC16C(cufftHandle plan, void *idata, void *odata, int direction) {
    CufftPlan p;
    if (!lookupPlan(plan, p)) return CUFFT_INVALID_PLAN;
    if (p.type != CUFFT_C16C) return CUFFT_INVALID_TYPE;
    return execC16C(p, idata, odata, direction);
}

cufftResult_t cufftExecC16BFC(cufftHandle plan, void *idata, void *odata, int direction) {
    CufftPlan p;
    if (!lookupPlan(plan, p)) return CUFFT_INVALID_PLAN;
    if (p.type != CUFFT_C16BFC) return CUFFT_INVALID_TYPE;
    auto *in  = static_cast<cufftBF16Complex*>(idata);
    auto *out = static_cast<cufftBF16Complex*>(odata);
    if (!in || !out || !isValidDirection(direction)) return CUFFT_INVALID_VALUE;
    int64_t total = logicalElementCount(p);
    for (int b = 0; b < p.batch; ++b) {
        std::vector<std::complex<float>> tmp(static_cast<size_t>(total));
        std::vector<std::complex<float>> res(static_cast<size_t>(total));
        for (int64_t i = 0; i < total; ++i) {
            const cufftBF16Complex &v = in[inputIndex(p, b, i)];
            // BF16 → float32: place the 16-bit value in the upper 2 bytes of a float
            uint32_t rx = static_cast<uint32_t>(v.x) << 16;
            uint32_t ry = static_cast<uint32_t>(v.y) << 16;
            float fx, fy;
            memcpy(&fx, &rx, sizeof(float));
            memcpy(&fy, &ry, sizeof(float));
            tmp[static_cast<size_t>(i)] = {fx, fy};
        }
        if (p.rank == 1) fft1d(tmp.data(), res.data(), p.nx, direction);
        else if (p.rank == 2) fft2d(tmp.data(), res.data(), p.ny, p.nx, direction);
        else if (p.rank == 3) fft3d(tmp.data(), res.data(), p.nz, p.ny, p.nx, direction);
        else return CUFFT_INVALID_PLAN;
        for (int64_t i = 0; i < total; ++i) {
            cufftBF16Complex &v = out[outputIndex(p, b, i)];
            // float32 → BF16: take the upper 16 bits (round-to-nearest-even)
            uint32_t ur, ui;
            float re = res[static_cast<size_t>(i)].real();
            float im = res[static_cast<size_t>(i)].imag();
            memcpy(&ur, &re, sizeof(float));
            memcpy(&ui, &im, sizeof(float));
            // Round: add 0x7FFF + round bit, then truncate
            ur = (ur + 0x7FFF + ((ur >> 16) & 1)) >> 16;
            ui = (ui + 0x7FFF + ((ui >> 16) & 1)) >> 16;
            v.x = static_cast<uint16_t>(ur);
            v.y = static_cast<uint16_t>(ui);
        }
    }
    return CUFFT_SUCCESS;
}

cufftResult_t cufftExecDCT2(cufftHandle plan, float *idata, float *odata) {
    CufftPlan p;
    if (!lookupPlan(plan, p)) return CUFFT_INVALID_PLAN;
    if (p.type != CUFFT_DCT2) return CUFFT_INVALID_TYPE;
    if (!idata || !odata) return CUFFT_INVALID_VALUE;
    if (p.rank != 1) return CUFFT_INVALID_PLAN;
    for (int b = 0; b < p.batch; ++b) {
        cufftResult_t st = dct2_1d<float>(idata + b * p.nx, odata + b * p.nx, p.nx);
        if (st != CUFFT_SUCCESS) return st;
    }
    return CUFFT_SUCCESS;
}

cufftResult_t cufftExecDCT3(cufftHandle plan, float *idata, float *odata) {
    CufftPlan p;
    if (!lookupPlan(plan, p)) return CUFFT_INVALID_PLAN;
    if (p.type != CUFFT_DCT3) return CUFFT_INVALID_TYPE;
    if (!idata || !odata) return CUFFT_INVALID_VALUE;
    if (p.rank != 1) return CUFFT_INVALID_PLAN;
    for (int b = 0; b < p.batch; ++b) {
        cufftResult_t st = dct3_1d<float>(idata + b * p.nx, odata + b * p.nx, p.nx);
        if (st != CUFFT_SUCCESS) return st;
    }
    return CUFFT_SUCCESS;
}

} // extern "C"
