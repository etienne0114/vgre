#include "vgre/runtime/vector_engine.h"
#include "vgre/common/logger.h"

#include <cstdint>
#include <cstring>
#include <cmath>
#include <chrono>
#include <sstream>

// __x86_64__ is GCC/Clang-only; MSVC (cl.exe) defines _M_X64/_M_AMD64
// instead and never defines __x86_64__ at all. Every CPUID-based capability
// check in this file was previously gated on __x86_64__ alone, so on an
// MSVC build detectCapabilities()'s entire body silently compiled out —
// caps_ stayed zero-initialized (all false) and every kernel fell back to
// scalar-only execution, with nothing indicating this was wrong beyond the
// "SIMD: none (scalar only)" log line at startup.
#if defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64)
#define VGRE_VECENGINE_X86_64 1
#else
#define VGRE_VECENGINE_X86_64 0
#endif

#if VGRE_VECENGINE_X86_64
#if defined(_MSC_VER)
#include <intrin.h>
// __get_cpuid/__get_cpuid_count (GCC/Clang, <cpuid.h>) return eax/ebx/ecx/edx
// via separate out-params; MSVC's __cpuid/__cpuidex (<intrin.h>) return the
// same four values packed into one int[4]. Wrap both behind the GCC/Clang
// call shape so the capability-bit logic below needs no per-compiler branches.
static inline int __get_cpuid(unsigned leaf, unsigned* eax, unsigned* ebx,
                               unsigned* ecx, unsigned* edx) {
    int regs[4];
    __cpuid(regs, static_cast<int>(leaf));
    *eax = static_cast<unsigned>(regs[0]);
    *ebx = static_cast<unsigned>(regs[1]);
    *ecx = static_cast<unsigned>(regs[2]);
    *edx = static_cast<unsigned>(regs[3]);
    return 1;
}
static inline int __get_cpuid_count(unsigned leaf, unsigned subleaf,
                                     unsigned* eax, unsigned* ebx,
                                     unsigned* ecx, unsigned* edx) {
    int regs[4];
    __cpuidex(regs, static_cast<int>(leaf), static_cast<int>(subleaf));
    *eax = static_cast<unsigned>(regs[0]);
    *ebx = static_cast<unsigned>(regs[1]);
    *ecx = static_cast<unsigned>(regs[2]);
    *edx = static_cast<unsigned>(regs[3]);
    return 1;
}
#else
#include <cpuid.h>
#endif
#endif

#if defined(__linux__) && defined(__x86_64__)
#include "vgre/common/os_backend.h"
#include <sys/syscall.h>  // SYS_arch_prctl — AMX tile-data state
#ifndef ARCH_REQ_XCOMP_PERM
#define ARCH_REQ_XCOMP_PERM  0x1023
#endif
#ifndef XFEATURE_XTILEDATA
#define XFEATURE_XTILEDATA   18
#endif
#else
#include "vgre/common/os_backend.h"
#endif

// SIMD is RUNTIME-DISPATCHED: the per-ISA kernels below are compiled
// unconditionally via __attribute__((target(...))) and selected at runtime by
// CPUID (vgre::cpu::supports), so one portable binary runs the widest ISA the
// host actually has — no ISA is baked into the object at compile time.
#include "vgre/common/cpu_features.h"
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__)) && !defined(_MSC_VER)
#  define VGRE_VEC_X86 1
#  include <immintrin.h>
#endif

namespace vgre {
namespace runtime {

namespace {
enum class VecIsa { Scalar, Avx2, Avx512 };
VecIsa vecIsa() {
    static const VecIsa v = [] {
#if defined(VGRE_VEC_X86)
        if (vgre::cpu::supports("avx512f")) return VecIsa::Avx512;
        if (vgre::cpu::supports("avx2") && vgre::cpu::supports("fma")) return VecIsa::Avx2;
#endif
        return VecIsa::Scalar;
    }();
    return v;
}
}  // namespace


VectorEngine::VectorEngine() {
    detectCapabilities();
    VGRE_LOG_INFO("VectorEngine",
                  "Initialized — " + getCapabilityString());
}

VectorEngine::~VectorEngine() = default;

// ── CPU feature detection ──────────────────────────────────────────────────
void VectorEngine::detectCapabilities() {
    #if VGRE_VECENGINE_X86_64
    unsigned int eax, ebx, ecx, edx;

    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        caps_.hasSSE2 = (edx >> 26) & 1;
        caps_.hasSSE4 = (ecx >> 19) & 1;
        caps_.hasAVX  = (ecx >> 28) & 1;
        caps_.hasFMA  = (ecx >> 12) & 1;
    }

    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        caps_.hasAVX2    = (ebx >> 5)  & 1;
        caps_.hasAVX512  = (ebx >> 16) & 1;
        caps_.hasVNNI    = (ecx >> 11) & 1;  // AVX-512 VNNI (Ice Lake+)
        caps_.hasAMXBF16 = (edx >> 22) & 1;  // AMX-BF16: leaf 7.0 EDX[22]
        caps_.hasAMXTile = (edx >> 24) & 1;  // AMX-TILE: leaf 7.0 EDX[24]
    }
    // AVX-VNNI without AVX-512: CPUID leaf 7.1 EAX[4] (Alder Lake / Raptor Lake)
    if (__get_cpuid_count(7, 1, &eax, &ebx, &ecx, &edx)) {
        caps_.hasAVXVNNI = (eax >> 4) & 1;
    }
    // Treat either VNNI variant as usable for INT8 acceleration
    if (caps_.hasAVXVNNI) caps_.hasVNNI = true;

    // AMX requires both CPUID bits AND OS permission.
    // Verify via XGETBV that the OS has enabled XTILECFG (bit 17) and
    // XTILEDATA (bit 18) in XCR0, then request XTILEDATA via arch_prctl.
    if (caps_.hasAMXTile && caps_.hasAMXBF16) {
#if defined(__linux__)
        // XGETBV: read XCR0 to confirm OS-granted tile state
        uint64_t xcr0 = 0;
        __asm__ volatile(
            "xgetbv"
            : "=A"(xcr0)   // EAX:EDX → xcr0 (64-bit)
            : "c"(0)        // XCR index 0
        );
        bool osTileConfigOk = (xcr0 >> 17) & 1; // XTILECFG
        bool osTileDataOk   = (xcr0 >> 18) & 1; // XTILEDATA

        if (!osTileDataOk) {
            // Ask the OS to enable XTILEDATA for this thread (and its children).
            // Without this, the first AMX instruction raises SIGILL.
            int rc = static_cast<int>(syscall(SYS_arch_prctl,
                                               ARCH_REQ_XCOMP_PERM,
                                               XFEATURE_XTILEDATA));
            if (rc == 0) {
                // Re-read XCR0 after enablement
                __asm__ volatile("xgetbv" : "=A"(xcr0) : "c"(0));
                osTileConfigOk = (xcr0 >> 17) & 1;
                osTileDataOk   = (xcr0 >> 18) & 1;
            }
        }

        caps_.amxEnabled = osTileConfigOk && osTileDataOk;
        if (caps_.amxEnabled) {
            VGRE_LOG_INFO("VectorEngine", "Intel AMX tile state enabled via arch_prctl");
        } else {
            VGRE_LOG_WARN("VectorEngine",
                "AMX CPUID bits set but OS tile state unavailable — AMX path disabled");
        }
#else
        // Windows / macOS: AMX not yet supported by the OS (as of 2026)
        caps_.amxEnabled = false;
#endif
    }

    caps_.hasAMX = caps_.amxEnabled;  // legacy alias
    #else
    // Non-x86: no SIMD detection
    caps_ = {};
    #endif
}

const SIMDCapabilities& VectorEngine::getCapabilities() const {
    return caps_;
}

std::string VectorEngine::getCapabilityString() const {
    std::ostringstream oss;
    oss << "SIMD: ";
    if (caps_.amxEnabled)  oss << "AMX-BF16(active) ";
    else if (caps_.hasAMXTile) oss << "AMX(hw-only/disabled) ";
    if (caps_.hasAVXVNNI) oss << "AVX-VNNI ";
    else if (caps_.hasVNNI) oss << "AVX512-VNNI ";
    if (caps_.hasAVX512) oss << "AVX-512F ";
    if (caps_.hasAVX2)   oss << "AVX2 ";
    if (caps_.hasAVX)    oss << "AVX ";
    if (caps_.hasFMA)    oss << "FMA ";
    if (caps_.hasSSE4)   oss << "SSE4.1 ";
    if (caps_.hasSSE2)   oss << "SSE2 ";
    if (!caps_.hasSSE2 && !caps_.hasAVX) oss << "none (scalar only)";
    return oss.str();
}

// ── High-Performance GFLOPS Benchmark ────────────────────────────────────────
#if defined(VGRE_VEC_X86)
__attribute__((target("avx512f")))
static size_t fma_bench_avx512(const float* pa, const float* pb, const float* pc,
                               float* pout, size_t n, int iterations) {
    size_t nAligned = n & ~size_t(15);
    for (int iter = 0; iter < iterations; ++iter) {
        for (size_t i = 0; i < nAligned; i += 16) {
            __m512 va = _mm512_loadu_ps(&pa[i]);
            __m512 vb = _mm512_loadu_ps(&pb[i]);
            __m512 vc = _mm512_loadu_ps(&pc[i]);
            for (int k = 0; k < 32; ++k) vc = _mm512_fmadd_ps(va, vb, vc);  // stay in regs
            _mm512_storeu_ps(&pout[i], vc);
        }
    }
    return nAligned;
}
__attribute__((target("avx2,fma")))
static size_t fma_bench_avx2(const float* pa, const float* pb, const float* pc,
                             float* pout, size_t n, int iterations) {
    size_t nAligned = n & ~size_t(7);
    for (int iter = 0; iter < iterations; ++iter) {
        for (size_t i = 0; i < nAligned; i += 8) {
            __m256 va = _mm256_loadu_ps(&pa[i]);
            __m256 vb = _mm256_loadu_ps(&pb[i]);
            __m256 vc = _mm256_loadu_ps(&pc[i]);
            for (int k = 0; k < 32; ++k) vc = _mm256_fmadd_ps(va, vb, vc);
            _mm256_storeu_ps(&pout[i], vc);
        }
    }
    return nAligned;
}
#endif

double VectorEngine::benchmarkFMA(size_t n, int iterations) {
    if (n == 0 || iterations == 0) return 0.0;
    // Guard against callers passing huge n that would exhaust memory (OOM DoS).
    // 4 vectors × n × 4 bytes; cap at 128 MB total (32M floats per vector).
    static constexpr size_t kMaxBenchN = 32 * 1024 * 1024ULL;
    if (n > kMaxBenchN) n = kMaxBenchN;
    // Cap iterations to prevent double overflow in the FLOP counter below.
    // Max representable: (32M × 32 × 2 × kMaxIter) must stay < DBL_MAX (~1.8e308).
    // At n=32M, 32×2=64 FLOPs/elem → max safe iterations = floor(1.8e308 / (32M × 64)) ≈ 8e292.
    // Practical cap at 1M iterations to keep benchmark runtime sane.
    static constexpr int kMaxIterations = 1'000'000;
    if (iterations > kMaxIterations) iterations = kMaxIterations;

    std::vector<float> a(n, 1.1f), b(n, 2.2f), c(n, 3.3f), out(n, 0.0f);
    float* pa = a.data();
    float* pb = b.data();
    float* pc = c.data();
    float* pout = out.data();

    auto start = std::chrono::steady_clock::now();

    // Returns the element count processed per iteration (aligned n) so the
    // caller can compute FLOPs. 32 FMAs/elem in the SIMD paths, 1 MAC/elem scalar.
    size_t elemsPerIter = 0;
    double flopsPerElem = 2.0;   // scalar: 1 mul + 1 add
    switch (vecIsa()) {
#if defined(VGRE_VEC_X86)
        case VecIsa::Avx512: elemsPerIter = fma_bench_avx512(pa, pb, pc, pout, n, iterations); flopsPerElem = 64.0; break;
        case VecIsa::Avx2:   elemsPerIter = fma_bench_avx2(pa, pb, pc, pout, n, iterations);   flopsPerElem = 64.0; break;
#endif
        default:
            for (int iter = 0; iter < iterations; ++iter)
                for (size_t i = 0; i < n; ++i) pout[i] = pa[i] * pb[i] + pc[i];
            elemsPerIter = n;
            break;
    }

    auto end = std::chrono::steady_clock::now();
    double seconds = std::chrono::duration<double>(end - start).count();

    double totalFlops = static_cast<double>(elemsPerIter) * flopsPerElem * iterations;
    return totalFlops / (seconds * 1e9);
}

// Escape a SIMD register via an asm barrier so the compiler cannot dead-code-
// eliminate the FMA chain. Zero-instruction hint — no load/store, no aliasing.
#if defined(__GNUC__) || defined(__clang__)
#define VGRE_ESCAPE_XMM(v128) __asm__ volatile("" : "+x"(v128))
#else
#define VGRE_ESCAPE_XMM(v128) (void)(v128)
#endif
#if defined(VGRE_VEC_X86)
__attribute__((target("avx512f")))
static size_t bf16_bench_avx512(const vgre_bf16* pa, const vgre_bf16* pb, size_t n, int iterations) {
    size_t nAligned = n & ~size_t(15);
    for (int iter = 0; iter < iterations; ++iter) {
        for (size_t i = 0; i < nAligned; i += 16) {
            __m256i raw_a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&pa[i]));
            __m256i raw_b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&pb[i]));
            __m512i i_a = _mm512_slli_epi32(_mm512_cvtepu16_epi32(raw_a), 16);
            __m512i i_b = _mm512_slli_epi32(_mm512_cvtepu16_epi32(raw_b), 16);
            __m512 va = _mm512_castsi512_ps(i_a), vb = _mm512_castsi512_ps(i_b);
            __m512 vsum = _mm512_setzero_ps();
            for (int k = 0; k < 32; ++k) vsum = _mm512_fmadd_ps(va, vb, vsum);
            __m128 v128 = _mm512_castps512_ps128(vsum);
            VGRE_ESCAPE_XMM(v128);
        }
    }
    return nAligned;
}
__attribute__((target("avx2,fma")))
static size_t bf16_bench_avx2(const vgre_bf16* pa, const vgre_bf16* pb, size_t n, int iterations) {
    size_t nAligned = n & ~size_t(7);
    for (int iter = 0; iter < iterations; ++iter) {
        for (size_t i = 0; i < nAligned; i += 8) {
            __m128i raw_a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&pa[i]));
            __m128i raw_b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&pb[i]));
            __m256i i_a = _mm256_slli_epi32(_mm256_cvtepu16_epi32(raw_a), 16);
            __m256i i_b = _mm256_slli_epi32(_mm256_cvtepu16_epi32(raw_b), 16);
            __m256 va = _mm256_castsi256_ps(i_a), vb = _mm256_castsi256_ps(i_b);
            __m256 vsum = _mm256_setzero_ps();
            for (int k = 0; k < 32; ++k) vsum = _mm256_fmadd_ps(va, vb, vsum);
            __m128 v128 = _mm256_castps256_ps128(vsum);
            VGRE_ESCAPE_XMM(v128);
        }
    }
    return nAligned;
}
#endif
#undef VGRE_ESCAPE_XMM

double VectorEngine::benchmarkBF16(size_t n, int iterations) {
    if (n == 0 || iterations == 0) return 0.0;
    static constexpr size_t kMaxBenchN    = 32 * 1024 * 1024ULL;
    static constexpr int    kMaxIterations = 1'000'000;
    if (n > kMaxBenchN)          n = kMaxBenchN;
    if (iterations > kMaxIterations) iterations = kMaxIterations;

    std::vector<vgre_bf16> a(n, fp32_to_bf16(1.1f)), b(n, fp32_to_bf16(2.2f));
    vgre_bf16* pa = a.data();
    vgre_bf16* pb = b.data();

    auto start = std::chrono::steady_clock::now();

    size_t elemsPerIter = 0;
    double flopsPerElem = 2.0;   // scalar
    switch (vecIsa()) {
#if defined(VGRE_VEC_X86)
        case VecIsa::Avx512: elemsPerIter = bf16_bench_avx512(pa, pb, n, iterations); flopsPerElem = 64.0; break;
        case VecIsa::Avx2:   elemsPerIter = bf16_bench_avx2(pa, pb, n, iterations);   flopsPerElem = 64.0; break;
#endif
        default: {
            volatile float sink = 0.0f;
            for (int iter = 0; iter < iterations; ++iter) {
                float sum = 0.0f;
                for (size_t i = 0; i < n; ++i)
                    sum += bf16_to_fp32(pa[i]) * bf16_to_fp32(pb[i]);
                sink = sum;
            }
            (void)sink;
            elemsPerIter = n;
            break;
        }
    }

    auto end = std::chrono::steady_clock::now();
    double seconds = std::chrono::duration<double>(end - start).count();
    // Minimum floor: guard against sub-resolution timer readings that would
    // produce physically impossible GFLOPS values and mislead calibration.
    if (seconds < 1e-4) seconds = 1e-4;

    double totalFlops = static_cast<double>(elemsPerIter) * flopsPerElem * iterations;
    return totalFlops / (seconds * 1e9);
}

// ── Memory operations ──────────────────────────────────────────────────────
#if defined(VGRE_VEC_X86)
__attribute__((target("avx512f")))
static void fill_avx512(float* dst, float value, size_t n) {
    __m512 vval = _mm512_set1_ps(value);
    size_t i = 0, na = n & ~size_t(15);
    for (; i < na; i += 16) _mm512_storeu_ps(&dst[i], vval);
    for (; i < n; ++i) dst[i] = value;
}
__attribute__((target("avx2,fma")))
static void fill_avx2(float* dst, float value, size_t n) {
    __m256 vval = _mm256_set1_ps(value);
    size_t i = 0, na = n & ~size_t(7);
    for (; i < na; i += 8) _mm256_storeu_ps(&dst[i], vval);
    for (; i < n; ++i) dst[i] = value;
}
#endif
void VectorEngine::vectorFill(float* dst, float value, size_t n) {
#if defined(VGRE_VEC_X86)
    switch (vecIsa()) {
        case VecIsa::Avx512: fill_avx512(dst, value, n); return;
        case VecIsa::Avx2:   fill_avx2(dst, value, n);   return;
        default: break;
    }
#endif
    for (size_t i = 0; i < n; ++i) dst[i] = value;
}

void VectorEngine::vectorCopy(const float* src, float* dst, size_t n) {
    memcpy(dst, src, n * sizeof(float));
}

// ── BF16 dot product ───────────────────────────────────────────────────────
#if defined(VGRE_VEC_X86)
__attribute__((target("avx512f")))
static float bf16_dot_avx512(const vgre_bf16* a, const vgre_bf16* b, size_t n) {
    __m512 vsum = _mm512_setzero_ps();
    size_t i = 0, na = n & ~size_t(15);
    for (; i < na; i += 16) {
        __m256i raw_a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&a[i]));
        __m256i raw_b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&b[i]));
        // BF16 → FP32: zero-extend to 32b then shift the bits into the high half
        __m512i i_a = _mm512_slli_epi32(_mm512_cvtepu16_epi32(raw_a), 16);
        __m512i i_b = _mm512_slli_epi32(_mm512_cvtepu16_epi32(raw_b), 16);
        vsum = _mm512_fmadd_ps(_mm512_castsi512_ps(i_a), _mm512_castsi512_ps(i_b), vsum);
    }
    float sum = _mm512_reduce_add_ps(vsum);   // AVX512F-only horizontal reduce
    for (; i < n; ++i) sum += bf16_to_fp32(a[i]) * bf16_to_fp32(b[i]);
    return sum;
}
__attribute__((target("avx2,fma")))
static float bf16_dot_avx2(const vgre_bf16* a, const vgre_bf16* b, size_t n) {
    __m256 vsum = _mm256_setzero_ps();
    size_t i = 0, na = n & ~size_t(7);
    for (; i < na; i += 8) {
        __m128i raw_a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i]));
        __m128i raw_b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&b[i]));
        __m256i i_a = _mm256_slli_epi32(_mm256_cvtepu16_epi32(raw_a), 16);
        __m256i i_b = _mm256_slli_epi32(_mm256_cvtepu16_epi32(raw_b), 16);
        vsum = _mm256_fmadd_ps(_mm256_castsi256_ps(i_a), _mm256_castsi256_ps(i_b), vsum);
    }
    __m128 hi = _mm256_extractf128_ps(vsum, 1);
    __m128 lo = _mm256_castps256_ps128(vsum);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    float sum = _mm_cvtss_f32(s);
    for (; i < n; ++i) sum += bf16_to_fp32(a[i]) * bf16_to_fp32(b[i]);
    return sum;
}
#endif
float VectorEngine::vectorDot(const vgre_bf16* a, const vgre_bf16* b, size_t n) {
#if defined(VGRE_VEC_X86)
    switch (vecIsa()) {
        case VecIsa::Avx512: return bf16_dot_avx512(a, b, n);
        case VecIsa::Avx2:   return bf16_dot_avx2(a, b, n);
        default: break;
    }
#endif
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) sum += bf16_to_fp32(a[i]) * bf16_to_fp32(b[i]);
    return sum;
}

// ── Singleton ──────────────────────────────────────────────────────────────
VectorEngine& VectorEngine::instance() {
    static VectorEngine* engine = new VectorEngine();
    return *engine;
}


} // namespace runtime
} // namespace vgre
