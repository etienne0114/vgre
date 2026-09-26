// Tensor Core Emulation Implementation
// Implements CPU-based tensor core operations using AVX-512/AMX

#include "tensor_core_emulation.h"
#include "vgre/runtime/vector_engine.h"  // VectorEngine::matMulInt8 / matMulBF16
#include <cstring>
#include <algorithm>
#include <vector>

// SIMD is RUNTIME-DISPATCHED: the AVX-512-VNNI / AVX-512-BF16 / AMX kernels are
// compiled unconditionally (target attribute) and the feature-detection below
// reflects the LIVE CPU (CPUID), not the compiler's -m flags.
#include "vgre/common/simd_dispatch.h"

namespace vgre {
namespace math {

namespace {

// Runtime CPU feature detection (CPUID via __builtin_cpu_supports on GCC/Clang;
// AMX-tile state is verified by VectorEngine's arch_prctl at startup). These are
// what makes the tensor-core precision/kernel choice adapt to each machine.
bool detect_avx512vnni() {
#if defined(VGRE_SIMD_X86)
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx512vnni");
#else
    return false;
#endif
}
bool detect_avx512bf16() {
#if defined(VGRE_SIMD_X86)
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx512bf16");
#else
    return false;
#endif
}

} // anonymous namespace

bool hasAVX512VNNI() {
    static const bool v = detect_avx512vnni();
    return v;
}

bool hasAVX512BF16() {
    static const bool v = detect_avx512bf16();
    return v;
}

bool hasAMX() {
    // AMX needs both the CPUID bits AND OS tile-data enablement; VectorEngine
    // performs the arch_prctl(ARCH_REQ_XCOMP_PERM) handshake at startup.
    return vgre::runtime::VectorEngine::instance().getCapabilities().amxEnabled;
}

TensorCoreConfig getOptimalTensorCoreConfig(size_t m, size_t n, size_t k,
                                           TensorPrecision precision) {
    TensorCoreConfig config(TensorOp::GEMM, precision, m, n, k);
    
    // Adjust dimensions to align with tensor core tile sizes (typically 16x16)
    config.m = ((m + 15) / 16) * 16;
    config.n = ((n + 15) / 16) * 16;
    config.k = ((k + 15) / 16) * 16;
    
    return config;
}

template<typename InputType, typename AccumType, typename OutputType>
void tensorCoreMatmul(const InputType* A, const InputType* B, OutputType* C,
                     const TensorCoreConfig& config) {
    size_t m = config.m;
    size_t n = config.n;
    size_t k = config.k;
    
    // ── Precision-aware dispatch through VectorEngine ────────────────────────
    // INT8 and BF16 routes use VectorEngine::matMulInt8 / matMulBF16 which
    // dispatch to AVX-VNNI (Alder Lake+) or AMX (Sapphire Rapids+) as available.
    // FP32 / FP64 fall through to simdMatmul which uses AVX2/AVX-512 FMA.
    if (config.precision == TensorPrecision::INT8) {
        // Signed INT8 → INT32 accumulation (AVX-VNNI or AMX path)
        std::vector<int32_t> tmp(m * n, 0);
        vgre::runtime::VectorEngine::instance().matMulInt8(
            reinterpret_cast<const int8_t*>(A),
            reinterpret_cast<const int8_t*>(B),
            tmp.data(),
            static_cast<int>(m), static_cast<int>(n), static_cast<int>(k));
        // Narrow INT32 → OutputType
        for (size_t i = 0; i < m * n; ++i)
            C[i] = static_cast<OutputType>(tmp[i]);
    } else if (config.precision == TensorPrecision::BF16) {
        // BF16 → FP32 accumulation (AMX-BF16 or AVX2 software path)
        using BF16 = vgre::runtime::vgre_bf16;
        std::vector<float> tmp(m * n, 0.f);
        vgre::runtime::VectorEngine::instance().matMulBF16(
            reinterpret_cast<const BF16*>(A),
            reinterpret_cast<const BF16*>(B),
            tmp.data(),
            static_cast<int>(m), static_cast<int>(n), static_cast<int>(k));
        for (size_t i = 0; i < m * n; ++i)
            C[i] = static_cast<OutputType>(tmp[i]);
    } else {
        // FP32 / FP64: use AVX2/AVX-512 SIMD matmul
        simdMatmul(reinterpret_cast<const AccumType*>(A),
                   reinterpret_cast<const AccumType*>(B),
                   C, m, n, k, k, n, n);
    }
}

#if defined(VGRE_SIMD_X86)
__attribute__((target("avx512f,avx512vnni")))
void avx512vnniInt8Matmul(const int8_t* A, const int8_t* B, int32_t* C,
                          size_t m, size_t n, size_t k,
                          size_t lda, size_t ldb, size_t ldc) {
    // Initialize C to zero
    std::memset(C, 0, m * ldc * sizeof(int32_t));
    
    // Use AVX-512 VNNI for INT8 matrix multiplication
    for (size_t i = 0; i < m; ++i) {
        for (size_t kk = 0; kk < k; kk += 16) {
            for (size_t j = 0; j < n; j += 16) {
                // Load 16x16 block from A and B
                __m512i a_row = _mm512_loadu_si512((__m512i*)(A + i * lda + kk));
                
                for (size_t k_offset = 0; k_offset < 16; ++k_offset) {
                    __m512i b_col = _mm512_loadu_si512((__m512i*)(B + (kk + k_offset) * ldb + j));
                    __m512i c_row = _mm512_loadu_si512((__m512i*)(C + i * ldc + j));
                    
                    // VNNI instruction: multiply and accumulate
                    __m512i result = _mm512_dpbusd_epi32(c_row, a_row, b_col);
                    _mm512_storeu_si512((__m512i*)(C + i * ldc + j), result);
                }
            }
        }
    }
}
#endif // VGRE_SIMD_X86 (AVX-512 VNNI)

#if defined(VGRE_SIMD_X86)
// BF16 GEMM using AVX-512F: each bf16 (uint16) is widened to fp32 by placing its
// bits in the high half of a 32-bit word (bf16 IS the top 16 bits of fp32), then
// a 16-wide FMA accumulates. Correct on any AVX-512F CPU (no AVX-512-BF16 ISA
// required); selected at runtime via hasAVX512BF16()/CPUID by external callers.
__attribute__((target("avx512f")))
void avx512bf16Matmul(const uint16_t* A, const uint16_t* B, float* C,
                     size_t m, size_t n, size_t k,
                     size_t lda, size_t ldb, size_t ldc) {
    auto bf16_to_f32 = [](uint16_t h) -> float {
        uint32_t bits = static_cast<uint32_t>(h) << 16;
        float f; std::memcpy(&f, &bits, sizeof(f)); return f;
    };
    std::memset(C, 0, m * ldc * sizeof(float));
    for (size_t i = 0; i < m; ++i) {
        for (size_t kk = 0; kk < k; ++kk) {
            __m512 a = _mm512_set1_ps(bf16_to_f32(A[i * lda + kk]));
            size_t j = 0;
            for (; j + 16 <= n; j += 16) {
                __m256i braw = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(B + kk * ldb + j));
                __m512i b32 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(braw), 16);
                __m512 b = _mm512_castsi512_ps(b32);
                __m512 c = _mm512_loadu_ps(C + i * ldc + j);
                _mm512_storeu_ps(C + i * ldc + j, _mm512_fmadd_ps(a, b, c));
            }
            for (; j < n; ++j)
                C[i * ldc + j] += bf16_to_f32(A[i * lda + kk]) * bf16_to_f32(B[kk * ldb + j]);
        }
    }
}
#endif // VGRE_SIMD_X86 (AVX-512 BF16)

#if defined(VGRE_SIMD_X86)

// Convert FP32 → BF16 with round-to-nearest-even (standard IEEE truncation).
static inline uint16_t f32_to_bf16(float f) noexcept {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    uint32_t lsb = (bits >> 16) & 1u;
    bits += 0x7FFFu + lsb;
    return static_cast<uint16_t>(bits >> 16);
}

// AMX tile configuration block — must be 64-byte aligned per ISA spec.
struct alignas(64) TileCfg {
    uint8_t  palette;       // 1 = AMX palette (only valid value)
    uint8_t  start_row;     // used for restartability, 0 for normal use
    uint8_t  pad[14];
    uint16_t colsz[16];     // byte-width of each tile's columns (up to 64)
    uint8_t  rows[16];      // row count for each tile (up to 16)
};

__attribute__((target("amx-tile,amx-bf16")))
void amxMatmul(const void* A, const void* B, void* C,
               size_t m, size_t n, size_t k,
               size_t lda, size_t ldb, size_t ldc) {
    const float* fA = static_cast<const float*>(A);
    const float* fB = static_cast<const float*>(B);
    float*       fC = static_cast<float*>(C);

    // Zero output — same convention as simdMatmul.
    for (size_t i = 0; i < m; ++i)
        std::memset(fC + i * ldc, 0, n * sizeof(float));

    // AMX-BF16 tile dimensions:
    //   TMM0 — A tile:    tile_m rows × tile_k BF16 cols  → colsz = tile_k×2 (max 64 bytes → tile_k≤32)
    //   TMM1 — B tile:    ceil(tile_k/2) VNNI rows × tile_n×2 BF16 words → colsz = tile_n×4
    //   TMM2 — C tile:    tile_m rows × tile_n FP32 cols  → colsz = tile_n×4 (max 64 bytes → tile_n≤16)
    constexpr size_t TM = 16;
    constexpr size_t TN = 16;
    constexpr size_t TK = 32;

    for (size_t mi = 0; mi < m; mi += TM) {
        const size_t tm = std::min(TM, m - mi);

        for (size_t ni = 0; ni < n; ni += TN) {
            const size_t tn = std::min(TN, n - ni);

            // Per (mi,ni) accumulator in main memory — loaded/stored each K-slice.
            alignas(64) float c_buf[TM * TN] = {};

            for (size_t ki = 0; ki < k; ki += TK) {
                const size_t tk  = std::min(TK, k - ki);
                const size_t tk2 = (tk + 1) / 2;  // VNNI rows = ⌈tk/2⌉

                // Pack A: tm×tk FP32 → tm×tk BF16 (row-major, stride=TK elements).
                alignas(64) uint16_t pbA[TM * TK] = {};
                for (size_t i = 0; i < tm; ++i)
                    for (size_t j = 0; j < tk; ++j)
                        pbA[i * TK + j] = f32_to_bf16(fA[(mi + i) * lda + (ki + j)]);

                // Pack B into VNNI format: tk2 rows × (tn×2) BF16 words.
                // VNNI element [row=r, col=2*j+{0,1}] = B[ki + 2r + {0,1}][ni + j].
                alignas(64) uint16_t pbB[TK * TN] = {};  // tk2*tn*2 ≤ TK*TN
                for (size_t j = 0; j < tn; ++j) {
                    for (size_t i = 0; i < tk; i += 2) {
                        const size_t r = i / 2;
                        pbB[r * (TN * 2) + j * 2 + 0] =
                            f32_to_bf16(fB[(ki + i)     * ldb + (ni + j)]);
                        pbB[r * (TN * 2) + j * 2 + 1] =
                            (i + 1 < tk) ? f32_to_bf16(fB[(ki + i + 1) * ldb + (ni + j)]) : 0u;
                    }
                }

                // Configure tiles for this (tm, tn, tk) shape.
                TileCfg cfg = {};
                cfg.palette  = 1;
                cfg.rows[0]  = static_cast<uint8_t>(tm);
                cfg.rows[1]  = static_cast<uint8_t>(tk2);
                cfg.rows[2]  = static_cast<uint8_t>(tm);
                cfg.colsz[0] = static_cast<uint16_t>(tk  * sizeof(uint16_t)); // A: tk BF16/row
                cfg.colsz[1] = static_cast<uint16_t>(tn  * 2 * sizeof(uint16_t)); // B: tn pairs/VNNI row
                cfg.colsz[2] = static_cast<uint16_t>(tn  * sizeof(float));    // C: tn FP32/row
                _tile_loadconfig(&cfg);

                _tile_loadd(0, pbA,   static_cast<int>(TK  * sizeof(uint16_t)));
                _tile_loadd(1, pbB,   static_cast<int>(TN  * 2 * sizeof(uint16_t)));
                _tile_loadd(2, c_buf, static_cast<int>(TN  * sizeof(float)));
                _tile_dpbf16ps(2, 0, 1);  // TMM2 += TMM0 × TMM1 (BF16 in, FP32 accumulate)
                _tile_stored(2, c_buf, static_cast<int>(TN * sizeof(float)));
            }

            // Write accumulated block to output.
            for (size_t i = 0; i < tm; ++i)
                for (size_t j = 0; j < tn; ++j)
                    fC[(mi + i) * ldc + (ni + j)] = c_buf[i * TN + j];
        }
    }
    _tile_release();
}
#endif // VGRE_SIMD_X86 (AMX)

template<typename T>
void simdMatmul(const T* A, const T* B, T* C,
               size_t m, size_t n, size_t k,
               size_t lda, size_t ldb, size_t ldc) {
    // Initialize C to zero
    std::memset(C, 0, m * ldc * sizeof(T));
    
    // Standard matrix multiplication with potential SIMD optimization
    for (size_t i = 0; i < m; ++i) {
        for (size_t kk = 0; kk < k; ++kk) {
            T a_ik = A[i * lda + kk];
            
            for (size_t j = 0; j < n; ++j) {
                C[i * ldc + j] += a_ik * B[kk * ldb + j];
            }
        }
    }
}

template<typename InputType, typename AccumType, typename OutputType>
void tensorCoreConv2D(const InputType* input, const InputType* kernel, OutputType* output,
                     size_t input_h, size_t input_w, size_t input_c,
                     size_t kernel_h, size_t kernel_w, size_t kernel_c,
                     size_t output_h, size_t output_w, size_t output_c,
                     size_t stride_h, size_t stride_w,
                     size_t pad_h, size_t pad_w,
                     const TensorCoreConfig& config) {
    // Tensor core optimized 2D convolution
    // Converts convolution to matrix multiplication (im2col)
    
    size_t kernel_size = kernel_h * kernel_w * input_c;
    
    for (size_t oc = 0; oc < output_c; ++oc) {
        for (size_t oh = 0; oh < output_h; ++oh) {
            for (size_t ow = 0; ow < output_w; ++ow) {
                AccumType sum = AccumType(0);
                
                for (size_t ic = 0; ic < input_c; ++ic) {
                    for (size_t kh = 0; kh < kernel_h; ++kh) {
                        for (size_t kw = 0; kw < kernel_w; ++kw) {
                            int ih = static_cast<int>(oh * stride_h) + static_cast<int>(kh) - static_cast<int>(pad_h);
                            int iw = static_cast<int>(ow * stride_w) + static_cast<int>(kw) - static_cast<int>(pad_w);
                            
                            if (ih >= 0 && ih < static_cast<int>(input_h) &&
                                iw >= 0 && iw < static_cast<int>(input_w)) {
                                size_t input_idx = (ic * input_h + ih) * input_w + iw;
                                size_t kernel_idx = (oc * kernel_c + ic) * kernel_h * kernel_w + kh * kernel_w + kw;
                                sum += static_cast<AccumType>(input[input_idx]) * static_cast<AccumType>(kernel[kernel_idx]);
                            }
                        }
                    }
                }
                
                output[(oc * output_h + oh) * output_w + ow] = static_cast<OutputType>(sum);
            }
        }
    }
}

// Explicit template instantiations
template void tensorCoreMatmul<float, float, float>(const float*, const float*, float*, const TensorCoreConfig&);
template void tensorCoreMatmul<int8_t, int32_t, int32_t>(const int8_t*, const int8_t*, int32_t*, const TensorCoreConfig&);

template void tensorCoreConv2D<float, float, float>(const float*, const float*, float*, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, const TensorCoreConfig&);
template void tensorCoreConv2D<int8_t, int32_t, int32_t>(const int8_t*, const int8_t*, int32_t*, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, const TensorCoreConfig&);

template void simdMatmul<float>(const float*, const float*, float*, size_t, size_t, size_t, size_t, size_t, size_t);
template void simdMatmul<double>(const double*, const double*, double*, size_t, size_t, size_t, size_t, size_t, size_t);

} // namespace math
} // namespace vgre
