// Track 9 — warp-collective tensor-core mma.sync correctness.
//
// The raw-PTX mma.sync.* helpers (vgre_mma_*) are WARP-COLLECTIVE: a single lane
// holds only a *fragment* of A/B/C (e.g. m16n8k16.f16: 8 of A's 256 elements), so
// its 4 outputs cannot be computed in isolation. VGRE runs them with one OS
// thread per lane sharing a per-warp scratch buffer + a block barrier — each lane
// deposits its fragment, barriers, the full tile GEMM is reconstructed, and this
// lane's outputs are scattered back.
//
// This test drives the SAME runtime path the JIT codegen sets up for `usesMma`
// kernels: it dispatches 32 lanes via vgre_jit_block_dispatch (which installs the
// block barrier exactly as a real kernel launch does), points each lane's TLS at
// a shared fragment buffer, and calls the helper. The host packs A/B/C into
// fragments using the canonical PTX ISA layout (ISA 9.7.14.4.x) and checks the
// reassembled D == A·B + C from an INDEPENDENT reference. A wrong fragment→lane
// map yields a wrong D — so this proves the layout, not just "it ran".

#include "vgre/compiler/wmma_emulation.h"   // vgre_mma_* (warp-collective helpers)

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

extern "C" {
  void vgre_jit_block_dispatch(int, void (*)(int, void*), void*);
  void** vgre_jit_get_mma_buffer();
  vgre::dim3* vgre_jit_get_threadIdx();
  vgre::dim3* vgre_jit_get_blockDim();
}

static int g_pass = 0, g_total = 0;
static void check(const char *name, bool ok) {
    ++g_total;
    printf(ok ? "  PASS  %s\n" : "  FAIL  %s\n", name);
    if (ok) ++g_pass;
}

static inline uint16_t f16bits(float f) { return vgre_cuda::__half(f).__x; }
static inline float    f16val (float f) { return float(vgre_cuda::__half(f)); }
static inline uint32_t pack2(uint16_t lo, uint16_t hi) {
    return (uint32_t)lo | ((uint32_t)hi << 16);
}
static inline uint32_t packs8(int8_t a, int8_t b, int8_t c, int8_t d) {
    return (uint32_t)(uint8_t)a | ((uint32_t)(uint8_t)b << 8) |
           ((uint32_t)(uint8_t)c << 16) | ((uint32_t)(uint8_t)d << 24);
}
static inline uint32_t pack4fp8(uint8_t a, uint8_t b, uint8_t c, uint8_t d) {
    return (uint32_t)a | ((uint32_t)b << 8) | ((uint32_t)c << 16) | ((uint32_t)d << 24);
}

// Shared block scratch: two warps × 32 lanes × 16 u32 slots per lane.
static uint32_t g_mma_scratch[64 * 16];

// ── m16n8k16 f16·f16→f32 ─────────────────────────────────────────────────────
struct F16Job {
    const uint32_t *Areg, *Breg; const float *Creg; float *Dreg;  // [32×..]
};
static void f16_lane(int tid, void *arg) {
    auto *j = static_cast<F16Job *>(arg);
    *vgre_jit_get_mma_buffer() = g_mma_scratch;
    *vgre_jit_get_threadIdx()  = vgre::dim3((uint32_t)tid, 0, 0);
    *vgre_jit_get_blockDim()   = vgre::dim3(32, 1, 1);
    const uint32_t *A = j->Areg + tid * 4;
    const uint32_t *B = j->Breg + tid * 2;
    const float    *C = j->Creg + tid * 4;
    float d0, d1, d2, d3;
    vgre_mma_m16n8k16_f32_f16(d0, d1, d2, d3, A[0], A[1], A[2], A[3], B[0], B[1],
                              C[0], C[1], C[2], C[3]);
    float *D = j->Dreg + tid * 4;
    D[0] = d0; D[1] = d1; D[2] = d2; D[3] = d3;
}

static bool runF16() {
    const int M = 16, N = 8, K = 16;
    std::mt19937 rng(20260611);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> A(M * K), B(K * N), C(M * N);
    for (auto &x : A) x = f16val(nd(rng));
    for (auto &x : B) x = f16val(nd(rng));
    for (auto &x : C) x = nd(rng);

    std::vector<uint32_t> Areg(32 * 4), Breg(32 * 2);
    std::vector<float>    Creg(32 * 4), Dreg(32 * 4, 0.0f);
    for (int lane = 0; lane < 32; ++lane) {
        int g = lane >> 2, t = lane & 3;
        auto Ab = [&](int m, int k){ return f16bits(A[m * K + k]); };
        Areg[lane*4+0] = pack2(Ab(g,   2*t+0), Ab(g,   2*t+1));
        Areg[lane*4+1] = pack2(Ab(g+8, 2*t+0), Ab(g+8, 2*t+1));
        Areg[lane*4+2] = pack2(Ab(g,   2*t+8), Ab(g,   2*t+9));
        Areg[lane*4+3] = pack2(Ab(g+8, 2*t+8), Ab(g+8, 2*t+9));
        auto Bb = [&](int k, int n){ return f16bits(B[k * N + n]); };
        Breg[lane*2+0] = pack2(Bb(2*t+0, g), Bb(2*t+1, g));
        Breg[lane*2+1] = pack2(Bb(2*t+8, g), Bb(2*t+9, g));
        Creg[lane*4+0] = C[(g)   * N + 2*t+0]; Creg[lane*4+1] = C[(g)   * N + 2*t+1];
        Creg[lane*4+2] = C[(g+8) * N + 2*t+0]; Creg[lane*4+3] = C[(g+8) * N + 2*t+1];
    }

    F16Job job{Areg.data(), Breg.data(), Creg.data(), Dreg.data()};
    std::memset(g_mma_scratch, 0, sizeof(g_mma_scratch));
    vgre_jit_block_dispatch(32, f16_lane, &job);

    std::vector<float> D(M * N, 0.0f);
    for (int lane = 0; lane < 32; ++lane) {
        int g = lane >> 2, t = lane & 3;
        D[(g)   * N + 2*t+0] = Dreg[lane*4+0]; D[(g)   * N + 2*t+1] = Dreg[lane*4+1];
        D[(g+8) * N + 2*t+0] = Dreg[lane*4+2]; D[(g+8) * N + 2*t+1] = Dreg[lane*4+3];
    }
    double maxErr = 0.0; int at = 0;
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float ref = C[m * N + n];
            for (int kk = 0; kk < K; ++kk) ref += f16val(A[m*K+kk]) * f16val(B[kk*N+n]);
            double e = std::fabs(D[m * N + n] - ref);
            if (e > maxErr) { maxErr = e; at = m * N + n; }
        }
    printf("  [info] m16n8k16.f16 max abs err vs A*B+C = %.2e at (%d,%d): D=%.4f\n",
           maxErr, at / N, at % N, D[at]);
    return maxErr < 1e-2;
}

// ── m16n8k32 s8·s8→s32 ───────────────────────────────────────────────────────
struct S8Job {
    const uint32_t *Areg, *Breg; const int *Creg; int *Dreg;
};
static void s8_lane(int tid, void *arg) {
    auto *j = static_cast<S8Job *>(arg);
    *vgre_jit_get_mma_buffer() = g_mma_scratch;
    *vgre_jit_get_threadIdx()  = vgre::dim3((uint32_t)tid, 0, 0);
    *vgre_jit_get_blockDim()   = vgre::dim3(32, 1, 1);
    const uint32_t *A = j->Areg + tid * 4;
    const uint32_t *B = j->Breg + tid * 2;
    const int      *C = j->Creg + tid * 4;
    int d0, d1, d2, d3;
    vgre_mma_m16n8k32_s8(d0, d1, d2, d3, A[0], A[1], A[2], A[3], B[0], B[1],
                         C[0], C[1], C[2], C[3]);
    int *D = j->Dreg + tid * 4;
    D[0] = d0; D[1] = d1; D[2] = d2; D[3] = d3;
}

static bool runS8() {
    const int M = 16, N = 8, K = 32;
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> di(-8, 7);
    std::vector<int8_t> A(M * K), B(K * N);
    std::vector<int>    C(M * N);
    for (auto &x : A) x = (int8_t)di(rng);
    for (auto &x : B) x = (int8_t)di(rng);
    for (size_t i = 0; i < C.size(); ++i)
        C[i] = (i & 1) ? std::numeric_limits<int>::min() + static_cast<int>(i)
                       : std::numeric_limits<int>::max() - static_cast<int>(i);

    std::vector<uint32_t> Areg(32 * 4), Breg(32 * 2);
    std::vector<int>      Creg(32 * 4), Dreg(32 * 4, 0);
    for (int lane = 0; lane < 32; ++lane) {
        int g = lane >> 2, t = lane & 3;
        auto Ae = [&](int m, int kc){ return A[m * K + kc]; };
        Areg[lane*4+0] = packs8(Ae(g,   4*t+0), Ae(g,   4*t+1), Ae(g,   4*t+2), Ae(g,   4*t+3));
        Areg[lane*4+1] = packs8(Ae(g+8, 4*t+0), Ae(g+8, 4*t+1), Ae(g+8, 4*t+2), Ae(g+8, 4*t+3));
        Areg[lane*4+2] = packs8(Ae(g,   4*t+16),Ae(g,   4*t+17),Ae(g,   4*t+18),Ae(g,   4*t+19));
        Areg[lane*4+3] = packs8(Ae(g+8, 4*t+16),Ae(g+8, 4*t+17),Ae(g+8, 4*t+18),Ae(g+8, 4*t+19));
        auto Be = [&](int kc, int n){ return B[kc * N + n]; };
        Breg[lane*2+0] = packs8(Be(4*t+0, g),  Be(4*t+1, g),  Be(4*t+2, g),  Be(4*t+3, g));
        Breg[lane*2+1] = packs8(Be(4*t+16, g), Be(4*t+17, g), Be(4*t+18, g), Be(4*t+19, g));
        Creg[lane*4+0] = C[(g)   * N + 2*t+0]; Creg[lane*4+1] = C[(g)   * N + 2*t+1];
        Creg[lane*4+2] = C[(g+8) * N + 2*t+0]; Creg[lane*4+3] = C[(g+8) * N + 2*t+1];
    }

    S8Job job{Areg.data(), Breg.data(), Creg.data(), Dreg.data()};
    std::memset(g_mma_scratch, 0, sizeof(g_mma_scratch));
    vgre_jit_block_dispatch(32, s8_lane, &job);

    std::vector<int> D(M * N, 0);
    for (int lane = 0; lane < 32; ++lane) {
        int g = lane >> 2, t = lane & 3;
        D[(g)   * N + 2*t+0] = Dreg[lane*4+0]; D[(g)   * N + 2*t+1] = Dreg[lane*4+1];
        D[(g+8) * N + 2*t+0] = Dreg[lane*4+2]; D[(g+8) * N + 2*t+1] = Dreg[lane*4+3];
    }
    int64_t maxErr = 0;
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            uint32_t bits;
            std::memcpy(&bits, &C[m * N + n], sizeof(bits));
            for (int kk = 0; kk < K; ++kk)
                bits += static_cast<uint32_t>(static_cast<int32_t>(A[m*K+kk])) *
                        static_cast<uint32_t>(static_cast<int32_t>(B[kk*N+n]));
            int32_t ref;
            std::memcpy(&ref, &bits, sizeof(ref));
            const int64_t err = static_cast<int64_t>(ref) - D[m * N + n];
            maxErr = std::max(maxErr, err < 0 ? -err : err);
        }
    printf("  [info] m16n8k32.s8 max abs err vs A*B+C = %lld (exact)\n",
           static_cast<long long>(maxErr));
    return maxErr == 0;
}

enum class M8IntOp { S4, U4, AndPopc, XorPopc };
struct M8IntJob {
    M8IntOp op;
    const uint32_t* Areg;
    const uint32_t* Breg;
    const int32_t* Creg;
    int32_t* Dreg;
};

static void m8int_lane(int tid, void* arg) {
    auto* job = static_cast<M8IntJob*>(arg);
    *vgre_jit_get_mma_buffer() = g_mma_scratch;
    *vgre_jit_get_threadIdx() = vgre::dim3(static_cast<uint32_t>(tid), 0, 0);
    *vgre_jit_get_blockDim() = vgre::dim3(64, 1, 1);
    const uint32_t a = job->Areg[tid], b = job->Breg[tid];
    const int32_t c0 = job->Creg[2 * tid], c1 = job->Creg[2 * tid + 1];
    int d0 = 0, d1 = 0;
    switch (job->op) {
    case M8IntOp::S4:       vgre_mma_m8n8k32_s4(d0, d1, a, b, c0, c1); break;
    case M8IntOp::U4:       vgre_mma_m8n8k32_u4(d0, d1, a, b, c0, c1); break;
    case M8IntOp::AndPopc:  vgre_mma_m8n8k128_b1_and(d0, d1, a, b, c0, c1); break;
    case M8IntOp::XorPopc:  vgre_mma_m8n8k128_b1_xor(d0, d1, a, b, c0, c1); break;
    }
    job->Dreg[2 * tid] = d0;
    job->Dreg[2 * tid + 1] = d1;
}

static bool runM8IntegerMma() {
    constexpr int kWarps = 2, kLanes = 32, kM = 8, kN = 8;
    constexpr int kThreads = kWarps * kLanes;
    for (M8IntOp op : {M8IntOp::S4, M8IntOp::U4, M8IntOp::AndPopc, M8IntOp::XorPopc}) {
        const bool binary = op == M8IntOp::AndPopc || op == M8IntOp::XorPopc;
        const bool signed4 = op == M8IntOp::S4;
        const int K = binary ? 128 : 32;
        std::vector<uint32_t> Areg(kThreads), Breg(kThreads);
        std::vector<int32_t> Creg(kThreads * 2), Dreg(kThreads * 2);
        std::vector<int32_t> A(kWarps * kM * K), B(kWarps * K * kN);
        std::vector<int32_t> C(kWarps * kM * kN), D(kWarps * kM * kN);

        for (int warp = 0; warp < kWarps; ++warp) {
            for (int m = 0; m < kM; ++m) {
                for (int k = 0; k < K; ++k) {
                    if (binary) {
                        A[warp*kM*K + m*K + k] = (m % 2 == 0) || ((k + m) % 3 == 0);
                    } else if (signed4) {
                        A[warp*kM*K + m*K + k] = ((m * 7 + k * 3 + warp) % 16) - 8;
                    } else {
                        A[warp*kM*K + m*K + k] = (m * 7 + k * 3 + warp) % 16;
                    }
                }
            }
            for (int k = 0; k < K; ++k) {
                for (int n = 0; n < kN; ++n) {
                    if (binary) {
                        B[warp*K*kN + k*kN + n] = (n % 2 == 0) || ((k + 2*n) % 5 == 0);
                    } else if (signed4) {
                        B[warp*K*kN + k*kN + n] = ((k * 5 + n * 3 + warp) % 16) - 8;
                    } else {
                        B[warp*K*kN + k*kN + n] = (k * 5 + n * 3 + warp) % 16;
                    }
                }
            }
            for (int m = 0; m < kM; ++m) {
                for (int n = 0; n < kN; ++n) {
                    const int index = warp*kM*kN + m*kN + n;
                    if (binary) C[index] = std::numeric_limits<int32_t>::max() - 32;
                    else C[index] = ((m + n) & 1)
                        ? std::numeric_limits<int32_t>::min() + 1000
                        : std::numeric_limits<int32_t>::max() - 1000;
                }
            }

            for (int lane = 0; lane < kLanes; ++lane) {
                const int group = lane >> 2, pair = lane & 3;
                const int tid = warp*kLanes + lane;
                uint32_t aw = 0, bw = 0;
                const int fragmentK = binary ? 32 : 8;
                for (int i = 0; i < fragmentK; ++i) {
                    const int k = pair*fragmentK + i;
                    const int av = A[warp*kM*K + group*K + k];
                    const int bv = B[warp*K*kN + k*kN + group];
                    if (binary) {
                        aw |= static_cast<uint32_t>(av) << i;
                        bw |= static_cast<uint32_t>(bv) << i;
                    } else {
                        aw |= (static_cast<uint32_t>(av) & 0xfu) << (4*i);
                        bw |= (static_cast<uint32_t>(bv) & 0xfu) << (4*i);
                    }
                }
                Areg[tid] = aw;
                Breg[tid] = bw;
                Creg[2*tid] = C[warp*kM*kN + group*kN + 2*pair];
                Creg[2*tid+1] = C[warp*kM*kN + group*kN + 2*pair + 1];
            }

            for (int m = 0; m < kM; ++m) {
                for (int n = 0; n < kN; ++n) {
                    const int out = warp*kM*kN + m*kN + n;
                    if (binary) {
                        uint32_t acc;
                        std::memcpy(&acc, &C[out], sizeof(acc));
                        for (int k = 0; k < K; ++k) {
                            const uint32_t av = static_cast<uint32_t>(A[warp*kM*K + m*K + k]);
                            const uint32_t bv = static_cast<uint32_t>(B[warp*K*kN + k*kN + n]);
                            acc += op == M8IntOp::AndPopc ? (av & bv) : (av ^ bv);
                        }
                        std::memcpy(&D[out], &acc, sizeof(acc));
                    } else {
                        int64_t acc = C[out];
                        for (int k = 0; k < K; ++k)
                            acc += static_cast<int64_t>(A[warp*kM*K + m*K + k]) *
                                   B[warp*K*kN + k*kN + n];
                        acc = std::max<int64_t>(std::numeric_limits<int32_t>::min(),
                                                std::min<int64_t>(std::numeric_limits<int32_t>::max(), acc));
                        D[out] = static_cast<int32_t>(acc);
                    }
                }
            }
        }

        M8IntJob job{op, Areg.data(), Breg.data(), Creg.data(), Dreg.data()};
        std::memset(g_mma_scratch, 0, sizeof(g_mma_scratch));
        vgre_jit_block_dispatch(kThreads, m8int_lane, &job);
        std::vector<int32_t> actual(kWarps*kM*kN);
        for (int tid = 0; tid < kThreads; ++tid) {
            const int warp = tid / kLanes, lane = tid % kLanes;
            const int group = lane >> 2, pair = lane & 3;
            actual[warp*kM*kN + group*kN + 2*pair] = Dreg[2*tid];
            actual[warp*kM*kN + group*kN + 2*pair + 1] = Dreg[2*tid + 1];
        }
        if (actual != D) {
            std::printf("  [FAIL] m8n8k%d integer warp result mismatch (op=%d)\n", K,
                        static_cast<int>(op));
            return false;
        }
    }
    return true;
}

struct F64Job {
    const double* Areg;
    const double* Breg;
    const double* Creg;
    double* Dreg;
};

static void f64_lane(int tid, void* arg) {
    auto* job = static_cast<F64Job*>(arg);
    *vgre_jit_get_mma_buffer() = g_mma_scratch;
    *vgre_jit_get_threadIdx() = vgre::dim3(static_cast<uint32_t>(tid), 0, 0);
    *vgre_jit_get_blockDim() = vgre::dim3(64, 1, 1);
    double d0, d1;
    vgre_mma_m8n8k4_f64(d0, d1, job->Areg[tid], job->Breg[tid],
                         job->Creg[2*tid], job->Creg[2*tid + 1]);
    job->Dreg[2*tid] = d0;
    job->Dreg[2*tid + 1] = d1;
}

static bool runM8F64Mma() {
    constexpr int kWarps = 2, kLanes = 32, kThreads = kWarps * kLanes;
    std::vector<double> Areg(kThreads), Breg(kThreads), Creg(kThreads*2), Dreg(kThreads*2);
    std::vector<double> A(kWarps*8*4), B(kWarps*4*8), C(kWarps*8*8), expected(kWarps*8*8);
    for (int warp = 0; warp < kWarps; ++warp) {
        for (int m = 0; m < 8; ++m)
            for (int k = 0; k < 4; ++k)
                A[warp*32 + m*4 + k] = (warp + 1) * (m - 3) + k * 0.5;
        for (int k = 0; k < 4; ++k)
            for (int n = 0; n < 8; ++n)
                B[warp*32 + k*8 + n] = (n - 2) * 0.25 + (k + warp) * 0.75;
        for (int m = 0; m < 8; ++m)
            for (int n = 0; n < 8; ++n)
                C[warp*64 + m*8 + n] = warp * 0.5 + m - n * 0.25;
        for (int m = 0; m < 8; ++m) {
            for (int n = 0; n < 8; ++n) {
                double sum = C[warp*64 + m*8 + n];
                for (int k = 0; k < 4; ++k)
                    sum += A[warp*32 + m*4 + k] * B[warp*32 + k*8 + n];
                expected[warp*64 + m*8 + n] = sum;
            }
        }
        for (int lane = 0; lane < kLanes; ++lane) {
            const int group = lane >> 2, pair = lane & 3, tid = warp*kLanes + lane;
            Areg[tid] = A[warp*32 + group*4 + pair];
            Breg[tid] = B[warp*32 + pair*8 + group];
            Creg[2*tid] = C[warp*64 + group*8 + 2*pair];
            Creg[2*tid + 1] = C[warp*64 + group*8 + 2*pair + 1];
        }
    }
    F64Job job{Areg.data(), Breg.data(), Creg.data(), Dreg.data()};
    std::memset(g_mma_scratch, 0, sizeof(g_mma_scratch));
    vgre_jit_block_dispatch(kThreads, f64_lane, &job);
    std::vector<double> actual(kWarps*64);
    for (int tid = 0; tid < kThreads; ++tid) {
        const int warp = tid / kLanes, lane = tid % kLanes;
        const int group = lane >> 2, pair = lane & 3;
        actual[warp*64 + group*8 + 2*pair] = Dreg[2*tid];
        actual[warp*64 + group*8 + 2*pair + 1] = Dreg[2*tid + 1];
    }
    return actual == expected;
}

// ── m16n8k32 FP8 warp-collective helpers ─────────────────────────────────────
using Fp8MmaFn = void (*)(float&, float&, float&, float&,
                          unsigned, unsigned, unsigned, unsigned,
                          unsigned, unsigned, float, float, float, float);
struct Fp8Job {
    const uint32_t *Areg, *Breg;
    const float *Creg;
    float *Dreg;
    Fp8MmaFn mma;
};

static void fp8_lane(int tid, void *arg) {
    auto *job = static_cast<Fp8Job *>(arg);
    *vgre_jit_get_mma_buffer() = g_mma_scratch;
    *vgre_jit_get_threadIdx() = vgre::dim3(static_cast<uint32_t>(tid), 0, 0);
    *vgre_jit_get_blockDim() = vgre::dim3(32, 1, 1);
    const uint32_t *a = job->Areg + tid * 4;
    const uint32_t *b = job->Breg + tid * 2;
    const float *c = job->Creg + tid * 4;
    float d0, d1, d2, d3;
    job->mma(d0, d1, d2, d3, a[0], a[1], a[2], a[3], b[0], b[1],
             c[0], c[1], c[2], c[3]);
    float *d = job->Dreg + tid * 4;
    d[0] = d0; d[1] = d1; d[2] = d2; d[3] = d3;
}

static bool runFp8(const char *name,
                   uint8_t (*encodeA)(float), uint8_t (*encodeB)(float),
                   float (*decodeA)(uint8_t), float (*decodeB)(uint8_t),
                   Fp8MmaFn mma) {
    constexpr int M = 16, N = 8, K = 32;
    std::mt19937 rng(20261007);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    std::vector<uint8_t> A(M * K), B(K * N);
    for (auto &x : A) x = encodeA(dist(rng));
    for (auto &x : B) x = encodeB(dist(rng));
    std::vector<float> C(M * N);
    for (auto &x : C) x = dist(rng);

    std::vector<uint32_t> Areg(32 * 4), Breg(32 * 2);
    std::vector<float> Creg(32 * 4), Dreg(32 * 4, 0.0f);
    for (int lane = 0; lane < 32; ++lane) {
        const int group = lane >> 2, pair = lane & 3;
        auto ae = [&](int row, int k) { return A[row * K + k]; };
        Areg[lane * 4 + 0] = pack4fp8(ae(group,     4*pair), ae(group,     4*pair+1),
                                     ae(group,     4*pair+2), ae(group,     4*pair+3));
        Areg[lane * 4 + 1] = pack4fp8(ae(group + 8, 4*pair), ae(group + 8, 4*pair+1),
                                     ae(group + 8, 4*pair+2), ae(group + 8, 4*pair+3));
        Areg[lane * 4 + 2] = pack4fp8(ae(group,     4*pair+16), ae(group,     4*pair+17),
                                     ae(group,     4*pair+18), ae(group,     4*pair+19));
        Areg[lane * 4 + 3] = pack4fp8(ae(group + 8, 4*pair+16), ae(group + 8, 4*pair+17),
                                     ae(group + 8, 4*pair+18), ae(group + 8, 4*pair+19));
        auto be = [&](int k, int col) { return B[k * N + col]; };
        Breg[lane * 2 + 0] = pack4fp8(be(4*pair, group), be(4*pair+1, group),
                                     be(4*pair+2, group), be(4*pair+3, group));
        Breg[lane * 2 + 1] = pack4fp8(be(4*pair+16, group), be(4*pair+17, group),
                                     be(4*pair+18, group), be(4*pair+19, group));
        Creg[lane * 4 + 0] = C[group * N + 2*pair];
        Creg[lane * 4 + 1] = C[group * N + 2*pair + 1];
        Creg[lane * 4 + 2] = C[(group + 8) * N + 2*pair];
        Creg[lane * 4 + 3] = C[(group + 8) * N + 2*pair + 1];
    }

    Fp8Job job{Areg.data(), Breg.data(), Creg.data(), Dreg.data(), mma};
    std::memset(g_mma_scratch, 0, sizeof(g_mma_scratch));
    vgre_jit_block_dispatch(32, fp8_lane, &job);

    std::vector<float> D(M * N, 0.0f);
    for (int lane = 0; lane < 32; ++lane) {
        const int group = lane >> 2, pair = lane & 3;
        D[group * N + 2*pair] = Dreg[lane * 4 + 0];
        D[group * N + 2*pair + 1] = Dreg[lane * 4 + 1];
        D[(group + 8) * N + 2*pair] = Dreg[lane * 4 + 2];
        D[(group + 8) * N + 2*pair + 1] = Dreg[lane * 4 + 3];
    }

    double maxErr = 0.0;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            double ref = C[m * N + n];
            double absProducts = 0.0;
            for (int k = 0; k < K; ++k) {
                const double product = static_cast<double>(decodeA(A[m*K+k])) *
                                       static_cast<double>(decodeB(B[k*N+n]));
                ref += product;
                absProducts += std::fabs(product);
            }
            maxErr = std::max(maxErr, std::fabs(static_cast<double>(D[m*N+n]) - ref));
            const double bound = 4.0 * K * std::numeric_limits<float>::epsilon() * absProducts + 2e-6;
            if (std::fabs(static_cast<double>(D[m*N+n]) - ref) > bound) {
                std::printf("  [FAIL] %s at (%d,%d): got %.8g ref %.8g bound %.3g\n",
                            name, m, n, D[m*N+n], ref, bound);
                return false;
            }
        }
    }
    std::printf("  [info] %s max abs error vs independent decoded-matrix reference = %.3g\n",
                name, maxErr);
    return true;
}

// ── TMA store (shared → global) — the JIT-path helpers in wmma_emulation.h ────
// cp.async.bulk.tensor.2d.global.shared::cta.bulk_group lowers to
// vgre_tma_store_2d_b. Validates the interior box scatter, out-of-bounds clipping
// (real TMA store drops OOB elements — must not overrun the tensor), and that the
// store is the exact inverse of the load (load→store reproduces the tensor).
static void runTmaStore() {
    const int H = 8, W = 8, bh = 4, bw = 4, GUARD = 4;
    std::vector<float> G(H * W + GUARD, -1.0f);
    for (int i = 0; i < GUARD; ++i) G[H * W + i] = 7777.0f;   // guard cells past the tensor

    VgreTMADescriptor d{};
    d.baseAddr  = G.data();
    d.elemBytes = sizeof(float);
    d.dim[0] = W; d.dim[1] = H;                                // global bounds
    d.stride[0] = (uint32_t)(W * sizeof(float));               // row byte stride (2d convention)
    d.boxDim[0] = bw; d.boxDim[1] = bh;

    // 1) Interior store lands the box at (col=2,row=1); other cells untouched.
    float tile[bh * bw];
    for (int i = 0; i < bh * bw; ++i) tile[i] = 100.0f + i;
    vgre_tma_store_2d_b(&d, tile, /*col*/2, /*row*/1);
    bool ok1 = (G[0] == -1.0f);
    for (int i = 0; i < bh; ++i)
        for (int j = 0; j < bw; ++j)
            if (G[(1 + i) * W + (2 + j)] != tile[i * bw + j]) ok1 = false;
    check("TMA 2d store writes the interior box to global", ok1);

    // 2) Box overhanging the tensor at (col=6,row=6): only [6..7]x[6..7] valid; the
    //    OOB quadrant must be dropped and the guard cells must be intact.
    std::fill(G.begin(), G.begin() + H * W, -1.0f);
    for (int i = 0; i < bh * bw; ++i) tile[i] = 500.0f + i;
    vgre_tma_store_2d_b(&d, tile, 6, 6);
    bool ok2 = true;
    for (int i = 0; i < bh; ++i)
        for (int j = 0; j < bw; ++j) {
            int gr = 6 + i, gc = 6 + j;
            if (gr < H && gc < W && G[gr * W + gc] != tile[i * bw + j]) ok2 = false;
        }
    for (int i = 0; i < GUARD; ++i) if (G[H * W + i] != 7777.0f) ok2 = false;   // no overrun
    check("TMA 2d store clips out-of-bounds (no overrun, real TMA padding)", ok2);

    // 3) load→store round-trip reproduces the whole tensor exactly.
    std::vector<float> src(H * W), dstG(H * W, 0.0f);
    for (int i = 0; i < H * W; ++i) src[i] = (float)i * 0.5f - 3.0f;
    VgreTMADescriptor ds = d; ds.baseAddr = src.data();
    VgreTMADescriptor dd = d; dd.baseAddr = dstG.data();
    float box[bh * bw];
    for (int y = 0; y < H; y += bh)
        for (int x = 0; x < W; x += bw) {
            vgre_tma_load_2d_b(box, &ds, x, y);
            vgre_tma_store_2d_b(&dd, box, x, y);
        }
    bool ok3 = true;
    for (int i = 0; i < H * W; ++i) if (dstG[i] != src[i]) ok3 = false;
    check("TMA load->store round-trip reproduces the tensor exactly", ok3);
}

int main() {
    printf("=== Tensor-core mma.sync warp-collective correctness (Track 9) ===\n");
    check("m16n8k16 f16*f16->f32 matches A*B+C reference", runF16());
    check("m16n8k32 s8*s8->s32 matches A*B+C reference (exact)", runS8());
    check("m8n8k32 s4/u4 and m8n8k128 b1 full two-warp matrix references",
          runM8IntegerMma());
    check("m8n8k4 f64 full two-warp matrix reference", runM8F64Mma());
    check("m16n8k32 e4m3*e4m3->f32 full warp reference",
          runFp8("FP8 E4M3×E4M3", vgre_f32_to_fp8e4m3, vgre_f32_to_fp8e4m3,
                 vgre_fp8e4m3_to_f32, vgre_fp8e4m3_to_f32,
                 vgre_mma_m16n8k32_f32_e4m3));
    check("m16n8k32 e5m2*e5m2->f32 full warp reference",
          runFp8("FP8 E5M2×E5M2", vgre_f32_to_fp8e5m2, vgre_f32_to_fp8e5m2,
                 vgre_fp8e5m2_to_f32, vgre_fp8e5m2_to_f32,
                 vgre_mma_m16n8k32_f32_e5m2));
    check("m16n8k32 e4m3*e5m2->f32 full warp reference",
          runFp8("FP8 E4M3×E5M2", vgre_f32_to_fp8e4m3, vgre_f32_to_fp8e5m2,
                 vgre_fp8e4m3_to_f32, vgre_fp8e5m2_to_f32,
                 vgre_mma_m16n8k32_f32_e4m3e5m2));
    check("m16n8k32 e5m2*e4m3->f32 full warp reference",
          runFp8("FP8 E5M2×E4M3", vgre_f32_to_fp8e5m2, vgre_f32_to_fp8e4m3,
                 vgre_fp8e5m2_to_f32, vgre_fp8e4m3_to_f32,
                 vgre_mma_m16n8k32_f32_e5m2e4m3));
    runTmaStore();
    printf("\n%d / %d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
