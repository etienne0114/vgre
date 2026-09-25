#ifndef VGRE_CORE_SSM_H
#define VGRE_CORE_SSM_H

// Phase 3, Track P3-12 — Mamba / state-space-model selective scan.
//
// SSMs (Mamba-2, Jamba) replace attention with a *selective scan* — a linear
// recurrence with input-dependent coefficients. For one channel with N state
// dims over a length-L sequence (already discretized):
//     h_t[n] = Ā_t[n] · h_{t-1}[n] + B̄x_t[n],   y_t = Σ_n C_t[n] · h_t[n]
// The recurrence h_t = a_t·h_{t-1} + b_t is an *associative* operation:
//     (a₁,b₁) ∘ (a₂,b₂) = (a₁·a₂,  a₂·b₁ + b₂)   (segment 1 then segment 2),
// so the whole sequence parallelizes with a prefix scan over that monoid (the
// Mamba parallel-scan trick) — O(L log L) work, O(log L) depth — instead of the
// O(L) sequential loop. This module provides both; they must agree.

#include <cstddef>
#include <vector>

namespace vgre {
namespace ssm {

// Sequential reference recurrence. Abar/Bx/C are [L,N] row-major; y is [L].
inline void selective_scan_seq(float* y, const float* Abar, const float* Bx,
                               const float* C, int L, int N) {
    std::vector<float> h(static_cast<size_t>(N), 0.0f);
    for (int t = 0; t < L; ++t) {
        float yt = 0.0f;
        for (int n = 0; n < N; ++n) {
            h[n] = Abar[static_cast<size_t>(t) * N + n] * h[n] +
                   Bx[static_cast<size_t>(t) * N + n];
            yt += C[static_cast<size_t>(t) * N + n] * h[n];
        }
        y[t] = yt;
    }
}

// Parallel selective scan: per state dim n, an inclusive Hillis-Steele scan over
// the linear-recurrence monoid yields h_t[n] (= the cumulative b, with h_0 = 0);
// then y_t = Σ_n C_t[n]·h_t[n]. Same result as the sequential recurrence.
inline void selective_scan_parallel(float* y, const float* Abar, const float* Bx,
                                    const float* C, int L, int N) {
    std::vector<float> H(static_cast<size_t>(L) * N, 0.0f);
    std::vector<float> a(L), b(L), na(L), nb(L);
    for (int n = 0; n < N; ++n) {
        for (int t = 0; t < L; ++t) {
            a[t] = Abar[static_cast<size_t>(t) * N + n];
            b[t] = Bx[static_cast<size_t>(t) * N + n];
        }
        // Inclusive scan over (a,b): combine earlier [..t-d] with later [t-d+1..t].
        for (int d = 1; d < L; d <<= 1) {
            for (int t = 0; t < L; ++t) {
                if (t >= d) { na[t] = a[t] * a[t - d];          // a_later · a_earlier
                              nb[t] = a[t] * b[t - d] + b[t]; }  // a_later · b_earlier + b_later
                else        { na[t] = a[t]; nb[t] = b[t]; }
            }
            a.swap(na); b.swap(nb);
        }
        for (int t = 0; t < L; ++t) H[static_cast<size_t>(t) * N + n] = b[t];  // h_t (h_0 = 0)
    }
    for (int t = 0; t < L; ++t) {
        float yt = 0.0f;
        for (int n = 0; n < N; ++n) yt += C[static_cast<size_t>(t) * N + n] *
                                          H[static_cast<size_t>(t) * N + n];
        y[t] = yt;
    }
}

// ── Mamba-3 MIMO (matrix state) selective scan ───────────────────────────────
// Mamba-2/3's SSD form carries a MATRIX state H ∈ R^{N×P} per channel (N state
// dims × P head channels) instead of a vector. Each step adds the outer product of
// the input projection B_t ∈ R^N and the P-dim input x_t, decays by the diagonal
// Ā_t ∈ R^N (shared across the P columns), and reads out with C_t ∈ R^N:
//     H_t[n,p] = Ā_t[n]·H_{t-1}[n,p] + B_t[n]·x_t[p],   y_t[p] = Σ_n C_t[n]·H_t[n,p]
// This is the multi-input / multi-output ("matrix-matrix") update; P=1 with
// B_t[n]=B̄x_t[n]/x_t is exactly the SISO scan above. Abar/B/C are [L,N] row-major,
// x and y are [L,P] row-major.
//
// Sequential reference recurrence.
inline void selective_scan_mimo_seq(float* y, const float* Abar, const float* B,
                                    const float* x, const float* C,
                                    int L, int N, int P) {
    std::vector<float> H(static_cast<size_t>(N) * P, 0.0f);   // matrix state, h_0 = 0
    for (int t = 0; t < L; ++t) {
        const float* at = Abar + static_cast<size_t>(t) * N;
        const float* bt = B    + static_cast<size_t>(t) * N;
        const float* ct = C    + static_cast<size_t>(t) * N;
        const float* xt = x    + static_cast<size_t>(t) * P;
        float* yt = y + static_cast<size_t>(t) * P;
        for (int p = 0; p < P; ++p) yt[p] = 0.0f;
        for (int n = 0; n < N; ++n) {
            float* Hn = &H[static_cast<size_t>(n) * P];
            const float a = at[n], b = bt[n], c = ct[n];
            for (int p = 0; p < P; ++p) {
                Hn[p] = a * Hn[p] + b * xt[p];
                yt[p] += c * Hn[p];
            }
        }
    }
}

// Parallel MIMO scan. For each state dim n the decay a_t=Ā_t[n] is shared across
// the P columns, so the state factorizes into P independent scalar linear
// recurrences over the monoid (a, b) with per-column input b_t^{(p)} = B_t[n]·x_t[p]:
// scan a once and carry the P-vector b through the same Hillis-Steele combine.
inline void selective_scan_mimo_parallel(float* y, const float* Abar, const float* B,
                                         const float* x, const float* C,
                                         int L, int N, int P) {
    for (int t = 0; t < L; ++t)
        for (int p = 0; p < P; ++p) y[static_cast<size_t>(t) * P + p] = 0.0f;
    std::vector<float> a(L), na(L);
    std::vector<float> b(static_cast<size_t>(L) * P), nb(static_cast<size_t>(L) * P);
    for (int n = 0; n < N; ++n) {
        for (int t = 0; t < L; ++t) {
            a[t] = Abar[static_cast<size_t>(t) * N + n];
            const float bn = B[static_cast<size_t>(t) * N + n];
            const float* xt = x + static_cast<size_t>(t) * P;
            float* bt = &b[static_cast<size_t>(t) * P];
            for (int p = 0; p < P; ++p) bt[p] = bn * xt[p];
        }
        // Inclusive scan over (a, b): (a_e,b_e)∘(a_l,b_l) = (a_l·a_e, a_l·b_e + b_l).
        for (int d = 1; d < L; d <<= 1) {
            for (int t = 0; t < L; ++t) {
                float* nbt = &nb[static_cast<size_t>(t) * P];
                const float* bt = &b[static_cast<size_t>(t) * P];
                if (t >= d) {
                    na[t] = a[t] * a[t - d];
                    const float al = a[t];
                    const float* be = &b[static_cast<size_t>(t - d) * P];
                    for (int p = 0; p < P; ++p) nbt[p] = al * be[p] + bt[p];
                } else {
                    na[t] = a[t];
                    for (int p = 0; p < P; ++p) nbt[p] = bt[p];
                }
            }
            a.swap(na); b.swap(nb);
        }
        // b[t] now holds H_t[n,·]; accumulate the readout y_t += C_t[n]·H_t[n,·].
        for (int t = 0; t < L; ++t) {
            const float c = C[static_cast<size_t>(t) * N + n];
            const float* bt = &b[static_cast<size_t>(t) * P];
            float* yt = y + static_cast<size_t>(t) * P;
            for (int p = 0; p < P; ++p) yt[p] += c * bt[p];
        }
    }
}

} // namespace ssm
} // namespace vgre

#endif // VGRE_CORE_SSM_H
