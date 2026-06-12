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

} // namespace ssm
} // namespace vgre

#endif // VGRE_CORE_SSM_H
