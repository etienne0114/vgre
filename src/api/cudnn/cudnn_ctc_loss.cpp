// cuDNN CTC Loss — reference CPU implementation.
//
// Supports forward CTC loss and gradient computation for sequences.
// This is a functional reference intended for correctness testing.
// For production speed with long sequences, link against a GPU-backed
// cuDNN or use an optimized CTC library (e.g., warp-ctc).

#include "cudnn_internal.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>

// ── CTC descriptor ───────────────────────────────────────────────────────────
struct CTCLossDesc {
    int blankLabel = 0;
    int gradMode = 0; // CUDNN_CTC_ZERO_DUPLICATE = 0
};

// ── Internal helpers ─────────────────────────────────────────────────────────

namespace {

// Log-sum-exp for numerical stability
inline float logSumExp(float a, float b) {
    if (a == -std::numeric_limits<float>::infinity()) return b;
    if (b == -std::numeric_limits<float>::infinity()) return a;
    float m = std::max(a, b);
    return m + std::log(std::exp(a - m) + std::exp(b - m));
}

// Extended label sequence with blanks: e.g. "ab" -> _ a _ b _
// Returns length = 2*L + 1
std::vector<int> extendLabels(const int* labels, int L, int blank) {
    std::vector<int> ext(2 * L + 1);
    for (int i = 0; i <= L; ++i) {
        ext[2 * i] = blank;
        if (i < L) ext[2 * i + 1] = labels[i];
    }
    return ext;
}

// Forward CTC (alpha) variables — log domain
// probs: [T][C]  row-major
// extLabels: extended label sequence, length S = 2*L+1
// alpha: output [T][S]
void ctcForward(int T, int C, const float* probs,
                const std::vector<int>& extLabels, int S,
                float* alpha, float* lossOut) {
    const float negInf = -std::numeric_limits<float>::infinity();

    // t = 0
    alpha[0 * S + 0] = std::log(probs[0 * C + extLabels[0]]);
    if (S > 1)
        alpha[0 * S + 1] = std::log(probs[0 * C + extLabels[1]]);
    for (int s = 2; s < S; ++s)
        alpha[0 * S + s] = negInf;

    // t > 0
    for (int t = 1; t < T; ++t) {
        for (int s = 0; s < S; ++s) {
            float prev = alpha[(t - 1) * S + s];
            // Stay at same state
            float best = prev;
            // Move from previous state
            if (s > 0) {
                best = logSumExp(best, alpha[(t - 1) * S + (s - 1)]);
            }
            // Skip blank (s >= 2 and not same label as s-2)
            if (s >= 2 && extLabels[s] != extLabels[s - 2]) {
                best = logSumExp(best, alpha[(t - 1) * S + (s - 2)]);
            }
            alpha[t * S + s] = best + std::log(probs[t * C + extLabels[s]]);
        }
    }

    // Loss = -log(sum of final alpha over valid end states)
    float finalSum = alpha[(T - 1) * S + (S - 1)];
    if (S > 1)
        finalSum = logSumExp(finalSum, alpha[(T - 1) * S + (S - 2)]);
    *lossOut = -finalSum;
}

// Backward CTC (beta) variables — log domain
// beta: [T][S]
void ctcBackward(int T, int C, const float* probs,
                 const std::vector<int>& extLabels, int S,
                 float* beta) {
    const float negInf = -std::numeric_limits<float>::infinity();

    // t = T-1
    for (int s = 0; s < S; ++s)
        beta[(T - 1) * S + s] = 0.f; // log(1) = 0

    // t < T-1
    for (int t = T - 2; t >= 0; --t) {
        for (int s = 0; s < S; ++s) {
            float best = beta[(t + 1) * S + s] + std::log(probs[(t + 1) * C + extLabels[s]]);
            if (s + 1 < S) {
                best = logSumExp(best, beta[(t + 1) * S + (s + 1)] + std::log(probs[(t + 1) * C + extLabels[s + 1]]));
            }
            if (s + 2 < S && extLabels[s + 2] != extLabels[s]) {
                best = logSumExp(best, beta[(t + 1) * S + (s + 2)] + std::log(probs[(t + 1) * C + extLabels[s + 2]]));
            }
            beta[t * S + s] = best;
        }
    }
}

} // namespace

// ── Public cuDNN CTC API ────────────────────────────────────────────────────

extern "C" {

cudnnStatus_t cudnnCreateCTCLossDescriptor(void** ctcDesc) {
    if (!ctcDesc) return CUDNN_STATUS_INVALID_VALUE;
    *ctcDesc = new CTCLossDesc();
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyCTCLossDescriptor(void* ctcDesc) {
    delete static_cast<CTCLossDesc*>(ctcDesc);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetCTCLossDescriptor(void* ctcDesc, int blankLabel) {
    if (!ctcDesc) return CUDNN_STATUS_INVALID_VALUE;
    static_cast<CTCLossDesc*>(ctcDesc)->blankLabel = blankLabel;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetCTCLossDescriptor(void* ctcDesc, int* blankLabel) {
    if (!ctcDesc || !blankLabel) return CUDNN_STATUS_INVALID_VALUE;
    *blankLabel = static_cast<CTCLossDesc*>(ctcDesc)->blankLabel;
    return CUDNN_STATUS_SUCCESS;
}

// Extended descriptor (gradMode)
cudnnStatus_t cudnnSetCTCLossDescriptorEx(void* ctcDesc, int blankLabel,
                                          int /*gradMode*/, int /*paddingMode*/) {
    if (!ctcDesc) return CUDNN_STATUS_INVALID_VALUE;
    static_cast<CTCLossDesc*>(ctcDesc)->blankLabel = blankLabel;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetCTCLossDescriptorEx(void* ctcDesc, int* blankLabel,
                                          int* /*gradMode*/, int* /*paddingMode*/) {
    if (!ctcDesc || !blankLabel) return CUDNN_STATUS_INVALID_VALUE;
    *blankLabel = static_cast<CTCLossDesc*>(ctcDesc)->blankLabel;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetCTCLossWorkspaceSize(
    cudnnHandle_t /*handle*/,
    const void* /*probsDesc*/,
    const void* /*gradientsDesc*/,
    const int* /*labels*/,
    const int* /*labelLengths*/,
    const int* /*inputLengths*/,
    void* ctcDesc,
    size_t* sizeInBytes)
{
    if (!sizeInBytes || !ctcDesc) return CUDNN_STATUS_INVALID_VALUE;
    // Workspace is not used in our reference implementation (allocs internally)
    *sizeInBytes = 0;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCTCLoss(
    cudnnHandle_t /*handle*/,
    const void* probsDesc,
    const void* probs,
    const int* labels,
    const int* labelLengths,
    const int* inputLengths,
    void* ctcDesc,
    void* /*workspace*/,
    size_t /*workSpaceSizeInBytes*/,
    void* costs,
    const void* gradientsDesc,
    void* gradients)
{
    if (!probsDesc || !probs || !labels || !labelLengths || !inputLengths ||
        !ctcDesc || !costs || !gradientsDesc || !gradients)
        return CUDNN_STATUS_INVALID_VALUE;

    const CTCLossDesc& desc = *static_cast<CTCLossDesc*>(ctcDesc);
    const TensorDesc& td = *static_cast<const TensorDesc*>(probsDesc);
    const TensorDesc& gd = *static_cast<const TensorDesc*>(gradientsDesc);
    (void)gd; // gradient descriptor has same shape as probs

    // probsDesc is [T, N, C] or [N, T, C] depending on format.
    // For simplicity, assume [T, N, C] layout (batch = N, time = T, classes = C).
    int T = td.n; // time steps
    int N = td.c; // batch size
    int C = td.h * td.w; // number of classes (flatten h*w)
    if (C == 0) C = td.h;

    float* costsF = static_cast<float*>(costs);
    float* gradsF = static_cast<float*>(gradients);
    const float* probsF = static_cast<const float*>(probs);

    // Process each batch item
    for (int b = 0; b < N; ++b) {
        int L = labelLengths[b];
        int Tb = inputLengths[b];
        if (Tb <= 0 || L < 0 || L > Tb) {
            costsF[b] = 0.f;
            continue;
        }

        const int* batchLabels = labels + b * L; // labels packed per batch
        std::vector<int> extLabels = extendLabels(batchLabels, L, desc.blankLabel);
        int S = static_cast<int>(extLabels.size());

        // Extract probabilities for this batch item: [Tb][C]
        std::vector<float> batchProbs(Tb * C);
        for (int t = 0; t < Tb; ++t)
            for (int c = 0; c < C; ++c)
                batchProbs[t * C + c] = probsF[((t * N + b) * C + c)];

        // Forward (alpha)
        std::vector<float> alpha(Tb * S);
        float loss = 0.f;
        ctcForward(Tb, C, batchProbs.data(), extLabels, S, alpha.data(), &loss);
        costsF[b] = loss;

        // Backward (beta)
        std::vector<float> beta(Tb * S);
        ctcBackward(Tb, C, batchProbs.data(), extLabels, S, beta.data());

        // Compute gradient
        // For each time t and class c:
        //   grad[t][c] = y[t][c] - sum_{s: extLabels[s]==c} (alpha[t][s]*beta[t][s]) / Z[t]
        // where Z[t] = sum_s alpha[t][s]*beta[t][s]
        for (int t = 0; t < Tb; ++t) {
            // Compute Z[t] in log domain then convert
            float logZ = -std::numeric_limits<float>::infinity();
            for (int s = 0; s < S; ++s) {
                logZ = logSumExp(logZ, alpha[t * S + s] + beta[t * S + s]);
            }
            float Z = std::exp(logZ);

            // Accumulate per-class numerator
            std::vector<float> numer(C, 0.f);
            for (int s = 0; s < S; ++s) {
                int c = extLabels[s];
                numer[c] += std::exp(alpha[t * S + s] + beta[t * S + s]);
            }

            for (int c = 0; c < C; ++c) {
                float y = batchProbs[t * C + c];
                float grad = y - numer[c] / Z;
                gradsF[((t * N + b) * C + c)] = grad;
            }
        }
    }

    return CUDNN_STATUS_SUCCESS;
}

} // extern "C"
