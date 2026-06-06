// cuDNN CTC Loss correctness test.
// Validates: terminal-state constraint in backward pass (the bug we fixed),
// forward loss value, and gradient sum-to-zero property.
//
// Uses inline type stubs matching the existing test pattern (no cudnn.h required).

#include <cstdio>
#include <cassert>
#include <cmath>
#include <vector>
#include <limits>
#include <algorithm>
#include <cstring>

extern "C" {

typedef int cudnnStatus_t;
#define CUDNN_STATUS_SUCCESS 0
#define CUDNN_STATUS_INVALID_VALUE 8

typedef void* cudnnHandle_t;

// Minimal TensorDesc that matches the internal layout used by cudnn_ctc_loss.cpp
struct TensorDesc4D { int n, c, h, w, format; };

cudnnStatus_t cudnnCreate(cudnnHandle_t* h);
cudnnStatus_t cudnnDestroy(cudnnHandle_t h);
cudnnStatus_t cudnnCreateCTCLossDescriptor(void** ctcDesc);
cudnnStatus_t cudnnDestroyCTCLossDescriptor(void* ctcDesc);
cudnnStatus_t cudnnSetCTCLossDescriptor(void* ctcDesc, int blankLabel);
cudnnStatus_t cudnnCTCLoss(
    cudnnHandle_t handle,
    const void* probsDesc, const void* probs,
    const int* labels, const int* labelLengths, const int* inputLengths,
    void* ctcDesc, void* workspace, size_t workSpaceSizeInBytes,
    void* costs, const void* gradientsDesc, void* gradients);

} // extern "C"

static bool approx(float a, float b, float tol = 1e-3f) {
    return std::fabs(a - b) <= tol + tol * std::fabs(b);
}

int main() {
    printf("=== CTC Loss Tests ===\n");

    cudnnHandle_t handle;
    cudnnCreate(&handle);

    // ── Test 1: Deterministic forward loss check ──────────────────────────────
    // T=2, N=1, C=2, blank=0, labels=[1]
    // probs[t=0]: blank=0.8, label=0.2
    // probs[t=1]: blank=0.3, label=0.7
    // Valid paths (decode to "1"):
    //   "0,1" → 0.8*0.7 = 0.56
    //   "1,0" → 0.2*0.3 = 0.06
    //   "1,1" → 0.2*0.7 = 0.14
    // P("1") = 0.76; loss = -log(0.76) ≈ 0.2744
    {
        const int T=2, N=1, C=2;
        // Layout [T][N][C] = [t*N*C + b*C + c]
        float probs[4] = {0.8f, 0.2f, 0.3f, 0.7f};

        void* ctcDesc;
        cudnnCreateCTCLossDescriptor(&ctcDesc);
        cudnnSetCTCLossDescriptor(ctcDesc, 0);

        TensorDesc4D probsDesc = {T, N, C, 1, 0};
        TensorDesc4D gradsDesc = {T, N, C, 1, 0};

        int labels[1]    = {1};
        int labelLens[1] = {1};
        int inputLens[1] = {T};

        float costs[1] = {0.f};
        float grads[4] = {};

        cudnnStatus_t st = cudnnCTCLoss(handle, &probsDesc, probs,
                                        labels, labelLens, inputLens,
                                        ctcDesc, nullptr, 0,
                                        costs, &gradsDesc, grads);
        assert(st == CUDNN_STATUS_SUCCESS);

        float expected_loss = -std::log(0.76f);
        printf("  loss=%.5f expected=%.5f err=%.2e\n",
               costs[0], expected_loss, std::fabs(costs[0]-expected_loss));
        assert(approx(costs[0], expected_loss) && "CTC forward loss wrong");

        // Gradient sum over classes at each timestep = 0
        for (int t = 0; t < T; ++t) {
            float sg = grads[t*C+0] + grads[t*C+1];
            assert(std::fabs(sg) < 5e-4f && "CTC gradient sum != 0");
        }
        assert(std::isfinite(grads[0]) && std::isfinite(grads[3]) && "CTC gradient NaN");

        cudnnDestroyCTCLossDescriptor(ctcDesc);
        printf("  [PASS] CTC T=2 deterministic: loss=%.4f, gradients finite\n", costs[0]);
    }

    // ── Test 2: Uniform probs, T=3 ───────────────────────────────────────────
    // T=3, N=1, C=2, blank=0, labels=[1], all probs=0.5
    // Valid alignments (decode to "1"): count manually
    //   001→"1"✓, 010→"1"✓, 011→"1"✓, 100→"1"✓, 110→"1"✓, 111→"1"✓
    //   101: de-dup=[1,0,1]→no consecutive dups→remove blanks=[1,1]="11"✗
    //   000: → ""✗
    //   6 paths * (0.5)^3 = 0.75; loss = -log(0.75) ≈ 0.2877
    {
        const int T=3, N=1, C=2;
        std::vector<float> probs(T*N*C, 0.5f);

        void* ctcDesc;
        cudnnCreateCTCLossDescriptor(&ctcDesc);
        cudnnSetCTCLossDescriptor(ctcDesc, 0);

        TensorDesc4D probsDesc = {T, N, C, 1, 0};
        TensorDesc4D gradsDesc = {T, N, C, 1, 0};

        int labels[1]    = {1};
        int labelLens[1] = {1};
        int inputLens[1] = {T};

        float costs[1] = {0.f};
        std::vector<float> grads(T*N*C, 0.f);

        cudnnStatus_t st = cudnnCTCLoss(handle, &probsDesc, probs.data(),
                                        labels, labelLens, inputLens,
                                        ctcDesc, nullptr, 0,
                                        costs, &gradsDesc, grads.data());
        assert(st == CUDNN_STATUS_SUCCESS);

        float expected_loss = -std::log(0.75f);
        printf("  loss=%.5f expected=%.5f err=%.2e\n",
               costs[0], expected_loss, std::fabs(costs[0]-expected_loss));
        assert(approx(costs[0], expected_loss) && "CTC T=3 loss wrong");

        // Verify terminal-state bug is fixed: at t=T-1, only the two valid terminal
        // states contribute to the gradient.  With the bug, Z[T-1] was inflated by
        // the invalid state alpha[T-1,0] (the first blank), making the loss correct
        // but the gradient wrong.
        for (int t = 0; t < T; ++t) {
            float sg = grads[t*C+0] + grads[t*C+1];
            assert(std::fabs(sg) < 5e-4f && "CTC gradient sum != 0");
        }

        cudnnDestroyCTCLossDescriptor(ctcDesc);
        printf("  [PASS] CTC T=3 uniform: loss=%.4f, terminal-state constraint OK\n", costs[0]);
    }

    cudnnDestroy(handle);
    printf("ALL CTC Loss tests PASSED.\n");
    return 0;
}
