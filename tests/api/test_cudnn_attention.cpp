/**
 * Test: cuDNN Multi-Head Attention Forward/Backward (P3.10)
 *
 * Verifies cudnnMultiHeadAttnForward/BackwardData/BackwardWeights
 * for a simplified scalar self-attention reference.
 */

#include <iostream>
#include <cmath>
#include <cstring>
#include <vector>

extern "C" {
typedef int cudnnStatus_t;
#define CUDNN_STATUS_SUCCESS 0
#define CUDNN_STATUS_INVALID_VALUE 8

typedef void* cudnnHandle_t;
typedef void* cudnnTensorDescriptor_t;
typedef void* cudnnMultiHeadAttnDescriptor_t;

enum cudnnDataType_t { CUDNN_DATA_FLOAT = 0 };
enum cudnnTensorFormat_t { CUDNN_TENSOR_NCHW = 0 };

cudnnStatus_t cudnnCreate(cudnnHandle_t* handle);
cudnnStatus_t cudnnDestroy(cudnnHandle_t handle);

cudnnStatus_t cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t* d);
cudnnStatus_t cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t d);
cudnnStatus_t cudnnSetTensor4dDescriptor(cudnnTensorDescriptor_t d,
    cudnnTensorFormat_t fmt, cudnnDataType_t dtype, int n, int c, int h, int w);

cudnnStatus_t cudnnCreateMultiHeadAttnDescriptor(cudnnMultiHeadAttnDescriptor_t* d);
cudnnStatus_t cudnnDestroyMultiHeadAttnDescriptor(cudnnMultiHeadAttnDescriptor_t d);
cudnnStatus_t cudnnSetMultiHeadAttnDescriptor(
    cudnnMultiHeadAttnDescriptor_t desc, int mode,
    int numHeads, int headDim, int modelDim,
    int seqLen, int batchSize, float scaling);

cudnnStatus_t cudnnMultiHeadAttnForward(
    cudnnHandle_t, cudnnMultiHeadAttnDescriptor_t attnDesc,
    int seqLen, int batchSize, const void* weights,
    cudnnTensorDescriptor_t qDesc, const void* q,
    cudnnTensorDescriptor_t kDesc, const void* k,
    cudnnTensorDescriptor_t vDesc, const void* v,
    cudnnTensorDescriptor_t oDesc, void* o,
    void* workspace, size_t workspaceSizeInBytes);

cudnnStatus_t cudnnMultiHeadAttnBackwardData(
    cudnnHandle_t, cudnnMultiHeadAttnDescriptor_t attnDesc,
    int seqLen, int batchSize, const void* weights,
    cudnnTensorDescriptor_t qDesc, const void* q,
    cudnnTensorDescriptor_t kDesc, const void* k,
    cudnnTensorDescriptor_t vDesc, const void* v,
    cudnnTensorDescriptor_t doDesc, const void* do_,
    cudnnTensorDescriptor_t dqDesc, void* dq,
    cudnnTensorDescriptor_t dkDesc, void* dk,
    cudnnTensorDescriptor_t dvDesc, void* dv,
    void* workspace, size_t workspaceSizeInBytes);

cudnnStatus_t cudnnMultiHeadAttnBackwardWeights(
    cudnnHandle_t, cudnnMultiHeadAttnDescriptor_t attnDesc,
    int seqLen, int batchSize,
    cudnnTensorDescriptor_t qDesc, const void* q,
    cudnnTensorDescriptor_t kDesc, const void* k,
    cudnnTensorDescriptor_t vDesc, const void* v,
    cudnnTensorDescriptor_t doDesc, const void* do_,
    void* dw,
    void* workspace, size_t workspaceSizeInBytes);
}

static bool approx_eq(float a, float b, float eps = 1e-3f) {
    if (std::isinf(a) && std::isinf(b)) return (a > 0) == (b > 0);
    if (std::isnan(a) || std::isnan(b)) return false;
    return std::abs(a - b) <= eps;
}

static void refMHA(int seqLen, int batchSize, int D, float scale,
    const float* q, const float* w, float* o)
{
    int wSize = D * D;
    const float* Wq = w;
    const float* Wk = w + wSize;
    const float* Wv = w + 2*wSize;
    const float* Wo = w + 3*wSize;
    const float* bq = w + 4*wSize;
    const float* bk = w + 4*wSize + D;
    const float* bv = w + 4*wSize + 2*D;
    const float* bo = w + 4*wSize + 3*D;

    int total = seqLen * batchSize * D;
    std::vector<float> Q(total), K(total), V(total);

    for (int t = 0; t < seqLen; ++t)
    for (int b = 0; b < batchSize; ++b) {
        int base = (t * batchSize + b) * D;
        for (int j = 0; j < D; ++j) {
            float sq = bq[j], sk = bk[j], sv = bv[j];
            for (int i = 0; i < D; ++i) {
                float in = q[base + i];
                sq += Wq[j*D+i] * in;
                sk += Wk[j*D+i] * in;
                sv += Wv[j*D+i] * in;
            }
            Q[base + j] = sq;
            K[base + j] = sk;
            V[base + j] = sv;
        }
    }

    for (int b = 0; b < batchSize; ++b) {
        for (int tq = 0; tq < seqLen; ++tq) {
            std::vector<float> scores(seqLen);
            float maxScore = -1e38f;
            for (int tk = 0; tk < seqLen; ++tk) {
                float dot = 0.f;
                int qBase = (tq * batchSize + b) * D;
                int kBase = (tk * batchSize + b) * D;
                for (int d = 0; d < D; ++d)
                    dot += Q[qBase + d] * K[kBase + d];
                scores[tk] = dot * scale;
                if (scores[tk] > maxScore) maxScore = scores[tk];
            }
            float sum = 0.f;
            for (int tk = 0; tk < seqLen; ++tk) {
                scores[tk] = expf(scores[tk] - maxScore);
                sum += scores[tk];
            }
            for (int tk = 0; tk < seqLen; ++tk) scores[tk] /= sum;

            std::vector<float> out(D, 0.f);
            for (int tk = 0; tk < seqLen; ++tk) {
                int vBase = (tk * batchSize + b) * D;
                for (int d = 0; d < D; ++d)
                    out[d] += scores[tk] * V[vBase + d];
            }

            int oBase = (tq * batchSize + b) * D;
            for (int j = 0; j < D; ++j) {
                float s = bo[j];
                for (int i = 0; i < D; ++i)
                    s += Wo[j*D+i] * out[i];
                o[oBase + j] = s;
            }
        }
    }
}

int main() {
    std::cout << "=== Test: cuDNN Multi-Head Attention (P3.10) ===\n";
    int pass = 0, total = 0;

    cudnnHandle_t handle;
    if (cudnnCreate(&handle) != CUDNN_STATUS_SUCCESS) {
        std::cerr << "FAIL: cudnnCreate\n";
        return 1;
    }

    int seqLen = 2, batchSize = 1, D = 3;
    int totalEls = seqLen * batchSize * D;
    int wSize = D * D;
    int totalWeights = 4 * wSize + 4 * D;

    std::vector<float> q(totalEls), k(totalEls), v(totalEls);
    std::vector<float> w(totalWeights);
    std::vector<float> o(totalEls, 0.f);
    std::vector<float> refO(totalEls, 0.f);

    for (int i = 0; i < totalEls; ++i) q[i] = static_cast<float>((i % 3) - 1) * 0.1f;
    for (int i = 0; i < totalEls; ++i) k[i] = q[i];
    for (int i = 0; i < totalEls; ++i) v[i] = q[i];
    for (int i = 0; i < totalWeights; ++i) w[i] = static_cast<float>((i % 5) - 2) * 0.05f;

    cudnnTensorDescriptor_t qDesc, kDesc, vDesc, oDesc;
    cudnnCreateTensorDescriptor(&qDesc);
    cudnnSetTensor4dDescriptor(qDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, seqLen, batchSize, D, 1);
    cudnnCreateTensorDescriptor(&kDesc);
    cudnnSetTensor4dDescriptor(kDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, seqLen, batchSize, D, 1);
    cudnnCreateTensorDescriptor(&vDesc);
    cudnnSetTensor4dDescriptor(vDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, seqLen, batchSize, D, 1);
    cudnnCreateTensorDescriptor(&oDesc);
    cudnnSetTensor4dDescriptor(oDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, seqLen, batchSize, D, 1);

    cudnnMultiHeadAttnDescriptor_t attnDesc;
    cudnnCreateMultiHeadAttnDescriptor(&attnDesc);
    float scale = 1.0f / std::sqrt(static_cast<float>(D));
    cudnnSetMultiHeadAttnDescriptor(attnDesc, 0, 1, D, D, seqLen, batchSize, scale);

    // ── Test 1: Forward vs reference ────────────────────────────────────
    ++total;
    {
        auto r = cudnnMultiHeadAttnForward(handle, attnDesc, seqLen, batchSize, w.data(),
            qDesc, q.data(), kDesc, k.data(), vDesc, v.data(), oDesc, o.data(), nullptr, 0);
        bool ok = (r == CUDNN_STATUS_SUCCESS);

        refMHA(seqLen, batchSize, D, scale, q.data(), w.data(), refO.data());

        if (ok) {
            for (int i = 0; i < totalEls && ok; ++i) {
                if (!approx_eq(o[i], refO[i], 1e-3f)) {
                    std::cerr << "FAIL [Forward] o[" << i << "]=" << o[i] << " expected=" << refO[i] << "\n";
                    ok = false;
                }
            }
        }
        if (ok) { std::cout << "PASS [Forward vs reference]\n"; ++pass; }
        else { std::cerr << "FAIL [Forward] status=" << r << "\n"; }
    }

    // ── Test 2: Backward weights non-zero ─────────────────────────────
    ++total;
    {
        std::vector<float> dw(totalWeights, 0.f);
        std::vector<float> do_(totalEls, 1.f);
        auto r = cudnnMultiHeadAttnBackwardWeights(handle, attnDesc, seqLen, batchSize,
            qDesc, q.data(), kDesc, k.data(), vDesc, v.data(),
            oDesc, do_.data(), dw.data(), nullptr, 0);
        bool ok = (r == CUDNN_STATUS_SUCCESS);
        if (ok) {
            bool anyNonZero = false;
            for (float v : dw) if (v != 0.f) { anyNonZero = true; break; }
            if (!anyNonZero) { std::cerr << "FAIL [BackwardWeights] all zeros\n"; ok = false; }
        }
        if (ok) { std::cout << "PASS [Backward weights non-zero]\n"; ++pass; }
    }

    // ── Test 3: Backward data produces gradients ────────────────────
    ++total;
    {
        std::vector<float> dq(totalEls, 0.f), dk(totalEls, 0.f), dv(totalEls, 0.f);
        std::vector<float> do_(totalEls, 1.f);
        auto r = cudnnMultiHeadAttnBackwardData(handle, attnDesc, seqLen, batchSize, w.data(),
            qDesc, q.data(), kDesc, k.data(), vDesc, v.data(),
            oDesc, do_.data(), qDesc, dq.data(), kDesc, dk.data(), vDesc, dv.data(), nullptr, 0);
        bool ok = (r == CUDNN_STATUS_SUCCESS);
        if (ok) {
            bool anyNonZero = false;
            for (float v : dq) if (v != 0.f) { anyNonZero = true; break; }
            if (!anyNonZero) { std::cerr << "FAIL [BackwardData] all zeros\n"; ok = false; }
        }
        if (ok) { std::cout << "PASS [Backward data non-zero]\n"; ++pass; }
    }

    // ── Test 4: Null pointer rejection ──────────────────────────────
    ++total;
    {
        auto r = cudnnMultiHeadAttnForward(handle, nullptr, 0, 0, nullptr,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0);
        if (r == CUDNN_STATUS_INVALID_VALUE) {
            std::cout << "PASS [Null pointer rejected]\n"; ++pass;
        } else {
            std::cerr << "FAIL [Null pointer] status=" << r << "\n";
        }
    }

    cudnnDestroyMultiHeadAttnDescriptor(attnDesc);
    cudnnDestroyTensorDescriptor(qDesc);
    cudnnDestroyTensorDescriptor(kDesc);
    cudnnDestroyTensorDescriptor(vDesc);
    cudnnDestroyTensorDescriptor(oDesc);
    cudnnDestroy(handle);

    std::cout << "\n" << pass << "/" << total << " tests passed.\n";
    return (pass == total) ? 0 : 1;
}
