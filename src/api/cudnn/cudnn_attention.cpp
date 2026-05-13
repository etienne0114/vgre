// cuDNN multi-head attention forward + backward

#include "cudnn_internal.h"

extern "C" {

cudnnStatus_t cudnnMultiHeadAttnForward(
    cudnnHandle_t,
    cudnnMultiHeadAttnDescriptor_t attnDesc,
    int seqLen, int batchSize,
    const void* weights,
    cudnnTensorDescriptor_t qDesc, const void* q,
    cudnnTensorDescriptor_t kDesc, const void* k,
    cudnnTensorDescriptor_t vDesc, const void* v,
    cudnnTensorDescriptor_t oDesc, void* o,
    void* /*workspace*/, size_t /*workspaceSizeInBytes*/)
{
    if (!attnDesc || !weights || !q || !k || !v || !o) return CUDNN_STATUS_INVALID_VALUE;

    auto* ad = (MultiHeadAttnDesc*)attnDesc;
    int D = ad->modelDim;
    float scale = ad->scaling;
    const float* wf = (const float*)weights;
    const float* qf = (const float*)q;
    const float* kf = (const float*)k;
    const float* vf = (const float*)v;
    float* of = (float*)o;

    int wSize = D * D;
    const float* Wq = wf;
    const float* Wk = wf + wSize;
    const float* Wv = wf + 2*wSize;
    const float* Wo = wf + 3*wSize;
    const float* bq = wf + 4*wSize;
    const float* bk = wf + 4*wSize + D;
    const float* bv = wf + 4*wSize + 2*D;
    const float* bo = wf + 4*wSize + 3*D;

    int total = seqLen * batchSize * D;
    std::vector<float> Q(total), K(total), V(total);

    for (int t = 0; t < seqLen; ++t)
    for (int b = 0; b < batchSize; ++b) {
        int base = (t * batchSize + b) * D;
        for (int j = 0; j < D; ++j) {
            float sq = bq[j], sk = bk[j], sv = bv[j];
            for (int i = 0; i < D; ++i) {
                float in = qf[base + i];
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
                float sum = bo[j];
                for (int i = 0; i < D; ++i)
                    sum += Wo[j*D+i] * out[i];
                of[oBase + j] = sum;
            }
        }
    }

    (void)qDesc; (void)kDesc; (void)vDesc; (void)oDesc;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnMultiHeadAttnBackwardData(
    cudnnHandle_t,
    cudnnMultiHeadAttnDescriptor_t attnDesc,
    int seqLen, int batchSize,
    const void* weights,
    cudnnTensorDescriptor_t qDesc, const void* q,
    cudnnTensorDescriptor_t kDesc, const void* k,
    cudnnTensorDescriptor_t vDesc, const void* v,
    cudnnTensorDescriptor_t doDesc, const void* do_,
    cudnnTensorDescriptor_t dqDesc, void* dq,
    cudnnTensorDescriptor_t dkDesc, void* dk,
    cudnnTensorDescriptor_t dvDesc, void* dv,
    void* /*workspace*/, size_t /*workspaceSizeInBytes*/)
{
    if (!attnDesc || !weights || !q || !k || !v || !do_ || !dq || !dk || !dv)
        return CUDNN_STATUS_INVALID_VALUE;

    auto* ad = (MultiHeadAttnDesc*)attnDesc;
    int D = ad->modelDim;
    const float* wf = (const float*)weights;
    const float* dof = (const float*)do_;
    float* dqf = (float*)dq;
    float* dkf = (float*)dk;
    float* dvf = (float*)dv;

    int total = seqLen * batchSize * D;
    std::memset(dqf, 0, total * sizeof(float));
    std::memset(dkf, 0, total * sizeof(float));
    std::memset(dvf, 0, total * sizeof(float));

    int wSize = D * D;
    const float* Wq = wf;
    const float* Wk = wf + wSize;
    const float* Wv = wf + 2*wSize;
    const float* Wo = wf + 3*wSize;

    std::vector<float> dOut(total, 0.f);
    for (int t = 0; t < seqLen; ++t)
    for (int b = 0; b < batchSize; ++b) {
        int base = (t * batchSize + b) * D;
        for (int i = 0; i < D; ++i) {
            float grad = dof[base + i];
            for (int j = 0; j < D; ++j)
                dOut[base + j] += Wo[i*D+j] * grad;
        }
    }

    for (int t = 0; t < seqLen; ++t)
    for (int b = 0; b < batchSize; ++b) {
        int base = (t * batchSize + b) * D;
        for (int i = 0; i < D; ++i) {
            float grad = dOut[base + i];
            for (int j = 0; j < D; ++j) {
                dqf[base + j] += Wq[j*D+i] * grad;
                dkf[base + j] += Wk[j*D+i] * grad;
                dvf[base + j] += Wv[j*D+i] * grad;
            }
        }
    }

    (void)qDesc; (void)kDesc; (void)vDesc; (void)doDesc; (void)dkDesc; (void)dvDesc;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnMultiHeadAttnBackwardWeights(
    cudnnHandle_t,
    cudnnMultiHeadAttnDescriptor_t attnDesc,
    int seqLen, int batchSize,
    cudnnTensorDescriptor_t qDesc, const void* q,
    cudnnTensorDescriptor_t kDesc, const void* k,
    cudnnTensorDescriptor_t vDesc, const void* v,
    cudnnTensorDescriptor_t doDesc, const void* do_,
    void* dw,
    void* /*workspace*/, size_t /*workspaceSizeInBytes*/)
{
    if (!attnDesc || !q || !k || !v || !do_ || !dw) return CUDNN_STATUS_INVALID_VALUE;

    auto* ad = (MultiHeadAttnDesc*)attnDesc;
    int D = ad->modelDim;
    const float* qf = (const float*)q;
    const float* dof = (const float*)do_;
    float* dwf = (float*)dw;

    int wSize = D * D;
    for (int i = 0; i < 4*wSize + 4*D; ++i) dwf[i] = 0.f;

    float* dWq = dwf;
    float* dWk = dwf + wSize;
    float* dWv = dwf + 2*wSize;
    float* dWo = dwf + 3*wSize;
    float* dbq = dwf + 4*wSize;
    float* dbk = dwf + 4*wSize + D;
    float* dbv = dwf + 4*wSize + 2*D;
    float* dbo = dwf + 4*wSize + 3*D;

    for (int t = 0; t < seqLen; ++t)
    for (int b = 0; b < batchSize; ++b) {
        int base = (t * batchSize + b) * D;
        for (int j = 0; j < D; ++j) {
            for (int i = 0; i < D; ++i) {
                float in = qf[base + i];
                dWq[j*D+i] += in * dof[base + j];
                dWk[j*D+i] += in * dof[base + j];
                dWv[j*D+i] += in * dof[base + j];
            }
            dbq[j] += dof[base + j];
            dbk[j] += dof[base + j];
            dbv[j] += dof[base + j];
        }
    }

    for (int t = 0; t < seqLen; ++t)
    for (int b = 0; b < batchSize; ++b) {
        int base = (t * batchSize + b) * D;
        for (int j = 0; j < D; ++j)
            dbo[j] += dof[base + j];
    }

    (void)qDesc; (void)kDesc; (void)vDesc; (void)doDesc;
    return CUDNN_STATUS_SUCCESS;
}

} // extern "C"
