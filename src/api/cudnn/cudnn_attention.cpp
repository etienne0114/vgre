// cuDNN multi-head attention forward + backward

#include "cudnn_internal.h"
#include "vgre/common/openmp_helper.h"

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

    #ifdef _OPENMP
    #pragma omp parallel for OMP_COLLAPSE(2) if (seqLen * batchSize > 4)
    #endif
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

    #ifdef _OPENMP
    #pragma omp parallel for OMP_COLLAPSE(2) if (batchSize * seqLen > 4)
    #endif
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

// Backward through attention: full gradient through softmax(Q·K^T·scale)·V.
//
// Forward recap (per batch b, query position tq):
//   Q[b,tq,:] = Wq · input[b,tq,:] + bq
//   K[b,tk,:] = Wk · input[b,tk,:] + bk
//   V[b,tk,:] = Wv · input[b,tk,:] + bv
//   S[b,tq,tk] = scale · ⟨Q[b,tq], K[b,tk]⟩
//   A[b,tq,:]  = softmax(S[b,tq,:])
//   C[b,tq,:] = Σ_tk A[b,tq,tk] · V[b,tk,:]
//   output[b,tq,:] = Wo · C[b,tq,:] + bo
//
// Backward:
//   dC      = Wo^T · d_output
//   dV_proj += A^T · dC             (accumulate over query positions)
//   dA      = dC · V_proj^T
//   dS      = A ⊙ (dA - Σ_tk A[tq,tk]·dA[tq,tk])   (softmax Jacobian)
//   dQ_proj = scale · dS · K_proj
//   dK_proj = scale · dS^T · Q_proj
//   dq      = Wq^T · dQ_proj + Wk^T · dK_proj + Wv^T · dV_proj
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
    float scale = ad->scaling;
    const float* wf  = (const float*)weights;
    const float* qf  = (const float*)q;
    const float* dof = (const float*)do_;
    float* dqf = (float*)dq;
    float* dkf = (float*)dk;
    float* dvf = (float*)dv;

    int total = seqLen * batchSize * D;
    memset(dqf, 0, total * sizeof(float));
    memset(dkf, 0, total * sizeof(float));
    memset(dvf, 0, total * sizeof(float));

    int wSize = D * D;
    const float* Wq = wf;
    const float* Wk = wf + wSize;
    const float* Wv = wf + 2*wSize;
    const float* Wo = wf + 3*wSize;
    const float* bq = wf + 4*wSize;
    const float* bk = wf + 4*wSize + D;
    const float* bv = wf + 4*wSize + 2*D;

    // ── Step 1: Recompute Q, K, V projections (same as forward) ────────────────
    std::vector<float> Q(total), K_(total), V_(total);
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
            Q[base + j]  = sq;
            K_[base + j] = sk;
            V_[base + j] = sv;
        }
    }

    // ── Step 2: backward through output projection — dC = Wo^T · do ────────────
    std::vector<float> dC(total, 0.f);
    for (int t = 0; t < seqLen; ++t)
    for (int b = 0; b < batchSize; ++b) {
        int base = (t * batchSize + b) * D;
        for (int i = 0; i < D; ++i) {
            float g = dof[base + i];
            for (int j = 0; j < D; ++j)
                dC[base + j] += Wo[i*D+j] * g;
        }
    }

    // ── Steps 3-5: per-batch backward through attention scores ──────────────────
    // dQ_proj, dK_proj, dV_proj accumulate per-position gradients.
    std::vector<float> dQp(total, 0.f), dKp(total, 0.f), dVp(total, 0.f);

    for (int b = 0; b < batchSize; ++b) {
        // Recompute attention weights A[tq, tk] for this batch element.
        // A is seqLen × seqLen.
        std::vector<float> A(static_cast<size_t>(seqLen) * seqLen);
        for (int tq = 0; tq < seqLen; ++tq) {
            int qBase = (tq * batchSize + b) * D;
            float maxS = -1e38f;
            for (int tk = 0; tk < seqLen; ++tk) {
                int kBase = (tk * batchSize + b) * D;
                float dot = 0.f;
                for (int d = 0; d < D; ++d) dot += Q[qBase + d] * K_[kBase + d];
                A[static_cast<size_t>(tq) * seqLen + tk] = dot * scale;
                if (A[static_cast<size_t>(tq) * seqLen + tk] > maxS)
                    maxS = A[static_cast<size_t>(tq) * seqLen + tk];
            }
            float sum = 0.f;
            for (int tk = 0; tk < seqLen; ++tk) {
                A[static_cast<size_t>(tq) * seqLen + tk] = std::exp(A[static_cast<size_t>(tq) * seqLen + tk] - maxS);
                sum += A[static_cast<size_t>(tq) * seqLen + tk];
            }
            for (int tk = 0; tk < seqLen; ++tk)
                A[static_cast<size_t>(tq) * seqLen + tk] /= sum;
        }

        // dV_proj[tk] += Σ_tq A[tq,tk] · dC[tq]
        for (int tq = 0; tq < seqLen; ++tq) {
            int cBase = (tq * batchSize + b) * D;
            for (int tk = 0; tk < seqLen; ++tk) {
                int vBase = (tk * batchSize + b) * D;
                float attn = A[static_cast<size_t>(tq) * seqLen + tk];
                for (int d = 0; d < D; ++d)
                    dVp[vBase + d] += attn * dC[cBase + d];
            }
        }

        // dA[tq,tk] = Σ_d dC[tq,d] · V_proj[tk,d]
        // dS[tq,tk] = A[tq,tk] · (dA[tq,tk] - Σ_t A[tq,t]·dA[tq,t])
        for (int tq = 0; tq < seqLen; ++tq) {
            int cBase = (tq * batchSize + b) * D;
            std::vector<float> dA_row(seqLen, 0.f);
            for (int tk = 0; tk < seqLen; ++tk) {
                int vBase = (tk * batchSize + b) * D;
                for (int d = 0; d < D; ++d)
                    dA_row[tk] += dC[cBase + d] * V_[vBase + d];
            }
            // Jacobian of softmax: dS_ij = A_ij * (dA_ij - dot(A_row, dA_row))
            float dot_AdA = 0.f;
            for (int tk = 0; tk < seqLen; ++tk)
                dot_AdA += A[static_cast<size_t>(tq) * seqLen + tk] * dA_row[tk];
            // dQ_proj[tq] += scale · Σ_tk dS[tq,tk] · K_proj[tk]
            // dK_proj[tk] += scale · dS[tq,tk] · Q_proj[tq]
            int qBase = (tq * batchSize + b) * D;
            for (int tk = 0; tk < seqLen; ++tk) {
                float dS = A[static_cast<size_t>(tq) * seqLen + tk] * (dA_row[tk] - dot_AdA);
                float dS_scaled = scale * dS;
                int kBase = (tk * batchSize + b) * D;
                for (int d = 0; d < D; ++d) {
                    dQp[qBase + d] += dS_scaled * K_[kBase + d];
                    dKp[kBase + d] += dS_scaled * Q[qBase + d];
                }
            }
        }
    }

    // ── Step 6: backward through projections Wq, Wk, Wv → input grad ──────────
    // The forward only reads from `q` (self-attention: q is the sole input source
    // for all three projections). Therefore all gradient paths accumulate into dq.
    // For a proper cross-attention caller, dq/dk/dv map to their respective inputs;
    // here we keep dk/dv as zero and fold all paths into dq to match the forward.
    for (int t = 0; t < seqLen; ++t)
    for (int b = 0; b < batchSize; ++b) {
        int base = (t * batchSize + b) * D;
        for (int j = 0; j < D; ++j) {
            for (int i = 0; i < D; ++i) {
                dqf[base + i] += Wq[j*D+i] * dQp[base + j]
                               + Wk[j*D+i] * dKp[base + j]
                               + Wv[j*D+i] * dVp[base + j];
            }
        }
    }

    (void)qDesc; (void)kDesc; (void)vDesc; (void)doDesc; (void)dkDesc; (void)dvDesc; (void)dqDesc;
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

    #ifdef _OPENMP
    #pragma omp parallel for OMP_COLLAPSE(2) if (seqLen * batchSize > 4)
    #endif
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

    #ifdef _OPENMP
    #pragma omp parallel for OMP_COLLAPSE(2) if (seqLen * batchSize > 4)
    #endif
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
