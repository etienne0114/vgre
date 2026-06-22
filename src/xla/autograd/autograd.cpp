// VGRE-Autograd — see include/vgre/xla/autograd.h.

#include "vgre/xla/autograd.h"
#include "vgre/xla/intree_gemm.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <stdexcept>
#include <unordered_set>

namespace vgre {
namespace xla {
namespace autograd {

namespace {
constexpr float kPi = 3.14159265358979323846f;

// Allocate a result node with zeroed data + grad and record its parents.
Var newNode(std::vector<int64_t> shape, std::vector<Var> parents, bool requires_grad) {
    auto n = std::make_shared<Node>();
    n->shape = std::move(shape);
    n->data.assign((size_t)n->size(), 0.0f);
    n->grad.assign(n->data.size(), 0.0f);
    n->parents = std::move(parents);
    n->requires_grad = requires_grad;
    return n;
}

bool anyReq(std::initializer_list<const Var*> vs) {
    for (auto* v : vs) if ((*v)->requires_grad) return true;
    return false;
}
}  // namespace

Var make(std::vector<int64_t> shape, bool requires_grad) {
    auto n = std::make_shared<Node>();
    n->shape = std::move(shape);
    n->data.assign((size_t)n->size(), 0.0f);
    n->grad.assign(n->data.size(), 0.0f);
    n->requires_grad = requires_grad;
    return n;
}

Var make(std::vector<int64_t> shape, std::vector<float> data, bool requires_grad) {
    auto n = make(std::move(shape), requires_grad);
    if ((int64_t)data.size() != n->size())
        throw std::runtime_error("autograd::make: data size mismatch");
    n->data = std::move(data);
    return n;
}

// ── matmul: C[M,N] = A[M,K] · B[K,N] ─────────────────────────────────────────
Var matmul(const Var& a, const Var& b) {
    if (a->shape.size() != 2 || b->shape.size() != 2)
        throw std::runtime_error("matmul: operands must be rank-2");
    const int64_t M = a->shape[0], K = a->shape[1], N = b->shape[1];
    if (b->shape[0] != K) throw std::runtime_error("matmul: inner dim mismatch");

    Var out = newNode({M, N}, {a, b}, anyReq({&a, &b}));
    // Use the multi-threaded GEMM: model weight matmuls are large, and for small
    // matrices the pool runs the single row block inline (no thread overhead).
    intree::gemm_f32_threaded(false, false, M, N, K,
                              a->data.data(), b->data.data(), out->data.data());

    Node* op = out.get();
    Var A = a, B = b;
    out->backward_fn = [op, A, B, M, N, K]() {
        // dA[M,K] += dC[M,N] · Bᵀ  → gemm(tA=F, tB=T, M, K, N, dC, B)
        if (A->requires_grad) {
            std::vector<float> dA((size_t)M * K, 0.0f);
            intree::gemm_f32_threaded(false, true, M, K, N,
                                      op->grad.data(), B->data.data(), dA.data());
            for (size_t i = 0; i < dA.size(); ++i) A->grad[i] += dA[i];
        }
        // dB[K,N] += Aᵀ · dC[M,N]  → gemm(tA=T, tB=F, K, N, M, A, dC)
        if (B->requires_grad) {
            std::vector<float> dB((size_t)K * N, 0.0f);
            intree::gemm_f32_threaded(true, false, K, N, M,
                                      A->data.data(), op->grad.data(), dB.data());
            for (size_t i = 0; i < dB.size(); ++i) B->grad[i] += dB[i];
        }
    };
    return out;
}

// ── add: same-shape, or 1-D bias [N] broadcast over rows of a[M,N] ───────────
Var add(const Var& a, const Var& b) {
    Var out = newNode(a->shape, {a, b}, anyReq({&a, &b}));
    const int64_t n = a->size();
    if (a->shape == b->shape) {
        for (int64_t i = 0; i < n; ++i) out->data[i] = a->data[i] + b->data[i];
        Node* op = out.get(); Var A = a, B = b;
        out->backward_fn = [op, A, B, n]() {
            if (A->requires_grad) for (int64_t i = 0; i < n; ++i) A->grad[i] += op->grad[i];
            if (B->requires_grad) for (int64_t i = 0; i < n; ++i) B->grad[i] += op->grad[i];
        };
    } else if (a->shape.size() == 2 && b->shape.size() == 1 &&
               b->shape[0] == a->shape[1]) {
        const int64_t M = a->shape[0], N = a->shape[1];
        for (int64_t i = 0; i < M; ++i)
            for (int64_t j = 0; j < N; ++j) out->data[i * N + j] = a->data[i * N + j] + b->data[j];
        Node* op = out.get(); Var A = a, B = b;
        out->backward_fn = [op, A, B, M, N]() {
            if (A->requires_grad)
                for (int64_t i = 0; i < M * N; ++i) A->grad[i] += op->grad[i];
            if (B->requires_grad)
                for (int64_t i = 0; i < M; ++i)
                    for (int64_t j = 0; j < N; ++j) B->grad[j] += op->grad[i * N + j];
        };
    } else {
        throw std::runtime_error("add: incompatible shapes");
    }
    return out;
}

Var mul(const Var& a, const Var& b) {
    if (a->shape != b->shape) throw std::runtime_error("mul: shape mismatch");
    Var out = newNode(a->shape, {a, b}, anyReq({&a, &b}));
    const int64_t n = a->size();
    for (int64_t i = 0; i < n; ++i) out->data[i] = a->data[i] * b->data[i];
    Node* op = out.get(); Var A = a, B = b;
    out->backward_fn = [op, A, B, n]() {
        if (A->requires_grad) for (int64_t i = 0; i < n; ++i) A->grad[i] += op->grad[i] * B->data[i];
        if (B->requires_grad) for (int64_t i = 0; i < n; ++i) B->grad[i] += op->grad[i] * A->data[i];
    };
    return out;
}

Var scale(const Var& a, float s) {
    Var out = newNode(a->shape, {a}, a->requires_grad);
    const int64_t n = a->size();
    for (int64_t i = 0; i < n; ++i) out->data[i] = a->data[i] * s;
    Node* op = out.get(); Var A = a;
    out->backward_fn = [op, A, s, n]() {
        if (A->requires_grad) for (int64_t i = 0; i < n; ++i) A->grad[i] += op->grad[i] * s;
    };
    return out;
}

Var relu(const Var& x) {
    Var out = newNode(x->shape, {x}, x->requires_grad);
    const int64_t n = x->size();
    for (int64_t i = 0; i < n; ++i) out->data[i] = x->data[i] > 0.0f ? x->data[i] : 0.0f;
    Node* op = out.get(); Var X = x;
    out->backward_fn = [op, X, n]() {
        if (X->requires_grad)
            for (int64_t i = 0; i < n; ++i) X->grad[i] += (X->data[i] > 0.0f) ? op->grad[i] : 0.0f;
    };
    return out;
}

// GELU (tanh approximation): 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
Var gelu(const Var& x) {
    Var out = newNode(x->shape, {x}, x->requires_grad);
    const int64_t n = x->size();
    const float c = std::sqrt(2.0f / kPi);
    for (int64_t i = 0; i < n; ++i) {
        const float v = x->data[i];
        const float u = c * (v + 0.044715f * v * v * v);
        out->data[i] = 0.5f * v * (1.0f + std::tanh(u));
    }
    Node* op = out.get(); Var X = x;
    out->backward_fn = [op, X, n, c]() {
        if (!X->requires_grad) return;
        for (int64_t i = 0; i < n; ++i) {
            const float v = X->data[i];
            const float v3 = v * v * v;
            const float u = c * (v + 0.044715f * v3);
            const float t = std::tanh(u);
            const float dudv = c * (1.0f + 3.0f * 0.044715f * v * v);
            // d/dv [0.5 v (1+t)] = 0.5(1+t) + 0.5 v (1-t²) du/dv
            const float dv = 0.5f * (1.0f + t) + 0.5f * v * (1.0f - t * t) * dudv;
            X->grad[i] += op->grad[i] * dv;
        }
    };
    return out;
}

Var silu(const Var& x) {
    Var out = newNode(x->shape, {x}, x->requires_grad);
    const int64_t n = x->size();
    for (int64_t i = 0; i < n; ++i) {
        const float s = 1.0f / (1.0f + std::exp(-x->data[i]));
        out->data[i] = x->data[i] * s;
    }
    Node* op = out.get(); Var X = x;
    out->backward_fn = [op, X, n]() {
        if (!X->requires_grad) return;
        for (int64_t i = 0; i < n; ++i) {
            const float s = 1.0f / (1.0f + std::exp(-X->data[i]));
            // d/dx [x·σ] = σ + x·σ(1-σ)
            X->grad[i] += op->grad[i] * (s + X->data[i] * s * (1.0f - s));
        }
    };
    return out;
}

// ── RMSNorm over last dim of x[M,D]; y = x / rms(x) * weight ──────────────────
Var rms_norm(const Var& x, const Var& weight, float eps) {
    if (x->shape.size() != 2) throw std::runtime_error("rms_norm: x must be [M,D]");
    const int64_t M = x->shape[0], D = x->shape[1];
    if (weight->shape.size() != 1 || weight->shape[0] != D)
        throw std::runtime_error("rms_norm: weight must be [D]");
    Var out = newNode({M, D}, {x, weight}, anyReq({&x, &weight}));
    std::vector<float> inv(M);  // 1/rms per row
    for (int64_t i = 0; i < M; ++i) {
        float ss = 0.0f;
        for (int64_t j = 0; j < D; ++j) { float v = x->data[i * D + j]; ss += v * v; }
        inv[i] = 1.0f / std::sqrt(ss / (float)D + eps);
        for (int64_t j = 0; j < D; ++j)
            out->data[i * D + j] = x->data[i * D + j] * inv[i] * weight->data[j];
    }
    Node* op = out.get(); Var X = x, W = weight;
    out->backward_fn = [op, X, W, M, D, inv]() {
        for (int64_t i = 0; i < M; ++i) {
            const float r = inv[i];
            // For weight: dW[j] += dy[i,j] * x[i,j] * r
            if (W->requires_grad)
                for (int64_t j = 0; j < D; ++j)
                    W->grad[j] += op->grad[i * D + j] * X->data[i * D + j] * r;
            if (X->requires_grad) {
                // g_j = dy[i,j]*w[j]; dx[i,j] = r*g_j - (r³/D)*x[i,j]*Σ_k g_k*x[i,k]
                float dot = 0.0f;
                for (int64_t j = 0; j < D; ++j)
                    dot += op->grad[i * D + j] * W->data[j] * X->data[i * D + j];
                const float r3 = r * r * r / (float)D;
                for (int64_t j = 0; j < D; ++j) {
                    const float g = op->grad[i * D + j] * W->data[j];
                    X->grad[i * D + j] += r * g - r3 * X->data[i * D + j] * dot;
                }
            }
        }
    };
    return out;
}

// ── LayerNorm over last dim of x[M,D]; y = (x-µ)/√(σ²+eps) * w + b ────────────
Var layer_norm(const Var& x, const Var& weight, const Var& bias, float eps) {
    if (x->shape.size() != 2) throw std::runtime_error("layer_norm: x must be [M,D]");
    const int64_t M = x->shape[0], D = x->shape[1];
    if (weight->shape.size() != 1 || weight->shape[0] != D ||
        bias->shape.size() != 1 || bias->shape[0] != D)
        throw std::runtime_error("layer_norm: weight/bias must be [D]");
    Var out = newNode({M, D}, {x, weight, bias}, anyReq({&x, &weight, &bias}));
    std::vector<float> mu(M), inv(M);                  // mean, 1/√(var+eps) per row
    std::vector<float> xhat((size_t)M * D);            // normalized, cached
    for (int64_t i = 0; i < M; ++i) {
        float m = 0.0f;
        for (int64_t j = 0; j < D; ++j) m += x->data[i * D + j];
        m /= (float)D;
        float v = 0.0f;
        for (int64_t j = 0; j < D; ++j) { float d = x->data[i * D + j] - m; v += d * d; }
        v /= (float)D;
        mu[i] = m; inv[i] = 1.0f / std::sqrt(v + eps);
        for (int64_t j = 0; j < D; ++j) {
            const float xh = (x->data[i * D + j] - m) * inv[i];
            xhat[i * D + j] = xh;
            out->data[i * D + j] = xh * weight->data[j] + bias->data[j];
        }
    }
    Node* op = out.get(); Var X = x, W = weight, Bs = bias;
    out->backward_fn = [op, X, W, Bs, M, D, inv, xhat]() {
        for (int64_t i = 0; i < M; ++i) {
            if (W->requires_grad)
                for (int64_t j = 0; j < D; ++j) W->grad[j] += op->grad[i * D + j] * xhat[i * D + j];
            if (Bs->requires_grad)
                for (int64_t j = 0; j < D; ++j) Bs->grad[j] += op->grad[i * D + j];
            if (X->requires_grad) {
                // g_j = dy_j * w_j ; dx_j = inv/D * (D*g_j - Σg - xhat_j*Σ(g*xhat))
                float sumG = 0.0f, sumGX = 0.0f;
                for (int64_t j = 0; j < D; ++j) {
                    const float g = op->grad[i * D + j] * W->data[j];
                    sumG += g; sumGX += g * xhat[i * D + j];
                }
                const float s = inv[i] / (float)D;
                for (int64_t j = 0; j < D; ++j) {
                    const float g = op->grad[i * D + j] * W->data[j];
                    X->grad[i * D + j] += s * ((float)D * g - sumG - xhat[i * D + j] * sumGX);
                }
            }
        }
    };
    return out;
}

// ── RoPE on x[T, H*Dh]: rotate dim-pairs by angle (pos · base^(-2i/Dh)) ───────
Var rope(const Var& x, int num_heads, float base) {
    if (x->shape.size() != 2) throw std::runtime_error("rope: x must be [T, H*Dh]");
    const int64_t T = x->shape[0], D = x->shape[1];
    const int64_t Dh = D / num_heads;
    if (Dh * num_heads != D || (Dh & 1)) throw std::runtime_error("rope: bad head_dim");
    Var out = newNode({T, D}, {x}, x->requires_grad);
    const int64_t half = Dh / 2;
    for (int64_t t = 0; t < T; ++t)
        for (int h = 0; h < num_heads; ++h) {
            const int64_t off = t * D + (int64_t)h * Dh;
            for (int64_t i = 0; i < half; ++i) {
                const float theta = (float)t * std::pow(base, -2.0f * (float)i / (float)Dh);
                const float cs = std::cos(theta), sn = std::sin(theta);
                const float a = x->data[off + 2 * i], b = x->data[off + 2 * i + 1];
                out->data[off + 2 * i]     = a * cs - b * sn;
                out->data[off + 2 * i + 1] = a * sn + b * cs;
            }
        }
    Node* op = out.get(); Var X = x;
    out->backward_fn = [op, X, T, D, Dh, num_heads, half, base]() {
        if (!X->requires_grad) return;
        for (int64_t t = 0; t < T; ++t)
            for (int h = 0; h < num_heads; ++h) {
                const int64_t off = t * D + (int64_t)h * Dh;
                for (int64_t i = 0; i < half; ++i) {
                    const float theta = (float)t * std::pow(base, -2.0f * (float)i / (float)Dh);
                    const float cs = std::cos(theta), sn = std::sin(theta);
                    const float da = op->grad[off + 2 * i], db = op->grad[off + 2 * i + 1];
                    // Inverse rotation (transpose of the orthogonal rotation).
                    X->grad[off + 2 * i]     += da * cs + db * sn;
                    X->grad[off + 2 * i + 1] += -da * sn + db * cs;
                }
            }
    };
    return out;
}

// ── Multi-head scaled-dot-product attention (optionally causal) ──────────────
Var attention(const Var& q, const Var& k, const Var& v, int num_heads, bool causal) {
    if (q->shape.size() != 2 || q->shape != k->shape || q->shape != v->shape)
        throw std::runtime_error("attention: Q/K/V must share shape [T, H*Dh]");
    const int64_t T = q->shape[0], D = q->shape[1];
    const int64_t Dh = D / num_heads;
    if (Dh * num_heads != D) throw std::runtime_error("attention: bad head_dim");
    const float scale = 1.0f / std::sqrt((float)Dh);

    Var out = newNode({T, D}, {q, k, v}, anyReq({&q, &k, &v}));
    // Cache the per-head softmax probs P[h][T*T] for backward.
    auto P = std::make_shared<std::vector<float>>((size_t)num_heads * T * T, 0.0f);

    std::vector<float> Qh(T * Dh), Kh(T * Dh), Vh(T * Dh), S(T * T), Oh(T * Dh);
    for (int h = 0; h < num_heads; ++h) {
        const int64_t col = (int64_t)h * Dh;
        for (int64_t t = 0; t < T; ++t)
            for (int64_t d = 0; d < Dh; ++d) {
                Qh[t * Dh + d] = q->data[t * D + col + d];
                Kh[t * Dh + d] = k->data[t * D + col + d];
                Vh[t * Dh + d] = v->data[t * D + col + d];
            }
        // S = scale * Qh · Khᵀ
        intree::gemm_f32_rows(false, true, T, T, Dh, Qh.data(), Kh.data(), S.data(), 0, T);
        float* Ph = P->data() + (size_t)h * T * T;
        for (int64_t i = 0; i < T; ++i) {
            const int64_t lim = causal ? i : (T - 1);
            float mx = -1e30f;
            for (int64_t j = 0; j <= lim; ++j) { S[i * T + j] *= scale; mx = std::max(mx, S[i * T + j]); }
            float sum = 0.0f;
            for (int64_t j = 0; j <= lim; ++j) { float e = std::exp(S[i * T + j] - mx); Ph[i * T + j] = e; sum += e; }
            const float invs = 1.0f / sum;
            for (int64_t j = 0; j <= lim; ++j) Ph[i * T + j] *= invs;
            for (int64_t j = lim + 1; j < T; ++j) Ph[i * T + j] = 0.0f;  // masked
        }
        // Oh = Ph · Vh
        intree::gemm_f32_rows(false, false, T, Dh, T, Ph, Vh.data(), Oh.data(), 0, T);
        for (int64_t t = 0; t < T; ++t)
            for (int64_t d = 0; d < Dh; ++d) out->data[t * D + col + d] = Oh[t * Dh + d];
    }

    Node* op = out.get(); Var Q = q, K = k, V = v;
    out->backward_fn = [op, Q, K, V, P, T, D, Dh, num_heads, scale]() {
        std::vector<float> Qh(T * Dh), Kh(T * Dh), Vh(T * Dh), dO(T * Dh);
        std::vector<float> dV(T * Dh), dP(T * T), dS(T * T), dQ(T * Dh), dK(T * Dh);
        for (int h = 0; h < num_heads; ++h) {
            const int64_t col = (int64_t)h * Dh;
            const float* Ph = P->data() + (size_t)h * T * T;
            for (int64_t t = 0; t < T; ++t)
                for (int64_t d = 0; d < Dh; ++d) {
                    Qh[t * Dh + d] = Q->data[t * D + col + d];
                    Kh[t * Dh + d] = K->data[t * D + col + d];
                    Vh[t * Dh + d] = V->data[t * D + col + d];
                    dO[t * Dh + d] = op->grad[t * D + col + d];
                }
            // dV = Pᵀ · dO
            intree::gemm_f32_rows(true, false, T, Dh, T, Ph, dO.data(), dV.data(), 0, T);
            // dP = dO · Vhᵀ
            intree::gemm_f32_rows(false, true, T, T, Dh, dO.data(), Vh.data(), dP.data(), 0, T);
            // dS = softmax-backward(P, dP), row-wise: dS_ij = P_ij(dP_ij - Σ_k P_ik dP_ik)
            for (int64_t i = 0; i < T; ++i) {
                float dot = 0.0f;
                for (int64_t j = 0; j < T; ++j) dot += Ph[i * T + j] * dP[i * T + j];
                for (int64_t j = 0; j < T; ++j)
                    dS[i * T + j] = Ph[i * T + j] * (dP[i * T + j] - dot) * scale;
            }
            // dQ = dS · K ; dK = dSᵀ · Q  (scale already folded into dS)
            intree::gemm_f32_rows(false, false, T, Dh, T, dS.data(), Kh.data(), dQ.data(), 0, T);
            intree::gemm_f32_rows(true, false, T, Dh, T, dS.data(), Qh.data(), dK.data(), 0, T);
            for (int64_t t = 0; t < T; ++t)
                for (int64_t d = 0; d < Dh; ++d) {
                    if (Q->requires_grad) Q->grad[t * D + col + d] += dQ[t * Dh + d];
                    if (K->requires_grad) K->grad[t * D + col + d] += dK[t * Dh + d];
                    if (V->requires_grad) V->grad[t * D + col + d] += dV[t * Dh + d];
                }
        }
    };
    return out;
}

// ── Embedding: weight[V,D], ids[M] -> [M,D] ──────────────────────────────────
Var embedding(const Var& weight, const std::vector<int>& ids) {
    if (weight->shape.size() != 2) throw std::runtime_error("embedding: weight must be [V,D]");
    const int64_t V = weight->shape[0], D = weight->shape[1];
    const int64_t M = (int64_t)ids.size();
    Var out = newNode({M, D}, {weight}, weight->requires_grad);
    for (int64_t i = 0; i < M; ++i) {
        const int id = ids[i];
        if (id < 0 || id >= V) throw std::runtime_error("embedding: id out of range");
        std::memcpy(&out->data[i * D], &weight->data[(int64_t)id * D], sizeof(float) * D);
    }
    Node* op = out.get(); Var W = weight;
    std::vector<int> ids_copy = ids;
    out->backward_fn = [op, W, ids_copy, M, D]() {
        if (!W->requires_grad) return;
        for (int64_t i = 0; i < M; ++i) {
            float* dst = &W->grad[(int64_t)ids_copy[i] * D];
            const float* src = &op->grad[i * D];
            for (int64_t j = 0; j < D; ++j) dst[j] += src[j];
        }
    };
    return out;
}

// ── Fused softmax + cross-entropy over rows of logits[M,V] ───────────────────
Var softmax_cross_entropy(const Var& logits, const std::vector<int>& targets) {
    if (logits->shape.size() != 2) throw std::runtime_error("sce: logits must be [M,V]");
    const int64_t M = logits->shape[0], V = logits->shape[1];
    if ((int64_t)targets.size() != M) throw std::runtime_error("sce: targets size mismatch");
    Var out = newNode({1}, {logits}, logits->requires_grad);

    std::vector<float> probs((size_t)M * V);  // softmax, cached for backward
    double loss = 0.0;
    for (int64_t i = 0; i < M; ++i) {
        const float* row = &logits->data[i * V];
        float mx = row[0];
        for (int64_t j = 1; j < V; ++j) mx = std::max(mx, row[j]);
        float sum = 0.0f;
        for (int64_t j = 0; j < V; ++j) { float e = std::exp(row[j] - mx); probs[i * V + j] = e; sum += e; }
        const float inv = 1.0f / sum;
        for (int64_t j = 0; j < V; ++j) probs[i * V + j] *= inv;
        const int t = targets[i];
        if (t < 0 || t >= V) throw std::runtime_error("sce: target out of range");
        loss += -std::log(std::max(probs[i * V + t], 1e-30f));
    }
    out->data[0] = (float)(loss / (double)M);

    Node* op = out.get(); Var L = logits;
    std::vector<int> tgt = targets;
    out->backward_fn = [op, L, probs, tgt, M, V]() {
        if (!L->requires_grad) return;
        const float g = op->grad[0] / (float)M;  // mean reduction
        for (int64_t i = 0; i < M; ++i)
            for (int64_t j = 0; j < V; ++j) {
                float d = probs[i * V + j] - (j == tgt[i] ? 1.0f : 0.0f);
                L->grad[i * V + j] += g * d;
            }
    };
    return out;
}

Var dropout(const Var& x, float p) {
    if (p <= 0.0f) return x;                 // identity (no node added)
    if (p >= 1.0f) p = 0.999f;               // guard against div-by-zero
    Var out = newNode(x->shape, {x}, x->requires_grad);
    static thread_local std::mt19937 rng(0xD0D0);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    const int64_t n = x->size();
    const float scale = 1.0f / (1.0f - p);
    auto mask = std::make_shared<std::vector<uint8_t>>((size_t)n);
    for (int64_t i = 0; i < n; ++i) {
        const uint8_t keep = (u(rng) >= p) ? 1 : 0;
        (*mask)[i] = keep;
        out->data[i] = keep ? x->data[i] * scale : 0.0f;
    }
    Node* op = out.get(); Var X = x;
    out->backward_fn = [op, X, mask, scale, n]() {
        if (!X->requires_grad) return;
        for (int64_t i = 0; i < n; ++i)
            if ((*mask)[i]) X->grad[i] += op->grad[i] * scale;
    };
    return out;
}

Var mean(const Var& x) {
    Var out = newNode({1}, {x}, x->requires_grad);
    const int64_t n = x->size();
    double s = 0.0;
    for (int64_t i = 0; i < n; ++i) s += x->data[i];
    out->data[0] = (float)(s / (double)n);
    Node* op = out.get(); Var X = x;
    out->backward_fn = [op, X, n]() {
        if (!X->requires_grad) return;
        const float g = op->grad[0] / (float)n;
        for (int64_t i = 0; i < n; ++i) X->grad[i] += g;
    };
    return out;
}

// ── Reverse-mode engine ──────────────────────────────────────────────────────
void backward(const Var& loss) {
    if (loss->size() != 1) throw std::runtime_error("backward: loss must be scalar");

    // Reverse-topological order via iterative post-order DFS over parents.
    std::vector<Node*> order;
    std::unordered_set<Node*> visited;
    std::vector<std::pair<Node*, size_t>> stack;
    stack.push_back({loss.get(), 0});
    visited.insert(loss.get());
    while (!stack.empty()) {
        auto& [node, idx] = stack.back();
        if (idx < node->parents.size()) {
            Node* p = node->parents[idx].get();
            ++idx;
            if (p && visited.insert(p).second) stack.push_back({p, 0});
        } else {
            order.push_back(node);
            stack.pop_back();
        }
    }

    loss->grad.assign(1, 1.0f);
    for (auto it = order.rbegin(); it != order.rend(); ++it)
        if ((*it)->backward_fn) (*it)->backward_fn();
}

void zero_grad(const std::vector<Var>& params) {
    for (const auto& p : params) std::fill(p->grad.begin(), p->grad.end(), 0.0f);
}

}  // namespace autograd
}  // namespace xla
}  // namespace vgre
