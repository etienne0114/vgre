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

// ── linear_tied: out[M,V] = x[M,D] · Wᵀ, W stored [V,D] (weight tying) ───────
Var linear_tied(const Var& x, const Var& w) {
    if (x->shape.size() != 2 || w->shape.size() != 2)
        throw std::runtime_error("linear_tied: operands must be rank-2");
    const int64_t M = x->shape[0], D = x->shape[1], V = w->shape[0];
    if (w->shape[1] != D) throw std::runtime_error("linear_tied: dim mismatch");

    Var out = newNode({M, V}, {x, w}, anyReq({&x, &w}));
    // out[m,v] = Σ_d x[m,d]·W[v,d]  → gemm(tA=F, tB=T, M, V, D, x, W)
    intree::gemm_f32_threaded(false, true, M, V, D, x->data.data(), w->data.data(), out->data.data());

    Node* op = out.get();
    Var X = x, W = w;
    out->backward_fn = [op, X, W, M, V, D]() {
        // dX[M,D] += dOut[M,V] · W[V,D]   → gemm(F, F, M, D, V, dOut, W)
        if (X->requires_grad) {
            std::vector<float> dX((size_t)M * D, 0.0f);
            intree::gemm_f32_threaded(false, false, M, D, V, op->grad.data(), W->data.data(), dX.data());
            for (size_t i = 0; i < dX.size(); ++i) X->grad[i] += dX[i];
        }
        // dW[V,D] += dOutᵀ[V,M] · X[M,D]  → gemm(T, F, V, D, M, dOut, X)
        if (W->requires_grad) {
            std::vector<float> dW((size_t)V * D, 0.0f);
            intree::gemm_f32_threaded(true, false, V, D, M, op->grad.data(), X->data.data(), dW.data());
            for (size_t i = 0; i < dW.size(); ++i) W->grad[i] += dW[i];
        }
    };
    return out;
}

// ── bmm: batched matmul a[B,M,K] · b[B,K,N] -> [B,M,N] ───────────────────────
Var bmm(const Var& a, const Var& b) {
    if (a->shape.size() != 3 || b->shape.size() != 3)
        throw std::runtime_error("bmm: operands must be rank-3 [B,M,K]·[B,K,N]");
    const int64_t B = a->shape[0], M = a->shape[1], K = a->shape[2];
    const int64_t N = b->shape[2];
    if (b->shape[0] != B || b->shape[1] != K)
        throw std::runtime_error("bmm: batch/inner dim mismatch");

    Var out = newNode({B, M, N}, {a, b}, anyReq({&a, &b}));
    for (int64_t bi = 0; bi < B; ++bi)
        intree::gemm_f32_threaded(false, false, M, N, K,
                                  a->data.data() + bi * M * K,
                                  b->data.data() + bi * K * N,
                                  out->data.data() + bi * M * N);

    Node* op = out.get();
    Var A = a, Bv = b;
    out->backward_fn = [op, A, Bv, B, M, N, K]() {
        for (int64_t bi = 0; bi < B; ++bi) {
            const float* dC = op->grad.data() + bi * M * N;
            if (A->requires_grad) {
                // dA[M,K] += dC[M,N] · Bᵀ  → gemm(F, T, M, K, N, dC, B)
                std::vector<float> dA((size_t)M * K, 0.0f);
                intree::gemm_f32_threaded(false, true, M, K, N, dC,
                                          Bv->data.data() + bi * K * N, dA.data());
                float* g = A->grad.data() + bi * M * K;
                for (size_t i = 0; i < dA.size(); ++i) g[i] += dA[i];
            }
            if (Bv->requires_grad) {
                // dB[K,N] += Aᵀ · dC[M,N]  → gemm(T, F, K, N, M, A, dC)
                std::vector<float> dB((size_t)K * N, 0.0f);
                intree::gemm_f32_threaded(true, false, K, N, M,
                                          A->data.data() + bi * M * K, dC, dB.data());
                float* g = Bv->grad.data() + bi * K * N;
                for (size_t i = 0; i < dB.size(); ++i) g[i] += dB[i];
            }
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

// Exact GELU: 0.5·x·(1 + erf(x/√2)).
//   d/dx = 0.5·(1 + erf(x/√2)) + x·(1/√(2π))·exp(-x²/2)
Var gelu(const Var& x) {
    Var out = newNode(x->shape, {x}, x->requires_grad);
    const int64_t n = x->size();
    const float kInvSqrt2  = 0.70710678118654752440f;   // 1/√2
    const float kInvSqrt2Pi = 0.39894228040143267794f;  // 1/√(2π)
    for (int64_t i = 0; i < n; ++i) {
        const float v = x->data[i];
        out->data[i] = 0.5f * v * (1.0f + std::erf(v * kInvSqrt2));
    }
    Node* op = out.get(); Var X = x;
    out->backward_fn = [op, X, n, kInvSqrt2, kInvSqrt2Pi]() {
        if (!X->requires_grad) return;
        for (int64_t i = 0; i < n; ++i) {
            const float v = X->data[i];
            const float cdf = 0.5f * (1.0f + std::erf(v * kInvSqrt2));
            const float pdf = kInvSqrt2Pi * std::exp(-0.5f * v * v);
            X->grad[i] += op->grad[i] * (cdf + v * pdf);
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

Var sigmoid(const Var& x) {
    Var out = newNode(x->shape, {x}, x->requires_grad);
    const int64_t n = x->size();
    for (int64_t i = 0; i < n; ++i) out->data[i] = 1.0f / (1.0f + std::exp(-x->data[i]));
    Node* op = out.get(); Var X = x;
    out->backward_fn = [op, X, n]() {
        if (!X->requires_grad) return;
        for (int64_t i = 0; i < n; ++i) { const float s = op->data[i]; X->grad[i] += op->grad[i] * s * (1.0f - s); }
    };
    return out;
}

Var tanh_(const Var& x) {
    Var out = newNode(x->shape, {x}, x->requires_grad);
    const int64_t n = x->size();
    for (int64_t i = 0; i < n; ++i) out->data[i] = std::tanh(x->data[i]);
    Node* op = out.get(); Var X = x;
    out->backward_fn = [op, X, n]() {
        if (!X->requires_grad) return;
        for (int64_t i = 0; i < n; ++i) { const float t = op->data[i]; X->grad[i] += op->grad[i] * (1.0f - t * t); }
    };
    return out;
}

// ── RMSNorm over last dim of x[M,D]; y = x / rms(x) * weight ──────────────────
Var rms_norm(const Var& x, const Var& weight, float eps) {
    if (x->shape.size() != 2 && x->shape.size() != 3)
        throw std::runtime_error("rms_norm: x must be rank-2 [M,D] or rank-3 [B,T,D]");
    const int64_t D = x->shape.back();          // normalize over the last dim
    const int64_t M = x->size() / D;            // leading dims fold into rows
    if (weight->shape.size() != 1 || weight->shape[0] != D)
        throw std::runtime_error("rms_norm: weight must be [D]");
    Var out = newNode(x->shape, {x, weight}, anyReq({&x, &weight}));
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
    if (x->shape.size() != 2 && x->shape.size() != 3)
        throw std::runtime_error("layer_norm: x must be rank-2 [M,D] or rank-3 [B,T,D]");
    const int64_t D = x->shape.back();
    const int64_t M = x->size() / D;
    if (weight->shape.size() != 1 || weight->shape[0] != D ||
        bias->shape.size() != 1 || bias->shape[0] != D)
        throw std::runtime_error("layer_norm: weight/bias must be [D]");
    Var out = newNode(x->shape, {x, weight, bias}, anyReq({&x, &weight, &bias}));
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

// ── RoPE on x[T,H*Dh] or [B,T,H*Dh]: rotate dim-pairs by angle pos·base^(-2i/Dh)
Var rope(const Var& x, int num_heads, float base) {
    if (x->shape.size() != 2 && x->shape.size() != 3)
        throw std::runtime_error("rope: x must be [T,H*Dh] or [B,T,H*Dh]");
    const bool batched = x->shape.size() == 3;
    const int64_t B = batched ? x->shape[0] : 1;
    const int64_t T = x->shape[batched ? 1 : 0];
    const int64_t D = x->shape[batched ? 2 : 1];
    const int64_t Dh = D / num_heads;
    if (Dh * num_heads != D || (Dh & 1)) throw std::runtime_error("rope: bad head_dim");
    Var out = newNode(x->shape, {x}, x->requires_grad);
    const int64_t half = Dh / 2;
    for (int64_t b = 0; b < B; ++b)
        for (int64_t t = 0; t < T; ++t)
            for (int h = 0; h < num_heads; ++h) {
                const int64_t off = (b * T + t) * D + (int64_t)h * Dh;
                for (int64_t i = 0; i < half; ++i) {
                    const float theta = (float)t * std::pow(base, -2.0f * (float)i / (float)Dh);
                    const float cs = std::cos(theta), sn = std::sin(theta);
                    const float a = x->data[off + 2 * i], bb = x->data[off + 2 * i + 1];
                    out->data[off + 2 * i]     = a * cs - bb * sn;
                    out->data[off + 2 * i + 1] = a * sn + bb * cs;
                }
            }
    Node* op = out.get(); Var X = x;
    out->backward_fn = [op, X, B, T, D, Dh, num_heads, half, base]() {
        if (!X->requires_grad) return;
        for (int64_t b = 0; b < B; ++b)
        for (int64_t t = 0; t < T; ++t)
            for (int h = 0; h < num_heads; ++h) {
                const int64_t off = (b * T + t) * D + (int64_t)h * Dh;
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
// Accepts rank-2 [T, H*Dh] (single sequence) or rank-3 [B, T, H*Dh] (batched);
// the batch dim folds into an outer loop so the per-(batch,head) math is shared.
Var attention(const Var& q, const Var& k, const Var& v, int num_heads, bool causal) {
    if ((q->shape.size() != 2 && q->shape.size() != 3) || q->shape != k->shape || q->shape != v->shape)
        throw std::runtime_error("attention: Q/K/V must share shape [T,H*Dh] or [B,T,H*Dh]");
    const bool batched = q->shape.size() == 3;
    const int64_t B  = batched ? q->shape[0] : 1;
    const int64_t T  = q->shape[batched ? 1 : 0];
    const int64_t D  = q->shape[batched ? 2 : 1];
    const int64_t Dh = D / num_heads;
    if (Dh * num_heads != D) throw std::runtime_error("attention: bad head_dim");
    const float scale = 1.0f / std::sqrt((float)Dh);

    Var out = newNode(q->shape, {q, k, v}, anyReq({&q, &k, &v}));
    // Per (batch,head) softmax probs cached for backward.
    auto P = std::make_shared<std::vector<float>>((size_t)B * num_heads * T * T, 0.0f);

    std::vector<float> Qh(T * Dh), Kh(T * Dh), Vh(T * Dh), S(T * T), Oh(T * Dh);
    for (int64_t b = 0; b < B; ++b)
    for (int h = 0; h < num_heads; ++h) {
        const int64_t col = b * T * D + (int64_t)h * Dh;     // base offset into [B,T,D]
        for (int64_t t = 0; t < T; ++t)
            for (int64_t d = 0; d < Dh; ++d) {
                Qh[t * Dh + d] = q->data[col + t * D + d];
                Kh[t * Dh + d] = k->data[col + t * D + d];
                Vh[t * Dh + d] = v->data[col + t * D + d];
            }
        intree::gemm_f32_rows(false, true, T, T, Dh, Qh.data(), Kh.data(), S.data(), 0, T);
        float* Ph = P->data() + ((size_t)b * num_heads + h) * T * T;
        for (int64_t i = 0; i < T; ++i) {
            const int64_t lim = causal ? i : (T - 1);
            float mx = -1e30f;
            for (int64_t j = 0; j <= lim; ++j) { S[i * T + j] *= scale; mx = std::max(mx, S[i * T + j]); }
            float sum = 0.0f;
            for (int64_t j = 0; j <= lim; ++j) { float e = std::exp(S[i * T + j] - mx); Ph[i * T + j] = e; sum += e; }
            const float invs = 1.0f / sum;
            for (int64_t j = 0; j <= lim; ++j) Ph[i * T + j] *= invs;
            for (int64_t j = lim + 1; j < T; ++j) Ph[i * T + j] = 0.0f;
        }
        intree::gemm_f32_rows(false, false, T, Dh, T, Ph, Vh.data(), Oh.data(), 0, T);
        for (int64_t t = 0; t < T; ++t)
            for (int64_t d = 0; d < Dh; ++d) out->data[col + t * D + d] = Oh[t * Dh + d];
    }

    Node* op = out.get(); Var Q = q, K = k, V = v;
    out->backward_fn = [op, Q, K, V, P, B, T, D, Dh, num_heads, scale]() {
        std::vector<float> Qh(T * Dh), Kh(T * Dh), Vh(T * Dh), dO(T * Dh);
        std::vector<float> dV(T * Dh), dP(T * T), dS(T * T), dQ(T * Dh), dK(T * Dh);
        for (int64_t b = 0; b < B; ++b)
        for (int h = 0; h < num_heads; ++h) {
            const int64_t col = b * T * D + (int64_t)h * Dh;
            const float* Ph = P->data() + ((size_t)b * num_heads + h) * T * T;
            for (int64_t t = 0; t < T; ++t)
                for (int64_t d = 0; d < Dh; ++d) {
                    Qh[t * Dh + d] = Q->data[col + t * D + d];
                    Kh[t * Dh + d] = K->data[col + t * D + d];
                    Vh[t * Dh + d] = V->data[col + t * D + d];
                    dO[t * Dh + d] = op->grad[col + t * D + d];
                }
            intree::gemm_f32_rows(true, false, T, Dh, T, Ph, dO.data(), dV.data(), 0, T);
            intree::gemm_f32_rows(false, true, T, T, Dh, dO.data(), Vh.data(), dP.data(), 0, T);
            for (int64_t i = 0; i < T; ++i) {
                float dot = 0.0f;
                for (int64_t j = 0; j < T; ++j) dot += Ph[i * T + j] * dP[i * T + j];
                for (int64_t j = 0; j < T; ++j)
                    dS[i * T + j] = Ph[i * T + j] * (dP[i * T + j] - dot) * scale;
            }
            intree::gemm_f32_rows(false, false, T, Dh, T, dS.data(), Kh.data(), dQ.data(), 0, T);
            intree::gemm_f32_rows(true, false, T, Dh, T, dS.data(), Qh.data(), dK.data(), 0, T);
            for (int64_t t = 0; t < T; ++t)
                for (int64_t d = 0; d < Dh; ++d) {
                    if (Q->requires_grad) Q->grad[col + t * D + d] += dQ[t * Dh + d];
                    if (K->requires_grad) K->grad[col + t * D + d] += dK[t * Dh + d];
                    if (V->requires_grad) V->grad[col + t * D + d] += dV[t * Dh + d];
                }
        }
    };
    return out;
}

// ── Flash (online-softmax) attention ─────────────────────────────────────────
// Same math as attention() but stores only O[T,Dh] and the per-row logsumexp
// L[T] per head (O(T) extra) instead of the full T×T probabilities; the backward
// recomputes scores on the fly (FA2-style, using D_i = Σ_d dO[i,d]·O[i,d]).
Var flash_attention(const Var& q, const Var& k, const Var& v, int num_heads, bool causal) {
    if (q->shape.size() != 2 || q->shape != k->shape || q->shape != v->shape)
        throw std::runtime_error("flash_attention: Q/K/V must share shape [T, H*Dh]");
    const int64_t T = q->shape[0], Dm = q->shape[1];
    const int64_t Dh = Dm / num_heads;
    if (Dh * num_heads != Dm) throw std::runtime_error("flash_attention: bad head_dim");
    const float scale = 1.0f / std::sqrt((float)Dh);

    Var out = newNode({T, Dm}, {q, k, v}, anyReq({&q, &k, &v}));
    // Per-head logsumexp L[T], cached for the backward recompute.
    auto Lcache = std::make_shared<std::vector<float>>((size_t)num_heads * T, 0.0f);

    const float* Q = q->data.data();
    const float* K = k->data.data();
    const float* V = v->data.data();
    for (int h = 0; h < num_heads; ++h) {
        const int64_t col = (int64_t)h * Dh;
        float* L = Lcache->data() + (size_t)h * T;
        for (int64_t i = 0; i < T; ++i) {
            const int64_t lim = causal ? i : (T - 1);
            const float* qi = Q + i * Dm + col;
            float m = -1e30f, l = 0.0f;
            std::vector<float> acc((size_t)Dh, 0.0f);
            for (int64_t j = 0; j <= lim; ++j) {
                const float* kj = K + j * Dm + col;
                float s = 0.0f;
                for (int64_t d = 0; d < Dh; ++d) s += qi[d] * kj[d];
                s *= scale;
                const float m_new = std::max(m, s);
                const float corr = std::exp(m - m_new);
                const float p = std::exp(s - m_new);
                l = l * corr + p;
                const float* vj = V + j * Dm + col;
                for (int64_t d = 0; d < Dh; ++d) acc[d] = acc[d] * corr + p * vj[d];
                m = m_new;
            }
            const float inv = (l > 0.0f) ? 1.0f / l : 0.0f;
            float* oi = out->data.data() + i * Dm + col;
            for (int64_t d = 0; d < Dh; ++d) oi[d] = acc[d] * inv;
            L[i] = m + std::log(l > 0.0f ? l : 1.0f);   // logsumexp
        }
    }

    Node* op = out.get(); Var Qv = q, Kv = k, Vv = v;
    out->backward_fn = [op, Qv, Kv, Vv, Lcache, T, Dm, Dh, num_heads, scale, causal]() {
        const float* Q = Qv->data.data();
        const float* K = Kv->data.data();
        const float* V = Vv->data.data();
        const float* O = op->data.data();
        const float* dO = op->grad.data();
        for (int h = 0; h < num_heads; ++h) {
            const int64_t col = (int64_t)h * Dh;
            const float* L = Lcache->data() + (size_t)h * T;
            for (int64_t i = 0; i < T; ++i) {
                const int64_t lim = causal ? i : (T - 1);
                const float* qi = Q + i * Dm + col;
                const float* dOi = dO + i * Dm + col;
                const float* Oi = O + i * Dm + col;
                float Di = 0.0f;                          // FA2: D_i = Σ_d dO·O
                for (int64_t d = 0; d < Dh; ++d) Di += dOi[d] * Oi[d];
                for (int64_t j = 0; j <= lim; ++j) {
                    const float* kj = K + j * Dm + col;
                    const float* vj = V + j * Dm + col;
                    float s = 0.0f, dp = 0.0f;
                    for (int64_t d = 0; d < Dh; ++d) { s += qi[d] * kj[d]; dp += dOi[d] * vj[d]; }
                    const float p  = std::exp(s * scale - L[i]);   // P[i,j]
                    const float ds = p * (dp - Di) * scale;        // dS[i,j]
                    if (Qv->requires_grad) { float* dq = Qv->grad.data() + i * Dm + col; for (int64_t d = 0; d < Dh; ++d) dq[d] += ds * kj[d]; }
                    if (Kv->requires_grad) { float* dk = Kv->grad.data() + j * Dm + col; for (int64_t d = 0; d < Dh; ++d) dk[d] += ds * qi[d]; }
                    if (Vv->requires_grad) { float* dv = Vv->grad.data() + j * Dm + col; for (int64_t d = 0; d < Dh; ++d) dv[d] += p * dOi[d]; }
                }
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

// ── softmax / transpose / concat ─────────────────────────────────────────────
Var softmax(const Var& x) {
    // Softmax over the LAST dim; accepts rank-2 [M,N] or rank-3 [B,M,N] (the
    // leading dims fold into the row count M since each row of N is contiguous).
    if (x->shape.size() != 2 && x->shape.size() != 3)
        throw std::runtime_error("softmax: x must be rank-2 [M,N] or rank-3 [B,M,N]");
    const int64_t N = x->shape.back();
    const int64_t M = x->size() / N;
    Var out = newNode(x->shape, {x}, x->requires_grad);
    for (int64_t i = 0; i < M; ++i) {
        const float* r = &x->data[i * N];
        float mx = r[0]; for (int64_t j = 1; j < N; ++j) mx = std::max(mx, r[j]);
        float s = 0.0f;
        for (int64_t j = 0; j < N; ++j) { float e = std::exp(r[j] - mx); out->data[i * N + j] = e; s += e; }
        const float inv = 1.0f / s;
        for (int64_t j = 0; j < N; ++j) out->data[i * N + j] *= inv;
    }
    Node* op = out.get(); Var X = x;
    out->backward_fn = [op, X, M, N]() {
        if (!X->requires_grad) return;
        for (int64_t i = 0; i < M; ++i) {
            const float* p = &op->data[i * N];
            const float* dy = &op->grad[i * N];
            float dot = 0.0f; for (int64_t j = 0; j < N; ++j) dot += p[j] * dy[j];
            for (int64_t j = 0; j < N; ++j) X->grad[i * N + j] += p[j] * (dy[j] - dot);
        }
    };
    return out;
}

Var transpose(const Var& x) {
    // rank-2: [M,N] -> [N,M].  rank-3: [B,M,N] -> [B,N,M] (swap the last two dims,
    // per batch — what batched bilinear / attention need).
    if (x->shape.size() == 2) {
        const int64_t M = x->shape[0], N = x->shape[1];
        Var out = newNode({N, M}, {x}, x->requires_grad);
        for (int64_t i = 0; i < M; ++i)
            for (int64_t j = 0; j < N; ++j) out->data[j * M + i] = x->data[i * N + j];
        Node* op = out.get(); Var X = x;
        out->backward_fn = [op, X, M, N]() {
            if (!X->requires_grad) return;
            for (int64_t i = 0; i < M; ++i)
                for (int64_t j = 0; j < N; ++j) X->grad[i * N + j] += op->grad[j * M + i];
        };
        return out;
    }
    if (x->shape.size() == 3) {
        const int64_t B = x->shape[0], M = x->shape[1], N = x->shape[2];
        Var out = newNode({B, N, M}, {x}, x->requires_grad);
        for (int64_t b = 0; b < B; ++b)
            for (int64_t i = 0; i < M; ++i)
                for (int64_t j = 0; j < N; ++j)
                    out->data[(b * N + j) * M + i] = x->data[(b * M + i) * N + j];
        Node* op = out.get(); Var X = x;
        out->backward_fn = [op, X, B, M, N]() {
            if (!X->requires_grad) return;
            for (int64_t b = 0; b < B; ++b)
                for (int64_t i = 0; i < M; ++i)
                    for (int64_t j = 0; j < N; ++j)
                        X->grad[(b * M + i) * N + j] += op->grad[(b * N + j) * M + i];
        };
        return out;
    }
    throw std::runtime_error("transpose: x must be rank-2 [M,N] or rank-3 [B,M,N]");
}

Var concat(const Var& a, const Var& b, int axis) {
    if (a->shape.size() != 2 || b->shape.size() != 2) throw std::runtime_error("concat: rank-2 only");
    const int64_t Ma = a->shape[0], Na = a->shape[1], Mb = b->shape[0], Nb = b->shape[1];
    if (axis == 0) {
        if (Na != Nb) throw std::runtime_error("concat axis 0: col mismatch");
        Var out = newNode({Ma + Mb, Na}, {a, b}, anyReq({&a, &b}));
        std::copy(a->data.begin(), a->data.end(), out->data.begin());
        std::copy(b->data.begin(), b->data.end(), out->data.begin() + a->data.size());
        Node* op = out.get(); Var A = a, B = b;
        out->backward_fn = [op, A, B]() {
            if (A->requires_grad) for (size_t i = 0; i < A->grad.size(); ++i) A->grad[i] += op->grad[i];
            if (B->requires_grad) for (size_t i = 0; i < B->grad.size(); ++i) B->grad[i] += op->grad[A->grad.size() + i];
        };
        return out;
    } else {  // axis 1 (columns)
        if (Ma != Mb) throw std::runtime_error("concat axis 1: row mismatch");
        const int64_t M = Ma, N = Na + Nb;
        Var out = newNode({M, N}, {a, b}, anyReq({&a, &b}));
        for (int64_t i = 0; i < M; ++i) {
            for (int64_t j = 0; j < Na; ++j) out->data[i * N + j] = a->data[i * Na + j];
            for (int64_t j = 0; j < Nb; ++j) out->data[i * N + Na + j] = b->data[i * Nb + j];
        }
        Node* op = out.get(); Var A = a, B = b;
        out->backward_fn = [op, A, B, M, N, Na, Nb]() {
            for (int64_t i = 0; i < M; ++i) {
                if (A->requires_grad) for (int64_t j = 0; j < Na; ++j) A->grad[i * Na + j] += op->grad[i * N + j];
                if (B->requires_grad) for (int64_t j = 0; j < Nb; ++j) B->grad[i * Nb + j] += op->grad[i * N + Na + j];
            }
        };
        return out;
    }
}

// ── Vision: conv2d (im2col + GEMM), max_pool2d, reshape ──────────────────────
namespace {
// im2col for one image: input[Ci,H,W] → cols[Ci*Kh*Kw, Ho*Wo].
void im2col(const float* in, int Ci, int H, int W, int Kh, int Kw,
            int stride, int pad, int Ho, int Wo, float* cols) {
    const int K = Ci * Kh * Kw, P = Ho * Wo;
    for (int ci = 0; ci < Ci; ++ci)
        for (int kh = 0; kh < Kh; ++kh)
            for (int kw = 0; kw < Kw; ++kw) {
                const int row = (ci * Kh + kh) * Kw + kw;
                for (int ho = 0; ho < Ho; ++ho) {
                    const int h = ho * stride + kh - pad;
                    for (int wo = 0; wo < Wo; ++wo) {
                        const int w = wo * stride + kw - pad;
                        const bool in_b = (h >= 0 && h < H && w >= 0 && w < W);
                        cols[(int64_t)row * P + ho * Wo + wo] =
                            in_b ? in[((int64_t)ci * H + h) * W + w] : 0.0f;
                    }
                }
                (void)K;
            }
}
// col2im: scatter-add cols[Ci*Kh*Kw, Ho*Wo] back into dInput[Ci,H,W].
void col2im(const float* cols, int Ci, int H, int W, int Kh, int Kw,
            int stride, int pad, int Ho, int Wo, float* dIn) {
    const int P = Ho * Wo;
    for (int ci = 0; ci < Ci; ++ci)
        for (int kh = 0; kh < Kh; ++kh)
            for (int kw = 0; kw < Kw; ++kw) {
                const int row = (ci * Kh + kh) * Kw + kw;
                for (int ho = 0; ho < Ho; ++ho) {
                    const int h = ho * stride + kh - pad;
                    if (h < 0 || h >= H) continue;
                    for (int wo = 0; wo < Wo; ++wo) {
                        const int w = wo * stride + kw - pad;
                        if (w < 0 || w >= W) continue;
                        dIn[((int64_t)ci * H + h) * W + w] += cols[(int64_t)row * P + ho * Wo + wo];
                    }
                }
            }
}
}  // namespace

Var conv2d(const Var& input, const Var& weight, const Var& bias, int stride, int pad) {
    if (input->shape.size() != 4 || weight->shape.size() != 4)
        throw std::runtime_error("conv2d: input[N,Ci,H,W], weight[Co,Ci,Kh,Kw]");
    const int N = (int)input->shape[0], Ci = (int)input->shape[1];
    const int H = (int)input->shape[2], W = (int)input->shape[3];
    const int Co = (int)weight->shape[0], Kh = (int)weight->shape[2], Kw = (int)weight->shape[3];
    if ((int)weight->shape[1] != Ci) throw std::runtime_error("conv2d: channel mismatch");
    const bool hasBias = bias && bias->size() == Co;
    const int Ho = (H + 2 * pad - Kh) / stride + 1;
    const int Wo = (W + 2 * pad - Kw) / stride + 1;
    if (Ho <= 0 || Wo <= 0) throw std::runtime_error("conv2d: output size <= 0");
    const int K = Ci * Kh * Kw, P = Ho * Wo;

    std::vector<Var> parents = {input, weight};
    if (hasBias) parents.push_back(bias);
    Var out = newNode({N, Co, Ho, Wo}, parents,
                      anyReq({&input, &weight}) || (hasBias && bias->requires_grad));

    std::vector<float> cols((size_t)K * P);
    for (int n = 0; n < N; ++n) {
        im2col(input->data.data() + (int64_t)n * Ci * H * W, Ci, H, W, Kh, Kw, stride, pad, Ho, Wo, cols.data());
        // out[n][Co,P] = weight[Co,K] · cols[K,P]
        intree::gemm_f32_threaded(false, false, Co, P, K,
                                  weight->data.data(), cols.data(),
                                  out->data.data() + (int64_t)n * Co * P);
        if (hasBias)
            for (int co = 0; co < Co; ++co) {
                float* o = out->data.data() + ((int64_t)n * Co + co) * P;
                const float b = bias->data[co];
                for (int p = 0; p < P; ++p) o[p] += b;
            }
    }

    Node* op = out.get();
    Var In = input, Wt = weight, Bs = hasBias ? bias : Var{};
    out->backward_fn = [op, In, Wt, Bs, N, Ci, H, W, Co, Kh, Kw, stride, pad, Ho, Wo, K, P]() {
        std::vector<float> cols((size_t)K * P);
        for (int n = 0; n < N; ++n) {
            const float* dout = op->grad.data() + (int64_t)n * Co * P;   // [Co,P]
            im2col(In->data.data() + (int64_t)n * Ci * H * W, Ci, H, W, Kh, Kw, stride, pad, Ho, Wo, cols.data());
            if (Wt->requires_grad) {
                // dW[Co,K] += dout[Co,P] · colsᵀ[P,K]
                std::vector<float> dW((size_t)Co * K, 0.0f);
                intree::gemm_f32_threaded(false, true, Co, K, P, dout, cols.data(), dW.data());
                for (size_t i = 0; i < dW.size(); ++i) Wt->grad[i] += dW[i];
            }
            if (In->requires_grad) {
                // dcols[K,P] = Wtᵀ[K,Co] · dout[Co,P]; then col2im → dInput
                std::vector<float> dcols((size_t)K * P, 0.0f);
                intree::gemm_f32_threaded(true, false, K, P, Co, Wt->data.data(), dout, dcols.data());
                col2im(dcols.data(), Ci, H, W, Kh, Kw, stride, pad, Ho, Wo,
                       In->grad.data() + (int64_t)n * Ci * H * W);
            }
            if (Bs && Bs->requires_grad)
                for (int co = 0; co < Co; ++co) {
                    float s = 0.0f;
                    for (int p = 0; p < P; ++p) s += dout[(int64_t)co * P + p];
                    Bs->grad[co] += s;
                }
        }
    };
    return out;
}

Var max_pool2d(const Var& input, int kernel, int stride) {
    if (input->shape.size() != 4) throw std::runtime_error("max_pool2d: input[N,C,H,W]");
    const int N = (int)input->shape[0], C = (int)input->shape[1];
    const int H = (int)input->shape[2], W = (int)input->shape[3];
    const int Ho = (H - kernel) / stride + 1, Wo = (W - kernel) / stride + 1;
    if (Ho <= 0 || Wo <= 0) throw std::runtime_error("max_pool2d: output size <= 0");
    Var out = newNode({N, C, Ho, Wo}, {input}, input->requires_grad);
    auto argmax = std::make_shared<std::vector<int64_t>>((size_t)N * C * Ho * Wo);  // flat input index
    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c) {
            const float* in = input->data.data() + ((int64_t)n * C + c) * H * W;
            for (int ho = 0; ho < Ho; ++ho)
                for (int wo = 0; wo < Wo; ++wo) {
                    float best = -1e30f; int64_t bi = 0;
                    for (int kh = 0; kh < kernel; ++kh)
                        for (int kw = 0; kw < kernel; ++kw) {
                            const int h = ho * stride + kh, w = wo * stride + kw;
                            const int64_t idx = (int64_t)h * W + w;
                            if (in[idx] > best) { best = in[idx]; bi = idx; }
                        }
                    const int64_t o = (((int64_t)n * C + c) * Ho + ho) * Wo + wo;
                    out->data[o] = best;
                    (*argmax)[o] = ((int64_t)n * C + c) * H * W + bi;   // flat input index
                }
        }
    Node* op = out.get(); Var In = input;
    out->backward_fn = [op, In, argmax]() {
        if (!In->requires_grad) return;
        for (size_t o = 0; o < argmax->size(); ++o) In->grad[(*argmax)[o]] += op->grad[o];
    };
    return out;
}

Var avg_pool2d(const Var& input, int kernel, int stride) {
    if (input->shape.size() != 4) throw std::runtime_error("avg_pool2d: input[N,C,H,W]");
    const int N = (int)input->shape[0], C = (int)input->shape[1];
    const int H = (int)input->shape[2], W = (int)input->shape[3];
    const int Ho = (H - kernel) / stride + 1, Wo = (W - kernel) / stride + 1;
    if (Ho <= 0 || Wo <= 0) throw std::runtime_error("avg_pool2d: output size <= 0");
    Var out = newNode({N, C, Ho, Wo}, {input}, input->requires_grad);
    const float invK = 1.0f / (float)(kernel * kernel);
    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c) {
            const float* in = input->data.data() + ((int64_t)n * C + c) * H * W;
            float* o = out->data.data() + ((int64_t)n * C + c) * Ho * Wo;
            for (int ho = 0; ho < Ho; ++ho)
                for (int wo = 0; wo < Wo; ++wo) {
                    float s = 0.0f;
                    for (int kh = 0; kh < kernel; ++kh)
                        for (int kw = 0; kw < kernel; ++kw)
                            s += in[(int64_t)(ho * stride + kh) * W + (wo * stride + kw)];
                    o[ho * Wo + wo] = s * invK;
                }
        }
    Node* op = out.get(); Var In = input;
    out->backward_fn = [op, In, N, C, H, W, Ho, Wo, kernel, stride, invK]() {
        if (!In->requires_grad) return;
        for (int n = 0; n < N; ++n)
            for (int c = 0; c < C; ++c) {
                float* dIn = In->grad.data() + ((int64_t)n * C + c) * H * W;
                const float* dO = op->grad.data() + ((int64_t)n * C + c) * Ho * Wo;
                for (int ho = 0; ho < Ho; ++ho)
                    for (int wo = 0; wo < Wo; ++wo) {
                        const float g = dO[ho * Wo + wo] * invK;
                        for (int kh = 0; kh < kernel; ++kh)
                            for (int kw = 0; kw < kernel; ++kw)
                                dIn[(int64_t)(ho * stride + kh) * W + (wo * stride + kw)] += g;
                    }
            }
    };
    return out;
}

// Batch norm over [N,C,H,W], per channel. Training uses batch stats and updates
// the running buffers (EMA); eval uses the running buffers.
Var batch_norm2d(const Var& x, const Var& gamma, const Var& beta,
                 std::vector<float>& running_mean, std::vector<float>& running_var,
                 bool training, float momentum, float eps) {
    if (x->shape.size() != 4) throw std::runtime_error("batch_norm2d: x[N,C,H,W]");
    const int N = (int)x->shape[0], C = (int)x->shape[1];
    const int H = (int)x->shape[2], W = (int)x->shape[3];
    if ((int)gamma->shape[0] != C || (int)beta->shape[0] != C)
        throw std::runtime_error("batch_norm2d: gamma/beta must be [C]");
    if ((int)running_mean.size() != C) { running_mean.assign(C, 0.0f); running_var.assign(C, 1.0f); }
    const int64_t M = (int64_t)N * H * W;   // elements per channel

    Var out = newNode(x->shape, {x, gamma, beta}, anyReq({&x, &gamma, &beta}));
    auto inv = std::make_shared<std::vector<float>>(C);    // 1/√(var+eps) per channel
    auto xhat = std::make_shared<std::vector<float>>(x->data.size());

    for (int c = 0; c < C; ++c) {
        float mean, var;
        if (training) {
            double sum = 0.0;
            for (int n = 0; n < N; ++n) {
                const float* xc = x->data.data() + ((int64_t)n * C + c) * H * W;
                for (int64_t i = 0; i < (int64_t)H * W; ++i) sum += xc[i];
            }
            mean = (float)(sum / (double)M);
            double v = 0.0;
            for (int n = 0; n < N; ++n) {
                const float* xc = x->data.data() + ((int64_t)n * C + c) * H * W;
                for (int64_t i = 0; i < (int64_t)H * W; ++i) { double d = xc[i] - mean; v += d * d; }
            }
            var = (float)(v / (double)M);
            running_mean[c] = (1.0f - momentum) * running_mean[c] + momentum * mean;
            running_var[c]  = (1.0f - momentum) * running_var[c]  + momentum * var;
        } else {
            mean = running_mean[c]; var = running_var[c];
        }
        (*inv)[c] = 1.0f / std::sqrt(var + eps);
        for (int n = 0; n < N; ++n) {
            const float* xc = x->data.data() + ((int64_t)n * C + c) * H * W;
            float* oc = out->data.data() + ((int64_t)n * C + c) * H * W;
            for (int64_t i = 0; i < (int64_t)H * W; ++i) {
                const float xh = (xc[i] - mean) * (*inv)[c];
                (*xhat)[((int64_t)n * C + c) * H * W + i] = xh;
                oc[i] = gamma->data[c] * xh + beta->data[c];
            }
        }
    }

    Node* op = out.get(); Var X = x, G = gamma, B = beta;
    out->backward_fn = [op, X, G, B, inv, xhat, N, C, H, W, M, training]() {
        const int64_t HW = (int64_t)H * W;
        for (int c = 0; c < C; ++c) {
            // dGamma = Σ dy·xhat ; dBeta = Σ dy
            double dg = 0.0, db = 0.0;
            for (int n = 0; n < N; ++n) {
                const float* dy = op->grad.data() + ((int64_t)n * C + c) * HW;
                const float* xh = xhat->data() + ((int64_t)n * C + c) * HW;
                for (int64_t i = 0; i < HW; ++i) { dg += (double)dy[i] * xh[i]; db += dy[i]; }
            }
            if (G->requires_grad) G->grad[c] += (float)dg;
            if (B->requires_grad) B->grad[c] += (float)db;
            if (!X->requires_grad) continue;
            // dx = (gamma·inv/M)·(M·dy − Σdy − xhat·Σ(dy·xhat))   [batch stats]
            // In eval mode batch stats are constant ⇒ dx = gamma·inv·dy.
            const float gi = G->data[c] * (*inv)[c];
            if (!training) {
                for (int n = 0; n < N; ++n) {
                    const float* dy = op->grad.data() + ((int64_t)n * C + c) * HW;
                    float* dx = X->grad.data() + ((int64_t)n * C + c) * HW;
                    for (int64_t i = 0; i < HW; ++i) dx[i] += gi * dy[i];
                }
            } else {
                const float s = gi / (float)M;
                for (int n = 0; n < N; ++n) {
                    const float* dy = op->grad.data() + ((int64_t)n * C + c) * HW;
                    const float* xh = xhat->data() + ((int64_t)n * C + c) * HW;
                    float* dx = X->grad.data() + ((int64_t)n * C + c) * HW;
                    for (int64_t i = 0; i < HW; ++i)
                        dx[i] += s * ((float)M * dy[i] - (float)db - xh[i] * (float)dg);
                }
            }
        }
    };
    return out;
}

Var reshape(const Var& x, std::vector<int64_t> shape) {
    int64_t n = 1; for (int64_t d : shape) n *= d;
    if (n != x->size()) throw std::runtime_error("reshape: size mismatch");
    Var out = newNode(shape, {x}, x->requires_grad);
    out->data = x->data;                       // same values, new shape
    Node* op = out.get(); Var X = x;
    out->backward_fn = [op, X]() {
        if (!X->requires_grad) return;
        for (size_t i = 0; i < X->grad.size(); ++i) X->grad[i] += op->grad[i];
    };
    return out;
}

// ── Reverse-mode engine ──────────────────────────────────────────────────────
namespace {
// Run backprop from `root` (its grad already seeded) in reverse-topo order.
// Traversal does not descend into `stops` and does not call their backward_fn —
// they are treated as the segment's leaves (gradients accumulate INTO them).
// This backs both the full backward() (empty stops) and checkpoint's bounded
// recompute (stops = the checkpointed segment's inputs).
void runBackward(Node* root, const std::unordered_set<Node*>& stops) {
    std::vector<Node*> order;
    std::unordered_set<Node*> visited;
    std::vector<std::pair<Node*, size_t>> stack;
    stack.push_back({root, 0});
    visited.insert(root);
    while (!stack.empty()) {
        auto& [node, idx] = stack.back();
        const bool isStop = (node != root) && stops.count(node);
        if (!isStop && idx < node->parents.size()) {
            Node* p = node->parents[idx].get();
            ++idx;
            if (p && visited.insert(p).second) stack.push_back({p, 0});
        } else {
            order.push_back(node);
            stack.pop_back();
        }
    }
    for (auto it = order.rbegin(); it != order.rend(); ++it)
        if (!stops.count(*it) && (*it)->backward_fn) (*it)->backward_fn();
}
}  // namespace

void backward(const Var& loss) {
    if (loss->size() != 1) throw std::runtime_error("backward: loss must be scalar");
    loss->grad.assign(1, 1.0f);
    runBackward(loss.get(), /*stops=*/{});
}

// ── Gradient checkpointing ───────────────────────────────────────────────────
// Run `fn(inputs)` but DROP the segment's intermediate activations after forward;
// recompute them in backward (trading compute for memory). Output matches
// fn(inputs) exactly; gradients are identical. The recompute backprops only
// within the segment, accumulating into the real `inputs`' grads.
Var checkpoint(const std::function<Var(const std::vector<Var>&)>& fn,
               const std::vector<Var>& inputs) {
    Var fwd = fn(inputs);                 // forward value; this tape is discarded below
    // The output must require grad so upstream ops fill its grad: the recompute
    // back-props into ALL leaves that require grad inside fn — including params
    // captured by fn that are not in `inputs` (each leaf still gates on its own
    // requires_grad, so non-grad leaves are unaffected).
    Var out = newNode(fwd->shape, inputs, /*requires_grad=*/true);
    out->data = fwd->data;                // keep ONLY the output activation
    // `fwd` (and the segment's intermediates) are released when this returns.

    Node* op = out.get();
    std::vector<Var> ins = inputs;
    auto fnCopy = fn;
    out->backward_fn = [op, fnCopy, ins]() {
        Var re = fnCopy(ins);             // recompute the segment (fresh tape)
        re->grad = op->grad;              // seed with the incoming gradient
        std::unordered_set<Node*> stops;
        for (auto& in : ins) stops.insert(in.get());
        runBackward(re.get(), stops);     // backprop within the segment → ins' grads
    };
    return out;
}

void zero_grad(const std::vector<Var>& params) {
    for (const auto& p : params) std::fill(p->grad.begin(), p->grad.end(), 0.0f);
}

}  // namespace autograd
}  // namespace xla
}  // namespace vgre
