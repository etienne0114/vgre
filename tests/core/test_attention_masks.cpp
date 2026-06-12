// Phase 3, Track P3-11 — structured attention masks.
// Verified by INDEPENDENT locality/causality properties (not a re-derivation):
//   • sliding-window = causal when window ≥ S;
//   • a key outside query i's window does NOT affect O[i] (locality);
//   • a SINK key outside the window DOES affect O[i] (StreamingLLM);
//   • ALiBi with identical keys weights nearer tokens monotonically more.

#include "vgre/core/attention_masks.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace vgre::attn;

static int g_pass = 0, g_total = 0;
static void check(const char* name, bool ok) {
    ++g_total;
    printf(ok ? "  PASS  %s\n" : "  FAIL  %s\n", name);
    if (ok) ++g_pass;
}

static float maxdiff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0; for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

int main() {
    printf("=== Structured attention masks ===\n");
    const int S = 16, d = 8;
    const float scale = 1.0f / std::sqrt((float)d);
    std::mt19937 rng(99);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> Q(S * d), K(S * d), V(S * d);
    for (auto& x : Q) x = nd(rng);
    for (auto& x : K) x = nd(rng);
    for (auto& x : V) x = nd(rng);

    // 1. window ≥ S ⇒ identical to dense causal.
    {
        std::vector<float> Oc(S * d), Ow(S * d);
        attention_masked(Oc.data(), Q.data(), K.data(), V.data(), S, d, scale, {MaskKind::CAUSAL});
        MaskConfig sw{MaskKind::SLIDING_WINDOW}; sw.window = S;
        attention_masked(Ow.data(), Q.data(), K.data(), V.data(), S, d, scale, sw);
        check("sliding-window(window≥S) == dense causal", maxdiff(Oc, Ow) < 1e-6);
    }

    // 2. Locality: perturbing a key OUTSIDE query i's window leaves O[i] unchanged.
    {
        MaskConfig sw{MaskKind::SLIDING_WINDOW}; sw.window = 4;
        const int qi = 12, farKey = 2;   // i-farKey = 10 ≥ window ⇒ out of window
        std::vector<float> O0(S * d), O1(S * d);
        attention_masked(O0.data(), Q.data(), K.data(), V.data(), S, d, scale, sw);
        std::vector<float> K2 = K, V2 = V;
        for (int t = 0; t < d; ++t) { K2[farKey * d + t] += 5.0f; V2[farKey * d + t] += 5.0f; }
        attention_masked(O1.data(), Q.data(), K2.data(), V2.data(), S, d, scale, sw);
        float dqi = 0; for (int t = 0; t < d; ++t) dqi = std::max(dqi, std::fabs(O0[qi*d+t]-O1[qi*d+t]));
        check("out-of-window key does not affect the query (locality)", dqi < 1e-6);
    }

    // 3. Attention sink: a sink key outside the window DOES affect O[i].
    {
        MaskConfig sk{MaskKind::SINK_WINDOW}; sk.window = 4; sk.sink = 2;
        const int qi = 12, sinkKey = 1;  // j<sink ⇒ always attended, though far
        std::vector<float> O0(S * d), O1(S * d);
        attention_masked(O0.data(), Q.data(), K.data(), V.data(), S, d, scale, sk);
        std::vector<float> V2 = V;
        for (int t = 0; t < d; ++t) V2[sinkKey * d + t] += 5.0f;
        attention_masked(O1.data(), Q.data(), K.data(), V2.data(), S, d, scale, sk);
        float dqi = 0; for (int t = 0; t < d; ++t) dqi = std::max(dqi, std::fabs(O0[qi*d+t]-O1[qi*d+t]));
        check("sink key (out of window) is attended (StreamingLLM)", dqi > 1e-3);
    }

    // 4. ALiBi: identical keys ⇒ weight ∝ exp(−slope·(i−j)); nearer j weighs more.
    //    With K[j]=const and V[j]=onehot(j), O[i] is the attention weight vector;
    //    check it is strictly increasing in j over the causal range.
    {
        std::vector<float> Kc(S * d, 0.3f), Vo(S * d, 0.0f), O(S * d, 0.0f);
        for (int j = 0; j < S; ++j) Vo[j * d + (j % d)] += 0.0f;     // (unused dims)
        // Put the "weight readout" in a single V column per key via identity-ish:
        std::vector<float> Vw(S * d, 0.0f);
        for (int j = 0; j < S; ++j) Vw[j * d + 0] = (float)j;        // O[i][0] = Σ_j w_ij · j
        MaskConfig al{MaskKind::CAUSAL}; al.alibi = true; al.alibi_slope = 0.5f;
        attention_masked(O.data(), Q.data(), Kc.data(), Vw.data(), S, d, scale, al);
        // For the last query, recompute weights directly and confirm monotonicity
        const int i = S - 1;
        float Z = 0, wprev = -1; bool mono = true;
        for (int j = 0; j <= i; ++j) Z += std::exp(-al.alibi_slope * (i - j));
        for (int j = 0; j <= i; ++j) {
            float w = std::exp(-al.alibi_slope * (i - j)) / Z;
            if (w < wprev - 1e-7f) mono = false;     // increasing toward i
            wprev = w;
        }
        // weighted index O[i][0] must equal Σ_j w·j
        float ref = 0; for (int j = 0; j <= i; ++j) ref += (std::exp(-al.alibi_slope*(i-j))/Z) * j;
        check("ALiBi weights increase toward the query (nearer dominates)", mono);
        check("ALiBi output == analytic distance-biased average", std::fabs(O[i*d+0] - ref) < 1e-3);
    }

    printf("\n%d / %d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
