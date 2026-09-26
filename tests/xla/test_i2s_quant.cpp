// BitNet I2_S ternary GGUF dequant (ggml type 36). The on-disk format (microsoft/
// BitNet i2_s spec, confirmed end-to-end on the real bitnet-b1.58-2B checkpoint):
//   scale = 1/mean(|w|);  q = round(w*scale).clamp(-1,1) ∈ {-1,0,+1};
//   codes q+1 ∈ {0,1,2} packed 4/byte MSB-first  byte = (c0<<6)|(c1<<4)|(c2<<2)|c3;
//   one trailing f32 scale;  dequant w = (code-1)/scale.
// This encodes known weights into that exact layout and checks vgre::xla's dequant
// recovers the quantized values bit-for-bit.
#undef NDEBUG

#include "vgre/xla/quant.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace vgre::xla;

static int g_fail = 0;
#define CHECK(c,m) do{ if(!(c)){ std::printf("FAIL: %s\n",(m)); ++g_fail; } }while(0)

int main() {
    // A weight vector with a clear ternary structure after absmean quantization.
    std::vector<float> w = {
        0.9f, -0.1f, 0.0f, -0.8f,  0.7f, 0.05f, -0.6f, 0.75f,
        -0.9f, 0.02f, 0.85f, -0.03f, 0.5f, -0.55f, 0.01f, 0.95f };
    const int64_t n = (int64_t)w.size();

    // Encode exactly as the I2_S spec.
    double absmean = 0; for (float v : w) absmean += std::fabs(v); absmean /= n;
    const float scale = (float)(1.0 / absmean);          // = 1/mean|w|
    std::vector<int> q(n);
    for (int64_t i = 0; i < n; ++i) {
        int r = (int)std::lround(w[i] * scale);
        q[i] = r < -1 ? -1 : r > 1 ? 1 : r;              // clamp to {-1,0,1}
    }
    std::vector<uint8_t> buf(i2sTensorBytes(n), 0);
    for (int64_t i = 0; i < n; ++i) {
        const int code = q[i] + 1;                        // {0,1,2}
        const int shift = 6 - 2 * (int)(i & 3);           // MSB-first: i%4==0 → bits 6-7
        buf[i >> 2] |= (uint8_t)(code << shift);
    }
    std::memcpy(buf.data() + i2sPackedBytes(n), &scale, 4);

    CHECK(i2sTensorBytes(n) == i2sPackedBytes(n) + 4, "tensor bytes = packed + f32 scale");
    CHECK(i2sPackedBytes(16) == 4, "16 weights pack into 4 bytes (2 bits each)");

    // Dequant via the library and check it reproduces q/scale exactly.
    std::vector<float> out(n, -999.0f);
    dequant_i2_s_tensor(buf.data(), n, out.data());
    int bad = 0;
    for (int64_t i = 0; i < n; ++i) {
        const float expect = (float)q[i] / scale;         // (code-1)/scale
        if (std::fabs(out[i] - expect) > 1e-6f) { if (bad < 4) std::printf("  i=%lld got=%.5f exp=%.5f\n", (long long)i, out[i], expect); ++bad; }
    }
    CHECK(bad == 0, "I2_S dequant recovers (code-1)/scale for every weight");

    // The recovered values are exactly ternary * absmean.
    bool ternary = true;
    for (int64_t i = 0; i < n; ++i) {
        float r = out[i] * scale;                          // back to {-1,0,1}
        if (std::fabs(r - std::lround(r)) > 1e-4f || std::fabs(r) > 1.001f) ternary = false;
    }
    CHECK(ternary, "recovered weights are ternary {-1,0,+1} * (1/scale)");
    std::printf("  I2_S: n=%lld packed=%lld bytes, scale=%.4f, dequant exact (%d bad)\n",
                (long long)n, (long long)i2sPackedBytes(n), scale, bad);

    if (g_fail == 0) std::printf("PASS: BitNet I2_S ternary GGUF dequant\n");
    else std::printf("FAILED: %d\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
