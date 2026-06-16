// GPTQ / AWQ 4-bit dequantization — see include/vgre/xla/gptq.h.

#include "vgre/xla/gptq.h"

namespace vgre {
namespace xla {

namespace {
// AWQ packs 8 nibbles per int32 in the order [0,2,4,6,1,3,5,7]; column (8k+t)
// lives at nibble position awq_inv[t]. (Inverse of the pack order.)
constexpr int kAwqInv[8] = {0, 4, 1, 5, 2, 6, 3, 7};

inline int nib(int32_t packed, int pos) { return (int)((uint32_t)packed >> (4 * pos)) & 0xF; }
}  // namespace

Literal gptqDequantize(const GptqTensors& q) {
    const int64_t in = q.in_features, out = q.out_features, G = q.groups;
    const int64_t out8 = (out + 7) / 8;
    const int64_t group_size = (G > 0) ? (in / G) : in;

    Literal W; W.shape.dims = {in, out}; W.data.resize((size_t)(in * out));
    for (int64_t i = 0; i < in; ++i) {
        const int64_t g = q.g_idx.empty() ? (i / group_size) : (int64_t)q.g_idx[(size_t)i];
        for (int64_t j = 0; j < out; ++j) {
            // qweight nibble: packed along `in` (8 rows per int32), lsb-first.
            const int32_t qwp = q.qweight[(size_t)((i / 8) * out + j)];
            const int qw = nib(qwp, (int)(i % 8));
            // qzero nibble: packed along `out`, +1 (AutoGPTQ convention).
            const int32_t qzp = q.qzeros[(size_t)(g * out8 + j / 8)];
            const int z = nib(qzp, (int)(j % 8)) + 1;
            const float s = q.scales[(size_t)(g * out + j)];
            W.data[(size_t)(i * out + j)] = s * (float)(qw - z);
        }
    }
    return W;
}

Literal awqDequantize(const GptqTensors& q) {
    const int64_t in = q.in_features, out = q.out_features, G = q.groups;
    const int64_t out8 = (out + 7) / 8;
    const int64_t group_size = (G > 0) ? (in / G) : in;

    Literal W; W.shape.dims = {in, out}; W.data.resize((size_t)(in * out));
    for (int64_t i = 0; i < in; ++i) {
        const int64_t g = i / group_size;
        for (int64_t j = 0; j < out; ++j) {
            const int pos = kAwqInv[j % 8];                       // AWQ nibble order
            const int qw = nib(q.qweight[(size_t)(i * out8 + j / 8)], pos);
            const int z  = nib(q.qzeros[(size_t)(g * out8 + j / 8)], pos);
            const float s = q.scales[(size_t)(g * out + j)];
            W.data[(size_t)(i * out + j)] = (float)(qw - z) * s;  // no +1
        }
    }
    return W;
}

}  // namespace xla
}  // namespace vgre
