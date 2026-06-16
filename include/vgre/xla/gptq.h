// GPTQ / AWQ 4-bit weight dequantization — the int4 path for Hugging Face
// *safetensors* models (distinct from GGUF). A quantized linear layer is stored
// as four tensors; this turns them back into an f32 [in_features, out_features]
// weight the engine can multiply (or quantize to GGUF, etc.).
//
// GPTQ (AutoGPTQ layout, 4-bit):
//   qweight int32 [in/8, out]   — 8 nibbles packed along `in`, lsb-first
//   qzeros  int32 [groups, out/8] — 8 nibbles packed along `out`, lsb-first
//   scales  f32   [groups, out]
//   g_idx   int32 [in]           — group index per input row (empty → in/group_size)
//   W[i,j] = scales[g,j] · (qweight[i,j] − (qzeros[g,j] + 1)),  g = g_idx[i]
//
// AWQ (4-bit): same tensors, but qweight/qzeros pack 8 nibbles along `out` with
// the AWQ order [0,2,4,6,1,3,5,7], group is i/group_size, and the zero point is
// subtracted directly (no +1):
//   W[i,j] = (qweight[i,j] − qzeros[g,j]) · scales[g,j]
#ifndef VGRE_XLA_GPTQ_H
#define VGRE_XLA_GPTQ_H

#include <cstdint>
#include <vector>

#include "vgre/xla/hlo.h"

namespace vgre {
namespace xla {

struct GptqTensors {
    std::vector<int32_t> qweight;   // [in/8, out]
    std::vector<int32_t> qzeros;    // [groups, out/8]
    std::vector<float>   scales;    // [groups, out]
    std::vector<int32_t> g_idx;     // [in]  (optional; empty → i / group_size)
    int64_t in_features = 0, out_features = 0, groups = 0;
    int bits = 4;
};

// Dequantize a GPTQ linear → f32 Literal [in_features, out_features].
Literal gptqDequantize(const GptqTensors& q);

// Dequantize an AWQ linear → f32 Literal [in_features, out_features].
// group_size = in_features / groups.
Literal awqDequantize(const GptqTensors& q);

}  // namespace xla
}  // namespace vgre

#endif  // VGRE_XLA_GPTQ_H
