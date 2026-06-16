// gptq_dequant — load a GPTQ/AWQ-quantized linear's tensors from a real
// safetensors file (via vgre::xla::SafeTensors::loadInt32) and dequantize to f32,
// printing sum + first values. Validates the int4-safetensors path against an
// independent reference (tools/validate_gptq.py).
//
//   gptq_dequant <file.safetensors> <prefix> <gptq|awq> <in> <out> <groups>
#include "vgre/xla/safetensors.h"
#include "vgre/xla/gptq.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 7) { std::fprintf(stderr, "usage: gptq_dequant <file> <prefix> <gptq|awq> <in> <out> <groups>\n"); return 2; }
    std::string file = argv[1], prefix = argv[2], mode = argv[3];
    auto st = vgre::xla::SafeTensors::open(file);
    if (!st) { std::fprintf(stderr, "open failed\n"); return 1; }

    vgre::xla::GptqTensors q;
    q.in_features = std::atoll(argv[4]);
    q.out_features = std::atoll(argv[5]);
    q.groups = std::atoll(argv[6]);
    q.bits = 4;

    std::vector<int64_t> sh;
    if (!st->loadInt32(prefix + "qweight", q.qweight, sh) ||
        !st->loadInt32(prefix + "qzeros", q.qzeros, sh)) {
        std::fprintf(stderr, "missing qweight/qzeros\n"); return 1;
    }
    vgre::xla::Literal scales;
    if (!st->load(prefix + "scales", scales)) { std::fprintf(stderr, "missing scales\n"); return 1; }
    q.scales = scales.data;
    std::vector<int32_t> gidx;
    if (st->loadInt32(prefix + "g_idx", gidx, sh)) q.g_idx = gidx;

    vgre::xla::Literal W = (mode == "awq") ? vgre::xla::awqDequantize(q) : vgre::xla::gptqDequantize(q);
    double sum = 0; for (float v : W.data) sum += v;
    std::printf("%zu\t%.6f", W.data.size(), sum);
    for (int i = 0; i < 5 && i < (int)W.data.size(); ++i) std::printf("\t%.6f", W.data[i]);
    std::printf("\n");
    return 0;
}
