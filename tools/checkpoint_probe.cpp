// checkpoint_probe — load a real safetensors/GGUF checkpoint via the VGRE
// loaders and print per-tensor stats, so they can be diffed against the
// reference libraries (safetensors / gguf). Validates the loaders on real model
// files, not just synthesized ones.
//
//   checkpoint_probe st   <file.safetensors> [tensor]
//   checkpoint_probe gguf <file.gguf>        [tensor]
//
// Output per tensor (tab-separated, machine-diffable):
//   name  dtype  shape  numel  sum  v0 v1 v2 v3 v4
#include "vgre/xla/safetensors.h"
#include "vgre/xla/gguf.h"

#include <cstdio>
#include <string>
#include <vector>

static std::string shapeStr(const std::vector<int64_t>& d) {
    std::string s = "[";
    for (size_t i = 0; i < d.size(); ++i) { s += std::to_string(d[i]); if (i + 1 < d.size()) s += ","; }
    return s + "]";
}

static void printLit(const std::string& name, const std::string& dtype,
                     const std::vector<int64_t>& shape, const vgre::xla::Literal& f32) {
    double sum = 0.0;
    for (float v : f32.data) sum += (double)v;
    std::printf("%s\t%s\t%s\t%zu\t%.6f", name.c_str(), dtype.c_str(),
                shapeStr(shape).c_str(), f32.data.size(), sum);
    for (int i = 0; i < 5 && i < (int)f32.data.size(); ++i) std::printf("\t%.6f", f32.data[i]);
    std::printf("\n");
}

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: checkpoint_probe st|gguf <file> [tensor]\n"); return 2; }
    std::string kind = argv[1], path = argv[2];
    std::string only = (argc > 3) ? argv[3] : "";

    if (kind == "st") {
        auto st = vgre::xla::SafeTensors::open(path);
        if (!st) { std::fprintf(stderr, "FAILED to open %s\n", path.c_str()); return 1; }
        for (const auto& n : st->names()) {
            if (!only.empty() && n != only) continue;
            const auto* ti = st->info(n);
            vgre::xla::Literal l;
            if (!st->load(n, l)) { std::fprintf(stderr, "load failed: %s\n", n.c_str()); continue; }
            printLit(n, vgre::xla::SafeTensors::dtypeName(ti->dtype), ti->shape, l);
        }
    } else if (kind == "gguf") {
        auto g = vgre::xla::GGUF::open(path);
        if (!g) { std::fprintf(stderr, "FAILED to open %s\n", path.c_str()); return 1; }
        std::fprintf(stderr, "# gguf v%u, arch=%s\n", g->version(),
                     g->metadataString("general.architecture").c_str());
        for (const auto& n : g->names()) {
            if (!only.empty() && n != only) continue;
            const auto* ti = g->info(n);
            vgre::xla::Literal l;
            if (!g->load(n, l)) { std::fprintf(stderr, "unsupported/failed: %s (ggml_type %d)\n",
                                                n.c_str(), ti->ggml_type); continue; }
            printLit(n, std::string("ggml") + std::to_string(ti->ggml_type), ti->shape, l);
        }
    } else {
        std::fprintf(stderr, "unknown kind '%s'\n", kind.c_str()); return 2;
    }
    return 0;
}
