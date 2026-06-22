// VGRE-LM checkpoint I/O — see include/vgre/xla/model.h.
//
// Saves the model's named parameters as a standard safetensors file: an 8-byte
// little-endian header length, a JSON header { name → {dtype,shape,
// data_offsets:[b,e]}, "__metadata__":{...} }, then the contiguous F32 blob.
// The Config lives in __metadata__. Because it is real safetensors, the existing
// vgre::xla::SafeTensors reader (and the inference path) load it unchanged —
// which is exactly how load_checkpoint() reads it back.

#include "vgre/xla/model.h"
#include "vgre/xla/safetensors.h"
#include "vgre/xla/hlo.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>

namespace vgre {
namespace xla {
namespace model {

bool save_checkpoint(GPT& model, const std::string& path) {
    auto named = model.named_parameters();
    const Config& c = model.config();

    // Build the JSON header and accumulate the F32 blob in parameter order.
    std::ostringstream hdr;
    hdr << '{';
    uint64_t off = 0;
    for (const auto& [name, p] : named) {
        const uint64_t bytes = (uint64_t)p->size() * sizeof(float);
        hdr << '"' << name << "\":{\"dtype\":\"F32\",\"shape\":[";
        for (size_t i = 0; i < p->shape.size(); ++i) {
            if (i) hdr << ',';
            hdr << p->shape[i];
        }
        hdr << "],\"data_offsets\":[" << off << ',' << (off + bytes) << "]},";
        off += bytes;
    }
    // Config in __metadata__ (safetensors requires all-string values).
    hdr << "\"__metadata__\":{"
        << "\"format\":\"vgre-lm\","
        << "\"vocab\":\""     << c.vocab     << "\","
        << "\"n_layer\":\""   << c.n_layer   << "\","
        << "\"d_model\":\""   << c.d_model   << "\","
        << "\"n_head\":\""    << c.n_head    << "\","
        << "\"d_ff\":\""      << c.ff()      << "\","
        << "\"max_seq\":\""   << c.max_seq   << "\""
        << "}}";
    std::string header = hdr.str();

    // safetensors aligns the data blob to 8 bytes by padding the header.
    while ((8 + header.size()) % 8 != 0) header.push_back(' ');

    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    const uint64_t hlen = header.size();
    out.write(reinterpret_cast<const char*>(&hlen), 8);   // LE on x86/arm64
    out.write(header.data(), (std::streamsize)header.size());
    for (const auto& [name, p] : named)
        out.write(reinterpret_cast<const char*>(p->data.data()),
                  (std::streamsize)((uint64_t)p->size() * sizeof(float)));
    return (bool)out;
}

bool load_checkpoint(GPT& model, const std::string& path) {
    auto st = SafeTensors::open(path);
    if (!st) return false;
    for (auto& [name, p] : model.named_parameters()) {
        Literal lit;
        if (!st->load(name, lit)) return false;
        if ((int64_t)lit.data.size() != p->size()) return false;  // shape/Config mismatch
        p->data = lit.data;   // overwrite parameter weights in place
    }
    return true;
}

}  // namespace model
}  // namespace xla
}  // namespace vgre
