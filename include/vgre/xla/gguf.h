// Memory-mapped GGUF checkpoint loader (llama.cpp weight format) for the HLO
// engine — the int4/int8 path of milestone L3.
//
// GGUF is: "GGUF" magic, a u32 version, tensor + metadata counts, a typed
// key/value metadata table, then per-tensor info (name, dims, ggml type, blob
// offset), an alignment pad, and the tensor data blob. This loader mmaps the
// file once and materializes a requested tensor into the engine's Literal,
// either dequantized to f32 or — with keepNative — kept at native quantized
// width (Q4_0/Q4_1 ≈ 4.5 bits/weight, ~4.5 GB for an 8B model; the engine
// dequantizes to f32 only when the op consuming it runs).
//
// Supported tensor types: F32, F16, and the block-quant formats Q8_0 / Q4_0 /
// Q4_1 (the canonical ggml layouts). Other ggml types are listed but load()
// returns false for them.
#ifndef VGRE_XLA_GGUF_H
#define VGRE_XLA_GGUF_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "vgre/xla/hlo.h"

namespace vgre {
namespace xla {

class GGUF {
public:
    struct TensorInfo {
        std::string name;
        int ggml_type = -1;            // ggml type id (0=F32,1=F16,2=Q4_0,3=Q4_1,8=Q8_0,…)
        std::vector<int64_t> shape;    // row-major (GGUF ne[] reversed)
        uint64_t offset = 0;           // byte offset within the tensor-data blob
        int64_t numel() const {
            int64_t n = 1;
            for (int64_t d : shape) n *= d;
            return shape.empty() ? 1 : n;
        }
    };

    static std::unique_ptr<GGUF> open(const std::string& path);  // nullptr on error; never throws
    ~GGUF();
    GGUF(const GGUF&) = delete;
    GGUF& operator=(const GGUF&) = delete;

    uint32_t version() const { return version_; }
    std::vector<std::string> names() const;
    const TensorInfo* info(const std::string& name) const;

    // String metadata value (e.g. "general.architecture"); empty if absent/non-string.
    std::string metadataString(const std::string& key) const;

    // Materialize `name` into `out`. Default → row-major f32 Literal (dequantized).
    // keepNative=true keeps Q*_*/F16 tensors at native width (the engine dequants
    // on use). False if absent or the ggml type is unsupported.
    bool load(const std::string& name, Literal& out, bool keepNative = false) const;

    uint64_t mappedBytes() const { return map_len_; }

private:
    GGUF() = default;

    const uint8_t* base_ = nullptr;
    uint64_t map_len_ = 0;
    uint64_t data_off_ = 0;            // byte offset of the tensor-data blob
    uint32_t version_ = 0;
    std::vector<TensorInfo> tensors_;
    std::vector<std::pair<std::string, std::string>> meta_str_;  // string metadata only

    void* os_handle_ = nullptr;        // Windows mapping handle
    void* os_file_ = nullptr;          // Windows file handle
};

}  // namespace xla
}  // namespace vgre

#endif  // VGRE_XLA_GGUF_H
