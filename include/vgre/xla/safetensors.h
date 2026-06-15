// Memory-mapped safetensors checkpoint loader for the HLO engine.
//
// safetensors (the Hugging Face weight format) is: an 8-byte little-endian
// header length, a JSON header mapping each tensor name to {dtype, shape,
// data_offsets:[begin,end]}, then one contiguous row-major data blob. This
// loader mmaps the file once (no copy of the — potentially many-GB — weights)
// and materializes a requested tensor into the engine's f32 Literal on demand,
// dequantizing F16/BF16 (and widening F64/int dtypes) as it goes. This is the
// L1 "weight loading at scale" path: feed real checkpoint weights to the engine
// as parameters instead of baking them into the program as constants.
#ifndef VGRE_XLA_SAFETENSORS_H
#define VGRE_XLA_SAFETENSORS_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "vgre/xla/hlo.h"

namespace vgre {
namespace xla {

class SafeTensors {
public:
    enum class DType { F64, F32, F16, BF16, I64, I32, I16, I8, U8, BOOL, Unknown };

    struct TensorInfo {
        DType dtype = DType::Unknown;
        std::vector<int64_t> shape;
        uint64_t begin = 0;   // byte offset into the data blob (inclusive)
        uint64_t end = 0;     // byte offset into the data blob (exclusive)
        int64_t numel() const {
            int64_t n = 1;
            for (int64_t d : shape) n *= d;
            return shape.empty() ? 1 : n;
        }
    };

    // mmap and parse `path`. Returns nullptr on any error (missing file, bad
    // magic/header, offsets out of range). Never throws.
    static std::unique_ptr<SafeTensors> open(const std::string& path);
    ~SafeTensors();

    SafeTensors(const SafeTensors&) = delete;
    SafeTensors& operator=(const SafeTensors&) = delete;

    std::vector<std::string> names() const;
    bool contains(const std::string& name) const;
    const TensorInfo* info(const std::string& name) const;

    // Materialize `name` into `out` as a row-major f32 Literal (shape preserved),
    // dequantizing F16/BF16 and widening F64/int. False if absent or malformed.
    bool load(const std::string& name, Literal& out) const;

    // Total bytes of the mapped file (header + blob).
    uint64_t mappedBytes() const { return map_len_; }

    static const char* dtypeName(DType d);

private:
    SafeTensors() = default;

    struct Entry { std::string name; TensorInfo info; };

    const uint8_t* base_ = nullptr;   // start of the mmap
    uint64_t map_len_ = 0;
    uint64_t blob_off_ = 0;           // byte offset of the data blob within the map
    std::vector<Entry> tensors_;      // header order

    // platform handle(s) for unmap
    void* os_handle_ = nullptr;       // Windows: file mapping handle; POSIX: unused
    void* os_file_ = nullptr;         // Windows: file handle; POSIX: unused
};

}  // namespace xla
}  // namespace vgre

#endif  // VGRE_XLA_SAFETENSORS_H
