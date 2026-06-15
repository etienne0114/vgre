// Memory-mapped GGUF loader — see include/vgre/xla/gguf.h.

#include "vgre/xla/gguf.h"

#include <cstring>

#include "vgre/xla/quant.h"

#if defined(_WIN32)
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace vgre {
namespace xla {

namespace {

// GGUF metadata value-type ids.
enum GgufVT {
    VT_U8 = 0, VT_I8 = 1, VT_U16 = 2, VT_I16 = 3, VT_U32 = 4, VT_I32 = 5,
    VT_F32 = 6, VT_BOOL = 7, VT_STRING = 8, VT_ARRAY = 9, VT_U64 = 10,
    VT_I64 = 11, VT_F64 = 12
};

// Bounds-checked little-endian cursor over the mmap. Any out-of-range read sets
// ok_=false; callers check ok() and bail (→ open() returns nullptr).
struct Cursor {
    const uint8_t* p;
    const uint8_t* end;
    bool ok_ = true;

    bool ok() const { return ok_; }
    bool need(uint64_t n) {
        if (!ok_ || (uint64_t)(end - p) < n) { ok_ = false; return false; }
        return true;
    }
    template <class T> T read() {
        T v{};
        if (need(sizeof(T))) { std::memcpy(&v, p, sizeof(T)); p += sizeof(T); }
        return v;
    }
    std::string readString() {
        uint64_t n = read<uint64_t>();
        if (!need(n)) return {};
        std::string s(reinterpret_cast<const char*>(p), (size_t)n);
        p += n;
        return s;
    }
    void skipBytes(uint64_t n) { if (need(n)) p += n; }
};

int vtFixedBytes(uint32_t t) {
    switch (t) {
        case VT_U8: case VT_I8: case VT_BOOL: return 1;
        case VT_U16: case VT_I16: return 2;
        case VT_U32: case VT_I32: case VT_F32: return 4;
        case VT_U64: case VT_I64: case VT_F64: return 8;
        default: return 0;
    }
}

// Advance past one metadata value of type `t` (recurses for arrays).
void skipValue(Cursor& c, uint32_t t) {
    if (t == VT_STRING) { c.readString(); return; }
    if (t == VT_ARRAY) {
        uint32_t et = c.read<uint32_t>();
        uint64_t n = c.read<uint64_t>();
        for (uint64_t i = 0; i < n && c.ok(); ++i) skipValue(c, et);
        return;
    }
    int b = vtFixedBytes(t);
    if (b == 0) { c.ok_ = false; return; }  // unknown type → cannot advance safely
    c.skipBytes((uint64_t)b);
}

int64_t typeStorageBytes(int ggml_type, int64_t n) {
    switch (ggml_type) {
        case 0: return n * 4;                       // F32
        case 1: return n * 2;                       // F16
        case 2: case 3: case 8:                     // Q4_0 / Q4_1 / Q8_0
            return quantStorageBytes(ggml_type, n);
        default: return -1;                         // unsupported / unknown size
    }
}

}  // namespace

std::unique_ptr<GGUF> GGUF::open(const std::string& path) {
    std::unique_ptr<GGUF> g(new GGUF());

#if defined(_WIN32)
    HANDLE fh = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (fh == INVALID_HANDLE_VALUE) return nullptr;
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(fh, &sz) || sz.QuadPart < 24) { CloseHandle(fh); return nullptr; }
    HANDLE mh = CreateFileMappingA(fh, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!mh) { CloseHandle(fh); return nullptr; }
    void* addr = MapViewOfFile(mh, FILE_MAP_READ, 0, 0, 0);
    if (!addr) { CloseHandle(mh); CloseHandle(fh); return nullptr; }
    g->base_ = static_cast<const uint8_t*>(addr);
    g->map_len_ = (uint64_t)sz.QuadPart;
    g->os_handle_ = mh; g->os_file_ = fh;
#else
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return nullptr;
    struct stat sb;
    if (fstat(fd, &sb) != 0 || sb.st_size < 24) { ::close(fd); return nullptr; }
    void* addr = mmap(nullptr, (size_t)sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (addr == MAP_FAILED) return nullptr;
    g->base_ = static_cast<const uint8_t*>(addr);
    g->map_len_ = (uint64_t)sb.st_size;
#endif

    Cursor c{g->base_, g->base_ + g->map_len_};
    uint8_t magic[4];
    for (int i = 0; i < 4; ++i) magic[i] = c.read<uint8_t>();
    if (!c.ok() || std::memcmp(magic, "GGUF", 4) != 0) return nullptr;
    g->version_ = c.read<uint32_t>();
    if (g->version_ != 2 && g->version_ != 3) return nullptr;  // v1 used u32 counts
    const uint64_t n_tensors = c.read<uint64_t>();
    const uint64_t n_kv = c.read<uint64_t>();
    if (!c.ok() || n_tensors > (1u << 24) || n_kv > (1u << 24)) return nullptr;  // sanity

    // ── metadata KV (parse to advance; keep strings + general.alignment) ────
    uint32_t alignment = 32;
    for (uint64_t i = 0; i < n_kv && c.ok(); ++i) {
        std::string key = c.readString();
        uint32_t vt = c.read<uint32_t>();
        if (!c.ok()) return nullptr;
        if (vt == VT_STRING) {
            std::string v = c.readString();
            g->meta_str_.emplace_back(key, v);
        } else if (vt == VT_U32 && key == "general.alignment") {
            alignment = c.read<uint32_t>();
        } else {
            skipValue(c, vt);
        }
    }
    if (!c.ok() || alignment == 0) return nullptr;

    // ── tensor infos ────────────────────────────────────────────────────────
    g->tensors_.reserve((size_t)n_tensors);
    for (uint64_t i = 0; i < n_tensors && c.ok(); ++i) {
        TensorInfo ti;
        ti.name = c.readString();
        uint32_t nd = c.read<uint32_t>();
        if (!c.ok() || nd > 8) return nullptr;
        std::vector<int64_t> ne(nd);
        for (uint32_t d = 0; d < nd; ++d) ne[d] = (int64_t)c.read<uint64_t>();
        ti.ggml_type = (int)c.read<uint32_t>();
        ti.offset = c.read<uint64_t>();
        // GGUF stores ne[0] as the fastest dim; reverse to row-major shape.
        ti.shape.assign(ne.rbegin(), ne.rend());
        g->tensors_.push_back(std::move(ti));
    }
    if (!c.ok()) return nullptr;

    // ── data blob starts at the next `alignment` boundary after the header ──
    uint64_t pos = (uint64_t)(c.p - g->base_);
    uint64_t pad = (alignment - (pos % alignment)) % alignment;
    g->data_off_ = pos + pad;
    if (g->data_off_ > g->map_len_) return nullptr;
    const uint64_t blob = g->map_len_ - g->data_off_;

    // Validate each tensor's byte range against the blob (for known types).
    for (const auto& ti : g->tensors_) {
        int64_t bytes = typeStorageBytes(ti.ggml_type, ti.numel());
        if (bytes < 0) continue;  // unsupported type: leave listed, load() will reject
        if (ti.offset > blob || (uint64_t)bytes > blob - ti.offset) return nullptr;
    }
    return g;
}

GGUF::~GGUF() {
#if defined(_WIN32)
    if (base_) UnmapViewOfFile(const_cast<uint8_t*>(base_));
    if (os_handle_) CloseHandle((HANDLE)os_handle_);
    if (os_file_) CloseHandle((HANDLE)os_file_);
#else
    if (base_ && map_len_) munmap(const_cast<uint8_t*>(base_), (size_t)map_len_);
#endif
}

std::vector<std::string> GGUF::names() const {
    std::vector<std::string> n;
    n.reserve(tensors_.size());
    for (const auto& t : tensors_) n.push_back(t.name);
    return n;
}

const GGUF::TensorInfo* GGUF::info(const std::string& name) const {
    for (const auto& t : tensors_)
        if (t.name == name) return &t;
    return nullptr;
}

std::string GGUF::metadataString(const std::string& key) const {
    for (const auto& kv : meta_str_)
        if (kv.first == key) return kv.second;
    return {};
}

bool GGUF::load(const std::string& name, Literal& out, bool keepNative) const {
    const TensorInfo* ti = info(name);
    if (!ti) return false;
    const uint8_t* src = base_ + data_off_ + ti->offset;
    const int64_t n = ti->numel();
    out.shape.dims = ti->shape;
    out.data.clear();
    out.half.clear();
    out.quant.clear();

    const int gt = ti->ggml_type;

    if (keepNative && gt == 1) {  // F16 native
        out.dtype = DType::F16;
        out.half.resize((size_t)n);
        std::memcpy(out.half.data(), src, (size_t)n * 2);
        return true;
    }
    if (keepNative && (gt == 2 || gt == 3 || gt == 8)) {  // Q4_0 / Q4_1 / Q8_0 native
        const int64_t bytes = quantStorageBytes(gt, n);
        if (bytes <= 0) return false;
        out.dtype = (gt == 8) ? DType::Q8_0 : (gt == 2 ? DType::Q4_0 : DType::Q4_1);
        out.quant.resize((size_t)bytes);
        std::memcpy(out.quant.data(), src, (size_t)bytes);
        return true;
    }

    // Dequantize / widen to f32.
    out.dtype = DType::F32;
    out.data.resize((size_t)n);
    switch (gt) {
        case 0:  // F32
            std::memcpy(out.data.data(), src, (size_t)n * 4);
            return true;
        case 1:  // F16
            for (int64_t i = 0; i < n; ++i) {
                uint16_t h; std::memcpy(&h, src + (size_t)i * 2, 2);
                out.data[(size_t)i] = f16_to_f32(h);
            }
            return true;
        case 2: case 3: case 8:  // Q4_0 / Q4_1 / Q8_0
            return dequantBlock(gt, src, n, out.data.data());
        default:
            return false;  // unsupported ggml type
    }
}

}  // namespace xla
}  // namespace vgre
