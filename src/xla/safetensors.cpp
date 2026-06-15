// Memory-mapped safetensors loader — see include/vgre/xla/safetensors.h.

#include "vgre/xla/safetensors.h"

#include <cstring>

#include "vgre/common/json.h"

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

SafeTensors::DType parseDType(const std::string& s) {
    using D = SafeTensors::DType;
    if (s == "F64") return D::F64;
    if (s == "F32") return D::F32;
    if (s == "F16") return D::F16;
    if (s == "BF16") return D::BF16;
    if (s == "I64") return D::I64;
    if (s == "I32") return D::I32;
    if (s == "I16") return D::I16;
    if (s == "I8") return D::I8;
    if (s == "U8") return D::U8;
    if (s == "BOOL") return D::BOOL;
    return D::Unknown;
}

int dtypeBytes(SafeTensors::DType d) {
    using D = SafeTensors::DType;
    switch (d) {
        case D::F64: case D::I64: return 8;
        case D::F32: case D::I32: return 4;
        case D::F16: case D::BF16: case D::I16: return 2;
        case D::I8: case D::U8: case D::BOOL: return 1;
        default: return 0;
    }
}

template <class T>
void widen(const uint8_t* src, int64_t n, std::vector<float>& out) {
    out.resize((size_t)n);
    T v;
    for (int64_t i = 0; i < n; ++i) {
        std::memcpy(&v, src + (size_t)i * sizeof(T), sizeof(T));
        out[(size_t)i] = (float)v;
    }
}

}  // namespace

const char* SafeTensors::dtypeName(DType d) {
    switch (d) {
        case DType::F64: return "F64";
        case DType::F32: return "F32";
        case DType::F16: return "F16";
        case DType::BF16: return "BF16";
        case DType::I64: return "I64";
        case DType::I32: return "I32";
        case DType::I16: return "I16";
        case DType::I8: return "I8";
        case DType::U8: return "U8";
        case DType::BOOL: return "BOOL";
        default: return "Unknown";
    }
}

std::unique_ptr<SafeTensors> SafeTensors::open(const std::string& path) {
    std::unique_ptr<SafeTensors> st(new SafeTensors());

    // ── mmap the whole file read-only ──────────────────────────────────────
#if defined(_WIN32)
    HANDLE fh = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (fh == INVALID_HANDLE_VALUE) return nullptr;
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(fh, &sz) || sz.QuadPart < 8) { CloseHandle(fh); return nullptr; }
    HANDLE mh = CreateFileMappingA(fh, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!mh) { CloseHandle(fh); return nullptr; }
    void* addr = MapViewOfFile(mh, FILE_MAP_READ, 0, 0, 0);
    if (!addr) { CloseHandle(mh); CloseHandle(fh); return nullptr; }
    st->base_ = static_cast<const uint8_t*>(addr);
    st->map_len_ = (uint64_t)sz.QuadPart;
    st->os_handle_ = mh;
    st->os_file_ = fh;
#else
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return nullptr;
    struct stat sb;
    if (fstat(fd, &sb) != 0 || sb.st_size < 8) { ::close(fd); return nullptr; }
    void* addr = mmap(nullptr, (size_t)sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);  // mapping survives close
    if (addr == MAP_FAILED) return nullptr;
    st->base_ = static_cast<const uint8_t*>(addr);
    st->map_len_ = (uint64_t)sb.st_size;
#endif

    // ── header length (u64 LE) + JSON header ───────────────────────────────
    uint64_t header_len = 0;
    std::memcpy(&header_len, st->base_, 8);  // safetensors is little-endian on disk
    if (8 + header_len > st->map_len_ || header_len == 0) return nullptr;
    st->blob_off_ = 8 + header_len;

    std::string header(reinterpret_cast<const char*>(st->base_ + 8), (size_t)header_len);
    common::json::Value root;
    if (!common::json::parse(header, root) || !root.isObject()) return nullptr;

    const uint64_t blob_size = st->map_len_ - st->blob_off_;
    for (const auto& kv : root.obj) {
        if (kv.first == "__metadata__") continue;  // free-form string map; skip
        const common::json::Value& t = kv.second;
        if (!t.isObject()) return nullptr;
        const common::json::Value* dt = t.find("dtype");
        const common::json::Value* shp = t.find("shape");
        const common::json::Value* off = t.find("data_offsets");
        if (!dt || !dt->isString() || !off || !off->isArray() || off->arr.size() != 2)
            return nullptr;

        Entry e;
        e.name = kv.first;
        e.info.dtype = parseDType(dt->asString());
        if (e.info.dtype == DType::Unknown) return nullptr;
        if (shp && shp->isArray())
            for (const auto& d : shp->arr) e.info.shape.push_back((int64_t)d.asNumber());
        e.info.begin = off->arr[0].asUint64();
        e.info.end = off->arr[1].asUint64();

        // Validate: ordered, within blob, and byte count matches shape·dtype.
        if (e.info.end < e.info.begin || e.info.end > blob_size) return nullptr;
        const uint64_t bytes = e.info.end - e.info.begin;
        const int esz = dtypeBytes(e.info.dtype);
        if (esz == 0 || bytes != (uint64_t)e.info.numel() * (uint64_t)esz) return nullptr;

        st->tensors_.push_back(std::move(e));
    }
    return st;
}

SafeTensors::~SafeTensors() {
#if defined(_WIN32)
    if (base_) UnmapViewOfFile(const_cast<uint8_t*>(base_));
    if (os_handle_) CloseHandle((HANDLE)os_handle_);
    if (os_file_) CloseHandle((HANDLE)os_file_);
#else
    if (base_ && map_len_) munmap(const_cast<uint8_t*>(base_), (size_t)map_len_);
#endif
}

std::vector<std::string> SafeTensors::names() const {
    std::vector<std::string> n;
    n.reserve(tensors_.size());
    for (const auto& e : tensors_) n.push_back(e.name);
    return n;
}

bool SafeTensors::contains(const std::string& name) const { return info(name) != nullptr; }

const SafeTensors::TensorInfo* SafeTensors::info(const std::string& name) const {
    for (const auto& e : tensors_)
        if (e.name == name) return &e.info;
    return nullptr;
}

bool SafeTensors::load(const std::string& name, Literal& out, bool keepNative) const {
    const TensorInfo* ti = info(name);
    if (!ti) return false;
    const uint8_t* src = base_ + blob_off_ + ti->begin;
    const int64_t n = ti->numel();

    out.shape.dims = ti->shape;

    // Native-width path: keep BF16/F16 as 16-bit codes (half the RAM); the engine
    // dequantizes to f32 only when the tensor is consumed.
    if (keepNative && (ti->dtype == DType::BF16 || ti->dtype == DType::F16)) {
        out.data.clear();
        out.dtype = (ti->dtype == DType::BF16) ? vgre::xla::DType::BF16 : vgre::xla::DType::F16;
        out.half.resize((size_t)n);
        std::memcpy(out.half.data(), src, (size_t)n * 2);
        return true;
    }
    out.dtype = vgre::xla::DType::F32;
    out.half.clear();

    switch (ti->dtype) {
        case DType::F32:
            out.data.resize((size_t)n);
            std::memcpy(out.data.data(), src, (size_t)n * 4);
            break;
        case DType::F64: widen<double>(src, n, out.data); break;
        case DType::I64: widen<int64_t>(src, n, out.data); break;
        case DType::I32: widen<int32_t>(src, n, out.data); break;
        case DType::I16: widen<int16_t>(src, n, out.data); break;
        case DType::I8:  widen<int8_t>(src, n, out.data); break;
        case DType::U8:
        case DType::BOOL: widen<uint8_t>(src, n, out.data); break;
        case DType::BF16: {
            out.data.resize((size_t)n);
            for (int64_t i = 0; i < n; ++i) {
                uint16_t h;
                std::memcpy(&h, src + (size_t)i * 2, 2);
                out.data[(size_t)i] = bf16_to_f32(h);
            }
            break;
        }
        case DType::F16: {
            out.data.resize((size_t)n);
            for (int64_t i = 0; i < n; ++i) {
                uint16_t h;
                std::memcpy(&h, src + (size_t)i * 2, 2);
                out.data[(size_t)i] = f16_to_f32(h);
            }
            break;
        }
        default:
            return false;
    }
    return true;
}

}  // namespace xla
}  // namespace vgre
