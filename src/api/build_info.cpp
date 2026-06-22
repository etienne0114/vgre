// Track 5 — vgre_get_build_info(): version, git hash, build type, enabled
// features, and the active SIMD level. Lets operators confirm exactly what they
// are running (e.g. `vgre --version`).

#include "vgre/api/vgre_c_api.h"
#include "vgre/common/build_config.h"  // generated: version, git hash, build type

#include <string>
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#  include <intrin.h>
#endif

#ifndef VGRE_VERSION_STRING
#define VGRE_VERSION_STRING "0.0.0"
#endif
#ifndef VGRE_GIT_HASH
#define VGRE_GIT_HASH "unknown"
#endif
#ifndef VGRE_BUILD_TYPE
#define VGRE_BUILD_TYPE "unknown"
#endif

namespace {

const char *simdLevel() {
#if (defined(__x86_64__) || defined(__i386__)) && !defined(_MSC_VER)
    // GCC/Clang: runtime detection via __builtin_cpu_supports
    if (__builtin_cpu_supports("avx512f")) return "avx512";
    if (__builtin_cpu_supports("avx2"))    return "avx2";
    if (__builtin_cpu_supports("avx"))     return "avx";
    if (__builtin_cpu_supports("sse4.2"))  return "sse4.2";
    return "baseline";
#elif defined(_M_X64) || defined(_M_IX86)
    // MSVC x86/x64: runtime detection via __cpuid
    int regs[4];
    __cpuid(regs, 1);
    bool hasSSE42   = (regs[2] & (1 << 20)) != 0;
    bool hasAVX     = (regs[2] & (1 << 28)) != 0;
    bool hasOSXSAVE = (regs[2] & (1 << 27)) != 0;
    bool avxOk      = hasAVX && hasOSXSAVE;
    bool hasAVX2    = false;
    bool hasAVX512F = false;
    if (avxOk) {
        int ext[4];
        __cpuid(ext, 7);
        hasAVX2    = (ext[1] & (1 << 5))  != 0;
        hasAVX512F = (ext[1] & (1 << 16)) != 0;
    }
    if (avxOk && hasAVX512F) return "avx512";
    if (avxOk && hasAVX2)    return "avx2";
    if (avxOk)               return "avx";
    if (hasSSE42)            return "sse4.2";
    return "baseline";
#elif defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64)
    return "neon";
#else
    return "scalar";
#endif
}

std::string buildInfoJson() {
    auto b = [](bool v) { return v ? "true" : "false"; };
    bool has_sqlite =
#ifdef VGRE_HAS_SQLITE
        true;
#else
        false;
#endif
    bool has_grpc =
#if defined(VGRE_ENABLE_GRPC) || defined(ENABLE_VGRE_GRPC)
        true;
#else
        false;
#endif
    bool has_rdma =
#if defined(VGRE_ENABLE_RDMA) || defined(ENABLE_VGRE_RDMA)
        true;
#else
        false;
#endif
    bool has_fusion =
#ifdef ENABLE_VGRE_KERNEL_FUSION
        true;
#else
        false;
#endif

    std::string s = "{";
    s += "\"version\":\"" VGRE_VERSION_STRING "\",";
    s += "\"git\":\"" VGRE_GIT_HASH "\",";
    s += "\"build_type\":\"" VGRE_BUILD_TYPE "\",";
    s += std::string("\"simd\":\"") + simdLevel() + "\",";
    s += "\"features\":{";
    s += std::string("\"sqlite\":") + b(has_sqlite) + ",";
    s += std::string("\"grpc\":") + b(has_grpc) + ",";
    s += std::string("\"rdma\":") + b(has_rdma) + ",";
    s += std::string("\"kernel_fusion\":") + b(has_fusion);
    s += "}}";
    return s;
}

} // namespace

extern "C" const char *vgre_get_build_info(void) {
    // Built once, returned as a stable C string for FFI consumers.
    static const std::string info = buildInfoJson();
    return info.c_str();
}
