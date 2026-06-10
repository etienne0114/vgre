// Track 5 — vgre_get_build_info(): version, git hash, build type, enabled
// features, and the active SIMD level. Lets operators confirm exactly what they
// are running (e.g. `vgre --version`).

#include "vgre/api/vgre_c_api.h"
#include "vgre/common/build_config.h"  // generated: version, git hash, build type

#include <string>

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
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
    if (__builtin_cpu_supports("avx512f")) return "avx512";
    if (__builtin_cpu_supports("avx2"))    return "avx2";
    if (__builtin_cpu_supports("avx"))     return "avx";
    if (__builtin_cpu_supports("sse4.2"))  return "sse4.2";
    return "baseline";
#elif defined(__aarch64__) || defined(__arm64__)
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
