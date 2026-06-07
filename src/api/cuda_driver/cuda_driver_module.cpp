// CUDA Driver API — cuda driver module

#include "cuda_driver_internal.h"
#include "vgre/api/fatbinary_utils.h"
#include "vgre/common/platform.h"
#include "../../compiler/sass/sass_decoder.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_set>
#include <vector>

extern "C" {

extern "C" void *vgre_register_module_data(const void *data, size_t size);
extern "C" bool vgre_unregister_module_data(void *handle);
extern "C" const char *vgre_get_module_source(void *handle);
extern "C" void *vgre_lookup_symbol(void *handle, const char *name, size_t *size);
extern "C" void *vgre_lookup_texture_ref(void *handle, const char *name);

VGRE_PUBLIC_API
CUresult cuModuleLoadData(CUmodule *module, const void *image) {
    if (!module || !image) return CUDA_ERROR_INVALID_VALUE;

    const uint32_t *header = reinterpret_cast<const uint32_t *>(image);
    const uint32_t ELF_MAGIC     = 0x464c457fu;
    const uint32_t FATBIN_MAGIC  = 0xba55ed50u;
    const uint32_t FATBIN_OLD    = 0xba55ed01u;
    const uint32_t WRAPPER_MAGIC = 0x466243b1u; // __fatBinC_Wrapper_t

    size_t len = 0;
    if (header[0] == ELF_MAGIC) {
        vgre::common::ELFReader reader(image, 1024 * 1024);
        len = reader.getTotalSize();
        if (len == 0) len = 8 * 1024 * 1024;
    } else if (header[0] == FATBIN_MAGIC || header[0] == FATBIN_OLD) {
        // Fatbinary container: read total size from the 16-byte container header.
        // Header layout: magic(4) + version(2) + headerSize(2) + fatSize(8).
        struct FbHdr { uint32_t magic; uint16_t version; uint16_t headerSize; uint64_t fatSize; };
        const auto* fh = reinterpret_cast<const FbHdr*>(image);
        len = static_cast<size_t>(fh->headerSize) + static_cast<size_t>(fh->fatSize);
        if (len < 16 || len > 256 * 1024 * 1024u) len = 256 * 1024; // sanity cap
    } else if (header[0] == WRAPPER_MAGIC) {
        // __fatBinC_Wrapper_t: follow data pointer at bytes 8-15.
        const uint64_t* wrap = reinterpret_cast<const uint64_t*>(image);
        const uint8_t* inner = reinterpret_cast<const uint8_t*>(
            static_cast<uintptr_t>(wrap[1]));
        if (inner) {
            struct FbHdr { uint32_t magic; uint16_t version; uint16_t headerSize; uint64_t fatSize; };
            const auto* fh = reinterpret_cast<const FbHdr*>(inner);
            if (fh->magic == FATBIN_MAGIC || fh->magic == FATBIN_OLD)
                len = static_cast<size_t>(fh->headerSize) + static_cast<size_t>(fh->fatSize);
        }
        if (len == 0) len = 4 * 1024 * 1024;
        len = std::max(len, (size_t)32);
    } else {
        // Plain PTX text (null-terminated)
        len = strlen(reinterpret_cast<const char *>(image));
    }

    if (len == 0) return CUDA_ERROR_INVALID_VALUE;
    void *h = vgre_register_module_data(image, len);
    if (!h) return CUDA_ERROR_INVALID_VALUE;
    *module = reinterpret_cast<CUmodule>(h);
    return CUDA_SUCCESS;
}

VGRE_PUBLIC_API
CUresult cuModuleLoadDataEx(CUmodule *module, const void *image,
                            unsigned int numOptions, int *options, void **optionValues) {
  (void)numOptions;
  (void)options;
  (void)optionValues;
  return cuModuleLoadData(module, image);
}

CUresult cuModuleGetGlobal(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod, const char *name) {
  if (!dptr || !hmod || !name) return CUDA_ERROR_INVALID_VALUE;
  void *ptr = vgre_lookup_symbol(hmod, name, bytes);
  if (!ptr) return CUDA_ERROR_UNKNOWN;
  *dptr = ptr;
  return CUDA_SUCCESS;
}

VGRE_PUBLIC_API
CUresult cuModuleLoad(CUmodule *module, const char *fname) {
  if (!module || !fname) return CUDA_ERROR_INVALID_VALUE;
  auto err = vgre::api::CUDAInterceptor::instance().moduleLoad(module, fname);
  return toCU(err);
}

CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name) {
  if (!hfunc || !hmod || !name) return CUDA_ERROR_INVALID_VALUE;
  // If module is from memory PTX, compile source and register kernel on the fly.
  const char *src = vgre_get_module_source(hmod);
  if (src && strlen(src) > 0) {
    // SASS-only binary: no PTX is available for JIT compilation.
    static const char kSassMarker[] = "__VGRE_SASS_ONLY__";
    if (std::strncmp(src, kSassMarker, sizeof(kSassMarker) - 1) == 0) {
      return CUDA_ERROR_NO_BINARY_FOR_GPU; // 209
    }
    vgre::KernelId id = 0;
    auto r = vgre::core::RuntimeEngine::instance().registerKernel(name, src, id);
    if (r != vgre::VGREResult::SUCCESS) return CUDA_ERROR_INVALID_PTX;
    *hfunc = static_cast<CUfunction>(id);
    return CUDA_SUCCESS;
  }

  auto err = vgre::api::CUDAInterceptor::instance().moduleGetFunction(hfunc, hmod, name);
  return toCU(err);
}

CUresult cuModuleUnload(CUmodule hmod) {
  if (!hmod) return CUDA_ERROR_INVALID_VALUE;
  if (vgre_unregister_module_data(hmod)) {
    return CUDA_SUCCESS;
  }
  auto err = vgre::api::CUDAInterceptor::instance().moduleUnload(hmod);
  return toCU(err);
}

CUresult cuLaunchKernel(CUfunction f,
                        unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                        unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                        unsigned int sharedMemBytes, CUstream hStream,
                        void **kernelParams, void **extra) {
  (void)extra;
  if (f == 0) return CUDA_ERROR_INVALID_VALUE;
  vgre::dim3 grid(gridDimX, gridDimY, gridDimZ);
  vgre::dim3 block(blockDimX, blockDimY, blockDimZ);
  auto r = vgre::core::RuntimeEngine::instance().launchKernel(
      static_cast<vgre::KernelId>(f), grid, block, kernelParams, sharedMemBytes, hStream);
  return (r == vgre::VGREResult::SUCCESS) ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
}

CUresult cuLaunchCooperativeKernel(CUfunction f,
                                   unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                                   unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                                   unsigned int sharedMemBytes, CUstream hStream,
                                   void **kernelParams) {
  // In CPU emulation cooperative launch behaves identically to regular launch
  // because all blocks are already scheduled by the runtime engine.
  return cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ,
                        blockDimX, blockDimY, blockDimZ,
                        sharedMemBytes, hStream, kernelParams, nullptr);
}

CUresult cuLaunchCooperativeKernelMultiDevice(
    void *launchParamsList, unsigned int numDevices, unsigned int flags) {
  (void)flags;
  if (!launchParamsList || numDevices == 0) return CUDA_ERROR_INVALID_VALUE;
  // VGRE is single-device emulation; launch only on device 0.
  // The launchParamsList is an array of CUDA_LAUNCH_PARAMS structs.
  // For CPU emulation we ignore the multi-device aspect and launch sequentially.
  struct CudaLaunchParams {
    void *func;
    unsigned int gridDimX, gridDimY, gridDimZ;
    unsigned int blockDimX, blockDimY, blockDimZ;
    unsigned int sharedMemBytes;
    void *hStream;
    void **kernelParams;
  };
  auto *params = static_cast<CudaLaunchParams*>(launchParamsList);
  for (unsigned int i = 0; i < numDevices; ++i) {
    CUresult r = cuLaunchCooperativeKernel(
        reinterpret_cast<CUfunction>(params[i].func),
        params[i].gridDimX, params[i].gridDimY, params[i].gridDimZ,
        params[i].blockDimX, params[i].blockDimY, params[i].blockDimZ,
        params[i].sharedMemBytes,
        reinterpret_cast<CUstream>(params[i].hStream),
        params[i].kernelParams);
    if (r != CUDA_SUCCESS) return r;
  }
  return CUDA_SUCCESS;
}

// ── cuLaunchKernelEx ──────────────────────────────────────────────────────────
// Extended kernel launch with a config struct (CUDA 11.3+).
// CUlaunchConfig extends cuLaunchKernel with cluster dims and attributes;
// in VGRE's CPU model we ignore cluster/attribute extensions and delegate
// to the standard cuLaunchKernel path.
struct CUlaunchConfig {
    unsigned int gridDimX;
    unsigned int gridDimY;
    unsigned int gridDimZ;
    unsigned int blockDimX;
    unsigned int blockDimY;
    unsigned int blockDimZ;
    unsigned int sharedMemBytes;
    CUstream     hStream;
    void*        attrs;         // CUlaunchAttribute array — ignored in CPU model
    unsigned int numAttrs;
};

CUresult cuLaunchKernelEx(const CUlaunchConfig *config, CUfunction f,
                          void **kernelParams, void **extra) {
    if (!config || f == 0) return CUDA_ERROR_INVALID_VALUE;
    return cuLaunchKernel(f,
        config->gridDimX, config->gridDimY, config->gridDimZ,
        config->blockDimX, config->blockDimY, config->blockDimZ,
        config->sharedMemBytes, config->hStream,
        kernelParams, extra);
}

// ── cuModuleLoadFatBinary ─────────────────────────────────────────────────────
// Fatbinary format wraps multiple PTX/SASS cubins for different SM targets.
// In VGRE's CPU model there are no GPU hardware cubins, so we extract the
// embedded PTX (if any) and register it, or succeed silently if the fatbinary
// contains only SASS (which VGRE cannot execute — the caller will call
// cuModuleGetFunction which will return CUDA_ERROR_INVALID_PTX for SASS-only).
CUresult cuModuleLoadFatBinary(CUmodule *module, const void *fatCubin) {
    if (!module || !fatCubin) return CUDA_ERROR_INVALID_VALUE;
    // Attempt to treat the fatbinary as a raw PTX/ELF image
    // (the most common case in framework-generated code).
    CUresult r = cuModuleLoadData(module, fatCubin);
    if (r == CUDA_SUCCESS) return CUDA_SUCCESS;
    // If not valid PTX/ELF: allocate a sentinel handle so callers don't crash,
    // and return success.  cuModuleGetFunction will handle the lookup.
    static unsigned long long g_sentinel_counter = 0;
    *module = reinterpret_cast<CUmodule>(
                  static_cast<uintptr_t>(0xFAB10000ULL + (++g_sentinel_counter)));
    return CUDA_SUCCESS;
}

// ── cuModuleLink* ─────────────────────────────────────────────────────────────
// Module linker APIs are used to link PTX or device object files at runtime.
// VGRE does not have a CUDA linker; we provide stubs that collect PTX data
// and finish by creating a module via cuModuleLoadData.

struct CUlinkStateImpl {
    std::vector<std::vector<uint8_t>> ptxBuffers;
};

CUresult cuLinkCreate(unsigned int numOptions, int *options, void **optionValues,
                      CUlinkState *stateOut) {
    (void)numOptions; (void)options; (void)optionValues;
    if (!stateOut) return CUDA_ERROR_INVALID_VALUE;
    *stateOut = reinterpret_cast<CUlinkState>(new CUlinkStateImpl());
    return CUDA_SUCCESS;
}

CUresult cuLinkCreate_v2(unsigned int numOptions, int *options, void **optionValues,
                         CUlinkState *stateOut) {
    return cuLinkCreate(numOptions, options, optionValues, stateOut);
}

CUresult cuLinkAddData(CUlinkState state, int type, void *data, size_t size,
                       const char * /*name*/, unsigned int /*numOptions*/,
                       int * /*options*/, void ** /*optionValues*/) {
    if (!state || !data || size == 0) return CUDA_ERROR_INVALID_VALUE;
    auto *ls = reinterpret_cast<CUlinkStateImpl*>(state);

    // CU_JIT_INPUT_PTX (1): raw PTX text bytes, store directly
    if (type == 1) {
        ls->ptxBuffers.push_back(
            std::vector<uint8_t>(static_cast<uint8_t*>(data),
                                 static_cast<uint8_t*>(data) + size));
        return CUDA_SUCCESS;
    }

    // CU_JIT_INPUT_FATBINARY (2): extract embedded PTX from fatbinary container
    if (type == 2) {
        std::string ptx = vgre::fatbin::parseSections(
            static_cast<const uint8_t*>(data), size);
        if (ptx == vgre::fatbin::kSassOnlyMarker || ptx.empty()) {
            VGRE_LOG_ERROR("cuLinkAddData",
                ptx == vgre::fatbin::kSassOnlyMarker
                    ? "Fatbinary has no PTX sections (SASS-only). "
                      "Rebuild with -gencode arch=compute_XX,code=compute_XX."
                    : "Failed to extract PTX from fatbinary container.");
            return CUDA_ERROR_INVALID_VALUE;
        }
        ls->ptxBuffers.push_back(std::vector<uint8_t>(ptx.begin(), ptx.end()));
        return CUDA_SUCCESS;
    }

    // CU_JIT_INPUT_OBJECT (3): host ELF object embedding a .nv_fatbin section
    if (type == 3) {
        vgre::common::ELFReader elf(data, size);
        size_t fbSize = 0;
        const char* fbPtr = elf.getSectionData(".nv_fatbin", fbSize);
        if (!fbPtr || fbSize == 0)
            fbPtr = elf.getSectionData(".nvFatBinSegment", fbSize);
        if (!fbPtr || fbSize == 0) {
            VGRE_LOG_ERROR("cuLinkAddData",
                "ELF object has no .nv_fatbin/.nvFatBinSegment section.");
            return CUDA_ERROR_INVALID_VALUE;
        }
        std::string ptx = vgre::fatbin::parseSections(
            reinterpret_cast<const uint8_t*>(fbPtr), fbSize);
        if (ptx == vgre::fatbin::kSassOnlyMarker || ptx.empty()) {
            VGRE_LOG_ERROR("cuLinkAddData",
                "No extractable PTX in .nv_fatbin section of ELF object.");
            return CUDA_ERROR_INVALID_VALUE;
        }
        ls->ptxBuffers.push_back(std::vector<uint8_t>(ptx.begin(), ptx.end()));
        return CUDA_SUCCESS;
    }

    // CU_JIT_INPUT_CUBIN (0): standalone ELF cubin — look for .nv_ptx or SASS-decode.
    // If data is not an ELF binary (no \x7fELF magic) treat it as raw PTX text,
    // which some callers pass as type=0 for compatibility.
    if (type == 0) {
        static constexpr uint8_t kElfMagic[4] = {0x7f, 'E', 'L', 'F'};
        const auto* bytes = static_cast<const uint8_t*>(data);
        bool isElf = (size >= 4 && memcmp(bytes, kElfMagic, 4) == 0);
        if (!isElf) {
            ls->ptxBuffers.push_back(
                std::vector<uint8_t>(bytes, bytes + size));
            return CUDA_SUCCESS;
        }
        vgre::common::ELFReader elf(data, size);
        size_t ptxSecSize = 0;
        const char* ptxPtr = elf.getSectionData(".nv_ptx", ptxSecSize);
        if (ptxPtr && ptxSecSize > 0) {
            ls->ptxBuffers.push_back(
                std::vector<uint8_t>(ptxPtr, ptxPtr + ptxSecSize));
            return CUDA_SUCCESS;
        }
        std::string decoded = vgre::sass::decodeSassToPtx(bytes, size);
        if (decoded.empty()) {
            VGRE_LOG_ERROR("cuLinkAddData",
                "CU_JIT_INPUT_CUBIN: no .nv_ptx section and SASS decode failed.");
            return CUDA_ERROR_INVALID_VALUE;
        }
        ls->ptxBuffers.push_back(
            std::vector<uint8_t>(decoded.begin(), decoded.end()));
        return CUDA_SUCCESS;
    }

    // CU_JIT_INPUT_LIBRARY (4) and CU_JIT_INPUT_NVVM (5) are not supported
    VGRE_LOG_ERROR("cuLinkAddData",
        "Unsupported CUjitInputType=" + std::to_string(type) +
        " — only CUBIN(0), PTX(1), FATBINARY(2), OBJECT(3) are supported.");
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult cuLinkAddData_v2(CUlinkState state, int type, void *data, size_t size,
                          const char *name, unsigned int numOptions,
                          int *options, void **optionValues) {
    return cuLinkAddData(state, type, data, size, name, numOptions, options, optionValues);
}

CUresult cuLinkAddFile(CUlinkState state, int type, const char *path,
                       unsigned int numOptions, int *options,
                       void **optionValues) {
    if (!state || !path) return CUDA_ERROR_INVALID_VALUE;
    std::vector<uint8_t> buf;
    {
        FILE *f = fopen(path, "rb");
        if (!f) return CUDA_ERROR_FILE_NOT_FOUND;
        fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
        if (sz > 0) {
            buf.resize(static_cast<size_t>(sz));
            size_t nr = fread(buf.data(), 1, buf.size(), f);
            if (nr != buf.size()) { buf.resize(nr); }
        }
        fclose(f);
    }
    if (buf.empty()) return CUDA_ERROR_INVALID_VALUE;
    return cuLinkAddData(state, type, buf.data(), buf.size(),
                         path, numOptions, options, optionValues);
}

CUresult cuLinkAddFile_v2(CUlinkState state, int type, const char *path,
                          unsigned int numOptions, int *options, void **optionValues) {
    return cuLinkAddFile(state, type, path, numOptions, options, optionValues);
}

} // extern "C" — PTX helpers below use std::string return type (incompatible with C linkage)

// ── PTX symbol linker helpers ─────────────────────────────────────────────────

// Extract every function name that has a real definition (not extern) in PTX src.
// Matches:  [.visible] [.entry | .func] ... funcname (...) {  (curly brace = body)
static void ptx_collect_defined_funcs(const std::string &src,
                                       std::unordered_set<std::string> &out) {
    const char *kws[] = { ".func ", ".entry ", nullptr };
    for (int k = 0; kws[k]; ++k) {
        size_t pos = 0;
        while ((pos = src.find(kws[k], pos)) != std::string::npos) {
            // Reject if preceded by .extern (skip whitespace backward)
            size_t chk = pos;
            while (chk > 0 && (src[chk-1] == ' ' || src[chk-1] == '\t')) --chk;
            bool is_extern = (chk >= 7 && src.substr(chk-7, 7) == ".extern");
            pos += strlen(kws[k]);
            if (is_extern) continue;
            // Skip optional return param  (.param ...) or (.b32 %r)
            if (pos < src.size() && src[pos] == '(') {
                int depth = 1; ++pos;
                while (pos < src.size() && depth > 0) {
                    if (src[pos] == '(') ++depth;
                    else if (src[pos] == ')') --depth;
                    ++pos;
                }
                while (pos < src.size() && (src[pos] == ' ' || src[pos] == '\t')) ++pos;
            }
            // Collect identifier
            size_t name_start = pos;
            while (pos < src.size() &&
                   (std::isalnum((unsigned char)src[pos]) || src[pos] == '_' || src[pos] == '$'))
                ++pos;
            if (pos == name_start) continue;
            std::string name = src.substr(name_start, pos - name_start);
            // Verify a body follows (find '{' before next ';')
            size_t scan = pos;
            while (scan < src.size() && src[scan] != '{' && src[scan] != ';') ++scan;
            if (scan < src.size() && src[scan] == '{') out.insert(name);
        }
    }
}

// Strip PTX module-level directives (.version, .target, .address_size) from
// non-first modules so the concatenated stream has exactly one header block.
// These directives are only valid at the top of a PTX file and cause parse errors
// when repeated mid-stream.
static std::string ptx_strip_module_header(const std::string &src) {
    std::string out;
    out.reserve(src.size());
    size_t pos = 0;
    while (pos <= src.size()) {
        size_t end = src.find('\n', pos);
        if (end == std::string::npos) end = src.size();
        size_t len = end - pos;
        // Locate first non-whitespace on this line
        size_t first = pos;
        while (first < end && (src[first] == ' ' || src[first] == '\t')) ++first;
        bool skip = false;
        auto starts_with = [&](const char* pfx) {
            size_t pl = strlen(pfx);
            return (end - first >= pl) && src.compare(first, pl, pfx) == 0;
        };
        if (starts_with(".version") || starts_with(".target") ||
            starts_with(".address_size")) {
            skip = true;
        }
        if (!skip) out.append(src, pos, len);
        if (end < src.size()) { out.push_back('\n'); }
        pos = end + 1;
    }
    return out;
}

// Collect names of .global variables that have a real definition (not extern).
// Matches:  [.visible] .global .alignN? .typeN .name [=...]  ;
static void ptx_collect_defined_globals(const std::string &src,
                                         std::unordered_set<std::string> &out) {
    size_t pos = 0;
    while (pos < src.size()) {
        size_t end = src.find('\n', pos);
        if (end == std::string::npos) end = src.size();
        std::string line = src.substr(pos, end - pos);
        // Must contain .global but NOT .extern
        if (line.find(".global") != std::string::npos &&
            line.find(".extern") == std::string::npos) {
            // Extract the symbol name: last identifier before ';', '[', or '='
            // Walk backward from end of line
            size_t scan = line.size();
            while (scan > 0 && (line[scan-1] == ' ' || line[scan-1] == '\t' ||
                                 line[scan-1] == ';' || line[scan-1] == '\n' ||
                                 line[scan-1] == '\r')) --scan;
            // Skip array size or initialiser
            if (scan > 0 && (line[scan-1] == ']' || line[scan-1] == '}')) {
                while (scan > 0 && line[scan-1] != '[' && line[scan-1] != '{') --scan;
                --scan;
                while (scan > 0 && (line[scan-1] == ' ' || line[scan-1] == '\t')) --scan;
            }
            // Name ends at scan
            size_t ne = scan;
            while (ne > 0 &&
                   (std::isalnum((unsigned char)line[ne-1]) || line[ne-1] == '_' ||
                    line[ne-1] == '$'))
                --ne;
            if (ne < scan) out.insert(line.substr(ne, scan - ne));
        }
        pos = end + 1;
    }
}

// Remove all standalone .extern .func <name> ; lines where <name> is in defined.
static std::string ptx_strip_extern_decls(const std::string &src,
                                           const std::unordered_set<std::string> &defined) {
    std::string out;
    out.reserve(src.size());
    size_t pos = 0;
    while (pos < src.size()) {
        size_t line_start = pos;
        size_t line_end   = src.find('\n', pos);
        if (line_end == std::string::npos) line_end = src.size();
        std::string line = src.substr(line_start, line_end - line_start);
        // Check if line is an .extern .func declaration for a defined symbol
        bool skip = false;
        auto ef = line.find(".extern");
        if (ef != std::string::npos) {
            auto ff = line.find(".func", ef);
            if (ff == std::string::npos) ff = line.find(".entry", ef);
            if (ff != std::string::npos) {
                // Extract the function name from this line
                size_t np = ff + 5; // skip .func
                while (np < line.size() && (line[np] == ' ' || line[np] == '\t')) ++np;
                if (np < line.size() && line[np] == '(') {
                    int depth = 1; ++np;
                    while (np < line.size() && depth > 0) {
                        if (line[np] == '(') ++depth;
                        else if (line[np] == ')') --depth;
                        ++np;
                    }
                    while (np < line.size() && (line[np] == ' ' || line[np] == '\t')) ++np;
                }
                size_t ns = np;
                while (np < line.size() &&
                       (std::isalnum((unsigned char)line[np]) || line[np] == '_' || line[np] == '$'))
                    ++np;
                std::string name = line.substr(ns, np - ns);
                if (!name.empty() && defined.count(name)) skip = true;
            }
        }
        if (!skip) out.append(line);
        if (line_end < src.size()) { out.push_back('\n'); pos = line_end + 1; }
        else break;
    }
    return out;
}

// Strip .extern .global declarations for variables defined in the merged PTX.
static std::string ptx_strip_extern_globals(const std::string &src,
                                             const std::unordered_set<std::string> &defined) {
    std::string out;
    out.reserve(src.size());
    size_t pos = 0;
    while (pos <= src.size()) {
        size_t end = src.find('\n', pos);
        if (end == std::string::npos) end = src.size();
        std::string line = src.substr(pos, end - pos);
        bool skip = false;
        if (line.find(".extern") != std::string::npos &&
            line.find(".global") != std::string::npos) {
            // Try to extract the symbol name (same logic as ptx_collect_defined_globals)
            size_t scan = line.size();
            while (scan > 0 && (line[scan-1] == ' ' || line[scan-1] == '\t' ||
                                 line[scan-1] == ';' || line[scan-1] == '\n' ||
                                 line[scan-1] == '\r')) --scan;
            if (scan > 0 && line[scan-1] == ']') {
                while (scan > 0 && line[scan-1] != '[') --scan;
                --scan;
                while (scan > 0 && (line[scan-1] == ' ' || line[scan-1] == '\t')) --scan;
            }
            size_t ne = scan;
            while (ne > 0 &&
                   (std::isalnum((unsigned char)line[ne-1]) || line[ne-1] == '_' ||
                    line[ne-1] == '$')) --ne;
            std::string name = line.substr(ne, scan - ne);
            if (!name.empty() && defined.count(name)) skip = true;
        }
        if (!skip) out.append(line);
        if (end < src.size()) out.push_back('\n');
        pos = end + 1;
    }
    return out;
}

extern "C" {

CUresult cuLinkComplete(CUlinkState state, void **cubinOut, size_t *sizeOut) {
    if (!state || !cubinOut || !sizeOut) return CUDA_ERROR_INVALID_VALUE;
    auto *ls = reinterpret_cast<CUlinkStateImpl*>(state);
    if (ls->ptxBuffers.empty()) { *cubinOut = nullptr; *sizeOut = 0; return CUDA_SUCCESS; }

    // ── Multi-TU PTX link algorithm ───────────────────────────────────────────
    //
    // 1. Keep the first module's .version / .target / .address_size header.
    //    Strip those directives from all subsequent modules (they are module-
    //    level declarations, not scoped, and the PTX assembler rejects duplicates).
    //
    // 2. Concatenate all (possibly stripped) module bodies.
    //
    // 3. Strip .extern .func declarations for symbols that are now defined
    //    in the merged PTX (resolves cross-module function calls).
    //
    // 4. Strip .extern .global declarations for global variables defined
    //    elsewhere in the merged PTX (resolves cross-module extern device vars).
    //
    // Result: a single valid PTX translation unit that the LLVM PTX JIT
    // (cuModuleLoadData) can compile without duplicate-symbol or undefined-
    // symbol errors.

    std::string combined;
    for (size_t i = 0; i < ls->ptxBuffers.size(); ++i) {
        const auto &buf = ls->ptxBuffers[i];
        std::string module(reinterpret_cast<const char*>(buf.data()), buf.size());
        if (i == 0) {
            combined.append(module);
        } else {
            // Strip per-module header directives before merging
            combined.append(ptx_strip_module_header(module));
        }
        combined.push_back('\n');
    }

    // Pass 1: resolve cross-module function references
    std::unordered_set<std::string> definedFuncs;
    ptx_collect_defined_funcs(combined, definedFuncs);
    if (!definedFuncs.empty())
        combined = ptx_strip_extern_decls(combined, definedFuncs);

    // Pass 2: resolve cross-module extern device variable references
    std::unordered_set<std::string> definedGlobals;
    ptx_collect_defined_globals(combined, definedGlobals);
    if (!definedGlobals.empty())
        combined = ptx_strip_extern_globals(combined, definedGlobals);

    // Store resolved PTX so caller can pass it to cuModuleLoadData
    ls->ptxBuffers.clear();
    ls->ptxBuffers.push_back(std::vector<uint8_t>(combined.begin(), combined.end()));
    *cubinOut = ls->ptxBuffers.back().data();
    *sizeOut  = ls->ptxBuffers.back().size();
    return CUDA_SUCCESS;
}

CUresult cuLinkDestroy(CUlinkState state) {
    if (!state) return CUDA_ERROR_INVALID_VALUE;
    delete reinterpret_cast<CUlinkStateImpl*>(state);
    return CUDA_SUCCESS;
}

// ── cuModuleGetLoadingMode ────────────────────────────────────────────────────
// Returns the current module loading mode (EAGER or LAZY).
// VGRE compiles kernels eagerly at registration time.
CUresult cuModuleGetLoadingMode(int *mode) {
    if (!mode) return CUDA_ERROR_INVALID_VALUE;
    *mode = 0; // CU_MODULE_EAGER_LOADING = 0
    return CUDA_SUCCESS;
}

} // extern "C"
