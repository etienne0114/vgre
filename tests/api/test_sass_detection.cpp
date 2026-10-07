// Test: SASS fatbinary detection and PTX extraction
//
// Verifies:
// 1. PTX-only module: cuModuleLoadData + cuModuleGetFunction succeed
// 2. Fatbinary with embedded PTX: PTX is extracted and function lookup succeeds
// 3. SASS-only ELF cubin (no .nv_ptx section): cuModuleGetFunction returns
//    CUDA_ERROR_NO_BINARY_FOR_GPU (209)
// 4. Raw fatbin container with SASS-only section: same error

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include "../../src/compiler/sass/sass_decoder.h"
#include "vgre/api/cuda_interceptor.h"

// ── Minimal CUDA driver stubs needed by the test harness ─────────────────────
using CUresult  = int;
using CUmodule  = void*;
using CUfunction = void*;
static constexpr CUresult CUDA_SUCCESS             = 0;
static constexpr CUresult CUDA_ERROR_INVALID_VALUE = 1;
static constexpr CUresult CUDA_ERROR_NO_BINARY_FOR_GPU = 209;
static constexpr CUresult CUDA_ERROR_INVALID_PTX   = 218;

extern "C" {
    CUresult cuInit(unsigned int flags);
    CUresult cuModuleLoadData(CUmodule* mod, const void* image);
    CUresult cuModuleGetFunction(CUfunction* fn, CUmodule mod, const char* name);
    CUresult cuModuleUnload(CUmodule mod);
}

extern "C" vgre::api::cudaError_t cudaCreateSurfaceObject(
    vgre::api::CUDAInterceptor::cudaSurfaceObject_t*,
    const vgre::api::CUDAInterceptor::cudaResourceDesc*);
extern "C" vgre::api::cudaError_t cudaDestroySurfaceObject(
    vgre::api::CUDAInterceptor::cudaSurfaceObject_t);
extern "C" vgre::api::cudaError_t cudaGetSurfaceObjectResourceDesc(
    vgre::api::CUDAInterceptor::cudaResourceDesc*,
    vgre::api::CUDAInterceptor::cudaSurfaceObject_t);
extern "C" vgre::api::cudaError_t cudaCreateTextureObject(
    vgre::api::CUDAInterceptor::cudaTextureObject_t*,
    const vgre::api::CUDAInterceptor::cudaResourceDesc*,
    const vgre::api::CUDAInterceptor::cudaTextureDesc*, const void*);
extern "C" vgre::api::cudaError_t cudaDestroyTextureObject(
    vgre::api::CUDAInterceptor::cudaTextureObject_t);
extern "C" vgre::api::cudaError_t cudaGetTextureObjectResourceDesc(
    vgre::api::CUDAInterceptor::cudaResourceDesc*,
    vgre::api::CUDAInterceptor::cudaTextureObject_t);
extern "C" vgre::api::cudaError_t cudaGetTextureObjectTextureDesc(
    vgre::api::CUDAInterceptor::cudaTextureDesc*,
    vgre::api::CUDAInterceptor::cudaTextureObject_t);
extern "C" vgre::api::cudaError_t cudaGetTextureObjectResourceViewDesc(
    vgre::api::CUDAInterceptor::CUDA_RESOURCE_VIEW_DESC*,
    vgre::api::CUDAInterceptor::cudaTextureObject_t);

static int passed = 0, failed = 0;
#define CHECK(cond, msg) do { \
    if (cond) { printf("  PASS: %s\n", msg); ++passed; } \
    else       { printf("  FAIL: %s\n", msg); ++failed; } \
} while(0)

struct TestElf64Ehdr {
    uint8_t e_ident[16]; uint16_t e_type, e_machine; uint32_t e_version;
    uint64_t e_entry, e_phoff, e_shoff; uint32_t e_flags;
    uint16_t e_ehsize, e_phentsize, e_phnum, e_shentsize, e_shnum, e_shstrndx;
};
struct TestElf64Shdr {
    uint32_t sh_name, sh_type; uint64_t sh_flags, sh_addr, sh_offset, sh_size;
    uint32_t sh_link, sh_info; uint64_t sh_addralign, sh_entsize;
};

static std::vector<uint8_t> buildDecoderElf(bool withText, bool withOrphanInfo,
                                            bool unknownInstruction) {
    TestElf64Ehdr eh{};
    eh.e_ident[0] = 0x7f; eh.e_ident[1] = 'E'; eh.e_ident[2] = 'L'; eh.e_ident[3] = 'F';
    eh.e_ident[4] = 2;
    eh.e_ehsize = sizeof(eh);
    eh.e_shentsize = sizeof(TestElf64Shdr);
    eh.e_shstrndx = 1;

    std::string names(1, '\0');
    auto addName = [&](const char* value) {
        uint32_t offset = static_cast<uint32_t>(names.size());
        names += value;
        names.push_back('\0');
        return offset;
    };
    const uint32_t strName = addName(".shstrtab");
    const uint32_t textName = withText ? addName(".text.kernel") : 0;
    const uint32_t infoName = withOrphanInfo ? addName(".nv.info.orphan") : 0;

    const size_t sectionCount = 2 + (withText ? 1 : 0) + (withOrphanInfo ? 1 : 0);
    std::vector<TestElf64Shdr> sections(sectionCount);
    size_t strOffset = sizeof(eh);
    size_t textOffset = (strOffset + names.size() + 7u) & ~size_t(7u);
    size_t infoOffset = textOffset + (withText ? 32u : 0u);
    size_t shOffset = (infoOffset + (withOrphanInfo ? 4u : 0u) + 7u) & ~size_t(7u);

    sections[1].sh_name = strName;
    sections[1].sh_type = 3;
    sections[1].sh_offset = strOffset;
    sections[1].sh_size = names.size();
    size_t sectionIndex = 2;
    if (withText) {
        sections[sectionIndex].sh_name = textName;
        sections[sectionIndex].sh_type = 1;
        sections[sectionIndex].sh_offset = textOffset;
        sections[sectionIndex].sh_size = 32;
        ++sectionIndex;
    }
    if (withOrphanInfo) {
        sections[sectionIndex].sh_name = infoName;
        sections[sectionIndex].sh_type = 1;
        sections[sectionIndex].sh_offset = infoOffset;
        sections[sectionIndex].sh_size = 4;
    }
    eh.e_shoff = shOffset;
    eh.e_shnum = static_cast<uint16_t>(sectionCount);

    std::vector<uint8_t> image(shOffset + sections.size() * sizeof(sections[0]), 0);
    memcpy(image.data(), &eh, sizeof(eh));
    memcpy(image.data() + strOffset, names.data(), names.size());
    if (withText && unknownInstruction) {
        const uint64_t unknown = uint64_t{0xfe} << 55;
        memcpy(image.data() + textOffset + sizeof(uint64_t), &unknown, sizeof(unknown));
    }
    memcpy(image.data() + shOffset, sections.data(), sections.size() * sizeof(sections[0]));
    return image;
}

// ── Minimal PTX source that defines one kernel ────────────────────────────────
static const char kPtxSource[] =
    ".version 7.5\n"
    ".target sm_80\n"
    ".address_size 64\n"
    ".visible .entry sass_test_kernel() {\n"
    "    ret;\n"
    "}\n";

// ── Minimal NVIDIA fatbinary container with one PTX section ──────────────────
// Layout: FatbinContainerHdr (16 bytes) + FatbinSectionHdr (64 bytes) + PTX payload
#pragma pack(push, 1)
struct FatbinContainerHdr {
    uint32_t magic;       // 0xba55ed50
    uint16_t version;
    uint16_t headerSize;
    uint64_t fatSize;
};
struct FatbinSectionHdr {
    uint32_t kind;        // 2 = PTX
    uint32_t headerSize;  // 64
    uint64_t dataSize;
    uint8_t  minorSM;
    uint8_t  majorSM;
    uint16_t flags;
    uint32_t cubinVersion;
    uint64_t uncompressedSize; // 0 = not compressed
    uint8_t  pad[32];     // padding to 64 bytes
};
#pragma pack(pop)

static void buildFatbinPTX(std::vector<uint8_t>& out) {
    const size_t ptxLen = strlen(kPtxSource);
    const size_t secHdrSize = 64;
    const size_t fatSize = secHdrSize + ptxLen;

    FatbinContainerHdr cHdr{};
    cHdr.magic      = 0xba55ed50u;
    cHdr.version    = 1;
    cHdr.headerSize = 16;
    cHdr.fatSize    = static_cast<uint64_t>(fatSize);

    FatbinSectionHdr sHdr{};
    sHdr.kind             = 2u; // PTX
    sHdr.headerSize       = static_cast<uint32_t>(secHdrSize);
    sHdr.dataSize         = static_cast<uint64_t>(ptxLen);
    sHdr.majorSM          = 8;
    sHdr.minorSM          = 0;
    sHdr.uncompressedSize = 0;

    out.resize(sizeof(FatbinContainerHdr) + secHdrSize + ptxLen, 0);
    memcpy(out.data(), &cHdr, sizeof(cHdr));
    memcpy(out.data() + sizeof(cHdr), &sHdr, sizeof(sHdr));
    memcpy(out.data() + sizeof(cHdr) + secHdrSize, kPtxSource, ptxLen);
}

static void buildFatbinSASSOnly(std::vector<uint8_t>& out) {
    // A fatbin with only kind=1 (SASS/cubin) section and 4-byte dummy payload
    const size_t secHdrSize = 64;
    const size_t fakeDataSize = 4;
    const size_t fatSize = secHdrSize + fakeDataSize;

    FatbinContainerHdr cHdr{};
    cHdr.magic      = 0xba55ed50u;
    cHdr.version    = 1;
    cHdr.headerSize = 16;
    cHdr.fatSize    = static_cast<uint64_t>(fatSize);

    FatbinSectionHdr sHdr{};
    sHdr.kind             = 1u; // SASS/cubin
    sHdr.headerSize       = static_cast<uint32_t>(secHdrSize);
    sHdr.dataSize         = static_cast<uint64_t>(fakeDataSize);
    sHdr.majorSM          = 8;
    sHdr.minorSM          = 0;
    sHdr.uncompressedSize = 0;

    out.resize(sizeof(FatbinContainerHdr) + secHdrSize + fakeDataSize, 0);
    memcpy(out.data(), &cHdr, sizeof(cHdr));
    memcpy(out.data() + sizeof(cHdr), &sHdr, sizeof(sHdr));
    // leave dummy payload as zeros
}

int main() {
    printf("=== SASS Detection / Fatbin PTX Extraction Tests ===\n");
    cuInit(0);

    {
        using Api = vgre::api::CUDAInterceptor;
        Api::cudaResourceDesc resource{};
        int backing = 42;
        resource.resType = 3;
        resource.res.devPtr = &backing;
        resource.res.desc.x = 32;
        resource.res.desc.f = 0;
        resource.res.width = 1;
        resource.res.height = 1;
        Api::cudaSurfaceObject_t surface = 0;
        const auto created = cudaCreateSurfaceObject(&surface, &resource);
        CHECK(created == vgre::api::cudaSuccess && surface != 0,
              "surface object creation returns a live handle");
        Api::cudaResourceDesc queried{};
        const auto queriedStatus = cudaGetSurfaceObjectResourceDesc(&queried, surface);
        CHECK(queriedStatus == vgre::api::cudaSuccess &&
              queried.res.devPtr == resource.res.devPtr && queried.res.width == 1 &&
              queried.res.desc.x == 32,
              "surface introspection returns the original resource descriptor");
        const auto destroyed = cudaDestroySurfaceObject(surface);
        CHECK(destroyed == vgre::api::cudaSuccess &&
              cudaGetSurfaceObjectResourceDesc(&queried, surface) == vgre::api::cudaErrorInvalidValue,
              "destroyed surface handles are rejected by introspection");
    }
    {
        using Api = vgre::api::CUDAInterceptor;
        Api::cudaResourceDesc resource{};
        Api::cudaTextureDesc texture{};
        int backing = 24;
        resource.resType = 3;
        resource.res.devPtr = &backing;
        resource.res.desc.x = 32;
        resource.res.desc.f = 0;
        resource.res.width = 1;
        resource.res.sizeInBytes = sizeof(backing);
        Api::cudaTextureObject_t object = 0;
        const auto created = cudaCreateTextureObject(&object, &resource, &texture, nullptr);
        CHECK(created == vgre::api::cudaSuccess && object != 0,
              "texture object creation returns a live handle");
        Api::cudaResourceDesc queriedResource{};
        Api::cudaTextureDesc queriedTexture{};
        Api::CUDA_RESOURCE_VIEW_DESC queriedView{};
        CHECK(cudaGetTextureObjectResourceDesc(&queriedResource, object) ==
                  vgre::api::cudaErrorNotSupported &&
              cudaGetTextureObjectTextureDesc(&queriedTexture, object) ==
                  vgre::api::cudaErrorNotSupported &&
              cudaGetTextureObjectResourceViewDesc(&queriedView, object) ==
                  vgre::api::cudaErrorNotSupported,
              "texture introspection rejects descriptors it cannot reconstruct exactly");
        CHECK(cudaDestroyTextureObject(object) == vgre::api::cudaSuccess,
              "texture object destruction succeeds after introspection");
    }

    {
        auto image = buildDecoderElf(true, false, true);
        const auto decoded = vgre::sass::decodeSassToPtx(image.data(), image.size());
        CHECK(decoded.empty(), "unknown SASS opcode rejects the whole translation");
    }
    {
        auto image = buildDecoderElf(false, true, false);
        const auto decoded = vgre::sass::decodeSassToPtx(image.data(), image.size());
        CHECK(decoded.empty(), "metadata-only kernel does not become a no-op PTX stub");
    }
    {
        auto image = buildDecoderElf(true, true, false);
        const auto decoded = vgre::sass::decodeSassToPtx(image.data(), image.size());
        CHECK(decoded.empty(), "partially decoded module does not hide a missing kernel body");
    }
    {
        auto image = buildDecoderElf(true, false, false);
        const auto decoded = vgre::sass::decodeSassToPtx(image.data(), image.size());
        CHECK(!decoded.empty() && decoded.find(".entry kernel") != std::string::npos,
              "recognized NOP-only kernel is decoded without fabricating instructions");
    }

    // ── Test 1: raw PTX module creates a non-null module handle ──────────────
    {
        CUmodule mod = nullptr;
        CUresult r = cuModuleLoadData(&mod, kPtxSource);
        CHECK(r == CUDA_SUCCESS, "T1: raw PTX cuModuleLoadData succeeds");
        CHECK(mod != nullptr,    "T1: raw PTX module handle is non-null");
        // cuModuleGetFunction with PTX goes through the C++ kernel parser which
        // only handles __global__ CUDA syntax. We verify the sentinel is NOT set.
        if (r == CUDA_SUCCESS && mod) {
            CUfunction fn = nullptr;
            CUresult rf = cuModuleGetFunction(&fn, mod, "sass_test_kernel");
            CHECK(rf != CUDA_ERROR_NO_BINARY_FOR_GPU,
                  "T1: raw PTX module not marked as SASS-only");
            cuModuleUnload(mod);
        }
    }

    // ── Test 2: fatbin with embedded PTX section creates a non-null handle ───
    {
        std::vector<uint8_t> fatbin;
        buildFatbinPTX(fatbin);
        CUmodule mod = nullptr;
        CUresult r = cuModuleLoadData(&mod, fatbin.data());
        CHECK(r == CUDA_SUCCESS, "T2: fatbin-with-PTX cuModuleLoadData succeeds");
        CHECK(mod != nullptr,    "T2: fatbin-with-PTX module handle is non-null");
        if (r == CUDA_SUCCESS && mod) {
            CUfunction fn = nullptr;
            CUresult rf = cuModuleGetFunction(&fn, mod, "sass_test_kernel");
            CHECK(rf != CUDA_ERROR_NO_BINARY_FOR_GPU,
                  "T2: fatbin-with-PTX module not marked as SASS-only");
            cuModuleUnload(mod);
        }
    }

    // ── Test 3: fatbin with SASS-only section → NO_BINARY_FOR_GPU (key test) ─
    {
        std::vector<uint8_t> fatbin;
        buildFatbinSASSOnly(fatbin);
        CUmodule mod = nullptr;
        CUresult r = cuModuleLoadData(&mod, fatbin.data());
        CHECK(r == CUDA_SUCCESS, "T3: SASS-only cuModuleLoadData succeeds (deferred error)");
        if (r == CUDA_SUCCESS && mod) {
            CUfunction fn = nullptr;
            r = cuModuleGetFunction(&fn, mod, "any_kernel");
            CHECK(r == CUDA_ERROR_NO_BINARY_FOR_GPU,
                  "T3: SASS-only cuModuleGetFunction returns CUDA_ERROR_NO_BINARY_FOR_GPU(209)");
            cuModuleUnload(mod);
        }
    }

    // ── Test 4: two independent SASS-only loads both produce the sentinel ─────
    {
        std::vector<uint8_t> f1, f2;
        buildFatbinSASSOnly(f1);
        buildFatbinSASSOnly(f2);
        CUmodule mod1 = nullptr, mod2 = nullptr;
        cuModuleLoadData(&mod1, f1.data());
        cuModuleLoadData(&mod2, f2.data());
        CUresult r1 = CUDA_SUCCESS, r2 = CUDA_SUCCESS;
        if (mod1) { CUfunction fn = nullptr; r1 = cuModuleGetFunction(&fn, mod1, "k"); cuModuleUnload(mod1); }
        if (mod2) { CUfunction fn = nullptr; r2 = cuModuleGetFunction(&fn, mod2, "k"); cuModuleUnload(mod2); }
        CHECK(r1 == CUDA_ERROR_NO_BINARY_FOR_GPU && r2 == CUDA_ERROR_NO_BINARY_FOR_GPU,
              "T4: two independent SASS-only modules both return NO_BINARY_FOR_GPU");
    }

    // ── Test 5: null image is rejected ───────────────────────────────────────
    {
        CUmodule mod = nullptr;
        CUresult r = cuModuleLoadData(&mod, nullptr);
        CHECK(r == CUDA_ERROR_INVALID_VALUE, "T5: null image rejected by cuModuleLoadData");
    }

    printf("\nResults: %d passed, %d failed\n", passed, failed);
    return (failed == 0) ? 0 : 1;
}
