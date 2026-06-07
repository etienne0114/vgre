// Shared NVIDIA fatbinary container parsing utilities.
// Used by both the CUDART shim and the CUDA Driver Module to extract embedded
// PTX text from fatbinary containers, ELF host objects, and cubin files.
#pragma once

#include "vgre/common/logger.h"
#ifdef VGRE_HAS_ZLIB
#  include <zlib.h>
#endif
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace vgre {
namespace fatbin {

// ── Container-level constants ─────────────────────────────────────────────────
static constexpr uint32_t kMagic    = 0xba55ed50u;
static constexpr uint32_t kMagicOld = 0xba55ed01u;
static constexpr uint32_t kWrapper  = 0x466243b1u;  // __fatBinC_Wrapper_t
static constexpr uint32_t kKindSASS = 1u;
static constexpr uint32_t kKindPTX  = 2u;

// Returned by parseSections when the container has SASS-only sections.
// The caller should treat this as "no PTX available, SASS decode required."
static const std::string kSassOnlyMarker = "__VGRE_SASS_ONLY__";

// ── Wire-format structs ───────────────────────────────────────────────────────
#pragma pack(push, 1)
struct ContainerHdr {
    uint32_t magic;
    uint16_t version;
    uint16_t headerSize;
    uint64_t fatSize;
};

struct SectionHdr {
    uint32_t kind;        // 1=SASS, 2=PTX
    uint32_t headerSize;
    uint64_t dataSize;
    uint8_t  minorSM;
    uint8_t  majorSM;
    uint16_t flags;
    uint32_t cubinVersion;
    uint64_t uncompressedSize;  // > 0 means zlib-compressed payload
};
#pragma pack(pop)

// ── parseSections ─────────────────────────────────────────────────────────────
// Walk the section list inside a fatbinary container (ptr must point at the
// ContainerHdr). Returns the first PTX section payload as a string.
// Returns kSassOnlyMarker if only SASS sections are present.
// Returns "" if the container is malformed or empty.
inline std::string parseSections(const uint8_t* ptr, size_t totalBytes) {
    if (totalBytes < sizeof(ContainerHdr)) return "";
    const auto* hdr = reinterpret_cast<const ContainerHdr*>(ptr);
    if (hdr->magic != kMagic && hdr->magic != kMagicOld) return "";
    if (hdr->headerSize < 16 || hdr->headerSize > 64) return "";

    size_t off    = hdr->headerSize;
    size_t fatEnd = hdr->headerSize + static_cast<size_t>(hdr->fatSize);
    if (fatEnd > totalBytes) fatEnd = totalBytes;

    bool hasSASS = false;
    while (off + sizeof(SectionHdr) <= fatEnd) {
        const auto* sec = reinterpret_cast<const SectionHdr*>(ptr + off);
        if (sec->headerSize < 32 || sec->headerSize > 4096) break;
        if (sec->dataSize > totalBytes) break;

        size_t payloadOff = off + sec->headerSize;
        size_t payloadEnd = payloadOff + static_cast<size_t>(sec->dataSize);
        if (payloadEnd > totalBytes) break;

        if (sec->kind == kKindPTX && sec->dataSize > 0) {
            const char* ptxData = reinterpret_cast<const char*>(ptr + payloadOff);
            VGRE_LOG_INFO("FatbinUtils",
                "PTX section found (" + std::to_string(sec->dataSize) +
                " bytes, sm_" + std::to_string(sec->majorSM) +
                std::to_string(sec->minorSM) + ")");

            if (sec->uncompressedSize > 0) {
#ifdef VGRE_HAS_ZLIB
                uLongf destLen = static_cast<uLongf>(sec->uncompressedSize) + 1;
                std::vector<char> decompressed(destLen + 1, '\0');
                int ret = uncompress(
                    reinterpret_cast<Bytef*>(decompressed.data()), &destLen,
                    reinterpret_cast<const Bytef*>(ptxData),
                    static_cast<uLong>(sec->dataSize));
                if (ret == Z_OK && destLen > 0) {
                    VGRE_LOG_INFO("FatbinUtils",
                        "Decompressed PTX: " + std::to_string(destLen) + " bytes");
                    return std::string(decompressed.data(), destLen);
                }
                VGRE_LOG_WARN("FatbinUtils",
                    "zlib decompress failed (ret=" + std::to_string(ret) +
                    ") — fallback to linear scan");
#else
                VGRE_LOG_WARN("FatbinUtils",
                    "Compressed PTX section requires zlib "
                    "(rebuild with -DVGRE_ENABLE_ZLIB=ON)");
#endif
                return "";  // signal caller to try linear scan
            }
            return std::string(ptxData, static_cast<size_t>(sec->dataSize));

        } else if (sec->kind == kKindSASS) {
            hasSASS = true;
            VGRE_LOG_INFO("FatbinUtils",
                "SASS section found (sm_" +
                std::to_string(sec->majorSM) + std::to_string(sec->minorSM) +
                ", " + std::to_string(sec->dataSize) + " bytes)");
        }
        off = payloadEnd;
    }

    if (hasSASS) {
        VGRE_LOG_WARN("FatbinUtils",
            "Fatbinary has SASS-only sections (no PTX). "
            "Rebuild with -gencode arch=compute_XX,code=compute_XX to embed PTX.");
        return kSassOnlyMarker;
    }
    return "";
}

// ── extractFromWrapper ────────────────────────────────────────────────────────
// Handle __fatBinC_Wrapper_t: the data pointer is at the second uint64 slot.
inline std::string extractFromWrapper(const void* image, size_t maxBytes) {
    const uint64_t* w = reinterpret_cast<const uint64_t*>(image);
    uint64_t dataPtr = w[1];
    if (!dataPtr) return "";
    const uint8_t* inner = reinterpret_cast<const uint8_t*>(
        static_cast<uintptr_t>(dataPtr));
    return parseSections(inner, maxBytes);
}

} // namespace fatbin
} // namespace vgre
