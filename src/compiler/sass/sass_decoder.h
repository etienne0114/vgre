#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace vgre::sass {

// Attempt to decode a SASS-only cubin ELF and synthesize equivalent PTX.
// Returns empty string if decoding fails (not a valid SASS ELF or unsupported ISA).
std::string decodeSassToPtx(const uint8_t* data, size_t size);

} // namespace vgre::sass
