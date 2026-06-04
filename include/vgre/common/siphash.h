#pragma once
#include <cstdint>
#include <cstring>
#include <string>

namespace vgre {
namespace common {

// SipHash-2-4: 2 compression rounds, 4 finalization rounds.
// ε-Δ-universal hash function; security: ε = 2^{-64} for adversarial inputs.
// k0, k1: 128-bit secret key; must be random per process to prevent hash flooding.
// Math: SipRound = v0+=v1; v1=rotl(v1,13); v1^=v0; v0=rotl(v0,32); ...
struct SipHash24 {
    uint64_t k0, k1;

    // Default constructor: borrows randomized keys from the process-wide singleton.
    // This allows SipHash24 to be used as a default-constructible Hash in templates.
    SipHash24();

    explicit SipHash24(uint64_t k0_, uint64_t k1_) : k0(k0_), k1(k1_) {}

    uint64_t operator()(const std::string& s) const;
    uint64_t hash(const void* data, size_t len) const;
};

// Process-wide singleton with randomized keys (set once at startup).
// Returns a reference to the process-global SipHash24 instance.
SipHash24& processHasher();

} // namespace common
} // namespace vgre
