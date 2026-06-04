// LSD (least-significant-digit) radix sort for uint32_t.
// 8-bit radix, 4 passes → covers all 32 bits.
// Each pass: counting sort on one byte using a 256-entry histogram.
// Complexity: O(4*(n + 256)) = O(n). Space: O(n + 256).
// Math invariant: after pass p ∈ {0,1,2,3}, the array is sorted with respect
// to the (p+1) least-significant bytes; i.e., keys[0..n-1] are non-decreasing
// when masked to bits 0..8*(p+1)-1.

#include "vgre/core/radix_sort.h"
#include <array>
#include <vector>

namespace vgre {

namespace {

// One pass of counting sort on the byte at bit offset 'shift' (0, 8, 16, 24).
static void counting_sort_pass(const uint32_t* in, uint32_t* out,
                               const uint32_t* vals_in, uint32_t* vals_out,
                               std::size_t n, int shift) {
    std::array<std::size_t, 256> cnt{};
    // Count occurrences of each 8-bit digit
    for (std::size_t i = 0; i < n; ++i)
        ++cnt[(in[i] >> shift) & 0xFF];
    // Exclusive prefix sum over counts → stable output positions
    std::size_t total = 0;
    for (int d = 0; d < 256; ++d) {
        std::size_t c = cnt[d];
        cnt[d] = total;
        total += c;
    }
    // Scatter: place each element at its counted position
    for (std::size_t i = 0; i < n; ++i) {
        int d = (int)((in[i] >> shift) & 0xFF);
        out[cnt[d]] = in[i];
        if (vals_out && vals_in)
            vals_out[cnt[d]] = vals_in[i];
        ++cnt[d];
    }
}

} // anonymous namespace

void radix_sort_u32(uint32_t* keys, std::size_t n) {
    if (n <= 1) return;
    std::vector<uint32_t> tmp(n);
    // 4 passes for 32-bit keys; alternate src/dst to avoid extra copy.
    uint32_t* src = keys;
    uint32_t* dst = tmp.data();
    for (int pass = 0; pass < 4; ++pass) {
        counting_sort_pass(src, dst, nullptr, nullptr, n, pass * 8);
        std::swap(src, dst);
    }
    // After 4 passes (even number), result is back in 'keys'.
    // src now points to keys (swapped back after pass 3).
    if (src != keys)
        std::copy(src, src + n, keys);
}

void radix_sort_u32_kv(uint32_t* keys, uint32_t* values, std::size_t n) {
    if (n <= 1) return;
    std::vector<uint32_t> tmp_keys(n), tmp_vals(n);
    uint32_t* sk = keys,          *sv = values;
    uint32_t* dk = tmp_keys.data(), *dv = tmp_vals.data();
    for (int pass = 0; pass < 4; ++pass) {
        counting_sort_pass(sk, dk, sv, dv, n, pass * 8);
        std::swap(sk, dk);
        std::swap(sv, dv);
    }
    if (sk != keys) {
        std::copy(sk, sk + n, keys);
        std::copy(sv, sv + n, values);
    }
}

} // namespace vgre
