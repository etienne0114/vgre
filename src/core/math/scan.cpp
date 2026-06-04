// Blelloch (1990) work-efficient CPU prefix scan.
// Algorithm: up-sweep (reduce) + down-sweep gives exclusive prefix sum.
// Complexity: O(n log n) work, O(log n) depth.
// Math invariant: after down-sweep, out[i] = sum(in[0..i-1]) for all i.
//                 out[0] = 0 (identity element).
//
// For non-power-of-2 n, pad to next power-of-2 in a temp buffer.

#include "vgre/core/scan.h"
#include <algorithm>
#include <cstring>
#include <vector>

namespace vgre {

namespace {

// Next power of 2 >= n.
static inline std::size_t next_pow2(std::size_t n) {
    std::size_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

// In-place Blelloch exclusive scan on buf[0..m-1] where m is a power of 2.
// After return: buf[i] = sum(original buf[0..i-1]), buf[0] = 0.
template<typename T>
static void blelloch_exclusive_inplace(T* buf, std::size_t m) {
    // Up-sweep (reduce): buf[2^(d+1)*i - 1] += buf[2^d*(2i-1)]
    for (std::size_t stride = 1; stride < m; stride <<= 1) {
        for (std::size_t k = 2 * stride - 1; k < m; k += 2 * stride)
            buf[k] += buf[k - stride];  // math: subtree sum stored at right child
    }
    // Down-sweep: insert identity at root, propagate sums down
    buf[m - 1] = T(0);
    for (std::size_t stride = m >> 1; stride >= 1; stride >>= 1) {
        for (std::size_t k = 2 * stride - 1; k < m; k += 2 * stride) {
            T t      = buf[k - stride];
            buf[k - stride] = buf[k];       // left child gets parent value
            buf[k]  += t;                   // right child gets parent + old left
        }
    }
    // Post-condition: buf[i] = sum(original[0..i-1])
}

template<typename T>
static void exclusive_scan_impl(const T* in, T* out, std::size_t n) {
    if (n == 0) return;
    std::size_t m = next_pow2(n);
    std::vector<T> buf(m, T(0));
    std::copy(in, in + n, buf.data());
    blelloch_exclusive_inplace(buf.data(), m);
    std::copy(buf.data(), buf.data() + n, out);
}

template<typename T>
static void inclusive_scan_impl(const T* in, T* out, std::size_t n) {
    if (n == 0) return;
    exclusive_scan_impl(in, out, n);
    // inclusive[i] = exclusive[i] + in[i]  (shift by original element)
    for (std::size_t i = 0; i < n; ++i)
        out[i] += in[i];
}

} // anonymous namespace

void exclusive_scan(const float* in, float* out, std::size_t n) {
    exclusive_scan_impl(in, out, n);
}
void exclusive_scan(const double* in, double* out, std::size_t n) {
    exclusive_scan_impl(in, out, n);
}
void exclusive_scan(const int32_t* in, int32_t* out, std::size_t n) {
    exclusive_scan_impl(in, out, n);
}

void inclusive_scan(const float* in, float* out, std::size_t n) {
    inclusive_scan_impl(in, out, n);
}
void inclusive_scan(const double* in, double* out, std::size_t n) {
    inclusive_scan_impl(in, out, n);
}
void inclusive_scan(const int32_t* in, int32_t* out, std::size_t n) {
    inclusive_scan_impl(in, out, n);
}

} // namespace vgre
