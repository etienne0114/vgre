#pragma once
#include <cstdint>
#include <cstddef>

namespace vgre {

// LSD radix sort for uint32_t keys — O(4n) = O(n) time, O(n + 256) space.
// 4 passes, 8-bit radix: each pass counting-sorts on one 8-bit digit.
// Math invariant: after pass i, array is sorted by digits 0..i (least significant).
// Stable sort: equal keys preserve their original relative order.
void radix_sort_u32(uint32_t* keys, std::size_t n);

// Sort key–value pairs by key (keys reordered in-place, values follow).
void radix_sort_u32_kv(uint32_t* keys, uint32_t* values, std::size_t n);

} // namespace vgre
