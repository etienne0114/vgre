#pragma once
#include <cstdint>
#include <vector>

namespace vgre {

// Blelloch (1990) work-efficient prefix scan — O(n log n) work, O(log n) depth.
// Math: after down-sweep, out[i] = sum(in[0..i-1])  (exclusive prefix sum).
//       inclusive:  out[i] = sum(in[0..i]).

// Exclusive prefix scan: out[i] = sum(in[0..i-1]), out[0] = 0. O(n log n).
void exclusive_scan(const float*  in, float*  out, std::size_t n);
void exclusive_scan(const double* in, double* out, std::size_t n);
void exclusive_scan(const int32_t* in, int32_t* out, std::size_t n);

// Inclusive prefix scan: out[i] = sum(in[0..i]). O(n log n).
void inclusive_scan(const float*  in, float*  out, std::size_t n);
void inclusive_scan(const double* in, double* out, std::size_t n);
void inclusive_scan(const int32_t* in, int32_t* out, std::size_t n);

} // namespace vgre
