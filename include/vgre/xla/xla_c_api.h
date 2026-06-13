// VGRE XLA backend — executable C ABI (docs/missingFeatures.md §4.2, step 2).
// The compile→executable→execute lifecycle over the HLO engine, exposed as a
// stable C ABI. A PJRT plugin's PJRT_Client_Compile / PJRT_LoadedExecutable_Execute
// map directly onto these: compile a (serialized) HLO module into an executable
// handle, then execute it on flat float32 input buffers.
#ifndef VGRE_XLA_XLA_C_API_H
#define VGRE_XLA_XLA_C_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Compile a serialized HLO module (from vgre_xla_serialize / HloModule) into an
// executable. Returns a non-zero handle, or 0 on parse error.
uint64_t vgre_xla_compile(const void* blob, size_t len);

// Execute `exe`. `in_data`/`in_numel` are parallel arrays of length `n_in`, one
// flat float32 buffer per parameter (in param-index order). The root result is
// written into `out_data` (capacity `out_capacity` floats). Returns the number
// of output elements written, or -1 on error (bad handle / capacity too small).
int64_t vgre_xla_execute(uint64_t exe, const float* const* in_data, const int64_t* in_numel,
                         int n_in, float* out_data, int64_t out_capacity);

// Number of elements the root result will produce (for sizing out_data), or -1.
int64_t vgre_xla_output_numel(uint64_t exe);

void vgre_xla_free(uint64_t exe);

#ifdef __cplusplus
}
#endif

#endif // VGRE_XLA_XLA_C_API_H
