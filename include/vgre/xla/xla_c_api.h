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

// ── builder C ABI ────────────────────────────────────────────────────────────
// Construct an HLO module op-by-op (the target a StableHLO→HLO translator emits
// into, avoiding any duplicate serialization format on the frontend side). Each
// op returns the new instruction's id (>=0) or -1 on error. `op` codes are the
// HloOp enum ordinals. dims/perm arrays are int64; pass ndim=0 for a scalar.

uint64_t vgre_xla_builder_new(void);

int vgre_xla_b_parameter(uint64_t b, int index, const int64_t* dims, int ndim);
int vgre_xla_b_constant(uint64_t b, const int64_t* dims, int ndim, const float* data, int n);
int vgre_xla_b_binary(uint64_t b, int op, int lhs, int rhs);
int vgre_xla_b_unary(uint64_t b, int op, int x);
int vgre_xla_b_broadcast(uint64_t b, int x, const int64_t* out_dims, int n_out,
                         const int64_t* bcast_dims, int n_bd);
int vgre_xla_b_reshape(uint64_t b, int x, const int64_t* out_dims, int n_out);
int vgre_xla_b_transpose(uint64_t b, int x, const int64_t* perm, int n_perm);
int vgre_xla_b_dot(uint64_t b, int lhs, int rhs);
int vgre_xla_b_reduce(uint64_t b, int x, const int64_t* dims, int n_dims,
                      const char* kind, float init);
int vgre_xla_b_compare(uint64_t b, int lhs, int rhs, const char* dir);
int vgre_xla_b_select(uint64_t b, int pred, int on_true, int on_false);

// — extended ops (real-model coverage) —
int vgre_xla_b_dot_general(uint64_t b, int lhs, int rhs,
                           const int64_t* lb, int nlb, const int64_t* rb, int nrb,
                           const int64_t* lc, int nlc, const int64_t* rc, int nrc,
                           const int64_t* out_dims, int n_out);
int vgre_xla_b_concatenate(uint64_t b, const int* xs, int n_xs, int64_t dim,
                           const int64_t* out_dims, int n_out);
int vgre_xla_b_slice(uint64_t b, int x, const int64_t* starts, const int64_t* limits,
                     const int64_t* strides, int n, const int64_t* out_dims, int n_out);
int vgre_xla_b_pad(uint64_t b, int x, int pad_val, const int64_t* low, const int64_t* high,
                   const int64_t* interior, int n, const int64_t* out_dims, int n_out);
int vgre_xla_b_convolution(uint64_t b, int lhs, int rhs, const int64_t* out_dims, int n_out,
                           int in_batch, int in_feat, int k_out, int k_in,
                           int out_batch, int out_feat,
                           const int64_t* in_sp, const int64_t* k_sp, const int64_t* out_sp, int n_sp,
                           const int64_t* strides, const int64_t* pad_lo, const int64_t* pad_hi,
                           const int64_t* rhs_dil, int groups);
int vgre_xla_b_gather(uint64_t b, int operand, int indices, const int64_t* out_dims, int n_out,
                      const int64_t* offset_dims, int n_off, const int64_t* collapsed, int n_col,
                      const int64_t* start_map, int n_sm, const int64_t* slice_sizes, int n_ss,
                      int index_vector_dim);
int vgre_xla_b_reduce_window(uint64_t b, int x, int init, const char* kind,
                             const int64_t* out_dims, int n_out,
                             const int64_t* win_dims, const int64_t* win_strides,
                             const int64_t* pad_lo, const int64_t* pad_hi,
                             const int64_t* base_dil, const int64_t* win_dil, int n);

// — control flow —
// `cond`/`body` are builder handles (from vgre_xla_builder_new) that this call
// consumes: their modules become the loop's cond/body sub-computations.
int vgre_xla_b_while(uint64_t b, const int* inits, int n_init, uint64_t cond, uint64_t body,
                     const int64_t* primary_dims, int n_primary);
int vgre_xla_b_tuple(uint64_t b, const int* elems, int n);
int vgre_xla_b_get_tuple_element(uint64_t b, int src, int index,
                                 const int64_t* out_dims, int n_out);
int vgre_xla_b_dynamic_slice(uint64_t b, int operand, const int* starts, int n_starts,
                             const int64_t* sizes, int n_sizes,
                             const int64_t* out_dims, int n_out);
int vgre_xla_b_dynamic_update_slice(uint64_t b, int operand, int update,
                                    const int* starts, int n_starts,
                                    const int64_t* out_dims, int n_out);
int vgre_xla_b_reverse(uint64_t b, int x, const int64_t* dims, int n);
int vgre_xla_b_iota(uint64_t b, const int64_t* out_dims, int n_out, int64_t dim);
// `cmp`/`body` are builder handles (consumed) for the comparator / reducer.
int vgre_xla_b_sort(uint64_t b, const int* operands, int n_ops, int64_t dim, uint64_t cmp);
int vgre_xla_b_reduce_general(uint64_t b, const int* operands, int n_ops,
                              const int64_t* dims, int n_dims, uint64_t body,
                              const int64_t* primary_dims, int n_primary);

void vgre_xla_b_set_root(uint64_t b, int id);

// Finalize the builder into an executable (consumes the builder). Returns an
// executable handle usable with vgre_xla_execute / _output_numel / _free, or 0.
uint64_t vgre_xla_b_compile(uint64_t b);
// Discard a builder that was never compiled.
void vgre_xla_b_free(uint64_t b);

#ifdef __cplusplus
}
#endif

#endif // VGRE_XLA_XLA_C_API_H
