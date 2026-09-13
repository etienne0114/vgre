// From-scratch HNSW approximate-nearest-neighbor index for the lightweight
// (LLVM/BLAS/CUDA-free) libvgre_nn — the retrieval half of a local RAG stack.
//
// Real Malkov–Yashunin HNSW (arXiv:1603.09320): multi-layer navigable
// small-world graph, geometric level assignment, best-first layer search, and
// the neighbor-selection heuristic (Algorithm 4). No external vector-DB
// dependency — embed with the in-tree model, index and search here, all CPU.
//
// Metrics: 0 = cosine (vectors are L2-normalized on insert/query; distance is
// 1 − dot), 1 = squared L2. Exact for small sets (search degenerates to scan
// when ef ≥ n), sublinear for large ones.
#ifndef VGRE_XLA_VECTOR_INDEX_H
#define VGRE_XLA_VECTOR_INDEX_H

#include <cstdint>
#include <stddef.h>

#include "vgre/common/platform.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void* vgre_vindex;

// Create an index. dim > 0; M (graph degree, typical 8–48; 0 → 16);
// ef_construction (build-time beam, typical 100–400; 0 → 200);
// metric: 0 = cosine, 1 = squared L2. NULL on invalid args.
VGRE_PUBLIC_API vgre_vindex vgre_vindex_create(int dim, int M,
                                               int ef_construction, int metric);

// Insert one vector under a caller-chosen 64-bit id. Returns 0 on success.
VGRE_PUBLIC_API int vgre_vindex_add(vgre_vindex h, const float* vec, int64_t id);

// k-NN search. ef is the query-time beam (clamped to ≥ k; 0 → max(64, k)).
// Fills up to k (id, distance) pairs, nearest first; returns the count found.
VGRE_PUBLIC_API int vgre_vindex_search(vgre_vindex h, const float* query,
                                       int k, int ef,
                                       int64_t* out_ids, float* out_dists);

VGRE_PUBLIC_API int64_t vgre_vindex_size(vgre_vindex h);
VGRE_PUBLIC_API int     vgre_vindex_dim(vgre_vindex h);

// Persist / restore the full graph (portable little-endian binary).
VGRE_PUBLIC_API int         vgre_vindex_save(vgre_vindex h, const char* path);
VGRE_PUBLIC_API vgre_vindex vgre_vindex_load(const char* path);

VGRE_PUBLIC_API void vgre_vindex_free(vgre_vindex h);

#ifdef __cplusplus
}
#endif

#endif  // VGRE_XLA_VECTOR_INDEX_H
