// Block Sparse Matrix Multiplication Implementation
// Implements SIMD-optimized block-sparse matrix operations

#include "block_sparse.h"
#include <cstring>
#include <algorithm>
#include <vector>

#ifdef __AVX512F__
#include <immintrin.h>
#endif

#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace vgre {
namespace math {

template<typename T, typename IndexType>
void csrToBlockSparse(const T* values, const IndexType* col_indices,
                     const IndexType* row_offsets, IndexType num_rows, IndexType num_cols,
                     IndexType block_size, BlockSparseMatrix<T, IndexType>& bsm) {
    // Count non-zero blocks
    IndexType num_row_blocks = (num_rows + block_size - 1) / block_size;
    IndexType num_col_blocks = (num_cols + block_size - 1) / block_size;
    
    std::vector<bool> block_nonzero(num_row_blocks * num_col_blocks, false);
    
    for (IndexType i = 0; i < num_rows; ++i) {
        IndexType row_start = row_offsets[i];
        IndexType row_end = row_offsets[i + 1];
        
        for (IndexType j = row_start; j < row_end; ++j) {
            IndexType col = col_indices[j];
            IndexType block_row = i / block_size;
            IndexType block_col = col / block_size;
            block_nonzero[block_row * num_col_blocks + block_col] = true;
        }
    }
    
    IndexType num_blocks = 0;
    for (bool nonzero : block_nonzero) {
        if (nonzero) num_blocks++;
    }
    
    // Allocate block sparse matrix
    bsm.num_rows = num_rows;
    bsm.num_cols = num_cols;
    bsm.block_size = block_size;
    bsm.num_blocks = num_blocks;
    bsm.block_row_indices = new IndexType[num_blocks];
    bsm.block_col_indices = new IndexType[num_blocks];
    bsm.block_values = new T[num_blocks * block_size * block_size];
    
    // Fill block sparse matrix
    IndexType block_idx = 0;
    for (IndexType br = 0; br < num_row_blocks; ++br) {
        for (IndexType bc = 0; bc < num_col_blocks; ++bc) {
            if (!block_nonzero[br * num_col_blocks + bc]) continue;
            
            bsm.block_row_indices[block_idx] = br;
            bsm.block_col_indices[block_idx] = bc;
            
            // Initialize block to zero
            T* block = bsm.block_values + block_idx * block_size * block_size;
            std::memset(block, 0, block_size * block_size * sizeof(T));
            
            // Fill block from CSR
            IndexType row_start = br * block_size;
            IndexType row_end = std::min((br + 1) * block_size, num_rows);
            IndexType col_start = bc * block_size;
            IndexType col_end = std::min((bc + 1) * block_size, num_cols);
            
            for (IndexType i = row_start; i < row_end; ++i) {
                IndexType csr_start = row_offsets[i];
                IndexType csr_end = row_offsets[i + 1];
                
                for (IndexType j = csr_start; j < csr_end; ++j) {
                    IndexType col = col_indices[j];
                    if (col >= col_start && col < col_end) {
                        IndexType block_i = i - row_start;
                        IndexType block_j = col - col_start;
                        block[block_i * block_size + block_j] = values[j];
                    }
                }
            }
            
            block_idx++;
        }
    }
}

template<typename T>
void simdBlockMultiply(const T* A_block, const T* B_block, T* C_block,
                      size_t block_size) {
    // Initialize C_block to zero
    std::memset(C_block, 0, block_size * block_size * sizeof(T));
    
    // Standard matrix multiplication for the block
    for (size_t i = 0; i < block_size; ++i) {
        for (size_t k = 0; k < block_size; ++k) {
            T a_ik = A_block[i * block_size + k];
            for (size_t j = 0; j < block_size; ++j) {
                C_block[i * block_size + j] += a_ik * B_block[k * block_size + j];
            }
        }
    }
}

#ifdef __AVX512F__
void avx512BlockMultiplyF(const float* A_block, const float* B_block, float* C_block,
                         size_t block_size) {
    // Initialize C_block to zero
    std::memset(C_block, 0, block_size * block_size * sizeof(float));
    
    // Use AVX-512 for block sizes that are multiples of 16
    if (block_size >= 16) {
        for (size_t i = 0; i < block_size; ++i) {
            for (size_t k = 0; k < block_size; ++k) {
                __m512 a_ik = _mm512_set1_ps(A_block[i * block_size + k]);
                
                for (size_t j = 0; j < block_size; j += 16) {
                    __m512 b_kj = _mm512_loadu_ps(B_block + k * block_size + j);
                    __m512 c_ij = _mm512_loadu_ps(C_block + i * block_size + j);
                    __m512 result = _mm512_fmadd_ps(a_ik, b_kj, c_ij);
                    _mm512_storeu_ps(C_block + i * block_size + j, result);
                }
            }
        }
    } else {
        // Fall back to standard implementation for small blocks
        simdBlockMultiply(A_block, B_block, C_block, block_size);
    }
}
#endif

#ifdef __AVX2__
void avx2BlockMultiplyF(const float* A_block, const float* B_block, float* C_block,
                       size_t block_size) {
    // Initialize C_block to zero
    std::memset(C_block, 0, block_size * block_size * sizeof(float));
    
    // Use AVX2 for block sizes that are multiples of 8
    if (block_size >= 8) {
        for (size_t i = 0; i < block_size; ++i) {
            for (size_t k = 0; k < block_size; ++k) {
                __m256 a_ik = _mm256_set1_ps(A_block[i * block_size + k]);
                
                for (size_t j = 0; j < block_size; j += 8) {
                    __m256 b_kj = _mm256_loadu_ps(B_block + k * block_size + j);
                    __m256 c_ij = _mm256_loadu_ps(C_block + i * block_size + j);
                    __m256 result = _mm256_fmadd_ps(a_ik, b_kj, c_ij);
                    _mm256_storeu_ps(C_block + i * block_size + j, result);
                }
            }
        }
    } else {
        // Fall back to standard implementation for small blocks
        simdBlockMultiply(A_block, B_block, C_block, block_size);
    }
}
#endif

template<typename T, typename IndexType>
void blockSparseMV(const BlockSparseMatrix<T, IndexType>& bsm,
                  const T* x, T* y) {
    // Initialize y to zero
    std::memset(y, 0, bsm.num_rows * sizeof(T));
    
    IndexType block_size = bsm.block_size;
    
    for (IndexType b = 0; b < bsm.num_blocks; ++b) {
        IndexType block_row = bsm.block_row_indices[b];
        IndexType block_col = bsm.block_col_indices[b];
        const T* block = bsm.block_values + b * block_size * block_size;
        
        IndexType row_start = block_row * block_size;
        IndexType row_end = std::min((block_row + 1) * block_size, bsm.num_rows);
        IndexType col_start = block_col * block_size;
        IndexType col_end = std::min((block_col + 1) * block_size, bsm.num_cols);
        
        for (IndexType i = row_start; i < row_end; ++i) {
            IndexType block_i = i - row_start;
            T sum = T(0);
            
            for (IndexType j = col_start; j < col_end; ++j) {
                IndexType block_j = j - col_start;
                sum += block[block_i * block_size + block_j] * x[j];
            }
            
            y[i] += sum;
        }
    }
}

template<typename T, typename IndexType>
void blockSparseMM(const BlockSparseMatrix<T, IndexType>& A,
                  const BlockSparseMatrix<T, IndexType>& B,
                  T* C, IndexType C_rows, IndexType C_cols) {
    // Initialize C to zero
    std::memset(C, 0, C_rows * C_cols * sizeof(T));
    
    IndexType block_size = A.block_size;
    
    for (IndexType b_a = 0; b_a < A.num_blocks; ++b_a) {
        IndexType block_row_a = A.block_row_indices[b_a];
        IndexType block_col_a = A.block_col_indices[b_a];
        const T* block_a = A.block_values + b_a * block_size * block_size;
        
        for (IndexType b_b = 0; b_b < B.num_blocks; ++b_b) {
            IndexType block_row_b = B.block_row_indices[b_b];
            IndexType block_col_b = B.block_col_indices[b_b];
            
            // Only multiply if column of A matches row of B
            if (block_col_a != block_row_b) continue;
            
            const T* block_b = B.block_values + b_b * block_size * block_size;
            
            IndexType row_start = block_row_a * block_size;
            IndexType row_end = std::min((block_row_a + 1) * block_size, C_rows);
            IndexType col_start = block_col_b * block_size;
            IndexType col_end = std::min((block_col_b + 1) * block_size, C_cols);
            
            // Temporary block for multiplication result
            T temp_block[16 * 16];  // Max block size 16x16
            simdBlockMultiply(block_a, block_b, temp_block, block_size);
            
            // Add to C
            for (IndexType i = row_start; i < row_end; ++i) {
                IndexType block_i = i - row_start;
                for (IndexType j = col_start; j < col_end; ++j) {
                    IndexType block_j = j - col_start;
                    C[i * C_cols + j] += temp_block[block_i * block_size + block_j];
                }
            }
        }
    }
}

template<typename T, typename IndexType>
void nmSparseMV(const NMSparseMatrix<T, IndexType>& nsm,
               const T* x, T* y) {
    // Initialize y to zero
    std::memset(y, 0, nsm.num_rows * sizeof(T));
    
    for (IndexType i = 0; i < nsm.nnz; ++i) {
        IndexType row = nsm.row_indices[i];
        IndexType col = nsm.col_indices[i];
        T val = nsm.values[i];
        
        y[row] += val * x[col];
    }
}

// Explicit template instantiations
template void csrToBlockSparse<float, int32_t>(const float*, const int32_t*, const int32_t*, int32_t, int32_t, int32_t, BlockSparseMatrix<float, int32_t>&);
template void csrToBlockSparse<double, int32_t>(const double*, const int32_t*, const int32_t*, int32_t, int32_t, int32_t, BlockSparseMatrix<double, int32_t>&);
template void csrToBlockSparse<float, int64_t>(const float*, const int64_t*, const int64_t*, int64_t, int64_t, int64_t, BlockSparseMatrix<float, int64_t>&);
template void csrToBlockSparse<double, int64_t>(const double*, const int64_t*, const int64_t*, int64_t, int64_t, int64_t, BlockSparseMatrix<double, int64_t>&);

template void blockSparseMV<float, int32_t>(const BlockSparseMatrix<float, int32_t>&, const float*, float*);
template void blockSparseMV<double, int32_t>(const BlockSparseMatrix<double, int32_t>&, const double*, double*);
template void blockSparseMV<float, int64_t>(const BlockSparseMatrix<float, int64_t>&, const float*, float*);
template void blockSparseMV<double, int64_t>(const BlockSparseMatrix<double, int64_t>&, const double*, double*);

template void blockSparseMM<float, int32_t>(const BlockSparseMatrix<float, int32_t>&, const BlockSparseMatrix<float, int32_t>&, float*, int32_t, int32_t);
template void blockSparseMM<double, int32_t>(const BlockSparseMatrix<double, int32_t>&, const BlockSparseMatrix<double, int32_t>&, double*, int32_t, int32_t);
template void blockSparseMM<float, int64_t>(const BlockSparseMatrix<float, int64_t>&, const BlockSparseMatrix<float, int64_t>&, float*, int64_t, int64_t);
template void blockSparseMM<double, int64_t>(const BlockSparseMatrix<double, int64_t>&, const BlockSparseMatrix<double, int64_t>&, double*, int64_t, int64_t);

template void nmSparseMV<float, int32_t>(const NMSparseMatrix<float, int32_t>&, const float*, float*);
template void nmSparseMV<double, int32_t>(const NMSparseMatrix<double, int32_t>&, const double*, double*);
template void nmSparseMV<float, int64_t>(const NMSparseMatrix<float, int64_t>&, const float*, float*);
template void nmSparseMV<double, int64_t>(const NMSparseMatrix<double, int64_t>&, const double*, double*);

} // namespace math
} // namespace vgre
