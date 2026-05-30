// Block Sparse Matrix Multiplication with SIMD
// Implements block-sparse matrix multiplication optimized for AVX-512/AVX2
// Based on research from SELLPACK, VBSF, and ALBUS formats

#ifndef VGRE_BLOCK_SPARSE_H
#define VGRE_BLOCK_SPARSE_H

#include <cstdint>
#include <cstddef>

namespace vgre {
namespace math {

// Block sparse matrix storage format
// Stores non-zero blocks in a compressed format for efficient SIMD processing
template<typename T = float, typename IndexType = int32_t>
struct BlockSparseMatrix {
    IndexType num_rows;      // Number of rows
    IndexType num_cols;      // Number of columns
    IndexType block_size;    // Block size (e.g., 4, 8, 16)
    IndexType num_blocks;    // Number of non-zero blocks
    
    IndexType* block_row_indices;  // Row index of each block
    IndexType* block_col_indices;  // Column index of each block
    T* block_values;               // Block values (block_size * block_size per block)
    
    BlockSparseMatrix()
        : num_rows(0), num_cols(0), block_size(1), num_blocks(0),
          block_row_indices(nullptr), block_col_indices(nullptr), block_values(nullptr) {}
    
    ~BlockSparseMatrix() {
        delete[] block_row_indices;
        delete[] block_col_indices;
        delete[] block_values;
    }
};

// Convert CSR to block-sparse format
template<typename T, typename IndexType>
void csrToBlockSparse(const T* values, const IndexType* col_indices,
                     const IndexType* row_offsets, IndexType num_rows, IndexType num_cols,
                     IndexType block_size, BlockSparseMatrix<T, IndexType>& bsm);

// Block sparse matrix-vector multiplication (SpMV) with SIMD
template<typename T, typename IndexType>
void blockSparseMV(const BlockSparseMatrix<T, IndexType>& bsm,
                  const T* x, T* y);

// Block sparse matrix-matrix multiplication with SIMD
template<typename T, typename IndexType>
void blockSparseMM(const BlockSparseMatrix<T, IndexType>& A,
                  const BlockSparseMatrix<T, IndexType>& B,
                  T* C, IndexType C_rows, IndexType C_cols);

// SIMD-optimized block multiplication
template<typename T>
void simdBlockMultiply(const T* A_block, const T* B_block, T* C_block,
                      size_t block_size);

// AVX-512 optimized block multiplication (float)
#ifdef __AVX512F__
void avx512BlockMultiplyF(const float* A_block, const float* B_block, float* C_block,
                         size_t block_size);
#endif

// AVX2 optimized block multiplication (float)
#ifdef __AVX2__
void avx2BlockMultiplyF(const float* A_block, const float* B_block, float* C_block,
                       size_t block_size);
#endif

// N:M structured sparsity support (e.g., 2:4, 4:8)
// Used for tensor core optimization on GPUs, emulated on CPU
template<typename T, typename IndexType>
struct NMSparseMatrix {
    IndexType num_rows;
    IndexType num_cols;
    IndexType nnz;
    IndexType n_ratio;  // N in N:M (e.g., 2 for 2:4)
    IndexType m_ratio;  // M in N:M (e.g., 4 for 2:4)
    
    IndexType* row_indices;
    IndexType* col_indices;
    T* values;
    uint8_t* metadata;  // Sparsity pattern metadata
    
    NMSparseMatrix()
        : num_rows(0), num_cols(0), nnz(0), n_ratio(0), m_ratio(0),
          row_indices(nullptr), col_indices(nullptr), values(nullptr), metadata(nullptr) {}
    
    ~NMSparseMatrix() {
        delete[] row_indices;
        delete[] col_indices;
        delete[] values;
        delete[] metadata;
    }
};

// N:M sparse matrix-vector multiplication
template<typename T, typename IndexType>
void nmSparseMV(const NMSparseMatrix<T, IndexType>& nsm,
               const T* x, T* y);

} // namespace math
} // namespace vgre

#endif // VGRE_BLOCK_SPARSE_H
