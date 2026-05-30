// Zero-Copy Sparse Matrix View System
// Implements lightweight view-based sparse matrix format conversions
// avoiding deep copies and memory allocations

#ifndef VGRE_SPARSE_VIEW_H
#define VGRE_SPARSE_VIEW_H

#include <cstdint>
#include <cstddef>

namespace vgre {
namespace sparse {

// Sparse matrix format types
enum class SparseFormat {
    CSR,  // Compressed Sparse Row
    CSC,  // Compressed Sparse Column
    COO,  // Coordinate format
    BSR,  // Block Sparse Row
    BSC   // Block Sparse Column
};

// Zero-copy view descriptor for sparse matrices
template<typename T = float, typename IndexType = int32_t>
struct SparseMatrixView {
    void* rowOffsets;      // Pointer to row offset array (CSR/BSR) or column offset (CSC/BSC)
    void* colIndices;      // Pointer to column index array (CSR/BSR) or row index (CSC/BSC)
    void* values;          // Pointer to values array
    IndexType rows;        // Number of rows
    IndexType cols;        // Number of columns
    IndexType nnz;         // Number of non-zero elements
    IndexType blockSize;   // Block size for BSR/BSC (1 for CSR/CSC/COO)
    SparseFormat format;   // Matrix format
    bool isView;           // True if this is a view (no ownership)
    
    // Default constructor
    SparseMatrixView() 
        : rowOffsets(nullptr), colIndices(nullptr), values(nullptr),
          rows(0), cols(0), nnz(0), blockSize(1), format(SparseFormat::CSR),
          isView(true) {}
    
    // Constructor from existing data (zero-copy view)
    SparseMatrixView(void* rowOff, void* colInd, void* vals,
                     IndexType r, IndexType c, IndexType n,
                     IndexType bs = 1, SparseFormat fmt = SparseFormat::CSR)
        : rowOffsets(rowOff), colIndices(colInd), values(vals),
          rows(r), cols(c), nnz(n), blockSize(bs), format(fmt),
          isView(true) {}
    
    // Convert view to different format (zero-copy if possible)
    SparseMatrixView<T, IndexType> toView(SparseFormat targetFormat) const;
    
    // Check if zero-copy conversion is possible
    bool canConvertZeroCopy(SparseFormat targetFormat) const;
};

// View-based format conversion functions
template<typename T, typename IndexType>
SparseMatrixView<T, IndexType> csrToCscView(const SparseMatrixView<T, IndexType>& csr);

template<typename T, typename IndexType>
SparseMatrixView<T, IndexType> csrToBsrView(const SparseMatrixView<T, IndexType>& csr, IndexType blockSize);

template<typename T, typename IndexType>
SparseMatrixView<T, IndexType> cooToCsrView(const SparseMatrixView<T, IndexType>& coo);

} // namespace sparse
} // namespace vgre

#endif // VGRE_SPARSE_VIEW_H
