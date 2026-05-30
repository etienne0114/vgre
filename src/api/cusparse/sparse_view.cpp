// Zero-Copy Sparse Matrix View System Implementation
// Implements lightweight view-based sparse matrix format conversions

#include "sparse_view.h"
#include <cstring>
#include <algorithm>

namespace vgre {
namespace sparse {

template<typename T, typename IndexType>
bool SparseMatrixView<T, IndexType>::canConvertZeroCopy(SparseFormat targetFormat) const {
    // Zero-copy conversions are possible for:
    // 1. CSR <-> CSC (just reinterpret pointers and swap rows/cols)
    // 2. CSR -> BSR with blockSize=1 (same structure)
    // 3. COO -> CSR requires building row offsets (not zero-copy)
    
    if (format == targetFormat) return true;
    
    if (format == SparseFormat::CSR && targetFormat == SparseFormat::CSC) {
        return true; // Can reinterpret as CSC
    }
    
    if (format == SparseFormat::CSC && targetFormat == SparseFormat::CSR) {
        return true; // Can reinterpret as CSR
    }
    
    if (format == SparseFormat::CSR && targetFormat == SparseFormat::BSR) {
        return blockSize == 1; // BSR with block size 1 is same as CSR
    }
    
    return false;
}

template<typename T, typename IndexType>
SparseMatrixView<T, IndexType> SparseMatrixView<T, IndexType>::toView(SparseFormat targetFormat) const {
    if (format == targetFormat) {
        return *this;
    }
    
    // CSR <-> CSC zero-copy conversion
    if (format == SparseFormat::CSR && targetFormat == SparseFormat::CSC) {
        SparseMatrixView<T, IndexType> cscView;
        cscView.rowOffsets = colIndices;  // CSC column offsets = CSR column indices
        cscView.colIndices = rowOffsets;  // CSC row indices = CSR row offsets
        cscView.values = values;
        cscView.rows = cols;  // Swap dimensions
        cscView.cols = rows;
        cscView.nnz = nnz;
        cscView.blockSize = blockSize;
        cscView.format = SparseFormat::CSC;
        cscView.isView = true;
        return cscView;
    }
    
    if (format == SparseFormat::CSC && targetFormat == SparseFormat::CSR) {
        SparseMatrixView<T, IndexType> csrView;
        csrView.rowOffsets = colIndices;  // CSR row offsets = CSC row indices
        csrView.colIndices = rowOffsets;  // CSR column indices = CSC column offsets
        csrView.values = values;
        csrView.rows = cols;  // Swap dimensions
        csrView.cols = rows;
        csrView.nnz = nnz;
        csrView.blockSize = blockSize;
        csrView.format = SparseFormat::CSR;
        csrView.isView = true;
        return csrView;
    }
    
    // CSR -> BSR with blockSize=1 (zero-copy)
    if (format == SparseFormat::CSR && targetFormat == SparseFormat::BSR && blockSize == 1) {
        SparseMatrixView<T, IndexType> bsrView;
        bsrView.rowOffsets = rowOffsets;
        bsrView.colIndices = colIndices;
        bsrView.values = values;
        bsrView.rows = rows;
        bsrView.cols = cols;
        bsrView.nnz = nnz;
        bsrView.blockSize = 1;
        bsrView.format = SparseFormat::BSR;
        bsrView.isView = true;
        return bsrView;
    }
    
    // For other conversions, return empty view (caller should use full conversion)
    return SparseMatrixView<T, IndexType>();
}

template<typename T, typename IndexType>
SparseMatrixView<T, IndexType> csrToCscView(const SparseMatrixView<T, IndexType>& csr) {
    return csr.toView(SparseFormat::CSC);
}

template<typename T, typename IndexType>
SparseMatrixView<T, IndexType> csrToBsrView(const SparseMatrixView<T, IndexType>& csr, IndexType blockSize) {
    if (blockSize == 1) {
        return csr.toView(SparseFormat::BSR);
    }
    // For block size > 1, need full conversion (not zero-copy)
    return SparseMatrixView<T, IndexType>();
}

template<typename T, typename IndexType>
SparseMatrixView<T, IndexType> cooToCsrView(const SparseMatrixView<T, IndexType>& coo) {
    // COO to CSR requires building row offsets (not zero-copy)
    // This would need a full conversion implementation
    return SparseMatrixView<T, IndexType>();
}

// Explicit template instantiations for common types
template struct SparseMatrixView<float, int32_t>;
template struct SparseMatrixView<double, int32_t>;
template struct SparseMatrixView<float, int64_t>;
template struct SparseMatrixView<double, int64_t>;

// canConvertZeroCopy is already instantiated via the `template struct` above

// toView is already instantiated via the `template struct` above

template SparseMatrixView<float, int32_t> csrToCscView(const SparseMatrixView<float, int32_t>&);
template SparseMatrixView<double, int32_t> csrToCscView(const SparseMatrixView<double, int32_t>&);
template SparseMatrixView<float, int64_t> csrToCscView(const SparseMatrixView<float, int64_t>&);
template SparseMatrixView<double, int64_t> csrToCscView(const SparseMatrixView<double, int64_t>&);

} // namespace sparse
} // namespace vgre
