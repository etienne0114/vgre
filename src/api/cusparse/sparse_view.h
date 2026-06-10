// Sparse Matrix View / Conversion System
// Zero-copy views where the layout is genuinely identical; real (owned)
// conversions otherwise. Conversions are correct (they represent the SAME
// matrix in the target format, not its transpose).

#ifndef VGRE_SPARSE_VIEW_H
#define VGRE_SPARSE_VIEW_H

#include <cstdint>
#include <cstddef>
#include <memory>
#include <vector>

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

// Sparse matrix descriptor. When isView==true the pointer fields alias
// caller-owned arrays (zero-copy). When isView==false the matrix owns its arrays
// via the shared_ptr buffers below and the pointer fields alias into them, so a
// converted matrix is self-contained and copyable.
//
// Array meanings by format (length in []):
//   CSR : rowOffsets[rows+1] = row pointers, colIndices[nnz] = column indices
//   CSC : rowOffsets[cols+1] = column pointers, colIndices[nnz] = row indices
//   COO : rowOffsets[nnz]    = row indices,    colIndices[nnz] = column indices
//   BSR : rowOffsets[rows/bs+1] block-row pointers, colIndices = block columns
template<typename T = float, typename IndexType = int32_t>
struct SparseMatrixView {
    void* rowOffsets;
    void* colIndices;
    void* values;
    IndexType rows;
    IndexType cols;
    IndexType nnz;
    IndexType blockSize;
    SparseFormat format;
    bool isView;

    // Owned backing storage (non-null only when isView==false).
    std::shared_ptr<std::vector<IndexType>> ownRow;
    std::shared_ptr<std::vector<IndexType>> ownCol;
    std::shared_ptr<std::vector<T>>         ownVal;

    SparseMatrixView()
        : rowOffsets(nullptr), colIndices(nullptr), values(nullptr),
          rows(0), cols(0), nnz(0), blockSize(1), format(SparseFormat::CSR),
          isView(true) {}

    SparseMatrixView(void* rowOff, void* colInd, void* vals,
                     IndexType r, IndexType c, IndexType n,
                     IndexType bs = 1, SparseFormat fmt = SparseFormat::CSR)
        : rowOffsets(rowOff), colIndices(colInd), values(vals),
          rows(r), cols(c), nnz(n), blockSize(bs), format(fmt),
          isView(true) {}

    // Convert to a different format. Returns a zero-copy view when the layout is
    // identical, otherwise a newly-allocated owned matrix. An empty matrix
    // (nnz==0, all null) is returned for an unsupported format pair.
    SparseMatrixView<T, IndexType> toView(SparseFormat targetFormat) const;

    // True only when the target shares this matrix's exact array layout, so the
    // conversion needs no allocation or data movement.
    bool canConvertZeroCopy(SparseFormat targetFormat) const;

    bool valid() const { return values != nullptr || nnz == 0; }
};

// Convenience wrappers (all delegate to toView and return correct results).
template<typename T, typename IndexType>
SparseMatrixView<T, IndexType> csrToCscView(const SparseMatrixView<T, IndexType>& csr);

template<typename T, typename IndexType>
SparseMatrixView<T, IndexType> csrToBsrView(const SparseMatrixView<T, IndexType>& csr, IndexType blockSize);

template<typename T, typename IndexType>
SparseMatrixView<T, IndexType> cooToCsrView(const SparseMatrixView<T, IndexType>& coo);

} // namespace sparse
} // namespace vgre

#endif // VGRE_SPARSE_VIEW_H
