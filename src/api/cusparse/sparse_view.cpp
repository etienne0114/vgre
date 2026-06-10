// Sparse Matrix View / Conversion System — implementation.
//
// Conversions that change the array layout (COO→CSR, CSR↔CSC, CSR↔COO) build new
// owned arrays and are numerically correct: the result is the SAME matrix in the
// target format. The only genuine zero-copy cases are an identity conversion and
// CSR↔BSR with blockSize==1 (which share the exact CSR layout).

#include "sparse_view.h"
#include <algorithm>

namespace vgre {
namespace sparse {

namespace {

// Build an owned matrix from freshly-computed arrays, wiring the void* fields to
// point into the owned buffers.
template <typename T, typename IndexType>
SparseMatrixView<T, IndexType>
makeOwned(std::vector<IndexType> row, std::vector<IndexType> col,
          std::vector<T> val, IndexType rows, IndexType cols, IndexType nnz,
          IndexType blockSize, SparseFormat fmt) {
    SparseMatrixView<T, IndexType> out;
    out.ownRow = std::make_shared<std::vector<IndexType>>(std::move(row));
    out.ownCol = std::make_shared<std::vector<IndexType>>(std::move(col));
    out.ownVal = std::make_shared<std::vector<T>>(std::move(val));
    out.rowOffsets = out.ownRow->data();
    out.colIndices = out.ownCol->data();
    out.values     = out.ownVal->data();
    out.rows = rows;
    out.cols = cols;
    out.nnz = nnz;
    out.blockSize = blockSize;
    out.format = fmt;
    out.isView = false;
    return out;
}

// COO (rowIdx[nnz], colIdx[nnz], val[nnz]) → CSR. Stable counting sort by row so
// within-row column order is preserved; O(nnz + rows).
template <typename T, typename IndexType>
SparseMatrixView<T, IndexType> cooToCsr(const SparseMatrixView<T, IndexType>& m) {
    const auto* rowIdx = static_cast<const IndexType*>(m.rowOffsets);
    const auto* colIdx = static_cast<const IndexType*>(m.colIndices);
    const auto* val    = static_cast<const T*>(m.values);
    const IndexType n  = m.nnz;
    const IndexType R  = m.rows;
    if (!rowIdx || !colIdx || !val || R < 0 || n < 0)
        return SparseMatrixView<T, IndexType>();

    std::vector<IndexType> rowPtr(static_cast<size_t>(R) + 1, 0);
    for (IndexType k = 0; k < n; ++k) {
        IndexType r = rowIdx[k];
        if (r < 0 || r >= R) return SparseMatrixView<T, IndexType>();
        rowPtr[static_cast<size_t>(r) + 1]++;
    }
    for (IndexType i = 0; i < R; ++i)
        rowPtr[static_cast<size_t>(i) + 1] += rowPtr[static_cast<size_t>(i)];

    std::vector<IndexType> outCol(static_cast<size_t>(n));
    std::vector<T>         outVal(static_cast<size_t>(n));
    std::vector<IndexType> cursor(rowPtr.begin(), rowPtr.end() - 1);  // per-row write head
    for (IndexType k = 0; k < n; ++k) {
        IndexType r   = rowIdx[k];
        IndexType dst = cursor[static_cast<size_t>(r)]++;
        outCol[static_cast<size_t>(dst)] = colIdx[k];
        outVal[static_cast<size_t>(dst)] = val[k];
    }
    return makeOwned<T, IndexType>(std::move(rowPtr), std::move(outCol),
                                   std::move(outVal), R, m.cols, n, 1,
                                   SparseFormat::CSR);
}

// Transpose a compressed matrix: CSR(A)→CSC(A) and CSC(A)→CSR(A) are the same
// algorithm (count per secondary index, prefix-sum, scatter). Produces the SAME
// matrix A in the target format — NOT A^T.
template <typename T, typename IndexType>
SparseMatrixView<T, IndexType>
transposeCompressed(const SparseMatrixView<T, IndexType>& m, SparseFormat target) {
    const auto* ptr = static_cast<const IndexType*>(m.rowOffsets); // primary pointers
    const auto* idx = static_cast<const IndexType*>(m.colIndices); // secondary indices
    const auto* val = static_cast<const T*>(m.values);
    const IndexType n = m.nnz;
    if (!ptr || !idx || !val) return SparseMatrixView<T, IndexType>();

    // primaryDim = number of compressed lines in the source;
    // secondaryDim = range of the secondary index → number of lines in the result.
    const IndexType primaryDim   = (m.format == SparseFormat::CSR) ? m.rows : m.cols;
    const IndexType secondaryDim = (m.format == SparseFormat::CSR) ? m.cols : m.rows;
    if (primaryDim < 0 || secondaryDim < 0) return SparseMatrixView<T, IndexType>();

    std::vector<IndexType> outPtr(static_cast<size_t>(secondaryDim) + 1, 0);
    for (IndexType k = 0; k < n; ++k) {
        IndexType s = idx[k];
        if (s < 0 || s >= secondaryDim) return SparseMatrixView<T, IndexType>();
        outPtr[static_cast<size_t>(s) + 1]++;
    }
    for (IndexType i = 0; i < secondaryDim; ++i)
        outPtr[static_cast<size_t>(i) + 1] += outPtr[static_cast<size_t>(i)];

    std::vector<IndexType> outIdx(static_cast<size_t>(n));
    std::vector<T>         outVal(static_cast<size_t>(n));
    std::vector<IndexType> cursor(outPtr.begin(), outPtr.end() - 1);
    for (IndexType p = 0; p < primaryDim; ++p) {
        for (IndexType k = ptr[p]; k < ptr[p + 1]; ++k) {
            IndexType s   = idx[k];
            IndexType dst = cursor[static_cast<size_t>(s)]++;
            outIdx[static_cast<size_t>(dst)] = p;       // the primary line becomes secondary
            outVal[static_cast<size_t>(dst)] = val[k];
        }
    }
    return makeOwned<T, IndexType>(std::move(outPtr), std::move(outIdx),
                                   std::move(outVal), m.rows, m.cols, n, 1, target);
}

// CSR(A) → COO(A): expand the row pointers into explicit row indices.
template <typename T, typename IndexType>
SparseMatrixView<T, IndexType> csrToCoo(const SparseMatrixView<T, IndexType>& m) {
    const auto* ptr = static_cast<const IndexType*>(m.rowOffsets);
    const auto* col = static_cast<const IndexType*>(m.colIndices);
    const auto* val = static_cast<const T*>(m.values);
    if (!ptr || !col || !val) return SparseMatrixView<T, IndexType>();

    std::vector<IndexType> rowIdx(static_cast<size_t>(m.nnz));
    std::vector<IndexType> colIdx(col, col + m.nnz);
    std::vector<T>         outVal(val, val + m.nnz);
    for (IndexType r = 0; r < m.rows; ++r)
        for (IndexType k = ptr[r]; k < ptr[r + 1]; ++k)
            rowIdx[static_cast<size_t>(k)] = r;
    return makeOwned<T, IndexType>(std::move(rowIdx), std::move(colIdx),
                                   std::move(outVal), m.rows, m.cols, m.nnz, 1,
                                   SparseFormat::COO);
}

} // namespace

template<typename T, typename IndexType>
bool SparseMatrixView<T, IndexType>::canConvertZeroCopy(SparseFormat targetFormat) const {
    if (format == targetFormat) return true;
    // CSR↔BSR with blockSize 1 share the exact CSR layout. Nothing else is
    // zero-copy: CSR↔CSC needs a transpose, COO↔CSR needs offset construction.
    if (format == SparseFormat::CSR && targetFormat == SparseFormat::BSR)
        return blockSize == 1;
    if (format == SparseFormat::BSR && targetFormat == SparseFormat::CSR)
        return blockSize == 1;
    return false;
}

template<typename T, typename IndexType>
SparseMatrixView<T, IndexType>
SparseMatrixView<T, IndexType>::toView(SparseFormat targetFormat) const {
    if (format == targetFormat) return *this;

    // Zero-copy: CSR↔BSR(blockSize 1) — identical layout, just relabel.
    if (((format == SparseFormat::CSR && targetFormat == SparseFormat::BSR) ||
         (format == SparseFormat::BSR && targetFormat == SparseFormat::CSR)) &&
        blockSize == 1) {
        SparseMatrixView<T, IndexType> v(*this);
        v.format = targetFormat;
        v.blockSize = 1;
        v.isView = true;
        return v;
    }

    // Real (owned) conversions — correct, allocate as needed.
    if (format == SparseFormat::COO && targetFormat == SparseFormat::CSR)
        return cooToCsr(*this);
    if (format == SparseFormat::CSR && targetFormat == SparseFormat::CSC)
        return transposeCompressed(*this, SparseFormat::CSC);
    if (format == SparseFormat::CSC && targetFormat == SparseFormat::CSR)
        return transposeCompressed(*this, SparseFormat::CSR);
    if (format == SparseFormat::CSR && targetFormat == SparseFormat::COO)
        return csrToCoo(*this);
    if (format == SparseFormat::COO && targetFormat == SparseFormat::CSC) {
        auto csr = cooToCsr(*this);
        return csr.toView(SparseFormat::CSC);
    }

    // Unsupported pair.
    return SparseMatrixView<T, IndexType>();
}

template<typename T, typename IndexType>
SparseMatrixView<T, IndexType> csrToCscView(const SparseMatrixView<T, IndexType>& csr) {
    return csr.toView(SparseFormat::CSC);
}

template<typename T, typename IndexType>
SparseMatrixView<T, IndexType> csrToBsrView(const SparseMatrixView<T, IndexType>& csr, IndexType blockSize) {
    if (blockSize == 1) return csr.toView(SparseFormat::BSR);
    // Block size > 1 changes the array shape; fall back to the owned path is not
    // yet defined for arbitrary blocking, so signal "no conversion" explicitly.
    return SparseMatrixView<T, IndexType>();
}

template<typename T, typename IndexType>
SparseMatrixView<T, IndexType> cooToCsrView(const SparseMatrixView<T, IndexType>& coo) {
    return coo.toView(SparseFormat::CSR);
}

// Explicit template instantiations for common types
template struct SparseMatrixView<float, int32_t>;
template struct SparseMatrixView<double, int32_t>;
template struct SparseMatrixView<float, int64_t>;
template struct SparseMatrixView<double, int64_t>;

template SparseMatrixView<float, int32_t>  csrToCscView(const SparseMatrixView<float, int32_t>&);
template SparseMatrixView<double, int32_t> csrToCscView(const SparseMatrixView<double, int32_t>&);
template SparseMatrixView<float, int64_t>  csrToCscView(const SparseMatrixView<float, int64_t>&);
template SparseMatrixView<double, int64_t> csrToCscView(const SparseMatrixView<double, int64_t>&);

template SparseMatrixView<float, int32_t>  cooToCsrView(const SparseMatrixView<float, int32_t>&);
template SparseMatrixView<double, int32_t> cooToCsrView(const SparseMatrixView<double, int32_t>&);

template SparseMatrixView<float, int32_t>  csrToBsrView(const SparseMatrixView<float, int32_t>&, int32_t);

} // namespace sparse
} // namespace vgre
