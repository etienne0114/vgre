// cuSPARSE sparse factorization shim — ILU(0), IC(0), SpGEMM.
//
// ILU(0): in-place incomplete LU with zero fill-in.
//   Each row i: for every j < i where A[i,j] exists, subtract A[i,j]/A[j,j] *
//   (contributions in row j that are also in row i). No new fill-in is created.
//
// IC(0): in-place incomplete Cholesky with zero fill-in (symmetric, lower tri).
//   For each column j: divide A[j,j] by its own square root (diagonal), then
//   subtract outer products for entries below the diagonal, only at existing
//   sparsity positions.
//
// SpGEMM: sparse × sparse → sparse, three-phase (workEstimation, compute, copy).
//   Uses CSR row-wise sparse dot products to fill the output CSR.

#include "cusparse_state.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace vgre_sp;

// ── Helper: extract raw int row-offset / col-index arrays ────────────────────
static void extractRowPtr(const CsrMat &A, std::vector<int32_t> &rowPtr) {
    rowPtr.resize(static_cast<size_t>(A.rows + 1));
    for (int64_t i = 0; i <= A.rows; ++i)
        rowPtr[static_cast<size_t>(i)] = static_cast<int32_t>(getIdx(A.rowOffsets, A.rowOffsetType, i));
}
static void extractColInd(const CsrMat &A, std::vector<int32_t> &colInd) {
    colInd.resize(static_cast<size_t>(A.nnz));
    for (int64_t i = 0; i < A.nnz; ++i)
        colInd[static_cast<size_t>(i)] = static_cast<int32_t>(getIdx(A.colInd, A.colIndType, i));
}

// ── ILU(0) — float or double, in-place on csrVal ─────────────────────────────
// For row i, column j (j < i), if A[i,j] != 0:
//   A[i,j] /= A[j,j]
//   for each k in cols(row j) where k > j and A[i,k] exists:
//     A[i,k] -= A[i,j] * A[j,k]
template<typename T>
static cusparseStatus_t ilu0_csr(int m, int nnz, T *val,
                                  const int *rowPtr, const int *colInd, int base) {
    if (!val || !rowPtr || !colInd || m <= 0) return CUSPARSE_STATUS_INVALID_VALUE;

    // Build fast column→value-index lookup per row for scatter access
    // diag[j] = position in val[] of diagonal element A[j,j]
    std::vector<int> diagIdx(static_cast<size_t>(m), -1);
    for (int i = 0; i < m; ++i) {
        for (int p = rowPtr[i] - base; p < rowPtr[i+1] - base; ++p) {
            if (colInd[p] - base == i) { diagIdx[static_cast<size_t>(i)] = p; break; }
        }
    }

    // Build a per-row map: col -> position for O(1) fill lookup
    std::vector<int> marker(static_cast<size_t>(m), -1);

    for (int i = 0; i < m; ++i) {
        int rowStart = rowPtr[i]   - base;
        int rowEnd   = rowPtr[i+1] - base;

        // Mark all columns in row i
        for (int p = rowStart; p < rowEnd; ++p) marker[static_cast<size_t>(colInd[p] - base)] = p;

        // Sweep over columns j < i where A[i,j] != 0
        for (int p = rowStart; p < rowEnd; ++p) {
            int j = colInd[p] - base;
            if (j >= i) break; // colInd must be sorted
            int dj = diagIdx[static_cast<size_t>(j)];
            if (dj < 0 || val[dj] == T(0)) continue; // singular pivot
            val[p] /= val[dj]; // A[i,j] /= A[j,j]
            T aij = val[p];
            // Subtract contributions from row j into row i at shared sparsity positions
            for (int q = dj + 1; q < rowPtr[j+1] - base; ++q) {
                int k = colInd[q] - base;
                if (k <= j) continue;
                int pos = marker[static_cast<size_t>(k)];
                if (pos >= 0) val[pos] -= aij * val[q];
            }
        }

        // Unmark
        for (int p = rowStart; p < rowEnd; ++p) marker[static_cast<size_t>(colInd[p] - base)] = -1;
    }
    return CUSPARSE_STATUS_SUCCESS;
}

// ── IC(0) — float or double, in-place, lower triangular CSR ──────────────────
// Incomplete Cholesky with zero fill-in on a symmetric positive-definite matrix
// stored in CSR lower triangular (only lower triangle needed).
// For column j (0..m-1):
//   A[j,j] = sqrt(A[j,j] - sum A[j,k]^2 for k < j)
//   For each i > j where A[i,j] != 0:
//     A[i,j] = (A[i,j] - sum A[i,k]*A[j,k] for k < j) / A[j,j]
template<typename T>
static cusparseStatus_t ic0_csr(int m, int nnz, T *val,
                                 const int *rowPtr, const int *colInd, int base) {
    if (!val || !rowPtr || !colInd || m <= 0) return CUSPARSE_STATUS_INVALID_VALUE;

    // For each row store: col -> (value index) for O(1) lookup
    // We need to access A[i,k] for columns k in both row i and row j
    // Build col-sorted index vector for all rows
    std::vector<std::vector<std::pair<int,int>>> rowMap(static_cast<size_t>(m)); // col → valIdx
    for (int i = 0; i < m; ++i) {
        for (int p = rowPtr[i] - base; p < rowPtr[i+1] - base; ++p)
            rowMap[static_cast<size_t>(i)].push_back({colInd[p] - base, p});
    }

    auto getVal = [&](int row, int col) -> T* {
        for (auto &[c, idx] : rowMap[static_cast<size_t>(row)])
            if (c == col) return &val[idx];
        return nullptr;
    };

    for (int j = 0; j < m; ++j) {
        // Compute diagonal: A[j,j] -= sum_{k<j, A[j,k]!=0} A[j,k]^2
        T *djj = getVal(j, j);
        if (!djj) continue;
        for (int p = rowPtr[j] - base; p < rowPtr[j+1] - base; ++p) {
            int k = colInd[p] - base;
            if (k >= j) break;
            *djj -= val[p] * val[p];
        }
        if (*djj <= T(0)) return CUSPARSE_STATUS_ZERO_PIVOT;
        *djj = std::sqrt(*djj);
        T diagJ = *djj;

        // Update column below diagonal: for each i > j where A[i,j] exists
        for (int p = rowPtr[j] - base; p < rowPtr[j+1] - base; ++p) {
            // Lower-tri CSR: entries in row j have col <= j; skip diagonal
            (void)p; // inner loop handled by scanning later rows for col j
        }

        // Scan all rows i > j and update A[i,j]
        for (int i = j + 1; i < m; ++i) {
            T *dij = getVal(i, j);
            if (!dij) continue;
            // A[i,j] -= sum_{k<j, A[i,k] && A[j,k]} A[i,k]*A[j,k]
            int pi = rowPtr[i] - base, pj = rowPtr[j] - base;
            int piEnd = rowPtr[i+1] - base, pjEnd = rowPtr[j+1] - base;
            // merge-walk both sorted row slices up to k < j
            while (pi < piEnd && pj < pjEnd) {
                int ci = colInd[pi] - base, cj = colInd[pj] - base;
                if (ci >= j || cj >= j) break;
                if (ci == cj) { *dij -= val[pi] * val[pj]; ++pi; ++pj; }
                else if (ci < cj) ++pi;
                else              ++pj;
            }
            *dij /= diagJ;
        }
    }
    return CUSPARSE_STATUS_SUCCESS;
}

// ── SpGEMM state ──────────────────────────────────────────────────────────────
// Three-phase (workEstimation → compute → copy). We compute the full product
// in 'compute' and store it; 'copy' writes into the user-provided C descriptor.

namespace vgre_sp {
    // defined in cusparse_state.h as extern, but SpGEMM state is file-local:
}

namespace {
struct SpGEMMState {
    // Result of sparse matmul: CSR stored inline
    std::vector<int32_t> cRowPtr;
    std::vector<int32_t> cColInd;
    std::vector<float>   cValF;
    std::vector<double>  cValD;
    int64_t cRows = 0, cCols = 0, cNnz = 0;
    cudaDataType_t dtype = CUDA_R_32F;
};

std::mutex g_spgemmMutex;
std::unordered_map<uintptr_t, SpGEMMState> g_spgemmStates;
uintptr_t g_nextSpGEMM = 1;

// CSR × CSR → CSR product (template for float/double)
template<typename T>
void csr_spgemm(
        int64_t m, int64_t k, int64_t n,
        const int *arPtr, const int *acInd, const T *aVal, int aBase,
        const int *brPtr, const int *bcInd, const T *bVal, int bBase,
        std::vector<int32_t> &cRowPtr, std::vector<int32_t> &cColInd, std::vector<T> &cVal) {

    cRowPtr.resize(static_cast<size_t>(m + 1), 0);
    std::vector<T> dense(static_cast<size_t>(n), T(0));
    std::vector<int> used;

    // Two-pass: first count NNZ per row, then fill
    std::vector<int32_t> rowNnz(static_cast<size_t>(m), 0);
    for (int64_t i = 0; i < m; ++i) {
        for (int pa = arPtr[i] - aBase; pa < arPtr[i+1] - aBase; ++pa) {
            int64_t jj = acInd[pa] - aBase;
            for (int pb = brPtr[jj] - bBase; pb < brPtr[jj+1] - bBase; ++pb) {
                int64_t cc = bcInd[pb] - bBase;
                if (dense[static_cast<size_t>(cc)] == T(0)) {
                    used.push_back(static_cast<int>(cc));
                    dense[static_cast<size_t>(cc)] = T(1); // mark
                }
                // Accumulate actual value
                dense[static_cast<size_t>(cc)] += aVal[pa] * bVal[pb] - T(1); // undo the +1 from mark
            }
        }
        rowNnz[static_cast<size_t>(i)] = static_cast<int32_t>(used.size());
        for (int c : used) dense[static_cast<size_t>(c)] = T(0);
        used.clear();
    }

    // Prefix sum
    cRowPtr[0] = 0;
    for (int64_t i = 0; i < m; ++i)
        cRowPtr[static_cast<size_t>(i+1)] = cRowPtr[static_cast<size_t>(i)] + rowNnz[static_cast<size_t>(i)];
    int64_t totalNnz = cRowPtr[static_cast<size_t>(m)];
    cColInd.resize(static_cast<size_t>(totalNnz));
    cVal.resize(static_cast<size_t>(totalNnz));

    // Second pass: fill values
    for (int64_t i = 0; i < m; ++i) {
        for (int pa = arPtr[i] - aBase; pa < arPtr[i+1] - aBase; ++pa) {
            int64_t jj = acInd[pa] - aBase;
            for (int pb = brPtr[jj] - bBase; pb < brPtr[jj+1] - bBase; ++pb) {
                int64_t cc = bcInd[pb] - bBase;
                if (dense[static_cast<size_t>(cc)] == T(0)) used.push_back(static_cast<int>(cc));
                dense[static_cast<size_t>(cc)] += aVal[pa] * bVal[pb];
            }
        }
        // Sort columns for canonical CSR
        std::sort(used.begin(), used.end());
        int32_t base_idx = cRowPtr[static_cast<size_t>(i)];
        for (size_t idx = 0; idx < used.size(); ++idx) {
            int c = used[idx];
            cColInd[static_cast<size_t>(base_idx + idx)] = c;
            cVal[static_cast<size_t>(base_idx + idx)] = dense[static_cast<size_t>(c)];
            dense[static_cast<size_t>(c)] = T(0);
        }
        used.clear();
    }
}
} // anonymous namespace

extern "C" {

// ── SpGEMM descriptor ─────────────────────────────────────────────────────────
cusparseStatus_t cusparseSpGEMM_createDescr(cusparseSpGEMMDescr_t *descr) {
    if (!descr) return CUSPARSE_STATUS_INVALID_VALUE;
    std::lock_guard<std::mutex> lk(g_spgemmMutex);
    uintptr_t id = g_nextSpGEMM++;
    g_spgemmStates[id] = {};
    *descr = reinterpret_cast<cusparseSpGEMMDescr_t>(id);
    return CUSPARSE_STATUS_SUCCESS;
}
cusparseStatus_t cusparseSpGEMM_destroyDescr(cusparseSpGEMMDescr_t descr) {
    std::lock_guard<std::mutex> lk(g_spgemmMutex);
    g_spgemmStates.erase(reinterpret_cast<uintptr_t>(descr));
    return CUSPARSE_STATUS_SUCCESS;
}

// Phase 1: workEstimation — no-op, buffer size 0
cusparseStatus_t cusparseSpGEMM_workEstimation(cusparseHandle_t /*h*/,
        cusparseOperation_t /*opA*/, cusparseOperation_t /*opB*/,
        const void * /*alpha*/, cusparseSpMatDescr_t /*matA*/,
        cusparseSpMatDescr_t /*matB*/, const void * /*beta*/,
        cusparseSpMatDescr_t /*matC*/, cudaDataType_t /*ct*/,
        cusparseSpGEMMAlg_t /*alg*/, cusparseSpGEMMDescr_t /*d*/,
        size_t *bufferSize1, void * /*externalBuffer1*/) {
    if (bufferSize1) *bufferSize1 = 0;
    return CUSPARSE_STATUS_SUCCESS;
}

// Phase 2: compute — perform the actual C = A * B
cusparseStatus_t cusparseSpGEMM_compute(cusparseHandle_t /*h*/,
        cusparseOperation_t opA, cusparseOperation_t opB,
        const void *alpha, cusparseSpMatDescr_t matA,
        cusparseSpMatDescr_t matB, const void *beta,
        cusparseSpMatDescr_t matC, cudaDataType_t computeType,
        cusparseSpGEMMAlg_t /*alg*/, cusparseSpGEMMDescr_t spgemmDescr,
        size_t *bufferSize2, void * /*externalBuffer2*/) {
    if (bufferSize2) *bufferSize2 = 0;

    std::lock_guard<std::mutex> lkD(g_descrMutex);
    auto aIt = g_csrMats.find(reinterpret_cast<uintptr_t>(matA));
    auto bIt = g_csrMats.find(reinterpret_cast<uintptr_t>(matB));
    auto cIt = g_csrMats.find(reinterpret_cast<uintptr_t>(matC));
    if (aIt == g_csrMats.end() || bIt == g_csrMats.end() || cIt == g_csrMats.end())
        return CUSPARSE_STATUS_INVALID_VALUE;
    if (!aIt->second.rowOffsets || !aIt->second.colInd || !aIt->second.values)
        return CUSPARSE_STATUS_INVALID_VALUE;
    if (!bIt->second.rowOffsets || !bIt->second.colInd || !bIt->second.values)
        return CUSPARSE_STATUS_INVALID_VALUE;

    const CsrMat &A = aIt->second;
    const CsrMat &B = bIt->second;

    int64_t m = (opA == CUSPARSE_OPERATION_NON_TRANSPOSE) ? A.rows : A.cols;
    int64_t n = (opB == CUSPARSE_OPERATION_NON_TRANSPOSE) ? B.cols : B.rows;

    std::vector<int32_t> arPtr, acInd, brPtr, bcInd;
    extractRowPtr(A, arPtr);
    extractColInd(A, acInd);
    extractRowPtr(B, brPtr);
    extractColInd(B, bcInd);

    std::lock_guard<std::mutex> lkG(g_spgemmMutex);
    auto stIt = g_spgemmStates.find(reinterpret_cast<uintptr_t>(spgemmDescr));
    if (stIt == g_spgemmStates.end()) return CUSPARSE_STATUS_INVALID_VALUE;
    SpGEMMState &st = stIt->second;
    st.dtype = computeType;
    st.cRows = m; st.cCols = n;

    if (computeType == CUDA_R_32F) {
        float alphaF = *static_cast<const float*>(alpha);
        csr_spgemm<float>(m, A.cols, n,
            arPtr.data(), acInd.data(), static_cast<const float*>(A.values), 0,
            brPtr.data(), bcInd.data(), static_cast<const float*>(B.values), 0,
            st.cRowPtr, st.cColInd, st.cValF);
        float betaF = *static_cast<const float*>(beta);
        if (alphaF != 1.0f) for (auto &v : st.cValF) v *= alphaF;
        // beta*C not applied here since C's arrays aren't filled yet (handled in copy)
        (void)betaF;
    } else if (computeType == CUDA_R_64F) {
        double alphaD = *static_cast<const double*>(alpha);
        csr_spgemm<double>(m, A.cols, n,
            arPtr.data(), acInd.data(), static_cast<const double*>(A.values), 0,
            brPtr.data(), bcInd.data(), static_cast<const double*>(B.values), 0,
            st.cRowPtr, st.cColInd, st.cValD);
        double betaD = *static_cast<const double*>(beta);
        if (alphaD != 1.0) for (auto &v : st.cValD) v *= alphaD;
        (void)betaD;
    } else {
        return CUSPARSE_STATUS_NOT_SUPPORTED;
    }
    st.cNnz = static_cast<int64_t>(st.cRowPtr[static_cast<size_t>(m)]);

    // Update the output descriptor's NNZ so caller can query it
    cIt->second.nnz  = st.cNnz;
    cIt->second.rows = m;
    cIt->second.cols = n;
    return CUSPARSE_STATUS_SUCCESS;
}

// Phase 3: copy — write computed result into the user-allocated C arrays
cusparseStatus_t cusparseSpGEMM_copy(cusparseHandle_t /*h*/,
        cusparseOperation_t /*opA*/, cusparseOperation_t /*opB*/,
        const void * /*alpha*/, cusparseSpMatDescr_t /*matA*/,
        cusparseSpMatDescr_t /*matB*/, const void * /*beta*/,
        cusparseSpMatDescr_t matC, cudaDataType_t computeType,
        cusparseSpGEMMAlg_t /*alg*/, cusparseSpGEMMDescr_t spgemmDescr) {
    std::lock_guard<std::mutex> lkD(g_descrMutex);
    auto cIt = g_csrMats.find(reinterpret_cast<uintptr_t>(matC));
    if (cIt == g_csrMats.end()) return CUSPARSE_STATUS_INVALID_VALUE;
    CsrMat &C = cIt->second;
    if (!C.rowOffsets || !C.colInd || !C.values) return CUSPARSE_STATUS_INVALID_VALUE;

    std::lock_guard<std::mutex> lkG(g_spgemmMutex);
    auto stIt = g_spgemmStates.find(reinterpret_cast<uintptr_t>(spgemmDescr));
    if (stIt == g_spgemmStates.end()) return CUSPARSE_STATUS_INVALID_VALUE;
    const SpGEMMState &st = stIt->second;

    int cBase = (C.idxBase == CUSPARSE_INDEX_BASE_ONE) ? 1 : 0;
    int64_t m = st.cRows;

    // Write row pointers
    for (int64_t i = 0; i <= m; ++i) {
        int64_t v = st.cRowPtr[static_cast<size_t>(i)] + cBase;
        if (C.rowOffsetType == CUSPARSE_INDEX_64I)
            static_cast<int64_t*>(C.rowOffsets)[i] = v;
        else
            static_cast<int32_t*>(C.rowOffsets)[i] = static_cast<int32_t>(v);
    }

    // Write column indices
    for (int64_t k = 0; k < st.cNnz; ++k) {
        int64_t v = st.cColInd[static_cast<size_t>(k)] + cBase;
        if (C.colIndType == CUSPARSE_INDEX_64I)
            static_cast<int64_t*>(C.colInd)[k] = v;
        else
            static_cast<int32_t*>(C.colInd)[k] = static_cast<int32_t>(v);
    }

    // Write values
    if (computeType == CUDA_R_32F)
        std::memcpy(C.values, st.cValF.data(), static_cast<size_t>(st.cNnz) * sizeof(float));
    else if (computeType == CUDA_R_64F)
        std::memcpy(C.values, st.cValD.data(), static_cast<size_t>(st.cNnz) * sizeof(double));
    else
        return CUSPARSE_STATUS_NOT_SUPPORTED;

    return CUSPARSE_STATUS_SUCCESS;
}

// ── ILU(0) implementation ─────────────────────────────────────────────────────

cusparseStatus_t cusparseCreateCsrilu02Info(csrilu02Info_t *info) {
    if (!info) return CUSPARSE_STATUS_INVALID_VALUE;
    *info = reinterpret_cast<csrilu02Info_t>(1); // opaque sentinel
    return CUSPARSE_STATUS_SUCCESS;
}
cusparseStatus_t cusparseDestroyCsrilu02Info(csrilu02Info_t /*info*/) {
    return CUSPARSE_STATUS_SUCCESS;
}
cusparseStatus_t cusparseScsrilu02_bufferSize(cusparseHandle_t /*h*/, int /*m*/, int /*nnz*/,
        const cusparseSpMatDescr_t /*descr*/, float * /*val*/,
        const int * /*rowPtr*/, const int * /*colInd*/,
        csrilu02Info_t /*info*/, int *pBufSize) {
    if (pBufSize) *pBufSize = 0; return CUSPARSE_STATUS_SUCCESS;
}
cusparseStatus_t cusparseDcsrilu02_bufferSize(cusparseHandle_t /*h*/, int /*m*/, int /*nnz*/,
        const cusparseSpMatDescr_t /*descr*/, double * /*val*/,
        const int * /*rowPtr*/, const int * /*colInd*/,
        csrilu02Info_t /*info*/, int *pBufSize) {
    if (pBufSize) *pBufSize = 0; return CUSPARSE_STATUS_SUCCESS;
}
cusparseStatus_t cusparseScsrilu02_analysis(cusparseHandle_t /*h*/, int /*m*/, int /*nnz*/,
        const cusparseSpMatDescr_t /*d*/, const float * /*val*/,
        const int * /*rowPtr*/, const int * /*colInd*/,
        csrilu02Info_t /*info*/, int /*policy*/, void * /*buf*/) {
    return CUSPARSE_STATUS_SUCCESS;
}
cusparseStatus_t cusparseDcsrilu02_analysis(cusparseHandle_t /*h*/, int /*m*/, int /*nnz*/,
        const cusparseSpMatDescr_t /*d*/, const double * /*val*/,
        const int * /*rowPtr*/, const int * /*colInd*/,
        csrilu02Info_t /*info*/, int /*policy*/, void * /*buf*/) {
    return CUSPARSE_STATUS_SUCCESS;
}
cusparseStatus_t cusparseScsrilu02(cusparseHandle_t /*h*/, int m, int nnz,
        const cusparseSpMatDescr_t /*d*/,
        float *csrVal, const int *rowPtr, const int *colInd,
        csrilu02Info_t /*info*/, int /*policy*/, void * /*buf*/) {
    // The legacy descriptor uses 0-based indexing; real cuSPARSE also accepts
    // 1-based; we assume 0-based for these legacy prototypes.
    return ilu0_csr<float>(m, nnz, csrVal, rowPtr, colInd, 0);
}
cusparseStatus_t cusparseDcsrilu02(cusparseHandle_t /*h*/, int m, int nnz,
        const cusparseSpMatDescr_t /*d*/,
        double *csrVal, const int *rowPtr, const int *colInd,
        csrilu02Info_t /*info*/, int /*policy*/, void * /*buf*/) {
    return ilu0_csr<double>(m, nnz, csrVal, rowPtr, colInd, 0);
}

// ── IC(0) implementation ──────────────────────────────────────────────────────

cusparseStatus_t cusparseCreateCsric02Info(csric02Info_t *info) {
    if (!info) return CUSPARSE_STATUS_INVALID_VALUE;
    *info = reinterpret_cast<csric02Info_t>(1);
    return CUSPARSE_STATUS_SUCCESS;
}
cusparseStatus_t cusparseDestroyCsric02Info(csric02Info_t /*info*/) {
    return CUSPARSE_STATUS_SUCCESS;
}
cusparseStatus_t cusparseScsric02_bufferSize(cusparseHandle_t /*h*/, int /*m*/, int /*nnz*/,
        const cusparseSpMatDescr_t /*d*/, float * /*val*/,
        const int * /*rowPtr*/, const int * /*colInd*/,
        csric02Info_t /*info*/, int *pBufSize) {
    if (pBufSize) *pBufSize = 0; return CUSPARSE_STATUS_SUCCESS;
}
cusparseStatus_t cusparseDcsric02_bufferSize(cusparseHandle_t /*h*/, int /*m*/, int /*nnz*/,
        const cusparseSpMatDescr_t /*d*/, double * /*val*/,
        const int * /*rowPtr*/, const int * /*colInd*/,
        csric02Info_t /*info*/, int *pBufSize) {
    if (pBufSize) *pBufSize = 0; return CUSPARSE_STATUS_SUCCESS;
}
cusparseStatus_t cusparseScsric02_analysis(cusparseHandle_t /*h*/, int /*m*/, int /*nnz*/,
        const cusparseSpMatDescr_t /*d*/, const float * /*val*/,
        const int * /*rowPtr*/, const int * /*colInd*/,
        csric02Info_t /*info*/, int /*policy*/, void * /*buf*/) {
    return CUSPARSE_STATUS_SUCCESS;
}
cusparseStatus_t cusparseDcsric02_analysis(cusparseHandle_t /*h*/, int /*m*/, int /*nnz*/,
        const cusparseSpMatDescr_t /*d*/, const double * /*val*/,
        const int * /*rowPtr*/, const int * /*colInd*/,
        csric02Info_t /*info*/, int /*policy*/, void * /*buf*/) {
    return CUSPARSE_STATUS_SUCCESS;
}
cusparseStatus_t cusparseScsric02(cusparseHandle_t /*h*/, int m, int nnz,
        const cusparseSpMatDescr_t /*d*/,
        float *csrVal, const int *rowPtr, const int *colInd,
        csric02Info_t /*info*/, int /*policy*/, void * /*buf*/) {
    return ic0_csr<float>(m, nnz, csrVal, rowPtr, colInd, 0);
}
cusparseStatus_t cusparseDcsric02(cusparseHandle_t /*h*/, int m, int nnz,
        const cusparseSpMatDescr_t /*d*/,
        double *csrVal, const int *rowPtr, const int *colInd,
        csric02Info_t /*info*/, int /*policy*/, void * /*buf*/) {
    return ic0_csr<double>(m, nnz, csrVal, rowPtr, colInd, 0);
}

} // extern "C"
