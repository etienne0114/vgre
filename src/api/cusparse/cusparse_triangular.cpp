// cuSPARSE sparse triangular solve (SpSV) — forward and backward substitution
// for lower- and upper-triangular CSR matrices, with conjugate/transpose support.
//
// Algorithm: row-by-row gather for direct solves; scatter-based for transposed.

#include "cusparse_state.h"

using namespace vgre_sp;

extern "C" {

cusparseStatus_t cusparseSpSV_createDescr(cusparseSpSVDescr_t *descr) {
    if (!descr) return CUSPARSE_STATUS_INVALID_VALUE;
    std::lock_guard<std::mutex> lk(g_descrMutex);
    uintptr_t id = g_nextSpSV++;
    g_spsvDescrs[id] = {};
    *descr = reinterpret_cast<cusparseSpSVDescr_t>(id);
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSpSV_destroyDescr(cusparseSpSVDescr_t descr) {
    std::lock_guard<std::mutex> lk(g_descrMutex);
    g_spsvDescrs.erase(reinterpret_cast<uintptr_t>(descr));
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSpSV_bufferSize(cusparseHandle_t /*h*/,
                                          cusparseOperation_t /*opA*/, const void * /*alpha*/,
                                          cusparseSpMatDescr_t /*matA*/,
                                          cusparseDnVecDescr_t /*vecX*/, cusparseDnVecDescr_t /*vecY*/,
                                          cudaDataType_t /*ct*/, cusparseSpSVAlg_t /*alg*/,
                                          cusparseSpSVDescr_t /*d*/, size_t *bufferSize) {
    if (bufferSize) *bufferSize = 0;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSpSV_analysis(cusparseHandle_t /*h*/,
                                        cusparseOperation_t /*opA*/, const void * /*alpha*/,
                                        cusparseSpMatDescr_t /*matA*/,
                                        cusparseDnVecDescr_t /*vecX*/, cusparseDnVecDescr_t /*vecY*/,
                                        cudaDataType_t /*ct*/, cusparseSpSVAlg_t /*alg*/,
                                        cusparseSpSVDescr_t descr, void * /*buffer*/) {
    std::lock_guard<std::mutex> lk(g_descrMutex);
    auto it = g_spsvDescrs.find(reinterpret_cast<uintptr_t>(descr));
    if (it != g_spsvDescrs.end()) it->second.analyzed = true;
    return CUSPARSE_STATUS_SUCCESS;
}

// Solves op(A) * y = alpha * x where A is a triangular CSR matrix.
//
// Direct lower triangular (op = N, fill = LOWER):
//   Forward substitution row-by-row: gather already-solved y[j < i] from row.
//
// Direct upper triangular (op = N, fill = UPPER):
//   Backward substitution row-by-row: gather already-solved y[j > i] from row.
//
// Transposed lower triangular (op = T, fill = LOWER) → upper solve:
//   Scatter-based backward pass: for each row i processed in reverse, solve y[i],
//   then subtract A[i,j]*y[i] from y[j] for all j < i in that row.
//
// Transposed upper triangular (op = T, fill = UPPER) → lower solve:
//   Scatter-based forward pass: for each row i processed in order, solve y[i],
//   then subtract A[i,j]*y[i] from y[j] for all j > i in that row.

cusparseStatus_t cusparseSpSV_solve(cusparseHandle_t /*h*/,
                                     cusparseOperation_t opA, const void *alpha,
                                     cusparseSpMatDescr_t matA,
                                     cusparseDnVecDescr_t vecX, cusparseDnVecDescr_t vecY,
                                     cudaDataType_t computeType, cusparseSpSVAlg_t /*alg*/,
                                     cusparseSpSVDescr_t /*descr*/) {
    std::lock_guard<std::mutex> lk(g_descrMutex);
    auto aIt = g_csrMats.find(reinterpret_cast<uintptr_t>(matA));
    auto xIt = g_dnVecs.find(reinterpret_cast<uintptr_t>(vecX));
    auto yIt = g_dnVecs.find(reinterpret_cast<uintptr_t>(vecY));
    if (aIt == g_csrMats.end() || xIt == g_dnVecs.end() || yIt == g_dnVecs.end())
        return CUSPARSE_STATUS_INVALID_VALUE;
    const CsrMat &A = aIt->second;
    const DnVec  &X = xIt->second;
    DnVec        &Y = yIt->second;
    if (!A.rowOffsets || !A.colInd || !A.values || !X.values || !Y.values)
        return CUSPARSE_STATUS_INVALID_VALUE;

    // FP16 / BF16: widen vectors and matrix values to float, solve, narrow back.
    // This avoids duplicating the triangular-solve logic for reduced-precision types.
    if (computeType == CUDA_R_16F || computeType == CUDA_R_16BF) {
        // Widen X → float
        int64_t n64 = A.rows;
        std::vector<float> xF(n64), yF(n64), aF(A.nnz);
        float alphaF = 0.0f;
        auto h2f = [](uint16_t h) -> float {
            uint32_t bits = (static_cast<uint32_t>(h & 0x8000) << 16) |
                ((static_cast<uint32_t>((h >> 10) & 0x1f) == 0 ? 0 :
                  (static_cast<uint32_t>((h >> 10) & 0x1f) + 112u)) << 23) |
                (static_cast<uint32_t>(h & 0x3ff) << 13);
            float f; std::memcpy(&f, &bits, 4); return f;
        };
        auto bf2f = [](uint16_t h) -> float {
            uint32_t bits = static_cast<uint32_t>(h) << 16;
            float f; std::memcpy(&f, &bits, 4); return f;
        };
        auto f2h = [](float f) -> uint16_t {
            uint32_t bits; std::memcpy(&bits, &f, 4);
            uint16_t sign = (bits >> 16) & 0x8000;
            int exp = ((bits >> 23) & 0xff) - 127 + 15;
            if (exp <= 0) return sign;
            if (exp >= 31) return sign | 0x7c00;
            return sign | (static_cast<uint16_t>(exp) << 10) | ((bits >> 13) & 0x3ff);
        };
        auto f2bf = [](float f) -> uint16_t {
            uint32_t bits; std::memcpy(&bits, &f, 4); return static_cast<uint16_t>(bits >> 16);
        };

        bool isFP16 = (computeType == CUDA_R_16F);
        alphaF = isFP16 ? h2f(*static_cast<const uint16_t*>(alpha))
                        : bf2f(*static_cast<const uint16_t*>(alpha));
        const uint16_t *xH = static_cast<const uint16_t*>(X.values);
        uint16_t *yH = static_cast<uint16_t*>(Y.values);
        for (int64_t i = 0; i < n64; ++i) xF[i] = isFP16 ? h2f(xH[i]) : bf2f(xH[i]);
        for (int64_t i = 0; i < n64; ++i) yF[i] = isFP16 ? h2f(yH[i]) : bf2f(yH[i]);
        if (A.valueType == CUDA_R_16F || A.valueType == CUDA_R_16BF) {
            bool aFP16 = (A.valueType == CUDA_R_16F);
            const uint16_t *aH = static_cast<const uint16_t*>(A.values);
            for (int64_t i = 0; i < A.nnz; ++i) aF[i] = aFP16 ? h2f(aH[i]) : bf2f(aH[i]);
        } else if (A.valueType == CUDA_R_32F) {
            std::memcpy(aF.data(), A.values, static_cast<size_t>(A.nnz) * sizeof(float));
        }
        // Build widened descriptors for recursive call (float compute)
        CsrMat Af = A; Af.valueType = CUDA_R_32F;

        // Inline float triangular solve on widened data (g_descrMutex already held)
        float alphaF_val = alphaF;
        bool lowerTri = (Af.fillMode == CUSPARSE_FILL_MODE_LOWER);
        bool unitDiag = (Af.diagType == CUSPARSE_DIAG_TYPE_UNIT);
        bool transposed = (opA != CUSPARSE_OPERATION_NON_TRANSPOSE);
        int base = (Af.idxBase == CUSPARSE_INDEX_BASE_ONE) ? 1 : 0;
        for (int64_t i = 0; i < n64; ++i) yF[i] = alphaF_val * xF[i];
        auto getA2 = [&](int64_t i) -> double { return static_cast<double>(aF[i]); };
        auto getY2 = [&](int64_t i) -> double { return static_cast<double>(yF[i]); };
        auto setY2 = [&](int64_t i, double v) { yF[i] = static_cast<float>(v); };
        if (!transposed) {
            if (lowerTri) {
                for (int64_t r = 0; r < n64; ++r) {
                    int64_t rs = getIdx(Af.rowOffsets, Af.rowOffsetType, r) - base;
                    int64_t re = getIdx(Af.rowOffsets, Af.rowOffsetType, r+1) - base;
                    double rhs = getY2(r), diag = 1.0;
                    for (int64_t k = rs; k < re; ++k) {
                        int64_t c = getIdx(Af.colInd, Af.colIndType, k) - base;
                        if (c < r) rhs -= getA2(k) * getY2(c);
                        else if (c == r && !unitDiag) diag = getA2(k);
                    }
                    setY2(r, rhs / diag);
                }
            } else {
                for (int64_t r = n64-1; r >= 0; --r) {
                    int64_t rs = getIdx(Af.rowOffsets, Af.rowOffsetType, r) - base;
                    int64_t re = getIdx(Af.rowOffsets, Af.rowOffsetType, r+1) - base;
                    double rhs = getY2(r), diag = 1.0;
                    for (int64_t k = rs; k < re; ++k) {
                        int64_t c = getIdx(Af.colInd, Af.colIndType, k) - base;
                        if (c > r) rhs -= getA2(k) * getY2(c);
                        else if (c == r && !unitDiag) diag = getA2(k);
                    }
                    setY2(r, rhs / diag);
                }
            }
        } else {
            if (lowerTri) {
                for (int64_t r = n64-1; r >= 0; --r) {
                    int64_t rs = getIdx(Af.rowOffsets, Af.rowOffsetType, r) - base;
                    int64_t re = getIdx(Af.rowOffsets, Af.rowOffsetType, r+1) - base;
                    double diag = 1.0;
                    if (!unitDiag) for (int64_t k = rs; k < re; ++k) if (getIdx(Af.colInd, Af.colIndType, k)-base == r) { diag = getA2(k); break; }
                    double yr = getY2(r) / diag; setY2(r, yr);
                    for (int64_t k = rs; k < re; ++k) { int64_t c = getIdx(Af.colInd, Af.colIndType, k)-base; if (c < r) setY2(c, getY2(c) - getA2(k)*yr); }
                }
            } else {
                for (int64_t r = 0; r < n64; ++r) {
                    int64_t rs = getIdx(Af.rowOffsets, Af.rowOffsetType, r) - base;
                    int64_t re = getIdx(Af.rowOffsets, Af.rowOffsetType, r+1) - base;
                    double diag = 1.0;
                    if (!unitDiag) for (int64_t k = rs; k < re; ++k) if (getIdx(Af.colInd, Af.colIndType, k)-base == r) { diag = getA2(k); break; }
                    double yr = getY2(r) / diag; setY2(r, yr);
                    for (int64_t k = rs; k < re; ++k) { int64_t c = getIdx(Af.colInd, Af.colIndType, k)-base; if (c > r) setY2(c, getY2(c) - getA2(k)*yr); }
                }
            }
        }
        // Narrow back
        for (int64_t i = 0; i < n64; ++i) yH[i] = isFP16 ? f2h(yF[i]) : f2bf(yF[i]);
        return CUSPARSE_STATUS_SUCCESS;
    }

    if (computeType != CUDA_R_32F && computeType != CUDA_R_64F)
        return CUSPARSE_STATUS_NOT_SUPPORTED;

    int64_t n        = A.rows;
    bool isDouble    = (computeType == CUDA_R_64F);
    bool lowerTri    = (A.fillMode == CUSPARSE_FILL_MODE_LOWER);
    bool unitDiag    = (A.diagType == CUSPARSE_DIAG_TYPE_UNIT);
    bool transposed  = (opA != CUSPARSE_OPERATION_NON_TRANSPOSE);
    int  base        = (A.idxBase == CUSPARSE_INDEX_BASE_ONE) ? 1 : 0;

    double alphaD = isDouble ? *static_cast<const double*>(alpha)
                             : static_cast<double>(*static_cast<const float*>(alpha));

    auto getA = [&](int64_t i) -> double {
        return isDouble ? static_cast<const double*>(A.values)[i]
                        : static_cast<double>(static_cast<const float*>(A.values)[i]);
    };
    auto getX = [&](int64_t i) -> double {
        return isDouble ? static_cast<const double*>(X.values)[i]
                        : static_cast<double>(static_cast<const float*>(X.values)[i]);
    };
    auto getY = [&](int64_t i) -> double {
        return isDouble ? static_cast<const double*>(Y.values)[i]
                        : static_cast<double>(static_cast<const float*>(Y.values)[i]);
    };
    auto setY = [&](int64_t i, double v) {
        if (isDouble) static_cast<double*>(Y.values)[i] = v;
        else          static_cast<float*>(Y.values)[i]  = static_cast<float>(v);
    };

    // Initialise y = alpha * x
    for (int64_t i = 0; i < n; ++i) setY(i, alphaD * getX(i));

    if (!transposed) {
        if (lowerTri) {
            // Forward substitution: y[i] = (y[i] - sum A[i,j]*y[j], j<i) / A[i,i]
            for (int64_t r = 0; r < n; ++r) {
                int64_t rs = getIdx(A.rowOffsets, A.rowOffsetType, r)     - base;
                int64_t re = getIdx(A.rowOffsets, A.rowOffsetType, r + 1) - base;
                double rhs = getY(r), diag = 1.0;
                for (int64_t k = rs; k < re; ++k) {
                    int64_t c = getIdx(A.colInd, A.colIndType, k) - base;
                    if (c < r)            rhs -= getA(k) * getY(c);
                    else if (c == r && !unitDiag) diag = getA(k);
                }
                setY(r, rhs / diag);
            }
        } else {
            // Backward substitution: y[i] = (y[i] - sum A[i,j]*y[j], j>i) / A[i,i]
            for (int64_t r = n - 1; r >= 0; --r) {
                int64_t rs = getIdx(A.rowOffsets, A.rowOffsetType, r)     - base;
                int64_t re = getIdx(A.rowOffsets, A.rowOffsetType, r + 1) - base;
                double rhs = getY(r), diag = 1.0;
                for (int64_t k = rs; k < re; ++k) {
                    int64_t c = getIdx(A.colInd, A.colIndType, k) - base;
                    if (c > r)            rhs -= getA(k) * getY(c);
                    else if (c == r && !unitDiag) diag = getA(k);
                }
                setY(r, rhs / diag);
            }
        }
    } else {
        // Transposed solve — scatter approach:
        // After solving y[r], scatter A[r,j]*y[r] to all off-diagonal y[j].
        if (lowerTri) {
            // A^T is upper triangular → backward pass (r = n-1 down to 0)
            for (int64_t r = n - 1; r >= 0; --r) {
                int64_t rs = getIdx(A.rowOffsets, A.rowOffsetType, r)     - base;
                int64_t re = getIdx(A.rowOffsets, A.rowOffsetType, r + 1) - base;
                double diag = 1.0;
                if (!unitDiag)
                    for (int64_t k = rs; k < re; ++k)
                        if (getIdx(A.colInd, A.colIndType, k) - base == r) { diag = getA(k); break; }
                double yr = getY(r) / diag;
                setY(r, yr);
                for (int64_t k = rs; k < re; ++k) {
                    int64_t c = getIdx(A.colInd, A.colIndType, k) - base;
                    if (c < r) setY(c, getY(c) - getA(k) * yr);
                }
            }
        } else {
            // A^T is lower triangular → forward pass (r = 0 to n-1)
            for (int64_t r = 0; r < n; ++r) {
                int64_t rs = getIdx(A.rowOffsets, A.rowOffsetType, r)     - base;
                int64_t re = getIdx(A.rowOffsets, A.rowOffsetType, r + 1) - base;
                double diag = 1.0;
                if (!unitDiag)
                    for (int64_t k = rs; k < re; ++k)
                        if (getIdx(A.colInd, A.colIndType, k) - base == r) { diag = getA(k); break; }
                double yr = getY(r) / diag;
                setY(r, yr);
                for (int64_t k = rs; k < re; ++k) {
                    int64_t c = getIdx(A.colInd, A.colIndType, k) - base;
                    if (c > r) setY(c, getY(c) - getA(k) * yr);
                }
            }
        }
    }
    return CUSPARSE_STATUS_SUCCESS;
}

} // extern "C"
